"""
main.py — entry point for the transcription-to-soundboard pipeline.

Wires the five stages together and runs the live loop:

    capture -> ASR -> transcript buffer -> regex matcher -> sound playback

Everything is driven by config/config.yaml (loaded once via config_loader). The
GPU/CPU ASR choice, buffer size, keyword->sound map, cooldowns, and playback
settings all come from there — this file just assembles the pieces and runs the
loop, keeping the hot path tight.

Usage
-----
On startup (except --list-devices) the pipeline asks two questions — a name for
the recording, then the transcription language — before the model loads. The
name is used for the per-run latency CSV (logs/<name>_<language>_logs.csv); the
language sets Whisper's `language` parameter (blank -> Hungarian, 'hu'). Pass
--name and/or --language to skip the prompts (e.g. scripted/non-interactive runs).

    python main.py                         # run with config/config.yaml
    python main.py --config path/to.yaml   # custom config
    python main.py --list-devices          # list input devices and exit
    python main.py --log-level DEBUG       # override log level
    python main.py --transcribe-only       # print transcripts, play no sound
    python main.py --transcribe-only --show-matches  # + show would-be triggers

Latency & transcript log
------------------------
Timing is recorded at the stage boundaries via the telemetry helper. The headline
number is per-trigger transcript->playback latency (from the moment the matched
ASR result was produced to the moment the sound is dispatched); it is logged to
the console as each trigger fires, with a summary at shutdown. Capture-stage
stats (dropped/overflow/queue depth) are logged periodically. In normal runs a
per-run CSV (logs/<name>_<language>_logs.csv) records one row per ASR result —
every partial AND every final — plus a row per fired trigger. Columns: is_final,
transcription, triggered, sound, matched_text, latency_ms (latency filled only on
triggered rows). Logging partials too is deliberate: the evolving/"wrong" text
(e.g. how English names get transcribed into Hungarian) is the raw material for
discovering new keyword variants. The file is therefore large and repetitive;
filter is_final=1 for the clean transcript.

Run this from the src/ directory (or with src/ on PYTHONPATH) so the stage
packages import cleanly.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import sys
import threading
from datetime import datetime
from typing import List

from config_loader import load_config, PipelineConfig
from capture import AudioCapture, list_input_devices
from asr import build_asr
from buffer import TranscriptBuffer
from matcher import KeywordMatcher
from playback import SoundPlayer
from telemetry import Telemetry

logger = logging.getLogger("pipeline.main")


_DEFAULT_LANGUAGE = "hu"  # Hungarian: used when no language is entered/passed.


def _safe_component(text: str, fallback: str) -> str:
    """Sanitize a string for safe use as one component of a filename.

    Keeps letters/digits/dot/underscore/hyphen; every other character (spaces,
    slashes, the Windows-illegal set < > : " / \\ | ? *, etc.) becomes '_'.
    Trims leading/trailing separators; falls back if the result is empty.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _ask(prompt: str) -> str:
    """Prompt on stdin, returning '' if input isn't available (non-TTY/EOF)."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _prompt_run_metadata(name_arg, language_arg):
    """Ask the two startup questions; return (run_name, language).

    Question 1 names the recording (used for the CSV filename); question 2 sets
    the transcription language. A provided flag (name_arg/language_arg) pre-fills
    and skips its prompt. An empty name becomes a timestamped default; an empty
    language becomes Hungarian ('hu').
    """
    # Q1: recording name (used for the CSV filename).
    name = name_arg if name_arg is not None else _ask("Name this recording: ")
    if not name:
        name = datetime.now().strftime("recording_%Y%m%d_%H%M%S")
        print(f"  (no name given — using '{name}')", flush=True)

    # Q2: language (Whisper's language parameter). Blank -> Hungarian.
    language = (
        language_arg if language_arg is not None
        else _ask("Language for transcription (e.g. 'hu' for Hungarian) [default: hu]: ")
    )
    if not language:
        language = _DEFAULT_LANGUAGE
        print(f"  (no language given — defaulting to '{language}', Hungarian)", flush=True)
    return name, language.lower()


def _distinct_sounds(cfg: PipelineConfig) -> List[str]:
    """The set of sound files referenced by the keyword rules (for preload).

    A keyword's `sound` may be a single filename or a list of filenames (the
    random-choice set). Flatten either form so every individual clip is
    preloaded, and de-duplicate across keywords that share a clip.
    """
    seen: List[str] = []
    for kw in cfg.matcher.keywords:
        names = kw.sound if isinstance(kw.sound, (list, tuple)) else [kw.sound]
        for name in names:
            if name not in seen:
                seen.append(name)
    return seen


def build_pipeline(
    cfg: PipelineConfig,
    build_player: bool = True,
    csv_path: "str | None" = None,
    run_name: str = "",
    language: str = "",
):
    """Construct every stage from a loaded config. Returns the components.

    When build_player is False (transcribe-only mode) the playback stage is not
    constructed at all, so no clips are decoded and no output stream is opened —
    you can verify transcription before any .wav files exist. `player` is then
    None.

    csv_path/run_name/language are handed to the telemetry helper: when csv_path
    is set, every finalized transcript and every fired trigger is appended to
    that CSV (run_name/language are recorded in the shutdown summary).
    """
    # Stage 2: ASR (factory picks Vosk/Whisper from cfg.resolved_device).
    asr = build_asr(cfg)

    # Stage 3: rolling transcript buffer.
    tbuf = TranscriptBuffer(
        max_chars=cfg.buffer.max_chars,
        include_partials=cfg.buffer.include_partials,
    )

    # Stage 4: regex matcher (compiles patterns once).
    matcher = KeywordMatcher(
        cfg.matcher.keywords,
        default_cooldown_seconds=cfg.matcher.default_cooldown_seconds,
        case_insensitive=cfg.matcher.case_insensitive,
        match_on_partials=cfg.matcher.match_on_partials,
    )

    # Stage 5: playback (preload clips, one mixing output stream).
    # channels=1 matches capture's mono default; clips must be at capture SR
    # (SoundPlayer warns otherwise). Skipped entirely in transcribe-only mode.
    if build_player:
        player = SoundPlayer(
            sounds_dir=cfg.playback.sounds_dir,
            sounds=_distinct_sounds(cfg),
            output_device=cfg.playback.output_device,
            preload=cfg.playback.preload,
            max_polyphony=cfg.playback.max_polyphony,
            volume=cfg.playback.volume,
            samplerate=cfg.capture.samplerate,
            channels=1,
        )
    else:
        player = None

    # Telemetry across boundaries. With a csv_path (normal runs), each finalized
    # transcript and each fired trigger is appended as a row to that CSV.
    tel = Telemetry(
        enabled=cfg.telemetry.enabled,
        log_stage_latency=cfg.telemetry.log_stage_latency,
        log_every_n_chunks=cfg.telemetry.log_every_n_chunks,
        csv_path=csv_path,
        run_name=run_name,
        language=language,
    )

    # Stage 1: capture (its config IS a CaptureConfig on cfg.capture).
    capture = AudioCapture(cfg.capture)

    return capture, asr, tbuf, matcher, player, tel


def run(
    cfg: PipelineConfig,
    transcribe_only: bool = False,
    show_matches: bool = False,
    run_name: str = "",
    language: str = "",
    csv_path: "str | None" = None,
) -> None:
    """Run the live pipeline until interrupted (Ctrl+C).

    transcribe_only : if True, print each transcript (partial/final) to stdout
        and do NOT play sounds. The playback stage isn't even built, so this
        works before any .wav clips exist — use it to check that the ASR hears
        you correctly.
    show_matches : only meaningful with transcribe_only. If True, also print
        which keyword(s) WOULD have fired (no sound), so you can tune patterns
        without making noise.
    run_name/language/csv_path : passed through to telemetry for the per-run
        latency CSV (see build_pipeline).
    """
    capture, asr, tbuf, matcher, player, tel = build_pipeline(
        cfg, build_player=not transcribe_only,
        csv_path=csv_path, run_name=run_name, language=language,
    )

    stop = {"flag": False}

    def _handle_sigint(signum, frame):
        logger.info("interrupt received, shutting down…")
        stop["flag"] = True
        capture.stop()  # unblocks the chunks() generator

    # Only main-thread execution may install signal handlers. Guarding this lets
    # the pipeline also be driven/embedded from another thread without crashing;
    # in that case the caller is responsible for stopping capture.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _handle_sigint)
    else:
        logger.info("not on main thread; SIGINT handler not installed")

    logger.info(
        "pipeline starting: device=%s (resolved), %d keyword(s), "
        "capture ~%.1f ms/chunk%s",
        cfg.resolved_device, len(cfg.matcher.keywords),
        cfg.capture.chunk_latency_ms,
        " [TRANSCRIBE-ONLY: no sound]" if transcribe_only else "",
    )
    capture.start()

    def _process(results):
        """Fold ASR results through buffer + matcher; play or print per mode."""
        for result in results:
            text = tbuf.update(result)

            if transcribe_only:
                # Live transcript view. '~' = partial (may change), '=' = final.
                marker = "=" if result.is_final else "~"
                print(f"{marker} {result.text}", flush=True)
                if show_matches:
                    for ev in matcher.scan(text, is_final=result.is_final,
                                           base_offset=tbuf.base_offset):
                        print(f"    [would fire: {ev.sound} on '{ev.matched_text}']",
                              flush=True)
                continue

            # Normal mode: scan, play, and log.
            events = matcher.scan(text, is_final=result.is_final,
                                  base_offset=tbuf.base_offset)
            for ev in events:
                player.play(ev.sound)
                # Log the fired trigger (triggered=1 row). The sound is already
                # dispatched above, so this logging never delays it.
                tel.on_trigger(
                    result.timestamp, result.text, ev.sound, ev.matched_text,
                    is_final=result.is_final,
                )
            # Log every non-triggering result — partials AND finals — so the
            # CSV captures the full evolving transcript, including "wrong"
            # partials. This is deliberate: mining mangled/partial text (e.g.
            # how English names get transcribed) is how new keyword variants
            # are found.
            if not events:
                tel.on_transcript(result.text, is_final=result.is_final)

    try:
        for chunk in capture.chunks(timeout=1.0):
            if stop["flag"]:
                break
            results = asr.feed(chunk)
            _process(results)
            tel.on_chunk(capture.stats())
    finally:
        # Flush the ASR's final tail through the rest of the pipeline.
        try:
            _process(asr.flush())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("flush failed: %s", exc)
        capture.stop()
        try:
            asr.close()
        except Exception:
            pass
        if player is not None:
            player.close()
        tel.log_summary()
        tel.close()
        logger.info("pipeline stopped. capture stats: %s", capture.stats())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Transcription-to-soundboard pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="path to config.yaml")
    parser.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    parser.add_argument("--log-level", default=None, help="override log level (DEBUG/INFO/…)")
    parser.add_argument(
        "--transcribe-only", action="store_true",
        help="print live transcripts and DO NOT play sounds (no clips needed)",
    )
    parser.add_argument(
        "--show-matches", action="store_true",
        help="with --transcribe-only, also print which keywords would fire",
    )
    parser.add_argument(
        "--name", default=None,
        help="recording name; skips the name prompt. Used in the CSV filename.",
    )
    parser.add_argument(
        "--language", default=None,
        help="transcription language (e.g. 'hu'); skips the language prompt. "
             "Blank/omitted defaults to Hungarian ('hu').",
    )
    args = parser.parse_args(argv)

    if args.list_devices:
        # Configure minimal logging so the device list is readable.
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        try:
            for d in list_input_devices():
                print(f"[{d['index']}] {d['name']} "
                      f"(in_channels={d['max_input_channels']}, "
                      f"default_sr={d['default_samplerate']})")
        except Exception as exc:
            print(f"could not list devices: {exc}", file=sys.stderr)
            return 1
        return 0

    # Two startup questions (recording name, then language) before the model
    # loads. --name/--language pre-fill and skip the matching prompt.
    run_name, language = _prompt_run_metadata(args.name, args.language)

    cfg = load_config(args.config)

    # The chosen language drives Whisper's `language` parameter. (The Vosk CPU
    # backend uses a fixed model and ignores this — see the note logged below.)
    cfg.asr.gpu.language = language

    level = (args.log_level or cfg.telemetry.level or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("recording=%r, language=%r (Whisper language parameter)",
                run_name, language)
    if cfg.resolved_device == "cpu":
        logger.info(
            "note: resolved device is cpu (Vosk); language %r applies to the "
            "Whisper backend only, not Vosk.", language,
        )

    # Per-run CSV path: <log_dir>/<name>_<language>_logs.csv (transcript +
    # triggers). Skipped in transcribe-only mode and when telemetry is disabled.
    csv_path = None
    if cfg.telemetry.enabled and not args.transcribe_only:
        safe_name = _safe_component(run_name, "recording")
        safe_lang = _safe_component(language, _DEFAULT_LANGUAGE)
        csv_path = os.path.join(
            cfg.telemetry.log_dir, f"{safe_name}_{safe_lang}_logs.csv"
        )

    run(
        cfg,
        transcribe_only=args.transcribe_only,
        show_matches=args.show_matches,
        run_name=run_name,
        language=language,
        csv_path=csv_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
