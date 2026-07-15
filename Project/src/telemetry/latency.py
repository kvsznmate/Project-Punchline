"""
telemetry/latency.py — transcript + trigger logging (and stage timing).

Latency-first project: latency must be measured, not guessed. This helper
records timestamps at the pipeline's stage boundaries, logs the
transcript->playback latency of each trigger to the console, and appends rows
to a per-run CSV that doubles as a tuning log.

Boundaries: capture -> asr partial/final -> regex match -> sound playback.

Headline latency number:
    trigger latency = time the sound is dispatched - time the matched
                      TranscriptResult was produced by the ASR.

CSV log
-------
The CSV is a readable transcript annotated with what fired, not just a trigger
ledger. Columns:

    is_final, transcription, triggered, sound, matched_text, latency_ms

A row is written for EVERY ASR result — partials (is_final=0) and finals
(is_final=1) — via on_transcript, plus a row per trigger via on_trigger. Logging
partials is deliberate: the evolving/"wrong" hypotheses (e.g. mangled English
names) are the raw material for finding new keyword variants. The trade-off is a
large, repetitive file — filter is_final=1 for the clean transcript. On triggered
rows, sound/matched_text/latency_ms are filled; on plain transcript rows they are
blank. (The ASR already suppresses consecutive identical partials, so rows change
as the hypothesis changes rather than repeating verbatim every step.)

The file is opened lazily on the first logged row (so a run that logs nothing
leaves no empty file), in append mode (reusing a run name adds rows rather than
clobbering). Writing is off the playback critical path: main.py dispatches the
sound before calling on_trigger. All timings use time.monotonic().
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

logger = logging.getLogger("pipeline.telemetry")


# Column order for the per-run CSV (transcript + triggers).
_CSV_HEADER = [
    "is_final",       # 1 if the row reflects finalized ASR text, 0 if a partial
    "transcription",  # the ASR text for this result
    "triggered",      # 1 if a sound fired on this text, 0 otherwise
    "sound",          # sound file dispatched (blank if not triggered)
    "matched_text",   # exact substring that fired (blank if not triggered)
    "latency_ms",     # transcript->playback latency (blank if not triggered)
]


@dataclass
class LatencyStats:
    """Running latency stats for one boundary (all values in milliseconds)."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        if ms < self.min_ms:
            self.min_ms = ms
        if ms > self.max_ms:
            self.max_ms = ms

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def summary(self) -> str:
        if not self.count:
            return "no samples"
        return (
            f"n={self.count} avg={self.avg_ms:.1f}ms "
            f"min={self.min_ms:.1f}ms max={self.max_ms:.1f}ms"
        )


class Telemetry:
    """Transcript/trigger logger + latency recorder for the pipeline.

    Parameters
    ----------
    enabled : bool
        If False, all methods are cheap no-ops.
    log_stage_latency : bool
        If True, log each trigger's transcript->playback latency to the console.
    log_every_n_chunks : int
        Emit a periodic capture-stage line every N chunks.
    csv_path : Optional[str]
        If set, append transcript/trigger rows to this CSV (opened lazily).
        None disables the CSV log.
    run_name, language : str
        Recording metadata (already encoded in the CSV filename); recorded in
        the shutdown summary line for context.
    """

    def __init__(
        self,
        enabled: bool = True,
        log_stage_latency: bool = True,
        log_every_n_chunks: int = 50,
        csv_path: Optional[str] = None,
        run_name: str = "",
        language: str = "",
    ) -> None:
        self.enabled = bool(enabled)
        self.log_stage_latency = bool(log_stage_latency)
        self.log_every_n_chunks = max(1, int(log_every_n_chunks))
        self.csv_path = csv_path
        self.run_name = run_name
        self.language = language
        self._chunks = 0
        self._trigger_stats = LatencyStats()
        self._csv_file: Optional[TextIO] = None
        self._csv_writer = None

    # -- per-chunk ---------------------------------------------------------

    def on_chunk(self, capture_stats: Optional[dict] = None) -> None:
        """Call once per audio chunk. Periodically logs capture-stage stats."""
        if not self.enabled:
            return
        self._chunks += 1
        if capture_stats is not None and self._chunks % self.log_every_n_chunks == 0:
            logger.info(
                "capture stage @ chunk %d: %s",
                self._chunks, _fmt_stats(capture_stats),
            )

    # -- csv ---------------------------------------------------------------

    def _ensure_csv(self) -> None:
        """Open the CSV on first use; write header if the file is new/empty.

        Append mode preserves data when a run name is reused (a warning is
        logged). Any failure disables the CSV for the rest of the run so we
        don't retry on every row.
        """
        if self._csv_writer is not None or not self.csv_path:
            return
        path = Path(self.csv_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            existed = path.exists() and path.stat().st_size > 0
            self._csv_file = path.open("a", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            if existed:
                logger.warning("log CSV %s already exists; appending", path)
            else:
                self._csv_writer.writerow(_CSV_HEADER)
                self._csv_file.flush()
            logger.info("logging transcript + triggers to %s", path)
        except Exception as exc:  # pragma: no cover - fs/permission issues
            logger.warning("could not open log CSV %s: %s", self.csv_path, exc)
            self._csv_file = None
            self._csv_writer = None
            self.csv_path = None

    def _write_row(self, row: list) -> None:
        """Append one row to the CSV (lazily opening it first). Best-effort."""
        if not self.csv_path:
            return
        self._ensure_csv()
        if self._csv_writer is None:
            return
        try:
            self._csv_writer.writerow(row)
            self._csv_file.flush()  # survive a crash mid-stream
        except Exception as exc:  # pragma: no cover
            logger.warning("failed writing CSV row: %s", exc)

    # -- per-result --------------------------------------------------------

    def on_trigger(
        self,
        result_timestamp: float,
        transcription: str,
        sound: str,
        matched_text: str,
        is_final: bool = False,
    ) -> float:
        """Log a fired trigger (triggered=1 row); return the latency (ms).

        `result_timestamp` is the monotonic timestamp of the TranscriptResult
        that produced the match; "now" is the moment the sound is dispatched.
        `transcription` is the ASR text for that result; `matched_text` is the
        exact substring that fired.
        """
        now = time.monotonic()
        latency_ms = (now - result_timestamp) * 1000.0
        if not self.enabled:
            return latency_ms
        self._trigger_stats.add(latency_ms)
        if self.log_stage_latency:
            logger.info(
                "TRIGGER %s on %r — transcript->playback %.1f ms",
                sound, matched_text, latency_ms,
            )
        self._write_row([
            int(bool(is_final)),
            transcription,
            1,
            sound,
            matched_text,
            f"{latency_ms:.1f}",
        ])
        return latency_ms

    def on_transcript(self, transcription: str, is_final: bool = True) -> None:
        """Log a non-triggering transcript row (triggered=0, no sound/latency).

        Called for every non-triggering ASR result — partials and finals — so
        the CSV captures the full evolving transcript (the raw material for
        finding new keyword variants), not just a trigger ledger. Empty
        transcriptions are skipped.
        """
        if not self.enabled or not self.csv_path:
            return
        if not transcription or not transcription.strip():
            return
        self._write_row([
            int(bool(is_final)),
            transcription,
            0,
            "",
            "",
            "",
        ])

    # -- shutdown ----------------------------------------------------------

    def summary(self) -> str:
        return f"trigger latency: {self._trigger_stats.summary()}"

    def log_summary(self) -> None:
        if self.enabled:
            logger.info(
                "telemetry summary — run=%r lang=%r %s",
                self.run_name, self.language, self.summary(),
            )

    def close(self) -> None:
        """Flush and close the CSV log. Safe to call multiple times."""
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:  # pragma: no cover
                pass
            finally:
                self._csv_file = None
                self._csv_writer = None


def _fmt_stats(stats: dict) -> str:
    """Compactly format the capture stats dict for a log line."""
    return ", ".join(f"{k}={v}" for k, v in stats.items())
