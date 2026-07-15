# Project Punchline

A latency focused pipeline that listens to a live audio  (e.g. a commentator
during a live event), transcribes it in real time, scans the running transcript
for keyword patterns, and **plays a sound effect the moment a trigger word is
spoken**. The entire design is optimized for minimizing the delay
between a word being said and the corresponding sound firing. If it lags, the
joke is dead.

The current configuration is tuned for **Hungarian football commentary** running
on **GPU (faster-whisper)**.

> **Command reference:** this file covers architecture, setup, and configuration.
> For the full list of commands and flags (running modes, the audio-diagnostic
> tools, Windows loopback and  setup), see **[CLI.md](CLI.md)**.

---

## How it works

Audio moves through five stages in a line each stage hands its result to the
next. A separate telemetry layer times every hand-off, so any delay is easy to
spot:

```
  mic / system loopback
          │
          ▼
  ┌───────────────┐   float32 mono chunks   ┌───────────────┐
  │ 1. CAPTURE    │ ──────────────────────► │ 2. ASR        │
  │ chunked audio │                         │ streaming     │
  └───────────────┘                         │ (GPU or CPU)  │
                                            └───────┬───────┘
                                                    │ TranscriptResult
                                                    │ (partial / final)
                                                    ▼
                                            ┌───────────────┐
                                            │ 3. BUFFER     │
                                            │ rolling text  │
                                            │ window        │
                                            └───────┬───────┘
                                                    │ scannable text
                                                    ▼
                                            ┌───────────────┐
                                            │ 4. MATCHER    │
                                            │ regex +       │
                                            │ cooldown      │
                                            └───────┬───────┘
                                                    │ MatchEvent (sound to play)
                                                    ▼
                                            ┌───────────────┐
                                            │ 5. PLAYBACK   │
                                            │ mix + fire    │
                                            └───────────────┘

   telemetry taps every boundary → per-trigger latency + transcript/trigger CSV
```

**1. Capture**  Listens to one microphone or system-audio device and slices the
incoming sound into small, equal-sized chunks. If the rest of the pipeline falls behind, it
throws away the *oldest* chunk for a live soundboard, stale audio is worse than
a tiny gap.

**2. ASR (speech-to-text)** Turns those chunks into text as they arrive,
without waiting for whole sentences. It produces two kinds of results:
- **partials**  the current best guess, which can still change as more words come in.
- **finals**  text the model has settled on and won't revise.

Two engines are available, and you pick one in the config: **Whisper on GPU**,
the default here and **Vosk on CPU**  lighter, used only when
you set `device: cpu` (needs a Vosk model on disk; see [CPU_SETUP](CPU_SETUP.md)).

**3. Buffer the rolling window** Holds a short, always-current snapshot of the
most recent transcript text for the matcher to read. See
[The rolling window](#the-rolling-window) below for how it stays live.

**4. Matcher**  Checks the window's text against the keyword patterns. Each
keyword points to a sound (or a list of sounds, one picked at random). A
per-keyword **cooldown** stops the same word from firing over and over so a
commentator yelling "gól gól gól!" triggers the sound once, not three times. The
matcher only reports *what to play*; it doesn't play the sound itself.

**5. Playback**  Plays the sound. Every clip is loaded into memory when the
pipeline starts, so there's no disk delay when a trigger fires. Overlapping
sounds really do play at the same time, up to a set limit in the config.yaml (`max_polyphony`).

**Telemetry**  Times every stage and writes a CSV log, so latency is measured,
not guessed (see [Logging](#logging-the-csv)).

---

### The rolling window

The buffer keeps a small **rolling window** of the most recent transcript text, a short window that slides along the speech as it's spoken, always showing
the latest words while older ones scroll off the back. It only keeps about the
last 240 characters, because
that's all the matcher needs to see.

Two behaviors are what make it feel *live*:

1. **New final text is added to the end**, and once the window is full the oldest
   text drops off the front, so it's always current and never grows without bound.
2. **While a word is still being figured out, each new partial replaces the
   unfinished tail** instead of stacking up. If the model first hears `gól` and
   then revises its guess to `gólja`, the window shows just `gólja`, not
   `gól gólja`. Once the text finalizes it's locked in, and the next partial
   starts a fresh tail.

That "rewrite the tail" behavior is exactly what lets you watch the transcription
update in real time: the end of the window keeps correcting itself to the model's
latest guess, much like live TV captions that visibly fix themselves mid-sentence.
And because the matcher reads this same window, it can catch a trigger word the
instant it appears in a partial, before the sentence is even finished.

---

## Folder structure

```
Project/
├── README.md                     ← this file (architecture, setup, config)
├── CLI.md                        full command / flag reference
├── config/
│   └── config.yaml               single source of truth for every stage
├── src/
│   ├── main.py                   entry point: prompts, wires stages, runs loop
│   ├── config_loader.py          loads + validates config, resolves device:auto
│   ├── audio_diag.py             diagnostic: which input devices open?
│   ├── levelmeter.py             diagnostic: is a device hearing audio?
│   ├── probe7.py                 diagnostic: deep-probe specific devices
│   ├── capture/                  Stage 1 — chunked audio input
│   ├── asr/                      Stage 2 — streaming ASR (base + gpu + cpu + factory)
│   ├── buffer/                   Stage 3 — rolling transcript window
│   ├── matcher/                  Stage 4 — regex keyword matcher
│   ├── playback/                 Stage 5 — sound trigger / mixing playback
│   └── telemetry/                cross-stage latency + transcript/trigger logging
├── assets/
│   └── sounds/                   keyword sound clips (present: see Setup)
├── logs/                         per-run CSV log directory (see Logging)
└── tests/                        (empty — no tests written yet)
```

Each stage is its own package with a single clear responsibility. The dependency
direction is one-way (capture → asr → buffer → matcher → playback);
`config_loader` is imported by `main.py` and the asr factory but doesn't import
any backend, so the GPU/CPU modules stay decoupled from config parsing.

---



## Running (quickstart)


```bash
cd src

# List input devices (find an index for capture.device)
python main.py --list-devices --config ../config/config.yaml

# Run the pipeline
python main.py --config ../config/config.yaml
```

On startup (except `--list-devices`) the pipeline asks **two questions before the
model loads**:

1. **Name this recording**  used for the log filename. Blank → a timestamped
   default like `recording_20260710_211407`.
2. **Language for transcription** sets Whisper's `language`. Blank → Hungarian
   (`hu`).

Pass `--name` and/or `--language` to skip the matching prompt (useful for
scripted runs). Stop with **Ctrl+C**, the pipeline flushes the final
transcript, closes the audio streams, writes the last log rows, and prints a
telemetry summary.

For every mode, flag, diagnostic tool, and the Windows loopback / CUDA setup, see
**[CLI.md](CLI.md)**.

---

## Logging (the CSV)

In a normal run (not `--transcribe-only`, telemetry enabled), each run writes a
CSV named **`<name>_<language>_logs.csv`**. Columns:

| Column | Meaning |
|---|---|
| `is_final` | `1` if the row reflects finalized ASR text, `0` if a partial |
| `transcription` | the ASR text for that result |
| `triggered` | `1` if a sound fired on this text, `0` otherwise |
| `sound` | the sound file dispatched (blank if not triggered) |
| `matched_text` | the exact substring that fired (blank if not triggered) |
| `latency_ms` | transcript → playback latency (blank if not triggered) |



**Where it lands:** the file is written to `src/logs/…`. 

---

## Configuration

Everything is driven by `config/config.yaml` there are no scattered constants
in code. The latency-relevant knobs, and the trade each one makes:

| Setting | Default (current) | Effect on latency / behavior |
|---|---|---|
| `device` | `auto` | `gpu` / `cpu` / `auto`. `auto` uses the **Whisper** backend on GPU if a CUDA GPU is detected, else Whisper on CPU; **Vosk is used only for explicit `cpu`**. |
| `asr.gpu.model` | `small` | Whisper model size; larger = more accurate, higher latency. |
| `asr.gpu.compute_type` | `float16` | GPU compute type. On CPU it is auto-downgraded to `int8` (`float16` is GPU-only). |
| `asr.gpu.language` | `hu` | Whisper decode language. Overridden at startup by the language prompt / `--language`. |
| `asr.gpu.beam_size` | `1` | Greedy (1) is fastest; higher beams are more accurate but slower. |
| `capture.blocksize` | `1600` | **Primary latency knob.** Latency ≈ `blocksize / samplerate` (1600 @ 16 kHz ≈ 100 ms). Smaller = lower latency, higher glitch risk. |
| `capture.device` | `1` | Input device index or name substring (`--list-devices` to find it). |
| `capture.queue_maxsize` | `8` | Bounds worst-case backlog to `queue_maxsize × chunk_latency`. Small on purpose. |
| `asr.emit_partials` | `true` | Emit in-progress hypotheses so triggers can fire before an utterance ends. |
| `buffer.max_chars` | `240` | Keep near the longest keyword phrase; larger = more text scanned per match. |
| `matcher.match_on_partials` | `true` | Fire on unstable partial text (lowest latency) vs. wait for finals (safer). |
| `matcher.default_cooldown_seconds` | `3.0` | Per-keyword debounce; a keyword's own `cooldown` overrides it. |
| `playback.max_polyphony` | `4` | Max simultaneous sounds; new triggers over the cap evict the oldest. |
| `telemetry.log_every_n_chunks` | `50` | How often capture-stage stats are logged. |

### Keyword → sound mapping
Under `matcher.keywords`, each rule has a regex `pattern`, a `sound` (a filename,
or a **list** of filenames, one chosen at random per fire), and an optional
`cooldown`. The current Hungarian rules:

```yaml
matcher:
  keywords:
    - pattern: "\\bNesszi\\b"
      sound: "messi.wav"
      cooldown: 3.0
    - pattern: "\\b(gól|gol|gool|bevágta|betalált)\\b"
      sound: ["goals1.wav", "goals2.wav", "goals3.wav", "goals4.wav"]
      cooldown: 500.0        # one random goal song per celebration; long lockout
    - pattern: "\\bMesszi\\b"
      sound: "goat.wav"
      cooldown: 3.0
    - pattern: "\\b(sárga|piros|sárgalap|pirosalap)\\b"
      sound: "siren.wav"
      cooldown: 3.0
```

Patterns are validated at startup, an invalid regex fails loudly rather than
mid-stream. After any config edit, just restart; the config is read fresh.

---
