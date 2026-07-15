# CLI Reference

Every command for running and troubleshooting the pipeline. For architecture,
setup, and configuration, see **[README.md](README.md)**.

> **Where to run these:** all commands run from the **`src/`** directory with your
> Python environment active (e.g. the `livesounds` conda env). The stage packages
> import correctly only from `src/`.
>
> **Config path:** `main.py`'s default `--config config/config.yaml` is relative
> to `Project/`, but you launch from `src/`. So pass the config explicitly —
> either `--config ../config/config.yaml` or the full absolute path. Examples
> below use `../config/config.yaml`.
>
> **Log location:** because you launch from `src/`, the per-run CSV lands in
> **`src/logs/`** (the relative `telemetry.log_dir` resolves against the current
> directory). The repo's top-level `logs/` is used only if you launch from
> `Project/`.

---

## `main.py` — the pipeline

### All flags

| Flag | Default | Purpose |
|---|---|---|
| `--config PATH` | `config/config.yaml` | Path to the YAML config. Pass `../config/config.yaml` from `src/`. |
| `--list-devices` | — | List input devices and exit (no prompts, no pipeline). |
| `--log-level LEVEL` | config `telemetry.level` (`INFO`) | Override log verbosity: `DEBUG` / `INFO` / `WARNING` / `ERROR`. |
| `--transcribe-only` | off | Print live transcripts, play **no** sound, write **no** CSV. |
| `--show-matches` | off | With `--transcribe-only`, also print which keywords *would* fire. |
| `--name NAME` | (prompted) | Recording name; skips the name prompt. Used in the CSV filename. |
| `--language CODE` | (prompted → `hu`) | Whisper decode language; skips the language prompt. Blank/omitted → Hungarian. |

### Startup prompts

On every run **except `--list-devices`**, before the model loads, the pipeline
asks two questions:

```
Name this recording: <you type a name>
Language for transcription (e.g. 'hu' for Hungarian) [default: hu]: <you type or press Enter>
```

- **Name** — used to build the log filename. Blank → a timestamped default such as
  `recording_20260710_211407`.
- **Language** — sets Whisper's `language` parameter. Blank → `hu` (Hungarian).

`--name` and/or `--language` pre-fill and skip the matching prompt, so a fully
non-interactive launch is possible:

```bash
python main.py --name testrun --language hu --config ../config/config.yaml
```

### 1. Run the full pipeline (transcribe + play sounds)

Normal mode: transcribe live audio and fire sound effects on keyword matches.

```bash
python main.py --config ../config/config.yaml
```

Requires the sound clips in `assets/sounds/` (present) and — for the current GPU
config — the CUDA libraries (see [GPU vs CPU](#gpu-vs-cpu-and-cuda-setup)). Stop
with **Ctrl+C**; the pipeline flushes the final transcript, closes audio, writes
the final CSV rows, and prints a telemetry summary.

**Produces:** `src/logs/<name>_<language>_logs.csv` — one row per ASR result
(every partial and every final) plus one per trigger. See
[README → Logging](README.md#logging-the-csv) for the columns and how to mine it
for new keyword variants.

### 2. Transcription-only mode (no sound, no CSV)

Prints live transcripts and plays **nothing** — the playback stage isn't even
started, so it works even if `assets/sounds/` is empty. Use it to check the ASR
understands the audio before caring about clips. (No CSV is written in this mode.)

```bash
python main.py --transcribe-only --config ../config/config.yaml
```

Output format:
- `~ some text` — a **partial** (in-progress hypothesis, may still change)
- `= some text` — a **final** (settled transcript for that utterance)

You are still prompted for a name and language (the language still affects
transcription).

#### See which keywords *would* fire (still no sound)

Adds an indented note whenever a pattern matches, so you can tune regexes
silently:

```bash
python main.py --transcribe-only --show-matches --config ../config/config.yaml
```

Prints e.g. `    [would fire: goals3.wav on 'gól']`.

#### Quieten the logs

Startup logs share the console with transcript lines. To leave mostly
transcripts:

```bash
python main.py --transcribe-only --log-level WARNING --config ../config/config.yaml
```

### 3. List audio input devices

Shows every input-capable device with its index, name, channel count, and default
samplerate. Use it to find a device index/name for `capture.device`.

```bash
python main.py --list-devices --config ../config/config.yaml
```

> **Device indices are not stable** — they can change between runs (plugging /
> unplugging gear, changing the Windows default device). If a device stops
> working, re-list and check its current index.

---

## Diagnostic tools

Three standalone helper scripts for getting audio capture working. They are
**not** part of the pipeline — troubleshooting only — and are safe to delete once
everything runs. Run them from `src/`.

### `levelmeter.py` — is the device actually hearing sound?

Opens a capture device the same way the pipeline does and prints input loudness
(RMS + peak) per chunk with a live bar, then reports whether it heard audio or
only silence. The single best check for "capture opens but transcripts are
empty," which usually means the device is capturing silence.

```bash
# Play some audio, then run (device 1 = current capture device in config):
python levelmeter.py --device 1

# Options:
python levelmeter.py --device 1 --seconds 30      # listen longer (default 10s)
python levelmeter.py --device 1 --samplerate 16000 --channels 1
```

| Flag | Default |
|---|---|
| `--device` | `1` |
| `--samplerate` | `16000` |
| `--channels` | `1` |
| `--seconds` | `10.0` |

Bars moving + "audio WAS captured" = the device works. Flat bars + "only silence"
= it isn't receiving your playback (Windows routing / wrong default output — see
[Capturing system audio](#capturing-system-audio-on-windows)).

### `audio_diag.py` — which devices can actually be opened?

Lists every input device and *tries to open each one* at the pipeline's settings
(and at the device's native channel count), reporting OK or the exact error.
Turns a cryptic PortAudio error into a concrete answer (wrong device, wrong
channels, or unsupported samplerate).

```bash
python audio_diag.py                              # probe at 16000 Hz, 1 channel
python audio_diag.py --samplerate 44100 --channels 2
```

| Flag | Default |
|---|---|
| `--samplerate` | `16000` |
| `--channels` | `1` |

### `probe7.py` — deep-probe specific devices

Focuses on a few device indices and reports each one's reported capabilities plus
concrete open attempts at 16000 Hz and its native rate, across channel counts.
Use when you need to know exactly what settings a particular device accepts.

```bash
python probe7.py                     # probes devices 7, 1, 8, 0
python probe7.py --devices 1         # just device 1
python probe7.py --devices 7 1 8     # a specific set
```

| Flag | Default |
|---|---|
| `--devices` | `7 1 8 0` |

---

## Capturing system audio on Windows

To transcribe audio playing on the PC (a stream/video) rather than a mic, capture
a **loopback** device — on this machine, **Stereo Mix (Realtek)**. Set
`capture.device` accordingly. Hard-won lessons:

- **Use the device that opens under MME or DirectSound, not WDM-KS.** WDM-KS
  copies fail with *"Blocking API not supported yet"*. The working Stereo Mix was
  the MME entry.
- **Index numbers shift.** Stereo Mix moved between indices during setup. If
  capture breaks, re-run `python main.py --list-devices` (or `audio_diag.py`) and
  update `capture.device`.
- **Stereo Mix only hears the Realtek output.** If audio plays out of HDMI / USB /
  Bluetooth, Stereo Mix captures silence. Set **Speakers (Realtek(R) Audio)** as
  the default *playback* device, and make sure Stereo Mix is enabled and unmuted
  in the Windows **Recording** tab (Levels up).
- **Confirm with `levelmeter.py`** before debugging anything downstream — it tells
  you in 10 seconds whether real audio is arriving.

Current working `capture` settings: `device: 1`, `channels: 1`, `samplerate: 16000`.

---

## GPU vs CPU (and CUDA setup)

The ASR backend is chosen by the top-level `device` in `config.yaml`; the Whisper
compute device by `asr.gpu.device`:

- **`device: auto`** (current) → always uses **faster-whisper**, on the **GPU
  (CUDA)** if one is detected, else Whisper on CPU. Vosk is **not** used in auto.
- **`device: gpu`** → faster-whisper; `asr.gpu.device: "cuda"` runs it on the GPU.
- **`device: cpu`** → **Vosk** (needs a Vosk model at `asr.cpu.model_path`, which
  is not currently installed).

### Running Whisper on GPU
Set `asr.gpu.device: "cuda"` and `asr.gpu.compute_type: "float16"` (the current
config). Needs the CUDA 12 runtime libraries. If you hit
`Library cublas64_12.dll is not found`, install them via pip:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

If a *different* missing-DLL error appears afterward (e.g. a `cudnn_*` file), it's
a cuDNN version-match issue — note the exact DLL name; that names the fix.

> The startup line *"no usable GPU was detected"* comes from a separate torch
> check and is **cosmetic** — what matters is whether faster-whisper's engine
> (CTranslate2) finds CUDA. If transcripts appear fast with no `transcribe failed`
> warning, the GPU is working regardless of that message.

### Running Whisper on CPU (fallback)
Set `asr.gpu.device: "cpu"` and a CPU-valid `compute_type` (`int8` or `float32` —
**not** `float16`, which is GPU-only; the loader auto-downgrades `float16` → `int8`
on CPU anyway). Works with no CUDA libraries but is much slower (~10 s latency
observed), so it's a fallback, not the target for live use.

---

## Config quick reference

Everything is driven by `config/config.yaml`. Most-used values (current):

| Setting | Current | Meaning |
|---|---|---|
| `device` | `auto` | Backend selection: `auto` (Whisper, GPU if present) / `gpu` (Whisper) / `cpu` (Vosk). |
| `asr.gpu.device` | `cuda` | Whisper compute device: `cuda` or `cpu`. |
| `asr.gpu.compute_type` | `float16` | `float16` (GPU) or `int8`/`float32` (CPU). |
| `asr.gpu.model` | `small` | Whisper model size; larger = more accurate, slower. |
| `asr.gpu.language` | `hu` | Decode language (Hungarian). Overridable via the prompt / `--language`. |
| `capture.device` | `1` | Input device index or name substring. |
| `capture.samplerate` | `16000` | Capture rate (Whisper/Vosk want 16000). |
| `capture.blocksize` | `1600` | Chunk size; latency ≈ blocksize / samplerate (~100 ms). |
| `matcher.keywords` | 4 rules | Keyword regex → sound mapping (Hungarian; see README). |

To change trigger words, edit `matcher.keywords` — each entry needs a `pattern`
(regex) and `sound` (a filename in `assets/sounds/`, or a list to pick from at
random), with an optional `cooldown`. Restart after any config edit; the config
is read fresh at startup.
