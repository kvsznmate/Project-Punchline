# Audio Capture (Stage 1)

The entry point of the transcription-to-soundboard pipeline. This is the **only**
place live audio enters the system. It streams audio from a microphone or a
system-loopback device in small, fixed-size chunks and hands them to the ASR
stage (Stage 2) with as little added latency as possible.

```
[ mic / system loopback ] --> (Stage 1: capture) --> [ ASR ] --> [ buffer ] --> [ matcher ] --> [ playback ]
                                     you are here
```

## Role in the pipeline

Capture sets the **latency floor** for everything downstream: a sound can never
fire sooner than the audio that triggers it is captured. Its job is narrow and
deliberate:

1. Open one input device at a known sample rate.
2. Receive audio in small chunks from PortAudio's realtime callback.
3. Downmix to mono `float32` and put each chunk on a bounded queue.
4. Let the consumer (ASR) pull chunks off that queue.

It does **not** transcribe, resample, detect speech, or do any analysis — that
belongs to later stages. Keeping this stage minimal is what keeps its latency
bounded and predictable.

## How it works

### Chunked callback, not file/blocking reads
`sounddevice` (PortAudio) calls a callback every `blocksize` frames. The latency
added by this stage is approximately:

```
chunk_latency ≈ blocksize / samplerate
```

| blocksize | samplerate | ≈ latency/chunk |
|-----------|------------|-----------------|
| 800       | 16000      | 50 ms           |
| 1600      | 16000      | 100 ms          |
| 3200      | 16000      | 200 ms          |

This is the **primary latency knob** for the stage, which is why it lives in
config, not in code. Smaller chunks = lower latency, but more callbacks per
second and a higher risk of input overflow (audio glitches) on a busy machine.
Tune it empirically against your hardware.

### The realtime callback does almost nothing
The callback runs on PortAudio's high-priority audio thread. If it blocks or
does heavy work, PortAudio can't hand over the next block and you get **input
overflow** — dropped audio and glitches. So the callback only:
- downmixes to mono if needed,
- copies the block (PortAudio reuses its buffer after the callback returns),
- drops it on a queue and returns.

All real work happens on the consumer thread that reads `chunks()`.

### Bounded, freshest-first queue
The handoff queue has a small `queue_maxsize`. This bounds the worst-case
backlog latency to `queue_maxsize * chunk_latency`. If the consumer falls
behind and the queue fills, the capture callback **drops the oldest chunk** to
make room for the newest.

For a live soundboard, **stale audio is worse than a small gap** — a late sound
is a dead joke, so we favor freshness over completeness. Every drop is counted
and exposed via `stats()` so this tradeoff is visible, never silent.

### Audio format emitted
Mono `float32` numpy arrays in `[-1.0, 1.0]`, shape `(blocksize,)`. `float32` is
what most streaming ASR frontends expect, so we avoid a later cast. If a
specific backend needs `int16`, convert at the ASR boundary — not here.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Exposes the public API: `AudioCapture`, `CaptureConfig`, `list_input_devices`. |
| `audio_capture.py` | The implementation: config dataclass, capture class, device listing. |
| `README.md` | This document. |

### Public API
- **`CaptureConfig`** — resolved settings for a session (mirrors the `capture:`
  block of `config/config.yaml`). Has a `chunk_latency_ms` property for quick
  latency readouts.
- **`AudioCapture`** — the streaming capture object. Key methods:
  - `start()` / `stop()` — open/close the device (also works as a context manager).
  - `chunks(timeout=1.0)` — generator yielding chunks until `stop()`.
  - `read(timeout=None)` — pull a single chunk (or `None` on timeout).
  - `stats()` — counters: captured, dropped, overflow events, queue depth, chunk latency.
- **`list_input_devices()`** — enumerate input-capable devices (index, name,
  channels, default samplerate), including loopback/monitor devices.

## How to use it

### 1. Install the dependency
```bash
pip install sounddevice numpy
```
`sounddevice` needs PortAudio. On Debian/Ubuntu: `sudo apt install libportaudio2`.
On macOS (Homebrew): `brew install portaudio`. On Windows it ships with the wheel.

### 2. Find your input device
```python
from capture import list_input_devices
for d in list_input_devices():
    print(d["index"], d["name"], d["max_input_channels"])
```
Note the index or a unique substring of the name to put in config.

### 3. Stream chunks
```python
from capture import AudioCapture, CaptureConfig

cfg = CaptureConfig(
    samplerate=16000,   # match your ASR model
    blocksize=1600,     # ~100 ms/chunk — the latency knob
    channels=1,
    device=None,        # None = system default input
    queue_maxsize=8,
)

with AudioCapture(cfg) as cap:      # start()/stop() handled for you
    for chunk in cap.chunks():
        # chunk: float32 mono numpy array, shape (blocksize,)
        asr.feed(chunk)             # hand off to Stage 2
```

Manual lifecycle instead of the context manager:
```python
cap = AudioCapture(cfg)
cap.start()
try:
    while cap.running:
        chunk = cap.read(timeout=1.0)
        if chunk is not None:
            asr.feed(chunk)
finally:
    cap.stop()
```

### Capturing system audio (a live stream, not your mic)
Select a **loopback/monitor** device as `device`:
- **Windows**: use a WASAPI loopback device, or enable "Stereo Mix".
- **macOS**: install a virtual loopback device (e.g. BlackHole) and select it.
- **Linux (PulseAudio/PipeWire)**: pick the `.monitor` source of your output.

Use `list_input_devices()` to find its index/name, then set it in config.

## Configuration reference

These map to the `capture:` section of `config/config.yaml`:

| Field | Default | Meaning / latency note |
|-------|---------|------------------------|
| `samplerate` | `16000` | Hz. Match the ASR model; resampling later costs latency + a copy. |
| `blocksize` | `1600` | Frames/chunk. **Latency ≈ blocksize/samplerate.** Main knob. |
| `channels` | `1` | Input channels; >1 is downmixed to mono. |
| `device` | `None` | Index or name substring; `None` = default input. Loopback here for system audio. |
| `dtype` | `"float32"` | Delivered sample dtype; `float32` avoids a later cast. |
| `queue_maxsize` | `8` | Max buffered chunks. Bounds backlog latency to `maxsize * chunk_latency`. |

## Tuning for latency
- **Start** at `blocksize=1600` (~100 ms). If your machine keeps up cleanly,
  step down to `800` (~50 ms), then `400`, watching for overflow.
- If `stats()["overflow_events"]` climbs or you hear glitches, your blocksize is
  too small for the machine — step back up.
- Keep `queue_maxsize` small (4–8). A large queue hides backlog as **latency**
  instead of surfacing it as drops.

## Troubleshooting
- **`RuntimeError: sounddevice is not available`** — not installed, or PortAudio
  is missing. See install step above.
- **Frequent overflow warnings / rising `overflow_events`** — blocksize too
  small, or the consumer (ASR) is too slow. Increase blocksize, or speed up /
  offload the consumer.
- **Rising `dropped_chunks`** — the consumer can't keep up with real time. This
  is the design working (favoring fresh audio), but sustained drops mean a
  downstream stage is the bottleneck — check ASR latency logs.
- **Capturing silence** — wrong `device`, or you selected an *output* device
  with no input channels. Re-check `list_input_devices()`.

## Observability
`stats()` returns a dict you can log at the capture→ASR boundary:
```python
{
  "captured_chunks": int,   # chunks delivered by PortAudio
  "dropped_chunks": int,    # chunks discarded due to a full queue (consumer behind)
  "overflow_events": int,   # PortAudio input overflows (blocksize too small)
  "queue_depth": int,       # chunks currently waiting
  "chunk_latency_ms": float # blocksize/samplerate in ms
}
```
Log these periodically so capture-stage latency and backpressure stay visible
rather than guessed at.
