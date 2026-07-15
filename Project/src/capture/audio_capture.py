"""
audio_capture.py — Stage 1: streaming audio capture.

Captures live audio from an input device (microphone or system loopback) and
makes it available as a stream of small, fixed-size chunks for the downstream
ASR stage.

Why it is built this way (latency notes)
----------------------------------------
The whole pipeline is latency-first, and capture sets the floor for everything
after it. Two decisions follow from that:

1. Chunked callback, not full-file / blocking reads.
   PortAudio (via `sounddevice`) invokes a callback every `blocksize` frames.
   The added latency at this stage is roughly:

       chunk_latency ≈ blocksize / samplerate

   e.g. blocksize=1600 @ 16 kHz ≈ 100 ms; blocksize=800 ≈ 50 ms. This is the
   biggest tunable knob here, so it is exposed in config rather than hardcoded.
   Smaller = lower latency but more callbacks/overhead and higher glitch risk
   on a busy machine. Tune empirically.

2. The realtime callback does *no work* beyond enqueueing.
   The callback runs on PortAudio's high-priority audio thread. Any heavy work
   there (or blocking) causes input overflow = dropped audio + glitches. So the
   callback only copies the block into a bounded queue and returns immediately.
   The consumer thread (ASR) drains the queue. This keeps capture latency
   bounded and jitter low.

The queue is bounded. If the consumer falls behind, the *oldest* chunk is
dropped so we always process the freshest audio — for a live soundboard, stale
audio is worse than a gap. Drops are counted and logged so the tradeoff is
visible, not silent.

Dependency
----------
    sounddevice  (PortAudio bindings) — sub-ms callback overhead; the dominant
                 latency term is blocksize/samplerate as above.

Audio format
------------
Emits mono float32 numpy arrays in [-1.0, 1.0] by default, shape (blocksize,).
float32 is what most streaming ASR frontends want; converting later would add a
copy. If a backend needs int16, convert at the ASR boundary, not here.
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - import-time environment issue
    sd = None
    _IMPORT_ERROR: Optional[Exception] = exc
else:
    _IMPORT_ERROR = None


logger = logging.getLogger("pipeline.capture")


@dataclass
class CaptureConfig:
    """Resolved settings for one capture session.

    These mirror the `capture:` block in config/config.yaml. Nothing here is
    hardcoded elsewhere — construct this from the loaded config.

    Attributes
    ----------
    samplerate : int
        Sampling rate in Hz. Should match what the ASR model expects (Vosk and
        most streaming models want 16000). Resampling later costs latency and a
        copy, so match the model here.
    blocksize : int
        Frames per chunk. Latency ≈ blocksize / samplerate. Primary latency
        knob for this stage.
    channels : int
        Input channel count. 1 (mono) is expected downstream; >1 will be
        downmixed to mono before enqueueing.
    device : int | str | None
        Input device index or name substring. None = system default input.
        For system loopback (capturing what you hear, e.g. a stream), select a
        loopback/monitor device here — see README.
    dtype : str
        Sample dtype delivered by PortAudio. "float32" avoids a later cast.
    queue_maxsize : int
        Max chunks buffered between capture and consumer. Small on purpose:
        bounds worst-case backlog latency to queue_maxsize * chunk_latency.
        When full, the oldest chunk is dropped.
    """

    samplerate: int = 16000
    blocksize: int = 1600
    channels: int = 1
    device: Optional[object] = None
    dtype: str = "float32"
    queue_maxsize: int = 8

    @property
    def chunk_latency_ms(self) -> float:
        """Approx latency added per chunk, in milliseconds."""
        return 1000.0 * self.blocksize / float(self.samplerate)


def list_input_devices() -> list[dict]:
    """Return available input-capable devices as a list of dicts.

    Each dict has: index, name, max_input_channels, default_samplerate.
    Use this to discover a device index/name to put in config (including
    loopback/monitor devices for capturing system audio).
    """
    _require_sounddevice()
    devices = sd.query_devices()
    out: list[dict] = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            out.append(
                {
                    "index": idx,
                    "name": dev.get("name", ""),
                    "max_input_channels": dev.get("max_input_channels", 0),
                    "default_samplerate": dev.get("default_samplerate", None),
                }
            )
    return out


class AudioCapture:
    """Streaming audio capture with a bounded, freshest-first chunk queue.

    Lifecycle:
        cap = AudioCapture(cfg)
        cap.start()                 # opens the stream, callback begins
        for chunk in cap.chunks():  # yields float32 mono arrays
            ...
        cap.stop()                  # closes the stream

    Or as a context manager:
        with AudioCapture(cfg) as cap:
            for chunk in cap.chunks():
                ...

    Thread model:
        - PortAudio calls `_callback` on its realtime thread; it only enqueues.
        - Your loop consuming `chunks()` runs on whatever thread you call it.
    """

    def __init__(self, config: Optional[CaptureConfig] = None) -> None:
        self.config = config or CaptureConfig()
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(
            maxsize=self.config.queue_maxsize
        )
        self._stream: Optional["sd.InputStream"] = None
        self._running = False
        # Observability counters (surfaced via stats()).
        self._dropped_chunks = 0
        self._overflow_events = 0
        self._captured_chunks = 0
        self._t_first_chunk: Optional[float] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Open the input stream and begin capturing.

        Idempotent-ish: raises if already running to surface logic errors.
        """
        _require_sounddevice()
        if self._running:
            raise RuntimeError("AudioCapture.start() called while already running")

        logger.info(
            "capture starting: samplerate=%d blocksize=%d (~%.1f ms/chunk) "
            "channels=%d device=%r queue_maxsize=%d",
            self.config.samplerate,
            self.config.blocksize,
            self.config.chunk_latency_ms,
            self.config.channels,
            self.config.device,
            self.config.queue_maxsize,
        )

        self._stream = sd.InputStream(
            samplerate=self.config.samplerate,
            blocksize=self.config.blocksize,
            channels=self.config.channels,
            device=self.config.device,
            dtype=self.config.dtype,
            callback=self._callback,
        )
        self._running = True
        self._stream.start()

    def stop(self) -> None:
        """Stop capturing and release the device. Safe to call more than once."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self._running = False
        logger.info(
            "capture stopped: captured=%d dropped=%d overflow_events=%d",
            self._captured_chunks,
            self._dropped_chunks,
            self._overflow_events,
        )

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # -- the realtime callback (keep this cheap!) --------------------------

    def _callback(self, indata, frames, time_info, status) -> None:
        """PortAudio realtime callback. Must be fast and non-blocking.

        Do not add logging of large objects, allocation-heavy work, or blocking
        calls here — it runs on the audio thread and stalling it drops audio.
        """
        if status:
            # status is truthy on xruns (input overflow, etc.).
            if getattr(status, "input_overflow", False):
                self._overflow_events += 1
            logger.warning("capture stream status: %s", status)

        # Downmix to mono if needed; produce a contiguous float32 (blocksize,).
        if indata.ndim == 2 and indata.shape[1] > 1:
            chunk = indata.mean(axis=1).astype(np.float32, copy=False)
        else:
            chunk = np.ascontiguousarray(indata).reshape(-1).astype(
                np.float32, copy=False
            )
        # Copy: PortAudio reuses `indata`'s buffer after the callback returns.
        chunk = chunk.copy()

        self._captured_chunks += 1
        if self._t_first_chunk is None:
            self._t_first_chunk = time.monotonic()

        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # Consumer is behind. Drop the OLDEST to keep audio fresh.
            try:
                self._queue.get_nowait()
                self._dropped_chunks += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                self._dropped_chunks += 1

    # -- consuming the stream ---------------------------------------------

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the next chunk, or None if none arrives within `timeout`.

        timeout=None blocks until a chunk is available.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def chunks(self, timeout: Optional[float] = 1.0) -> Iterator[np.ndarray]:
        """Yield chunks as they arrive, until stop() is called.

        Yields float32 mono arrays of shape (blocksize,). Between chunks it
        wakes every `timeout` seconds to re-check the running flag, so the loop
        exits promptly after stop().
        """
        while self._running or not self._queue.empty():
            chunk = self.read(timeout=timeout)
            if chunk is not None:
                yield chunk

    # -- observability -----------------------------------------------------

    def stats(self) -> dict:
        """Return capture counters for logging / latency debugging."""
        return {
            "captured_chunks": self._captured_chunks,
            "dropped_chunks": self._dropped_chunks,
            "overflow_events": self._overflow_events,
            "queue_depth": self._queue.qsize(),
            "chunk_latency_ms": round(self.config.chunk_latency_ms, 2),
        }

    @property
    def running(self) -> bool:
        return self._running


def _require_sounddevice() -> None:
    """Raise a clear error if sounddevice/PortAudio isn't available."""
    if sd is None:
        raise RuntimeError(
            "sounddevice is not available. Install it with "
            "`pip install sounddevice` (needs PortAudio; on Debian/Ubuntu "
            "also `apt install libportaudio2`). "
            f"Original import error: {_IMPORT_ERROR!r}"
        )
