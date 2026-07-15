"""
asr/base.py — Stage 2 interface: streaming ASR.

Defines the contract every ASR backend (GPU or CPU) must implement, plus the
result type they emit. Concrete backends live alongside this file
(`gpu_backend.py`, `cpu_backend.py`) and are selected via config — nothing in
the pipeline should import a specific backend directly; go through the factory
/ config so the GPU-vs-CPU choice stays a config switch.

Where this sits
---------------
    [capture] --feed(chunk)--> (Stage 2: ASR) --TranscriptResult--> [buffer] --> [matcher] --> [playback]

Design decisions (latency-first)
--------------------------------
1. `feed()` must return quickly and NOT block on utterance completion.
   The capture consumer loop calls `feed(chunk)` once per audio chunk
   (~blocksize/samplerate ms apart). If `feed()` blocked until an utterance
   ended, capture's bounded queue would back up and drop audio. So `feed()`
   returns the transcript results produced *so far* for the audio seen to date
   — often `[]` for a given chunk, sometimes one or more partials, occasionally
   a final. Heavy model work should happen inline only if it fits the real-time
   budget; otherwise a backend may run the model on its own thread and have
   `feed()` return whatever is ready. Either way, `feed()` returns fast.

2. Partial vs final is explicit (`TranscriptResult.is_final`).
   Streaming ASR emits unstable in-progress hypotheses (partials) that can
   change as more audio arrives, then a stable final when the utterance
   settles. The matcher downstream needs this distinction: matching on partials
   is lower-latency but can misfire on words that later change; matching only
   on finals is safe but slower. Encoding `is_final` on every result lets Stage
   4 make that tradeoff per keyword rather than us deciding here.

3. Every result carries a monotonic `timestamp`.
   Set at the moment the backend produces the result, using
   `time.monotonic()`. This lets the telemetry stage measure the
   capture→ASR-partial boundary without guessing.

Audio input contract (must match Stage 1 / capture)
----------------------------------------------------
Capture emits mono float32 numpy arrays in [-1.0, 1.0], shape (blocksize,), at
`samplerate` Hz. That is the canonical input to `feed()`. Some engines (e.g.
Vosk) want int16 PCM bytes instead — converting is the *backend's*
responsibility, done at this boundary, not in capture. `SAMPLE_DTYPE` and the
`float32_to_int16_bytes()` helper below exist for exactly that.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# Canonical audio format delivered by the capture stage. Backends convert from
# this if their engine needs something else.
SAMPLE_DTYPE = "float32"


@dataclass
class TranscriptResult:
    """One transcript emission from an ASR backend.

    Attributes
    ----------
    text : str
        The transcript text. For a partial, this is the current best (unstable)
        hypothesis for the in-progress utterance. For a final, it is the settled
        text for the completed utterance/segment.
    is_final : bool
        False = partial (may still change), True = final (stable). The matcher
        uses this to decide whether it is safe to trigger on this text.
    timestamp : float
        Monotonic time (time.monotonic()) when the backend produced this result.
        Used by telemetry to measure stage latency. Defaults to now.
    confidence : float | None
        Optional model confidence in [0, 1] if the backend provides one; None if
        not available. Never assume presence.
    """

    text: str
    is_final: bool
    timestamp: float = field(default_factory=time.monotonic)
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        # Normalize once so downstream stages don't each re-strip.
        self.text = self.text.strip()


class StreamingASR(ABC):
    """Abstract base for a streaming ASR backend.

    Contract for implementors:
      - `feed(chunk)` accepts one mono float32 array (shape (blocksize,)) and
        returns a list of TranscriptResult produced so far. Return `[]` when the
        chunk produced no new emission. MUST return promptly (see module docs).
      - `flush()` signals end-of-stream: return any remaining/forced-final
        results (e.g. finalize the current partial). Called once when capture
        stops.
      - `reset()` clears internal decoder state to start a fresh utterance
        stream without recreating the (expensive) model.
      - Model construction/loading happens in the concrete __init__, so the
        first `feed()` is not paying the load cost. State the model's rough
        latency contribution in the backend's docstring.

    Subclasses should set `self.samplerate` so callers can assert it matches the
    capture samplerate (mismatched rates mean silent resampling or garbage).
    """

    #: Sampling rate (Hz) the backend expects. Concrete backends must set this.
    samplerate: int = 16000

    @abstractmethod
    def feed(self, chunk: np.ndarray) -> List[TranscriptResult]:
        """Feed one audio chunk; return transcript results produced so far.

        Parameters
        ----------
        chunk : np.ndarray
            Mono float32, shape (blocksize,), values in [-1.0, 1.0], at
            `self.samplerate` Hz — i.e. exactly what the capture stage yields.

        Returns
        -------
        list[TranscriptResult]
            Zero or more results. Empty list is normal and expected for many
            chunks. Must return without blocking on utterance completion.
        """
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> List[TranscriptResult]:
        """End-of-stream: return any final results still pending.

        Called once when the audio stream stops. Implementations should finalize
        the in-progress utterance (if any) and return it as a final result.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset decoder state for a fresh stream, keeping the loaded model.

        Default is a no-op; override if the engine holds streaming state.
        """
        return None

    # -- shared helpers ----------------------------------------------------

    def _check_samplerate(self, expected: int) -> None:
        """Raise if the configured samplerate doesn't match this backend.

        Call from a backend __init__ with the capture samplerate to fail loudly
        instead of silently mis-decoding.
        """
        if int(expected) != int(self.samplerate):
            raise ValueError(
                f"samplerate mismatch: capture={expected} Hz but "
                f"{type(self).__name__} expects {self.samplerate} Hz. "
                "Match capture.samplerate to the model, or resample explicitly."
            )

    def __enter__(self) -> "StreamingASR":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Give backends a chance to release model/GPU resources.
        self.close()

    def close(self) -> None:
        """Release backend resources (model, GPU memory). Default no-op."""
        return None


def float32_to_int16_bytes(chunk: np.ndarray) -> bytes:
    """Convert a mono float32 [-1,1] array to little-endian int16 PCM bytes.

    Helper for engines (e.g. Vosk) that take int16 PCM rather than float32. This
    conversion is the backend's responsibility at the ASR boundary — capture
    always emits float32. Cost is a single vectorized pass over the chunk
    (sub-millisecond for typical block sizes); negligible vs. model inference.
    """
    if chunk.dtype != np.float32:
        chunk = chunk.astype(np.float32, copy=False)
    clipped = np.clip(chunk, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()
