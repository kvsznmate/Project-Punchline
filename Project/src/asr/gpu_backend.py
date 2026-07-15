"""
asr/gpu_backend.py — Stage 2 GPU backend: faster-whisper (sliding window).

Whisper is highly accurate but is NOT a natively streaming model: it is built to
transcribe a complete audio segment at once, not to emit words incrementally as
audio arrives. To use it in a latency-first live pipeline we wrap it in a
sliding-window scheme and run it off the capture thread.

How streaming is faked (and the latency trade-off)
--------------------------------------------------
- We keep a rolling window of the most recent `window_seconds` of audio.
- A background worker re-transcribes that window every ~`step_seconds` and
  publishes the text as a PARTIAL result. Because the window keeps sliding,
  Whisper may revise earlier words as more context arrives — so partials here
  are genuinely unstable, and matching on them can occasionally misfire. That
  is the documented cost of getting Whisper to react quickly.
- When we detect a pause (a run of low-energy audio), the current window is
  transcribed once more and emitted as a FINAL, then the window is cleared so
  the next utterance starts clean.

Why a background thread
-----------------------
Whisper inference on a window takes far longer than one audio chunk, so running
it inside feed() would stall the capture consumer loop and back up capture's
bounded queue (dropping audio). Instead, feed() only appends the chunk to the
rolling window (cheap) and returns whatever results the worker has already
published. Inference happens on its own thread. This keeps feed() non-blocking,
per the StreamingASR contract.

Net latency: a spoken word surfaces in a partial within roughly one `step`
interval plus one inference duration, rather than at end-of-utterance. Tune
`step_seconds` down for lower latency (more frequent, more expensive inference)
or up to reduce GPU load.

NOTE — config gap: window_seconds / step_seconds / silence thresholds are the
GPU latency knobs but are NOT in asr.gpu config yet (which only has model,
compute_type, language, beam_size). They are exposed as constructor args with
defaults here and should be promoted into config.yaml later.

Dependency
----------
    faster-whisper (pip install faster-whisper). Runs on GPU with CUDA when
    device="cuda"; compute_type e.g. float16/int8_float16 trades accuracy for
    speed/VRAM. Model load happens in __init__.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from typing import List, Optional

import numpy as np

from .base import StreamingASR, TranscriptResult

logger = logging.getLogger("pipeline.asr.gpu")


def _register_cuda_dll_dirs() -> None:
    """Ensure the pip-installed NVIDIA CUDA libs are loadable on Windows.

    Delegates to config_loader.ensure_cuda_on_path(), which prepends the
    nvidia\\*\\bin folders to PATH (the mechanism CTranslate2's compiled
    extension actually honors) AND registers them via os.add_dll_directory().
    Merely calling os.add_dll_directory() is NOT enough on its own: CTranslate2
    loads CUDA through the OS loader, which uses PATH. No-op off Windows or when
    the CUDA wheels aren't installed.

    Kept as a thin wrapper (rather than inlining) so the PATH logic lives in one
    place; falls back silently if config_loader can't be imported standalone.
    """
    if sys.platform != "win32":
        return
    try:
        from config_loader import ensure_cuda_on_path  # type: ignore
        ensure_cuda_on_path()
    except Exception as exc:  # pragma: no cover - standalone/import edge case
        logger.debug("ensure_cuda_on_path unavailable (%s); CUDA libs may need PATH set manually", exc)


class WhisperASR(StreamingASR):
    """Streaming-ish GPU ASR backed by faster-whisper + a sliding window.

    Parameters
    ----------
    model : str
        Whisper model size/name (asr.gpu.model), e.g. "small".
    samplerate : int
        Capture samplerate; must match capture (16000 typical; Whisper wants 16k).
    emit_partials : bool
        If False, only finals (on detected pauses) are emitted.
    compute_type : str
        faster-whisper compute type (asr.gpu.compute_type), e.g. "float16".
    language : str
        Forced decode language (asr.gpu.language), e.g. "en".
    beam_size : int
        Decoding beam (asr.gpu.beam_size). 1 = greedy = fastest.
    device : str
        "cuda" for GPU (default) or "cpu". The factory passes "cuda" for the GPU
        backend; kept configurable so the class isn't hard-wired to CUDA.
    window_seconds : float
        Length of the rolling audio window re-transcribed each step.
    step_seconds : float
        How often the worker re-transcribes the window (latency knob).
    silence_rms : float
        RMS below this counts as silence for pause detection.
    silence_seconds : float
        Contiguous silence needed to finalize the current utterance.
    """

    def __init__(
        self,
        model: str = "small",
        samplerate: int = 16000,
        emit_partials: bool = True,
        compute_type: str = "float16",
        language: str = "en",
        beam_size: int = 1,
        device: str = "cuda",
        window_seconds: float = 4.0,
        step_seconds: float = 0.5,
        silence_rms: float = 0.010,
        silence_seconds: float = 0.6,
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "faster-whisper is not installed. Install it with "
                "`pip install faster-whisper`. Original import error: "
                f"{exc!r}"
            ) from exc

        self.samplerate = int(samplerate)
        self.emit_partials = bool(emit_partials)
        self._language = language
        self._beam_size = int(beam_size)
        self._window_samples = int(window_seconds * self.samplerate)
        self._step_seconds = float(step_seconds)
        self._silence_rms = float(silence_rms)
        self._silence_samples = int(silence_seconds * self.samplerate)

        # Model load is the expensive step; do it in __init__ so the first
        # feed() isn't paying for it.
        # On Windows, make the pip-installed CUDA libs discoverable first, or
        # CTranslate2 can't find cublas64_12.dll / cudnn64_9.dll on the GPU path.
        if device == "cuda":
            _register_cuda_dll_dirs()
        logger.info(
            "loading faster-whisper model=%s device=%s compute_type=%s",
            model, device, compute_type,
        )
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

        # Rolling window + synchronization.
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._pending: List[TranscriptResult] = []       # results worker published
        self._silence_run = 0                              # contiguous silent samples
        self._want_final = False                           # pause detected -> finalize
        self._stop = threading.Event()
        self._last_partial = ""

        # Background inference worker.
        self._worker = threading.Thread(target=self._run, name="whisper-asr", daemon=True)
        self._worker.start()
        logger.info(
            "WhisperASR ready: samplerate=%d emit_partials=%s window=%.1fs step=%.2fs",
            self.samplerate, self.emit_partials, window_seconds, step_seconds,
        )

    # -- StreamingASR contract --------------------------------------------

    def feed(self, chunk: np.ndarray) -> List[TranscriptResult]:
        """Append chunk to the rolling window; return any published results.

        Non-blocking: no inference happens here. Pause detection (cheap RMS) is
        done inline so the worker knows when to emit a final.
        """
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)

        # Cheap silence tracking for utterance boundary detection.
        rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        with self._lock:
            self._buf = np.concatenate([self._buf, chunk])
            # Trim to the window length to bound memory and inference cost.
            if self._buf.size > self._window_samples:
                self._buf = self._buf[-self._window_samples:]

            if rms < self._silence_rms:
                self._silence_run += chunk.size
                if self._silence_run >= self._silence_samples:
                    self._want_final = True
            else:
                self._silence_run = 0

            out = self._pending
            self._pending = []
        return out

    def flush(self) -> List[TranscriptResult]:
        """End-of-stream: transcribe whatever remains as a final, synchronously."""
        with self._lock:
            audio = self._buf.copy()
            self._buf = np.zeros(0, dtype=np.float32)
            self._want_final = False
            self._silence_run = 0
        results: List[TranscriptResult] = []
        text = self._transcribe(audio)
        if text:
            results.append(TranscriptResult(text=text, is_final=True))
        return results

    def reset(self) -> None:
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
            self._pending = []
            self._silence_run = 0
            self._want_final = False
            self._last_partial = ""

    def close(self) -> None:
        """Stop the worker and release the model."""
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._model = None

    # -- background worker -------------------------------------------------

    def _run(self) -> None:
        """Periodically transcribe the window; publish partials and finals."""
        while not self._stop.is_set():
            time.sleep(self._step_seconds)
            with self._lock:
                audio = self._buf.copy()
                want_final = self._want_final
                self._want_final = False
                if want_final:
                    # Consume the window on finalize so the next utterance is clean.
                    self._buf = np.zeros(0, dtype=np.float32)

            if audio.size == 0:
                continue

            text = self._transcribe(audio)
            if not text:
                continue

            if want_final:
                result = TranscriptResult(text=text, is_final=True)
                self._last_partial = ""
                with self._lock:
                    self._pending.append(result)
            elif self.emit_partials and text != self._last_partial:
                self._last_partial = text
                result = TranscriptResult(text=text, is_final=False)
                with self._lock:
                    self._pending.append(result)

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run the model on an audio window and return concatenated text."""
        if audio.size == 0 or self._model is None:
            return ""
        try:
            segments, _info = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=self._beam_size,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:  # pragma: no cover - runtime/inference issues
            logger.warning("whisper transcribe failed: %s", exc)
            return ""
