"""
asr/cpu_backend.py — Stage 2 CPU backend: Vosk streaming ASR.

The real-time CPU fallback. Vosk is a lightweight, natively-streaming recognizer
(Kaldi under the hood): you feed it PCM incrementally and it returns partial
hypotheses as words arrive, plus a final result when it detects an utterance
boundary. That streaming nature is exactly what a latency-first pipeline wants,
which is why it is the CPU engine of choice here.

Latency contribution (rough order of magnitude)
------------------------------------------------
- Per-chunk decode with the small English model runs comfortably faster than
  real time on a modern CPU core (the decode of one ~100 ms chunk takes well
  under 100 ms), so the backend keeps up with the live stream.
- End-to-end added latency at this stage is therefore ~one chunk
  (blocksize/samplerate, e.g. ~100 ms) plus Vosk's decode time for that chunk.
  A partial for a just-spoken word typically appears within one or two chunks.
- The float32 -> int16 conversion is a sub-millisecond vectorized pass and is
  negligible next to decode.

Trade-off vs. the GPU backend: lower accuracy than Whisper, but far lower and
more predictable latency on CPU and no GPU dependency. For triggering short
keywords this is usually the right trade.

Dependency
----------
    vosk  (pip install vosk) + an acoustic model directory on disk. The model
    path comes from config (asr.cpu.model_path). Download a model separately,
    e.g. a "vosk-model-small-en-us" build for real-time CPU use.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import numpy as np

from .base import StreamingASR, TranscriptResult, float32_to_int16_bytes

logger = logging.getLogger("pipeline.asr.cpu")


class VoskASR(StreamingASR):
    """Streaming CPU ASR backed by Vosk.

    Parameters
    ----------
    model_path : str
        Filesystem path to a Vosk model directory (asr.cpu.model_path).
    samplerate : int
        Capture samplerate; must match what the model expects (16000 typical).
    emit_partials : bool
        If False, only final results are emitted (partials are suppressed).
        Mirrors asr.emit_partials — disabling raises latency to utterance end.
    """

    def __init__(
        self,
        model_path: str,
        samplerate: int = 16000,
        emit_partials: bool = True,
    ) -> None:
        # Lazy import so importing this module (e.g. for the factory) doesn't
        # hard-require vosk unless the CPU backend is actually constructed.
        try:
            from vosk import Model, KaldiRecognizer  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "vosk is not installed. Install it with `pip install vosk` and "
                "download a model, then set asr.cpu.model_path in config.yaml. "
                f"Original import error: {exc!r}"
            ) from exc

        self.samplerate = int(samplerate)
        self.emit_partials = bool(emit_partials)
        self._model_path = model_path

        import os
        if not os.path.isdir(model_path):
            raise RuntimeError(
                f"Vosk model directory not found: {model_path!r}. Download a "
                "model and point asr.cpu.model_path at it."
            )

        # Model load is the expensive step; do it here in __init__ so the first
        # feed() call isn't paying for it.
        logger.info("loading Vosk model from %s", model_path)
        self._Model = Model
        self._KaldiRecognizer = KaldiRecognizer
        self._model = Model(model_path)
        self._rec = KaldiRecognizer(self._model, self.samplerate)
        self._rec.SetWords(True)
        # Track the last partial text so we don't emit the same partial twice.
        self._last_partial = ""
        logger.info(
            "VoskASR ready: samplerate=%d emit_partials=%s",
            self.samplerate, self.emit_partials,
        )

    # -- StreamingASR contract --------------------------------------------

    def feed(self, chunk: np.ndarray) -> List[TranscriptResult]:
        """Feed one mono float32 chunk; return results produced so far."""
        pcm = float32_to_int16_bytes(chunk)
        results: List[TranscriptResult] = []

        if self._rec.AcceptWaveform(pcm):
            # An utterance boundary was reached -> a final result is ready.
            text = self._extract(self._rec.Result(), key="text")
            self._last_partial = ""
            if text:
                results.append(TranscriptResult(text=text, is_final=True))
        else:
            # Still mid-utterance -> a (possibly updated) partial is available.
            if self.emit_partials:
                partial = self._extract(self._rec.PartialResult(), key="partial")
                if partial and partial != self._last_partial:
                    self._last_partial = partial
                    results.append(TranscriptResult(text=partial, is_final=False))
        return results

    def flush(self) -> List[TranscriptResult]:
        """End-of-stream: force-finalize whatever is buffered in the recognizer."""
        results: List[TranscriptResult] = []
        text = self._extract(self._rec.FinalResult(), key="text")
        self._last_partial = ""
        if text:
            results.append(TranscriptResult(text=text, is_final=True))
        return results

    def reset(self) -> None:
        """Start a fresh recognizer without reloading the (expensive) model."""
        self._rec = self._KaldiRecognizer(self._model, self.samplerate)
        self._rec.SetWords(True)
        self._last_partial = ""

    def close(self) -> None:
        """Release recognizer/model references."""
        self._rec = None
        self._model = None

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _extract(raw_json: str, key: str) -> str:
        """Pull a text field out of a Vosk JSON result string, safely."""
        try:
            obj = json.loads(raw_json)
        except (ValueError, TypeError):
            return ""
        return str(obj.get(key, "")).strip()
