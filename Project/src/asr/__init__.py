"""
asr package — Stage 2 (streaming ASR).

Exposes the interface and result type from `base`, both concrete backends, and a
`build_asr()` factory that constructs the right backend from a loaded
PipelineConfig. The config loader intentionally does NOT import the backends; it
only resolves *which* device to use. This factory is where that decision becomes
a concrete object, so the GPU/CPU modules stay decoupled from config parsing.

Typical use::

    from config_loader import load_config
    from asr import build_asr

    cfg = load_config("config/config.yaml")
    asr = build_asr(cfg)            # picks Vosk or Whisper per cfg.resolved_device
    for chunk in capture.chunks():
        for result in asr.feed(chunk):
            ...
"""

from __future__ import annotations

import logging

from .base import (
    StreamingASR,
    TranscriptResult,
    SAMPLE_DTYPE,
    float32_to_int16_bytes,
)

logger = logging.getLogger("pipeline.asr")

__all__ = [
    "StreamingASR",
    "TranscriptResult",
    "SAMPLE_DTYPE",
    "float32_to_int16_bytes",
    "build_asr",
]


def build_asr(config) -> StreamingASR:
    """Construct the ASR backend for a loaded PipelineConfig.

    Selects on `config.resolved_device` (already resolved from device: auto by
    the loader). Backends are imported lazily inside each branch so that, e.g.,
    a CPU-only machine never needs faster-whisper installed, and vice versa.
    """
    device = getattr(config, "resolved_device", "cpu")
    asr_cfg = config.asr
    samplerate = config.capture.samplerate

    if device == "gpu":
        from .gpu_backend import WhisperASR
        g = asr_cfg.gpu
        logger.info("build_asr: constructing GPU backend (faster-whisper) on device=%s", g.device)
        return WhisperASR(
            model=g.model,
            samplerate=samplerate,
            emit_partials=asr_cfg.emit_partials,
            compute_type=g.compute_type,
            language=g.language,
            beam_size=g.beam_size,
            device=g.device,
        )

    if device == "cpu":
        c = asr_cfg.cpu
        if c.engine != "vosk":
            raise ValueError(
                f"unsupported CPU ASR engine {c.engine!r}; only 'vosk' is "
                "implemented. Update asr.cpu.engine in config.yaml."
            )
        from .cpu_backend import VoskASR
        logger.info("build_asr: constructing CPU backend (Vosk)")
        return VoskASR(
            model_path=c.model_path,
            samplerate=samplerate,
            emit_partials=asr_cfg.emit_partials,
        )

    raise ValueError(f"unknown resolved_device {device!r}; expected 'gpu' or 'cpu'")
