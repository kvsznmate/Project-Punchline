"""
config_loader.py — load, validate, and resolve the pipeline config.

Reads config/config.yaml (the single source of truth) into typed config objects
and resolves the `device: auto` choice into a concrete gpu/cpu decision. Every
stage should obtain its settings from here rather than reading the YAML or
hardcoding constants.

Location note: this module lives under src/ next to the `capture` package it
imports; config/ holds only config.yaml (data), not Python. Load the pipeline
config with, e.g.:

    from config_loader import load_config
    cfg = load_config("config/config.yaml")

Why typed objects instead of passing the raw dict around
--------------------------------------------------------
A bad or missing key should fail *at startup* with a clear message, not as a
KeyError deep in the hot path while live audio is flowing. Parsing into
dataclasses (and reusing capture's own CaptureConfig) gives one validation point
and lets the rest of the code rely on attribute access with known types.

Device resolution (`device: auto`)
-----------------------------------
The GPU-vs-CPU choice is a config switch, never hardcoded. `resolve_device()`:
  - "gpu" -> use GPU (error later, in the backend, if it isn't actually usable)
  - "cpu" -> use CPU
  - "auto" -> use GPU if a usable one is detected, else fall back to CPU
The resolved value is stored as `PipelineConfig.resolved_device` next to the
requested `device`, and the decision (plus the reason) is logged. This loader
does NOT import the ASR backends — it only decides *which* to use; constructing
the backend is the factory's job, so the GPU/CPU modules stay decoupled.

Latency note
------------
Loading/validation is one-time startup cost (milliseconds) and adds nothing to
the per-chunk hot path. GPU detection may briefly touch the CUDA runtime once at
startup; it is not on the audio path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml

# Reuse capture's own config object so the capture block cannot drift from it.
from capture import CaptureConfig  # type: ignore


logger = logging.getLogger("pipeline.config")


VALID_DEVICES = ("auto", "gpu", "cpu")


# --------------------------------------------------------------------------- #
# Typed sub-configs
# --------------------------------------------------------------------------- #

@dataclass
class GpuAsrConfig:
    model: str = "small"
    compute_type: str = "float16"
    language: str = "en"
    beam_size: int = 1
    device: str = "cuda"   # compute device for Whisper: "cuda" (GPU) or "cpu".
                           # Note: this is separate from the top-level `device`,
                           # which selects the backend (Whisper vs Vosk).


@dataclass
class CpuAsrConfig:
    engine: str = "vosk"
    model_path: str = "models/vosk-model-small-en-us"


@dataclass
class AsrConfig:
    emit_partials: bool = True
    gpu: GpuAsrConfig = field(default_factory=GpuAsrConfig)
    cpu: CpuAsrConfig = field(default_factory=CpuAsrConfig)


@dataclass
class BufferConfig:
    max_chars: int = 240
    include_partials: bool = True


@dataclass
class KeywordRule:
    pattern: str
    sound: Union[str, List[str]]  # one filename, or a list to pick from at random
    cooldown: Optional[float] = None  # None -> use matcher default_cooldown_seconds


@dataclass
class MatcherConfig:
    match_on_partials: bool = True
    case_insensitive: bool = True
    default_cooldown_seconds: float = 3.0
    keywords: List[KeywordRule] = field(default_factory=list)


@dataclass
class PlaybackConfig:
    sounds_dir: str = "assets/sounds"
    output_device: Optional[object] = None
    preload: bool = True
    max_polyphony: int = 4
    volume: float = 1.0


@dataclass
class TelemetryConfig:
    enabled: bool = True
    log_dir: str = "logs"
    log_stage_latency: bool = True
    log_every_n_chunks: int = 50
    level: str = "INFO"


@dataclass
class PipelineConfig:
    """Top-level resolved config for the whole pipeline."""

    device: str = "auto"            # as requested in the YAML
    resolved_device: str = "cpu"    # concrete gpu/cpu after resolution
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)


# --------------------------------------------------------------------------- #
# Device resolution
# --------------------------------------------------------------------------- #

def ensure_cuda_on_path() -> list:
    """On Windows, prepend the pip-installed NVIDIA CUDA lib folders to PATH.

    This is the fix that actually makes CTranslate2 (faster-whisper's engine)
    find cuBLAS/cuDNN. Registering the dirs via os.add_dll_directory() is NOT
    sufficient: that only affects DLLs loaded directly by Python, whereas
    CTranslate2's compiled extension loads CUDA through the OS loader, which
    honors PATH. So we put the folders on PATH here, before any CUDA detection
    or model load happens.

    Discovers folders from the installed `nvidia` package (not a hardcoded
    path). No-op off Windows or if the packages aren't installed. Idempotent —
    won't add a folder that's already on PATH. Returns the folders it added.
    """
    import os, sys
    if sys.platform != "win32":
        return []
    try:
        import nvidia
        nv_paths = list(getattr(nvidia, "__path__", []))
    except Exception:
        nv_paths = []
    dirs = []
    for base in nv_paths:
        if not os.path.isdir(base):
            continue
        for pkg in os.listdir(base):
            pkgdir = os.path.join(base, pkg)
            if not os.path.isdir(pkgdir):
                continue
            bindir = os.path.join(pkgdir, "bin")
            dirs.append(bindir if os.path.isdir(bindir) else pkgdir)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep)
    added = []
    for d in dirs:
        if d not in parts:
            added.append(d)
    if added:
        os.environ["PATH"] = os.pathsep.join(added + [current])
        # Also register for Python's own DLL loading (belt and suspenders).
        add = getattr(os, "add_dll_directory", None)
        if add is not None:
            for d in added:
                try:
                    add(d)
                except OSError:
                    pass
        logger.info("added %d CUDA dir(s) to PATH: %s", len(added), added)
    return added


def gpu_is_available() -> bool:
    """Best-effort detection of a usable CUDA GPU for ASR, via CTranslate2.

    We query CTranslate2 (the engine faster-whisper actually runs on) rather
    than torch: torch may be absent or a CPU-only build, which would wrongly
    report no GPU even when CTranslate2 can use CUDA. Ensures the CUDA libs are
    on PATH first so the count is accurate. Runs once at startup, not on the
    audio hot path. Any failure conservatively reports no GPU (CPU always works).
    """
    ensure_cuda_on_path()
    try:
        import ctranslate2  # type: ignore
    except Exception:
        logger.info("gpu detection: ctranslate2 not importable -> assuming no GPU")
        return False
    try:
        count = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:  # pragma: no cover - runtime/driver oddities
        logger.warning("gpu detection: ctranslate2 CUDA check failed (%s) -> no GPU", exc)
        return False
    logger.info("gpu detection: ctranslate2 CUDA device count = %d", count)
    return count > 0


def resolve_device(requested: str) -> str:
    """Resolve a requested device ('auto'|'gpu'|'cpu') to concrete 'gpu'|'cpu'.

    Logs the decision and the reason. 'auto' prefers GPU and falls back to CPU.
    """
    req = str(requested).strip().lower()
    if req not in VALID_DEVICES:
        raise ValueError(
            f"config 'device' must be one of {VALID_DEVICES}, got {requested!r}"
        )

    if req == "cpu":
        logger.info("device resolved: cpu (requested explicitly)")
        return "cpu"

    if req == "gpu":
        # Honor the explicit request; the backend will hard-error at load if the
        # GPU truly isn't usable. We still note detection for visibility.
        if not gpu_is_available():
            logger.warning(
                "device resolved: gpu (requested explicitly) but no usable GPU "
                "was detected — the GPU backend may fail to initialize."
            )
        else:
            logger.info("device resolved: gpu (requested explicitly, GPU detected)")
        return "gpu"

    # auto
    if gpu_is_available():
        logger.info("device resolved: gpu (auto: GPU detected)")
        return "gpu"
    logger.info("device resolved: cpu (auto: no GPU detected, falling back)")
    return "cpu"


# --------------------------------------------------------------------------- #
# Loading + validation
# --------------------------------------------------------------------------- #

def _require_mapping(value, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config section '{name}' must be a mapping, got {type(value).__name__}")
    return value


def _parse_keywords(raw_list) -> List[KeywordRule]:
    if raw_list is None:
        return []
    if not isinstance(raw_list, list):
        raise ValueError("matcher.keywords must be a list")
    rules: List[KeywordRule] = []
    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            raise ValueError(f"matcher.keywords[{i}] must be a mapping")
        if "pattern" not in item or "sound" not in item:
            raise ValueError(
                f"matcher.keywords[{i}] needs both 'pattern' and 'sound' "
                f"(got keys {sorted(item.keys())})"
            )
        pattern = str(item["pattern"])
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"matcher.keywords[{i}] pattern {pattern!r} is not valid regex: {exc}"
            ) from exc
        cd = item.get("cooldown", None)
        if cd is not None:
            cd = float(cd)
            if cd < 0:
                raise ValueError(f"matcher.keywords[{i}] cooldown must be >= 0")
        # `sound` may be a single filename (str) or a list of filenames to pick
        # from at random when the keyword fires. Validate both forms here so a
        # malformed entry fails at startup, not mid-stream.
        raw_sound = item["sound"]
        if isinstance(raw_sound, (list, tuple)):
            if len(raw_sound) == 0:
                raise ValueError(
                    f"matcher.keywords[{i}] 'sound' list must not be empty"
                )
            sound: Union[str, List[str]] = []
            for j, s in enumerate(raw_sound):
                if isinstance(s, (list, tuple, dict)):
                    raise ValueError(
                        f"matcher.keywords[{i}] 'sound'[{j}] must be a filename "
                        f"string, got {type(s).__name__}"
                    )
                sound.append(str(s))
        else:
            sound = str(raw_sound)
        rules.append(KeywordRule(pattern=pattern, sound=sound, cooldown=cd))
    return rules


def load_config(path: str | Path) -> PipelineConfig:
    """Load, validate, and resolve the pipeline config from a YAML file.

    Raises FileNotFoundError if the file is missing and ValueError on any
    invalid/malformed setting, so problems surface at startup.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")

    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError("top-level config must be a mapping")

    # --- capture: reuse CaptureConfig, validated by construction ---
    cap_raw = _require_mapping(raw.get("capture"), "capture")
    known_cap = {f for f in CaptureConfig.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(cap_raw) - known_cap
    if unknown:
        raise ValueError(f"unknown key(s) in capture: {sorted(unknown)}")
    capture = CaptureConfig(**cap_raw)
    if capture.samplerate <= 0:
        raise ValueError("capture.samplerate must be > 0")
    if capture.blocksize <= 0:
        raise ValueError("capture.blocksize must be > 0")
    if capture.channels < 1:
        raise ValueError("capture.channels must be >= 1")
    if capture.queue_maxsize < 1:
        raise ValueError("capture.queue_maxsize must be >= 1")

    # --- asr ---
    asr_raw = _require_mapping(raw.get("asr"), "asr")
    gpu_raw = _require_mapping(asr_raw.get("gpu"), "asr.gpu")
    cpu_raw = _require_mapping(asr_raw.get("cpu"), "asr.cpu")
    asr = AsrConfig(
        emit_partials=bool(asr_raw.get("emit_partials", True)),
        gpu=GpuAsrConfig(
            model=str(gpu_raw.get("model", "small")),
            compute_type=str(gpu_raw.get("compute_type", "float16")),
            language=str(gpu_raw.get("language", "en")),
            beam_size=int(gpu_raw.get("beam_size", 1)),
            device=str(gpu_raw.get("device", "cuda")),
        ),
        cpu=CpuAsrConfig(
            engine=str(cpu_raw.get("engine", "vosk")),
            model_path=str(cpu_raw.get("model_path", "models/vosk-model-small-en-us")),
        ),
    )
    if asr.gpu.beam_size < 1:
        raise ValueError("asr.gpu.beam_size must be >= 1")

    # --- buffer ---
    buf_raw = _require_mapping(raw.get("buffer"), "buffer")
    buffer = BufferConfig(
        max_chars=int(buf_raw.get("max_chars", 240)),
        include_partials=bool(buf_raw.get("include_partials", True)),
    )
    if buffer.max_chars < 1:
        raise ValueError("buffer.max_chars must be >= 1")

    # --- matcher ---
    m_raw = _require_mapping(raw.get("matcher"), "matcher")
    default_cd = float(m_raw.get("default_cooldown_seconds", 3.0))
    if default_cd < 0:
        raise ValueError("matcher.default_cooldown_seconds must be >= 0")
    matcher = MatcherConfig(
        match_on_partials=bool(m_raw.get("match_on_partials", True)),
        case_insensitive=bool(m_raw.get("case_insensitive", True)),
        default_cooldown_seconds=default_cd,
        keywords=_parse_keywords(m_raw.get("keywords")),
    )

    # --- playback ---
    pb_raw = _require_mapping(raw.get("playback"), "playback")
    playback = PlaybackConfig(
        sounds_dir=str(pb_raw.get("sounds_dir", "assets/sounds")),
        output_device=pb_raw.get("output_device", None),
        preload=bool(pb_raw.get("preload", True)),
        max_polyphony=int(pb_raw.get("max_polyphony", 4)),
        volume=float(pb_raw.get("volume", 1.0)),
    )
    if playback.max_polyphony < 1:
        raise ValueError("playback.max_polyphony must be >= 1")
    if not (0.0 <= playback.volume <= 1.0):
        raise ValueError("playback.volume must be in [0.0, 1.0]")

    # --- telemetry ---
    tel_raw = _require_mapping(raw.get("telemetry"), "telemetry")
    telemetry = TelemetryConfig(
        enabled=bool(tel_raw.get("enabled", True)),
        log_dir=str(tel_raw.get("log_dir", "logs")),
        log_stage_latency=bool(tel_raw.get("log_stage_latency", True)),
        log_every_n_chunks=int(tel_raw.get("log_every_n_chunks", 50)),
        level=str(tel_raw.get("level", "INFO")).upper(),
    )
    if telemetry.log_every_n_chunks < 1:
        raise ValueError("telemetry.log_every_n_chunks must be >= 1")

    # --- device resolution ---
    requested_device = str(raw.get("device", "auto"))
    resolved = resolve_device(requested_device)

    # Coordinate the two device switches so `auto` behaves sensibly.
    #
    # Background: the top-level `device` picks the BACKEND (gpu=Whisper,
    # cpu=Vosk), while asr.gpu.device picks Whisper's compute device
    # (cuda|cpu). Naively, `auto` with no GPU would resolve to the Vosk
    # backend — but Vosk needs a downloaded model and has no Hungarian model,
    # so that's the wrong fallback for a Whisper user. Instead we treat `auto`
    # as Whisper-oriented: always use the Whisper backend, and let GPU
    # detection choose cuda vs cpu.
    #
    # Explicit `device: gpu` or `device: cpu` is always honored as-is (gpu ->
    # Whisper, cpu -> Vosk). Explicit asr.gpu.device is never overridden.
    gpu_device_explicit = "device" in gpu_raw
    if requested_device.strip().lower() == "auto":
        # Force the Whisper backend for auto; pick its compute device by
        # detection (gpu_is_available already ran inside resolve_device).
        resolved = "gpu"
        if not gpu_device_explicit:
            asr.gpu.device = "cuda" if gpu_is_available() else "cpu"
            if asr.gpu.device == "cpu":
                logger.info("auto: no CUDA GPU usable -> running Whisper on CPU")
            else:
                logger.info("auto: CUDA GPU detected -> running Whisper on GPU")
    elif resolved == "gpu" and not gpu_device_explicit:
        # Explicit `device: gpu`: honor it, align Whisper's compute device.
        asr.gpu.device = "cuda" if gpu_is_available() else "cpu"

    # Guard the compute_type against the device: float16 is GPU-only and errors
    # on CPU. If we're on CPU with a GPU-only compute_type, downgrade it.
    if asr.gpu.device == "cpu" and asr.gpu.compute_type in ("float16", "int8_float16"):
        logger.info(
            "asr.gpu.compute_type=%s is GPU-only; using int8 for CPU",
            asr.gpu.compute_type,
        )
        asr.gpu.compute_type = "int8"

    cfg = PipelineConfig(
        device=requested_device,
        resolved_device=resolved,
        capture=capture,
        asr=asr,
        buffer=buffer,
        matcher=matcher,
        playback=playback,
        telemetry=telemetry,
    )

    # Cross-stage sanity: warn if the buffer can't hold the longest keyword.
    if matcher.keywords:
        longest = max(len(k.pattern) for k in matcher.keywords)
        if longest > buffer.max_chars:
            logger.warning(
                "buffer.max_chars=%d is smaller than the longest keyword pattern "
                "(%d chars); multi-word triggers may be truncated.",
                buffer.max_chars, longest,
            )

    logger.info(
        "config loaded from %s: device requested=%s resolved=%s, %d keyword(s), "
        "capture ~%.1f ms/chunk",
        p, cfg.device, cfg.resolved_device, len(matcher.keywords),
        capture.chunk_latency_ms,
    )
    return cfg
