"""
telemetry package — cross-stage latency instrumentation.

Exposes Telemetry, which records stage-boundary timings and logs the
transcript->playback latency of each trigger, so latency regressions are visible
rather than guessed at.
"""

from .latency import Telemetry, LatencyStats

__all__ = ["Telemetry", "LatencyStats"]
