"""
Audio capture stage (stage 1 of the transcription-to-soundboard pipeline).

This package owns the *only* point where live audio enters the system. It
streams audio from a mic or system-loopback device in small fixed-size chunks
and hands them off to the ASR stage with as little added latency as possible.

Public API
----------
    AudioCapture      - the streaming capture class (start/stop, chunk queue)
    CaptureConfig     - resolved config for a capture session
    list_input_devices - helper to enumerate available input devices

Typical use::

    from capture import AudioCapture, CaptureConfig

    cfg = CaptureConfig(samplerate=16000, blocksize=1600, channels=1)
    cap = AudioCapture(cfg)
    cap.start()
    for chunk in cap.chunks():        # blocks until audio arrives
        process(chunk)                # hand to ASR
    cap.stop()

See README.md in this directory for the full explanation.
"""

from .audio_capture import (
    AudioCapture,
    CaptureConfig,
    list_input_devices,
)

__all__ = [
    "AudioCapture",
    "CaptureConfig",
    "list_input_devices",
]
