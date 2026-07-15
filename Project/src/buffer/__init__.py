"""
buffer package — Stage 3 (rolling transcript window).

Exposes the TranscriptBuffer that sits between the ASR stage and the matcher,
holding a small window of recent transcript text sized to the keyword patterns.
"""

from .transcript_buffer import TranscriptBuffer

__all__ = ["TranscriptBuffer"]
