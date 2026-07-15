"""
playback package — Stage 5 (sound trigger / playback).

Exposes SoundPlayer, which preloads clips and plays them on match via a single
persistent, mixing output stream (real polyphony, non-blocking triggers).
"""

from .sound_player import SoundPlayer

__all__ = ["SoundPlayer"]
