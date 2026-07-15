"""
matcher package — Stage 4 (regex keyword matcher).

Exposes KeywordMatcher (scans transcript text, applies per-keyword cooldown) and
MatchEvent (what a fired match hands to the playback stage).
"""

from .keyword_matcher import KeywordMatcher, MatchEvent

__all__ = ["KeywordMatcher", "MatchEvent"]
