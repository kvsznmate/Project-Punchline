"""
matcher/keyword_matcher.py — Stage 4: regex keyword matcher.

Scans the transcript buffer (Stage 3) for keyword patterns and, when one fires,
returns a match event naming the keyword and the sound to play. It does NOT play
audio itself — that is Stage 5's job. Keeping matching pure (text in, events
out) makes it fast and testable.

Latency-relevant design
------------------------
1. Patterns are compiled ONCE at construction, not per scan. Compilation is the
   expensive part of regex; the hot path (`scan`) only runs pre-compiled
   patterns over a small window.
2. Per-keyword cooldown (debounce). A commentator repeating "goal goal goal"
   should not fire the airhorn three times. Each keyword tracks the monotonic
   time it last fired; within its cooldown window, re-matches are suppressed.
3. Partial safety. Matching on partial (unstable) transcripts is lower latency
   but can misfire on words that later change. `match_on_partials` gates this:
   when False, the matcher only fires on final text. The buffer still decides
   whether partial text is even present; this is the second, independent guard.
4. Occurrence dedup. Cooldown alone can't stop a *single* spoken word from
   firing more than once: the word lingers in the rolling window for as long as
   it takes newer speech to push it out (often far longer than a short
   cooldown), and a naive scan re-matches it on every slide. So each scan is
   told the absolute position of the window's text (`base_offset` from the
   buffer) and we only fire on matches that end *beyond* the furthest offset
   already scanned (`_scanned_through`). A word therefore fires exactly once,
   when it first appears; cooldown is then left to do only its real job —
   debouncing genuinely repeated speech.

How scanning works
------------------
The buffer is tiny, so we still run each compiled pattern over the whole window
(cheap, and it keeps multi-word patterns that straddle an internal boundary
matchable). The novelty is only in what we FIRE on: `finditer` enumerates every
match, but a match is emitted only if (a) it ends past `_scanned_through` — i.e.
it is new text, not old text sliding by — and (b) its keyword is outside its
cooldown. After each scan we advance `_scanned_through` to the end of the window
so nothing currently visible can re-fire later.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

logger = logging.getLogger("pipeline.matcher")


@dataclass
class MatchEvent:
    """A fired keyword match, handed to playback.

    Attributes
    ----------
    pattern : str
        The regex pattern (source) that matched.
    sound : str
        The sound filename to play (relative to playback.sounds_dir).
    matched_text : str
        The exact substring that matched, for logging/telemetry.
    timestamp : float
        Monotonic time the match fired (for latency measurement downstream).
    """

    pattern: str
    sound: str
    matched_text: str
    timestamp: float = field(default_factory=time.monotonic)


class _CompiledKeyword:
    """Internal: a compiled pattern plus its cooldown state.

    `sounds` is always stored as a list of filenames. A keyword configured with
    a single string sound becomes a one-element list; a keyword configured with
    a list keeps all options. On each fire, one filename is chosen at random via
    `pick_sound()` (a uniform choice; for a single-element list that's just that
    element). Random selection happens only on the rare fire event, never in the
    scan loop, so it adds nothing to per-scan latency.
    """

    __slots__ = ("pattern_src", "sounds", "cooldown", "regex", "_last_fired")

    def __init__(self, pattern_src: str, sounds, cooldown: float, flags: int):
        self.pattern_src = pattern_src
        # Normalize to a non-empty list of filenames.
        if isinstance(sounds, (list, tuple)):
            self.sounds = [str(s) for s in sounds]
        else:
            self.sounds = [str(sounds)]
        if not self.sounds:
            raise ValueError(f"keyword {pattern_src!r} has no sound(s)")
        self.cooldown = float(cooldown)
        self.regex = re.compile(pattern_src, flags)
        self._last_fired: Optional[float] = None

    def pick_sound(self) -> str:
        """Choose one sound filename for this fire (random if multiple)."""
        if len(self.sounds) == 1:
            return self.sounds[0]
        return random.choice(self.sounds)

    def ready(self, now: float) -> bool:
        """True if this keyword is outside its cooldown window."""
        if self._last_fired is None:
            return True
        return (now - self._last_fired) >= self.cooldown

    def mark_fired(self, now: float) -> None:
        self._last_fired = now


class KeywordMatcher:
    """Regex keyword matcher with per-keyword cooldown.

    Parameters
    ----------
    keywords : list
        List of KeywordRule-like objects, each with `.pattern`, `.sound`, and
        `.cooldown` (None -> use default_cooldown_seconds).
    default_cooldown_seconds : float
        Cooldown applied to any keyword whose own cooldown is None.
    case_insensitive : bool
        Compile patterns with re.IGNORECASE.
    match_on_partials : bool
        If False, `scan` fires only when called with is_final=True text.
    """

    def __init__(
        self,
        keywords,
        default_cooldown_seconds: float = 3.0,
        case_insensitive: bool = True,
        match_on_partials: bool = True,
    ) -> None:
        self.match_on_partials = bool(match_on_partials)
        flags = re.IGNORECASE if case_insensitive else 0
        self._keywords: List[_CompiledKeyword] = []
        for kw in keywords:
            cooldown = kw.cooldown if kw.cooldown is not None else default_cooldown_seconds
            self._keywords.append(
                _CompiledKeyword(kw.pattern, kw.sound, cooldown, flags)
            )
        # Highest absolute transcript offset we've already scanned. Matches that
        # end at or before this are old text lingering in the window and must
        # NOT re-fire; only matches ending beyond it are genuinely new. Init to
        # -1 so the very first character (offset 0) counts as new.
        self._scanned_through = -1
        logger.info(
            "KeywordMatcher ready: %d keyword(s), match_on_partials=%s",
            len(self._keywords), self.match_on_partials,
        )

    def scan(
        self, text: str, is_final: bool = True, base_offset: int = 0
    ) -> List[MatchEvent]:
        """Scan `text` for keyword matches; return events for those that fire.

        Parameters
        ----------
        text : str
            The current transcript window (from the buffer).
        is_final : bool
            Whether this text reflects a final transcript. If the matcher is
            configured with match_on_partials=False, non-final calls return no
            events (we wait for stable text before firing).
        base_offset : int
            Absolute position, in the buffer's monotonic transcript stream, of
            `text[0]` (i.e. `TranscriptBuffer.base_offset`). Used to tell new
            speech from old text still lingering in the window: a match fires
            only if it ends beyond everything already scanned. Defaults to 0
            for callers passing a standalone string.

        Returns
        -------
        list[MatchEvent]
            One event per keyword that matched fresh text and was outside its
            cooldown.
        """
        if not text:
            return []
        if not is_final and not self.match_on_partials:
            return []

        now = time.monotonic()
        # Snapshot the high-water mark for this scan; every comparison below is
        # against the SAME threshold, and we advance it once at the end.
        threshold = self._scanned_through
        events: List[MatchEvent] = []
        for kw in self._keywords:
            for m in kw.regex.finditer(text):
                abs_end = base_offset + m.end()
                if abs_end <= threshold:
                    continue  # old text sliding through the window; already seen
                # Genuinely new occurrence.
                if not kw.ready(now):
                    continue  # within cooldown, suppress
                kw.mark_fired(now)
                chosen = kw.pick_sound()
                events.append(
                    MatchEvent(
                        pattern=kw.pattern_src,
                        sound=chosen,
                        matched_text=m.group(0),
                        timestamp=now,
                    )
                )
                logger.debug(
                    "match: %r -> %s (%r) @abs[%d:%d]",
                    kw.pattern_src, chosen, m.group(0),
                    base_offset + m.start(), abs_end,
                )
        # Everything visible in this scan is now "seen": mark the whole window
        # consumed so lingering text can't re-fire on a later slide. New fires
        # come only from text that appears beyond this offset.
        window_end = base_offset + len(text)
        if window_end > self._scanned_through:
            self._scanned_through = window_end
        return events

    def reset_cooldowns(self) -> None:
        """Clear per-keyword cooldown state (e.g. on a new stream).

        Note: this does NOT reset the occurrence-dedup high-water mark
        (`_scanned_through`). That is keyed to the buffer's monotonic offsets,
        which never move backwards, so resetting it here would risk re-firing on
        text still sitting in the window. For a clean slate, build a fresh
        matcher (and buffer) instead.
        """
        for kw in self._keywords:
            kw._last_fired = None
