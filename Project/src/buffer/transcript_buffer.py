"""
buffer/transcript_buffer.py — Stage 3: rolling transcript window.

A small rolling window of recent transcript text that the regex matcher (Stage
4) scans. Sized to the regex patterns in use: big enough to hold the longest
multi-word trigger, no bigger — a larger window is just more text to scan on
every match attempt.

The one subtlety: partials vs finals
------------------------------------
A streaming ASR emits a sequence of PARTIALS for the same in-progress utterance,
each one a *revision* of the last ("goal" -> "goalie" -> "goalie makes"), then a
FINAL that settles it. If we naively appended every partial we'd get
"goal goalie goalie makes" in the window and fire triggers on text the speaker
never actually completed.

So the buffer holds two parts:
  - `committed`: text from FINAL results, appended as utterances complete.
  - `partial_tail`: the *latest* partial for the current in-progress utterance,
    which REPLACES the previous partial rather than appending.
The scannable text is `committed + partial_tail`, trimmed to `max_chars`. When a
final arrives, the partial tail is folded into committed and cleared.

If `include_partials` is False, partials are ignored entirely and only finalized
text is ever scanned (safer, but you only match once an utterance ends —
higher latency).

Latency note
------------
All operations here are O(max_chars) string slicing on a deliberately small
window — negligible next to ASR inference. Keeping `max_chars` near the longest
keyword keeps both this and the matcher's per-scan cost minimal.
"""

from __future__ import annotations

import logging
from typing import Optional

from asr.base import TranscriptResult  # type: ignore

logger = logging.getLogger("pipeline.buffer")


class TranscriptBuffer:
    """Rolling window of recent transcript text for the matcher to scan.

    Parameters
    ----------
    max_chars : int
        Maximum length of the scannable window (buffer.max_chars). The window is
        trimmed from the left to this many characters.
    include_partials : bool
        If True, the current partial tail is included in the scannable text so
        matches can fire before an utterance finalizes. If False, only committed
        (final) text is scanned.
    """

    def __init__(self, max_chars: int = 240, include_partials: bool = True) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        self.max_chars = int(max_chars)
        self.include_partials = bool(include_partials)
        self._committed = ""     # settled text from finals (trimmed to window)
        self._partial_tail = ""  # latest partial for the in-progress utterance
        # Monotonic count of ALL characters ever committed, including any that
        # have since scrolled out of the trimmed window. This is what lets us
        # hand the matcher a STABLE absolute position for the window's contents
        # (see base_offset) so it can tell genuinely new speech from old text
        # still lingering in the window. Never reset (see clear()).
        self._committed_total = 0

    def update(self, result: TranscriptResult) -> str:
        """Fold a TranscriptResult into the window; return the scannable text.

        Final results append to committed text and clear the partial tail.
        Partial results replace the partial tail (or are ignored if
        include_partials is False).
        """
        if result.is_final:
            # Commit: append the finalized text, drop the working partial.
            text = result.text
            if text:
                added = ""
                if self._committed and not self._committed.endswith(" "):
                    added = " "
                added += text
                self._committed += added
                # Grow the monotonic counter by exactly what we appended (the
                # separator counts too) so a word keeps the same absolute offset
                # across the partial->final transition — the matcher then sees
                # the final as the SAME occurrence, not a new one.
                self._committed_total += len(added)
            self._partial_tail = ""
            self._trim_committed()
        else:
            # Partial: replace the working tail (do NOT append).
            if self.include_partials:
                self._partial_tail = result.text
        return self.text

    def _render(self):
        """Build the scannable window plus the absolute offset of its 1st char.

        Returns ``(window_text, base_offset)``. ``base_offset`` is the position
        of ``window_text[0]`` in the monotonic committed+partial character
        stream. The separator between committed text and the partial tail is
        applied here exactly as it is on commit, so offsets stay consistent
        across finalization.
        """
        committed = self._committed
        base = self._committed_total - len(committed)
        if self.include_partials and self._partial_tail:
            sep = " " if (committed and not committed.endswith(" ")) else ""
            combined = committed + sep + self._partial_tail
        else:
            combined = committed
        # Trim from the left to the window size, advancing the base to match.
        if len(combined) > self.max_chars:
            drop = len(combined) - self.max_chars
            combined = combined[drop:]
            base += drop
        return combined, base

    @property
    def text(self) -> str:
        """The current scannable window: committed text plus partial tail."""
        return self._render()[0]

    @property
    def base_offset(self) -> int:
        """Absolute offset of ``text[0]`` in the monotonic transcript stream.

        Pass this to ``KeywordMatcher.scan`` alongside ``text`` so the matcher
        can fire once per real occurrence instead of re-firing every time the
        window slides. Monotonic non-decreasing across the buffer's lifetime.
        """
        return self._render()[1]

    def clear(self) -> None:
        """Reset the *visible window* (e.g. on ASR reset / new stream).

        NOTE: this intentionally does NOT reset ``_committed_total``. Keeping
        that counter monotonic means absolute offsets never move backwards, so
        the matcher's dedup state stays valid and won't re-fire on text that
        happens to reuse an old offset. For a truly clean slate, build a fresh
        TranscriptBuffer together with a fresh KeywordMatcher.
        """
        self._committed = ""
        self._partial_tail = ""

    def _trim_committed(self) -> None:
        """Keep committed text from growing without bound between finals."""
        if len(self._committed) > self.max_chars:
            self._committed = self._committed[-self.max_chars:]
