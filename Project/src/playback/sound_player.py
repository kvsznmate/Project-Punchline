"""
playback/sound_player.py — Stage 5: sound trigger / playback.

Fires a sound effect the instant the matcher (Stage 4) reports a hit, mixed into
the local audio output. This is the last stage, and the whole pipeline exists to
make it fast — so the design minimizes everything between "match fired" and
"speaker moves".

Latency-critical design
-----------------------
1. Preload + decode at startup. Every clip named by a keyword is read from disk
   and decoded into a numpy array ONCE, when the player is built. A trigger then
   just references an already-decoded array — no disk read, no decode on the hot
   path. (preload=False decodes lazily on first use; that first trigger pays it.)
2. One persistent output stream + callback mixer. We open a single
   sounddevice.OutputStream at construction and keep it running. The audio
   library pulls frames from our callback on its own realtime thread; `play()`
   just adds a voice to the active list and returns. Nothing blocks the caller,
   and — crucially — multiple active voices are SUMMED in the callback, so
   sounds genuinely overlap (real polyphony), unlike sd.play() which has a
   single global stream and would cut the previous sound off.
3. Polyphony cap. At most `max_polyphony` voices mix at once; a new trigger over
   the cap evicts the oldest. Bounds mixing cost and avoids a wall of sound.

Expected latency contribution: `play()` itself is sub-millisecond (append a
voice under a short lock). The real floor is the output stream's blocksize /
samplerate buffer, typically a few to low-tens of ms depending on backend.

Dependencies
------------
    soundfile   (pip install soundfile) to read/decode wav/ogg/flac.
    sounddevice (pip install sounddevice) for the output stream; both imported
    lazily so importing this module doesn't hard-require them.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("pipeline.playback")


@dataclass
class _Voice:
    """An actively-playing clip and how far into it we are."""
    data: np.ndarray   # shape (n,) mono or (n, ch)
    pos: int = 0


class SoundPlayer:
    """Preloading, non-blocking, mixing sound player with a polyphony cap.

    Parameters
    ----------
    sounds_dir : str
        Directory containing the sound files (playback.sounds_dir).
    sounds : list[str]
        Filenames to make playable (the distinct sounds the matcher references).
        With preload=True these are decoded now.
    output_device : int | str | None
        Output device index/name, or None for system default.
    preload : bool
        Decode all `sounds` at construction (recommended for low latency).
    max_polyphony : int
        Max simultaneous voices; a new trigger over this evicts the oldest.
    volume : float
        Output gain in [0, 1] applied to the mix.
    samplerate : int
        Output stream samplerate. Clips whose native rate differs are flagged.
    channels : int
        Output channel count for the stream (default 1 = mono).
    blocksize : int
        Output stream blocksize (frames per callback). Smaller = lower output
        latency, more callbacks. Left modest by default.
    """

    def __init__(
        self,
        sounds_dir: str,
        sounds: List[str],
        output_device=None,
        preload: bool = True,
        max_polyphony: int = 4,
        volume: float = 1.0,
        samplerate: int = 16000,
        channels: int = 1,
        blocksize: int = 256,
    ) -> None:
        if max_polyphony < 1:
            raise ValueError("max_polyphony must be >= 1")
        if not (0.0 <= volume <= 1.0):
            raise ValueError("volume must be in [0, 1]")

        try:
            import sounddevice as sd  # type: ignore
            import soundfile as sf    # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "playback needs 'sounddevice' and 'soundfile' "
                "(pip install sounddevice soundfile). "
                f"Original import error: {exc!r}"
            ) from exc

        self._sd = sd
        self._sf = sf
        self.sounds_dir = sounds_dir
        self.output_device = output_device
        self.preload = bool(preload)
        self.max_polyphony = int(max_polyphony)
        self.volume = float(volume)
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.blocksize = int(blocksize)

        self._clips: Dict[str, np.ndarray] = {}
        self._voices: List[_Voice] = []
        self._lock = threading.Lock()

        if self.preload:
            for name in sounds:
                self._load(name)

        # One persistent output stream; the callback mixes active voices.
        self._stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            device=self.output_device,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "SoundPlayer ready: %d clip(s) preloaded, max_polyphony=%d "
            "volume=%.2f sr=%d ch=%d blocksize=%d",
            len(self._clips), self.max_polyphony, self.volume,
            self.samplerate, self.channels, self.blocksize,
        )

    # -- public API --------------------------------------------------------

    def play(self, sound: str) -> None:
        """Start playing `sound` immediately; return without waiting.

        Hot path: fetch decoded array, add a voice (evicting oldest over the
        cap), return. The callback thread does the actual mixing/output.
        """
        data = self._clips.get(sound)
        if data is None:
            data = self._load(sound)
            if data is None:
                return
        with self._lock:
            if len(self._voices) >= self.max_polyphony:
                self._voices.pop(0)  # evict oldest
            self._voices.append(_Voice(data=data, pos=0))

    def stop_all(self) -> None:
        with self._lock:
            self._voices.clear()

    def close(self) -> None:
        """Stop and close the output stream."""
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # pragma: no cover
            pass
        self.stop_all()

    # -- realtime callback (keep cheap) ------------------------------------

    def _callback(self, outdata, frames, time_info, status) -> None:
        """Mix active voices into the output buffer. Runs on the audio thread."""
        if status:
            logger.debug("playback stream status: %s", status)
        # Start from silence, shape (frames, channels).
        mix = np.zeros((frames, self.channels), dtype=np.float32)

        with self._lock:
            still: List[_Voice] = []
            for v in self._voices:
                remaining = v.data.shape[0] - v.pos
                if remaining <= 0:
                    continue
                n = min(frames, remaining)
                seg = v.data[v.pos:v.pos + n]
                # Shape the segment to (n, channels).
                if seg.ndim == 1:
                    seg2 = np.repeat(seg[:, None], self.channels, axis=1)
                else:
                    # Clip has channels; match/truncate to output channels.
                    if seg.shape[1] >= self.channels:
                        seg2 = seg[:, : self.channels]
                    else:
                        seg2 = np.repeat(seg[:, :1], self.channels, axis=1)
                mix[:n] += seg2
                v.pos += n
                if v.pos < v.data.shape[0]:
                    still.append(v)  # keep voices that still have audio
            self._voices = still

        if self.volume != 1.0:
            mix *= self.volume
        np.clip(mix, -1.0, 1.0, out=mix)  # avoid wrap on overlapping sums
        outdata[:] = mix

    # -- internals ---------------------------------------------------------

    def _load(self, name: str) -> Optional[np.ndarray]:
        path = os.path.join(self.sounds_dir, name)
        if not os.path.isfile(path):
            logger.warning("sound file not found: %s", path)
            return None
        try:
            data, sr = self._sf.read(path, dtype="float32", always_2d=False)
        except Exception as exc:
            logger.warning("failed to decode %s: %s", path, exc)
            return None
        if sr != self.samplerate:
            logger.warning(
                "clip %s is %d Hz but output is %d Hz; pitch/speed will be off. "
                "Provide clips at the output samplerate.",
                name, sr, self.samplerate,
            )
        self._clips[name] = data
        return data
