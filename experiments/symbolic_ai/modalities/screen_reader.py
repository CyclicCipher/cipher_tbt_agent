"""screen_reader.py -- Phase S.0c: Unified screen reading interface.

Combines GlyphReader + TextScanner into a single API used by GenericModality.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from modalities.glyph_reader import GlyphReader
from modalities.text_scanner import TextScanner, FrameReading


class ScreenReader:
    """The visual reading capability.

    Wraps GlyphReader (character recognition) + TextScanner (layout + saccade)
    into a single interface.

    Usage::

        reader = ScreenReader.train_and_save("glyph_reader.pkl")
        reader = ScreenReader.load("glyph_reader.pkl")
        reading = reader.read(frame)
        print(reading.narrative)
        print(reading.buttons)
    """

    def __init__(self, glyph_reader: GlyphReader, foveal_size: int = 16) -> None:
        self._glyph   = glyph_reader
        self._scanner = TextScanner(glyph_reader, foveal_size=foveal_size)
        self._last_reading: Optional[FrameReading] = None

    @classmethod
    def train_and_save(
        cls,
        path:       str  = "glyph_reader.pkl",
        n_clusters: int  = 128,
        verbose:    bool = True,
    ) -> "ScreenReader":
        """Train GlyphReader from scratch; save to disk; return ScreenReader."""
        reader = GlyphReader(n_clusters=n_clusters, model_path=path)
        reader.train(verbose=verbose)
        reader.save(path)
        return cls(reader)

    @classmethod
    def load(cls, path: str = "glyph_reader.pkl", foveal_size: int = 16) -> "ScreenReader":
        """Load pre-trained GlyphReader from disk."""
        reader = GlyphReader.load(path)
        return cls(reader, foveal_size=foveal_size)

    def calibrate(self, frame: np.ndarray, verbose: bool = False) -> None:
        """Fine-tune GlyphReader centroids to this game font rendering."""
        self._glyph.calibrate(frame, verbose=verbose)

    def read(self, frame: np.ndarray) -> FrameReading:
        """Full pipeline: layout detection -> saccadic reading -> FrameReading."""
        reading = self._scanner.read_frame(frame)
        self._last_reading = reading
        return reading

    @property
    def last_reading(self) -> Optional[FrameReading]:
        """The FrameReading from the most recent read() call."""
        return self._last_reading

    def summary(self) -> str:
        s = self._glyph.summary()
        if self._last_reading is not None:
            r = self._last_reading
            s += (
                f"\n  last reading: {len(r.buttons)} buttons, "
                f"{len(r.narrative)} chars narrative, "
                f"{len(r.raw_regions)} regions"
            )
        return s
