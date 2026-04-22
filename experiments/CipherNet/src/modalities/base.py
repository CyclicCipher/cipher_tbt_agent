"""SensorModality — abstract base for all input modalities.

A modality owns the full input-to-SDR pipeline for one sensor layer.
It has two responsibilities:

  preprocess(raw_input) -> prepared
    Expensive one-time preparation applied once per stimulus before
    any fixation loop starts.  Examples:
      Vision  : DoG (difference of Gaussians) retinal filter
      Audio   : STFT spectrogram
      Text    : tokenise + embed

  encode(prepared, fixation) -> list[np.ndarray | None]
    Per-fixation feature extraction.  Returns one SDR (or None for a
    silent/blank slot) per column in the sensor layer, in row-major
    order matching the layer's (grid_h x grid_w) layout.  None entries
    are silently skipped by the cortex forward sweep.

The cortex stores a SensorModality on each sensor Layer and calls
these two methods — no eye.py, no codebook, no audio libraries bleed
into cortex.py itself.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SensorModality(ABC):
    """Abstract sensor modality.

    Subclasses implement preprocess() and encode() for a specific
    input type (vision, audio, text, …).
    """

    @abstractmethod
    def preprocess(self, raw_input: Any) -> Any:
        """One-time stimulus preparation.

        Called once per input before the fixation loop.
        The return value is passed to every encode() call for this
        stimulus, so it should capture all expensive shared computation.

        Args:
            raw_input: The raw stimulus (e.g. np.ndarray for images).

        Returns:
            Prepared form passed to encode().
        """

    @abstractmethod
    def encode(self, prepared: Any, fixation: tuple) -> list:
        """Per-fixation feature extraction.

        Args:
            prepared: Output of preprocess() for this stimulus.
            fixation: (x, y) or general position tuple for this fixation.

        Returns:
            list of length n_columns, each entry np.ndarray (int8 SDR)
            or None (silent/blank — cortex skips it automatically).
            Order matches the layer's row-major column layout.
        """

    @property
    def n_columns(self) -> int:
        """Number of columns (features) produced per fixation."""
        raise NotImplementedError
