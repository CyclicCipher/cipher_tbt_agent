"""VisualModality — vision-specific sensor adapter.

Wraps Eye (retinal preprocessing + fixation) and an SDR encoder
(Gabor filter bank or HOG) into the SensorModality interface.

The cortex calls preprocess() once per image and encode() once per
fixation; it never touches Eye or the encoder directly.

Patch layout
------------
Patches are extracted from the retinal window in row-major order
matching the sensor layer's (grid_h x grid_w) column grid:

  patch index k = gy * grid_w + gx
  top-left pixel in retina: (x0 = gx * stride, y0 = gy * stride)

This matches the order in which MacroColumns are created in
Cortex.from_config so column i always receives patch i.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from modalities.base import SensorModality


class VisualModality(SensorModality):
    """Vision modality: DoG retina + patch encoder.

    Args:
        eye:     Eye instance (preprocess + fixate + sample).
        encoder: SDR encoder (GaborFilterBank or HOGEncoder).
                 Must implement encode_batch(patches) -> list[np.ndarray|None].
        grid_h:  Sensor layer column grid height.
        grid_w:  Sensor layer column grid width.
        patch_size: Patch side length in retinal pixels.
        stride:     Stride between patch centres in retinal pixels.
    """

    def __init__(self, eye, encoder, grid_h: int, grid_w: int,
                 patch_size: int, stride: int):
        self._eye        = eye
        self._encoder    = encoder
        self._grid_h     = grid_h
        self._grid_w     = grid_w
        self._patch_size = patch_size
        self._stride     = stride
        self._n_cols     = grid_h * grid_w

    # ------------------------------------------------------------------
    # SensorModality interface
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply retinal DoG filter.

        Returns a float32 contrast-coded image (same shape as input).
        This is the 'dog' image passed to encode() for every fixation.
        """
        return self._eye.preprocess(image)

    def encode(self, dog: np.ndarray, fixation: tuple) -> list:
        """Extract and encode patches for one fixation.

        Moves the eye to fixation, crops the retinal window, slices it
        into (grid_h x grid_w) patches, and passes the batch to the
        encoder.

        Returns:
            list[np.ndarray | None] of length n_columns.
            None = blank/silent patch (below encoder activity threshold).
        """
        self._eye.fixate(float(fixation[0]), float(fixation[1]))
        retina = self._eye.sample(dog)

        ps = self._patch_size
        st = self._stride

        patches = np.empty((self._n_cols, ps, ps), dtype=np.float32)
        k = 0
        for gy in range(self._grid_h):
            y0 = gy * st
            for gx in range(self._grid_w):
                x0 = gx * st
                patches[k] = retina[y0:y0 + ps, x0:x0 + ps]
                k += 1

        return self._encoder.encode_batch(patches)

    @property
    def n_columns(self) -> int:
        return self._n_cols
