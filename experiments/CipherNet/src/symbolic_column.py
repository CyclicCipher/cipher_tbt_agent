"""Symbolic cortical column with receptive fields.

Each column has a POSITION (from wiring) and learns FEATURES at that
position (from signal content). Position and feature are pre-separated
by the topographic mapping (labeled line principle).

Memory accumulates votes: the same feature can map to multiple targets
(e.g., a pixel patch appearing in different digit classes). Prediction
returns the mode (most frequent target). This is domain-general:
succession has 1 target per feature, classification has many.
"""
from __future__ import annotations

from typing import Any
import numpy as np


class SymbolicColumn:
    """A cortical column with a fixed receptive field.

    Position = which input positions this column receives from.
    Feature = what signal content arrives at those positions.
    Memory = {feature → {target: count}} (accumulated votes).
    """

    def __init__(self, name: str, receptive_field: Any = None,
                 position: tuple = (0.0, 0.0)):
        self.name = name
        self.receptive_field = receptive_field
        self.position = position
        # Memory: feature → {target: count}. Accumulates, never overwrites.
        self.memory: dict[str, dict[str, int]] = {}
        # Transient state.
        self.current_input: str | None = None
        self.prediction: str | None = None
        self.error: bool = False
        self.vote: str | None = None

    def observe(self, feature: str) -> bool:
        """Observe a feature. Compare against prediction."""
        self.current_input = feature
        self.prediction = self.predict()
        self.error = (self.prediction is None)
        self.vote = feature
        return self.error

    def teach(self, feature: str, target: str):
        """Accumulate: feature → target gets +1 vote."""
        if feature not in self.memory:
            self.memory[feature] = {}
        counts = self.memory[feature]
        counts[target] = counts.get(target, 0) + 1

    def predict(self) -> str | None:
        """Return the most frequent target for current input (mode)."""
        if self.current_input is None:
            return None
        counts = self.memory.get(self.current_input)
        if not counts:
            return None
        return max(counts, key=counts.get)

    def predict_dist(self) -> dict[str, float] | None:
        """Return full distribution P(target | current_input)."""
        if self.current_input is None:
            return None
        counts = self.memory.get(self.current_input)
        if not counts:
            return None
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def reset(self):
        """Reset transient state. Memory preserved."""
        self.current_input = None
        self.prediction = None
        self.error = False
        self.vote = None

    def __repr__(self):
        n_features = len(self.memory)
        return f"SymbolicColumn({self.name}, rf={self.receptive_field}, {n_features} features)"


class ColumnSheet2D:
    """A 2D grid of symbolic columns (visual cortex).

    Each column covers a rectangular patch of the input image.
    Receptive fields can overlap with neighbors (shared pixels
    create implicit lateral connections).

    For MNIST (28×28): 7×7 grid, 4×4 patches, stride 4.
    For Danganronpa (1920×1080): configurable grid, same code.
    """

    def __init__(self, name: str, image_shape: tuple[int, int],
                 patch_size: int = 4, stride: int = 4):
        self.name = name
        self.image_h, self.image_w = image_shape
        self.patch_size = patch_size
        self.stride = stride
        self.grid_h = (self.image_h - patch_size) // stride + 1
        self.grid_w = (self.image_w - patch_size) // stride + 1

        # Create 2D grid of columns.
        self.columns: list[list[SymbolicColumn]] = []
        for gy in range(self.grid_h):
            row = []
            for gx in range(self.grid_w):
                # Receptive field in pixel coordinates.
                y0 = gy * stride
                x0 = gx * stride
                rf = (y0, x0, y0 + patch_size, x0 + patch_size)
                col = SymbolicColumn(
                    name=f"{name}:{gy},{gx}",
                    receptive_field=rf,
                    position=(float(gx), float(gy)),
                )
                row.append(col)
            self.columns.append(row)

    def n_columns(self) -> int:
        return self.grid_h * self.grid_w

    def all_columns(self):
        """Iterate over all columns (flat)."""
        for row in self.columns:
            yield from row

    def extract_patches(self, image: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
        """Extract patches from an image. Returns (gy, gx, patch) tuples."""
        patches = []
        ps = self.patch_size
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                y0 = gy * self.stride
                x0 = gx * self.stride
                patch = image[y0:y0 + ps, x0:x0 + ps]
                patches.append((gy, gx, patch))
        return patches

    def __repr__(self):
        return (f"ColumnSheet2D({self.name}, {self.grid_h}x{self.grid_w} columns, "
                f"patch={self.patch_size}, stride={self.stride})")


class SuccessionEngine:
    """Z/10Z successor morphism for multi-digit succession."""

    SUCC = {}
    for _d in range(10):
        for _c in (False, True):
            _val = _d + 1 + (1 if _c else 0)
            SUCC[(_d, _c)] = (_val % 10, _val >= 10)

    @staticmethod
    def successor(number_str: str) -> str:
        """Compute successor of a number string."""
        digits = [int(d) for d in number_str]
        result = []
        carry = False
        for i in range(len(digits) - 1, -1, -1):
            d = digits[i]
            if i == len(digits) - 1:
                out_d, carry = SuccessionEngine.SUCC[(d, False)]
            else:
                if carry:
                    out_d, carry = SuccessionEngine.SUCC[(d, False)]
                else:
                    out_d = d
                    carry = False
            result.append(str(out_d))
        if carry:
            result.append("1")
        result.reverse()
        return ''.join(result)
