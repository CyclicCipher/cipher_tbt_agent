"""Foveal eye with constrained saccade vocabulary.

The eye produces a RETINAL IMAGE: a 2D crop around the fixation point.
V1 columns tile this retinal image with patch receptive fields.

Saccades use a FIXED VOCABULARY of ~8 vectors (4 cardinal + 4 diagonal)
at quantized amplitudes. This matches biology: the superior colliculus
has a topographic map creating a discrete grid of saccade vectors,
saccades are 60% horizontal / 30% vertical / 10% oblique, and
85% of saccades are short (<15°).

The constrained vocabulary creates the regularity needed for category
discovery to find the Z² spatial structure.
"""
from __future__ import annotations

import numpy as np


class Eye:
    """Foveal eye that produces a retinal image around fixation.

    The retinal image is a fixed-size 2D array. The center (fovea)
    maps 1:1 to the source image pixels. The periphery could be
    downsampled in a future version (for now: simple crop).
    """

    def __init__(self, retina_size: int = 19):
        """
        Args:
            retina_size: side length of the square retinal image.
                Must be odd (center pixel = fixation point).
                Default 19: captures a 19×19 patch around fixation,
                enough context for a single MNIST digit.
        """
        assert retina_size % 2 == 1, "retina_size must be odd"
        self.retina_size = retina_size
        self.half = retina_size // 2
        self.fixation: tuple[float, float] = (0.0, 0.0)
        self.last_displacement: tuple[float, float] = (0.0, 0.0)

    def fixate(self, x: float, y: float):
        """Set fixation point. No saccade signal."""
        self.fixation = (x, y)
        self.last_displacement = (0.0, 0.0)

    def saccade(self, dx: float, dy: float):
        """Move fixation by (dx, dy). Produces displacement."""
        fx, fy = self.fixation
        self.fixation = (fx + dx, fy + dy)
        self.last_displacement = (dx, dy)

    def saccade_to(self, x: float, y: float):
        """Move fixation to absolute position. Produces displacement."""
        fx, fy = self.fixation
        self.last_displacement = (x - fx, y - fy)
        self.fixation = (x, y)

    def sample(self, image: np.ndarray) -> np.ndarray:
        """Produce retinal image: a crop around fixation point.

        Returns a (retina_size, retina_size) float32 array.
        Pixels outside the source image are 0 (black border).
        """
        h, w = image.shape[:2]
        fx, fy = int(round(self.fixation[0])), int(round(self.fixation[1]))
        r = self.half

        retina = np.zeros((self.retina_size, self.retina_size), dtype=np.float32)

        # Source and destination ranges (clipped to image bounds).
        src_y0 = max(0, fy - r)
        src_y1 = min(h, fy + r + 1)
        src_x0 = max(0, fx - r)
        src_x1 = min(w, fx + r + 1)

        dst_y0 = src_y0 - (fy - r)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = src_x0 - (fx - r)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        if src_y1 > src_y0 and src_x1 > src_x0:
            retina[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]

        return retina

    # --- Constrained saccade vocabulary ---
    # 8 vectors: 4 cardinal + 4 diagonal at a fixed step size.
    # Matches biology: SC topographic map, cardinal bias, quantized amplitudes.

    SACCADE_VOCAB = [
        (1, 0), (-1, 0), (0, 1), (0, -1),    # cardinal
        (1, 1), (1, -1), (-1, 1), (-1, -1),   # diagonal
    ]

    def grid_scan(self, image: np.ndarray, step: int = 5,
                  center: tuple[float, float] | None = None
                  ) -> list[tuple[float, float]]:
        """Generate a systematic grid scan pattern over the image.

        Returns list of fixation points (x, y) in a regular grid
        centered on the image (or specified center). Uses cardinal
        saccade directions at fixed step size.

        This produces the regular displacement patterns needed for
        category discovery to find Z² structure.
        """
        h, w = image.shape[:2]
        if center is None:
            cx, cy = w / 2.0, h / 2.0
        else:
            cx, cy = center

        fixations = [(cx, cy)]  # start at center

        # Spiral outward from center using cardinal steps.
        for radius in range(1, max(h, w) // step + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue  # only perimeter of current ring
                    fx = cx + dx * step
                    fy = cy + dy * step
                    if 0 <= fx < w and 0 <= fy < h:
                        fixations.append((fx, fy))

        return fixations

    def cardinal_scan(self, image: np.ndarray, step: int = 5,
                      n_fixations: int = 9) -> list[tuple[float, float]]:
        """Generate fixations using cardinal/diagonal saccade vocabulary.

        Starts at image center, saccades using the 8-vector vocabulary
        at the given step size. Returns up to n_fixations points.
        """
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        fixations = [(cx, cy)]

        for dx, dy in self.SACCADE_VOCAB:
            if len(fixations) >= n_fixations:
                break
            fx = cx + dx * step
            fy = cy + dy * step
            if 0 <= fx < w and 0 <= fy < h:
                fixations.append((fx, fy))

        return fixations

    def __repr__(self):
        return f"Eye(retina={self.retina_size}x{self.retina_size})"


class SalienceMap:
    """Bottom-up salience for guiding saccades."""

    @staticmethod
    def compute(image: np.ndarray) -> np.ndarray:
        """Gradient magnitude salience."""
        h, w = image.shape[:2]
        sal = np.zeros((h, w), dtype=np.float32)
        if h > 2 and w > 2:
            gx = np.abs(np.diff(image, axis=1))
            gy = np.abs(np.diff(image, axis=0))
            min_h = min(gx.shape[0], gy.shape[0])
            min_w = min(gx.shape[1], gy.shape[1])
            sal[1:1+min_h, 1:1+min_w] = np.sqrt(
                gx[:min_h, :min_w]**2 + gy[:min_h, :min_w]**2)
        return sal

    @staticmethod
    def suggest_fixations(image: np.ndarray, n: int = 5,
                          min_distance: int = 4) -> list[tuple[int, int]]:
        """Suggest N fixation points, highest salience first."""
        sal = SalienceMap.compute(image)
        fixations = []
        for _ in range(n):
            if sal.max() < 1e-6:
                break
            idx = np.unravel_index(sal.argmax(), sal.shape)
            y, x = int(idx[0]), int(idx[1])
            fixations.append((x, y))
            y0 = max(0, y - min_distance)
            y1 = min(sal.shape[0], y + min_distance + 1)
            x0 = max(0, x - min_distance)
            x1 = min(sal.shape[1], x + min_distance + 1)
            sal[y0:y1, x0:x1] = 0
        return fixations
