"""Foveal eye — a sensor that looks at images with variable resolution.

The eye is a SEPARATE OBJECT from the brain. It has:
- A fixation point (where it's looking in the image)
- A fovea (high-resolution center)
- Logarithmically decreasing resolution toward the periphery
- Saccade capability (move fixation point → displacement signal)

V1 columns read from the EYE (retinotopic), not the image directly.
When the eye saccades, the image shifts on the retina, but V1 columns
stay fixed — they always read from the same retinal position.

The eye produces:
- A retinal image (foveal sampling of the world image)
- A displacement vector (efference copy from saccade)
"""
from __future__ import annotations

import numpy as np


class Eye:
    """A foveal eye that samples images with log-polar resolution.

    The retina is a 1D array of samples radiating from the fovea.
    Resolution is highest at the center (fovea) and decreases
    logarithmically toward the periphery.

    For a 2D image, the retina samples along concentric rings
    at increasing radii with decreasing angular density.
    """

    def __init__(self, fovea_radius: int = 3, n_rings: int = 5,
                 samples_per_ring: int = 8):
        """
        Args:
            fovea_radius: pixel radius of the high-resolution fovea.
            n_rings: number of concentric rings beyond the fovea.
                Each ring is at 2× the radius of the previous (log spacing).
            samples_per_ring: angular samples per ring.
        """
        self.fovea_radius = fovea_radius
        self.n_rings = n_rings
        self.samples_per_ring = samples_per_ring

        # Current fixation point in image coordinates.
        self.fixation: tuple[float, float] = (0.0, 0.0)

        # Build sampling grid: list of (dx, dy, ring_index) offsets from fixation.
        # Ring 0 = fovea (dense grid within fovea_radius).
        # Ring 1+ = concentric rings at exponentially increasing radii.
        self._sample_offsets: list[tuple[float, float, int]] = []
        self._build_sampling_grid()

        # Last saccade displacement (efference copy).
        self.last_displacement: tuple[float, float] = (0.0, 0.0)

    def _build_sampling_grid(self):
        """Build the retinal sampling positions."""
        self._sample_offsets = []

        # Fovea: dense grid within fovea_radius.
        r = self.fovea_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    self._sample_offsets.append((float(dx), float(dy), 0))

        # Peripheral rings: log-spaced radii, fixed angular samples.
        for ring in range(1, self.n_rings + 1):
            radius = self.fovea_radius * (2.0 ** ring)
            for s in range(self.samples_per_ring):
                angle = 2.0 * np.pi * s / self.samples_per_ring
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)
                self._sample_offsets.append((dx, dy, ring))

    @property
    def n_samples(self) -> int:
        """Total number of retinal samples."""
        return len(self._sample_offsets)

    @property
    def n_foveal(self) -> int:
        """Number of foveal (high-resolution) samples."""
        return sum(1 for _, _, ring in self._sample_offsets if ring == 0)

    def fixate(self, x: float, y: float):
        """Move fixation to (x, y) in image coordinates. No saccade signal."""
        self.fixation = (x, y)
        self.last_displacement = (0.0, 0.0)

    def saccade(self, dx: float, dy: float):
        """Move fixation by (dx, dy). Produces displacement (efference copy)."""
        fx, fy = self.fixation
        self.fixation = (fx + dx, fy + dy)
        self.last_displacement = (dx, dy)

    def saccade_to(self, x: float, y: float):
        """Move fixation to (x, y). Produces displacement."""
        fx, fy = self.fixation
        self.last_displacement = (x - fx, y - fy)
        self.fixation = (x, y)

    def sample(self, image: np.ndarray) -> np.ndarray:
        """Sample the image through the retina.

        Returns a 1D array of pixel values, one per retinal sample.
        Samples outside the image boundary return 0 (black).

        The ordering matches _sample_offsets: foveal samples first,
        then peripheral rings in order.
        """
        h, w = image.shape[:2]
        fx, fy = self.fixation
        result = np.zeros(self.n_samples, dtype=np.float32)

        for i, (dx, dy, ring) in enumerate(self._sample_offsets):
            # Image coordinates (nearest-neighbor sampling).
            ix = int(round(fx + dx))
            iy = int(round(fy + dy))
            if 0 <= ix < w and 0 <= iy < h:
                result[i] = image[iy, ix]

        return result

    def get_ring_indices(self, ring: int) -> list[int]:
        """Get sample indices for a specific ring (0=fovea, 1+=peripheral)."""
        return [i for i, (_, _, r) in enumerate(self._sample_offsets) if r == ring]

    def get_sample_positions(self) -> list[tuple[float, float, int]]:
        """Get (dx, dy, ring) for each retinal sample."""
        return list(self._sample_offsets)

    def __repr__(self):
        return (f"Eye(fovea_r={self.fovea_radius}, rings={self.n_rings}, "
                f"samples={self.n_samples}, foveal={self.n_foveal})")


class SalienceMap:
    """Simple bottom-up salience for guiding saccades.

    Computes "interestingness" at each image position based on
    local contrast (gradient magnitude). The eye is drawn to
    high-salience locations.

    This is NOT a command — it's a suggestion. The brain can
    override salience with top-down attention (intentional saccades).
    """

    @staticmethod
    def compute(image: np.ndarray) -> np.ndarray:
        """Compute salience map from an image.

        Returns a same-size array of salience values (higher = more salient).
        Uses gradient magnitude as a simple contrast measure.
        """
        h, w = image.shape[:2]
        salience = np.zeros((h, w), dtype=np.float32)

        # Gradient magnitude (Sobel-like).
        if h > 2 and w > 2:
            gx = np.abs(image[1:-1, 2:] - image[1:-1, :-2])
            gy = np.abs(image[2:, 1:-1] - image[:-2, 1:-1])
            salience[1:-1, 1:-1] = np.sqrt(gx[:, :min(gx.shape[1], gy.shape[1])]**2 +
                                            gy[:min(gx.shape[0], gy.shape[0]), :]**2)

        return salience

    @staticmethod
    def suggest_fixations(image: np.ndarray, n: int = 5,
                          min_distance: int = 4) -> list[tuple[int, int]]:
        """Suggest N fixation points based on salience.

        Returns [(x, y), ...] in image coordinates, highest salience first.
        Enforces minimum distance between fixations to avoid clustering.
        """
        sal = SalienceMap.compute(image)
        fixations = []

        for _ in range(n):
            if sal.max() < 1e-6:
                break
            # Find peak.
            idx = np.unravel_index(sal.argmax(), sal.shape)
            y, x = int(idx[0]), int(idx[1])
            fixations.append((x, y))
            # Suppress neighborhood.
            y0 = max(0, y - min_distance)
            y1 = min(sal.shape[0], y + min_distance + 1)
            x0 = max(0, x - min_distance)
            x1 = min(sal.shape[1], x + min_distance + 1)
            sal[y0:y1, x0:x1] = 0

        return fixations
