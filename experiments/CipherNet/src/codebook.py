"""Visual feature encoding: Gabor filter bank + sparse coding.

Replaces VQ codebook (hard assignment to nearest centroid) with
biologically correct encoding:

1. Gabor filter bank: fixed oriented edge detectors at multiple
   orientations and spatial frequencies (like V1 simple cells).
2. Graded response: each patch activates many filters with
   continuous response strengths (not one-hot).
3. Sparse coding: threshold to keep only top-K active filters.
   The active filter IDs become the SDR bits.

This is what V1 actually does: overcomplete basis of Gabor-like
filters → sparse distributed population code.
"""
from __future__ import annotations

import numpy as np


class GaborFilterBank:
    """Fixed Gabor filter bank for patch encoding (V1 simple cells).

    Generates oriented edge filters at multiple orientations and
    spatial frequencies. Each filter is a 2D Gabor wavelet.

    Encoding: patch → response vector (one per filter) → sparse
    code (top-K active filter IDs as string key).
    """

    def __init__(self, patch_size: int = 5,
                 n_orientations: int = 8,
                 n_frequencies: int = 2,
                 top_k: int = 4):
        """
        Args:
            patch_size: size of input patches (square).
            n_orientations: number of orientation bins (8 = every 22.5°).
            n_frequencies: number of spatial frequency bands.
            top_k: number of active filters in sparse code.
        """
        self.patch_size = patch_size
        self.n_orientations = n_orientations
        self.n_frequencies = n_frequencies
        self.top_k = top_k
        self.n_filters = n_orientations * n_frequencies

        # Build Gabor filter bank.
        self.filters = self._build_filters()

    def _build_filters(self) -> np.ndarray:
        """Build the Gabor filter bank: (n_filters, patch_size, patch_size)."""
        ps = self.patch_size
        filters = []

        for fi in range(self.n_frequencies):
            # Spatial frequency: higher fi = finer detail.
            frequency = (fi + 1) * 0.15
            sigma = ps / (2.0 * (fi + 1))

            for oi in range(self.n_orientations):
                theta = oi * np.pi / self.n_orientations

                # Generate Gabor kernel.
                kernel = np.zeros((ps, ps), dtype=np.float32)
                center = ps / 2.0
                for y in range(ps):
                    for x in range(ps):
                        dx = x - center
                        dy = y - center
                        # Rotated coordinates.
                        x_theta = dx * np.cos(theta) + dy * np.sin(theta)
                        y_theta = -dx * np.sin(theta) + dy * np.cos(theta)
                        # Gabor: Gaussian × cosine.
                        gauss = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
                        kernel[y, x] = gauss * np.cos(2 * np.pi * frequency * x_theta)

                # Normalize.
                norm = np.sqrt(np.sum(kernel**2))
                if norm > 1e-6:
                    kernel /= norm
                filters.append(kernel)

        return np.array(filters, dtype=np.float32)

    @staticmethod
    def _normalize(flat_patch: np.ndarray) -> np.ndarray:
        """Zero-mean, unit-variance normalisation.

        Makes Gabor responses depend only on the spatial pattern of
        contrast within the patch, not on absolute brightness or
        overall contrast level.  After DoG preprocessing, interior
        (uniform) patches are already near-zero; normalization makes
        them consistently zero so they produce a stable, repeatable
        sparse code rather than amplifying numerical noise.

        Truly flat patches (std < 1e-6) are left as all-zeros — their
        code will be consistent across images (all zero input → same
        top-K by index), so they contribute correctly to overlap.
        """
        mean = flat_patch.mean()
        std  = flat_patch.std()
        if std > 1e-6:
            return (flat_patch - mean) / std
        return flat_patch - mean   # flat → zeros

    def encode(self, patch: np.ndarray) -> str:
        """Encode a single patch → sparse code string."""
        flat_patch = self._normalize(patch.flatten().astype(np.float32))
        flat_filters = self.filters.reshape(self.n_filters, -1)
        responses = np.abs(flat_filters @ flat_patch)
        top_idx = np.sort(np.argsort(responses)[-self.top_k:])
        return "g" + "_".join(str(i) for i in top_idx)

    def encode_batch(self, patches: np.ndarray) -> list[str]:
        """Encode batch of patches → list of sparse code strings."""
        n = patches.shape[0]
        flat = patches.reshape(n, -1).astype(np.float32)
        # Normalise each patch independently (zero-mean, unit-variance).
        # np.maximum avoids div-by-zero without triggering numpy warnings.
        means = flat.mean(axis=1, keepdims=True)
        flat  = flat - means
        stds  = np.sqrt((flat ** 2).mean(axis=1, keepdims=True))
        flat  = flat / np.maximum(stds, 1e-6)

        responses = np.abs(flat @ self.filters.reshape(self.n_filters, -1).T)
        codes = []
        for i in range(n):
            top_idx = np.sort(np.argsort(responses[i])[-self.top_k:])
            codes.append("g" + "_".join(str(j) for j in top_idx))
        return codes

    def fit(self, patches: np.ndarray = None, verbose: bool = True):
        """No-op: Gabor filters are fixed (not learned). API compat."""
        if verbose:
            print(f"  Gabor bank: {self.n_filters} filters "
                  f"({self.n_orientations} orientations x {self.n_frequencies} frequencies), "
                  f"top-{self.top_k} sparse code")

    def __repr__(self):
        return (f"GaborFilterBank(patch={self.patch_size}, "
                f"filters={self.n_filters}, top_k={self.top_k})")


class HOGEncoder:
    """Histogram of Oriented Gradients patch encoder.

    Designed for DoG-preprocessed (contrast-coded) input, exactly as V1
    simple cells receive from the retina.

    For each patch:
      1. Compute central-difference gradients (gx, gy).
      2. Compute unsigned orientation = arctan2(gy, gx) % π  →  [0, π).
         Unsigned because we want "horizontal edge" to code the same
         regardless of polarity (light-on-dark vs dark-on-light).
      3. Build an orientation histogram weighted by gradient magnitude.
      4. L2-normalise the histogram.
      5. Return the top-K dominant bin indices as a code string.

    Vocabulary size = C(n_bins, top_k).
      n_bins=8, top_k=3  →  56 codes.
      Gabor top-K gave C(24, 4) = 10,626 codes.
    The smaller vocabulary means far more code reuse across images of the
    same class, so the minicolumn WTA converges with far less training data.

    Flat patches (DoG ≈ 0, magnitude ≈ 0) produce a consistent all-zero
    gradient → stable code "h5_h6_h7" (argsort of all-zero vector) on
    every image.  They contribute reliably to overlap without adding noise.
    """

    def __init__(self, patch_size: int = 5,
                 n_bins: int = 8,
                 top_k: int = 3):
        self.patch_size = patch_size
        self.n_bins     = n_bins
        self.top_k      = top_k

    # ------------------------------------------------------------------

    def _hog(self, patch: np.ndarray) -> str:
        p = patch.astype(np.float32)

        # Central differences (forward/backward at borders).
        gx = np.empty_like(p)
        gy = np.empty_like(p)
        if p.shape[1] >= 2:
            gx[:, 1:-1] = p[:, 2:] - p[:, :-2]
            gx[:, 0]    = p[:, 1]  - p[:, 0]
            gx[:, -1]   = p[:, -1] - p[:, -2]
        else:
            gx[:] = 0.0
        if p.shape[0] >= 2:
            gy[1:-1, :] = p[2:, :] - p[:-2, :]
            gy[0, :]    = p[1, :]  - p[0, :]
            gy[-1, :]   = p[-1, :] - p[-2, :]
        else:
            gy[:] = 0.0

        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # Unsigned orientation in [0, π).
        angle = np.arctan2(gy, gx) % np.pi

        # Hard-bin: which of n_bins equally-spaced orientation bins?
        bin_idx = np.clip(
            (angle / np.pi * self.n_bins).astype(np.int32),
            0, self.n_bins - 1,
        )

        # Weighted histogram.
        hist = np.zeros(self.n_bins, dtype=np.float32)
        np.add.at(hist, bin_idx.ravel(), magnitude.ravel())

        # L2 normalise (+ ε to keep flat patches at zero, not inf).
        norm = np.sqrt((hist ** 2).sum()) + 1e-6
        hist /= norm

        # Top-K dominant bins → code string.
        top_idx = np.sort(np.argsort(hist)[-self.top_k:])
        return "h" + "_".join(str(i) for i in top_idx)

    def encode(self, patch: np.ndarray) -> str:
        return self._hog(patch)

    def encode_batch(self, patches: np.ndarray) -> list[str]:
        return [self._hog(p) for p in patches]

    def fit(self, patches: np.ndarray = None, verbose: bool = True):
        """No-op: HOG is parameter-free."""
        if verbose:
            from math import comb
            print(f"  HOG: {self.n_bins} orientation bins, "
                  f"top-{self.top_k}  "
                  f"(vocab = C({self.n_bins},{self.top_k}) = "
                  f"{comb(self.n_bins, self.top_k)} codes)")

    def __repr__(self) -> str:
        return (f"HOGEncoder(patch={self.patch_size}, "
                f"bins={self.n_bins}, top_k={self.top_k})")


# Keep PatchCodebook for backward compatibility.
class PatchCodebook(GaborFilterBank):
    """Alias for API compatibility. Uses Gabor filters instead of VQ."""

    def __init__(self, n_codes: int = 256, seed: int = 42,
                 patch_size: int = 5):
        # Map n_codes to reasonable Gabor parameters.
        n_ori = 8
        n_freq = max(2, n_codes // 64)
        top_k = 4
        super().__init__(patch_size=patch_size,
                         n_orientations=n_ori,
                         n_frequencies=n_freq,
                         top_k=top_k)
