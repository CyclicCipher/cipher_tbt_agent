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

    def encode(self, patch: np.ndarray) -> str:
        """Encode a single patch → sparse code string.

        Computes response to all Gabor filters, takes top-K active
        filter IDs as the sparse code.
        """
        flat_patch = patch.flatten().astype(np.float32)
        flat_filters = self.filters.reshape(self.n_filters, -1)
        responses = np.abs(flat_filters @ flat_patch)  # rectified response

        # Top-K sparse code.
        top_idx = np.argsort(responses)[-self.top_k:]
        top_idx = np.sort(top_idx)  # canonical ordering
        return "g" + "_".join(str(i) for i in top_idx)

    def encode_batch(self, patches: np.ndarray) -> list[str]:
        """Encode batch of patches → list of sparse code strings.

        Vectorized: all patches × all filters in one matmul.
        """
        n = patches.shape[0]
        flat_patches = patches.reshape(n, -1).astype(np.float32)
        flat_filters = self.filters.reshape(self.n_filters, -1)

        # (n_patches, n_filters) response matrix.
        responses = np.abs(flat_patches @ flat_filters.T)

        # Top-K per patch.
        codes = []
        for i in range(n):
            top_idx = np.argsort(responses[i])[-self.top_k:]
            top_idx = np.sort(top_idx)
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
