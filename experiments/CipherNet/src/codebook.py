"""Visual feature encoding — 2048-bit SDR output.

FEATURE REPRESENTATION
----------------------
All encoders return np.ndarray(dtype=int8, shape=(n_filters,)) — a binary
Sparse Distributed Representation (SDR) with exactly top_k active bits.
Silent patches (insufficient gradient energy) return None.

WHY 2048 BITS / 40 ACTIVE
--------------------------
Monty (Numenta) uses 2048-bit SDRs with 40 active bits (1.95% sparsity).
The sparsity level is critical for the union model to work:

  P(bit j still 0 after n training examples) = (1 - top_k/n_filters)^n

  With 8-bit/3-active HOG (37.5% density):
    n=5 → P ≈ 9%  — union saturates after ~5 examples
    All 10 class unions become all-1s → every test SDR scores 1.0 everywhere
    → winner = highest boost = class with fewest training examples

  With 2048-bit/40-active Gabor (1.95% density):
    n=100 → P ≈ 14%  — union retains ~86% zeros after 100 examples
    n=8   → P ≈ 85%  — after 8 examples, only 15% of bits set
    Different classes produce distinct unions → overlap is discriminative

WHY GABOR FILTERS
-----------------
V1 simple cells are Gabor filters: they respond to oriented edges at a
specific spatial frequency, phase, and retinal position.  The V1 population
code for a patch is the binary activity vector across hundreds of such cells.
Only the cells whose tuning closely matches the patch's dominant gradient fire
— naturally sparse.

To replicate this with 2048 filters in a 5×5 patch:
  16 orientations (every 11.25°)
  ×  8 spatial frequencies (log-spaced, 2px–4px wavelength)
  ×  2 phases (0° cosine / 90° sine — quadrature pair, ON/OFF-like)
  ×  8 sub-positions (slightly offset filter centres within the patch,
                      modelling different V1 cells covering same region)
  = 2048 filters exactly.

BLANK PATCH SUPPRESSION (STRUCTURAL)
-------------------------------------
encode() returns None when normalised patch energy < ACTIVITY_THRESHOLD.
After zero-mean/unit-variance normalisation, a truly flat patch has all
pixels ≈ 0 → all Gabor responses ≈ 0 → below threshold → None.
No cortex-level conditional needed.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# GaborFilterBank — primary encoder, 2048-bit SDR
# ---------------------------------------------------------------------------

class GaborFilterBank:
    """2048-filter Gabor bank → 2048-bit SDR with 40 active bits.

    Biologically grounded V1 population code:
      16 orientations × 8 frequencies × 2 phases × 8 sub-positions = 2048.

    For each patch:
      1. Zero-mean / unit-variance normalisation.
      2. Compute |response| of all 2048 filters (vectorised matmul).
      3. Return None if mean |response| < ACTIVITY_THRESHOLD (blank patch).
      4. Set top-40 response indices to 1 in a 2048-bit int8 array.

    The filter bank is fixed (not learned) and built once in __init__.
    """

    ACTIVITY_THRESHOLD: float = 0.02

    def __init__(self,
                 patch_size:     int = 5,
                 n_orientations: int = 16,
                 n_frequencies:  int = 8,
                 n_phases:       int = 2,
                 n_positions:    int = 8,
                 top_k:          int = 40):
        self.patch_size     = patch_size
        self.n_orientations = n_orientations
        self.n_frequencies  = n_frequencies
        self.n_phases       = n_phases
        self.n_positions    = n_positions
        self.top_k          = top_k
        self.n_filters      = n_orientations * n_frequencies * n_phases * n_positions
        # filters: (n_filters, patch_size²) — used as flat dot-product bank
        self.filters = self._build_filters()  # (2048, 25) for default params

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    @staticmethod
    def _make_positions(n_pos: int, ps: int) -> list[tuple[float, float]]:
        """n_pos (y, x) centres spread over the patch interior [0.5, ps-0.5]²."""
        side   = int(np.ceil(np.sqrt(n_pos)))
        coords = np.linspace(0.5, ps - 0.5, side, dtype=np.float32)
        pts    = [(float(y), float(x)) for y in coords for x in coords]
        return pts[:n_pos]

    def _build_filters(self) -> np.ndarray:
        """Build all n_filters Gabor kernels as a (n_filters, ps²) matrix.

        Outer loops: sub-position → frequency → orientation → phase.
        Each kernel is L2-normalised to unit energy so response magnitude
        is comparable across filters.
        """
        ps        = self.patch_size
        positions = self._make_positions(self.n_positions, ps)

        # Log-spaced wavelengths: 2px (fine) → ps*0.8 px (coarse)
        lambdas = np.logspace(np.log10(2.0),
                              np.log10(ps * 0.8),
                              self.n_frequencies,
                              dtype=np.float32)

        # Phase offsets: 0 (cosine/even) and π/2 (sine/odd)
        phases = np.linspace(0.0, np.pi, self.n_phases,
                             endpoint=False, dtype=np.float32)

        # Pixel grid (vectorised over all patch pixels simultaneously)
        yi, xi = np.meshgrid(np.arange(ps, dtype=np.float32),
                             np.arange(ps, dtype=np.float32),
                             indexing='ij')          # (ps, ps)
        y_flat = yi.ravel()   # (ps²,)
        x_flat = xi.ravel()   # (ps²,)

        all_kernels: list[np.ndarray] = []

        for cy, cx in positions:
            dy = y_flat - cy   # (ps²,)
            dx = x_flat - cx   # (ps²,)

            for lam in lambdas:
                sigma = float(lam) * 0.56   # bandwidth ≈ 1 octave
                freq  = 1.0 / float(lam)

                for oi in range(self.n_orientations):
                    theta  = oi * np.pi / self.n_orientations
                    cos_t  = np.cos(theta)
                    sin_t  = np.sin(theta)
                    x_th   =  dx * cos_t + dy * sin_t    # rotated x
                    y_th   = -dx * sin_t + dy * cos_t    # rotated y

                    # Gaussian envelope (aspect ratio γ=0.5 — elongated)
                    gauss = np.exp(
                        -(x_th ** 2 + (0.5 * y_th) ** 2)
                        / (2.0 * sigma ** 2)
                    )

                    for phase in phases:
                        kernel = gauss * np.cos(
                            2.0 * np.pi * freq * x_th + phase)
                        norm = np.sqrt((kernel ** 2).sum())
                        if norm > 1e-8:
                            kernel = kernel / norm
                        all_kernels.append(kernel.astype(np.float32))

        assert len(all_kernels) == self.n_filters, (
            f"Expected {self.n_filters} filters, built {len(all_kernels)}")
        return np.array(all_kernels, dtype=np.float32)  # (n_filters, ps²)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, patch: np.ndarray) -> np.ndarray | None:
        """Encode one patch → 2048-bit SDR or None (blank)."""
        flat = patch.ravel().astype(np.float32)
        mean = flat.mean()
        flat = flat - mean
        std  = np.sqrt((flat ** 2).mean())
        if std > 1e-6:
            flat = flat / std

        if np.abs(flat).mean() < self.ACTIVITY_THRESHOLD:
            return None

        responses = np.abs(self.filters @ flat)   # (n_filters,)
        sdr = np.zeros(self.n_filters, dtype=np.int8)
        sdr[np.argsort(responses)[-self.top_k:]] = 1
        return sdr

    def encode_batch(self, patches: np.ndarray) -> list[np.ndarray | None]:
        """Encode a batch of patches → list[2048-bit SDR | None].

        Vectorised: one matmul for the entire batch.
        """
        n    = patches.shape[0]
        flat = patches.reshape(n, -1).astype(np.float32)

        # Per-patch zero-mean / unit-variance normalisation
        means = flat.mean(axis=1, keepdims=True)
        flat  = flat - means
        stds  = np.sqrt((flat ** 2).mean(axis=1, keepdims=True))
        flat  = flat / np.maximum(stds, 1e-6)

        # Blank-patch energy check (scalar per patch)
        energy = np.abs(flat).mean(axis=1)           # (n,)

        # Batch response: (n, n_filters)
        responses = np.abs(flat @ self.filters.T)    # (n, 2048)

        codes: list[np.ndarray | None] = []
        for i in range(n):
            if energy[i] < self.ACTIVITY_THRESHOLD:
                codes.append(None)
            else:
                sdr = np.zeros(self.n_filters, dtype=np.int8)
                sdr[np.argsort(responses[i])[-self.top_k:]] = 1
                codes.append(sdr)
        return codes

    def fit(self, patches: np.ndarray = None, verbose: bool = True) -> None:
        """No-op: Gabor filters are fixed (not learned)."""
        if verbose:
            sparsity = self.top_k / self.n_filters * 100
            print(f"  Gabor bank: {self.n_filters} filters "
                  f"({self.n_orientations} ori x {self.n_frequencies} freq x "
                  f"{self.n_phases} phase x {self.n_positions} pos), "
                  f"top-{self.top_k} -> {self.n_filters}-bit SDR, "
                  f"{sparsity:.1f}% sparsity")

    def __repr__(self) -> str:
        return (f"GaborFilterBank(patch={self.patch_size}, "
                f"filters={self.n_filters}, top_k={self.top_k})")


# ---------------------------------------------------------------------------
# HOGEncoder — retained for reference; not used in primary configs
# ---------------------------------------------------------------------------

class HOGEncoder:
    """8-bin HOG encoder.

    DEPRECATED as primary encoder: 8-bit SDR (37.5% density) saturates
    the union model after ~5 training examples.  Use GaborFilterBank.

    Retained for ablation comparisons only.
    """

    ACTIVITY_THRESHOLD: float = 0.02

    def __init__(self, patch_size: int = 5,
                 n_bins: int = 8,
                 top_k: int = 3):
        self.patch_size = patch_size
        self.n_bins     = n_bins
        self.top_k      = top_k

    def _hog(self, patch: np.ndarray) -> np.ndarray | None:
        p  = patch.astype(np.float32)
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
        if magnitude.mean() < self.ACTIVITY_THRESHOLD:
            return None
        angle   = np.arctan2(gy, gx) % np.pi
        bin_idx = np.clip(
            (angle / np.pi * self.n_bins).astype(np.int32),
            0, self.n_bins - 1)
        hist = np.zeros(self.n_bins, dtype=np.float32)
        np.add.at(hist, bin_idx.ravel(), magnitude.ravel())
        norm = np.sqrt((hist ** 2).sum()) + 1e-6
        hist /= norm
        sdr = np.zeros(self.n_bins, dtype=np.int8)
        sdr[np.argsort(hist)[-self.top_k:]] = 1
        return sdr

    def encode(self, patch: np.ndarray) -> np.ndarray | None:
        return self._hog(patch)

    def encode_batch(self, patches: np.ndarray) -> list[np.ndarray | None]:
        return [self._hog(p) for p in patches]

    def fit(self, patches: np.ndarray = None, verbose: bool = True) -> None:
        if verbose:
            from math import comb
            print(f"  HOG (deprecated): {self.n_bins} bins, top-{self.top_k} "
                  f"→ {self.n_bins}-bit SDR, "
                  f"{self.top_k/self.n_bins*100:.0f}% sparsity "
                  f"[USE GaborFilterBank FOR PRODUCTION]")

    def __repr__(self) -> str:
        return (f"HOGEncoder(patch={self.patch_size}, "
                f"bins={self.n_bins}, top_k={self.top_k})")


# Backward-compat alias
class PatchCodebook(GaborFilterBank):
    def __init__(self, n_codes: int = 256, seed: int = 42,
                 patch_size: int = 5):
        super().__init__(patch_size=patch_size)
