"""Vision modality for SymbolicAI.

Provides biologically-justified visual primitives modelling early visual
cortex processing — an analogue of the innate visual system priors that
humans have from birth (retinal ganglion cells, V1 simple/complex cells).

All primitives are resolution-agnostic: they work for both CIFAR-10
(32×32) and full-screen 1920×1080 images (or any other size).

Biologically grounded primitives:
    img_load(path)              — load image from file path; returns np.ndarray
    img_from_array(arr)         — wrap an existing array as an image
    img_shape(img)              — (H, W) or (H, W, C) as a Python tuple
    img_height(img)             — H
    img_width(img)              — W
    img_channels(img)           — C (1 for grayscale, 3 for RGB)
    img_get(img, y, x)          — pixel value(s) at (y, x)
    img_resize(img, h, w)       — bilinear resize to h×w
    img_to_gray(img)            — ITU-R BT.601 luminance: 0.299R + 0.587G + 0.114B
    img_normalize(img)          — scale pixel values to [0.0, 1.0]
    img_crop(img, y0, x0, h, w) — spatial crop

Retinal model (center-surround):
    img_dog(img, sigma1, sigma2)
        — Difference of Gaussians (DoG): G(σ₁) − G(σ₂).
          Models retinal ganglion cell ON-center/OFF-surround receptive fields.
          sigma1 < sigma2 for ON-center; sigma1 > sigma2 for OFF-center.
          Returns float array of same shape, values in (−1, 1) after clipping.

V1 simple cells (oriented edge detection):
    img_gabor(img, theta, sigma, freq)
        — Real part of a 2-D Gabor filter.
          theta: orientation in radians (0 = horizontal)
          sigma: Gaussian envelope width in pixels
          freq:  spatial frequency in cycles/pixel
          Returns float array of same shape as input.

V1 complex cells (phase-invariant energy):
    img_gabor_energy(img, theta, sigma, freq)
        — sqrt(Re(Gabor)² + Im(Gabor)²) — phase-invariant energy.
          Corresponds to complex cell responses in V1.
          Returns non-negative float array.

    img_max_pool(img, pool_h, pool_w)
        — Spatial max-pooling with kernel (pool_h, pool_w); no padding.
          Models spatial summation / sub-sampling in LGN and V1.
          pool_h and pool_w must divide img height and width exactly,
          or are silently rounded down.

Statistical summaries:
    img_mean(img)               — scalar mean over all pixels
    img_std(img)                — scalar standard deviation
    img_patch(img, y, x, h, w)  — sub-image crop (same as img_crop)
    img_flatten(img)            — 1-D float list (row-major)

Dependencies: numpy (always available), scipy (optional, used for convolve2d).
If scipy is unavailable, a pure-numpy fallback is used (slower but correct).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import convolve as _scipy_convolve
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from modalities.base import Modality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(img: np.ndarray) -> np.ndarray:
    """Return float32 copy of img, scaling uint8 [0,255] → [0.0, 1.0]."""
    arr = np.asarray(img, dtype=np.float32)
    if img.dtype == np.uint8:
        arr = arr / 255.0
    return arr


def _ensure_2d(img: np.ndarray) -> np.ndarray:
    """Return 2-D array (H, W); collapse colour channels if present."""
    if img.ndim == 3:
        # ITU-R luminance
        return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
    return np.asarray(img, dtype=np.float32)


def _gaussian_kernel_2d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Build a 2-D Gaussian kernel with standard deviation sigma."""
    radius = max(1, int(math.ceil(truncate * sigma)))
    size = 2 * radius + 1
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g1d = np.exp(-0.5 * (x / sigma) ** 2)
    g1d /= g1d.sum()
    kernel = np.outer(g1d, g1d)
    return kernel.astype(np.float32)


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve 2-D image with kernel (reflect padding)."""
    if _HAS_SCIPY:
        return _scipy_convolve(img, kernel, mode='reflect').astype(np.float32)
    # Pure-numpy fallback: manual reflect padding + np.convolve via reshape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
    H, W = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + H, j:j + W]
    return out


def _gabor_kernel(theta: float, sigma: float, freq: float,
                  n_stds: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (real, imag) 2-D Gabor kernels."""
    radius = max(1, int(math.ceil(n_stds * sigma)))
    size = 2 * radius + 1
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x_r = x * cos_t + y * sin_t
    y_r = -x * sin_t + y * cos_t
    envelope = np.exp(-0.5 * (x_r ** 2 + y_r ** 2) / sigma ** 2)
    carrier_re = np.cos(2 * math.pi * freq * x_r)
    carrier_im = np.sin(2 * math.pi * freq * x_r)
    return (envelope * carrier_re).astype(np.float32), (envelope * carrier_im).astype(np.float32)


# ---------------------------------------------------------------------------
# VisionModality
# ---------------------------------------------------------------------------

class VisionModality(Modality):
    """Biologically-grounded visual primitives for SymbolicAI.

    Works for any image size: CIFAR-10 (32×32), ImageNet (224×224),
    or full-screen (1920×1080).  All operations are spatial, not
    architecture-specific.

    Typical usage in a process block:
        img    = img_load(path)
        gray   = img_to_gray(img)
        norm   = img_normalize(gray)
        dog    = img_dog(norm, 1.0, 2.0)
        gabors = img_gabor(norm, 0.0, 2.0, 0.2)
        energy = img_gabor_energy(norm, 0.0, 2.0, 0.2)
        feat   = img_mean(energy)
        emit(feat)
    """

    @property
    def name(self) -> str:
        return 'vision'

    @property
    def primitives(self) -> Dict[str, Callable]:
        return {
            # I/O
            'img_load':         self._img_load,
            'img_from_array':   self._img_from_array,
            # Shape
            'img_shape':        self._img_shape,
            'img_height':       self._img_height,
            'img_width':        self._img_width,
            'img_channels':     self._img_channels,
            # Pixel access
            'img_get':          self._img_get,
            # Spatial transforms
            'img_resize':       self._img_resize,
            'img_crop':         self._img_crop,
            'img_patch':        self._img_crop,    # alias
            'img_flatten':      self._img_flatten,
            # Colour / normalisation
            'img_to_gray':      self._img_to_gray,
            'img_normalize':    self._img_normalize,
            # Retinal model
            'img_dog':          self._img_dog,
            # V1 simple cells
            'img_gabor':        self._img_gabor,
            # V1 complex cells
            'img_gabor_energy': self._img_gabor_energy,
            'img_max_pool':     self._img_max_pool,
            # Statistics
            'img_mean':         self._img_mean,
            'img_std':          self._img_std,
            # ── Innate face understanding ──────────────────────────────
            # Biologically equivalent to newborn face preference (Goren 1975)
            # and subcortical face detection (superior colliculus pathway).
            # These are "built in" like succ/pred — not learned.
            'img_face_schematic':   self._img_face_schematic,
            'img_face_detect':      self._img_face_detect,
            'img_face_align':       self._img_face_align,
            'img_eye_response':     self._img_eye_response,
            'img_face_score':       self._img_face_score,
        }

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _img_load(self, path: str) -> np.ndarray:
        """Load image from file path.  Returns float32 H×W or H×W×C array."""
        if _HAS_PIL:
            pil = _PILImage.open(str(path))
            return _to_float(np.array(pil))
        # Fallback: try numpy/matplotlib
        try:
            import matplotlib.pyplot as plt
            arr = plt.imread(str(path))
            return _to_float(arr)
        except Exception as exc:
            raise RuntimeError(
                f"img_load: cannot load {path!r}. "
                "Install Pillow (pip install Pillow) for full support."
            ) from exc

    def _img_from_array(self, arr: Any) -> np.ndarray:
        """Wrap a numpy array (or list) as a float32 image."""
        return _to_float(np.asarray(arr))

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def _img_shape(self, img: np.ndarray) -> tuple:
        return tuple(int(d) for d in img.shape)

    def _img_height(self, img: np.ndarray) -> int:
        return int(img.shape[0])

    def _img_width(self, img: np.ndarray) -> int:
        return int(img.shape[1])

    def _img_channels(self, img: np.ndarray) -> int:
        return int(img.shape[2]) if img.ndim == 3 else 1

    # ------------------------------------------------------------------
    # Pixel access
    # ------------------------------------------------------------------

    def _img_get(self, img: np.ndarray, y: int, x: int) -> Any:
        """Return pixel value(s) at position (y, x)."""
        val = img[int(y), int(x)]
        if isinstance(val, np.ndarray):
            return tuple(float(v) for v in val)
        return float(val)

    # ------------------------------------------------------------------
    # Spatial transforms
    # ------------------------------------------------------------------

    def _img_resize(self, img: np.ndarray, h: int, w: int) -> np.ndarray:
        """Bilinear resize to h×w.  Handles both grayscale and colour."""
        h, w = int(h), int(w)
        if _HAS_PIL:
            arr = (img * 255).clip(0, 255).astype(np.uint8)
            if arr.ndim == 2:
                pil = _PILImage.fromarray(arr, mode='L')
            else:
                pil = _PILImage.fromarray(arr, mode='RGB')
            pil = pil.resize((w, h), _PILImage.BILINEAR)
            return _to_float(np.array(pil))
        # Numpy fallback — nearest-neighbour
        H0, W0 = img.shape[:2]
        ys = (np.arange(h) * H0 / h).astype(int)
        xs = (np.arange(w) * W0 / w).astype(int)
        return img[np.ix_(ys, xs)].astype(np.float32)

    def _img_crop(self, img: np.ndarray, y0: int, x0: int,
                  h: int, w: int) -> np.ndarray:
        y0, x0, h, w = int(y0), int(x0), int(h), int(w)
        return img[y0:y0 + h, x0:x0 + w].astype(np.float32)

    def _img_flatten(self, img: np.ndarray) -> list:
        return img.astype(np.float32).flatten().tolist()

    # ------------------------------------------------------------------
    # Colour / normalisation
    # ------------------------------------------------------------------

    def _img_to_gray(self, img: np.ndarray) -> np.ndarray:
        """ITU-R BT.601 luminance — same coefficients as standard TV."""
        arr = _to_float(img)
        return _ensure_2d(arr)

    def _img_normalize(self, img: np.ndarray) -> np.ndarray:
        """Scale to [0.0, 1.0].  No-op if already in range."""
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            return (arr - lo) / (hi - lo)
        return np.zeros_like(arr)

    # ------------------------------------------------------------------
    # Retinal model — Difference of Gaussians
    # ------------------------------------------------------------------

    def _img_dog(self, img: np.ndarray,
                 sigma1: float, sigma2: float) -> np.ndarray:
        """Difference of Gaussians: G(σ₁) − G(σ₂).

        Models ON-center retinal ganglion cells (σ₁ < σ₂).
        Both sigmas are in pixels.  Output is clipped to [−1, 1].
        """
        arr = _ensure_2d(_to_float(img))
        k1 = _gaussian_kernel_2d(float(sigma1))
        k2 = _gaussian_kernel_2d(float(sigma2))
        blurred1 = _convolve2d(arr, k1)
        blurred2 = _convolve2d(arr, k2)
        dog = blurred1 - blurred2
        return np.clip(dog, -1.0, 1.0)

    # ------------------------------------------------------------------
    # V1 simple cells — Gabor filter (oriented, phase-sensitive)
    # ------------------------------------------------------------------

    def _img_gabor(self, img: np.ndarray,
                   theta: float, sigma: float, freq: float) -> np.ndarray:
        """Real-part Gabor filter response.

        theta: orientation in radians (0 = horizontal grating)
        sigma: Gaussian envelope std in pixels
        freq:  spatial frequency in cycles/pixel (e.g. 0.2 for 5px period)
        """
        arr = _ensure_2d(_to_float(img))
        k_re, _ = _gabor_kernel(float(theta), float(sigma), float(freq))
        return _convolve2d(arr, k_re)

    # ------------------------------------------------------------------
    # V1 complex cells — phase-invariant Gabor energy
    # ------------------------------------------------------------------

    def _img_gabor_energy(self, img: np.ndarray,
                          theta: float, sigma: float, freq: float) -> np.ndarray:
        """Phase-invariant Gabor energy: sqrt(Re² + Im²).

        Corresponds to complex cell responses in primary visual cortex (V1).
        Output is non-negative; values represent local edge energy at
        orientation theta and spatial frequency freq.
        """
        arr = _ensure_2d(_to_float(img))
        k_re, k_im = _gabor_kernel(float(theta), float(sigma), float(freq))
        resp_re = _convolve2d(arr, k_re)
        resp_im = _convolve2d(arr, k_im)
        return np.sqrt(resp_re ** 2 + resp_im ** 2)

    def _img_max_pool(self, img: np.ndarray,
                      pool_h: int, pool_w: int) -> np.ndarray:
        """Spatial max-pooling with kernel (pool_h, pool_w).

        Output shape: (H // pool_h, W // pool_w).
        Models spatial sub-sampling in LGN and striate cortex.
        """
        arr = _ensure_2d(_to_float(img))
        ph, pw = int(pool_h), int(pool_w)
        H, W = arr.shape
        H2, W2 = H // ph, W // pw
        arr = arr[:H2 * ph, :W2 * pw]
        return arr.reshape(H2, ph, W2, pw).max(axis=(1, 3))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _img_mean(self, img: np.ndarray) -> float:
        return float(np.asarray(img, dtype=np.float32).mean())

    def _img_std(self, img: np.ndarray) -> float:
        return float(np.asarray(img, dtype=np.float32).std())

    # ------------------------------------------------------------------
    # Innate face understanding
    #
    # Models the subcortical face detection pathway present from birth:
    #   Morton & Johnson (1991) — newborns prefer face-like configurations
    #   Goren et al. (1975) — newborns track schematic faces, not scrambled
    #   Johnson et al. (1991) — superior colliculus, not cortex, drives this
    #
    # The "face template" is the 2-dark-above-1-dark configuration.
    # This is innate prior knowledge, not learned — treating it like
    # succ/pred in the arithmetic domain.
    # ------------------------------------------------------------------

    def _img_face_schematic(self, patch: np.ndarray) -> float:
        """Score how well the patch matches the schematic face template.

        Template: two dark regions (eyes) in the upper half,
                  one dark region (mouth/nose) in the lower half,
                  on a lighter background.

        This is the innate pattern newborns prefer (Goren et al. 1975).
        Returns a confidence score in [0, 1].
        """
        arr = _ensure_2d(_to_float(patch))
        arr = self._img_resize(arr, 32, 32)  # canonical scale
        H, W = arr.shape

        # Invert: dark regions become bright (eyes/mouth are dark features)
        inv = 1.0 - arr

        # Smooth to find blobs (eye scale ≈ W/8)
        eye_sigma = max(1.0, W / 8.0)
        k_eye = _gaussian_kernel_2d(eye_sigma)
        smooth = _convolve2d(inv, k_eye)

        # Top half: expect two peaks — one in left quarter, one in right
        top = smooth[:H // 2, :]
        left_eye_score  = float(top[:, :W // 2].max())
        right_eye_score = float(top[:, W // 2:].max())

        # Bottom half: expect one central peak — nose/mouth
        bot = smooth[H // 2:, :]
        mouth_score = float(bot[:, W // 4: 3 * W // 4].max())

        # Both eyes must be present (symmetry requirement)
        symmetry  = 1.0 - abs(left_eye_score - right_eye_score)
        eye_score = min(left_eye_score, right_eye_score)  # both needed

        # Geometric mean — all three components must be present
        product = eye_score * mouth_score * max(0.0, symmetry)
        return float(product ** (1.0 / 3.0)) if product > 0 else 0.0

    def _img_face_detect(self, img: np.ndarray) -> tuple:
        """Detect the most face-like region in the image.

        Models subcortical (superior colliculus) face detection:
        operates on low-resolution, coarse spatial information.

        Returns (confidence, norm_y, norm_x) where norm_y/x ∈ [0,1]
        are the approximate face-center location as fractions of H, W.
        """
        arr = _ensure_2d(_to_float(img))

        # Work at coarse scale (subcortical processing is low-res)
        small = self._img_resize(arr, 32, 32)
        H, W = small.shape
        size = 16  # detection window = half the downsampled image

        best_score = 0.0
        best_cy    = H / 2.0
        best_cx    = W / 2.0

        for y in range(0, H - size + 1, 4):
            for x in range(0, W - size + 1, 4):
                patch = small[y:y + size, x:x + size]
                score = self._img_face_schematic(patch)
                if score > best_score:
                    best_score = score
                    best_cy    = y + size / 2.0
                    best_cx    = x + size / 2.0

        return float(best_score), float(best_cy / H), float(best_cx / W)

    def _img_face_align(self, img: np.ndarray,
                        center_y_norm: float,
                        center_x_norm: float,
                        size_norm: float = 0.5) -> np.ndarray:
        """Extract and normalize a face region to a canonical 32×32 patch.

        center_y_norm, center_x_norm: face center as fractions of image H, W
        size_norm: face box half-width as fraction of image width

        Implements the crop + rescale step of face canonicalization,
        analogous to cortical face normalization (FFA input normalization).
        """
        arr = _to_float(img)
        H, W = arr.shape[:2]

        cy   = int(float(center_y_norm) * H)
        cx   = int(float(center_x_norm) * W)
        half = max(8, int(float(size_norm) * W))

        y0 = max(0, cy - half)
        x0 = max(0, cx - half)
        y1 = min(H, cy + half)
        x1 = min(W, cx + half)

        patch = arr[y0:y1, x0:x1]
        return self._img_resize(patch, 32, 32)

    def _img_eye_response(self, patch: np.ndarray) -> float:
        """DoG response in the upper third of a (face) patch at eye scale.

        Models the V1/V2 response to eye-like features:
        high center-surround contrast at the correct scale and location.
        Returns mean absolute DoG response in the expected eye region.
        """
        arr = _ensure_2d(_to_float(patch))
        H, W = arr.shape
        upper = arr[:H // 3, :]  # eyes are in upper 1/3 of face
        dog   = self._img_dog(upper, 1.0, 3.0)  # eye-scale DoG
        return float(np.abs(dog).mean())

    def _img_face_score(self, patch: np.ndarray) -> float:
        """Composite face quality score combining schematic + eye response.

        Uses both the innate schematic template and the V1 eye response.
        This is a stronger signal than either alone and models the early
        cortical enhancement of subcortical face detection signals.
        """
        schematic = self._img_face_schematic(patch)
        eye_resp  = self._img_eye_response(patch)
        # Normalize eye response to [0, 1] (typical range 0–0.3)
        eye_norm  = min(1.0, eye_resp / 0.3)
        return float((schematic + eye_norm) / 2.0)

    # ------------------------------------------------------------------
    # preprocess hook
    # ------------------------------------------------------------------

    def preprocess(self, raw: Any) -> np.ndarray:
        """Convert raw input (path, list, or array) to float32 array."""
        if isinstance(raw, str):
            return self._img_load(raw)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.dtype == np.uint8 or arr.max() > 1.0:
            arr = arr / 255.0
        return arr.astype(np.float32)
