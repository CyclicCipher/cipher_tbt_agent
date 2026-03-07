"""vision_cortex.py — Fixed retinotopic V1 processing pipeline.

Architecture
------------
This is infrastructure, not knowledge.  It implements the fixed portion of
the visual hierarchy (up to and including V1) as a deterministic, parameter-
free computation.  CTKG concepts sit above this and learn to interpret the
feature map produced here.

The hard break point: everything below is wired at birth (like primate V1);
everything above is learned from experience (like primate IT).

Pipeline
--------
  Raw frame  (H, W, 3) uint8
      |
  Grayscale  (H, W) float32
      |
  Gabor bank  16 filters (4 orientations × 4 scales) applied to full frame
      →  gabor_maps: (16, H, W)
      |
  Log-polar sampling  centered on gaze (default = frame center)
      →  feature_map: (n_rings, n_angles, 16) — the retinotopic V1 output
      |
  CTKG concepts take over here (texture_patch, block_face, block_type, ...)

Log-polar geometry
------------------
Ring 0    = 1 pixel from gaze     → fovea centralis (highest resolution)
Ring N-1  = r_max pixels from gaze → peripheral vision (lowest resolution)
Radii are log-spaced: each ring represents ~twice the area of the previous.

This gives foveal magnification (more rings near center) and makes the
representation invariant to scale (a block at 2m vs 4m shifts radially
by a fixed amount, independent of distance).

Gaze = frame center
-------------------
In Minecraft, the crosshair is always at the frame center.  The agent
changes what is fixated by issuing mc_turn(Δpitch, Δyaw) commands.
So vc_encode(frame) always uses (H//2, W//2) as the gaze point.
vc_pixel_to_look(y, x) converts a pixel coordinate to the mc_turn delta
needed to bring that point to the center.

Saliency (dorsal stream)
------------------------
vc_saliency(frame, prev_frame) returns a (H, W) map combining:
  - Edge saliency:  DoG magnitude (retinal ON-center response)
  - Motion saliency: |frame − prev_frame| (temporal difference)
  - Optional top-down bias: hue match amplifies task-relevant regions

Two-stream summary
------------------
  Ventral (what):  vc_encode → CTKG texture_patch → block_face → block_type
  Dorsal  (where): vc_saliency → vc_next_gaze → mc_turn (saccade)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import convolve as _scipy_convolve
    from scipy.ndimage import map_coordinates as _scipy_map_coords
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Public constants (used by MinecraftModality for FOV conversion)
# ---------------------------------------------------------------------------

# Minecraft default FOV (degrees).  Used to convert pixel offsets → turn deltas.
MINECRAFT_VFOV_DEG = 70.0
MINECRAFT_HFOV_DEG = 80.0

# Default log-polar parameters (good balance for 160×256 frames)
DEFAULT_N_RINGS  = 32
DEFAULT_N_ANGLES = 64


# ---------------------------------------------------------------------------
# VisualCortex
# ---------------------------------------------------------------------------

class VisualCortex:
    """Fixed retinotopic V1 processing: log-polar sampling + Gabor bank.

    Instantiate once; the precomputed grids are reused every step.

    Args:
        frame_h:   Frame height in pixels  (default 160)
        frame_w:   Frame width in pixels   (default 256)
        n_rings:   Number of log-polar rings  (default 32)
        n_angles:  Number of angular samples  (default 64)
    """

    def __init__(
        self,
        frame_h:  int = 160,
        frame_w:  int = 256,
        n_rings:  int = DEFAULT_N_RINGS,
        n_angles: int = DEFAULT_N_ANGLES,
    ) -> None:
        self._h        = frame_h
        self._w        = frame_w
        self._n_rings  = n_rings
        self._n_angles = n_angles

        # Gaze = frame center (crosshair is always at center in Minecraft)
        self._gaze_y = frame_h / 2.0
        self._gaze_x = frame_w / 2.0

        # Precomputed log-polar sample coordinates (absolute pixel positions)
        # Shape: (n_rings, n_angles) each — float32
        self._sample_y, self._sample_x = _build_logpolar_coords(
            frame_h, frame_w, n_rings, n_angles,
            self._gaze_y, self._gaze_x,
        )

        # Precomputed Gabor filter bank: 4 orientations × 4 scales = 16 filters
        self._gabors = _build_gabor_bank()   # list of 16 (kh, kw) float32 arrays

    # -------------------------------------------------------------------------
    # Ventral stream: encode
    # -------------------------------------------------------------------------

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """V1 encode: log-polar Gabor feature map centred on gaze (frame center).

        Args:
            frame: (H, W, 3) uint8 or float32 RGB frame.

        Returns:
            features: (n_rings, n_angles, 16) float32
                      Retinotopic feature map.  Axis 0 = ring (fovea→periphery),
                      axis 1 = angle, axis 2 = Gabor channel.
        """
        gray = _to_gray_f32(frame)                        # (H, W)
        gabor_maps = _apply_gabor_bank(gray, self._gabors) # (16, H, W)

        features = np.empty(
            (self._n_rings, self._n_angles, 16), dtype=np.float32
        )
        for c in range(16):
            features[..., c] = _bilinear_sample(
                gabor_maps[c], self._sample_y, self._sample_x
            )
        return features

    # -------------------------------------------------------------------------
    # Dorsal stream: saliency + gaze control
    # -------------------------------------------------------------------------

    def saliency(
        self,
        frame:      np.ndarray,
        prev_frame: np.ndarray,
        target_hue: Optional[float] = None,
    ) -> np.ndarray:
        """Compute (H, W) saliency map for saccade planning.

        Combines edge saliency (DoG) and motion saliency (frame diff).
        Optional top-down bias: pass target_hue in [0, 1] (HSV) to amplify
        regions matching the current goal colour (e.g. 0.08 for wood brown).

        Args:
            frame:      Current frame  (H, W, 3) uint8 or float32.
            prev_frame: Previous frame (H, W, 3) uint8 or float32.
            target_hue: HSV hue in [0, 1] or None.

        Returns:
            sal: (H, W) float32, values ≥ 0.
        """
        gray      = _to_gray_f32(frame)       # (H, W) in [0, 1]
        prev_gray = _to_gray_f32(prev_frame)  # (H, W) in [0, 1]

        # Edge saliency: absolute DoG response (retinal ON + OFF centers)
        dog       = _dog(gray, sigma1=1.0, sigma2=3.0)
        edge_sal  = np.abs(dog)

        # Motion saliency: absolute frame difference
        motion_sal = np.abs(gray - prev_gray)

        sal = edge_sal + motion_sal

        # Top-down hue bias (goal-directed attention)
        if target_hue is not None:
            hue_map = _rgb_hue(frame)           # (H, W) in [0, 1]
            # Gaussian bump around target hue (bandwidth = 0.05)
            hue_match = np.exp(-((hue_map - target_hue) ** 2) / (2 * 0.05 ** 2))
            sal = sal * (1.0 + hue_match)

        return sal.astype(np.float32)

    def next_gaze(self, saliency_map: np.ndarray) -> Tuple[int, int]:
        """Return (y, x) pixel of peak saliency — next saccade target.

        Args:
            saliency_map: (H, W) float32 from self.saliency().

        Returns:
            (y, x) integer pixel coordinates.
        """
        idx = int(np.argmax(saliency_map))
        y, x = divmod(idx, saliency_map.shape[1])
        return y, x

    def pixel_to_look(self, target_y: int, target_x: int) -> Tuple[float, float]:
        """Convert a pixel coordinate to mc_turn(delta_pitch, delta_yaw) args.

        The crosshair is always at the frame center.  To bring (target_y,
        target_x) to the center, the agent must turn by:
            delta_pitch = (target_y − gaze_y) / H × VFOV
            delta_yaw   = (target_x − gaze_x) / W × HFOV

        Args:
            target_y, target_x: Pixel coordinate of the desired fixation point.

        Returns:
            (delta_pitch, delta_yaw) in degrees, suitable for mc_turn().
        """
        delta_pitch = (target_y - self._gaze_y) / self._h * MINECRAFT_VFOV_DEG
        delta_yaw   = (target_x - self._gaze_x) / self._w * MINECRAFT_HFOV_DEG
        return float(delta_pitch), float(delta_yaw)

    # -------------------------------------------------------------------------
    # Feature summary helpers (used by vc_foveal_mean / vc_foveal_std primitives)
    # -------------------------------------------------------------------------

    @property
    def n_foveal_rings(self) -> int:
        """Number of rings considered 'foveal' (inner quarter)."""
        return max(1, self._n_rings // 4)

    def foveal_mean(self, features: np.ndarray) -> float:
        """Mean Gabor response over foveal (inner) rings."""
        return float(features[:self.n_foveal_rings].mean())

    def foveal_std(self, features: np.ndarray) -> float:
        """Std of Gabor response over foveal (inner) rings."""
        return float(features[:self.n_foveal_rings].std())

    # -------------------------------------------------------------------------
    # Innate priors (hardwired, not learned — Layer 0.75 in minecraft.ctkg)
    # -------------------------------------------------------------------------

    def brightness_gradient(self, frame: np.ndarray) -> float:
        """Top-strip brightness minus bottom-strip brightness.

        Ramachandran (1988) light-from-above prior: humans assume sunlight
        comes from above, so surfaces that are brighter on top are perceived
        as convex (facing up).  In Minecraft, the top face of any block
        receives direct skylight; side faces are shaded; bottom faces are dark.

        Returns:
            float: top_quarter_mean - bottom_quarter_mean.
                   Positive  = lit from above = top face of block.
                   Near zero = side face or uniform surface.
                   Negative  = bottom face (rare in normal play).
        """
        gray = _to_gray_f32(frame)
        h  = gray.shape[0]
        h4 = max(1, h // 4)
        top_mean = float(gray[:h4].mean())
        bot_mean = float(gray[3 * h4:].mean())
        return top_mean - bot_mean

    def looming(self, frame: np.ndarray, prev_frame: np.ndarray) -> float:
        """Looming score: central motion excess over peripheral motion.

        Gibson (1950) looming / optical expansion: an approaching surface
        produces a radially expanding texture flow — the center of the visual
        field moves more than the periphery.  Newborns show avoidance responses
        to looming stimuli (Bower 1977).

        In Minecraft: walking into a wall or block shows high central frame-diff
        with low peripheral diff.  Lateral motion shows equal diff everywhere.

        Returns:
            float: center_diff_mean - full_frame_diff_mean.
                   Positive = approaching surface (looming) → slow or turn.
                   Near zero = lateral motion or hovering.
        """
        gray      = _to_gray_f32(frame)
        prev_gray = _to_gray_f32(prev_frame)
        diff = np.abs(gray - prev_gray)       # (H, W)
        h, w = diff.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 6                   # foveal-center radius ≈ 1/6 frame
        y0 = max(0, cy - r);  y1 = min(h, cy + r)
        x0 = max(0, cx - r);  x1 = min(w, cx + r)
        center     = float(diff[y0:y1, x0:x1].mean())
        full_frame = float(diff.mean())
        return center - full_frame

    def sky_fraction(self, frame: np.ndarray) -> float:
        """Fraction of the top strip matching Minecraft sky hue.

        Rubin (1915) figure-ground: the larger/surrounding region is perceived
        as background.  The sky is always background; blocks are figure.
        Minecraft sky hue is blue-cyan (HSV hue ≈ 0.55–0.65).

        Returns:
            float in [0, 1]: 0.0 = cave / interior, 1.0 = open sky background.
            Used as a figure-ground cue: high sky_fraction → outdoor context.
        """
        h = frame.shape[0]
        top = frame[:max(1, h // 5)]    # top 20% of frame
        hue = _rgb_hue(top)             # (H//5, W) in [0, 1]
        sky_mask = ((hue > 0.50) & (hue < 0.70)).astype(np.float32)
        return float(sky_mask.mean())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_gray_f32(frame: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) uint8/float32 RGB to (H, W) float32 in [0, 1]."""
    if frame.ndim == 2:
        arr = frame.astype(np.float32)
    else:
        arr = frame.astype(np.float32)
        if arr.max() > 1.5:          # uint8 range
            arr /= 255.0
        # ITU-R BT.601 luminance
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def _dog(gray: np.ndarray, sigma1: float = 1.0, sigma2: float = 3.0) -> np.ndarray:
    """Difference of Gaussians: G(σ₁) − G(σ₂).  Result in (−1, 1)."""
    g1 = _gaussian_blur(gray, sigma1)
    g2 = _gaussian_blur(gray, sigma2)
    dog = g1 - g2
    peak = np.abs(dog).max()
    if peak > 1e-8:
        dog /= peak
    return dog


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """1D-separable Gaussian blur (numpy only, no scipy required)."""
    radius = int(math.ceil(3 * sigma))
    size   = 2 * radius + 1
    x      = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()

    # Separable: blur rows then columns
    out = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode='same'), axis=1, arr=img
    )
    out = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode='same'), axis=0, arr=out
    )
    return out.astype(np.float32)


def _build_logpolar_coords(
    frame_h: int, frame_w: int,
    n_rings: int, n_angles: int,
    gaze_y: float, gaze_x: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute absolute pixel coordinates for the log-polar sampling grid.

    Returns:
        sample_y: (n_rings, n_angles) float32 — y pixel coords
        sample_x: (n_rings, n_angles) float32 — x pixel coords
    """
    r_min = 1.0
    r_max = min(frame_h, frame_w) / 2.0

    # Log-spaced radii
    radii  = np.exp(np.linspace(np.log(r_min), np.log(r_max), n_rings))
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)

    # Broadcast to (n_rings, n_angles)
    R, A = np.meshgrid(radii, angles, indexing='ij')

    sample_y = (gaze_y + R * np.sin(A)).astype(np.float32)
    sample_x = (gaze_x + R * np.cos(A)).astype(np.float32)

    # Clamp to valid pixel range
    sample_y = np.clip(sample_y, 0.0, frame_h - 1.001)
    sample_x = np.clip(sample_x, 0.0, frame_w - 1.001)

    return sample_y, sample_x


def _gabor_kernel(theta: float, sigma: float, n_stds: int = 3) -> np.ndarray:
    """Real part of a 2D Gabor filter kernel.

    Args:
        theta:  Orientation in radians (0 = horizontal edges).
        sigma:  Gaussian envelope width in pixels.
        n_stds: Half-width of kernel in units of sigma.

    Returns:
        kernel: (2k+1, 2k+1) float32, zero-mean.
    """
    k = int(math.ceil(n_stds * sigma))
    y, x = np.mgrid[-k:k + 1, -k:k + 1].astype(np.float32)

    # Rotated coordinates
    xp =  x * math.cos(theta) + y * math.sin(theta)

    # Spatial frequency: one cycle per 2σ
    freq   = 1.0 / (2.0 * sigma)
    gauss  = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    carrier = np.cos(2.0 * math.pi * freq * xp)

    kernel  = gauss * carrier
    kernel -= kernel.mean()                     # remove DC component
    norm    = np.sqrt((kernel ** 2).sum())
    if norm > 1e-8:
        kernel /= norm                          # unit energy
    return kernel.astype(np.float32)


def _build_gabor_bank():
    """4 orientations × 4 scales = 16 Gabor filters.

    Orientations: 0°, 45°, 90°, 135°
    Scales (σ):   1, 2, 4, 8 pixels
    """
    orientations = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    sigmas       = [1.0, 2.0, 4.0, 8.0]
    filters = []
    for sigma in sigmas:
        for theta in orientations:
            filters.append(_gabor_kernel(theta, sigma))
    return filters  # 16 kernels


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution with 'same' padding.  Uses scipy if available."""
    if _HAS_SCIPY:
        return _scipy_convolve(img, kernel, mode='reflect').astype(np.float32)
    # Pure-numpy fallback: pad + stride-1 convolution (slower)
    kh, kw  = kernel.shape
    ph, pw  = kh // 2, kw // 2
    padded  = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
    h, w    = img.shape
    out     = np.zeros((h, w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            out[r, c] = (padded[r:r + kh, c:c + kw] * kernel).sum()
    return out


def _apply_gabor_bank(gray: np.ndarray, gabors) -> np.ndarray:
    """Apply all 16 Gabor filters to a grayscale image.

    Args:
        gray:   (H, W) float32 image.
        gabors: List of 16 (kh, kw) float32 kernels.

    Returns:
        maps: (16, H, W) float32 — one response map per filter.
    """
    h, w = gray.shape
    maps = np.empty((16, h, w), dtype=np.float32)
    for i, kernel in enumerate(gabors):
        maps[i] = _convolve2d(gray, kernel)
    return maps


def _bilinear_sample(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Bilinear interpolation at floating-point pixel coordinates.

    Args:
        img: (H, W) float32 source image.
        ys:  (n_rings, n_angles) float32 y coordinates.
        xs:  (n_rings, n_angles) float32 x coordinates.

    Returns:
        out: (n_rings, n_angles) float32 sampled values.
    """
    if _HAS_SCIPY:
        coords = np.stack([ys, xs], axis=0)            # (2, n_rings, n_angles)
        return _scipy_map_coords(
            img, coords, order=1, mode='nearest'
        ).astype(np.float32)

    # Pure-numpy bilinear fallback
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, img.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, img.shape[1] - 1)
    y0 = np.clip(y0,      0, img.shape[0] - 1)
    x0 = np.clip(x0,      0, img.shape[1] - 1)

    dy = (ys - y0).astype(np.float32)
    dx = (xs - x0).astype(np.float32)

    return (
        img[y0, x0] * (1 - dy) * (1 - dx) +
        img[y0, x1] * (1 - dy) * dx       +
        img[y1, x0] * dy       * (1 - dx) +
        img[y1, x1] * dy       * dx
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# FovealAttention  (Phase R4 — AIF Level 1 visual loop)
# ---------------------------------------------------------------------------

class FovealAttention:
    """AIF Level 1: visual attention that saccades to maximise information gain.

    Architecture
    ============
    This implements the fastest level of the Active Inference hierarchy:

        Level 1 (FovealAttention):  where to look   → ~100ms/saccade
        Level 2 (AIFEngine):        what to do       → ~1s/action
        Level 3 (deliberate):       what goal        → ~10s/decision

    Each step:
      1. Extract high-res foveal patch at the current fixation point.
      2. Extract low-res peripheral view of the full image (for context).
      3. Compute visual prediction error = surprise at this fixation.
      4. Optionally saccade: move fixation to the most surprising candidate.

    Visual prediction error
    =======================
    The agent builds a running-mean model of each screen region it has
    visited.  Novel regions (never seen) have maximum prediction error.
    Familiar regions have low prediction error unless something has changed.

        prediction_error = MSE(observed_foveal, expected_foveal)

    Expected = running mean of recent patches at this screen cell.
    This is a purely self-supervised model: no labels, no explicit generative
    model specification.  The agent learns what to expect from experience.

    Saccade policy (AIF Level 1)
    ============================
    The agent evaluates N candidate fixation points and moves to the one
    with the highest prediction error.  This IS the AIF principle at the
    visual level: minimise expected free energy by exploring uncertain regions.

    Text region detection
    =====================
    ``is_text_region(foveal)`` detects whether the foveal patch likely
    contains text, using a heuristic based on fine-scale edge statistics.
    Phase R6 will replace this with learned visual symbol recognition.

    Design principle
    ================
    **THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK. THE MODEL MUST
    BE GENERAL.**  This class contains zero game-specific logic.  It operates
    on arbitrary (H, W, C) image arrays.

    Parameters
    ----------
    image_h, image_w
        Screen dimensions in pixels.  Used to set default fixation and
        compute candidate fixation grids.
    foveal_size
        Side length of the square foveal patch in pixels.  Default 64.
        Should be large enough to capture a few text characters or a
        UI element, but small enough to be discriminative.
    peripheral_res
        Side length of the downsampled peripheral view.  Default 32.
        Provides low-resolution context (where are the salient regions?).
    n_candidates
        Number of candidate fixation points evaluated each saccade.
        Includes a regular grid + random jitter.  Default 16.
    history_len
        Number of recent foveal patches retained per screen cell for the
        running-mean self-prediction model.  Default 20.
    grid_cells
        Coarseness of the screen-cell grid for self-prediction.  A value
        of 8 divides the screen into an 8×8 grid of cells; each cell
        has its own independent prediction model.  Default 8.
    """

    def __init__(
        self,
        image_h:       int,
        image_w:       int,
        foveal_size:   int   = 64,
        peripheral_res: int  = 32,
        n_candidates:  int   = 16,
        history_len:   int   = 20,
        grid_cells:    int   = 8,
    ) -> None:
        self._h            = image_h
        self._w            = image_w
        self._foveal_size  = foveal_size
        self._peri_res     = peripheral_res
        self._n_cands      = n_candidates
        self._history_len  = history_len
        self._grid_cells   = grid_cells

        # Current fixation point (y, x) — starts at image centre.
        self.fixation: Tuple[int, int] = (image_h // 2, image_w // 2)

        # Self-prediction model: grid_cell_key → list of recent foveal patches.
        # Running mean is computed on demand.
        # {(cell_y, cell_x): [np.ndarray, ...]}
        self._history: Dict[Tuple[int, int], list] = {}

        # Step counter (for diagnostics).
        self._step: int = 0

    # ------------------------------------------------------------------
    # Core patch extraction

    def foveal_patch(
        self,
        image:    np.ndarray,
        fixation: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Extract a square foveal patch centred on fixation.

        Parameters
        ----------
        image
            (H, W) or (H, W, C) array, any dtype.
        fixation
            (y, x) centre of the patch.  None → use self.fixation.

        Returns
        -------
        patch  (foveal_size, foveal_size) or (foveal_size, foveal_size, C)
               Padded with edge values if the fixation is near the border.
        """
        fy, fx = fixation if fixation is not None else self.fixation
        h, w   = image.shape[:2]
        half   = self._foveal_size // 2

        # Clamp centre so patch fits within image bounds.
        cy = int(np.clip(fy, half, h - half))
        cx = int(np.clip(fx, half, w - half))

        if image.ndim == 2:
            patch = image[cy - half: cy + half, cx - half: cx + half]
        else:
            patch = image[cy - half: cy + half, cx - half: cx + half, :]

        # Ensure correct size (in case of border clamping edge case).
        fs = self._foveal_size
        if patch.shape[0] != fs or patch.shape[1] != fs:
            # Fallback: extract with padding.
            pad = [(max(0, half - cy), max(0, cy + half - h)),
                   (max(0, half - cx), max(0, cx + half - w))]
            if image.ndim == 3:
                pad.append((0, 0))
            patch = np.pad(
                image[max(0, cy - half): cy + half,
                      max(0, cx - half): cx + half],
                pad, mode='edge',
            )[:fs, :fs]
        return patch.copy()

    def peripheral_patch(self, image: np.ndarray) -> np.ndarray:
        """Downsample the full image to the peripheral resolution.

        Returns a (peripheral_res, peripheral_res) grayscale float32 array
        representing the low-resolution context around the fixation.

        Parameters
        ----------
        image  (H, W) or (H, W, C) array.
        """
        gray = _to_gray_f32(image)           # (H, W) in [0, 1]
        res  = self._peri_res
        h, w = gray.shape
        # Box-sample at res×res grid points.
        ys = np.linspace(0, h - 1, res).astype(np.float32)
        xs = np.linspace(0, w - 1, res).astype(np.float32)
        Y, X = np.meshgrid(ys, xs, indexing='ij')   # (res, res) each
        return _bilinear_sample(gray, Y, X)

    # ------------------------------------------------------------------
    # Self-prediction model

    def _cell_key(self, fixation: Tuple[int, int]) -> Tuple[int, int]:
        """Map (y, x) pixel fixation to grid cell key."""
        fy, fx = fixation
        cy = int(fy * self._grid_cells / self._h)
        cx = int(fx * self._grid_cells / self._w)
        return (
            int(np.clip(cy, 0, self._grid_cells - 1)),
            int(np.clip(cx, 0, self._grid_cells - 1)),
        )

    def _running_mean(self, key: Tuple[int, int]) -> Optional[np.ndarray]:
        """Return mean foveal patch for a grid cell, or None if never visited."""
        patches = self._history.get(key)
        if not patches:
            return None
        return np.mean(np.stack(patches, axis=0), axis=0).astype(np.float32)

    def _record(self, fixation: Tuple[int, int], patch: np.ndarray) -> None:
        """Add a foveal patch to the history for this grid cell."""
        key = self._cell_key(fixation)
        lst = self._history.setdefault(key, [])
        lst.append(_to_gray_f32(patch).copy())
        if len(lst) > self._history_len:
            lst.pop(0)

    def visual_prediction_error(
        self,
        foveal:   np.ndarray,
        expected: Optional[np.ndarray] = None,
    ) -> float:
        """Measure surprise (prediction error) at the current foveal patch.

        If ``expected`` is provided, uses it directly.
        Otherwise, uses the running mean at the current grid cell.
        Novel cells (never visited) return 1.0 (maximum surprise).

        Returns
        -------
        float in [0, 1].  0 = perfectly predicted; 1 = completely novel.
        """
        obs_gray = _to_gray_f32(foveal)

        if expected is None:
            key = self._cell_key(self.fixation)
            expected = self._running_mean(key)

        if expected is None:
            return 1.0   # never seen this region → maximum surprise

        mse = float(np.mean((obs_gray - expected.astype(np.float32)) ** 2))
        # Normalise: MSE of uniformly random patches ≈ 0.08 for [0,1] images.
        return float(np.clip(mse / 0.08, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Saccade planning

    def _candidate_fixations(
        self,
        extra: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """Generate candidate fixation points covering the screen.

        Returns a list of (y, x) integer pixel positions:
          - n_grid points on a regular sub-grid
          - n_random random positions for exploration
          - extra positions (e.g. from top-down saliency)
        """
        n_grid   = max(1, self._n_cands * 3 // 4)
        n_random = self._n_cands - n_grid
        half     = self._foveal_size // 2

        # Regular sub-grid (avoids image borders).
        grid_size = max(2, int(math.ceil(math.sqrt(n_grid))))
        ys = np.linspace(half, self._h - half - 1, grid_size, dtype=int)
        xs = np.linspace(half, self._w - half - 1, grid_size, dtype=int)
        grid = [(int(y), int(x)) for y in ys for x in xs]

        # Random candidates (exploration).
        rng_state = np.random.RandomState(self._step)
        rand_ys = rng_state.randint(half, max(half + 1, self._h - half), n_random)
        rand_xs = rng_state.randint(half, max(half + 1, self._w - half), n_random)
        random_cands = [(int(y), int(x)) for y, x in zip(rand_ys, rand_xs)]

        candidates = (grid + random_cands)[: self._n_cands]
        if extra:
            candidates.extend(extra)
        return candidates

    def update_fixation(
        self,
        image:    np.ndarray,
        bias:     Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[int, int]:
        """Saccade to the candidate fixation with highest prediction error.

        Parameters
        ----------
        image    Full screen image (H, W) or (H, W, C).
        bias     Optional list of (y, x) candidate fixations from top-down
                 attention (e.g. from VisualCortex.next_gaze() saliency).
                 These are added to the candidate pool.

        Returns
        -------
        (y, x) new fixation point (also stored as self.fixation).

        AIF interpretation
        ------------------
        This implements ``argmax_fixation visual_prediction_error(fixation)``
        which is equivalent to minimising expected visual free energy:
        the agent prefers to look at regions that will most reduce its
        uncertainty about the visual world.
        """
        candidates = self._candidate_fixations(extra=bias)

        best_fix = self.fixation
        best_pe  = -1.0

        for candidate in candidates:
            patch = self.foveal_patch(image, fixation=candidate)
            key   = self._cell_key(candidate)
            exp   = self._running_mean(key)
            pe    = self.visual_prediction_error(patch, expected=exp)
            if pe > best_pe:
                best_pe  = pe
                best_fix = candidate

        self.fixation = best_fix
        return best_fix

    # ------------------------------------------------------------------
    # Full attention step

    def step(
        self,
        image:       np.ndarray,
        update_gaze: bool = True,
    ) -> dict:
        """Execute one foveal attention step.

        Extracts foveal + peripheral patches, computes prediction error,
        optionally saccades, and updates the self-prediction model.

        Parameters
        ----------
        image        Full screen image (H, W) or (H, W, C).
        update_gaze  If True (default), saccade to the most surprising
                     candidate before extracting the foveal patch.
                     Set False to extract at the current fixation only.

        Returns
        -------
        dict with keys:
          'foveal':            np.ndarray (foveal_size, foveal_size) gray
          'peripheral':        np.ndarray (peripheral_res, peripheral_res) gray
          'fixation':          (y, x) current fixation after saccade
          'prediction_error':  float in [0, 1] — visual surprise
          'text_region':       bool — heuristic: likely contains text
          'step':              int — monotone step counter
        """
        self._step += 1

        if update_gaze:
            self.update_fixation(image)

        foveal     = self.foveal_patch(image)
        peripheral = self.peripheral_patch(image)

        # Compute prediction error BEFORE updating the history.
        pe = self.visual_prediction_error(foveal)

        # Update self-prediction model.
        self._record(self.fixation, foveal)

        return {
            'foveal':           _to_gray_f32(foveal),
            'peripheral':       peripheral,
            'fixation':         self.fixation,
            'prediction_error': pe,
            'text_region':      self.is_text_region(foveal),
            'step':             self._step,
        }

    # ------------------------------------------------------------------
    # Text region heuristic (Phase R4 — to be replaced by R6 learning)

    @staticmethod
    def is_text_region(foveal: np.ndarray) -> bool:
        """Heuristic: does this foveal patch likely contain text?

        Text characters have:
          - High fine-scale edge density (many stroke boundaries)
          - Strong contrast between ink and background
          - Complex edge structure (strokes go in many directions)

        This is a pure image-statistics heuristic: no labels, no training.
        Phase R6 (visual symbol learning) will replace this with a learned
        distributional model, analogous to Phase O for text.

        Returns
        -------
        True  if the patch likely contains text or other high-contrast
              alphanumeric symbols.
        False if the patch looks like a flat surface, sky, or simple texture.
        """
        gray     = _to_gray_f32(foveal)
        contrast = float(np.std(gray))

        # Low contrast = uniform region, not text.
        if contrast < 0.05:
            return False

        # Fine-scale DoG response: strong in character stroke regions.
        dog = _dog(gray, sigma1=0.5, sigma2=1.5)
        edge_std = float(np.std(dog))

        # Coarse-scale DoG: low in text (characters are smaller than blobs).
        dog_coarse = _dog(gray, sigma1=2.0, sigma2=6.0)
        edge_coarse_std = float(np.std(dog_coarse))

        # Text heuristic: fine edges present, coarser structure moderate.
        # Calibrated to separate text from: flat surfaces (low fine edges),
        # complex textures (high fine AND coarse edges), photos (medium both).
        fine_present  = edge_std > 0.10
        coarse_low    = edge_coarse_std < 0.35
        return bool(fine_present and coarse_low)

    # ------------------------------------------------------------------
    # Diagnostics

    def coverage(self) -> float:
        """Fraction of grid cells visited at least once (in [0, 1])."""
        total = self._grid_cells * self._grid_cells
        return len(self._history) / total

    def reset(self, image_h: Optional[int] = None, image_w: Optional[int] = None) -> None:
        """Reset fixation to centre and clear the self-prediction model."""
        if image_h is not None:
            self._h = image_h
        if image_w is not None:
            self._w = image_w
        self.fixation = (self._h // 2, self._w // 2)
        self._history.clear()
        self._step = 0

    def __repr__(self) -> str:
        fy, fx = self.fixation
        cov = self.coverage()
        return (
            f'FovealAttention(fixation=({fy},{fx}), '
            f'foveal={self._foveal_size}px, '
            f'coverage={cov:.1%})'
        )


def _rgb_hue(frame: np.ndarray) -> np.ndarray:
    """Compute HSV hue channel (H, W) in [0, 1] from RGB frame."""
    arr = frame.astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    hue = np.zeros_like(r)
    eps = 1e-8

    # Red sector
    mask = (cmax == r) & (delta > eps)
    hue[mask] = ((g[mask] - b[mask]) / (delta[mask] + eps)) % 6.0
    # Green sector
    mask = (cmax == g) & (delta > eps)
    hue[mask] = (b[mask] - r[mask]) / (delta[mask] + eps) + 2.0
    # Blue sector
    mask = (cmax == b) & (delta > eps)
    hue[mask] = (r[mask] - g[mask]) / (delta[mask] + eps) + 4.0

    return (hue / 6.0).astype(np.float32)
