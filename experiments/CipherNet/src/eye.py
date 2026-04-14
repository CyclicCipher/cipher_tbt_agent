"""Foveal eye with biologically grounded preprocessing and fixation.

Three biological facts drive every design decision here:

1. THE RETINA SENDS CONTRAST, NOT PIXELS.
   Retinal ganglion cells have center-surround (DoG) receptive fields.
   They fire when their center is brighter OR darker than their surround.
   Uniform regions produce no output.  The brain never sees raw luminance.
   Implementation: Eye.preprocess() applies DoG before any patch is sampled.

2. THE EYE FIXATES ON THE OBJECT, NOT ON IMAGE COORDINATES.
   The first saccade lands the fovea on the object (superior colliculus).
   Subsequent saccades explore relative to that anchor point.
   This makes all location keys object-relative, not image-relative.
   Without this, the same feature of a "3" gets a different location key
   in every image depending on where the digit happens to sit.
   Implementation: first fixation = centroid of non-background pixels.

3. SUBSEQUENT SACCADES GO TO SALIENT POINTS.
   After landing on the object, the eye moves to the most informative
   locations — edges, corners, stroke endpoints — not to a fixed grid.
   Salience is computed from the DoG image (the retinal output).
   Implementation: SalienceMap.suggest_fixations on the DoG image.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Gaussian utilities (no scipy dependency)
# ---------------------------------------------------------------------------

def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur using np.convolve. No scipy needed."""
    size = max(3, int(4 * sigma + 1) | 1)   # odd, covers ±2σ
    k = np.arange(size, dtype=np.float32) - size // 2
    kernel = np.exp(-k ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    img = image.astype(np.float32)
    # Horizontal pass
    out = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode='same'), 1, img)
    # Vertical pass
    out = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode='same'), 0, out)
    return out


# ---------------------------------------------------------------------------
# Eye
# ---------------------------------------------------------------------------

class Eye:
    """Foveal eye: DoG retinal preprocessing + fixation-based sampling.

    Workflow per image:
        dog = eye.preprocess(raw_image)      # retinal output (once)
        cx, cy = eye.centroid(raw_image)     # object anchor (once)
        retina = eye.sample(dog, fixation)   # retinal crop (per fixation)
    """

    def __init__(self, retina_size: int = 19):
        assert retina_size % 2 == 1, "retina_size must be odd"
        self.retina_size = retina_size
        self.half = retina_size // 2
        self.fixation: tuple[float, float] = (0.0, 0.0)
        self.last_displacement: tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Biological retinal preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Retinal DoG: center-surround contrast coding.

        Mimics retinal ganglion cell (ON/OFF center-surround) responses.
        Removes absolute luminance — only contrast (edges) survives.
        This is what the optic nerve actually sends to the brain.

        sigma1=1.0 (center), sigma2=2.0 (surround).
        Output is float32 in roughly [-0.3, 0.3] for a 0–1 image.
        Positive = bright-center, negative = dark-center.
        Both signs drive Gabor filters (which take abs response).
        """
        img = image.astype(np.float32)
        if img.max() > 1.5:          # allow both 0–255 and 0–1 inputs
            img = img / 255.0
        return _gaussian_blur(img, 1.0) - _gaussian_blur(img, 2.0)

    def centroid(self, image: np.ndarray) -> tuple[float, float]:
        """Object centroid: center of mass of supra-threshold pixels.

        Uses the raw image (not DoG) so we anchor to the object's mass,
        not its edges.  Threshold = 10% of peak value, which works for
        both 0–255 and 0–1 images and gracefully handles background.
        """
        img = image.astype(np.float32)
        threshold = max(1e-3, float(img.max()) * 0.10)
        mask = img > threshold
        if not mask.any():
            h, w = img.shape[:2]
            return float(w / 2.0), float(h / 2.0)
        ys, xs = np.where(mask)
        return float(xs.mean()), float(ys.mean())

    # ------------------------------------------------------------------
    # Fixation control
    # ------------------------------------------------------------------

    def fixate(self, x: float, y: float) -> None:
        self.fixation = (x, y)
        self.last_displacement = (0.0, 0.0)

    def saccade(self, dx: float, dy: float) -> None:
        fx, fy = self.fixation
        self.fixation = (fx + dx, fy + dy)
        self.last_displacement = (dx, dy)

    def saccade_to(self, x: float, y: float) -> None:
        fx, fy = self.fixation
        self.last_displacement = (x - fx, y - fy)
        self.fixation = (x, y)

    # ------------------------------------------------------------------
    # Retinal sampling
    # ------------------------------------------------------------------

    def sample(self, dog_image: np.ndarray) -> np.ndarray:
        """Crop a retina_size × retina_size window from the DoG image.

        Accepts the OUTPUT of preprocess() (float32, contrast-coded).
        Pixels outside the image boundary are 0 (no signal = silence).
        """
        h, w = dog_image.shape[:2]
        fx = int(round(self.fixation[0]))
        fy = int(round(self.fixation[1]))
        r  = self.half

        retina = np.zeros((self.retina_size, self.retina_size),
                          dtype=np.float32)

        src_y0 = max(0, fy - r);  src_y1 = min(h, fy + r + 1)
        src_x0 = max(0, fx - r);  src_x1 = min(w, fx + r + 1)
        dst_y0 = src_y0 - (fy - r)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = src_x0 - (fx - r)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        if src_y1 > src_y0 and src_x1 > src_x0:
            retina[dst_y0:dst_y1, dst_x0:dst_x1] = \
                dog_image[src_y0:src_y1, src_x0:src_x1]

        return retina

    # ------------------------------------------------------------------
    # Saccade vocabulary
    # ------------------------------------------------------------------

    SACCADE_VOCAB = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    def __repr__(self) -> str:
        return f"Eye(retina={self.retina_size}x{self.retina_size})"


# ---------------------------------------------------------------------------
# FovealExplorer
# ---------------------------------------------------------------------------

class FovealExplorer:
    """Drives the eye through a biologically grounded fixation sequence.

    Fixation 0: object centroid (eye lands on the object first).
    Fixations 1–N: highest-salience points from the DoG image
                   (edges, corners, stroke endpoints), spread out by
                   min_distance to avoid redundant nearby samples.

    This is the biological saccade strategy: land on object → explore
    its most informative features.  All location keys produced by the
    cortex are therefore object-relative (centroid-anchored).
    """

    def __init__(self, eye: Eye, n_fixations: int = 9, step: int = 5):
        self.eye = eye
        self.n_fixations = n_fixations
        self.step = step

    def get_fixations(self, image: np.ndarray) -> list[tuple]:
        """Centroid first, then cardinal/diagonal offsets from centroid.

        All fixation points are object-relative: expressed as displacements
        from the digit's center of mass rather than absolute image coords.
        This ensures the same structural feature of a digit maps to the
        same location key across all instances, regardless of where the
        digit sits in the image.

        Using the cardinal vocabulary (not salience) keeps the fixation
        pattern CONSISTENT across instances — the same 9 offsets are
        visited every time, so the same location keys are produced and
        the object model converges.  Salience-driven saccades would vary
        per image and scatter location keys, defeating the model.
        """
        cx, cy = self.eye.centroid(image)
        fixations: list[tuple] = [(cx, cy)]
        for dx, dy in self.eye.SACCADE_VOCAB:
            if len(fixations) >= self.n_fixations:
                break
            fixations.append((cx + dx * self.step,
                               cy + dy * self.step))
        return fixations[:self.n_fixations]


# ---------------------------------------------------------------------------
# StaticEye
# ---------------------------------------------------------------------------

class StaticEye(Eye):
    """Non-moving eye that sees the entire image at full resolution.

    Single fixation at the OBJECT CENTROID (not image center).
    The 6×6 column grid is therefore centered on the object, making
    all location keys object-relative — the same fix as the foveal eye,
    just without the need for multiple saccades.

    Use to isolate the feature-encoding and WTA pipeline from saccade
    complexity.  DoG preprocessing still applies (retina is retina).
    """

    def __init__(self, image_size: int = 28):
        retina_size = image_size if image_size % 2 == 1 else image_size + 1
        super().__init__(retina_size=retina_size)
        self.image_size = image_size

    def get_fixations(self, image: np.ndarray) -> list[tuple]:
        """Single fixation at the object centroid."""
        cx, cy = self.centroid(image)
        return [(cx, cy)]


# ---------------------------------------------------------------------------
# SalienceMap
# ---------------------------------------------------------------------------

class SalienceMap:
    """Bottom-up salience from the DoG (retinal) image."""

    @staticmethod
    def compute(dog_image: np.ndarray) -> np.ndarray:
        """Gradient magnitude of the DoG image.

        High salience = rapid spatial change in contrast =
        edges, corners, stroke endpoints in the original image.
        """
        h, w = dog_image.shape[:2]
        sal = np.zeros((h, w), dtype=np.float32)
        if h > 2 and w > 2:
            gx = np.abs(np.diff(dog_image.astype(np.float32), axis=1))
            gy = np.abs(np.diff(dog_image.astype(np.float32), axis=0))
            mh = min(gx.shape[0], gy.shape[0])
            mw = min(gx.shape[1], gy.shape[1])
            sal[1:1+mh, 1:1+mw] = np.sqrt(
                gx[:mh, :mw] ** 2 + gy[:mh, :mw] ** 2)
        return sal

    @staticmethod
    def suggest_fixations(dog_image: np.ndarray, n: int = 8,
                          min_distance: int = 4) -> list[tuple[float, float]]:
        """N highest-salience fixation points, spatially spread out."""
        sal = SalienceMap.compute(dog_image)
        fixations: list[tuple[float, float]] = []
        for _ in range(n):
            if sal.max() < 1e-6:
                break
            idx = np.unravel_index(sal.argmax(), sal.shape)
            y, x = int(idx[0]), int(idx[1])
            fixations.append((float(x), float(y)))
            y0 = max(0, y - min_distance)
            y1 = min(sal.shape[0], y + min_distance + 1)
            x0 = max(0, x - min_distance)
            x1 = min(sal.shape[1], x + min_distance + 1)
            sal[y0:y1, x0:x1] = 0.0
        return fixations
