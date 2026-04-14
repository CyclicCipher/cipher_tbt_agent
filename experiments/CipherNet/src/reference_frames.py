"""Innate reference frames for cortical columns.

Every cortical column in a biological brain is initialized with one of
these frames. The frame type is determined by anatomical connectivity
(i.e. what the genome specifies), NOT by learning.

The frame tracks the column's current position in its reference space
and updates via path integration: displacement signals (efference copies
of motor commands) are integrated to maintain position without requiring
access to absolute coordinates.

Path integration is the biologically correct mechanism:
  - The column never knows its absolute position in the world.
  - It only knows where it STARTED and how far it has MOVED.
  - position_new = position_old + displacement

Supported frames (matching all known innate cortical maps):
  retinotopic   - 2D visual field (V1, V2, V4, MT)
  tonotopic     - 1D frequency axis (A1, A2, belt areas)
  somatotopic   - 2D body surface (S1, S2)
  proprioceptive - N-D joint angle space (M1, cerebellum)
  egocentric    - 3D body-centered (parietal cortex, PPC)
  allocentric   - 2D/3D world-centered (entorhinal, hippocampus)
  temporal      - 1D time axis (prefrontal, hippocampus)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ReferenceFrame(ABC):
    """Abstract reference frame.

    A frame tracks a position (as a hashable key) and updates it via
    path integration when a displacement arrives.
    """

    @abstractmethod
    def position_key(self) -> tuple:
        """Current position as a hashable tuple (used as dict key)."""

    @abstractmethod
    def update(self, displacement: tuple) -> None:
        """Path-integrate one displacement step."""

    @abstractmethod
    def set_position(self, position: tuple) -> None:
        """Hard-set position (used at sequence start / reset)."""

    @abstractmethod
    def frame_type(self) -> str:
        """Canonical name matching FRAME_REGISTRY keys."""


# ---------------------------------------------------------------------------
# Retinotopic — 2D visual field (V1, V2, V4, MT, MST)
# ---------------------------------------------------------------------------

class RetinotopicFrame(ReferenceFrame):
    """2D retinal/visual-field reference frame.

    Initialization (innate / genome-specified):
      Each column in the retinotopic map starts at the position
      corresponding to its grid location within the retinal image.
      Patch center in retinal coords = gx*stride + (patch_size-1)/2.
      Allocentric position = fixation + patch_offset - image_center.

    Path integration:
      When the eye moves by displacement D, allocentric position
      shifts by D (same for every column — the whole map translates).

    Adaptation:
      Over development, the map can be refined by experience
      (e.g. monocular deprivation shifts column boundaries), but
      the initial topology is fully specified by anatomy.
    """

    def __init__(self, grid_pos: tuple[int, int],
                 stride: int, patch_size: int,
                 retina_size: int,
                 image_size: tuple[int, int] = (28, 28),
                 encoding: str = 'grid'):
        gx, gy = grid_pos
        rc = (retina_size - 1) / 2
        # Offset of this column's patch center from retina center
        self._offset = (
            gx * stride + (patch_size - 1) / 2 - rc,
            gy * stride + (patch_size - 1) / 2 - rc,
        )
        self._image_cx = (image_size[0] - 1) / 2
        self._image_cy = (image_size[1] - 1) / 2
        self._encoding = encoding
        # Start at image center fixation
        self._fx: float = self._image_cx
        self._fy: float = self._image_cy
        self._pos = self._compute()

    # Spatial resolution of the location key in pixels.
    # Positions within ±(RESOLUTION/2) px map to the same key.
    # Motivated by grid-cell firing field width: nearby positions activate
    # the same grid cell, so the reference frame has finite precision.
    # Practically: absorbs the ±2–3px centroid variation across MNIST
    # instances so same-class images produce identical location keys and
    # the plasticity gate lets them reinforce each other.
    RESOLUTION: int = 2

    # Fourier encoding parameters.
    # Three spatial wavelengths (coarse→fine); per wavelength, four
    # components: cos(2π x/λ), sin(2π x/λ), cos(2π y/λ), sin(2π y/λ).
    # Total: 3 × 4 = 12 integers per position key.
    # FOURIER_Q: quantisation half-range; cos ∈ [-1,1] → int in [-Q,Q].
    # Nearby positions share more components → higher _loc_sim weight.
    FOURIER_LAMBDAS: tuple = (28.0, 14.0, 7.0)
    FOURIER_Q: int = 2

    def _compute(self) -> tuple:
        """Allocentric position key, quantised.

        'grid'   : 2-integer bin (original behaviour, RESOLUTION-pixel bins).
        'fourier': 12-integer multi-scale Fourier key.
        """
        if self._encoding == 'fourier':
            return self._compute_fourier()
        raw_x = self._fx + self._offset[0] - self._image_cx
        raw_y = self._fy + self._offset[1] - self._image_cy
        r = self.RESOLUTION
        return (round(raw_x / r), round(raw_y / r))

    def _compute_fourier(self) -> tuple:
        """12-component Fourier location key.

        Three spatial scales (wavelengths 28, 14, 7 px).
        Per scale: cos and sin of 2π·pos/λ for both x and y.
        Each component quantised to [-FOURIER_Q, +FOURIER_Q].

        Nearby positions agree on more components (especially the low-
        frequency ones) and therefore receive higher _loc_sim weights in
        MiniColumn.overlap_score, giving automatic spatial generalisation.
        Displacement arithmetic is undefined on these components, so
        MiniColumn.predict() returns None for keys longer than 2 elements.
        """
        import math
        raw_x = self._fx + self._offset[0] - self._image_cx
        raw_y = self._fy + self._offset[1] - self._image_cy
        Q = self.FOURIER_Q
        components: list[int] = []
        for lam in self.FOURIER_LAMBDAS:
            components.append(round(math.cos(2 * math.pi * raw_x / lam) * Q))
            components.append(round(math.sin(2 * math.pi * raw_x / lam) * Q))
            components.append(round(math.cos(2 * math.pi * raw_y / lam) * Q))
            components.append(round(math.sin(2 * math.pi * raw_y / lam) * Q))
        return tuple(components)

    def set_position(self, position: tuple) -> None:
        """Set absolute fixation position (sequence start)."""
        self._fx, self._fy = float(position[0]), float(position[1])
        self._pos = self._compute()

    def update(self, displacement: tuple) -> None:
        """Path-integrate eye movement."""
        self._fx += displacement[0]
        self._fy += displacement[1]
        self._pos = self._compute()

    def position_key(self) -> tuple:
        return self._pos

    def frame_type(self) -> str:
        return 'retinotopic'


# ---------------------------------------------------------------------------
# Tonotopic — 1D frequency axis (A1, A2, belt auditory areas)
# ---------------------------------------------------------------------------

class TonotopicFrame(ReferenceFrame):
    """1D tonotopic (frequency) reference frame.

    Initialization:
      Each auditory column is initialized at a characteristic frequency
      determined by its position along the cochleotopic axis.
      Frequency is represented on a log scale (octaves).

    Path integration:
      Displacement = shift in octaves (e.g. from an attention shift or
      auditory streaming process).
    """

    def __init__(self, frequency_hz: float,
                 min_hz: float = 20.0, max_hz: float = 20000.0,
                 n_bins: int = 128):
        import math
        self._log_min = math.log2(min_hz)
        self._log_max = math.log2(max_hz)
        self._n_bins = n_bins
        self._log_freq = math.log2(max(frequency_hz, min_hz))
        self._pos = self._compute()

    def _compute(self) -> tuple[int]:
        import math
        span = self._log_max - self._log_min
        idx = round((self._log_freq - self._log_min) / span * (self._n_bins - 1))
        return (max(0, min(self._n_bins - 1, idx)),)

    def set_position(self, position: tuple) -> None:
        """Set log-frequency directly."""
        import math
        self._log_freq = math.log2(max(position[0], 1e-3))
        self._pos = self._compute()

    def update(self, displacement: tuple) -> None:
        """Displacement = change in log2(Hz) (octaves)."""
        self._log_freq += displacement[0]
        self._pos = self._compute()

    def position_key(self) -> tuple:
        return self._pos

    def frame_type(self) -> str:
        return 'tonotopic'


# ---------------------------------------------------------------------------
# Somatotopic — 2D body surface (S1, S2)
# ---------------------------------------------------------------------------

class SomatotopicFrame(ReferenceFrame):
    """2D somatotopic (body surface) reference frame.

    Initialization:
      Each column is initialized at a specific body surface location
      determined by somatotopic anatomy (homunculus).
      Position = (x, y) on the 2D unfolded body surface map.

    Path integration:
      Displacement = movement of the tactile sensor (finger tip, skin).
    """

    def __init__(self, body_pos: tuple[float, float],
                 scale: float = 1.0):
        self._x, self._y = float(body_pos[0]), float(body_pos[1])
        self._scale = scale
        self._pos = self._compute()

    def _compute(self) -> tuple[int, int]:
        return (round(self._x / self._scale), round(self._y / self._scale))

    def set_position(self, position: tuple) -> None:
        self._x, self._y = float(position[0]), float(position[1])
        self._pos = self._compute()

    def update(self, displacement: tuple) -> None:
        self._x += displacement[0]
        self._y += displacement[1]
        self._pos = self._compute()

    def position_key(self) -> tuple:
        return self._pos

    def frame_type(self) -> str:
        return 'somatotopic'


# ---------------------------------------------------------------------------
# Proprioceptive — joint angle space (M1, premotor, cerebellum)
# ---------------------------------------------------------------------------

class ProprioceptiveFrame(ReferenceFrame):
    """N-dimensional joint angle reference frame.

    Initialization:
      Each column is initialized at a resting joint angle configuration
      (anatomically determined default posture).

    Path integration:
      Displacement = change in joint angles (efference copy of motor command).
      One component per joint.
    """

    def __init__(self, joint_angles: tuple[float, ...],
                 resolution: float = 5.0):
        """
        joint_angles: initial angles in degrees for each joint
        resolution:   quantization step in degrees
        """
        self._angles = list(joint_angles)
        self._res = resolution
        self._pos = self._compute()

    def _compute(self) -> tuple:
        return tuple(round(a / self._res) for a in self._angles)

    def set_position(self, position: tuple) -> None:
        self._angles = [float(a) * self._res for a in position]
        self._pos = self._compute()

    def update(self, displacement: tuple) -> None:
        for i, d in enumerate(displacement):
            if i < len(self._angles):
                self._angles[i] += d
        self._pos = self._compute()

    def position_key(self) -> tuple:
        return self._pos

    def frame_type(self) -> str:
        return 'proprioceptive'


# ---------------------------------------------------------------------------
# Egocentric — body-centered 3D (PPC, premotor, frontal eye fields)
# ---------------------------------------------------------------------------

class EgocentricFrame(ReferenceFrame):
    """3D body-centered (head-centered / eye-centered) reference frame.

    Initialization:
      Columns are initialized at positions relative to the body midline.
      Coordinates: (azimuth_deg, elevation_deg, depth_m).

    Path integration:
      Displacement = change in (azimuth, elevation, depth) from
      vestibular / proprioceptive efference copy.
    """

    def __init__(self, position: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 resolution: tuple[float, float, float] = (5.0, 5.0, 0.1)):
        self._pos_f = list(position)
        self._res = resolution
        self._pos = self._compute()

    def _compute(self) -> tuple:
        return tuple(round(p / r) for p, r in zip(self._pos_f, self._res))

    def set_position(self, position: tuple) -> None:
        self._pos_f = [float(p) for p in position]
        self._pos = self._compute()

    def update(self, displacement: tuple) -> None:
        for i, d in enumerate(displacement):
            if i < len(self._pos_f):
                self._pos_f[i] += d
        self._pos = self._compute()

    def position_key(self) -> tuple:
        return self._pos

    def frame_type(self) -> str:
        return 'egocentric'


# ---------------------------------------------------------------------------
# Allocentric — world-centered (entorhinal cortex, hippocampus)
# ---------------------------------------------------------------------------

class AllocentricFrame(ReferenceFrame):
    """World-centered (allocentric) reference frame.

    Initialization:
      Columns start at the origin of the environment map (0, 0) or at a
      known landmark position. The reference frame is anchored to the
      world, not the observer.

    Path integration:
      Displacement = self-motion vector (from vestibular + proprioceptive
      signals). Grid cells in entorhinal cortex implement this.
    """

    def __init__(self, position: tuple[float, ...] = (0.0, 0.0),
                 resolution: float = 1.0):
        self._pos_f = list(position)
        self._res = resolution
        self._pos = self._compute()

    def _compute(self) -> tuple:
        return tuple(round(p / self._res) for p in self._pos_f)

    def set_position(self, position: tuple) -> None:
        self._pos_f = [float(p) for p in position]
        self._pos = self._compute()

    def update(self, displacement: tuple) -> None:
        for i, d in enumerate(displacement):
            if i < len(self._pos_f):
                self._pos_f[i] += d
        self._pos = self._compute()

    def position_key(self) -> tuple:
        return self._pos

    def frame_type(self) -> str:
        return 'allocentric'


# ---------------------------------------------------------------------------
# Temporal — 1D time axis (prefrontal, hippocampus, basal ganglia)
# ---------------------------------------------------------------------------

class TemporalFrame(ReferenceFrame):
    """1D temporal reference frame.

    Initialization:
      Column starts at time offset 0 relative to a sequence anchor
      (e.g. start of a word, start of an action sequence).

    Path integration:
      Displacement = time step (1 = next event, larger = skip ahead).
    """

    def __init__(self, t: int = 0, max_t: int = 1024):
        self._t = t
        self._max_t = max_t
        self._pos = (self._t,)

    def set_position(self, position: tuple) -> None:
        self._t = int(position[0]) % self._max_t
        self._pos = (self._t,)

    def update(self, displacement: tuple) -> None:
        self._t = (self._t + int(displacement[0])) % self._max_t
        self._pos = (self._t,)

    def position_key(self) -> tuple:
        return self._pos

    def frame_type(self) -> str:
        return 'temporal'


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

FRAME_REGISTRY: dict[str, type[ReferenceFrame]] = {
    'retinotopic':    RetinotopicFrame,
    'tonotopic':      TonotopicFrame,
    'somatotopic':    SomatotopicFrame,
    'proprioceptive': ProprioceptiveFrame,
    'egocentric':     EgocentricFrame,
    'allocentric':    AllocentricFrame,
    'temporal':       TemporalFrame,
}


def make_frame(frame_type: str, params: dict[str, Any]) -> ReferenceFrame:
    """Instantiate a reference frame from a type name and parameter dict.

    The params dict is passed directly as keyword arguments to the
    frame constructor. This is what the cortex config loader calls.
    """
    cls = FRAME_REGISTRY.get(frame_type)
    if cls is None:
        raise ValueError(
            f"Unknown reference frame type '{frame_type}'. "
            f"Available: {sorted(FRAME_REGISTRY)}"
        )
    return cls(**params)
