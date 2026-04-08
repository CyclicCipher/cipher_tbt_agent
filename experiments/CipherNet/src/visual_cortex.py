"""Retinotopic visual cortex with parietal coordinate transform.

V1 columns read patches from the retinal image. The parietal cortex
(a simple function, not neurons) transforms retinal positions to
object-centered coordinates: location = retinal_pos + eye_fixation - object_origin.

This gives each feature binding a CANONICAL location on the object,
independent of which saccade path was used to explore it. Two images
of "7" explored with different saccade sequences produce the SAME
object-centered locations for the same digit features.
"""
from __future__ import annotations

import numpy as np

from symbolic_column import SymbolicColumn, CorticalMessage
from sdr import SDR, SDREncoder
from eye import Eye, SalienceMap
from codebook import PatchCodebook


def parietal_transform(retinal_pos: tuple[float, float],
                       eye_fixation: tuple[float, float],
                       object_origin: tuple[float, float]) -> tuple[float, float]:
    """Symbolic parietal cortex: compute object-centered coordinates.

    head_centered = retinal_pos + eye_fixation
    object_centered = head_centered - object_origin

    The brain does this with 300 gain-modulated neurons (because
    neurons can't add). We just add.
    """
    hx = retinal_pos[0] + eye_fixation[0]
    hy = retinal_pos[1] + eye_fixation[1]
    return (hx - object_origin[0], hy - object_origin[1])


class RetinotopicV1:
    """V1 cortex: columns with patch RFs + parietal coordinate transform."""

    def __init__(self, eye: Eye, codebook: PatchCodebook,
                 patch_size: int = 5, stride: int = 3,
                 sdr_n: int = 256, sdr_w: int = 10):
        self.eye = eye
        self.codebook = codebook
        self.patch_size = patch_size
        self.stride = stride

        rs = eye.retina_size
        self.grid_h = (rs - patch_size) // stride + 1
        self.grid_w = (rs - patch_size) // stride + 1

        self.encoder = SDREncoder(n=sdr_n, w=sdr_w)

        # One column per retinal grid position.
        # retinal_pos = center of the column's patch on the retina.
        self.columns: list[SymbolicColumn] = []
        self._retinal_positions: list[tuple[float, float]] = []
        half = rs / 2.0
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                # Retinal position: center of patch, relative to retina center.
                ry = gy * stride + patch_size / 2.0 - half
                rx = gx * stride + patch_size / 2.0 - half
                col = SymbolicColumn(
                    name=f"V1:{gy},{gx}",
                    receptive_field=(gy * stride, gx * stride,
                                    gy * stride + patch_size,
                                    gx * stride + patch_size),
                    position=(rx, ry),
                )
                self.columns.append(col)
                self._retinal_positions.append((rx, ry))

        # Object origin for parietal transform (set per exploration).
        self._object_origin: tuple[float, float] = (0.0, 0.0)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def set_object_origin(self, origin: tuple[float, float]):
        """Set the object reference frame origin (first fixation point)."""
        self._object_origin = origin

    def observe(self, image: np.ndarray) -> list[SDR]:
        """Sample image, encode patches as SDRs, compute object-centered locations."""
        retina = self.eye.sample(image)
        ps, st = self.patch_size, self.stride

        # Extract all patches (vectorized).
        n = self.grid_h * self.grid_w
        patches = np.empty((n, ps, ps), dtype=np.float32)
        idx = 0
        for gy in range(self.grid_h):
            y0 = gy * st
            for gx in range(self.grid_w):
                x0 = gx * st
                patches[idx] = retina[y0:y0+ps, x0:x0+ps]
                idx += 1

        codes = self.codebook.encode_batch(patches)
        eye_fix = self.eye.fixation

        features = []
        for i, col in enumerate(self.columns):
            sdr_feat = self.encoder.encode(f"v{codes[i]}")
            # Parietal transform: object-centered location.
            rp = self._retinal_positions[i]
            loc = parietal_transform(rp, eye_fix, self._object_origin)
            # Quantize location to grid (avoid float precision noise).
            qloc = (round(loc[0]), round(loc[1]))
            col.set_location(qloc)
            col.observe(sdr_feat)
            features.append(sdr_feat)

        return features

    def learn_object(self, object_id: str):
        """All columns learn: bind current (feature, location) to object."""
        for col in self.columns:
            col.learn(object_id)

    def recognize(self) -> tuple[str | None, float]:
        """Aggregate recognition votes across columns."""
        votes: dict[str, float] = {}
        for col in self.columns:
            if col.recognized is not None and col.confidence > 0.0:
                votes[col.recognized] = votes.get(col.recognized, 0.0) + col.confidence
        if not votes:
            return None, 0.0
        winner = max(votes, key=votes.get)
        total = sum(votes.values())
        return winner, votes[winner] / total if total > 0 else 0.0


class FovealExplorer:
    """Explores images through saccades with parietal coordinate transform."""

    def __init__(self, eye: Eye, v1: RetinotopicV1, n_fixations: int = 3):
        self.eye = eye
        self.v1 = v1
        self.n_fixations = n_fixations

    def explore(self, image: np.ndarray, object_id: str | None = None,
                learn: bool = True) -> tuple[str | None, float]:
        """Explore an image via saccades.

        The first fixation defines the object reference frame origin.
        All subsequent locations are object-centered (relative to
        first fixation). This makes locations CANONICAL — the same
        for the same object regardless of saccade path.
        """
        h, w = image.shape[:2]
        fixations = SalienceMap.suggest_fixations(
            image, n=self.n_fixations,
            min_distance=max(3, self.eye.retina_size // 4),
        )
        if not fixations:
            fixations = [(w // 2, h // 2)]

        # Object origin = image center (canonical, consistent across images).
        # Using first fixation as origin makes locations saccade-dependent.
        # Using image center makes locations absolute → same digit features
        # at same image positions get the same object-centered coordinates.
        self.v1.set_object_origin((w / 2.0, h / 2.0))
        fx, fy = fixations[0]
        self.eye.fixate(float(fx), float(fy))
        self.v1.observe(image)

        if learn and object_id is not None:
            self.v1.learn_object(object_id)

        # Subsequent fixations: saccade → observe → learn.
        for fx, fy in fixations[1:]:
            self.eye.saccade_to(float(fx), float(fy))
            self.v1.observe(image)
            if learn and object_id is not None:
                self.v1.learn_object(object_id)

        return self.v1.recognize()
