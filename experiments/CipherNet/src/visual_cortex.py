"""Retinotopic visual cortex with SDR features and foveal exploration.

V1 columns read patches from the retinal image, VQ-encode them,
then encode VQ codes as SDRs. Feature-location bindings stored
in object models (not feature→label dicts).

When the eye saccades, displacement updates each column's location
in the reference frame. Columns accumulate (feature_SDR, location)
bindings per object through sensorimotor exploration.
"""
from __future__ import annotations

import numpy as np

from symbolic_column import SymbolicColumn, CorticalMessage
from sdr import SDR, SDREncoder
from eye import Eye, SalienceMap
from codebook import PatchCodebook


class RetinotopicV1:
    """V1 cortex: columns with patch RFs, SDR features, location binding."""

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

        # SDR encoder: VQ code strings → SDR features.
        self.encoder = SDREncoder(n=sdr_n, w=sdr_w)

        # One column per grid position on retina.
        self.columns: list[SymbolicColumn] = []
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                y0, x0 = gy * stride, gx * stride
                col = SymbolicColumn(
                    name=f"V1:{gy},{gx}",
                    receptive_field=(y0, x0, y0 + patch_size, x0 + patch_size),
                    position=(float(gx), float(gy)),
                )
                self.columns.append(col)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def observe(self, image: np.ndarray) -> list[SDR]:
        """Sample image through eye, extract patches, encode as SDRs."""
        retina = self.eye.sample(image)
        ps = self.patch_size
        st = self.stride

        # Extract and encode all patches (vectorized).
        n = self.grid_h * self.grid_w
        patches = np.empty((n, ps, ps), dtype=np.float32)
        idx = 0
        for gy in range(self.grid_h):
            y0 = gy * st
            for gx in range(self.grid_w):
                x0 = gx * st
                patches[idx] = retina[y0:y0+ps, x0:x0+ps]
                idx += 1

        # Batch VQ encode → code indices.
        codes = self.codebook.encode_batch(patches)

        # Encode each code as SDR and observe at column.
        features = []
        for i, col in enumerate(self.columns):
            sdr_feat = self.encoder.encode(f"v{codes[i]}")
            col.observe(sdr_feat)
            features.append(sdr_feat)

        return features

    def update_locations(self, dx: float, dy: float):
        """Update all columns' reference frame locations by displacement."""
        for col in self.columns:
            col.displace(dx, dy)

    def reset_locations(self):
        """Reset all columns to origin of reference frame."""
        for col in self.columns:
            col.set_location((0.0, 0.0))

    def learn_object(self, object_id: str):
        """All columns learn: bind current (feature, location) to object."""
        for col in self.columns:
            col.learn(object_id)

    def recognize(self) -> tuple[str | None, float]:
        """Aggregate recognition across all columns.

        Each column votes for its best matching object model.
        Majority vote with confidence weighting.
        """
        votes: dict[str, float] = {}
        for col in self.columns:
            if col.recognized is not None and col.confidence > 0.0:
                obj = col.recognized
                votes[obj] = votes.get(obj, 0.0) + col.confidence

        if not votes:
            return None, 0.0
        winner = max(votes, key=votes.get)
        total = sum(votes.values())
        return winner, votes[winner] / total if total > 0 else 0.0

    def get_messages(self) -> list[CorticalMessage]:
        return [col.message() for col in self.columns]


class FovealExplorer:
    """Explores images through saccades, building object models."""

    def __init__(self, eye: Eye, v1: RetinotopicV1, n_fixations: int = 3):
        self.eye = eye
        self.v1 = v1
        self.n_fixations = n_fixations

    def explore(self, image: np.ndarray, object_id: str | None = None,
                learn: bool = True) -> tuple[str | None, float]:
        """Explore an image via saccades. Learn or recognize.

        Training: object_id provided → learn (feature, location) bindings.
        Testing: object_id=None → recognize from stored models.

        Returns (recognized_object_id, confidence).
        """
        h, w = image.shape[:2]
        fixations = SalienceMap.suggest_fixations(
            image, n=self.n_fixations,
            min_distance=max(3, self.eye.retina_size // 4),
        )
        if not fixations:
            fixations = [(w // 2, h // 2)]

        # Reset reference frame for this object.
        self.v1.reset_locations()

        # First fixation (origin of reference frame).
        fx, fy = fixations[0]
        self.eye.fixate(float(fx), float(fy))
        self.v1.observe(image)

        if learn and object_id is not None:
            self.v1.learn_object(object_id)

        # Subsequent fixations: saccade → update location → observe → learn.
        for fx, fy in fixations[1:]:
            self.eye.saccade_to(float(fx), float(fy))
            dx, dy = self.eye.last_displacement
            self.v1.update_locations(dx, dy)
            self.v1.observe(image)

            if learn and object_id is not None:
                self.v1.learn_object(object_id)

        # Recognition: aggregate column votes.
        return self.v1.recognize()
