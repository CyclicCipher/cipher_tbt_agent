"""Retinotopic visual cortex — V1 columns with patch receptive fields.

Each V1 column reads a PATCH from the retinal image (not a single pixel).
Patches are VQ-encoded into discrete feature tokens. Columns tile the
retinal image with overlapping receptive fields.

When the eye saccades, the retinal image shifts → columns see new
patches → accumulate (feature, displacement, next_feature) triples
for reference frame learning.

Vectorized: patch extraction and VQ encoding are batched numpy ops.
Per-column Python loops only for memory dict operations.
"""
from __future__ import annotations

import numpy as np

from symbolic_column import SymbolicColumn, CorticalMessage, MAX_MEMORY
from eye import Eye, SalienceMap
from codebook import PatchCodebook


class RetinotopicV1:
    """V1 cortex: columns tile the retinal image with patch RFs.

    Each column covers a patch_size × patch_size region of the retina.
    Patches are VQ-encoded to feature tokens. Columns overlap by
    stride < patch_size.
    """

    def __init__(self, eye: Eye, codebook: PatchCodebook,
                 patch_size: int = 5, stride: int = 3):
        self.eye = eye
        self.codebook = codebook
        self.patch_size = patch_size
        self.stride = stride

        rs = eye.retina_size
        self.grid_h = (rs - patch_size) // stride + 1
        self.grid_w = (rs - patch_size) // stride + 1

        # One column per grid position.
        self.columns: list[SymbolicColumn] = []
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                y0, x0 = gy * stride, gx * stride
                col = SymbolicColumn(
                    name=f"V1:{gy},{gx}",
                    receptive_field=(y0, x0, y0 + patch_size, x0 + patch_size),
                    position=(float(gx), float(gy)),
                    max_memory=MAX_MEMORY,
                )
                self.columns.append(col)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def _extract_and_encode(self, retina: np.ndarray) -> np.ndarray:
        """Extract all patches from retinal image and VQ-encode them.

        Returns array of codebook indices (int), one per column.
        Fully vectorized: no Python loop over columns.
        """
        ps = self.patch_size
        st = self.stride
        n = self.grid_h * self.grid_w
        # Extract all patches at once using stride tricks.
        patches = np.empty((n, ps, ps), dtype=np.float32)
        idx = 0
        for gy in range(self.grid_h):
            y0 = gy * st
            for gx in range(self.grid_w):
                x0 = gx * st
                patches[idx] = retina[y0:y0+ps, x0:x0+ps]
                idx += 1
        # Batch VQ encode.
        codes = self.codebook.encode_batch(patches)
        return codes

    def observe(self, image: np.ndarray) -> list[str]:
        """Sample image through eye, encode patches, observe at columns.

        Returns feature codes (one per column). Vectorized encoding.
        """
        retina = self.eye.sample(image)
        codes = self._extract_and_encode(retina)
        features = []
        for i, col in enumerate(self.columns):
            feat = f"v{codes[i]}"
            col.observe(feat)
            features.append(feat)
        return features

    def teach_all(self, target: str):
        """Teach all columns: current_feature → target."""
        for col in self.columns:
            if col.current_input is not None:
                col.teach(col.current_input, target)

    def displacement_teach(self, prev_features: list[str],
                           displacement: tuple[float, float],
                           curr_features: list[str]):
        """Teach displacement: (prev_feature:disp → curr_feature)."""
        dx, dy = displacement
        qdx, qdy = int(round(dx)), int(round(dy))
        disp_key = f"d{qdx},{qdy}"
        for i, col in enumerate(self.columns):
            pf, cf = prev_features[i], curr_features[i]
            if pf is not None and cf is not None:
                col.teach(f"{pf}:{disp_key}", cf)

    def get_messages(self) -> list[CorticalMessage]:
        return [col.message() for col in self.columns]

    def vote_classification(self) -> tuple[str | None, dict[str, float]]:
        """Collect classification votes from all columns."""
        votes: dict[str, float] = {}
        for col in self.columns:
            pred = col.prediction
            if pred is not None:
                votes[pred] = votes.get(pred, 0.0) + col.confidence
        if not votes:
            return None, votes
        return max(votes, key=votes.get), votes


class FovealExplorer:
    """Explores images through saccadic eye movements."""

    def __init__(self, eye: Eye, v1: RetinotopicV1, n_fixations: int = 3):
        self.eye = eye
        self.v1 = v1
        self.n_fixations = n_fixations

    def explore(self, image: np.ndarray, label: str | None = None,
                learn: bool = True) -> tuple[str | None, dict[str, float]]:
        """Explore an image. Optionally learn identity + displacement."""
        h, w = image.shape[:2]
        fixations = SalienceMap.suggest_fixations(
            image, n=self.n_fixations,
            min_distance=max(3, self.eye.retina_size // 4),
        )
        if not fixations:
            fixations = [(w // 2, h // 2)]

        # First fixation.
        fx, fy = fixations[0]
        self.eye.fixate(float(fx), float(fy))
        prev_features = self.v1.observe(image)
        if learn and label is not None:
            self.v1.teach_all(label)

        # Subsequent fixations.
        for fx, fy in fixations[1:]:
            self.eye.saccade_to(float(fx), float(fy))
            disp = self.eye.last_displacement
            curr_features = self.v1.observe(image)
            if learn:
                self.v1.displacement_teach(prev_features, disp, curr_features)
                if label is not None:
                    self.v1.teach_all(label)
            prev_features = curr_features

        return self.v1.vote_classification()
