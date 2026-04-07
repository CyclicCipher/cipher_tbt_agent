"""Symbolic Brain — hierarchical cortical columns.

Same SymbolicColumn at every level. Lower levels' votes become
higher levels' features. This is TBT's cortical messaging protocol.

No domain-specific code: succession and vision use the same columns.
"""
from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from graph import Graph
from prior_loader import load_priors
from token_io import TokenIO
from symbolic_column import (
    SymbolicColumn, ColumnSheet, ColumnHierarchy, SuccessionEngine,
)
from codebook import PatchCodebook


class SymbolicBrain:
    """Brain with hierarchical symbolic cortical columns."""

    def __init__(self):
        self.graph, self.priors = load_priors()
        self.tio = TokenIO(self.graph, self.priors)

        # Succession column (token domain).
        self.succession = SymbolicColumn("succession")

        # Visual hierarchy (created on demand).
        self.visual: ColumnHierarchy | None = None
        self.codebook: PatchCodebook | None = None
        self._vis_patch_size: int = 4
        self._vis_stride: int = 4

        # Output node lookup.
        self._output_nodes: dict[str, int] = {}
        for key, nid in self.priors.get('output_cortex', {}).items():
            node = self.graph.get_node(nid)
            if node and node.meta.get('token'):
                self._output_nodes[node.meta['token']] = nid

    # ----- Succession -----

    def train_succession(self, pairs: list[tuple[str, str]]):
        for token, next_token in pairs:
            self.succession.teach(token, next_token)

    def predict_successor(self, token: str) -> str | None:
        self.succession.observe(token)
        return self.succession.predict()

    def predict_number_successor(self, number_str: str) -> str:
        return SuccessionEngine.successor(number_str)

    # ----- Visual hierarchy -----

    def init_visual(self, image_shape: tuple[int, int] = (28, 28),
                    patch_size: int = 4, stride: int = 4,
                    n_codes: int = 512, n_levels: int = 2,
                    pool: int = 2):
        """Create a visual cortex hierarchy with N levels.

        Level 0: image pixels → VQ codes. Grid = image/stride.
        Level 1+: pools lower level columns. Grid = lower/pool.

        Args:
            image_shape: (H, W) of input images.
            patch_size: pixel patch size for level 0.
            stride: pixel stride for level 0.
            n_codes: VQ codebook size.
            n_levels: number of hierarchy levels.
            pool: pooling factor between levels (pool×pool lower → 1 higher).
        """
        self._vis_patch_size = patch_size
        self._vis_stride = stride
        self.codebook = PatchCodebook(n_codes=n_codes)

        h, w = image_shape
        grid_h = (h - patch_size) // stride + 1
        grid_w = (w - patch_size) // stride + 1

        self.visual = ColumnHierarchy()

        # Level 0: raw VQ codes from pixel patches.
        level0 = ColumnSheet(
            "V1", grid_h, grid_w,
            rf_size=(patch_size, patch_size),
            stride=(stride, stride),
        )
        self.visual.add_level(level0, pool_h=pool, pool_w=pool)

        # Higher levels: each pools from the level below.
        cur_h, cur_w = grid_h, grid_w
        for lev in range(1, n_levels):
            next_h = max(1, cur_h // pool)
            next_w = max(1, cur_w // pool)
            sheet = ColumnSheet(f"V{lev+1}", next_h, next_w)
            self.visual.add_level(sheet, pool_h=pool, pool_w=pool)
            cur_h, cur_w = next_h, next_w

        print(f"Visual hierarchy: {self.visual}")
        for i, lev in enumerate(self.visual.levels):
            print(f"  Level {i}: {lev}")

    def train_codebook(self, images: np.ndarray, max_patches: int = 50000,
                       verbose: bool = True):
        """Pre-train VQ codebook on sampled patches."""
        ps = self._vis_patch_size
        st = self._vis_stride
        level0 = self.visual.levels[0]
        rng = np.random.RandomState(42)

        n_sample = min(max_patches, len(images) * level0.n_columns())
        sample_idx = rng.randint(0, len(images), size=n_sample)
        sample_gy = rng.randint(0, level0.grid_h, size=n_sample)
        sample_gx = rng.randint(0, level0.grid_w, size=n_sample)

        patches = np.empty((n_sample, ps, ps), dtype=np.float32)
        for i in range(n_sample):
            y0, x0 = sample_gy[i] * st, sample_gx[i] * st
            patches[i] = images[sample_idx[i], y0:y0+ps, x0:x0+ps]

        if verbose:
            print(f"  Sampled {n_sample} patches")
        self.codebook.fit(patches, verbose=verbose)

    def _feed_image(self, image: np.ndarray):
        """Feed an image through the visual hierarchy.

        Level 0: encode pixel patches → VQ codes → observe.
        Level 1+: compose lower-level messages → observe.
        """
        self.visual.reset()

        # Level 0: pixel patches → VQ codes.
        level0 = self.visual.levels[0]
        ps, st = self._vis_patch_size, self._vis_stride
        patches = []
        coords = []
        for gy in range(level0.grid_h):
            for gx in range(level0.grid_w):
                y0, x0 = gy * st, gx * st
                patches.append(image[y0:y0+ps, x0:x0+ps])
                coords.append((gy, gx))

        codes = self.codebook.encode_batch(np.array(patches))
        for (gy, gx), code in zip(coords, codes):
            level0.columns[gy][gx].observe(f"v{code}")

        # Propagate upward through hierarchy.
        self.visual.propagate()

    def train_image(self, image: np.ndarray, label: int):
        """One-shot: feed image, teach all levels the label."""
        self._feed_image(image)
        self.visual.teach_all(str(label))

    def classify_image(self, image: np.ndarray) -> tuple[str | None, dict]:
        """Classify: feed image, collect votes from all levels."""
        self._feed_image(image)
        return self.visual.vote()

    # ----- Output cortex -----

    def read_output(self, prediction: str | None) -> tuple[str | None, bool]:
        if prediction is None:
            return None, False
        self.tio.clear_output()
        nid = self._output_nodes.get(prediction)
        if nid is not None:
            self.graph.activate(nid, 1.0)
            self.graph.step()
        token, act = self.tio.read_output()
        return token, act > 0.01
