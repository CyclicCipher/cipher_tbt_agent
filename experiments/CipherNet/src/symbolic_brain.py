"""Symbolic Brain — domain-general cortical column system.

Same SymbolicColumn handles succession AND image classification.
No domain-specific code: teach() accumulates votes, predict() returns
the mode. The column doesn't know if it's processing tokens or pixels.
"""
from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from graph import Graph
from prior_loader import load_priors
from token_io import TokenIO
from symbolic_column import SymbolicColumn, ColumnSheet2D, SuccessionEngine
from codebook import PatchCodebook


class SymbolicBrain:
    """Brain with symbolic cortical columns.

    Succession: single column, teach(token, next_token).
    Vision: 2D column sheet + VQ codebook, teach(patch_code, label).
    Same SymbolicColumn for both. No domain-specific code.
    """

    def __init__(self):
        self.graph, self.priors = load_priors()
        self.tio = TokenIO(self.graph, self.priors)

        # Succession column.
        self.succession = SymbolicColumn("succession")

        # Visual cortex (created on demand).
        self.visual_cortex: ColumnSheet2D | None = None
        self.codebook: PatchCodebook | None = None

        # Output node lookup.
        self._output_nodes: dict[str, int] = {}
        for key, nid in self.priors.get('output_cortex', {}).items():
            node = self.graph.get_node(nid)
            if node and node.meta.get('token'):
                self._output_nodes[node.meta['token']] = nid

    # ----- Succession (tokens) -----

    def train_succession(self, pairs: list[tuple[str, str]]):
        for token, next_token in pairs:
            self.succession.teach(token, next_token)

    def predict_successor(self, token: str) -> str | None:
        self.succession.observe(token)
        return self.succession.predict()

    def predict_number_successor(self, number_str: str) -> str:
        return SuccessionEngine.successor(number_str)

    # ----- Vision (images) -----

    def init_visual_cortex(self, image_shape: tuple[int, int] = (28, 28),
                           patch_size: int = 4, stride: int = 4,
                           n_codes: int = 256):
        """Create visual cortex and codebook."""
        self.visual_cortex = ColumnSheet2D(
            "visual_cortex", image_shape,
            patch_size=patch_size, stride=stride,
        )
        self.codebook = PatchCodebook(n_codes=n_codes)
        print(f"Visual cortex: {self.visual_cortex}")

    def train_codebook(self, images: np.ndarray, max_patches: int = 50000,
                       verbose: bool = True):
        """Pre-train VQ codebook on a SAMPLE of image patches.

        Samples max_patches random patches (not all 2.9M) to avoid
        memory explosion. 50K patches is more than enough for 256 centroids.
        """
        ps = self.visual_cortex.patch_size
        stride = self.visual_cortex.stride
        gh, gw = self.visual_cortex.grid_h, self.visual_cortex.grid_w
        n_per_image = gh * gw
        total_patches = len(images) * n_per_image

        # Sample random image indices + patch positions.
        rng = np.random.RandomState(42)
        n_sample = min(max_patches, total_patches)
        sample_idx = rng.randint(0, len(images), size=n_sample)
        sample_gy = rng.randint(0, gh, size=n_sample)
        sample_gx = rng.randint(0, gw, size=n_sample)

        patches = np.empty((n_sample, ps, ps), dtype=np.float32)
        for i in range(n_sample):
            y0 = sample_gy[i] * stride
            x0 = sample_gx[i] * stride
            patches[i] = images[sample_idx[i], y0:y0+ps, x0:x0+ps]

        if verbose:
            print(f"  Sampled {n_sample} patches (of {total_patches} total)")
        self.codebook.fit(patches, verbose=verbose)

    def train_image(self, image: np.ndarray, label: int):
        """Teach all visual cortex columns: patch_at_position → label."""
        label_str = str(label)
        patches_info = self.visual_cortex.extract_patches(image)
        # Batch encode all patches at once.
        patch_array = np.array([p for _, _, p in patches_info])
        codes = self.codebook.encode_batch(patch_array)
        for i, (gy, gx, _) in enumerate(patches_info):
            self.visual_cortex.columns[gy][gx].teach(f"v{codes[i]}", label_str)

    def classify_image(self, image: np.ndarray) -> tuple[str | None, dict[str, int]]:
        """Classify an image by TBT column voting."""
        patches_info = self.visual_cortex.extract_patches(image)
        patch_array = np.array([p for _, _, p in patches_info])
        codes = self.codebook.encode_batch(patch_array)
        votes: dict[str, int] = {}
        for i, (gy, gx, _) in enumerate(patches_info):
            col = self.visual_cortex.columns[gy][gx]
            feature = f"v{codes[i]}"
            col.observe(feature)
            pred = col.predict()
            if pred is not None:
                votes[pred] = votes.get(pred, 0) + 1
        if not votes:
            return None, votes
        return max(votes, key=votes.get), votes

    # ----- Output cortex interface -----

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
