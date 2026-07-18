"""hippocampus/dg.py — dentate gyrus PATTERN SEPARATION → orthogonal chart keys (DESIGN §2/§3, slice 4).

Marr/Treves/Rolls: the dentate gyrus performs pattern SEPARATION "by competitive learning to produce sparse representations
suitable for setting up new representations in CA3... neurons with place-like fields from entorhinal grid cells." Its job here
is to turn an environment SIGNATURE into a sparse, orthogonalized CHART KEY (which map am I in?). It DECORRELATES: distinct
environments get well-separated (near-disjoint) keys, so their memories do not interfere in CA3, while similar environments
keep proportionally more key overlap (monotonic) — the base for recognising a chart on return and for multi-chart REMAPPING
(slice 5). DG separation is exactly what keeps CA3 (slice 3) from cross-talking on overlapping memories.

REUSES the k-WTA `SpatialPooler` (ARCHITECTURE: extract one canonical component and USE it, never reimplement): a fixed sparse
projection + winner-take-all IS pattern separation — sparse codes over a random-ish expansion rarely collide, and the top-w
inhibition sharpens small input differences into large output ones. Deterministic with `learn=False`, so the SAME environment
always yields the SAME key (a chart is recognised on return); competitive LEARNING (`learn=True`) sharpens it on exposure.
Not an ANN — HTM competitive sparse coding, the same substrate the columns use.
"""

from __future__ import annotations

import numpy as np

from ..encoders import SpatialPooler


class DG:
    """Dentate gyrus pattern separation: `separate(signature)` → a sparse CHART KEY (a frozenset of key bits). A signature is
    a set of active input bits describing the environment; the key is the pooler's sparse, decorrelated code of it."""

    def __init__(self, n_inputs: int, n_cols: int = 1024, w: int = 24, seed: int = 0) -> None:
        self.n_inputs = int(n_inputs)
        self.pooler = SpatialPooler(n_inputs=self.n_inputs, n_cols=n_cols, w=w, seed=seed)

    def separate(self, signature, learn: bool = False) -> frozenset:
        """An environment signature (a set of active input bits) → its sparse CHART KEY. Deterministic by default, so a
        returning agent gets the SAME key for the SAME environment; `learn=True` sharpens the separation on repeated exposure."""
        x = np.zeros(self.n_inputs)
        idx = [i for i in signature if 0 <= i < self.n_inputs]
        if idx:
            x[idx] = 1.0
        return frozenset(self.pooler.encode(x, learn=learn).active)
