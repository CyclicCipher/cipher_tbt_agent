"""Phase-2 prototype — a multi-modal CONCEPT as a UNION of convex regions, with MDL discovering the
number of modes. Closes the convex limit the shape benchmark exposed (two_blobs): a single convex
polytope cannot capture disjoint senses ("bank" = riverbank ∪ finance), so a concept becomes a SET
of regions, membership = OR.

The number of modes is not specified — MDL picks it. For each candidate M we cluster the positives
(k-means), fit a half-space concept per cluster (vs all negatives), and score the union's two-part
code. More modes cost model bits and are bought only when they separate positives a single convex
region cannot — so a convex concept stays M=1 (parsimony), a two-blob concept becomes M=2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .halfspace import HalfspaceConcept, fit_halfspace_concept, model_bits

_EPS = 1e-9


@dataclass(frozen=True)
class UnionConcept:
    regions: Tuple[HalfspaceConcept, ...]

    def contains(self, x) -> bool:
        return any(r.contains(x) for r in self.regions)

    @property
    def n_modes(self) -> int:
        return len(self.regions)

    @property
    def n_faces(self) -> int:
        return sum(r.n_faces for r in self.regions)


def _kmeans(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25) -> np.ndarray:
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for step in range(iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = d2.argmin(axis=1)
        if step > 0 and np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            pts = X[labels == c]
            if len(pts):
                centers[c] = pts.mean(axis=0)
    return labels


def _inside_any(regions, pts: np.ndarray) -> np.ndarray:
    hit = np.zeros(len(pts), dtype=bool)
    for r in regions:
        if r.W.shape[0] == 0:
            return np.ones(len(pts), dtype=bool)
        hit |= np.all(pts @ r.W.T <= r.b[None, :] + _EPS, axis=1)
    return hit


def _cell_bits(pos: int, neg: int) -> float:
    total = pos + neg
    if total == 0:
        return 0.0
    bits = 0.0
    for c in (pos, neg):
        if c > 0:
            bits -= c * math.log2(c / total)
    return bits


def _union_mdl(regions, P: np.ndarray, N: np.ndarray) -> float:
    ip = int(_inside_any(regions, P).sum())
    inn = int(_inside_any(regions, N).sum())
    data = _cell_bits(ip, inn) + _cell_bits(len(P) - ip, len(N) - inn)
    return sum(model_bits(r) for r in regions) + data


def fit_union_concept(positives, negatives, n_features: int,
                      max_modes: int = 4, seed: int = 0) -> UnionConcept:
    """Discover a multi-modal concept: search M = 1..max_modes, cluster the positives, fit a
    convex (half-space) region per mode, and keep the M with the lowest union MDL. M is found,
    never given; a convex concept collapses back to M=1."""
    P = np.asarray(positives, dtype=float)
    N = np.asarray(negatives, dtype=float)
    rng = np.random.default_rng(seed)
    best = None
    for m in range(1, max_modes + 1):
        if m > len(P):
            break
        labels = np.zeros(len(P), dtype=int) if m == 1 else _kmeans(P, m, rng)
        regions = [fit_halfspace_concept(P[labels == c], N, n_features)
                   for c in range(m) if (labels == c).any()]
        mdl = _union_mdl(regions, P, N)
        if best is None or mdl < best[0] - _EPS:
            best = (mdl, regions)
    return UnionConcept(tuple(best[1]))
