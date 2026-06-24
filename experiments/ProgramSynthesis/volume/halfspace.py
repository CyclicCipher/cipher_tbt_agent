"""Phase-2 prototype — a CONCEPT as an intersection of K half-spaces, with MDL discovering BOTH the
number of faces AND their orientation. The region SHAPE is not chosen; it emerges.

This is the bitter-lesson treatment of region shape (docs/phase2/VOLUME_CONCEPTS.md §8 Q1 / the shape
discussion): instead of picking a family (box / ellipsoid / cone), we offer ONE general primitive —
the half-space (a linear inequality `w·x ≤ b`) — and let the SAME MDL objective that picks a
concept's subspace also pick how many half-spaces it needs and how they are tilted.

The asymmetry that makes shape *emerge*: an AXIS-ALIGNED half-space is cheap to encode (which axis +
sign + one offset), an ORIENTED one is expensive (a whole normal vector). So MDL defaults to a box and
only buys orientation when a tilted face removes enough negatives to pay for itself. A box is the
special case recovered when the data is boxy; a rotated polytope is what you get when it isn't —
neither is decreed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

_RES = 64        # coordinate resolution (bits per quantized number = log2 _RES)
_EPS = 1e-9


@dataclass(frozen=True)
class HalfspaceConcept:
    W: np.ndarray            # (K, d) outward normals
    b: np.ndarray            # (K,) offsets — membership is W @ x <= b
    axis: Tuple[bool, ...]   # per-face: is it an axis-aligned cut? (cheap) — for MDL + readability

    def contains(self, x) -> bool:
        if self.W.shape[0] == 0:
            return True
        return bool(np.all(self.W @ np.asarray(x, float) <= self.b + _EPS))

    @property
    def n_faces(self) -> int:
        return int(self.W.shape[0])

    @property
    def n_oriented(self) -> int:
        return sum(1 for a in self.axis if not a)


def _inside(pts: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    if W.shape[0] == 0:
        return np.ones(len(pts), dtype=bool)
    return np.all(pts @ W.T <= b[None, :] + _EPS, axis=1)


def _cell_bits(pos: int, neg: int) -> float:
    total = pos + neg
    if total == 0:
        return 0.0
    bits = 0.0
    for c in (pos, neg):
        if c > 0:
            bits -= c * math.log2(c / total)
    return bits


def _model_bits(axis_flags: Sequence[bool], d: int) -> float:
    per_axis = math.log2(2 * d) + math.log2(_RES)            # which axis+sign, + offset
    per_oriented = d * math.log2(_RES) + math.log2(_RES)     # full normal, + offset
    return sum(per_axis if a else per_oriented for a in axis_flags)


def model_bits(concept: HalfspaceConcept) -> float:
    """Description length (bits) of a concept's *model* — its faces, by the axis-cheap /
    oriented-expensive code. Used by the union fitter (union.py) to price extra modes."""
    return _model_bits(concept.axis, concept.W.shape[1])


def _mdl(P, N, W, b, axis_flags) -> float:
    ip = int(_inside(P, W, b).sum())
    inn = int(_inside(N, W, b).sum())
    data = _cell_bits(ip, inn) + _cell_bits(len(P) - ip, len(N) - inn)
    return _model_bits(axis_flags, P.shape[1]) + data


def fit_halfspace_concept(positives, negatives, n_features: int,
                          max_faces: int = 24) -> HalfspaceConcept:
    """Greedy forward selection of half-spaces under the two-part MDL code. Every candidate keeps
    ALL positives inside (its offset is the tightest that does), so faces only ever cut negatives.
    Candidates each round: the 2·d cheap axis-aligned cuts, plus one oriented cut per still-inside
    negative (normal = direction from its nearest positive). The number and tilt of faces — the
    shape — are discovered, never specified."""
    P = np.asarray(positives, dtype=float)
    N = np.asarray(negatives, dtype=float)
    W = np.zeros((0, n_features))
    b = np.zeros((0,))
    axis: List[bool] = []
    best = _mdl(P, N, W, b, axis)

    while len(axis) < max_faces:
        inside_neg = N[_inside(N, W, b)]
        if len(inside_neg) == 0:
            break

        candidates: List[Tuple[np.ndarray, float, bool]] = []
        for dd in range(n_features):       # cheap axis-aligned cuts (both signs)
            e = np.zeros(n_features); e[dd] = 1.0
            candidates.append((e.copy(), float((P @ e).max()), True))
            candidates.append((-e, float((P @ -e).max()), True))
        for n in inside_neg[:48]:          # expensive oriented cuts, one per inside-negative
            j = int(np.argmin(((P - n) ** 2).sum(axis=1)))
            w = n - P[j]
            nrm = float(np.linalg.norm(w))
            if nrm < _EPS:
                continue
            w = w / nrm
            candidates.append((w, float((P @ w).max()), False))

        pick = None
        for w, bv, is_axis in candidates:
            W2 = np.vstack([W, w]); b2 = np.append(b, bv); ax2 = axis + [is_axis]
            m = _mdl(P, N, W2, b2, ax2)
            if pick is None or m < pick[0]:
                pick = (m, W2, b2, ax2)

        if pick is None or pick[0] >= best - _EPS:
            break
        best, W, b, axis = pick

    return HalfspaceConcept(W, b, tuple(axis))
