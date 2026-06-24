"""Phase-2 prototype — a CONCEPT as a *box* (volume) discovered in its OWN low-dimensional
subspace by an MDL two-part code, never handed the relevant dimensions.

This is the smallest test of `docs/phase2/VOLUME_CONCEPTS.md`:
  - **Resolution 1** — a concept lives in its own small subspace, *not* the global space. A box is
    unconstrained on every dimension it does not care about.
  - **"MDL decides the geometry"** — both *which* dimensions a concept needs and the region within
    them. A box fit on the right K of N dimensions ignores the N-K noise dimensions purely because
    encoding an extra dimension costs more bits than the misclassifications it removes.

No dimension is given. The subspace is discovered by greedy forward selection under a two-part code
(model = the box; data = the exceptions). This is the discrete, single-concept seed of the
"graph of local spaces" — one node, fit from data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Vector = Sequence[float]

_RES = 64  # assumed coordinate resolution (bits per bound = log2 _RES); only relative DL matters


@dataclass(frozen=True)
class BoxConcept:
    """A region = an axis-aligned box over a chosen subspace `dims` (indices into the full space).
    Unconstrained on every dimension not in `dims` — that is the whole point: the concept is
    defined only where it cares. Empty `dims` ⇒ contains everything (the "no structure" concept)."""

    dims: Tuple[int, ...]
    lo: Tuple[float, ...]
    hi: Tuple[float, ...]

    def contains(self, x: Vector) -> bool:
        return all(self.lo[i] <= x[d] <= self.hi[i] for i, d in enumerate(self.dims))

    @property
    def n_dims(self) -> int:
        return len(self.dims)


def _enclosing_box(positives: Sequence[Vector], dims: Sequence[int]) -> BoxConcept:
    """Tightest axis-aligned box on `dims` that contains every positive."""
    dims = tuple(dims)
    if len(positives) == 0:
        return BoxConcept(dims, tuple(0.0 for _ in dims), tuple(0.0 for _ in dims))
    lo = tuple(min(p[d] for p in positives) for d in dims)
    hi = tuple(max(p[d] for p in positives) for d in dims)
    return BoxConcept(dims, lo, hi)


def _errors(box: BoxConcept, positives, negatives) -> int:
    fp = sum(1 for n in negatives if box.contains(n))      # negative inside the box
    fn = sum(1 for p in positives if not box.contains(p))  # positive outside (0 on training fit)
    return fp + fn


def _cell_label_bits(pos: int, neg: int) -> float:
    """Bits to encode the labels of one region cell = total · H(label) = -Σ c·log2(c/total)."""
    total = pos + neg
    if total == 0:
        return 0.0
    bits = 0.0
    for c in (pos, neg):
        if c > 0:
            bits -= c * math.log2(c / total)
    return bits


def _description_length(box: BoxConcept, positives, negatives, n_features: int) -> float:
    """Two-part MDL code, in bits: model (the box) + data (the labels given the box).

    The data term is the *conditional label entropy* — the cost of encoding each example's
    label given only whether it is inside or outside the box. This rewards a dimension only when
    it genuinely separates positives from negatives; it does NOT reward shaving the distribution
    tails (the bug that let a box hallucinate structure from noise). `model` charges each used
    dimension for two bounds and its identity."""
    bits_per_dim = 2 * math.log2(_RES) + math.log2(max(n_features, 2))
    model_bits = box.n_dims * bits_per_dim
    in_p = sum(1 for p in positives if box.contains(p))
    in_n = sum(1 for n in negatives if box.contains(n))
    out_p, out_n = len(positives) - in_p, len(negatives) - in_n
    data_bits = _cell_label_bits(in_p, in_n) + _cell_label_bits(out_p, out_n)
    return model_bits + data_bits


def fit_box_concept(positives: Sequence[Vector], negatives: Sequence[Vector],
                    n_features: int) -> BoxConcept:
    """Discover the concept: greedy forward selection of the subspace that minimizes the two-part
    MDL code. The relevant dimensions are *found*, never supplied. Returns a `BoxConcept` whose
    `dims` is the discovered subspace (possibly empty, if the labels carry no boxable structure)."""
    chosen: List[int] = []
    best_box = _enclosing_box(positives, chosen)  # empty subspace: contains everything
    best_dl = _description_length(best_box, positives, negatives, n_features)
    improving = True
    while improving:
        improving = False
        best_candidate = None
        for d in range(n_features):
            if d in chosen:
                continue
            box = _enclosing_box(positives, chosen + [d])
            dl = _description_length(box, positives, negatives, n_features)
            if dl < best_dl - 1e-9 and (best_candidate is None or dl < best_candidate[0]):
                best_candidate = (dl, d, box)
        if best_candidate is not None:
            best_dl, d, best_box = best_candidate
            chosen.append(d)
            improving = True
    return best_box


if __name__ == "__main__":  # quick visual sanity check
    import random

    rng = random.Random(0)
    rel, rng_ = [1, 4], {1: (0.3, 0.7), 4: (0.6, 0.9)}
    pos, neg = [], []
    while len(pos) < 250 or len(neg) < 250:
        x = [rng.random() for _ in range(6)]
        ok = all(rng_[d][0] <= x[d] <= rng_[d][1] for d in rel)
        (pos if ok and len(pos) < 250 else neg if not ok and len(neg) < 250 else []).append(x)
    box = fit_box_concept(pos, neg, n_features=6)
    print("planted relevant dims :", rel, rng_)
    print("discovered dims       :", box.dims)
    print("discovered region     :", {d: (round(box.lo[i], 2), round(box.hi[i], 2))
                                       for i, d in enumerate(box.dims)})
