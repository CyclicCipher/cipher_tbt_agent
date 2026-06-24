"""Phase-2 prototype — the SHAPE BENCHMARK (docs/phase2/VOLUME_CONCEPTS.md §8 Q1).

Synthetic concepts of *known* true shape (axis box, rotated box/diamond, triangle, disk, two blobs),
each fit by two competing concept families:
  - the axis-aligned box (`volume.box`)            — a fixed shape,
  - the MDL half-space intersection (`volume.halfspace`) — shape discovered, not chosen.

The decisive question (the bitter-lesson test): does letting MDL choose the shape *match the box when
the truth is boxy* and *beat it when the truth is tilted/curved* — without us ever naming the shape?
Each fixed family should only win on concepts matching its bias; MDL-selection should track the best
across the whole battery. `two_blobs` is non-convex: it shows the convex limit (→ motivates `union`).
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .box import fit_box_concept
from .halfspace import fit_halfspace_concept

# --- known-truth shape predicates on the 2-D domain [-1, 1]^2 -------------------------------------

SHAPES: Dict[str, Callable[[np.ndarray], bool]] = {
    "axis_box":    lambda p: max(abs(p[0]), abs(p[1])) <= 0.5,          # 4 axis faces
    "rotated_box": lambda p: abs(p[0]) + abs(p[1]) <= 0.6,              # 4 oriented faces (diamond)
    "triangle":    lambda p: p[0] >= -0.6 and p[1] >= -0.6 and p[0] + p[1] <= 0.0,  # 2 axis + 1 oriented
    "disk":        lambda p: p[0] ** 2 + p[1] ** 2 <= 0.5 ** 2,         # many oriented faces
    "two_blobs":   lambda p: (np.hypot(p[0] + 0.5, p[1] + 0.5) <= 0.3   # non-convex (needs union)
                              or np.hypot(p[0] - 0.5, p[1] - 0.5) <= 0.3),
}


def sample(pred, n: int, rng: np.random.Generator, d: int = 2):
    pos, neg = [], []
    while len(pos) < n or len(neg) < n:
        x = rng.uniform(-1.0, 1.0, size=d)
        if pred(x):
            if len(pos) < n:
                pos.append(x)
        elif len(neg) < n:
            neg.append(x)
    return np.array(pos), np.array(neg)


def accuracy(concept, pos, neg) -> float:
    c = sum(concept.contains(p) for p in pos) + sum(not concept.contains(n) for n in neg)
    return c / (len(pos) + len(neg))


def evaluate(shape: str, seed: int = 0, n_train: int = 250, n_test: int = 200) -> Dict:
    rng = np.random.default_rng(seed)
    pred = SHAPES[shape]
    tr_pos, tr_neg = sample(pred, n_train, rng)
    te_pos, te_neg = sample(pred, n_test, rng)

    box = fit_box_concept(tr_pos, tr_neg, n_features=2)
    hs = fit_halfspace_concept(tr_pos, tr_neg, n_features=2)
    return {
        "shape": shape,
        "box_acc": accuracy(box, te_pos, te_neg),
        "hs_acc": accuracy(hs, te_pos, te_neg),
        "hs_faces": hs.n_faces,
        "hs_oriented": hs.n_oriented,
    }


def run(seed: int = 0) -> List[Dict]:
    return [evaluate(s, seed) for s in SHAPES]


if __name__ == "__main__":
    print(f"{'shape':<12} {'box_acc':>8} {'hs_acc':>8} {'faces':>6} {'oriented':>9}   winner")
    for r in run():
        win = "tie" if abs(r["hs_acc"] - r["box_acc"]) < 0.02 else (
            "half-space" if r["hs_acc"] > r["box_acc"] else "box")
        print(f"{r['shape']:<12} {r['box_acc']:>8.3f} {r['hs_acc']:>8.3f} "
              f"{r['hs_faces']:>6} {r['hs_oriented']:>9}   {win}")
