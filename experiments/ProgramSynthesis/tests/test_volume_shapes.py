"""Phase-2 prototype — the shape benchmark (docs/phase2/VOLUME_CONCEPTS.md §8 Q1): does letting MDL
choose a concept's shape (number + orientation of half-spaces from one general primitive) recover a
box when the truth is boxy, and beat the fixed box when it isn't — without ever naming the shape?
`evaluate(shape, seed=0)` is deterministic."""

import random

from volume.benchmark import evaluate
from volume.halfspace import fit_halfspace_concept


def test_axis_box_recovered_without_inventing_orientation():
    r = evaluate("axis_box")
    assert r["hs_oriented"] == 0                 # parsimony: no orientation bought when not needed
    assert r["hs_acc"] >= 0.95
    assert abs(r["hs_acc"] - r["box_acc"]) <= 0.03   # ties the fixed box on its home turf


def test_rotated_box_discovers_orientation_and_beats_the_box():
    r = evaluate("rotated_box")
    assert r["hs_oriented"] >= 1                 # orientation discovered only because it pays
    assert r["hs_acc"] > r["box_acc"]
    assert r["hs_acc"] >= 0.95


def test_triangle_mixes_axis_and_oriented_faces_and_wins():
    r = evaluate("triangle")
    assert r["hs_oriented"] >= 1                          # the hypotenuse
    assert r["hs_faces"] - r["hs_oriented"] >= 1          # plus cheap axis faces
    assert r["hs_acc"] > r["box_acc"]
    assert r["hs_acc"] >= 0.95


def test_two_blobs_exposes_the_convex_limit():
    # a single convex polytope cannot separate two disjoint blobs — better than the box, but far
    # from solved. This is the honest motivation for `union` (a set of regions), the deferred op.
    r = evaluate("two_blobs")
    assert r["hs_acc"] > r["box_acc"]
    assert r["hs_acc"] < evaluate("axis_box")["hs_acc"]   # clearly worse than a convex truth


def test_noise_labels_buy_no_faces():
    rng = random.Random(3)
    pos, neg = [], []
    for _ in range(300):
        x = [rng.random() for _ in range(4)]
        (pos if rng.random() < 0.5 else neg).append(x)
    concept = fit_halfspace_concept(pos, neg, n_features=4)
    assert concept.n_faces == 0                  # invents no structure from noise
