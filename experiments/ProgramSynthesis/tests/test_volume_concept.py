"""Phase-2 prototype tests — does MDL discover a concept's own low-dim subspace + region,
without being told the dimensions? (docs/phase2/VOLUME_CONCEPTS.md §0, Resolution 1.)"""

import random

from volume.box import fit_box_concept


def _make_data(rng, n, n_features, rel_dims, ranges):
    """Each sample is uniform[0,1] on every dim. POSITIVE iff x[d] in ranges[d] for all relevant d;
    the other dimensions are pure noise the concept must learn to ignore."""
    pos, neg = [], []
    while len(pos) < n or len(neg) < n:
        x = [rng.random() for _ in range(n_features)]
        is_pos = all(ranges[d][0] <= x[d] <= ranges[d][1] for d in rel_dims)
        if is_pos and len(pos) < n:
            pos.append(x)
        elif not is_pos and len(neg) < n:
            neg.append(x)
    return pos, neg


def test_recovers_low_dim_subspace_and_region():
    rng = random.Random(0)
    rel_dims, ranges = [1, 4], {1: (0.30, 0.70), 4: (0.60, 0.90)}
    pos, neg = _make_data(rng, 250, 6, rel_dims, ranges)

    box = fit_box_concept(pos, neg, n_features=6)

    # discovered exactly the relevant subspace — ignored the 4 noise dimensions:
    assert set(box.dims) == set(rel_dims)
    # the region brackets the planted ranges (tight enclosing box of the positives):
    bound = {d: (box.lo[i], box.hi[i]) for i, d in enumerate(box.dims)}
    assert 0.28 <= bound[1][0] and bound[1][1] <= 0.72
    assert 0.58 <= bound[4][0] and bound[4][1] <= 0.92


def test_generalizes_to_held_out():
    rng = random.Random(1)
    rel_dims, ranges = [0, 3], {0: (0.20, 0.50), 3: (0.50, 0.80)}
    train_pos, train_neg = _make_data(rng, 300, 6, rel_dims, ranges)

    box = fit_box_concept(train_pos, train_neg, n_features=6)

    test_pos, test_neg = _make_data(rng, 200, 6, rel_dims, ranges)
    correct = (sum(box.contains(p) for p in test_pos)
               + sum(not box.contains(n) for n in test_neg))
    acc = correct / (len(test_pos) + len(test_neg))
    assert set(box.dims) == set(rel_dims)
    assert acc > 0.9


def test_invents_no_structure_from_noise_labels():
    """If the label is independent of the features, no subspace pays for itself — the concept
    declines to invent structure (the MDL knife-edge: minimize priors, don't hallucinate a region)."""
    rng = random.Random(2)
    pos, neg = [], []
    for _ in range(300):
        x = [rng.random() for _ in range(5)]
        (pos if rng.random() < 0.5 else neg).append(x)

    box = fit_box_concept(pos, neg, n_features=5)

    assert box.n_dims <= 1
