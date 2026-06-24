"""Phase-2 prototype — multi-modal concepts via UNION (docs/phase2/VOLUME_CONCEPTS.md §9 P0):
MDL discovers the number of modes — a convex concept stays one region, a two-blob concept splits
into two. Closes the convex limit the shape benchmark exposed. `seed=0` everywhere → deterministic."""

import numpy as np

from volume.benchmark import SHAPES, accuracy, sample
from volume.halfspace import HalfspaceConcept, fit_halfspace_concept
from volume.union import UnionConcept, fit_union_concept


def _train_test(shape, n_train=250, n_test=200):
    rng = np.random.default_rng(0)
    tr = sample(SHAPES[shape], n_train, rng)
    te = sample(SHAPES[shape], n_test, rng)
    return tr, te


def test_convex_concept_stays_one_mode():
    (tr_p, tr_n), (te_p, te_n) = _train_test("axis_box")
    u = fit_union_concept(tr_p, tr_n, n_features=2)
    assert u.n_modes == 1                         # parsimony: don't split a convex concept
    assert accuracy(u, te_p, te_n) >= 0.95


def test_disk_stays_one_mode():
    (tr_p, tr_n), (te_p, te_n) = _train_test("disk")
    u = fit_union_concept(tr_p, tr_n, n_features=2)
    assert u.n_modes == 1
    assert accuracy(u, te_p, te_n) >= 0.95


def test_two_blobs_discovers_two_modes_and_beats_single_convex():
    (tr_p, tr_n), (te_p, te_n) = _train_test("two_blobs")
    u = fit_union_concept(tr_p, tr_n, n_features=2)
    single = fit_halfspace_concept(tr_p, tr_n, n_features=2)   # one convex region, same data
    assert u.n_modes == 2                          # multi-modality discovered
    assert accuracy(u, te_p, te_n) >= 0.93
    assert accuracy(u, te_p, te_n) > accuracy(single, te_p, te_n)  # union beats the convex limit


def test_union_membership_is_or():
    def _box(b):  # axis-aligned box via four half-spaces: x<=b0, x>=b1, y<=b2, y>=b3
        W = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        return HalfspaceConcept(W, np.array(b), (True, True, True, True))
    a = _box([1.0, 0.0, 1.0, 0.0])     # [0,1] x [0,1]
    b = _box([3.0, -2.0, 3.0, -2.0])   # [2,3] x [2,3]
    u = UnionConcept((a, b))
    assert u.contains([0.5, 0.5])      # in A
    assert u.contains([2.5, 2.5])      # in B
    assert not u.contains([1.5, 1.5])  # in neither
    assert u.n_modes == 2
