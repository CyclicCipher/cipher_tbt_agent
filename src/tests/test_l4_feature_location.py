"""Layer 4 — feature-at-location. The label-free content codebook (absorbing the retina's patch vocabulary, grown
online with no hard wall), the rotation-INVARIANT feature descriptor (the ventral 'what', complementary to L5's
equivariant displacements), and predict_feature -- the PREDICT half of predict-then-compare seated where the
feature lives (L4 inherits L6's location, forms none of its own)."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.l4_feature_location import L4_FeatureLocation, invariant_sig  # noqa: E402
from tbt.l5_displacement import apply_pose, local_disps               # noqa: E402


def test_codebook_is_label_free_and_recurs():
    """Same descriptor -> same id; a novel one -> the next id. The content vocabulary is discovered by watching,
    never injected (the bitter lesson) -- the recurrence the column needs."""
    l4 = L4_FeatureLocation(n_entities=8, feat_dim=64, seed=0)
    assert l4.encode(("patchA",)) == 0
    assert l4.encode(("patchB",)) == 1
    assert l4.encode(("patchA",)) == 0               # recurs, does not grow
    assert len(l4.codebook) == 2


def test_codebook_grows_past_capacity_no_wall():
    """The sparse codebook grows on demand -- the content vocabulary has no hard wall (cortical capacity)."""
    l4 = L4_FeatureLocation(n_entities=2, feat_dim=64, seed=0)
    assert [l4.encode((i,)) for i in range(5)] == [0, 1, 2, 3, 4]   # 5 distinct > capacity 2
    assert l4.E.shape[0] >= 5                          # E grew with the vocabulary
    _ = l4.bind(4, torch.nn.functional.normalize(torch.randn(16), dim=0))  # a grown code is usable (no index error)


def test_feature_at_location_binds_and_reads_back():
    """bind a feature at a place, readout recovers it (argmax) -- feature-at-location, crosstalk cleaned by the
    sparse code."""
    l4 = L4_FeatureLocation(n_entities=8, feat_dim=128, seed=1)
    d_mem = 16
    places = torch.nn.functional.normalize(torch.randn(3, d_mem), dim=1)
    S = torch.zeros(l4.feat_dim, d_mem)
    for fid, p in zip((2, 5, 7), places):
        S = S + l4.bind(fid, p)
    for fid, p in zip((2, 5, 7), places):
        assert int(l4.readout(S, p).argmax()) == fid


def test_predict_feature_reads_the_object():
    """predict_feature(S, place) = the feature expected at place under the object memory S -- the predict half of
    predict-then-compare, seated in L4."""
    l4 = L4_FeatureLocation(n_entities=8, feat_dim=128, seed=2)
    d_mem = 16
    p0 = torch.nn.functional.normalize(torch.randn(d_mem), dim=0)
    p1 = torch.nn.functional.normalize(torch.randn(d_mem), dim=0)
    S = l4.bind(3, p0) + l4.bind(6, p1)
    assert l4.predict_feature(S, p0) == 3
    assert l4.predict_feature(S, p1) == 6


def test_invariant_sig_is_rotation_invariant():
    """L4's feature descriptor is the rotation-INVARIANT 'what' (complementary to L5's equivariant local_disps):
    the same shape at any orientation yields the same feature, so content recurs across pose."""
    cloud = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0)]
    base = [np.asarray(c, float) for c in cloud]
    sig0 = invariant_sig(local_disps(base, 0, radius=3.0))
    for theta in (0.5, np.pi / 2, 2.0):
        rotated = [np.asarray(p, float) for p in apply_pose(cloud, theta, (4.0, -3.0))]
        assert invariant_sig(local_disps(rotated, 0, radius=3.0)) == sig0


def test_invariant_sig_exposed_on_the_layer():
    """L4 the LAYER exposes the feature descriptor (the column reads the 'what' through it)."""
    assert L4_FeatureLocation.invariant_sig is invariant_sig
