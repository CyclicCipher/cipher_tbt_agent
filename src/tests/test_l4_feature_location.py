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

from tbt.l4_feature_location import L4_FeatureLocation, invariant_sig, view_signature  # noqa: E402
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


def test_view_signature_is_pose_invariant_and_colour_aware():
    """L4's WHOLE-VIEW content descriptor (P1 slice 1): rotation+translation-invariant (the same shape+colours at any
    pose/position -> the same descriptor -> the same content id) AND colour-aware + shape-aware (a different colouring or
    shape -> a different id). This is the 'what' P2's operator needs -- content invariant to where/how it is."""
    l4 = L4_FeatureLocation(n_entities=8, feat_dim=64, seed=0)
    shape = [(0, 0), (1, 0), (2, 0), (2, 1)]                        # an L-shaped tetromino stub
    view = [(x, y, 7) for (x, y) in shape]                          # colour 7
    sig = view_signature(view)
    for theta in (np.pi / 2, np.pi, 2.0):                          # rotate (incl. a non-axis angle) + translate far away
        moved = [(px, py, 7) for (px, py) in apply_pose(shape, theta, (5.0, -4.0))]
        assert view_signature(moved) == sig                        # pose-invariant descriptor
        assert l4.encode(view_signature(moved)) == l4.encode(sig)  # -> the same content id (recurs across pose)
    other_colour = [(x, y, 3) for (x, y) in shape]                 # same shape, different colour
    assert view_signature(other_colour) != sig
    assert l4.encode(view_signature(other_colour)) != l4.encode(sig)
    diff_shape = [(0, 0, 7), (1, 0, 7), (0, 1, 7), (1, 1, 7)]      # a 2x2 block (same colour, different shape)
    assert view_signature(diff_shape) != sig
    assert L4_FeatureLocation.view_signature is view_signature      # exposed on the layer
