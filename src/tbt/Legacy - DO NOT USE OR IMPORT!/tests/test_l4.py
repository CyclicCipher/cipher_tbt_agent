"""Layer 4 — the sensorimotor INPUT layer (MICROCIRCUIT.md Stage 2), in isolation. L4 is the object's feature-at-location
MAP over the shared HTM mechanism (`tbt.l4.L4Layer` on `tbt.htm.HTMLayer`): minicolumns = feature-SDR bits, cells = the
L6 LOCATION context (basal segments onto the grid SDR). It learns the map, RECALLS the feature at a location without
sensing it (imagination), and BURSTS when a feature is sensed where a different one was learned (surprise). Not wired into
the live loop yet (Stage 4)."""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.encoders import GridEncoder  # noqa: E402
from tbt.l4 import L4Layer  # noqa: E402


def test_l4_feature_at_location_recall_imagination_and_burst():
    """L4 learns an object's feature-at-location map, then (a) RECALLS the feature at each mapped location as a pure query
    — the IMAGINATION: predict what is there before sensing it — and a far/unmapped location predicts (near) nothing;
    (b) BURSTS when a feature is sensed where a DIFFERENT feature was learned (the surprise / object-mismatch signal), and
    does NOT burst for the correct feature. Basal = the L6 grid SDR of the location; the SAME HTM mechanism as the temporal
    memory, driven by location instead of the previous cells (rule 1)."""
    g = GridEncoder(scales=(5, 7, 11), dims=2, mw=3, bounds=[(0, 15), (0, 15)])

    def loc(x, y):
        return g.encode((x, y)).active

    feats = {"a": set(range(0, 8)), "b": set(range(8, 16)), "c": set(range(16, 24))}

    def decode(cols):
        return max(feats, key=lambda f: len(feats[f] & cols)) if cols else None

    l4 = L4Layer(cells_per_column=4, activation_threshold=10, min_threshold=4, init_perm=0.5)
    obj = {(1, 1): "a", (5, 8): "b", (11, 3): "c"}                  # a 3-feature object at well-separated locations
    for _ in range(3):
        for (x, y), f in obj.items():
            l4.observe(feats[f], loc(x, y))

    # RECALL / IMAGINATION: predict the feature at each mapped location WITHOUT sensing it
    for (x, y), f in obj.items():
        assert decode(l4.predict_feature(loc(x, y))) == f, ((x, y), f)
    # a far / unmapped location depolarises (near) nothing — the map does not cover it
    assert len(l4.predict_feature(loc(14, 14))) <= 2, l4.predict_feature(loc(14, 14))

    # BURST = surprise: a feature sensed where a DIFFERENT feature was learned
    l4.observe(feats["a"], loc(11, 3))                             # 'a' where 'c' lives -> not predicted -> burst
    assert l4.burst()
    l4.observe(feats["c"], loc(11, 3))                             # the correct feature -> predicted -> no burst
    assert not l4.burst()
