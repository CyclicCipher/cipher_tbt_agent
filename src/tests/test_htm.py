"""Standalone tests for `tbt.htm.HTMLayer` — the ONE cortical-layer mechanism (was `HTMLayer` + `SequenceMemory`, now merged).

Two parts: (1) the dendrite substrate directly (connected-segment depolarisation, basal-determined burst winner, the one
Hebbian rule, apical as a pure tiebreak); (2) the layer STEP as a next-element predictor over SDRs (predicts a learned
sequence, HIGH-ORDER context disambiguation, burst on novelty, the pluggable feature-at-location context, and — the
salvage's point — GENERALISATION over a similar/overlapping context, graded by overlap). No column, no layer wiring.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.htm import HTMLayer, PopulationReadout  # noqa: E402


# ---- (1) the dendrite substrate -------------------------------------------------------------------------------
def test_connected_segment_depolarises_a_cell():
    h = HTMLayer(cells_per_column=4, activation_threshold=2)
    h.seg[(5, 0)] = [{"a": 0.6, "b": 0.6, "z": 0.6}]            # 2 connected synapses onto {a,b} present in the context
    pred, segs = h._predicted({"a", "b", "c"})
    assert (5, 0) in pred and (5, 0) in segs
    assert (5, 0) not in h._predicted({"x", "y"})[0]           # context absent → not depolarised


def test_burst_winner_is_basal_determined():
    h = HTMLayer(cells_per_column=4, min_threshold=1)
    h.seg[(5, 0)] = [{"a": 0.4, "b": 0.4}]                     # a matching (potential) segment on cell 0
    w, s = h._winner(5, {"a", "b"})
    assert w == (5, 0) and s is not None                      # best potential match wins
    w2, s2 = h._winner(9, {"a"})                              # column with no segments → deterministic least-used cell
    assert w2 == (9, 0) and s2 is None


def test_hebbian_reinforces_grows_prunes():
    h = HTMLayer(init_perm=0.4, perm_inc=0.2, perm_dec=0.6, max_syn=5)
    seg = {"a": 0.5, "b": 0.5}
    h._learn_segment(seg, active={"a"}, grow={"c"})
    assert abs(seg["a"] - 0.7) < 1e-9                          # onto-active → reinforced
    assert "b" not in seg                                      # not-active → weakened past 0 → pruned
    assert abs(seg["c"] - 0.4) < 1e-9                          # grown at init_perm


def test_apical_is_a_pure_tiebreak():
    h = HTMLayer(activation_threshold=2)
    h.apical_seg[(3, 0)] = [{"f1": 0.6, "f2": 0.6}]
    apic = h._apical_predicted({"f1", "f2"})
    assert (3, 0) in apic
    basal = [(3, 0), (3, 1)]
    assert h._apical_narrow(basal, apic) == [(3, 0)]           # narrow to the apically-supported cell
    assert h._apical_narrow(basal, set()) == basal            # no support → keep ALL basal


# ---- (2) the layer step as an SDR sequence predictor ----------------------------------------------------------
A = set(range(0, 10))
B = set(range(10, 20))
C = set(range(20, 30))
D = set(range(30, 40))
E = set(range(40, 50))
B_SIM = set(range(10, 18)) | {95, 96}          # overlaps B by 8 / 10
B_FAR = {10, 11} | set(range(80, 88))          # overlaps B by 2 / 10


def _train(h, seq, n):
    for _ in range(n):
        h.reset()
        for sdr in seq:
            h.observe(sdr)


def test_predicts_a_learned_sequence():
    h = HTMLayer(order=2)
    _train(h, [A, B, C], 12)
    h.reset()
    h.observe(A)
    h.observe(B)
    assert len(h.predict() & C) >= 8, h.predict()
    assert h.predict_best([C, D, E]) == C


def test_high_order_context_disambiguates_a_shared_element():
    h = HTMLayer(order=2)
    for _ in range(20):                                        # A B C and D B E — B is shared, only the context tells C from E
        h.reset(); [h.observe(x) for x in (A, B, C)]
        h.reset(); [h.observe(x) for x in (D, B, E)]
    h.reset(); h.observe(A); h.observe(B)
    assert len(h.predict() & C) >= 6 and len(h.predict() & E) <= 2, h.predict()
    h.reset(); h.observe(D); h.observe(B)
    assert len(h.predict() & E) >= 6 and len(h.predict() & C) <= 2, h.predict()


def test_novel_transition_bursts():
    h = HTMLayer(order=2)
    _train(h, [A, B, C], 12)
    h.reset()
    h.observe(A)
    h.observe(B)
    h.observe(C)
    h.observe(E)                                             # C → E was never seen
    assert h.bursting()


def test_generalises_over_a_similar_context_graded_by_overlap():
    """The salvage's point: a SIMILAR context still predicts (overlap → shared basal cells → same next), OVERLAP-GRADED."""
    h = HTMLayer(order=2)
    _train(h, [A, B, C], 12)
    h.reset(); h.observe(A); h.observe(B_SIM)                 # overlaps B by 8
    assert len(h.predict() & C) >= 6, h.predict()
    h.reset(); h.observe(A); h.observe(B_FAR)               # overlaps B by 2 (< activation_threshold)
    assert len(h.predict() & C) == 0, h.predict()


def test_pluggable_context_binds_feature_at_location():
    """The pluggable basal context (the L4 / sensorimotor use): drive the dendrites from an EXTERNAL context (a location) so
    the SAME mechanism binds a feature AT a location and reads it back with `predict_at` — before a column exists."""
    h = HTMLayer(order=1)
    loc, feat = set(range(50, 60)), set(range(60, 70))
    for _ in range(6):
        h.reset()
        h.observe(feat, context=loc)
    assert len(h.predict_at(loc) & feat) >= 6
    assert len(h.predict_at(set(range(70, 80))) & feat) == 0


# ── the POPULATION READ-OUT ─────────────────────────────────────────────────────────────────────────────────────────────
# The canonical pipeline's third stage, VECTOR-valued: encoder → SP → TM → read-out. We had only ever built the discrete
# variant (a softmax over cells → one bucket); this is the continuous one, and it is the mechanism that lets a cell assembly
# be an IDENTITY code and a METRIC code at once — which cells fire, and what they decode to.

def test_population_readout_decodes_a_vector_one_shot():
    """A population vector: the summed preferred vectors of the active cells, exact after one observation at lr=1."""
    r = PopulationReadout(2)
    r.learn({1, 2, 3, 4}, (1.0, 0.0))
    assert all(abs(a - b) < 1e-9 for a, b in zip(r.decode({1, 2, 3, 4}), (1.0, 0.0)))


def test_population_readout_generalises_by_overlap():
    """SDR overlap makes the decode metric: a half-overlapping assembly inherits half the learned vector, and a disjoint one
    inherits nothing (an unlearned cell contributes zero — no evidence, no effect). Smooth generalisation for free."""
    r = PopulationReadout(2)
    r.learn({1, 2, 3, 4}, (1.0, 0.0))
    assert abs(r.decode({1, 2, 9, 10})[0] - 0.5) < 1e-9, "half the cells ⇒ half the vector"
    assert all(abs(v) < 1e-9 for v in r.decode({7, 8})), "no learned cell ⇒ no decode"


def test_population_readout_separates_contexts():
    """Different assemblies decode to different vectors — so the SAME quantity read under two contexts is two values, which
    is what makes the conjunction an HTM layer forms usable as a transform."""
    r = PopulationReadout(2)
    r.learn({1, 2, 3, 4}, (1.0, 0.0))
    r.learn({5, 6, 7, 8}, (0.0, 1.0))
    assert all(abs(a - b) < 1e-9 for a, b in zip(r.decode({1, 2, 3, 4}), (1.0, 0.0)))
    assert all(abs(a - b) < 1e-9 for a, b in zip(r.decode({5, 6, 7, 8}), (0.0, 1.0)))


def test_population_readout_shares_error_across_cells():
    """The delta rule normalised by the active count — which is cue competition at CELL granularity, and the reason a
    spurious cue is blocked once the predictive cells explain the quantity."""
    r = PopulationReadout(2, lr=0.5)
    for _ in range(40):
        r.learn({1, 2, 3}, (0.0, 0.0))       # cells 1-3 together ⇒ nothing
        r.learn({2, 3}, (0.0, 1.0))          # 2-3 alone ⇒ down
    assert all(abs(v) < 0.05 for v in r.decode({1, 2, 3}))
    assert abs(r.decode({2, 3})[1] - 1.0) < 0.05
