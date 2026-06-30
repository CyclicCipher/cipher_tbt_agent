"""The Goal State Generator (GSG), disambiguation-first (Monty's graph-mismatch hypothesis-test). When the top-2
(object, pose) hypotheses still compete, the GSG proposes the WORLD point where they most DISAGREE -- present in one
model's predicted cloud, far from the other -- the maximally-discriminating place to sample. Domain-general: it reads
only the column's own hypotheses. The column wraps it into a message-shaped GoalState (heterarchy-ready)."""

from __future__ import annotations

import os
import sys

import numpy as np

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.column import CorticalColumn, GoalState     # noqa: E402
from tbt.l23_object import L23_Object, _Hyp          # noqa: E402
from tbt.l5_displacement import local_disps          # noqa: E402

A = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (0, 2)]   # a 1x4 bar + a LEFT appendage (up the left end)
B = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)]   # the SAME bar + a RIGHT appendage -- ambiguous from the bar alone


def test_graph_mismatch_picks_the_discriminating_point():
    """The GSG core: with two object hypotheses at the same (sensed) pose -- sharing the bar, differing in the
    appendage -- the proposed goal is the appendage TIP where they most disagree (on one cloud, far from the other)."""
    l23 = L23_Object()
    oa, ob = l23.learn(A, name="A"), l23.learn(B, name="B")
    l23.hyps = [_Hyp(oa, 0.0, np.array([0.0, 0.0]), oa.locs[0], 1.0),     # both aligned to the shared bar, tied
                _Hyp(ob, 0.0, np.array([0.0, 0.0]), ob.locs[0], 1.0)]
    goal = l23.disambiguation_goal()
    assert goal is not None
    g = np.asarray(goal, float)
    da = min(np.linalg.norm(g - np.asarray(p, float)) for p in oa.cells_at(0.0, (0.0, 0.0)))
    db = min(np.linalg.norm(g - np.asarray(p, float)) for p in ob.cells_at(0.0, (0.0, 0.0)))
    assert min(da, db) < 0.1 and max(da, db) > 1.5, f"goal {goal} not discriminating: dA={da:.2f} dB={db:.2f}"
    assert goal in {(0.0, 2.0), (3.0, 2.0)}, goal      # the appendage TIP -- the single most-distant point


def test_margin_gate_suppresses_the_goal_when_one_hypothesis_leads():
    """A clear leader (> margin evidence ahead) -> nothing to resolve -> no goal (Monty fires the test on genuine
    competition, not on a settled hypothesis)."""
    l23 = L23_Object()
    oa, ob = l23.learn(A, name="A"), l23.learn(B, name="B")
    l23.hyps = [_Hyp(oa, 0.0, np.array([0.0, 0.0]), oa.locs[0], 5.0),     # A leads by 4 (> margin 1.5)
                _Hyp(ob, 0.0, np.array([0.0, 0.0]), ob.locs[0], 1.0)]
    assert l23.disambiguation_goal() is None
    l23.start()
    assert l23.disambiguation_goal() is None                              # no hypotheses at all


def test_gsg_fires_in_a_real_recognition_session():
    """End to end: presenting A and sensing the AMBIGUOUS shared bar leaves >= 2 competing hypotheses, and the GSG
    proposes a discriminating sample point (the recognition dynamics produce the ambiguity; the GSG resolves it)."""
    l23 = L23_Object()
    l23.learn(A, name="A"); l23.learn(B, name="B")
    locs = [np.asarray(c, float) for c in A]
    l23.start()
    i = A.index((1, 0))                                                   # one sense of the shared bar middle
    l23.sense(locs[i], local_disps(locs, i, l23.radius))
    assert len(l23.hyps) >= 2
    goal = l23.disambiguation_goal()
    assert goal is not None
    g = np.asarray(goal, float)
    h1, h2 = sorted(l23.hyps, key=lambda h: h.ev, reverse=True)[:2]
    da = min(np.linalg.norm(g - np.asarray(p, float)) for p in h1.obj.cells_at(h1.theta, h1.t))
    db = min(np.linalg.norm(g - np.asarray(p, float)) for p in h2.obj.cells_at(h2.theta, h2.t))
    assert min(da, db) < 0.1 and max(da, db) >= 0.9, f"goal {goal} not discriminating: d1={da:.2f} d2={db:.2f}"


def test_column_gsg_proposes_a_message_shaped_goal_state():
    """The column's GSG wraps the disambiguation target into a message-shaped GoalState (self-generated now,
    receivable from a connected column later -- the heterarchy is just where the message comes from)."""
    col = CorticalColumn(n_entities=8, seed=0)
    col.learn_object(A, name="A"); col.learn_object(B, name="B")
    locs = [np.asarray(c, float) for c in A]
    col.L23.start()
    i = A.index((1, 0))
    col.L23.sense(locs[i], local_disps(locs, i, col.L23.radius))
    goal = col.propose_goal()
    assert isinstance(goal, GoalState) and goal.kind == "disambiguate" and goal.target is not None
    col.L23.start()
    assert col.propose_goal() is None                                    # nothing to resolve -> no goal


def test_sense_absent_falsifies_only_the_predicting_hypothesis():
    """The ABSENT half of a hypothesis-test: sampling a location ONE hypothesis predicts (but the other does not)
    and finding it EMPTY penalises only the predictor -- so a graph-mismatch sample discriminates even by ABSENCE."""
    l23 = L23_Object()
    oa, ob = l23.learn(A, name="A"), l23.learn(B, name="B")
    l23.hyps = [_Hyp(oa, 0.0, np.array([0.0, 0.0]), oa.locs[0], 1.0),
                _Hyp(ob, 0.0, np.array([0.0, 0.0]), ob.locs[0], 1.0)]
    l23.sense_absent((3.0, 2.0))                          # (3,2) is in B's cloud (right tip) but NOT A's
    evA = next((h.ev for h in l23.hyps if h.obj.name == "A"), None)
    evB = next((h.ev for h in l23.hyps if h.obj.name == "B"), -99.0)
    assert evA == 1.0 and evB <= 0.0, f"A {evA} (should be 1.0, unaffected), B {evB} (should be falsified)"


def test_examine_actively_recognises_the_object():
    """examine -- the GSG DIRECTING the motor -- actively recognises the true object: begin on a shared (ambiguous)
    cell, then the GSG covertly picks discriminating sample points and the motor samples them until one hypothesis
    wins. The mechanism: think (graph-mismatch, no action), then act (sample the target)."""
    col = CorticalColumn(n_entities=8, seed=0)
    col.learn_object(A, name="A"); col.learn_object(B, name="B")
    truth = [np.asarray(c, float) for c in A]

    def sense_at(target):
        t = np.asarray(target, float)
        d = [np.linalg.norm(t - p) for p in truth]
        i = int(np.argmin(d))
        return (truth[i], local_disps(truth, i, col.L23.radius)) if d[i] < 0.6 else None  # cell here, or empty

    first = (truth[1], local_disps(truth, 1, col.L23.radius))   # start on a shared bar cell (ambiguous)
    best, n = col.examine(sense_at, first)
    assert best is not None and best[0] == "A", f"misrecognised: {best}"
    assert n <= 12


def _ambiguous_column():
    col = CorticalColumn(n_entities=8, seed=0)
    col.learn_object(A, name="A"); col.learn_object(B, name="B")
    locs = [np.asarray(c, float) for c in A]
    col.L23.start()
    i = A.index((1, 0))
    col.L23.sense(locs[i], local_disps(locs, i, col.L23.radius))   # one ambiguous glance -> disambiguate available
    return col


def test_propose_goals_lists_act_always_and_disambiguate_when_ambiguous():
    col = _ambiguous_column()
    assert [g.kind for g, _ in col.propose_goals(act_value=0.2, g_value=0.5)] == ["act", "disambiguate"]
    col.L23.start()                                              # no competition -> only the act goal
    assert [g.kind for g, _ in col.propose_goals(act_value=0.2, g_value=0.5)] == ["act"]


def test_basal_ganglia_arbitrates_act_vs_disambiguate_and_learns():
    """GD3: the basal ganglia arbitrates the column's goal candidates by EFE value (Go the higher), and dopamine-RPE
    makes a consistently-valuable goal type win even when its critic value later dips -- the urge that resolves the
    affordance competition (Cisek). On a single column the gate is a thin arbiter; it becomes consensus across
    columns (and over received goal-messages) in the heterarchy."""
    from tbt.basal_ganglia import BasalGanglia
    kinds = ["act", "disambiguate"]
    assert kinds[BasalGanglia(n_columns=1).gate(kinds, [0.2, 0.5])] == "disambiguate"   # higher EFE -> Go
    assert kinds[BasalGanglia(n_columns=1).gate(kinds, [0.8, 0.3])] == "act"
    bg = BasalGanglia(n_columns=1)                              # dopamine-RPE learning
    for _ in range(5):
        bg.gate(kinds, [0.3, 0.6])                              # disambiguate consistently more valuable
    assert kinds[bg.gate(kinds, [0.5, 0.4])] == "disambiguate"  # learned affinity carries it past a value dip
