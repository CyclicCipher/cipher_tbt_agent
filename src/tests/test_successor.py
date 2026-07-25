"""test_successor.py — the LEARNED frame (`tbt.successor.SuccessorFrame`).

`GridEncoder` is a frame you are GIVEN, and it works because physical space is metric. This is the frame you LEARN, for
spaces with no coordinates to hand it. Place cells ARE the successor representation and grid cells are its eigenvectors
(`reference_grid_sr_eigenbasis`), so the same machinery run on a different transition structure yields that structure's
code — which is what lets a region above the sensorimotor one have an L6a at all.

Three things it deliberately does NOT do, each from a recorded cost or measured failure: no eigendecomposition (O(n^3),
prohibitive online, and the drive built on it was dropped as redundant); no matrix operator over SR rows (built once,
path-integrated [2,2,2,4,4,4] where [1,2,3,4,5,6] was needed); no orthogonalisation (SR codes are MEANT to be correlated —
the overlap IS the topology).
"""

from __future__ import annotations

import itertools
import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.successor import SuccessorFrame     # noqa: E402


def _line(n=8, reps=60):
    sr = SuccessorFrame()
    for _ in range(reps):
        for i in range(n - 1):
            sr.observe(i, "r", i + 1)
            sr.observe(i + 1, "l", i)
    return sr


def _keys():
    """A genuinely NON-METRIC task space: states are SETS of collected keys, which are not points in R^n."""
    sr = SuccessorFrame()
    e, a, b, ab = frozenset(), frozenset("a"), frozenset("b"), frozenset("ab")
    for _ in range(80):
        for s, act, n in [(e, "a", a), (e, "b", b), (a, "b", ab), (b, "a", ab)]:
            sr.observe(s, act, n)
    return sr, e, a, b, ab


def test_topology_falls_out_of_transitions():
    """On a line, similarity decreases monotonically with graph distance — nobody supplied a coordinate, only transitions."""
    sr = _line()
    sims = [sr.similarity(0, s) for s in range(8)]
    assert all(sims[i] >= sims[i + 1] - 1e-9 for i in range(7)), f"must fall off with distance, got {sims}"
    assert sims[0] > sims[-1], "and the far end must be clearly less similar than the near one"


def test_it_works_where_there_is_no_metric_to_assume():
    """THE POINT. States are sets of keys — no coordinates exist to hand a GridEncoder — and the frame still places
    one-key-collected NEARER to none-collected than both-collected is. Traversability, not distance."""
    sr, e, a, _b, ab = _keys()
    assert sr.similarity(e, a) > sr.similarity(e, ab), "one action away must be nearer than two"


def test_value_is_a_dot_product_and_a_moved_reward_needs_no_relearning():
    """`V = M.R` — the SR's whole payoff as a planner. Value rises toward the goal across a non-metric space, and moving
    the reward re-values everything WITHOUT re-learning the space, which is what a cached policy cannot do."""
    sr, e, a, _b, ab = _keys()
    at_both = {ab: 1.0}
    assert sr.value(e, at_both) < sr.value(a, at_both) < sr.value(ab, at_both), "value must rise toward the goal"

    at_one = {a: 1.0}                                     # move the reward; no further observation
    assert sr.value(e, at_one) > 0.0, "the space re-values instantly under a new reward"
    assert sr.value(ab, at_one) == 0.0, "and correctly reports zero where the goal is unreachable"


def test_a_state_only_ever_arrived_at_still_carries_value():
    """The bug this caught: a state that is never acted FROM — a terminal, or a goal — kept an empty row, so reward placed
    there propagated nowhere and `V = M.R` read zero everywhere. Being in a state means occupying it."""
    sr = SuccessorFrame()
    for _ in range(40):
        sr.observe("start", "go", "terminal")
    assert sr.value("start", {"terminal": 1.0}) > 0.0, "reward at a never-departed state must still be seen from before it"


def test_it_stays_sparse_and_cheap_online():
    """No eigendecomposition and no dense matrix: rows are sparse dicts and every update is incremental, which is the whole
    reason this is affordable in the live loop."""
    sr = SuccessorFrame()
    states = [frozenset(c) for r in range(4) for c in itertools.combinations("abc", r)]
    for _ in range(40):
        for i, s in enumerate(states[:-1]):
            sr.observe(s, "n", states[i + 1])
    assert max(len(r) for r in sr.M.values()) <= len(states), "rows never exceed the reachable state count"
    assert not hasattr(sr, "eig"), "there is no eigendecomposition here, by design"
