"""End-to-end test of the L4↔L6a loop (ARCHITECTURE.md §8, STATUS "Next" #2): predict the FEATURE at the path-integrated
LOCATION — the composition of the two primitives (TRANSFORM: the operator supplies WHERE; ASSOCIATE: L4 supplies WHAT).

The claim being tested is ORDER-INVARIANCE, the defining TBT property (`htm.py`): because L4 predicts the feature from the
LOCATION (its L6a basal context), not from the previous feature, an object learned in one traversal order is predicted
correctly in a DIFFERENT order — which a pure temporal (previous-feature) predictor cannot do. This exercises both
primitives together: the operator dead-reckons to each location in a new order, and L4 reads the feature bound there.

ARCHITECTURALLY: the feature-at-location binding is INTRACOLUMNAR (L6a→L4 basal context = the column's own
`observe(feature, location)` contract), NOT the thalamus's cross-column content⊗location bind (that is for VOTING across
columns — a later, multi-column slice). A single sensory column needs no thalamus here.

RULES #3 acceptance: the agent predicts what it will sense at a place it path-integrates to, in an order it never learned.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent  # noqa: E402

ACTIONS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
# The OBJECT = a 3x3 patch of DISTINCT features (colours 1..9) at distinct locations.
OBJECT = {(x, y): 1 + (x - 10) + 3 * (y - 10) for x in range(10, 13) for y in range(10, 13)}


def _move(p, a):
    dx, dy = ACTIONS[a]
    return (p[0] + dx, p[1] + dy)


def _teach_operator(agent: Agent) -> None:
    """Pre-learn the 4 unit actions (the TRANSFORM primitive) so path integration is exact — tested separately."""
    for a in ACTIONS:
        for x in range(9, 14):
            for y in range(9, 14):
                agent.learn_move(a, (x, y), _move((x, y), a))


def _learn_object(agent: Agent, passes: int = 5) -> None:
    """Bind each feature to its location (L4↔L6a). Learned by ANCHORING at the true location (`locate`) — the binding is
    location-based, so the LEARNING order is irrelevant (that is the whole point). A few passes let the HTM synapses connect."""
    for _ in range(passes):
        for cell, colour in OBJECT.items():
            agent.locate(cell)
            agent.sense_at(colour)


def _direct_binding_accuracy(agent: Agent):
    ok = 0
    for cell, colour in OBJECT.items():
        agent.locate(cell)
        ok += (agent.predict_feature() == colour)
    return ok, len(OBJECT)


def _order_invariance_accuracy(agent: Agent):
    """Traverse the object in a DIFFERENT order from learning (reverse boustrophedon), dead-reckoning by the operator, and
    predict each cell's feature BEFORE sensing it — purely from the path-integrated location."""
    start = (12, 12)
    path = ["W", "W", "S", "E", "E", "S", "W", "W"]          # visits all 9 cells, an order never used in learning
    agent.locate(start)
    cell = start
    ok = (agent.predict_feature() == OBJECT[start])           # predict at the start location
    for a in path:
        agent.path_integrate(a)                               # operator supplies the next LOCATION (no sensory input)
        cell = _move(cell, a)
        ok += (agent.predict_feature() == OBJECT[cell])       # L4 supplies the FEATURE at that location
    return ok, len(path) + 1


def test_feature_at_location_binds():
    agent = _fresh()
    _teach_operator(agent)
    _learn_object(agent)
    ok, tot = _direct_binding_accuracy(agent)
    assert ok == tot, f"feature-at-location binding wrong at {tot - ok}/{tot} cells — L4 must predict the feature from L6a"


def test_prediction_is_order_invariant():
    agent = _fresh()
    _teach_operator(agent)
    _learn_object(agent)
    ok, tot = _order_invariance_accuracy(agent)
    assert ok == tot, (
        f"order-invariant prediction {ok}/{tot} — L4 predicts feature from LOCATION (operator ⊕ L4), so a new traversal "
        "order must still be predicted correctly")


def test_unbound_location_predicts_nothing():
    agent = _fresh()
    _teach_operator(agent)
    _learn_object(agent)
    agent.locate((40, 40))                                    # a location with no object cell
    assert agent.predict_feature() is None, "an unbound location must predict no feature"


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


if __name__ == "__main__":
    ag = _fresh()
    _teach_operator(ag)
    _learn_object(ag)
    d_ok, d_tot = _direct_binding_accuracy(ag)
    o_ok, o_tot = _order_invariance_accuracy(ag)
    print(f"direct feature-at-location binding: {d_ok}/{d_tot}")
    print(f"order-invariant prediction (new traversal order): {o_ok}/{o_tot}")
