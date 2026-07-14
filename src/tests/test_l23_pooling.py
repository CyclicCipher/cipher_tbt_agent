"""End-to-end test of L2/3 TEMPORAL POOLING (ARCHITECTURE.md §8, STATUS "Next" #2): pool the L4 feature-at-location stream
into a STABLE object IDENTITY (`pooler.ColumnPooler`), driven from `agent.py`.

The defining property (`reference_tbt_layers_4_23`): L4 changes every fixation (a different feature-at-location), but L2/3
stays the SAME — one sparse identity per object, persisting across the traversal, recognised INVARIANT to fixation order,
and minted ONCE (not a new object every frame — the duplication bug the pooler fixes). Recognition is associative recall
over the object library, not a from-scratch recompute.

Two distinct objects (distinct colours, distinct places). Locations are anchored with `locate` (a sensory fix) to isolate
the pooler; the operator ⊕ L4 composition is already covered by `test_feature_at_location`. RULES #3 acceptance: the agent
now recognises WHICH object it is sensing, stable across the sweep — something it could not do before.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent  # noqa: E402

OBJ_A = {(x, y): 1 + (x - 10) + 2 * (y - 10) for x in range(10, 12) for y in range(10, 12)}   # colours 1..4 near (10,10)
OBJ_B = {(x, y): 5 + (x - 20) + 2 * (y - 20) for x in range(20, 22) for y in range(20, 22)}   # colours 5..8 near (20,20)
PASSES = 6


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _traverse(agent: Agent, obj: dict, order, learn: bool):
    """Sweep the object in `order`, sensing each feature at its location and pooling into L2/3. Returns (labels per fixation,
    L4 codes per fixation)."""
    agent.reset_object()
    labels, l4_codes = [], []
    for cell in order:
        agent.locate(cell)
        agent.sense_at(obj[cell], learn=learn)
        labels.append(agent.perceive_object(learn=learn))
        l4_codes.append(frozenset(agent._nav_col().layers["L4"].htm._active))
    return labels, l4_codes


def _learn(agent: Agent):
    for _ in range(PASSES):
        _traverse(agent, OBJ_A, list(OBJ_A), learn=True)
        _traverse(agent, OBJ_B, list(OBJ_B), learn=True)


def test_one_identity_per_object_no_duplication():
    agent = _fresh()
    _learn(agent)
    assert len(agent._nav_col().pooler.objects) == 2, (
        f"minted {len(agent._nav_col().pooler.objects)} objects over {PASSES} passes of 2 objects — L2/3 must pool one "
        "STABLE identity per object (re-pool only on error), not a new object per fixation")


def test_recognition_is_stable_and_order_invariant():
    agent = _fresh()
    _learn(agent)
    a_labels, a_l4 = _traverse(agent, OBJ_A, list(reversed(list(OBJ_A))), learn=False)   # a NEW order, frozen (infer)
    b_labels, _ = _traverse(agent, OBJ_B, list(OBJ_B), learn=False)
    assert set(a_labels) == {0}, f"object A recognised as {a_labels} — must be the SAME identity (0) at every fixation"
    assert set(b_labels) == {1}, f"object B recognised as {b_labels} — must be the SAME identity (1) at every fixation"
    assert a_labels[0] != b_labels[0], "the two objects must have DISTINCT L2/3 identities"
    assert len(set(a_l4)) > 1, "L4 must CHANGE across fixations (feature-at-location) while the L2/3 identity stays constant"
    assert len(agent._nav_col().pooler.objects) == 2, "inference must NOT mint new objects (recognise, don't relearn)"


if __name__ == "__main__":
    ag = _fresh()
    _learn(ag)
    a, a_l4 = _traverse(ag, OBJ_A, list(reversed(list(OBJ_A))), learn=False)
    b, _ = _traverse(ag, OBJ_B, list(OBJ_B), learn=False)
    print(f"library size (should be 2): {len(ag._nav_col().pooler.objects)}")
    print(f"object A sweep (new order) → identities {a}  (distinct L4 codes: {len(set(a_l4))}/4)")
    print(f"object B sweep → identities {b}")
