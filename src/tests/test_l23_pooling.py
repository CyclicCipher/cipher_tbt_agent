"""End-to-end test of L2/3 TEMPORAL POOLING (ARCHITECTURE.md §8, STATUS "Next" #2): pool the L4 feature-at-location stream
into a STABLE object IDENTITY (`pooler.ColumnPooler`), driven from `agent.py`.

The defining property (`reference_tbt_layers_4_23`): L4 changes every fixation (a different feature-at-location), but L2/3
stays the SAME — one sparse identity per object, persisting across the traversal, recognised INVARIANT to fixation order,
and minted ONCE (not a new object every frame — the duplication bug the pooler fixes). Recognition is associative recall
over the object library, not a from-scratch recompute.

Two distinct objects (distinct colours, distinct places). Locations are anchored with `locate` (a sensory fix) to isolate
the pooler; the operator ⊕ L4 composition is already covered by `test_feature_at_location`. RULES #3 acceptance: the agent
now recognises WHICH object it is sensing, stable across the sweep — something it could not do before.

LEARNING is an EPISODE (`start_object` → `sense_sweep` × n → `commit`), not a per-fixation act: L2/3 commits once the whole
sweep is in, because at the first fixation two objects that share a feature are indistinguishable and committing there
merged them (see `test_shared_features_do_not_merge`). INFERENCE stays per-fixation (`perceive`) — that is L2/3's output.
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
# Two objects that SHARE a feature-at-location (colour 9 at (30,30)) and differ elsewhere — the case that used to merge.
SHARE_P = {(30, 30): 9, (31, 30): 1, (30, 31): 2}
SHARE_Q = {(30, 30): 9, (31, 30): 1, (30, 29): 3}     # same FIRST TWO fixations as P, then diverges
PASSES = 6


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _study(agent: Agent, obj: dict, order=None) -> int:
    """LEARN one object: an episode — onset, buffer each fixation, then COMMIT (recognise-or-mint + bind)."""
    agent.start_object()
    for cell in (order or list(obj)):
        agent.locate(cell)
        agent.sense_sweep(obj[cell])
    return agent.commit()


def _traverse(agent: Agent, obj: dict, order):
    """INFER: sweep the object, reading L2/3's identity at each fixation. Returns (labels, L4 codes) per fixation."""
    agent.start_object()
    labels, l4_codes = [], []
    for cell in order:
        agent.locate(cell)
        labels.append(agent.perceive(obj[cell]))
        l4_codes.append(frozenset(agent._nav_col().layers["L4"].htm._active))
    return labels, l4_codes


def _learn(agent: Agent, *objects: dict):
    for _ in range(PASSES):
        for obj in (objects or (OBJ_A, OBJ_B)):
            _study(agent, obj)


def test_one_identity_per_object_no_duplication():
    agent = _fresh()
    _learn(agent)
    assert len(agent._nav_col().pooler.objects) == 2, (
        f"minted {len(agent._nav_col().pooler.objects)} objects over {PASSES} passes of 2 objects — L2/3 must pool one "
        "STABLE identity per object (recognise + reinforce a known one), not a new object per pass")


def test_recognition_is_stable_and_order_invariant():
    agent = _fresh()
    _learn(agent)
    a_labels, a_l4 = _traverse(agent, OBJ_A, list(reversed(list(OBJ_A))))   # a NEW order, inference only
    b_labels, _ = _traverse(agent, OBJ_B, list(OBJ_B))
    assert set(a_labels) == {0}, f"object A recognised as {a_labels} — must be the SAME identity (0) at every fixation"
    assert set(b_labels) == {1}, f"object B recognised as {b_labels} — must be the SAME identity (1) at every fixation"
    assert a_labels[0] != b_labels[0], "the two objects must have DISTINCT L2/3 identities"
    assert len(set(a_l4)) > 1, "L4 must CHANGE across fixations (feature-at-location) while the L2/3 identity stays constant"
    assert len(agent._nav_col().pooler.objects) == 2, "inference must NOT mint new objects (recognise, don't relearn)"


def test_shared_features_do_not_merge():
    """THE R5 REGRESSION. P and Q share their first TWO feature-at-locations (9@(30,30), 1@(31,30)) and diverge only at the
    third. Committing per FIXATION merged them: L2/3 recognised P at fixation 1 — correct inference, since at that moment
    they are indistinguishable — then persisted unconditionally and bound Q's third feature into P, yielding ONE chimeric
    identity holding both objects (measured: 2 objects → 1). Committing per EPISODE lets the third fixation refute P, because
    only the whole sweep carries the object's EXTENT."""
    agent = _fresh()
    _learn(agent, SHARE_P, SHARE_Q)
    assert len(agent._nav_col().pooler.objects) == 2, (
        f"P and Q share 2 of 3 feature-at-locations and must stay DISTINCT objects, got "
        f"{len(agent._nav_col().pooler.objects)} identities — a merge (1) means L2/3 never revised its first guess")
    p, _ = _traverse(agent, SHARE_P, list(SHARE_P))
    q, _ = _traverse(agent, SHARE_Q, list(SHARE_Q))
    assert p[-1] != q[-1], f"the diverging fixation must resolve P vs Q, got {p} / {q}"


def test_a_partial_view_of_a_known_object_is_not_a_new_object():
    """The flip side of the regression above, and why refutation must be EVIDENCE rather than "anything unexplained is new":
    sweeping only PART of a known object is all matches and no contradiction, so it must recognise + reinforce — not mint a
    duplicate. A rule that minted on any incomplete explanation would fragment every object it ever saw partially."""
    agent = _fresh()
    _learn(agent)
    before = len(agent._nav_col().pooler.objects)
    label = _study(agent, OBJ_A, order=list(OBJ_A)[:2])          # only 2 of A's 4 features
    assert len(agent._nav_col().pooler.objects) == before, "a PARTIAL sweep of a known object must not mint a duplicate"
    assert label == 0, f"a partial sweep of A must recognise A, got {label}"


if __name__ == "__main__":
    ag = _fresh()
    _learn(ag)
    a, a_l4 = _traverse(ag, OBJ_A, list(reversed(list(OBJ_A))))
    b, _ = _traverse(ag, OBJ_B, list(OBJ_B))
    print(f"library size (should be 2): {len(ag._nav_col().pooler.objects)}")
    print(f"object A sweep (new order) → identities {a}  (distinct L4 codes: {len(set(a_l4))}/4)")
    print(f"object B sweep → identities {b}")
    sh = _fresh()
    _learn(sh, SHARE_P, SHARE_Q)
    print(f"P/Q share 2 of 3 feature-at-locations → {len(sh._nav_col().pooler.objects)} identities (must be 2, was 1)")
