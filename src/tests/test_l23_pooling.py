"""End-to-end test of L2/3 TEMPORAL POOLING (ARCHITECTURE.md §8, STATUS "Next" #2): pool the L4 feature-at-location stream
into a STABLE object IDENTITY (`pooler.ColumnPooler`), driven from `agent.py`.

The defining property (`reference_tbt_layers_4_23`): L4 changes every fixation (a different feature-at-location), but L2/3
stays the SAME — one sparse identity per object, persisting across the traversal, recognised INVARIANT to fixation order,
and minted ONCE (not a new object every frame — the duplication bug the pooler fixes). Recognition is associative recall
over the object library, not a from-scratch recompute.

Two objects with the same SHAPE and different colours — identity is the features AT the locations, so this isolates the
pooler; the operator ⊕ L4 composition is covered by `test_feature_at_location`, and placement-independence by
`test_object_centric`. Fixations are placed with `locate` (a sensory fix) rather than path integration, in the object's OWN
frame: an object's frame origin is the first fixation of the episode that learned it (`reference_tbt_object_frame_bootstrap`),
so a sweep must be presented in object-relative coordinates — absolute ones would learn a model no later sweep could match.
RULES #3 acceptance: the agent recognises WHICH object it is sensing, stable across the sweep.

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

OBJ_A = {(0, 0): 1, (1, 0): 2, (0, 1): 3, (1, 1): 4}   # colours 1..4
OBJ_B = {(0, 0): 5, (1, 0): 6, (0, 1): 7, (1, 1): 8}   # colours 5..8, the same shape — identity is the FEATURES at them
# Two objects that SHARE their first two feature-at-locations and diverge at the third — the case that used to merge.
SHARE_P = {(0, 0): 9, (1, 0): 1, (0, 1): 2}
SHARE_Q = {(0, 0): 9, (1, 0): 1, (0, -1): 3}           # same FIRST TWO fixations as P, then diverges
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


# ── the COMPOSITIONAL region: feature-at-location one level up ─────────────────────────────────────────────────────────
# `Column.place_object(..., identity=)` drives the higher region's L4 with an OBJECT's identity at its POSE — the same
# L4↔L6a loop a sensory column runs on colours-at-cells. Until 2026-07-23 this method only wrote a dict entry, so the
# compositional column had no cortical input at all and the "hierarchy" was sibling columns calling each other.

def test_the_compositional_region_learns_objects_at_poses():
    """Objects are the higher region's FEATURES and their poses its LOCATIONS. After seeing two objects at two poses the
    column predicts which object belongs where — and does NOT predict one at the other's pose, so the code is genuinely
    location-specific rather than a bag of identities."""
    from tbt.agent import Agent
    from tbt.operator import eye
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    from tbt.encoders import SDR
    pose = lambda c: (tuple(float(x) for x in c), eye(2))
    ident = lambda k: SDR(64, range(k * 8, k * 8 + 8))      # stand-in for a lower region's settled identity
    for _ in range(4):
        a.place_object(6, pose((3, 2)), identity=ident(0))
        a.place_object(7, pose((6, 4)), identity=ident(1))
    col = a._scene_col()

    def predicts(cell, k):
        col.set_pose(tuple(float(x) for x in cell), eye(2))
        return len(set(col.predict_feature()) & ident(k).active)

    assert predicts((3, 2), 0) == 8, "the object learned at this pose must be predicted there"
    assert predicts((6, 4), 1) == 8, "and likewise the other"
    assert predicts((6, 4), 0) == 0, "but not an object that was never at this pose"


def test_the_sensory_region_has_an_output_and_the_edge_is_cortical():
    """THE EDGE, closed. The sensory region was built WITHOUT a frame, so it had no pooler and therefore nothing to send —
    `edges()` was empty and the compositional column was driven by a transduced colour reaching two levels up. Giving it a
    frame gives it an L2/3 pooler, its settled identity is what now drives the higher region, and the heterarchy reports a
    real region → region edge. It is a SEPARATE region from `nav` on purpose: sweeping an object re-anchors L6a, which would
    wreck the body pose nav path-integrates."""
    from tbt.agent import Agent
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    a._scene_col()
    assert a.sensory.pooler is not None, "the sensory region must have an L2/3 output to send"
    assert a.sensory is not a._nav_col(), "and be distinct from the region path-integrating the body"
    assert ("sensory", "scene") in a.hierarchy.edges(), f"a cortical edge must be declared, got {a.hierarchy.edges()}"


def test_a_settled_identity_is_pose_invariant():
    """What makes the identity worth sending: the same shape seen somewhere else settles on the SAME identity, so what
    travels up the hierarchy is the region's CONCLUSION about what a thing is, factored from where it is. `commit` defers
    on early looks — until L4 predicts part of the sweep there is nothing to ground an identity on — so this takes a few."""
    import numpy as np
    from tbt.agent import Agent
    from tbt.perceive import segment
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    g = np.zeros((6, 8), dtype=int); g[2, 3] = 6; g[2, 4] = 6
    obj = [o for o in segment(g.tolist()) if o.color == 6][0]
    for _ in range(3):
        here = a._object_identity(obj)
    assert here, "a repeated look must eventually mint an identity"

    g2 = np.zeros((6, 8), dtype=int); g2[4, 1] = 6; g2[4, 2] = 6
    moved = [o for o in segment(g2.tolist()) if o.color == 6][0]
    assert a._object_identity(moved) == here, "the same shape elsewhere is the SAME object, not a new one"


def test_same_coloured_objects_are_told_apart_by_shape():
    """`_positions` used to return NOTHING for a repeated colour — it keyed on "exactly one object of this colour", so two
    same-coloured objects lost BOTH, and its docstring deferred the fix to recognition. Recognition is wired now: each is
    swept, L2/3 settles an identity, and the differing SHAPES give differing identities, so both are tracked."""
    import numpy as np
    from tbt.agent import Agent
    from tbt.perceive import segment
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    g = np.zeros((8, 10), dtype=int)
    g[1, 1] = 6; g[1, 2] = 6                          # a domino …
    g[5, 5] = 6; g[5, 6] = 6; g[6, 5] = 6             # … and an L-tromino, SAME colour
    objs = segment(g.tolist())
    for _ in range(4):                                 # `commit` defers on early looks while L4 learns
        pos = a._positions(objs)
    assert len(pos) == 2, f"both same-coloured objects must be tracked, got {pos}"
    assert {v for v in pos.values()} == {(1, 1), (5, 5)}, "and at their own anchors"


def test_one_object_per_colour_still_keys_on_the_colour():
    """The common case is untouched: a colour realised once keys on the colour itself, so nothing that depended on plain
    feature handles changes."""
    import numpy as np
    from tbt.agent import Agent
    from tbt.perceive import segment
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    g = np.zeros((6, 8), dtype=int); g[2, 3] = 6; g[4, 6] = 7
    pos = a._positions(segment(g.tolist()))
    assert pos == {6: (3, 2), 7: (6, 4)}, f"unique colours key on the colour, got {pos}"
