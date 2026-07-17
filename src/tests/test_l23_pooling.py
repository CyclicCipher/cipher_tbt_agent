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

from tbt.agent import Agent          # noqa: E402
from tbt.pooler import ColumnPooler  # noqa: E402

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


def test_ART_the_two_normalisations_do_two_DIFFERENT_jobs():
    """The ART cut-over, tested where the mechanism lives. Two categories both explain the input PERFECTLY and neither is
    refuted: a SMALL model whose every cell is present, and a BIG one of which only half is present.

    * **VIGILANCE** `M_j = |I ∧ w_j| / |I|` is normalised by the **INPUT** — "how much of what I am seeing does this account
      for?" Both score 1.0. It CANNOT tell them apart, and that is correct: a partial view of a known object must not
      fragment it (`test_a_partial_view_of_a_known_object_is_not_a_new_object`). Our "nothing REFUTES it" bar IS this, at
      ρ=1.0; `rho < 1` is the deferred sensor-noise knob.
    * **CHOICE** `T_j = |I ∧ w_j| / (α + |w_j|)` is normalised by the **CATEGORY** — it prefers the SMALLEST model that
      explains the input (Grossberg's conservative limit = Tenenbaum's size principle: seeing exactly the small model's cells
      and none of the big one's others is a *suspicious coincidence*).

    We had only one number (raw accumulated evidence, a SUM) and were making it do both jobs — which is why a blob explaining
    2 of its 4 cells TIED an object explaining 2 of 2, and measurably absorbed it forever. Two jobs need two normalisations."""
    pooler = ColumnPooler(seed=0)
    small, big = pooler.mint(), pooler.mint()
    seen = frozenset(range(16))                       # the input I: the cells this sweep actually activated
    pooler.bind(seen, small)                          # the SMALL model is exactly the input
    pooler.bind(seen | frozenset(range(16, 32)), big)  # the BIG model is the input PLUS as much again, unseen
    assert pooler.match(seen, small) == 1.0 and pooler.match(seen, big) == 1.0, \
        "VIGILANCE must NOT separate them — both explain every cell of what is being seen (that is the partial-view rule)"
    assert pooler.choice(seen, small) > pooler.choice(seen, big), (
        f"CHOICE must prefer the SMALLER model: {pooler.choice(seen, small):.3f} vs {pooler.choice(seen, big):.3f} — "
        f"this is the term we lacked, and its absence is what let a blob absorb its own parts")


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
