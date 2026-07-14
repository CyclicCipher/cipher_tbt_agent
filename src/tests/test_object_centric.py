"""End-to-end test of the OBJECT-CENTRIC frame + the emergent object boundary (ARCHITECTURE.md §8; research:
`notes/tbt_object_frame_and_bootstrap_research.md`).

TWO coupled behaviours, ONE event. Per Lewis et al. 2019, allocating a fresh frame origin for an object is the SAME act as
individuating it — so on our substrate a single RECOGNITION-FAILURE event both (a) re-anchors the L6a frame (sensing becomes
OBJECT-RELATIVE → translation-invariant) and (b) starts a fresh L2/3 identity. No symbolic segmenter: the trigger IS the
recognition mechanism failing (`Column.perceive` / `start_object`).

- **Translation invariance:** an object is recognised by its RELATIVE feature arrangement, independent of where it sits —
  because `start_object` re-origins the frame. The allocentric flow (world coords) canNOT do this: the same shape at a
  different world position is a NEW identity — the bug this fixes (contrasted below).
- **Emergent boundary:** a continuous two-object sweep with ONE onset finds the object boundary by recognition failure
  (the first object's model stops predicting) → re-anchor + recognise the next object — no explicit reset at the boundary.

Learning-time boundaries use an explicit `start_object` (the honest minimal episode cue TBT itself uses); the INFERENCE-time
boundary is fully emergent. RULES #3: the agent now recognises objects by structure, invariant to placement.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent  # noqa: E402

A_FEATS = [1, 2, 3]                                   # object A = these colours at relative (0,0),(1,0),(2,0)
B_FEATS = [4, 5, 6]                                   # object B = a different shape (distinct colours)


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _teach_operator(agent: Agent) -> None:
    """Learn the "E" step so relative path integration works (over the small object-relative range)."""
    for x in range(0, 9):
        agent.learn_move("E", (x, 0), (x + 1, 0))


def _sweep(agent: Agent, feats, learn: bool):
    """One OBJECT-CENTRIC sweep: onset (re-anchor to the origin), then step +E sensing each feature object-relative."""
    agent.start_object()
    ids = [agent.perceive(feats[0], learn)]
    for f in feats[1:]:
        agent.path_integrate("E")
        ids.append(agent.perceive(f, learn))
    return ids


def _learn(agent: Agent, passes: int = 6) -> None:
    for _ in range(passes):
        _sweep(agent, A_FEATS, learn=True)
        _sweep(agent, B_FEATS, learn=True)


def test_object_centric_is_translation_invariant():
    agent = _fresh()
    _teach_operator(agent)
    _learn(agent)
    a = _sweep(agent, A_FEATS, learn=False)           # re-present A (placement-independent — the frame is relative)
    b = _sweep(agent, B_FEATS, learn=False)
    assert set(a) == {0}, f"object A must be recognised as ONE identity regardless of placement, got {a}"
    assert set(b) == {1}, f"object B must be recognised as identity 1, got {b}"
    assert len(agent._nav_col().pooler.objects) == 2, "re-presenting a known object must NOT mint a new one"


def test_emergent_boundary_segments_a_continuous_scene():
    agent = _fresh()
    _teach_operator(agent)
    _learn(agent)
    agent.start_object()                              # ONE onset for the whole scene
    ids = [agent.perceive(A_FEATS[0], learn=False)]   # at the origin → recognise A
    for f in A_FEATS[1:]:
        agent.path_integrate("E")
        ids.append(agent.perceive(f, learn=False))
    for f in B_FEATS:                                 # keep moving INTO B — NO explicit boundary
        agent.path_integrate("E")
        ids.append(agent.perceive(f, learn=False))
    assert ids[:3] == [0, 0, 0], f"first object should read as identity 0 throughout, got {ids[:3]}"
    assert ids[3:] == [1, 1, 1], f"the boundary must be found emergently → identity 1, got {ids[3:]}"
    assert len(agent._nav_col().pooler.objects) == 2, "the emergent boundary must not mint spurious objects"


def test_relative_arrangement_is_load_bearing():
    """The object-centric LOCATION must do real work: two objects with the SAME features in a DIFFERENT relative arrangement
    (P = 7-then-8, Q = 8-then-7) must get DIFFERENT identities. Feature-only recognition could not tell {7,8} from {7,8};
    only feature-AT-relative-location can — so this proves the re-anchored frame is load-bearing, not incidental."""
    agent = _fresh()
    _teach_operator(agent)
    P, Q = [7, 8], [8, 7]                             # same features, swapped arrangement
    for _ in range(8):
        _sweep(agent, P, learn=True)
        _sweep(agent, Q, learn=True)
    p = _sweep(agent, P, learn=False)
    q = _sweep(agent, Q, learn=False)
    assert len(set(p)) == 1 and len(set(q)) == 1, f"each arrangement must read as ONE stable identity, got {p} / {q}"
    assert p[0] != q[0], f"same features, swapped arrangement → DIFFERENT identities (frame load-bearing), got {p[0]}=={q[0]}"


if __name__ == "__main__":
    ag = _fresh()
    _teach_operator(ag)
    _learn(ag)
    print(f"object-centric re-present A: {_sweep(ag, A_FEATS, False)}  B: {_sweep(ag, B_FEATS, False)}  "
          f"(objects: {len(ag._nav_col().pooler.objects)})")
