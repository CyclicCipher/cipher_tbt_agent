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


def _sweep(agent: Agent, feats):
    """INFER: one OBJECT-CENTRIC sweep — onset (re-anchor to the origin), then step +E reading L2/3 object-relative."""
    agent.start_object()
    ids = [agent.perceive(feats[0])]
    for f in feats[1:]:
        agent.path_integrate("E")
        ids.append(agent.perceive(f))
    return ids


def _study(agent: Agent, feats) -> int:
    """LEARN: the same object-centric sweep, buffered, then COMMITTED as one episode (the R5 learning path)."""
    agent.start_object()
    agent.sense_sweep(feats[0])
    for f in feats[1:]:
        agent.path_integrate("E")
        agent.sense_sweep(f)
    return agent.commit()


def _learn(agent: Agent, *objects, passes: int = 6) -> None:
    for _ in range(passes):
        for feats in (objects or (A_FEATS, B_FEATS)):
            _study(agent, feats)


def test_object_centric_is_translation_invariant():
    agent = _fresh()
    _teach_operator(agent)
    _learn(agent)
    a = _sweep(agent, A_FEATS)                        # re-present A (placement-independent — the frame is relative)
    b = _sweep(agent, B_FEATS)
    assert set(a) == {0}, f"object A must be recognised as ONE identity regardless of placement, got {a}"
    assert set(b) == {1}, f"object B must be recognised as identity 1, got {b}"
    assert len(agent._nav_col().pooler.objects) == 2, "re-presenting a known object must NOT mint a new one"


def test_emergent_boundary_segments_a_continuous_scene():
    agent = _fresh()
    _teach_operator(agent)
    _learn(agent)
    agent.start_object()                              # ONE onset for the whole scene
    ids = [agent.perceive(A_FEATS[0])]                # at the origin → recognise A
    for f in A_FEATS[1:]:
        agent.path_integrate("E")
        ids.append(agent.perceive(f))
    for f in B_FEATS:                                 # keep moving INTO B — NO explicit boundary
        agent.path_integrate("E")
        ids.append(agent.perceive(f))
    assert ids[:3] == [0, 0, 0], f"first object should read as identity 0 throughout, got {ids[:3]}"
    assert ids[3:] == [1, 1, 1], f"the boundary must be found emergently → identity 1, got {ids[3:]}"
    assert len(agent._nav_col().pooler.objects) == 2, "the emergent boundary must not mint spurious objects"


def test_ONLINE_recognition_SOLVES_its_place_on_the_object():
    """THE CROWN of the online pose-solving slice. `perceive` used to bind and recall at whatever coordinate the caller
    supplied, so it only recognised an object entered AT ITS LEARNED ORIGIN: measured, the same object shifted to (7,3) read
    `[-1,-1,-1]` and entered mid-object `[-1,-1]`, while the buffered path solved the identical presentation. Now the online
    path narrows a population of (object, pose) hypotheses, so it recognises the object ENTERED ANYWHERE and PLACED
    anywhere — no assumption, and the caller's coordinate frame stops being a silent contract."""
    agent = _fresh()
    _teach_operator(agent)
    _learn(agent)

    def read(feats, start):                           # sweep from an arbitrary world position
        agent.start_object()
        agent.locate(start)
        ids = [agent.perceive(feats[0])]
        for f in feats[1:]:
            agent.path_integrate("E")
            ids.append(agent.perceive(f))
        return ids

    assert read(A_FEATS, (0.0, 0.0)) == [0, 0, 0], "at the learned origin (the case that always worked)"
    assert read(A_FEATS, (7.0, 3.0)) == [0, 0, 0], "the SAME object placed elsewhere — was [-1,-1,-1] when the pose was assumed"
    assert read(B_FEATS, (-4.0, 9.0)) == [1, 1, 1], "and the other object, somewhere else again"
    # ENTERED MID-OBJECT: the sweep starts on A's SECOND feature, so the caller cannot be at the origin even by luck.
    agent.start_object()
    agent.locate((5.0, 5.0))
    mid = [agent.perceive(A_FEATS[1])]
    agent.path_integrate("E")
    mid.append(agent.perceive(A_FEATS[2]))
    assert mid == [0, 0], f"entered mid-object, origin unknowable — was [-1,-1] when the pose was assumed, got {mid}"


def test_a_continuous_LEARNING_sweep_splits_itself_at_the_boundary():
    """THE R7 CROWN: the caller no longer has to say where one object ends. Sweep A and B as ONE continuous episode — one
    onset, no boundary cue — and `commit` splits it: A is reinforced, B is reinforced, and no spurious A+B blob is minted.

    The signal is REFUTATION, which recognition already produced (`reference_tbt_segmentation_and_grouping`: "it relies on
    feature and morphology mismatch to implicitly detect boundaries"). The judgement is `_exhausts`: the prefix covered ALL
    of A's model, so A ENDED there — as opposed to a sweep that merely SHARES a prefix with A, which is one different object
    (`test_l23_pooling.test_shared_features_do_not_merge` — the same evidence, the opposite reading, told apart by whether
    the prefix reached the object's edge)."""
    agent = _fresh()
    _teach_operator(agent)
    _learn(agent)                                     # A and B learned as separate episodes
    before = len(agent._nav_col().pooler.objects)
    for _ in range(3):
        agent.start_object()                          # ONE onset for BOTH objects — no boundary cue anywhere
        agent.sense_sweep(A_FEATS[0])
        for f in A_FEATS[1:] + B_FEATS:               # sweep straight through A and on into B
            agent.path_integrate("E")
            agent.sense_sweep(f)
        agent.commit()
    assert len(agent._nav_col().pooler.objects) == before, (
        f"a continuous sweep over two KNOWN objects must split at the boundary and reinforce both, not mint a blob — "
        f"library went {before} → {len(agent._nav_col().pooler.objects)}")
    assert set(_sweep(agent, A_FEATS)) == {0} and set(_sweep(agent, B_FEATS)) == {1}, \
        "both objects must survive the continuous sweep intact"


def test_a_NOVEL_object_after_a_known_one_is_minted_not_absorbed():
    """The same split, but the remainder is UNKNOWN. Having exhausted A, the sweep has left it, so what follows is a new
    object — minted in ITS OWN frame (the origin is where the sweep first touched it, not the episode's anchor), which is
    what makes it recognisable later on its own."""
    agent = _fresh()
    _teach_operator(agent)
    _learn(agent, A_FEATS)                            # ONLY A is known
    for _ in range(4):
        agent.start_object()
        agent.sense_sweep(A_FEATS[0])
        for f in A_FEATS[1:] + B_FEATS:               # A, then straight into a NOVEL object
            agent.path_integrate("E")
            agent.sense_sweep(f)
        agent.commit()
    assert len(agent._nav_col().pooler.objects) == 2, (
        f"A was exhausted, so the novel remainder must become its OWN object, got "
        f"{len(agent._nav_col().pooler.objects)} identities")
    a, b = _sweep(agent, A_FEATS), _sweep(agent, B_FEATS)
    assert set(a) == {0}, f"A must still read as itself, got {a}"
    assert set(b) == {1}, f"the object learned WITHOUT ever being marked must be recognisable alone, got {b}"


def test_relative_arrangement_is_load_bearing():
    """The object-centric LOCATION must do real work: two objects with the SAME features in a DIFFERENT relative ARRANGEMENT
    must get DIFFERENT identities. Feature-only recognition cannot tell {7,8,9} from {7,8,9}; only feature-AT-relative-location
    can — so this proves the re-anchored frame is load-bearing, not incidental.

    NB the arrangements must not be related by a ROTATION. The original pair here was P=[7,8] / Q=[8,7], which R4 exposed as
    DEGENERATE: on a line, 8-then-7 is literally 7-then-8 rotated 180°, so a pose-invariant recogniser is RIGHT to call them
    one object at two poses — identity and pose are factored, and the orientation is not lost, it is reported. Three cells
    make the pair genuinely distinct: [7,8,9] rotated 180° reads 9-then-8-then-7, which is neither arrangement below.

    Since the online path SOLVES its place rather than assuming it, this now also shows the ARRANGEMENT doing the work in
    real time: the two read `[-1, -1, 0]` and `[-1, -1, 1]`. The leading -1s are not a failure, they are the honest answer.
    Feature 7 belongs to BOTH objects, and after two fixations [7,8] is still explained by P at 0° AND by Q at 180° (on a
    line, one IS the other rotated) — so only the third fixation separates them, and that is precisely the claim: features
    alone can NEVER tell these apart, the relative arrangement is what does it. Under the retired code this read `[0,0,0]`
    from the first fixation, but only because it ASSUMED the sensor sat on the object's origin: the assumption, not the
    evidence, was breaking the tie."""
    agent = _fresh()
    _teach_operator(agent)
    P, Q = [7, 8, 9], [8, 7, 9]                       # same features; NOT rotations of each other
    _learn(agent, P, Q, passes=8)
    p = _sweep(agent, P)
    q = _sweep(agent, Q)
    assert p[-1] != -1 and q[-1] != -1, f"the full arrangement must settle each object, got {p} / {q}"
    assert p[-1] != q[-1], f"same features, different arrangement → DIFFERENT identities (frame load-bearing), got {p} / {q}"
    assert p[0] == -1 and q[0] == -1, (
        f"a SHARED first feature cannot name the object — reporting one would be a guess, not inference; got {p} / {q}")


if __name__ == "__main__":
    ag = _fresh()
    _teach_operator(ag)
    _learn(ag)
    print(f"object-centric re-present A: {_sweep(ag, A_FEATS)}  B: {_sweep(ag, B_FEATS)}  "
          f"(objects: {len(ag._nav_col().pooler.objects)})")
