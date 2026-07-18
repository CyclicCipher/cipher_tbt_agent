"""End-to-end test of L5 DISPLACEMENT cells — relations between objects (ARCHITECTURE.md §9; ROADMAP Phase 3b).

TBT is explicit that a relation is a DIFFERENT cell type with a complementary job (`reference_tbt_layers_4_23`, Hawkins
2019): grid cells (L6a) do `location + movement → location`; DISPLACEMENT cells (L5PT, thick-tufted) do the INVERSE,
`location + location → the relation`, coding a RELATIVE vector between two frames — position- and orientation-invariant.

THE RELATION IS A DISPLACEMENT THAT STAYS STABLE AS THE PAIR MOVES. "Resting on", "part of", "attached" are exactly this,
and a compositional object is sub-objects at fixed relative displacements. This is [[feedback_prefer_generalize_then_correct]]
in miniature: the relation is ASSUMED fixed from the first view and DISSOLVED the moment the two objects move independently —
mis-generalise, then correct when the world contradicts it.

The object poses fed in are what `recognize` SOLVES (R4/R6), so perception → relations is end to end. NB L5 = IT + PT; this
is the PT thick-tufted displacement role, the L5IT associative integrator that gates into it deferred.
"""

from __future__ import annotations

import math
import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                                  # noqa: E402
from tbt.operator import eye, from_angle, rotate, to_angle   # noqa: E402

A = {(0.0, 0.0): 1, (2.0, 0.0): 2, (0.0, 2.0): 3}            # two distinct, asymmetric objects (identity is not the point here)
B = {(0.0, 0.0): 4, (2.0, 0.0): 5, (0.0, 2.0): 6}


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _learn(agent: Agent, obj: dict, passes: int = 6) -> None:
    for _ in range(passes):
        agent.start_object()
        for coord, feature in obj.items():
            agent.locate(coord)
            agent.sense_sweep(feature)
        agent.commit()


def _pose_of(agent: Agent, obj: dict, at, rotation=None):
    """Present the object at a world pose and let the agent SOLVE it — its (origin, R), the input a displacement needs."""
    rotation = rotation or eye(2)
    agent.start_object()
    for coord, feature in obj.items():
        p = rotate(rotation, coord)
        agent.locate((p[0] + at[0], p[1] + at[1]))
        agent.sense_sweep(feature)
    h = agent.recognize()[0]
    return h.label, (h.origin, h.rotation)


def _close(a, b, tol=1e-6) -> bool:
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_the_relation_is_INVARIANT_to_where_the_pair_is():
    """The displacement cell's defining property: the relative pose of B in A's frame does not depend on WHERE the pair sits.
    A relation learned for a pair therefore holds for that pair anywhere — the same generalisation-by-frame as everywhere
    else, now between two objects."""
    agent = _fresh()
    _learn(agent, A)
    _learn(agent, B)
    ref = None
    for shift in ((0.0, 0.0), (10.0, 0.0), (3.0, 7.0), (20.0, 20.0)):
        _, pa = _pose_of(agent, A, (shift[0] + 5.0, shift[1]))     # A five to the right of B, as a rigid pair
        _, pb = _pose_of(agent, B, shift)
        dp, dr = agent.relate(pa, pb)
        assert _close(dp, (-5.0, 0.0)), f"pair at {shift}: relative position {dp} (must be invariant)"
        assert abs(to_angle(dr)) < 1e-6, f"pair at {shift}: relative angle {to_angle(dr)}"
        ref = ref or (dp, dr)
    assert ref is not None


def test_the_relation_is_INVARIANT_to_the_pair_S_ORIENTATION():
    """Rotate the WHOLE pair rigidly and the relation is unchanged — a relation is about how two things are arranged relative
    to each other, not their shared orientation. (Needs the pose solve to recover each object's rotation, R4/R6.)"""
    agent = _fresh()
    _learn(agent, A)
    _learn(agent, B)
    for deg in (0.0, 40.0, 137.0):
        Q = from_angle(deg)
        _, pa = _pose_of(agent, A, rotate(Q, (5.0, 0.0)), Q)        # the pair, rotated rigidly by Q about B's origin
        _, pb = _pose_of(agent, B, (0.0, 0.0), Q)
        dp, _dr = agent.relate(pa, pb)
        assert _close(dp, (-5.0, 0.0)), f"pair rotated {deg}°: relative position {dp} must be unchanged"


def test_a_STABLE_relation_is_learned_when_the_pair_MOVES_TOGETHER():
    """The relation is CONFIRMED once it holds across views — 'resting on' as a fixed relative pose that survives the pair
    moving as one. This reuses common fate's 'moves together', but captures the RELATION, not just the grouping."""
    agent = _fresh()
    _learn(agent, A)
    _learn(agent, B)
    la = lb = None
    for shift in ((0.0, 0.0), (10.0, 0.0), (3.0, 7.0)):            # the pair moves rigidly, offset fixed
        la, pa = _pose_of(agent, A, (shift[0] + 5.0, shift[1]))
        lb, pb = _pose_of(agent, B, shift)
        agent.observe_relation(la, pa, lb, pb)
    rel = agent.relation_of(la, lb)
    assert rel is not None, "a relative pose that held across several rigid moves must be learned as a fixed relation"
    assert _close(rel[0], (-5.0, 0.0)), f"the learned relation is the fixed offset, got {rel[0]}"


def test_the_relation_DISSOLVES_when_they_move_INDEPENDENTLY():
    """The correction half. Once the two objects move to a DIFFERENT relative pose, they have no fixed relation — the
    assumption is retracted by the evidence, not defended."""
    agent = _fresh()
    _learn(agent, A)
    _learn(agent, B)
    la, pa = _pose_of(agent, A, (5.0, 0.0))
    lb, pb = _pose_of(agent, B, (0.0, 0.0))
    agent.observe_relation(la, pa, lb, pb)                          # first: a candidate relation
    agent.observe_relation(la, pa, lb, pb)                          # confirmed stable...
    assert agent.relation_of(la, lb) is not None, "held twice ⇒ provisionally a relation"
    la2, pa2 = _pose_of(agent, A, (40.0, 0.0))                      # ...then A moves independently (offset changes)
    agent.observe_relation(la2, pa2, lb, pb)
    assert agent.relation_of(la2, lb) is None, "moved apart ⇒ the relation must DISSOLVE (mis-generalise, then correct)"


if __name__ == "__main__":
    ag = _fresh()
    _learn(ag, A)
    _learn(ag, B)
    print("relative pose of A-in-B as the PAIR moves rigidly (must be invariant):")
    for shift in ((0.0, 0.0), (12.0, -4.0), (30.0, 30.0)):
        _, pa = _pose_of(ag, A, (shift[0] + 5.0, shift[1]))
        la, pb = _pose_of(ag, B, shift)
        dp, dr = ag.relate(pa, pb)
        print(f"  pair at {shift}: rel = pos {tuple(round(c, 2) for c in dp)}, angle {to_angle(dr):.1f}°")
