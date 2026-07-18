"""test_world_map.py — the hippocampal allocentric world-STATE (hippocampus/map.py; DESIGN §2, slice 1).

The rollout's block (DESIGN §0) was that nothing held a coherent, FORKABLE "world right now, including me": the agent's pose
lived in the nav column, the objects in the scene column, neither cheaply copyable to try a hypothetical. `WorldMap` binds
them into one forkable state that path-integrates the agent under the SHARED learned operator (borrowed by reference, never
re-learned) — the substrate replay.py will simulate in. These tests exercise exactly the new capability: assemble the state,
fork it, run the agent forward in a fork without disturbing the parent, and close the loop by re-anchoring to a landmark.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                     # noqa: E402
from tbt.hippocampus.map import WorldMap        # noqa: E402
from tbt.operator import eye                    # noqa: E402


def _close(a, b, tol=1e-6) -> bool:
    return all(abs(x - y) < tol for x, y in zip(a, b))


def _agent_that_can_move() -> Agent:
    """An agent localised at (10,10) that has learned ONE body action ('E' = east by 1) from a single observed move —
    position-invariant thereafter, so it applies at (10,10) though it was seen at the origin."""
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    a.set_pose((10.0, 10.0), eye(2))
    a.learn_pose_move("E", ((0.0, 0.0), eye(2)), ((1.0, 0.0), eye(2)))
    return a


def test_world_state_binds_agent_and_objects():
    """The state combines the agent's self-location (nav column) with the scene's objects (scene column) + the frame extent
    — one coherent world, where before they lived in two columns with no joint handle."""
    a = _agent_that_can_move()
    a.place_object(7, ((5.0, 5.0), eye(2)))
    a.place_object(9, ((20.0, 20.0), eye(2)))
    w = a.world_state()
    assert _close(w.agent[0], (10.0, 10.0)), "the agent's self-location is in the map"
    assert set(w.objects) == {7, 9}, "the scene's objects are in the map"
    assert _close(w.objects[7][0], (5.0, 5.0))
    assert w.in_bounds((10, 10)) and not w.in_bounds((100, 100)), "the boundary is the frame extent"


def test_agent_path_integrates_inside_the_map():
    """move_agent applies the SHARED learned operator to the agent within the map — the map borrows the model, it does not
    re-derive motion. It returns a NEW map (a fork), leaving the parent untouched, and objects are unmoved by the agent."""
    a = _agent_that_can_move()
    a.place_object(7, ((5.0, 5.0), eye(2)))
    w = a.world_state()
    w2 = w.move_agent("E")
    assert _close(w2.agent[0], (11.0, 10.0)), f"agent moved east under the shared operator, got {w2.agent[0]}"
    assert _close(w.agent[0], (10.0, 10.0)), "the parent map is UNCHANGED — move_agent forks, it does not mutate"
    assert w2.objects[7] == w.objects[7], "moving the agent does not move objects"


def test_the_fork_is_independent():
    """Simulability: edits to a forked branch (place/remove an object) never leak back to the parent — the property a rollout
    tree needs to branch hypotheticals safely."""
    a = _agent_that_can_move()
    a.place_object(7, ((5.0, 5.0), eye(2)))
    w = a.world_state()
    branch = w.move_agent("E")
    branch.place(8, ((0.0, 0.0), eye(2)))
    branch.remove(7)
    assert set(w.objects) == {7}, "mutating a fork must not touch the parent's object set"
    assert set(branch.objects) == {8}, "the fork sees its own edits"


def test_unlearned_action_is_the_identity():
    """An action the body operator has never seen leaves the agent put — the operator's own correct prior (predict staying),
    so a rollout may safely try any action label without special-casing the unknown ones."""
    a = _agent_that_can_move()
    w = a.world_state()
    w2 = w.move_agent("NORTH")
    assert _close(w2.agent[0], (10.0, 10.0)), "an unlearned action leaves the agent where it is"


def test_loop_closure_anchor_corrects_drift():
    """Dead-reckoning three steps east accumulates to (13,10); re-seeing the landmark and anchoring resets the position while
    keeping orientation — loop closure, what bounds path-integration drift (`reference_hippocampus`)."""
    a = _agent_that_can_move()
    drifted = a.world_state()
    for _ in range(3):
        drifted = drifted.move_agent("E")
    assert _close(drifted.agent[0], (13.0, 10.0)), f"three east steps → (13,10), got {drifted.agent[0]}"
    drifted.anchor((10.0, 10.0))
    assert _close(drifted.agent[0], (10.0, 10.0)), "anchoring resets the path-integrated position (loop closure)"
    assert drifted.agent[1] == eye(2), "orientation is kept through the anchor"


if __name__ == "__main__":
    ag = _agent_that_can_move()
    ag.place_object(7, ((5.0, 5.0), eye(2)))
    world = ag.world_state()
    print(f"world: agent at {world.agent[0]}, objects {[ (i, p[0]) for i, p in world.objects.items() ]}")
    for step in range(3):
        world = world.move_agent("E")
        print(f"  after E×{step + 1}: agent at {tuple(round(c, 1) for c in world.agent[0])}")
