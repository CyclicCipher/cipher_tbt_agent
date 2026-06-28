"""The planner (tbt.plan): compose the LEARNED forward model + LEARNED goal through the existing Neocortex achiever,
with no role schema and no hand-coded move geometry (the harness dissolved). A closed-loop test drives a simulated
self by its OWN learned operators toward a learned goal and checks it arrives; another teaches non-unit "hop"
operators to prove the planner uses whatever was learned, not an assumed DELTAS table. Pure stdlib."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.forward import ForwardModel       # noqa: E402
from tbt.goal import GoalModel             # noqa: E402
from tbt.plan import Planner               # noqa: E402


def _teach(fm, ops):
    """Teach the forward model one clean transition per operator (action_key -> (dx, dy))."""
    for a, (dx, dy) in ops.items():
        fm.observe((0, 0), a, (dx, dy))


def test_planner_navigates_to_a_learned_goal():
    fm = ForwardModel()
    _teach(fm, {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)})   # up/down/left/right, learned
    g = GoalModel()
    g.observe((9, 8), [("box", (10, 8))], 1)                     # goal: be one cell left of the (static) box
    p = Planner(fm, g)

    self_pose, others = (3, 8), [("box", (10, 8))]
    steps = 0
    while not g.is_goal(self_pose, others) and steps < 20:
        self_pose = fm.predict(self_pose, p.act(self_pose, others))
        steps += 1
    assert g.is_goal(self_pose, others)
    assert self_pose == (9, 8)
    assert steps <= 8                                            # ~optimal (6); slack for tie-breaking


def test_uses_learned_operators_not_an_assumed_move_set():
    """The planner has no notion of unit moves: teach (+2,0)/(-2,0) hop operators and it still reaches a goal six
    cells along, in three hops -- exactly because the geometry is read from forward.py, not a DELTAS table."""
    fm = ForwardModel()
    _teach(fm, {"hopR": (2, 0), "hopL": (-2, 0)})
    g = GoalModel()
    g.observe((6, 0), [("flag", (20, 0))], 1)                    # the flag is a static landmark giving position meaning
    p = Planner(fm, g)

    self_pose, others = (0, 0), [("flag", (20, 0))]
    seq = []
    while not g.is_goal(self_pose, others) and len(seq) < 10:
        a = p.act(self_pose, others)
        seq.append(a)
        self_pose = fm.predict(self_pose, a)
    assert self_pose == (6, 0)
    assert seq == ["hopR", "hopR", "hopR"]                       # three learned hops, no unit-move assumption


def test_cold_start_returns_a_valid_action_with_no_goal_yet():
    fm = ForwardModel()
    _teach(fm, {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)})
    g = GoalModel()                                              # nothing rewarded yet
    p = Planner(fm, g)
    a = p.act((5, 5), [("box", (10, 10))])
    assert a in (0, 1, 2, 3)                                     # wanders with a valid operator, does not crash
