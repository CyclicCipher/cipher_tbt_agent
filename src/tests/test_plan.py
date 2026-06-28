"""The planner (tbt.plan): one active-inference value over the object configuration, with NO self. The controllable
object EMERGES -- the planner rolls every object's operator, and the action-sensitive one moves toward the goal.
Babble (untried actions), explore (novel configs), exploit (goal config) are one value. Pure stdlib."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.forward import ForwardModel       # noqa: E402
from tbt.goal import GoalModel             # noqa: E402
from tbt.plan import Planner               # noqa: E402

MOVES = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}


def _mover():
    fm = ForwardModel()
    for a, (dx, dy) in MOVES.items():
        fm.observe((0, 0), a, (dx, dy))
    return fm


def test_moves_the_controllable_object_to_a_goal_with_no_self():
    """Object 0 is action-sensitive (controllable); object 1 (the larger anchor) has no operator, so it is static.
    The planner is never told which is the self -- it rolls both operators and the action moves object 0 to the goal."""
    forwards = {0: _mover()}
    g = GoalModel()
    g.observe({0: ((9, 8), 1), 1: ((10, 8), 4)}, 1)          # goal: object 0 one cell left of the big object
    p = Planner(g)
    objects = {0: ((3, 8), 1), 1: ((10, 8), 4)}
    steps = 0
    while not g.is_goal(objects) and steps < 25:
        a = p.act(objects, forwards, [0, 1, 2, 3], {0, 1, 2, 3})
        objects = {0: (forwards[0].predict(objects[0][0], a), 1), 1: ((10, 8), 4)}   # object 1 stays put (static)
        steps += 1
    assert g.is_goal(objects) and objects[0][0] == (9, 8) and steps <= 10


def test_babbles_an_untried_action():
    p = Planner(GoalModel())
    a = p.act({0: ((5, 5), 1), 1: ((10, 10), 4)}, {0: ForwardModel()}, [0, 1, 2, 3], set())
    assert a in (0, 1, 2, 3)                                  # nothing tried/learned -> returns an action to try


def test_exploits_a_reachable_goal_over_exploring():
    forwards = {0: _mover()}
    g = GoalModel()
    g.observe({0: ((9, 0), 1), 1: ((10, 0), 4)}, 1)
    p = Planner(g)
    a = p.act({0: ((3, 0), 1), 1: ((10, 0), 4)}, forwards, [0, 1, 2, 3], {0, 1, 2, 3})
    assert a == 3                                             # moves the controllable object right toward the goal
