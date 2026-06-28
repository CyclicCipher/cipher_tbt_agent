"""Goal / value (tbt.goal): learn the rewarding object CONFIGURATION from the sparse score -- no typed sub-goal and
no privileged self. Keyed on the relative arrangement (anchored on the largest object), so it transfers across board
position; object-removal is a distinct state for free. Reuses reward.py. Pure stdlib."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.goal import GoalModel, config_state  # noqa: E402


def test_config_state_translation_invariant_and_distinguishing():
    a = {0: ((5, 8), 1), 1: ((10, 8), 4)}
    b = {0: ((15, 18), 1), 1: ((20, 18), 4)}                  # same arrangement, shifted +10,+10
    assert config_state(a) == config_state(b)
    assert config_state(a) != config_state({0: ((6, 8), 1), 1: ((10, 8), 4)})   # an object moved one cell
    assert config_state(a) != config_state({0: ((5, 8), 2), 1: ((10, 8), 4)})   # identity (size) changed
    # left vs right of the anchor must differ -- the bug a mean-anchor + banker's rounding would hide
    assert config_state({0: ((9, 8), 1), 1: ((10, 8), 4)}) != config_state({0: ((11, 8), 1), 1: ((10, 8), 4)})


def test_goal_learned_from_the_score_and_transfers_across_position():
    g = GoalModel()
    g.observe({0: ((2, 8), 1), 1: ((10, 8), 4)}, 0)          # wandering, score flat
    goal = {0: ((9, 8), 1), 1: ((10, 8), 4)}                 # score rose with this arrangement -> THE goal
    g.observe(goal, 1)
    assert g.is_goal(goal)
    assert g.is_goal({0: ((29, 8), 1), 1: ((30, 8), 4)})     # SAME relative arrangement elsewhere
    assert not g.is_goal({0: ((3, 8), 1), 1: ((10, 8), 4)})  # a different arrangement


def test_object_removal_is_a_distinct_goal_with_no_special_case():
    g = GoalModel()
    g.observe({0: ((4, 8), 1), 1: ((8, 8), 4)}, 0)           # both present
    g.observe({0: ((8, 8), 1)}, 1)                           # the big object gone, score rose
    assert g.is_goal({0: ((8, 8), 1)})                       # the cleared board is the goal
    assert not g.is_goal({0: ((8, 8), 1), 1: ((12, 8), 4)})  # still-present is not


def test_reward_is_higher_at_a_learned_goal():
    g = GoalModel()
    goal = {0: ((9, 8), 1), 1: ((10, 8), 4)}
    g.observe(goal, 1)
    assert g.reward(goal) > g.reward({0: ((3, 8), 1), 1: ((10, 8), 4)})
