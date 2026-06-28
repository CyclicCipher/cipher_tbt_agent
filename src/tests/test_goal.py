"""Goal / value (tbt.goal): learn the rewarding object-configuration from the sparse score (no typed sub-goal), keyed
on the RELATIVE arrangement so it transfers across board position, with object-removal a distinct state for free.
Reuses reward.py for the value. Pure stdlib."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.goal import GoalModel, canonical_state  # noqa: E402


def test_canonical_state_is_translation_invariant_and_identity_sensitive():
    self_a, others_a = (5, 8), [("box", (8, 8)), ("door", (5, 3))]
    self_b, others_b = (15, 18), [("box", (18, 18)), ("door", (15, 13))]   # same arrangement, shifted +10,+10
    assert canonical_state(self_a, others_a) == canonical_state(self_b, others_b)
    swapped = [("door", (8, 8)), ("box", (5, 3))]                          # identities exchanged -> different state
    assert canonical_state(self_a, others_a) != canonical_state(self_a, swapped)
    moved = [("box", (9, 8)), ("door", (5, 3))]                            # box one cell further -> different state
    assert canonical_state(self_a, others_a) != canonical_state(self_a, moved)


def test_goal_is_learned_from_the_score_and_transfers_across_position():
    g = GoalModel()
    g.observe((2, 8), [("box", (8, 8))], 0)                 # wandering, score flat -> not a goal
    g.observe((5, 8), [("box", (8, 8))], 0)
    g.observe((7, 8), [("box", (8, 8))], 1)                 # score rose with the self next to the box -> THE goal
    assert g.is_goal((7, 8), [("box", (8, 8))])            # the exact configuration
    assert g.is_goal((27, 8), [("box", (28, 8))])          # SAME relative arrangement elsewhere -> still the goal
    assert not g.is_goal((3, 8), [("box", (8, 8))])        # a different arrangement -> not the goal


def test_reward_is_higher_at_a_learned_goal_than_elsewhere():
    g = GoalModel()
    g.observe((7, 8), [("box", (8, 8))], 1)                # learn the goal
    goal_r = g.reward((7, 8), [("box", (8, 8))])
    other_r = g.reward((3, 8), [("box", (8, 8))])
    assert goal_r > other_r                                 # extrinsic value lifts the rewarded configuration


def test_object_removal_is_a_distinct_goal_with_no_special_case():
    """A 'clear the object' goal: the score rises when the object is GONE. The absent object is simply missing from
    the tuple, so the board-cleared configuration is its own learned goal -- required-absent for free."""
    g = GoalModel()
    g.observe((4, 8), [("gem", (8, 8))], 0)                 # gem present, no score
    g.observe((8, 8), [], 1)                                # gem gone, score rose -> the cleared board is the goal
    assert g.is_goal((8, 8), [])                            # board-clear recognised
    assert not g.is_goal((8, 8), [("gem", (8, 8))])        # gem still present is NOT the goal
