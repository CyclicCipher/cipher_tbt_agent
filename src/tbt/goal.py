"""Goal and value -- learn WHAT the sparse score rewards, as an object-configuration, then value any configuration.

ARC-AGI-3 gives no goal, only a score that ticks up when a level is completed. The goal must be LEARNED, and -- the
bitter lesson -- it cannot be a typed sub-goal (cover / collect / fire / toggle); it is simply *the configuration of
objects on screen when the score rose*. This module is the bridge from the recognised scene to `reward.py`:

  * it encodes a scene as a position-CANONICAL state -- each object's pose RELATIVE to the controllable self, rounded
    to the cell grid, sorted -- so the same relative arrangement recurs wherever on the board it happens. That
    relational encoding is what generalises (reward.py's ValueLearner: features that CHANGE with the action, like the
    gap between the self and a target, generalise; absolute cell positions do not);
  * it ties the score to that state through the existing `RewardModel`: a score increment marks the reached
    configuration rewarding, value then propagates over the state by prioritized sweeping (in the planner, step 3).

A removed object is simply absent from the tuple, so a "required-absent" goal (clear the board of a colour) is a
distinct state with no special case -- the bitter lesson at the goal layer. DOMAIN-GENERAL (the Danganronpa litmus):
it consumes `(self pose, [(identity, pose), ...], score_delta)` and nothing about colours, a grid, or a mechanic. The
value machinery is reused from `reward.py`, never reimplemented. Pure stdlib.
"""

from __future__ import annotations

from .reward import RewardModel


def canonical_state(self_pose, others):
    """The object-configuration anchored on the controllable self: each other object's `(identity, integer pose
    relative to the self)`, sorted -> a hashable, translation-invariant state. The same arrangement elsewhere on the
    board yields the SAME state, so a goal learned once transfers across position. With no self (`self_pose=None`,
    e.g. a click game where absolute position IS the content) it falls back to absolute poses."""
    ax, ay = self_pose if self_pose is not None else (0.0, 0.0)
    return tuple(sorted((ident, (round(px - ax), round(py - ay))) for ident, (px, py) in others))


class GoalModel:
    """Learn the rewarding object-configuration from the sparse score, over canonical (self-anchored) states, and
    value a configuration the planner is considering. Wraps `reward.py`'s `RewardModel` (its `rm` attribute, which
    the planner drives with transitions from the forward model) and adds only the scene->state encoding."""

    def __init__(self, **rmkw):
        # N only sizes the (unused) VI sweep count; 0-init value (optimistic=False) is what reward.py says to use
        # when PLANNING over an enumerated configuration set, so undecayed optimism cannot outvalue real progress.
        self.rm = RewardModel(1, optimistic=False, **rmkw)
        self.goals: set = set()                              # the canonical configurations a score increment rewarded

    def state(self, self_pose, others):
        """The canonical, self-anchored state of a scene -- the key everything else is keyed on."""
        return canonical_state(self_pose, others)

    def observe(self, self_pose, others, score_delta) -> tuple:
        """One step: record the current configuration and, if the score rose, mark it the goal (reward.py infers the
        reward at the reached state). Returns the canonical state observed."""
        s = self.state(self_pose, others)
        self.rm.observe(s, score_delta)
        if score_delta > 0:
            self.goals.add(s)
        return s

    def is_goal(self, self_pose, others) -> bool:
        """Is this configuration one the score has rewarded? (Recognised by its RELATIVE arrangement, so it fires at
        a position never seen at learning time.)"""
        return self.state(self_pose, others) in self.goals

    def reward(self, self_pose, others) -> float:
        """The unified value of reaching this configuration: extrinsic (1.0 at a learned goal) plus the intrinsic
        novelty bonus (reward.py keeps exploration and exploitation in ONE value, no explore/exploit switch)."""
        return self.rm.reward_total(self.state(self_pose, others))
