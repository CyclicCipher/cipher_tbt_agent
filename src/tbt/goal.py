"""Goal and value -- learn WHAT the sparse score rewards as an object CONFIGURATION, with no privileged self.

ARC-AGI-3 gives no goal, only a score that ticks up on level completion. The goal must be LEARNED, and -- the bitter
lesson -- it is neither a typed sub-goal nor a self-relative target; it is simply *the configuration of objects on
screen when the score rose*. This module encodes a scene of objects as a position-CANONICAL state -- each object's
`(size, pose relative to the configuration's centroid)`, rounded and sorted -- so the same RELATIVE arrangement is one
state wherever it sits on the board (the relational encoding that generalises, reward.py's ValueLearner lesson), and
ties the score to it through the existing `RewardModel`.

There is no "self" here: the state is over ALL objects, anchored on the whole configuration, not on a controllable
agent -- so it works even in a task with no controllable object, and the controllable one (when there is one) is just
one term in the tuple. Identity is the object's SIZE (a shape proxy, instance-invariant), never its colour. A removed
object is simply absent from the tuple, so a "required-absent" goal needs no special case. DOMAIN-GENERAL (the
Danganronpa litmus); the value machinery is reused from `reward.py`. Pure stdlib.
"""

from __future__ import annotations

import math

from .reward import RewardModel


def _r(v) -> int:
    """Round half UP (monotonic) -- not Python's banker's `round`, whose half-to-even aliases adjacent half-integer
    positions (the killer when poses are object centroids)."""
    return int(math.floor(v + 0.5))


def config_state(objects):
    """The configuration of `objects = {id: (pose, size)}` as a hashable, translation-invariant state: each object's
    `(size, integer pose relative to the LARGEST object)`, sorted. The same arrangement elsewhere on the board yields
    the SAME state. The anchor is the largest object -- an emergent, stable reference (the big static structure on a
    real frame), NOT a privileged self, and NOT the configuration mean (whose symmetry can't tell left from right for
    two objects)."""
    items = list(objects.values())
    if not items:
        return ()
    (ax, ay), _ = max(items, key=lambda it: (it[1], it[0]))   # the largest object (ties by pose) is the reference
    return tuple(sorted((size, (_r(pose[0] - ax), _r(pose[1] - ay))) for pose, size in items))


class GoalModel:
    """Learn the rewarding object-configuration from the sparse score, over canonical (self-free) states, and value a
    configuration the planner is considering. Wraps `reward.py`'s `RewardModel` and adds only the scene->state
    encoding."""

    def __init__(self, **rmkw):
        self.rm = RewardModel(1, optimistic=False, **rmkw)
        self.goals: set = set()                              # the canonical configurations a score increment rewarded

    def state(self, objects):
        return config_state(objects)

    def observe(self, objects, score_delta) -> tuple:
        """Record the current configuration; if the score rose, mark it the goal."""
        s = self.state(objects)
        self.rm.observe(s, score_delta)
        if score_delta > 0:
            self.goals.add(s)
        return s

    def is_goal(self, objects) -> bool:
        """Is this configuration one the score has rewarded? (Recognised by its relative arrangement.)"""
        return self.state(objects) in self.goals

    def reward(self, objects) -> float:
        """Unified value of reaching this configuration: extrinsic (1.0 at a learned goal) plus the novelty bonus."""
        return self.rm.reward_total(self.state(objects))

    def visits(self, objects) -> int:
        """How many times the agent has really been in this configuration -- the epistemic/novelty drive."""
        return self.rm.visits.get(self.state(objects), 0)
