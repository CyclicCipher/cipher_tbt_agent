"""The planner -- choose the action by rolling the LEARNED forward model toward the LEARNED goal (active inference).

This is the harness dissolved. `perception/control.py`'s `NeocortexPlanner` reaches the same achiever, but only
after a fat role-aware glue layer: it decodes the scene into a `_Layout` (body / pushable / blocking / door colours,
from a decoded role schema), assumes a hand-coded 4-move `DELTAS` geometry, and builds an SR-frame `CorticalColumn`
over the enumerated navigable cells. None of that survives the rebuild's findings -- real games do not announce roles,
and the action set and its effects are LEARNED, not given.

So this planner keeps only the general core and feeds it learned parts:
  * the transition is `forward.predict` -- the per-action operators learned by `forward.py`, so the action set AND
    its geometry come from what was observed, never a `DELTAS` table;
  * the value is `goal.py` -- a score-rewarded object-configuration, so there is no role schema and no typed sub-goal;
  * the search is the existing `Neocortex.achieve` (reward.py's prioritized replay over signed value) -- reused
    whole, not reimplemented. The thalamus + basal ganglia it carries come online in the next step, to gate WHICH
    object the rollout factors on when a scene has several movers; here one controllable self navigates.

`act` rolls the self's operators from its current pose (others held fixed -- object interaction / contact is the
next step), values the reachable configurations by their learned goal-reward, and returns the operator that closes
the gap. Domain-general (the Danganronpa litmus): it consumes `(self pose, [(identity, pose), ...])` and nothing
about a grid, colours, or a mechanic. Pure stdlib over the reused achiever.
"""

from __future__ import annotations

from .neocortex import Neocortex


class Planner:
    """Active-inference action selection over the learned forward model + goal. Hold the (mutable, still-learning)
    `forward` and `goal` models; `act(self_pose, others)` returns the action key to take. `cap` bounds the rollout
    frontier so an unbounded board stays tractable (per-step re-planning then walks the goal in as progress is made,
    exactly as the achiever's `max_states` is used for Tetris)."""

    def __init__(self, forward, goal, cap: int = 600, gamma: float = 0.95, seed: int = 0):
        self.forward = forward
        self.goal = goal
        self.cap = cap
        self.neo = Neocortex(gamma=gamma, seed=seed)

    def reset(self):
        self.neo.reset()

    def act(self, self_pose, others, actions=None):
        """The operator to take from `self_pose` given the other objects `others = [(identity, pose), ...]`. Rolls
        each learned operator forward (others fixed), values the reached configurations by the learned goal, and
        returns the greedy operator key. With nothing learned to move toward, the achiever finds no terminal and
        returns an arbitrary valid action (the cold-start wander). Returns None only if no operator is known yet."""
        actions = list(actions) if actions is not None else sorted(self.forward.actions(), key=str)
        if not actions:
            return None
        others = [(ident, (round(px), round(py))) for ident, (px, py) in others]
        start = (round(self_pose[0]), round(self_pose[1]))
        forward, goal = self.forward, self.goal

        def step(state, a):
            nxt = forward.predict(state, actions[a])           # the LEARNED operator -- no DELTAS table
            done = goal.is_goal(nxt, others)                   # the LEARNED goal-configuration -- no role schema
            return nxt, (1.0 if done else 0.0), done

        return actions[self.neo.achieve(step, start, len(actions), max_states=self.cap)]
