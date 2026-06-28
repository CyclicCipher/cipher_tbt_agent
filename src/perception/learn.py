"""The world-model LEARNER — perception's online cold-start of the roles from self-directed play.

ARC-AGI-3 hands the agent no goal and no win condition: it must LEARN the world model from the pixels and the
sparse score alone (F's cold-start). This bundles the four canonical learners and re-decodes them into a SHARED
`WorldModel` IN PLACE, so an agent built on that world (its `Perception` + `NeocortexPlanner` both hold the same
reference) sees newly-learned roles live — body, pushable, blocking, the conditional effects (a key opens a
door), and the goal + its required-absent context. It is driven by `Agent.explore_and_learn` (the act-and-learn
play loop); this is where "perception owns its learning" actually lives.

  observe(prev_frame, action, frame) — one transition → the four learners (the ONLY task-format-aware step)
  refresh()                          — re-decode the learners into the shared world (build_world, in place)

The mechanics are LEARNED, not coded: the body is the efference copy, pushable/blocking come from motion, the
door-opening is a residual conditional effect (the column's dynamics faculty, the predicate search that found carry), and
the goal is whatever colour the score rewards — no key/door/pad/goal token anywhere in here.
"""

from __future__ import annotations

from tasks import GameState

from tbt.column import CorticalColumn

from .perceive import DynamicsPerceiver, GoalModel, ObjectPerceiver
from .scene import build_world


class WorldLearner:
    """Accumulate the world model online; expose it as a shared `WorldModel` that the agent's perception + planner
    read live (updated in place by `refresh`). The learners persist across episodes; only per-step/per-level
    perceiver state resets."""

    def __init__(self):
        self.perc = DynamicsPerceiver()
        self.dm = CorticalColumn(n_entities=1)                    # a column whose DYNAMICS faculty models world responses
        self.objp = ObjectPerceiver()
        self.goal = GoalModel()
        self.world = build_world(self.dm, self.objp, self.goal)   # empty until learned — shared by reference

    def observe(self, prev_frame, action, frame):
        """Learn from one transition (prev_frame --action--> frame): the body/effect/context features feed the
        dynamics model, object motion feeds the role learner, and the score feeds the goal model."""
        f, e, present = self.perc.observe(prev_frame, action, frame)
        if f is None:
            return
        self.dm.observe_effect(f, e)
        if action.is_movement and prev_frame.state == GameState.NOT_FINISHED and frame.level == prev_frame.level:
            self.objp.observe(prev_frame.grid, action.delta, frame.grid)
        stepped_on = f[0]
        if e == "score_up":                                       # a level just completed → the goal was reached
            self.goal.observe_win(present, stepped_on)
        elif stepped_on in self.goal.goal_colors:                 # reached the goal colour but did NOT win →
            self.goal.observe_reach_no_win(present)               # something else gates it (the required-absent)

    def refresh(self):
        """Re-decode the learners into the shared world, IN PLACE — so the agent's perception/planner (holding the
        same `world` reference) immediately plan with any newly-learned role."""
        self.dm.learn_dynamics()
        self.world.__dict__.update(build_world(self.dm, self.objp, self.goal).__dict__)

    def new_level(self):
        self.perc.new_level()
        self.objp.new_level()

    def reset(self):
        """New episode: clear the per-step perceiver state, KEEP the learned roles (they accrue across episodes)."""
        self.perc.reset()
