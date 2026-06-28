"""The playing agent -- the continuous online loop that assembles the rebuild into one agent.

Thin by construction (it wires faculties, it does not solve): each step it PERCEIVES the scene (`perceive.py`),
LEARNS from the transition just taken (the self's operator via `forward.py`, the goal via `goal.py` from the score),
and PLANS the next action (`plan.py`: babble → explore → exploit, all one value). The learned models PERSIST across
levels (cross-level transfer -- a goal learned on level 0 is exploited on level 1); only per-level position/tracking
state resets. This is the successor to the old `agent.py` + `perception/` harness, built on the new front-end; the old
path is dissolved (stage 5) once this plays a real game.

One continuous interaction, no episodes (the RHAE action budget rules out `explore_and_learn`'s many random episodes).
Generic over an env exposing `reset()/step(action)` and a frame with `grid / score / level / available / is_win() /
action_counter`, so the same loop runs the controlled scene and (through an adapter) the live games. Pure stdlib.
"""

from __future__ import annotations

import random

from .agent import Outcome
from .events import EventSegmenter
from .forward import ForwardModel
from .goal import GoalModel
from .perceive import ScenePerceiver
from .plan import Planner
from .retina import salient_cells


class Player:
    """The assembled agent. `act(grid, actions, score)` returns the next action; `run(env)` drives it to a win or the
    action budget. `reset()` is a new GAME (fresh learner -- no cross-game transfer, which avoids negative transfer);
    `new_level()` keeps the learned models (transfer) and resets only per-level tracking."""

    def __init__(self, cap: int = 600, gamma: float = 0.95, novelty: float = 0.05, seed: int = 0):
        self.cap, self.gamma, self.novelty, self.seed = cap, gamma, novelty, seed
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)
        self.perceiver = ScenePerceiver()
        self.forward = ForwardModel()
        self.goal = GoalModel()
        self.events = EventSegmenter()
        self.planner = Planner(self.forward, self.goal, cap=self.cap, gamma=self.gamma,
                               novelty=self.novelty, seed=self.seed)
        self._prev = self._last = self._prev_self = None
        self._prev_score = 0

    def new_level(self):
        """A level cleared: keep the learned forward/goal (and the known self identity) for transfer; re-localise."""
        self.planner.reset()
        self.events = EventSegmenter()
        self._prev = self._last = self._prev_self = None

    def act(self, grid, actions, score):
        """Perceive the current scene, learn from the transition into it, and plan the next action."""
        self_pose, others = self.perceiver.perceive(self._prev, self._last, grid)
        if self._prev is not None and self_pose is not None:
            boundary = self.events.is_boundary(len(salient_cells(self._prev, grid)))
            if not boundary and self._prev_self is not None:
                self.forward.observe(self._prev_self, self._last, self_pose)   # the self's operator (learned)
            self.goal.observe(self_pose, others, score - self._prev_score)     # the goal, if the score rose
        if self_pose is None:                                                  # self not yet located -> babble to move
            action = self.rng.choice(list(actions))
        else:
            action = self.planner.act(self_pose, others, actions)
        self._prev, self._last, self._prev_self, self._prev_score = grid, action, self_pose, score
        return action

    def run(self, env, max_steps: int = 2000):
        self.reset()
        frame = env.reset()
        while frame.action_counter < max_steps and not frame.is_win():
            action = self.act(frame.grid, frame.available, frame.score)
            nxt = env.step(action)
            if nxt.level != frame.level:                                       # level cleared -> re-localise, keep models
                self.new_level()
            frame = nxt
        return Outcome(won=frame.is_win(), levels=frame.score, actions=frame.action_counter)
