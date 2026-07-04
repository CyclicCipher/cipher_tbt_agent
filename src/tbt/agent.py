"""The agent — the sensorimotor discovery/exploit loop over ONE column. Thin by construction.

The location is the pose belief (a grid-cell SDR), value is successor features over it (`col.sf`), navigation is the
goal-oriented vector field (`col.navigate_vector`), and exploration is GOAL BABBLING (sample a target, reach it, learn —
`DISCOVERY.md`). Each turn: read the pose position, learn the SF value from the transition, and choose an action by the
priority

    exploit (a remembered reward goal) > cued (a salient target) > babble (a sampled target),

with a motor-babbling cold start (random actions until the operators are learned, so `navigate_vector` has displacements
to steer by). The model persists across levels; `new_episode()` resets only the per-episode belief linkage. Pure stdlib.
"""

from __future__ import annotations

import random


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class Agent:
    """One column, driven by the discovery/exploit loop. `step(...)` reads the pose the sensor path-integrated this turn,
    learns the SF value, and returns the next action. `complete()` remembers a reached goal (exploit target for later
    levels). `_integrate` (set by the policy) is kept for the sensory contract but the loop is position-based throughout."""

    def __init__(self, n_actions: int, n_entities: int = 256, seed: int = 0, board: int = 64, **_):
        from .column import CorticalColumn                                     # lazy: keep the import graph shallow
        self.actions = list(range(n_actions))
        self.rng = random.Random(seed)
        self.board = board
        self.col = CorticalColumn(n_entities=n_entities, seed=seed, board=board)
        self._integrate = False                                               # position mode (set by the policy); the loop is position-based regardless
        self._goal_pos_raw = None                                             # a remembered reward goal (exploit target) — persists across levels (transfer)
        self.new_episode()

    def new_episode(self):
        self._here = None                                                     # the current pose position (col.here_position())
        self._here_prev = None                                                # the previous pose position (the SF value transition)
        self._salient = None                                                  # a salient target this turn (cued discovery)
        self._babble_target = None                                            # the current goal-babbling target (resampled on arrival)
        self.goal = None                                                      # (kind, target) chosen this turn — for introspection
        self.surprised = False
        self.sensed_surprise = False

    def complete(self, score_delta: float = 1.0):
        """A level completed: remember the goal POSITION and reward it in the SF (a self-loop = a reward SOURCE, so value
        propagates back). On later levels the exploit branch beelines here (transfer)."""
        if self._here is not None:
            self._goal_pos_raw = self._here
            if self._here_prev is not None:
                self.col.learn_location_value(self._here_prev, self._here, 1.0)
            self.col.learn_location_value(self._here, self._here, 1.0)
        self.new_episode()

    def step(self, state=None, score_delta: float = 0.0, frame=None, feature=None, location=None, cloud=None,
             salient=None):
        """One turn: read the pose position the sensor path-integrated (via `col.perceive`), learn the SF value from the
        transition (reward from the score; negative = aversion, the ONE signed value), and choose an action. `state`/
        `feature`/`cloud` are accepted for the sensory contract but the loop reads the pose directly."""
        here, prev = self.col.here_position(), self._here
        if prev is not None and here is not None:                             # LEARN the SF value over the transition (reward from the score; <0 = aversion)
            self.col.learn_location_value(prev, here, float(score_delta) if score_delta < 0 else 0.0)
        self._here_prev, self._here = prev, here
        self._salient = salient
        return self.col.motor(self._choose(here))

    def _choose(self, here):
        """Priority: EXPLOIT a remembered reward goal > test a CUED salient target > goal-BABBLE toward a sampled target.
        Motor-babble (random) until the operators are learned, so `navigate_vector` has displacements to steer by."""
        acts = self.actions
        if here is None or len(self.col.pose_ops) < len(acts):               # cold start: babble until EVERY action's operator is learned (D3)
            self.goal = ("motor_babble", None)
            return self.rng.choice(acts)
        if self._goal_pos_raw is not None:                                    # EXPLOIT: beeline the remembered reward goal
            a = self.col.navigate_vector(here, self._goal_pos_raw, acts)
            self.goal = ("exploit", self._goal_pos_raw)
            if a is not None:
                return a
        if self._salient is not None:                                        # CUED discovery: a salient percept is a candidate goal to test
            a = self.col.navigate_vector(here, self._salient, acts)
            self.goal = ("cued", self._salient)
            if a is not None:
                return a
        if self._babble_target is None or _dist(here, self._babble_target) < 2.0:   # UNCUED: goal babbling — resample on arrival
            self._babble_target = (self.rng.uniform(0, self.board - 1), self.rng.uniform(0, self.board - 1))
        self.goal = ("babble", self._babble_target)
        a = self.col.navigate_vector(here, self._babble_target, acts)
        return a if a is not None else self.rng.choice(acts)
