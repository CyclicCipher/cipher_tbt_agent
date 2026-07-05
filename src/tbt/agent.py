"""The agent — the sensorimotor discovery/exploit loop over ONE column. Thin by construction.

The location is the pose belief (a grid-cell SDR), value is successor features over it (`col.sf`), navigation is the
goal-oriented vector field (`col.navigate_vector`), and exploration is GOAL BABBLING (sample a target, reach it, learn —
`DISCOVERY.md`). Each turn: read the pose position, learn the SF value from the transition, and choose an action by the
priority

    exploit (a remembered reward goal) > cued (a salient target) > babble (a sampled target),

with a motor-babbling cold start (random actions until each action's effect is learned — the hippocampus's gain field
generalises one observation to every heading, so no per-heading coverage sweep is needed). The model persists across levels;
`new_episode()` resets only the per-episode belief linkage. Pure stdlib.
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
        self._seen_lo = None                                                   # per-axis min/max of VISITED positions — bounds babble to the reachable region
        self._seen_hi = None                                                   # (the board is unknown a-priori; sampling 0..board wastes targets off the playable area)
        self.new_episode()

    def new_episode(self):
        self._here = None                                                     # the current pose position (col.here_position())
        self._here_prev = None                                                # the previous pose position (the SF value transition)
        self._salient = None                                                  # a salient target this turn (cued discovery)
        self._cued = None                                                     # the COMMITTED cued target — held until reached or stuck (the cue's salience flickers off when the body is adjacent)
        self._babble_target = None                                            # the current goal-babbling target (resampled on arrival/stall)
        self._stuck = 0                                                        # consecutive no-progress steps (a target we can't approach — abandon it)
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
        self._stuck = self._stuck + 1 if (prev is not None and here is not None and _dist(prev, here) < 0.5) else 0
        if here is not None:                                                  # track the reachable region (for bounded babble)
            if self._seen_lo is None:
                self._seen_lo, self._seen_hi = [here[0], here[1]], [here[0], here[1]]
            else:
                self._seen_lo = [min(self._seen_lo[0], here[0]), min(self._seen_lo[1], here[1])]
                self._seen_hi = [max(self._seen_hi[0], here[0]), max(self._seen_hi[1], here[1])]
        self._here_prev, self._here = prev, here
        self._salient = salient
        return self.col.motor(self._choose(here))

    def _sample_babble(self, margin: float = 6.0):
        """A goal-babbling target inside the VISITED region grown by `margin` (clamped to the board) — so targets stay
        REACHABLE (not stranded off the playable area) while the margin pushes the frontier outward, covering the board."""
        lo = self._seen_lo or [0.0, 0.0]
        hi = self._seen_hi or [self.board - 1, self.board - 1]
        return tuple(self.rng.uniform(max(0.0, lo[i] - margin), min(self.board - 1, hi[i] + margin)) for i in range(2))

    def _learned(self, a) -> bool:
        """Has action `a`'s effect been learned — a nonzero MOVE or TURN (L5's efference)? The gain field generalises ONE
        forward observation to every heading, so cold-start ends as soon as every action has shown its effect (no
        per-heading sweep). An action seen only while blocked (zero effect) keeps babbling until it is seen acting."""
        return self.col.L5.learned(a)

    def _choose(self, here):
        """Priority: EXPLOIT a remembered reward goal > test a CUED salient target > goal-BABBLE toward a sampled target.
        Cold start = motor babble (random) until every action's effect is learned, so `navigate_vector` (the hippocampus's
        inverse gain field) has displacements + turns to steer by."""
        acts = self.actions
        if here is None or not all(self._learned(a) for a in acts):
            self.goal = ("motor_babble", None)
            return self.rng.choice(acts)
        if self._goal_pos_raw is not None:                                    # EXPLOIT: beeline the remembered reward goal
            a = self.col.navigate_vector(here, self._goal_pos_raw, acts)
            self.goal = ("exploit", self._goal_pos_raw)
            if a is not None:
                return a
        if self._salient is not None:                                        # a salient percept this turn → COMMIT to it (the cue flickers off when the body is adjacent — hold it)
            self._cued = self._salient
        if self._cued is not None:                                           # CUED discovery: navigate to the committed target ("hold your horses")
            if _dist(here, self._cued) < 2.0 or self._stuck >= 4:            # reached it (the game scores here if it was the goal), or can't approach → release to re-acquire
                self._cued = None
            else:
                a = self.col.navigate_vector(here, self._cued, acts)
                if a is not None:
                    self.goal = ("cued", self._cued)
                    return a
                self._cued = None                                            # navigator gave up → release
        if self._babble_target is None or _dist(here, self._babble_target) < 2.0 or self._stuck >= 4:   # UNCUED goal babbling — resample on arrival OR when STALLED (a target we can't approach, e.g. past a wall)
            self._babble_target = self._sample_babble()
            self._stuck = 0
        self.goal = ("babble", self._babble_target)
        a = self.col.navigate_vector(here, self._babble_target, acts)
        return a if a is not None else self.rng.choice(acts)
