"""Path integration (step 7c): the egocentric view recurs but has NO allocentric position, so distinct board
locations ALIAS and the agent gets stuck (the live wall after 7b). Tracking the controllable object's path-integrated
position (efference + correct) separates the aliased views and lets novelty drive COVERAGE + navigation. Gated offline
by a NAVIGATION game whose goal is FAR (pure local sensing can't navigate there): path-integration solves it, pure
egocentric does not. A non-controllable (state-change) scene keeps the gate OFF, so its recurring local view survives."""

from __future__ import annotations

import os
import random
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from arc_sdk import TbtPolicy  # noqa: E402
from tbt.sensor import Sensor  # noqa: E402


class _St:
    def __init__(self, name):
        self.name = name


class NavGame:
    """Duck-typed like the raw frame. A 2x2 mover on a 24x24 board must reach a goal in the OPEN INTERIOR (no marker,
    far from any edge) -- so the egocentric window is uniform there and ALIASES with everywhere else; only allocentric
    POSITION can navigate to it. Moves are 2 cells. Reaching the goal completes the level; advances in place; last ->
    WIN. (A goal at an edge/corner would be solvable by local sensing alone -- the window sees the out-of-bounds edge.)"""

    N = 24
    STEP = 2
    GOAL = (12, 12)
    MOVES = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}

    def __init__(self, levels=8):
        self.n_levels = levels
        self.levels_completed = 0
        self.state = _St("NOT_PLAYED")
        self.actions_taken = 0
        self.mx, self.my = 0, 0

    @property
    def frame(self):
        g = [[0] * self.N for _ in range(self.N)]
        for dx in (0, 1):
            for dy in (0, 1):
                g[self.my + dy][self.mx + dx] = 7            # the 2x2 mover (the only object -- the goal is invisible)
        return [g]

    @property
    def available(self):
        return ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    def step(self, name, data=None):
        if name == "RESET":
            self.state = _St("NOT_FINISHED"); self.mx, self.my = 0, 0
            return self
        if self.state.name in ("WIN", "NOT_PLAYED"):
            return self
        self.actions_taken += 1
        dx, dy = self.MOVES.get(name, (0, 0))
        self.mx = min(max(self.mx + dx * self.STEP, 0), self.N - 2)
        self.my = min(max(self.my + dy * self.STEP, 0), self.N - 2)
        if abs(self.mx - self.GOAL[0]) <= 1 and abs(self.my - self.GOAL[1]) <= 1:   # reached the interior goal
            self.levels_completed += 1
            if self.levels_completed >= self.n_levels:
                self.state = _St("WIN")
            else:
                self.mx, self.my = 0, 0                       # next level, in place
        return self


def _drive(policy, game, budget):
    frame = game
    for _ in range(budget):
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        frame = game.step(name, coords)
    return game.levels_completed


def test_path_integration_navigates_where_pure_egocentric_stalls():
    """With path integration the agent covers the board and reaches the far goal repeatedly; pure egocentric sensing
    (no position) aliases the uniform interior and reaches it far less often."""
    nav = _drive(TbtPolicy(seed=0, local=True, integrate=True), NavGame(8), budget=3000)
    ego = _drive(TbtPolicy(seed=0, local=True, integrate=False), NavGame(8), budget=3000)
    assert nav >= 6, f"path-integration agent only completed {nav}/8 levels"
    assert nav >= 3 * max(ego, 1) and ego <= 2, f"path-integration {nav} not far above pure-egocentric {ego}"


def test_gate_stays_off_for_a_non_controllable_scene():
    """A scene whose change is NOT action-driven (in-place colour animation) keeps the position gate OFF -- the
    coarse position stays constant, so the recurring local view (the state-change game's signal) is preserved."""
    s = Sensor(local=True, integrate=True, window=5)
    rng = random.Random(0)
    N = 30
    for t in range(20):
        g = [[0] * N for _ in range(N)]
        for k in range(4):                                   # a fixed 2x2 block toggling colour in place (no movement)
            g[10 + (k // 2)][10 + (k % 2)] = rng.choice([2, 3])
        (_patch, pos), _change = s.read(g, action=(t % 4))
        last_pos = pos
    assert not s._controllable(), f"gate wrongly ON: learned deltas {s._delta}"
    assert last_pos == (0, 0), f"position should be the constant gate-off value, got {last_pos}"
