"""The generalization that matters (the user's question): does the agent LEARN that barriers must be routed around as
a RULE -- predicting a never-bumped instance blocks via RECOGNITION -- or only memorize each barrier's location? A nav
game places a recognizable bar (fixed shape -> one recognised identity) blocking the straight path at a DIFFERENT row
each level, so NO fixed route pre-avoids it: only recognising the bar generalises the avoidance. With the object-
behaviour faculty the agent bumps a bar a few times to LEARN the type, then routes around later instances WITHOUT
bumping; the baseline (barriers off) must re-discover each instance by bumping -- so it bumps ~twice as much."""

from __future__ import annotations

import os
import statistics
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from arc_sdk import TbtPolicy  # noqa: E402


class _St:
    def __init__(self, name):
        self.name = name


class BarrierNavGame:
    """Duck-typed raw frame. A 1-cell mover starts at the left and must reach the RIGHT edge; a fixed-shape vertical
    BAR (colour 5) blocks the straight path and must be gone around. The bar sits at a DIFFERENT row each level (same
    shape -> the same recognised object), so a fixed detour route can't pre-avoid it -- only recognising it as a barrier
    generalises. `bumps` counts moves into a bar cell (blocked)."""

    N = 14
    HALF = 2                                                 # bar half-height (a 5-cell bar -> a real detour)
    MOVES = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}
    ROWS = [5, 8, 6, 7, 5, 8]

    def __init__(self, levels=6):
        self.n_levels = levels
        self.levels_completed = 0
        self.state = _St("NOT_PLAYED")
        self.actions_taken = 0
        self.bumps = 0
        self._set()

    def _set(self):
        self.row = self.ROWS[self.levels_completed % len(self.ROWS)]
        self.ax, self.ay = 1, self.row
        self.wall = {(7, self.row + dy) for dy in range(-self.HALF, self.HALF + 1)}

    @property
    def frame(self):
        g = [[0] * self.N for _ in range(self.N)]
        for (x, y) in self.wall:
            g[y][x] = 5
        g[self.ay][self.ax] = 7
        return [g]

    @property
    def available(self):
        return ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    def step(self, name, data=None):
        if name == "RESET":
            self.state = _St("NOT_FINISHED"); self._set()
            return self
        if self.state.name in ("WIN", "NOT_PLAYED"):
            return self
        self.actions_taken += 1
        dx, dy = self.MOVES.get(name, (0, 0))
        tx, ty = self.ax + dx, self.ay + dy
        if (tx, ty) in self.wall:
            self.bumps += 1                                  # tried to move into the bar -> blocked (stayed)
        elif 0 <= tx < self.N and 0 <= ty < self.N:
            self.ax, self.ay = tx, ty
        if self.ax >= self.N - 2:                            # reached the right edge
            self.levels_completed += 1
            if self.levels_completed >= self.n_levels:
                self.state = _St("WIN")
            else:
                self._set()
        return self


def _run(policy, game, budget=10000):
    frame = game
    for _ in range(budget):
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        frame = game.step(name, coords)
    return game.levels_completed, game.bumps


def test_barrier_avoidance_generalises_across_instances():
    """Across seeds: both agents solve every level, but the object-behaviour agent bumps FAR fewer bars -- it learned
    the barrier as a recognised TYPE and routed around new-row instances without re-bumping, while the baseline had to
    re-discover each by bumping. This is the LEARNED, generalising avoidance (not hardcoded, not per-location memory)."""
    seeds = (0, 1, 2)
    on = [_run(TbtPolicy(seed=s, local=True, integrate=True, barriers=True), BarrierNavGame(6)) for s in seeds]
    off = [_run(TbtPolicy(seed=s, local=True, integrate=True, barriers=False), BarrierNavGame(6)) for s in seeds]
    assert all(lv == 6 for lv, _ in on), f"barriers-on did not solve all levels: {[lv for lv, _ in on]}"
    assert all(lv == 6 for lv, _ in off), f"baseline did not solve all levels: {[lv for lv, _ in off]}"
    mean_on = statistics.mean(b for _, b in on)
    mean_off = statistics.mean(b for _, b in off)
    assert mean_on < 0.7 * mean_off, f"no generalization: barriers-on bumps {mean_on:.1f} vs baseline {mean_off:.1f}"
