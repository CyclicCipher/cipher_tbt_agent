"""The assembled playing agent (tbt.play), self-free: the full chain on raw frames -- perceive objects (no self) ->
learn each object's operator (the controllable one EMERGES) -> explore (directed) -> reach a goal. The controlled
scene renders frames; this gates the loop end-to-end before a live game. (Finding one specific unmarked cell within
budget is the known, parked exploration-efficiency limit, so the goal here is a reliably-reachable region.) Pure stdlib."""

from __future__ import annotations

import os
import random
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.play import Player  # noqa: E402

MOVES = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
N = 20


def _clamp(v):
    return min(max(v, 0), N - 1)


class GridScene:
    """A 1-cell controllable object (colour 7), a 2x2 static landmark (colour 5, the config anchor), no self labelled.
    Reaching the OUTER RING scores -- a goal outward exploration reliably reaches."""

    def __init__(self, pos=(10, 10), landmark=(3, 3)):
        self.pos, self.L = pos, landmark
        self.score = 0
        self.actions = 0

    def render(self):
        g = [[0] * N for _ in range(N)]
        lx, ly = self.L
        for dx in (0, 1):
            for dy in (0, 1):
                g[ly + dy][lx + dx] = 5
        g[self.pos[1]][self.pos[0]] = 7
        return g

    def step(self, a):
        dx, dy = MOVES[a]
        self.pos = (_clamp(self.pos[0] + dx), _clamp(self.pos[1] + dy))
        self.actions += 1
        x, y = self.pos
        if x <= 1 or x >= N - 2 or y <= 1 or y >= N - 2:
            self.score += 1
        return self.render()


def test_loop_learns_operators_and_explores_on_frames():
    """End-to-end on rendered frames: the controllable object EMERGES (some tracked object learns an action-sensitive
    operator) and directed exploration covers more than a random walk -- with no self and no goal."""
    env = GridScene()
    p = Player(seed=0)
    p.reset()
    grid = env.render()
    seen = set()
    for _ in range(100):
        grid = env.step(p.act(grid, [0, 1, 2, 3], 0))         # score held 0 -> pure exploration, no goal
        seen.add(env.pos)
    assert any(len({fm.delta(a) for a in fm.actions()}) >= 2 for fm in p.forwards.values())   # the self emerged

    rng = random.Random(0)
    pos, walk = (10, 10), set()
    for _ in range(100):
        dx, dy = MOVES[rng.randrange(4)]
        pos = (_clamp(pos[0] + dx), _clamp(pos[1] + dy))
        walk.add(pos)
    assert len(seen) > len(walk)                              # directed exploration beats a random walk


class _Frame:
    """A frame in the contract Player.run consumes -- the same shape arc_run._LiveFrame provides (the live-adapter gate)."""

    def __init__(self, grid, score, level, action_counter, win):
        self.grid, self.score, self.level, self.action_counter, self._win = grid, score, level, action_counter, win
        self.available = [0, 1, 2, 3]

    def is_win(self):
        return self._win


class GridEnv:
    def __init__(self):
        self.scene = GridScene()

    def _frame(self):
        s = self.scene
        return _Frame(s.render(), s.score, s.score, s.actions, s.score >= 1)

    def reset(self):
        self.scene = GridScene()
        return self._frame()

    def step(self, action):
        self.scene.step(action)
        return self._frame()


def test_player_run_reaches_a_goal_via_the_env_contract():
    """Player.run drives the env through its full contract to a win -- the same contract the live adapter satisfies."""
    out = Player(seed=0).run(GridEnv(), max_steps=120)
    assert out.won and out.levels >= 1 and out.actions < 120
