"""The assembled playing agent (tbt.play): the full chain on raw frames -- perceive -> learn the operators (babble)
-> explore to find and learn the goal -> exploit it after a respawn (cross-level transfer). The controlled scene
renders frames (no walls in the way), so this gates the loop end-to-end before a live game. Pure stdlib."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.play import Player  # noqa: E402

MOVES = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}


class GridScene:
    """A controlled frame scene: a 1x1 self (colour 7) on a 20x20 grid, a static landmark L (colour 5), and a goal
    CELL (no marker). Reaching the goal cell raises the score. The goal is near the interior start, so exploration
    finds it without hitting an edge (the wall problem is parked)."""

    def __init__(self, n=20, pos=(10, 10), goal=(10, 7), landmark=(3, 3)):
        self.n, self.pos, self.goal, self.L = n, pos, goal, landmark
        self.score = 0
        self.actions = 0

    def render(self):
        g = [[0] * self.n for _ in range(self.n)]
        g[self.L[1]][self.L[0]] = 5
        g[self.pos[1]][self.pos[0]] = 7
        return g

    def step(self, a):
        dx, dy = MOVES[a]
        self.pos = (min(max(self.pos[0] + dx, 0), self.n - 1), min(max(self.pos[1] + dy, 0), self.n - 1))
        self.actions += 1
        if self.pos == self.goal:
            self.score += 1
        return self.render()


def _drive(player, env, target_score, budget):
    """Step the player on the env until the score reaches `target_score` (the goal frame is perceived & learned by
    the act() that sees the raised score) or the budget is spent. Returns the actions spent this phase."""
    grid = env.render()
    start = env.actions
    while env.actions - start < budget:
        a = player.act(grid, [0, 1, 2, 3], env.score)
        if env.score >= target_score:
            break
        grid = env.step(a)
    return env.actions - start


class _Frame:
    """A frame in the contract Player.run consumes -- the SAME shape arc_run._LiveFrame provides, so passing this
    test means the live adapter's contract is met (offline gate, no API)."""

    def __init__(self, grid, score, level, action_counter, win):
        self.grid, self.score, self.level, self.action_counter, self._win = grid, score, level, action_counter, win
        self.available = [0, 1, 2, 3]

    def is_win(self):
        return self._win


class GridEnv:
    """The controlled scene as a reset/step env returning `_Frame`s -- a single level won by reaching the goal."""

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


def test_player_run_completes_via_the_env_contract():
    """Player.run drives the env through its full contract to a win -- the same contract the live adapter satisfies."""
    out = Player(seed=0).run(GridEnv(), max_steps=150)
    assert out.won and out.levels >= 1 and out.actions < 150


def test_assembled_agent_finds_then_exploits_a_goal():
    env = GridScene()
    p = Player(seed=0)
    p.reset()

    find = _drive(p, env, target_score=1, budget=150)
    assert env.score == 1                                     # explored to the goal and the score rose
    assert {a: p.forward.delta(a) for a in p.forward.actions()} == {       # all four operators LEARNED, not assumed
        0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

    env.pos = (10, 13)                                        # respawn the self elsewhere -> "level 1" (same goal + L)
    p.new_level()
    exploit = _drive(p, env, target_score=2, budget=150)
    assert env.score == 2                                     # reached the goal again
    assert exploit <= 12                                      # ~optimal (6) -- it EXPLOITS the learned goal, no re-search
    assert exploit < find                                     # cross-level transfer: exploiting beats exploring
