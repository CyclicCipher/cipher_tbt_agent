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
    assert any(fm.is_action_sensitive() for fm in p.forwards.values())   # the self emerged (pose- or content-sensitive)

    rng = random.Random(0)
    pos, walk = (10, 10), set()
    for _ in range(100):
        dx, dy = MOVES[rng.randrange(4)]
        pos = (_clamp(pos[0] + dx), _clamp(pos[1] + dy))
        walk.add(pos)
    assert len(seen) > len(walk)                              # directed exploration beats a random walk


def test_explained_motion_is_not_a_boundary_but_a_scene_cut_is():
    """The events upgrade (reafference residual): once the operator is learned, even a BIG move is explained (residual
    ~0) so it is not a boundary and is learned cleanly; an unexplained SCENE-CUT has a huge residual -> boundary ->
    EXCLUDED from operator learning, so it cannot corrupt the operator."""
    p = Player(seed=0)
    p.reset()

    def frame(x):
        g = [[0] * 30 for _ in range(30)]
        g[10][x] = 7                                          # a 1-cell mover
        return g

    x = 2
    grid = frame(x)
    for _ in range(7):                                        # the mover jumps +3 in x each step under the only action
        p.act(grid, [3], 0)
        x += 3
        grid = frame(x)
    fm = next(f for f in p.forwards.values() if f.delta(3))
    assert fm.delta(3) == (3, 0)                              # the BIG move is learned cleanly, never flagged a boundary
    before = fm.confidence(3)

    cut = [[5 if (r + c) % 2 == 0 else 0 for c in range(30)] for r in range(30)]   # an unrelated full-frame replacement
    p.act(cut, [3], 0)
    assert fm.delta(3) == (3, 0) and fm.confidence(3) >= before   # the scene-cut did NOT corrupt the operator


class ToggleScene:
    """An ls20-like STATE-CHANGE game: a block that changes COLOUR in place (it never moves) under the actions, plus a
    static landmark. ACTION0 sets it to colour A, ACTION1 to colour B; the score ticks up whenever it is colour B (a
    goal that is a state, not a position). A pose-only agent is blind to this whole class -- nothing translates."""

    def __init__(self):
        self.colour = 3                                       # the block's state (colour A)
        self.score = 0
        self.actions = 0

    def render(self):
        g = [[0] * 20 for _ in range(20)]
        for dx in range(4):
            for dy in range(4):
                g[8 + dy][8 + dx] = self.colour              # a 4x4 block, FIXED position, colour = its state
        for dx in (0, 1):
            for dy in (0, 1):
                g[1 + dy][1 + dx] = 5                         # a static landmark (the configuration anchor)
        return g

    def step(self, a):
        self.actions += 1
        if a == 0:
            self.colour = 3                                   # action 0 -> colour A
        elif a == 1:
            self.colour = 6                                   # action 1 -> colour B (the goal state)
        if self.colour == 6:
            self.score += 1
        return self.render()


def test_solves_an_in_place_state_change_game():
    """The deadly assumption removed: the SAME agent that solves movement games now solves a game where the controllable
    object changes COLOUR in place (nothing translates). The content operator learns 'ACTION1 turns the block to the goal
    colour', the goal is a state configuration, and the planner reaches and HOLDS it -- the operator KIND emerged."""
    env = ToggleScene()
    p = Player(seed=0)
    p.reset()
    grid = env.render()
    for _ in range(60):
        grid = env.step(p.act(grid, [0, 1, 2, 3], env.score))
    assert any(fm.is_action_sensitive() for fm in p.forwards.values())   # the block emerged controllable via CONTENT
    assert env.colour == 6                                               # the agent reached AND holds the goal colour
    assert env.score >= 5                                                # held the state, not a one-off accident


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
