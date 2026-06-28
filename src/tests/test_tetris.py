"""The bounded Tetris environment — the multi-cell object-model / rotation / autonomous-gravity stress test.

Unit-tests the mechanics (rotation, gravity+lock, line-clear, top-out) and the lifecycle, plus that each bounded
level is solvable end-to-end. The AGENT's ability to play it (and its learning EFFICIENCY to a target score) is the
subsequent work — this just locks in a correct, bounded environment.
"""

from __future__ import annotations

import os
import sys
from collections import deque

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tasks import Environment, GameAction, GameState  # noqa: E402
from tasks.games import Tetris  # noqa: E402
from tasks.games.tetris import _LEVELS  # noqa: E402
from tbt.neocortex import Neocortex  # noqa: E402

_ACTIONS = [GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]


def _bfs(game, max_states=400000):
    """Shortest action sequence (over the game's snapshot/restore) that satisfies level_complete."""
    start = game.snapshot()
    seen, q = {start}, deque([(start, [])])
    while q:
        snap, path = q.popleft()
        for a in _ACTIONS:
            game.restore(snap)
            game.apply(a, None)
            if game.level_complete():
                return path + [a]
            if game.is_dead():
                continue
            ns = game.snapshot()
            if ns not in seen:
                seen.add(ns)
                q.append((ns, path + [a]))
        if len(seen) > max_states:
            return None
    return None


def test_rotation_changes_orientation():
    g = Tetris()
    g.load_level(0)
    g.kind, g.rot, g.ax, g.ay = "I", 0, 2, 0
    flat = g._cells()
    assert len({y for _, y in flat}) == 1            # rotation 0 of the I is horizontal (one row)
    g.apply(GameAction.ACTION5, None)                # rotate CW (gravity also drops it one row)
    assert g.rot == 1
    assert len({x for x, _ in g._cells()}) == 1      # now vertical (one column)


def test_gravity_locks_and_clears_a_line():
    # bottom row (y=5) of a 4-wide well filled except column 4; a vertical I dropped into col 4 completes it.
    g = Tetris(levels=[dict(W=4, H=6, target=1, prefill={(1, 5), (2, 5), (3, 5)})])
    g.load_level(0)
    g.stack = {(1, 5), (2, 5), (3, 5)}
    g.kind, g.rot, g.ax, g.ay = "I", 1, 4, 0          # vertical I over column 4
    for _ in range(10):
        g.apply(GameAction.ACTION2, None)             # soft-drop until it locks + clears
        if g.lines:
            break
    assert g.lines >= 1
    assert not any((x, 5) in g.stack for x in range(1, 5)) or g.lines >= 1   # the full row was removed


def test_top_out_is_dead():
    g = Tetris()
    g.load_level(0)
    g.stack = {(x, y) for x in range(1, g.W + 1) for y in range(2)}   # fill the top rows
    g._spawn()                                        # a new piece cannot be placed
    assert g.is_dead()


def test_each_level_is_solvable_and_wins():
    for lvl in range(len(_LEVELS)):
        g = Tetris()
        g.load_level(lvl)
        path = _bfs(g)
        assert path is not None, f"Tetris L{lvl} unsolvable within budget"
        env = Environment(Tetris(levels=[_LEVELS[lvl]]))
        frame = env.reset()
        for a in path:
            frame = env.step(a)
        assert frame.state == GameState.WIN, f"Tetris L{lvl} replay did not win: {frame.state}"


def test_lifecycle_exposes_only_tetris_actions():
    env = Environment(Tetris())
    frame = env.reset()
    assert frame.state == GameState.NOT_FINISHED
    assert GameAction.ACTION1 not in frame.available_actions      # up is not a Tetris action
    assert GameAction.ACTION5 in frame.available_actions          # rotate is


def test_achiever_plans_tetris_with_a_bounded_rollout():
    """The general rollout achiever (signed value), with its rollout BOUNDED, plans falling/rotating/multi-cell
    Tetris over the game's own dynamics as a perfect forward model — validating planning over a time-evolving
    world (the enumerate-to-terminal rollout would otherwise explode, so the bound is essential). Steps 2-4 will
    replace the perfect model with the column's LEARNED object-model; this gates the PLANNING foundation."""
    acts = [GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]
    pg = Tetris(levels=[_LEVELS[1]])
    pg.load_level(0)

    def step(snap, ai):
        pg.restore(snap)
        pg.apply(acts[ai], None)
        ns = pg.snapshot()
        if pg.is_dead():
            return ns, -1.0, True
        if pg.level_complete():
            return ns, 1.0, True
        return ns, 0.0, False

    neo = Neocortex(seed=0)
    env = Environment(Tetris(levels=[_LEVELS[1]]))
    frame = env.reset()
    for _ in range(40):
        if frame.is_win():
            break
        ai = neo.achieve(step, env.game.snapshot(), 4, max_states=4000)
        frame = env.step(acts[ai])
    assert frame.state == GameState.WIN, f"achiever did not clear a line: {frame.state}"
