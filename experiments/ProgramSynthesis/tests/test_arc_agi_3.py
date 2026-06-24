"""Tests for the ARC-AGI-3 replica harness and the LockPath game.

Run from anywhere:  pytest experiments/ProgramSynthesis/tests/test_arc_agi_3.py
"""

from __future__ import annotations

import os
import sys
from collections import deque
from typing import FrozenSet, List, Optional, Tuple

import pytest

# Make `import arc_agi_3` work regardless of pytest's rootdir.
_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from arc_agi_3 import (  # noqa: E402
    ActionNotAvailable,
    Environment,
    GameAction,
    GameResult,
    GameState,
    GRID_SIZE,
    RandomAgent,
    Scorecard,
    run_episode,
)
from arc_agi_3.games import LockPath  # noqa: E402

DIRECTIONS = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
]


# --- a BFS solver (test-only) over LockPath's internal state ----------------

_Dyn = Tuple[Tuple[int, int], FrozenSet, FrozenSet, bool, bool]


def _capture(game: LockPath) -> _Dyn:
    return (
        game.agent,
        frozenset(game.blocks),
        frozenset(game.keys),
        game.has_key,
        game._dead,
    )


def _restore(game: LockPath, dyn: _Dyn) -> None:
    game.agent = dyn[0]
    game.blocks = set(dyn[1])
    game.keys = set(dyn[2])
    game.has_key = dyn[3]
    game._dead = dyn[4]


def solve_level(game: LockPath) -> Optional[List[GameAction]]:
    """BFS for an action sequence that completes the current level.

    Mutates `game` while searching, then restores it to the level start, so the
    caller can replay the returned path through a real Environment.
    """
    level = game._level
    start = _capture(game)
    seen = {start}
    queue: deque = deque([(start, [])])
    solution = None
    while queue:
        state, path = queue.popleft()
        _restore(game, state)
        if game.level_complete():
            solution = path
            break
        for action in DIRECTIONS:
            _restore(game, state)
            game.apply(action, None)
            if game._dead:
                continue
            nxt = _capture(game)
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [action]))
    game.load_level(level)  # restore to start for replay
    return solution


def _advance_to_level(env: Environment, target: int):
    """Reset and solve levels until the environment is at `target`."""
    frame = env.reset()
    while env.level < target:
        path = solve_level(env.game)
        assert path is not None, f"level {env.level} is unsolvable"
        for action in path:
            frame = env.step(action)
    return frame


# --- lifecycle & gating -----------------------------------------------------


def test_reset_emits_well_formed_frame():
    env = Environment(LockPath())
    frame = env.reset()
    assert frame.state == GameState.NOT_FINISHED
    assert frame.level == 0
    assert frame.score == 0
    assert frame.action_counter == 0
    # Faithful observation: a single 64x64 grid of values 0..15.
    assert len(frame.frame) == 1
    assert len(frame.grid) == GRID_SIZE
    assert all(len(row) == GRID_SIZE for row in frame.grid)
    assert all(0 <= v <= 15 for row in frame.grid for v in row)
    assert GameAction.RESET in frame.available_actions
    assert GameAction.ACTION5 not in frame.available_actions  # LockPath: dirs only


def test_step_before_reset_raises():
    env = Environment(LockPath())
    with pytest.raises(ActionNotAvailable):
        env.step(GameAction.ACTION1)


def test_unavailable_action_raises():
    env = Environment(LockPath())
    env.reset()
    with pytest.raises(ActionNotAvailable):
        env.step(GameAction.ACTION5)  # not exposed by LockPath
    with pytest.raises(ActionNotAvailable):
        env.step(GameAction.ACTION6, (10, 10))  # coordinate action not exposed


def test_action_counter_counts_everything():
    env = Environment(LockPath())
    env.reset()
    env.step(GameAction.ACTION4)
    f = env.step(GameAction.ACTION3)
    assert f.action_counter == 2


# --- mechanics --------------------------------------------------------------


def test_level0_scripted_solution_advances():
    env = Environment(LockPath())
    frame = env.reset()
    # Agent (1,1) -> goal (6,4): five right, three down.
    for _ in range(5):
        frame = env.step(GameAction.ACTION4)
    for _ in range(3):
        frame = env.step(GameAction.ACTION2)
    assert frame.score == 1            # level 0 completed
    assert frame.level == 1            # advanced
    assert frame.state == GameState.NOT_FINISHED


def test_full_playthrough_reaches_win():
    env = Environment(LockPath())
    frame = env.reset()
    n_levels = env.game.level_count
    for _ in range(n_levels):
        path = solve_level(env.game)
        assert path is not None
        for action in path:
            frame = env.step(action)
    assert frame.state == GameState.WIN
    assert frame.score == n_levels


def test_composition_level_needs_both_mechanics():
    # Level 3 composes key+door and block+pad; the solver must use both, so the
    # solution must collect the key (else the door blocks it) and cover the pad.
    env = Environment(LockPath())
    _advance_to_level(env, 3)
    game = env.game
    path = solve_level(game)
    assert path is not None
    # Replay on a fresh copy of the level and check both sub-goals were achieved.
    game.load_level(3)
    for action in path:
        game.apply(action, None)
    assert game.has_key                      # key+door mechanic exercised
    assert game.pads.issubset(game.blocks)   # block+pad mechanic exercised
    assert game.level_complete()


def test_hazard_triggers_game_over_and_reset_recovers():
    env = Environment(LockPath())
    _advance_to_level(env, 3)
    assert env.score == 3
    # L3: agent (1,1) -> key/hazard row; down x3 to (1,4), right x3 onto hazard (4,4).
    for _ in range(3):
        frame = env.step(GameAction.ACTION2)
    for _ in range(3):
        frame = env.step(GameAction.ACTION4)
    assert frame.state == GameState.GAME_OVER
    assert frame.available_actions == [GameAction.RESET]
    with pytest.raises(ActionNotAvailable):
        env.step(GameAction.ACTION1)         # only RESET is legal now
    frame = env.step(GameAction.RESET)
    assert frame.state == GameState.NOT_FINISHED
    assert frame.level == 3                   # restarts the level, not the game
    assert frame.score == 3                   # completed levels are retained


# --- agents & scoring -------------------------------------------------------


def test_random_agent_fuzzes_without_crashing():
    env = Environment(LockPath())
    result = run_episode(env, RandomAgent(seed=0), max_actions=3000)
    assert isinstance(result, GameResult)
    assert result.game_id == "lp01"
    assert 0 <= result.levels_completed <= env.game.level_count
    assert result.total_actions <= 3000


def test_scorecard_aggregates():
    card = Scorecard()
    card.record(GameResult("lp01", won=True, levels_completed=4, total_actions=120))
    card.record(GameResult("lp02", won=False, levels_completed=2, total_actions=300))
    assert card.levels_completed == 6
    assert card.total_actions == 420
    assert card.games_won == 1
    assert "lp01" in card.summary()
