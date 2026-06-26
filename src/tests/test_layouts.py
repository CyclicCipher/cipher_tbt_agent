"""Tests for the procedural LockPath layout generators."""

from __future__ import annotations

import os
import sys

import pytest

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:

from tasks import Environment, GameState  # noqa: E402
from tasks.games import LockPath  # noqa: E402
from tasks.oracle import is_solvable, solve_level  # noqa: E402
from tasks.layouts import (  # noqa: E402
    GENERATORS,
    _loaded_game,
    make_game,
    sample_layouts,
    train_test_split,
)

MECHANICS = ["nav", "key_door", "block_pad", "compose"]


def _chars(layout):
    return set("".join(layout))


@pytest.mark.parametrize("mech", MECHANICS)
def test_sampled_layouts_are_solvable(mech):
    layouts = sample_layouts(mech, n=12, seed=0)
    assert len(layouts) == 12
    for layout in layouts:
        # Every layout has exactly one agent and one goal, and is BFS-solvable.
        flat = "".join(layout)
        assert flat.count("A") == 1
        assert flat.count("G") == 1
        game = _loaded_game(layout)
        assert is_solvable(game)
        assert solve_level(game), "solution must be non-trivial (>0 actions)"


def test_mechanic_specific_elements_present():
    assert {"K", "D"} <= _chars(sample_layouts("key_door", 6, seed=1)[0])
    assert {"B", "P"} <= _chars(sample_layouts("block_pad", 6, seed=1)[0])
    compose = _chars(sample_layouts("compose", 6, seed=1)[0])
    assert {"K", "D", "B", "P"} <= compose
    nav = _chars(sample_layouts("nav", 6, seed=1)[0])
    assert not ({"K", "D", "B", "P"} & nav)  # nav is bare navigation


def test_sampling_is_deterministic():
    a = sample_layouts("compose", 8, seed=42)
    b = sample_layouts("compose", 8, seed=42)
    c = sample_layouts("compose", 8, seed=43)
    assert a == b
    assert a != c


def test_train_test_split_is_disjoint():
    layouts = sample_layouts("block_pad", 20, seed=7)
    train, test = train_test_split(layouts, train_frac=0.7)
    assert len(train) == 14 and len(test) == 6
    assert not ({tuple(l) for l in train} & {tuple(l) for l in test})


def test_generated_game_is_playable_to_win():
    # A procedural game of one layout per mechanic should be solvable end to end.
    layouts = [sample_layouts(m, 1, seed=3)[0] for m in MECHANICS]
    env = Environment(make_game(layouts))
    frame = env.reset()
    for _ in range(len(layouts)):
        path = solve_level(env.game)
        assert path is not None
        for action in path:
            frame = env.step(action)
    assert frame.state == GameState.WIN
    assert frame.score == len(layouts)


def test_unknown_mechanic_raises():
    with pytest.raises(ValueError):
        sample_layouts("teleport", 1)
