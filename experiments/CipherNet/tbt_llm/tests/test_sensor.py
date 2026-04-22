"""Tests for sensor.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from maze_env import MazeEnv, DEFAULT_MAZE, DEFAULT_GOAL
from sensor import LocalSensor, BIT_N, BIT_S, BIT_E, BIT_W, BIT_GOAL, N_BITS


@pytest.fixture
def env():
    return MazeEnv()


@pytest.fixture
def sensor():
    return LocalSensor()


def test_output_shape(env, sensor):
    sdr = sensor.encode(env)
    assert sdr.shape == (N_BITS,)
    assert sdr.dtype == np.int8


def test_output_is_binary(env, sensor):
    sdr = sensor.encode(env)
    assert set(sdr.tolist()).issubset({0, 1})


def test_corner_walls_at_start(env, sensor):
    # (0,0): N=boundary(wall), W=boundary(wall)
    sdr = sensor.encode(env)
    assert sdr[BIT_N] == 1, "North should be wall at (0,0)"
    assert sdr[BIT_W] == 1, "West should be wall at (0,0)"


def test_no_goal_flag_at_start(env, sensor):
    sdr = sensor.encode(env)
    assert sdr[BIT_GOAL] == 0


def test_goal_flag_at_goal(env, sensor):
    env.reset_at(DEFAULT_GOAL)
    sdr = sensor.encode(env)
    assert sdr[BIT_GOAL] == 1


def test_open_directions_not_flagged(env, sensor):
    # Move to (1,0): row1col0 open. South (2,0) is open. North (0,0) is open.
    env.reset_at((1, 0))
    sdr = sensor.encode(env)
    assert sdr[BIT_N] == 0, "North from (1,0) is (0,0) which is open"
    assert sdr[BIT_S] == 0, "South from (1,0) is (2,0) which is open"


def test_wall_direction_flagged(env, sensor):
    # At (1,0): East is (1,1) which is a wall in DEFAULT_MAZE
    env.reset_at((1, 0))
    sdr = sensor.encode(env)
    assert sdr[BIT_E] == 1, "East from (1,0) is (1,1) wall"


def test_encode_at_does_not_move_agent(env, sensor):
    original_pos = env.pos
    sensor.encode_at(env, (2, 2))
    assert env.pos == original_pos


def test_encode_at_matches_encode_after_move(env, sensor):
    sdr_at = sensor.encode_at(env, (2, 2))
    env.reset_at((2, 2))
    sdr_direct = sensor.encode(env)
    np.testing.assert_array_equal(sdr_at, sdr_direct)


def test_distinct_cells_can_have_same_reading(env, sensor):
    """
    Two open interior cells may share the same sensor reading.
    This is by design — the column must use path integration to tell them apart.
    """
    readings = {}
    for pos in env.open_cells():
        sdr = sensor.encode_at(env, pos)
        key = tuple(sdr.tolist())
        if key not in readings:
            readings[key] = []
        readings[key].append(pos)
    # Find any reading shared by >1 cell (expect this in a 5x5 maze)
    ambiguous = {k: v for k, v in readings.items() if len(v) > 1}
    assert len(ambiguous) > 0, (
        "Expected some cells to share sensor readings in the default maze. "
        "If the maze is fully discriminated by sensor alone, path integration "
        "is not being tested."
    )


def test_boundary_treated_as_wall(env, sensor):
    # Top edge: north of row 0 is always a wall
    for c in range(env.W):
        if env.grid[0, c] == 0:
            env.reset_at((0, c))
            sdr = sensor.encode(env)
            assert sdr[BIT_N] == 1, f"North boundary at (0,{c}) should read as wall"
            break
