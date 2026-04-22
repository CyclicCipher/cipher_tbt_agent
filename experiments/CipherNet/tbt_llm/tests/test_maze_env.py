"""Tests for maze_env.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from maze_env import MazeEnv, DEFAULT_MAZE, DEFAULT_START, DEFAULT_GOAL, N, S, E, W


@pytest.fixture
def env():
    return MazeEnv()


def test_default_maze_shape():
    env = MazeEnv()
    assert env.H == 5 and env.W == 5


def test_start_and_goal_are_open():
    env = MazeEnv()
    assert env.grid[DEFAULT_START] == 0
    assert env.grid[DEFAULT_GOAL] == 0


def test_reset_returns_start(env):
    env.step(S)
    pos = env.reset()
    assert pos == DEFAULT_START
    assert env.pos == DEFAULT_START


def test_reset_at_arbitrary_cell(env):
    pos = env.reset_at((2, 2))
    assert pos == (2, 2)
    assert env.pos == (2, 2)


def test_reset_at_wall_raises(env):
    # (0,2) is a wall in DEFAULT_MAZE
    with pytest.raises(AssertionError):
        env.reset_at((0, 2))


def test_valid_step_moves_agent(env):
    # (0,0) can go S or E in default maze (row1col0=open, row0col1=open)
    new_pos, hit = env.step(S)
    assert hit is False
    assert new_pos == (1, 0)
    assert env.pos == (1, 0)


def test_wall_step_does_not_move_agent(env):
    # (0,0): North is out of bounds, West is out of bounds
    pos_before = env.pos
    new_pos, hit = env.step(N)
    assert hit is True
    assert new_pos == pos_before
    assert env.pos == pos_before


def test_prev_pos_updated_on_step(env):
    env.step(S)
    assert env.prev_pos == DEFAULT_START


def test_prev_pos_unchanged_on_wall_hit(env):
    # hitting a wall: prev_pos is still set to current pos before attempt
    env.step(N)   # hits wall
    assert env.prev_pos == DEFAULT_START


def test_valid_actions_excludes_walls(env):
    # At (0,0): N and W are boundaries (walls), S and E should be valid
    valid = env.valid_actions()
    assert N not in valid
    assert W not in valid
    assert S in valid or E in valid   # at least one should be open


def test_reached_goal_false_initially(env):
    assert not env.reached_goal()


def test_reached_goal_true_at_goal(env):
    env.reset_at(DEFAULT_GOAL)
    assert env.reached_goal()


def test_open_cells_count_matches_grid():
    env = MazeEnv()
    n_open = int((DEFAULT_MAZE == 0).sum())
    assert len(env.open_cells()) == n_open
    assert env.n_open() == n_open


def test_all_open_cells_are_actually_open(env):
    for pos in env.open_cells():
        assert env.grid[pos] == 0


def test_render_contains_agent_marker(env):
    r = env.render()
    assert 'A' in r


def test_render_contains_goal_marker(env):
    env.reset_at((2, 2))   # move agent away from goal
    r = env.render()
    assert 'G' in r


def test_render_wall_count(env):
    r = env.render()
    n_walls = sum(c == '#' for c in r)
    expected = int((DEFAULT_MAZE == 1).sum())
    assert n_walls == expected


def test_custom_maze():
    grid = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    env = MazeEnv(grid=grid, start=(0, 0), goal=(1, 1))
    assert env.H == 2 and env.W == 2
    env.reset()
    _, hit = env.step(E)
    assert not hit
    assert env.pos == (0, 1)


def test_full_path_to_goal():
    """Verify a known path through the default maze reaches the goal."""
    env = MazeEnv()
    env.reset()
    # One valid path: (0,0)→S→(1,0)→S→(2,0)→E→(2,1)→E→(2,2)→E→(2,3)→S→(3,3)→S→(4,3) — wait, (4,3) is a wall?
    # Let me trace manually:
    # DEFAULT_MAZE row 4: [0, 0, 0, 1, 0]
    # So (4,3) is a wall. (4,4) = goal.
    # Try: (2,3)→S→(3,3)→S→(4,3) — (4,3) is wall
    # Better: (2,3)→S→(3,3)→E→(3,4)→S→(4,4)
    path = [S, S, E, E, E, S, E, S]
    for action in path:
        env.step(action)
    assert env.reached_goal(), f"Did not reach goal, at {env.pos}"
