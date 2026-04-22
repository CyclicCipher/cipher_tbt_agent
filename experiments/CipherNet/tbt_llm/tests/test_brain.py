"""Tests for brain.py — SingleColumnBrain."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from maze_env import MazeEnv, DEFAULT_MAZE, DEFAULT_START, DEFAULT_GOAL, N, S, E, W
from brain import SingleColumnBrain, _entropy, _normalise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_brain(n_cells=20, min_coverage=0.0, epsilon=0.0):
    """Brain with coverage gate disabled so model-based policy always runs."""
    return SingleColumnBrain(n_cells=n_cells, min_coverage=min_coverage,
                             epsilon=epsilon)


# ---------------------------------------------------------------------------
# _entropy / _normalise utilities
# ---------------------------------------------------------------------------

def test_entropy_uniform():
    d = {0: 0.5, 1: 0.5}
    assert _entropy(d) == pytest.approx(np.log(2))


def test_entropy_point_mass():
    d = {0: 1.0, 1: 0.0}
    assert _entropy(d) == pytest.approx(0.0, abs=1e-9)


def test_normalise_sums_to_one():
    d = {'a': 3.0, 'b': 1.0}
    n = _normalise(d)
    assert sum(n.values()) == pytest.approx(1.0)
    assert n['a'] == pytest.approx(0.75)


def test_normalise_near_zero_gives_uniform():
    d = {'a': 1e-20, 'b': 1e-20}
    n = _normalise(d)
    assert n['a'] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_known_start_point_belief():
    brain = make_brain()
    brain.reset((2, 3), known_start=True)
    assert brain.belief == {(2, 3): 1.0}


def test_reset_unknown_start_no_map_gives_point():
    brain = make_brain()
    # No map yet → falls back to point mass even with known_start=False
    brain.reset((1, 1), known_start=False)
    assert brain.belief == {(1, 1): 1.0}


def test_reset_unknown_start_with_map_gives_uniform():
    brain = make_brain()
    # Pre-populate the map with a few positions
    brain.place_map.observe(np.array([1, 0, 0, 1, 0], dtype=np.int8), (0, 0))
    brain.place_map.observe(np.array([0, 1, 1, 0, 0], dtype=np.int8), (1, 1))
    brain.reset((0, 0), known_start=False)
    assert len(brain.belief) == 2
    assert sum(brain.belief.values()) == pytest.approx(1.0)
    for v in brain.belief.values():
        assert v == pytest.approx(0.5)


def test_reset_clears_belief_across_episodes():
    brain = make_brain()
    brain.reset((0, 0))
    brain.reset((3, 3))
    assert brain.belief == {(3, 3): 1.0}


# ---------------------------------------------------------------------------
# observe / _update_belief
# ---------------------------------------------------------------------------

def test_observe_writes_to_place_map():
    brain = make_brain()
    brain.reset((0, 0))
    sdr = np.array([1, 0, 0, 1, 0], dtype=np.int8)
    brain.observe(sdr)
    stored = brain.place_map.predict((0, 0))
    assert stored is not None
    np.testing.assert_array_equal(stored, sdr)


def test_belief_update_sharpens_on_consistent_sdr():
    """After observing an sdr consistent with pos A but not pos B, belief[A] rises."""
    brain = make_brain()
    # Prime the map: two positions with distinct patterns
    sdr_a = np.array([1, 0, 1, 0, 0], dtype=np.int8)
    sdr_b = np.array([0, 1, 0, 1, 0], dtype=np.int8)
    brain.place_map.observe(sdr_a, (0, 0))
    brain.place_map.observe(sdr_b, (1, 1))
    # Uniform prior over both
    brain.belief = {(0, 0): 0.5, (1, 1): 0.5}
    # Observe sdr_a — should push weight toward (0,0).
    # (1,1) may be dropped from belief (weight 0) or remain at 0.
    brain._update_belief(sdr_a)
    assert brain.belief[(0, 0)] > brain.belief.get((1, 1), 0.0)


def test_belief_update_normalises():
    brain = make_brain()
    sdr = np.array([1, 0, 0, 0, 0], dtype=np.int8)
    brain.place_map.observe(sdr, (0, 0))
    brain.belief = {(0, 0): 0.5, (1, 0): 0.5}
    brain._update_belief(sdr)
    assert sum(brain.belief.values()) == pytest.approx(1.0)


def test_belief_collapse_resets_to_uniform():
    """If no stored position matches the sdr, belief resets to uniform."""
    brain = make_brain()
    brain.place_map.observe(np.array([1, 1, 0, 0, 0], dtype=np.int8), (0, 0))
    brain.belief = {(0, 0): 1.0}
    # Observe an sdr with NO active bits → match is 0 for any stored pattern
    brain._update_belief(np.array([0, 0, 0, 0, 0], dtype=np.int8))
    assert sum(brain.belief.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# select_action
# ---------------------------------------------------------------------------

def test_select_action_returns_valid_action():
    env = MazeEnv()
    brain = make_brain()
    brain.reset(DEFAULT_START)
    action = brain.select_action(env.valid_actions())
    assert action in env.valid_actions()


def test_select_action_low_coverage_is_random(monkeypatch):
    """With min_coverage=0.9 and empty map, policy must be random (not model-based)."""
    brain = SingleColumnBrain(n_cells=20, min_coverage=0.9)
    brain.reset((0, 0))
    # Just check it doesn't raise and returns something from valid_actions
    valid = [N, S, E, W]
    action = brain.select_action(valid)
    assert action in valid


def test_select_action_no_valid_raises():
    brain = make_brain()
    brain.reset((0, 0))
    with pytest.raises(ValueError):
        brain.select_action([])


def test_select_action_single_option():
    brain = make_brain()
    brain.reset((0, 0))
    action = brain.select_action([S])
    assert action == S


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------

def test_step_returns_sdr_shape():
    env = MazeEnv()
    brain = make_brain()
    brain.reset(DEFAULT_START)
    sdr = brain.step(S, env)
    assert sdr.shape == (5,)
    assert sdr.dtype == np.int8


def test_step_updates_frame_on_success():
    env = MazeEnv()
    brain = make_brain()
    brain.reset(DEFAULT_START)
    brain.step(S, env)   # (0,0) → (1,0)
    assert brain.frame.position_key() == (1, 0)


def test_step_corrects_frame_on_wall_hit():
    env = MazeEnv()
    brain = make_brain()
    brain.reset(DEFAULT_START)
    # (0,0) has wall to North and West
    brain.step(N, env)   # hits wall → frame should stay at (0,0)
    assert brain.frame.position_key() == (0, 0)


def test_step_updates_place_map():
    env = MazeEnv()
    brain = make_brain()
    brain.reset(DEFAULT_START)
    brain.step(S, env)
    assert brain.place_map.predict((1, 0)) is not None


# ---------------------------------------------------------------------------
# best_estimate / belief_entropy / is_localised
# ---------------------------------------------------------------------------

def test_best_estimate_point_mass():
    brain = make_brain()
    brain.reset((2, 3))
    assert brain.best_estimate() == (2, 3)


def test_best_estimate_empty_belief():
    brain = make_brain()
    brain.belief = {}
    assert brain.best_estimate() is None


def test_belief_entropy_point_mass_is_zero():
    brain = make_brain()
    brain.reset((0, 0))
    assert brain.belief_entropy() == pytest.approx(0.0, abs=1e-9)


def test_belief_entropy_uniform_is_positive():
    brain = make_brain()
    brain.belief = {(0, 0): 0.5, (1, 0): 0.5}
    assert brain.belief_entropy() > 0.0


def test_is_localised_when_certain():
    brain = SingleColumnBrain(n_cells=10, confidence_threshold=1.0)
    brain.reset((0, 0))   # point mass → entropy 0 < threshold 1.0
    assert brain.is_localised()


def test_is_not_localised_when_uniform():
    brain = SingleColumnBrain(n_cells=10, confidence_threshold=0.1)
    brain.belief = {(0, 0): 0.5, (1, 0): 0.5}
    # entropy = log(2) ≈ 0.69 > threshold 0.1
    assert not brain.is_localised()


# ---------------------------------------------------------------------------
# Integration: short maze walk
# ---------------------------------------------------------------------------

def test_short_walk_builds_map():
    """Walk the default maze for 10 steps; map should have multiple entries."""
    env = MazeEnv()
    brain = make_brain(n_cells=env.n_open())
    brain.reset(DEFAULT_START)
    for _ in range(10):
        valid = env.valid_actions()
        action = brain.select_action(valid)
        brain.step(action, env)
    assert len(brain.place_map._model) > 1


def test_frame_tracks_true_position():
    """After a known sequence of moves, frame position should match env position."""
    env = MazeEnv()
    brain = make_brain()
    brain.reset(DEFAULT_START)
    path = [S, S, E, E]
    for action in path:
        brain.step(action, env)
    assert brain.frame.position_key() == env.pos
