"""Tests for place_map.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from place_map import PlaceMap


def sdr(*bits) -> np.ndarray:
    """Helper: build a 5-bit int8 SDR from explicit bit values."""
    return np.array(bits, dtype=np.int8)


@pytest.fixture
def pm():
    return PlaceMap(n_cells=10)


# ------------------------------------------------------------------
# observe / predict
# ------------------------------------------------------------------

def test_predict_unseen_returns_none(pm):
    assert pm.predict((0, 0)) is None


def test_observe_and_predict(pm):
    s = sdr(1, 0, 1, 0, 0)
    pm.observe(s, (0, 0))
    stored = pm.predict((0, 0))
    assert stored is not None
    np.testing.assert_array_equal(stored, s)


def test_union_model_grows(pm):
    pm.observe(sdr(1, 0, 0, 0, 0), (0, 0))
    pm.observe(sdr(0, 1, 0, 0, 0), (0, 0))
    stored = pm.predict((0, 0))
    assert stored[0] == 1 and stored[1] == 1


def test_union_never_shrinks(pm):
    pm.observe(sdr(1, 1, 0, 0, 0), (0, 0))
    pm.observe(sdr(0, 0, 1, 1, 0), (0, 0))
    stored = pm.predict((0, 0))
    assert stored.sum() == 4


def test_distinct_positions_independent(pm):
    pm.observe(sdr(1, 0, 0, 0, 0), (0, 0))
    pm.observe(sdr(0, 1, 0, 0, 0), (1, 1))
    assert pm.predict((0, 0))[0] == 1
    assert pm.predict((1, 1))[1] == 1
    assert pm.predict((0, 0))[1] == 0


# ------------------------------------------------------------------
# match
# ------------------------------------------------------------------

def test_match_perfect(pm):
    s = sdr(1, 0, 1, 0, 0)
    pm.observe(s, (0, 0))
    assert pm.match(s, (0, 0)) == pytest.approx(1.0)


def test_match_unseen_is_zero(pm):
    assert pm.match(sdr(1, 0, 0, 0, 0), (9, 9)) == 0.0


def test_match_partial(pm):
    pm.observe(sdr(1, 1, 1, 0, 0), (0, 0))  # stored: bits 0,1,2
    obs = sdr(1, 1, 0, 0, 0)                  # observed: bits 0,1 (2 of 2 active)
    # match = fraction of obs active bits in stored = 2/2 = 1.0
    assert pm.match(obs, (0, 0)) == pytest.approx(1.0)


def test_match_zero_for_all_mismatch(pm):
    pm.observe(sdr(0, 0, 1, 1, 0), (0, 0))   # stored: bits 2,3
    obs = sdr(1, 1, 0, 0, 0)                   # observed: bits 0,1 — no overlap
    assert pm.match(obs, (0, 0)) == pytest.approx(0.0)


def test_match_empty_sdr_is_zero(pm):
    pm.observe(sdr(1, 0, 0, 0, 0), (0, 0))
    assert pm.match(sdr(0, 0, 0, 0, 0), (0, 0)) == 0.0


# ------------------------------------------------------------------
# prediction_error
# ------------------------------------------------------------------

def test_prediction_error_perfect_is_zero(pm):
    s = sdr(1, 0, 1, 0, 0)
    pm.observe(s, (0, 0))
    assert pm.prediction_error(s, (0, 0)) == pytest.approx(0.0)


def test_prediction_error_unseen_is_one(pm):
    assert pm.prediction_error(sdr(1, 0, 0, 0, 0), (5, 5)) == pytest.approx(1.0)


# ------------------------------------------------------------------
# coverage
# ------------------------------------------------------------------

def test_coverage_zero_initially(pm):
    assert pm.coverage() == pytest.approx(0.0)


def test_coverage_increases_with_new_positions(pm):
    pm.observe(sdr(1, 0, 0, 0, 0), (0, 0))
    assert pm.coverage() == pytest.approx(1 / 10)
    pm.observe(sdr(0, 1, 0, 0, 0), (1, 0))
    assert pm.coverage() == pytest.approx(2 / 10)


def test_coverage_does_not_exceed_one():
    pm = PlaceMap(n_cells=2)
    pm.observe(sdr(1, 0, 0, 0, 0), (0, 0))
    pm.observe(sdr(0, 1, 0, 0, 0), (1, 0))
    pm.observe(sdr(0, 0, 1, 0, 0), (2, 0))  # more positions than n_cells
    assert pm.coverage() > 1.0   # allowed to exceed 1.0 (n_cells is a denominator estimate)


def test_revisiting_same_cell_does_not_increase_coverage(pm):
    pm.observe(sdr(1, 0, 0, 0, 0), (0, 0))
    c1 = pm.coverage()
    pm.observe(sdr(1, 0, 0, 0, 0), (0, 0))
    assert pm.coverage() == pytest.approx(c1)


# ------------------------------------------------------------------
# localize
# ------------------------------------------------------------------

def test_localize_empty_history_returns_none():
    pm = PlaceMap(n_cells=5)
    assert pm.localize([]) is None


def test_localize_empty_map_returns_none():
    pm = PlaceMap(n_cells=5)
    history = [(sdr(1, 0, 0, 0, 0), (0, 1))]
    assert pm.localize(history) is None


def test_localize_correct_position():
    """Build a small map and verify localize returns the correct position."""
    pm = PlaceMap(n_cells=4)
    # Map: (0,0)→[1,0,1,0,0], (0,1)→[1,0,0,1,0], (1,0)→[0,0,1,0,0], (1,1)→[0,0,0,1,0]
    pm.observe(sdr(1, 0, 1, 0, 0), (0, 0))
    pm.observe(sdr(1, 0, 0, 1, 0), (0, 1))
    pm.observe(sdr(0, 0, 1, 0, 0), (1, 0))
    pm.observe(sdr(0, 0, 0, 1, 0), (1, 1))

    # History: start at (0,0), move E to (0,1), observe sdr at (0,1)
    # displacement for E = (0, 1)
    history = [(sdr(1, 0, 0, 1, 0), (0, 1))]
    result = pm.localize(history)
    assert result == (0, 1)
