"""Tests for ctkg/inference/surprise.py (Stage 5 — Surprise Detection)."""

from __future__ import annotations

import math
import sys
import os

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from experiments.symbolic_ai_v2.ctkg.inference.surprise import (
    SurpriseDetector,
    SurpriseAnnotation,
    _kl_point_mass,
    _KL_INF_SUBSTITUTE,
)
from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph


# ---------------------------------------------------------------------------
# Minimal stub predictor
# ---------------------------------------------------------------------------

class ConstantPredictor:
    """Always predicts the same distribution regardless of prefix."""
    def __init__(self, dist: dict[str, float]):
        self._dist = dict(dist)

    def predict_next(self, prefix):
        return dict(self._dist)


class EmptyPredictor:
    """Always returns {} (no prediction)."""
    def predict_next(self, prefix):
        return {}


class PrefixPredictor:
    """Returns different distributions based on prefix length."""
    def __init__(self, dists: list[dict[str, float]]):
        self._dists = dists

    def predict_next(self, prefix):
        idx = min(len(prefix), len(self._dists) - 1)
        return dict(self._dists[idx])


# ---------------------------------------------------------------------------
# _kl_point_mass
# ---------------------------------------------------------------------------

class TestKLPointMass:
    def test_perfect_prediction(self):
        # p=1.0 → KL = -log(1) = 0
        assert _kl_point_mass("a", {"a": 1.0}) == pytest.approx(0.0)

    def test_half_probability(self):
        # p=0.5 → KL = -log(0.5) = log(2) ≈ 0.693
        assert _kl_point_mass("a", {"a": 0.5, "b": 0.5}) == pytest.approx(math.log(2))

    def test_token_absent_from_dist(self):
        assert _kl_point_mass("z", {"a": 0.5, "b": 0.5}) == _KL_INF_SUBSTITUTE

    def test_empty_distribution(self):
        assert _kl_point_mass("a", {}) == _KL_INF_SUBSTITUTE

    def test_zero_probability(self):
        assert _kl_point_mass("a", {"a": 0.0, "b": 1.0}) == _KL_INF_SUBSTITUTE

    def test_unnormalized_distribution(self):
        # dist sums to 2; normalised p("a") = 0.5 → KL = log(2)
        kl = _kl_point_mass("a", {"a": 1.0, "b": 1.0})
        assert kl == pytest.approx(math.log(2))

    def test_point_mass_distribution(self):
        # p("x") = 1.0 (only token)
        assert _kl_point_mass("x", {"x": 1.0}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SurpriseDetector construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_threshold(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred)
        assert sd.get_threshold() == pytest.approx(1.0)

    def test_custom_threshold(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred, threshold=0.5)
        assert sd.get_threshold() == pytest.approx(0.5)

    def test_accepts_existing_graph(self):
        mg = MorphismGraph()
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred, mg=mg)
        assert sd._mg is mg


# ---------------------------------------------------------------------------
# Threshold stored in MorphismGraph
# ---------------------------------------------------------------------------

class TestThresholdInGraph:
    def test_set_threshold_updates_graph(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred, threshold=1.0)
        sd.set_threshold(2.5)
        assert sd.get_threshold() == pytest.approx(2.5)

    def test_threshold_persists_across_detector_reads(self):
        mg = MorphismGraph()
        pred = ConstantPredictor({"a": 1.0})
        sd1 = SurpriseDetector(pred, mg=mg, threshold=0.7)
        # A second detector on the same graph should see the same threshold.
        sd2 = SurpriseDetector(pred, mg=mg, threshold=0.7)
        # Both reference the same meta object & morphism.
        assert sd1.get_threshold() == pytest.approx(sd2.get_threshold())


# ---------------------------------------------------------------------------
# compute_surprise
# ---------------------------------------------------------------------------

class TestComputeSurprise:
    def test_zero_surprise_for_certain_prediction(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred)
        s = sd.compute_surprise([], "a")
        assert s == pytest.approx(0.0)

    def test_nonzero_surprise_for_uncertain_prediction(self):
        pred = ConstantPredictor({"a": 0.5, "b": 0.5})
        sd = SurpriseDetector(pred)
        s = sd.compute_surprise([], "a")
        assert s == pytest.approx(math.log(2))

    def test_infinite_surprise_for_unpredicted_token(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred)
        s = sd.compute_surprise([], "z")
        assert s == _KL_INF_SUBSTITUTE

    def test_maximal_surprise_when_predictor_returns_empty(self):
        pred = EmptyPredictor()
        sd = SurpriseDetector(pred)
        s = sd.compute_surprise([], "x")
        assert s == _KL_INF_SUBSTITUTE


# ---------------------------------------------------------------------------
# is_surprising
# ---------------------------------------------------------------------------

class TestIsSurprising:
    def test_above_threshold_is_surprising(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred, threshold=0.5)
        assert sd.is_surprising(1.0) is True

    def test_below_threshold_is_not_surprising(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred, threshold=2.0)
        assert sd.is_surprising(1.0) is False

    def test_equal_to_threshold_is_not_surprising(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred, threshold=1.0)
        # strictly greater than threshold → not surprising at exactly threshold
        assert sd.is_surprising(1.0) is False


# ---------------------------------------------------------------------------
# scan_sequence
# ---------------------------------------------------------------------------

class TestScanSequence:
    def test_empty_sequence_returns_empty(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred)
        assert sd.scan_sequence([]) == []

    def test_certain_sequence_no_flags(self):
        # Predictor always says "a" with p=1 → surprise=0 < threshold=1.0
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred, threshold=1.0)
        flagged = sd.scan_sequence(["a", "a", "a"])
        assert flagged == []

    def test_unexpected_token_is_flagged(self):
        # Predictor says "a" with p=1; sequence has "b" at position 1
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred, threshold=1.0)
        flagged = sd.scan_sequence(["a", "b", "a"])
        assert len(flagged) == 1
        assert flagged[0].position == 1
        assert flagged[0].token == "b"
        assert flagged[0].surprise == _KL_INF_SUBSTITUTE

    def test_multiple_positions_flagged(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred, threshold=1.0)
        flagged = sd.scan_sequence(["b", "b", "a"])
        assert len(flagged) == 2
        assert flagged[0].position == 0
        assert flagged[1].position == 1

    def test_start_parameter_skips_early_positions(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred, threshold=0.1)
        # Positions 0, 1, 2 with start=2 → only check position 2
        flagged = sd.scan_sequence(["b", "b", "b"], start=2)
        assert len(flagged) == 1
        assert flagged[0].position == 2

    def test_annotation_contains_predicted_distribution(self):
        pred = ConstantPredictor({"a": 0.3, "b": 0.7})
        sd = SurpriseDetector(pred, threshold=0.1)
        flagged = sd.scan_sequence(["a"])
        assert len(flagged) == 1
        assert "a" in flagged[0].predicted
        assert "b" in flagged[0].predicted


# ---------------------------------------------------------------------------
# surprise_sequence
# ---------------------------------------------------------------------------

class TestSurpriseSequence:
    def test_returns_float_per_token(self):
        pred = ConstantPredictor({"a": 0.5, "b": 0.5})
        sd = SurpriseDetector(pred)
        s = sd.surprise_sequence(["a", "b"])
        assert len(s) == 2
        assert all(isinstance(v, float) for v in s)

    def test_values_match_kl_formula(self):
        pred = ConstantPredictor({"a": 0.5, "b": 0.5})
        sd = SurpriseDetector(pred)
        s = sd.surprise_sequence(["a"])
        assert s[0] == pytest.approx(math.log(2))

    def test_start_parameter(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred)
        s = sd.surprise_sequence(["a", "a", "a"], start=1)
        assert len(s) == 2  # positions 1, 2 only

    def test_empty_sequence(self):
        pred = ConstantPredictor({"a": 1.0})
        sd = SurpriseDetector(pred)
        assert sd.surprise_sequence([]) == []
