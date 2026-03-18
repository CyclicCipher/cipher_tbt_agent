"""Tests for ctkg/inference/revise.py (Stage 6 — Revision Engine)."""

from __future__ import annotations

import sys
import os

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from experiments.symbolic_ai_v2.ctkg.inference.revise import (
    RevisionEngine,
    RevisionCandidate,
)
from experiments.symbolic_ai_v2.ctkg.inference.surprise import SurpriseDetector
from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph


# ---------------------------------------------------------------------------
# Minimal stub predictors
# ---------------------------------------------------------------------------

class ConstantPredictor:
    def __init__(self, dist):
        self._dist = dict(dist)

    def predict_next(self, prefix):
        return dict(self._dist)


class PrefixPredictor:
    """Returns dist[0] for all positions — ignores prefix."""
    def __init__(self, dist):
        self._dist = dict(dist)

    def predict_next(self, prefix):
        return dict(self._dist)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(predictor, threshold=0.5, complexity=0.5):
    mg = MorphismGraph()
    sd = SurpriseDetector(predictor, mg=mg, threshold=threshold)
    return RevisionEngine(sd, mg, complexity_penalty=complexity), mg


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_creates_engine(self):
        pred = ConstantPredictor({"a": 1.0})
        eng, mg = _make_engine(pred)
        assert eng is not None
        assert eng._mg is mg


# ---------------------------------------------------------------------------
# generate_candidates
# ---------------------------------------------------------------------------

class TestGenerateCandidates:
    def test_no_anomalies_gives_no_candidates(self):
        pred = ConstantPredictor({"a": 1.0})
        eng, _ = _make_engine(pred, threshold=100.0)
        # No anomalies at very high threshold
        anomalies = eng._sd.scan_sequence(["a", "a", "a"])
        candidates = eng.generate_candidates(["a", "a", "a"], anomalies)
        assert candidates == []

    def test_single_anomaly_gives_one_candidate(self):
        # Predictor says "a"; sequence has "b" at position 1 → surprise
        pred = ConstantPredictor({"a": 1.0})
        eng, _ = _make_engine(pred, threshold=0.1, complexity=0.5)
        tokens = ["a", "b"]
        anomalies = eng._sd.scan_sequence(tokens)
        candidates = eng.generate_candidates(tokens, anomalies)
        assert len(candidates) == 1
        c = candidates[0]
        assert c.source_label == "a"
        assert c.target_label == "b"
        assert c.morph_type == "OBS_SEQ"

    def test_first_position_anomaly_no_candidate(self):
        # Anomaly at position 0 has no prev_token → cannot form bigram.
        pred = ConstantPredictor({"a": 1.0})
        eng, _ = _make_engine(pred, threshold=0.1, complexity=0.5)
        tokens = ["z"]  # anomalous at position 0
        anomalies = eng._sd.scan_sequence(tokens)
        # position 0 anomaly
        assert any(ann.position == 0 for ann in anomalies)
        candidates = eng.generate_candidates(tokens, anomalies)
        # No candidate because no prev token for position 0
        assert candidates == []

    def test_score_formula(self):
        # complexity=0.5, 1 anomaly explained → score = 1 - 0.5 = 0.5
        pred = ConstantPredictor({"a": 1.0})
        eng, _ = _make_engine(pred, threshold=0.1, complexity=0.5)
        tokens = ["a", "b"]
        anomalies = eng._sd.scan_sequence(tokens)
        candidates = eng.generate_candidates(tokens, anomalies)
        assert candidates[0].score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# revise
# ---------------------------------------------------------------------------

class TestRevise:
    def test_returns_none_when_no_anomaly(self):
        pred = ConstantPredictor({"a": 1.0})
        eng, _ = _make_engine(pred, threshold=100.0)
        result = eng.revise(["a", "a"])
        assert result is None

    def test_returns_candidate_on_anomaly(self):
        pred = ConstantPredictor({"a": 1.0})
        eng, _ = _make_engine(pred, threshold=0.1, complexity=0.5)
        result = eng.revise(["a", "b"])
        assert result is not None
        assert result.source_label == "a"
        assert result.target_label == "b"

    def test_adopted_candidate_creates_graph_edge(self):
        pred = ConstantPredictor({"a": 1.0})
        eng, mg = _make_engine(pred, threshold=0.1, complexity=0.5)
        eng.revise(["a", "b"])
        # There should now be an edge a→b in the graph.
        morphs = mg.morphisms(include_identity=False)
        obs_edges = [m for m in morphs if m.morph_type == "OBS_SEQ"]
        labels = [(mg._objects[m.source].label, mg._objects[m.target].label)
                  for m in obs_edges]
        assert ("a", "b") in labels

    def test_negative_score_candidate_not_adopted(self):
        # complexity=2.0 > 1 anomaly → score = 1 - 2 = -1 → not adopted
        pred = ConstantPredictor({"a": 1.0})
        eng, mg = _make_engine(pred, threshold=0.1, complexity=2.0)
        result = eng.revise(["a", "b"])
        assert result is None

    def test_second_revision_increments_evidence(self):
        pred = ConstantPredictor({"a": 1.0})
        eng, mg = _make_engine(pred, threshold=0.1, complexity=0.5)
        eng.revise(["a", "b"])
        eng.revise(["a", "b"])
        morphs = mg.morphisms(include_identity=False)
        obs = [m for m in morphs if m.morph_type == "OBS_SEQ"
               and mg._objects[m.source].label == "a"
               and mg._objects[m.target].label == "b"]
        assert len(obs) == 1
        assert obs[0].evidence_count == 2

    def test_start_parameter(self):
        pred = ConstantPredictor({"a": 1.0})
        eng, _ = _make_engine(pred, threshold=0.1, complexity=0.5)
        # "z" at position 0 is anomalous but skipped via start=1.
        result = eng.revise(["z", "a"], start=1)
        assert result is None


# ---------------------------------------------------------------------------
# Multiple candidates (A-2 scenario)
# ---------------------------------------------------------------------------

class TestMultipleCandidates:
    def test_best_candidate_adopted(self):
        """When two competing bigrams are anomalous, the one with more evidence wins."""
        pred = ConstantPredictor({"a": 1.0})
        eng, mg = _make_engine(pred, threshold=0.1, complexity=0.5)
        # Tokens: a→b (1 occurrence), a→c (1 occurrence)
        # Both have score 0.5; first one in candidates list is adopted.
        # Here we test that the revise() method returns a valid candidate.
        tokens = ["a", "b", "a", "c"]
        result = eng.revise(tokens)
        assert result is not None
        # Either "a→b" or "a→c" is the best candidate (both score 0.5).
        assert result.source_label == "a"
        assert result.target_label in ("b", "c")
