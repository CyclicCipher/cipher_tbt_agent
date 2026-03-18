"""Tests for ctkg/inference/deduct.py (Stage 4 — Deduction Engine)."""

from __future__ import annotations

import sys
import os

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from experiments.symbolic_ai_v2.ctkg.inference.deduct import DeductionEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    return DeductionEngine(rule_tok="rule", given_tok="given", conclude_tok="conclude")


@pytest.fixture()
def anon_engine():
    """Engine using anonymous Unicode symbols as role tokens."""
    return DeductionEngine(rule_tok="∀", given_tok="∃", conclude_tok="∂")


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_max_depth(self):
        e = DeductionEngine("r", "g", "c")
        assert e._max_depth == 10

    def test_custom_max_depth(self):
        e = DeductionEngine("r", "g", "c", max_depth=3)
        assert e._max_depth == 3


# ---------------------------------------------------------------------------
# predict() — basic firing conditions
# ---------------------------------------------------------------------------

class TestPredictFiring:
    def test_does_not_fire_on_empty_prefix(self, engine):
        assert engine.predict([]) is None

    def test_does_not_fire_without_conclude_tok(self, engine):
        assert engine.predict(["rule", "A", "B", "given", "A"]) is None

    def test_does_not_fire_when_conclude_not_last(self, engine):
        # conclude_tok is present but not last
        assert engine.predict(["rule", "A", "B", "conclude", "given", "A"]) is None

    def test_does_not_fire_without_given(self, engine):
        prefix = ["rule", "A", "B", "conclude"]
        # No given_tok → no premises → None
        assert engine.predict(prefix) is None

    def test_does_not_fire_without_rules(self, engine):
        prefix = ["given", "A", "conclude"]
        # No rules → no graph edges → None
        assert engine.predict(prefix) is None


# ---------------------------------------------------------------------------
# D-1: 1-hop deduction
# ---------------------------------------------------------------------------

class TestD1OnHop:
    def test_basic_modus_ponens(self, engine):
        # Rule: A → B.  Given: A.  Conclude: B.
        prefix = ["rule", "A", "B", "given", "A", "conclude"]
        result = engine.predict(prefix)
        assert result == {"B": 1.0}

    def test_modus_ponens_different_tokens(self, engine):
        prefix = ["rule", "cat", "dog", "given", "cat", "conclude"]
        result = engine.predict(prefix)
        assert result == {"dog": 1.0}

    def test_anonymous_symbols(self, anon_engine):
        # Same structure with Unicode anonymous symbols
        prefix = ["∀", "∁", "∆", "∃", "∁", "∂"]
        result = anon_engine.predict(prefix)
        assert result == {"∆": 1.0}

    def test_irrelevant_rule_ignored(self, engine):
        # Rule: X → Y (irrelevant), Rule: A → B.  Given: A.  Conclude: B.
        prefix = ["rule", "X", "Y", "rule", "A", "B", "given", "A", "conclude"]
        result = engine.predict(prefix)
        assert result == {"B": 1.0}

    def test_multiple_premises_not_supported_gracefully(self, engine):
        # Two given tokens — both are premises; only one has a rule
        prefix = ["rule", "A", "B", "given", "A", "given", "X", "conclude"]
        result = engine.predict(prefix)
        # A is reachable via 1 hop to B; X has no outgoing rule
        assert result == {"B": 1.0}


# ---------------------------------------------------------------------------
# D-2: 2-hop deduction
# ---------------------------------------------------------------------------

class TestD2TwoHop:
    def test_two_step_chain(self, engine):
        # A → B → C.  Given A.  Conclude C.
        prefix = ["rule", "A", "B", "rule", "B", "C", "given", "A", "conclude"]
        result = engine.predict(prefix)
        assert result == {"C": 1.0}

    def test_two_step_anonymous(self, anon_engine):
        # ∁ → ∆ → ∑.  Given ∁.  Conclude ∑.
        prefix = ["∀", "∁", "∆", "∀", "∆", "∑", "∃", "∁", "∂"]
        result = anon_engine.predict(prefix)
        assert result == {"∑": 1.0}

    def test_two_step_only_deepest_returned(self, engine):
        # A → B → C.  Deepest reachable = C (depth 2), not B (depth 1).
        prefix = ["rule", "A", "B", "rule", "B", "C", "given", "A", "conclude"]
        result = engine.predict(prefix)
        # C is depth 2, B is depth 1 → C is returned
        assert "C" in result
        assert "B" not in result


# ---------------------------------------------------------------------------
# D-3: 3-hop deduction
# ---------------------------------------------------------------------------

class TestD3ThreeHop:
    def test_three_step_chain(self, engine):
        # A → B → C → D.  Given A.  Conclude D.
        prefix = [
            "rule", "A", "B",
            "rule", "B", "C",
            "rule", "C", "D",
            "given", "A", "conclude"
        ]
        result = engine.predict(prefix)
        assert result == {"D": 1.0}

    def test_three_step_anonymous(self, anon_engine):
        prefix = [
            "∀", "∁", "∆",
            "∀", "∆", "∑",
            "∀", "∑", "∏",
            "∃", "∁", "∂",
        ]
        result = anon_engine.predict(prefix)
        assert result == {"∏": 1.0}


# ---------------------------------------------------------------------------
# max_depth enforcement
# ---------------------------------------------------------------------------

class TestMaxDepth:
    def test_depth_limit_stops_search(self):
        # Chain A→B→C→D, but max_depth=1.  Should only reach B.
        engine = DeductionEngine("rule", "given", "conclude", max_depth=1)
        prefix = [
            "rule", "A", "B",
            "rule", "B", "C",
            "rule", "C", "D",
            "given", "A", "conclude"
        ]
        result = engine.predict(prefix)
        assert result == {"B": 1.0}

    def test_depth_limit_2_reaches_two_hops(self):
        engine = DeductionEngine("rule", "given", "conclude", max_depth=2)
        prefix = [
            "rule", "A", "B",
            "rule", "B", "C",
            "rule", "C", "D",
            "given", "A", "conclude"
        ]
        result = engine.predict(prefix)
        assert result == {"C": 1.0}


# ---------------------------------------------------------------------------
# predict_chain
# ---------------------------------------------------------------------------

class TestPredictChain:
    def test_basic_chain(self, engine):
        rules = [("A", "B")]
        chain = engine.predict_chain(rules, ["A"])
        assert chain == ["A", "B"]

    def test_two_hop_chain(self, engine):
        rules = [("A", "B"), ("B", "C")]
        chain = engine.predict_chain(rules, ["A"])
        assert chain == ["A", "B", "C"]

    def test_three_hop_chain(self, engine):
        rules = [("A", "B"), ("B", "C"), ("C", "D")]
        chain = engine.predict_chain(rules, ["A"])
        assert chain == ["A", "B", "C", "D"]

    def test_no_rules_returns_none(self, engine):
        chain = engine.predict_chain([], ["A"])
        assert chain is None

    def test_no_premises_returns_none(self, engine):
        chain = engine.predict_chain([("A", "B")], [])
        assert chain is None

    def test_isolated_premise_returns_none(self, engine):
        # Premise has no outgoing rule
        chain = engine.predict_chain([("X", "Y")], ["A"])
        assert chain is None


# ---------------------------------------------------------------------------
# Symbol invariance
# ---------------------------------------------------------------------------

class TestSymbolInvariance:
    def test_different_role_symbols_same_result(self):
        """Same structure, different role token names → same deductive result."""
        content = ("P", "Q", "R")  # antecedent, intermediate, consequent
        result1 = DeductionEngine("rule", "given", "conclude").predict(
            ["rule", content[0], content[1],
             "rule", content[1], content[2],
             "given", content[0], "conclude"]
        )
        result2 = DeductionEngine("∀", "∃", "∂").predict(
            ["∀", content[0], content[1],
             "∀", content[1], content[2],
             "∃", content[0], "∂"]
        )
        assert result1 == result2 == {content[2]: 1.0}

    def test_anonymous_content_tokens(self, anon_engine):
        """Content tokens can be any arbitrary strings."""
        content = ("Ω", "Γ", "Λ")
        prefix = [
            "∀", content[0], content[1],
            "∀", content[1], content[2],
            "∃", content[0], "∂",
        ]
        result = anon_engine.predict(prefix)
        assert result == {content[2]: 1.0}
