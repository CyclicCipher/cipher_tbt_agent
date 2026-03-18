"""
Tests for TypedDeductionEngine (D-8) — value-dependent type checking.

D-8: "types as propositions; the graph IS the type context."
TypedDeductionEngine extends BFS deduction with predicates on node values.
When a node's value fails its predicate, outgoing edges are blocked.

This implements the dependent-type property: the set of reachable conclusions
from a node X depends on the NUMERIC VALUE of X, not just its name.

Test classes
------------
TestTypedDeductionBasic     : backward-compatible with untyped predict()
TestTypedDeductionConstraint: type constraints block and allow edges correctly
TestTypedDeductionVelocity  : SR-analog velocity bound use case
TestTypedDeductionCage      : anonymous tokens + type constraints (Iron Law)
TestTypedDeductionDefectProbe: targeted violation probes
"""
from __future__ import annotations

import sys
import os

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from experiments.symbolic_ai_v2.ctkg.inference.deduct import (
    DeductionEngine,
    TypedDeductionEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine() -> TypedDeductionEngine:
    return TypedDeductionEngine(
        rule_tok="rule", given_tok="given", conclude_tok="conclude"
    )


def _prefix(*rules_given) -> list[str]:
    """Build a prefix ending with 'conclude'. rules_given alternates rule triples
    and given pairs, terminated by 'conclude'."""
    return list(rules_given) + ["conclude"]


# ---------------------------------------------------------------------------
# TestTypedDeductionBasic
# ---------------------------------------------------------------------------

class TestTypedDeductionBasic:

    def test_untyped_same_as_base(self):
        """predict_with_types with no constraints gives same result as predict()."""
        eng = _engine()
        prefix = ["rule", "A", "B", "rule", "B", "C", "given", "A", "conclude"]
        base   = eng.predict(prefix)
        typed  = eng.predict_with_types(prefix)
        assert base == typed

    def test_returns_none_no_conclude(self):
        eng = _engine()
        prefix = ["rule", "A", "B", "given", "A"]
        assert eng.predict_with_types(prefix) is None

    def test_single_hop(self):
        eng = _engine()
        prefix = ["rule", "A", "B", "given", "A", "conclude"]
        result = eng.predict_with_types(prefix)
        assert result == {"B": 1.0}


# ---------------------------------------------------------------------------
# TestTypedDeductionConstraint
# ---------------------------------------------------------------------------

class TestTypedDeductionConstraint:

    def test_constraint_blocks_outgoing_edges(self):
        """If premise A fails its type constraint, edges from A are blocked."""
        eng = _engine()
        prefix = ["rule", "A", "B", "given", "A", "conclude"]
        # A has value 10.0 and constraint: value must be < 5.0
        vc = {"A": 10.0}
        tc = {"A": lambda v: v < 5.0}   # 10.0 fails
        result = eng.predict_with_types(prefix, vc, tc)
        assert result is None

    def test_constraint_allows_when_satisfied(self):
        """If premise A satisfies its type constraint, edges from A are followed."""
        eng = _engine()
        prefix = ["rule", "A", "B", "given", "A", "conclude"]
        vc = {"A": 3.0}
        tc = {"A": lambda v: v < 5.0}   # 3.0 satisfies
        result = eng.predict_with_types(prefix, vc, tc)
        assert result == {"B": 1.0}

    def test_intermediate_block_stops_chain(self):
        """Type failure on intermediate node stops further propagation."""
        eng = _engine()
        prefix = ["rule", "A", "B", "rule", "B", "C", "given", "A", "conclude"]
        # B fails its constraint: chain A→B allowed, B→C blocked
        vc = {"A": 2.0, "B": 20.0}
        tc = {"B": lambda v: v < 10.0}   # 20.0 fails
        result = eng.predict_with_types(prefix, vc, tc)
        # B is reachable (constraint checked before following outgoing, not before entering)
        # BFS: A (depth 0) → follows A→B → B at depth 1, B fails constraint → stops
        # So result should be B at depth 1 (it was reached; the block stops *outgoing* from B)
        assert result == {"B": 1.0}

    def test_constraint_on_absent_token_is_ok(self):
        """Token absent from value_context has value 0.0 by convention."""
        eng = _engine()
        prefix = ["rule", "A", "B", "given", "A", "conclude"]
        tc = {"A": lambda v: v == 0.0}   # absent → 0.0 → passes
        result = eng.predict_with_types(prefix, type_constraints=tc)
        assert result == {"B": 1.0}

    def test_no_constraints_same_as_no_typing(self):
        """Empty type_constraints dict = no filtering."""
        eng = _engine()
        prefix = ["rule", "A", "B", "rule", "B", "C", "given", "A", "conclude"]
        result = eng.predict_with_types(prefix, {}, {})
        assert result == {"C": 1.0}


# ---------------------------------------------------------------------------
# TestTypedDeductionVelocity
# ---------------------------------------------------------------------------

class TestTypedDeductionVelocity:
    """Analog of Special Relativity velocity bound.

    Rules encode: slow_velocity → medium_velocity → fast_velocity → ultra_velocity
    Type constraint: any velocity > c (= 0.9) is blocked (type violation).
    """

    _C = 0.9    # speed-of-light analog

    def _make_sr_prefix(self) -> list[str]:
        return [
            "rule", "slow", "medium",
            "rule", "medium", "fast",
            "rule", "fast", "ultra",
            "given", "slow",
            "conclude",
        ]

    def test_no_type_constraint_reaches_ultra(self):
        """Without type constraints, chain reaches ultra (depth 3)."""
        eng = _engine()
        prefix = self._make_sr_prefix()
        result = eng.predict_with_types(prefix)
        assert result == {"ultra": 1.0}

    def test_type_constraint_blocks_at_fast(self):
        """fast exceeds c → outgoing edges from fast are blocked → conclusion is fast."""
        eng = _engine()
        prefix = self._make_sr_prefix()
        vc = {"slow": 0.3, "medium": 0.6, "fast": 0.95, "ultra": 1.2}
        tc = {t: (lambda v: v <= self._C) for t in ("slow", "medium", "fast", "ultra")}
        # slow (0.3 ≤ 0.9) → medium (0.6 ≤ 0.9) → fast (0.95 > 0.9, blocks outgoing)
        result = eng.predict_with_types(prefix, vc, tc)
        assert result == {"fast": 1.0}, \
            f"SR analog: expected fast (bound violated), got {result}"

    def test_type_constraint_allows_medium(self):
        """With values slow=0.2, medium=0.5, fast=0.8 (all ≤ c=0.9), reaches ultra."""
        eng = _engine()
        prefix = self._make_sr_prefix()
        vc = {"slow": 0.2, "medium": 0.5, "fast": 0.8, "ultra": 1.0}
        tc = {
            "slow": lambda v: v <= self._C,
            "medium": lambda v: v <= self._C,
            "fast": lambda v: v <= self._C,
            "ultra": lambda v: v <= self._C,
        }
        # ultra (1.0 > 0.9) blocks outgoing, but ultra has no outgoing anyway
        result = eng.predict_with_types(prefix, vc, tc)
        # fast (0.8 passes) → ultra reached at depth 3; ultra blocks outgoing but is returned
        assert result == {"ultra": 1.0}

    def test_cage_anonymous_tokens(self):
        """Type constraint works with completely anonymous token names."""
        import random as _rng
        rng = _rng.Random(0)

        for seed in range(10):
            # Generate 4 anonymous tokens
            toks = [chr(0x2200 + rng.randint(0, 0xFE)) for _ in range(4)]
            t0, t1, t2, t3 = toks
            c_bound = 0.9

            prefix = [
                "rule", t0, t1,
                "rule", t1, t2,
                "rule", t2, t3,
                "given", t0,
                "conclude",
            ]
            vc = {t0: 0.3, t1: 0.6, t2: 0.95, t3: 1.2}
            tc = {t: (lambda v, b=c_bound: v <= b) for t in toks}

            eng = TypedDeductionEngine("rule", "given", "conclude")
            result = eng.predict_with_types(prefix, vc, tc)
            assert result == {t2: 1.0}, \
                f"seed {seed}: anonymous SR cage failed: got {result}"


# ---------------------------------------------------------------------------
# TestTypedDeductionDefectProbe
# ---------------------------------------------------------------------------

class TestTypedDeductionDefectProbe:

    def test_probe_no_string_comparison_in_constraint(self):
        """Type constraint predicate receives float value, not token string."""
        eng = _engine()
        received_type = []

        def recording_pred(v):
            received_type.append(type(v))
            return True

        prefix = ["rule", "A", "B", "given", "A", "conclude"]
        eng.predict_with_types(prefix, {"A": 3.14}, {"A": recording_pred})
        assert received_type, "Predicate was never called"
        assert received_type[0] is float, \
            f"PROBE: predicate received {received_type[0]}, expected float"

    def test_probe_backward_compatible_with_base_class(self):
        """TypedDeductionEngine.predict() (untyped) works identically to base class."""
        base = DeductionEngine("rule", "given", "conclude")
        typed = TypedDeductionEngine("rule", "given", "conclude")
        prefix = ["rule", "A", "B", "rule", "B", "C", "given", "A", "conclude"]
        assert base.predict(prefix) == typed.predict(prefix)

    def test_probe_type_constraint_not_checked_on_premise(self):
        """The PREMISE is checked before following edges.

        If the premise fails its type constraint, no conclusions are reachable.
        A correct implementation that starts BFS from the premise must check
        the premise's constraint before following its edges.
        """
        eng = _engine()
        prefix = ["rule", "A", "B", "given", "A", "conclude"]
        vc = {"A": 99.0}
        tc = {"A": lambda v: v < 10.0}
        result = eng.predict_with_types(prefix, vc, tc)
        assert result is None, \
            "PROBE: premise A fails type but edges were still followed"
