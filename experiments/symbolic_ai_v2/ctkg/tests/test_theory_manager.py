"""
Tests for Phase 4 — Theory Compartments and Cross-Domain Consistency.

Test classes
------------
TestTheoryManagerBasic (10 tests)
    - register / assign / theory_name
    - predict_under_theory with single and multiple laws
    - predict returns None for empty theory
    - assign_morphism idempotence

TestConsistencyCheck (8 tests)
    - two identical theories → consistent, gap=0
    - two divergent theories → inconsistent, gap > tolerance
    - theory with no laws → gap=inf, inconsistent
    - tolerance boundary

TestBlameTheory (10 tests)
    - single theory, single wrong morphism → blame points at that morphism
    - theory with 10 morphisms, 1 wrong → blame points at the 1 wrong one
    - blame returns morph_id (not theory_id!) — the key defect probe target
    - no candidate theories → None
    - all predictions correct → blame returns smallest-error morphism

TestBitterLessonCage (3 tests)
    - 10 anonymous symbol seeds: register, predict, check produce identical results
    - variance of predicted k < 1e-3
    - cage: blame_theory finds wrong morphism across all seeds

TestDefectProbe (5 tests)
    Probe 1: blame_theory returns BlameResult.morph_id, not BlameResult.theory_id
    Probe 2: 10-morphism theory, exactly 1 wrong → blame finds the wrong one
    Probe 3: consistency_check with identical theories gives gap ≈ 0
    Probe 4: consistency_check with theories predicting opposite constants → inconsistent
    Probe 5: predict_under_theory returns None for a theory with no FITTED_LAW morphisms
"""
from __future__ import annotations

import math
import random
import string
from typing import Any

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import node, atom, var, Expr
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import (
    FittedLaw,
    add_fitted_law,
)
from experiments.symbolic_ai_v2.ctkg.inference.theory import (
    TheoryManager,
    ConsistencyResult,
    BlameResult,
    TheoryId,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_mg() -> MorphismGraph:
    return MorphismGraph()


def _mul_ctx() -> tuple[EvalContext, int]:
    """EvalContext with 'mul' → a*b; returns (ctx, mul_nid)."""
    mul_nid = TOKEN_GRAPH.encode("mul")
    ctx = EvalContext({mul_nid: lambda a, b: a * b})
    return ctx, mul_nid


def _make_fitted_law(mul_nid: int, k: float) -> FittedLaw:
    """Build a FittedLaw for  out = k * x  (linear in x, constant k).

    formula is  mul_nid(atom("k"), var("x")); k is stored in .params.
    We use Expr directly because mul_nid is already a NodeId (int).
    """
    formula = Expr(head=mul_nid, args=(atom("k"), var("x")))
    schema = SchematicLaw(
        pattern=formula,
        conclusion=formula,
        params=frozenset(),
        variables=frozenset(["x"]),
        evidence=1,
    )
    return FittedLaw(schema=schema, params={"k": k}, residual=0.0)


def _register_law_in_theory(mg: MorphismGraph, tm: TheoryManager,
                              theory_id: TheoryId, law: FittedLaw,
                              label: str) -> int:
    """Store a FittedLaw in mg and return its morph_id."""
    mid = add_fitted_law(mg, label, law)
    tm.assign_morphism(mid, theory_id)
    return mid


# ---------------------------------------------------------------------------
# TestTheoryManagerBasic
# ---------------------------------------------------------------------------

class TestTheoryManagerBasic:

    def test_register_returns_theory_id(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        tid = tm.register_theory("Newton")
        assert isinstance(tid, int)

    def test_theory_name_roundtrip(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        tid = tm.register_theory("Hooke")
        assert tm.theory_name(tid) == "Hooke"

    def test_unknown_theory_name_is_none(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        assert tm.theory_name(9999) is None

    def test_predict_empty_theory_returns_none(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        tid = tm.register_theory("Empty")
        ctx, _ = _mul_ctx()
        result = tm.predict_under_theory(tid, {"x": 3.0}, ctx)
        assert result is None

    def test_predict_single_law(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        law = _make_fitted_law(mul_nid, k=5.0)
        tid = tm.register_theory("Spring", morph_ids=[])
        _register_law_in_theory(mg, tm, tid, law, "spring_law")
        pred = tm.predict_under_theory(tid, {"x": 2.0}, ctx)
        assert pred == pytest.approx(10.0, rel=1e-6)

    def test_predict_two_laws_mean(self):
        """Two laws k=4 and k=6 → mean = 5 for x=1."""
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        tid = tm.register_theory("TwoLaw")
        _register_law_in_theory(mg, tm, tid, _make_fitted_law(mul_nid, 4.0), "law4")
        _register_law_in_theory(mg, tm, tid, _make_fitted_law(mul_nid, 6.0), "law6")
        pred = tm.predict_under_theory(tid, {"x": 1.0}, ctx)
        assert pred == pytest.approx(5.0, rel=1e-6)

    def test_assign_morphism_idempotent(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        tid = tm.register_theory("T")
        law = _make_fitted_law(mul_nid, 3.0)
        mid = add_fitted_law(mg, "law3", law)
        tm.assign_morphism(mid, tid)
        tm.assign_morphism(mid, tid)  # second call should not duplicate
        members = mg.theory_members(tid)
        assert members.count(mid) == 1

    def test_assign_morphism_adds_to_existing(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        tid = tm.register_theory("T")
        law1 = _make_fitted_law(mul_nid, 1.0)
        law2 = _make_fitted_law(mul_nid, 2.0)
        mid1 = add_fitted_law(mg, "l1", law1)
        mid2 = add_fitted_law(mg, "l2", law2)
        tm.assign_morphism(mid1, tid)
        tm.assign_morphism(mid2, tid)
        members = mg.theory_members(tid)
        assert mid1 in members
        assert mid2 in members

    def test_all_theories_returns_registered(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        t1 = tm.register_theory("Alpha")
        t2 = tm.register_theory("Beta")
        ids = {tid for tid, _ in tm.all_theories()}
        assert t1 in ids
        assert t2 in ids

    def test_multiple_theories_independent(self):
        """Laws in T1 don't contaminate T2's predictions."""
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t1 = tm.register_theory("T1")
        t2 = tm.register_theory("T2")
        _register_law_in_theory(mg, tm, t1, _make_fitted_law(mul_nid, 10.0), "l10")
        _register_law_in_theory(mg, tm, t2, _make_fitted_law(mul_nid, 99.0), "l99")
        p1 = tm.predict_under_theory(t1, {"x": 1.0}, ctx)
        p2 = tm.predict_under_theory(t2, {"x": 1.0}, ctx)
        assert p1 == pytest.approx(10.0)
        assert p2 == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# TestConsistencyCheck
# ---------------------------------------------------------------------------

class TestConsistencyCheck:

    def test_identical_theories_consistent(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        law = _make_fitted_law(mul_nid, 7.0)
        t1 = tm.register_theory("T1")
        t2 = tm.register_theory("T2")
        _register_law_in_theory(mg, tm, t1, law, "ta")
        # Add the same law independently to t2
        law2 = _make_fitted_law(mul_nid, 7.0)
        _register_law_in_theory(mg, tm, t2, law2, "tb")
        result = tm.consistency_check(t1, t2, {"x": 3.0}, ctx)
        assert result.consistent is True
        assert result.gap == pytest.approx(0.0, abs=1e-9)

    def test_divergent_theories_inconsistent(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t1 = tm.register_theory("T1")
        t2 = tm.register_theory("T2")
        _register_law_in_theory(mg, tm, t1, _make_fitted_law(mul_nid, 1.0), "l1")
        _register_law_in_theory(mg, tm, t2, _make_fitted_law(mul_nid, 100.0), "l2")
        result = tm.consistency_check(t1, t2, {"x": 1.0}, ctx)
        assert result.consistent is False
        assert result.gap == pytest.approx(99.0, rel=1e-6)

    def test_empty_theory_gives_inf_gap(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t1 = tm.register_theory("T1")
        t2 = tm.register_theory("T2")
        _register_law_in_theory(mg, tm, t1, _make_fitted_law(mul_nid, 5.0), "l5")
        result = tm.consistency_check(t1, t2, {"x": 1.0}, ctx)
        assert result.gap == float("inf")
        assert result.consistent is False

    def test_result_carries_theory_ids(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t1 = tm.register_theory("A")
        t2 = tm.register_theory("B")
        result = tm.consistency_check(t1, t2, {}, ctx)
        assert result.theory_a_id == t1
        assert result.theory_b_id == t2

    def test_tolerance_boundary_consistent(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t1 = tm.register_theory("T1")
        t2 = tm.register_theory("T2")
        _register_law_in_theory(mg, tm, t1, _make_fitted_law(mul_nid, 5.0), "la")
        _register_law_in_theory(mg, tm, t2, _make_fitted_law(mul_nid, 5.0 + 1e-7), "lb")
        result = tm.consistency_check(t1, t2, {"x": 1.0}, ctx, tolerance=1e-6)
        assert result.consistent is True

    def test_tolerance_boundary_inconsistent(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t1 = tm.register_theory("T1")
        t2 = tm.register_theory("T2")
        _register_law_in_theory(mg, tm, t1, _make_fitted_law(mul_nid, 5.0), "lc")
        _register_law_in_theory(mg, tm, t2, _make_fitted_law(mul_nid, 5.0 + 1e-5), "ld")
        result = tm.consistency_check(t1, t2, {"x": 1.0}, ctx, tolerance=1e-6)
        assert result.consistent is False

    def test_pred_a_b_populated(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t1 = tm.register_theory("T1")
        t2 = tm.register_theory("T2")
        _register_law_in_theory(mg, tm, t1, _make_fitted_law(mul_nid, 3.0), "l3")
        _register_law_in_theory(mg, tm, t2, _make_fitted_law(mul_nid, 4.0), "l4")
        result = tm.consistency_check(t1, t2, {"x": 2.0}, ctx)
        assert result.pred_a == pytest.approx(6.0)
        assert result.pred_b == pytest.approx(8.0)

    def test_same_theory_self_consistent(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t1 = tm.register_theory("T1")
        _register_law_in_theory(mg, tm, t1, _make_fitted_law(mul_nid, 7.0), "l7")
        result = tm.consistency_check(t1, t1, {"x": 5.0}, ctx)
        assert result.consistent is True
        assert result.gap == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# TestBlameTheory
# ---------------------------------------------------------------------------

class TestBlameTheory:

    def test_single_wrong_morphism_blamed(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t = tm.register_theory("T")
        mid = _register_law_in_theory(mg, tm, t, _make_fitted_law(mul_nid, 99.0), "wrong")
        result = tm.blame_theory([t], {"x": 1.0}, observed=0.0, ctx=ctx)
        assert result is not None
        assert result.morph_id == mid

    def test_blame_returns_blame_result_type(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t = tm.register_theory("T")
        _register_law_in_theory(mg, tm, t, _make_fitted_law(mul_nid, 5.0), "l")
        result = tm.blame_theory([t], {"x": 1.0}, observed=0.0, ctx=ctx)
        assert isinstance(result, BlameResult)

    def test_blame_morph_id_is_fitted_law_morphism(self):
        """KEY DEFECT PROBE: blame.morph_id must be the FITTED_LAW morphism id."""
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t = tm.register_theory("T")
        mid = _register_law_in_theory(mg, tm, t, _make_fitted_law(mul_nid, 5.0), "lm")
        result = tm.blame_theory([t], {"x": 1.0}, observed=0.0, ctx=ctx)
        assert result is not None
        # morph_id must be the id of the FITTED_LAW morphism
        assert result.morph_id == mid
        blamed = mg.morphism_by_id(result.morph_id)
        assert blamed is not None
        assert blamed.morph_type == "FITTED_LAW"

    def test_blame_error_field(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t = tm.register_theory("T")
        _register_law_in_theory(mg, tm, t, _make_fitted_law(mul_nid, 10.0), "l10")
        result = tm.blame_theory([t], {"x": 1.0}, observed=0.0, ctx=ctx)
        assert result.error == pytest.approx(10.0)

    def test_blame_no_candidates_returns_none(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, _ = _mul_ctx()
        result = tm.blame_theory([], {"x": 1.0}, observed=5.0, ctx=ctx)
        assert result is None

    def test_blame_empty_theories_returns_none(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, _ = _mul_ctx()
        t1 = tm.register_theory("Empty1")
        t2 = tm.register_theory("Empty2")
        result = tm.blame_theory([t1, t2], {"x": 1.0}, observed=5.0, ctx=ctx)
        assert result is None

    def test_blame_all_correct_returns_minimum_error(self):
        """All morphisms correct → blame returns the one with the smallest error."""
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t = tm.register_theory("T")
        mid1 = _register_law_in_theory(mg, tm, t, _make_fitted_law(mul_nid, 5.0), "e5")
        mid2 = _register_law_in_theory(mg, tm, t, _make_fitted_law(mul_nid, 5.1), "e5.1")
        # observed=5.0, x=1 → pred1=5.0 (error=0), pred2=5.1 (error=0.1)
        # blame should return mid2 (higher error)
        result = tm.blame_theory([t], {"x": 1.0}, observed=5.0, ctx=ctx)
        assert result.morph_id == mid2

    def test_blame_across_two_theories(self):
        """Two theories; one morphism is very wrong → that one gets blamed."""
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t1 = tm.register_theory("T1")
        t2 = tm.register_theory("T2")
        _register_law_in_theory(mg, tm, t1, _make_fitted_law(mul_nid, 5.0), "g5")
        bad_mid = _register_law_in_theory(mg, tm, t2, _make_fitted_law(mul_nid, 500.0), "g500")
        result = tm.blame_theory([t1, t2], {"x": 1.0}, observed=5.0, ctx=ctx)
        assert result.morph_id == bad_mid
        assert result.theory_id == t2

    def test_blame_all_returns_sorted_list(self):
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t = tm.register_theory("T")
        _register_law_in_theory(mg, tm, t, _make_fitted_law(mul_nid, 1.0), "h1")
        _register_law_in_theory(mg, tm, t, _make_fitted_law(mul_nid, 5.0), "h5")
        _register_law_in_theory(mg, tm, t, _make_fitted_law(mul_nid, 10.0), "h10")
        results = tm.blame_theory_all([t], {"x": 1.0}, observed=0.0, ctx=ctx)
        errors = [r.error for r in results]
        assert errors == sorted(errors, reverse=True)

    def test_blame_theory_id_field(self):
        """BlameResult.theory_id correctly reflects which theory the morphism came from."""
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, mul_nid = _mul_ctx()
        t1 = tm.register_theory("Good")
        t2 = tm.register_theory("Bad")
        _register_law_in_theory(mg, tm, t1, _make_fitted_law(mul_nid, 1.0), "j1")
        _register_law_in_theory(mg, tm, t2, _make_fitted_law(mul_nid, 999.0), "j999")
        result = tm.blame_theory([t1, t2], {"x": 1.0}, observed=1.0, ctx=ctx)
        assert result.theory_id == t2


# ---------------------------------------------------------------------------
# TestBitterLessonCage
# ---------------------------------------------------------------------------

def _anon_mul_ctx(seed: int) -> tuple[EvalContext, int]:
    """EvalContext with an anonymous operator symbol (from Unicode block U+2200+)."""
    rng = random.Random(seed)
    symbol = chr(0x2200 + rng.randint(0, 0xFF))
    nid = TOKEN_GRAPH.encode(symbol)
    ctx = EvalContext({nid: lambda a, b: a * b})
    return ctx, nid


class TestBitterLessonCage:

    def test_cage_predict_consistent_across_seeds(self):
        """k=7.0, x=3.0 → prediction 21.0 regardless of operator symbol."""
        preds = []
        for seed in range(10):
            ctx, nid = _anon_mul_ctx(seed)
            mg = _fresh_mg()
            tm = TheoryManager(mg)
            law = _make_fitted_law(nid, k=7.0)
            tid = tm.register_theory("anon")
            _register_law_in_theory(mg, tm, tid, law, f"law_s{seed}")
            pred = tm.predict_under_theory(tid, {"x": 3.0}, ctx)
            preds.append(pred)
        # All seeds produce 21.0
        assert all(p == pytest.approx(21.0, rel=1e-6) for p in preds)

    def test_cage_consistency_zero_variance(self):
        """Two theories with k=5 and k=5 give gap=0 across all seeds."""
        gaps = []
        for seed in range(10):
            ctx, nid = _anon_mul_ctx(seed)
            mg = _fresh_mg()
            tm = TheoryManager(mg)
            t1 = tm.register_theory("A")
            t2 = tm.register_theory("B")
            _register_law_in_theory(mg, tm, t1, _make_fitted_law(nid, 5.0), "ca")
            _register_law_in_theory(mg, tm, t2, _make_fitted_law(nid, 5.0), "cb")
            r = tm.consistency_check(t1, t2, {"x": 2.0}, ctx)
            gaps.append(r.gap)
        assert all(g == pytest.approx(0.0, abs=1e-9) for g in gaps)

    def test_cage_blame_finds_wrong_morphism(self):
        """blame_theory correctly identifies the bad morphism across all seeds."""
        for seed in range(10):
            ctx, nid = _anon_mul_ctx(seed)
            mg = _fresh_mg()
            tm = TheoryManager(mg)
            t = tm.register_theory("theory")
            # 9 good morphisms (k ≈ 5) and 1 bad (k = 500)
            for i in range(9):
                _register_law_in_theory(mg, tm, t, _make_fitted_law(nid, 5.0), f"g{seed}_{i}")
            bad_mid = _register_law_in_theory(mg, tm, t, _make_fitted_law(nid, 500.0), f"bad_{seed}")
            result = tm.blame_theory([t], {"x": 1.0}, observed=5.0, ctx=ctx)
            assert result is not None, f"seed {seed}: blame returned None"
            assert result.morph_id == bad_mid, f"seed {seed}: wrong morphism blamed"


# ---------------------------------------------------------------------------
# TestDefectProbe
# ---------------------------------------------------------------------------

class TestDefectProbe:

    def test_probe1_morph_id_points_at_fitted_law(self):
        """blame_theory must return .morph_id that is a FITTED_LAW morphism.

        A naïve implementation might return the theory object id instead.
        We verify the returned morph_id is an actual FITTED_LAW morphism.
        """
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, nid = _mul_ctx()
        t = tm.register_theory("probe1")
        mid = _register_law_in_theory(mg, tm, t, _make_fitted_law(nid, 10.0), "p1law")
        result = tm.blame_theory([t], {"x": 1.0}, observed=0.0, ctx=ctx)
        assert result.morph_id == mid          # must be the specific FITTED_LAW morphism
        blamed_morph = mg.morphism_by_id(mid)
        assert blamed_morph is not None
        assert blamed_morph.morph_type == "FITTED_LAW"
        assert result.theory_id == t

    def test_probe2_ten_morphisms_one_wrong(self):
        """A theory with 10 morphisms, exactly 1 wrong: blame points at the wrong one."""
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, nid = _mul_ctx()
        t = tm.register_theory("probe2")
        mids = []
        for i in range(9):
            mid = _register_law_in_theory(mg, tm, t, _make_fitted_law(nid, 5.0), f"p2g_{i}")
            mids.append(mid)
        # The one wrong morphism
        bad_mid = _register_law_in_theory(mg, tm, t, _make_fitted_law(nid, 500.0), "p2_bad")
        result = tm.blame_theory([t], {"x": 1.0}, observed=5.0, ctx=ctx)
        assert result.morph_id == bad_mid

    def test_probe3_self_consistency(self):
        """A theory compared against itself must always be consistent (gap=0)."""
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, nid = _mul_ctx()
        t = tm.register_theory("probe3")
        _register_law_in_theory(mg, tm, t, _make_fitted_law(nid, 42.0), "p3law")
        result = tm.consistency_check(t, t, {"x": 2.0}, ctx)
        assert result.consistent is True
        assert result.gap == pytest.approx(0.0, abs=1e-9)

    def test_probe4_opposite_theories_inconsistent(self):
        """Two theories predicting +1 and -1 must be flagged as inconsistent."""
        mg = _fresh_mg()
        tm = TheoryManager(mg)
        nid = TOKEN_GRAPH.encode("neg_mul")
        ctx_neg = EvalContext({nid: lambda a, b: a * b * -1})
        ctx_pos, pos_nid = _mul_ctx()

        ta = tm.register_theory("Positive")
        tb = tm.register_theory("Negative")
        mg2 = mg  # same graph for both

        # Positive theory: k=1, x=1 → pred=1
        law_pos = _make_fitted_law(pos_nid, 1.0)
        mid_pos = add_fitted_law(mg, "pos", law_pos)
        tm.assign_morphism(mid_pos, ta)

        # Negative theory: uses neg_mul → pred = -1
        formula_neg = Expr(head=nid, args=(atom("k"), var("x")))
        schema_neg = SchematicLaw(
            pattern=formula_neg, conclusion=formula_neg,
            params=frozenset(), variables=frozenset(["x"]), evidence=1,
        )
        law_neg = FittedLaw(schema=schema_neg, params={"k": 1.0}, residual=0.0)
        mid_neg = add_fitted_law(mg, "neg", law_neg)
        tm.assign_morphism(mid_neg, tb)

        ctx_combined = EvalContext({
            pos_nid: lambda a, b: a * b,
            nid: lambda a, b: a * b * -1,
        })
        result = tm.consistency_check(ta, tb, {"x": 1.0}, ctx_combined)
        assert result.consistent is False
        assert result.gap == pytest.approx(2.0, rel=1e-6)

    def test_probe5_no_fitted_law_returns_none(self):
        """A theory containing only non-FITTED_LAW morphisms gives predict=None."""
        from experiments.symbolic_ai_v2.ctkg.core.expr_law import add_expr_law
        from experiments.symbolic_ai_v2.ctkg.core.expr_law import ExprLaw
        from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom as _atom

        mg = _fresh_mg()
        tm = TheoryManager(mg)
        ctx, _ = _mul_ctx()
        t = tm.register_theory("probe5")

        # Add an EXPR_LAW morphism (not FITTED_LAW)
        p = _atom("a")
        expr_law = add_expr_law(mg, "probe5_expr", p, p)
        tm.assign_morphism(expr_law.morph_id, t)

        result = tm.predict_under_theory(t, {"x": 1.0}, ctx)
        assert result is None
