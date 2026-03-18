"""
Tests for theory.py extensions — symmetry_group_check and cross_theory_inference.
Phase 9 Blockers 2 and 3.
"""
from __future__ import annotations

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import FittedLaw, add_fitted_law
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom, var, Expr
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager


def _mul_ctx():
    nid = TOKEN_GRAPH.encode("mul_ext")
    return EvalContext({nid: lambda a, b: a * b}), nid


def _fitted(nid: int, k: float) -> FittedLaw:
    formula = Expr(head=nid, args=(atom("k"), var("x")))
    sch = SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(), variables=frozenset(["x"]), evidence=1,
    )
    return FittedLaw(schema=sch, params={"k": k}, residual=0.0)


class TestSymmetryGroupCheck:

    def test_identity_transform_is_invariant(self):
        ctx, nid = _mul_ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "law_k2", _fitted(nid, 2.0))
        tm.assign_morphism(mid, t)

        identity = lambda inp: inp
        test_inputs = [{"x": float(i)} for i in range(1, 6)]
        result = tm.symmetry_group_check(t, identity, ctx, test_inputs)
        assert result.invariant is True
        assert result.max_deviation < 1e-9

    def test_scaling_breaks_linear_law(self):
        """Scaling x by 2 should change prediction for f(x) = 2x."""
        ctx, nid = _mul_ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "law_k2b", _fitted(nid, 2.0))
        tm.assign_morphism(mid, t)

        def scale_x(inp):
            return {"x": inp["x"] * 2.0}

        test_inputs = [{"x": float(i)} for i in range(1, 6)]
        result = tm.symmetry_group_check(t, scale_x, ctx, test_inputs, tolerance=0.1)
        assert result.invariant is False
        assert result.max_deviation > 0.1

    def test_n_tested_counts_evaluable_inputs(self):
        ctx, nid = _mul_ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "law_k3", _fitted(nid, 3.0))
        tm.assign_morphism(mid, t)

        test_inputs = [{"x": float(i)} for i in range(5)]
        result = tm.symmetry_group_check(t, lambda inp: inp, ctx, test_inputs)
        assert result.n_tested == 5


class TestCrossTheoryInference:

    def test_basic_composition(self):
        """Combined prediction = pred_a + pred_b."""
        ctx, nid = _mul_ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        tA = tm.register_theory("A")
        tB = tm.register_theory("B")
        midA = add_fitted_law(mg, "A_k2", _fitted(nid, 2.0))
        midB = add_fitted_law(mg, "B_k3", _fitted(nid, 3.0))
        tm.assign_morphism(midA, tA)
        tm.assign_morphism(midB, tB)

        result = tm.cross_theory_inference(
            tA, tB,
            {"x": 5.0}, {"x": 5.0},
            lambda a, b: a + b,
            ctx,
        )
        assert result is not None
        # pred_a = 2*5 = 10, pred_b = 3*5 = 15 → combined = 25
        assert abs(result.prediction - 25.0) < 1e-9

    def test_returns_none_if_theory_empty(self):
        ctx, nid = _mul_ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        tA = tm.register_theory("A")
        tB = tm.register_theory("B")  # no morphisms
        midA = add_fitted_law(mg, "A_k2c", _fitted(nid, 2.0))
        tm.assign_morphism(midA, tA)

        result = tm.cross_theory_inference(
            tA, tB,
            {"x": 5.0}, {"x": 5.0},
            lambda a, b: a + b,
            ctx,
        )
        assert result is None

    def test_composition_fn_is_applied(self):
        """composition_fn=lambda a,b: a*b should multiply predictions."""
        ctx, nid = _mul_ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        tA = tm.register_theory("A")
        tB = tm.register_theory("B")
        midA = add_fitted_law(mg, "A_k4", _fitted(nid, 4.0))
        midB = add_fitted_law(mg, "B_k5", _fitted(nid, 5.0))
        tm.assign_morphism(midA, tA)
        tm.assign_morphism(midB, tB)

        result = tm.cross_theory_inference(
            tA, tB,
            {"x": 2.0}, {"x": 2.0},
            lambda a, b: a * b,
            ctx,
        )
        # pred_a = 4*2=8, pred_b = 5*2=10 → product = 80
        assert result is not None
        assert abs(result.prediction - 80.0) < 1e-9


class TestTheoryExtensionsCage:

    def test_cage_symmetry_check_10_seeds(self):
        """Symmetry check result is symbol-invariant across 10 seeds."""
        import random
        results = []
        for seed in range(10):
            rng = random.Random(seed)
            sym = chr(0x2200 + rng.randint(0, 0xFE))
            nid = TOKEN_GRAPH.encode(sym)
            ctx = EvalContext({nid: lambda a, b: a * b})
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            t = tm.register_theory("T")
            formula = Expr(head=nid, args=(atom("k"), var("x")))
            sch = SchematicLaw(
                pattern=formula, conclusion=formula,
                params=frozenset(), variables=frozenset(["x"]), evidence=1,
            )
            law = FittedLaw(schema=sch, params={"k": 2.0}, residual=0.0)
            from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import add_fitted_law as afl
            mid = afl(mg, f"seed_{seed}", law)
            tm.assign_morphism(mid, t)

            test_inputs = [{"x": float(i)} for i in range(1, 6)]
            res = tm.symmetry_group_check(t, lambda inp: inp, ctx, test_inputs)
            results.append(res.invariant)

        assert all(results), "symmetry_group_check not symbol-invariant"
