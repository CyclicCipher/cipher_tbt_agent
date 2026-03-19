"""
Phase 15 Integration Test — Symmetry-Conflict Latent Hypothesis Generation.

Tests:
  - compare_symmetry_groups finds conflicts correctly
  - hypothesise_from_symmetry_conflict generates correct ConflictLatent
  - Defect probes verify Iron Law compliance
"""
from __future__ import annotations

import math
import random

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import FittedLaw, add_fitted_law
from experiments.symbolic_ai_v2.ctkg.core.prim_ops import make_prim_ctx
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom, var, Expr
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager, TheoryId
from experiments.symbolic_ai_v2.ctkg.inference.latent_conflict import (
    hypothesise_from_symmetry_conflict,
    ConflictLatent,
)
from experiments.symbolic_ai_v2.ctkg.inference.theory import SymmetryConflict


def _make_mg_tm():
    mg = MorphismGraph()
    tm = TheoryManager(mg)
    return mg, tm


def _make_linear_law(mg, nid: int, k: float) -> FittedLaw:
    """Create a FittedLaw for y = k * x."""
    formula = Expr(head=nid, args=(atom("p0"), var("x")))
    schema = SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset({"p0"}), variables=frozenset(["x"]), evidence=5,
    )
    return FittedLaw(schema=schema, params={"p0": k}, residual=0.0)


def _make_theory_with_law(mg, tm, k: float, seed: int = 0) -> TheoryId:
    """Register a theory with y = k * x law."""
    nid = TOKEN_GRAPH.encode(f"TEST_MUL_SEED{seed}_K{k}")
    law = _make_linear_law(mg, nid, k)
    mid = add_fitted_law(mg, f"law_k{k}_s{seed}", law)
    tid = tm.register_theory(f"T_k{k}_s{seed}")
    tm.assign_morphism(mid, tid)
    return tid


# ---------------------------------------------------------------------------
# TestSymmetryConflict
# ---------------------------------------------------------------------------

class TestSymmetryConflict:

    def test_compare_symmetry_groups_finds_conflict(self):
        """T_A invariant under G (scale x by 2), T_B not → a_only contains G."""
        mg, tm = _make_mg_tm()
        ctx = make_prim_ctx()

        # T_A: y = 1 * x (scale-invariant: f(2x) = 2x = 2*f(x))
        # But symmetry here is about PREDICTION invariance: f(T(x)) == f(x)
        # For y=x: f(2x) = 2x ≠ x, so it's NOT trivially symmetric
        # We need a theory where the PREDICTION doesn't change under T

        # T_A: y = 1.0 (constant) → invariant under any input transform
        nid_a = TOKEN_GRAPH.encode("TEST_CONST_A")
        formula_a = atom("1.0")
        schema_a = SchematicLaw(
            pattern=formula_a, conclusion=formula_a,
            params=frozenset(), variables=frozenset(["x"]), evidence=5,
        )
        law_a = FittedLaw(schema=schema_a, params={}, residual=0.0)
        mid_a = add_fitted_law(mg, "law_const_a", law_a)
        tid_a = tm.register_theory("T_const_a")
        tm.assign_morphism(mid_a, tid_a)

        # T_B: y = x (linear) → NOT constant, prediction changes under scale
        nid_b = TOKEN_GRAPH.encode("TEST_LINEAR_B")
        formula_b = var("x")
        schema_b = SchematicLaw(
            pattern=formula_b, conclusion=formula_b,
            params=frozenset(), variables=frozenset(["x"]), evidence=5,
        )
        law_b = FittedLaw(schema=schema_b, params={}, residual=0.0)
        mid_b = add_fitted_law(mg, "law_linear_b", law_b)
        tid_b = tm.register_theory("T_linear_b")
        tm.assign_morphism(mid_b, tid_b)

        # Transform G: scale x by 2
        G = lambda inp: {"x": inp["x"] * 2.0}
        test_inputs = [{"x": float(i)} for i in range(1, 6)]

        conflict = tm.compare_symmetry_groups(tid_a, tid_b, [G], ctx, test_inputs, tolerance=0.01)

        # T_A (constant) is invariant under G, T_B (linear) is NOT
        assert len(conflict.a_only) == 1, (
            f"Expected 1 a_only transform, got {len(conflict.a_only)}"
        )
        assert conflict.has_conflict

    def test_compare_symmetry_groups_no_conflict_when_shared(self):
        """Both theories invariant under G → a_only=[], b_only=[]."""
        mg, tm = _make_mg_tm()
        ctx = make_prim_ctx()

        # Both constant theories → both invariant under any transform
        for name, k in [("T_const_1", "1.0"), ("T_const_2", "1.0")]:
            nid = TOKEN_GRAPH.encode(f"TEST_CONST_{name}")
            formula = atom(k)
            schema = SchematicLaw(
                pattern=formula, conclusion=formula,
                params=frozenset(), variables=frozenset(["x"]), evidence=5,
            )
            law = FittedLaw(schema=schema, params={}, residual=0.0)
            mid = add_fitted_law(mg, f"law_{name}", law)
            tid = tm.register_theory(name)
            tm.assign_morphism(mid, tid)

        theories = {n: tid for tid, n in tm.all_theories()}
        tid_a = theories["T_const_1"]
        tid_b = theories["T_const_2"]

        G = lambda inp: {"x": inp["x"] * 2.0}
        test_inputs = [{"x": float(i)} for i in range(1, 6)]
        conflict = tm.compare_symmetry_groups(tid_a, tid_b, [G], ctx, test_inputs)
        assert not conflict.has_conflict
        assert len(conflict.a_only) == 0
        assert len(conflict.b_only) == 0

    def test_compare_symmetry_groups_b_only(self):
        """T_B invariant under G, T_A not → b_only contains G."""
        mg, tm = _make_mg_tm()
        ctx = make_prim_ctx()

        # T_A: y = x (linear, NOT invariant under scale)
        nid_a = TOKEN_GRAPH.encode("TEST_LINEAR_A2")
        formula_a = var("x")
        schema_a = SchematicLaw(
            pattern=formula_a, conclusion=formula_a,
            params=frozenset(), variables=frozenset(["x"]), evidence=5,
        )
        law_a = FittedLaw(schema=schema_a, params={}, residual=0.0)
        mid_a = add_fitted_law(mg, "law_linear_a2", law_a)
        tid_a = tm.register_theory("T_linear_a2")
        tm.assign_morphism(mid_a, tid_a)

        # T_B: y = 1.0 (constant, invariant)
        nid_b = TOKEN_GRAPH.encode("TEST_CONST_B2")
        formula_b = atom("1.0")
        schema_b = SchematicLaw(
            pattern=formula_b, conclusion=formula_b,
            params=frozenset(), variables=frozenset(["x"]), evidence=5,
        )
        law_b = FittedLaw(schema=schema_b, params={}, residual=0.0)
        mid_b = add_fitted_law(mg, "law_const_b2", law_b)
        tid_b = tm.register_theory("T_const_b2")
        tm.assign_morphism(mid_b, tid_b)

        G = lambda inp: {"x": inp["x"] * 2.0}
        test_inputs = [{"x": float(i)} for i in range(1, 6)]
        conflict = tm.compare_symmetry_groups(tid_a, tid_b, [G], ctx, test_inputs, tolerance=0.01)

        assert len(conflict.b_only) == 1, (
            f"Expected 1 b_only transform, got {len(conflict.b_only)}"
        )
        assert conflict.has_conflict


# ---------------------------------------------------------------------------
# TestConflictLatent
# ---------------------------------------------------------------------------

class TestConflictLatent:

    def _make_conflict_with_a_only(self, mg, tm, ctx):
        """Build a scenario where a_only is non-empty."""
        # T_A = constant (invariant), T_B = linear (not invariant)
        nid_a = TOKEN_GRAPH.encode("CLT_A_CONST")
        formula_a = atom("1.0")
        schema_a = SchematicLaw(
            pattern=formula_a, conclusion=formula_a,
            params=frozenset(), variables=frozenset(["x"]), evidence=5,
        )
        law_a = FittedLaw(schema=schema_a, params={}, residual=0.0)
        mid_a = add_fitted_law(mg, "clt_law_a", law_a)
        tid_a = tm.register_theory("CLT_T_A")
        tm.assign_morphism(mid_a, tid_a)

        nid_b = TOKEN_GRAPH.encode("CLT_B_LINEAR")
        formula_b = var("x")
        schema_b = SchematicLaw(
            pattern=formula_b, conclusion=formula_b,
            params=frozenset(), variables=frozenset(["x"]), evidence=5,
        )
        law_b = FittedLaw(schema=schema_b, params={}, residual=0.0)
        mid_b = add_fitted_law(mg, "clt_law_b", law_b)
        tid_b = tm.register_theory("CLT_T_B")
        tm.assign_morphism(mid_b, tid_b)

        G = lambda inp: {"x": inp["x"] * 2.0}
        test_inputs = [{"x": float(i)} for i in range(1, 6)]
        return tm.compare_symmetry_groups(tid_a, tid_b, [G], ctx, test_inputs, tolerance=0.01), tid_a, tid_b

    def test_hypothesise_generates_morphism(self):
        """Non-empty a_only → ConflictLatent with non-None hypothesis_morphism."""
        mg, tm = _make_mg_tm()
        ctx = make_prim_ctx()
        conflict, tid_a, tid_b = self._make_conflict_with_a_only(mg, tm, ctx)

        result = hypothesise_from_symmetry_conflict(conflict, tid_a, tid_b, mg, tm)
        assert result is not None
        assert result.hypothesis_morphism is not None
        assert result.hypothesis_morphism != -1

    def test_hypothesise_returns_none_when_no_conflict(self):
        """Empty a_only → returns None."""
        conflict = SymmetryConflict(a_only=[], b_only=[], shared=[])
        mg, tm = _make_mg_tm()
        tid_a = tm.register_theory("empty_a")
        tid_b = tm.register_theory("empty_b")
        result = hypothesise_from_symmetry_conflict(conflict, tid_a, tid_b, mg, tm)
        assert result is None

    def test_morphism_type_is_latent_concept(self):
        """Returned morphism has morph_type='LATENT_CONCEPT'."""
        mg, tm = _make_mg_tm()
        ctx = make_prim_ctx()
        conflict, tid_a, tid_b = self._make_conflict_with_a_only(mg, tm, ctx)

        result = hypothesise_from_symmetry_conflict(conflict, tid_a, tid_b, mg, tm)
        assert result is not None
        morph = mg.morphism_by_id(result.hypothesis_morphism)
        assert morph is not None
        assert morph.morph_type == "LATENT_CONCEPT"


# ---------------------------------------------------------------------------
# TestPhase15Cage
# ---------------------------------------------------------------------------

class TestPhase15Cage:

    def test_cage_5_seeds(self):
        """Anonymous theories with scale-x transform conflict → ConflictLatent for all 5 seeds."""
        ctx = make_prim_ctx()
        for seed in range(5):
            mg, tm = _make_mg_tm()

            # T_A = constant (always 1.0)
            nid_a = TOKEN_GRAPH.encode(f"CAGE_CONST_S{seed}")
            formula_a = atom("1.0")
            schema_a = SchematicLaw(
                pattern=formula_a, conclusion=formula_a,
                params=frozenset(), variables=frozenset(["x"]), evidence=5,
            )
            law_a = FittedLaw(schema=schema_a, params={}, residual=0.0)
            mid_a = add_fitted_law(mg, f"cage_law_a_s{seed}", law_a)
            tid_a = tm.register_theory(f"cage_T_A_s{seed}")
            tm.assign_morphism(mid_a, tid_a)

            # T_B = linear y = x
            nid_b = TOKEN_GRAPH.encode(f"CAGE_LINEAR_S{seed}")
            formula_b = var("x")
            schema_b = SchematicLaw(
                pattern=formula_b, conclusion=formula_b,
                params=frozenset(), variables=frozenset(["x"]), evidence=5,
            )
            law_b = FittedLaw(schema=schema_b, params={}, residual=0.0)
            mid_b = add_fitted_law(mg, f"cage_law_b_s{seed}", law_b)
            tid_b = tm.register_theory(f"cage_T_B_s{seed}")
            tm.assign_morphism(mid_b, tid_b)

            G = lambda inp: {"x": inp["x"] * 2.0}
            test_inputs = [{"x": float(i)} for i in range(1, 6)]
            conflict = tm.compare_symmetry_groups(
                tid_a, tid_b, [G], ctx, test_inputs, tolerance=0.01
            )
            result = hypothesise_from_symmetry_conflict(conflict, tid_a, tid_b, mg, tm)
            assert result is not None, f"seed={seed}: expected ConflictLatent, got None"


# ---------------------------------------------------------------------------
# TestPhase15DefectProbes
# ---------------------------------------------------------------------------

class TestPhase15DefectProbes:

    def test_probe1_no_latent_when_groups_match(self):
        """Both theories invariant under all transforms → returns None."""
        mg, tm = _make_mg_tm()
        ctx = make_prim_ctx()

        for name in ["D1A", "D1B"]:
            nid = TOKEN_GRAPH.encode(f"PROBE1_CONST_{name}")
            formula = atom("1.0")
            schema = SchematicLaw(
                pattern=formula, conclusion=formula,
                params=frozenset(), variables=frozenset(["x"]), evidence=5,
            )
            law = FittedLaw(schema=schema, params={}, residual=0.0)
            mid = add_fitted_law(mg, f"p1_law_{name}", law)
            tid = tm.register_theory(f"D1_{name}")
            tm.assign_morphism(mid, tid)

        theories = {n: tid for tid, n in tm.all_theories()}
        tid_a = theories["D1_D1A"]
        tid_b = theories["D1_D1B"]

        G1 = lambda inp: {"x": inp["x"] * 2.0}
        G2 = lambda inp: {"x": inp["x"] + 1.0}
        test_inputs = [{"x": float(i)} for i in range(1, 6)]
        conflict = tm.compare_symmetry_groups(tid_a, tid_b, [G1, G2], ctx, test_inputs)
        result = hypothesise_from_symmetry_conflict(conflict, tid_a, tid_b, mg, tm)
        assert result is None, "Both theories fully symmetric → no latent needed"

    def test_probe2_structural_not_nominal(self):
        """Two independent setups both produce ConflictLatent with LATENT_CONCEPT type."""
        ctx = make_prim_ctx()

        results = []
        for trial in range(2):
            mg, tm = _make_mg_tm()

            nid_a = TOKEN_GRAPH.encode(f"P2_CONST_{trial}")
            formula_a = atom("1.0")
            schema_a = SchematicLaw(
                pattern=formula_a, conclusion=formula_a,
                params=frozenset(), variables=frozenset(["x"]), evidence=5,
            )
            law_a = FittedLaw(schema=schema_a, params={}, residual=0.0)
            mid_a = add_fitted_law(mg, f"p2_law_a_{trial}", law_a)
            tid_a = tm.register_theory(f"P2_A_{trial}")
            tm.assign_morphism(mid_a, tid_a)

            nid_b = TOKEN_GRAPH.encode(f"P2_LINEAR_{trial}")
            formula_b = var("x")
            schema_b = SchematicLaw(
                pattern=formula_b, conclusion=formula_b,
                params=frozenset(), variables=frozenset(["x"]), evidence=5,
            )
            law_b = FittedLaw(schema=schema_b, params={}, residual=0.0)
            mid_b = add_fitted_law(mg, f"p2_law_b_{trial}", law_b)
            tid_b = tm.register_theory(f"P2_B_{trial}")
            tm.assign_morphism(mid_b, tid_b)

            G = lambda inp: {"x": inp["x"] * 3.0}
            test_inputs = [{"x": float(i)} for i in range(1, 6)]
            conflict = tm.compare_symmetry_groups(
                tid_a, tid_b, [G], ctx, test_inputs, tolerance=0.01
            )
            result = hypothesise_from_symmetry_conflict(conflict, tid_a, tid_b, mg, tm)
            assert result is not None
            morph = mg.morphism_by_id(result.hypothesis_morphism)
            assert morph.morph_type == "LATENT_CONCEPT"
            # Check payload has structural keys
            assert "a_only_count" in morph.payload
            assert "b_only_count" in morph.payload
            results.append(result)

        # Both should work independently (structural, not nominal)
        assert len(results) == 2
