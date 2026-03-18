"""
Phase 4 Integration Test — Theory Compartments and Cross-Domain Consistency.

This test chains Phases 1–4 end-to-end:
  Phase 1 → ExprLaw (structural laws in the graph)
  Phase 2 → SchematicLaw / discover_parametric_law (symbolic parameter extraction)
  Phase 3 → FittedLaw / fit_law (numerical OLS fitting)
  Phase 4 → TheoryManager (compartments, consistency, blame)

Scenario
--------
Two competing physical theories are fitted from observations:

  Theory A — Hooke's Law:     F = k_A × x   (spring constant k_A ≈ 50)
  Theory B — Coulomb's Law:   F = k_B × x   (Coulomb constant k_B ≈ 200)

Both use the same structural formula (linear in x).  They agree when x is
small but diverge at large x.

Pipeline:
  1. Discover SchematicLaw from families of (x, F) observations.
  2. Fit FittedLaw for each theory using OLS.
  3. Register theories in TheoryManager.
  4. consistency_check on a shared observable: theories are inconsistent.
  5. blame_theory on an anomalous observation: exactly one morphism is blamed.

Test classes
------------
TestPhase4Integration (7 tests)
    - full pipeline: discover → fit → register → predict → check → blame
    - Hooke prediction correct
    - Coulomb prediction correct
    - cross-theory inconsistency
    - blame returns a BlameResult (not None)
    - blame.morph_id is a real morphism in the graph
    - blame points at the Coulomb morphism when Hooke observation is given

TestPhase4Cage (3 tests)
    - same pipeline with 10 anonymous operator seeds
    - k_A and k_B recovered within 1%
    - consistency_check flags divergence across all seeds

TestPhase4DefectProbe (3 tests)
    Probe A: blame.morph_id != blame.theory_id (morphism locality)
    Probe B: two-theory graph, only one wrong → blame finds it
    Probe C: theories with identical k are consistent (gap < 1e-6)
"""
from __future__ import annotations

import random

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import node, atom, var, Expr
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import (
    discover_parametric_law,
    SchematicLaw,
    instantiate,
)
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import (
    FittedLaw,
    fit_law,
    add_fitted_law,
)
from experiments.symbolic_ai_v2.ctkg.inference.theory import (
    TheoryManager,
    ConsistencyResult,
    BlameResult,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_mul_ctx(mul_name: str = "mul") -> tuple[EvalContext, int]:
    mul_nid = TOKEN_GRAPH.encode(mul_name)
    ctx = EvalContext({mul_nid: lambda a, b: a * b})
    return ctx, mul_nid


def _linear_schema(mul_nid: int) -> SchematicLaw:
    """Build a SchematicLaw for  out = k * x  where k is the parameter.

    Uses Expr directly because mul_nid is already a NodeId (int).
    """
    formula = Expr(head=mul_nid, args=(var("k"), var("x")))
    return SchematicLaw(
        pattern=formula,
        conclusion=formula,
        params=frozenset(["k"]),
        variables=frozenset(["x"]),
        evidence=1,
    )


def _hooke_observations(n: int = 8) -> list[tuple[dict, float]]:
    """F = 50 * x, noiseless."""
    return [({("x"): float(i + 1)}, 50.0 * (i + 1)) for i in range(n)]


def _coulomb_observations(n: int = 8) -> list[tuple[dict, float]]:
    """F = 200 * x, noiseless."""
    return [({("x"): float(i + 1)}, 200.0 * (i + 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# TestPhase4Integration
# ---------------------------------------------------------------------------

class TestPhase4Integration:
    """End-to-end Phase 1→4 pipeline test."""

    def _build_pipeline(self):
        """Return (mg, tm, ctx, mul_nid, hooke_mid, coulomb_mid, t_hooke, t_coulomb)."""
        ctx, mul_nid = _build_mul_ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)

        schema = _linear_schema(mul_nid)

        # Fit Hooke's law
        hooke_fl = fit_law(schema, _hooke_observations(), ctx)
        hooke_mid = add_fitted_law(mg, "hooke", hooke_fl)

        # Fit Coulomb's law
        coulomb_fl = fit_law(schema, _coulomb_observations(), ctx)
        coulomb_mid = add_fitted_law(mg, "coulomb", coulomb_fl)

        # Register theories
        t_hooke = tm.register_theory("Hooke")
        t_coulomb = tm.register_theory("Coulomb")
        tm.assign_morphism(hooke_mid, t_hooke)
        tm.assign_morphism(coulomb_mid, t_coulomb)

        return mg, tm, ctx, mul_nid, hooke_mid, coulomb_mid, t_hooke, t_coulomb

    def test_hooke_prediction_correct(self):
        _, tm, ctx, _, _, _, t_hooke, _ = self._build_pipeline()
        pred = tm.predict_under_theory(t_hooke, {"x": 3.0}, ctx)
        assert pred == pytest.approx(150.0, rel=1e-4)  # 50 * 3

    def test_coulomb_prediction_correct(self):
        _, tm, ctx, _, _, _, _, t_coulomb = self._build_pipeline()
        pred = tm.predict_under_theory(t_coulomb, {"x": 3.0}, ctx)
        assert pred == pytest.approx(600.0, rel=1e-4)  # 200 * 3

    def test_theories_are_inconsistent(self):
        """Hooke and Coulomb give very different predictions → inconsistent."""
        _, tm, ctx, _, _, _, t_hooke, t_coulomb = self._build_pipeline()
        result = tm.consistency_check(t_hooke, t_coulomb, {"x": 1.0}, ctx)
        assert result.consistent is False
        assert result.gap == pytest.approx(150.0, rel=1e-4)  # |50 - 200|

    def test_same_theory_consistent(self):
        _, tm, ctx, _, _, _, t_hooke, _ = self._build_pipeline()
        result = tm.consistency_check(t_hooke, t_hooke, {"x": 2.0}, ctx)
        assert result.consistent is True
        assert result.gap == pytest.approx(0.0, abs=1e-9)

    def test_blame_returns_result(self):
        _, tm, ctx, _, _, _, t_hooke, t_coulomb = self._build_pipeline()
        result = tm.blame_theory([t_hooke, t_coulomb], {"x": 1.0},
                                  observed=50.0, ctx=ctx)
        assert result is not None
        assert isinstance(result, BlameResult)

    def test_blame_morph_id_is_real_morphism(self):
        mg, tm, ctx, _, _, _, t_hooke, t_coulomb = self._build_pipeline()
        result = tm.blame_theory([t_hooke, t_coulomb], {"x": 1.0},
                                  observed=50.0, ctx=ctx)
        assert mg.morphism_by_id(result.morph_id) is not None

    def test_blame_points_at_coulomb_morphism(self):
        """Given Hooke observation (F=50 at x=1), Coulomb is more wrong."""
        mg, tm, ctx, _, hooke_mid, coulomb_mid, t_hooke, t_coulomb = self._build_pipeline()
        result = tm.blame_theory([t_hooke, t_coulomb], {"x": 1.0},
                                  observed=50.0, ctx=ctx)
        # Hooke predicts 50 (error=0), Coulomb predicts 200 (error=150)
        assert result.morph_id == coulomb_mid
        assert result.theory_id == t_coulomb

    def test_full_pipeline_residuals_small(self):
        """Both fitted laws have near-zero residuals (noiseless data)."""
        ctx, mul_nid = _build_mul_ctx()
        mg = MorphismGraph()
        schema = _linear_schema(mul_nid)

        hooke_fl = fit_law(schema, _hooke_observations(), ctx)
        coulomb_fl = fit_law(schema, _coulomb_observations(), ctx)

        assert hooke_fl.residual == pytest.approx(0.0, abs=1e-8)
        assert coulomb_fl.residual == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# TestPhase4Cage
# ---------------------------------------------------------------------------

def _anon_ctx(seed: int) -> tuple[EvalContext, int]:
    rng = random.Random(seed)
    symbol = chr(0x2200 + rng.randint(0, 0xFF))
    nid = TOKEN_GRAPH.encode(symbol)
    return EvalContext({nid: lambda a, b: a * b}), nid


class TestPhase4Cage:

    def test_cage_k_hooke_recovered(self):
        """k_A ≈ 50 across all 10 anonymous symbol seeds."""
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            schema = _linear_schema(nid)
            fl = fit_law(schema, _hooke_observations(), ctx)
            assert fl.params["k"] == pytest.approx(50.0, rel=1e-3), \
                f"seed {seed}: k={fl.params['k']}"

    def test_cage_k_coulomb_recovered(self):
        """k_B ≈ 200 across all 10 anonymous symbol seeds."""
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            schema = _linear_schema(nid)
            fl = fit_law(schema, _coulomb_observations(), ctx)
            assert fl.params["k"] == pytest.approx(200.0, rel=1e-3), \
                f"seed {seed}: k={fl.params['k']}"

    def test_cage_inconsistency_flagged(self):
        """Hooke and Coulomb are always flagged inconsistent across all seeds."""
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            schema = _linear_schema(nid)

            hooke_mid = add_fitted_law(mg, f"h_{seed}", fit_law(schema, _hooke_observations(), ctx))
            coulomb_mid = add_fitted_law(mg, f"c_{seed}", fit_law(schema, _coulomb_observations(), ctx))

            t_h = tm.register_theory("H")
            t_c = tm.register_theory("C")
            tm.assign_morphism(hooke_mid, t_h)
            tm.assign_morphism(coulomb_mid, t_c)

            result = tm.consistency_check(t_h, t_c, {"x": 1.0}, ctx)
            assert not result.consistent, f"seed {seed}: expected inconsistent"
            assert result.gap == pytest.approx(150.0, rel=1e-3), \
                f"seed {seed}: gap={result.gap}"


# ---------------------------------------------------------------------------
# TestPhase4DefectProbe
# ---------------------------------------------------------------------------

class TestPhase4DefectProbe:

    def test_probe_a_blame_morph_id_not_theory_id(self):
        """blame.morph_id must identify the FITTED_LAW morphism, not the theory object.

        A naïve implementation might return theory_id as morph_id.  We catch this
        by checking that the returned morph_id is a real morphism (not a theory
        object) with morph_type=FITTED_LAW, and that it equals the known mid.
        """
        ctx, mul_nid = _build_mul_ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        schema = _linear_schema(mul_nid)

        fl = fit_law(schema, _coulomb_observations(), ctx)
        mid = add_fitted_law(mg, "probe_a", fl)
        t = tm.register_theory("ProbeA")
        tm.assign_morphism(mid, t)

        result = tm.blame_theory([t], {"x": 1.0}, observed=50.0, ctx=ctx)
        # Must point at the known FITTED_LAW morphism
        assert result.morph_id == mid
        # That morphism must exist in the graph and be a FITTED_LAW
        blamed_morph = mg.morphism_by_id(result.morph_id)
        assert blamed_morph is not None
        assert blamed_morph.morph_type == "FITTED_LAW"
        # The theory_id must be a theory object, not just the same int
        theory_obj = mg.object_by_id(result.theory_id)
        assert theory_obj is not None
        assert theory_obj.is_theory is True

    def test_probe_b_ten_morphisms_one_wrong(self):
        """Theory with 10 morphisms, exactly 1 wrong → blame finds the one."""
        ctx, mul_nid = _build_mul_ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        schema = _linear_schema(mul_nid)
        t = tm.register_theory("ProbeB")

        # 9 morphisms with k=50 (Hooke-correct)
        for i in range(9):
            fl = fit_law(schema, _hooke_observations(), ctx)
            mid = add_fitted_law(mg, f"probe_b_good_{i}", fl)
            tm.assign_morphism(mid, t)

        # 1 morphism with k=200 (Coulomb — wrong for Hooke observation)
        bad_fl = fit_law(schema, _coulomb_observations(), ctx)
        bad_mid = add_fitted_law(mg, "probe_b_bad", bad_fl)
        tm.assign_morphism(bad_mid, t)

        result = tm.blame_theory([t], {"x": 1.0}, observed=50.0, ctx=ctx)
        assert result.morph_id == bad_mid

    def test_probe_c_identical_k_consistent(self):
        """Two theories with the same k must always be consistent."""
        ctx, mul_nid = _build_mul_ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        schema = _linear_schema(mul_nid)

        fl1 = fit_law(schema, _hooke_observations(), ctx)
        fl2 = fit_law(schema, _hooke_observations(), ctx)

        mid1 = add_fitted_law(mg, "probe_c_1", fl1)
        mid2 = add_fitted_law(mg, "probe_c_2", fl2)

        t1 = tm.register_theory("PC1")
        t2 = tm.register_theory("PC2")
        tm.assign_morphism(mid1, t1)
        tm.assign_morphism(mid2, t2)

        result = tm.consistency_check(t1, t2, {"x": 5.0}, ctx)
        assert result.consistent is True
        assert result.gap < 1e-6
