"""
Phase 10 Integration Test — Paradigm Shift / Left Kan Extension (A-8).

Chains Phases 1–10 end-to-end.  Implements the Einstein test's core abductive
step: when an anomaly is irreconcilable with the current theory, propose a new
theory cluster (Left Kan Extension) rather than contaminating the old one.

Scenario A — Newtonian → Relativistic
----------------------------------------
Newton theory: single morphism k=5.
Anomaly stream: observations from k=50 process — fundamentally different regime.
- Phase 9 (preservation check) would reject any revision to the Newton theory
  because it breaks class-A (k=5) predictions.
- Phase 10 (paradigm shift) creates a new "Relativistic" theory cluster with a
  k≈50 explanation, linked to Newton by a PARADIGM_SHIFT bridge morphism.

Tests verify:
  1. The new theory covers the anomaly.
  2. The old theory's morphisms are unchanged.
  3. The bridge morphism exists with correct source/target.
  4. Held-out anomaly predictions are correct from the new theory.

Scenario B — Multiple anomaly sets: shared new explanation
------------------------------------------------------------
Three anomaly sets all from k=50; one outlier from k=100.
Paradigm shift creates a single new theory with ≥75% coverage.

Scenario C — Full pipeline: Phases 1–10
-----------------------------------------
Phase 4 (TheoryManager) + Phase 5 (ClosedLoopReviser) + Phase 6 (RetractEngine)
+ Phase 7 (LatentHypothesis) + Phase 8 (Coverage) + Phase 9 (Preservation)
+ Phase 10 (ParadigmShift).

Test classes
------------
TestPhase10Integration (8 tests)
    - paradigm shift returns non-None
    - new theory different from old
    - bridge morphism in graph with correct type
    - old theory unchanged
    - explanation covers anomaly
    - held-out prediction correct
    - query_paradigm_shifts returns result
    - full pipeline: preservation-blocked revision → paradigm shift

TestPhase10Cage (3 tests)
    - 10 seeds: non-None
    - 10 seeds: old theory unchanged
    - 5 seeds: coverage ≥ 0.5

TestPhase10DefectProbe (4 tests)
    Probe 1: old theory has no new morphisms (contamination prevention)
    Probe 2: new theory has explanation
    Probe 3: two symbol tables same coverage
    Probe 4: bridge morph type = PARADIGM_SHIFT
"""
from __future__ import annotations

import random

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom, var, Expr
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import (
    FittedLaw,
    add_fitted_law,
    predict_continuous,
)
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager
from experiments.symbolic_ai_v2.ctkg.inference.revision import ClosedLoopReviser
from experiments.symbolic_ai_v2.ctkg.inference.retract import RetractEngine
from experiments.symbolic_ai_v2.ctkg.inference.preservation import (
    PredictionLedger,
    propose_and_apply_safe,
)
from experiments.symbolic_ai_v2.ctkg.inference.paradigm import (
    propose_paradigm_shift,
    query_paradigm_shifts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(name: str = "mul") -> tuple[EvalContext, int]:
    nid = TOKEN_GRAPH.encode(name)
    return EvalContext({nid: lambda a, b: a * b}), nid


def _anon_ctx(seed: int) -> tuple[EvalContext, int]:
    rng = random.Random(seed)
    sym = chr(0x2200 + rng.randint(0, 0xFF))
    nid = TOKEN_GRAPH.encode(sym)
    return EvalContext({nid: lambda a, b: a * b}), nid


def _schema_g(nid: int) -> SchematicLaw:
    formula = Expr(head=nid, args=(var("k"), var("x")))
    return SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(["k"]), variables=frozenset(["x"]), evidence=1,
    )


def _schema_h(nid: int) -> SchematicLaw:
    formula = Expr(head=nid, args=(var("a"), var("z")))
    return SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(["a"]), variables=frozenset(["z"]), evidence=1,
    )


def _fitted(nid: int, k: float) -> FittedLaw:
    formula = Expr(head=nid, args=(atom("k"), var("x")))
    sch = SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(), variables=frozenset(["x"]), evidence=1,
    )
    return FittedLaw(schema=sch, params={"k": k}, residual=0.0)


def _obs_sets(k: float, n_sets: int = 3, n_per: int = 5) -> list[list[tuple[dict, float]]]:
    return [[({  "x": float(i + 1)}, k * (i + 1)) for i in range(n_per)]
            for _ in range(n_sets)]


# ---------------------------------------------------------------------------
# TestPhase10Integration
# ---------------------------------------------------------------------------

class TestPhase10Integration:

    def test_paradigm_shift_non_none(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t_newton = tm.register_theory("Newton")
        mid = add_fitted_law(mg, "newton_k5", _fitted(nid, 5.0))
        tm.assign_morphism(mid, t_newton)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_newton, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
            new_theory_name="SR", tolerance=0.10,
        )
        assert result is not None

    def test_new_theory_different_from_old(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t_newton = tm.register_theory("Newton")
        mid = add_fitted_law(mg, "newton_k5b", _fitted(nid, 5.0))
        tm.assign_morphism(mid, t_newton)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_newton, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        assert result is not None
        assert result.new_theory_id != result.old_theory_id

    def test_bridge_morphism_correct_type(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t_newton = tm.register_theory("Newton")
        mid = add_fitted_law(mg, "newton_k5c", _fitted(nid, 5.0))
        tm.assign_morphism(mid, t_newton)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_newton, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        assert result is not None
        m = mg.morphism_by_id(result.bridge_morph_id)
        assert m is not None
        assert m.morph_type == "PARADIGM_SHIFT"
        assert m.source == t_newton
        assert m.target == result.new_theory_id

    def test_old_theory_unchanged(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t_newton = tm.register_theory("Newton")
        mid = add_fitted_law(mg, "newton_k5d", _fitted(nid, 5.0))
        tm.assign_morphism(mid, t_newton)
        members_before = set(tm._mg.theory_members(t_newton))
        sets = _obs_sets(50.0)
        propose_paradigm_shift(
            t_newton, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        members_after = set(tm._mg.theory_members(t_newton))
        assert members_before == members_after

    def test_explanation_covers_anomaly(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t_newton = tm.register_theory("Newton")
        mid = add_fitted_law(mg, "newton_k5e", _fitted(nid, 5.0))
        tm.assign_morphism(mid, t_newton)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_newton, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
            tolerance=0.10,
        )
        assert result is not None
        assert result.anomaly_coverage >= 0.5

    def test_holdout_prediction_correct(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t_newton = tm.register_theory("Newton")
        mid = add_fitted_law(mg, "newton_k5f", _fitted(nid, 5.0))
        tm.assign_morphism(mid, t_newton)
        sets = _obs_sets(50.0, n_per=6)
        result = propose_paradigm_shift(
            t_newton, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
            tolerance=0.10,
        )
        assert result is not None
        assert result.explanation is not None
        # Held-out: x=8 → 50*8 = 400
        z    = predict_continuous(result.explanation.input_law,  {"x": 8.0}, ctx)
        pred = predict_continuous(result.explanation.output_law, {"z": z},   ctx)
        assert pred == pytest.approx(400.0, rel=0.15)

    def test_query_returns_result(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t_newton = tm.register_theory("Newton")
        mid = add_fitted_law(mg, "newton_k5g", _fitted(nid, 5.0))
        tm.assign_morphism(mid, t_newton)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_newton, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
            new_theory_name="SR_query",
        )
        assert result is not None
        shifts = query_paradigm_shifts(mg, t_newton)
        assert len(shifts) == 1
        assert shifts[0].new_theory_id == result.new_theory_id

    def test_full_pipeline_preservation_blocked_then_shift(self):
        """Preservation blocks revision → paradigm shift creates new theory."""
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
        eng = RetractEngine(rev, tm, mg)
        t_newton = tm.register_theory("Newton")

        # Phase 4: Newton theory with k=5 and k=10
        mid_a = add_fitted_law(mg, "full_newton_a", _fitted(nid, 5.0))
        mid_b = add_fitted_law(mg, "full_newton_b", _fitted(nid, 10.0))
        tm.assign_morphism(mid_a, t_newton)
        tm.assign_morphism(mid_b, t_newton)

        # Phase 9: ledger records class-A observations
        ledger = PredictionLedger()
        for i in range(4):
            ledger.record(t_newton, {"x": float(i + 1)}, 5.0 * (i + 1),
                          5.0 * (i + 1))

        schema = _schema_g(nid)
        anomalies = [({  "x": float(i + 1)}, 50.0 * (i + 1)) for i in range(4)]

        # Phase 9: try safe revision — should be rejected (breaks class-A)
        revision_result = propose_and_apply_safe(
            eng, rev, tm, mg, ledger, t_newton, anomalies, ctx, schema,
            label="full_newton_rev", tolerance=0.05,
        )
        # Should be rejected
        assert revision_result is None

        # Phase 10: paradigm shift
        members_before = set(mg.theory_members(t_newton))
        shift = propose_paradigm_shift(
            t_newton, [anomalies], [_schema_g(nid)], _schema_h(nid),
            ctx, mg, tm, new_theory_name="Relativistic", tolerance=0.10,
        )
        assert shift is not None
        # Old theory unchanged
        members_after = set(mg.theory_members(t_newton))
        assert members_before == members_after
        # New theory covers anomaly
        assert shift.anomaly_coverage >= 0.5


# ---------------------------------------------------------------------------
# TestPhase10Cage
# ---------------------------------------------------------------------------

class TestPhase10Cage:

    def test_cage_non_none(self):
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            t_old = tm.register_theory("T")
            mid = add_fitted_law(mg, f"cage10_{seed}", _fitted(nid, 5.0))
            tm.assign_morphism(mid, t_old)
            sets = _obs_sets(50.0)
            result = propose_paradigm_shift(
                t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
                tolerance=0.10, label_prefix=f"cage10_{seed}",
            )
            assert result is not None, f"seed {seed}: None"

    def test_cage_old_theory_unchanged(self):
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            t_old = tm.register_theory("T")
            mid = add_fitted_law(mg, f"cage10u_{seed}", _fitted(nid, 5.0))
            tm.assign_morphism(mid, t_old)
            members_before = set(mg.theory_members(t_old))
            sets = _obs_sets(50.0)
            propose_paradigm_shift(
                t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
                label_prefix=f"cage10u_{seed}",
            )
            members_after = set(mg.theory_members(t_old))
            assert members_before == members_after, f"seed {seed}: old theory changed"

    def test_cage_coverage_ge_half(self):
        for seed in range(5):
            ctx, nid = _anon_ctx(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            t_old = tm.register_theory("T")
            mid = add_fitted_law(mg, f"cage10c_{seed}", _fitted(nid, 5.0))
            tm.assign_morphism(mid, t_old)
            sets = _obs_sets(50.0)
            result = propose_paradigm_shift(
                t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
                min_coverage=0.5, tolerance=0.10,
            )
            assert result is not None, f"seed {seed}: None"
            assert result.anomaly_coverage >= 0.5, \
                f"seed {seed}: coverage {result.anomaly_coverage} < 0.5"


# ---------------------------------------------------------------------------
# TestPhase10DefectProbe
# ---------------------------------------------------------------------------

class TestPhase10DefectProbe:

    def test_probe1_old_theory_not_contaminated(self):
        """After paradigm shift, the old theory's morphism count is unchanged."""
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t_old = tm.register_theory("Newton")
        mid = add_fitted_law(mg, "p10d1_k5", _fitted(nid, 5.0))
        tm.assign_morphism(mid, t_old)
        n_before = len(list(mg.theory_members(t_old)))
        sets = _obs_sets(50.0)
        propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        n_after = len(list(mg.theory_members(t_old)))
        assert n_before == n_after, \
            f"PROBE 1: old theory gained {n_after - n_before} new morphisms"

    def test_probe2_new_theory_has_explanation(self):
        """The new theory cluster must contain at least one hypothesis morphism."""
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t_old = tm.register_theory("Newton")
        mid = add_fitted_law(mg, "p10d2_k5", _fitted(nid, 5.0))
        tm.assign_morphism(mid, t_old)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        assert result is not None
        new_members = list(mg.theory_members(result.new_theory_id))
        assert len(new_members) >= 1, \
            "PROBE 2: new theory has no morphisms"

    def test_probe3_two_tables_same_coverage(self):
        """Two symbol tables produce the same anomaly coverage fraction."""
        coverages = []
        for seed in range(2):
            ctx, nid = _anon_ctx(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            t_old = tm.register_theory("T")
            mid = add_fitted_law(mg, f"p10d3_{seed}", _fitted(nid, 5.0))
            tm.assign_morphism(mid, t_old)
            sets = _obs_sets(50.0)
            result = propose_paradigm_shift(
                t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
                tolerance=0.10, label_prefix=f"p10d3_{seed}",
            )
            assert result is not None, f"seed {seed}: None"
            coverages.append(result.anomaly_coverage)
        assert coverages[0] == pytest.approx(coverages[1], abs=0.01), \
            f"PROBE 3: coverage differs: {coverages[0]} vs {coverages[1]}"

    def test_probe4_bridge_type_is_paradigm_shift(self):
        """The bridge morphism must have type PARADIGM_SHIFT."""
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t_old = tm.register_theory("Newton")
        mid = add_fitted_law(mg, "p10d4_k5", _fitted(nid, 5.0))
        tm.assign_morphism(mid, t_old)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        assert result is not None
        m = mg.morphism_by_id(result.bridge_morph_id)
        assert m.morph_type == "PARADIGM_SHIFT", \
            f"PROBE 4: bridge type = {m.morph_type!r}, expected 'PARADIGM_SHIFT'"
