"""
Unit tests for Phase 10 — Paradigm Shift / Left Kan Extension (A-8).

TestParadigmShiftBasic (7 tests)
    - returns non-None for irreconcilable anomaly
    - new_theory_id != old_theory_id
    - bridge_morph_id in graph
    - bridge morph type is PARADIGM_SHIFT
    - bridge morph source = old_theory_id
    - bridge morph target = new_theory_id
    - old theory morphisms unchanged after shift

TestParadigmShiftExplanation (5 tests)
    - explanation is not None
    - explanation predicts held-out anomaly correctly
    - anomaly_coverage ≥ min_coverage threshold
    - new theory contains the hypothesis morphisms
    - two schema candidates: MDL selects simpler

TestQueryParadigmShifts (3 tests)
    - query returns empty list when no shifts
    - query returns stored shift
    - multiple shifts from same theory

TestBitterLessonCage (3 tests)
    - 10 seeds: shift returns non-None
    - 10 seeds: old theory unchanged
    - 5 seeds: query returns stored result

TestDefectProbe (4 tests)
    Probe 1: old theory has NO new morphisms after shift
    Probe 2: new theory has the explanation morphisms
    Probe 3: two symbol tables produce same coverage
    Probe 4: shift covers anomaly, old theory does not
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
from experiments.symbolic_ai_v2.ctkg.inference.coverage import score_coverage
from experiments.symbolic_ai_v2.ctkg.inference.paradigm import (
    ParadigmShiftResult,
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


def _schema_g3(nid: int) -> SchematicLaw:
    formula = Expr(head=nid, args=(var("a"), var("x")))
    return SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(["a", "b", "c"]),
        variables=frozenset(["x"]), evidence=1,
    )


def _fitted(nid: int, k: float) -> FittedLaw:
    formula = Expr(head=nid, args=(atom("k"), var("x")))
    sch = SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(), variables=frozenset(["x"]), evidence=1,
    )
    return FittedLaw(schema=sch, params={"k": k}, residual=0.0)


def _obs_sets(k: float, n_sets: int = 3, n_per: int = 4) -> list[list[tuple[dict, float]]]:
    return [[({  "x": float(i + 1)}, k * (i + 1)) for i in range(n_per)]
            for _ in range(n_sets)]


def _setup_old_theory(nid: int, k: float = 5.0):
    mg = MorphismGraph()
    tm = TheoryManager(mg)
    t_old = tm.register_theory("OldNewton")
    mid = add_fitted_law(mg, "old_k", _fitted(nid, k))
    tm.assign_morphism(mid, t_old)
    return mg, tm, t_old, mid


# ---------------------------------------------------------------------------
# TestParadigmShiftBasic
# ---------------------------------------------------------------------------

class TestParadigmShiftBasic:

    def test_returns_non_none(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
            new_theory_name="SR", tolerance=0.10,
        )
        assert result is not None

    def test_new_theory_different_from_old(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        assert result is not None
        assert result.new_theory_id != result.old_theory_id

    def test_bridge_morph_in_graph(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        assert result is not None
        m = mg.morphism_by_id(result.bridge_morph_id)
        assert m is not None

    def test_bridge_morph_type(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        assert result is not None
        m = mg.morphism_by_id(result.bridge_morph_id)
        assert m.morph_type == "PARADIGM_SHIFT"

    def test_bridge_source_is_old_theory(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        assert result is not None
        m = mg.morphism_by_id(result.bridge_morph_id)
        assert m.source == t_old

    def test_bridge_target_is_new_theory(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        assert result is not None
        m = mg.morphism_by_id(result.bridge_morph_id)
        assert m.target == result.new_theory_id

    def test_old_theory_morphisms_unchanged(self):
        """The old theory must have the same morphisms after the shift."""
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        members_before = set(tm._mg.theory_members(t_old))
        sets = _obs_sets(50.0)
        propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        members_after = set(tm._mg.theory_members(t_old))
        assert members_before == members_after, \
            "Old theory morphisms must not change after paradigm shift"


# ---------------------------------------------------------------------------
# TestParadigmShiftExplanation
# ---------------------------------------------------------------------------

class TestParadigmShiftExplanation:

    def test_explanation_not_none(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        assert result is not None
        assert result.explanation is not None

    def test_explanation_predicts_holdout(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0, n_per=5)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
            tolerance=0.10,
        )
        assert result is not None
        assert result.explanation is not None
        # Held-out: x=10 → f(10) = 50*10 = 500
        z    = predict_continuous(result.explanation.input_law,  {"x": 10.0}, ctx)
        pred = predict_continuous(result.explanation.output_law, {"z": z},    ctx)
        assert pred == pytest.approx(500.0, rel=0.10)

    def test_coverage_meets_threshold(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
            min_coverage=0.5, tolerance=0.10,
        )
        assert result is not None
        assert result.anomaly_coverage >= 0.5

    def test_new_theory_has_hypothesis(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        assert result is not None
        members = tm._mg.theory_members(result.new_theory_id)
        assert len(members) >= 1

    def test_two_schemas_mdl_selects_simpler(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0, n_per=6)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid), _schema_g3(nid)], _schema_h(nid),
            ctx, mg, tm, tolerance=0.10,
        )
        assert result is not None
        assert result.explanation is not None
        # 1-param schema has fewer total params
        n_params = (len(result.explanation.input_law.schema.params)
                    + len(result.explanation.output_law.schema.params))
        assert n_params <= 2


# ---------------------------------------------------------------------------
# TestQueryParadigmShifts
# ---------------------------------------------------------------------------

class TestQueryParadigmShifts:

    def test_empty_when_no_shifts(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        shifts = query_paradigm_shifts(mg, t_old)
        assert shifts == []

    def test_query_returns_stored_shift(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
            new_theory_name="SR1",
        )
        assert result is not None
        shifts = query_paradigm_shifts(mg, t_old)
        assert len(shifts) == 1
        assert shifts[0].new_theory_id == result.new_theory_id

    def test_multiple_shifts(self):
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets_a = _obs_sets(50.0)
        sets_b = _obs_sets(100.0)
        propose_paradigm_shift(t_old, sets_a, [_schema_g(nid)], _schema_h(nid),
                                ctx, mg, tm, new_theory_name="SR_a")
        propose_paradigm_shift(t_old, sets_b, [_schema_g(nid)], _schema_h(nid),
                                ctx, mg, tm, new_theory_name="SR_b")
        shifts = query_paradigm_shifts(mg, t_old)
        assert len(shifts) == 2


# ---------------------------------------------------------------------------
# TestBitterLessonCage
# ---------------------------------------------------------------------------

class TestBitterLessonCage:

    def test_cage_non_none(self):
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            mg, tm, t_old, mid = _setup_old_theory(nid)
            sets = _obs_sets(50.0)
            result = propose_paradigm_shift(
                t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
                tolerance=0.10, label_prefix=f"cage10_{seed}",
            )
            assert result is not None, f"seed {seed}: None"

    def test_cage_old_theory_unchanged(self):
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            mg, tm, t_old, mid = _setup_old_theory(nid)
            members_before = set(tm._mg.theory_members(t_old))
            sets = _obs_sets(50.0)
            propose_paradigm_shift(
                t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
                label_prefix=f"cage10u_{seed}",
            )
            members_after = set(tm._mg.theory_members(t_old))
            assert members_before == members_after, f"seed {seed}: old theory changed"

    def test_cage_query_returns_result(self):
        for seed in range(5):
            ctx, nid = _anon_ctx(seed)
            mg, tm, t_old, mid = _setup_old_theory(nid)
            sets = _obs_sets(50.0)
            result = propose_paradigm_shift(
                t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
                new_theory_name=f"SR_{seed}",
            )
            assert result is not None
            shifts = query_paradigm_shifts(mg, t_old)
            assert len(shifts) >= 1, f"seed {seed}: no shifts stored"


# ---------------------------------------------------------------------------
# TestDefectProbe
# ---------------------------------------------------------------------------

class TestDefectProbe:

    def test_probe1_old_theory_no_new_morphisms(self):
        """After propose_paradigm_shift, the old theory has NO new morphisms."""
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        members_before = set(tm._mg.theory_members(t_old))
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        members_after = set(tm._mg.theory_members(t_old))
        assert members_before == members_after, \
            "PROBE 1: old theory must not gain new morphisms after paradigm shift"

    def test_probe2_new_theory_has_morphisms(self):
        """The new theory contains the explanation morphisms."""
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
        )
        assert result is not None
        new_members = tm._mg.theory_members(result.new_theory_id)
        assert len(new_members) >= 1, \
            "PROBE 2: new theory must have at least one morphism"

    def test_probe3_two_tables_same_coverage(self):
        """Two symbol tables produce the same anomaly coverage."""
        coverages = []
        for seed in range(2):
            ctx, nid = _anon_ctx(seed)
            mg, tm, t_old, mid = _setup_old_theory(nid)
            sets = _obs_sets(50.0)
            result = propose_paradigm_shift(
                t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
                tolerance=0.10, label_prefix=f"p10d3_{seed}",
            )
            assert result is not None, f"seed {seed}: None"
            coverages.append(result.anomaly_coverage)
        assert coverages[0] == pytest.approx(coverages[1], abs=0.01), \
            f"PROBE 3: coverage differs: {coverages[0]} vs {coverages[1]}"

    def test_probe4_shift_covers_anomaly_old_does_not(self):
        """Old theory doesn't cover 50*x; new theory does."""
        ctx, nid = _ctx()
        mg, tm, t_old, mid = _setup_old_theory(nid, k=5.0)
        sets = _obs_sets(50.0)
        result = propose_paradigm_shift(
            t_old, sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm,
            tolerance=0.10,
        )
        assert result is not None
        # Old theory (k=5) should not cover 50*x anomalies well
        # We can check: old theory predicts 5*x, anomaly expects 50*x → bad coverage
        old_pred = tm.predict_under_theory(t_old, {"x": 1.0}, ctx)
        assert old_pred is not None
        obs = 50.0
        rel_err = abs(old_pred - obs) / obs
        assert rel_err > 0.5, "PROBE 4: old theory should not cover anomaly"
        # New theory explanation predicts ≈50*x
        assert result.anomaly_coverage >= 0.5, \
            "PROBE 4: new theory must cover the anomaly"
