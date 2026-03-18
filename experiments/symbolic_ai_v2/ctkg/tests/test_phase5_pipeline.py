"""
Phase 5 Integration Test — Closed-Loop Surprise and Revision.

This test chains Phases 1–5 end-to-end, adding the closed-loop revision
mechanism to the theory compartments from Phase 4.

Scenario
--------
A Newtonian theory (k=10) is in the graph.  A stream of anomalous
observations (true k=50) arrives.  The ClosedLoopReviser:

  1. Detects high surprise (squared-error >> threshold)
  2. Scores the anomaly positively (KL >> MDL_COST)
  3. Fits a new FITTED_LAW morphism with k≈50
  4. Verifies the revision reduces surprise (closed loop)
  5. Accepts the revision → prediction moves toward 50

After revision:
  - predict_under_theory returns a value between 10 and 50 (mean of two laws)
  - The new morphism is a FITTED_LAW (not OBS_SEQ)
  - Evidence accumulation: 5 observe() calls + flush() → k≈50 fitted from 5 obs

Test classes
------------
TestPhase5Integration (8 tests)
    - full pipeline: theory setup → revise_immediate → verify prediction change
    - revise_immediate returns non-None for anomalous observation
    - accepted result: surprise decreases
    - evidence_count == 1 for revise_immediate
    - flush from 5 observations: evidence_count == 5, k recovered
    - no OBS_SEQ created
    - candidate.morph_id is a real FITTED_LAW morphism
    - prediction improves toward true k after revision

TestPhase5Cage (3 tests)
    - cage: 10 anonymous seeds, revise_immediate non-None in all
    - cage: evidence_count == 5 in all flush results
    - cage: accepted in all

TestPhase5DefectProbe (5 tests: one per defect)
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
    fit_law,
    add_fitted_law,
)
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager
from experiments.symbolic_ai_v2.ctkg.inference.revision import (
    ClosedLoopReviser,
    RevisionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(name: str = "mul") -> tuple[EvalContext, int]:
    nid = TOKEN_GRAPH.encode(name)
    return EvalContext({nid: lambda a, b: a * b}), nid


def _schema(nid: int) -> SchematicLaw:
    formula = Expr(head=nid, args=(var("k"), var("x")))
    return SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(["k"]), variables=frozenset(["x"]), evidence=1,
    )


def _fitted(nid: int, k: float) -> FittedLaw:
    formula = Expr(head=nid, args=(atom("k"), var("x")))
    schema = SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(), variables=frozenset(["x"]), evidence=1,
    )
    return FittedLaw(schema=schema, params={"k": k}, residual=0.0)


def _setup(k_theory=10.0):
    ctx, nid = _ctx()
    mg = MorphismGraph()
    tm = TheoryManager(mg)
    schema = _schema(nid)
    law = _fitted(nid, k_theory)
    mid = add_fitted_law(mg, "newton", law)
    t = tm.register_theory("Newton")
    tm.assign_morphism(mid, t)
    reviser = ClosedLoopReviser(tm, mg, mdl_cost=2.0, threshold=3.0, sigma=1.0)
    return mg, tm, ctx, nid, t, schema, reviser


# ---------------------------------------------------------------------------
# TestPhase5Integration
# ---------------------------------------------------------------------------

class TestPhase5Integration:

    def test_revise_immediate_returns_non_none_for_anomaly(self):
        mg, tm, ctx, nid, t, schema, reviser = _setup(k_theory=10.0)
        # true k=50; predicted=10; observed=50 → surprise=1600 >> 3.0
        result = reviser.revise_immediate(t, {"x": 1.0}, observed=50.0,
                                          schema=schema, ctx=ctx)
        assert result is not None

    def test_accepted_and_surprise_decreases(self):
        mg, tm, ctx, nid, t, schema, reviser = _setup(k_theory=10.0)
        result = reviser.revise_immediate(t, {"x": 1.0}, observed=50.0,
                                          schema=schema, ctx=ctx)
        assert result is not None
        assert result.accepted is True
        assert result.surprise_after < result.surprise_before

    def test_evidence_count_one(self):
        mg, tm, ctx, nid, t, schema, reviser = _setup(k_theory=10.0)
        result = reviser.revise_immediate(t, {"x": 1.0}, observed=50.0,
                                          schema=schema, ctx=ctx)
        assert result is not None
        assert result.evidence_count == 1

    def test_no_obs_seq_created(self):
        mg, tm, ctx, nid, t, schema, reviser = _setup(k_theory=10.0)
        reviser.revise_immediate(t, {"x": 1.0}, observed=50.0,
                                  schema=schema, ctx=ctx)
        for m in mg.morphisms():
            assert m.morph_type != "OBS_SEQ"

    def test_candidate_morph_is_fitted_law(self):
        mg, tm, ctx, nid, t, schema, reviser = _setup(k_theory=10.0)
        result = reviser.revise_immediate(t, {"x": 1.0}, observed=50.0,
                                          schema=schema, ctx=ctx)
        assert result is not None
        mid = result.candidate.morph_id
        m = mg.morphism_by_id(mid)
        assert m is not None
        assert m.morph_type == "FITTED_LAW"

    def test_prediction_changes_toward_true_k(self):
        mg, tm, ctx, nid, t, schema, reviser = _setup(k_theory=10.0)
        p_before = tm.predict_under_theory(t, {"x": 1.0}, ctx)
        reviser.revise_immediate(t, {"x": 1.0}, observed=50.0,
                                  schema=schema, ctx=ctx)
        p_after = tm.predict_under_theory(t, {"x": 1.0}, ctx)
        # After adding k≈50 law, mean of k=10 and k≈50 is ≈30; closer to 50 than 10
        assert p_after > p_before
        assert p_after < 50.0 + 1e-6  # can't exceed true k with mean

    def test_flush_five_observations_evidence_count(self):
        mg, tm, ctx, nid, t, schema, reviser = _setup(k_theory=10.0)
        for i in range(5):
            reviser.observe(t, {"x": float(i + 1)}, 50.0 * (i + 1))
        result = reviser.flush(schema, ctx)
        assert result is not None
        assert result.evidence_count == 5

    def test_flush_five_recovers_k50(self):
        mg, tm, ctx, nid, t, schema, reviser = _setup(k_theory=10.0)
        for i in range(5):
            reviser.observe(t, {"x": float(i + 1)}, 50.0 * (i + 1))
        result = reviser.flush(schema, ctx)
        assert result is not None
        fitted_k = result.candidate.law.params.get("k")
        assert fitted_k == pytest.approx(50.0, rel=0.05)


# ---------------------------------------------------------------------------
# TestPhase5Cage
# ---------------------------------------------------------------------------

def _anon(seed: int) -> tuple[EvalContext, int]:
    rng = random.Random(seed)
    sym = chr(0x2200 + rng.randint(0, 0xFF))
    nid = TOKEN_GRAPH.encode(sym)
    return EvalContext({nid: lambda a, b: a * b}), nid


class TestPhase5Cage:

    def test_cage_revise_non_none_all_seeds(self):
        for seed in range(10):
            ctx, nid = _anon(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            sch = _schema(nid)
            law = _fitted(nid, 10.0)
            mid = add_fitted_law(mg, f"n{seed}", law)
            t = tm.register_theory("T")
            tm.assign_morphism(mid, t)
            rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
            result = rev.revise_immediate(t, {"x": 1.0}, observed=50.0,
                                          schema=sch, ctx=ctx)
            assert result is not None, f"seed {seed}: returned None"

    def test_cage_flush_evidence_count_five(self):
        for seed in range(10):
            ctx, nid = _anon(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            sch = _schema(nid)
            law = _fitted(nid, 10.0)
            mid = add_fitted_law(mg, f"f{seed}", law)
            t = tm.register_theory("T")
            tm.assign_morphism(mid, t)
            rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
            for i in range(5):
                rev.observe(t, {"x": float(i + 1)}, 50.0 * (i + 1))
            result = rev.flush(sch, ctx)
            assert result is not None
            assert result.evidence_count == 5, f"seed {seed}: ec={result.evidence_count}"

    def test_cage_accepted_all_seeds(self):
        for seed in range(10):
            ctx, nid = _anon(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            sch = _schema(nid)
            law = _fitted(nid, 10.0)
            mid = add_fitted_law(mg, f"a{seed}", law)
            t = tm.register_theory("T")
            tm.assign_morphism(mid, t)
            rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
            result = rev.revise_immediate(t, {"x": 1.0}, observed=50.0,
                                          schema=sch, ctx=ctx)
            assert result is not None
            assert result.accepted, f"seed {seed}: not accepted"


# ---------------------------------------------------------------------------
# TestPhase5DefectProbe
# ---------------------------------------------------------------------------

class TestPhase5DefectProbe:

    def test_probe1_single_anomaly_not_ignored(self):
        mg, tm, ctx, nid, t, schema, reviser = _setup(k_theory=10.0)
        result = reviser.revise_immediate(t, {"x": 1.0}, observed=50.0,
                                          schema=schema, ctx=ctx)
        assert result is not None, "Defect 1: single anomaly was ignored"

    def test_probe2_no_obs_seq_morphism(self):
        mg, tm, ctx, nid, t, schema, reviser = _setup(k_theory=10.0)
        reviser.revise_immediate(t, {"x": 1.0}, observed=50.0,
                                  schema=schema, ctx=ctx)
        obs_seq = [m for m in mg.morphisms() if m.morph_type == "OBS_SEQ"]
        assert len(obs_seq) == 0, f"Defect 2: OBS_SEQ morphisms created: {obs_seq}"

    def test_probe3_five_separate_observations_flush(self):
        mg, tm, ctx, nid, t, schema, reviser = _setup(k_theory=10.0)
        for i in range(5):
            reviser.observe(t, {"x": float(i + 1)}, 50.0 * (i + 1))
        result = reviser.flush(schema, ctx)
        assert result is not None
        assert result.evidence_count == 5, \
            f"Defect 3: evidence_count={result.evidence_count}, expected 5"

    def test_probe4_closed_loop_reduces_surprise(self):
        mg, tm, ctx, nid, t, schema, reviser = _setup(k_theory=10.0)
        result = reviser.revise_immediate(t, {"x": 1.0}, observed=50.0,
                                          schema=schema, ctx=ctx)
        assert result is not None
        assert result.accepted is True
        assert result.surprise_after < result.surprise_before, \
            "Defect 4: closed loop did not verify surprise decrease"

    def test_probe5_blame_finds_wrong_morphism(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        # Two morphisms: correct and wrong
        mid_c = add_fitted_law(mg, "dp5_c", _fitted(nid, 50.0))
        mid_w = add_fitted_law(mg, "dp5_w", _fitted(nid, 10.0))
        tm.assign_morphism(mid_c, t)
        tm.assign_morphism(mid_w, t)
        # Blame should find the wrong one (k=10 is further from observed=50)
        blame = tm.blame_theory([t], {"x": 1.0}, observed=50.0, ctx=ctx)
        assert blame.morph_id == mid_w, \
            f"Defect 5: blame={blame.morph_id}, expected wrong={mid_w}"
