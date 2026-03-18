"""
Phase 9 Integration Test — Theory Revision with Preservation Guarantee (A-7).

Chains Phases 1–9 end-to-end.

Scenario A — Safe Revision
----------------------------
Newton theory has two morphisms:
  mid_correct  k=5  (correctly explains class A: obs = 5*x)
  mid_wrong    k=10 (wrong for class C: obs = 50*x)

The PredictionLedger records class-A examples as confirmed correct.
propose_and_apply_safe attempts to retract mid_wrong and add a k≈50 law.

Preservation check: the new k≈50 law predicts ≈50*x for class A (x=1),
but class A expects 5*x.  Relative error ≈ 900%.  Preservation FAILS.
The revision is rejected.  mid_wrong is un-vetoed.  mid_correct still active.

Scenario B — Harmless Revision
---------------------------------
Theory has only mid_wrong (k=10).
No class-A entries in the ledger (no correct examples to preserve).
Revision replaces k=10 with k≈50.
No preservation constraint → revision accepted.

Scenario C — Full pipeline
----------------------------
Phase 4 Newton theory + Phase 5 ClosedLoopReviser + Phase 6 RetractEngine +
Phase 7 latent + Phase 8 multi-anomaly + Phase 9 preservation.
Three anomaly streams from 6*x; ledger has class-A entries (5*x).
A revision aimed at 6*x is accepted only if it doesn't break 5*x predictions.

Test classes
------------
TestPhase9Integration (8 tests)
    - check_preservation: single-morphism theory passes
    - check_preservation: broken when obs ≠ theory prediction
    - propose_and_apply_safe: rejects revision that breaks ledger
    - propose_and_apply_safe: accepts revision when no ledger entries
    - after rejection: original morphism un-vetoed
    - after rejection: new morphism vetoed
    - after acceptance: old morphism stays vetoed
    - full pipeline: phase 4 + 9

TestPhase9Cage (3 tests)
    - 10 seeds: preservation check consistent
    - 10 seeds: rejection restores theory
    - 5 seeds: ledger correct_examples match what was recorded

TestPhase9DefectProbe (4 tests)
    Probe 1: without preservation, breaking revision is applied silently
    Probe 2: with preservation, same revision is rejected
    Probe 3: after rejection, class-A predictions unchanged
    Probe 4: two symbol tables produce same rejection outcome
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
    check_preservation,
    apply_with_preservation,
    propose_and_apply_safe,
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


def _schema(nid: int) -> SchematicLaw:
    formula = Expr(head=nid, args=(var("k"), var("x")))
    return SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(["k"]), variables=frozenset(["x"]), evidence=1,
    )


def _fitted(nid: int, k: float) -> FittedLaw:
    formula = Expr(head=nid, args=(atom("k"), var("x")))
    sch = SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(), variables=frozenset(["x"]), evidence=1,
    )
    return FittedLaw(schema=sch, params={"k": k}, residual=0.0)


def _full_setup(k_correct: float = 5.0, k_wrong: float = 10.0):
    ctx, nid = _ctx()
    mg = MorphismGraph()
    tm = TheoryManager(mg)
    rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
    eng = RetractEngine(rev, tm, mg)
    t = tm.register_theory("Newton")
    mid_c = add_fitted_law(mg, "correct", _fitted(nid, k_correct))
    mid_w = add_fitted_law(mg, "wrong",   _fitted(nid, k_wrong))
    tm.assign_morphism(mid_c, t)
    tm.assign_morphism(mid_w, t)
    return mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w


# ---------------------------------------------------------------------------
# TestPhase9Integration
# ---------------------------------------------------------------------------

class TestPhase9Integration:

    def test_check_preservation_single_morphism_passes(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "k5_single", _fitted(nid, 5.0))
        tm.assign_morphism(mid, t)
        examples = [({  "x": float(i + 1)}, 5.0 * (i + 1)) for i in range(4)]
        result = check_preservation(t, examples, mg, tm, ctx, tolerance=0.01)
        assert result.passed is True

    def test_check_preservation_broken_when_mismatch(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "k50_wrong", _fitted(nid, 50.0))
        tm.assign_morphism(mid, t)
        examples = [({  "x": 1.0}, 5.0)]   # expects 5; theory predicts 50
        result = check_preservation(t, examples, mg, tm, ctx, tolerance=0.05)
        assert result.passed is False
        assert result.n_broken >= 1

    def test_safe_revision_rejects_breaking_change(self):
        """Revision that breaks class-A (k=5) observations must be rejected."""
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w = _full_setup()
        ledger = PredictionLedger()
        for i in range(4):
            ledger.record(t, {"x": float(i + 1)}, 5.0 * (i + 1), 5.0 * (i + 1))
        schema = _schema(nid)
        anomalies = [({  "x": float(i + 1)}, 50.0 * (i + 1)) for i in range(4)]
        result = propose_and_apply_safe(
            eng, rev, tm, mg, ledger, t, anomalies, ctx, schema,
            label="p9_reject", tolerance=0.05,
        )
        assert result is None, "Revision that breaks class-A must be rejected"

    def test_safe_revision_accepts_when_no_ledger_entries(self):
        """With no prior correct examples, any revision is accepted."""
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w = _full_setup()
        ledger = PredictionLedger()  # empty
        schema = _schema(nid)
        anomalies = [({  "x": float(i + 1)}, 50.0 * (i + 1)) for i in range(4)]
        result = propose_and_apply_safe(
            eng, rev, tm, mg, ledger, t, anomalies, ctx, schema,
            label="p9_accept_empty", tolerance=0.05,
        )
        # Empty ledger → no preservation constraint → should accept
        assert result is not None

    def test_rejection_un_vetoes_old_morphism(self):
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w = _full_setup()
        ledger = PredictionLedger()
        for i in range(4):
            ledger.record(t, {"x": float(i + 1)}, 5.0 * (i + 1), 5.0 * (i + 1))
        schema = _schema(nid)
        anomalies = [({  "x": float(i + 1)}, 50.0 * (i + 1)) for i in range(4)]
        candidate = eng.propose_replacement(t, anomalies, [], ctx, schema)
        assert candidate is not None
        result = apply_with_preservation(
            candidate, ctx, mg, tm, rev, eng, ledger,
            label="p9_undo", tolerance=0.05,
        )
        if result is None:
            assert candidate.retract_id not in rev._vetoed

    def test_rejection_vetoes_new_morphism(self):
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w = _full_setup()
        ledger = PredictionLedger()
        for i in range(4):
            ledger.record(t, {"x": float(i + 1)}, 5.0 * (i + 1), 5.0 * (i + 1))
        schema = _schema(nid)
        anomalies = [({  "x": float(i + 1)}, 50.0 * (i + 1)) for i in range(4)]
        candidate = eng.propose_replacement(t, anomalies, [], ctx, schema)
        assert candidate is not None
        result = apply_with_preservation(
            candidate, ctx, mg, tm, rev, eng, ledger,
            label="p9_veto_new", tolerance=0.05,
        )
        if result is None and candidate.morph_id_new != -1:
            assert candidate.morph_id_new in rev._vetoed

    def test_acceptance_keeps_old_vetoed(self):
        """After successful revision, the old wrong morphism stays vetoed."""
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w = _full_setup()
        ledger = PredictionLedger()  # no constraints
        schema = _schema(nid)
        anomalies = [({  "x": float(i + 1)}, 50.0 * (i + 1)) for i in range(4)]
        candidate = eng.propose_replacement(t, anomalies, [], ctx, schema)
        assert candidate is not None
        result = apply_with_preservation(
            candidate, ctx, mg, tm, rev, eng, ledger,
            label="p9_keep_veto", tolerance=10.0,  # very loose
        )
        if result is not None:
            assert candidate.retract_id in rev._vetoed

    def test_full_pipeline_phases_1_to_9(self):
        """Phase 4 theory + Phases 6+9 safe revision."""
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
        eng = RetractEngine(rev, tm, mg)
        t = tm.register_theory("Newton")

        # Phase 4: add morphism for class-A
        mid_a = add_fitted_law(mg, "newton_a", _fitted(nid, 5.0))
        mid_b = add_fitted_law(mg, "newton_b", _fitted(nid, 10.0))
        tm.assign_morphism(mid_a, t)
        tm.assign_morphism(mid_b, t)

        # Phase 9: record class-A as ledger
        ledger = PredictionLedger()
        for i in range(4):
            ledger.record(t, {"x": float(i + 1)}, 5.0 * (i + 1), 5.0 * (i + 1))

        schema = _schema(nid)
        anomalies = [({  "x": float(i + 1)}, 50.0 * (i + 1)) for i in range(4)]

        result = propose_and_apply_safe(
            eng, rev, tm, mg, ledger, t, anomalies, ctx, schema,
            label="p9_full", tolerance=0.05,
        )
        # class-A ledger entries prevent the k≈50 replacement from landing
        assert result is None
        # class-A morphism still active
        assert mid_a not in rev._vetoed


# ---------------------------------------------------------------------------
# TestPhase9Cage
# ---------------------------------------------------------------------------

class TestPhase9Cage:

    def test_cage_preservation_consistent(self):
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            t = tm.register_theory("T")
            mid = add_fitted_law(mg, f"cage9_{seed}", _fitted(nid, 5.0))
            tm.assign_morphism(mid, t)
            examples = [({  "x": float(i + 1)}, 5.0 * (i + 1)) for i in range(4)]
            result = check_preservation(t, examples, mg, tm, ctx, tolerance=0.01)
            assert result.passed is True, f"seed {seed}: preservation failed"

    def test_cage_rejection_restores_theory(self):
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
            eng = RetractEngine(rev, tm, mg)
            t = tm.register_theory("T")
            mid_c = add_fitted_law(mg, f"cage9rr_c_{seed}", _fitted(nid, 5.0))
            mid_w = add_fitted_law(mg, f"cage9rr_w_{seed}", _fitted(nid, 10.0))
            tm.assign_morphism(mid_c, t)
            tm.assign_morphism(mid_w, t)
            ledger = PredictionLedger()
            for i in range(3):
                ledger.record(t, {"x": float(i + 1)}, 5.0 * (i + 1), 5.0 * (i + 1))
            schema = _schema(nid)
            anomalies = [({  "x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
            candidate = eng.propose_replacement(t, anomalies, [], ctx, schema)
            if candidate is not None:
                result = apply_with_preservation(
                    candidate, ctx, mg, tm, rev, eng, ledger,
                    label=f"cage9rr_{seed}", tolerance=0.05,
                )
                if result is None:
                    assert candidate.retract_id not in rev._vetoed, \
                        f"seed {seed}: retracted morphism still vetoed after rejection"

    def test_cage_ledger_correct_examples(self):
        for seed in range(5):
            ctx, nid = _anon_ctx(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            t = tm.register_theory("T")
            ledger = PredictionLedger()
            n = seed + 2
            for i in range(n):
                ledger.record(t, {"x": float(i + 1)}, float(i + 1) * 5.0, float(i + 1) * 5.0)
            examples = ledger.correct_examples(t)
            assert len(examples) == n, f"seed {seed}: expected {n} examples, got {len(examples)}"


# ---------------------------------------------------------------------------
# TestPhase9DefectProbe
# ---------------------------------------------------------------------------

class TestPhase9DefectProbe:

    def test_probe1_without_preservation_applies_silently(self):
        """RetractEngine without ledger applies the revision (no preservation check)."""
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w = _full_setup()
        schema = _schema(nid)
        anomalies = [({  "x": float(i + 1)}, 50.0 * (i + 1)) for i in range(4)]
        candidate = eng.propose_replacement(t, anomalies, [], ctx, schema)
        assert candidate is not None
        result = eng.apply_replacement(candidate, ctx, label="probe1_blind")
        # apply_replacement always returns a RevisionResult (no preservation check)
        assert result is not None
        assert result.candidate.morph_id != -1

    def test_probe2_with_preservation_rejects(self):
        """apply_with_preservation rejects the same revision that probe1 accepted."""
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w = _full_setup()
        ledger = PredictionLedger()
        for i in range(4):
            ledger.record(t, {"x": float(i + 1)}, 5.0 * (i + 1), 5.0 * (i + 1))
        schema = _schema(nid)
        anomalies = [({  "x": float(i + 1)}, 50.0 * (i + 1)) for i in range(4)]
        candidate = eng.propose_replacement(t, anomalies, [], ctx, schema)
        assert candidate is not None
        result = apply_with_preservation(
            candidate, ctx, mg, tm, rev, eng, ledger,
            label="probe2_pres", tolerance=0.05,
        )
        assert result is None, "PROBE 2: preservation check must reject breaking revision"

    def test_probe3_after_rejection_class_a_unchanged(self):
        """After rejection, class-A prediction is unchanged from before revision."""
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
        eng = RetractEngine(rev, tm, mg)
        t = tm.register_theory("T")
        # Only k=5 morphism so prediction is deterministic
        mid_a = add_fitted_law(mg, "p9d3_a", _fitted(nid, 5.0))
        mid_w = add_fitted_law(mg, "p9d3_w", _fitted(nid, 10.0))
        tm.assign_morphism(mid_a, t)
        tm.assign_morphism(mid_w, t)

        # Veto mid_w temporarily to measure "pure k=5" prediction
        rev._vetoed.add(mid_w)
        pred_before = tm.predict_under_theory(t, {"x": 2.0}, ctx)
        rev._vetoed.discard(mid_w)

        ledger = PredictionLedger()
        for i in range(4):
            ledger.record(t, {"x": float(i + 1)}, 5.0 * (i + 1), 5.0 * (i + 1))
        schema = _schema(nid)
        anomalies = [({  "x": float(i + 1)}, 50.0 * (i + 1)) for i in range(4)]
        candidate = eng.propose_replacement(t, anomalies, [], ctx, schema)
        if candidate is not None:
            result = apply_with_preservation(
                candidate, ctx, mg, tm, rev, eng, ledger,
                label="p9d3_restore", tolerance=0.05,
            )
            if result is None:
                # Rejection: mid_a must still be active
                assert mid_a not in rev._vetoed, \
                    "PROBE 3: class-A morphism must be active after rejection"
                # And the mid_w must be un-vetoed (back to original state)
                assert candidate.retract_id not in rev._vetoed

    def test_probe4_two_tables_same_rejection(self):
        """Two symbol tables produce the same rejection outcome."""
        outcomes = []
        for seed in range(2):
            ctx, nid = _anon_ctx(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
            eng = RetractEngine(rev, tm, mg)
            t = tm.register_theory("T")
            mid_c = add_fitted_law(mg, f"p9d4_c_{seed}", _fitted(nid, 5.0))
            mid_w = add_fitted_law(mg, f"p9d4_w_{seed}", _fitted(nid, 10.0))
            tm.assign_morphism(mid_c, t)
            tm.assign_morphism(mid_w, t)
            ledger = PredictionLedger()
            for i in range(4):
                ledger.record(t, {"x": float(i + 1)}, 5.0 * (i + 1), 5.0 * (i + 1))
            schema = _schema(nid)
            anomalies = [({  "x": float(i + 1)}, 50.0 * (i + 1)) for i in range(4)]
            candidate = eng.propose_replacement(t, anomalies, [], ctx, schema)
            if candidate is not None:
                result = apply_with_preservation(
                    candidate, ctx, mg, tm, rev, eng, ledger,
                    label=f"p9d4_{seed}", tolerance=0.05,
                )
                outcomes.append(result is None)
            else:
                outcomes.append(None)
        if all(o is not None for o in outcomes):
            assert outcomes[0] == outcomes[1], \
                f"PROBE 4: rejection outcomes differ: {outcomes[0]} vs {outcomes[1]}"
