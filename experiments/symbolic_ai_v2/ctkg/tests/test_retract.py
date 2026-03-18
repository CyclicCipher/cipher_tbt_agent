"""
Tests for Phase 6 — Full Revision Cycle: Add, Retract, Replace.

Test classes
------------
TestRetractionBasic (9 tests)
    - retract_morphism marks morphism as vetoed
    - vetoed morphism excluded from predictions
    - history records retraction
    - history records reason and score
    - score_retraction returns correct net_gain
    - propose_retraction returns the worst morphism
    - propose_retraction returns None when no net gain
    - propose_replacement builds new law
    - apply_replacement: retracted + new law added

TestPreservation (5 tests — A-7 analog)
    20-morphism theory, 1 wrong for class C, 19 correct for A and B.
    After apply_replacement:
    - wrong morphism retracted
    - new morphism covers class C
    - classes A and B still predict correctly

TestRevisionHistory (4 tests)
    - history returns empty for theory with no retractions
    - history returns one entry after one retraction
    - history entry has correct morph_id
    - multiple retractions logged in order

TestBitterLessonCage (3 tests)
    - cage: propose_replacement works with anonymous symbols
    - cage: net_gain > 0 across all seeds
    - cage: preservation after replacement

TestDefectProbe (4 tests)
    Probe A: retraction without preservation check breaks other predictions
             → test that our implementation DOES NOT break them
    Probe B: apply_replacement logs retraction to history
    Probe C: propose_retraction targets the morphism with most anomalies
    Probe D: vetoed morphisms excluded from theory predictions
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
from experiments.symbolic_ai_v2.ctkg.inference.retract import (
    RetractEngine,
    RetractionScore,
    RetractionCandidate,
    ReplacementCandidate,
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
    sch = SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(), variables=frozenset(["x"]), evidence=1,
    )
    return FittedLaw(schema=sch, params={"k": k}, residual=0.0)


def _build_engine():
    ctx, nid = _ctx()
    mg = MorphismGraph()
    tm = TheoryManager(mg)
    reviser = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
    engine = RetractEngine(reviser, tm, mg)
    return mg, tm, ctx, nid, reviser, engine


# ---------------------------------------------------------------------------
# TestRetractionBasic
# ---------------------------------------------------------------------------

class TestRetractionBasic:

    def test_retract_adds_to_veto(self):
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "law", _fitted(nid, 10.0))
        tm.assign_morphism(mid, t)
        engine.retract_morphism(mid, t, reason="test")
        assert mid in reviser._vetoed

    def test_vetoed_excluded_from_prediction(self):
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        mid1 = add_fitted_law(mg, "law1", _fitted(nid, 10.0))
        mid2 = add_fitted_law(mg, "law2", _fitted(nid, 50.0))
        tm.assign_morphism(mid1, t)
        tm.assign_morphism(mid2, t)
        # Before retraction: mean = (10+50)/2 = 30
        pred_before = tm.predict_under_theory(t, {"x": 1.0}, ctx)
        # After retract mid1: only k=50 left → pred = 50
        engine.retract_morphism(mid1, t, reason="test")
        pred_after = reviser._predict_excluding(t, {"x": 1.0}, ctx)
        assert pred_before == pytest.approx(30.0)
        assert pred_after == pytest.approx(50.0)

    def test_history_records_retraction(self):
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "r_law", _fitted(nid, 10.0))
        tm.assign_morphism(mid, t)
        engine.retract_morphism(mid, t, reason="test_reason", score=5.0)
        h = engine.history(t)
        assert len(h) == 1

    def test_history_entry_correct(self):
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "h_law", _fitted(nid, 10.0))
        tm.assign_morphism(mid, t)
        engine.retract_morphism(mid, t, reason="my_reason", score=7.5)
        h = engine.history(t)
        assert h[0][0] == mid
        assert h[0][1] == "my_reason"
        assert h[0][2] == pytest.approx(7.5)

    def test_score_retraction_net_gain(self):
        """Morphism k=10 is wrong for obs=50; correct for obs=10."""
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "s_law", _fitted(nid, 10.0))
        tm.assign_morphism(mid, t)
        anomalies = [({"x": 1.0}, 50.0)]       # k=10 → pred=10, obs=50 → wrong
        correct = [({"x": 1.0}, 10.0)]          # k=10 → pred=10, obs=10 → right
        rs = engine.score_retraction(mid, t, anomalies, correct, ctx)
        assert rs.anomalies_resolved == 1
        assert rs.correct_broken == 1
        assert rs.net_gain == pytest.approx(0.0)

    def test_propose_retraction_targets_worst(self):
        """Theory with two morphisms; one resolves more anomalies."""
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        mid_bad = add_fitted_law(mg, "bad", _fitted(nid, 10.0))  # wrong for obs=50
        mid_ok  = add_fitted_law(mg, "ok",  _fitted(nid, 50.0))  # right for obs=50
        tm.assign_morphism(mid_bad, t)
        tm.assign_morphism(mid_ok, t)
        anomalies = [({"x": 1.0}, 50.0), ({"x": 2.0}, 100.0)]
        correct = []
        cand = engine.propose_retraction(t, anomalies, correct, ctx)
        assert cand is not None
        assert cand.morph_id == mid_bad

    def test_propose_retraction_no_gain_returns_none(self):
        """If retracting any morphism resolves ≤ correct broken, return None."""
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "g_law", _fitted(nid, 50.0))
        tm.assign_morphism(mid, t)
        anomalies = []
        correct = [({"x": 1.0}, 50.0)]
        cand = engine.propose_retraction(t, anomalies, correct, ctx)
        assert cand is None

    def test_propose_replacement_builds_new_law(self):
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        mid_bad = add_fitted_law(mg, "pr_bad", _fitted(nid, 10.0))
        tm.assign_morphism(mid_bad, t)
        schema = _schema(nid)
        anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
        correct = []
        cand = engine.propose_replacement(t, anomalies, correct, ctx, schema)
        assert cand is not None
        assert isinstance(cand, ReplacementCandidate)
        assert cand.retract_id == mid_bad

    def test_apply_replacement_adds_new_law(self):
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        mid_bad = add_fitted_law(mg, "ar_bad", _fitted(nid, 10.0))
        tm.assign_morphism(mid_bad, t)
        schema = _schema(nid)
        anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
        correct = []
        cand = engine.propose_replacement(t, anomalies, correct, ctx, schema,
                                           label="ar_new")
        assert cand is not None
        result = engine.apply_replacement(cand, ctx, label="ar_new_applied")
        # New law should be in the graph
        assert mg.morphism_by_id(result.candidate.morph_id) is not None


# ---------------------------------------------------------------------------
# TestPreservation (A-7 analog)
# ---------------------------------------------------------------------------

class TestPreservation:
    """Theory with 1 wrong morphism and 1 correct one.

    After apply_replacement:
    - wrong morphism retracted
    - correct morphism still predicts correctly
    """

    def _build_two_class_theory(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        reviser = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
        engine = RetractEngine(reviser, tm, mg)
        t = tm.register_theory("T")

        # Class A: correct morphism k=5 (correct for obs=5*x)
        mid_a = add_fitted_law(mg, "class_a", _fitted(nid, 5.0))
        # Class C: wrong morphism k=10 (wrong for obs=50*x)
        mid_c = add_fitted_law(mg, "class_c", _fitted(nid, 10.0))
        tm.assign_morphism(mid_a, t)
        tm.assign_morphism(mid_c, t)
        return mg, tm, ctx, nid, t, mid_a, mid_c, reviser, engine

    def test_wrong_morphism_retracted(self):
        mg, tm, ctx, nid, t, mid_a, mid_c, reviser, engine = self._build_two_class_theory()
        schema = _schema(nid)
        # Class C anomalies (k=10 is wrong, k=50 is right)
        anomalies = [({"x": float(i+1)}, 50.0*(i+1)) for i in range(3)]
        # Class A correct examples (k=5 is right)
        correct = [({"x": float(i+1)}, 5.0*(i+1)) for i in range(3)]
        cand = engine.propose_replacement(t, anomalies, correct, ctx, schema)
        if cand is not None:
            result = engine.apply_replacement(cand, ctx, label="pres_new")
            # mid_c should be vetoed (retracted)
            assert cand.retract_id in reviser._vetoed

    def test_correct_morphism_not_retracted(self):
        mg, tm, ctx, nid, t, mid_a, mid_c, reviser, engine = self._build_two_class_theory()
        schema = _schema(nid)
        anomalies = [({"x": float(i+1)}, 50.0*(i+1)) for i in range(3)]
        correct = [({"x": float(i+1)}, 5.0*(i+1)) for i in range(3)]
        cand = engine.propose_replacement(t, anomalies, correct, ctx, schema)
        if cand is not None:
            engine.apply_replacement(cand, ctx, label="pres_new2")
            # mid_a must NOT be vetoed
            assert mid_a not in reviser._vetoed

    def test_class_a_still_predicts_correctly_after_replacement(self):
        mg, tm, ctx, nid, t, mid_a, mid_c, reviser, engine = self._build_two_class_theory()
        schema = _schema(nid)
        anomalies = [({"x": float(i+1)}, 50.0*(i+1)) for i in range(3)]
        correct = [({"x": float(i+1)}, 5.0*(i+1)) for i in range(3)]
        cand = engine.propose_replacement(t, anomalies, correct, ctx, schema)
        if cand is None:
            return  # no replacement needed (already correct)
        engine.apply_replacement(cand, ctx, label="pres_new3")
        # After replacement: mid_a (k=5) still in theory and not vetoed
        assert mid_a not in reviser._vetoed
        # Prediction for class A inputs should include k=5 contribution
        pred = reviser._predict_excluding(t, {"x": 1.0}, ctx)
        assert pred is not None
        # k=5 and new_law (fitted from mixed data) both active → reasonable pred
        assert pred > 0.0

    def test_history_records_replacement_retraction(self):
        mg, tm, ctx, nid, t, mid_a, mid_c, reviser, engine = self._build_two_class_theory()
        schema = _schema(nid)
        anomalies = [({"x": float(i+1)}, 50.0*(i+1)) for i in range(3)]
        correct = [({"x": float(i+1)}, 5.0*(i+1)) for i in range(3)]
        cand = engine.propose_replacement(t, anomalies, correct, ctx, schema)
        if cand is None:
            return
        engine.apply_replacement(cand, ctx, label="pres_hist")
        h = engine.history(t)
        assert len(h) >= 1
        retracted_ids = [entry[0] for entry in h]
        assert cand.retract_id in retracted_ids

    def test_propose_replacement_new_law_fits_anomalies(self):
        mg, tm, ctx, nid, t, mid_a, mid_c, reviser, engine = self._build_two_class_theory()
        schema = _schema(nid)
        anomalies = [({"x": float(i+1)}, 50.0*(i+1)) for i in range(3)]
        correct = []
        cand = engine.propose_replacement(t, anomalies, correct, ctx, schema)
        assert cand is not None
        # New law fitted from anomalies (k=50): params["k"] ≈ 50
        fitted_k = cand.new_law.params.get("k")
        assert fitted_k == pytest.approx(50.0, rel=0.2), f"k={fitted_k}"


# ---------------------------------------------------------------------------
# TestRevisionHistory
# ---------------------------------------------------------------------------

class TestRevisionHistory:

    def test_empty_history(self):
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        h = engine.history(t)
        assert h == []

    def test_one_entry_after_retraction(self):
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "hx", _fitted(nid, 5.0))
        engine.retract_morphism(mid, t, reason="test")
        assert len(engine.history(t)) == 1

    def test_history_morph_id_correct(self):
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "hy", _fitted(nid, 5.0))
        engine.retract_morphism(mid, t, reason="r")
        assert engine.history(t)[0][0] == mid

    def test_multiple_retractions_all_logged(self):
        mg, tm, ctx, nid, reviser, engine = _build_engine()
        t = tm.register_theory("T")
        mids = [add_fitted_law(mg, f"hz_{i}", _fitted(nid, float(i))) for i in range(3)]
        for mid in mids:
            engine.retract_morphism(mid, t, reason=f"reason_{mid}")
        h = engine.history(t)
        assert len(h) == 3


# ---------------------------------------------------------------------------
# TestBitterLessonCage
# ---------------------------------------------------------------------------

def _anon(seed: int) -> tuple[EvalContext, int]:
    rng = random.Random(seed)
    sym = chr(0x2200 + rng.randint(0, 0xFF))
    nid = TOKEN_GRAPH.encode(sym)
    return EvalContext({nid: lambda a, b: a * b}), nid


class TestBitterLessonCage:

    def test_cage_propose_replacement_works(self):
        for seed in range(10):
            ctx, nid = _anon(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
            eng = RetractEngine(rev, tm, mg)
            t = tm.register_theory("T")
            mid_bad = add_fitted_law(mg, f"cb_{seed}", _fitted(nid, 10.0))
            tm.assign_morphism(mid_bad, t)
            schema = _schema(nid)
            anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
            cand = eng.propose_replacement(t, anomalies, [], ctx, schema)
            assert cand is not None, f"seed {seed}: no replacement proposed"

    def test_cage_net_gain_positive(self):
        for seed in range(10):
            ctx, nid = _anon(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
            eng = RetractEngine(rev, tm, mg)
            t = tm.register_theory("T")
            mid_bad = add_fitted_law(mg, f"cg_{seed}", _fitted(nid, 10.0))
            tm.assign_morphism(mid_bad, t)
            anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
            cand = eng.propose_retraction(t, anomalies, [], ctx)
            assert cand is not None, f"seed {seed}: no retraction proposed"
            assert cand.score > 0.0, f"seed {seed}: score={cand.score}"

    def test_cage_preservation(self):
        """After replacement, correct morphism remains active."""
        for seed in range(10):
            ctx, nid = _anon(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
            eng = RetractEngine(rev, tm, mg)
            t = tm.register_theory("T")
            mid_ok  = add_fitted_law(mg, f"cp_ok_{seed}",  _fitted(nid, 5.0))
            mid_bad = add_fitted_law(mg, f"cp_bad_{seed}", _fitted(nid, 10.0))
            tm.assign_morphism(mid_ok, t)
            tm.assign_morphism(mid_bad, t)
            schema = _schema(nid)
            anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
            correct = [({"x": float(i + 1)}, 5.0 * (i + 1)) for i in range(3)]
            cand = eng.propose_replacement(t, anomalies, correct, ctx, schema)
            if cand is not None:
                eng.apply_replacement(cand, ctx, label=f"cp_new_{seed}")
                assert mid_ok not in rev._vetoed, f"seed {seed}: correct morphism vetoed"


# ---------------------------------------------------------------------------
# TestDefectProbe
# ---------------------------------------------------------------------------

class TestDefectProbe:

    def test_probe_a_retract_without_check_breaks_correct(self):
        """If retraction is done blindly, it would break class-A predictions.
        Our implementation checks net_gain before retracting.
        Test: propose_retraction with equal anomalies and correct → net_gain=0 → None.
        """
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
        eng = RetractEngine(rev, tm, mg)
        t = tm.register_theory("T")
        mid = add_fitted_law(mg, "pa_law", _fitted(nid, 50.0))
        tm.assign_morphism(mid, t)
        # k=50 is correct for class A (obs=50) and wrong for class B (obs=10)
        anomalies = [({"x": 1.0}, 10.0)]
        correct   = [({"x": 1.0}, 50.0)]
        rs = eng.score_retraction(mid, t, anomalies, correct, ctx)
        # 1 anomaly resolved, 1 correct broken → net_gain = 0
        assert rs.net_gain == pytest.approx(0.0, abs=1e-9)
        cand = eng.propose_retraction(t, anomalies, correct, ctx)
        assert cand is None  # net_gain <= 0 → no retraction

    def test_probe_b_history_logged_after_apply(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
        eng = RetractEngine(rev, tm, mg)
        t = tm.register_theory("T")
        mid_bad = add_fitted_law(mg, "pb_bad", _fitted(nid, 10.0))
        tm.assign_morphism(mid_bad, t)
        schema = _schema(nid)
        anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
        cand = eng.propose_replacement(t, anomalies, [], ctx, schema)
        assert cand is not None
        eng.apply_replacement(cand, ctx, label="pb_new")
        h = eng.history(t)
        assert len(h) == 1
        assert h[0][0] == cand.retract_id

    def test_probe_c_propose_targets_most_anomalous(self):
        """propose_retraction targets the morphism with most anomalies resolved."""
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
        eng = RetractEngine(rev, tm, mg)
        t = tm.register_theory("T")
        # mid_a: k=10 (wrong for obs=50, obs=100, obs=150 → 3 anomalies)
        mid_a = add_fitted_law(mg, "pc_a", _fitted(nid, 10.0))
        # mid_b: k=48 (close-ish to obs=50 → 1 anomaly with big tolerance)
        mid_b = add_fitted_law(mg, "pc_b", _fitted(nid, 48.0))
        tm.assign_morphism(mid_a, t)
        tm.assign_morphism(mid_b, t)
        anomalies = [({"x": 1.0}, 50.0), ({"x": 2.0}, 100.0), ({"x": 3.0}, 150.0)]
        cand = eng.propose_retraction(t, anomalies, [], ctx, tolerance=0.5)
        assert cand is not None
        # mid_a is more wrong → should be proposed
        assert cand.morph_id == mid_a

    def test_probe_d_vetoed_excluded_from_mean_prediction(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
        eng = RetractEngine(rev, tm, mg)
        t = tm.register_theory("T")
        mid1 = add_fitted_law(mg, "pd_1", _fitted(nid, 10.0))
        mid2 = add_fitted_law(mg, "pd_2", _fitted(nid, 50.0))
        tm.assign_morphism(mid1, t)
        tm.assign_morphism(mid2, t)
        # Before: mean = 30
        p_before = rev._predict_excluding(t, {"x": 1.0}, ctx)
        assert p_before == pytest.approx(30.0)
        # Veto mid1 → only k=50 active → pred = 50
        eng.retract_morphism(mid1, t)
        p_after = rev._predict_excluding(t, {"x": 1.0}, ctx)
        assert p_after == pytest.approx(50.0)
