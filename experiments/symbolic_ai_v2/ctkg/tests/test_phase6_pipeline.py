"""
Phase 6 Integration Test — Full Revision Cycle: Add, Retract, Replace.

Scenario
--------
A Newton theory has two morphisms: a correct one (k=5 for class A) and a wrong
one (k=10, wrong for class C where true k=50).

The RetractEngine:
  1. propose_retraction: identifies mid_wrong (k=10) as the morphism to remove
  2. propose_replacement: fits a new law from class-C anomalies
  3. apply_replacement: retracts mid_wrong, adds new law, verifies improvement
  4. After replacement: mid_correct (k=5) still active; mid_wrong is vetoed;
     new law predicts class C correctly.

Test classes
------------
TestPhase6Integration (8 tests)
    - propose_retraction returns mid_wrong
    - propose_replacement not None
    - replacement.retract_id == mid_wrong
    - apply_replacement: mid_wrong vetoed
    - apply_replacement: new morphism in graph
    - apply_replacement: history logged
    - class A still predicted correctly
    - class C better predicted after replacement

TestPhase6Cage (3 tests)
    10 anonymous seeds, all pass.

TestPhase6DefectProbe (4 tests)
    Probe: retraction without preservation check — caught by net_gain scoring.
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
)
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager
from experiments.symbolic_ai_v2.ctkg.inference.revision import ClosedLoopReviser
from experiments.symbolic_ai_v2.ctkg.inference.retract import RetractEngine


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


def _setup():
    ctx, nid = _ctx()
    mg = MorphismGraph()
    tm = TheoryManager(mg)
    rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
    eng = RetractEngine(rev, tm, mg)
    t = tm.register_theory("Newton")

    # Correct morphism: k=5 (class A: obs = 5*x)
    mid_correct = add_fitted_law(mg, "correct", _fitted(nid, 5.0))
    # Wrong morphism: k=10 (wrong for class C: obs = 50*x)
    mid_wrong = add_fitted_law(mg, "wrong", _fitted(nid, 10.0))
    tm.assign_morphism(mid_correct, t)
    tm.assign_morphism(mid_wrong, t)

    schema = _schema(nid)
    # Class C anomalies (k=10 is wrong; true=50)
    anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(4)]
    # Class A correct examples (k=5 is right)
    correct = [({"x": float(i + 1)}, 5.0 * (i + 1)) for i in range(4)]

    return mg, tm, ctx, nid, rev, eng, t, mid_correct, mid_wrong, schema, anomalies, correct


# ---------------------------------------------------------------------------
# TestPhase6Integration
# ---------------------------------------------------------------------------

class TestPhase6Integration:

    def test_propose_retraction_finds_wrong_morphism(self):
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w, schema, anomalies, correct = _setup()
        cand = eng.propose_retraction(t, anomalies, correct, ctx)
        assert cand is not None
        assert cand.morph_id == mid_w

    def test_propose_replacement_not_none(self):
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w, schema, anomalies, correct = _setup()
        cand = eng.propose_replacement(t, anomalies, correct, ctx, schema)
        assert cand is not None

    def test_replacement_retract_id_is_wrong_morphism(self):
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w, schema, anomalies, correct = _setup()
        cand = eng.propose_replacement(t, anomalies, correct, ctx, schema)
        assert cand is not None
        assert cand.retract_id == mid_w

    def test_apply_vetoes_wrong_morphism(self):
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w, schema, anomalies, correct = _setup()
        cand = eng.propose_replacement(t, anomalies, correct, ctx, schema)
        assert cand is not None
        eng.apply_replacement(cand, ctx, label="phase6_new")
        assert mid_w in rev._vetoed

    def test_apply_adds_new_morphism(self):
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w, schema, anomalies, correct = _setup()
        cand = eng.propose_replacement(t, anomalies, correct, ctx, schema)
        assert cand is not None
        result = eng.apply_replacement(cand, ctx, label="phase6_new2")
        assert mg.morphism_by_id(result.candidate.morph_id) is not None
        assert mg.morphism_by_id(result.candidate.morph_id).morph_type == "FITTED_LAW"

    def test_apply_logs_history(self):
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w, schema, anomalies, correct = _setup()
        cand = eng.propose_replacement(t, anomalies, correct, ctx, schema)
        assert cand is not None
        eng.apply_replacement(cand, ctx, label="phase6_hist")
        h = eng.history(t)
        assert len(h) >= 1

    def test_correct_morphism_not_vetoed(self):
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w, schema, anomalies, correct = _setup()
        cand = eng.propose_replacement(t, anomalies, correct, ctx, schema)
        assert cand is not None
        eng.apply_replacement(cand, ctx, label="phase6_pres")
        assert mid_c not in rev._vetoed

    def test_class_c_prediction_improves(self):
        mg, tm, ctx, nid, rev, eng, t, mid_c, mid_w, schema, anomalies, correct = _setup()
        # Before: mean of k=5 and k=10 = 7.5 → pred = 7.5 for x=1; obs=50
        pred_before = rev._predict_excluding(t, {"x": 1.0}, ctx)
        cand = eng.propose_replacement(t, anomalies, correct, ctx, schema)
        assert cand is not None
        eng.apply_replacement(cand, ctx, label="phase6_c")
        pred_after = rev._predict_excluding(t, {"x": 1.0}, ctx)
        # After: k=10 vetoed, new law k≈approx from combined obs; pred should move up
        assert pred_after > pred_before


# ---------------------------------------------------------------------------
# TestPhase6Cage
# ---------------------------------------------------------------------------

def _anon(seed: int) -> tuple[EvalContext, int]:
    rng = random.Random(seed)
    sym = chr(0x2200 + rng.randint(0, 0xFF))
    nid = TOKEN_GRAPH.encode(sym)
    return EvalContext({nid: lambda a, b: a * b}), nid


class TestPhase6Cage:

    def test_cage_propose_replacement(self):
        for seed in range(10):
            ctx, nid = _anon(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
            eng = RetractEngine(rev, tm, mg)
            t = tm.register_theory("T")
            mid_w = add_fitted_law(mg, f"cage_w_{seed}", _fitted(nid, 10.0))
            tm.assign_morphism(mid_w, t)
            schema = _schema(nid)
            anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
            cand = eng.propose_replacement(t, anomalies, [], ctx, schema)
            assert cand is not None, f"seed {seed}: no replacement proposed"

    def test_cage_retraction_vetoes_wrong(self):
        for seed in range(10):
            ctx, nid = _anon(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
            eng = RetractEngine(rev, tm, mg)
            t = tm.register_theory("T")
            mid_w = add_fitted_law(mg, f"cage_rv_{seed}", _fitted(nid, 10.0))
            tm.assign_morphism(mid_w, t)
            schema = _schema(nid)
            anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
            cand = eng.propose_replacement(t, anomalies, [], ctx, schema)
            if cand is not None:
                eng.apply_replacement(cand, ctx, label=f"cage_new_{seed}")
                assert mid_w in rev._vetoed, f"seed {seed}: wrong morphism not vetoed"

    def test_cage_preservation_correct_not_vetoed(self):
        for seed in range(10):
            ctx, nid = _anon(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
            eng = RetractEngine(rev, tm, mg)
            t = tm.register_theory("T")
            mid_ok = add_fitted_law(mg, f"cage_ok_{seed}", _fitted(nid, 5.0))
            mid_w  = add_fitted_law(mg, f"cage_bw_{seed}", _fitted(nid, 10.0))
            tm.assign_morphism(mid_ok, t)
            tm.assign_morphism(mid_w, t)
            schema = _schema(nid)
            anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
            correct = [({"x": float(i + 1)}, 5.0 * (i + 1)) for i in range(3)]
            cand = eng.propose_replacement(t, anomalies, correct, ctx, schema)
            if cand is not None:
                eng.apply_replacement(cand, ctx, label=f"cage_pnv_{seed}")
                assert mid_ok not in rev._vetoed, f"seed {seed}: correct morphism vetoed"


# ---------------------------------------------------------------------------
# TestPhase6DefectProbe
# ---------------------------------------------------------------------------

class TestPhase6DefectProbe:

    def test_probe_retraction_respects_net_gain(self):
        """propose_retraction returns None when net_gain ≤ 0 (preserves correct)."""
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
        eng = RetractEngine(rev, tm, mg)
        t = tm.register_theory("T")
        # k=50 is correct for A and "wrong" for B (tolerance=0.01 strict)
        mid = add_fitted_law(mg, "dp_law", _fitted(nid, 50.0))
        tm.assign_morphism(mid, t)
        anomalies = [({"x": 1.0}, 10.0)]   # 1 anomaly
        correct = [({"x": 1.0}, 50.0)]      # 1 correct
        cand = eng.propose_retraction(t, anomalies, correct, ctx)
        # 1 - 1 = 0 → no retraction
        assert cand is None

    def test_probe_replacement_adds_fitted_law_not_obs_seq(self):
        """apply_replacement must write FITTED_LAW, never OBS_SEQ."""
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
        eng = RetractEngine(rev, tm, mg)
        t = tm.register_theory("T")
        mid_w = add_fitted_law(mg, "dp2_w", _fitted(nid, 10.0))
        tm.assign_morphism(mid_w, t)
        schema = _schema(nid)
        anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
        cand = eng.propose_replacement(t, anomalies, [], ctx, schema)
        assert cand is not None
        eng.apply_replacement(cand, ctx, label="dp2_new")
        for m in mg.morphisms():
            assert m.morph_type != "OBS_SEQ", \
                f"OBS_SEQ morphism created by RetractEngine: {m}"

    def test_probe_history_after_apply(self):
        """history must contain the retracted morphism after apply_replacement."""
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
        eng = RetractEngine(rev, tm, mg)
        t = tm.register_theory("T")
        mid_w = add_fitted_law(mg, "dp3_w", _fitted(nid, 10.0))
        tm.assign_morphism(mid_w, t)
        schema = _schema(nid)
        anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
        cand = eng.propose_replacement(t, anomalies, [], ctx, schema)
        assert cand is not None
        eng.apply_replacement(cand, ctx, label="dp3_new")
        h = eng.history(t)
        retracted_ids = [e[0] for e in h]
        assert cand.retract_id in retracted_ids

    def test_probe_correct_morphism_survives(self):
        """After apply_replacement, a morphism covering class A must still be active."""
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
        eng = RetractEngine(rev, tm, mg)
        t = tm.register_theory("T")
        mid_a = add_fitted_law(mg, "dp4_a", _fitted(nid, 5.0))   # correct for A
        mid_w = add_fitted_law(mg, "dp4_w", _fitted(nid, 10.0))  # wrong for C
        tm.assign_morphism(mid_a, t)
        tm.assign_morphism(mid_w, t)
        schema = _schema(nid)
        anomalies = [({"x": float(i + 1)}, 50.0 * (i + 1)) for i in range(3)]
        correct = [({"x": float(i + 1)}, 5.0 * (i + 1)) for i in range(3)]
        cand = eng.propose_replacement(t, anomalies, correct, ctx, schema)
        if cand is not None:
            eng.apply_replacement(cand, ctx, label="dp4_new")
            assert mid_a not in rev._vetoed
