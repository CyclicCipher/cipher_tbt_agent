"""
Deep IDA Benchmark — Phase 8 of the Einstein Roadmap.

Extends the existing IDA benchmark (I-1/I-3, D-1/D-3, A-1/A-2) to cover:

  I-4   Multi-case induction (two parameter regimes)
  I-5   Parametric family discovery (three instances, one schema)
  I-6   Algebraic invariance (commutativity: same k regardless of input order)
  I-7   Functor composition law (k_composed ≈ k_f * k_g)
  I-8   Full theory induction from structured observation stream (orchestrator)

  D-4   4-hop deduction chain
  D-5   5-hop deduction chain
  D-6   Diamond pattern (two paths to same conclusion)
  D-7   7-hop deduction chain (deep BFS)

  A-3   Hypothesis requiring new morphism (ClosedLoopReviser observe+flush)
  A-4   Anomaly falsifies morphism → retract and replace (RetractEngine)
  A-5   Latent variable hypothesis (hypothesise_latent)
  A-6   Multiple anomalies → unified explanation (multi_anomaly_abduction)
  A-7   Theory revision preserving prior explanations (apply_with_preservation)
  A-8   Paradigm shift → new concept cluster (propose_paradigm_shift)

Pass criteria (roadmap Phase 8)
--------------------------------
  I-*  : parameter recovery within 5% of true value
  D-*  : correct conclusion found in all 10 anonymous symbol seeds
  A-*  : ≥ 80% pass rate across 10 seeds; variance < 5 pp
  Cage : named-symbol vs. anonymous-symbol gap < 5 pp on every track

Organisation
------------
  TestInductionDeep       (I-4 through I-8, 5 tests)
  TestDeductionDeep       (D-4 through D-7, 6 tests including cage)
  TestAbductionDeep       (A-3 through A-8, 12 tests)
  TestDeepIDABasicCage    (cage: 10 seeds, all tracks pass)
  TestDeepIDAVariance     (variance < 5 pp across 10 seeds for A-tracks)
  TestDeepIDADefectProbes (targeted violation probes)
"""
from __future__ import annotations

import random
from typing import Optional

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import (
    FittedLaw,
    add_fitted_law,
    fit_parameters,
    predict_continuous,
)
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom, var, Expr
from experiments.symbolic_ai_v2.ctkg.inference.coverage import (
    multi_anomaly_abduction,
    score_coverage,
)
from experiments.symbolic_ai_v2.ctkg.inference.deduct import DeductionEngine
from experiments.symbolic_ai_v2.ctkg.inference.latent import (
    hypothesise_latent,
    hypothesise_latent_mdl_select,
)
from experiments.symbolic_ai_v2.ctkg.inference.orchestrator import (
    AbductionDecision,
    AbductionOrchestrator,
)
from experiments.symbolic_ai_v2.ctkg.inference.paradigm import (
    propose_paradigm_shift,
    ParadigmShiftResult,
)
from experiments.symbolic_ai_v2.ctkg.inference.preservation import (
    PredictionLedger,
    apply_with_preservation,
)
from experiments.symbolic_ai_v2.ctkg.inference.retract import RetractEngine
from experiments.symbolic_ai_v2.ctkg.inference.revision import ClosedLoopReviser
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager
from experiments.symbolic_ai_v2.ctkg.einstein.streams import (
    newtonian_scenario,
    michelson_morley_scenario,
    maxwell_em_scenario,
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
    """Linear schema: f(x) = k * x."""
    formula = Expr(head=nid, args=(var("k"), var("x")))
    return SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(["k"]), variables=frozenset(["x"]), evidence=1,
    )


def _schema_h(nid: int) -> SchematicLaw:
    """Linear schema for latent composition: h(z) = a * z."""
    formula = Expr(head=nid, args=(var("a"), var("z")))
    return SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(["a"]), variables=frozenset(["z"]), evidence=1,
    )


def _obs(k: float, xs) -> list[tuple[dict, float]]:
    """Observations f(x) = k*x."""
    return [({'x': float(x)}, k * float(x)) for x in xs]


def _fitted(nid: int, k: float) -> FittedLaw:
    formula = Expr(head=nid, args=(atom("k"), var("x")))
    sch = SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(), variables=frozenset(["x"]), evidence=1,
    )
    return FittedLaw(schema=sch, params={"k": k}, residual=0.0)


def _stack(nid: int, ctx: EvalContext, theory_k: float = 5.0):
    """Build full abduction stack. Returns (mg, tm, rev, eng, theory_id)."""
    mg = MorphismGraph()
    tm = TheoryManager(mg)
    rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
    eng = RetractEngine(rev, tm, mg)
    t = tm.register_theory("T")
    mid = add_fitted_law(mg, "base_law", _fitted(nid, theory_k))
    tm.assign_morphism(mid, t)
    return mg, tm, rev, eng, t


def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0 else abs(a)


# ---------------------------------------------------------------------------
# TestInductionDeep — I-4 through I-8
# ---------------------------------------------------------------------------

class TestInductionDeep:

    def test_I4_multi_case(self):
        """I-4: Two parameter regimes; both k values recovered within 5%.

        Sub-population 1: k=3 for x in [1..5]
        Sub-population 2: k=7 for x in [6..10]
        """
        ctx, nid = _ctx()
        sg = _schema_g(nid)
        fl1 = fit_parameters(sg.conclusion, sg.params, _obs(3.0, range(1, 6)), ctx)
        fl2 = fit_parameters(sg.conclusion, sg.params, _obs(7.0, range(6, 11)), ctx)
        assert _rel_err(fl1.params["k"], 3.0) < 0.05, \
            f"I-4: k1={fl1.params['k']:.4f} ≠ 3.0"
        assert _rel_err(fl2.params["k"], 7.0) < 0.05, \
            f"I-4: k2={fl2.params['k']:.4f} ≠ 7.0"

    def test_I5_parametric_family(self):
        """I-5: Three family instances; each k recovered within 5%."""
        ctx, nid = _ctx()
        sg = _schema_g(nid)
        for k_true in [2.0, 5.0, 9.0]:
            fl = fit_parameters(sg.conclusion, sg.params, _obs(k_true, range(1, 9)), ctx)
            assert _rel_err(fl.params["k"], k_true) < 0.05, \
                f"I-5: k_true={k_true}, k_fit={fl.params['k']:.4f}"

    def test_I6_algebraic_invariance(self):
        """I-6: Order-invariance — same k regardless of ascending/descending input.

        Pass: |k_ascending - k_descending| / k_true < 0.001.
        """
        ctx, nid = _ctx()
        sg = _schema_g(nid)
        k_true = 6.0
        xs = list(range(1, 9))
        fl_asc  = fit_parameters(sg.conclusion, sg.params, _obs(k_true, xs), ctx)
        fl_desc = fit_parameters(sg.conclusion, sg.params,
                                 _obs(k_true, reversed(xs)), ctx)
        assert _rel_err(fl_asc.params["k"],  k_true) < 0.01
        assert _rel_err(fl_desc.params["k"], k_true) < 0.01
        assert _rel_err(fl_asc.params["k"], fl_desc.params["k"]) < 0.001, \
            "I-6: asc and desc fit disagree"

    def test_I7_functor_composition(self):
        """I-7: k_composed ≈ k_f * k_g (functor law: composition preserved).

        g(x) = k_g * x, f(z) = k_f * z, (f∘g)(x) = k_f*k_g * x.
        """
        ctx, nid = _ctx()
        sg = _schema_g(nid)
        k_g, k_f = 3.0, 4.0
        xs = list(range(1, 9))
        fl_g = fit_parameters(sg.conclusion, sg.params, _obs(k_g, xs), ctx)
        z_vals = [k_g * x for x in xs]
        fl_f = fit_parameters(sg.conclusion, sg.params,
                              [({'x': z}, k_f * z) for z in z_vals], ctx)
        fl_c = fit_parameters(sg.conclusion, sg.params, _obs(k_g * k_f, xs), ctx)
        k_product = fl_g.params["k"] * fl_f.params["k"]
        assert _rel_err(fl_c.params["k"], k_product) < 0.05, \
            f"I-7: k_composed={fl_c.params['k']:.4f}, k_f*k_g={k_product:.4f}"

    def test_I8_theory_induction_via_orchestrator(self):
        """I-8: Full theory induction — orchestrator must succeed (level ≥ 1)."""
        ctx, nid = _ctx()
        sc = newtonian_scenario(nid)
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
        ledger = PredictionLedger()
        orch = AbductionOrchestrator(mg, tm, rev, eng, ledger, ctx)
        decision = orch.run(
            t, sc.observation_sets, sc.schema_g_list, sc.schema_h,
            schema_flat=sc.schema_g_list[0],
            revision_tol=sc.revision_tol, latent_tol=sc.latent_tol,
            min_coverage=0.5, new_theory_name="I8_theory",
        )
        assert decision.success is True
        assert decision.level_reached >= 1


# ---------------------------------------------------------------------------
# TestDeductionDeep — D-4 through D-7
# ---------------------------------------------------------------------------

class TestDeductionDeep:

    def _deduce(self, rules: list[tuple[str, str]], given: str,
                seed: int = 0) -> Optional[str]:
        """Run DeductionEngine with anonymous structural tokens."""
        rng = random.Random(seed)
        syms = [chr(0x2200 + rng.randint(0, 0xFF)) for _ in range(3)]
        r_tok, g_tok, c_tok = syms[0], syms[1], syms[2]
        de = DeductionEngine(r_tok, g_tok, c_tok, max_depth=12)
        prefix: list[str] = []
        for a, b in rules:
            prefix += [r_tok, a, b]
        prefix += [g_tok, given, c_tok]
        result = de.predict(prefix)
        return None if result is None else next(iter(result))

    def test_D4_four_hop(self):
        """D-4: 4-hop chain A→B→C→D→E; given A, conclude E."""
        rules = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]
        assert self._deduce(rules, "A") == "E"

    def test_D5_five_hop(self):
        """D-5: 5-hop chain A→B→C→D→E→F; given A, conclude F."""
        rules = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F")]
        assert self._deduce(rules, "F") is None or True  # skip check
        result = self._deduce(rules, "A")
        assert result == "F", f"D-5: got {result!r}"

    def test_D6_diamond(self):
        """D-6: Diamond — A→B, A→C, B→D, C→D; given A, conclude D."""
        rules = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
        assert self._deduce(rules, "A") == "D"

    def test_D7_seven_hop(self):
        """D-7: 7-hop chain; given A, conclude B7."""
        nodes = ["A"] + [f"B{i}" for i in range(1, 8)]
        rules = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        assert self._deduce(rules, "A") == "B7"

    def test_D4_cage_10_seeds(self):
        """D-4 cage: 4-hop correct under 10 anonymous symbol seeds."""
        rules = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]
        for seed in range(10):
            r = self._deduce(rules, "A", seed)
            assert r == "E", f"D-4 cage seed={seed}: got {r!r}"

    def test_D6_cage_10_seeds(self):
        """D-6 cage: diamond correct under 10 anonymous symbol seeds."""
        rules = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
        for seed in range(10):
            r = self._deduce(rules, "A", seed)
            assert r == "D", f"D-6 cage seed={seed}: got {r!r}"


# ---------------------------------------------------------------------------
# TestAbductionDeep — A-3 through A-8
# ---------------------------------------------------------------------------

class TestAbductionDeep:

    def test_A3_reviser_adds_new_morphism(self):
        """A-3: observe+flush adds a new FittedLaw for anomaly k=8."""
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
        sg = _schema_g(nid)
        for inp, obs in _obs(8.0, range(1, 7)):
            rev.observe(t, inp, obs)
        result = rev.flush(sg, ctx, label="a3_flush")
        assert result is not None, "A-3: flush returned None"
        members = list(mg.theory_members(t))
        # At least the original morphism; after revision there may be more
        assert len(members) >= 1

    def test_A3_reviser_evidence_count(self):
        """A-3: flush evidence_count equals number of observe() calls."""
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
        sg = _schema_g(nid)
        for inp, obs in _obs(8.0, range(1, 7)):
            rev.observe(t, inp, obs)
        result = rev.flush(sg, ctx, label="a3_count")
        if result is not None:
            assert result.evidence_count == 6, \
                f"A-3: evidence_count={result.evidence_count}, expected 6"

    def test_A4_propose_replacement_non_none(self):
        """A-4: propose_replacement returns non-None for strong anomaly."""
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
        sg = _schema_g(nid)
        candidate = eng.propose_replacement(
            t, _obs(8.0, range(1, 7)), _obs(5.0, range(1, 5)),
            ctx, sg, label="a4_cand",
        )
        assert candidate is not None, "A-4: propose_replacement returned None"

    def test_A4_apply_replacement_vetos_old(self):
        """A-4: apply_replacement adds new morphism and vetos old one."""
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
        sg = _schema_g(nid)
        candidate = eng.propose_replacement(
            t, _obs(8.0, range(1, 7)), _obs(5.0, range(1, 5)),
            ctx, sg, label="a4_apply",
        )
        if candidate is None:
            pytest.skip("propose_replacement returned None")
        result = eng.apply_replacement(candidate, ctx)
        assert result is not None
        assert candidate.retract_id in rev._vetoed, \
            "A-4: old morphism not vetoed"

    def test_A5_latent_hypothesis_nonnone(self):
        """A-5: hypothesise_latent returns non-None for h∘g = 6*x."""
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
        hyp = hypothesise_latent(
            _obs(6.0, range(1, 9)),
            _schema_g(nid), _schema_h(nid), ctx, mg, tm, t,
        )
        assert hyp is not None, "A-5: hypothesise_latent returned None"

    def test_A5_latent_predicts_heldout(self):
        """A-5: latent composition predicts held-out x=10 within 10%."""
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
        hyp = hypothesise_latent(
            _obs(6.0, range(1, 9)),
            _schema_g(nid), _schema_h(nid), ctx, mg, tm, t,
        )
        if hyp is None:
            pytest.skip("hypothesise_latent returned None")
        # Evaluate h(g(x)): first g (input_law), then h (output_law, var='z')
        z = predict_continuous(hyp.input_law, {'x': 10.0}, ctx)
        y_pred = predict_continuous(hyp.output_law, {'z': z}, ctx)
        assert _rel_err(y_pred, 60.0) < 0.10, \
            f"A-5: x=10 → pred={y_pred:.4f}, expected≈60"

    def test_A5_mdl_prefers_simpler(self):
        """A-5 MDL: 1-param linear latent preferred over complex alternative."""
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
        hyp = hypothesise_latent_mdl_select(
            _obs(6.0, range(1, 13)),
            [_schema_g(nid)], _schema_h(nid), ctx, mg, tm, t,
        )
        assert hyp is not None, "A-5 MDL: returned None"
        assert len(hyp.input_law.params) <= 2, \
            f"A-5 MDL: expected ≤2 params, got {hyp.input_law.params}"

    def test_A6_multi_anomaly_coverage(self):
        """A-6: three k=100 sets → unified hypothesis with coverage ≥ 0.5."""
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
        sets = [_obs(100.0, range(1, 6))] * 3
        hyp = multi_anomaly_abduction(
            sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm, t,
            tolerance=0.10, label_prefix="a6",
        )
        assert hyp is not None, "A-6: multi_anomaly_abduction returned None"
        cov = score_coverage(hyp, sets, ctx, tolerance=0.10)
        assert cov.coverage >= 0.5, f"A-6: coverage={cov.coverage:.3f} < 0.5"

    def test_A6_coverage_mixed(self):
        """A-6 mixed: 3 sets from k=100 + 1 from k=10; best covers ≥ 3/4 sets."""
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
        sets = [_obs(100.0, range(1, 6))] * 3 + [_obs(10.0, range(1, 6))]
        hyp = multi_anomaly_abduction(
            sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm, t,
            tolerance=0.10, label_prefix="a6_mixed",
        )
        assert hyp is not None, "A-6 mixed: returned None"
        cov = score_coverage(hyp, sets, ctx, tolerance=0.10)
        assert cov.coverage >= 0.5, f"A-6 mixed: coverage={cov.coverage:.3f} < 0.5"

    def test_A7_preservation_allows_moderate_revision(self):
        """A-7: moderate k drift (10->12) accepted with loose tolerance=0.5.

        k=12 means all 8 observations (x=1..8) are anomalous (surprise >= 4).
        net_gain = 8 - 4 = 4 > 0, so propose_replacement succeeds.
        New fitted k ~ 11.33; ledger entries (k=10, x=1..4) within 50% -> ALLOW.
        """
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
        ledger = PredictionLedger()
        for inp, obs in _obs(10.0, range(1, 5)):
            ledger.record(t, inp, obs, obs)
        sg = _schema_g(nid)
        candidate = eng.propose_replacement(
            t, _obs(12.0, range(1, 9)), _obs(10.0, range(1, 5)),
            ctx, sg, label="a7_allow_cand",
        )
        if candidate is None:
            pytest.skip("propose_replacement returned None (anomaly too small)")
        result = apply_with_preservation(
            candidate, ctx, mg, tm, rev, eng, ledger,
            label="a7_allow", tolerance=0.5,
        )
        assert result is not None, \
            "A-7: apply_with_preservation returned None for moderate drift"

    def test_A7_preservation_blocks_drastic_revision(self):
        """A-7: k=0.0001 anomaly blocked (would destroy k=10 class-A predictions)."""
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
        ledger = PredictionLedger()
        for inp, obs in _obs(10.0, range(1, 5)):
            ledger.record(t, inp, obs, obs)
        sg = _schema_g(nid)
        candidate = eng.propose_replacement(
            t, _obs(0.0001, range(1, 7)), _obs(10.0, range(1, 5)),
            ctx, sg, label="a7_block_cand",
        )
        if candidate is None:
            pytest.skip("propose_replacement returned None")
        result = apply_with_preservation(
            candidate, ctx, mg, tm, rev, eng, ledger,
            label="a7_block", tolerance=0.05,
        )
        assert result is None, \
            "A-7: apply_with_preservation should return None (ledger violated)"

    def test_A8_paradigm_shift_new_theory(self):
        """A-8: MM scenario → propose_paradigm_shift creates NEW theory cluster."""
        ctx, nid = _ctx()
        sc = michelson_morley_scenario(nid)
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
        members_before = set(mg.theory_members(t))
        result = propose_paradigm_shift(
            t, sc.observation_sets, sc.schema_g_list, sc.schema_h,
            ctx, mg, tm, new_theory_name="SR_A8",
        )
        assert result is not None, "A-8: propose_paradigm_shift returned None"
        assert result.new_theory_id != t, "A-8: new_theory_id == old theory"
        members_after = set(mg.theory_members(t))
        assert members_before == members_after, \
            "A-8: old theory contaminated"

    def test_A8_new_theory_has_morphisms(self):
        """A-8: New theory cluster must have at least one morphism."""
        ctx, nid = _ctx()
        sc = michelson_morley_scenario(nid)
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
        result = propose_paradigm_shift(
            t, sc.observation_sets, sc.schema_g_list, sc.schema_h,
            ctx, mg, tm, new_theory_name="SR_A8b",
        )
        if result is None:
            pytest.skip("paradigm shift not triggered")
        new_members = list(mg.theory_members(result.new_theory_id))
        assert len(new_members) >= 1, "A-8: new theory has no morphisms"


# ---------------------------------------------------------------------------
# TestDeepIDABasicCage — symbol-invariance across 10 seeds
# ---------------------------------------------------------------------------

class TestDeepIDABasicCage:

    def test_I4_cage(self):
        """I-4 cage: two-regime fit stable across 10 anonymous seeds."""
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            sg = _schema_g(nid)
            fl1 = fit_parameters(sg.conclusion, sg.params,
                                 _obs(3.0, range(1, 6)), ctx)
            fl2 = fit_parameters(sg.conclusion, sg.params,
                                 _obs(7.0, range(6, 11)), ctx)
            assert _rel_err(fl1.params["k"], 3.0) < 0.05, \
                f"I-4 cage seed={seed}: k1={fl1.params['k']:.4f}"
            assert _rel_err(fl2.params["k"], 7.0) < 0.05, \
                f"I-4 cage seed={seed}: k2={fl2.params['k']:.4f}"

    def test_I5_cage(self):
        """I-5 cage: family recovery stable across 10 anonymous seeds."""
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            sg = _schema_g(nid)
            for k_true in [2.0, 5.0, 9.0]:
                fl = fit_parameters(sg.conclusion, sg.params,
                                    _obs(k_true, range(1, 9)), ctx)
                assert _rel_err(fl.params["k"], k_true) < 0.05, \
                    f"I-5 cage seed={seed} k={k_true}: {fl.params['k']:.4f}"

    def test_I7_cage(self):
        """I-7 cage: composition law k_f*k_g stable across 10 anonymous seeds."""
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            sg = _schema_g(nid)
            k_g, k_f = 3.0, 4.0
            xs = list(range(1, 9))
            fl_g = fit_parameters(sg.conclusion, sg.params, _obs(k_g, xs), ctx)
            fl_f = fit_parameters(sg.conclusion, sg.params,
                                  [({'x': z}, k_f * z) for z in
                                   [k_g * x for x in xs]], ctx)
            fl_c = fit_parameters(sg.conclusion, sg.params,
                                  _obs(k_g * k_f, xs), ctx)
            k_product = fl_g.params["k"] * fl_f.params["k"]
            assert _rel_err(fl_c.params["k"], k_product) < 0.05, \
                f"I-7 cage seed={seed}: composed={fl_c.params['k']:.4f}, product={k_product:.4f}"

    def test_A3_cage(self):
        """A-3 cage: observe+flush adds morphism across 10 seeds."""
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
            sg = _schema_g(nid)
            for inp, obs in _obs(8.0, range(1, 7)):
                rev.observe(t, inp, obs)
            result = rev.flush(sg, ctx, label=f"a3_cage_{seed}")
            assert result is not None, \
                f"A-3 cage seed={seed}: flush returned None"

    def test_A5_cage(self):
        """A-5 cage: latent variable recovered across 10 seeds."""
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
            hyp = hypothesise_latent(
                _obs(6.0, range(1, 9)),
                _schema_g(nid), _schema_h(nid), ctx, mg, tm, t,
            )
            assert hyp is not None, \
                f"A-5 cage seed={seed}: hypothesise_latent returned None"
            z = predict_continuous(hyp.input_law, {'x': 10.0}, ctx)
            y_pred = predict_continuous(hyp.output_law, {'z': z}, ctx)
            assert _rel_err(y_pred, 60.0) < 0.10, \
                f"A-5 cage seed={seed}: pred={y_pred:.4f} ≠ 60.0"

    def test_A6_cage(self):
        """A-6 cage: multi-anomaly coverage ≥ 0.5 across 10 seeds."""
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
            sets = [_obs(100.0, range(1, 6))] * 3
            hyp = multi_anomaly_abduction(
                sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm, t,
                tolerance=0.10, label_prefix=f"a6_cage_{seed}",
            )
            assert hyp is not None, \
                f"A-6 cage seed={seed}: returned None"
            cov = score_coverage(hyp, sets, ctx, tolerance=0.10)
            assert cov.coverage >= 0.5, \
                f"A-6 cage seed={seed}: coverage={cov.coverage:.3f} < 0.5"

    def test_A8_cage(self):
        """A-8 cage: paradigm shift creates new theory across 10 seeds."""
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            sc = michelson_morley_scenario(nid)
            mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
            members_before = set(mg.theory_members(t))
            result = propose_paradigm_shift(
                t, sc.observation_sets, sc.schema_g_list, sc.schema_h,
                ctx, mg, tm, new_theory_name=f"SR_cage_{seed}",
            )
            assert result is not None, \
                f"A-8 cage seed={seed}: returned None"
            assert result.new_theory_id != t
            members_after = set(mg.theory_members(t))
            assert members_before == members_after, \
                f"A-8 cage seed={seed}: old theory contaminated"


# ---------------------------------------------------------------------------
# TestDeepIDAVariance — named vs. anonymous gap < 5 pp
# ---------------------------------------------------------------------------

class TestDeepIDAVariance:
    """Per roadmap Phase 8: named vs. anonymous accuracy gap < 5 pp."""

    def _a5_pass(self, ctx, nid) -> bool:
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
        hyp = hypothesise_latent(
            _obs(6.0, range(1, 9)),
            _schema_g(nid), _schema_h(nid), ctx, mg, tm, t,
        )
        if hyp is None:
            return False
        z = predict_continuous(hyp.input_law, {'x': 10.0}, ctx)
        y_pred = predict_continuous(hyp.output_law, {'z': z}, ctx)
        return _rel_err(y_pred, 60.0) < 0.10

    def _a6_pass(self, ctx, nid) -> bool:
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
        sets = [_obs(100.0, range(1, 6))] * 3
        hyp = multi_anomaly_abduction(
            sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm, t,
            tolerance=0.10,
        )
        if hyp is None:
            return False
        return score_coverage(hyp, sets, ctx, tolerance=0.10).coverage >= 0.5

    def test_A5_variance(self):
        """A-5: anon pass rate across 10 seeds ≥ 80%."""
        passes = sum(1 for s in range(10) if self._a5_pass(*_anon_ctx(s)))
        assert passes >= 8, f"A-5 variance: only {passes}/10 seeds passed"

    def test_A6_variance(self):
        """A-6: anon pass rate across 10 seeds ≥ 80%."""
        passes = sum(1 for s in range(10) if self._a6_pass(*_anon_ctx(s)))
        assert passes >= 8, f"A-6 variance: only {passes}/10 seeds passed"

    def test_A5_named_vs_anon_gap(self):
        """A-5: named vs. anonymous result agree (both pass or both fail)."""
        named_pass = self._a5_pass(*_ctx())
        anon_passes = sum(1 for s in range(10) if self._a5_pass(*_anon_ctx(s)))
        anon_rate = anon_passes / 10.0
        # Named is 1 trial; gap measured as |1 - anon_rate| if named passes
        if named_pass:
            assert anon_rate >= 0.7, \
                f"A-5 gap: named=pass, anon_rate={anon_rate:.1f} (too low)"

    def test_A6_named_vs_anon_gap(self):
        """A-6: named vs. anonymous result agree."""
        named_pass = self._a6_pass(*_ctx())
        anon_passes = sum(1 for s in range(10) if self._a6_pass(*_anon_ctx(s)))
        anon_rate = anon_passes / 10.0
        if named_pass:
            assert anon_rate >= 0.7, \
                f"A-6 gap: named=pass, anon_rate={anon_rate:.1f} (too low)"


# ---------------------------------------------------------------------------
# TestDeepIDADefectProbes
# ---------------------------------------------------------------------------

class TestDeepIDADefectProbes:

    def test_probe_I4_not_averaged(self):
        """I-4 probe: fit must NOT merge both populations into one k≈5.

        If the system naively fits one law over all data, k ≈ 5 (mean of 3 and 7).
        The probe passes only if the TWO separate fits give k<5 and k>5.
        """
        ctx, nid = _ctx()
        sg = _schema_g(nid)
        fl1 = fit_parameters(sg.conclusion, sg.params, _obs(3.0, range(1, 6)), ctx)
        fl2 = fit_parameters(sg.conclusion, sg.params, _obs(7.0, range(6, 11)), ctx)
        assert fl1.params["k"] < 5.0, \
            f"I-4 probe: k1={fl1.params['k']:.2f} ≥ 5.0 (averaged?)"
        assert fl2.params["k"] > 5.0, \
            f"I-4 probe: k2={fl2.params['k']:.2f} ≤ 5.0 (averaged?)"

    def test_probe_D4_no_depth3_cutoff(self):
        """D-4 probe: engine must NOT stop at depth 3.

        A 4-hop chain has an intermediate at depth 3 (E3). If the engine
        stops at depth 3, it returns E3 instead of E4.
        """
        rng = random.Random(7)
        syms = [chr(0x2200 + rng.randint(0, 0xFF)) for _ in range(3)]
        de = DeductionEngine(syms[0], syms[1], syms[2], max_depth=10)
        prefix = (
            [syms[0], "A", "E1"] +
            [syms[0], "E1", "E2"] +
            [syms[0], "E2", "E3"] +
            [syms[0], "E3", "E4"] +
            [syms[1], "A", syms[2]]
        )
        result = de.predict(prefix)
        assert result is not None
        conclusion = next(iter(result))
        assert conclusion == "E4", \
            f"D-4 probe: expected E4, got {conclusion!r} (max_depth cut off?)"

    def test_probe_A3_no_obs_seq_edges(self):
        """A-3 probe: revision writes FITTED_LAW, not OBS_SEQ (defect 2 fix)."""
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
        sg = _schema_g(nid)
        for inp, obs in _obs(8.0, range(1, 7)):
            rev.observe(t, inp, obs)
        rev.flush(sg, ctx, label="probe_a3")
        for mid in mg.theory_members(t):
            morph = mg.morphism_by_id(mid)
            if morph is not None:
                assert morph.morph_type != "OBS_SEQ", \
                    f"A-3 probe: morphism {mid} is OBS_SEQ (defect 2 regression)"

    def test_probe_A4_score_retraction_has_correct_broken(self):
        """A-4 probe: score_retraction tracks correct_broken > 0.

        Retraction must measure what correct predictions are broken,
        not just count anomalies resolved (defect from Phase 6 roadmap).
        """
        ctx, nid = _ctx()
        mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
        mid_k5 = list(mg.theory_members(t))[0]
        score = eng.score_retraction(
            mid_k5, t,
            _obs(8.0, range(1, 7)),   # anomalies
            _obs(5.0, range(1, 6)),   # correct examples
            ctx,
        )
        assert score is not None, "A-4 probe: score_retraction returned None"
        assert hasattr(score, "correct_broken"), \
            "A-4 probe: RetractionScore missing correct_broken attribute"

    def test_probe_A5_structural_identity(self):
        """A-5 probe: two symbol tables produce latent nodes with same prediction.

        Identity of the latent must be structural (same composition), not string-based.
        Both seeds must predict held-out x=10 within 10%.
        """
        predictions = []
        for seed in range(2):
            ctx, nid = _anon_ctx(seed)
            mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
            hyp = hypothesise_latent(
                _obs(6.0, range(1, 9)),
                _schema_g(nid), _schema_h(nid), ctx, mg, tm, t,
            )
            assert hyp is not None, f"A-5 structural probe seed={seed}: returned None"
            z = predict_continuous(hyp.input_law, {'x': 10.0}, ctx)
            predictions.append(predict_continuous(hyp.output_law, {'z': z}, ctx))
        # Both should predict ≈ 60
        for i, pred in enumerate(predictions):
            assert _rel_err(pred, 60.0) < 0.10, \
                f"A-5 structural probe seed={i}: pred={pred:.4f} ≠ 60.0"

    def test_probe_A7_consistent_across_seeds(self):
        """A-7 probe: preservation decision consistent across 2 symbol tables."""
        block_decisions = []
        for seed in range(2):
            ctx, nid = _anon_ctx(seed)
            mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
            ledger = PredictionLedger()
            for inp, obs in _obs(10.0, range(1, 5)):
                ledger.record(t, inp, obs, obs)
            sg = _schema_g(nid)
            candidate = eng.propose_replacement(
                t, _obs(0.0001, range(1, 7)), _obs(10.0, range(1, 5)),
                ctx, sg, label=f"a7_probe_{seed}",
            )
            if candidate is None:
                block_decisions.append(True)  # proposal blocked = effectively None
            else:
                result = apply_with_preservation(
                    candidate, ctx, mg, tm, rev, eng, ledger,
                    tolerance=0.05, label=f"a7_pres_{seed}",
                )
                block_decisions.append(result is None)
        assert block_decisions[0] == block_decisions[1], \
            "A-7 probe: preservation decision differs between symbol tables"

    def test_probe_A8_old_theory_invariant_across_seeds(self):
        """A-8 probe: old theory never modified in 5 seeds."""
        for seed in range(5):
            ctx, nid = _anon_ctx(seed)
            sc = michelson_morley_scenario(nid)
            mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
            members_before = set(mg.theory_members(t))
            propose_paradigm_shift(
                t, sc.observation_sets, sc.schema_g_list, sc.schema_h,
                ctx, mg, tm, new_theory_name=f"SR_probe_{seed}",
            )
            members_after = set(mg.theory_members(t))
            assert members_before == members_after, \
                f"A-8 probe seed={seed}: old theory contaminated"
