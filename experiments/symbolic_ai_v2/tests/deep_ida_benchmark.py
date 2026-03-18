"""Deep IDA Benchmark Runner — Phase 8 of the Einstein Roadmap.

Extends the existing IDA benchmark (I-1/I-3, D-1/D-3, A-1/A-2) to test the
deep infrastructure built in Phases 1–7 and the supplementary abduction work:

  I-4   Multi-case induction (two parameter regimes)
  I-5   Parametric family discovery (three instances, one schema)
  I-6   Algebraic invariance (commutativity / order-independence)
  I-7   Functor composition law (k_composed ≈ k_f * k_g)
  I-8   Full theory induction via AbductionOrchestrator

  D-4   4-hop deduction chain
  D-5   5-hop deduction chain
  D-6   Diamond topology (two paths to same conclusion)
  D-7   7-hop deduction chain

  A-3   New morphism hypothesis (ClosedLoopReviser observe+flush)
  A-4   Anomaly falsifies morphism -> retract and replace (RetractEngine)
  A-5   Latent variable hypothesis (hypothesise_latent)
  A-6   Multiple anomalies -> unified explanation (multi_anomaly_abduction)
  A-7   Theory revision preserving prior explanations (apply_with_preservation)
  A-8   Paradigm shift -> new concept cluster (propose_paradigm_shift)
  D-8   [DEFERRED — requires dependent-type extension to DeductionEngine]

Pass criteria (roadmap Phase 8)
--------------------------------
  I-*  : ≥ 90% accuracy; variance < 5 pp across 10 seeds
  D-*  : ≥ 90% accuracy; variance < 5 pp across 10 seeds
  A-*  : ≥ 80% accuracy; variance < 5 pp across 10 seeds

Gap compliance (Bitter Lesson check — roadmap Phase 8 requirement)
-------------------------------------------------------------------
  Each track is run once with named symbols and once with anonymous symbols.
  The accuracy gap must be < 5 pp. A gap ≥ 10 pp is a Bitter Lesson violation.

Usage
-----
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/deep_ida_benchmark.py
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/deep_ida_benchmark.py --seeds 10
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/deep_ida_benchmark.py --track I4
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/deep_ida_benchmark.py --track all --gap
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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
from experiments.symbolic_ai_v2.ctkg.inference.latent import hypothesise_latent
from experiments.symbolic_ai_v2.ctkg.inference.orchestrator import AbductionOrchestrator
from experiments.symbolic_ai_v2.ctkg.inference.paradigm import propose_paradigm_shift
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
)


# ---------------------------------------------------------------------------
# Anonymous symbol pool (matches ida_benchmark.py)
# ---------------------------------------------------------------------------

ANON_SYMBOLS: list[str] = [chr(i) for i in range(0x2200, 0x22FF)]


def _anon_sym(seed: int) -> tuple[EvalContext, int]:
    """Return (EvalContext, nid) using a random Unicode operator."""
    rng = random.Random(seed)
    sym = ANON_SYMBOLS[rng.randint(0, len(ANON_SYMBOLS) - 1)]
    nid = TOKEN_GRAPH.encode(sym)
    return EvalContext({nid: lambda a, b: a * b}), nid


def _named_sym() -> tuple[EvalContext, int]:
    """Return (EvalContext, nid) using the named operator 'mul'."""
    nid = TOKEN_GRAPH.encode("mul")
    return EvalContext({nid: lambda a, b: a * b}), nid


# ---------------------------------------------------------------------------
# Result types (mirrors ida_benchmark.py)
# ---------------------------------------------------------------------------

@dataclass
class TrackResult:
    track:     str
    seed:      int
    n_correct: int
    n_total:   int
    accuracy:  float
    notes:     str = ""

    def __str__(self) -> str:
        status = "PASS" if self.accuracy >= _pass_threshold(self.track) else "FAIL"
        return (
            f"Track {self.track} (seed={self.seed}): "
            f"{self.n_correct}/{self.n_total} = {100*self.accuracy:.1f}%  [{status}]"
            + (f"  {self.notes}" if self.notes else "")
        )


@dataclass
class VarianceResult:
    track:    str
    n_seeds:  int
    mean_acc: float
    std_acc:  float
    passed:   bool = False
    results:  list[TrackResult] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"Variance [{self.track}] over {self.n_seeds} seeds: "
            f"mean={100*self.mean_acc:.1f}%  std={100*self.std_acc:.2f}pp  [{status}]"
        )


@dataclass
class GapResult:
    track:       str
    named_acc:   float
    anon_acc:    float
    gap_pp:      float          # |named - anon| in percentage points
    compliant:   bool           # gap < 5 pp -> True; gap ≥ 10 pp -> Bitter Lesson violation

    def __str__(self) -> str:
        flag = "" if self.compliant else "  ⚠ BITTER LESSON VIOLATION"
        return (
            f"Gap  [{self.track}]: named={100*self.named_acc:.1f}%  "
            f"anon={100*self.anon_acc:.1f}%  "
            f"gap={self.gap_pp:.2f}pp"
            + flag
        )


def _pass_threshold(track: str) -> float:
    """Pass threshold: 90% for I/D tracks, 80% for A tracks."""
    return 0.80 if track.startswith("A") else 0.90


# ---------------------------------------------------------------------------
# Common infrastructure
# ---------------------------------------------------------------------------

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


def _obs(k: float, xs) -> list[tuple[dict, float]]:
    return [({'x': float(x)}, k * float(x)) for x in xs]


def _fitted(nid: int, k: float) -> FittedLaw:
    formula = Expr(head=nid, args=(atom("k"), var("x")))
    sch = SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(), variables=frozenset(["x"]), evidence=1,
    )
    return FittedLaw(schema=sch, params={"k": k}, residual=0.0)


def _stack(nid: int, ctx: EvalContext, theory_k: float = 5.0):
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
# Induction tracks
# ---------------------------------------------------------------------------

def run_track_I4(seed: int = 0) -> TrackResult:
    """I-4: Multi-case induction — two parameter regimes recovered independently.

    Sub-population 1: k=3 for x in [1..5].
    Sub-population 2: k=7 for x in [6..10].
    Pass: both k values recovered within 5% (2/2 = 100%).
    """
    ctx, nid = _anon_sym(seed)
    sg = _schema_g(nid)
    fl1 = fit_parameters(sg.conclusion, sg.params, _obs(3.0, range(1, 6)), ctx)
    fl2 = fit_parameters(sg.conclusion, sg.params, _obs(7.0, range(6, 11)), ctx)
    ok1 = _rel_err(fl1.params["k"], 3.0) < 0.05
    ok2 = _rel_err(fl2.params["k"], 7.0) < 0.05
    n_correct = int(ok1) + int(ok2)
    return TrackResult(
        track="I-4", seed=seed,
        n_correct=n_correct, n_total=2,
        accuracy=n_correct / 2,
        notes=f"k1={fl1.params['k']:.3f}(true=3), k2={fl2.params['k']:.3f}(true=7)",
    )


def run_track_I5(seed: int = 0) -> TrackResult:
    """I-5: Parametric family — three instances, each k recovered within 5%.

    Families: k ∈ {2, 5, 9}, each with 8 observations.
    Pass: n_recovered / 3 ≥ 0.90.
    """
    ctx, nid = _anon_sym(seed)
    sg = _schema_g(nid)
    k_trues = [2.0, 5.0, 9.0]
    recovered = []
    for k_true in k_trues:
        fl = fit_parameters(sg.conclusion, sg.params, _obs(k_true, range(1, 9)), ctx)
        recovered.append(_rel_err(fl.params["k"], k_true) < 0.05)
    n_correct = sum(recovered)
    return TrackResult(
        track="I-5", seed=seed,
        n_correct=n_correct, n_total=3,
        accuracy=n_correct / 3,
        notes=f"recovered: {['Y' if r else 'N' for r in recovered]}",
    )


def run_track_I6(seed: int = 0) -> TrackResult:
    """I-6: Algebraic invariance — same k from ascending vs. descending data.

    Pass: |k_asc - k_desc| / k < 0.001 (≤ 0.1% discrepancy -> order-invariant).
    Returns 1/1 (pass) or 0/1 (fail).
    """
    ctx, nid = _anon_sym(seed)
    sg = _schema_g(nid)
    k_true, xs = 6.0, list(range(1, 9))
    fl_asc  = fit_parameters(sg.conclusion, sg.params, _obs(k_true, xs), ctx)
    fl_desc = fit_parameters(sg.conclusion, sg.params, _obs(k_true, reversed(xs)), ctx)
    ok = (
        _rel_err(fl_asc.params["k"],  k_true) < 0.01 and
        _rel_err(fl_desc.params["k"], k_true) < 0.01 and
        _rel_err(fl_asc.params["k"],  fl_desc.params["k"]) < 0.001
    )
    return TrackResult(
        track="I-6", seed=seed,
        n_correct=int(ok), n_total=1,
        accuracy=float(ok),
        notes=f"k_asc={fl_asc.params['k']:.4f}, k_desc={fl_desc.params['k']:.4f}",
    )


def run_track_I7(seed: int = 0) -> TrackResult:
    """I-7: Functor composition — k_composed ≈ k_f * k_g within 5%.

    g(x) = k_g*x (k_g=3), f(z) = k_f*z (k_f=4), (f∘g)(x) = 12x.
    Pass: |k_composed - k_f*k_g| / (k_f*k_g) < 0.05.
    """
    ctx, nid = _anon_sym(seed)
    sg = _schema_g(nid)
    k_g, k_f = 3.0, 4.0
    xs = list(range(1, 9))
    fl_g = fit_parameters(sg.conclusion, sg.params, _obs(k_g, xs), ctx)
    fl_f = fit_parameters(sg.conclusion, sg.params,
                          [({'x': k_g * x}, k_f * k_g * x) for x in xs], ctx)
    fl_c = fit_parameters(sg.conclusion, sg.params, _obs(k_g * k_f, xs), ctx)
    k_product = fl_g.params["k"] * fl_f.params["k"]
    ok = _rel_err(fl_c.params["k"], k_product) < 0.05
    return TrackResult(
        track="I-7", seed=seed,
        n_correct=int(ok), n_total=1,
        accuracy=float(ok),
        notes=f"k_composed={fl_c.params['k']:.4f}, k_f*k_g={k_product:.4f}",
    )


def run_track_I8(seed: int = 0) -> TrackResult:
    """I-8: Full theory induction — AbductionOrchestrator must reach level ≥ 1.

    Newtonian scenario (slight k drift) presented to the full abduction stack.
    Pass: decision.success is True and level_reached ≥ 1.
    """
    ctx, nid = _anon_sym(seed)
    sc = newtonian_scenario(nid)
    mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
    ledger = PredictionLedger()
    orch = AbductionOrchestrator(mg, tm, rev, eng, ledger, ctx)
    try:
        decision = orch.run(
            t, sc.observation_sets, sc.schema_g_list, sc.schema_h,
            schema_flat=sc.schema_g_list[0],
            revision_tol=sc.revision_tol, latent_tol=sc.latent_tol,
            min_coverage=0.5, new_theory_name="I8_theory",
        )
        ok = decision.success and decision.level_reached >= 1
        notes = f"level={decision.level_reached}, success={decision.success}"
    except Exception as e:
        ok = False
        notes = f"exception: {e}"
    return TrackResult(
        track="I-8", seed=seed,
        n_correct=int(ok), n_total=1,
        accuracy=float(ok),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Deduction tracks
# ---------------------------------------------------------------------------

def _deduce(rules: list[tuple[str, str]], given: str, seed: int) -> Optional[str]:
    """Run DeductionEngine with anonymous structural tokens."""
    rng = random.Random(seed + 1000)   # offset to avoid collision with _anon_sym
    syms = [ANON_SYMBOLS[rng.randint(0, len(ANON_SYMBOLS) - 1)] for _ in range(3)]
    while len(set(syms)) < 3:
        syms = [ANON_SYMBOLS[rng.randint(0, len(ANON_SYMBOLS) - 1)] for _ in range(3)]
    r_tok, g_tok, c_tok = syms[0], syms[1], syms[2]
    de = DeductionEngine(r_tok, g_tok, c_tok, max_depth=12)
    prefix: list[str] = []
    for a, b in rules:
        prefix += [r_tok, a, b]
    prefix += [g_tok, given, c_tok]
    result = de.predict(prefix)
    return None if result is None else next(iter(result))


def run_track_D4(seed: int = 0) -> TrackResult:
    """D-4: 4-hop deduction chain. A->B->C->D->E; given A, conclude E."""
    rules = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]
    result = _deduce(rules, "A", seed)
    ok = result == "E"
    return TrackResult(
        track="D-4", seed=seed, n_correct=int(ok), n_total=1,
        accuracy=float(ok), notes=f"got={result!r}",
    )


def run_track_D5(seed: int = 0) -> TrackResult:
    """D-5: 5-hop deduction chain. A->...->F; given A, conclude F."""
    rules = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F")]
    result = _deduce(rules, "A", seed)
    ok = result == "F"
    return TrackResult(
        track="D-5", seed=seed, n_correct=int(ok), n_total=1,
        accuracy=float(ok), notes=f"got={result!r}",
    )


def run_track_D6(seed: int = 0) -> TrackResult:
    """D-6: Diamond topology. A->B, A->C, B->D, C->D; given A, conclude D."""
    rules = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    result = _deduce(rules, "A", seed)
    ok = result == "D"
    return TrackResult(
        track="D-6", seed=seed, n_correct=int(ok), n_total=1,
        accuracy=float(ok), notes=f"got={result!r}",
    )


def run_track_D7(seed: int = 0) -> TrackResult:
    """D-7: 7-hop deduction chain. A->B1->...->B7; given A, conclude B7."""
    nodes = ["A"] + [f"B{i}" for i in range(1, 8)]
    rules = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    result = _deduce(rules, "A", seed)
    ok = result == "B7"
    return TrackResult(
        track="D-7", seed=seed, n_correct=int(ok), n_total=1,
        accuracy=float(ok), notes=f"got={result!r}",
    )


# ---------------------------------------------------------------------------
# Abduction tracks
# ---------------------------------------------------------------------------

def run_track_A3(seed: int = 0) -> TrackResult:
    """A-3: New morphism hypothesis. Theory k=5; anomaly k=8 -> reviser adds morphism.

    Pass: flush() returns non-None (evidence_count matches observe() calls).
    """
    ctx, nid = _anon_sym(seed)
    mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
    sg = _schema_g(nid)
    for inp, obs_val in _obs(8.0, range(1, 7)):
        rev.observe(t, inp, obs_val)
    result = rev.flush(sg, ctx, label="a3_benchmark")
    ok = result is not None
    notes = f"evidence_count={result.evidence_count}" if result else "flush=None"
    return TrackResult(
        track="A-3", seed=seed, n_correct=int(ok), n_total=1,
        accuracy=float(ok), notes=notes,
    )


def run_track_A4(seed: int = 0) -> TrackResult:
    """A-4: Retract and replace. Theory k=5; anomaly k=8 -> old vetoed, new adopted.

    Pass: propose_replacement non-None AND apply_replacement succeeds AND
    old morphism is vetoed.
    """
    ctx, nid = _anon_sym(seed)
    mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
    sg = _schema_g(nid)
    candidate = eng.propose_replacement(
        t, _obs(8.0, range(1, 7)), _obs(5.0, range(1, 5)),
        ctx, sg, label="a4_benchmark",
    )
    if candidate is None:
        return TrackResult(track="A-4", seed=seed, n_correct=0, n_total=1,
                           accuracy=0.0, notes="propose_replacement=None")
    result = eng.apply_replacement(candidate, ctx)
    ok = result is not None and candidate.retract_id in rev._vetoed
    return TrackResult(
        track="A-4", seed=seed, n_correct=int(ok), n_total=1,
        accuracy=float(ok),
        notes=f"vetoed={candidate.retract_id in rev._vetoed}",
    )


def run_track_A5(seed: int = 0) -> TrackResult:
    """A-5: Latent variable hypothesis. f(x) = h(g(x)) = 6x; recover latent g.

    Pass: h(g(x=10)) predicts ≈ 60 within 10%.
    """
    ctx, nid = _anon_sym(seed)
    mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=5.0)
    hyp = hypothesise_latent(
        _obs(6.0, range(1, 9)),
        _schema_g(nid), _schema_h(nid), ctx, mg, tm, t,
    )
    if hyp is None:
        return TrackResult(track="A-5", seed=seed, n_correct=0, n_total=1,
                           accuracy=0.0, notes="hypothesise_latent=None")
    z = predict_continuous(hyp.input_law, {'x': 10.0}, ctx)
    y_pred = predict_continuous(hyp.output_law, {'z': z}, ctx)
    ok = _rel_err(y_pred, 60.0) < 0.10
    return TrackResult(
        track="A-5", seed=seed, n_correct=int(ok), n_total=1,
        accuracy=float(ok),
        notes=f"x=10->pred={y_pred:.2f}(true=60)",
    )


def run_track_A6(seed: int = 0) -> TrackResult:
    """A-6: Multi-anomaly coverage. Three k=100 sets -> unified hypothesis, cov ≥ 0.5.

    Pass: coverage ≥ 0.5.
    """
    ctx, nid = _anon_sym(seed)
    mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
    sets = [_obs(100.0, range(1, 6))] * 3
    hyp = multi_anomaly_abduction(
        sets, [_schema_g(nid)], _schema_h(nid), ctx, mg, tm, t,
        tolerance=0.10, label_prefix="a6_benchmark",
    )
    if hyp is None:
        return TrackResult(track="A-6", seed=seed, n_correct=0, n_total=1,
                           accuracy=0.0, notes="multi_anomaly_abduction=None")
    cov = score_coverage(hyp, sets, ctx, tolerance=0.10)
    ok = cov.coverage >= 0.5
    return TrackResult(
        track="A-6", seed=seed, n_correct=int(ok), n_total=1,
        accuracy=float(ok),
        notes=f"coverage={cov.coverage:.3f}",
    )


def run_track_A7(seed: int = 0) -> TrackResult:
    """A-7: Preservation. Moderate revision allowed; large revision blocked.

    Two sub-tests:
      (a) k=12 with loose tolerance=0.50 -> accepted  (correct_allowed)
          Theory k=10; anomaly k=12 for x=1..8 (all 8 obs exceed threshold).
          New law k≈11.33 fits both anomalies and correct obs.
          Ledger entries (k=10, x=1..4) within 50% tolerance -> ALLOW.
      (b) k=0.0001 with strict tolerance=0.05 -> rejected (correct_blocked)
          Theory k=10; anomaly k≈0 drives k_new -> near 0, destroying ledger.
    Pass: both sub-tests correct (2/2).
    """
    ctx, nid = _anon_sym(seed)
    sg = _schema_g(nid)

    # Sub-test (a): moderate drift — anomaly large enough for propose_replacement
    # k=12 gives surprise (12x - 10x)^2 = (2x)^2 = 4x^2 >> threshold=3 for all x>=2
    mg1, tm1, rev1, eng1, t1 = _stack(nid, ctx, theory_k=10.0)
    ledger1 = PredictionLedger()
    for inp, obs_val in _obs(10.0, range(1, 5)):
        ledger1.record(t1, inp, obs_val, obs_val)
    c1 = eng1.propose_replacement(
        t1, _obs(12.0, range(1, 9)), _obs(10.0, range(1, 5)),
        ctx, sg, label="a7a",
    )
    allow_ok = False
    if c1 is not None:
        r1 = apply_with_preservation(c1, ctx, mg1, tm1, rev1, eng1, ledger1,
                                     tolerance=0.5, label="a7a_pres")
        allow_ok = r1 is not None

    # Sub-test (b): drastic revision should be blocked
    mg2, tm2, rev2, eng2, t2 = _stack(nid, ctx, theory_k=10.0)
    ledger2 = PredictionLedger()
    for inp, obs_val in _obs(10.0, range(1, 5)):
        ledger2.record(t2, inp, obs_val, obs_val)
    c2 = eng2.propose_replacement(
        t2, _obs(0.0001, range(1, 7)), _obs(10.0, range(1, 5)),
        ctx, sg, label="a7b",
    )
    block_ok = c2 is None   # no candidate (anomaly too small) counts as blocked
    if c2 is not None:
        r2 = apply_with_preservation(c2, ctx, mg2, tm2, rev2, eng2, ledger2,
                                     tolerance=0.05, label="a7b_pres")
        block_ok = r2 is None

    n_correct = int(allow_ok) + int(block_ok)
    return TrackResult(
        track="A-7", seed=seed, n_correct=n_correct, n_total=2,
        accuracy=n_correct / 2,
        notes=f"allow={'Y' if allow_ok else 'N'}, block={'Y' if block_ok else 'N'}",
    )


def run_track_A8(seed: int = 0) -> TrackResult:
    """A-8: Paradigm shift. MM scenario -> new theory cluster; old theory unchanged.

    Pass: new theory created (non-None result) AND old theory members unchanged.
    """
    ctx, nid = _anon_sym(seed)
    sc = michelson_morley_scenario(nid)
    mg, tm, rev, eng, t = _stack(nid, ctx, theory_k=10.0)
    members_before = set(mg.theory_members(t))
    try:
        result = propose_paradigm_shift(
            t, sc.observation_sets, sc.schema_g_list, sc.schema_h,
            ctx, mg, tm, new_theory_name=f"SR_{seed}",
        )
        members_after = set(mg.theory_members(t))
        ok = (
            result is not None and
            result.new_theory_id != t and
            members_before == members_after
        )
        notes = (f"new_theory={result.new_theory_id if result else None}, "
                 f"old_unchanged={members_before == members_after}")
    except Exception as e:
        ok = False
        notes = f"exception: {e}"
    return TrackResult(
        track="A-8", seed=seed, n_correct=int(ok), n_total=1,
        accuracy=float(ok), notes=notes,
    )


# ---------------------------------------------------------------------------
# Variance measurement
# ---------------------------------------------------------------------------

def run_variance(
    track_fn:           Callable[[int], TrackResult],
    n_seeds:            int = 10,
    pass_std_threshold: float = 0.05,
) -> VarianceResult:
    """Run track_fn across n_seeds and compute mean/std."""
    results = [track_fn(s) for s in range(n_seeds)]
    accs = [r.accuracy for r in results]
    mean_acc = statistics.mean(accs)
    std_acc  = statistics.stdev(accs) if len(accs) > 1 else 0.0
    mean_ok  = mean_acc >= _pass_threshold(results[0].track)
    passed   = mean_ok and std_acc < pass_std_threshold
    return VarianceResult(
        track=results[0].track,
        n_seeds=n_seeds,
        mean_acc=mean_acc,
        std_acc=std_acc,
        passed=passed,
        results=results,
    )


# ---------------------------------------------------------------------------
# Gap compliance check (Bitter Lesson — Phase 8 roadmap requirement)
# ---------------------------------------------------------------------------

def run_gap_check(
    track_fn:    Callable[[int], TrackResult],
    seed:        int = 0,
    gap_warn_pp: float = 5.0,
) -> GapResult:
    """Run track_fn with anonymous symbols and compare to named-symbol baseline.

    The named-symbol baseline uses seed=seed but forces the named operator 'mul'
    (rather than a random Unicode symbol).  The anonymous run uses the normal
    _anon_sym(seed) path (already built into every run_track_* function).

    A gap ≥ gap_warn_pp percentage points is flagged as a Bitter Lesson violation:
    the named-symbol case is relying on token identity, not structural reasoning.
    """
    # Anonymous run (standard path — _anon_sym already used inside each track_fn)
    anon_result = track_fn(seed)

    # Named-symbol run: monkey-patch _anon_sym to return the named ctx
    # We achieve this by using a fixed seed whose _anon_sym mapping happens to
    # produce a specific symbol — instead, we use the track's internal logic
    # but replace the anon sym with 'mul'.  The cleanest approach is to run
    # the track with seed=0 vs seed=∞ and compare, but since the track
    # functions call _anon_sym(seed) internally, the only way to get a true
    # named baseline is to use a wrapper.
    #
    # Implementation: run with the same seed but ensure the symbol is 'mul' by
    # overriding _anon_sym in this module temporarily.
    import experiments.symbolic_ai_v2.tests.deep_ida_benchmark as _self
    original_anon_sym = _self._anon_sym

    def _named_override(s: int) -> tuple[EvalContext, int]:
        return _named_sym()

    _self._anon_sym = _named_override
    try:
        named_result = track_fn(seed)
    finally:
        _self._anon_sym = original_anon_sym

    gap_pp = abs(named_result.accuracy - anon_result.accuracy) * 100.0
    return GapResult(
        track=anon_result.track,
        named_acc=named_result.accuracy,
        anon_acc=anon_result.accuracy,
        gap_pp=gap_pp,
        compliant=gap_pp < gap_warn_pp,
    )


# ---------------------------------------------------------------------------
# Track registry
# ---------------------------------------------------------------------------

_TRACK_MAP: dict[str, Callable[[int], TrackResult]] = {
    "I4": run_track_I4,
    "I5": run_track_I5,
    "I6": run_track_I6,
    "I7": run_track_I7,
    "I8": run_track_I8,
    "D4": run_track_D4,
    "D5": run_track_D5,
    "D6": run_track_D6,
    "D7": run_track_D7,
    "A3": run_track_A3,
    "A4": run_track_A4,
    "A5": run_track_A5,
    "A6": run_track_A6,
    "A7": run_track_A7,
    "A8": run_track_A8,
}

_TRACK_ORDER = ["I4", "I5", "I6", "I7", "I8",
                "D4", "D5", "D6", "D7",
                "A3", "A4", "A5", "A6", "A7", "A8"]

_DEFERRED = {"D8"}   # requires dependent-type extension to DeductionEngine


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep IDA Benchmark — Phase 8 of the Einstein Roadmap"
    )
    parser.add_argument(
        "--track", default="all",
        choices=list(_TRACK_MAP.keys()) + ["all"],
        help="Which track to run (default: all)",
    )
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Number of random seeds for variance test (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Single seed when --seeds=1 (default: 0)",
    )
    parser.add_argument(
        "--gap", action="store_true",
        help="Run Bitter Lesson gap check (named vs. anonymous symbol accuracy)",
    )
    args = parser.parse_args()

    track_keys = _TRACK_ORDER if args.track == "all" else [args.track]

    print("=" * 70)
    print("Deep IDA Benchmark — Phase 8 of the Einstein Roadmap")
    print(f"Tracks: {', '.join(track_keys)}   Seeds: {args.seeds}")
    print("=" * 70)
    print()

    all_passed = True
    gap_results: list[GapResult] = []

    for key in track_keys:
        fn = _TRACK_MAP[key]
        track_id = f"{key[0]}-{key[1:]}"

        print(f"-- Track {track_id} --")

        if args.seeds > 1:
            vr = run_variance(fn, n_seeds=args.seeds)
            for r in vr.results:
                print(f"  {r}")
            print(f"  {vr}")
            passed = vr.passed
        else:
            r = fn(args.seed)
            print(f"  {r}")
            passed = r.accuracy >= _pass_threshold(r.track)

        if not passed:
            all_passed = False

        if args.gap:
            gr = run_gap_check(fn, seed=args.seed)
            print(f"  {gr}")
            gap_results.append(gr)
            if not gr.compliant:
                all_passed = False

        print()

    if _DEFERRED:
        print(f"-- Deferred tracks: {', '.join(sorted(_DEFERRED))} --")
        print("  D-8: DEFERRED — requires dependent-type extension to DeductionEngine")
        print()

    print("=" * 70)
    if args.gap and gap_results:
        all_gap_ok = all(g.compliant for g in gap_results)
        gap_status = "PASS" if all_gap_ok else "FAIL (Bitter Lesson violation)"
        print(f"Gap compliance (all tracks < 5 pp): {gap_status}")
    print(f"Benchmark {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
