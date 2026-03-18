"""Abduction Routing Benchmark — Structural scaffold for Phase 9 development.

THIS IS NOT THE EINSTEIN TEST.

Runs four SYNTHETIC LINEAR scenarios (f(x) = k*x) through the abduction
orchestrator to verify routing behaviour only.  These scenarios are scalar
magnitude proxies; they do not constitute discovery of general relativity.

Scenarios (routing proxies — not physics content)
-------------------------------------------------
  newtonian       : small-drift routing — slight k change, level 1 expected
  michelson_morley: preservation-block routing — k≈0 vs k=10, level >= 2 expected
  mercury         : minor-correction routing — k=10.3 vs k=10, level 1 expected
  maxwell_em      : multi-set coverage routing — 3 independent k=100 sets, level >= 2

Pass criteria (abduction routing verification only)
---------------------------------------------------
  Each scenario: success == True across all seeds
  michelson_morley: level_reached >= 2 on all seeds (escalates beyond revision)
  newtonian/mercury: level_reached <= 3 (not over-engineered to paradigm shift)
  maxwell_em: coverage >= 0.5 when level >= 2

Gap compliance (Bitter Lesson check)
-------------------------------------
  Each scenario run in named mode (op='mul') and anon mode (op=random Unicode).
  Gap = |named_success_rate - anon_success_rate| must be < 5 pp.

Usage
-----
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/einstein_benchmark.py
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/einstein_benchmark.py --seeds 10
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/einstein_benchmark.py --scenario michelson_morley
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/einstein_benchmark.py --gap
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from dataclasses import dataclass
from typing import Callable, Optional

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import FittedLaw, add_fitted_law
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom, var, Expr
from experiments.symbolic_ai_v2.ctkg.inference.coverage import score_coverage
from experiments.symbolic_ai_v2.ctkg.inference.orchestrator import (
    AbductionDecision,
    AbductionOrchestrator,
)
from experiments.symbolic_ai_v2.ctkg.inference.paradigm import propose_paradigm_shift
from experiments.symbolic_ai_v2.ctkg.inference.preservation import PredictionLedger
from experiments.symbolic_ai_v2.ctkg.inference.retract import RetractEngine
from experiments.symbolic_ai_v2.ctkg.inference.revision import ClosedLoopReviser
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager
from experiments.symbolic_ai_v2.ctkg.einstein.streams import (
    EinsteinScenario,
    all_scenarios,
    newtonian_scenario,
    michelson_morley_scenario,
    mercury_precession_scenario,
    maxwell_em_scenario,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario_name: str
    seed: int
    success: bool
    level_reached: int
    coverage: float          # coverage if applicable, else 1.0
    anon: bool               # True if run with anonymous symbols


@dataclass
class GapResult:
    scenario_name: str
    named_rate: float
    anon_rate: float
    gap_pp: float
    compliant: bool          # gap_pp < 5.0


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _ctx_named() -> tuple[EvalContext, int]:
    nid = TOKEN_GRAPH.encode("mul")
    return EvalContext({nid: lambda a, b: a * b}), nid


def _ctx_anon(seed: int) -> tuple[EvalContext, int]:
    rng = random.Random(seed)
    sym = chr(0x2200 + rng.randint(0, 0xFE))
    nid = TOKEN_GRAPH.encode(sym)
    return EvalContext({nid: lambda a, b: a * b}), nid


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def _fitted_law(nid: int, k: float) -> FittedLaw:
    formula = Expr(head=nid, args=(atom("k"), var("x")))
    sch = SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(), variables=frozenset(["x"]), evidence=1,
    )
    return FittedLaw(schema=sch, params={"k": k}, residual=0.0)


def run_scenario(
    scenario: EinsteinScenario,
    nid: int,
    ctx: EvalContext,
    theory_k: float = 10.0,
    seed: int = 0,
    anon: bool = False,
) -> ScenarioResult:
    """Run a single Einstein scenario and return a ScenarioResult."""
    mg = MorphismGraph()
    tm = TheoryManager(mg)
    rev = ClosedLoopReviser(tm, mg, threshold=3.0, sigma=1.0)
    eng = RetractEngine(rev, tm, mg)
    t = tm.register_theory("Newton")
    mid = add_fitted_law(mg, f"newton_k_{seed}", _fitted_law(nid, theory_k))
    tm.assign_morphism(mid, t)

    ledger = PredictionLedger()
    for inp, obs in scenario.ledger_examples:
        ledger.record(t, inp, obs, obs)

    orch = AbductionOrchestrator(mg, tm, rev, eng, ledger, ctx)
    label = f"es_{scenario.name}_{seed}_{'a' if anon else 'n'}"

    decision = orch.run(
        t,
        scenario.observation_sets,
        scenario.schema_g_list,
        scenario.schema_h,
        schema_flat=scenario.schema_g_list[0],
        revision_tol=scenario.revision_tol,
        latent_tol=scenario.latent_tol,
        min_coverage=0.5,
        new_theory_name=f"new_theory_{label}",
        label_prefix=label,
    )

    # Compute coverage if the decision has a coverage_hyp or latent_hyp
    coverage = 1.0
    if decision.coverage_hyp is not None:
        cov = score_coverage(
            decision.coverage_hyp, scenario.observation_sets, ctx,
            tolerance=scenario.latent_tol,
        )
        coverage = cov.coverage
    elif decision.latent_hyp is not None:
        cov = score_coverage(
            decision.latent_hyp, scenario.observation_sets, ctx,
            tolerance=scenario.latent_tol,
        )
        coverage = cov.coverage

    return ScenarioResult(
        scenario_name=scenario.name,
        seed=seed,
        success=decision.success,
        level_reached=decision.level_reached,
        coverage=coverage,
        anon=anon,
    )


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_scenario_seeds(
    factory: Callable[[int], EinsteinScenario],
    n_seeds: int,
    anon: bool,
    theory_k: float = 10.0,
) -> list[ScenarioResult]:
    results = []
    for seed in range(n_seeds):
        if anon:
            ctx, nid = _ctx_anon(seed)
        else:
            ctx, nid = _ctx_named()
        sc = factory(nid)
        results.append(run_scenario(sc, nid, ctx, theory_k=theory_k, seed=seed, anon=anon))
    return results


# ---------------------------------------------------------------------------
# Gap check
# ---------------------------------------------------------------------------

def run_gap_check(
    factory: Callable[[int], EinsteinScenario],
    n_seeds: int = 10,
    theory_k: float = 10.0,
    gap_warn_pp: float = 5.0,
) -> GapResult:
    named = run_scenario_seeds(factory, n_seeds, anon=False, theory_k=theory_k)
    anon  = run_scenario_seeds(factory, n_seeds, anon=True,  theory_k=theory_k)
    named_rate = sum(r.success for r in named) / n_seeds
    anon_rate  = sum(r.success for r in anon)  / n_seeds
    gap_pp = abs(named_rate - anon_rate) * 100.0
    sc_name = named[0].scenario_name if named else "unknown"
    return GapResult(
        scenario_name=sc_name,
        named_rate=named_rate,
        anon_rate=anon_rate,
        gap_pp=gap_pp,
        compliant=gap_pp < gap_warn_pp,
    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

_SCENARIO_MAP: dict[str, Callable[[int], EinsteinScenario]] = {
    "newtonian":        newtonian_scenario,
    "michelson_morley": michelson_morley_scenario,
    "mercury":          mercury_precession_scenario,
    "maxwell_em":       maxwell_em_scenario,
}

# Level constraints for pass-criteria check
_MIN_LEVEL: dict[str, int] = {
    "newtonian":        1,
    "michelson_morley": 2,
    "mercury":          1,
    "maxwell_em":       2,
}
_MAX_LEVEL: dict[str, int] = {
    "newtonian":        3,
    "michelson_morley": 4,
    "mercury":          3,
    "maxwell_em":       4,
}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _bar(width: int = 70) -> str:
    return "=" * width


def _check(ok: bool) -> str:
    return "Y" if ok else "N"


def _print_scenario_block(
    name: str,
    results: list[ScenarioResult],
    n_seeds: int,
) -> bool:
    success_rate = sum(r.success for r in results) / n_seeds
    levels = [r.level_reached for r in results]
    min_lev = _MIN_LEVEL[name]
    max_lev = _MAX_LEVEL[name]
    level_ok = all(min_lev <= l <= max_lev for l in levels)
    level_std = statistics.stdev(levels) if len(levels) > 1 else 0.0

    pass_ok = success_rate >= 1.0 and level_ok

    print(f"  {name:<20} success={success_rate*100:.1f}%  "
          f"levels={min(levels)}-{max(levels)}  std={level_std:.2f}  "
          f"[{'PASS' if pass_ok else 'FAIL'}]")
    return pass_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Einstein Benchmark — Phase 9")
    parser.add_argument("--scenario", default="all",
                        choices=list(_SCENARIO_MAP) + ["all"],
                        help="Which scenario to run (default: all)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of anonymous symbol seeds (default: 10)")
    parser.add_argument("--gap", action="store_true",
                        help="Also run gap compliance check (named vs. anonymous)")
    args = parser.parse_args()

    n_seeds = args.seeds
    do_gap = args.gap
    scenario_names = (list(_SCENARIO_MAP) if args.scenario == "all"
                      else [args.scenario])

    print(_bar())
    print("Abduction Routing Benchmark (NOT the Einstein test - synthetic scalar proxies only)")
    print(f"Scenarios: {', '.join(scenario_names)}  Seeds: {n_seeds}")
    print(_bar())

    all_pass = True
    all_results: dict[str, list[ScenarioResult]] = {}

    for name in scenario_names:
        factory = _SCENARIO_MAP[name]
        results = run_scenario_seeds(factory, n_seeds, anon=True)
        all_results[name] = results
        ok = _print_scenario_block(name, results, n_seeds)
        if not ok:
            all_pass = False

    print(_bar())

    # Overall stats
    all_successes = [r.success for rs in all_results.values() for r in rs]
    overall_rate = sum(all_successes) / len(all_successes) if all_successes else 0.0
    print(f"All scenarios: mean={overall_rate*100:.1f}%  "
          f"[{'PASS' if all_pass else 'FAIL'}]")

    # Gap check
    if do_gap:
        print()
        print("Gap compliance (named vs. anonymous, < 5 pp = PASS):")
        gap_ok = True
        for name in scenario_names:
            factory = _SCENARIO_MAP[name]
            gr = run_gap_check(factory, n_seeds)
            sym = "*" if not gr.compliant else " "
            print(f"  Gap [{sym}] {name:<20} named={gr.named_rate*100:.1f}%  "
                  f"anon={gr.anon_rate*100:.1f}%  gap={gr.gap_pp:.2f}pp  "
                  f"[{'PASS' if gr.compliant else 'FAIL'}]")
            if not gr.compliant:
                gap_ok = False

        print(_bar())
        print(f"Gap compliance (all scenarios < 5 pp): "
              f"{'PASS' if gap_ok else 'FAIL'}")
        all_pass = all_pass and gap_ok

    print(_bar())
    print(f"Benchmark {'PASSED' if all_pass else 'FAILED'}")
    print(_bar())

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
