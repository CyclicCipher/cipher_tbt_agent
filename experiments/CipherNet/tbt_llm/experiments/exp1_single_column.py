"""Experiment 1 — Single Column Maze Navigation.

Overview
--------
Tests whether a single TBT cortical column, equipped only with a 5-bit local
sensor and a motor efference copy (no GPS, no global map), can:

  Phase 1 — Exploration:  build a place map by random walk until coverage ≥ 95%.
  Phase 2 — Localisation: starting with unknown position (uniform belief), localise
             via sensorimotor integration without moving to goal.
  Phase 3 — Navigation:   from a known start position, use the model-based
             (information-gain) policy to navigate to the goal.

Metrics
-------
  Phase 1: steps to 95% map coverage; prediction error over time.
  Phase 2: steps to localisation (belief entropy < threshold); accuracy
            of best estimate vs true position.
  Phase 3: steps to goal; comparison against random-walk baseline.

Output
------
  Prints a per-phase summary table.
  Renders the maze with the agent's best-estimate overlay at each phase end.
  Writes metrics to stdout (no file I/O required for Experiment 1).

TBT Grounding
-------------
  PlaceMap  ≈ L4 place cells       (feature-at-location model)
  frame     ≈ L6 grid cells        (path integration via efference copies)
  belief    ≈ minicolumn evidence   (Bayesian hypothesis distribution)
  select_action ≈ L5 goal-state output (information-gain policy)

See tbt_llm/PLAN.md for the full theoretical background.
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

# Allow running from repo root or from this file's directory
_HERE = Path(__file__).parent
_SRC  = _HERE.parent / 'src'
sys.path.insert(0, str(_SRC))

from maze_env import MazeEnv, DEFAULT_START, DEFAULT_GOAL
from brain import SingleColumnBrain

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED                  = 42
N_PHASE1_MAX_STEPS    = 2_000   # cap on exploration steps
COVERAGE_TARGET       = 0.95    # stop exploring when this fraction is mapped
N_PHASE2_EPISODES     = 20      # how many random-start localisation tests
LOCALISE_MAX_STEPS    = 100     # max steps per localisation episode
LOCALISE_THRESHOLD    = 0.5     # entropy (nats) below which "localised"
N_PHASE3_EPISODES     = 20      # navigation trials
NAV_MAX_STEPS         = 200     # max steps per navigation episode
N_BASELINE_EPISODES   = 20      # random-walk baseline for comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _render_with_estimate(env: MazeEnv, estimate: Optional[tuple]) -> str:
    """Render the maze and mark the brain's position estimate with '?'."""
    overlay = {}
    if estimate is not None:
        overlay[estimate] = '?'
    return env.render(overlay)


def _steps_to_goal_random(env: MazeEnv, max_steps: int, rng: random.Random) -> Optional[int]:
    """Random walk from env.pos to goal; returns step count or None if timeout."""
    for step in range(max_steps):
        if env.reached_goal():
            return step
        action = rng.choice(env.valid_actions())
        env.step(action)
    return None


# ---------------------------------------------------------------------------
# Phase 1 — Exploration
# ---------------------------------------------------------------------------

def run_phase1(brain: SingleColumnBrain, env: MazeEnv,
               rng: random.Random) -> dict:
    """Curiosity-driven exploration using the free energy policy.

    The brain uses select_action() throughout — no separate random-walk phase.
    Curiosity (epistemic bonus for unseen cells) drives exploration naturally.
    The same policy that will later navigate to the goal is running here;
    it just has nothing to navigate to yet, so curiosity dominates.

    Returns
    -------
    dict with keys: steps, final_coverage, pred_errors, coverage_curve
    """
    env.reset()
    brain.reset(DEFAULT_START)

    pred_errors: list[float] = []
    coverage_curve: list[float] = []
    steps = 0

    while steps < N_PHASE1_MAX_STEPS:
        cov = brain.place_map.coverage()
        coverage_curve.append(cov)
        if cov >= COVERAGE_TARGET:
            break

        valid  = env.valid_actions()
        action = brain.select_action(valid)   # free energy policy: curiosity-driven
        sdr    = brain.step(action, env)

        pos = brain.frame.position_key()
        pe  = brain.place_map.prediction_error(sdr, pos)
        pred_errors.append(pe)
        steps += 1

    return {
        'steps'          : steps,
        'final_coverage' : brain.place_map.coverage(),
        'pred_errors'    : pred_errors,
        'coverage_curve' : coverage_curve,
    }


# ---------------------------------------------------------------------------
# Phase 2 — Localisation
# ---------------------------------------------------------------------------

def run_phase2(brain: SingleColumnBrain, env: MazeEnv,
               rng: random.Random) -> dict:
    """Test localisation from uniform prior over multiple random-start episodes.

    The brain starts each episode with uniform belief over all known positions.
    It then takes steps, updating belief via sensorimotor integration, until
    entropy < LOCALISE_THRESHOLD or step limit reached.

    Returns
    -------
    dict: localised_count, total_episodes, steps_list (per episode),
          accuracy_list (fraction of episodes where best_estimate == true_pos)
    """
    open_cells = env.open_cells()
    steps_list: list[Optional[int]] = []
    accurate: list[bool] = []

    for _ in range(N_PHASE2_EPISODES):
        # Random start position
        true_start = rng.choice(open_cells)
        env.reset_at(true_start)

        # Initialise brain at the true start position in the frame, but with
        # uniform belief — simulating "woke up in maze, don't know where I am."
        brain.frame.set_position((float(true_start[0]), float(true_start[1])))
        known = brain.place_map.known_positions()
        brain.belief = {p: 1.0 / len(known) for p in known} if known else {true_start: 1.0}

        # First observation at start (no movement)
        from sensor import LocalSensor
        _sensor = LocalSensor()
        sdr = _sensor.encode(env)
        brain.observe(sdr)

        localised_step: Optional[int] = None
        for step in range(LOCALISE_MAX_STEPS):
            if brain.belief_entropy() < LOCALISE_THRESHOLD:
                localised_step = step
                break
            valid = env.valid_actions()
            brain.step(rng.choice(valid), env)

        steps_list.append(localised_step)
        estimate = brain.best_estimate()
        # Compare frame position at localisation to true env position
        accurate.append(estimate == env.pos)

    localised_count = sum(1 for s in steps_list if s is not None)
    return {
        'localised_count' : localised_count,
        'total_episodes'  : N_PHASE2_EPISODES,
        'steps_list'      : steps_list,
        'accuracy_list'   : accurate,
        'mean_steps'      : (sum(s for s in steps_list if s is not None)
                             / max(localised_count, 1)),
        'accuracy'        : sum(accurate) / N_PHASE2_EPISODES,
    }


# ---------------------------------------------------------------------------
# Phase 3 — Navigation
# ---------------------------------------------------------------------------

def run_phase3(brain: SingleColumnBrain, env: MazeEnv,
               rng: random.Random) -> dict:
    """Model-based navigation from various start positions to goal.

    Uses information-gain (model-based) policy once map coverage >= min_coverage.
    Compares against a random-walk baseline.

    Returns
    -------
    dict: model_steps (list), baseline_steps (list), goal_reached (int),
          baseline_reached (int)
    """
    open_cells = [c for c in env.open_cells() if c != DEFAULT_GOAL]
    model_steps: list[Optional[int]] = []
    baseline_steps: list[Optional[int]] = []

    for _ in range(N_PHASE3_EPISODES):
        start = rng.choice(open_cells)

        # --- Model-based run ---
        env.reset_at(start)
        brain.reset(start, known_start=True)
        reached = False
        for step in range(NAV_MAX_STEPS):
            if env.reached_goal():
                reached = True
                break
            valid = env.valid_actions()
            action = brain.select_action(valid)
            brain.step(action, env)
        model_steps.append(step + 1 if reached else None)

        # --- Random-walk baseline ---
        env.reset_at(start)
        result = _steps_to_goal_random(env, NAV_MAX_STEPS, rng)
        baseline_steps.append(result)

    model_reached   = sum(1 for s in model_steps if s is not None)
    baseline_reached = sum(1 for s in baseline_steps if s is not None)

    def _mean(lst):
        vals = [v for v in lst if v is not None]
        return sum(vals) / len(vals) if vals else float('nan')

    return {
        'model_steps'      : model_steps,
        'baseline_steps'   : baseline_steps,
        'model_reached'    : model_reached,
        'baseline_reached' : baseline_reached,
        'total'            : N_PHASE3_EPISODES,
        'model_mean_steps' : _mean(model_steps),
        'baseline_mean'    : _mean(baseline_steps),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(SEED)
    env = MazeEnv()

    # Only count cells reachable from the default start — the maze has 3
    # isolated cells unreachable from (0,0) that can never be mapped.
    reachable = env.reachable_cells()
    n_reachable = len(reachable)

    print(f"\nDefault maze ({env.H}x{env.W})  --  {env.n_open()} open cells  "
          f"({n_reachable} reachable from start)")
    print(env.render())

    brain = SingleColumnBrain(
        n_cells              = n_reachable,   # coverage denominator = reachable only
        min_coverage         = 0.0,           # model-based policy always active
        confidence_threshold = LOCALISE_THRESHOLD,
        epsilon              = 0.0,
        goal                 = DEFAULT_GOAL,  # enable goal-directed navigation
    )

    # ------------------------------------------------------------------
    _print_section("PHASE 1 — Exploration (random walk, build map)")
    t0 = time.perf_counter()
    p1 = run_phase1(brain, env, rng)
    t1 = time.perf_counter()

    print(f"  Steps taken:       {p1['steps']}")
    print(f"  Final coverage:    {p1['final_coverage']:.1%}  "
          f"({len(brain.place_map._model)}/{n_reachable} reachable cells)")
    if p1['pred_errors']:
        first10 = p1['pred_errors'][:10]
        last10  = p1['pred_errors'][-10:]
        print(f"  Pred-error first10 avg:  {sum(first10)/len(first10):.3f}")
        print(f"  Pred-error last10  avg:  {sum(last10)/len(last10):.3f}")
    print(f"  Wall time:         {t1-t0:.2f}s")

    print()
    print("  Map at end of exploration (A=agent, G=goal):")
    env.reset()
    print(env.render())

    # ------------------------------------------------------------------
    _print_section("PHASE 2 — Localisation (uniform prior, find position)")
    t0 = time.perf_counter()
    p2 = run_phase2(brain, env, rng)
    t1 = time.perf_counter()

    print(f"  Episodes:          {p2['total_episodes']}")
    print(f"  Localised:         {p2['localised_count']}/{p2['total_episodes']}  "
          f"({p2['localised_count']/p2['total_episodes']:.0%})")
    print(f"  Mean steps to loc: {p2['mean_steps']:.1f}")
    print(f"  Estimate accuracy: {p2['accuracy']:.0%}  "
          f"(best_estimate == true_pos)")
    print(f"  Wall time:         {t1-t0:.2f}s")

    # ------------------------------------------------------------------
    # Switch from exploration to navigation mode: BFS fallback now plans
    # toward the goal instead of the nearest unmapped frontier cell.
    brain._explore_mode = False

    _print_section("PHASE 3 — Navigation (model-based vs random-walk)")
    t0 = time.perf_counter()
    p3 = run_phase3(brain, env, rng)
    t1 = time.perf_counter()

    print(f"  Episodes:           {p3['total']}")
    print(f"  Model reached goal: {p3['model_reached']}/{p3['total']}  "
          f"({p3['model_reached']/p3['total']:.0%})")
    print(f"  Random reached:     {p3['baseline_reached']}/{p3['total']}  "
          f"({p3['baseline_reached']/p3['total']:.0%})")
    if not math.isnan(p3['model_mean_steps']):
        print(f"  Model mean steps:   {p3['model_mean_steps']:.1f}")
    if not math.isnan(p3['baseline_mean']):
        print(f"  Random mean steps:  {p3['baseline_mean']:.1f}")
        if not math.isnan(p3['model_mean_steps']):
            improvement = (p3['baseline_mean'] - p3['model_mean_steps']) / p3['baseline_mean']
            print(f"  Improvement:        {improvement:+.1%}")
    print(f"  Wall time:          {t1-t0:.2f}s")

    # ------------------------------------------------------------------
    _print_section("SUMMARY")
    all_pass = (
        p1['final_coverage'] >= COVERAGE_TARGET
        and p2['localised_count'] / p2['total_episodes'] >= 0.75
        and p3['model_reached'] / p3['total'] >= p3['baseline_reached'] / p3['total']
    )
    print(f"  Coverage target ({COVERAGE_TARGET:.0%}):          "
          f"{'PASS' if p1['final_coverage'] >= COVERAGE_TARGET else 'FAIL'}")
    print(f"  Localisation rate >= 75%:       "
          f"{'PASS' if p2['localised_count']/p2['total_episodes'] >= 0.75 else 'FAIL'}")
    print(f"  Navigation >= random baseline:  "
          f"{'PASS' if p3['model_reached']/p3['total'] >= p3['baseline_reached']/p3['total'] else 'FAIL'}")
    print()
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES - see above'}")
    print()


if __name__ == '__main__':
    main()
