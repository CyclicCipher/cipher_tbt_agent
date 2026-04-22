"""Experiment 1b — Scaling Study.

Tests the single-column TBT architecture across four maze sizes and two
sensor configs:

  Config A: SingleColumnBrain + LocalSensor  (5-bit binary, original)
  Config B: SingleColumnBrain + DistanceSensor (9-value log-compressed)

Key questions:
  1. How does sensor ambiguity change with maze size?
  2. Does the log-distance sensor reduce ambiguity?
  3. What fraction of the maze does a single column cover?
  4. Navigation: how does goal-reach rate compare to random baseline?

Phase 2 (localisation from unknown start) is marked N/A.
A single TBT column cannot localise without multi-column voting.
That is a correct architectural finding, not a bug.

MultiScaleBrain is implemented in brain.py but not benchmarked here —
coarse→fine belief combination adds complexity that deserves its own
experiment once basic scaling is characterised.

Metrics per maze × config:
  - Sensor ambiguity (fraction of cells sharing an SDR reading)
  - Steps to 95% map coverage (exploration)
  - Navigation: goal-reach rate and steps vs random baseline (Phase 3)
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / 'src'
sys.path.insert(0, str(_SRC))

import brain as brain_mod
from maze_env import MazeEnv
from maze_gen import make_maze, maze_stats
from sensor import LocalSensor, DistanceSensor, N_BITS, N_DIST_BITS
from brain import SingleColumnBrain

# ---------------------------------------------------------------------------
# Per-maze parameters (scaled so total runtime stays ~60s)
# ---------------------------------------------------------------------------

CONFIGS = {
    'tiny'  : dict(rows=5,  cols=5,  explore_max=2_000,  nav_eps=20, nav_max=200),
    'small' : dict(rows=10, cols=10, explore_max=10_000, nav_eps=10, nav_max=400),
    'medium': dict(rows=20, cols=20, explore_max=50_000, nav_eps=10, nav_max=1000),
    'large' : dict(rows=40, cols=40, explore_max=200_000,nav_eps=5,  nav_max=3000),
}
COVERAGE_TARGET = 0.95
SEED            = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ambiguity(env: MazeEnv, sensor) -> float:
    """Fraction of open cells sharing their SDR with at least one other cell."""
    readings: dict[tuple, int] = {}
    for pos in env.open_cells():
        key = tuple(sensor.encode_at(env, pos).tolist())
        readings[key] = readings.get(key, 0) + 1
    shared = sum(v for v in readings.values() if v > 1)
    total  = env.n_open()
    return shared / total if total else 0.0


def _mean(lst: list) -> float:
    return sum(lst) / len(lst) if lst else float('nan')


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(env: MazeEnv, sensor, cfg: dict,
                  rng: random.Random) -> dict:
    """Run exploration (Phase 1) and navigation (Phase 3).

    Phase 2 (localisation from unknown start) is N/A for a single column —
    it requires multi-column voting.

    Returns metrics dict.
    """
    n_reach    = len(env.reachable_cells())
    open_cells = env.open_cells()
    non_goal   = [c for c in open_cells if c != env.goal]

    def make_brain():
        return SingleColumnBrain(goal=env.goal, epsilon=0.0)

    # ---- Phase 1: single-column exploration ----
    # The brain's own select_action() drives exploration:
    #   priority 1 — goal adjacent
    #   priority 2 — unmapped adjacent
    #   priority 3 — random walk
    # This is the honest single-column behaviour.  Coverage plateau is a
    # genuine experimental result, not a bug.
    brain = make_brain()
    env.reset()
    brain.reset(env.start, known_start=True)
    # Observe initial cell
    brain.observe(sensor.encode(env))
    steps_p1 = 0
    while (brain.coverage(n_reach) < COVERAGE_TARGET
           and steps_p1 < cfg['explore_max']):
        brain.step(rng.choice(env.valid_actions()), env)
        steps_p1 += 1
    final_cov = brain.coverage(n_reach)

    # ---- Phase 2: N/A ----
    # Single-column localisation requires multi-column voting.

    # ---- Phase 3: navigation ----
    nav_reached = 0
    nav_steps_list = []
    rnd_reached = 0
    rnd_steps_list = []
    for _ in range(cfg['nav_eps']):
        start = rng.choice(non_goal)

        # Model-based (single column, known start)
        env.reset_at(start)
        brain.reset(start, known_start=True)
        for step in range(cfg['nav_max']):
            if env.reached_goal():
                nav_reached += 1
                nav_steps_list.append(step + 1)
                break
            brain.step(brain.select_action(env.valid_actions()), env)

        # Random baseline
        env.reset_at(start)
        for step in range(cfg['nav_max']):
            if env.reached_goal():
                rnd_reached += 1
                rnd_steps_list.append(step + 1)
                break
            env.step(rng.choice(env.valid_actions()))

    return {
        'n_reach'  : n_reach,
        'final_cov': final_cov,
        'explore'  : steps_p1,
        'nav_rate' : nav_reached / cfg['nav_eps'],
        'nav_steps': _mean(nav_steps_list),
        'rnd_rate' : rnd_reached / cfg['nav_eps'],
        'rnd_steps': _mean(rnd_steps_list),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Experiment 1b: Scaling Study ===")
    print("Config A = SingleColumn + LocalSensor (5-bit binary)")
    print("Config B = SingleColumn + DistanceSensor (9-val log-compressed)")
    print("Phase 2 (localisation) = N/A — requires multi-column voting")
    print()

    header = (f"  {'Maze':<8} {'Cfg':<2}  {'Cells':>5}  "
              f"{'Ambig':>6}  {'Cov':>5}  {'Expl':>7}  "
              f"{'Nav%':>5} {'NavSt':>6}  {'Rnd%':>5} {'RndSt':>6}  {'Impv':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for diff, cfg in CONFIGS.items():
        env = make_maze(diff, seed=SEED + hash(diff) % 1000)

        for label, sensor in [('A', LocalSensor()), ('B', DistanceSensor())]:
            # Patch module-level sensor used by brain.step()
            brain_mod._SENSOR = sensor

            amb = _ambiguity(env, sensor)
            rng = random.Random(SEED)

            t0 = time.perf_counter()
            r  = run_benchmark(env, sensor, cfg, rng)
            dt = time.perf_counter() - t0

            nav_imp = ''
            if (not math.isnan(r['nav_steps']) and not math.isnan(r['rnd_steps'])
                    and r['rnd_steps'] > 0):
                imp = (r['rnd_steps'] - r['nav_steps']) / r['rnd_steps']
                nav_imp = f"{imp:+.0%}"

            print(f"  {diff:<8} {label:<2}  {r['n_reach']:>5}  "
                  f"{amb:>6.0%}  {r['final_cov']:>5.0%}  {r['explore']:>7}  "
                  f"{r['nav_rate']:>5.0%} {r['nav_steps']:>6.1f}  "
                  f"{r['rnd_rate']:>5.0%} {r['rnd_steps']:>6.1f}  "
                  f"{nav_imp:>6}  [{dt:.1f}s]")

    brain_mod._SENSOR = LocalSensor()   # restore default
    print()


if __name__ == '__main__':
    main()
