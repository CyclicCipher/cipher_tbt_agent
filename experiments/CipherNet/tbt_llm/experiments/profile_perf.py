"""Performance profiler for TBT column components.

Measures per-operation cost for each hot path across maze sizes.
Run this directly; does NOT run the full experiment.
"""
from __future__ import annotations

import sys
import time
import random
from pathlib import Path

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / 'src'
sys.path.insert(0, str(_SRC))

from maze_gen import generate_maze
from brain import SingleColumnBrain
from sensor import LocalSensor

SEED = 42


def time_n(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6   # microseconds per call


def profile_maze(rows, cols, label):
    rng   = random.Random(SEED)
    env   = generate_maze(rows, cols, seed=SEED)
    n     = len(env.reachable_cells())
    brain = SingleColumnBrain(goal=env.goal)

    print(f"\n{'='*60}")
    print(f"  {label}  ({rows}x{cols}, {n} reachable cells)")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Sensor encode
    # ------------------------------------------------------------------
    sensor = LocalSensor()
    env.reset()
    us = time_n(lambda: sensor.encode(env), 1000)
    print(f"  sensor.encode()                {us:8.2f} µs/call")

    # ------------------------------------------------------------------
    # 2. mini_column.learn_one  (single cell write)
    # ------------------------------------------------------------------
    import numpy as np
    sdr = sensor.encode(env)
    us = time_n(lambda: brain.mini_column.learn_one(sdr, (0, 0)), 1000)
    print(f"  mini_column.learn_one()        {us:8.2f} µs/call")

    # ------------------------------------------------------------------
    # 3. mini_column._model.get  (exact position lookup, hot path)
    # ------------------------------------------------------------------
    brain.mini_column.learn_one(sdr, (0, 0))
    us = time_n(lambda: brain.mini_column._model.get((0, 0)), 100_000)
    print(f"  mini_column._model.get()       {us:8.2f} µs/call  <-- O(1) hot path")

    # ------------------------------------------------------------------
    # 4. Build a realistic map (full exploration)
    # ------------------------------------------------------------------
    brain.reset(env.start, known_start=True)
    env.reset()
    steps = 0
    while brain.coverage(n) < 1.0 and steps < 200_000:
        valid  = env.valid_actions()
        action = brain.select_action(valid)
        brain.step(action, env)
        steps += 1
    cov      = brain.coverage(n)
    n_mapped = brain.n_mapped()
    print(f"  Exploration: {steps} steps -> {cov:.0%} coverage ({n_mapped} cells mapped)")

    # ------------------------------------------------------------------
    # 5. select_action (k=1 mapped case, all neighbours known)
    # ------------------------------------------------------------------
    brain.reset(env.start, known_start=True)
    env.reset()
    valid = env.valid_actions()
    us = time_n(lambda: brain.select_action(valid), 10_000)
    print(f"  select_action (mapped)         {us:8.2f} µs/call")

    # ------------------------------------------------------------------
    # 6. Full brain.step() (navigation, model trained)
    # ------------------------------------------------------------------
    brain.reset(env.start, known_start=True)
    env.reset()

    def _one_step():
        if env.reached_goal():
            env.reset()
            brain.reset(env.start, known_start=True)
        valid  = env.valid_actions()
        action = brain.select_action(valid)
        brain.step(action, env)

    us = time_n(_one_step, 500)
    print(f"  full brain.step() nav          {us:8.2f} µs/call")

    # ------------------------------------------------------------------
    # 7. Full brain.step() during exploration (unmapped neighbours present)
    # ------------------------------------------------------------------
    brain2 = SingleColumnBrain(goal=env.goal)
    brain2.reset(env.start, known_start=True)
    env.reset()

    def _one_step_exp():
        if brain2.coverage(n) >= 1.0:
            brain2.reset(env.start, known_start=True)
            env.reset()
        valid  = env.valid_actions()
        action = brain2.select_action(valid)
        brain2.step(action, env)

    us = time_n(_one_step_exp, 500)
    print(f"  full brain.step() explore      {us:8.2f} µs/call")


if __name__ == '__main__':
    for rows, cols, label in [
        (5,  5,  'tiny'),
        (10, 10, 'small'),
        (20, 20, 'medium'),
        (40, 40, 'large'),
    ]:
        profile_maze(rows, cols, label)
    print()
