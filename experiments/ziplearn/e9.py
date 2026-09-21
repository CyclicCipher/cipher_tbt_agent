"""E9 — planning in the written map (DESIGN.md §16, the outer objective's first test).

Environment: 6-digit sequences over V = 5; the seven primitives of the composition domain are the ACTIONS (reverse,
rotate left/right, swap pairs, swap halves, inc, negate). Phase 1: the agent watches each action k times on random
states and describes it with the core loop (one two-layer description per action: a position permutation and a
value map). Phase 2: a goal is a target sequence reachable from a random start in <= 3 actions; the agent plans by
breadth-first search over words in its LEARNED action descriptions, and the plan is executed on the true
environment. Nothing is learned from reward; there is no value function; the plan's price is its bits.

Measures (pre-registered): fraction of goals reached; plan length against an oracle search on the true actions; bits;
for k = 1, 2, 3, 8 demonstrations per action. Control: the same learned actions with no composition (single actions).
Pass: >= 95% reached at k = 3 with plans no longer than the oracle's. Refute: < 80% at k = 3, or plans longer.

    python experiments/ziplearn/e9.py            (seconds, CPU) -> runs/e9/e9.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "transformers"))
from ziplearner import TwoLayer   # noqa: E402
import h1_lid as H                # noqa: E402

L, V = H.L, H.V
ACTIONS = list(H.NAMES)           # the seven primitives


def true_apply(name, s):
    return tuple(int(v) for v in H.PRIMS[name](torch.tensor(s)).tolist())


def learn_actions(k, rng):
    """One description per action from k observed (state, next state) pairs."""
    models = {}
    for name in ACTIONS:
        learner = TwoLayer(L, V)
        for _ in range(k):
            s = tuple(int(v) for v in rng.integers(0, V, L))
            learner.observe_demo(np.array(s), np.array(true_apply(name, s)))
        models[name] = learner
    return models


def learned_apply(models, name, s):
    out = models[name].predict_seq(np.array(s))
    return None if any(v is None for v in out) else tuple(int(v) for v in out)


def plan(apply_fn, start, goal, max_depth):
    """Shortest word of actions taking start to goal under apply_fn, by breadth-first search; None if none found."""
    if start == goal:
        return []
    seen = {start: []}
    frontier = deque([start])
    while frontier:
        s = frontier.popleft()
        if len(seen[s]) >= max_depth:
            continue
        for a in ACTIONS:
            n = apply_fn(a, s)
            if n is None or n in seen:
                continue
            seen[n] = seen[s] + [a]
            if n == goal:
                return seen[n]
            frontier.append(n)
    return None


def execute(word, start):
    s = start
    for a in word:
        s = true_apply(a, s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e9"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bits_per_action = math.log2(len(ACTIONS))
    # the goals: a random start and a random word of 1..3 actions, the same set for every condition
    goals = []
    while len(goals) < args.goals:
        start = tuple(int(v) for v in rng.integers(0, V, L))
        word = [ACTIONS[int(i)] for i in rng.integers(0, len(ACTIONS), int(rng.integers(1, 4)))]
        goal = execute(word, start)
        if goal != start:
            goals.append((start, goal))
    oracle = [plan(true_apply, s, g, 4) for s, g in goals]
    print(f"E9: {len(goals)} goals within 3 actions (oracle mean shortest plan {np.mean([len(w) for w in oracle]):.2f} actions)\n")
    print(f"{'k demos/action':>15} | {'reached':>8} {'= oracle length':>16} {'mean plan bits':>15} | {'single-action control':>22}")
    res = {}
    for k in (1, 2, 3, 8):
        models = learn_actions(k, rng)
        reached, same_len, bits, single = 0, 0, [], 0
        for (s, g), orc in zip(goals, oracle):
            w = plan(lambda a, st: learned_apply(models, a, st), s, g, 4)
            if w is not None and execute(w, s) == g:
                reached += 1
                same_len += int(len(w) == len(orc))
                bits.append(len(w) * bits_per_action)
            w1 = plan(lambda a, st: learned_apply(models, a, st), s, g, 1)
            if w1 is not None and execute(w1, s) == g:
                single += 1
        res[k] = dict(reached=reached / len(goals), same_length_as_oracle=same_len / max(1, reached),
                      mean_plan_bits=float(np.mean(bits)) if bits else None, single_action_control=single / len(goals))
        print(f"{k:>15} | {res[k]['reached']:>8.3f} {res[k]['same_length_as_oracle']:>16.3f} "
              f"{(res[k]['mean_plan_bits'] or 0):>15.2f} | {res[k]['single_action_control']:>22.3f}")
    r3 = res[3]
    verdict = "PASS" if (r3["reached"] >= 0.95 and r3["same_length_as_oracle"] >= 0.999) else "REFUTED" if r3["reached"] < 0.8 else "INCONCLUSIVE"
    print(f"\nE9 verdict: {verdict} (k = 3: reached {r3['reached']:.3f}, plans as short as the oracle's {r3['same_length_as_oracle']:.3f})")
    json.dump(dict(results={str(k): v for k, v in res.items()}, oracle_mean_length=float(np.mean([len(w) for w in oracle])),
                   verdict=verdict), open(out / "e9.json", "w"), indent=1)


if __name__ == "__main__":
    main()
