"""E11 — non-bijective actions (DESIGN.md §16 item 2).

E9's environment plus three actions that are NOT one-to-one: copy01 (position 1 := position 0), set0 (position 0 := 0),
fill0 (every position := position 0). They destroy information: no inverse exists, some states cannot be undone, and
a goal can become unreachable after one wrong move. The library gains one structure for them -- a position MAP, where
an output position either reads any input position (repeats allowed) or is written a constant -- and nothing else
changes: the same core loop describes them, and the same forward search plans with them.

Measured (pre-registered):
  A. description: the effect of each action, learned from k observations, tested on 200 random states -- fraction
     exactly right, for the seven one-to-one actions and the three many-to-one ones, k = 1..4, 8;
  B. planning: goals reachable in <= 3 actions using all ten; forward search over the learned effects; fraction
     reached and plan length against the oracle, as in E9;
  C. irreversibility: the same goals with E10's price-driven tries, unknown actions including the writes -- how many
     goals are lost because a TRY erased what the goal needed (a try that makes the goal unreachable within the
     budget), against the same count with the one-to-one unknowns of E10. Reported, not judged.
Pass: A >= 0.95 exact at k = 4 for the many-to-one actions; B >= 0.95 reached at k = 4 with plans no longer than the
oracle's. Refute: A < 0.8 or B < 0.8 at k = 4.

    python experiments/ziplearn/e11.py           (CPU, ~2 min) -> runs/e11/e11.json
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
from ziplearner import TwoLayer, PositionPerm, PositionMap   # noqa: E402
import h1_lid as H                                            # noqa: E402

L, V = H.L, H.V
BIJ = list(H.NAMES)
NONBIJ = ["copy01", "set0", "fill0"]
ACTIONS = BIJ + NONBIJ


def true_apply(name, s):
    s = tuple(int(v) for v in s)
    if name == "copy01":
        return (s[0], s[0]) + s[2:]
    if name == "set0":
        return (0,) + s[1:]
    if name == "fill0":
        return (s[0],) * L
    return tuple(int(v) for v in H.PRIMS[name](torch.tensor(s)).tolist())


def learn(name, k, rng):
    learner = TwoLayer(L, V)
    for _ in range(k):
        s = tuple(int(v) for v in rng.integers(0, V, L))
        learner.observe_demo(np.array(s), np.array(true_apply(name, s)))
    return learner


def learned_apply(learner, s):
    out = learner.predict_seq(np.array(s))
    return None if any(v is None for v in out) else tuple(int(v) for v in out)


def exactness(learner, name, rng, n=200):
    ok = 0
    for _ in range(n):
        s = tuple(int(v) for v in rng.integers(0, V, L))
        ok += int(learned_apply(learner, s) == true_apply(name, s))
    return ok / n


def plan(apply_fn, start, goal, max_depth):
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


def execute(word, s):
    for a in word:
        s = true_apply(a, s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", type=int, default=200)
    ap.add_argument("--n_learn", type=int, default=20, help="independent learners per (action, k) for part A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e11"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}

    # ── A. description of one-to-one and many-to-one actions ───────────────────────────────────────────────────────
    print("A. exact effect (200 random states) after k observations, mean over actions and 20 learners:")
    print(f"   {'k':>3} | {'one-to-one (7)':>15} | {'many-to-one (3)':>16} | kept structure for the many-to-one")
    A = {}
    for k in (1, 2, 3, 4, 8):
        ex = {}
        kept = {}
        for name in ACTIONS:
            vals, ks = [], []
            for _ in range(args.n_learn):
                lr = learn(name, k, rng)
                vals.append(exactness(lr, name, rng))
                ks.append(lr.best()[2].name)
            ex[name] = float(np.mean(vals))
            kept[name] = max(set(ks), key=ks.count)
        A[k] = dict(per_action=ex, kept=kept, bij=float(np.mean([ex[a] for a in BIJ])), nonbij=float(np.mean([ex[a] for a in NONBIJ])))
        print(f"   {k:>3} | {A[k]['bij']:>15.3f} | {A[k]['nonbij']:>16.3f} | " +
              ", ".join(f"{a}: {kept[a]}" for a in NONBIJ))
    res["A"] = {str(k): v for k, v in A.items()}

    # ── B. planning with all ten actions ────────────────────────────────────────────────────────────────────────────
    goals = []
    while len(goals) < args.goals:
        start = tuple(int(v) for v in rng.integers(0, V, L))
        word = [ACTIONS[int(i)] for i in rng.integers(0, len(ACTIONS), int(rng.integers(1, 4)))]
        g = execute(word, start)
        if g != start:
            goals.append((start, g))
    oracle = [plan(true_apply, s, g, 4) for s, g in goals]
    uses_nonbij = sum(1 for w in oracle if any(a in NONBIJ for a in w))
    print(f"\nB. {len(goals)} goals (oracle mean shortest plan {np.mean([len(w) for w in oracle]):.2f}; "
          f"{uses_nonbij} of them need a many-to-one action in every shortest plan found)")
    print(f"   {'k':>3} | {'reached':>8} {'= oracle length':>16}")
    B = {}
    for k in (1, 2, 3, 4, 8):
        models = {a: learn(a, k, rng) for a in ACTIONS}
        reached, same = 0, 0
        for (s, g), orc in zip(goals, oracle):
            w = plan(lambda a, st: learned_apply(models[a], st), s, g, 4)
            if w is not None and execute(w, s) == g:
                reached += 1
                same += int(len(w) == len(orc))
        B[k] = dict(reached=reached / len(goals), same_length=same / max(1, reached))
        print(f"   {k:>3} | {B[k]['reached']:>8.3f} {B[k]['same_length']:>16.3f}")
    res["B"] = {str(k): v for k, v in B.items()}

    # ── C. irreversibility: tries that erase the goal ───────────────────────────────────────────────────────────────
    # A try of an action `a` from state s is "fatal" if, after it, the goal is no longer reachable within the remaining
    # budget under the TRUE actions. Count fatal tries for one-to-one unknowns (E10's) and many-to-one unknowns.
    def fatal_rate(unknown, trials=300):
        fatal = 0
        for _ in range(trials):
            s, g = goals[int(rng.integers(len(goals)))]
            a = unknown[int(rng.integers(len(unknown)))]
            s2 = true_apply(a, s)
            w = plan(true_apply, s2, g, 5)                              # 6-step budget minus the try
            fatal += int(w is None)
        return fatal / trials
    C = dict(one_to_one_unknowns=fatal_rate(["reverse", "rot_right", "swap_halves"]), many_to_one_unknowns=fatal_rate(NONBIJ))
    print(f"\nC. a try that makes the goal unreachable within the budget: one-to-one unknowns {C['one_to_one_unknowns']:.3f}, "
          f"many-to-one unknowns {C['many_to_one_unknowns']:.3f} of tries")
    res["C"] = C

    a4, b4 = A[4]["nonbij"], B[4]
    verdict = ("PASS" if (a4 >= 0.95 and b4["reached"] >= 0.95 and b4["same_length"] >= 0.999)
               else "REFUTED" if (a4 < 0.8 or b4["reached"] < 0.8) else "INCONCLUSIVE")
    print(f"\nE11 verdict: {verdict} (k = 4: many-to-one exact {a4:.3f}; reached {b4['reached']:.3f}, optimal {b4['same_length']:.3f})")
    res["verdict"] = verdict
    json.dump(res, open(out / "e11.json", "w"), indent=1)


if __name__ == "__main__":
    main()
