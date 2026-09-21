"""E23 — the cache is the library (DESIGN.md §16, OPEN-10 first form): macro-actions shrink the search.

E9's environment with goals farther away (4..6 actions). Two planners over the same learned action descriptions
(4 observations each): breadth-first search over the seven primitive actions; and the same search with the library's
named items (E3's 17 recurring permutations, each a word of primitives) offered as single MACRO-actions alongside the
primitives. A macro costs its word's bits and executes as its word. Measured: goals reached, nodes expanded per goal,
plan length in primitive steps against the oracle. Pre-registered: with macros, nodes expanded fall by at least 2x
at no loss of goals reached, and plans are at most one primitive step longer than the oracle's on average. Refute:
no reduction in nodes, or fewer goals reached.

    python experiments/ziplearn/e23.py           (CPU, ~1 min) -> runs/e23/e23.json
"""
from __future__ import annotations

import argparse
import itertools
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
ACTIONS = list(H.NAMES)
POS = ACTIONS[:5]


def true_apply(name, s):
    return tuple(int(v) for v in H.PRIMS[name](torch.tensor(s)).tolist())


def execute(word, s):
    for a in word:
        s = true_apply(a, s)
    return s


def learn(k, rng):
    models = {}
    for a in ACTIONS:
        lr = TwoLayer(L, V)
        for _ in range(k):
            s = tuple(int(v) for v in rng.integers(0, V, L))
            lr.observe_demo(np.array(s), np.array(true_apply(a, s)))
        models[a] = lr
    return models


def learned_apply(models, a, s):
    out = models[a].predict_seq(np.array(s))
    return None if any(v is None for v in out) else tuple(int(v) for v in out)


def apply_word(models, word, s):
    for a in word:
        s = learned_apply(models, a, s)
        if s is None:
            return None
    return s


def search(models, moves, start, goal, max_depth):
    """moves: name -> word of primitives. Returns (plan as a list of primitives, nodes expanded)."""
    if start == goal:
        return [], 0
    seen = {start: []}
    frontier = deque([(start, 0)])
    nodes = 0
    while frontier:
        s, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        for name, word in moves.items():
            nodes += 1
            n = apply_word(models, word, s)
            if n is None or n in seen:
                continue
            seen[n] = seen[s] + list(word)
            if n == goal:
                return seen[n], nodes
            frontier.append((n, depth + 1))
    return None, nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e23"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = learn(4, rng)
    named = [tuple(p) for p in json.load(open(HERE / "runs" / "e3" / "e3.json"))["named"]]
    # each named permutation as a word of position primitives (the shortest of length <= 2 that produces it)
    probe = tuple(range(L))
    def perm_of(word):
        s = probe
        for a in word:
            s = true_apply(a, s)
        return s
    macros = {}
    for pi in named:
        for n in (1, 2):
            for w in itertools.product(POS, repeat=n):
                if perm_of(w) == pi:
                    macros[f"macro:{'∘'.join(w)}"] = list(w)
                    break
            else:
                continue
            break
    prim_moves = {a: [a] for a in ACTIONS}
    macro_moves = dict(prim_moves, **macros)
    goals = []
    while len(goals) < args.goals:
        start = tuple(int(v) for v in rng.integers(0, V, L))
        word = [ACTIONS[int(i)] for i in rng.integers(0, len(ACTIONS), int(rng.integers(4, 7)))]
        g = execute(word, start)
        if g != start:
            goals.append((start, g))
    oracle = [search({a: models[a] for a in ACTIONS}, prim_moves, s, g, 6)[0] for s, g in goals]
    print(f"E23: {len(goals)} goals 4–6 actions away; {len(macros)} macro-actions from the library's named permutations\n")
    print(f"{'planner':<22}{'reached':>9}{'nodes/goal':>12}{'plan steps':>12}{'oracle':>8}")
    res = {}
    for name, moves in (("primitives", prim_moves), ("primitives + macros", macro_moves)):
        reached, nodes, steps, orc = 0, [], [], []
        for (s, g), o in zip(goals, oracle):
            w, n = search(models, moves, s, g, 6)
            nodes.append(n)
            if w is not None and execute(w, s) == g:
                reached += 1
                steps.append(len(w))
                orc.append(len(o) if o is not None else len(w))
        res[name] = dict(reached=reached / len(goals), nodes=float(np.mean(nodes)), steps=float(np.mean(steps)) if steps else None,
                         oracle=float(np.mean(orc)) if orc else None)
        print(f"{name:<22}{res[name]['reached']:>9.3f}{res[name]['nodes']:>12.1f}{res[name]['steps'] or 0:>12.2f}{res[name]['oracle'] or 0:>8.2f}")
    p, m = res["primitives"], res["primitives + macros"]
    ok = m["reached"] >= p["reached"] and m["nodes"] <= p["nodes"] / 2 and (m["steps"] or 99) <= (m["oracle"] or 0) + 1.0
    verdict = "PASS" if ok else "REFUTED" if (m["nodes"] >= p["nodes"] or m["reached"] < p["reached"]) else "INCONCLUSIVE"
    print(f"\nE23 verdict: {verdict}")
    res["verdict"] = verdict
    res["macros"] = macros
    json.dump(res, open(out / "e23.json", "w"), indent=1)


if __name__ == "__main__":
    main()
