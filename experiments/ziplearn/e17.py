"""E17 — goals that are relations, not states (DESIGN.md §16, OPEN-12).

"Make the output the reverse of the input" is not a target state; it is a TRANSFORMATION the plan must implement for
every input. In E9's environment (seven actions, learned from 4 observations each), a goal is now given as two
demonstration pairs (x -> T(x)) of an unknown relation T -- a composition of 1..3 actions -- and the agent must
return a plan whose composed effect equals T on inputs it has never seen.

Two planners on the same goals. STATE planner (E9): treat the first demonstration as a state goal, search for any
word taking x1 to T(x1), return it. RELATION planner: identify T from the demonstrations (E2's TwoLayer -- a position
structure and a value map, from two pairs), then search for the shortest word of learned actions whose composed
effect agrees with T on a set of probe states (E12's fingerprint), i.e. the word that IS the relation. Both plans
are executed on 20 fresh inputs; a goal is solved when the plan's effect matches T on all of them.

Pass (pre-registered): the relation planner solves >= 90% of relational goals; the state planner solves fewer, since
a word that happens to take x1 to T(x1) need not be T (the gap is the measure of why relations are their own kind of
goal). Refute: the relation planner < 70%.

    python experiments/ziplearn/e17.py           (CPU, ~1 min) -> runs/e17/e17.json
"""
from __future__ import annotations

import argparse
import itertools
import json
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


def true_apply(name, s):
    return tuple(int(v) for v in H.PRIMS[name](torch.tensor(s)).tolist())


def execute(word, s):
    for a in word:
        s = true_apply(a, s)
    return s


def learn_actions(k, rng):
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


def state_plan(models, start, goal, max_depth=4):
    if start == goal:
        return []
    seen = {start: []}
    frontier = deque([start])
    while frontier:
        s = frontier.popleft()
        if len(seen[s]) >= max_depth:
            continue
        for a in ACTIONS:
            n = learned_apply(models, a, s)
            if n is None or n in seen:
                continue
            seen[n] = seen[s] + [a]
            if n == goal:
                return seen[n]
            frontier.append(n)
    return None


def relation_plan(models, demos, probes, max_depth=3):
    """Identify T from the demonstrations, then the shortest word whose effect on the probes equals T's."""
    ident = TwoLayer(L, V)
    for x, y in demos:
        ident.observe_demo(np.array(x), np.array(y))
    target = tuple(tuple(v) if v is not None and all(u is not None for u in v) else None for v in (ident.predict_seq(np.array(p)) for p in probes))
    if any(t is None for t in target):
        return None
    for n in range(1, max_depth + 1):
        for word in itertools.product(ACTIONS, repeat=n):
            if tuple(apply_word(models, list(word), p) for p in probes) == target:
                return list(word)
    return None


def relation_plan_direct(models, demos, max_depth=3):
    """The shortest word whose learned effect matches every demonstration pair -- no identification step."""
    for n in range(1, max_depth + 1):
        for word in itertools.product(ACTIONS, repeat=n):
            if all(apply_word(models, list(word), x) == y for x, y in demos):
                return list(word)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--goals", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e17"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = learn_actions(args.k, rng)
    probes = [tuple(int(v) for v in rng.integers(0, V, L)) for _ in range(3)]
    res = {"state": dict(solved=0, first_pair_only=0), "relation": dict(solved=0, none=0), "relation_direct": dict(solved=0, none=0)}
    lengths = {"state": [], "relation": [], "relation_direct": []}
    for _ in range(args.goals):
        word = [ACTIONS[int(i)] for i in rng.integers(0, len(ACTIONS), int(rng.integers(1, 4)))]
        demos = []
        for _ in range(2):
            x = tuple(int(v) for v in rng.integers(0, V, L))
            demos.append((x, execute(word, x)))
        tests = [tuple(int(v) for v in rng.integers(0, V, L)) for _ in range(20)]
        # the state planner: from demo 1's input to demo 1's output
        w_state = state_plan(models, demos[0][0], demos[0][1])
        if w_state is not None:
            if execute(w_state, demos[0][0]) == demos[0][1]:
                res["state"]["first_pair_only"] += 1
            if all(execute(w_state, x) == execute(word, x) for x in tests):
                res["state"]["solved"] += 1
                lengths["state"].append(len(w_state))
        w_rel = relation_plan(models, demos, probes)
        if w_rel is None:
            res["relation"]["none"] += 1
        elif all(execute(w_rel, x) == execute(word, x) for x in tests):
            res["relation"]["solved"] += 1
            lengths["relation"].append(len(w_rel))
        w_dir = relation_plan_direct(models, demos)
        if w_dir is None:
            res["relation_direct"]["none"] += 1
        elif all(execute(w_dir, x) == execute(word, x) for x in tests):
            res["relation_direct"]["solved"] += 1
            lengths["relation_direct"].append(len(w_dir))
    n = args.goals
    print(f"E17: {n} relational goals (a composition of 1–3 actions, given as 2 demonstration pairs); actions learned from {args.k} observations\n")
    print(f"   state planner    : reaches demo 1's output {res['state']['first_pair_only'] / n:.2f}; implements the relation on 20 fresh inputs **{res['state']['solved'] / n:.2f}**")
    print(f"   relation planner : implements the relation on 20 fresh inputs **{res['relation']['solved'] / n:.2f}**; no word found {res['relation']['none'] / n:.2f}; "
          f"mean plan length {np.mean(lengths['relation']) if lengths['relation'] else float('nan'):.2f}")
    print(f"   relation planner, direct (words matched to the demonstrations, no identification): "
          f"**{res['relation_direct']['solved'] / n:.2f}**; no word found {res['relation_direct']['none'] / n:.2f}")
    rs, rr = res["state"]["solved"] / n, res["relation"]["solved"] / n
    verdict = "PASS" if (rr >= 0.9 and rs < rr) else "REFUTED" if rr < 0.7 else "INCONCLUSIVE"
    print(f"\nE17 verdict: {verdict}")
    json.dump(dict(results=res, n=n, verdict=verdict, mean_len_relation=float(np.mean(lengths["relation"])) if lengths["relation"] else None),
              open(out / "e17.json", "w"), indent=1)


if __name__ == "__main__":
    main()
