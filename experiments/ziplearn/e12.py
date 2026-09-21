"""E12 — abelian vs non-abelian action sets: what commutativity buys, and whether the learner can find it.

Two action sets of four over 6-digit sequences (V = 5), each learned from k observations per action (TwoLayer):
  abelian      rot_left, rot_right, inc, dec        -- rotations and value shifts commute (Z6 x Z5, order 30)
  non-abelian  rot_left, reverse, inc, negate       -- reverse does not commute with rotation, negate not with inc
Parts:
  A. discovery: from the LEARNED maps alone, does each pair of actions commute (a(b(s)) == b(a(s)) on random states)?
  B. summarisability: for words of length <= 3, do words with the same letter counts have the same effect? (In an
     abelian set the effect of a word is its counts -- a vector; in a non-abelian set it is the word itself.)
  C. planning three ways on goals reachable in <= 3 actions: breadth-first search over words (E9); a COORDINATE
     planner for the abelian set (identify the needed rotation and shift, then the plan is arithmetic: how many
     rotations, how many shifts, choosing the shorter direction); a WORD-TABLE planner for the non-abelian set (every
     reachable transformation enumerated once with its shortest word, then matched). Measured: goals reached, plan
     optimality, states examined per goal, and the bits it takes to describe each planner (two cycle orders and a
     generator each, against a table of words).
Pass (pre-registered): A finds all 6 abelian pairs commuting and at least 2 non-abelian pairs not; B 1.00 vs < 1.00;
C the coordinate planner reaches >= 95% of abelian goals with optimal plans and costs fewer bits than the word table
the non-abelian set needs. Refute: A wrong, or the coordinate planner < 80%.

    python experiments/ziplearn/e12.py           (CPU, ~1 min) -> runs/e12/e12.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "transformers"))
from ziplearner import TwoLayer   # noqa: E402
import h1_lid as H                # noqa: E402

L, V = H.L, H.V
SETS = {"abelian": ["rot_left", "rot_right", "inc", "dec"], "non-abelian": ["rot_left", "reverse", "inc", "negate"]}


def true_apply(name, s):
    s = tuple(int(v) for v in s)
    if name == "dec":
        return tuple((v - 1) % V for v in s)
    return tuple(int(v) for v in H.PRIMS[name](torch.tensor(s)).tolist())


def learn(name, k, rng):
    lr = TwoLayer(L, V)
    for _ in range(k):
        s = tuple(int(v) for v in rng.integers(0, V, L))
        lr.observe_demo(np.array(s), np.array(true_apply(name, s)))
    return lr


def learned_apply(models, name, s):
    out = models[name].predict_seq(np.array(s))
    return None if any(v is None for v in out) else tuple(int(v) for v in out)


def apply_word(models, word, s):
    for a in word:
        s = learned_apply(models, a, s)
        if s is None:
            return None
    return s


def bfs(models, actions, start, goal, max_depth):
    """Shortest word and the number of states examined."""
    if start == goal:
        return [], 0
    seen = {start: []}
    frontier = deque([start])
    examined = 0
    while frontier:
        s = frontier.popleft()
        if len(seen[s]) >= max_depth:
            continue
        for a in actions:
            n = learned_apply(models, a, s)
            examined += 1
            if n is None or n in seen:
                continue
            seen[n] = seen[s] + [a]
            if n == goal:
                return seen[n], examined
            frontier.append(n)
    return None, examined


def fingerprint(models, word, probes):
    return tuple(apply_word(models, word, p) for p in probes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--goals", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e12"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    probes = [tuple(int(v) for v in rng.integers(0, V, L)) for _ in range(4)]
    for set_name, actions in SETS.items():
        models = {a: learn(a, args.k, rng) for a in actions}
        # A. commutation, from the learned maps
        comm = {}
        for a, b in itertools.combinations(actions, 2):
            agree = 0
            for _ in range(40):
                s = tuple(int(v) for v in rng.integers(0, V, L))
                x = learned_apply(models, b, learned_apply(models, a, s) or s)
                y = learned_apply(models, a, learned_apply(models, b, s) or s)
                agree += int(x is not None and x == y)
            comm[f"{a},{b}"] = agree / 40
        commuting = sum(1 for v in comm.values() if v >= 0.99)
        # B. words with equal letter counts, equal effects?
        words = [w for n in (1, 2, 3) for w in itertools.product(actions, repeat=n)]
        by_counts = {}
        for w in words:
            by_counts.setdefault(tuple(sorted(Counter(w).items())), set()).add(fingerprint(models, w, probes))
        summarisable = float(np.mean([len(v) == 1 for v in by_counts.values()]))
        group = {}
        for w in words:
            group.setdefault(fingerprint(models, w, probes), w)
        # C. planning
        goals = []
        while len(goals) < args.goals:
            start = tuple(int(v) for v in rng.integers(0, V, L))
            w = [actions[int(i)] for i in rng.integers(0, len(actions), int(rng.integers(1, 4)))]
            g = start
            for a in w:
                g = true_apply(a, g)
            if g != start:
                goals.append((start, g))
        # C1 search
        reached, optimal, examined = 0, 0, []
        oracle_len = []
        for s, g in goals:
            w, ex = bfs(models, actions, s, g, 4)
            wo, _ = bfs({a: models[a] for a in actions}, actions, s, g, 4)     # same maps: the search optimum
            oracle_len.append(len(wo) if wo is not None else None)
            examined.append(ex)
            if w is not None and apply_word({a: models[a] for a in actions}, w, s) == g:
                gs = s
                for a in w:
                    gs = true_apply(a, gs)
                if gs == g:
                    reached += 1
                    optimal += int(len(w) == len(wo))
        C = dict(search=dict(reached=reached / len(goals), optimal=optimal / max(1, reached), states_examined=float(np.mean(examined))))
        # C2: the coordinate planner (abelian) or the word-table planner (non-abelian)
        if set_name == "abelian":
            # coordinates from the learned maps: the order of the rotation generator and of the shift generator
            def order(a):
                s = probes[0]
                for n in range(1, 20):
                    s = learned_apply(models, a, s)
                    if s == probes[0]:
                        return n
                return None
            orders = {a: order(a) for a in actions}
            n_rot, n_val = orders["rot_left"], orders["inc"]
            r2, o2 = 0, 0
            ex2 = []
            for s, g in goals:
                found = None
                checks = 0
                for i in range(n_rot):                                       # identify (i, j): 30 checks, no search
                    for j in range(n_val):
                        checks += 1
                        if apply_word(models, ["rot_left"] * i + ["inc"] * j, s) == g:
                            found = (i, j)
                            break
                    if found:
                        break
                ex2.append(checks)
                if found is None:
                    continue
                i, j = found
                plan = (["rot_left"] * i if i <= n_rot - i else ["rot_right"] * (n_rot - i)) + \
                       (["inc"] * j if j <= n_val - j else ["dec"] * (n_val - j))
                gs = s
                for a in plan:
                    gs = true_apply(a, gs)
                if gs == g:
                    r2 += 1
                    wo, _ = bfs(models, actions, s, g, 6)
                    o2 += int(wo is not None and len(plan) == len(wo))
            bits = math.log2(len(actions)) * 2 + math.log2(n_rot) + math.log2(n_val)       # two generators, two orders
            C["coordinates"] = dict(reached=r2 / len(goals), optimal=o2 / max(1, r2), states_examined=float(np.mean(ex2)),
                                    planner_bits=bits, orders=orders)
        else:
            # the word table: every distinct transformation reachable within 6 letters, with its shortest word
            table = {}
            for n in range(1, 7):
                for w in itertools.product(actions, repeat=n):
                    fp = fingerprint(models, w, probes)
                    if fp not in table:
                        table[fp] = w
            r2, o2, ex2 = 0, 0, []
            for s, g in goals:
                checks, found = 0, None
                for fp, w in table.items():
                    checks += 1
                    if apply_word(models, w, s) == g:
                        found = w
                        break
                ex2.append(checks)
                if found is None:
                    continue
                gs = s
                for a in found:
                    gs = true_apply(a, gs)
                if gs == g:
                    r2 += 1
                    wo, _ = bfs(models, actions, s, g, 6)
                    o2 += int(wo is not None and len(found) == len(wo))
            bits = sum(len(w) * math.log2(len(actions)) + math.log2(1 + len(w)) for w in table.values())
            C["word table"] = dict(reached=r2 / len(goals), optimal=o2 / max(1, r2), states_examined=float(np.mean(ex2)),
                                   planner_bits=bits, table_size=len(table))
        res[set_name] = dict(commutation=comm, commuting_pairs=commuting, summarisable=summarisable,
                             distinct_effects_len_le_3=len(group), planning=C)
        print(f"{set_name} {actions}")
        print(f"   A. commuting pairs (learned maps): {commuting}/6  {comm}")
        print(f"   B. words with equal letter counts have equal effects: {summarisable:.2f}; distinct effects of words <= 3: {len(group)}")
        for pl, r in C.items():
            extra = f" | planner bits {r['planner_bits']:.1f}" if "planner_bits" in r else ""
            extra += f" | table {r['table_size']}" if "table_size" in r else ""
            print(f"   C. {pl:<12} reached {r['reached']:.3f} optimal {r['optimal']:.3f} states examined/goal {r['states_examined']:.1f}{extra}")
        print()
    ab, nab = res["abelian"], res["non-abelian"]
    a_ok = ab["commuting_pairs"] == 6 and nab["commuting_pairs"] <= 4
    b_ok = ab["summarisable"] >= 0.999 and nab["summarisable"] < 0.999
    c_ok = (ab["planning"]["coordinates"]["reached"] >= 0.95 and ab["planning"]["coordinates"]["optimal"] >= 0.999 and
            ab["planning"]["coordinates"]["planner_bits"] < nab["planning"]["word table"]["planner_bits"])
    verdict = "PASS" if (a_ok and b_ok and c_ok) else "REFUTED" if (not a_ok or ab["planning"]["coordinates"]["reached"] < 0.8) else "INCONCLUSIVE"
    print(f"E12 verdict: {verdict} (A {a_ok}, B {b_ok}, C {c_ok})")
    res["verdict"] = verdict
    json.dump(res, open(out / "e12.json", "w"), indent=1)


if __name__ == "__main__":
    main()
