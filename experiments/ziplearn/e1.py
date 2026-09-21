"""E1 — one layer, arithmetic in weights (DESIGN.md §11).

All 36 rules x -> (a*x + b) mod 11, one 11 x 11 matrix each, library {table, identity, shift, affine, permutation}.
Pairs (x -> y) arrive one at a time from random 6-digit demonstrations. After every pair we record which structure the
layer keeps, the fraction of UNSEEN rows of its written matrix that are correct, and whether the written matrix equals
the true rule on all eleven rows; after every whole demonstration, exact-match on a fresh 6-digit query.

Pass (pre-registered): 100% of unseen rows correct for every rule once two pairs with different x have been seen, with
affine kept (shift when a = 1, identity when a = 1 and b = 0). Refute: the table is kept, or unseen rows are wrong.
Controls: a random but CONSISTENT function of x (nothing but the table can describe it; unseen rows must stay
unanswered) and a random INCONSISTENT mapping (every structure pays escapes; the price per digit must exceed guessing).

    python experiments/ziplearn/e1.py            (seconds, CPU) -> runs/e1/e1.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "inner_objective"))
from ziplearner import Layer                                          # noqa: E402
import tasks as TASKS                                                 # noqa: E402

L = TASKS.L


def run_rule(fn, V, K, n, rng, library):
    """fn: x -> y (may be stochastic for the noise control). Returns per-pair and per-demo statistics over n instances."""
    P = K * L
    unseen = np.zeros(P)
    unseen_n = np.zeros(P)
    exact_matrix = np.zeros(P)
    kept = [Counter() for _ in range(P)]
    at2_unseen, at2_exact, at2_kept = [], [], Counter()
    demo_exact = np.zeros(K)
    bits_kept, bits_table = [], []
    for _ in range(n):
        layer = Layer(V, library)
        truth = np.array([fn(x, rng, final=True) for x in range(V)])   # the rule's own answer per input (for grading)
        x_seq = rng.integers(0, V, (K + 1, L))
        c, done2 = 0, False
        for k in range(K):
            for j in range(L):
                x = int(x_seq[k, j])
                y = int(fn(x, rng))
                layer.observe(x, y)
                W = layer.matrix()
                rows_ok = np.array([W[t].sum() > 0 and W[t].argmax() == truth[t] for t in range(V)])
                un = [t for t in range(V) if t not in layer.seen]
                if un:
                    unseen[c] += rows_ok[un].mean()
                    unseen_n[c] += 1
                exact_matrix[c] += float(rows_ok.all())
                kept[c][layer.keep().name] += 1
                if not done2 and len(layer.seen) >= 2:
                    done2 = True
                    at2_unseen.append(rows_ok[un].mean() if un else 1.0)
                    at2_exact.append(float(rows_ok.all()))
                    at2_kept[layer.keep().name] += 1
                c += 1
            pred = layer.predict_seq(x_seq[K])
            demo_exact[k] += float(all(p is not None and p == truth[x] for p, x in zip(pred, x_seq[K])))
        bits_kept.append(layer.price(layer.keep()) / P)
        bits_table.append(layer.price(next(s for s in layer.structs if s.name == "table")) / P)
    return dict(unseen_by_pair=(unseen / np.maximum(unseen_n, 1)).tolist(),
                exact_matrix_by_pair=(exact_matrix / n).tolist(),
                kept_by_pair=[dict(k) for k in kept],
                after_two_distinct=dict(unseen=float(np.mean(at2_unseen)), exact=float(np.mean(at2_exact)), kept=dict(at2_kept)),
                demo_exact=(demo_exact / n).tolist(),
                bits_per_digit_kept=float(np.mean(bits_kept)), bits_per_digit_table=float(np.mean(bits_table)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e1"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    task = TASKS.make("affine", args.seed)
    V, K = task.V, args.k
    lib = ("table", "identity", "shift", "affine", "permutation")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rules = [(a, b) for a in task.A_SET for b in task.B_SET]
    print(f"E1: {len(rules)} rules (a*x+b mod {V}), {args.n} instances each, K={K} demonstrations of {L} digits, "
          f"library {lib}\n")
    res = {}
    for a, b in rules:
        fn = lambda x, rng, final=False, a=a, b=b: (a * x + b) % V
        res[f"{a}x+{b}"] = run_rule(fn, V, K, args.n, rng, lib)
    # controls
    def make_consistent():
        table = rng.integers(0, V, V)
        return lambda x, rng, final=False: int(table[x])
    def inconsistent(x, rng, final=False):
        return int(rng.integers(0, V))
    controls = {"random consistent function": run_rule(make_consistent(), V, K, args.n, rng, lib),
                "random inconsistent mapping": run_rule(inconsistent, V, K, args.n, rng, lib)}

    # ── the pre-registered numbers ──────────────────────────────────────────────────────────────────────────────────
    held = {f"{a}x+{b}" for a, b in task.test}
    at2 = np.array([r["after_two_distinct"]["unseen"] for r in res.values()])
    ex2 = np.array([r["after_two_distinct"]["exact"] for r in res.values()])
    kept2 = Counter()
    ok_kept = 0
    for name, r in res.items():
        a, b = name.split("x+")
        a, b = int(a), int(b)
        want = "identity" if (a, b) == (1, 0) else "shift" if a == 1 else "affine"
        k = r["after_two_distinct"]["kept"]
        top = max(k, key=k.get)
        kept2[top] += 1
        ok_kept += (top == want)
    worst = min(res.items(), key=lambda kv: kv[1]["after_two_distinct"]["unseen"])
    print("after the first two pairs with DIFFERENT x:")
    print(f"   unseen rows correct: mean {at2.mean():.3f}, min {at2.min():.3f} (rule {worst[0]}); "
          f"written matrix == true rule on all 11 rows: {ex2.mean():.3f}")
    print(f"   kept structure (most common per rule): {dict(kept2)}; matches the expected structure for {ok_kept}/{len(res)} rules")
    demo = np.mean([r["demo_exact"] for r in res.values()], axis=0)
    demo_held = np.mean([r["demo_exact"] for n, r in res.items() if n in held], axis=0)
    print(f"   exact-match on a fresh query after k demonstrations (all 36): " + " ".join(f"{x:.2f}" for x in demo))
    print(f"   same, the 12 held-out rules of the transformer runs:        " + " ".join(f"{x:.2f}" for x in demo_held))
    unseen_curve = np.mean([r["unseen_by_pair"] for r in res.values()], axis=0)
    print(f"   unseen rows correct after 1..6 pairs (mean over rules):     " + " ".join(f"{x:.2f}" for x in unseen_curve[:6]))
    bk = np.mean([r["bits_per_digit_kept"] for r in res.values()])
    bt = np.mean([r["bits_per_digit_table"] for r in res.values()])
    print(f"   price per digit at the end: kept structure {bk:.2f} bits vs the table {bt:.2f} bits (guessing {math.log2(V):.2f})")
    print("\ncontrols:")
    for name, r in controls.items():
        k = r["after_two_distinct"]["kept"]
        print(f"   {name:<28} kept {dict(k)} | unseen rows correct after 6 pairs {r['unseen_by_pair'][5]:.2f} | "
              f"query exact after 8 demos {r['demo_exact'][-1]:.2f} | price per digit {r['bits_per_digit_kept']:.2f} "
              f"(guessing {math.log2(V):.2f})")
    passed = at2.min() >= 0.999 and ok_kept == len(res)
    verdict = "PASS" if passed else "REFUTED"
    print(f"\nE1 verdict: {verdict}")
    json.dump(dict(rules=res, controls=controls, summary=dict(after_two_distinct_unseen_mean=float(at2.mean()),
                   after_two_distinct_unseen_min=float(at2.min()), worst_rule=worst[0], exact_matrix_after_two=float(ex2.mean()),
                   kept_after_two=dict(kept2), kept_matches_expected=ok_kept, n_rules=len(res),
                   demo_exact_all=demo.tolist(), demo_exact_held=demo_held.tolist(), unseen_curve=unseen_curve.tolist(),
                   bits_kept=float(bk), bits_table=float(bt), verdict=verdict)),
              open(out / "e1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
