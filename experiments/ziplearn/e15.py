"""E15 — the capacity budget (DESIGN.md §8 change 3, §9 rule 3) and parameter precision (§8 change 2).

A. Six rules arrive in sequence with no labels (E6's protocol): A gets 60 pairs, the other five 20 each. The continual
   layer must stay under a budget of C bits. When it does not fit, it first counts same-kind blocks as one template
   (consolidate) and then drops the block worth least -- evidence x bits saved over a table (rule 3). Compared with
   the same budget under FIFO (drop the oldest) and with no budget. Measured after the stream: bits (consolidated)
   against C; which rules survive; retention on every rule, and evidence-weighted retention (a rule's retention
   weighted by how often it was seen -- what the price rule is supposed to protect).
   Pass: the price rule ends under C, keeps A (the most evidence) where FIFO drops it, and has evidence-weighted
   retention >= FIFO's; refute: it ends over C or drops A.
B. Precision: a real-valued offset b estimated from n noisy observations (y = x + b + noise, sigma = 1) is written
   at a grid step delta; the total code length -- log2(span / delta) for the parameter plus the residuals coded at
   the rounded b -- is minimised over delta for each n. Pass: the best delta shrinks like 1 / sqrt(n) (slope of
   log delta against log n within 0.5 +/- 0.15), which is what precision_bits assumes. Refute: no such slope.

    python experiments/ziplearn/e15.py           (CPU, ~20 s) -> runs/e15/e15.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ziplearner import ContinualLayer, enforce_capacity, block_value, consolidate, precision_bits   # noqa: E402

RULES = {"A": (1, 3), "B": (1, 5), "C": (2, 1), "D": (1, 8), "E": (3, 4), "F": (1, 10)}
STRETCH = {"A": 60, "B": 20, "C": 20, "D": 20, "E": 20, "F": 20}


def apply(rule, x, V):
    a, b = rule
    return (a * x + b) % V


def retention(layer, V, ctx_len, rng):
    out = {}
    for name, rule in RULES.items():
        xs = rng.integers(0, V, ctx_len)
        ctx = [(int(x), apply(rule, int(x), V)) for x in xs]
        out[name] = float(np.mean([layer.predict_in_context(ctx, x) == apply(rule, x, V) for x in range(V)]))
    return out


def run(policy, capacity, seed, V=11):
    rng = np.random.default_rng(seed)
    layer = ContinualLayer(V)
    order = list(RULES)
    dropped_log = []
    for name in order:
        for _ in range(STRETCH[name]):
            x = int(rng.integers(V))
            layer.observe(x, apply(RULES[name], x, V))
        if capacity is not None:
            if policy == "price":
                enforce_capacity(layer, capacity, dropped_log)
            elif policy == "fifo":
                while layer.blocks and layer.total_bits(consolidated=True) > capacity:
                    victim = layer.blocks.pop(0)
                    dropped_log.append(dict(dropped=victim["layer"].keep().name, evidence=victim["n"]))
    ret = retention(layer, V, 3, rng)
    w = {k: STRETCH[k] for k in RULES}
    ew = sum(ret[k] * w[k] for k in RULES) / sum(w.values())
    return dict(bits=layer.total_bits(consolidated=True), blocks=layer.describe(), retention=ret, evidence_weighted=ew,
                dropped=dropped_log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=float, default=48.0)
    ap.add_argument("--streams", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e15"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    print(f"A. six rules, A seen 60 times and the others 20; budget C = {args.capacity} bits (no budget: all six ≈ 70 bits)\n")
    print(f"{'policy':<10}{'bits':>7}{'A':>7}{'B':>7}{'C':>7}{'D':>7}{'E':>7}{'F':>7}{'ev.-weighted':>14}   kept")
    for policy in ("none", "fifo", "price"):
        rows = [run(policy, None if policy == "none" else args.capacity, args.seed + i) for i in range(args.streams)]
        agg = dict(bits=float(np.mean([r["bits"] for r in rows])),
                   retention={k: float(np.mean([r["retention"][k] for r in rows])) for k in RULES},
                   evidence_weighted=float(np.mean([r["evidence_weighted"] for r in rows])),
                   example_blocks=rows[0]["blocks"], example_dropped=rows[0]["dropped"])
        res[policy] = agg
        print(f"{policy:<10}{agg['bits']:>7.1f}" + "".join(f"{agg['retention'][k]:>7.2f}" for k in RULES) +
              f"{agg['evidence_weighted']:>14.3f}   {[b.split(' (')[0] for b in agg['example_blocks']]}")
    print(f"   example of what the price rule dropped: {res['price']['example_dropped']}")
    print(f"   example of what FIFO dropped:           {res['fifo']['example_dropped']}")
    a_ok = (res["price"]["bits"] <= args.capacity + 1e-6 and res["price"]["retention"]["A"] >= 0.999 and
            res["price"]["evidence_weighted"] >= res["fifo"]["evidence_weighted"] and res["fifo"]["retention"]["A"] < 0.999)

    # ── B. precision ────────────────────────────────────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    sigma, span, b_true = 1.0, 64.0, 17.37
    best_delta = {}
    print("\nB. a real-valued offset from n noisy observations: the grid step that minimises the total code length")
    deltas = np.logspace(-3, 1, 120)
    for n in (4, 16, 64, 256, 1024):
        # the EXPECTED total code length at each grid step, over 200 random offsets: a single offset can sit by luck
        # next to a coarse grid point, which is not precision, just fortune
        totals = np.zeros(len(deltas))
        for _ in range(200):
            b = rng.uniform(-span / 2, span / 2)
            resid0 = rng.normal(0, sigma, n)                          # y - x - b_true
            b_hat = b + resid0.mean()
            for di, delta in enumerate(deltas):
                b_q = round(b_hat / delta) * delta
                r = resid0 + (b - b_q)
                totals[di] += math.log2(span / delta) + n * 0.5 * math.log2(2 * math.pi * math.e * sigma ** 2) + float((r ** 2).sum()) / (2 * sigma ** 2 * math.log(2))
        best = float(deltas[int(np.argmin(totals))])
        best_delta[n] = best
        print(f"   n = {n:>5}: best delta {best:.4f}  (sigma*sqrt(12/n) = {sigma * math.sqrt(12 / n):.4f}); precision_bits = {precision_bits(n, span, sigma):.1f}")
    ns = np.log(list(best_delta)); ds = np.log(list(best_delta.values()))
    slope = float(np.polyfit(ns, ds, 1)[0])
    b_ok = abs(slope + 0.5) <= 0.15
    print(f"   slope of log(best delta) against log(n): {slope:.3f} (the 1/sqrt(n) rule predicts −0.5)")
    verdict = "PASS" if (a_ok and b_ok) else "REFUTED" if (res["price"]["bits"] > args.capacity + 1e-6 or res["price"]["retention"]["A"] < 0.999) else "INCONCLUSIVE"
    print(f"\nE15 verdict: {verdict} (A {a_ok}, B {b_ok})")
    res["precision"] = dict(best_delta=best_delta, slope=slope)
    res["verdict"] = verdict
    json.dump(res, open(out / "e15.json", "w"), indent=1)


if __name__ == "__main__":
    main()
