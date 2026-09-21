"""E4 — the lossy price (DESIGN.md §8 change 1, §11).

Two prices for the same descriptions. v1 ("flat"): a correct prediction costs log2(1 + 1/(1+4n)), an exception a flat
1 bit plus the digit. "rate": the exception FLAG is coded by an adaptive two-symbol code (Krichevsky-Trofimov,
P(exception) = (n_wrong + 1/2)/(n + 1)), so n observations at exception rate eps cost about n*H(eps) + n_exc*log2(V-1):
exceptions are paid for by their rate, not remembered one by one.

Measured, on "shift by 3 with an exception rate eps" for eps in {0, 0.1, 0.3} and on a random mapping:
  (a) is each price a valid code? -- the implied probabilities of {right, each wrong digit} must sum to at most 1;
  (b) the price per observation as n grows, against the entropy bound H(eps) + eps*log2(V-1);
  (c) the exception rate the kept description reports, against the true eps;
  (d) which structure is kept.
Pass (pre-registered): the rate price stays flat per observation, reports eps ~ 0.10 and keeps the shift.
The pre-registration also expected the flat price to "grow and eventually prefer the table"; that expectation is
tested here and reported either way (the table pays the same exceptions, so it may simply not happen).

    python experiments/ziplearn/e4.py            (seconds, CPU) -> runs/e4/e4.json
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
from ziplearner import Layer, flag_bits   # noqa: E402


def entropy(e):
    return 0.0 if e in (0.0, 1.0) else -(e * math.log2(e) + (1 - e) * math.log2(1 - e))


def implied_mass(price, V, n_right, n_wrong):
    """Sum of the probabilities a price implies for one observation: 2^-cost(right) + (V-1) * 2^-cost(one wrong digit)."""
    if price == "flat":
        p_right = 2.0 ** -math.log2(1 + 1.0 / (1 + 4 * n_right))
        p_wrong_each = 2.0 ** -(math.log2(V) + 1.0)
    else:
        p_right = 2.0 ** -flag_bits(False, n_right, n_wrong)
        p_wrong_each = 2.0 ** -(flag_bits(True, n_right, n_wrong) + math.log2(V - 1))
    return p_right + (V - 1) * p_wrong_each


def run(price, eps, V, n_obs, n_inst, rng, random_map=False):
    per_obs = np.zeros(n_obs)
    kept, rates = Counter(), []
    for _ in range(n_inst):
        layer = Layer(V, price=price)
        for i in range(n_obs):
            x = int(rng.integers(V))
            y = int(rng.integers(V)) if random_map else (x + 3) % V
            if not random_map and rng.random() < eps:
                y = int((y + 1 + rng.integers(V - 1)) % V)             # a wrong digit, never the right one
            layer.observe(x, y)
            per_obs[i] += layer.price(layer.keep()) / (i + 1)
        s = layer.keep()
        kept[s.name] += 1
        rates.append(s.rate)
    return dict(price_per_obs=(per_obs / n_inst).tolist(), kept=dict(kept), reported_rate=float(np.mean(rates)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_obs", type=int, default=200)
    ap.add_argument("--n_inst", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e4"))
    args = ap.parse_args()
    V = 11
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    print("(a) is the price a valid code?  implied probability mass per observation (must be <= 1):")
    for price in ("flat", "rate"):
        masses = [implied_mass(price, V, n, w) for n, w in ((0, 0), (1, 0), (5, 0), (20, 0), (20, 2), (100, 10))]
        res[f"mass_{price}"] = masses
        print(f"   {price:5} " + " ".join(f"{m:.3f}" for m in masses) + "   (at n_right,n_wrong = 0,0 1,0 5,0 20,0 20,2 100,10)")
    print(f"\n(b)-(d) shift by 3 with exception rate eps; {args.n_obs} observations, {args.n_inst} instances; bound = H(eps) + eps*log2(V-1)")
    print(f"   {'eps':>5} {'price':>5} | bits/obs at n=25, 50, 100, 200 | bound | reported eps | kept")
    for eps in (0.0, 0.1, 0.3):
        for price in ("flat", "rate"):
            r = run(price, eps, V, args.n_obs, args.n_inst, rng)
            res[f"eps{eps}_{price}"] = r
            b = r["price_per_obs"]
            bound = entropy(eps) + eps * math.log2(V - 1)
            print(f"   {eps:5.2f} {price:>5} | {b[24]:.2f} {b[49]:.2f} {b[99]:.2f} {b[199]:.2f} | {bound:.2f} | "
                  f"{r['reported_rate']:.3f} | {r['kept']}")
    print("\n   random mapping (no rule at all; the noisy TV):")
    for price in ("flat", "rate"):
        r = run(price, 0.0, V, args.n_obs, args.n_inst, rng, random_map=True)
        res[f"random_{price}"] = r
        b = r["price_per_obs"]
        print(f"   {'--':>5} {price:>5} | {b[24]:.2f} {b[49]:.2f} {b[99]:.2f} {b[199]:.2f} | {math.log2(V):.2f} | "
              f"{r['reported_rate']:.3f} | {r['kept']}")
    r = res["eps0.1_rate"]
    b = r["price_per_obs"]
    flat_per_obs = abs(b[199] - b[99]) < 0.05
    passed = flat_per_obs and abs(r["reported_rate"] - 0.10) < 0.03 and r["kept"].get("shift", 0) >= 0.9 * args.n_inst
    verdict = "PASS" if passed else "REFUTED"
    valid_flat = max(res["mass_flat"]) <= 1.0 + 1e-9
    print(f"\nthe flat price is {'a valid' if valid_flat else 'NOT a valid'} code (mass up to {max(res['mass_flat']):.3f}); "
          f"the rate price's mass is {max(res['mass_rate']):.3f}")
    print(f"E4 verdict: {verdict}")
    res["verdict"] = verdict
    json.dump(res, open(out / "e4.json", "w"), indent=1)


if __name__ == "__main__":
    main()
