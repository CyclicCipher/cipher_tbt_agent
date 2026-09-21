"""E22 — annealing the budget (DESIGN.md §8, the "raise λ gradually" reading), against a hard cap.

E15's stream (six rules; A seen 60 times, the others 20) under two ways of arriving at the same final budget of
40 bits: HARD -- the cap is 40 from the first stretch; ANNEALED -- the cap starts generous (80) and tightens by 8
bits after each stretch (80, 72, 64, 56, 48, 40), so consolidation and evidence have time to accumulate before
anything is forced out. Measured at the end: bits, blocks kept, evidence-weighted retention, and which rules
survived. Pre-registered: annealing keeps at least the evidence-weighted retention of the hard cap and never more
bits; refute: the hard cap retains more.

    python experiments/ziplearn/e22.py           (CPU, ~20 s) -> runs/e22/e22.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ziplearner import ContinualLayer, enforce_capacity   # noqa: E402
from e15 import RULES, STRETCH, apply, retention           # noqa: E402


def run(schedule, seed, V=11):
    rng = np.random.default_rng(seed)
    layer = ContinualLayer(V)
    dropped = []
    for i, name in enumerate(RULES):
        for _ in range(STRETCH[name]):
            x = int(rng.integers(V))
            layer.observe(x, apply(RULES[name], x, V))
        enforce_capacity(layer, schedule[i], dropped)
    ret = retention(layer, V, 3, rng)
    ew = sum(ret[k] * STRETCH[k] for k in RULES) / sum(STRETCH.values())
    return dict(bits=layer.total_bits(consolidated=True), blocks=layer.describe(), retention=ret, evidence_weighted=ew,
                dropped=[d["dropped"] + f"(n={d['evidence']})" for d in dropped])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e22"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    schedules = {"hard cap 40": [40] * 6, "annealed 80→40": [80, 72, 64, 56, 48, 40]}
    res = {}
    print(f"E22: E15's stream under a final budget of 40 bits, reached at once or gradually; {args.streams} streams\n")
    print(f"{'schedule':<16}{'bits':>7}{'blocks':>8}{'ev.-weighted':>14}   retention A..F / dropped (one stream)")
    for name, sched in schedules.items():
        rows = [run(sched, args.seed + i) for i in range(args.streams)]
        agg = dict(bits=float(np.mean([r["bits"] for r in rows])), blocks=float(np.mean([len(r["blocks"]) for r in rows])),
                   evidence_weighted=float(np.mean([r["evidence_weighted"] for r in rows])),
                   retention={k: float(np.mean([r["retention"][k] for r in rows])) for k in RULES}, example_dropped=rows[0]["dropped"])
        res[name] = agg
        print(f"{name:<16}{agg['bits']:>7.1f}{agg['blocks']:>8.1f}{agg['evidence_weighted']:>14.3f}   "
              + " ".join(f"{agg['retention'][k]:.2f}" for k in RULES) + f" / {agg['example_dropped']}")
    a, h = res["annealed 80→40"], res["hard cap 40"]
    verdict = "PASS" if (a["evidence_weighted"] >= h["evidence_weighted"] - 1e-9 and a["bits"] <= h["bits"] + 1e-6) else "REFUTED" if a["evidence_weighted"] < h["evidence_weighted"] else "INCONCLUSIVE"
    print(f"\nE22 verdict: {verdict}")
    res["verdict"] = verdict
    json.dump(res, open(out / "e22.json", "w"), indent=1)


if __name__ == "__main__":
    main()
