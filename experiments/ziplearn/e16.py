"""E16 — blocks across layers (DESIGN.md §9, OPEN-7): continual learning of whole two-layer tasks with no labels.

E6's protocol on compositions: rule A (rot_left then inc), B (reverse then negate), C (swap_pairs then inc), then A
again, each for a stretch of demonstrations, no task labels, no replay. A block is a TwoLayer description (a position
layer and a value layer); routing and minting are the digit-level layer's, with a demonstration as the unit.
Measured after every stretch: retention -- every rule seen so far, given 2 of its demonstrations as context, exact
match on a fresh query -- the number of blocks, and total bits. Pass (pre-registered): retention >= 0.95 for every
rule after every stretch; exactly 3 blocks at the end in >= 90% of streams. Refute: retention < 0.8 or blocks
multiplying (> 4 on average). OPEN-7's question is the block count.

    python experiments/ziplearn/e16.py           (CPU, ~1 min) -> runs/e16/e16.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "inner_objective"))
sys.path.insert(0, str(HERE.parent / "transformers"))
from ziplearner import ContinualTwoLayer   # noqa: E402
import tasks as TASKS                       # noqa: E402
import h1_lid as H                          # noqa: E402

L, V = TASKS.L, H.V
NAMES = list(H.NAMES)
RULES = {"A": (NAMES.index("rot_left"), NAMES.index("inc")), "B": (NAMES.index("reverse"), NAMES.index("negate")),
         "C": (NAMES.index("swap_pairs"), NAMES.index("inc"))}


def apply(pair, x):
    return H.apply_pair(torch.tensor(x), pair).numpy()


def retention(layer, rng, ctx=2, n=20):
    out = {}
    for name, pair in RULES.items():
        ok = 0
        for _ in range(n):
            xs = rng.integers(0, V, (ctx + 1, L))
            demos = [(xs[i], apply(pair, xs[i])) for i in range(ctx)]
            pred = layer.predict_in_context(demos, xs[ctx])
            truth = apply(pair, xs[ctx])
            ok += int(pred is not None and all(p is not None and p == int(t) for p, t in zip(pred, truth)))
        out[name] = ok / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stretch", type=int, default=25)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e16"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    order = ["A", "B", "C", "A"]
    ret_after = {i: {r: [] for r in RULES} for i in range(len(order))}
    blocks_after = {i: [] for i in range(len(order))}
    bits_after = {i: [] for i in range(len(order))}
    example = None
    print(f"E16: two-layer rules {order}, {args.stretch} demonstrations each, {args.n} streams\n")
    for s in range(args.n):
        layer = ContinualTwoLayer(L, V)
        for i, name in enumerate(order):
            for _ in range(args.stretch):
                x = rng.integers(0, V, L)
                layer.observe_demo(x, apply(RULES[name], x))
            r = retention(layer, rng)
            for k in order[:i + 1]:
                ret_after[i][k].append(r[k])
            blocks_after[i].append(len(layer.blocks))
            bits_after[i].append(layer.total_bits())
        example = example or layer.describe()
    print(f"{'after stretch':<16}{'A':>7}{'B':>7}{'C':>7} | blocks | bits")
    summary = {}
    for i, name in enumerate(order):
        cells = {k: float(np.mean(ret_after[i][k])) for k in RULES if ret_after[i][k]}
        nb = float(np.mean(blocks_after[i]))
        summary[f"after {i + 1} ({name})"] = dict(retention=cells, blocks=nb, bits=float(np.mean(bits_after[i])))
        print(f"{i + 1} ({name}){'':<10}" + "".join(f"{cells.get(k, float('nan')):>7.2f}" for k in RULES) + f" | {nb:5.2f}  | {np.mean(bits_after[i]):6.1f}")
    print(f"\nblocks in one stream at the end: {example}")
    ok_ret = all(min(ret_after[i][k]) >= 0.95 for i in range(len(order)) for k in order[:i + 1] if ret_after[i][k])
    three = np.mean([b == 3 for b in blocks_after[3]])
    verdict = "PASS" if (ok_ret and three >= 0.9) else "REFUTED" if (np.mean(blocks_after[3]) > 4 or any(
        np.mean(ret_after[i][k]) < 0.8 for i in range(len(order)) for k in order[:i + 1])) else "INCONCLUSIVE"
    print(f"\nE16 verdict: {verdict} (retention ≥ 0.95 everywhere: {ok_ret}; exactly 3 blocks at the end in {three:.0%} of streams)")
    json.dump(dict(summary=summary, example=example, verdict=verdict, three_blocks=float(three)), open(out / "e16.json", "w"), indent=1)


if __name__ == "__main__":
    main()
