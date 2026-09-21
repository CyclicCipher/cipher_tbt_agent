"""E6 — sequential learning, no replay (DESIGN.md §9, §11).

Rules arrive one at a time with no task labels -- A (shift 3), B (shift 5), C (affine 2x+1), then A again -- each for
a stretch of observations. Nothing from a finished stretch is kept except the descriptions and their evidence counts:
no examples, no replay. After every stretch, every rule seen so far is tested by giving the layer a short CONTEXT of
that rule (a few of its pairs, from which it learns nothing) and reading the whole 11-row matrix of the block it
selects.

Measured: retention (rows correct for every earlier rule) after each stretch; the number of blocks; total bits against a
single table forced to describe the whole stream. Pass (pre-registered): 100% retention throughout; 3 blocks (A's
return reuses its block); total bits below the table's. Refute: retention drops after any stretch, or a fourth block.

    python experiments/ziplearn/e6.py            (seconds, CPU) -> runs/e6/e6.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ziplearner import ContinualLayer, Layer   # noqa: E402

RULES = {"A": (1, 3), "B": (1, 5), "C": (2, 1)}


def apply(rule, x, V):
    a, b = rule
    return (a * x + b) % V


def retention(layer, V, ctx_len, rng):
    """For each rule: give ctx_len pairs of it as context, read the selected block's matrix, score all 11 rows."""
    out = {}
    for name, rule in RULES.items():
        xs = rng.integers(0, V, ctx_len)
        ctx = [(int(x), apply(rule, int(x), V)) for x in xs]
        rows = [layer.predict_in_context(ctx, x) == apply(rule, x, V) for x in range(V)]
        out[name] = float(np.mean(rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stretch", type=int, default=40)
    ap.add_argument("--ctx", type=int, default=3, help="context pairs given at test time (learning nothing)")
    ap.add_argument("--n", type=int, default=20, help="independent streams")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e6"))
    args = ap.parse_args()
    V = 11
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    order = ["A", "B", "C", "A"]
    print(f"E6: stream {' -> '.join(order)} ({args.stretch} pairs each, no labels), {args.n} independent streams, "
          f"context at test time {args.ctx} pairs\n")
    ret_after = {i: {r: [] for r in RULES} for i in range(len(order))}
    blocks_after = {i: [] for i in range(len(order))}
    bits_after, table_bits_after = {i: [] for i in range(len(order))}, {i: [] for i in range(len(order))}
    example = None
    for s in range(args.n):
        layer = ContinualLayer(V)
        table = Layer(V, ("table",))                                   # one lossless table over the whole stream
        for i, name in enumerate(order):
            rule = RULES[name]
            for _ in range(args.stretch):
                x = int(rng.integers(V))
                y = apply(rule, x, V)
                layer.observe(x, y)
                table.observe(x, y)
            seen = set(order[:i + 1])
            r = retention(layer, V, args.ctx, rng)
            for k in seen:
                ret_after[i][k].append(r[k])
            blocks_after[i].append(len(layer.blocks))
            bits_after[i].append(layer.total_bits())
            table_bits_after[i].append(table.price(table.keep()))
        if example is None:
            example = layer.describe()
    print(f"{'after stretch':<16}{'retention A':>12}{'B':>8}{'C':>8} | blocks | bits (table alone)")
    summary = {}
    for i, name in enumerate(order):
        seen = order[:i + 1]
        cells = {k: float(np.mean(ret_after[i][k])) for k in RULES if k in seen}
        nb = float(np.mean(blocks_after[i]))
        summary[f"after {i + 1} ({name})"] = dict(retention=cells, blocks=nb, bits=float(np.mean(bits_after[i])),
                                                 table_bits=float(np.mean(table_bits_after[i])),
                                                 blocks_hist={str(b): blocks_after[i].count(b) for b in sorted(set(blocks_after[i]))})
        print(f"{i + 1} ({name}){'':<10}" + "".join(f"{cells.get(k, float('nan')):>12.3f}" if k == 'A' else f"{cells.get(k, float('nan')):>8.3f}" for k in RULES)
              + f" | {nb:5.2f}  | {np.mean(bits_after[i]):7.1f} ({np.mean(table_bits_after[i]):7.1f})")
    print(f"\nblocks in one example stream at the end: {example}")
    final = summary["after 4 (A)"]
    ok_ret = all(min(ret_after[i][k]) >= 0.999 for i in range(len(order)) for k in order[:i + 1])
    ok_blocks = all(b == 3 for b in blocks_after[3])
    verdict = "PASS" if ok_ret and ok_blocks and final["bits"] < final["table_bits"] else "REFUTED"
    print(f"\nE6 verdict: {verdict} (retention always 100%: {ok_ret}; exactly 3 blocks at the end in every stream: {ok_blocks}; "
          f"bits {final['bits']:.1f} vs table {final['table_bits']:.1f})")
    json.dump(dict(summary=summary, example=example, verdict=verdict), open(out / "e6.json", "w"), indent=1)


if __name__ == "__main__":
    main()
