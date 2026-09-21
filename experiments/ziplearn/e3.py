"""E3 — library discovery (DESIGN.md §7 step 3, §11).

Tasks from the composition domain arrive one after another (K = 8 demonstrations each). The library starts with NO
named position permutation -- only "identity" and "any permutation, solved from scratch" -- so every primitive
(reverse, rotate, swap ...) is a hidden item. After each task the solved permutation is recorded; the second time the
same one is solved it becomes a NAMED item, which later tasks can select for log2(#named) bits instead of solving
its entries for ~9.5. Measured per task: whether its permutation was already named when it arrived, the bits the kept
description cost, and the demonstrations until the query was exact.

Pass (pre-registered): recurring permutations are found, and a task whose permutation is named costs at least the
price of the entries (log2(6!) ≈ 9.5 bits minus the naming bits) less than the same task did before. Refute: nothing
recurs, or no drop.

    python experiments/ziplearn/e3.py            (seconds, CPU) -> runs/e3/e3.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "inner_objective"))
from ziplearner import PermLibrary, TwoLayerWithLibrary, PositionPerm   # noqa: E402
import tasks as TASKS                                                     # noqa: E402

L = TASKS.L


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tasks", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e3"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    task = TASKS.make("compose", args.seed)
    V, K = task.V, args.k
    pairs = task.train + task.test
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lib = PermLibrary()
    log = []
    print(f"E3: a stream of {args.tasks} tasks drawn from {len(pairs)} compositions, V={V}, L={L}, K={K}; the library starts empty\n")
    for i in range(args.tasks):
        pair = pairs[int(rng.integers(len(pairs)))]
        x = rng.integers(0, V, (K + 1, L))
        y = task.apply(torch.tensor(x), pair).numpy()
        learner = TwoLayerWithLibrary(L, V, lib)
        ttc = None
        for k in range(K):
            learner.observe_demo(x[k], y[k])
            pred = learner.predict_seq(x[K])
            if ttc is None and all(p is not None and p == int(t) for p, t in zip(pred, y[K])):
                ttc = k + 1
        _name, _ab, pos = learner.best()
        pi = learner.solved_pi()
        was_named = (pi in lib.named) if pi is not None else None
        bits_pos = pos.cost + learner.bits_pos_name
        promoted = lib.record(pi) if (pi is not None and isinstance(pos, PositionPerm)) else False
        log.append(dict(i=i, task=task.describe(pair), kept=pos.name, pi=pi, was_named=was_named,
                        bits_position_layer=round(bits_pos, 2), demos_to_exact=ttc, promoted=promoted, n_named=len(lib.named)))
        if promoted or i < 4 or i % 10 == 0:
            print(f"   task {i:3d} {task.describe(pair):<24} kept {pos.name:<20} bits(position layer) {bits_pos:6.2f} "
                  f"demos-to-exact {ttc}  named items {len(lib.named)}{'  <- promoted' if promoted else ''}")
    # ── the pre-registered numbers ──────────────────────────────────────────────────────────────────────────────────
    named_rows = [r for r in log if r["was_named"]]
    unnamed_rows = [r for r in log if r["was_named"] is False and r["kept"] == "position permutation"]
    print(f"\nnamed items at the end: {len(lib.named)}")
    b_named, b_unnamed = mean([r["bits_position_layer"] for r in named_rows]), mean([r["bits_position_layer"] for r in unnamed_rows])
    d_named, d_unnamed = mean([r["demos_to_exact"] for r in named_rows]), mean([r["demos_to_exact"] for r in unnamed_rows])
    print(f"tasks whose permutation was already named ({len(named_rows)}): position layer {b_named:.2f} bits, demos-to-exact {d_named:.2f}")
    print(f"tasks solved from scratch ({len(unnamed_rows)}):                 position layer {b_unnamed:.2f} bits, demos-to-exact {d_unnamed:.2f}")
    drops = []
    for pi in lib.named:
        before = [r["bits_position_layer"] for r in log if r["pi"] == pi and r["was_named"] is False]
        after = [r["bits_position_layer"] for r in log if r["pi"] == pi and r["was_named"]]
        if before and after:
            drops.append(float(np.mean(before) - np.mean(after)))
    need = math.log2(math.factorial(L)) - math.log2(max(1, len(lib.named)))
    print(f"price drop per named permutation (before naming -> after): mean {mean(drops):.2f} bits, min {min(drops) if drops else float('nan'):.2f}, "
          f"over {len(drops)} items; pass needs every drop >= {need:.2f}")
    verdict = "PASS" if (lib.named and drops and min(drops) >= need) else "REFUTED" if not lib.named else "INCONCLUSIVE"
    print(f"\nE3 verdict: {verdict}")
    json.dump(dict(log=log, named=[list(p) for p in lib.named],
                   summary=dict(n_named=len(lib.named), bits_named=b_named, bits_unnamed=b_unnamed, demos_named=d_named,
                                demos_unnamed=d_unnamed, drops=drops, need=need, verdict=verdict)),
              open(out / "e3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
