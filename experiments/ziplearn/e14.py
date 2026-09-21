"""E14 — the order dependence of the price of a collection (DESIGN.md §8 change 3, §7; the curriculum gap).

The 25 compositions of the composition domain, each once, K = 8 demonstrations, as a stream in several orders — from
an empty library every time (E3's machinery: solved position permutations are named on their second occurrence and
later tasks may select them by name). Orders: PARTS FIRST (tasks whose position permutation is a single primitive
come first, then tasks whose permutation is a product of two primitives), WHOLES FIRST (the reverse), and eight random
orders. Measured per order: the total bits paid over the stream (the sum of each task's kept description's price);
the final library (named items) and its price as generators + words (E8); and the price of the same 25 tasks
presented again with the library frozen -- the value of what was learned for the future.

For an ideal compressor the total is order-independent (symmetry of information); the spread between orders measures
this learner's distance from ideal. Pass (pre-registered): parts-first is cheapest; the random orders spread by more
than 10% of their mean; the final libraries differ across orders and so does the frozen-library price. Refute: totals
agree within noise.

    python experiments/ziplearn/e14.py           (CPU, ~1 min) -> runs/e14/e14.json
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
sys.path.insert(0, str(HERE.parent / "transformers"))
from ziplearner import PermLibrary, TwoLayerWithLibrary, PositionPerm, WordLibrary, NamedPerm   # noqa: E402
import tasks as TASKS                                                                       # noqa: E402
import h1_lid as H                                                                          # noqa: E402

L = TASKS.L


def task_price(learner):
    name, ab, pos = learner.best()
    bits = next(b for (n, a, b), *_ in learner.runs if n == name and a == ab)
    return bits + learner.bits_value_name + pos.cost + learner.bits_pos_name


def run_stream(order, task, K, rng, lib=None, freeze=False, words=False):
    """Present the tasks in `order`; return total bits and the library. With freeze=True nothing is named. With
    words=True a task may also select any permutation in the group the named items generate, at its word price (E8):
    the compositional reuse that could make an order matter."""
    from ziplearner import TwoLayerWithWords
    lib = lib if lib is not None else PermLibrary()
    total = 0.0
    for pair in order:
        x = rng.integers(0, task.V, (K, L))
        y = task.apply(torch.tensor(x), pair).numpy()
        if words and lib.named:
            learner = TwoLayerWithWords(L, task.V, WordLibrary(lib.named, L, max_generators=3))
        else:
            learner = TwoLayerWithLibrary(L, task.V, lib)
        for k in range(K):
            learner.observe_demo(x[k], y[k])
        total += task_price(learner)
        if not freeze:
            _n, _ab, pos = learner.best()
            pi = learner.solved_pi()
            if pi is not None and isinstance(pos, PositionPerm):
                lib.record(pi)
    return total, lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--random_orders", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--words", type=int, default=0, help="1: tasks may use words in the named items (E8)")
    ap.add_argument("--out", default=str(HERE / "runs" / "e14"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    task = TASKS.make("compose", args.seed)
    tasks = task.train + task.test
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    value_prims = {5, 6}                                                  # inc, negate: value maps, not positions
    parts = [p for p in tasks if (p[0] in value_prims) != (p[1] in value_prims) or (p[0] in value_prims and p[1] in value_prims)]
    wholes = [p for p in tasks if p not in parts]                         # two position primitives composed
    orders = {"parts first": parts + wholes, "wholes first": wholes + parts}
    for i in range(args.random_orders):
        orders[f"random {i}"] = [tasks[j] for j in rng.permutation(len(tasks))]
    res = {}
    print(f"E14: {len(tasks)} tasks ({len(parts)} whose permutation is a single primitive, {len(wholes)} products), K={args.k}\n")
    print(f"{'order':<14}{'total bits':>11}{'named':>7}{'library as words':>18}{'frozen re-pass':>15}")
    for name, order in orders.items():
        rs = np.random.default_rng(args.seed + 7)                        # the same demonstrations for every order
        total, lib = run_stream(order, task, args.k, rs, words=bool(args.words))
        wl = WordLibrary(lib.named, L, max_generators=3) if lib.named else None
        rs2 = np.random.default_rng(args.seed + 99)
        frozen, _ = run_stream(tasks, task, args.k, rs2, lib=lib, freeze=True, words=bool(args.words))
        res[name] = dict(total_bits=total, n_named=len(lib.named), named=[list(p) for p in lib.named],
                         library_words_bits=(wl.price if wl else 0.0), frozen_repass_bits=frozen)
        print(f"{name:<14}{total:>11.1f}{len(lib.named):>7}{(wl.price if wl else 0.0):>18.1f}{frozen:>15.1f}")
    rand = [res[f"random {i}"]["total_bits"] for i in range(args.random_orders)]
    spread = (max(rand) - min(rand)) / float(np.mean(rand))
    libs = {tuple(sorted(map(tuple, res[n]["named"]))) for n in res}
    parts_cheapest = res["parts first"]["total_bits"] <= min(rand) and res["parts first"]["total_bits"] < res["wholes first"]["total_bits"]
    frozen_vals = [res[n]["frozen_repass_bits"] for n in res]
    print(f"\nrandom orders: mean {np.mean(rand):.1f}, spread (max−min)/mean {spread:.1%}; distinct final libraries {len(libs)} of {len(res)}; "
          f"frozen re-pass from {min(frozen_vals):.1f} to {max(frozen_vals):.1f} bits")
    print(f"curriculum gap (worst − best total): {max(r['total_bits'] for r in res.values()) - min(r['total_bits'] for r in res.values()):.1f} bits "
          f"(an ideal compressor: 0)")
    ok = parts_cheapest and spread > 0.10 and len(libs) > 1 and (max(frozen_vals) - min(frozen_vals)) > 1.0
    verdict = "PASS" if ok else "REFUTED" if spread < 0.03 else "INCONCLUSIVE"
    print(f"\nE14 verdict: {verdict} (parts-first cheapest {parts_cheapest}, spread {spread:.1%}, libraries differ {len(libs) > 1})")
    res["summary"] = dict(spread=spread, distinct_libraries=len(libs), parts_cheapest=parts_cheapest, verdict=verdict)
    json.dump(res, open(out / ("e14_words.json" if args.words else "e14.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
