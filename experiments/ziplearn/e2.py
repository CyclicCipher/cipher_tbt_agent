"""E2 — two layers, composition (DESIGN.md §11).

The composition domain of `h1_lid`: 7 primitives on 6-digit sequences over V = 5 -- five move digits between positions
(reverse, rotate left/right, swap pairs, swap halves), two change values (inc, negate) -- and a task is one primitive
followed by another (25 distinct maps: 17 trained, 8 held out). Every task is a position permutation followed by a
value map, so the written network is two layers: layer 1 a 6 x 6 position permutation, layer 2 a value map applied at
every position. Measured: exact-match on a fresh query after k demonstrations; and whether the plain sweep of §6
(start from identity, freeze one layer, solve the other, alternate) converges, against the sweep restarted from every
value-layer description (`TwoLayer`).

Pass (pre-registered): >= 80% of held-out tasks exact after <= 2 demonstrations (today's ZipLearner: 0%), sweep stable
within 3 rounds. Refute: < 50%, or oscillation.

    python experiments/ziplearn/e2.py            (seconds, CPU) -> runs/e2/e2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "inner_objective"))
from ziplearner import Layer, PositionIdentity, PositionPerm, TwoLayer, inv   # noqa: E402
import tasks as TASKS                                                          # noqa: E402

L = TASKS.L


def plain_sweep(demos, L, V, rounds=3):
    """§6 as written: layer 2 starts as the identity; alternate solving layer 1 (positions) given layer 2 and layer 2
    (values, per digit) given layer 1. Returns (pos, value (a, b) or None if the value layer is not invertible, rounds
    until the pair stopped changing, or None if it never did)."""
    value = (1, 0)
    prev, stable_at = None, None
    pos = None
    for r in range(1, rounds + 1):
        pid, pperm = PositionIdentity(L, V), PositionPerm(L, V)
        for x, y in demos:
            tgt = [(inv(value[0], V) * (int(v) - value[1])) % V for v in y]
            pid.observe_demo(x, tgt)
            pperm.observe_demo(x, tgt)
        pos = min((pid, pperm), key=lambda s: (round(s.cost, 9), s.n_params))
        lay = Layer(V)
        for x, y in demos:
            mid = pos.predict(x)
            for m, v in zip(mid, y):
                if m is not None:
                    lay.observe(m, int(v))
        s = lay.keep()
        if s.name == "identity":
            value = (1, 0)
        elif s.name == "shift" and s.b is not None:
            value = (1, s.b)
        elif s.name == "affine" and s.a is not None:
            value = (s.a, s.b)
        else:
            return pos, None, None                              # table / permutation: no closed-form inverse -> deadlock
        state = (pos.name, tuple(sorted((j, pos.pi(j)) for j in range(L)) if isinstance(pos, PositionPerm) else ()), value)
        if state == prev and stable_at is None:
            stable_at = r
        prev = state
    return pos, value, stable_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e2"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    task = TASKS.make("compose", args.seed)
    V, K = task.V, args.k
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"E2: {len(task.train)} trained + {len(task.test)} held-out compositions, V={V}, L={L}, {args.n} instances each, K={K}\n")
    res = {}
    for split, pairs in (("trained", task.train), ("held-out", task.test)):
        acc = np.zeros(K)
        plain_acc = np.zeros(K)
        stable = Counter()
        descs = Counter()
        for pair in pairs:
            for _ in range(args.n):
                x = rng.integers(0, V, (K + 1, L))
                y = task.apply(torch.tensor(x), pair).numpy()
                learner = TwoLayer(L, V)
                demos = []
                for k in range(K):
                    learner.observe_demo(x[k], y[k])
                    demos.append((x[k], y[k]))
                    pred = learner.predict_seq(x[K])
                    acc[k] += float(all(p is not None and p == int(t) for p, t in zip(pred, y[K])))
                    pos, value, st = plain_sweep(demos, L, V)
                    if value is not None:
                        mid = pos.predict(x[K])
                        pp = [None if m is None else (value[0] * m + value[1]) % V for m in mid]
                        plain_acc[k] += float(all(p is not None and p == int(t) for p, t in zip(pp, y[K])))
                    if k == 1:
                        stable[st] += 1
                descs[task.describe(pair) + " -> " + learner.describe()] += 1
        n = len(pairs) * args.n
        res[split] = dict(exact_by_demos=(acc / n).tolist(), plain_sweep_exact_by_demos=(plain_acc / n).tolist(),
                          plain_sweep_rounds_to_stable_at_2_demos={str(k): v for k, v in stable.items()},
                          descriptions=dict(descs))
        print(f"{split:9} restarted sweep, exact after k demos: " + " ".join(f"{a:.2f}" for a in acc / n))
        print(f"{'':9} plain sweep from identity:        " + " ".join(f"{a:.2f}" for a in plain_acc / n)
              + f"   rounds-to-stable at 2 demos: {dict(stable)}")
    print("\nwhat the learner wrote for each held-out task (after 8 demonstrations, most common):")
    for d, c in sorted(res["held-out"]["descriptions"].items()):
        print(f"   {d}  ({c}/{args.n})")
    h2 = res["held-out"]["exact_by_demos"][1]
    verdict = "PASS" if h2 >= 0.8 else "REFUTED" if h2 < 0.5 else "INCONCLUSIVE"
    print(f"\nE2 verdict: {verdict} (held-out exact after 2 demonstrations {h2:.2f})")
    res["verdict"] = verdict
    json.dump(res, open(out / "e2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
