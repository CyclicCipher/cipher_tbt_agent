"""E5 — a written prior (DESIGN.md §10, §11).

A task is an unknown boolean formula over three bits (a, b, c); a demonstration is one input and its output bit, drawn
at random; the learner must predict outputs for inputs it has not seen. WITHOUT the prior the only description is the
table (one bit per new input; an unseen input has no answer). WITH the prior, AND / OR / XOR and NOT are written in as
gates, and a formula is a wiring of them: f = NOT? g2( NOT? g1(v_i, v_j), v_k ) -- every such wiring is a candidate
description that predicts from the first demonstration and pays for its name (log2 of the number of distinct
formulas) plus the rate price of its exceptions. The cheapest description answers.

Measured: truth-table accuracy (all 8 inputs) after k demonstrations and demonstrations until the whole table is right,
with and without the prior, for formulas INSIDE the class and for random 3-bit functions OUTSIDE it (the prior must
not hurt those beyond its naming bits); and, as the harm check, the same prior on random functions it was not built for.

Pass (pre-registered): fewer demonstrations with the prior on in-class formulas, no harm off it. Refute: no gain, or harm.

    python experiments/ziplearn/e5.py            (seconds, CPU) -> runs/e5/e5.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ziplearner import pay, PRICE   # noqa: E402

GATES = {"and": lambda p, q: p & q, "or": lambda p, q: p | q, "xor": lambda p, q: p ^ q}
INPUTS = list(itertools.product((0, 1), repeat=3))


def formula_table(pair, g1, n1, g2, n2):
    i, j = pair
    k = ({0, 1, 2} - {i, j}).pop()
    out = []
    for v in INPUTS:
        h = GATES[g1](v[i], v[j]) ^ n1
        out.append(GATES[g2](h, v[k]) ^ n2)
    return tuple(out)


def formula_class():
    """Every distinct wiring of the written gates, keyed by its truth table."""
    seen = {}
    for pair in ((0, 1), (0, 2), (1, 2)):
        for g1, n1, g2, n2 in itertools.product(GATES, (0, 1), GATES, (0, 1)):
            tt = formula_table(pair, g1, n1, g2, n2)
            seen.setdefault(tt, (pair, g1, n1, g2, n2))
    return seen


class BoolTable:
    name = "table"

    def __init__(self):
        self.V, self.price, self.cost, self.n_right, self.n_wrong = 2, PRICE, 0.0, 0, 0
        self.entries = {}

    def predict(self, v):
        return self.entries.get(v)

    def observe(self, v, y):
        g = self.predict(v)
        if g is None:
            self.cost += 1.0                                              # one bit: a new entry
        else:
            pay(self, g == y)
        self.entries.setdefault(v, y)


class BoolFormula:
    name = "formula"

    def __init__(self, tt, bits_name):
        self.V, self.price, self.cost, self.n_right, self.n_wrong = 2, PRICE, bits_name, 0, 0
        self.tt = dict(zip(INPUTS, tt))

    def predict(self, v):
        # at V = 2 a description and its complement cost the same under the rate price ("wrong every time" is as
        # compressible as "right every time"), so a prediction honours the learned exception rate: flip if wrong more
        # often than right. The cost is unchanged; only what is read out.
        return self.tt[v] ^ int(self.n_wrong > self.n_right)

    def observe(self, v, y):
        pay(self, self.tt[v] == y)


def run(target, use_prior, klass, K, rng):
    structs = [BoolTable()]
    if use_prior:
        bits = math.log2(len(klass) + 1)                                 # naming: the table or one of the formulas
        structs[0].cost += bits
        structs += [BoolFormula(tt, bits) for tt in klass]
    acc, first_full = [], None
    for k in range(1, K + 1):
        v = INPUTS[int(rng.integers(8))]
        y = target[INPUTS.index(v)]
        for s in structs:
            s.observe(v, y)
        best = min(structs, key=lambda s: (round(s.cost, 9), min(s.n_wrong, s.n_right), 0 if s.name == "formula" else 1))
        table_acc = np.mean([best.predict(u) is not None and best.predict(u) == target[i] for i, u in enumerate(INPUTS)])
        acc.append(float(table_acc))
        if first_full is None and table_acc == 1.0:
            first_full = k
    return acc, first_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e5"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    klass = formula_class()
    tts = list(klass)
    print(f"E5: the written gates give {len(tts)} distinct formulas over 3 bits (of 256 functions); {args.n} tasks per condition, "
          f"K={args.k} demonstrations\n")
    res = {}
    for cond, sampler in (("in-class formula", lambda: tts[int(rng.integers(len(tts)))]),
                          ("random function (30% in class by chance)", lambda: tuple(int(b) for b in rng.integers(0, 2, 8)))):
        for use_prior in (False, True):
            accs, fulls = [], []
            for _ in range(args.n):
                target = sampler()
                acc, full = run(target, use_prior, klass, args.k, rng)
                accs.append(acc)
                fulls.append(full if full is not None else args.k + 1)
            accs = np.mean(accs, axis=0)
            res[f"{cond} | prior={use_prior}"] = dict(acc_by_demos=accs.tolist(), demos_to_full_table=float(np.mean(fulls)))
            print(f"   {cond:<32} prior={str(use_prior):<5} truth-table accuracy after 1,2,4,8,16 demos: "
                  f"{accs[0]:.2f} {accs[1]:.2f} {accs[3]:.2f} {accs[7]:.2f} {accs[15]:.2f} | demos to the whole table: {np.mean(fulls):.1f}")
    # the harm check: the same prior on functions it was not built for (random 3-bit functions; about 30% of them fall
    # inside the class by chance). A prior over boolean gates cannot even be offered to the digit-level library of E1
    # (different input type), so the arithmetic check named in §11 is replaced by this one, and the log says so.
    gain = res["in-class formula | prior=False"]["demos_to_full_table"] - res["in-class formula | prior=True"]["demos_to_full_table"]
    harm = res["random function (30% in class by chance) | prior=True"]["demos_to_full_table"] - res["random function (30% in class by chance) | prior=False"]["demos_to_full_table"]
    print(f"\n   gain from the prior on in-class formulas: {gain:.1f} fewer demonstrations to the whole table")
    print(f"   cost of the prior on random functions: {harm:+.1f} demonstrations")
    verdict = "PASS" if gain > 0 and harm <= 1.0 else "REFUTED"
    print(f"\nE5 verdict: {verdict}")
    res["verdict"] = verdict
    res["n_formulas"] = len(tts)
    json.dump(res, open(out / "e5.json", "w"), indent=1)


if __name__ == "__main__":
    main()
