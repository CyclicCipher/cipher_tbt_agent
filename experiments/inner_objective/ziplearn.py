"""ZipLearn: a compression-style learner with no gradients, tested on modular arithmetic.

THE IDEA. A ZIP file learns without backprop: it builds a dictionary of what recurs and picks, per block, the filter
(PNG: none/sub/up/average) under which the block is smallest. Learning is counting; model selection is "which filter
compresses this best"; prediction is decompression. This file is that recipe applied to task learning:

  * a PPM-style DICTIONARY: counts of (input digit -> output digit) per task, one pass, no gradients (hypothesis H0);
  * generic FILTERS from data compression -- finite differences of order 1 and 2 across the input digit (H1, H2): if the
    order-k difference of the output table is constant, the table is a degree-k polynomial in the input and the whole
    table follows from k+1 observations. This is PNG's "sub" filter, not a hand-coded affine family: the learner is never
    told arithmetic exists, only that "differences may be constant" is a filter worth trying;
  * TWO-PART CODE selection per task: the hypothesis with the smallest description of the demonstrations seen so far
    (parameters cost log2(p) each; a prediction the hypothesis got wrong costs an escape); the query is decoded under it.

Succession, addition, subtraction, multiplication and division mod p are all degree-1, so the critical test -- "can it
learn the generalisable structure of modular arithmetic" -- is: from a few demonstrations of an UNSEEN (a, b), does it
predict unseen inputs exactly, and how many demonstrations does that take? Noise tasks are the control: no filter
compresses them, H0 stays at maximum entropy, and the learner reports that it has learned nothing -- immunity to the
noisy TV without a curriculum.

What is NOT learned: the filter set itself. Position-permuting tasks (reverse, rotate) need positional filters this
file does not have; that boundary is reported, not hidden.

    python experiments/inner_objective/ziplearn.py            (seconds, CPU)
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
import tasks as TASKS  # noqa: E402

L = TASKS.L


def inv(a, p):
    return pow(int(a), p - 2, p)                                   # p prime


class Hyp:
    """A hypothesis about one task: how output digits depend on input digits, with an MDL cost."""

    def __init__(self, p, order):
        self.p, self.order = p, order
        self.table = {}                                            # H0 dictionary: x -> counts over y
        self.points = {}                                           # distinct (x -> y) observations, for H1/H2
        self.cost = 0.0                                            # bits spent so far (the description of the demos)
        self.wrong = 0

    def predict(self, x):
        p = self.p
        if self.order == 0:
            c = self.table.get(x)
            return (max(c, key=c.get) if c else None)
        pts = sorted(self.points.items())
        if len(pts) < self.order + 1:
            return self.table.get(x) and max(self.table[x], key=self.table[x].get)
        if self.order == 1:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            a = (y1 - y0) * inv(x1 - x0, p) % p
            return (y0 + a * (x - x0)) % p
        (x0, y0), (x1, y1), (x2, y2) = pts[0], pts[1], pts[2]      # Lagrange, degree 2, over Z_p
        num = 0
        for xi, yi in ((x0, y0), (x1, y1), (x2, y2)):
            term = yi
            for xj, _ in ((x0, y0), (x1, y1), (x2, y2)):
                if xj != xi:
                    term = term * (x - xj) * inv(xi - xj, p) % p
            num = (num + term) % p
        return num

    def observe(self, x, y):
        """Pay the code length of this observation under the current state, then update."""
        p = self.p
        guess = self.predict(x)
        if guess is None:                                           # no prediction possible yet: a parameter, log2(p)
            self.cost += math.log2(p)
        elif guess == y:
            self.cost += math.log2(1 + 1.0 / (1 + 4 * len(self.points)))   # nearly free once the model is fixed
        else:
            self.cost += math.log2(p) + 1.0                         # escape + the symbol
            self.wrong += 1
        self.table.setdefault(x, {}).setdefault(y, 0)
        self.table[x][y] += 1
        if x not in self.points:
            self.points[x] = y


class ZipLearner:
    """Per task: three hypotheses observe the same stream; the cheapest description wins; the query is decoded."""

    def __init__(self, p):
        self.p = p
        self.hyps = [Hyp(p, 0), Hyp(p, 1), Hyp(p, 2)]

    def observe_demo(self, x_seq, y_seq):
        for x, y in zip(x_seq, y_seq):
            for h in self.hyps:
                h.observe(int(x), int(y))

    def best(self):
        return min(self.hyps, key=lambda h: (h.cost + 2.0 * h.order, h.order))   # +2 bits per order: L(filter)

    def predict_seq(self, x_seq):
        h = self.best()
        out = [h.predict(int(x)) for x in x_seq]
        return [o if o is not None else 0 for o in out], h.order, h.cost


def evaluate(task, pairs, K, n, rng, noise=False, learner=None):
    """Exact-match accuracy of the query after k demonstrations (k = 1..K), averaged over n instances per task."""
    learner = learner or ZipLearner
    p = task.V
    acc = np.zeros(K)
    orders = np.zeros(3)
    bits = []                                                   # the learner's own code length per digit at the end
    for pair in pairs:
        for _ in range(n):
            x = rng.integers(0, p, (K + 1, L))
            y = (task.apply(torch.tensor(x), pair).numpy() if not noise else rng.integers(0, p, (K + 1, L)))
            zl = learner(p)
            for k in range(K):
                zl.observe_demo(x[k], y[k])
                pred, order, _c = zl.predict_seq(x[K])
                acc[k] += float(np.array_equal(np.array(pred), y[K]))
            orders[order] += 1
            bits.append(zl.best().cost / (K * L))
    return acc / (len(pairs) * n), orders / orders.sum(), float(np.mean(bits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="affine", choices=["affine", "compose"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    task = TASKS.make(args.task, args.seed)
    print(f"ZipLearn on {task.name} (p = {task.V}): {len(task.train)} trained tasks, {len(task.test)} held out, "
          f"{args.n} instances each, K = {args.k} demonstrations\n")
    res = {}
    for name, pairs in (("trained tasks", task.train), ("HELD-OUT tasks", task.test)):
        acc, orders, bits = evaluate(task, pairs, args.k, args.n, rng)
        ttc = next((k + 1 for k, a in enumerate(acc) if a >= 0.8), None)
        res[name] = dict(acc_by_demos=acc.tolist(), ttc=ttc, filter_chosen=orders.tolist(), bits_per_digit=bits)
        print(f"{name:16} exact-match after k demos: " + " ".join(f"{a:.2f}" for a in acc) +
              f"  | demos to criterion: {ttc}  | filter (table/diff1/diff2): " + " ".join(f"{o:.2f}" for o in orders)
              + f"  | its own code length {bits:.2f} bits/digit")
    acc, orders, bits = evaluate(task, task.test[:4], args.k, args.n, rng, noise=True)
    res["noise tasks"] = dict(acc_by_demos=acc.tolist(), filter_chosen=orders.tolist(), bits_per_digit=bits)
    print(f"{'noise tasks':16} exact-match after k demos: " + " ".join(f"{a:.2f}" for a in acc) +
          f"  | filter: " + " ".join(f"{o:.2f}" for o in orders) + f"  | its own code length {bits:.2f} bits/digit "
          f"(max {math.log2(task.V):.2f}: it knows it learned nothing)")
    # a table-only ablation: the pure-ZIP dictionary without the difference filters
    class TableOnly(ZipLearner):
        def __init__(self, p):
            self.p = p
            self.hyps = [Hyp(p, 0)]
    acc_t, _o, _b = evaluate(task, task.test, args.k, args.n, rng, learner=TableOnly)
    res["held-out, table only"] = dict(acc_by_demos=acc_t.tolist())
    print(f"{'held-out, table':16} exact-match after k demos: " + " ".join(f"{a:.2f}" for a in acc_t) +
          "   (the dictionary alone: memorises digits it has seen, cannot extrapolate)")
    json.dump(res, open(HERE / "runs" / f"ziplearn_{task.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
