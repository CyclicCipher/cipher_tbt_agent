"""Emergent multi-column allocation — place value where NO column is told its role (architecture doc §12.3).

A POOL of identical columns + a basal-ganglia gate + the thalamus. Two structures — a digit number line and a
position line — are presented to the pool; the BG ALLOCATES each to a column by competition (random-init
symmetry break) + load-balancing + dopamine-RPE reinforcement. WHICH column becomes the digit column vs the
position column EMERGES (it changes with the seed); it is not hand-assigned (Mountcastle). The thalamus then
composes the two GATE-CHOSEN columns into place-value addition — the SAME generalization as factored.py, but
with the factorization ALLOCATED, not designed.

Honest scope: disentanglement — discovering THAT the task factors into digit × position — is still given; the
two structures arrive as separate streams (§12.3c, the open sub-problem). What is emergent here is the
ALLOCATION (which column, and the learned routing back to it), the part the BG / MoE gate owns. One column
would be forced to hold both structures and they would collide (the capacity wall); the gate spreads them.

Run:  python -m precursor.multicolumn      (from experiments/RecurrentWorldModel/)
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tbt.basal_ganglia import BasalGanglia                   # noqa: E402
from tbt.column import CorticalColumn                        # noqa: E402
from tbt.thalamus import Thalamus                            # noqa: E402


def _line(n):
    """Transitions of a 1-D line 0..n-1: action 0 = +1, action 1 = -1 (no wrap)."""
    t = []
    for s in range(n):
        if s + 1 < n:
            t.append((s, 0, s + 1))
        if s - 1 >= 0:
            t.append((s, 1, s - 1))
    return t


def _learn(col, transitions):
    for s, a, s2 in transitions:                             # observe records the edge graph (order-independent)
        col.observe(s, a, s2)
    col.consolidate()                                        # → SR-eigenvector frame + per-relation operators


def _acc(col, transitions):
    return sum(col.predict(s, a) == s2 for s, a, s2 in transitions) / len(transitions)


class EmergentPlaceValue:
    def __init__(self, base=10, n_places=12, n_columns=3, torus=22, seed=0):
        self.base = base
        self.thal = Thalamus()
        self.bg = BasalGanglia(n_columns, seed=seed)
        self.cols = [CorticalColumn(n_entities=2 * base, torus_size=torus, place_k=1, seed=seed * 10 + c)
                     for c in range(n_columns)]               # a POOL of identical columns (distinct seeds = niches)
        streams = {"digit": _line(2 * base), "position": _line(n_places)}   # given+identified; allocation is emergent
        self.assign = {}
        for key, trans in streams.items():
            c = self.bg.select(key)                          # the gate picks a column — roles NOT hand-assigned
            _learn(self.cols[c], trans)
            self.bg.reinforce(key, c, _acc(self.cols[c], trans))   # dopamine-RPE on how well it modelled the stream
            self.assign[key] = c
        self.digit_col = self.cols[self.assign["digit"]]     # whatever the gate chose
        self.pos_col = self.cols[self.assign["position"]]

    # place value via the GATE-CHOSEN columns, joined by the thalamus (cf. factored.py, but allocated)
    def encode(self, number):
        items, p = [], 0
        while number or p == 0:
            items.append((number % self.base, p))
            number //= self.base
            p += 1
        return self.thal.bind(self.digit_col, self.pos_col, items)

    def read_digit(self, R, place):
        return self.thal.read(R, self.digit_col, self.pos_col, place, default=0)

    def add(self, a, b):
        Ra, Rb = self.encode(a), self.encode(b)
        n_places = max(len(str(a)), len(str(b))) + 1
        out, carry = 0, 0
        for p in range(n_places):
            s = self.digit_col.add(self.read_digit(Ra, p), self.read_digit(Rb, p) + carry)
            out += (s % self.base) * self.base ** p
            carry = s // self.base
        return out


def test_addition(pv, n_digits, trials=200, seed=0):
    rng = random.Random(seed)
    hi = 10 ** n_digits - 1
    return sum(pv.add(a, b) == a + b for a, b in
               ((rng.randint(0, hi), rng.randint(0, hi)) for _ in range(trials))), trials


if __name__ == "__main__":
    print("emergent multi-column place value — the BG gate allocates columns; roles are NOT hand-assigned.\n")
    print(f"  {'seed':>4}  {'digit→col':>10}  {'position→col':>13}  {'reroute':>8}  {'add 2-digit':>12}  {'5-digit':>9}")
    for seed in (0, 1, 2):
        pv = EmergentPlaceValue(seed=seed)
        reroute = (pv.bg.select("digit") == pv.assign["digit"]
                   and pv.bg.select("position") == pv.assign["position"])   # the gate routes a known structure back
        c2, t2 = test_addition(pv, 2)
        c5, t5 = test_addition(pv, 5)
        print(f"  {seed:>4}  {pv.assign['digit']:>10}  {pv.assign['position']:>13}  {str(reroute):>8}"
              f"  {f'{c2}/{t2}':>12}  {f'{c5}/{t5}':>9}")
    print(f"\n  (pool of {len(pv.cols)} identical columns.) which column is the digit line vs the position line")
    print("  EMERGES (varies by seed) and the gate routes back to it; the thalamus composes")
    print("  the gate-chosen columns → place value, allocated not designed. Disentanglement (§12.3c) still given.")
