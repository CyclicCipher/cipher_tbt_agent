"""Stage 3 — the FACTORED (place-value) representation + the generalization test (architecture doc §14).

A number is stored IN THE COLUMN as `digit ⊗ place` bindings — the column's native What(L4 content) ×
Where(L6 location) binding, exactly the brain's place-value system (Grossberg: categorical-What × spatial-
Where; Dehaene neuronal recycling). Addition reads the digit at each place, adds it on the column's learned
number line (by navigation), and propagates a base-b carry.

The column learns ONLY the single-digit number line (0 .. 2b-1). It NEVER sees a multi-digit number, so
EVERY test addition is held-out — correctness = the factored representation GENERALISED (it composed 10
digit-symbols), it did not memorise. Contrast: the holistic line (stage 2) caps at one column's place capacity (~d_mem) and cannot
even represent a 3-digit number. The place-value rule (decompose digits / carry) is the symbolic layer that
the column's ANS digit arithmetic runs underneath — provided here, as it is culturally taught in humans.

Run:  python -m demos.factored      (run from src/ with PYTHONPATH=src)
"""

from __future__ import annotations

import os
import random
import sys


from tbt.agent import Agent                                       # noqa: E402
from tbt.thalamus import Thalamus                                 # noqa: E402

from demos.numberline import NumberLine                       # noqa: E402


class FactoredColumn:
    def __init__(self, base: int = 10, n_places: int = 12, torus: int = 22, seed: int = 0):
        self.base = base
        self.thal = Thalamus()                                   # joins the two columns (digit ⊗ position)
        n = 2 * base                                              # single-digit sums (+carry) land in 0..2b-1
        env = NumberLine(n=n, seed=seed, shuffle=False)
        self.ag = Agent(n_symbols=n, torus=torus, seed=seed).explore_and_learn(env, steps=8 * n * n, seed=seed)
        # positions (units→tens→hundreds) are their OWN 1-D structure; discover their frame the same way, so
        # place value = digit-frame (What, self.ag) ⊗ position-frame (Where, self.pos), the two joined by the
        # thalamus. Under approach A a column's frame spans ONLY what it explored, so the factorization
        # genuinely needs a second column + the thalamic channel — the multi-column claim, surfacing concretely.
        pos_env = NumberLine(n=n_places, seed=seed, shuffle=False)
        self.pos = Agent(n_symbols=n_places, torus=torus, seed=seed + 101).explore_and_learn(
            pos_env, steps=8 * n_places * n_places, seed=seed + 101)

    # ---- the factored representation = a thalamic register:  number  <->  Σ_p  digit_p ⊗ position_p -------
    def encode(self, number: int):
        items, p = [], 0
        while number or p == 0:
            items.append((number % self.base, p))                # (digit, position)
            number //= self.base
            p += 1
        return self.thal.bind(self.ag, self.pos, items)          # bind digit (What) at position (Where)

    def read_digit(self, R, place: int) -> int:
        return self.thal.read(R, self.ag, self.pos, place, default=0)   # unbound place reads 0 (no leading garbage)

    def decode(self, R, n_places: int) -> int:
        return sum(self.read_digit(R, p) * self.base ** p for p in range(n_places))

    # ---- addition: read each place from the factored reps, add on the number line, carry ---------------
    def add(self, a: int, b: int) -> int:
        Ra, Rb = self.encode(a), self.encode(b)
        n_places = max(len(str(a)), len(str(b))) + 1
        out, carry = 0, 0
        for p in range(n_places):
            s = self.ag.add(self.read_digit(Ra, p), self.read_digit(Rb, p) + carry)   # digit-column number line
            out += (s % self.base) * self.base ** p
            carry = s // self.base
        return out


def test_addition(fc, n_digits, trials=200, seed=0):
    rng = random.Random(seed)
    hi = 10 ** n_digits - 1
    correct = sum(fc.add(a, b) == a + b for a, b in
                  ((rng.randint(0, hi), rng.randint(0, hi)) for _ in range(trials)))
    return correct, trials


if __name__ == "__main__":
    print("stage 3 — FACTORED place-value addition; the column learned ONLY the single-digit number line.\n")
    fc = FactoredColumn(base=10)
    # sanity: round-trip a few unseen numbers through the column's digit⊗place representation
    rt = all(fc.decode(fc.encode(x), len(str(x))) == x for x in (0, 7, 42, 305, 9999, 80706))
    print(f"  encode→decode round-trip (unseen numbers): {'OK' if rt else 'FAIL'}\n")
    print(f"  {'digits':>7}  {'range':>14}  {'unseen additions correct':>26}")
    for nd in (1, 2, 3, 5, 8):
        c, t = test_addition(fc, nd)
        print(f"  {nd:>7}  {'0..' + str(10 ** nd - 1):>14}  {c:>12}/{t}")
    print("\n  every test number is unseen (no multi-digit was ever trained) — correctness = generalization")
    print("  from 10 digit-symbols + the place rule, not memorization (holistic caps at one column's place capacity).")
