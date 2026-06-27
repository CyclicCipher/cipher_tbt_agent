"""Stage 3b — can the CARRY RULE be learned (not provided)?

In factored.py the carry was hand-coded (`s % base`, `s // base`). Here it is not coded at all. The insight:
the carry is **modular** — it is what happens when a digit line *wraps* (9→0) — and grid cells are
intrinsically cyclic. So we learn a **cyclic** (mod-base) digit line from a cyclic environment; the
successor operator the column learns then includes the wrap edge (base−1 → 0). Single-digit addition is
navigation on that cyclic line:

    digit = where you land   (= (da+db) mod base, but never written as a %)
    carry = how many times you wrapped   (= wraps of the cyclic line crossed, never written as a //)

So the carry EMERGES from the learned modular structure. The column learned ONLY the cyclic 0..base−1 line;
every multi-digit test number is unseen. The place-value *decomposition* (reading a number's digits) is the
symbolic representation, as before; the CARRY is now learned.

Run:  python -m demos.carry      (run from src/ with PYTHONPATH=src)
"""

from __future__ import annotations

import os
import random
import sys


import torch                                                      # noqa: E402

from tbt.column_learner import ColumnLearner as Agent                                       # noqa: E402
from tbt.env import Environment, Step                             # noqa: E402


class CyclicLine(Environment):
    """A digit line that WRAPS: succ at base−1 returns to 0 (mod-base ring)."""

    def __init__(self, base: int = 10, seed: int = 0):
        self.base = base
        self.symbol = list(range(base))                           # symbol == digit, in order
        self.pos = 0

    def reset(self):
        self.pos = 0
        return self.symbol[self.pos]

    def step(self, action: int) -> Step:
        self.pos = (self.pos + (1 if action == 0 else -1)) % self.base
        return Step(self.symbol[self.pos], 0.0, False)

    @property
    def actions(self):
        return [0, 1]


class CyclicCarry:
    def __init__(self, base: int = 10, torus: int = 22, seed: int = 0):
        self.base = base
        env = CyclicLine(base, seed)
        self.ag = Agent(n_symbols=base, torus=torus, seed=seed).explore_and_learn(env, steps=30 * base * base, seed=seed)
        # the column DISCOVERS the topology itself: the SR-eigenvector frame of a RING has a cyclic spectrum
        # (distinct from a line), so rel[0] is already the cyclic successor operator — it includes the wrap
        # edge (base−1 → 0). Nothing here is told the line is cyclic.
        self.codes = self.ag.place[: base]
        self.pos_to_sym = {p: s for s, p in self.ag.loc.items()}
        self.discovered_wrap = self.digit_add(base - 1, 1) == (0, 1)   # succ of the top digit wraps to 0 (a carry)

    def _succ(self, pos: int) -> int:
        v = self.ag.col.L5.apply(self.ag.rel[0], self.codes[pos])
        return int((self.codes @ v).argmax())

    def digit_add(self, da: int, count: int):
        """Land = result digit; wraps = carry. Read off the DISCOVERED cyclic line — no % or // anywhere."""
        pos, carry = self.ag.loc[da], 0
        for _ in range(count):
            nxt = self._succ(pos)
            if self.pos_to_sym[nxt] < self.pos_to_sym[pos]:       # digit value wrapped (9 → 0) = a carry
                carry += 1
            pos = nxt
        return self.pos_to_sym[pos], carry

    def add(self, a: int, b: int) -> int:
        out, carry, place = 0, 0, 1
        while a or b or carry:
            d, carry = self.digit_add(a % self.base, b % self.base + carry)   # decompose = representation; carry = learned
            out += d * place
            a //= self.base
            b //= self.base
            place *= self.base
        return out


def test(adder, n_digits, trials=60, seed=0):
    rng = random.Random(seed)
    hi = 10 ** n_digits - 1
    correct = sum(adder.add(a, b) == a + b for a, b in
                  ((rng.randint(0, hi), rng.randint(0, hi)) for _ in range(trials)))
    return correct, trials


if __name__ == "__main__":
    print("stage 3b — LEARNED carry: the column DISCOVERS the cyclic line (SR frame of a ring), not told.\n")
    cc = CyclicCarry(base=10)
    print(f"  cyclic wrap discovered (succ of {cc.base - 1} -> 0, i.e. a carry): {cc.discovered_wrap}")
    sd = all(cc.digit_add(da, db) == (((da + db) % 10), ((da + db) // 10))
             for da in range(10) for db in range(11))
    print(f"  single-digit add+carry (all 110 da,db pairs) matches truth: {'OK' if sd else 'FAIL'}\n")
    print(f"  {'digits':>7}  {'range':>14}  {'unseen additions correct':>26}")
    for nd in (1, 2, 3, 5):
        c, t = test(cc, nd)
        print(f"  {nd:>7}  {'0..' + str(10 ** nd - 1):>14}  {c:>12}/{t}")
    print("\n  the carry was never coded (no % or //) — it is the wrap of a learned cyclic line.")
