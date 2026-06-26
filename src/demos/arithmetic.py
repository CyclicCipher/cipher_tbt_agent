"""Stage 2 — arithmetic on ONE column (architecture doc §14).

Addition = navigation on the learned number line: a + b = apply the successor operator b times from a.
The point is empirical, not theoretical: does ONE column do arithmetic, and where (if anywhere) does it
stop? We do NOT assume it needs a second column — we build it and watch.

Run:  python -m demos.arithmetic      (run from src/ with PYTHONPATH=src)
"""

from __future__ import annotations

import os
import sys


from tbt.agent import Agent                                       # noqa: E402

from demos.numberline import NumberLine                       # noqa: E402


def run(n: int, torus: int = 22, steps=None, seed: int = 0, max_b: int = 9):
    steps = steps or 80 * n
    env = NumberLine(n=n, seed=seed, shuffle=False)               # symbol == number, in order
    agent = Agent(n_symbols=n, torus=torus, seed=seed).explore_and_learn(env, steps=steps, seed=seed)
    correct = total = 0
    for a in range(n):
        if a not in agent.loc:
            continue
        for b in range(min(max_b, n - 1 - a) + 1):                # only a+b that fit on the line
            correct += int(agent.add(a, b) == a + b)
            total += 1
    return correct, total, len(agent.loc)


if __name__ == "__main__":
    print("stage 2 — addition on ONE column, scaling the number line (a + b by navigation):\n")
    print(f"  {'line':>6}  {'addition':>14}  {'placed':>9}")
    for n in (12, 96, 256):
        c, t, placed = run(n=n, torus=22, steps=5 * n * n, max_b=20)
        print(f"  {n:>6}  {f'{c}/{t}':>14}  {placed:>5}/{n}")
    print("\n  no feat_dim wall any more: SPARSE content codes (L4) give capacity >> feat_dim — a single line")
    print("  reaches ~d_mem = 512 (the LOCATION capacity, the place codes), then degrades gracefully. But raw")
    print("  capacity was never the real reason to factor: place value reuses 10 digit-symbols across positions")
    print("  = a LOGARITHMIC representation for ANY magnitude (factored.py), which no single line matches.")
