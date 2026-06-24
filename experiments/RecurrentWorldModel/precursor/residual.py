"""Recursive residual modelling, tested on STRUCTURALLY DIFFERENT problems with the SAME loop (RESEARCH.md R7).

The point is generality: one mechanism (model → residual → re-model the residual → recurse) should recover
carry, context-dependence, feature-triggered exceptions, AND refuse to memorise incompressible noise — without
a line of structure-specific code. The coordinates here are what disentanglement (disentangle.py) provides;
the residual loop learns the corrections on top.

  carry (2- and 3-digit) : a CROSS-coordinate coupling at a boundary, with NESTED corrections (carry, double
                           carry, ...) — tests the recursion.
  context-dependence     : the rule MAGNITUDE depends on a context coordinate (F_τ(C)).
  feature exceptions     : a feature value triggers a different move.
  random exceptions      : exceptions that share NO feature — must be REFUSED (the MDL stop), not memorised.

Run:  python -m precursor.residual      (from experiments/RecurrentWorldModel/)
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tbt.factorize import cyclic_coords                       # noqa: E402
from tbt.residual import predict, recursive_residual         # noqa: E402


def carry(b, d):
    """d-digit base-b numbers; coords = the digits (units, tens, ...); action = +1 successor (carries)."""
    n = b ** d
    coords = {x: tuple((x // b ** i) % b for i in range(d)) for x in range(n)}
    transitions = [(x, "succ", (x + 1) % n) for x in range(n)]
    return coords, transitions, ["succ"]


def context(P):
    """A RING whose STEP SIZE depends on a context coordinate: ctx=0 -> +1, ctx=1 -> +2 (wraps need ranges)."""
    states = [(p, c) for p in range(P) for c in (0, 1)]
    coords = {s: s for s in states}
    transitions = [((p, c), "step", ((p + (2 if c else 1)) % P, c)) for (p, c) in states]
    return coords, transitions, ["step"]


def exceptions(P):
    """A RING where one feature value (kind=2) triggers a jump (+3) instead of +1 (wraps need ranges)."""
    states = [(p, k) for p in range(P) for k in (0, 1, 2)]
    coords = {s: s for s in states}
    transitions = [((p, k), "step", ((p + (3 if k == 2 else 1)) % P, k)) for (p, k) in states]
    return coords, transitions, ["step"]


def random_exceptions(P, n_exc, seed=0):
    """A line where a RANDOM subset jumps (+5) — the exceptions all share the same delta but NO coordinate
    feature, so there is no compressing rule: the loop must REFUSE them (model the rest), not memorise."""
    coords = {p: (p,) for p in range(P)}
    exc = set(random.Random(seed).sample(range(P - 5), n_exc))   # p+5 stays in range → all jumps are +5
    transitions = [(p, "step", p + 5) if p in exc else (p, "step", p + 1) for p in range(P - 1)]
    return coords, transitions, ["step"]


def evaluate(coords, transitions, actions):
    rules = recursive_residual(coords, transitions, actions)
    correct = sum(predict(rules, coords, s, a) == coords[sn] for (s, a, sn) in transitions)
    nrules = sum(len(dl) for dl in rules.values())
    return correct, len(transitions), nrules, rules


def end_to_end_carry(b=10, d=2, seed=0):
    """Edge 2: from RAW shuffled-symbol transitions with unlabelled actions {+1 (succ), +b (factor)} — NO
    coordinates given — discover the coordinates (cyclic_coords) THEN the carry corrections (recursive_residual)."""
    n = b ** d
    syms = list(range(n))
    random.Random(seed).shuffle(syms)
    sym = {i: syms[i] for i in range(n)}
    graph = {}
    for i in range(n):
        graph.setdefault(sym[i], {})[0] = sym[(i + 1) % n]       # +1 succ (carries)
        graph.setdefault(sym[i], {})[1] = sym[(i + b) % n]       # +b factor (clean)
    coords, base = cyclic_coords(graph, succ=0, factor=1)        # disentangle -> coordinates (base discovered)
    transitions = [(sym[i], 0, sym[(i + 1) % n]) for i in range(n)]
    rules = recursive_residual(coords, transitions, [0])         # residual -> the carry corrections
    correct = sum(predict(rules, coords, s, a) == coords[sn] for (s, a, sn) in transitions)
    return correct, len(transitions), base


if __name__ == "__main__":
    print("recursive residual modelling — ONE loop, structurally different problems (RESEARCH.md R7):\n")
    print(f"  {'structure':>22}  {'states':>7}  {'predicted':>11}  {'rules':>6}")
    cases = [
        ("carry  (2-digit base-10)", *carry(10, 2)),
        ("carry  (3-digit base-10)", *carry(10, 3)),
        ("context-dependence",       *context(16)),
        ("feature exceptions",       *exceptions(16)),
        ("random exceptions (noise)", *random_exceptions(50, 8)),
    ]
    carry_rules = None
    for name, coords, transitions, actions in cases:
        c, t, nr, rules = evaluate(coords, transitions, actions)
        if name.startswith("carry  (2"):
            carry_rules = rules["succ"]
        print(f"  {name:>22}  {len(coords):>7}  {f'{c}/{t}':>11}  {nr:>6}")
    print("\n  the carry rules it discovered (no holonomy, no place-value given — just residual peeling):")
    for pred, desc, delta in carry_rules:
        print(f"    if {desc:>14}:  digit-delta {delta}")
    print("\n  ONE loop recovers carry (nested), context, and feature exceptions at 100%, and REFUSES the random")
    print("  exceptions (no shared feature -> no compressing rule -> the MDL stop, not a per-state lookup).")
    print("\n  END-TO-END (Edge 2): from RAW shuffled symbols + unlabelled {+1, +b} actions, NO coordinates given")
    print("  -> discover the coordinates (disentangle) -> discover the carry (residual):")
    for d in (2, 3):
        c, t, base = end_to_end_carry(b=10, d=d)
        print(f"    {d}-digit:  base discovered = {base},  carry predicted from data {c}/{t}")
