"""Coupled-carry disentanglement — the SEMIDIRECT-product case (place value with carry).

disentangle.py handled the clean DIRECT product (independent factors, transverse orbit-partitions). Place
value is COUPLED: units x tens, but carry ties them — at the units wrap (b-1 -> 0) the tens also increments.
So the orbit test no longer reports a clean product, and THAT is the signal: the transverse test FLAGS the
coupling and localizes it to the carry action.

Setup: 2-digit base-b numbers as a SINGLE cyclic structure 0..b^2-1, two actions — +1 (units, carries) and
+b (tens, clean). Note 0..b^2-1 is the cyclic group Z_{b^2}, which is NOT Z_b x Z_b (gcd(b,b)!=1): place
value is a non-split group extension, and the carry is exactly the non-splitting (the cocycle).

Result (this file):
  - +b is clean: its orbits partition into b units-classes -> a clean factor of size b.
  - +1 is coupled: its orbit is the WHOLE b^2-cycle (the carry spirals through tens), so discover_factors
    returns a TRIVIAL second factor (n=1) and is_product = FALSE (vs the torus's TRUE) — the coupling is
    flagged and localized to +1.
  - The carry is SPARSE: b of the b^2 +1-transitions (one per units-wrap). EXTRACTING it from data alone is
    the HOLONOMY of the +1-connection around the units-cycle (the extension's cocycle) — the next build.

Run:  python -m precursor.coupled      (from experiments/RecurrentWorldModel/)
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tbt.factorize import discover_factors                   # noqa: E402

from precursor.disentangle import torus_graph                # noqa: E402


def coupled_graph(b, seed=0):
    """2-digit base-b numbers (0..b^2-1) as shuffled symbols; actions 0 = +1 (units, carries), 1 = +b (tens)."""
    n = b * b
    syms = list(range(n))
    random.Random(seed).shuffle(syms)
    sym = {i: syms[i] for i in range(n)}
    graph = {}
    for i in range(n):
        graph.setdefault(sym[i], {})[0] = sym[(i + 1) % n]   # +1: units, carries at i % b == b-1
        graph.setdefault(sym[i], {})[1] = sym[(i + b) % n]   # +b: tens
    carries = sum(1 for i in range(n) if i % b == b - 1)     # ground-truth carries (units wraps) — what's to be found
    return graph, carries


def run(b, seed=0):
    graph, carries = coupled_graph(b, seed)
    factors, is_product = discover_factors(graph, actions=[0, 1])
    return factors, is_product, carries, b * b


if __name__ == "__main__":
    print("coupled-carry disentanglement — place value (units x tens) with carry = a SEMIDIRECT product.\n")
    print(f"  {'base':>5}  {'states':>7}  {'discover_factors':>18}  {'direct-product':>14}  {'carries (sparse)':>16}")
    for b in (3, 4, 10):
        factors, is_product, carries, n = run(b)
        sizes = " x ".join(str(f["n"]) for f in factors)
        print(f"  {b:>5}  {n:>7}  {sizes:>18}  {str(is_product):>14}  {f'{carries}/{n}':>16}")
    _, clean = discover_factors(torus_graph(10), actions=[0, 1, 2, 3])
    print(f"\n  contrast — clean 2-D torus: direct-product={clean}  (a true product; place value above is not)")
    print("\n  the transverse test FLAGS the coupling: place value gives a TRIVIAL second factor (n=1) and")
    print("  is_product=FALSE, because +1's orbit is the whole cycle (carry spirals through tens) — Z_{b^2} is")
    print("  NOT Z_b x Z_b. +b still gives a clean units factor (size b); the carry is localized to +1 and is")
    print("  sparse (b of b^2). Extracting it from data = the HOLONOMY of +1 around the units-cycle (next build).")
