"""Higgins-style disentanglement — discover that an ENTANGLED space factors, from action alone (doc §12.3c).

A 2-D torus N x N, but the agent sees only a SHUFFLED single symbol per cell (the factorization is invisible
in the observation) and 4 actions that are NOT labelled as to axis. From the transition graph it must DISCOVER
that the space factors into two independent rings — which it does the Higgins way (Locatello: you need
action): each action's orbit-partition is computed; actions with the same orbits are the same factor; the two
factors are a direct product iff their partitions are transverse. Then one column models each discovered
factor and the joint dynamics are predicted COMPOSITIONALLY (the action moves one factor, leaves the other).

The payoff (cf. factored.py / arithmetic.py stage-2 wall): two columns of N model the N^2 joint space, so once
N^2 exceeds a column's content capacity (feat_dim) a single holistic column can no longer even represent the
states, while the factored model is unaffected — capacity is the whole reason to disentangle.

Run:  python -m demos.disentangle      (run from src/ with PYTHONPATH=src)
"""

from __future__ import annotations

import os
import random
import sys


from tbt.column import CorticalColumn                        # noqa: E402
from tbt.factorize import discover_factors, factor_graph     # noqa: E402

_MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]                   # actions 0..3 (+x,-x,+y,-y) — UNKNOWN to the agent


def torus_graph(N, seed=0):
    """N x N torus, cells given SHUFFLED symbols; full transition graph {sym: {action: sym}}."""
    cells = [(x, y) for x in range(N) for y in range(N)]
    syms = list(range(N * N))
    random.Random(seed).shuffle(syms)
    sym = {c: syms[i] for i, c in enumerate(cells)}
    graph = {}
    for (x, y), s in sym.items():
        for a, (dx, dy) in enumerate(_MOVES):
            graph.setdefault(s, {})[a] = sym[((x + dx) % N, (y + dy) % N)]
    return graph


def _column(fgraph, n):
    col = CorticalColumn(n_entities=n, place_k=1, seed=0)
    for s, e in fgraph.items():
        for a, s2 in e.items():
            col.observe(s, a, s2)
    col.consolidate()
    return col


def factored_model(graph, factors):
    """One column per discovered factor + the map from per-factor coordinates back to the joint state."""
    cols = [_column(factor_graph(graph, f), f["n"]) for f in factors]
    coord_to_state = {tuple(f["coord"][s] for f in factors): s for s in graph}
    return cols, coord_to_state


def factored_predict(s, a, factors, cols, coord_to_state):
    """Compositional: the factor that owns action `a` advances its coordinate; the others stay fixed."""
    for i, (f, col) in enumerate(zip(factors, cols)):
        if a in f["actions"]:
            moved = col.predict(f["coord"][s], a)
            tup = tuple(moved if j == i else g["coord"][s] for j, g in enumerate(factors))
            return coord_to_state.get(tup)
    return None


def _accuracy(graph, predict):
    correct = total = 0
    for s, e in graph.items():
        for a, s2 in e.items():
            correct += int(predict(s, a) == s2)
            total += 1
    return correct, total


def run(N, seed=0):
    graph = torus_graph(N, seed)
    factors, is_product = discover_factors(graph, actions=[0, 1, 2, 3])
    cols, c2s = factored_model(graph, factors)
    fc, ft = _accuracy(graph, lambda s, a: factored_predict(s, a, factors, cols, c2s))
    hol = _column(graph, N * N)                                    # holistic baseline: one column, all N^2 symbols
    hc, ht = _accuracy(graph, lambda s, a: hol.predict(s, a))
    return factors, is_product, (fc, ft), (hc, ht)


if __name__ == "__main__":
    print("Higgins-style disentanglement — discover the factorization of an entangled torus from action alone.\n")
    for N in (6, 45):
        factors, is_product, (fc, ft), (hc, ht) = run(N)
        sizes = " x ".join(str(f["n"]) for f in factors)
        groups = " | ".join("actions " + str(f["actions"]) for f in factors)
        print(f"  torus {N}x{N}  (joint space {N * N};  sparse codes -> generous single-column capacity)")
        print(f"    discovered {len(factors)} factors  [{sizes}]  direct-product={is_product}   ({groups})")
        print(f"    factored model  (2 columns of {N})     joint-transition prediction: {fc}/{ft}")
        print(f"    holistic model  (1 column of {N * N})  joint-transition prediction: {hc}/{ht}")
        print()
    print("  the factorization is DISCOVERED from action-orbits, not declared (the symbol is shuffled, the")
    print("  actions unlabelled). With sparse coding (the capacity work) a single column now has generous")
    print("  capacity, so BOTH models fit here — but the factored cost is LINEAR (2 columns of N) where the")
    print("  holistic is QUADRATIC (one N^2-state column -> an N^2 x N^2 consolidation), so the holistic blows")
    print("  up in compute and eventually degrades (~N=90), while the factored stays cheap and exact. As in")
    print("  cortex: sparse capacity is generous; factorization is for SCALE (and compositional transfer).")
