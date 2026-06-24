"""Passive learning — a column learns a WORLD model by WATCHING (no actions), and ANTICIPATES.

Active learning (the agent's own actions, efference-tagged) is qualitatively better — it intervenes, breaks
spurious correlations, and yields the controllable factors (the Higgins orbit route, disentangle.py). But
passive learning gives a capability active cannot: a model of what the world does ON ITS OWN, learned by
observation, so the agent can ANTICIPATE an uncontrolled process — another agent, a cycle, physics it sees
but does not cause.

The efference copy is the switch (von Holst & Mittelstaedt, reafference vs exafference): a SELF-CAUSED
transition (efference-tagged) trains a controllable ACTION operator; an OBSERVED autonomous transition trains
a WORLD operator. Crucially these are the SAME machinery — observe / consolidate / predict / compose — so
passive learning needs NO new mechanism: it is just `observe(s, 'world', s2)` for transitions the agent did
not cause. Anticipation = composing the world operator forward (the same `add` that does arithmetic). Both
pathways coexist in ONE column, told apart only by the action token.

Run:  python -m precursor.passive      (from experiments/RecurrentWorldModel/)
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tbt.column import CorticalColumn                        # noqa: E402

WORLD = "world"                                              # the autonomous-transition token (no efference copy)


def _ring(n, seed):
    """A ring of n nodes with SHUFFLED symbols; returns the symbol list (advances are i -> (i+1) % n)."""
    syms = list(range(n))
    random.Random(seed).shuffle(syms)
    return syms


def passive(n=20, horizon=5, seed=0):
    """WATCH an autonomous ring advance +1 each tick (no actions). Learn it, then anticipate."""
    sym = _ring(n, seed)
    col = CorticalColumn(n_entities=n, place_k=1, seed=seed)
    for i in range(n):
        col.observe(sym[i], WORLD, sym[(i + 1) % n])         # observed, NOT self-caused
    col.consolidate()
    one = sum(col.predict(sym[i], WORLD) == sym[(i + 1) % n] for i in range(n))            # 1-step anticipation
    roll = sum(col.add(sym[i], horizon, WORLD) == sym[(i + horizon) % n] for i in range(n))  # K-step roll-out
    return one, roll, n


def active_and_passive(n=16, seed=0):
    """ONE column: an ACTIVE structure (the agent's own moves on ring A, efference-tagged action 0) PLUS a
    PASSIVE structure (an autonomous ring B the agent only watches, WORLD token). Both learned, both predicted."""
    A, B = _ring(n, seed), _ring(n, seed + 1)
    B = [b + n for b in B]                                   # distinct symbols from A
    col = CorticalColumn(n_entities=2 * n, place_k=1, seed=seed)
    for i in range(n):
        col.observe(A[i], 0, A[(i + 1) % n])                 # ACTIVE: self-caused (efference copy)
        col.observe(B[i], WORLD, B[(i + 1) % n])             # PASSIVE: autonomous (watched)
    col.consolidate()
    act = sum(col.predict(A[i], 0) == A[(i + 1) % n] for i in range(n))         # predict my own action's effect
    wld = sum(col.predict(B[i], WORLD) == B[(i + 1) % n] for i in range(n))     # anticipate the world's evolution
    return act, wld, n


if __name__ == "__main__":
    print("passive learning — a column learns a world model by watching, with no actions, and anticipates.\n")
    print(f"  {'autonomous ring (watched)':>28}  {'1-step':>8}  {'5-step roll-out':>16}")
    for seed in (0, 1, 2):
        one, roll, n = passive(seed=seed)
        print(f"  {f'n={n}, seed {seed}':>28}  {f'{one}/{n}':>8}  {f'{roll}/{n}':>16}")
    print("\n  active + passive in ONE column (efference copy = the switch):")
    for seed in (0, 1, 2):
        act, wld, n = active_and_passive(seed=seed)
        print(f"    seed {seed}:  own action (active) {act}/{n}    world evolution (passive) {wld}/{n}")
    print("\n  passive learning needs NO new machinery — a 'world' operator is learned exactly like an action")
    print("  operator; only the efference copy distinguishes what the agent CONTROLS from what it merely WATCHES.")
    print("  Active alone could never learn the autonomous process (you cannot act on what you do not control).")
