"""A binary tree — a NON-METRIC structure, handled by the SAME mechanism as the line / ring / 2-D grid.

Heap layout: node i has children 2i+1, 2i+2 and parent (i−1)//2. Actions: to-parent, to-left-child,
to-right-child. 'to-parent' is the inverse of BOTH child moves, and a child move is a DIFFERENT displacement
at every node — so NO consistent set of axis displacements embeds a tree on a grid (a metric embedding must
fail here). The SR-eigenvector frame has no such restriction: a complete eigenbasis gives every node a
distinct near-orthonormal code (mirror-image nodes are separated by the antisymmetric modes), and the
per-relation operators read each neighbor off directly — so the tree needs no special case.

Run:  python -m demos.tree      (run from src/ with PYTHONPATH=src)
"""

from __future__ import annotations

import os
import random
import sys


from tbt.agent import Agent                                       # noqa: E402
from tbt.env import Environment, Step                             # noqa: E402


class BinaryTree(Environment):
    def __init__(self, depth: int = 4, seed: int = 0):
        self.n = 2 ** depth - 1
        syms = list(range(self.n))
        random.Random(seed).shuffle(syms)
        self.sym = syms                                          # node index -> shuffled symbol
        self.node_of = {v: i for i, v in enumerate(syms)}
        self.node = 0

    def reset(self):
        self.node = 0
        return self.sym[0]

    def step(self, action: int) -> Step:
        i = self.node
        nxt = {0: (i - 1) // 2 if i > 0 else i,                 # to-parent
               1: 2 * i + 1 if 2 * i + 1 < self.n else i,        # to-left-child
               2: 2 * i + 2 if 2 * i + 2 < self.n else i}[action]
        self.node = nxt
        return Step(self.sym[self.node], 0.0, False)

    @property
    def actions(self):
        return [0, 1, 2]

    def neighbor(self, sym, action):
        i = self.node_of[sym]
        j = {0: (i - 1) // 2 if i > 0 else None,
             1: 2 * i + 1 if 2 * i + 1 < self.n else None,
             2: 2 * i + 2 if 2 * i + 2 < self.n else None}[action]
        return self.sym[j] if j is not None else None


def run(depth=4, steps=None, seed=0):
    env = BinaryTree(depth, seed)
    steps = steps or 400 * env.n
    agent = Agent(n_symbols=env.n, torus=16, seed=seed).explore_and_learn(env, steps=steps, seed=seed)
    correct = total = 0
    for sym in range(env.n):
        if sym not in agent.loc:
            continue
        for a in range(3):
            truth = env.neighbor(sym, a)
            if truth is None:
                continue
            correct += int(agent.predict(sym, a) == truth)
            total += 1
    return correct, total, len(agent.loc), env.n


if __name__ == "__main__":
    print("a binary tree (non-metric) — handled by the SR-eigenvector frame, same as line / ring / 2-D grid.\n")
    print(f"  {'depth':>6}  {'nodes':>6}  {'neighbor prediction':>20}  {'placed':>9}")
    for d in (3, 4, 5):
        c, t, placed, n = run(d)
        print(f"  {d:>6}  {n:>6}  {f'{c}/{t}':>20}  {placed:>4}/{n}")
    print("\n  perfect neighbor prediction on a non-metric tree: per-relation operators on the SR frame need no")
    print("  metric embedding, so there is no special case for trees.")
