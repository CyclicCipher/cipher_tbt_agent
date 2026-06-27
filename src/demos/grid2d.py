"""A 2-D grid — does the column discover a 2-D structure from transitions, with no geometry declared?

A W×H grid world: 4 actions (up/down/left/right); the observation is a SHUFFLED symbol per cell, so the 2-D
structure must be DISCOVERED from transitions, not read off the label. The column does this the universal
way — the SR-eigenvector frame of the transition graph (the 2-D structure surfaces as the grid's 2-D Fourier
modes) — then reads each neighbor off per-relation operators. Test: after learning, predict the neighbor
reached by each action from each cell. The same mechanism handles the 1-D line, the ring, and the tree.

Run:  python -m demos.grid2d      (run from src/ with PYTHONPATH=src)
"""

from __future__ import annotations

import os
import random
import sys


from tbt.column_learner import ColumnLearner as Agent                                       # noqa: E402
from tbt.env import Environment, Step                             # noqa: E402

_MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]                       # up, down, left, right


class Grid2D(Environment):
    def __init__(self, W: int = 6, H: int = 5, seed: int = 0):
        self.W, self.H = W, H
        cells = [(x, y) for y in range(H) for x in range(W)]
        syms = list(range(W * H))
        random.Random(seed).shuffle(syms)
        self.sym = {c: syms[i] for i, c in enumerate(cells)}     # cell -> shuffled symbol
        self.cell_of = {v: c for c, v in self.sym.items()}
        self.x = self.y = 0

    def reset(self):
        self.x = self.y = 0
        return self.sym[(0, 0)]

    def step(self, action: int) -> Step:
        dx, dy = _MOVES[action]
        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx < self.W and 0 <= ny < self.H:
            self.x, self.y = nx, ny
        return Step(self.sym[(self.x, self.y)], 0.0, False)

    @property
    def actions(self):
        return [0, 1, 2, 3]

    def neighbor(self, sym, action):
        x, y = self.cell_of[sym]
        dx, dy = _MOVES[action]
        nx, ny = x + dx, y + dy
        return self.sym[(nx, ny)] if (0 <= nx < self.W and 0 <= ny < self.H) else None


def run(W=6, H=5, steps=None, seed=0):
    env = Grid2D(W, H, seed)
    steps = steps or 200 * W * H
    agent = Agent(n_symbols=W * H, torus=16, seed=seed).explore_and_learn(env, steps=steps, seed=seed)
    correct = total = 0
    for sym in range(W * H):
        if sym not in agent.loc:
            continue
        for a in range(4):
            truth = env.neighbor(sym, a)
            if truth is None:
                continue                                          # a wall — no edge to predict
            correct += int(agent.predict(sym, a) == truth)
            total += 1
    return correct, total, len(agent.loc), W * H


if __name__ == "__main__":
    print("a 2-D grid — discover the 2-D structure from transitions, predict neighbours:\n")
    print(f"  {'grid':>7}  {'neighbor prediction':>20}  {'placed':>9}")
    for W, H in [(5, 5), (6, 5), (8, 6)]:
        c, t, placed, n = run(W, H)
        print(f"  {W}x{H:<4}  {f'{c}/{t}':>20}  {placed:>4}/{n}")
    print("\n  the 2-D geometry was discovered (SR eigenvectors of the transition graph), not declared.")
