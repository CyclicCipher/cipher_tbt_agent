"""Number-line precursor — architecture doc §14, stage 1.

A 1-D ordered world: position 0..n-1; action 0 = successor (+1), action 1 = predecessor (-1). The
OBSERVATION is a SHUFFLED symbol label, not the position — so the agent must DISCOVER the ordering from
transitions, COLD. No reward (stage 1 = unsupervised structure learning).

This settles the R1 claim empirically: does the column learn the number line from scratch, online, through
the reusable Environment + Agent — with no meta-prior? Success = recovering each symbol's true successor.

Run:  python -m demos.numberline      (run from src/ with PYTHONPATH=src)
"""

from __future__ import annotations

import os
import random
import sys


from tbt.agent import Agent                                       # noqa: E402
from tbt.env import Environment, Step                             # noqa: E402


class NumberLine(Environment):
    def __init__(self, n: int = 12, seed: int = 0, shuffle: bool = True):
        self.n = n
        self.symbol = list(range(n))                              # position -> symbol
        if shuffle:
            random.Random(seed).shuffle(self.symbol)              # order not readable from the label
        self.pos = 0

    def reset(self):
        self.pos = 0
        return self.symbol[self.pos]

    def step(self, action: int) -> Step:
        if action == 0:
            self.pos = min(self.pos + 1, self.n - 1)
        elif action == 1:
            self.pos = max(self.pos - 1, 0)
        return Step(self.symbol[self.pos], 0.0, False)

    @property
    def actions(self):
        return [0, 1]

    def true_successor(self, symbol):
        p = self.symbol.index(symbol)
        return self.symbol[p + 1] if p < self.n - 1 else None


def run_stage1(n: int = 12, steps: int = 600, seed: int = 0):
    env = NumberLine(n=n, seed=seed)
    agent = Agent(n_symbols=n, seed=seed).explore_and_learn(env, steps=steps, seed=seed)
    correct = total = 0
    for p in range(n - 1):
        s = env.symbol[p]
        if s in agent.loc:
            correct += int(agent.predict(s, 0) == env.true_successor(s))   # succ via the column
            total += 1
    return correct, total, len(agent.loc), n


if __name__ == "__main__":
    print("stage 1 — learn the number line cold, through the Environment + Agent (successor prediction):\n")
    for seed in range(5):
        c, t, placed, n = run_stage1(seed=seed)
        print(f"  seed {seed}:  successor {c}/{t}   symbols placed {placed}/{n}")
