"""Scaling probe: how does the flat reward/exploration system scale as goal depth grows?

The "context goal" was ONE subgoal (visit switch, then goal) — a 2-level hierarchy. Real long-horizon
tasks are CONJUNCTIVE: reach the goal only after visiting K subgoals (collect K keys, flip K switches).
The flat method represents progress as a context = the SUBSET of subgoals achieved → 2^K contexts. This
probe measures state-space size, completions, and backups (compute) as K grows. Throwaway diagnostic.
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict

from tbt.reward import RewardModel, MOVES


class ConjGrid:
    """Goal pays out only after ALL K switches have been visited (any order)."""

    def __init__(self, N, switches, goal):
        self.N, self.switches, self.goal = N, set(switches), goal
        self.reset()

    def reset(self):
        self.pos, self.visited = (0, 0), frozenset()
        return self.pos, self.visited

    def step(self, a):
        dx, dy = MOVES[a]
        x, y = self.pos
        self.pos = (min(max(x + dx, 0), self.N - 1), min(max(y + dy, 0), self.N - 1))
        if self.pos in self.switches:
            self.visited = self.visited | {self.pos}
        d = 1 if (self.pos == self.goal and len(self.visited) == len(self.switches)) else 0
        return (self.pos, self.visited), d


def conj_transitions(N, switches, goal):
    """World model over (cell, visited-subset) — 2^K subsets times N*N cells."""
    T, preds = {}, defaultdict(list)
    subsets = [frozenset(c) for k in range(len(switches) + 1) for c in itertools.combinations(switches, k)]
    for x in range(N):
        for y in range(N):
            for vis in subsets:
                row = []
                for a in range(4):
                    dx, dy = MOVES[a]
                    nx, ny = min(max(x + dx, 0), N - 1), min(max(y + dy, 0), N - 1)
                    nvis = vis | ({(nx, ny)} if (nx, ny) in switches else set())
                    row.append(((nx, ny), frozenset(nvis)))
                s = ((x, y), vis)
                T[s] = row
                for nxt in row:
                    preds[nxt].append(s)
    return T, preds


def run_conj(N, switches, goal, steps, seed=0):
    rng = random.Random(seed)
    T, preds = conj_transitions(N, switches, goal)
    rm = RewardModel(N, prioritized=True)
    env = ConjGrid(N, switches, goal)
    state = env.reset()
    completions = 0
    for _ in range(steps):
        a = rm.act(state, T, preds, rng)
        state, d = env.step(a)
        rm.observe(state, d)
        if d > 0:
            completions += 1
            state = env.reset()
    return completions, len(T), rm.backups


if __name__ == "__main__":
    N, goal, steps = 7, (6, 6), 3000
    pool = [(0, 6), (6, 0), (3, 3), (5, 1), (1, 5)]                # candidate subgoal cells
    print(f"conjunctive K-subgoal goal, N={N}, {steps} steps — flat prioritized-sweeping reward model\n")
    print(f"  {'K':>2}  {'states':>8}  {'completions':>12}  {'backups':>10}")
    for K in range(0, 5):
        c, ns, b = run_conj(N, pool[:K], goal, steps)
        print(f"  {K:>2}  {ns:>8}  {c:>12}  {b:>10}")
