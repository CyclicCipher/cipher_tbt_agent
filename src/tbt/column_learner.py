"""The precursor COLUMN-LEARNER — drive an Environment, feed every transition to a column, consolidate.

This is NOT the agent (that is `tbt/agent.py` — the thin game-playing env-driver over a planner). It is the
number-line / arithmetic precursors' learning helper: wrap a `CorticalColumn`, learn a structure by random
exploration, then delegate attribute access to the column so a demo reads the learned model
(`.loc` / `.predict` / `.add`). Used only by the precursor demos (numberline, arithmetic, carry, factored,
grid2d, tree); kept separate so `tbt/agent.py` is the ONE agent.
"""

from __future__ import annotations

import random

from tbt.column import CorticalColumn


class ColumnLearner:
    def __init__(self, n_symbols: int, torus: int = 16, scales=(11, 13, 17), seed: int = 0):
        self.col = CorticalColumn(n_entities=n_symbols, torus_size=torus, scales=scales, place_k=1, seed=seed)

    def explore_and_learn(self, env, steps: int = 600, seed: int = 0):
        """Drive the Environment; feed every transition to the column; consolidate the discovered structure."""
        rng = random.Random(seed)
        s = env.reset()
        for _ in range(steps):
            a = rng.choice(env.actions)
            s2 = env.step(a).observation
            self.col.observe(s, a, s2)
            s = s2
        self.col.consolidate()
        return self

    def __getattr__(self, name):
        if name == "col":                                      # avoid recursion before col is set
            raise AttributeError(name)
        return getattr(self.col, name)                         # delegate learned-structure access to the column
