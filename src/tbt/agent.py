"""The agentic wrapper — a thin ENV DRIVER over a column.

The column owns the model: it learns a structure from transitions (`column.observe` / `column.consolidate`),
and predicts from it (`column.predict` / `column.add`). This wrapper only does the part the column should
NOT know about — driving an `Environment` (choosing actions, stepping it) and feeding the observed
transitions to the column. Everything else (the SR-eigenvector frame discovery, L5 operators, L4/L23
content) lives in the column, beside `learn_domain`, so there is one place that owns "learn a model".

Attribute access delegates to the column, so `agent.loc`, `agent.place`, `agent.predict`, `agent.add`,
`agent.graph`, `agent.rel` all reach the column's learned structure.
"""

from __future__ import annotations

import random

from .column import CorticalColumn


class Agent:
    def __init__(self, n_symbols: int, torus: int = 16, scales=(11, 13, 17), seed: int = 0):
        self.col = CorticalColumn(n_entities=n_symbols, torus_size=torus, scales=scales, place_k=1, seed=seed)

    def explore_and_learn(self, env, steps: int = 600, seed: int = 0):
        """Drive the Environment; feed every transition to the column; consolidate the discovered structure."""
        rng = random.Random(seed)
        s = env.reset()
        for _ in range(steps):
            a = rng.choice(env.actions)
            s2 = env.step(a).observation
            self.col.observe(s, a, s2)                          # the column learns the structure
            s = s2
        self.col.consolidate()
        return self

    def __getattr__(self, name):
        if name == "col":                                      # avoid recursion before col is set
            raise AttributeError(name)
        return getattr(self.col, name)                         # delegate learned-structure access to the column
