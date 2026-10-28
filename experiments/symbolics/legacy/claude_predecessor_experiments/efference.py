"""An efference copy for the body's actions (docs/AGENT.md, part 16): what an action is meant to do, kept apart from
what happened.

Each (body kind, action) holds hypotheses over the intended displacement: the 25 translations within 2 cells,
including none, with a uniform prior. A move that happened is evidence for exactly its displacement. A move that
didn't happen is evidence against every displacement that would have landed on a free in-grid cell. For the others
(off the board, or into something) it says nothing about the action: the block is the world's.
"""
import math
from collections import Counter

D = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)]
EPS = 0.01


class Efference:
    def __init__(self, prior=None):
        self.lw, self.moves = {}, Counter()                 # moves: observed moves per (kind, action)
        self.prior = prior                                  # part 19: prior(a) -> {d: p} for a body never seen

    def _w(self, kind, a):
        if (kind, a) not in self.lw:
            pri = self.prior(a) if self.prior else None
            self.lw[(kind, a)] = [math.log(pri[d]) for d in D] if pri else [0.0] * len(D)
        return self.lw[(kind, a)]

    def observe(self, kind, a, t0, e, cells, free):
        """t0: the body's translation before the step; e: the observed element (dx, dy, r); cells: the body's cells at
        translation (0, 0); free(x, y): the cell is on the board and holds nothing (background, no frame)."""
        w = self._w(kind, a)
        moved = (e[0], e[1]) != (0, 0) or e[2] != 0
        self.moves[(kind, a)] += moved
        for i, d in enumerate(D):
            if moved:
                lik = 1 - EPS if e[2] == 0 and (e[0], e[1]) == d else EPS / len(D)
            elif d == (0, 0):
                lik = 1 - EPS
            else:
                dest = [(x + t0[0] + d[0], y + t0[1] + d[1]) for x, y in cells]
                lik = EPS if all(free(x, y) for x, y in dest) else 1 - EPS
            w[i] += math.log(lik)

    def posterior(self, kind, a):
        w = self._w(kind, a)
        m = max(w)
        p = [math.exp(v - m) for v in w]
        z = sum(p)
        return {D[i]: p[i] / z for i in range(len(D))}

    def intended(self, kind, a):
        """The most probable displacement, its probability, and the posterior's entropy in bits."""
        post = self.posterior(kind, a)
        d = max(post, key=post.get)
        return d, post[d], -sum(p * math.log2(p) for p in post.values() if p > 0)
