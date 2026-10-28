"""Overhypotheses (docs/AGENT.md, parts 19 and 22): priors learned across games (Kemp, Perfors & Tenenbaum 2007).
In the arc_agent it holds the prior over goal types (executive.Goals), learned from WINs within a game and from
warm-up play on other games.

A slot is a variable that recurs in every game (what action 2 does; how a one-cell object responds to contact). Each
game contributes one confident value per slot, a draw. The prior for a new game is a Dirichlet-multinomial,
P(v) = (n_v + alpha * b_v) / (N + alpha), with b the flat prior. alpha, how much games vary, is learned: the prediction
is averaged over a grid of alphas, each weighted by how well it predicted the past draws in order.
"""
import math
from collections import Counter

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


class Overhypothesis:
    def __init__(self):
        self.draws = {}                                     # slot -> [values, in the order the games came]

    def add(self, slot, value):
        self.draws.setdefault(slot, []).append(value)

    def alpha_posterior(self, slot, values):
        b = 1.0 / len(values)
        lw = []
        for a in ALPHAS:
            n, w = Counter(), 0.0
            for i, v in enumerate(self.draws.get(slot, [])):
                w += math.log((n[v] + a * b) / (i + a))
                n[v] += 1
            lw.append(w)
        m = max(lw)
        p = [math.exp(w - m) for w in lw]
        z = sum(p)
        return [x / z for x in p]

    def prior(self, slot, values):
        """P(v) for a new game, over the slot's possible values."""
        draws, b = self.draws.get(slot, []), 1.0 / len(values)
        n, N = Counter(draws), len(draws)
        post = self.alpha_posterior(slot, values)
        return {v: sum(w * (n[v] + a * b) / (N + a) for w, a in zip(post, ALPHAS)) for v in values}
