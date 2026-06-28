"""L6 — the ONLINE successor-representation location code (TD-learned, no eigendecomposition).

The reference frame learned the way a brain learns it: the successor representation itself, by temporal difference
(Dayan 1993), NOT by a batch eigendecomposition of the whole transition graph. Per observed transition s -> s':

    M[s] <- M[s] + alpha * ( e_s + gamma * M[s'] - M[s] )

The fixed point is the exact SR, M = (I - gamma*T)^-1 (the Bellman equation M = I + gamma*T*M); the update reaches it
incrementally at O(visited states) per step, with no recompute -- the online replacement for `column._sr_frame` /
`consolidate`. The SR ROWS are place-cell-like codes that already encode reachability / topology (Stachenfeld 2017:
place cells ~ SR rows; grid cells ~ its eigenvectors, which a Hebbian layer can extract online later if the grid's
multi-scale / vector-navigation benefits are needed). States are discovered on the fly: a never-seen symbol gets a
fresh zero row/column. Pure numpy.

This is the metric/eigendecomposition-free L6: cheap enough to run every step on a live game, and biologically faithful
(no organism runs a batch eigh). It pairs with factored state (a small graph; the sensor's job) and the L5 operators
(action-conditioned displacement, learned separately and likewise online).
"""

from __future__ import annotations

import numpy as np


class OnlineSR:
    """The online successor representation over discovered states. `observe(s, s2)` does one TD-SR update; `code(s)`
    returns state `s`'s place code (its normalized SR row). `gamma` = the SR horizon; `alpha` = the TD step size
    (alpha=1 with ordered sweeps is exact Gauss-Seidel on the SR Bellman equation; alpha<1 is the online/stochastic
    case). Action is not used here -- the SR is over the state graph; action-conditioning lives in the L5 operators."""

    def __init__(self, gamma: float = 0.95, alpha: float = 0.3):
        self.gamma = gamma
        self.alpha = alpha
        self.idx: dict = {}                                   # state symbol -> row index
        self.M = np.zeros((0, 0))                             # the SR matrix, grown as states are discovered

    def _ensure(self, s) -> int:
        """Index of state `s`, allocating a fresh zero row+column the first time it is seen (online state discovery)."""
        if s not in self.idx:
            n = len(self.idx)
            self.idx[s] = n
            grown = np.zeros((n + 1, n + 1))
            if n:
                grown[:n, :n] = self.M
            self.M = grown
        return self.idx[s]

    def observe(self, s, s2) -> None:
        """One transition s -> s': the TD-SR update of M[s] toward e_s + gamma * M[s']. O(number of states)."""
        i = self._ensure(s)
        j = self._ensure(s2)
        n = len(self.idx)
        e = np.zeros(n)
        e[i] = 1.0
        target = e + self.gamma * self.M[j]
        self.M[i] += self.alpha * (target - self.M[i])

    def code(self, s):
        """The place code for state `s` = its SR row, L2-normalized (nearby-in-graph states get similar codes)."""
        v = self.M[self.idx[s]]
        nrm = float(np.linalg.norm(v))
        return v / nrm if nrm > 0 else v

    def sr(self):
        """The current SR matrix (rows ordered by discovery; `idx` maps symbol -> row)."""
        return self.M
