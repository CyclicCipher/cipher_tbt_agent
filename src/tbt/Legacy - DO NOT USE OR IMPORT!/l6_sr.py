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


class SuccessorFeatures:
    """Successor FEATURES over an SDR encoding — the SDR-native replacement for the localist-symbol-indexed `OnlineSR`
    (ARCHITECTURE §2 L6, §10 P4a/P5). The predictive map is linear in an SDR feature `phi = encoder.encode(obs).dense()`:
    `psi(s) = W·phi(s)` estimates the discounted future feature occupancy, learned online by TD (Dayan 1993; Barreto et
    al. 2017; *Neurobiological successor features*, 2021). Because states with OVERLAPPING phi share columns of W,
    value/occupancy GENERALISE across nearby states BEFORE each is individually visited — the a-priori metric
    generalisation the localist `OnlineSR` cannot give (an unvisited symbol has a zero row). Value `V(s) = w·psi(s)`,
    with reward weights `w` s.t. `r(s) ~ w·phi(s)`, learned online. The operator path-integrates `phi` (the SDR), not a
    symbol. Pure numpy."""

    def __init__(self, d: int, gamma: float = 0.95, alpha: float = 0.1, beta: float = 0.5):
        self.d = int(d)
        self.gamma, self.alpha, self.beta = gamma, alpha, beta
        self.W = np.zeros((d, d))                             # psi(s) = W·phi(s): the successor-feature operator
        self.w = np.zeros(d)                                 # reward weights: r(s) ~ w·phi(s)

    def psi(self, phi):
        """The successor features of `phi` = the discounted future feature occupancy."""
        return self.W @ np.asarray(phi, dtype=float)

    def value(self, phi) -> float:
        """V(s) = w·psi(s) = w·(W·phi) — expected discounted future reward, read from the SDR (no rollout)."""
        phi = np.asarray(phi, dtype=float)
        return float(self.w @ (self.W @ phi))

    def observe(self, phi, phi_next, reward: float = 0.0) -> None:
        """One TD update for a transition phi -> phi_next carrying immediate `reward`: regress the reward weights `w`
        (r ~ w·phi) and take an LMS step on the SF operator `W` toward the Bellman target phi + gamma·W·phi_next."""
        phi = np.asarray(phi, dtype=float)
        phi_next = np.asarray(phi_next, dtype=float)
        nrm = float(phi @ phi) + 1e-9
        self.w += self.beta * (reward - self.w @ phi) * phi / nrm            # reward regression
        err = (phi + self.gamma * (self.W @ phi_next)) - self.W @ phi        # SF Bellman TD error
        self.W += self.alpha * np.outer(err, phi) / nrm                      # LMS gradient step


def hex_code(disp, scales=(11, 13, 17), lattice: str = "hex"):
    """The HEX reference frame's INITIAL-STATE descriptor, WITHIN the one L6 module (the collapsed `l6_grid`). A specific
    frame whose geometry is known a-priori -- the spatial column's innate entorhinal grid -- is described here by the MINIMAL
    data + code: the multi-scale plane-wave code at a (relative) displacement `disp=(dx,dy)`, each module = plane waves at
    0/120/240° (hex) with an incommensurate scale (Wei-Fiete), giving a metric, path-integrable code over `lcm(scales)`. It
    is a valid ABELIAN representation of translation (`hex_code(a+b)` = the phase-composition of a and b), so it doubles as a
    code space for operator learning. NB the SR above is the LEARNED substrate that runs live; this is only the OPTIONAL
    metric prior for an a-priori-known frame -- one L6 script, the frame as an initial-state descriptor, not a parallel class."""
    import math
    disp = np.asarray(disp, dtype=float)
    angles = [0.0, 120.0, 240.0] if lattice == "hex" else [0.0, 90.0]
    dirs = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in angles])
    W = np.concatenate([(2.0 * math.pi / s) * dirs for s in scales], axis=0)    # (M, 2) grid frequencies (incommensurate)
    ph = disp @ W.T                                                             # (M,) phase increment per module
    return np.stack([np.cos(ph), np.sin(ph)], axis=-1).reshape(-1)              # (2M,) the hex grid code
