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
        self._grid_U = None                                   # cached left singular vectors of M (the grid cells)
        self._grid_n = -1                                     # state count the grid cache was built at (recompute when it grows)

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

    def _states_by_index(self):
        states = [None] * len(self.idx)
        for st, i in self.idx.items():
            states[i] = st
        return states

    def value(self, s, reward_map):
        """The SR VALUE V(s) = M[s] · R = Σ_g M[s,g]·R[g] over the (typically SPARSE) rewarding states g -- the
        expected DISCOUNTED FUTURE REWARD (Dayan 1993; Stachenfeld 2017; Gershman 2018) as a few cached lookups: the
        deep multi-step propagation is PRECOMPUTED into the SR row M[s] (the future-occupancy), so there is NO rollout.
        Iterating the NONZERO rewards (not the full row) keeps it O(rewarding states), so it scales as the map grows.
        `reward_map` maps a state symbol to its immediate reward; an unknown state -> 0 (reference_brain_planning)."""
        if s not in self.idx:
            return 0.0
        i = self.idx[s]
        return float(sum(self.M[i, self.idx[g]] * r for g, r in reward_map.items() if r and g in self.idx))

    def values(self, reward_map):
        """V(s) = M·R for EVERY known state at once -- `{state: value}` from a single matrix-vector product (the cheap
        deep value over the whole map; the per-plan form when scoring many states)."""
        if not self.idx:
            return {}
        R = np.array([float(reward_map.get(st, 0.0)) for st in self._states_by_index()])
        V = self.M @ R
        return {st: float(V[i]) for st, i in self.idx.items()}

    def grid(self, k: int = 5, refresh: bool = False):
        """The LEARNED multi-scale GRID CELLS = the top-k left singular vectors of the SR matrix `M` (Stachenfeld: grid
        cells ARE the SR eigenvectors; mesh size ∝ eigenvalue -> a handful of geometric scales, `k≈5`). Returns
        `(k, n)`: each ROW is one grid cell (one eigenvector over the states); they encode the diffusive / bottleneck
        structure (separate clusters -> the eigenoption / vector-nav substrate). Cached, rebuilt when the state set GROWS.

        Uses the full `np.linalg.svd` (DETERMINISTic). NB a truncated `svds` was tried and REVERTED: it is only O(n²k)
        vs O(n³), but (a) `eigenpurpose` CAPS grid use at n≤400 where the full SVD is already fast, so the win is moot,
        and (b) `svds` uses a RANDOM start vector and, on the degenerate singular subspaces common in a small SR, returns
        DIFFERENT individual vectors run-to-run -> a nondeterministic, shifted eigenpurpose (it flaked MultiKey/Tetris).
        The truly-online grid (deferred) must preserve determinism + the exact vectors. See BASAL_GANGLIA_PLAN §6."""
        n = self.M.shape[0]
        if n == 0:
            return np.zeros((0, 0))
        if refresh or self._grid_n != n or self._grid_U is None:
            self._grid_U = np.linalg.svd(self.M)[0]
            self._grid_n = n
        return self._grid_U[:, :min(k, n)].T

    def eigenpurpose(self, visits, scale: float = 1.0, k: int = 5, cap: int = 400):
        """The per-state EIGENPURPOSE intrinsic reward from the GRID (top-k SR eigenvectors): high at the UNDER-visited
        extreme of the eigenstructure (the frontier / bottleneck), oriented by anti-correlation with `visits` (a
        state->count map) and normalized to [0, 1] x `scale`. The DIRECTED-exploration drive out of a reward-less,
        locally-exhausted pocket (the EFE dead-zone; reference_eigenoptions_subgoals). `{}` for a graph too small (no
        structure) or too large (SVD is O(n^3); Oja/Sanger streaming is the scale fix). Was `agent._eigenpurpose`."""
        n = self.M.shape[0]
        if n < 3 or n > cap:
            return {}
        inv = {i: s for s, i in self.idx.items()}
        vis = np.array([visits.get(inv[i], 0) for i in range(n)], dtype=float)
        acc = np.zeros(n)
        for e in self.grid(k):                                            # each grid cell (eigenvector over states)
            if e.std() < 1e-9:
                continue
            if vis.std() > 0 and np.corrcoef(e, vis)[0, 1] > 0:          # orient: HIGH value = UNDER-visited (the frontier extreme)
                e = -e
            e = e - e.min()
            if e.max() > 0:
                acc += e / e.max()
        if acc.max() > 0:
            acc /= acc.max()                                             # normalize the eigenpurpose to [0, 1]
        return {inv[i]: scale * float(acc[i]) for i in range(n)}

    def grid_code(self, s, k: int = 5):
        """State `s`'s GRID-CELL CODE -- its coordinates in the top-k SR eigenbasis (the grid-cell activations at `s`).
        The location signal L4 binds features to and L5 path-integrates; `None` for an unknown state."""
        if s not in self.idx:
            return None
        return self.grid(k)[:, self.idx[s]]

    def sr(self):
        """The current SR matrix (rows ordered by discovery; `idx` maps symbol -> row)."""
        return self.M


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
