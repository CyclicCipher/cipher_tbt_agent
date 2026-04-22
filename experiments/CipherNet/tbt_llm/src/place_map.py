"""PlaceMap — the column's internal model of its environment.

TBT context
-----------
This is the L4 abstraction: a map from reference-frame position (L6 output)
to expected sensory SDR. It implements the column's learned world model —
the claim TBT makes is that every cortical column maintains exactly this
kind of position→feature mapping for its input domain.

Learning rule: union (bitwise OR)
---------------------------------
When the column visits a position, it writes the observed sensor SDR via
bitwise OR into the stored pattern. The union grows monotonically. For a
noiseless maze the union stabilises after one visit. For noisy sensors,
the union covers all observed variants, making the model robust.

This matches MiniColumn.learn_one() in the vision system — the same
biological principle: the stored model is the union of all features ever
observed at that location.

Bayesian belief update
----------------------
The belief distribution P(pos) is maintained here and updated after each
observation. The update rule is:

    belief[p] ∝ belief[p] × match(observed_sdr, p)

where match() = fraction of active observed bits present in stored union.
This is exact Bayesian inference for a discrete uniform prior over positions,
assuming the stored union is the likelihood model.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


class PlaceMap:
    """Column's map: position_key → stored sensor SDR (union model).

    Parameters
    ----------
    n_cells : int
        Total number of open cells in the maze (for coverage denominator).
    """

    def __init__(self, n_cells: int = 0) -> None:
        self.n_cells = n_cells   # 0 = unknown; set externally for coverage()
        self._model:  dict[tuple, np.ndarray] = {}   # pos → union SDR
        self._visits: dict[tuple, int]         = {}   # pos → visit count

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def observe(self, sdr: np.ndarray, pos: tuple) -> None:
        """Write one (sdr, pos) observation — OR into the union."""
        if pos not in self._model:
            self._model[pos] = np.zeros(len(sdr), dtype=np.int8)
        np.bitwise_or(self._model[pos], sdr, out=self._model[pos])
        self._visits[pos] = self._visits.get(pos, 0) + 1

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def predict(self, pos: tuple) -> Optional[np.ndarray]:
        """Return stored SDR at pos, or None if unseen."""
        return self._model.get(pos)

    def match(self, sdr: np.ndarray, pos: tuple) -> float:
        """Fraction of active sdr bits present in stored union at pos.

        Returns 0.0 if pos is unseen or sdr has no active bits.
        Always in [0.0, 1.0].

        Used for Bayesian belief update:
            belief[p] *= place_map.match(observed_sdr, p)
        """
        stored = self._model.get(pos)
        if stored is None:
            return 0.0
        n_active = int(sdr.sum())
        if n_active == 0:
            return 0.0
        return float(np.bitwise_and(sdr, stored).sum()) / n_active

    def prediction_error(self, sdr: np.ndarray, pos: tuple) -> float:
        """1.0 - match(sdr, pos). 0.0 = perfect prediction."""
        return 1.0 - self.match(sdr, pos)

    def match_all(self, sdr: np.ndarray) -> tuple[list[tuple], np.ndarray]:
        """Vectorised match against all known positions.

        Returns
        -------
        positions : list of known position keys (length N)
        scores    : float64 array of shape (N,) — match score per position

        Much faster than calling match() in a loop when N is large (uses
        numpy matrix operations instead of per-element Python iteration).
        Used by _update_belief in SingleColumnBrain for large mazes.
        """
        if not self._model:
            return [], np.empty(0, dtype=np.float64)

        positions = list(self._model.keys())
        n_active  = int(sdr.sum())
        if n_active == 0:
            return positions, np.zeros(len(positions), dtype=np.float64)

        # Stack stored SDRs into (N, D) matrix
        sdr_dim = len(sdr)
        matrix  = np.stack([self._model[p] for p in positions])  # (N, D)
        # AND each row with sdr, sum along D, divide by n_active
        scores  = (matrix & sdr).sum(axis=1).astype(np.float64) / n_active
        return positions, scores

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def coverage(self) -> float:
        """Fraction of total maze cells seen at least once."""
        return len(self._model) / max(1, self.n_cells)

    def known_positions(self) -> list[tuple]:
        return list(self._model.keys())

    # ------------------------------------------------------------------
    # Localisation helper (diagnostics)
    # ------------------------------------------------------------------

    def localize(
        self,
        sdr_history: list[tuple[np.ndarray, tuple]],
    ) -> Optional[tuple]:
        """Estimate current position from recent (sdr, displacement) history.

        Replays each displacement through each known position in the map,
        accumulates match scores for each candidate trajectory, and returns
        the position that best explains the full history.

        sdr_history: list of (observed_sdr, displacement) pairs,
                     oldest first, most recent last.
                     displacement = DELTA[action] that produced this sdr.

        Returns the best-matching current position, or None if the map
        is empty.
        """
        if not self._model or not sdr_history:
            return None

        # Candidate starting positions = all known positions
        candidates: dict[tuple, float] = {
            p: 1.0 for p in self._model
        }

        for sdr, displacement in sdr_history:
            # Advance each candidate by the displacement
            new_candidates: dict[tuple, float] = {}
            for p, score in candidates.items():
                pr, pc = p
                dr, dc = displacement
                next_p = (pr + dr, pc + dc)
                m = self.match(sdr, next_p)
                if m > 0.0:
                    new_candidates[next_p] = score * m
            if not new_candidates:
                # All candidates eliminated — return None (lost)
                return None
            # Renormalise
            total = sum(new_candidates.values())
            candidates = {p: s / total for p, s in new_candidates.items()}

        if not candidates:
            return None
        return max(candidates, key=candidates.__getitem__)
