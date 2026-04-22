"""Cortical column hierarchy: MiniColumn → MacroColumn.

THOUSAND BRAINS THEORY — CROSS-FIXATION EVIDENCE ACCUMULATION
-------------------------------------------------------------
Biology processes one image as a temporal SEQUENCE of fixations.  The
column that best explains the ENTIRE sequence wins — not the column with
the best overlap on any single fixation.

Algorithm per image:
  1. begin_image()           — reset evidence accumulators (seeded with boost)
  2. observe(sdr, loc)       — called once per fixation; adds overlap_score
                               to every minicolumn's evidence; temporal bonus
                               added to the previous leader if its prediction
                               matched the current SDR
  3. commit(write)           — winner = argmax(evidence); optionally write
                               all observed (sdr, loc) pairs to winner's model

For supervised layers (IT):
  commit_supervised(label, write) — force minicolumn `label` to win.

FEATURE REPRESENTATION: SDRs
-----------------------------
Features are np.ndarray(dtype=int8, shape=(n_bits,)) — binary Sparse
Distributed Representations where ~top_k bits are active.

MiniColumn model at each location = union (bitwise OR) of all training SDRs
observed there.  This naturally covers within-class variation: if digit "3"
produces orientation pattern (0,1,2) in most examples and (0,1,3) in some,
the union at that location has bits {0,1,2,3} active.  A test observation
with pattern (0,1,3) scores overlap = 2/2 = 1.0 against the union, vs. 0.0
with string exact-match.

Overlap score = (test_sdr & union).sum() / test_sdr.sum()
             = fraction of test feature's active bits present in model union.

HIERARCHY
---------
V1  — sensor input, unsupervised (commit)
IT  — V1 population input, supervised (commit_supervised; mini idx = class)
"""
from __future__ import annotations

import numpy as np

from reference_frames import ReferenceFrame
from cortical_message import CorticalMessage


# ---------------------------------------------------------------------------
# MiniColumn
# ---------------------------------------------------------------------------

class MiniColumn:
    """One minicolumn: stores exactly one object model as SDR unions.

    model:      location_key → union SDR (np.ndarray int8, shape (n_bits,))
    loc_total:  location_key → number of training observations written there
    _n_wins:    cumulative commit-wins (drives homeostatic boost decay)

    UNION MODEL
    -----------
    learn_one(sdr, loc) performs  model[loc] |= sdr  (bitwise OR).
    The union grows monotonically: once a bit is set at a location it stays.
    This is the correct implementation of the SDR model: the stored pattern
    represents every orientation that has ever been observed at that location
    across all training examples of this class.

    OVERLAP SCORE
    -------------
    overlap_score(sdr, loc) = max over nearby stored locations of:
        (sdr & union).sum() / sdr.sum()  ×  spatial_similarity(loc, stored)

    Result is always in [0.0, 1.0].  A mismatch contributes 0.0 — never
    negative.  This matches Monty's feature evidence model exactly: evidence
    is clipped to [0, ∞) so poor matches simply add nothing rather than
    subtracting.  "Hypothesis intersection" emerges naturally from
    accumulation: the correct class scores positively at most locations while
    wrong classes score 0 at most locations, so the gap grows over fixations.

    The ±1 neighbour spatial smoothing absorbs centroid estimation errors
    (sub-pixel variation between training instances).
    """

    __slots__ = ('_model', '_loc_total', '_n_wins')

    def __init__(self) -> None:
        self._model:     dict[tuple, np.ndarray] = {}
        self._loc_total: dict[tuple, int]         = {}
        self._n_wins:    int                      = 0

    # ------------------------------------------------------------------
    # Boost (homeostatic, decays per win not per observation)
    # ------------------------------------------------------------------

    @property
    def _boost(self) -> float:
        return 0.25 / (self._n_wins + 1)

    # ------------------------------------------------------------------
    # Spatial similarity
    # ------------------------------------------------------------------

    _NEIGHBOURS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))

    @staticmethod
    def _loc_sim(loc_a: tuple, loc_b: tuple) -> float:
        """Spatial similarity between two location keys.

        2-component grid keys: Chebyshev-distance decay 1/(1+dist).
          dist=0 → 1.0,  dist=1 (≈±2 px) → 0.5,  dist=2 → 0.33
        Other keys: fraction of matching components.
        """
        if len(loc_a) != len(loc_b) or not loc_a:
            return 0.0
        if len(loc_a) == 2:
            dist = max(abs(loc_a[0] - loc_b[0]), abs(loc_a[1] - loc_b[1]))
            return 1.0 / (1 + dist)
        return sum(a == b for a, b in zip(loc_a, loc_b)) / len(loc_a)

    # ------------------------------------------------------------------
    # Overlap (feedforward match quality)
    # ------------------------------------------------------------------

    def overlap_score(self, sdr: np.ndarray, loc: tuple) -> float:
        """Fraction of sdr's active bits present in the stored union at loc.

        Iterates all stored locations, weights by spatial similarity, returns
        the best weighted overlap.  Result always in [0.0, 1.0].

        Returns 0.0 when:
          - sdr has no active bits (blank — should not reach here)
          - no stored location is spatially nearby
          - the feature has never been seen near this location
        """
        n_active = int(sdr.sum())
        if n_active == 0:
            return 0.0
        best = 0.0
        for stored_loc, union in self._model.items():
            sim = self._loc_sim(loc, stored_loc)
            if sim == 0.0:
                continue
            score = int(np.bitwise_and(sdr, union).sum()) / n_active * sim
            if score > best:
                best = score
        return best

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn_one(self, sdr: np.ndarray, loc: tuple) -> None:
        """Write one (sdr, loc) observation — OR into the union."""
        if loc not in self._model:
            self._model[loc] = np.zeros(len(sdr), dtype=np.int8)
        np.bitwise_or(self._model[loc], sdr, out=self._model[loc])
        self._loc_total[loc] = self._loc_total.get(loc, 0) + 1

    def unlearn_one(self, sdr: np.ndarray, loc: tuple) -> None:
        """CHL contrastive step.

        The union model is monotonically increasing (OR only).  Exact
        reversal of a union step would require storing all individual SDRs
        and recomputing the union minus the unlearned sample — deferred.
        For the current supervised-IT architecture this is never called.
        """
        pass

    # ------------------------------------------------------------------
    # Prediction (group law — used for temporal bonus)
    # ------------------------------------------------------------------

    def predict(self, location: tuple, displacement: tuple) -> np.ndarray | None:
        """Predicted SDR at location + displacement.

        Returns the stored union SDR at the target location (with ±1
        neighbour search for grid keys), or None if no model exists there.
        Only meaningful for 2-component grid location keys.
        """
        if len(location) != 2:
            return None
        tx = location[0] + displacement[0]
        ty = location[1] + displacement[1]
        for dx, dy in self._NEIGHBOURS:
            result = self._model.get((tx + dx, ty + dy))
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def n_locations(self) -> int:
        return len(self._model)


# ---------------------------------------------------------------------------
# MacroColumn
# ---------------------------------------------------------------------------

class MacroColumn:
    """One macrocolumn: N minicolumns + TBT cross-fixation evidence.

    Usage per image:
        col.begin_image()
        for fix in fixations:
            col.observe(sdr, loc)          # sensor layers
            # or
            col.observe_multi([(sdr, loc), ...])   # higher layers
        winner = col.commit(write=True)            # unsupervised
        # or
        winner = col.commit_supervised(label, write=True)   # IT layer

    Features are np.ndarray(dtype=int8) SDRs from the encoder.
    None features (blank patches) must be filtered before calling observe().
    """

    TEMPORAL_BONUS:   float = 1.0    # correct SDR prediction → lock leader
    CONTINUITY_BONUS: float = 0.30   # partial/no prediction → keep leader primed

    # Minimum SDR overlap fraction to count as a correct prediction.
    PREDICTION_MATCH_THRESHOLD: float = 0.5

    def __init__(self, frame: ReferenceFrame, n_mini: int = 10,
                 miss_penalty: float = 0.0) -> None:
        self.frame        = frame
        self.N_MINI       = n_mini
        # miss_penalty is retained for API compat but no longer used —
        # natural zero SDR overlap at known locations is the elimination signal.
        self.minicolumns: list[MiniColumn] = [MiniColumn() for _ in range(n_mini)]

        self._active:        list[int]   = []
        self._active_scores: list[float] = []

        # Per-image state (reset by begin_image)
        self._evidence:    list[float]                  = []
        self._obs:         list[tuple[np.ndarray, tuple]] = []
        self._prev_loc:    tuple | None                  = None
        self._prev_leader: int   | None                  = None

    # ------------------------------------------------------------------
    # Per-image API
    # ------------------------------------------------------------------

    def begin_image(self) -> None:
        """Reset accumulators.  Evidence seeds with homeostatic boost."""
        self._evidence    = [mc._boost for mc in self.minicolumns]
        self._obs         = []
        self._prev_loc    = None
        self._prev_leader = None

    def observe(self, sdr: np.ndarray, loc: tuple) -> None:
        """Accumulate evidence from ONE (sdr, loc) observation.

        For sensor layers (V1): called once per fixation with the HOG SDR
        and the object-relative retinal location.

        1. Add overlap_score(sdr, loc) to every minicolumn's evidence.
        2. Temporal bonus for the previous-fixation leader:
             TEMPORAL_BONUS    if its prediction overlaps sdr ≥ threshold
             CONTINUITY_BONUS  otherwise (keeps leader primed, cold-start)
        3. Update prev_leader and prev_loc.
        """
        for i, mc in enumerate(self.minicolumns):
            self._evidence[i] += mc.overlap_score(sdr, loc)

        if self._prev_leader is not None and self._prev_loc is not None:
            disp     = tuple(int(l) - int(p)
                             for l, p in zip(loc, self._prev_loc))
            mc_prev  = self.minicolumns[self._prev_leader]
            predicted = mc_prev.predict(self._prev_loc, disp)
            if predicted is not None:
                n_active = int(sdr.sum())
                if n_active > 0:
                    match = int(np.bitwise_and(predicted, sdr).sum()) / n_active
                    if match >= self.PREDICTION_MATCH_THRESHOLD:
                        self._evidence[self._prev_leader] += self.TEMPORAL_BONUS
                    else:
                        self._evidence[self._prev_leader] += self.CONTINUITY_BONUS

        self._obs.append((sdr, loc))
        self._prev_loc    = loc
        self._prev_leader = max(range(self.N_MINI),
                                key=lambda i: self._evidence[i])

    def observe_multi(self, observations: list[tuple[np.ndarray, tuple]]) -> None:
        """Accumulate evidence from MULTIPLE independent (sdr, loc) pairs.

        For higher layers (IT): called once per fixation with one entry per
        lower-layer column in this column's receptive field (None entries
        already filtered out before calling observe_multi()).

        Evidence is normalised by RF size so each fixation contributes the
        same total regardless of how many active slots remain after blank
        suppression.
        """
        n = len(observations)
        if n == 0:
            return
        scale = 1.0 / n

        for sdr, loc in observations:
            for i, mc in enumerate(self.minicolumns):
                self._evidence[i] += mc.overlap_score(sdr, loc) * scale

        # Temporal bonus: check how many of current observations the
        # pre-fixation leader predicted correctly (displacement = (0,0)
        # since RF-local positions are constant across fixations).
        if self._prev_leader is not None:
            mc_prev = self.minicolumns[self._prev_leader]
            n_correct = 0
            n_predicted = 0
            for sdr, loc in observations:
                predicted = mc_prev.predict(loc, (0, 0))
                if predicted is None:
                    continue
                n_predicted += 1
                n_active = int(sdr.sum())
                if n_active > 0:
                    match = (int(np.bitwise_and(predicted, sdr).sum())
                             / n_active)
                    if match >= self.PREDICTION_MATCH_THRESHOLD:
                        n_correct += 1
            if n_correct > 0:
                self._evidence[self._prev_leader] += (
                    self.TEMPORAL_BONUS * n_correct / n)
            elif n_predicted > 0:
                self._evidence[self._prev_leader] += self.CONTINUITY_BONUS

        self._obs.extend(observations)
        self._prev_leader = max(range(self.N_MINI),
                                key=lambda i: self._evidence[i])

    def tentative_winner(self) -> int:
        """Current evidence leader (before image ends)."""
        if not self._evidence:
            return 0
        return max(range(self.N_MINI), key=lambda i: self._evidence[i])

    def commit(self, write: bool = True) -> int:
        """Finalise winner.  Optionally write all observations to its model."""
        winner = self.tentative_winner()
        if write:
            for sdr, loc in self._obs:
                self.minicolumns[winner].learn_one(sdr, loc)
            self.minicolumns[winner]._n_wins += 1
        ev = self._evidence[winner] if self._evidence else 0.0
        self._active        = [winner]
        self._active_scores = [ev]
        return winner

    def commit_supervised(self, label: int, write: bool = True) -> int:
        """Force minicolumn `label` to win (supervised IT layer)."""
        if write:
            for sdr, loc in self._obs:
                self.minicolumns[label].learn_one(sdr, loc)
            self.minicolumns[label]._n_wins += 1
        ev = self._evidence[label] if self._evidence else 0.0
        self._active        = [label]
        self._active_scores = [ev]
        return label

    def commit_chl(self, clamped_label: int, write: bool = True) -> int:
        """Contrastive Hebbian Learning commit.

        Clamped phase: write to minicolumn (clamped_label % N_MINI).
        Free-phase correction: unlearn_one is a no-op for union SDR models
        (deferred — see MiniColumn.unlearn_one).
        """
        free_winner    = self.tentative_winner()
        clamped_winner = clamped_label % self.N_MINI
        if write:
            for sdr, loc in self._obs:
                self.minicolumns[clamped_winner].learn_one(sdr, loc)
            self.minicolumns[clamped_winner]._n_wins += 1
            # Contrastive unlearn deferred (union model is write-only for now).
        ev = self._evidence[clamped_winner] if self._evidence else 0.0
        self._active        = [clamped_winner]
        self._active_scores = [ev]
        return clamped_winner

    def receive_feedback_by_index(self, mini_idx: int,
                                   bonus: float) -> None:
        """Top-down prediction: boost evidence for minicolumn mini_idx."""
        if self._evidence and 0 <= mini_idx < self.N_MINI:
            self._evidence[mini_idx] += bonus

    def apply_lateral_input(self, neighbor_winners: list[int],
                             bonus: float) -> None:
        """Add lateral evidence from neighbouring columns' tentative winners."""
        if not self._evidence:
            return
        for winner in neighbor_winners:
            if 0 <= winner < self.N_MINI:
                self._evidence[winner] += bonus

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def sdr(self) -> frozenset[int]:
        return frozenset(self._active)

    def confidence(self) -> float:
        return max(self._active_scores) if self._active_scores else 0.0

    def emit(self) -> CorticalMessage:
        return CorticalMessage(
            location=self.frame.position_key(),
            feature=str(sorted(self._active)),
            confidence=self.confidence(),
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        used       = sum(1 for mc in self.minicolumns if mc.n_locations() > 0)
        total_locs = sum(mc.n_locations() for mc in self.minicolumns)
        return {
            'used_mini':       used,
            'total_locations': total_locs,
            'avg_locations':   total_locs / max(1, used),
            'confidence':      self.confidence(),
        }
