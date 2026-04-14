"""Cortical column hierarchy: MiniColumn → MacroColumn.

THOUSAND BRAINS THEORY — CROSS-FIXATION EVIDENCE ACCUMULATION
-------------------------------------------------------------
Biology processes one image as a temporal SEQUENCE of fixations.  The
column that best explains the ENTIRE sequence wins — not the column with
the best overlap on any single fixation.

Algorithm per image:
  1. begin_image()     — reset evidence accumulators (seeded with boost)
  2. observe(feat,loc) — called once per fixation; adds overlap_score to
                         every minicolumn's evidence; temporal bonus added
                         to the previous leader if its prediction matched
  3. commit(write)     — winner = argmax(evidence); optionally write all
                         observed (feat,loc) pairs to winner's model

For supervised layers (IT):
  commit_supervised(label, write) — force minicolumn `label` to win;
  minicolumn index == digit class (label supervision from hippocampus).

TEMPORAL CONTEXT
----------------
Two tiers:
  TEMPORAL_BONUS (1.0) — leader from previous fixation predicted feat
    correctly via group law.  Locks the leader through the saccade.
  CONTINUITY_BONUS (0.30) — leader from previous fixation predicted
    incorrectly (cold start: no model at new location yet).  Keeps it
    primed above the fresh-minicolumn boost (0.25 / 1 = 0.25), preventing
    a different minicolumn from hijacking the sequence before learning begins.

HIERARCHY
---------
V1  — sensor input, unsupervised (commit)
V2  — V1 population input, unsupervised (commit)
IT  — V2 population input, supervised (commit_supervised; mini idx = class)
"""
from __future__ import annotations

from collections import Counter

from reference_frames import ReferenceFrame
from cortical_message import CorticalMessage


# ---------------------------------------------------------------------------
# MiniColumn
# ---------------------------------------------------------------------------

class MiniColumn:
    """One minicolumn: stores exactly one object model.

    model:     location_key → Counter[feature_str]
    loc_total: location_key → total observation count
    _best:     location_key → most-common feature (for predict())
    _n_activations: cumulative commit-wins (drives boost decay)

    SOFT OVERLAP
    ------------
    overlap_score(feat, loc) = P(feat | loc, this minicolumn)
    Checked over the focal location and its ±1 cardinal neighbours to
    absorb centroid variation between training and test instances.
    """

    __slots__ = ('_model', '_loc_total', '_best', '_n_wins')

    def __init__(self) -> None:
        self._model: dict[tuple, Counter] = {}
        self._loc_total: dict[tuple, int] = {}
        self._best: dict[tuple, str] = {}
        self._n_wins: int = 0

    # ------------------------------------------------------------------
    # Boost (homeostatic, decays per WIN not per observation)
    # ------------------------------------------------------------------

    @property
    def _boost(self) -> float:
        return 0.25 / (self._n_wins + 1)

    # ------------------------------------------------------------------
    # Overlap (feedforward match quality)
    # ------------------------------------------------------------------

    _NEIGHBOURS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))

    @staticmethod
    def _loc_sim(loc_a: tuple, loc_b: tuple) -> float:
        """Spatial similarity between two location keys.

        For 2-component grid keys: Chebyshev-distance decay — 1/(1+dist).
          dist=0 (exact) → 1.0
          dist=1 (±1 bin ≈ ±2 px) → 0.5
          dist=2 (±2 bins ≈ ±4 px) → 0.33
        This gives smooth spatial generalisation while keeping the number
        of unique stored keys small (nearby centroids bin together).

        For N-component keys (other frame types): fraction of matching
        components (original fallback).
        """
        if len(loc_a) != len(loc_b) or not loc_a:
            return 0.0
        if len(loc_a) == 2:
            dist = max(abs(loc_a[0] - loc_b[0]), abs(loc_a[1] - loc_b[1]))
            return 1.0 / (1 + dist)
        return sum(a == b for a, b in zip(loc_a, loc_b)) / len(loc_a)

    def overlap_score(self, feat: str, loc: tuple) -> float:
        """Similarity-weighted max P(feat | stored_loc).

        Iterates all stored locations, weights each by _loc_sim, and returns
        the highest weighted probability. Replaces the hardcoded ±1 neighbour
        search: for 2D keys the behaviour is equivalent; for Fourier keys it
        gives automatic smooth generalisation over the component space.
        """
        best = 0.0
        for stored_loc, total in self._loc_total.items():
            if total == 0:
                continue
            sim = self._loc_sim(loc, stored_loc)
            if sim == 0.0:
                continue
            s = self._model[stored_loc].get(feat, 0) / total * sim
            if s > best:
                best = s
        return best

    def overlap_score_signed(self, feat: str, loc: tuple,
                             miss_penalty: float) -> float:
        """Like overlap_score but subtracts miss_penalty when the minicolumn
        has a model at this location yet the feature doesn't match.

        Positive return = the minicolumn expects this feature here.
        Negative return = the minicolumn has a conflicting expectation.
        Zero            = no model at this location (no signal either way).
        """
        best = 0.0
        has_model = False
        for stored_loc, total in self._loc_total.items():
            if total == 0:
                continue
            sim = self._loc_sim(loc, stored_loc)
            if sim == 0.0:
                continue
            has_model = True
            s = self._model[stored_loc].get(feat, 0) / total * sim
            if s > best:
                best = s
        if has_model and best == 0.0:
            return -miss_penalty
        return best

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn_one(self, feat: str, loc: tuple) -> None:
        """Write a single (feat, loc) observation."""
        if loc not in self._model:
            self._model[loc] = Counter()
        self._model[loc][feat] += 1
        self._loc_total[loc] = self._loc_total.get(loc, 0) + 1
        if self._model[loc][feat] >= self._model[loc].get(
                self._best.get(loc, ''), 0):
            self._best[loc] = self._model[loc].most_common(1)[0][0]

    def unlearn_one(self, feat: str, loc: tuple) -> None:
        """Subtract one count — CHL free-phase correction.

        Called on the WTA winner when CHL detects it differs from the
        clamped (correct) minicolumn.  Prevents incorrect associations
        from accumulating after the free phase selects the wrong winner.
        Zero-count entries are removed to keep most_common() clean.
        """
        if loc not in self._model:
            return
        cnt = self._model[loc].get(feat, 0)
        if cnt <= 0:
            return
        if cnt == 1:
            del self._model[loc][feat]
        else:
            self._model[loc][feat] -= 1
        self._loc_total[loc] = max(0, self._loc_total.get(loc, 0) - 1)
        remaining = self._model[loc].most_common(1)
        self._best[loc] = remaining[0][0] if remaining else ''

    # ------------------------------------------------------------------
    # Prediction (group law)
    # ------------------------------------------------------------------

    def predict(self, location: tuple, displacement: tuple) -> str | None:
        """Best feature at location + displacement (with ±1 neighbour search).

        Only defined for 2-component grid keys where displacement arithmetic
        is geometrically meaningful. Returns None for Fourier or other keys
        (triggers CONTINUITY_BONUS instead of TEMPORAL_BONUS in observe()).
        """
        if len(location) != 2:
            return None
        tx = location[0] + displacement[0]
        ty = location[1] + displacement[1]
        for dx, dy in self._NEIGHBOURS:
            result = self._best.get((tx + dx, ty + dy))
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
            col.observe(feat, loc)
        winner = col.commit(write=True)   # unsupervised layers
        # or
        winner = col.commit_supervised(label, write=True)  # IT layer

    The winner is the minicolumn whose evidence (sum of overlap_score over
    all fixations, plus temporal bonuses) is highest.  For supervised IT,
    the label is forced regardless of evidence.
    """

    TEMPORAL_BONUS:    float = 1.0   # correct prediction → big bonus
    CONTINUITY_BONUS:  float = 0.30  # wrong/none prediction → keep leader primed

    def __init__(self, frame: ReferenceFrame, n_mini: int = 10,
                 miss_penalty: float = 0.0) -> None:
        self.frame        = frame
        self.N_MINI       = n_mini
        self.miss_penalty = miss_penalty
        self.minicolumns: list[MiniColumn] = [MiniColumn() for _ in range(n_mini)]

        # Populated by commit() so emit()/sdr()/confidence() keep working.
        self._active:        list[int]   = []
        self._active_scores: list[float] = []

        # Per-image state (reset by begin_image)
        self._evidence:    list[float]         = []
        self._obs:         list[tuple[str, tuple]] = []
        self._prev_loc:    tuple | None         = None
        self._prev_leader: int   | None         = None

    # ------------------------------------------------------------------
    # Per-image API (replaces step / reset_temporal)
    # ------------------------------------------------------------------

    def begin_image(self) -> None:
        """Reset accumulators.  Evidence seeds with homeostatic boost."""
        self._evidence    = [mc._boost for mc in self.minicolumns]
        self._obs         = []
        self._prev_loc    = None
        self._prev_leader = None

    def observe(self, feat: str, loc: tuple) -> None:
        """Accumulate evidence from ONE (feat, loc) observation.

        For sensor layers (V1): called once per fixation with the HOG
        feature and the object-relative retinal location.

        1. Add overlap_score(feat, loc) to every minicolumn's evidence.
        2. Temporal bonus for the previous-fixation leader:
             TEMPORAL_BONUS  if its group-law prediction matched feat
             CONTINUITY_BONUS otherwise (cold-start priming)
        3. Update prev_leader and prev_loc.
        """
        if self.miss_penalty > 0.0:
            for i, mc in enumerate(self.minicolumns):
                self._evidence[i] += mc.overlap_score_signed(feat, loc, self.miss_penalty)
        else:
            for i, mc in enumerate(self.minicolumns):
                self._evidence[i] += mc.overlap_score(feat, loc)

        if self._prev_leader is not None and self._prev_loc is not None:
            disp = tuple(l - p for l, p in zip(loc, self._prev_loc))
            mc_prev = self.minicolumns[self._prev_leader]
            predicted = mc_prev.predict(self._prev_loc, disp)
            if predicted == feat:
                self._evidence[self._prev_leader] += self.TEMPORAL_BONUS
            else:
                self._evidence[self._prev_leader] += self.CONTINUITY_BONUS

        self._obs.append((feat, loc))
        self._prev_loc    = loc
        self._prev_leader = max(range(self.N_MINI),
                                key=lambda i: self._evidence[i])

    def observe_multi(self, observations: list[tuple[str, tuple]]) -> None:
        """Accumulate evidence from MULTIPLE independent (feat, loc) pairs.

        For higher layers (V2, IT): called once per fixation with one entry
        per lower-layer column in this column's receptive field.

        Each (feat, loc) is an independent observation; evidence from all
        observations is summed.  The temporal bonus fires if the pre-fixation
        leader's predictions (at zero displacement — RF-local positions are
        stationary between fixations) match the current observations.
        The bonus is proportional to the fraction of correct predictions.

        _prev_loc is NOT used here (RF positions do not shift between
        fixations), so there is no cross-fixation location displacement.
        """
        # Feedforward: sum overlap, normalised by RF size so each fixation
        # contributes the same total evidence regardless of how many RF slots
        # this column covers.  Prevents higher layers from accumulating
        # disproportionate evidence and locking out unexplored minicolumns.
        scale = 1.0 / len(observations)
        if self.miss_penalty > 0.0:
            for feat, loc in observations:
                for i, mc in enumerate(self.minicolumns):
                    self._evidence[i] += mc.overlap_score_signed(
                        feat, loc, self.miss_penalty) * scale
        else:
            for feat, loc in observations:
                for i, mc in enumerate(self.minicolumns):
                    self._evidence[i] += mc.overlap_score(feat, loc) * scale

        # Temporal: check how many of the current observations the
        # pre-fixation leader correctly predicted (displacement = (0,0)
        # because RF-local positions are constant across saccades).
        if self._prev_leader is not None and observations:
            mc_prev = self.minicolumns[self._prev_leader]
            n_correct = sum(
                1 for feat, loc in observations
                if mc_prev.predict(loc, (0, 0)) == feat
            )
            n = len(observations)
            if n_correct > 0:
                self._evidence[self._prev_leader] += (
                    self.TEMPORAL_BONUS * n_correct / n)
            else:
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
            for feat, loc in self._obs:
                self.minicolumns[winner].learn_one(feat, loc)
            self.minicolumns[winner]._n_wins += 1
        ev = self._evidence[winner] if self._evidence else 0.0
        self._active        = [winner]
        self._active_scores = [ev]
        return winner

    def commit_supervised(self, label: int, write: bool = True) -> int:
        """Force minicolumn `label` to win (supervised IT layer).

        Minicolumn index IS the class label.  No WTA competition.
        """
        if write:
            for feat, loc in self._obs:
                self.minicolumns[label].learn_one(feat, loc)
            self.minicolumns[label]._n_wins += 1
        ev = self._evidence[label] if self._evidence else 0.0
        self._active        = [label]
        self._active_scores = [ev]
        return label

    def commit_chl(self, clamped_label: int, write: bool = True) -> int:
        """Contrastive Hebbian Learning commit.

        Clamped phase : write to minicolumn (clamped_label % N_MINI).
        Free-phase correction : unlearn from the WTA winner if it differs.

        The contrastive term (unlearn) is what distinguishes CHL from pure
        teacher forcing.  It removes associations from whichever minicolumn
        happened to win the free-phase competition, preventing incorrect
        models from accumulating silently over thousands of training images.
        """
        free_winner    = self.tentative_winner()
        clamped_winner = clamped_label % self.N_MINI

        if write:
            for feat, loc in self._obs:
                self.minicolumns[clamped_winner].learn_one(feat, loc)
            self.minicolumns[clamped_winner]._n_wins += 1

            if free_winner != clamped_winner:
                for feat, loc in self._obs:
                    self.minicolumns[free_winner].unlearn_one(feat, loc)

        ev = self._evidence[clamped_winner] if self._evidence else 0.0
        self._active        = [clamped_winner]
        self._active_scores = [ev]
        return clamped_winner

    def apply_lateral_input(self, neighbor_winners: list[int],
                             bonus: float) -> None:
        """Add lateral evidence from neighbouring columns' tentative winners.

        Called once per fixation after this column's observe step.
        Each entry in neighbor_winners is the current tentative winner index
        of one adjacent column.  The bonus is added to that minicolumn's
        accumulated evidence, biasing WTA toward the neighbours' consensus.
        """
        if not self._evidence:
            return
        for winner in neighbor_winners:
            if 0 <= winner < self.N_MINI:
                self._evidence[winner] += bonus

    # ------------------------------------------------------------------
    # Compatibility helpers (lateral messaging, stats)
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
