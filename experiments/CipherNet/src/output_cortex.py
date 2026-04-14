"""Output cortex — discrete action selection via basal ganglia gating.

BIOLOGY
-------
The motor cortex selects a discrete action (here: digit 0-9) through
the cortico-basal ganglia-thalamo-cortical (CBGTC) loop:

  Direct (Go) pathway:   cortex → striatum → GPi → thalamus → cortex
                         Net effect: disinhibits thalamus → action fires.
  Indirect (NoGo) path:  cortex → striatum → GPe → STN → GPi → thalamus
                         Net effect: inhibits thalamus → action suppressed.

Multiple action candidates compete. The candidate with the most
cortical support wins (Go), the rest are suppressed (NoGo).

In active inference terms: the motor system predicts "finger at position
3" and contracts muscles to minimize the proprioceptive prediction error.
There is no explicit inverse model — prediction error minimization IS the
motor command. (Adams, Shipp & Friston 2012 "Predictions not commands".)

IMPLEMENTATION
--------------
During learning:
  For each (macrocolumn_index, active_minicolumn_index) pair that fired
  while label L was present, increment the vote counter for L.

  This is Hebbian: cells that fire together wire together. The label
  cells in the output cortex strengthen their connection to the sensory
  minicolumns that were active at the same time.

During inference (WTA / basal ganglia gate):
  For each currently active (macro_idx, mini_idx) pair, look up which
  labels it was associated with during training. The label with the most
  total votes across ALL active minicolumns in ALL macrocolumns wins.
  This is the basal ganglia competition: one winner, all others suppressed.

No label lives inside any sensory minicolumn. The label association is
entirely here, in the output cortex.
"""
from __future__ import annotations

from collections import Counter


class OutputCortex:
    """Maps macrocolumn SDR patterns to discrete output labels.

    Voting is at the level of individual minicolumns, not whole SDR
    patterns. This makes the readout robust to slight variations in
    which minicolumns win the WTA from one image to the next.

    _mini_votes[(macro_idx, mini_idx)] = Counter[label]
        → "when this minicolumn was active, label L was present N times"
    """

    def __init__(self, n_outputs: int = 10):
        self.n_outputs = n_outputs
        # (macro_idx, mini_idx) → Counter[label]
        self._mini_votes: dict[tuple[int, int], Counter] = {}

    # ------------------------------------------------------------------
    # Learning  (Hebbian association: active minicolumn ↔ label)
    # ------------------------------------------------------------------

    def learn(self, macro_idx: int,
              sdr: frozenset[int], label: int) -> None:
        """Associate every active minicolumn in this SDR with label."""
        for mini_idx in sdr:
            key = (macro_idx, mini_idx)
            if key not in self._mini_votes:
                self._mini_votes[key] = Counter()
            self._mini_votes[key][label] += 1

    # ------------------------------------------------------------------
    # Inference  (WTA / basal ganglia gate)
    # ------------------------------------------------------------------

    def classify(self, active_per_macro: list[tuple[int, frozenset]]
                 ) -> tuple[int, Counter]:
        """Basal ganglia WTA: return winning label + full vote counts.

        active_per_macro: [(macro_idx, sdr_frozenset), ...]
          one entry per macrocolumn, each sdr = active minicolumn indices.

        For each active minicolumn, we look up which label it has been
        most strongly associated with during training, and cast one vote
        for that label. The label with the most votes wins.
        """
        votes: Counter = Counter()
        for macro_idx, sdr in active_per_macro:
            for mini_idx in sdr:
                key = (macro_idx, mini_idx)
                counter = self._mini_votes.get(key)
                if counter:
                    best_label = counter.most_common(1)[0][0]
                    votes[best_label] += 1

        if not votes:
            return -1, votes
        return votes.most_common(1)[0][0], votes

    def confidence(self, votes: Counter) -> float:
        """Fraction of total votes going to the winning label.

        1.0 = unanimous, 0.1 = barely above chance for 10 classes.
        """
        total = sum(votes.values())
        if total == 0:
            return 0.0
        return votes.most_common(1)[0][1] / total

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def n_associations(self) -> int:
        """Total (minicolumn, label) associations stored."""
        return len(self._mini_votes)

    def purity_stats(self) -> dict:
        """Fraction of minicolumns whose plurality label dominates clearly.

        Returns:
            mean_purity: average (plurality_count / total_votes) per minicolumn
            frac_pure80: fraction of minicolumns with purity ≥ 0.80
            frac_pure60: fraction of minicolumns with purity ≥ 0.60
        """
        if not self._mini_votes:
            return {'mean_purity': 0.0, 'frac_pure80': 0.0, 'frac_pure60': 0.0}
        purities = []
        for counter in self._mini_votes.values():
            total = sum(counter.values())
            if total > 0:
                purities.append(counter.most_common(1)[0][1] / total)
        if not purities:
            return {'mean_purity': 0.0, 'frac_pure80': 0.0, 'frac_pure60': 0.0}
        mean_p = sum(purities) / len(purities)
        return {
            'mean_purity': mean_p,
            'frac_pure80': sum(1 for p in purities if p >= 0.80) / len(purities),
            'frac_pure60': sum(1 for p in purities if p >= 0.60) / len(purities),
        }
