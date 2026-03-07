"""Extensional storage for concept examples and KL divergence metrics.

An ExampleStore holds observed (inputs, outputs) pairs for a concept
before or alongside consolidation into a process rule.

KL divergence measures how well a proposed rule explains the stored
examples — it is the core signal for when consolidation is needed:

    KL ≈ 0   ->  rule perfectly explains all examples (consolidated)
    KL -> inf ->  rule fails on every example (consolidation needed)

This mirrors the free-energy framing of the original predictive coding
project: surprise = KL divergence from the model's predictions.

Distributional concepts
-----------------------
For statistical concepts (next_word, next_action, next_note) where the
same inputs legitimately produce different outputs, KL never reaches 0
under a deterministic rule.  These concepts use freq_predict() /
freq_dist() instead of exact-match lookup, and mean_log_likelihood() as
the proper evaluation metric (cross-entropy under the empirical distribution).

The freq_predict / freq_dist methods are always available — they build
frequency statistics on the fly from the stored examples, requiring no
separate consolidation step.  engine.freq_consolidate() pre-builds a
snapshot of the freq table so that kl() can report the *residual*
entropy (the irreducible stochasticity of the task), not a synthesis failure.
"""

from __future__ import annotations

import collections
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class ExampleStore:
    """Extensional storage for a single concept's training examples.

    Examples are stored as (inputs, outputs) tuples.  Both sides are
    tuples of primitive values (integers, strings).

    Usage:
        store = ExampleStore('single_digit_addition')
        store.add((3, 'ADD', 4), (0, 7))
        store.add((8, 'ADD', 5), (1, 3))

        acc = store.accuracy(lambda inp: ai.ask('single_digit_addition', inp))
        kl  = store.kl_divergence(lambda inp: ai.ask('single_digit_addition', inp))
    """

    concept_name: str
    examples: List[Tuple[tuple, tuple]] = field(default_factory=list)

    def add(self, inputs: tuple, outputs: tuple) -> None:
        """Add one (inputs, outputs) training example."""
        self.examples.append((tuple(inputs), tuple(outputs)))

    def __len__(self) -> int:
        return len(self.examples)

    def __repr__(self) -> str:
        return f"ExampleStore({self.concept_name!r}, n={len(self.examples)})"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def accuracy(self, predict_fn: Callable[[tuple], Optional[tuple]]) -> float:
        """Fraction of stored examples correctly predicted by predict_fn.

        predict_fn(inputs) should return a tuple matching the stored
        outputs, or None if the concept cannot yet answer.
        """
        if not self.examples:
            return 0.0
        correct = sum(
            1 for inp, out in self.examples
            if predict_fn(inp) == out
        )
        return correct / len(self.examples)

    def kl_divergence(self, predict_fn: Callable[[tuple], Optional[tuple]]) -> float:
        """Surprisal of the current rule on the stored examples (in bits).

        For a deterministic rule:
            KL = -log2(accuracy)
            KL = 0.0   if rule gets every example right
            KL → inf   if rule fails on every example

        Uses accuracy as a proxy for the rule's empirical probability
        mass on the correct outputs.  This is an approximation valid
        when the rule is nearly deterministic (which it always is after
        consolidation).
        """
        acc = self.accuracy(predict_fn)
        if acc == 0.0:
            return float('inf')
        return -math.log2(acc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def sample(
        self,
        n: int,
        rng: Optional[random.Random] = None,
        replace: bool = False,
    ) -> 'ExampleStore':
        """Return a new ExampleStore with (at most) n examples.

        If replace=False (default), samples without replacement.
        If n >= len(self), returns a copy of the full store.
        """
        if rng is None:
            rng = random.Random()
        if replace:
            picked = [rng.choice(self.examples) for _ in range(n)]
        else:
            picked = rng.sample(self.examples, min(n, len(self.examples)))
        store = ExampleStore(self.concept_name)
        store.examples = list(picked)
        return store

    def all_inputs(self) -> List[tuple]:
        return [inp for inp, _ in self.examples]

    def all_outputs(self) -> List[tuple]:
        return [out for _, out in self.examples]

    # ------------------------------------------------------------------
    # Distributional methods (Phase L)
    # ------------------------------------------------------------------
    # These treat the ExampleStore as an empirical frequency table,
    # supporting statistical concepts where the same inputs legitimately
    # precede multiple different outputs (language, music, action planning).
    # ------------------------------------------------------------------

    def _count_table(self) -> Dict[tuple, collections.Counter]:
        """Build a raw count table: inputs -> Counter({output: count}).

        Called internally by freq_predict / freq_dist / build_full_freq_table.
        Re-computed each call; use build_full_freq_table() for a pre-built
        snapshot when query speed matters.
        """
        table: Dict[tuple, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for inp, out in self.examples:
            table[inp][out] += 1
        return table

    def freq_predict(self, inputs: tuple) -> Optional[tuple]:
        """Return the most frequent output observed for these inputs.

        Returns None if these inputs have never been seen.
        This is the mode (argmax) of the empirical output distribution —
        equivalent to the Markov model's top-1 prediction.
        """
        counter: collections.Counter = collections.Counter()
        for inp, out in self.examples:
            if inp == inputs:
                counter[out] += 1
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def freq_dist(self, inputs: tuple) -> Optional[Dict[tuple, float]]:
        """Return the full empirical output distribution for these inputs.

        Returns a dict mapping output tuples to probabilities (summing to 1),
        or None if these inputs have never been seen.
        """
        counter: collections.Counter = collections.Counter()
        for inp, out in self.examples:
            if inp == inputs:
                counter[out] += 1
        if not counter:
            return None
        total = sum(counter.values())
        return {out: count / total for out, count in counter.items()}

    def build_full_freq_table(self) -> Dict[tuple, Dict[tuple, float]]:
        """Build the complete empirical conditional distribution.

        Returns a dict: inputs -> {output: probability}.
        Iterates examples once; use for pre-building snapshots (freq_consolidate).
        """
        raw = self._count_table()
        table: Dict[tuple, Dict[tuple, float]] = {}
        for inp, counter in raw.items():
            total = sum(counter.values())
            table[inp] = {out: count / total for out, count in counter.items()}
        return table

    def mean_log_likelihood(
        self,
        dist_fn: Callable[[tuple], Optional[Dict[tuple, float]]],
    ) -> float:
        """Mean log-likelihood of stored outputs under predicted distributions.

        For each example (inp, out), computes:
            log2( P(out | inp) )   under dist_fn(inp)

        Returns the mean over all examples (in bits, negative = more surprise).

        Interpretation:
            0.0  = predicted the correct output with probability 1 (perfect)
            -1.0 = 50% probability assigned to the correct output on average
            -H   = where H is the empirical entropy of the outputs

        dist_fn should return a dict {output: prob} or None (treated as uniform).
        """
        if not self.examples:
            return 0.0
        total_ll = 0.0
        n_scored = 0
        for inp, out in self.examples:
            dist = dist_fn(inp)
            if dist is None:
                # Unseen context: use 1/vocab as uniform prior
                # (very small, but finite — avoids -inf)
                total_ll += math.log2(1e-6)
            else:
                prob = dist.get(out, 1e-9)   # unseen output: near-zero
                total_ll += math.log2(max(prob, 1e-9))
            n_scored += 1
        return total_ll / n_scored if n_scored else 0.0

    def empirical_entropy(self) -> float:
        """Mean empirical entropy H(output | input) in bits.

        Averages Shannon entropy over all unique input contexts seen.
        This is the theoretical lower bound on how well any model can
        predict — the irreducible stochasticity of the concept.

        KL between the freq_dist model and the true distribution = 0
        by construction (the freq_dist IS the empirical distribution).
        The residual entropy reported here is the genuine task difficulty.
        """
        raw = self._count_table()
        if not raw:
            return 0.0
        entropies = []
        for counter in raw.values():
            total = sum(counter.values())
            h = 0.0
            for count in counter.values():
                p = count / total
                if p > 0:
                    h -= p * math.log2(p)
            entropies.append(h)
        return sum(entropies) / len(entropies)

    def coverage(self) -> float:
        """Fraction of unique inputs that have been seen at least once.

        Returns 1.0 if every stored input is unique (no repeats — sparse coverage).
        Lower values indicate concentrated experience on few contexts.
        Used by agent_loop to decide when to move to the next corpus window.
        """
        if not self.examples:
            return 0.0
        unique_inputs = len({inp for inp, _ in self.examples})
        return unique_inputs / len(self.examples)
