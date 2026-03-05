"""Extensional storage for concept examples and KL divergence metrics.

An ExampleStore holds observed (inputs, outputs) pairs for a concept
before or alongside consolidation into a process rule.

KL divergence measures how well a proposed rule explains the stored
examples — it is the core signal for when consolidation is needed:

    KL ≈ 0   →  rule perfectly explains all examples (consolidated)
    KL → inf →  rule fails on every example (consolidation needed)

This mirrors the free-energy framing of the original predictive coding
project: surprise = KL divergence from the model's predictions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


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
