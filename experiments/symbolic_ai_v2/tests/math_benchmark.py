"""math_benchmark.py — SpectralPredictor accuracy on the math corpus.

For each level of the math corpus (counting → Bernoulli), trains a
SpectralPredictor on the training sequences and measures how often it
correctly predicts the final token of each test sequence.

Metric: top-1 accuracy on the final token, given the full prefix.
This is the hardest reasonable test: the model sees the complete
question (e.g. 'add 3 4 eq') and must predict the answer ('7').

Run:
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/math_benchmark.py

Output:
    Level            train   test    acc
    counting           23      7   xx.x%
    successor         ...
    ...
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.symbolic_ai_v2.corpus.math_generator import all_levels
from experiments.symbolic_ai_v2.core.spectral_predict import SpectralPredictor

K_MAX = 6


def accuracy_on_level(
    train_seqs: list[list[str]],
    test_seqs:  list[list[str]],
) -> float:
    """Train SpectralPredictor on train_seqs; measure final-token top-1 on test_seqs."""
    sp = SpectralPredictor.train(train_seqs, k_max=K_MAX)

    correct = 0
    for seq in test_seqs:
        if len(seq) < 2:
            continue
        prefix = seq[:-1]
        target = seq[-1]
        dist   = sp.predict(prefix)
        pred   = max(dist, key=dist.get) if dist else None
        if pred == target:
            correct += 1

    return correct / len(test_seqs) if test_seqs else 0.0


def main() -> None:
    print(f"\n{'Level':<18} {'train':>6} {'test':>6}  {'acc':>7}")
    print("-" * 42)

    for name, train, test in all_levels():
        acc = accuracy_on_level(train, test)
        print(f"{name:<18} {len(train):>6} {len(test):>6}  {acc:>6.1%}")

    print()


if __name__ == "__main__":
    main()
