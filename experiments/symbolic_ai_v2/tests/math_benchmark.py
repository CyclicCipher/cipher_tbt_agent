"""math_benchmark.py — SpectralPredictor accuracy on the math corpus.

For each level of the math corpus (counting → Bernoulli), trains a
SpectralPredictor on the training sequences and measures top-1 accuracy
on the final token of both training and test sequences.

Metric: top-1 accuracy on the final token, given the full prefix.
This is the hardest reasonable test: the model sees the complete
question (e.g. 'add 3 4 eq') and must predict the answer ('7').

Train vs test gap reveals memorisation vs generalisation:
  - train ≈ test  → generalising
  - train >> test → memorising (cannot extrapolate)
  - train ≈ 0     → structure not discoverable from these examples at all

Run:
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/math_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.symbolic_ai_v2.corpus.math_generator import all_levels
from experiments.symbolic_ai_v2.core.spectral_predict import SpectralPredictor

K_MAX = 6


def _top1(sp: SpectralPredictor, seqs: list[list[str]]) -> tuple[int, int]:
    """Return (n_correct, n_total) for final-token top-1 accuracy."""
    correct = 0
    total   = 0
    for seq in seqs:
        if len(seq) < 2:
            continue
        prefix = seq[:-1]
        target = seq[-1]
        dist   = sp.predict(prefix)
        pred   = max(dist, key=dist.get) if dist else None
        if pred == target:
            correct += 1
        total += 1
    return correct, total


def benchmark_level(
    train_seqs: list[list[str]],
    test_seqs:  list[list[str]],
) -> tuple[float, float]:
    """Return (train_acc, test_acc)."""
    sp = SpectralPredictor.train(train_seqs, k_max=K_MAX)
    tr_c, tr_n = _top1(sp, train_seqs)
    te_c, te_n = _top1(sp, test_seqs)
    return (tr_c / tr_n if tr_n else 0.0,
            te_c / te_n if te_n else 0.0)


def main() -> None:
    print(f"\n{'Level':<18} {'#train':>7} {'#test':>6}  {'train':>7} {'test':>7}  {'gap':>7}")
    print("-" * 60)

    for name, train, test in all_levels():
        tr_acc, te_acc = benchmark_level(train, test)
        gap = tr_acc - te_acc
        print(
            f"{name:<18} {len(train):>7} {len(test):>6}"
            f"  {tr_acc:>6.1%} {te_acc:>6.1%}  {gap:>+6.1%}"
        )

    print()


if __name__ == "__main__":
    main()
