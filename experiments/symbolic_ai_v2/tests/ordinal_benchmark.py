"""Digit-level ordinal recognition benchmark.

Train the CTKG Predictor on succ/pred for 0–99 (digit-level flat format), then
test on three magnitude ranges never seen in training:

    100–999    (+1 order of magnitude from training)
    1000–9999  (+2 orders)
    10000–99999 (+3 orders)

Metrics per range:
    strict      — fraction of test cases where ALL generated digits are correct
    per_digit   — fraction of individual digit positions correct (aligned right)
    length_ok   — fraction where the generated answer has the correct digit count
    no_carry    — strict accuracy on inputs whose last digit ≠ 9 (no carry at LSB)
    carry       — strict accuracy on inputs whose last digit = 9 (carry at LSB)

The process rule extracted from training (carry propagation table) is magnitude-
independent, so we expect ≥ 90% strict accuracy on all three ranges.

Usage:
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/ordinal_benchmark.py
"""

from __future__ import annotations

import sys
import os
import random

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.corpus.digit_math_generator import (
    digit_succ_pred_split,
)
from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.ctkg.learning.fca_discover import discover_concepts
from experiments.symbolic_ai_v2.ctkg.learning.morphism_discover import (
    discover_morphisms,
)
from experiments.symbolic_ai_v2.ctkg.learning.process_discover import (
    discover_processes,
)
from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_MAX = 99
N_SAMPLE = 200    # test cases per range (sampled from each range)
MAX_STEPS = 12    # max tokens generated per sequence
R = 1             # context radius


# ---------------------------------------------------------------------------
# Test ranges
# ---------------------------------------------------------------------------

TEST_RANGES = [
    (100,    999,   "100-999      (+1 order) "),
    (1000,   9999,  "1000-9999    (+2 orders)"),
    (10000,  99999, "10000-99999  (+3 orders)"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ground_truth_output(seq: list[str]) -> list[str]:
    """Extract the expected output digits from a ground-truth sequence."""
    try:
        eq_idx = seq.index("eq")
    except ValueError:
        return []
    out = []
    for t in seq[eq_idx + 1:]:
        if t == "<eos>":
            break
        if t in "0123456789":
            out.append(t)
    return out


def _generated_output(predictor: Predictor, seq: list[str]) -> list[str]:
    """Run the predictor on the prefix up to and including 'eq'."""
    try:
        eq_idx = seq.index("eq")
    except ValueError:
        return []
    prefix = seq[: eq_idx + 1]
    tokens = predictor.generate(prefix, eos="<eos>", max_steps=MAX_STEPS)
    # Strip <eos>
    return [t for t in tokens if t != "<eos>"]


def _per_digit_accuracy(pred: list[str], truth: list[str]) -> float:
    """Right-aligned per-digit accuracy."""
    if not truth:
        return 1.0 if not pred else 0.0
    # Right-align: pad shorter list with '?'
    n = max(len(pred), len(truth))
    pred_pad  = ["?"] * (n - len(pred)) + pred
    truth_pad = ["?"] * (n - len(truth)) + truth
    correct = sum(p == t for p, t in zip(pred_pad, truth_pad))
    return correct / n


def _is_carry_case(seq: list[str]) -> bool:
    """True if the input's last digit is '9' (carry propagation needed)."""
    try:
        eq_idx = seq.index("eq")
    except ValueError:
        return False
    input_digits = [t for t in seq[1:eq_idx] if t in "0123456789"]
    return bool(input_digits) and input_digits[-1] == "9"


def _sample_sequences(test_seqs: list[list[str]], n: int, seed: int = 0) -> list[list[str]]:
    """Sample n sequences from test_seqs (or return all if fewer than n)."""
    rng = random.Random(seed)
    if len(test_seqs) <= n:
        return list(test_seqs)
    return rng.sample(test_seqs, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    # ---- Training ----
    _, train_seqs, _ = digit_succ_pred_split(
        train_max=TRAIN_MAX,
        test_min=TRAIN_MAX + 1,
        test_max=TRAIN_MAX + 1,
    )

    print(f"Digit-level ordinal recognition (trained on succ/pred 0-{TRAIN_MAX})")
    print(f"r={R}, k_neighbours=5, max_steps={MAX_STEPS}, #train={len(train_seqs)}")
    print()

    # Build HankelCount
    hc = HankelCount(r_max=R)
    hc.update_batch(train_seqs)

    # FCA
    lattices = discover_concepts(
        hankel=hc,
        r_levels=[R],
        lambda_productivity=0.1,
        merge_threshold=0.15,
        min_support=2.0,
    )
    lattice = lattices[0]

    # Morphism discovery
    mg = discover_morphisms(train_seqs, hc, lattice, r=R)

    # Process discovery
    process_rules = discover_processes(train_seqs, op_atoms=["succ", "pred"])
    print(f"Process rules discovered: {len(process_rules)}")
    for rule in process_rules:
        print(f"  {rule}")
    print()

    # Build Predictor
    pred = Predictor(
        hankel=hc,
        lattice=lattice,
        morphism_graph=mg,
        process_rules=process_rules,
        k_neighbours=5,
        r=R,
    )

    # ---- Evaluation ----
    header = (
        f"{'Test range':<22}  {'#seqs':>5}  "
        f"{'strict':>7}  {'per_digit':>9}  "
        f"{'length_ok':>9}  {'no_carry':>8}  {'carry':>6}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for (test_min, test_max, label) in TEST_RANGES:
        # Generate test sequences for this range
        _, _, test_seqs = digit_succ_pred_split(
            train_max=TRAIN_MAX,
            test_min=test_min,
            test_max=test_max,
        )
        sample = _sample_sequences(test_seqs, N_SAMPLE)

        n_strict = 0
        n_per_digit_sum = 0.0
        n_length_ok = 0
        n_carry_correct = 0;  n_carry_total = 0
        n_nocarry_correct = 0; n_nocarry_total = 0

        for seq in sample:
            truth = _ground_truth_output(seq)
            gen   = _generated_output(pred, seq)

            strict    = (gen == truth)
            pd_acc    = _per_digit_accuracy(gen, truth)
            length_ok = (len(gen) == len(truth))

            n_strict       += strict
            n_per_digit_sum += pd_acc
            n_length_ok    += length_ok

            if _is_carry_case(seq):
                n_carry_total += 1
                n_carry_correct += strict
            else:
                n_nocarry_total += 1
                n_nocarry_correct += strict

        n = len(sample)
        strict_pct   = 100.0 * n_strict / n
        pd_pct       = 100.0 * n_per_digit_sum / n
        length_pct   = 100.0 * n_length_ok / n
        nocarry_pct  = (100.0 * n_nocarry_correct / n_nocarry_total
                        if n_nocarry_total else float("nan"))
        carry_pct    = (100.0 * n_carry_correct / n_carry_total
                        if n_carry_total else float("nan"))

        print(
            f"{label:<22}  {n:>5}  "
            f"{strict_pct:>6.1f}%  {pd_pct:>8.1f}%  "
            f"{length_pct:>8.1f}%  {nocarry_pct:>7.1f}%  {carry_pct:>5.1f}%"
        )

    print()
    print("Required: strict >= 90% on 100-999.")


if __name__ == "__main__":
    run_benchmark()
