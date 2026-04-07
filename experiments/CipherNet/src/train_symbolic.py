"""Symbolic column training — one-shot learning benchmarks.

Usage:
    python train_symbolic.py --stage succession
    python train_symbolic.py --stage carry
    python train_symbolic.py --stage ood
    python train_symbolic.py --stage all
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from symbolic_brain import SymbolicBrain
from symbolic_column import SuccessionColumn, PlaceValueColumn


# -----------------------------------------------------------------------
# Stage 1: Succession (single digit, one pass)
# -----------------------------------------------------------------------

def stage_succession(brain: SymbolicBrain):
    """Learn succession: 0→1, 1→2, ..., 8→9. ONE PASS."""
    print("\n=== Stage 1: Succession (1 pass, 9 examples) ===")

    # Ensure succession column exists.
    if 'succession' not in brain.columns:
        brain.add_column('succession', SuccessionColumn('succession'))

    # Train: one pass.
    for d in range(9):
        brain.feed(str(d))
        brain.teach(str(d + 1))

    # Test.
    correct = 0
    for d in range(9):
        brain.feed(str(d))
        token, _ = brain.read_output()
        expected = str(d + 1)
        ok = token == expected
        if ok:
            correct += 1
        print(f"  {d} -> {token} (expected {expected}) [{'OK' if ok else 'X'}]")
    print(f"  Result: {correct}/9")
    return correct


# -----------------------------------------------------------------------
# Stage 2: Multi-digit succession with carry
# -----------------------------------------------------------------------

def stage_carry(brain: SymbolicBrain):
    """Learn multi-digit succession: 10→11, 19→20, 99→100."""
    print("\n=== Stage 2: Multi-digit Carry ===")

    # Train single-digit rollover (9→10 requires carry).
    train_pairs = []

    # No-carry: ones digit 0-8 for tens 1-4.
    for tens in range(1, 5):
        for ones in range(9):
            train_pairs.append((f"{tens}{ones}", f"{tens}{ones + 1}"))

    # Carry: ones digit 9 for tens 1-4.
    for tens in range(1, 5):
        train_pairs.append((f"{tens}9", f"{tens + 1}0"))

    # Special: 9→10 (single to double digit).
    train_pairs.append(("9", "10"))

    # 99→100 (double to triple digit).
    train_pairs.append(("99", "100"))

    print(f"  Training {len(train_pairs)} examples...")
    for inp, out in train_pairs:
        brain.teach_number(inp, out)

    # Test trained examples.
    correct = 0
    for inp, expected in train_pairs:
        brain.feed_number(inp)
        produced = brain.read_number()
        ok = produced == expected
        if ok:
            correct += 1
    print(f"  Trained: {correct}/{len(train_pairs)} correct")

    # Holdout: unseen tens digits.
    holdout = []
    for tens in range(5, 9):
        for ones in range(9):
            holdout.append((f"{tens}{ones}", f"{tens}{ones + 1}"))
        holdout.append((f"{tens}9", f"{tens + 1}0"))
    h_correct = 0
    for inp, expected in holdout:
        brain.feed_number(inp)
        produced = brain.read_number()
        if produced == expected:
            h_correct += 1
    print(f"  Holdout (tens 5-8): {h_correct}/{len(holdout)} correct")

    return correct, h_correct


# -----------------------------------------------------------------------
# Stage 3: OOD evaluation (no training on these digit counts)
# -----------------------------------------------------------------------

def stage_ood(brain: SymbolicBrain):
    """Test out-of-distribution generalization."""
    print("\n=== Stage 3: OOD Evaluation ===")

    ood_tests = [
        # Near OOD: unseen 2-digit pairs
        ("59", "60"), ("69", "70"), ("79", "80"), ("89", "90"),
        ("51", "52"), ("67", "68"), ("83", "84"),
        # Far OOD: never-seen digit counts
        ("99", "100"),
        ("199", "200"), ("299", "300"), ("599", "600"),
        ("999", "1000"),
        ("9999", "10000"),
        ("99999", "100000"),
    ]

    correct = 0
    for inp, expected in ood_tests:
        brain.feed_number(inp)
        produced = brain.read_number()
        ok = produced == expected
        tag = "OK" if ok else "X"
        print(f"  {inp} -> {produced} (expected {expected}) [{tag}]")
        if ok:
            correct += 1

    print(f"\n  OOD total: {correct}/{len(ood_tests)}")
    return correct


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CipherNet Symbolic Training")
    parser.add_argument('--stage', type=str, default='all',
                        choices=['succession', 'carry', 'ood', 'all'])
    args = parser.parse_args()

    brain = SymbolicBrain()
    brain.add_column('succession', SuccessionColumn('succession'))

    if args.stage in ('succession', 'all'):
        stage_succession(brain)

    if args.stage in ('carry', 'all'):
        stage_carry(brain)

    if args.stage in ('ood', 'all'):
        stage_ood(brain)

    print(f"\n{'=' * 60}")
    print(f"Symbolic brain: {len(brain.columns)} columns")
    for name, col in brain.columns.items():
        print(f"  {name}: {len(col.memory)} memory entries")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
