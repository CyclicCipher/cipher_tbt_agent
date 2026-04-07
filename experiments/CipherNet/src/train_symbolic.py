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


def stage_succession(brain: SymbolicBrain):
    """Single-digit succession: 0→1, 1→2, ..., 8→9. One pass."""
    print("\n=== Stage 1: Succession (1 pass, 9 examples) ===")
    brain.train_succession([(str(d), str(d + 1)) for d in range(9)])

    correct = 0
    for d in range(9):
        pred = brain.predict_successor(str(d))
        token, _ = brain.read_output(pred)
        expected = str(d + 1)
        ok = token == expected
        if ok:
            correct += 1
        print(f"  {d} -> {token} (expected {expected}) [{'OK' if ok else 'X'}]")
    print(f"  Result: {correct}/9")
    return correct


def stage_carry(brain: SymbolicBrain):
    """Multi-digit succession with carry (uses Z/10Z morphism)."""
    print("\n=== Stage 2: Multi-digit Carry ===")

    # No-carry pairs.
    pairs = []
    for tens in range(1, 5):
        for ones in range(9):
            pairs.append((f"{tens}{ones}", f"{tens}{ones + 1}"))
    # Carry pairs.
    for tens in range(1, 5):
        pairs.append((f"{tens}9", f"{tens + 1}0"))
    pairs.append(("9", "10"))
    pairs.append(("99", "100"))

    correct = 0
    for inp, expected in pairs:
        produced = brain.predict_number_successor(inp)
        if produced == expected:
            correct += 1
    print(f"  Trained range: {correct}/{len(pairs)} correct")

    # Holdout: unseen tens digits.
    holdout = []
    for tens in range(5, 10):
        for ones in range(9):
            holdout.append((f"{tens}{ones}", f"{tens}{ones + 1}"))
        holdout.append((f"{tens}9", f"{tens + 1}0"))
    h_correct = sum(1 for i, e in holdout
                    if brain.predict_number_successor(i) == e)
    print(f"  Holdout (tens 5-9): {h_correct}/{len(holdout)} correct")
    return correct, h_correct


def stage_ood(brain: SymbolicBrain):
    """OOD: never-seen digit counts."""
    print("\n=== Stage 3: OOD Evaluation ===")
    tests = [
        ("59", "60"), ("69", "70"), ("89", "90"),
        ("51", "52"), ("67", "68"), ("83", "84"),
        ("99", "100"),
        ("199", "200"), ("299", "300"), ("599", "600"),
        ("999", "1000"),
        ("9999", "10000"),
        ("99999", "100000"),
        ("999999999", "1000000000"),
    ]
    correct = 0
    for inp, expected in tests:
        produced = brain.predict_number_successor(inp)
        ok = produced == expected
        if ok:
            correct += 1
        print(f"  {inp} -> {produced} (expected {expected}) [{'OK' if ok else 'X'}]")
    print(f"\n  OOD: {correct}/{len(tests)}")
    return correct


def main():
    parser = argparse.ArgumentParser(description="CipherNet Symbolic Training")
    parser.add_argument('--stage', type=str, default='all',
                        choices=['succession', 'carry', 'ood', 'all'])
    args = parser.parse_args()

    brain = SymbolicBrain()

    if args.stage in ('succession', 'all'):
        stage_succession(brain)
    if args.stage in ('carry', 'all'):
        stage_carry(brain)
    if args.stage in ('ood', 'all'):
        stage_ood(brain)

    print(f"\n{'=' * 50}")
    print(f"Succession memory: {len(brain.succession.memory)} entries")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
