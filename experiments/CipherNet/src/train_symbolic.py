"""Symbolic column training — succession + MNIST, same AI.

Usage:
    python train_symbolic.py --stage succession
    python train_symbolic.py --stage mnist
    python train_symbolic.py --stage ood
    python train_symbolic.py --stage all
"""
from __future__ import annotations

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from symbolic_brain import SymbolicBrain
import numpy as np


# -----------------------------------------------------------------------
# Stage 1: Succession
# -----------------------------------------------------------------------

def stage_succession(brain: SymbolicBrain):
    print("\n=== Stage 1: Succession (1 pass, 9 examples) ===")
    brain.train_succession([(str(d), str(d + 1)) for d in range(9)])
    correct = 0
    for d in range(9):
        pred = brain.predict_successor(str(d))
        token, _ = brain.read_output(pred)
        expected = str(d + 1)
        ok = token == expected
        if ok: correct += 1
        print(f"  {d} -> {token} (expected {expected}) [{'OK' if ok else 'X'}]")
    print(f"  Result: {correct}/9")
    return correct


# -----------------------------------------------------------------------
# Stage 2: OOD succession
# -----------------------------------------------------------------------

def stage_ood(brain: SymbolicBrain):
    print("\n=== Stage 2: OOD Succession ===")
    tests = [
        ("59", "60"), ("99", "100"), ("999", "1000"),
        ("9999", "10000"), ("999999999", "1000000000"),
    ]
    correct = 0
    for inp, expected in tests:
        produced = brain.predict_number_successor(inp)
        ok = produced == expected
        if ok: correct += 1
        print(f"  {inp} -> {produced} [{'OK' if ok else 'X'}]")
    print(f"  OOD: {correct}/{len(tests)}")
    return correct


# -----------------------------------------------------------------------
# Stage 3: MNIST
# -----------------------------------------------------------------------

def stage_mnist(brain: SymbolicBrain):
    print("\n=== Stage 3: MNIST Classification ===")

    from mnist_loader import load_mnist
    (train_img, train_lbl), (test_img, test_lbl) = load_mnist()

    # Initialize visual cortex.
    brain.init_visual_cortex(
        image_shape=(28, 28),
        patch_size=4, stride=4,
        n_codes=2048,
    )

    # Pre-train codebook on training patches.
    t0 = time.time()
    brain.train_codebook(train_img, verbose=True)
    print(f"  Codebook time: {time.time()-t0:.1f}s")

    # Train: one pass through training set.
    t0 = time.time()
    for i in range(len(train_img)):
        brain.train_image(train_img[i], int(train_lbl[i]))
        if (i + 1) % 10000 == 0:
            print(f"  Trained {i+1}/{len(train_img)}")
    print(f"  Training time: {time.time()-t0:.1f}s")

    # Memory stats.
    total_mem = sum(len(col.memory) for col in brain.visual_cortex.all_columns())
    print(f"  Total column memory entries: {total_mem}")

    # Test.
    t0 = time.time()
    correct = 0
    for i in range(len(test_img)):
        pred, votes = brain.classify_image(test_img[i])
        if pred is not None and int(pred) == test_lbl[i]:
            correct += 1

    acc = correct / len(test_img) * 100
    print(f"  Test time: {time.time()-t0:.1f}s")
    print(f"  MNIST accuracy: {correct}/{len(test_img)} = {acc:.2f}%")
    return acc


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CipherNet Symbolic Training")
    parser.add_argument('--stage', type=str, default='all',
                        choices=['succession', 'ood', 'mnist', 'all'])
    args = parser.parse_args()

    brain = SymbolicBrain()

    if args.stage in ('succession', 'all'):
        stage_succession(brain)
    if args.stage in ('ood', 'all'):
        stage_ood(brain)
    if args.stage in ('mnist', 'all'):
        stage_mnist(brain)

    print(f"\n{'=' * 50}")
    print(f"Succession memory: {len(brain.succession.memory)} entries")
    if brain.visual_cortex:
        total = sum(len(c.memory) for c in brain.visual_cortex.all_columns())
        print(f"Visual cortex: {brain.visual_cortex.n_columns()} columns, "
              f"{total} memory entries")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
