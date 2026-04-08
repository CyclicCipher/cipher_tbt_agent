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
    print("\n=== Stage 3: MNIST (Foveal Exploration) ===")

    from mnist_loader import load_mnist
    from eye import Eye
    from visual_cortex import RetinotopicV1, FovealExplorer
    from codebook import PatchCodebook

    (train_img, train_lbl), (test_img, test_lbl) = load_mnist()

    # Create eye (19×19 retinal image) + V1 with patch RFs + codebook.
    eye = Eye(retina_size=19)
    codebook = PatchCodebook(n_codes=256)
    v1 = RetinotopicV1(eye, codebook, patch_size=5, stride=3)
    explorer = FovealExplorer(eye, v1, n_fixations=3)
    print(f"  Eye: {eye}")
    print(f"  V1: {v1.n_columns} columns ({v1.grid_h}x{v1.grid_w})")

    # Pre-train codebook: extract patches from center-fixated images.
    t0 = time.time()
    rng = np.random.RandomState(42)
    ps = v1.patch_size
    sample_patches = []
    for idx in rng.choice(len(train_img), size=500, replace=False):
        eye.fixate(14.0, 14.0)  # center of 28×28
        retina = eye.sample(train_img[idx])
        for gy in range(v1.grid_h):
            for gx in range(v1.grid_w):
                y0, x0 = gy * v1.stride, gx * v1.stride
                sample_patches.append(retina[y0:y0+ps, x0:x0+ps])
    codebook.fit(np.array(sample_patches[:10000]), verbose=True)
    print(f"  Codebook time: {time.time()-t0:.1f}s")

    # Train: explore images with saccades.
    n_train = min(20000, len(train_img))
    t0 = time.time()
    for i in range(n_train):
        label = str(int(train_lbl[i]))
        explorer.explore(train_img[i], label=label, learn=True)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  Trained {i+1}/{n_train} ({elapsed:.1f}s)")
    print(f"  Training time: {time.time()-t0:.1f}s")

    # Memory stats.
    total_mem = sum(len(c.memory) for c in v1.columns)
    ident = sum(1 for c in v1.columns for k in c.memory if ':d' not in k)
    disp = sum(1 for c in v1.columns for k in c.memory if ':d' in k)
    print(f"  Memory: {total_mem} total ({ident} identity, {disp} displacement)")

    # Test.
    t0 = time.time()
    correct = 0
    for i in range(len(test_img)):
        pred, _ = explorer.explore(test_img[i], learn=False)
        if pred is not None and pred == str(int(test_lbl[i])):
            correct += 1
        if (i + 1) % 2000 == 0:
            print(f"  Tested {i+1}/{len(test_img)}: "
                  f"{correct}/{i+1} = {correct/(i+1)*100:.1f}%")

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
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
