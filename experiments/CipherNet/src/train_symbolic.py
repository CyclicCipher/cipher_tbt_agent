"""Symbolic column training — succession + MNIST with category discovery.

Usage:
    python train_symbolic.py --stage succession
    python train_symbolic.py --stage mnist
    python train_symbolic.py --stage all
"""
from __future__ import annotations

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from symbolic_brain import SymbolicBrain
from symbolic_column import SuccessionEngine
import numpy as np


def stage_succession(brain: SymbolicBrain):
    print("\n=== Stage 1: Succession (1 pass, 9 examples) ===")
    brain.train_succession([(str(d), str(d + 1)) for d in range(9)])
    correct = 0
    for d in range(9):
        pred = brain.predict_successor(str(d))
        token, _ = brain.read_output(pred)
        expected = str(d + 1)
        if token == expected:
            correct += 1
        print(f"  {d} -> {token} (expected {expected}) [{'OK' if token == expected else 'X'}]")
    print(f"  Result: {correct}/9")


def stage_ood(brain: SymbolicBrain):
    print("\n=== Stage 2: OOD Succession ===")
    tests = [("59","60"),("99","100"),("999","1000"),("9999","10000"),("999999999","1000000000")]
    correct = sum(1 for i, e in tests if brain.predict_number_successor(i) == e)
    for i, e in tests:
        p = brain.predict_number_successor(i)
        print(f"  {i} -> {p} [{'OK' if p == e else 'X'}]")
    print(f"  OOD: {correct}/{len(tests)}")


def stage_mnist(brain: SymbolicBrain):
    print("\n=== Stage 3: MNIST (Category Discovery) ===")

    from mnist_loader import load_mnist
    from eye import Eye
    from visual_cortex import HierarchicalV1, FovealExplorer
    from codebook import PatchCodebook

    (train_img, train_lbl), (test_img, test_lbl) = load_mnist()

    # Setup: eye + hierarchical cortex (2 levels).
    eye = Eye(retina_size=19)
    codebook = PatchCodebook(n_codes=256)
    cortex = HierarchicalV1(eye, codebook, patch_size=5, stride=3,
                            n_levels=2, pool=2, n_categories=32)
    explorer = FovealExplorer(eye, cortex, n_fixations=9)
    print(f"  Eye: {eye}")
    for i, lev in enumerate(cortex.levels):
        print(f"  Level {i} ({lev.name}): {lev.grid_h}x{lev.grid_w} = {lev.n_cols} columns")

    # Codebook.
    t0 = time.time()
    ps = cortex.patch_size
    patches = []
    for idx in range(200):
        eye.fixate(14.0, 14.0)
        r = eye.sample(train_img[idx])
        gh, gw = cortex.levels[0].grid_h, cortex.levels[0].grid_w
        st = cortex.stride
        for gy in range(gh):
            for gx in range(gw):
                patches.append(r[gy*st:gy*st+ps, gx*st:gx*st+ps])
    codebook.fit(np.array(patches[:5000]), verbose=True)
    print(f"  Codebook: {time.time()-t0:.1f}s")

    # Phase 1: Explore unlabeled → accumulate triples.
    t0 = time.time()
    n_explore = 1000
    for i in range(n_explore):
        explorer.explore_unlabeled(train_img[i])
    total_triples = sum(lev.total_triples() for lev in cortex.levels)
    print(f"  Phase 1 (explore {n_explore}): {time.time()-t0:.1f}s, {total_triples} triples")

    # Phase 2: Discover categories at all levels.
    t0 = time.time()
    cortex.discover_all(verbose=True)
    print(f"  Phase 2 (discover): {time.time()-t0:.1f}s")

    # Phase 3: Learn.
    t0 = time.time()
    n_train = 3000
    for i in range(n_train):
        explorer.learn(train_img[i], str(int(train_lbl[i])))
        if (i + 1) % 1000 == 0:
            print(f"  Trained {i+1}/{n_train}")
    print(f"  Phase 3 (learn {n_train}): {time.time()-t0:.1f}s")

    # Test.
    t0 = time.time()
    n_test = 1000
    correct = 0
    for i in range(n_test):
        pred, conf = explorer.recognize(test_img[i])
        if pred == str(int(test_lbl[i])):
            correct += 1
        if (i + 1) % 500 == 0:
            print(f"  Tested {i+1}/{n_test}: {correct}/{i+1} = {correct/(i+1)*100:.1f}%")

    acc = correct / n_test * 100
    print(f"  Test: {time.time()-t0:.1f}s")
    print(f"  MNIST accuracy: {correct}/{n_test} = {acc:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='all',
                        choices=['succession', 'ood', 'mnist', 'all'])
    args = parser.parse_args()

    brain = SymbolicBrain()

    if args.stage in ('succession', 'all'):
        stage_succession(brain)
    if args.stage in ('ood', 'all'):
        stage_ood(brain)
    if args.stage in ('mnist', 'all'):
        stage_mnist(brain)

    print(f"\n{'='*50}")
    print(f"Succession: {len(brain.succession.models)} models")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
