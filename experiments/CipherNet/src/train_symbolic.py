"""CipherNet training entry point.

Usage:
    python train_symbolic.py [--config configs/visual.yaml]
                             [--n_train 100] [--n_test 1000]
                             [--n_fixations 9]
                             [--confidence 0.6]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import yaml
import numpy as np

from mnist_loader import load_mnist
from eye import Eye, FovealExplorer
from cortex import Cortex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       default='configs/guided.yaml')
    parser.add_argument('--n_train',      type=int,   default=100)
    parser.add_argument('--n_test',       type=int,   default=1000)
    parser.add_argument('--n_fixations',  type=int,   default=9)
    parser.add_argument('--confidence',   type=float, default=0.6)
    parser.add_argument('--diagnose',     action='store_true',
                        help='Print full layer/column/minicolumn representation report after training')
    parser.add_argument('--diag_sample', type=int, default=200,
                        help='Images per diagnostic probe (default 200)')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    (train_img, train_lbl), (test_img, test_lbl) = load_mnist()

    eye_type = config.get('eye_type', 'foveal')
    if eye_type == 'static':
        from eye import StaticEye
        eye      = StaticEye(image_size=28)
        explorer = eye
    else:
        eye      = Eye(retina_size=19)
        explorer = FovealExplorer(eye, n_fixations=args.n_fixations)

    print(f"Building cortex from {args.config} (eye={eye_type}) ...")
    cortex = Cortex.from_config(config, eye=eye)
    for s in cortex.hierarchy_stats():
        sup = ' [supervised]' if s['supervised'] else ''
        print(f"  Layer '{s['id']}'{sup}: "
              f"{s['n_macrocolumns']} macrocolumns × {s['n_mini']} minicolumns")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nTraining on {args.n_train} images ...")
    t0 = time.time()
    for i in range(args.n_train):
        fixations = explorer.get_fixations(train_img[i])
        cortex.learn(train_img[i], int(train_lbl[i]), fixations)
        if (i + 1) % max(1, args.n_train // 5) == 0:
            print(f"  {i+1}/{args.n_train}: {time.time()-t0:.1f}s")

    print("\nStats after training:")
    for s in cortex.hierarchy_stats():
        print(f"  [{s['id']}] used_mini={s['used_mini']}  "
              f"total_locations={s['total_locations']}")

    purity = cortex._output_cortex.purity_stats()
    print(f"  OutputCortex: {cortex._output_cortex.n_associations()} associations  "
          f"mean_purity={purity['mean_purity']:.2f}  "
          f"pure80={purity['frac_pure80']:.0%}  "
          f"pure60={purity['frac_pure60']:.0%}")

    # ------------------------------------------------------------------
    # Diagnostics (probes run before the full test loop)
    # ------------------------------------------------------------------
    if args.diagnose:
        from diagnostics import run_diagnostics
        run_diagnostics(
            cortex, explorer,
            train_img[:args.n_train], train_lbl[:args.n_train],
            test_img[:args.n_test],   test_lbl[:args.n_test],
            n_sample=args.diag_sample,
            confidence=args.confidence,
        )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    print(f"\nTesting on {args.n_test} images "
          f"(confidence threshold={args.confidence}) ...")
    correct   = 0
    no_assoc  = 0   # OutputCortex returned -1 (active IT mini has no training associations)
    t0 = time.time()

    for i in range(args.n_test):
        fixations = explorer.get_fixations(test_img[i])
        pred, votes = cortex.classify(
            test_img[i], fixations,
            confidence_threshold=args.confidence)

        if pred == -1:
            no_assoc += 1
        elif pred == int(test_lbl[i]):
            correct += 1

        if (i + 1) % max(1, args.n_test // 4) == 0:
            elapsed = time.time() - t0
            done    = i + 1
            print(f"  {done}/{args.n_test}: {correct/done*100:.1f}%  "
                  f"({elapsed:.1f}s)")

    print(f"\nTest time: {time.time()-t0:.1f}s")
    if args.n_test > 0:
        denominator = args.n_test - no_assoc
        acc = correct / args.n_test * 100
        print(f"MNIST accuracy: {correct}/{args.n_test} = {acc:.2f}%  "
              f"(no-assoc: {no_assoc})")


if __name__ == '__main__':
    main()
