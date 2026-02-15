#!/usr/bin/env python3
"""
Phase 5d: Ablation testing for Naja.

Runs a grid of (preset × task) combinations and collects results
into a summary table. Each run is a subprocess for clean GPU state.

Usage:
  # Full ablation grid (~45 runs, ~8 min on RTX 3050 Ti)
  python run_ablations.py

  # Dry run (print commands without executing)
  python run_ablations.py --dry-run

  # Filter to specific tasks/presets
  python run_ablations.py --tasks parity associative_recall
  python run_ablations.py --presets mamba3_base naja_full delta_only

  # Quick run (fewer epochs)
  python run_ablations.py --epochs 20

  # Resume from a previous results file (skip completed runs)
  python run_ablations.py --resume
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# -------------------------------------------------------------------------
# Ablation grid
# -------------------------------------------------------------------------

# Core presets for feature isolation.
# Ablation logic: full model vs full-minus-one-feature.
#
#   Feature tested         | Control (ON)       | Ablated (OFF)
#   -----------------------|--------------------|-----------------
#   PoPE vs RoPE           | mamba3_base        | mamba3_rope
#   Delta rule             | delta_only         | mamba3_base
#   PoPE pair (B2)         | naja_full          | delta_per_channel
#   Per-channel decay      | naja_full          | pope_perp_only
#   Surprise gating        | surprise           | naja_full
#   MIMO r=2               | mimo2              | naja_full

ALL_PRESETS = [
    'mamba3_base',
    'mamba3_rope',
    'delta_only',
    'pope_perp_only',
    'per_channel_only',
    'delta_per_channel',
    'naja_full',
    'surprise',
    'mimo2',
]

ALL_TASKS = [
    'associative_recall',
    'parity',
    'multi_scale',
    'permutation_3',
    'permutation_4',
]


def parse_args():
    p = argparse.ArgumentParser(description='Run Naja ablation grid')
    p.add_argument('--presets', nargs='+', default=None,
                   help=f'Presets to run (default: all). Choices: {ALL_PRESETS}')
    p.add_argument('--tasks', nargs='+', default=None,
                   help=f'Tasks to run (default: all). Choices: {ALL_TASKS}')
    p.add_argument('--epochs', type=int, default=50,
                   help='Epochs per run (default: 50)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='auto',
                   help='Device for training (default: auto)')
    p.add_argument('--results_file', type=str, default='ablation_results.jsonl',
                   help='Results file (default: ablation_results.jsonl)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print commands without executing')
    p.add_argument('--resume', action='store_true',
                   help='Skip (preset, task) pairs already in results file')
    return p.parse_args()


def load_completed(results_file: str) -> set:
    """Load already-completed (preset, task) pairs from results file."""
    completed = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    completed.add((r['preset'], r['task']))
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def print_results_table(results_file: str):
    """Read results file and print a markdown summary table."""
    if not os.path.exists(results_file):
        print("No results file found.")
        return

    results = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not results:
        print("No results found.")
        return

    # Group by task
    tasks = sorted(set(r['task'] for r in results))
    presets = sorted(set(r['preset'] for r in results))

    # Build lookup
    lookup = {}
    for r in results:
        lookup[(r['preset'], r['task'])] = r

    # Print table header
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)

    # One table per task
    for task in tasks:
        print(f"\n### {task}\n")
        print(f"{'Preset':<22} {'Config':<45} {'TrAcc':>7} {'TstAcc':>7} {'Loss':>8}")
        print("-" * 95)

        for preset in presets:
            key = (preset, task)
            if key not in lookup:
                continue
            r = lookup[key]
            print(f"{r['preset']:<22} {r['config']:<45} "
                  f"{r['train_acc']:7.4f} {r['test_acc']:7.4f} {r['loss']:8.4f}")

    # Summary: best preset per task
    print(f"\n### Best test accuracy per task\n")
    print(f"{'Task':<22} {'Best Preset':<22} {'TstAcc':>7}")
    print("-" * 55)
    for task in tasks:
        task_results = [r for r in results if r['task'] == task]
        if task_results:
            best = max(task_results, key=lambda r: r['test_acc'])
            print(f"{task:<22} {best['preset']:<22} {best['test_acc']:7.4f}")

    # Feature contribution table
    print(f"\n### Feature contribution (naja_full - ablated)\n")
    print(f"{'Feature':<22} {'Task':<22} {'Full':>7} {'Ablated':>7} {'Delta':>7} {'Ablated Preset':<22}")
    print("-" * 95)

    comparisons = [
        ('PoPE vs RoPE',     'mamba3_base', 'mamba3_rope'),
        ('Delta rule',       'naja_full', 'mamba3_base'),
        ('PoPE pair (B2)',   'naja_full', 'delta_per_channel'),
        ('Per-ch decay',     'naja_full', 'pope_perp_only'),
        ('Surprise gate',    'surprise',  'naja_full'),
        ('MIMO r=2',         'mimo2',     'naja_full'),
    ]

    for feature, control, ablated in comparisons:
        for task in tasks:
            c = lookup.get((control, task))
            a = lookup.get((ablated, task))
            if c and a:
                delta = c['test_acc'] - a['test_acc']
                sign = '+' if delta >= 0 else ''
                print(f"{feature:<22} {task:<22} {c['test_acc']:7.4f} {a['test_acc']:7.4f} "
                      f"{sign}{delta:6.4f} {ablated:<22}")


def main():
    args = parse_args()

    presets = args.presets or ALL_PRESETS
    tasks = args.tasks or ALL_TASKS
    results_file = os.path.join(SCRIPT_DIR, args.results_file)

    # Validate
    for p in presets:
        if p not in ALL_PRESETS:
            print(f"Unknown preset: {p}. Choices: {ALL_PRESETS}")
            sys.exit(1)
    for t in tasks:
        if t not in ALL_TASKS:
            print(f"Unknown task: {t}. Choices: {ALL_TASKS}")
            sys.exit(1)

    grid = [(p, t) for p in presets for t in tasks]
    completed = load_completed(results_file) if args.resume else set()

    if args.resume and completed:
        grid = [(p, t) for p, t in grid if (p, t) not in completed]
        print(f"Resuming: {len(completed)} completed, {len(grid)} remaining")

    total = len(grid)
    print(f"Ablation grid: {len(presets)} presets x {len(tasks)} tasks = {total} runs")
    print(f"Epochs per run: {args.epochs}")
    print(f"Results file: {results_file}")

    if args.dry_run:
        print("\n--- DRY RUN (commands only) ---\n")
        for i, (preset, task) in enumerate(grid, 1):
            cmd = (f"python {SCRIPT_DIR / 'train_naja.py'} "
                   f"--preset {preset} --task {task} "
                   f"--epochs {args.epochs} --seed {args.seed} "
                   f"--device {args.device} "
                   f"--results_file {results_file} --diag_every 0")
            print(f"[{i}/{total}] {cmd}")
        print(f"\nTotal: {total} runs")
        return

    print()
    t_start = time.time()
    n_done = 0
    n_fail = 0

    for i, (preset, task) in enumerate(grid, 1):
        print(f"[{i}/{total}] preset={preset}  task={task}")

        cmd = [
            sys.executable, str(SCRIPT_DIR / 'train_naja.py'),
            '--preset', preset,
            '--task', task,
            '--epochs', str(args.epochs),
            '--seed', str(args.seed),
            '--device', args.device,
            '--results_file', results_file,
            '--diag_every', '0',  # no charts during ablation
            '--print_every', str(max(args.epochs // 5, 1)),
        ]

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, timeout=600)
            elapsed = time.time() - t0
            if proc.returncode == 0:
                n_done += 1
                print(f"  done in {elapsed:.1f}s\n")
            else:
                n_fail += 1
                print(f"  FAILED (exit code {proc.returncode}) in {elapsed:.1f}s\n")
        except subprocess.TimeoutExpired:
            n_fail += 1
            print(f"  TIMEOUT (>600s)\n")
        except KeyboardInterrupt:
            print("\nInterrupted. Printing partial results.\n")
            break

    total_time = time.time() - t_start
    print(f"\nCompleted: {n_done}/{total}  Failed: {n_fail}  "
          f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # Print summary table
    print_results_table(results_file)


if __name__ == '__main__':
    main()
