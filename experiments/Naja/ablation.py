#!/usr/bin/env python3
"""
Ablation testing runner for Naja.

Implements the ablation protocol from DESIGN.md:
  - Multiple configurations (base Mamba3 vs each Naja feature)
  - Multiple seeds (3x per configuration)
  - Multiple tasks (stage 1b, stage 2, associative recall, parity,
    multi-scale memory, permutation tracking)
  - Results table with mean +/- std

Usage:
  # Run full ablation suite
  python ablation.py

  # Run specific task only
  python ablation.py --tasks parity associative_recall

  # Run specific configs only
  python ablation.py --configs mamba3_base naja_full delta_only

  # Quick test (1 seed, fewer epochs)
  python ablation.py --seeds 1 --epochs 10

  # Custom output file
  python ablation.py --output results/ablation_2026-02-14.txt

Do NOT run on CPU (Mistake #36). This script is designed for GPU.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.Naja.train_naja import train, PRESETS
from experiments.Naja.tasks import ABLATION_TASKS


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

# Which configs to test (maps to PRESETS in train_naja.py)
DEFAULT_CONFIGS = [
    'mamba3_base',     # Control: no Naja features
    'delta_only',      # +delta rule only
    'pope_perp_only',  # +delta + PoPE perp
    'per_channel_only',  # +per-channel decay only
    'naja_full',       # All Naja features
    'stable_reparam',  # Full + StableSSM
    'surprise',        # Full + surprise gating
    'mimo2',           # Full + MIMO r=2
]

# Which tasks to test
DEFAULT_TASKS = [
    # (name, type, description)
    ('stage_1b', 'stage', 'Sanity check — all configs should pass'),
    ('stage_2', 'stage', 'Multi-rule generalization — the hard test'),
    ('associative_recall', 'task', 'Tests delta rule'),
    ('multi_scale', 'task', 'Tests per-channel decay'),
    ('permutation_3', 'task', 'Tests full architecture (3 elements)'),
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Naja ablation testing')
    p.add_argument('--configs', nargs='+', default=None,
                   help='Config presets to test (default: all)')
    p.add_argument('--tasks', nargs='+', default=None,
                   help='Tasks to test (default: all)')
    p.add_argument('--seeds', type=int, default=3,
                   help='Number of seeds per config (default: 3)')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override epoch count (default: task-dependent)')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--output', type=str, default=None,
                   help='Output file for results')
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--compile', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Run one configuration × task × seed
# ---------------------------------------------------------------------------

def make_train_args(config_name: str, task_name: str, task_type: str,
                    seed: int, epochs: int, device: str,
                    no_amp: bool, do_compile: bool):
    """Build an argparse.Namespace mimicking train_naja.py CLI args."""
    args = argparse.Namespace()

    # Stage/task
    if task_type == 'stage':
        stage_code = task_name.replace('stage_', '')
        args.stage = stage_code
        args.task = None
    else:
        args.stage = None
        args.task = task_name

    # Config preset
    args.preset = config_name

    # Architecture (defaults)
    args.d_model = 128
    args.d_state = 64
    args.n_layer = 4
    args.headdim = 64
    args.expand = 2
    args.mimo_rank = None  # use preset

    # Feature toggles (None = use preset)
    args.use_delta_rule = None
    args.no_delta_rule = False
    args.use_pope_perp = None
    args.no_pope_perp = False
    args.per_channel_decay = None
    args.no_per_channel_decay = False
    args.stable_reparam = None
    args.use_surprise_gate = None
    args.use_chunkwise = False
    args.chunk_size = 64

    # VRAM-aware: auto-enable chunkwise on small cards
    small_vram = False
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb <= 6.0:
            args.use_chunkwise = True
            args.chunk_size = 16
            small_vram = True
    except Exception:
        pass

    # Data
    args.seq_len = 64
    args.vocab_size = 16
    args.n_train = 5000
    args.n_test = 1000
    args.n_examples = 5
    args.n_rules = 5

    # Task-specific overrides
    if task_name == 'stage_2':
        args.n_train = 10000
        args.n_test = 2000
        args.batch_size = 32 if small_vram else 128
        args.seq_len = max(2 * args.n_examples + 2, 16)
    elif task_name in ABLATION_TASKS:
        info = ABLATION_TASKS[task_name]
        da = info['default_args']
        args.vocab_size = da.get('vocab_size', 16)
        args.seq_len = da.get('seq_len', 64)
    else:
        pass  # stage 1a/1b/1c: defaults are fine

    # Training
    task_epochs = {
        'stage_1b': 30, 'stage_1a': 30, 'stage_1c': 30,
        'stage_2': 50,
        'associative_recall': 50,
        'multi_scale': 50, 'permutation_3': 50, 'permutation_4': 80,
    }
    args.epochs = epochs if epochs is not None else task_epochs.get(task_name, 30)
    if not hasattr(args, 'batch_size'):
        args.batch_size = 8 if small_vram else 32
    args.lr = 1e-3
    args.warmup_epochs = 4
    args.w_clip = 1.0
    args.weight_decay = 0.01

    # Performance
    args.no_amp = no_amp
    args.compile = do_compile
    args.profile = False

    # Misc
    args.device = device
    args.seed = seed
    args.print_every = max(args.epochs // 5, 1)

    return args


# ---------------------------------------------------------------------------
# Main ablation loop
# ---------------------------------------------------------------------------

def run_ablation(args):
    configs = args.configs or [c for c, *_ in
                                [(name,) for name in DEFAULT_CONFIGS]]
    if args.configs is None:
        configs = [name for name in DEFAULT_CONFIGS]
    else:
        configs = args.configs

    if args.tasks is not None:
        tasks = [(t, 'task' if t in ABLATION_TASKS else 'stage',
                  '') for t in args.tasks]
    else:
        tasks = DEFAULT_TASKS

    seeds = list(range(42, 42 + args.seeds))

    # Results storage: {(config, task): [result_dicts]}
    results = {}

    total_runs = len(configs) * len(tasks) * len(seeds)
    print(f"=== Naja Ablation Suite ===")
    print(f"Configs: {configs}")
    print(f"Tasks: {[t[0] for t in tasks]}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {total_runs}")
    print()

    run_idx = 0
    t_start = time.perf_counter()

    for task_name, task_type, task_desc in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        if task_desc:
            print(f"  {task_desc}")
        print(f"{'='*60}")

        for config_name in configs:
            key = (config_name, task_name)
            results[key] = []

            for seed in seeds:
                run_idx += 1
                print(f"\n--- Run {run_idx}/{total_runs}: "
                      f"{config_name} x {task_name} x seed={seed} ---")

                train_args = make_train_args(
                    config_name, task_name, task_type, seed,
                    args.epochs, args.device, args.no_amp, args.compile,
                )

                try:
                    result = train(train_args)
                    results[key].append(result)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    results[key].append({
                        'train_acc': 0.0, 'test_acc': 0.0,
                        'loss': float('inf'), 'error': str(e),
                    })

    # --- Print results table ---
    print(f"\n\n{'='*80}")
    print(f"ABLATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<20s} {'Task':<20s} {'Train Acc':>12s} {'Test Acc':>12s}")
    print(f"{'-'*20} {'-'*20} {'-'*12} {'-'*12}")

    summary_lines = []
    for task_name, task_type, task_desc in tasks:
        for config_name in configs:
            key = (config_name, task_name)
            runs = results[key]
            if not runs:
                continue
            train_accs = [r['train_acc'] for r in runs if 'error' not in r]
            test_accs = [r['test_acc'] for r in runs if 'error' not in r]

            if train_accs:
                tr_mean = sum(train_accs) / len(train_accs)
                tr_std = (sum((x - tr_mean)**2 for x in train_accs)
                          / max(len(train_accs) - 1, 1)) ** 0.5
                te_mean = sum(test_accs) / len(test_accs)
                te_std = (sum((x - te_mean)**2 for x in test_accs)
                          / max(len(test_accs) - 1, 1)) ** 0.5
                tr_str = f"{tr_mean:.4f}+/-{tr_std:.4f}"
                te_str = f"{te_mean:.4f}+/-{te_std:.4f}"
            else:
                tr_str = "FAILED"
                te_str = "FAILED"
                tr_mean = te_mean = 0.0

            line = f"{config_name:<20s} {task_name:<20s} {tr_str:>12s} {te_str:>12s}"
            print(line)
            summary_lines.append(line)

        print()  # blank line between tasks

    elapsed = time.perf_counter() - t_start
    print(f"Total time: {elapsed:.1f}s")

    # --- Save to file ---
    output_path = args.output
    if output_path is None:
        os.makedirs('results', exist_ok=True)
        output_path = f"results/ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(output_path, 'w') as f:
        f.write(f"Naja Ablation Results\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Configs: {configs}\n")
        f.write(f"Tasks: {[t[0] for t in tasks]}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Total time: {elapsed:.1f}s\n\n")
        for line in summary_lines:
            f.write(line + '\n')
        f.write(f"\n\nRaw results:\n")
        # Serialize results (convert tuple keys to strings)
        serializable = {f"{k[0]}|{k[1]}": v for k, v in results.items()}
        f.write(json.dumps(serializable, indent=2, default=str))

    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    run_ablation(args)
