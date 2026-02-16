#!/usr/bin/env python3
"""
Test continual learning methods on the arithmetic curriculum.

Runs 3 configurations to Stage 6, comparing catastrophic forgetting:
  1. diff_lr  -- Differential learning rates by layer
  2. der      -- DER++ logit matching
  3. both     -- diff_lr + DER++ combined

EWC was tested and found counterproductive: its quadratic penalty makes
the model rigid after a few stages, causing training to fail. Removed.

Usage:
  python experiments/Mamba3/test_continual.py                 # Full test (GPU)
  python experiments/Mamba3/test_continual.py --epochs 3      # Quick sanity (CPU)
  python experiments/Mamba3/test_continual.py --target_stage 8 # Test further

Do NOT run full training on CPU (Mistake #36).
Quick mode (--epochs 3) is for verifying code correctness only.
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.Mamba3.mamba3_block import Mamba3Config, Mamba3LM
from experiments.Mamba3.arithmetic_tasks import (
    VOCAB_SIZE, get_stage_data, decode_tokens,
)
from experiments.Mamba3.continual import DERPlusPlus, build_layer_lr_groups
from experiments.Mamba3.train_arithmetic import (
    train_stage, evaluate_result_accuracy, make_lr_lambda,
)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    'diff_lr': dict(diff_lr=True, ewc=False, der=False),
    'der':     dict(diff_lr=False, ewc=False, der=True),
    'both':    dict(diff_lr=True, ewc=False, der=True),
}


def make_args(overrides, base_args):
    """Create a SimpleNamespace with training args, applying overrides."""
    d = dict(
        lr=base_args.lr,
        weight_decay=0.01,
        warmup_epochs=4,
        batch_size=64,
        grad_clip=1.0,
        advance_threshold=0.95,
        replay_fraction=0.25,
        print_every=base_args.print_every,
        # CL defaults
        diff_lr=False,
        lr_decay=base_args.lr_decay,
        ewc=False,
        ewc_lambda=0.0,
        der=False,
        der_alpha=base_args.der_alpha,
        der_samples=base_args.der_samples,
    )
    d.update(overrides)
    return SimpleNamespace(**d)


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_experiment(name, config_overrides, base_args, device):
    """Run one curriculum experiment. Returns dict of results."""
    args = make_args(config_overrides, base_args)

    torch.manual_seed(base_args.seed)

    cfg = Mamba3Config(
        d_model=base_args.d_model, d_state=base_args.d_state,
        n_layer=base_args.n_layer, headdim=base_args.headdim,
        chunk_size=base_args.chunk_size,
    )
    model = Mamba3LM(cfg, VOCAB_SIZE).to(device)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    amp_ctx = (lambda: torch.amp.autocast('cuda', dtype=torch.float16)) if use_amp else nullcontext

    # CL objects
    der_obj = DERPlusPlus(alpha=args.der_alpha, n_snapshot=args.der_samples) if args.der else None

    def make_loaders(stage):
        data = get_stage_data(stage, n_train=base_args.n_train,
                              n_test=base_args.n_test,
                              test_fraction=0.2,
                              seq_len=base_args.seq_len,
                              seed=base_args.data_seed)
        te = DataLoader(TensorDataset(data['test_seqs']),
                        batch_size=64, shuffle=False)
        return te, data

    def make_param_groups():
        if args.diff_lr:
            return build_layer_lr_groups(model, args.lr, args.lr_decay,
                                         args.weight_decay)
        return None

    tags = []
    if args.diff_lr: tags.append('diff_lr')
    if args.der: tags.append('der')
    tag_str = '+'.join(tags) if tags else 'none'

    print(f"\n{'='*60}")
    print(f"  Config: {name} ({tag_str})")
    print(f"{'='*60}")

    prev_loaders = {}
    prev_train_seqs = []
    last_stage = 0
    all_history = []
    t_start = time.perf_counter()

    for stage in range(1, base_args.target_stage + 1):
        te, data = make_loaders(stage)

        # Replay info
        if prev_train_seqs and args.replay_fraction > 0:
            n_replay = max(1, int(len(data['train_seqs']) * args.replay_fraction))
            per_stage = max(1, n_replay // len(prev_train_seqs))
            print(f"  Stage {stage}: {data['n_train_problems']} train / "
                  f"{data['n_test_problems']} test "
                  f"(+{per_stage * len(prev_train_seqs)} replay)")
        else:
            print(f"  Stage {stage}: {data['n_train_problems']} train / "
                  f"{data['n_test_problems']} test")

        history, passed = train_stage(
            model, data['train_seqs'], te, device, amp_ctx, scaler,
            data['n_result_tokens'], base_args.epochs, args,
            stage_label=f"S{stage}", prev_loaders=prev_loaders,
            prev_train_seqs=prev_train_seqs,
            ewc=None, der=der_obj, param_groups=make_param_groups())
        all_history.extend(history)

        if not passed:
            last_test = history[-1]['test_acc']
            print(f"  Stage {stage} FAILED (test={last_test:.4f})")
            last_stage = stage
            break

        prev_loaders[stage] = (te, data['n_result_tokens'])
        prev_train_seqs.append(data['train_seqs'])
        last_stage = stage

        # Register with CL methods
        if der_obj is not None:
            der_obj.register_stage(model, data['train_seqs'],
                                   device, amp_ctx,
                                   data['n_result_tokens'])

    elapsed = time.perf_counter() - t_start
    last = all_history[-1] if all_history else {}

    # Collect prev-stage accuracies from last record
    prev_accs = {}
    for s in range(1, last_stage + 1):
        key = f'stage{s}_acc'
        if key in last:
            prev_accs[s] = last[key]

    return dict(
        name=name,
        last_stage=last_stage,
        passed=last_stage == base_args.target_stage,
        train_acc=last.get('train_acc', 0),
        test_acc=last.get('test_acc', 0),
        test_per_token=last.get('test_per_token', []),
        prev_accs=prev_accs,
        elapsed=elapsed,
        n_epochs=len(all_history),
        history=all_history,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Test CL methods on arithmetic curriculum')
    p.add_argument('--target_stage', type=int, default=6)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_seed', type=int, default=42)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_state', type=int, default=64)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--headdim', type=int, default=64)
    p.add_argument('--chunk_size', type=int, default=16)
    p.add_argument('--n_train', type=int, default=5000)
    p.add_argument('--n_test', type=int, default=1000)
    p.add_argument('--seq_len', type=int, default=48)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_decay', type=float, default=0.5)
    p.add_argument('--der_alpha', type=float, default=0.5)
    p.add_argument('--der_samples', type=int, default=500)
    p.add_argument('--print_every', type=int, default=10)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--configs', type=str, nargs='+',
                   choices=list(CONFIGS.keys()),
                   default=list(CONFIGS.keys()),
                   help='Which configs to test (default: all)')
    base_args = p.parse_args()

    if base_args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(base_args.device)

    n_params = sum(p.numel() for p in
                   Mamba3LM(Mamba3Config(
                       d_model=base_args.d_model, d_state=base_args.d_state,
                       n_layer=base_args.n_layer, headdim=base_args.headdim),
                       VOCAB_SIZE).parameters())

    print(f"Continual Learning Comparison")
    print(f"  Model: {n_params:,} params | {device}")
    print(f"  Target: Stage {base_args.target_stage} | "
          f"{base_args.epochs} epochs/stage | seed={base_args.seed}")
    print(f"  Configs: {', '.join(base_args.configs)}")

    results = []
    for cfg_name in base_args.configs:
        result = run_experiment(cfg_name, CONFIGS[cfg_name], base_args, device)
        results.append(result)

    # --- Comparison table ---
    print(f"\n{'='*70}")
    print(f"  RESULTS COMPARISON (target=Stage {base_args.target_stage})")
    print(f"{'='*70}")
    print(f"  {'Config':<10} {'Last':>5} {'Train':>7} {'Test':>7} "
          f"{'Per-Token':<15} {'Time':>6} {'Epochs':>6}")
    print(f"  {'-'*10} {'-'*5} {'-'*7} {'-'*7} {'-'*15} {'-'*6} {'-'*6}")

    for r in results:
        pt_str = '|'.join(f'{x:.2f}' for x in r['test_per_token']) if r['test_per_token'] else 'n/a'
        status = 'PASS' if r['passed'] else f"S{r['last_stage']}"
        print(f"  {r['name']:<10} {status:>5} {r['train_acc']:>7.4f} "
              f"{r['test_acc']:>7.4f} [{pt_str:^13}] "
              f"{r['elapsed']:>5.0f}s {r['n_epochs']:>6d}")

    # Prev-stage stability for the last recorded epoch
    print(f"\n  Previous-stage accuracy (last epoch):")
    for r in results:
        if r['prev_accs']:
            parts = [f"S{s}={a:.2f}" for s, a in sorted(r['prev_accs'].items())]
            print(f"    {r['name']:<10} {', '.join(parts)}")

    print()


if __name__ == '__main__':
    main()
