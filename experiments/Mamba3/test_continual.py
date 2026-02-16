#!/usr/bin/env python3
"""
Test continual learning on the arithmetic curriculum.

Replay buffer provides prior-stage examples. Optionally uses differential
learning rates (lower LR for earlier layers) — pass --diff_lr to enable.

Tested and rejected:
  - EWC: quadratic penalty makes the model rigid after a few stages.
  - DER++: logit matching loss fights task learning (failed Stage 4,
    a task that otherwise passes in 1 epoch). Also 3x slower.
  - diff_lr: no measurable effect — replay alone prevents forgetting.

Usage:
  python experiments/Mamba3/test_continual.py                 # Replay only (GPU)
  python experiments/Mamba3/test_continual.py --diff_lr       # + diff learning rates
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
from experiments.Mamba3.continual import build_layer_lr_groups
from experiments.Mamba3.train_arithmetic import (
    train_stage, evaluate_result_accuracy, make_lr_lambda,
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_experiment(base_args, device):
    """Run curriculum with replay (+ optional diff_lr). Returns dict of results."""
    args = SimpleNamespace(
        lr=base_args.lr,
        weight_decay=0.01,
        warmup_epochs=4,
        batch_size=64,
        grad_clip=1.0,
        advance_threshold=0.95,
        replay_fraction=0.25,
        print_every=base_args.print_every,
        diff_lr=base_args.diff_lr,
        lr_decay=base_args.lr_decay,
        ewc=False,
        ewc_lambda=0.0,
        der=False,
        der_alpha=0.0,
        der_samples=0,
    )

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
            ewc=None, der=None, param_groups=make_param_groups())
        all_history.extend(history)

        if not passed:
            last_test = history[-1]['test_acc']
            print(f"  Stage {stage} FAILED (test={last_test:.4f})")
            last_stage = stage
            break

        prev_loaders[stage] = (te, data['n_result_tokens'])
        prev_train_seqs.append(data['train_seqs'])
        last_stage = stage

    elapsed = time.perf_counter() - t_start
    last = all_history[-1] if all_history else {}

    # Collect prev-stage accuracies from last record
    prev_accs = {}
    for s in range(1, last_stage + 1):
        key = f'stage{s}_acc'
        if key in last:
            prev_accs[s] = last[key]

    return dict(
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
    p = argparse.ArgumentParser(description='Test curriculum on arithmetic')
    p.add_argument('--target_stage', type=int, default=6)
    p.add_argument('--epochs', type=int, default=50)
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
    p.add_argument('--diff_lr', action='store_true',
                   help='Enable differential learning rates by layer')
    p.add_argument('--print_every', type=int, default=10)
    p.add_argument('--device', type=str, default='auto')
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

    mode = "diff_lr + replay" if base_args.diff_lr else "replay only"
    print(f"Arithmetic Curriculum ({mode})")
    print(f"  Model: {n_params:,} params | {device}")
    print(f"  Target: Stage {base_args.target_stage} | "
          f"{base_args.epochs} epochs/stage | seed={base_args.seed}")

    result = run_experiment(base_args, device)

    # --- Summary ---
    print(f"\n{'='*60}")
    pt_str = '|'.join(f'{x:.2f}' for x in result['test_per_token']) if result['test_per_token'] else 'n/a'
    status = 'PASS' if result['passed'] else f"FAILED at Stage {result['last_stage']}"
    print(f"  {status}  train={result['train_acc']:.4f}  test={result['test_acc']:.4f}  "
          f"per-token=[{pt_str}]  {result['elapsed']:.0f}s  {result['n_epochs']} epochs")

    if result['prev_accs']:
        parts = [f"S{s}={a:.2f}" for s, a in sorted(result['prev_accs'].items())]
        print(f"  Previous stages: {', '.join(parts)}")

    print()


if __name__ == '__main__':
    main()
