#!/usr/bin/env python3
"""
Curriculum training script for compositional arithmetic on Mamba3.

Modes:
  Single-stage:  python train_arithmetic.py --stage 4 --epochs 50
  Curriculum:    python train_arithmetic.py --curriculum --target_stage 6
  Direct:        python train_arithmetic.py --stage 6 --epochs 200

See CONTINUATION.md for experimental design.
Do NOT run full training on CPU (Mistake #36).
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.Mamba3.mamba3_block import Mamba3Config, Mamba3LM
from experiments.Mamba3.arithmetic_tasks import (
    VOCAB_SIZE, get_stage_data, decode_tokens,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Compositional Arithmetic Curriculum')

    # Mode
    p.add_argument('--stage', type=int, choices=list(range(1, 11)), default=None,
                   help='Single stage to train (direct mode)')
    p.add_argument('--curriculum', action='store_true',
                   help='Curriculum mode: train stages 1 -> target_stage')
    p.add_argument('--target_stage', type=int, choices=list(range(2, 11)), default=9,
                   help='Final stage for curriculum mode (default: 9 = two-digit arithmetic)')

    # Architecture
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_state', type=int, default=64)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--headdim', type=int, default=64)
    p.add_argument('--chunk_size', type=int, default=16)

    # Data
    p.add_argument('--n_train', type=int, default=5000)
    p.add_argument('--n_test', type=int, default=1000)
    p.add_argument('--test_fraction', type=float, default=0.2)
    p.add_argument('--seq_len', type=int, default=32)

    # Training
    p.add_argument('--epochs', type=int, default=50,
                   help='Max epochs per stage (curriculum) or total (direct)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup_epochs', type=int, default=4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--advance_threshold', type=float, default=0.95,
                   help='Test accuracy to advance in curriculum mode')
    p.add_argument('--replay_fraction', type=float, default=0.25,
                   help='Fraction of previous-stage data mixed into current stage (0=off)')

    # Output
    p.add_argument('--results_file', type=str, default=None)

    # Performance
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--compile', action='store_true')

    # Misc
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--print_every', type=int, default=10)

    args = p.parse_args()
    if args.stage is None and not args.curriculum:
        args.stage = 1
    return args


# ---------------------------------------------------------------------------
# LR schedule (cosine with warmup, same as train_naja.py)
# ---------------------------------------------------------------------------

def make_lr_lambda(warmup: int, total: int):
    def lr_lambda(epoch):
        if epoch < warmup:
            return 0.2 + 0.8 * epoch / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_result_accuracy(model, loader, device, amp_ctx, n_result_tokens):
    """Exact-match and per-token accuracy on the last n_result_tokens positions.

    Per Mistake #41: prediction for position p uses logits[:, p-1],
    which has NOT seen the token at position p.

    Returns dict:
        exact: float     — fraction of samples where ALL result tokens match
        per_token: list   — per-position accuracy (length n_result_tokens)
    """
    model.eval()
    correct = total = 0
    token_correct = [0] * n_result_tokens
    with torch.no_grad():
        for (seqs,) in loader:
            seqs = seqs.to(device)
            with amp_ctx():
                logits = model(seqs)
            match = torch.ones(seqs.shape[0], dtype=torch.bool, device=device)
            for k in range(n_result_tokens):
                pos = -(n_result_tokens - k)      # result token position
                pred = logits[:, pos - 1].argmax(-1)
                tok_match = (pred == seqs[:, pos])
                match &= tok_match
                token_correct[k] += tok_match.sum().item()
            correct += match.sum().item()
            total += seqs.shape[0]
    exact = correct / max(total, 1)
    per_token = [tc / max(total, 1) for tc in token_correct]
    return dict(exact=exact, per_token=per_token)


# ---------------------------------------------------------------------------
# Single-stage training loop
# ---------------------------------------------------------------------------

def train_stage(model, train_loader, test_loader, device, amp_ctx, scaler,
                n_result_tokens, epochs, args, stage_label="",
                prev_loaders=None):
    """Train one stage. Returns (history, passed) where passed indicates
    whether test accuracy reached the advance_threshold."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(args.warmup_epochs, epochs))

    history = []
    passed = False
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()

        for (seqs,) in train_loader:
            seqs = seqs.to(device)
            with amp_ctx():
                logits = model(seqs)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, VOCAB_SIZE),
                    seqs[:, 1:].reshape(-1),
                    ignore_index=0,
                )

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        train_result = evaluate_result_accuracy(
            model, train_loader, device, amp_ctx, n_result_tokens)
        test_result = evaluate_result_accuracy(
            model, test_loader, device, amp_ctx, n_result_tokens)

        train_acc = train_result['exact']
        test_acc = test_result['exact']

        ep_s = time.perf_counter() - t0

        record = dict(epoch=epoch, stage=stage_label,
                      loss=round(avg_loss, 4),
                      train_acc=round(train_acc, 4),
                      test_acc=round(test_acc, 4),
                      time=round(ep_s, 1))

        if n_result_tokens > 1:
            record['train_per_token'] = [round(x, 4) for x in train_result['per_token']]
            record['test_per_token'] = [round(x, 4) for x in test_result['per_token']]

        # Catastrophic-forgetting check on previous stages
        if prev_loaders:
            for s, (ldr, nr) in prev_loaders.items():
                prev_result = evaluate_result_accuracy(model, ldr, device, amp_ctx, nr)
                record[f'stage{s}_acc'] = round(prev_result['exact'], 4)

        history.append(record)

        if epoch % args.print_every == 0 or epoch == 1 or epoch == epochs:
            # Per-token diagnostic for multi-token results (test only)
            tok_str = ""
            if n_result_tokens > 1:
                te_tok = '|'.join(f'{x:.2f}' for x in test_result['per_token'])
                tok_str = f" [{te_tok}]"

            prev_str = ""
            if prev_loaders:
                parts = [f"S{s}={record[f'stage{s}_acc']:.2f}"
                         for s in sorted(prev_loaders)]
                prev_str = f"  prev=[{', '.join(parts)}]"
            print(f"  [{stage_label}] ep {epoch:3d}  loss={avg_loss:.4f}  "
                  f"train={train_acc:.4f}  test={test_acc:.4f}{tok_str}  "
                  f"{ep_s:.1f}s{prev_str}")

        if test_acc >= args.advance_threshold:
            print(f"  [{stage_label}] PASSED ep {epoch}  "
                  f"(test={test_acc:.4f} >= {args.advance_threshold})")
            passed = True
            break

    return history, passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    config = Mamba3Config(
        d_model=args.d_model, d_state=args.d_state,
        n_layer=args.n_layer, headdim=args.headdim,
        chunk_size=args.chunk_size,
    )
    model = Mamba3LM(config, VOCAB_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    use_amp = not args.no_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    amp_ctx = (lambda: torch.amp.autocast('cuda', dtype=torch.float16)) if use_amp else nullcontext

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile: ON")
        except Exception as e:
            print(f"torch.compile: FAILED ({e})")

    print(f"Mamba3LM | {n_params:,} params | {device}"
          f" | d={config.d_model} N={config.d_state}"
          f" | L={config.n_layer} H={config.nheads}"
          f" | chunk={config.chunk_size} seq={args.seq_len}")

    all_history = []

    # --- Helper to build loaders for a stage ---
    def make_loaders(stage):
        data = get_stage_data(stage, n_train=args.n_train, n_test=args.n_test,
                              test_fraction=args.test_fraction,
                              seq_len=args.seq_len, seed=args.seed)
        tr = DataLoader(TensorDataset(data['train_seqs']),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
        te = DataLoader(TensorDataset(data['test_seqs']),
                        batch_size=args.batch_size, shuffle=False)
        return tr, te, data

    if args.curriculum:
        print(f"\nCurriculum 1 -> {args.target_stage}  "
              f"(advance >= {args.advance_threshold})"
              f"  replay={args.replay_fraction}\n")
        prev_loaders = {}
        prev_train_seqs = []  # accumulate training tensors for replay

        for stage in range(1, args.target_stage + 1):
            tr, te, data = make_loaders(stage)

            # Experience replay: mix previous-stage samples into train loader
            if prev_train_seqs and args.replay_fraction > 0:
                all_prev = torch.cat(prev_train_seqs, dim=0)
                n_replay = max(1, int(len(data['train_seqs']) * args.replay_fraction))
                perm = torch.randperm(len(all_prev))[:n_replay]
                replay_seqs = all_prev[perm]
                combined = torch.cat([data['train_seqs'], replay_seqs], dim=0)
                tr = DataLoader(TensorDataset(combined),
                                batch_size=args.batch_size, shuffle=True,
                                drop_last=True)
                print(f"Stage {stage}: {data['n_train_problems']} train / "
                      f"{data['n_test_problems']} test problems"
                      f" (+{n_replay} replay)")
            else:
                print(f"Stage {stage}: {data['n_train_problems']} train / "
                      f"{data['n_test_problems']} test problems")

            # Print a few sample sequences for sanity
            for j in range(min(3, len(data['train_seqs']))):
                print(f"  sample: {decode_tokens(data['train_seqs'][j])}")

            history, passed = train_stage(
                model, tr, te, device, amp_ctx, scaler,
                data['n_result_tokens'], args.epochs, args,
                stage_label=f"S{stage}", prev_loaders=prev_loaders)
            all_history.extend(history)

            if not passed:
                last_test = history[-1]['test_acc']
                print(f"  Stage {stage} FAILED — test={last_test:.4f} "
                      f"< {args.advance_threshold}.  Curriculum halted.\n")
                break

            prev_loaders[stage] = (te, data['n_result_tokens'])
            prev_train_seqs.append(data['train_seqs'])
            print()
    else:
        stage = args.stage
        tr, te, data = make_loaders(stage)
        print(f"\nStage {stage}: {data['n_train_problems']} train / "
              f"{data['n_test_problems']} test problems")
        for j in range(min(3, len(data['train_seqs']))):
            print(f"  sample: {decode_tokens(data['train_seqs'][j])}")
        print()

        history, passed = train_stage(
            model, tr, te, device, amp_ctx, scaler,
            data['n_result_tokens'], args.epochs, args,
            stage_label=f"S{stage}")
        all_history.extend(history)

    # --- Save results ---
    if args.results_file:
        result = dict(
            mode='curriculum' if args.curriculum else 'direct',
            target_stage=args.target_stage if args.curriculum else args.stage,
            n_params=n_params, seed=args.seed,
            history=all_history,
        )
        with open(args.results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        print(f"Results -> {args.results_file}")

    # --- Final summary ---
    if all_history:
        last = all_history[-1]
        print(f"\nDone. Last: {last['stage']}  "
              f"train={last['train_acc']:.4f}  test={last['test_acc']:.4f}")


if __name__ == '__main__':
    main()
