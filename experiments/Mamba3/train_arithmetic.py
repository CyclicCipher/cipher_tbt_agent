#!/usr/bin/env python3
"""
Curriculum training script for compositional arithmetic on Mamba3.

Modes:
  Single-stage:  python train_arithmetic.py --stage 2 --epochs 50
  Curriculum:    python train_arithmetic.py --curriculum --target_stage 5
  Direct:        python train_arithmetic.py --stage 5 --epochs 200

Stages (via scratchpad framework):
  1: Query counting — "how many DOTs/TENs?" (n_result=1)
  2: Combined counting — count-up process with STOP (n_result=20)
  3: Single-digit +/- — carry + ones (n_result=2)
  4: Two-digit ± single-digit — column scratchpad (n_result=21)
  5: Two-digit ± two-digit — column scratchpad (n_result=21)

Diagnostics:
  Epiplexity (S_preq): area under loss curve above final loss, per stage.
    High S_preq = rich structure learned. Low S_preq = trivially memorizable.
    Ref: Alemi (2025) "Epiplexity and the Solomonoff Prior"

  Reverse problems (--reverse_fraction): mix in problems where one operand is
    missing and must be inducted from the result. Forces bidirectional understanding.
    Stage 3 only for now. Ref: Alemi (2025), factorization order experiment.

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
from experiments.scratchpad import Vocab, split_problems
from experiments.scratchpad.generators import (
    QueryCountingGenerator, CombinedCountingGenerator,
    SingleDigitArithmeticGenerator, TwoDigitSingleArithmeticGenerator,
    TwoDigitArithmeticGenerator,
)


# ---------------------------------------------------------------------------
# Scratchpad vocab + generators  (replaces arithmetic_tasks imports)
# ---------------------------------------------------------------------------

def _build_vocab():
    """Build deterministic vocab with all tokens for the arithmetic curriculum."""
    v = Vocab()                         # PAD=0, WORK=1, NOTE=2, SEP=3
    for d in range(10): v.add(str(d))   # 4-13
    v.add('+'); v.add('-'); v.add('=')   # 14-16
    v.add('DOT'); v.add('TEN')          # 17-18
    v.add('?')                          # 19  missing operand (reverse problems)
    v.add('STOP')                       # 20  counting terminator
    return v

VOCAB = _build_vocab()
VOCAB_SIZE = len(VOCAB)

def _build_generators(reverse_fraction=0.0):
    """Build stage generators. reverse_fraction applies to arithmetic stages."""
    return {
        1: QueryCountingGenerator(),
        2: CombinedCountingGenerator(),
        3: SingleDigitArithmeticGenerator(reverse_fraction=reverse_fraction),
        4: TwoDigitSingleArithmeticGenerator(),
        5: TwoDigitArithmeticGenerator(),
    }


def decode_tokens(tokens):
    """Compatibility shim: decode token IDs to readable string."""
    return VOCAB.decode_sequence(tokens)


from experiments.Mamba3.continual import EWC, DERPlusPlus, build_layer_lr_groups


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Compositional Arithmetic Curriculum')

    # Mode
    p.add_argument('--stage', type=int, choices=list(range(1, 6)), default=None,
                   help='Single stage to train (direct mode)')
    p.add_argument('--curriculum', action='store_true',
                   help='Curriculum mode: train stages 1 -> target_stage')
    p.add_argument('--target_stage', type=int, choices=list(range(2, 6)), default=5,
                   help='Final stage for curriculum mode (default: 5 = two-digit arithmetic)')

    # Architecture
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_state', type=int, default=64)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--headdim', type=int, default=64)
    p.add_argument('--chunk_size', type=int, default=16)
    p.add_argument('--stable_ssm', action='store_true',
                   help='Use StableSSM A-matrix reparameterization (Wang & Li 2024)')
    p.add_argument('--mhc', action='store_true',
                   help='Enable manifold-constrained hyperconnections (Xiao et al. 2025)')
    p.add_argument('--mhc_n_streams', type=int, default=4,
                   help='mHC expansion rate (number of residual streams, default 4)')

    # Data
    p.add_argument('--n_train', type=int, default=5000)
    p.add_argument('--n_test', type=int, default=1000)
    p.add_argument('--test_fraction', type=float, default=0.2)
    p.add_argument('--seq_len', type=int, default=48)

    # Training
    p.add_argument('--epochs', type=int, default=100,
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
    p.add_argument('--reverse_fraction', type=float, default=0.0,
                   help='Fraction of reverse problems in arithmetic stages (0=off, 0.3=30%%)')

    # Continual learning
    p.add_argument('--diff_lr', action='store_true',
                   help='Enable differential learning rates by layer')
    p.add_argument('--lr_decay', type=float, default=0.5,
                   help='Layer LR decay factor (default 0.5, lower = more protection)')
    p.add_argument('--ewc', action='store_true',
                   help='Enable Elastic Weight Consolidation')
    p.add_argument('--ewc_lambda', type=float, default=400.0,
                   help='EWC penalty weight (default 400)')
    p.add_argument('--der', action='store_true',
                   help='Enable DER++ logit matching')
    p.add_argument('--der_alpha', type=float, default=0.5,
                   help='DER++ MSE weight (default 0.5)')
    p.add_argument('--der_samples', type=int, default=500,
                   help='DER++ samples per stage snapshot (default 500)')

    # Output
    p.add_argument('--results_file', type=str, default=None)

    # Checkpointing
    p.add_argument('--checkpoint_dir', type=str, default=None,
                   help='Directory for stage checkpoints (saves after each passed stage)')
    p.add_argument('--no_resume', action='store_true',
                   help='Start fresh even if checkpoints exist')

    # Performance
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--compile', action='store_true')

    # Misc
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42,
                   help='Training seed (model init, batch order)')
    p.add_argument('--data_seed', type=int, default=42,
                   help='Data split seed (train/test problem partition, always fixed)')
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
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(checkpoint_dir, stage, model, prev_train_seqs, all_history):
    """Save model + replay buffer after a passed stage."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'stage_{stage}.pt')
    torch.save({
        'stage': stage,
        'model_state_dict': model.state_dict(),
        'prev_train_seqs': prev_train_seqs,
        'all_history': all_history,
    }, path)
    print(f"  Checkpoint saved -> {path}")


def load_latest_checkpoint(checkpoint_dir, model):
    """Find and load the highest-stage checkpoint. Returns (start_stage, prev_train_seqs, all_history) or None."""
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith('stage_') and fname.endswith('.pt'):
            try:
                s = int(fname[len('stage_'):-len('.pt')])
                checkpoints.append((s, os.path.join(checkpoint_dir, fname)))
            except ValueError:
                pass
    if not checkpoints:
        return None
    checkpoints.sort()
    best_stage, best_path = checkpoints[-1]
    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Resumed from checkpoint: {best_path} (stage {best_stage} passed)")
    return best_stage, ckpt['prev_train_seqs'], ckpt['all_history']


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

def _build_replay_loader(train_seqs, prev_train_seqs, replay_fraction, batch_size):
    """Build a DataLoader with freshly sampled stratified replay.

    Called each epoch so the model sees different replay samples,
    preventing memorization of a fixed replay buffer.
    """
    if not prev_train_seqs or replay_fraction <= 0:
        return DataLoader(TensorDataset(train_seqs),
                          batch_size=batch_size, shuffle=True, drop_last=True)
    n_replay = max(1, int(len(train_seqs) * replay_fraction))
    per_stage = max(1, n_replay // len(prev_train_seqs))
    replay_parts = []
    for pts in prev_train_seqs:
        perm = torch.randperm(len(pts))[:per_stage]
        replay_parts.append(pts[perm])
    replay_seqs = torch.cat(replay_parts, dim=0)
    combined = torch.cat([train_seqs, replay_seqs], dim=0)
    return DataLoader(TensorDataset(combined),
                      batch_size=batch_size, shuffle=True, drop_last=True)


def train_stage(model, train_seqs, test_loader, device, amp_ctx, scaler,
                n_result_tokens, epochs, args, stage_label="",
                prev_loaders=None, prev_train_seqs=None,
                ewc=None, der=None, param_groups=None):
    """Train one stage. Returns (history, passed) where passed indicates
    whether test accuracy reached the advance_threshold.

    If prev_train_seqs is provided, replay is resampled every epoch
    for better coverage of previous skills.

    Optional CL methods:
      param_groups: optimizer param groups (for differential LR)
      ewc: EWC object (adds quadratic penalty to loss)
      der: DERPlusPlus object (adds logit-matching loss)
    """
    if param_groups is not None:
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(args.warmup_epochs, epochs))

    history = []
    epoch_losses = []  # for epiplexity computation
    passed = False
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()

        # Resample replay every epoch for diverse coverage
        train_loader = _build_replay_loader(
            train_seqs, prev_train_seqs, args.replay_fraction, args.batch_size)

        for (seqs,) in train_loader:
            seqs = seqs.to(device)
            with amp_ctx():
                logits = model(seqs)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, VOCAB_SIZE),
                    seqs[:, 1:].reshape(-1),
                    ignore_index=0,
                )

            # Continual learning penalties
            if ewc is not None:
                loss = loss + ewc.penalty(model)
            if der is not None:
                loss = loss + der.loss(model, device, amp_ctx)

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
        epoch_losses.append(avg_loss)

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

        record['train_per_token'] = [round(x, 4) for x in train_result['per_token']]
        record['test_per_token'] = [round(x, 4) for x in test_result['per_token']]

        # Catastrophic-forgetting check on previous stages
        if prev_loaders:
            for s, (ldr, nr) in prev_loaders.items():
                prev_result = evaluate_result_accuracy(model, ldr, device, amp_ctx, nr)
                record[f'stage{s}_acc'] = round(prev_result['exact'], 4)

        history.append(record)

        if epoch % args.print_every == 0 or epoch == 1 or epoch == epochs:
            # Per-token diagnostic (always shown, even for n_result=1)
            tr_tok = '|'.join(f'{x:.2f}' for x in train_result['per_token'])
            te_tok = '|'.join(f'{x:.2f}' for x in test_result['per_token'])
            tr_tok_str = f" [{tr_tok}]"
            te_tok_str = f" [{te_tok}]"

            prev_str = ""
            if prev_loaders:
                parts = [f"S{s}={record[f'stage{s}_acc']:.2f}"
                         for s in sorted(prev_loaders)]
                prev_str = f"  prev=[{', '.join(parts)}]"
            print(f"  [{stage_label}] ep {epoch:3d}  loss={avg_loss:.4f}  "
                  f"train={train_acc:.4f}{tr_tok_str}  "
                  f"test={test_acc:.4f}{te_tok_str}  "
                  f"{ep_s:.1f}s{prev_str}")

        if test_acc >= args.advance_threshold:
            print(f"  [{stage_label}] PASSED ep {epoch}  "
                  f"(test={test_acc:.4f} >= {args.advance_threshold})")
            passed = True
            break

    # Epiplexity: area under loss curve above final loss
    # S_preq = sum(l_i - l_final) for all epochs
    # High S_preq = model slowly extracted rich structure (good)
    # Low S_preq = stage trivially memorizable or too simple (warning)
    # Ref: Alemi (2025) "Epiplexity and the Solomonoff Prior"
    if epoch_losses:
        l_final = epoch_losses[-1]
        epiplexity = sum(l - l_final for l in epoch_losses)
        n_epochs_run = len(epoch_losses)
        print(f"  [{stage_label}] Epiplexity: S_preq={epiplexity:.2f}  "
              f"(final_loss={l_final:.4f}, {n_epochs_run} epochs)")
        # Attach to last history record for downstream analysis
        if history:
            history[-1]['epiplexity'] = round(epiplexity, 4)
            history[-1]['final_loss'] = round(l_final, 4)
            history[-1]['n_epochs_run'] = n_epochs_run

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
        stable_ssm=args.stable_ssm,
        use_mhc=args.mhc,
        mhc_n_streams=args.mhc_n_streams,
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

    mhc_str = f" | mHC(n={config.mhc_n_streams})" if config.use_mhc else ""
    print(f"Mamba3LM | {n_params:,} params | {device}"
          f" | d={config.d_model} N={config.d_state}"
          f" | L={config.n_layer} H={config.nheads}"
          f" | chunk={config.chunk_size} seq={args.seq_len}{mhc_str}")

    all_history = []
    generators = _build_generators(reverse_fraction=args.reverse_fraction)

    if args.reverse_fraction > 0:
        print(f"Reverse problems: {args.reverse_fraction:.0%} of arithmetic stage samples")

    # --- CL method setup ---
    ewc_obj = EWC(lambda_ewc=args.ewc_lambda) if args.ewc else None
    der_obj = DERPlusPlus(alpha=args.der_alpha, n_snapshot=args.der_samples) if args.der else None

    cl_tags = []
    if args.diff_lr: cl_tags.append('diff_lr')
    if args.ewc: cl_tags.append(f'ewc(λ={args.ewc_lambda})')
    if args.der: cl_tags.append(f'der(α={args.der_alpha})')
    if cl_tags:
        print(f"CL methods: {', '.join(cl_tags)}")

    # --- Helper to build loaders for a stage ---
    def make_loaders(stage):
        gen = generators[stage]
        data = split_problems(gen, VOCAB, n_train=args.n_train, n_test=args.n_test,
                              test_fraction=args.test_fraction,
                              seq_len=args.seq_len, seed=args.data_seed)
        tr = DataLoader(TensorDataset(data['train_seqs']),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
        te = DataLoader(TensorDataset(data['test_seqs']),
                        batch_size=args.batch_size, shuffle=False)
        return tr, te, data

    def make_param_groups():
        if args.diff_lr:
            return build_layer_lr_groups(model, args.lr, args.lr_decay, args.weight_decay)
        return None

    if args.curriculum:
        print(f"\nCurriculum 1 -> {args.target_stage}  "
              f"(advance >= {args.advance_threshold})"
              f"  replay={args.replay_fraction}\n")
        prev_loaders = {}
        prev_train_seqs = []  # accumulate training tensors for replay
        start_stage = 1

        # Resume from checkpoint if available
        if args.checkpoint_dir and not args.no_resume:
            resumed = load_latest_checkpoint(args.checkpoint_dir, model)
            if resumed is not None:
                last_passed, prev_train_seqs, all_history = resumed
                start_stage = last_passed + 1
                # Rebuild prev_loaders for catastrophic-forgetting checks
                for s in range(1, last_passed + 1):
                    _, te, data = make_loaders(s)
                    prev_loaders[s] = (te, data['n_result_tokens'])
                print(f"  Skipping stages 1-{last_passed} (already passed)\n")

        for stage in range(start_stage, args.target_stage + 1):
            _, te, data = make_loaders(stage)

            # Replay info for logging
            if prev_train_seqs and args.replay_fraction > 0:
                n_replay = max(1, int(len(data['train_seqs']) * args.replay_fraction))
                per_stage = max(1, n_replay // len(prev_train_seqs))
                print(f"Stage {stage}: {data['n_train_specs']} train / "
                      f"{data['n_test_specs']} test problems"
                      f" (+{per_stage * len(prev_train_seqs)} replay/epoch, "
                      f"{per_stage}/stage, resampled)")
            else:
                print(f"Stage {stage}: {data['n_train_specs']} train / "
                      f"{data['n_test_specs']} test problems")

            # Print a few sample sequences for sanity
            for j in range(min(3, len(data['train_seqs']))):
                print(f"  sample: {decode_tokens(data['train_seqs'][j])}")

            # Replay is resampled each epoch inside train_stage
            history, passed = train_stage(
                model, data['train_seqs'], te, device, amp_ctx, scaler,
                data['n_result_tokens'], args.epochs, args,
                stage_label=f"S{stage}", prev_loaders=prev_loaders,
                prev_train_seqs=prev_train_seqs,
                ewc=ewc_obj, der=der_obj, param_groups=make_param_groups())
            all_history.extend(history)

            if not passed:
                last_test = history[-1]['test_acc']
                print(f"  Stage {stage} FAILED — test={last_test:.4f} "
                      f"< {args.advance_threshold}.  Curriculum halted.\n")
                break

            prev_loaders[stage] = (te, data['n_result_tokens'])
            prev_train_seqs.append(data['train_seqs'])

            # Register with CL methods after a stage passes
            if ewc_obj is not None:
                ewc_obj.register_stage(model, data['train_seqs'],
                                       device, amp_ctx)
            if der_obj is not None:
                der_obj.register_stage(model, data['train_seqs'],
                                       device, amp_ctx,
                                       data['n_result_tokens'])

            # Save checkpoint after each passed stage
            if args.checkpoint_dir:
                save_checkpoint(args.checkpoint_dir, stage, model,
                                prev_train_seqs, all_history)
            print()
    else:
        stage = args.stage
        _, te, data = make_loaders(stage)
        print(f"\nStage {stage}: {data['n_train_specs']} train / "
              f"{data['n_test_specs']} test problems")
        for j in range(min(3, len(data['train_seqs']))):
            print(f"  sample: {decode_tokens(data['train_seqs'][j])}")
        print()

        history, passed = train_stage(
            model, data['train_seqs'], te, device, amp_ctx, scaler,
            data['n_result_tokens'], args.epochs, args,
            stage_label=f"S{stage}",
            ewc=ewc_obj, der=der_obj, param_groups=make_param_groups())
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
