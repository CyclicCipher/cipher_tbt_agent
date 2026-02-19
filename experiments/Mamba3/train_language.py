#!/usr/bin/env python3
"""
Curriculum training script for syntax parsing on Mamba3 + WikiText-2.

Modes:
  Single-stage:  python train_language.py --stage 2 --epochs 50
  Curriculum:    python train_language.py --curriculum --target_stage 5
  Direct:        python train_language.py --stage 5 --epochs 200

Stages (via syntax scratchpad generators, matching CTKG english_syntax):
  1: POS tagging      — word → POS tag (N, V, DET, ADJ, ...)
  2: NP chunking      — word → BIO-NP tag (B_NP, I_NP, O_NP)
  3: PP chunking      — word → BIO-PP tag (B_PP, I_PP, O_PP)
  4: VP chunking      — word → BIO-VP tag (B_VP, I_VP, O_VP)
  5: Clause structure  — word → SUBJ/PRED/OTHER

All stages use the same WikiText-2 sentence pool. The output format
changes between stages. The model's learned representations carry
forward (curriculum hypothesis).

Key test: Does training stages 1→5 in order produce better stage 5
accuracy than training stage 5 directly?

Diagnostics same as train_arithmetic.py: epiplexity, per-token accuracy,
catastrophic forgetting checks.

Dependencies:
    pip install spacy datasets torch
    python -m spacy download en_core_web_sm

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
from experiments.scratchpad.generators.syntax import (
    PosTagGenerator, NpChunkGenerator, PpChunkGenerator,
    VpChunkGenerator, ClauseStructureGenerator,
    setup_syntax_vocab,
)
from experiments.language.wikitext2 import (
    load_or_annotate, build_word_list,
)


# ---------------------------------------------------------------------------
# Vocab construction
# ---------------------------------------------------------------------------

def _build_vocab(word_list):
    """Build deterministic vocab with grammar + word tokens."""
    v = Vocab()  # PAD=0, WORK=1, NOTE=2, SEP=3
    setup_syntax_vocab(v, word_list)
    return v


# ---------------------------------------------------------------------------
# Generator construction
# ---------------------------------------------------------------------------

def _build_generators(sentences, max_words, word_list):
    """Build stage generators. All share the same sentence pool."""
    return {
        1: PosTagGenerator(sentences, max_words=max_words, word_list=word_list),
        2: NpChunkGenerator(sentences, max_words=max_words, word_list=word_list),
        3: PpChunkGenerator(sentences, max_words=max_words, word_list=word_list),
        4: VpChunkGenerator(sentences, max_words=max_words, word_list=word_list),
        5: ClauseStructureGenerator(sentences, max_words=max_words, word_list=word_list),
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Syntax Parsing Curriculum on Mamba3')

    # Mode
    p.add_argument('--stage', type=int, choices=list(range(1, 6)), default=None,
                   help='Single stage to train (direct mode)')
    p.add_argument('--curriculum', action='store_true',
                   help='Curriculum mode: train stages 1 -> target_stage')
    p.add_argument('--target_stage', type=int, choices=list(range(2, 6)), default=5,
                   help='Final stage for curriculum mode')

    # Architecture
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_state', type=int, default=64)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--headdim', type=int, default=64)
    p.add_argument('--chunk_size', type=int, default=16)
    p.add_argument('--stable_ssm', action='store_true',
                   help='Use StableSSM A-matrix reparameterization')
    p.add_argument('--mhc', action='store_true',
                   help='Enable manifold-constrained hyperconnections')
    p.add_argument('--mhc_n_streams', type=int, default=4)

    # Data
    p.add_argument('--data_dir', type=str, default=None,
                   help='Directory for WikiText-2 data/cache')
    p.add_argument('--max_words', type=int, default=12,
                   help='Max words per sentence (shorter padded, longer filtered)')
    p.add_argument('--min_words', type=int, default=3)
    p.add_argument('--max_vocab', type=int, default=2000,
                   help='Word vocabulary size (top N by frequency)')
    p.add_argument('--n_train', type=int, default=5000)
    p.add_argument('--n_test', type=int, default=1000)
    p.add_argument('--test_fraction', type=float, default=0.2)
    p.add_argument('--seq_len', type=int, default=48)

    # Training
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup_epochs', type=int, default=4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--advance_threshold', type=float, default=0.95,
                   help='Test accuracy to advance in curriculum mode')
    p.add_argument('--replay_fraction', type=float, default=0.25)

    # Output
    p.add_argument('--results_file', type=str, default=None)

    # Performance
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--compile', action='store_true')

    # Misc
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_seed', type=int, default=42)
    p.add_argument('--print_every', type=int, default=10)

    args = p.parse_args()
    if args.stage is None and not args.curriculum:
        args.stage = 1
    return args


# ---------------------------------------------------------------------------
# LR schedule (cosine with warmup)
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

def evaluate_result_accuracy(model, loader, device, amp_ctx,
                             n_result_tokens, vocab_size):
    """Exact-match and per-token accuracy on the last n_result_tokens.

    Per Mistake #41: prediction for position p uses logits[:, p-1].
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
                pos = -(n_result_tokens - k)
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
# Replay loader
# ---------------------------------------------------------------------------

def _build_replay_loader(train_seqs, prev_train_seqs, replay_fraction,
                         batch_size):
    """Build DataLoader with stratified replay from previous stages."""
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


# ---------------------------------------------------------------------------
# Single-stage training
# ---------------------------------------------------------------------------

def train_stage(model, train_seqs, test_loader, device, amp_ctx, scaler,
                n_result_tokens, vocab_size, epochs, args,
                stage_label="", prev_loaders=None, prev_train_seqs=None):
    """Train one stage. Returns (history, passed)."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(args.warmup_epochs, epochs))

    history = []
    epoch_losses = []
    passed = False
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()

        train_loader = _build_replay_loader(
            train_seqs, prev_train_seqs, args.replay_fraction, args.batch_size)

        for (seqs,) in train_loader:
            seqs = seqs.to(device)
            with amp_ctx():
                logits = model(seqs)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab_size),
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
        epoch_losses.append(avg_loss)

        train_result = evaluate_result_accuracy(
            model, train_loader, device, amp_ctx, n_result_tokens, vocab_size)
        test_result = evaluate_result_accuracy(
            model, test_loader, device, amp_ctx, n_result_tokens, vocab_size)

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

        # Catastrophic-forgetting checks
        if prev_loaders:
            for s, (ldr, nr) in prev_loaders.items():
                prev_result = evaluate_result_accuracy(
                    model, ldr, device, amp_ctx, nr, vocab_size)
                record[f'stage{s}_acc'] = round(prev_result['exact'], 4)

        history.append(record)

        if epoch % args.print_every == 0 or epoch == 1 or epoch == epochs:
            tr_tok = '|'.join(f'{x:.2f}' for x in train_result['per_token'][:6])
            te_tok = '|'.join(f'{x:.2f}' for x in test_result['per_token'][:6])
            tr_tok_str = f" [{tr_tok}...]" if n_result_tokens > 6 else f" [{tr_tok}]"
            te_tok_str = f" [{te_tok}...]" if n_result_tokens > 6 else f" [{te_tok}]"

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

    # Epiplexity
    if epoch_losses:
        l_final = epoch_losses[-1]
        epiplexity = sum(l - l_final for l in epoch_losses)
        n_epochs_run = len(epoch_losses)
        print(f"  [{stage_label}] Epiplexity: S_preq={epiplexity:.2f}  "
              f"(final_loss={l_final:.4f}, {n_epochs_run} epochs)")
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

    # --- Device ---
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # --- Load & annotate WikiText-2 ---
    print("Loading WikiText-2...")
    sentences = load_or_annotate(
        data_dir=args.data_dir, split='train',
        max_words=args.max_words, min_words=args.min_words)
    print(f"  {len(sentences)} annotated sentences "
          f"({args.min_words}-{args.max_words} words)")

    # --- Build vocab ---
    word_list = build_word_list(sentences, max_vocab=args.max_vocab)
    VOCAB = _build_vocab(word_list)
    VOCAB_SIZE = len(VOCAB)
    print(f"  Vocab: {VOCAB_SIZE} tokens "
          f"({len(word_list)} words + grammar + special)")

    # --- Build generators ---
    generators = _build_generators(sentences, args.max_words, word_list)

    # --- Model ---
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
    amp_ctx = (lambda: torch.amp.autocast('cuda', dtype=torch.float16)) \
        if use_amp else nullcontext

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

    def decode_tokens(tokens):
        return VOCAB.decode_sequence(tokens)

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

    stage_names = {
        1: 'POS tagging', 2: 'NP chunking', 3: 'PP chunking',
        4: 'VP chunking', 5: 'Clause structure',
    }

    if args.curriculum:
        print(f"\nCurriculum 1 -> {args.target_stage}  "
              f"(advance >= {args.advance_threshold})"
              f"  replay={args.replay_fraction}\n")
        prev_loaders = {}
        prev_train_seqs = []

        for stage in range(1, args.target_stage + 1):
            _, te, data = make_loaders(stage)

            # Replay info
            if prev_train_seqs and args.replay_fraction > 0:
                n_replay = max(1, int(len(data['train_seqs']) * args.replay_fraction))
                per_stage = max(1, n_replay // len(prev_train_seqs))
                print(f"Stage {stage} ({stage_names[stage]}): "
                      f"{data['n_train_specs']} train / "
                      f"{data['n_test_specs']} test sentences"
                      f" (+{per_stage * len(prev_train_seqs)} replay/epoch)")
            else:
                print(f"Stage {stage} ({stage_names[stage]}): "
                      f"{data['n_train_specs']} train / "
                      f"{data['n_test_specs']} test sentences")

            # Print sample sequences
            for j in range(min(3, len(data['train_seqs']))):
                print(f"  sample: {decode_tokens(data['train_seqs'][j])}")

            history, passed = train_stage(
                model, data['train_seqs'], te, device, amp_ctx, scaler,
                data['n_result_tokens'], VOCAB_SIZE, args.epochs, args,
                stage_label=f"S{stage}", prev_loaders=prev_loaders,
                prev_train_seqs=prev_train_seqs)
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
        _, te, data = make_loaders(stage)
        print(f"\nStage {stage} ({stage_names[stage]}): "
              f"{data['n_train_specs']} train / "
              f"{data['n_test_specs']} test sentences")
        for j in range(min(3, len(data['train_seqs']))):
            print(f"  sample: {decode_tokens(data['train_seqs'][j])}")
        print()

        history, passed = train_stage(
            model, data['train_seqs'], te, device, amp_ctx, scaler,
            data['n_result_tokens'], VOCAB_SIZE, args.epochs, args,
            stage_label=f"S{stage}")
        all_history.extend(history)

    # --- Save results ---
    if args.results_file:
        result = dict(
            mode='curriculum' if args.curriculum else 'direct',
            target_stage=args.target_stage if args.curriculum else args.stage,
            n_params=n_params, seed=args.seed,
            vocab_size=VOCAB_SIZE, n_sentences=len(sentences),
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
