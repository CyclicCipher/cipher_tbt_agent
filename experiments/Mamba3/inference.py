#!/usr/bin/env python3
"""
Interactive inference for trained Mamba3 models.

Load a checkpoint and generate completions interactively.

Usage (arithmetic):
    python inference.py --domain arithmetic --checkpoint checkpoints/stage_5.pt

Usage (language):
    python inference.py --domain language --checkpoint checkpoints/stage_5.pt

Usage (language, with data dir):
    python inference.py --domain language --checkpoint checkpoints/stage_5.pt --data_dir ./data

Interactive commands:
    Type a prompt and press Enter to generate a completion.
    !temp <float>   — set sampling temperature (0 = greedy, default)
    !topk <int>     — set top-k sampling (0 = off, default)
    !topp <float>   — set top-p / nucleus sampling (1.0 = off, default)
    !maxlen <int>   — set max generation length (default: 64)
    !vocab          — print the full vocabulary
    !sample <N>     — show N random training sequences (requires data rebuild)
    !help           — print this help
    !quit / !exit   — exit

The prompt is tokenized by splitting on spaces. Unknown tokens are shown
as warnings. After the WORK token, the model generates autoregressively.
"""

import argparse
import os
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.Mamba3.mamba3_block import Mamba3Config, Mamba3LM
from experiments.scratchpad import Vocab


# ---------------------------------------------------------------------------
# Vocab builders (must match training scripts exactly)
# ---------------------------------------------------------------------------

def build_arithmetic_vocab():
    """Reproduce the arithmetic vocab from train_arithmetic.py."""
    v = Vocab()
    for d in range(10):
        v.add(str(d))
    v.add('+'); v.add('-'); v.add('=')
    v.add('DOT'); v.add('TEN')
    v.add('?')
    v.add('STOP')
    return v


def build_language_vocab(data_dir=None, max_words=2000, max_word_len=15, min_words=3):
    """Reproduce the language vocab from train_language.py.

    Requires WikiText-2 data + spaCy annotations (cached after first run).
    """
    from experiments.scratchpad.generators.syntax import setup_syntax_vocab
    from experiments.language.wikitext2 import load_or_annotate, build_word_list

    sentences = load_or_annotate(
        data_dir=data_dir, split='train',
        max_words=max_word_len, min_words=min_words)
    word_list = build_word_list(sentences, max_vocab=max_words)
    v = Vocab()
    setup_syntax_vocab(v, word_list)
    return v


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, input_ids, max_new_tokens=64, temperature=0.0,
             top_k=0, top_p=1.0, pad_id=0, stop_ids=None,
             device='cpu', amp_ctx=None):
    """Autoregressive generation from a Mamba3LM.

    Args:
        model: Mamba3LM in eval mode.
        input_ids: (1, prompt_len) tensor of token IDs.
        max_new_tokens: maximum tokens to generate.
        temperature: sampling temperature (0 = greedy argmax).
        top_k: top-k filtering (0 = off).
        top_p: nucleus sampling threshold (1.0 = off).
        pad_id: PAD token ID (stop if generated).
        stop_ids: optional set of token IDs that stop generation.
        device: torch device.
        amp_ctx: autocast context manager (or nullcontext).

    Returns:
        List of generated token IDs (not including the prompt).
    """
    if amp_ctx is None:
        amp_ctx = nullcontext

    model.eval()
    generated = []
    seq = input_ids.to(device)

    for _ in range(max_new_tokens):
        with amp_ctx():
            logits = model(seq)  # (1, seq_len, vocab_size)
        next_logits = logits[0, -1]  # (vocab_size,)

        if temperature <= 0:
            # Greedy
            next_id = next_logits.argmax().item()
        else:
            next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(0)))
                next_logits[next_logits < topk_vals[-1]] = -float('inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=0), dim=0)
                remove_mask = cumulative_probs > top_p
                # Shift right so the first token above threshold is kept
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0] = False
                sorted_logits[remove_mask] = -float('inf')
                next_logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=0)
            next_id = torch.multinomial(probs, 1).item()

        if next_id == pad_id:
            break
        if stop_ids and next_id in stop_ids:
            generated.append(next_id)
            break

        generated.append(next_id)
        next_token = torch.tensor([[next_id]], device=device)
        seq = torch.cat([seq, next_token], dim=1)

    return generated


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(path, config, vocab_size, device):
    """Load model from a training checkpoint.

    Supports two formats:
    1. New format (self-contained): has 'config', 'vocab' keys
    2. Legacy format: has 'model_state_dict' (requires config from args)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # New format: config embedded in checkpoint
    if 'config' in ckpt:
        cfg = ckpt['config']
        config = Mamba3Config(**cfg)
        vocab_size = ckpt.get('vocab_size', vocab_size)

    model = Mamba3LM(config, vocab_size).to(device)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # Maybe the checkpoint IS the state dict
        model.load_state_dict(ckpt)

    return model, config


# ---------------------------------------------------------------------------
# Token probability display
# ---------------------------------------------------------------------------

def show_token_probs(logits, vocab, top_n=10):
    """Show top-N token probabilities from logits."""
    probs = F.softmax(logits, dim=0)
    topk_probs, topk_ids = torch.topk(probs, min(top_n, len(probs)))
    lines = []
    for p, tid in zip(topk_probs.tolist(), topk_ids.tolist()):
        token = vocab.decode(tid)
        bar = '#' * int(p * 40)
        lines.append(f"  {token:>12s}  {p:6.3f}  {bar}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def interactive_loop(model, vocab, device, amp_ctx, seq_len=48):
    """Interactive prompt → generation loop."""
    temperature = 0.0
    top_k = 0
    top_p = 1.0
    max_new = 64
    pad_id = vocab.PAD

    print("\nInteractive inference ready.")
    print("Type a sequence of tokens (space-separated), then press Enter.")
    print("The model will generate a continuation autoregressively.")
    print("Type !help for commands, !quit to exit.\n")

    # Show vocab summary
    n = len(vocab)
    sample_tokens = [vocab.decode(i) for i in range(min(n, 20))]
    print(f"Vocab: {n} tokens. First 20: {' '.join(sample_tokens)}")
    print()

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        # Commands
        if line.startswith('!'):
            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ('!quit', '!exit'):
                print("Bye.")
                break
            elif cmd == '!help':
                print(__doc__)
            elif cmd == '!temp':
                temperature = float(parts[1]) if len(parts) > 1 else 0.0
                print(f"Temperature = {temperature}")
            elif cmd == '!topk':
                top_k = int(parts[1]) if len(parts) > 1 else 0
                print(f"Top-k = {top_k}")
            elif cmd == '!topp':
                top_p = float(parts[1]) if len(parts) > 1 else 1.0
                print(f"Top-p = {top_p}")
            elif cmd == '!maxlen':
                max_new = int(parts[1]) if len(parts) > 1 else 64
                print(f"Max generation length = {max_new}")
            elif cmd == '!vocab':
                for i in range(len(vocab)):
                    print(f"  {i:4d}: {vocab.decode(i)}")
            else:
                print(f"Unknown command: {cmd}. Type !help for available commands.")
            continue

        # Tokenize input
        tokens = line.split()
        ids = []
        unknown = []
        for t in tokens:
            if t in vocab:
                ids.append(vocab[t])
            else:
                unknown.append(t)
                # Try to match case-insensitively
                found = False
                for tok_str in [t.upper(), t.lower(), t.capitalize()]:
                    if tok_str in vocab:
                        ids.append(vocab[tok_str])
                        found = True
                        break
                if not found:
                    print(f"  WARNING: unknown token '{t}' — skipping")

        if not ids:
            print("  No valid tokens in input.")
            continue

        if unknown:
            print(f"  (resolved: {' '.join(vocab.decode(i) for i in ids)})")

        # Build input tensor
        input_ids = torch.tensor([ids], dtype=torch.long)

        # Generate
        gen_ids = generate(
            model, input_ids,
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_id=pad_id,
            device=device,
            amp_ctx=amp_ctx,
        )

        # Display result
        prompt_str = ' '.join(vocab.decode(i) for i in ids)
        gen_str = ' '.join(vocab.decode(i) for i in gen_ids) if gen_ids else '(empty)'
        print(f"  Input:  {prompt_str}")
        print(f"  Output: {prompt_str} | {gen_str}")

        # Show probabilities for next token after last generated
        if gen_ids:
            full_seq = ids + gen_ids
            full_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                with amp_ctx():
                    logits = model(full_tensor)
            # Probs for the position right after the full sequence
            print(f"\n  Next-token probabilities (after '{vocab.decode(gen_ids[-1])}'):")
            print(show_token_probs(logits[0, -1], vocab))
        print()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Interactive inference for trained Mamba3 models')

    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to .pt checkpoint file')
    p.add_argument('--domain', type=str, choices=['arithmetic', 'language'],
                   default='arithmetic',
                   help='Domain (determines vocab construction)')

    # Architecture (used only if checkpoint lacks config)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_state', type=int, default=64)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--headdim', type=int, default=64)
    p.add_argument('--chunk_size', type=int, default=16)
    p.add_argument('--stable_ssm', action='store_true')
    p.add_argument('--mhc', action='store_true')
    p.add_argument('--mhc_n_streams', type=int, default=4)

    # Language domain data
    p.add_argument('--data_dir', type=str, default=None,
                   help='WikiText-2 data directory (language domain)')
    p.add_argument('--max_vocab', type=int, default=2000,
                   help='Word vocabulary size (language domain)')
    p.add_argument('--max_words', type=int, default=15)
    p.add_argument('--min_words', type=int, default=3)

    # Generation defaults
    p.add_argument('--seq_len', type=int, default=48)

    # Device
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--no_amp', action='store_true')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Build vocab
    print(f"Domain: {args.domain}")
    if args.domain == 'arithmetic':
        vocab = build_arithmetic_vocab()
    else:
        print("Building language vocab (loading WikiText-2 + spaCy)...")
        vocab = build_language_vocab(
            data_dir=args.data_dir,
            max_words=args.max_vocab,
            max_word_len=args.max_words,
            min_words=args.min_words,
        )

    vocab_size = len(vocab)
    print(f"Vocab: {vocab_size} tokens")

    # Build config from args (checkpoint may override)
    config = Mamba3Config(
        d_model=args.d_model,
        d_state=args.d_state,
        n_layer=args.n_layer,
        headdim=args.headdim,
        chunk_size=args.chunk_size,
        stable_ssm=args.stable_ssm,
        use_mhc=args.mhc,
        mhc_n_streams=args.mhc_n_streams,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, config, vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | d={config.d_model} N={config.d_state} "
          f"L={config.n_layer} H={config.nheads} | {device}")

    # AMP setup
    use_amp = not args.no_amp and device.type == 'cuda'
    amp_ctx = (lambda: torch.amp.autocast('cuda', dtype=torch.float16)) \
        if use_amp else nullcontext

    model.eval()

    # Enter interactive loop
    interactive_loop(model, vocab, device, amp_ctx, seq_len=args.seq_len)


if __name__ == '__main__':
    main()
