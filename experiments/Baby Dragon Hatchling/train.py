"""Multi-task BDH training: sudoku + calendar.

Trains a single BDH model on interleaved sudoku puzzle-solving and
calendar date-arithmetic tasks. Both use byte-level tokenization
(vocab_size=256).

Checkpoint management: saves latest model to checkpoints/latest.pt
after each evaluation. Deletes previous checkpoint.

Usage:
    python experiments/Baby\ Dragon\ Hatchling/train.py
    python experiments/Baby\ Dragon\ Hatchling/train.py --max_iters 10  # smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

sys.path.insert(0, _SCRIPT_DIR)
from bdh import BDH, BDHConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=8)  # 8 fits 4GB VRAM with mlp_mult=64
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=64)  # 64 not 128 — fits 4GB VRAM
    p.add_argument("--sudoku_train_size", type=int, default=5000)
    p.add_argument("--sudoku_test_size", type=int, default=200)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(args) -> tuple[np.ndarray, np.ndarray]:
    """Build interleaved training corpus from sudoku + calendar data."""
    print("Generating sudoku puzzles...")
    from data.sudoku import (
        generate_dataset, format_for_training,
    )
    train_pairs, test_pairs = generate_dataset(
        n_train=args.sudoku_train_size,
        n_test=args.sudoku_test_size,
        difficulty="mixed",
    )
    sudoku_train_strs = [format_for_training(p, s) for p, s in train_pairs]
    sudoku_test_strs = [format_for_training(p, s) for p, s in test_pairs]

    print("Generating calendar sentences...")
    from data.calendar_data import (
        calendar_training_strings,
    )
    calendar_strs = calendar_training_strings()

    # Combine into single text blobs with newline separators.
    # Shuffle the training data so tasks are interleaved.
    import random
    rng = random.Random(42)
    all_train = sudoku_train_strs + calendar_strs
    rng.shuffle(all_train)
    train_text = "\n".join(all_train)

    all_test = sudoku_test_strs
    test_text = "\n".join(all_test)

    # Byte-level encoding (uint8 → int64 for torch).
    train_data = np.frombuffer(train_text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
    test_data = np.frombuffer(test_text.encode("utf-8"), dtype=np.uint8).astype(np.int64)

    print(f"Train tokens: {len(train_data):,}")
    print(f"Test tokens:  {len(test_data):,}")

    return train_data, test_data


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: torch.device):
    """Sample a random batch of sequences from the data."""
    max_start = len(data) - block_size - 1
    if max_start < 1:
        max_start = 1
    ix = np.random.randint(0, max_start, size=(batch_size,))
    x = torch.from_numpy(np.stack([data[i:i + block_size] for i in ix])).to(device)
    y = torch.from_numpy(np.stack([data[i + 1:i + block_size + 1] for i in ix])).to(device)
    return x, y


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")


def save_checkpoint(model, optimizer, step, loss, path=None):
    """Save model checkpoint. Deletes previous checkpoint."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, "latest.pt")

    # Delete previous checkpoint.
    if os.path.exists(path):
        os.remove(path)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "config": model.config,
    }, path)
    print(f"  Checkpoint saved: {path} (step {step}, loss {loss:.4f})")


def load_checkpoint(path=None):
    """Load a checkpoint. Returns dict or None."""
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, test_data, args, device):
    """Estimate loss on train and test data."""
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("test", test_data)]:
        batch_losses = []
        for _ in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[name] = sum(batch_losses) / len(batch_losses)
    model.train()
    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Device setup.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Precision.
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Using bfloat16")
        else:
            dtype = torch.float16
            print("Using float16")
    else:
        dtype = torch.float32
        print("Using float32")

    # Load data.
    train_data, test_data = load_data(args)

    # Create model.
    config = BDHConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_mult,
        vocab_size=256,
        dropout=0.1,
    )
    model = BDH(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    N = config.mlp_internal_dim_multiplier * config.n_embd // config.n_head
    print(f"\nModel: {n_params:,} params, N={N} neurons/head, "
          f"{config.n_layer} layers, {config.n_head} heads")

    # Optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1,
    )

    # Mixed precision.
    scaler = torch.amp.GradScaler(device.type, enabled=(dtype == torch.float16))
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype)

    # Training loop.
    print(f"\nTraining for {args.max_iters} iterations...")
    t0 = time.time()

    for step in range(args.max_iters):
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)

        with autocast_ctx:
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging.
        if step % args.log_freq == 0 or step == args.max_iters - 1:
            dt = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():.4f} | {dt:.1f}s")
            t0 = time.time()

        # Evaluation + checkpoint.
        if (step > 0 and step % args.eval_interval == 0) or step == args.max_iters - 1:
            losses = estimate_loss(model, train_data, test_data, args, device)
            print(f"  eval | train loss {losses['train']:.4f} | test loss {losses['test']:.4f}")
            save_checkpoint(model, optimizer, step, losses['test'])

    print("\nTraining complete.")

    # Final VRAM report.
    if device.type == "cuda":
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
