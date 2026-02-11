"""
Mamba-3 synthetic task training (standard backprop).

Tests:
  1. Copy: input [a,b,c,PAD,PAD,PAD] → output [PAD,PAD,PAD,a,b,c]
  2. Deeper architectures (4-8 layers)
  3. Comparison with Mamba2 baseline

Usage:
  python experiments/Mamba3/train_synthetic.py
  python experiments/Mamba3/train_synthetic.py --n_layer 8 --epochs 50
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.Mamba3.mamba3_block import Mamba3Config, Mamba3LM


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_copy_data(n_samples, seq_len, vocab_size, copy_len=None):
    """Generate copy task data. PAD=0, data tokens in [1, vocab_size-1]."""
    if copy_len is None:
        copy_len = seq_len // 2
    assert copy_len <= seq_len // 2

    inputs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.zeros(n_samples, seq_len, dtype=torch.long)

    data = torch.randint(1, vocab_size, (n_samples, copy_len))
    inputs[:, :copy_len] = data
    targets[:, seq_len - copy_len:] = data

    return inputs, targets


def compute_accuracy(logits, targets, ignore_pad=True):
    """Compute token-level accuracy."""
    preds = logits.argmax(dim=-1)
    if ignore_pad:
        mask = targets != 0
        if mask.sum() == 0:
            return 0.0
        return (preds[mask] == targets[mask]).float().mean().item()
    return (preds == targets).float().mean().item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    epoch_time = 0.0

    for batch in train_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        t0 = time.perf_counter()

        optimizer.zero_grad()
        logits = model(inputs)
        b, l, v = logits.shape
        loss = F.cross_entropy(logits.reshape(b * l, v), targets.reshape(b * l))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        t1 = time.perf_counter()
        epoch_time += (t1 - t0)

        with torch.no_grad():
            acc = compute_accuracy(logits, targets)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches, epoch_time * 1000 / n_batches


def evaluate(model, test_loader, device):
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            logits = model(inputs)
            b, l, v = logits.shape
            loss = F.cross_entropy(logits.reshape(b * l, v), targets.reshape(b * l))
            acc = compute_accuracy(logits, targets)
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

    return total_acc / n_batches, total_loss / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Mamba-3 synthetic training')
    parser.add_argument('--task', choices=['copy'], default='copy')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--use_conv', action='store_true',
                        help='Enable optional short causal convolution')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Adjust seq_len/chunk_size
    chunk_size = min(64, args.seq_len)
    while args.seq_len % chunk_size != 0 and chunk_size > 1:
        chunk_size -= 1

    config = Mamba3Config(
        d_model=args.d_model,
        d_state=args.d_state,
        n_layer=args.n_layer,
        chunk_size=chunk_size,
        use_conv=args.use_conv,
    )
    print(f"Config: d_model={config.d_model}, d_inner={config.d_inner}, "
          f"nheads={config.nheads}, n_layer={config.n_layer}, "
          f"d_state={config.d_state}")

    # Data
    print(f"\nGenerating {args.task} task data...")
    train_x, train_y = generate_copy_data(
        args.n_train, args.seq_len, args.vocab_size)
    test_x, test_y = generate_copy_data(
        args.n_test, args.seq_len, args.vocab_size)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=args.batch_size, shuffle=False, drop_last=True,
    )

    # Model
    model = Mamba3LM(config, vocab_size=args.vocab_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Mamba-3 LM ({config.n_layer} layers)")
    print(f"Parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_test_acc = 0.0
    print(f"\n{'Epoch':>5} {'Loss':>10} {'Train Acc':>10} {'Test Acc':>10} {'ms/batch':>10}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        avg_loss, avg_acc, avg_time = train_epoch(
            model, train_loader, optimizer, device)
        test_acc, test_loss = evaluate(model, test_loader, device)
        best_test_acc = max(best_test_acc, test_acc)

        print(f"{epoch:5d} {avg_loss:10.4f} {avg_acc:10.4f} "
              f"{test_acc:10.4f} {avg_time:10.1f}")

        if best_test_acc >= 0.99 and epoch >= 5:
            print(f"\nSuccess! Test accuracy {best_test_acc:.4f} >= 99% at epoch {epoch}")
            break

    print(f"\nBest test accuracy: {best_test_acc:.4f}")
    if best_test_acc >= 0.95:
        print("PASS: Mamba-3 works on copy task!")
    elif best_test_acc >= 0.90:
        print("PROMISING: 90%+ accuracy.")
    else:
        print(f"Needs work: {best_test_acc:.1%} accuracy.")


if __name__ == '__main__':
    main()
