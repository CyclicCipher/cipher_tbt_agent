"""Fast seed sweep for ePC-Mamba3 to find working initializations.

Tests many seeds for a few batches each, checking Newton convergence
to identify seeds where the rank-1 Newton step is effective.

Usage:
    python experiments/Mamba3/seed_sweep.py
    python experiments/Mamba3/seed_sweep.py --seeds 100 --batches 20
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.Mamba3.mamba3_block import Mamba3Config
from experiments.Mamba3.epc_model import ePCMamba3LM


def generate_copy_data(n_samples, seq_len, vocab_size, copy_len=None):
    if copy_len is None:
        copy_len = seq_len // 2
    inputs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.zeros(n_samples, seq_len, dtype=torch.long)
    data = torch.randint(1, vocab_size, (n_samples, copy_len))
    inputs[:, :copy_len] = data
    targets[:, seq_len - copy_len:] = data
    return inputs, targets


def test_seed(seed, device, n_batches=10, batch_size=32, seq_len=64,
              vocab_size=16, d_model=128, d_state=64, n_layer=4):
    """Test a single seed for n_batches, return diagnostics."""
    # Disable TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    config = Mamba3Config(
        d_model=d_model, d_state=d_state, n_layer=n_layer,
        chunk_size=min(64, seq_len),
    )

    # Generate data with this seed
    train_x, train_y = generate_copy_data(
        n_batches * batch_size, seq_len, vocab_size)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size, shuffle=False, drop_last=True,
    )

    model = ePCMamba3LM(
        config, vocab_size=vocab_size, iters=2, damping=0.1,
        precision_mode='geometric', precision_base=3.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    convergences = []
    accs = []

    model.train()
    t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        inputs = batch[0].to(device)
        targets = batch[1].to(device)

        E_val = model.ipc_train_step(
            inputs, targets, optimizer, batch_size, w_clip=1.0)

        diag = model.get_diagnostics()
        losses.append(E_val)
        convergences.append(diag['convergence'])

        with torch.no_grad():
            logits = model(inputs)
            preds = logits.argmax(dim=-1)
            mask = targets != 0
            if mask.sum() > 0:
                acc = (preds[mask] == targets[mask]).float().mean().item()
            else:
                acc = 0.0
            accs.append(acc)

    t1 = time.perf_counter()
    ms_total = (t1 - t0) * 1000

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        'seed': seed,
        'avg_loss': sum(losses) / len(losses) if losses else 0,
        'avg_convergence': sum(convergences) / len(convergences) if convergences else 0,
        'first_loss': losses[0] if losses else 0,
        'last_loss': losses[-1] if losses else 0,
        'first_conv': convergences[0] if convergences else 0,
        'last_conv': convergences[-1] if convergences else 0,
        'last_acc': accs[-1] if accs else 0,
        'ms_per_batch': ms_total / len(losses) if losses else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Seed sweep for ePC-Mamba3')
    parser.add_argument('--seeds', type=int, default=50,
                        help='Number of seeds to test (1 to N)')
    parser.add_argument('--batches', type=int, default=10,
                        help='Batches per seed')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Seed sweep: testing seeds 1-{args.seeds}, {args.batches} batches each")
    print(f"Device: {device}")
    print()
    print(f"{'Seed':>6} {'1st Loss':>10} {'Last Loss':>10} {'Avg Conv':>10} "
          f"{'Last Conv':>10} {'Last Acc':>10} {'ms/batch':>10}")
    print("-" * 76)

    results = []
    for seed in range(1, args.seeds + 1):
        r = test_seed(seed, device, n_batches=args.batches)
        results.append(r)
        flag = " ***" if r['avg_convergence'] > 100 else ""
        print(f"{r['seed']:6d} {r['first_loss']:10.1f} {r['last_loss']:10.1f} "
              f"{r['avg_convergence']:10.2f} {r['last_conv']:10.2f} "
              f"{r['last_acc']:10.4f} {r['ms_per_batch']:10.1f}{flag}")

    # Summary
    print("\n" + "=" * 76)
    positive = [r for r in results if r['avg_convergence'] > 0]
    good = [r for r in results if r['avg_convergence'] > 100]

    print(f"Seeds with positive avg convergence: {len(positive)}/{len(results)}")
    print(f"Seeds with avg convergence > 100:    {len(good)}/{len(results)}")

    if good:
        print(f"\nBest seeds (convergence > 100):")
        for r in sorted(good, key=lambda x: -x['avg_convergence']):
            print(f"  Seed {r['seed']}: conv={r['avg_convergence']:.2f}, "
                  f"loss={r['last_loss']:.1f}, acc={r['last_acc']:.4f}")
    else:
        print("\nNo seeds found with convergence > 100.")
        print("This suggests the issue is NOT initialization sensitivity,")
        print("but a structural problem with rank-1 Newton on Mamba3.")

        # Show best anyway
        best = sorted(results, key=lambda x: -x['avg_convergence'])[:5]
        print(f"\nTop 5 seeds by convergence:")
        for r in best:
            print(f"  Seed {r['seed']}: conv={r['avg_convergence']:.2f}, "
                  f"loss={r['last_loss']:.1f}, acc={r['last_acc']:.4f}")


if __name__ == '__main__':
    main()
