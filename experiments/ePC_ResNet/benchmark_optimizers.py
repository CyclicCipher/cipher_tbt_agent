"""
Benchmark weight optimizers for ePC on MNIST.

Compares Adam vs KRONOS (A-only) for weight updates, both using Newton T=2
for error optimization (the established best inference method).

Key questions:
1. Does KRONOS A-only (input whitening via LRPD) improve accuracy over Adam?
2. What's the wall-clock cost of the curvature estimation overhead?
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.ePC_ResNet.epc_model import PCE
from experiments.ePC_ResNet.architectures import get_mlp_mnist
from src.optimizers.kronos import KRONOS


def get_mnist_loaders(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def run_experiment(config, train_loader, test_loader, device, num_epochs=3):
    """Run one experiment configuration and return metrics."""
    name = config['name']
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"  error: {config['error_optim']} T={config['iters']} "
          f"damping={config.get('e_damping', 'N/A')}")
    print(f"  weight: {config['weight_optim']}")
    if config['weight_optim'] == 'kronos':
        print(f"    rank={config.get('kronos_rank', 32)}, "
              f"damping={config.get('kronos_damping', 0.01)}, "
              f"ema={config.get('kronos_ema', 0.95)}, "
              f"update_freq={config.get('kronos_update_freq', 10)}")
    print(f"{'='*60}")

    architecture = get_mlp_mnist(hidden_size=128, num_hidden=3)
    model = PCE(
        architecture,
        iters=config['iters'],
        e_lr=config.get('e_lr', 0.01),
        output_loss='ce',
        error_optim=config['error_optim'],
        damping=config.get('e_damping', 1.0),
    ).to(device)

    if config['weight_optim'] == 'kronos':
        weight_optim = KRONOS(
            model,
            lr=config.get('w_lr', 0.001),
            damping=config.get('kronos_damping', 0.01),
            rank=config.get('kronos_rank', 32),
            ema_decay=config.get('kronos_ema', 0.95),
            update_freq=config.get('kronos_update_freq', 10),
            momentum=config.get('kronos_momentum', 0.9),
            grad_clip=config.get('kronos_grad_clip', 1.0),
        )
    else:
        weight_optim = torch.optim.Adam(
            model.parameters(), lr=config.get('w_lr', 0.001))

    results = {
        'name': name,
        'train_accs': [],
        'test_accs': [],
        'convergences': [],
        'error_norms': [],
        'epoch_times': [],
        'batch_train_accs': [],
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_correct = 0
        epoch_total = 0
        epoch_convergences = []
        epoch_error_norms = []

        start = time.time()
        for data, target in train_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            batch_size = data.size(0)

            # Phase 1: Inference
            model(data, target)
            diag = model.get_diagnostics()
            epoch_convergences.append(diag['convergence'])
            if diag['error_norms']:
                epoch_error_norms.append(sum(diag['error_norms']))

            # Phase 2: Weight update
            weight_optim.zero_grad()
            loss = model.compute_weight_loss(data, target, batch_size)
            loss.backward()
            weight_optim.step()

            # Track accuracy
            with torch.no_grad():
                outputs = model(data)
                preds = outputs.argmax(dim=1)
                correct = (preds == target).sum().item()
                epoch_correct += correct
                epoch_total += batch_size
                results['batch_train_accs'].append(correct / batch_size)

        epoch_time = time.time() - start
        train_acc = epoch_correct / epoch_total
        results['train_accs'].append(train_acc)
        results['epoch_times'].append(epoch_time)
        results['convergences'].append(np.mean(epoch_convergences))
        if epoch_error_norms:
            results['error_norms'].append(np.mean(epoch_error_norms))

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(data.size(0), -1).to(device)
                target = target.to(device)
                outputs = model(data)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == target).sum().item()
                test_total += data.size(0)

        test_acc = test_correct / test_total
        results['test_accs'].append(test_acc)

        print(f"  Epoch {epoch+1}: Train {train_acc:.2%}, Test {test_acc:.2%}, "
              f"Time {epoch_time:.1f}s, Conv {results['convergences'][-1]:.2f}")

    return results


def plot_comparison(all_results, save_path='benchmark_optimizers.png'):
    """Plot comparison of all optimizer configurations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_results), 2)))

    # [0,0] Test Accuracy by Epoch
    ax = axes[0, 0]
    for i, r in enumerate(all_results):
        ax.plot(range(1, len(r['test_accs'])+1), r['test_accs'],
                'o-', color=colors[i], label=r['name'], linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,1] Training Time per Epoch
    ax = axes[0, 1]
    for i, r in enumerate(all_results):
        ax.plot(range(1, len(r['epoch_times'])+1), r['epoch_times'],
                'o-', color=colors[i], label=r['name'], linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Time per Epoch')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,2] Accuracy vs Wall-Clock Time
    ax = axes[0, 2]
    for i, r in enumerate(all_results):
        cum_time = np.cumsum(r['epoch_times'])
        ax.plot(cum_time, r['test_accs'],
                'o-', color=colors[i], label=r['name'], linewidth=2, markersize=6)
    ax.set_xlabel('Cumulative Time (seconds)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs Wall-Clock Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,0] Batch-level Training Accuracy (learning curves)
    ax = axes[1, 0]
    for i, r in enumerate(all_results):
        accs = r['batch_train_accs']
        if accs:
            ax.plot(accs, alpha=0.3, color=colors[i], linewidth=0.5)
            # Moving average
            if len(accs) > 50:
                window = min(50, len(accs) // 5)
                ma = np.convolve(accs, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(accs)), ma,
                        color=colors[i], linewidth=1.5, label=r['name'])
    ax.set_xlabel('Batch')
    ax.set_ylabel('Train Accuracy')
    ax.set_title('Batch-level Training Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,1] Inference Convergence
    ax = axes[1, 1]
    for i, r in enumerate(all_results):
        if r['convergences']:
            ax.plot(range(1, len(r['convergences'])+1), r['convergences'],
                    'o-', color=colors[i], label=r['name'], linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg E_0 - E_T')
    ax.set_title('Inference Convergence (avg per epoch)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,2] Summary Table
    ax = axes[1, 2]
    ax.axis('off')
    lines = [f"{'Config':<30} {'Best Test':>10} {'Time/ep':>10} {'Speedup':>10}"]
    lines.append("-" * 62)

    baseline_time = np.mean(all_results[0]['epoch_times']) if all_results else 1.0

    for r in all_results:
        best = max(r['test_accs']) if r['test_accs'] else 0
        avg_time = np.mean(r['epoch_times'])
        speedup = baseline_time / avg_time if avg_time > 0 else 0
        lines.append(f"{r['name']:<30} {best:>9.2%} {avg_time:>9.1f}s {speedup:>9.2f}x")

    ax.text(0.05, 0.5, '\n'.join(lines), fontsize=9,
            verticalalignment='center', family='monospace',
            transform=ax.transAxes)

    plt.suptitle('ePC Weight Optimizer Benchmark — Newton T=2 + Adam vs KRONOS A-only (MNIST)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nBenchmark chart saved to {save_path}")
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    num_epochs = 3

    # All configs use Newton T=2 d=0.1 for error optimization (established best)
    newton_base = {
        'error_optim': 'newton', 'iters': 2, 'e_damping': 0.1,
    }

    configs = [
        # Baseline: Newton errors + Adam weights
        {**newton_base,
         'name': 'Newton + Adam (baseline)',
         'weight_optim': 'adam', 'w_lr': 0.001},

        # KRONOS A-only preconditioning (no G factor — degenerate for ePC)
        # LR increased to compensate for removed G^{-1} ≈ 100*I amplification.
        # Clipping disabled — precond/raw ratios are healthy (1.7-27x).
        {**newton_base,
         'name': 'Newton + KRONOS A-only r=32',
         'weight_optim': 'kronos', 'w_lr': 0.3,
         'kronos_rank': 32, 'kronos_damping': 0.01,
         'kronos_ema': 0.95, 'kronos_update_freq': 10,
         'kronos_grad_clip': 0.0},
    ]

    all_results = []
    for config in configs:
        results = run_experiment(config, train_loader, test_loader, device, num_epochs)
        all_results.append(results)

    plot_comparison(all_results)

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    baseline_time = np.mean(all_results[0]['epoch_times'])
    for r in all_results:
        best = max(r['test_accs']) if r['test_accs'] else 0
        avg_time = np.mean(r['epoch_times'])
        speedup = baseline_time / avg_time
        print(f"  {r['name']:<35}: {best:.2%} test, "
              f"{avg_time:.1f}s/epoch, {speedup:.2f}x speedup")


if __name__ == "__main__":
    main()
