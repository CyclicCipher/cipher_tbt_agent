"""
Benchmark error optimizers for ePC on MNIST.

Compares SGD, Adam, and LRPD Newton for error optimization to determine:
1. Can Newton reduce T (inference iterations) from 5 to 1-2?
2. What's the wall-clock speedup?
3. Does accuracy hold up with fewer iterations?

The key hypothesis: the error Hessian H = I + J^T H_L J is LRPD
with rank <= n_output (10 for MNIST), so Woodbury-based Newton steps
should converge much faster than first-order methods.
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
    print(f"  error_optim={config['error_optim']}, iters={config['iters']}, "
          f"e_lr={config.get('e_lr', 'N/A')}, damping={config.get('damping', 'N/A')}")
    print(f"{'='*60}")

    architecture = get_mlp_mnist(hidden_size=128, num_hidden=3)
    model = PCE(
        architecture,
        iters=config['iters'],
        e_lr=config.get('e_lr', 0.01),
        output_loss='ce',
        error_optim=config['error_optim'],
        damping=config.get('damping', 1.0),
    ).to(device)

    weight_optim = torch.optim.Adam(model.parameters(), lr=0.001)

    results = {
        'name': name,
        'train_accs': [],
        'test_accs': [],
        'convergences': [],
        'error_norms': [],
        'epoch_times': [],
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
                epoch_correct += (preds == target).sum().item()
                epoch_total += batch_size

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

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

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

    # [0,1] Training Time
    ax = axes[0, 1]
    for i, r in enumerate(all_results):
        ax.plot(range(1, len(r['epoch_times'])+1), r['epoch_times'],
                'o-', color=colors[i], label=r['name'], linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Time per Epoch')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,2] Accuracy vs Time (efficiency frontier)
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

    # [1,0] Convergence
    ax = axes[1, 0]
    for i, r in enumerate(all_results):
        ax.plot(range(1, len(r['convergences'])+1), r['convergences'],
                'o-', color=colors[i], label=r['name'], linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg E_0 - E_T')
    ax.set_title('Inference Convergence (avg per epoch)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,1] Error Norms
    ax = axes[1, 1]
    for i, r in enumerate(all_results):
        if r['error_norms']:
            ax.plot(range(1, len(r['error_norms'])+1), r['error_norms'],
                    'o-', color=colors[i], label=r['name'], linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg sum(||e_i||)')
    ax.set_title('Error Magnitudes (avg per epoch)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,2] Summary Table
    ax = axes[1, 2]
    ax.axis('off')
    lines = [f"{'Config':<25} {'Best Test':>10} {'Time/ep':>10} {'Speedup':>10}"]
    lines.append("-" * 57)

    # Find baseline time (SGD T=5)
    baseline_time = None
    for r in all_results:
        if 'SGD T=5' in r['name']:
            baseline_time = np.mean(r['epoch_times'])
            break
    if baseline_time is None and all_results:
        baseline_time = np.mean(all_results[0]['epoch_times'])

    for r in all_results:
        best = max(r['test_accs']) if r['test_accs'] else 0
        avg_time = np.mean(r['epoch_times'])
        speedup = baseline_time / avg_time if avg_time > 0 else 0
        lines.append(f"{r['name']:<25} {best:>9.2%} {avg_time:>9.1f}s {speedup:>9.2f}x")

    ax.text(0.05, 0.5, '\n'.join(lines), fontsize=9,
            verticalalignment='center', family='monospace',
            transform=ax.transAxes)

    plt.suptitle('ePC Error Optimizer Benchmark (MNIST)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nBenchmark chart saved to {save_path}")
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    num_epochs = 3

    configs = [
        # Baseline: SGD T=5 (reference ePC paper)
        {'name': 'SGD T=5 (baseline)',
         'error_optim': 'sgd', 'iters': 5, 'e_lr': 0.01},

        # SGD T=2 (can we get away with fewer iters?)
        {'name': 'SGD T=2',
         'error_optim': 'sgd', 'iters': 2, 'e_lr': 0.01},

        # Adam T=5
        {'name': 'Adam T=5',
         'error_optim': 'adam', 'iters': 5, 'e_lr': 0.01},

        # Adam T=2
        {'name': 'Adam T=2',
         'error_optim': 'adam', 'iters': 2, 'e_lr': 0.01},

        # Newton T=2 (damping=1.0, conservative)
        {'name': 'Newton T=2 d=1.0',
         'error_optim': 'newton', 'iters': 2, 'damping': 1.0},

        # Newton T=2 (damping=0.1, aggressive)
        {'name': 'Newton T=2 d=0.1',
         'error_optim': 'newton', 'iters': 2, 'damping': 0.1},

        # Newton T=1 (single step, maximum speedup)
        {'name': 'Newton T=1 d=0.1',
         'error_optim': 'newton', 'iters': 1, 'damping': 0.1},
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
        print(f"  {r['name']:<25}: {best:.2%} test, "
              f"{avg_time:.1f}s/epoch, {speedup:.2f}x speedup")


if __name__ == "__main__":
    main()
