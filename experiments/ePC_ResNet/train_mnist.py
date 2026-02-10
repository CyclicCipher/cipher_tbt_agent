"""
Validate ePC on MNIST before CIFAR-10.

Confirms that pure ePC (error optimization + gradient descent weight updates)
achieves reasonable accuracy on MNIST. This validates the core algorithm before
tackling the more complex ResNet + CIFAR-10 setting.

Target: ~95% test accuracy (eBPC baseline: 95.74% with Hebbian updates)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

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


class Diagnostics:
    """Collect and plot training diagnostics for ePC MNIST."""

    def __init__(self, num_error_layers):
        self.num_error_layers = num_error_layers
        self.reset()

    def reset(self):
        self.train_accs = []
        self.train_losses = []
        self.test_accs = []
        self.test_losses = []
        self.layer_energies = [[] for _ in range(self.num_error_layers)]
        self.inference_convergence = []
        self.iters_used = []
        self.error_norms = [[] for _ in range(self.num_error_layers)]
        self.weight_magnitudes = {}

    def update_train(self, acc, loss, diagnostics, weight_mags):
        self.train_accs.append(acc)
        self.train_losses.append(loss)
        self.inference_convergence.append(diagnostics['convergence'])
        self.iters_used.append(diagnostics.get('iters_used', 0))

        for i, energy in enumerate(diagnostics['layer_energies']):
            if i < self.num_error_layers:
                self.layer_energies[i].append(energy)

        for i, norm in enumerate(diagnostics['error_norms']):
            if i < self.num_error_layers:
                self.error_norms[i].append(norm)

        for name, mag in weight_mags.items():
            if name not in self.weight_magnitudes:
                self.weight_magnitudes[name] = []
            self.weight_magnitudes[name].append(mag)

    def update_test(self, acc, loss):
        self.test_accs.append(acc)
        self.test_losses.append(loss)

    def plot(self, save_path):
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))

        # [0,0] Accuracy
        ax = axes[0, 0]
        if self.train_accs:
            ax.plot(self.train_accs, alpha=0.5, linewidth=0.5, label='Train (batch)')
        if self.test_accs:
            n_train = len(self.train_accs)
            n_test = len(self.test_accs)
            if n_test > 0:
                test_x = [(i + 1) * n_train / n_test for i in range(n_test)]
                ax.plot(test_x, self.test_accs, 'o-', linewidth=2,
                        markersize=6, label='Test')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [0,1] Cross-Entropy Loss
        ax = axes[0, 1]
        if self.train_losses:
            ax.plot(self.train_losses, alpha=0.5, linewidth=0.5, label='Train (batch)')
        if self.test_losses:
            n_train = len(self.train_losses)
            n_test = len(self.test_losses)
            if n_test > 0:
                test_x = [(i + 1) * n_train / n_test for i in range(n_test)]
                ax.plot(test_x, self.test_losses, 'o-', linewidth=2,
                        markersize=6, label='Test')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Cross-Entropy Loss (logging only)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [0,2] Per-Layer Energies
        ax = axes[0, 2]
        for i, energies in enumerate(self.layer_energies):
            if energies:
                ax.plot(energies, label=f'Layer {i+1}', alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Energy')
        ax.set_title('Per-Layer Energies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # [1,0] Inference Convergence
        ax = axes[1, 0]
        if self.inference_convergence:
            ax.plot(self.inference_convergence, alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('E_initial - E_final')
        ax.set_title('Inference Convergence (E_0 - E_T)')
        ax.grid(True, alpha=0.3)

        # [1,1] Error Magnitudes
        ax = axes[1, 1]
        for i, norms in enumerate(self.error_norms):
            if norms:
                ax.plot(norms, label=f'Layer {i+1}', alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('||e_i||')
        ax.set_title('Error Magnitudes (per layer)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # [1,2] Weight Magnitudes
        ax = axes[1, 2]
        for name, mags in self.weight_magnitudes.items():
            if mags:
                ax.plot(mags, label=name, alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('max|W|')
        ax.set_title('Weight Magnitudes (per layer)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [2,0] Summary
        ax = axes[2, 0]
        ax.axis('off')
        lines = []
        if self.test_accs:
            lines.append(f"Best Test Acc: {max(self.test_accs):.2%}")
            lines.append(f"Final Test Acc: {self.test_accs[-1]:.2%}")
        if self.train_accs:
            lines.append(f"Final Train Acc (batch): {self.train_accs[-1]:.2%}")
        if self.inference_convergence:
            avg_conv = np.mean(self.inference_convergence[-100:])
            lines.append(f"\nAvg Convergence (last 100): {avg_conv:.2f}")
        lines.append(f"\neBPC baseline: 95.74% (3 epochs)")
        lines.append(f"ePC target: ~95%")
        ax.text(0.1, 0.5, '\n'.join(lines), fontsize=12,
                verticalalignment='center', family='monospace')

        # [2,1] Per-layer stats
        ax = axes[2, 1]
        ax.axis('off')
        lines = []
        for i, norms in enumerate(self.error_norms):
            if norms:
                recent = norms[-100:] if len(norms) >= 100 else norms
                lines.append(f"Layer {i+1}: ||e|| mean={np.mean(recent):.4f}, "
                             f"max={np.max(recent):.4f}")
        if self.inference_convergence:
            recent = self.inference_convergence[-100:]
            lines.append(f"\nConvergence: mean={np.mean(recent):.3f}")
        ax.text(0.05, 0.5, '\n'.join(lines), fontsize=10,
                verticalalignment='center', family='monospace')

        # [2,2] Early stopping stats
        ax = axes[2, 2]
        ax.axis('off')
        lines = []
        if self.iters_used:
            recent = self.iters_used[-100:] if len(self.iters_used) >= 100 else self.iters_used
            lines.append(f"Avg iters used: {np.mean(recent):.2f}")
            max_iters = max(self.iters_used)
            early_stop_rate = sum(1 for i in recent if i < max_iters) / len(recent)
            lines.append(f"Early stop rate: {early_stop_rate:.0%}")
            lines.append(f"Max iters (T): {max_iters}")
            lines.append(f"\nAll iters distribution:")
            from collections import Counter
            counts = Counter(self.iters_used)
            for k in sorted(counts.keys()):
                lines.append(f"  T={k}: {counts[k]} ({counts[k]/len(self.iters_used):.0%})")
        ax.text(0.05, 0.5, '\n'.join(lines), fontsize=10,
                verticalalignment='center', family='monospace')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Diagnostics saved to {save_path}")
        plt.close()


def get_weight_magnitudes(model):
    """Extract max absolute weight per layer."""
    mags = {}
    for i, layer in enumerate(model.layers):
        max_val = 0.0
        has_params = False
        for p in layer.parameters():
            max_val = max(max_val, p.data.abs().max().item())
            has_params = True
        if has_params:
            mags[f'Layer {i+1}'] = max_val
    return mags


def train_epoch(model, weight_optim, scaler, train_loader, device, epoch, diagnostics):
    model.train()
    total_correct = 0
    total_samples = 0
    use_amp = device == 'cuda'

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for data, target in pbar:
        data = data.view(data.size(0), -1).to(device)
        target = target.to(device)
        batch_size = data.size(0)

        # Phase 1: Inference (optimize errors) — fp16 forward, fp32 Newton step
        with autocast(enabled=use_amp):
            energy = model(data, target)

        # Collect diagnostics
        diag = model.get_diagnostics()

        # Phase 2: Weight update (local learning via E_local) — fp16 + GradScaler
        weight_optim.zero_grad()
        with autocast(enabled=use_amp):
            loss = model.compute_weight_loss(data, target, batch_size)
        scaler.scale(loss).backward()
        scaler.step(weight_optim)
        scaler.update()

        # Track accuracy + CE loss from cached E_local prediction (no extra forward pass)
        preds = model._weight_phase_prediction.argmax(dim=1)
        correct = (preds == target).sum().item()
        total_correct += correct
        total_samples += batch_size

        acc = correct / batch_size
        weight_mags = get_weight_magnitudes(model)

        ce_loss = F.cross_entropy(model._weight_phase_prediction.float(), target).item()

        diagnostics.update_train(
            acc=acc, loss=ce_loss,
            diagnostics=diag, weight_mags=weight_mags,
        )

        pbar.set_postfix(acc=f"{acc:.2%}", T=diag.get('iters_used', '?'))

    return total_correct / total_samples


def evaluate(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    use_amp = device == 'cuda'

    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            with autocast(enabled=use_amp):
                outputs = model(data)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += data.size(0)
            total_loss += F.cross_entropy(outputs.float(), target, reduction='sum').item()

    return total_correct / total_samples, total_loss / total_samples


def main():
    # Hyperparameters — Error optimization (inference phase)
    error_optim = 'newton'  # 'sgd', 'adam', or 'newton'
    iters = 2               # Error optimization steps (Newton needs only 1-2)
    e_lr = 0.01             # Learning rate for errors (SGD/Adam only)
    e_damping = 0.1         # Newton damping (lower = more aggressive)

    # Hyperparameters — Weight optimization (learning phase)
    w_lr = 0.001
    batch_size = 128
    num_epochs = 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Enable TF32 for fp32 matmuls (Ampere+), FP16 AMP for mixed precision
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    scaler = GradScaler(enabled=(device == 'cuda'))

    print("=" * 60)
    print("ePC MNIST Validation")
    print(f"Architecture: [784, 128, 128, 128, 10], ReLU")
    print(f"Mixed precision: FP16 autocast + GradScaler" if device == 'cuda' else "No AMP (CPU)")
    print(f"Inference: {error_optim} errors, T={iters}, "
          f"{'damping='+str(e_damping) if error_optim == 'newton' else 'e_lr='+str(e_lr)}")
    print(f"Learning: Adam weights, w_lr={w_lr}")
    print(f"Output loss: cross-entropy")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
    print("=" * 60)

    train_loader, test_loader = get_mnist_loaders(batch_size)

    architecture = get_mlp_mnist(hidden_size=128, num_hidden=3)
    model = PCE(
        architecture, iters=iters, e_lr=e_lr, output_loss='ce',
        error_optim=error_optim, damping=e_damping,
        early_stop_threshold=0.02,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_error_layers = len(model.layers) - 1
    print(f"Parameters: {num_params:,}")

    weight_optim = torch.optim.Adam(model.parameters(), lr=w_lr)
    diagnostics = Diagnostics(num_error_layers)

    best_test_acc = 0.0
    for epoch in range(num_epochs):
        train_acc = train_epoch(
            model, weight_optim, scaler, train_loader, device, epoch, diagnostics,
        )
        test_acc, test_loss = evaluate(model, test_loader, device)
        diagnostics.update_test(test_acc, test_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train {train_acc:.2%}, Test {test_acc:.2%}")

        diagnostics.plot(f'diagnostics_epc_mnist_epoch_{epoch+1}.png')

    print(f"\nBest test accuracy: {best_test_acc:.2%}")
    print(f"eBPC baseline: 95.74% (3 epochs, Hebbian updates)")
    print(f"Target: ~95%")

    # Early stopping summary
    if diagnostics.iters_used:
        from collections import Counter
        counts = Counter(diagnostics.iters_used)
        total = len(diagnostics.iters_used)
        print(f"\nEarly stopping: avg iters={np.mean(diagnostics.iters_used):.2f}")
        for k in sorted(counts.keys()):
            print(f"  T={k}: {counts[k]}/{total} ({counts[k]/total:.0%})")

    diagnostics.plot('diagnostics_epc_mnist_final.png')


if __name__ == "__main__":
    main()
