"""
Train ePC ResNet-18 on CIFAR-10.

Uses PCESkipConnection with ResNet-18 architecture matching the ePC paper
(Goemaere et al. 2025). Two-phase training: SGD error optimization for
inference, then local weight updates via E_local with Adam optimizer.

Target: 92.17% test accuracy (ePC paper)
Backprop baseline: 92.36%

Reference hyperparameters (from ePC repo cifar branch):
  batch_size=256, epochs=50, iters=5, e_lr=0.001
  w_lr=0.0001, w_decay=0.0, output_loss='mse'
  LR schedule: linear warmup (10%) + cosine decay
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import math
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from experiments.ePC_ResNet.epc_model import PCESkipConnection, quantize_model_weights
from experiments.ePC_ResNet.architectures import get_resnet18_cifar10
from experiments.ePC_ResNet.ada_woodbury import AdaWoodbury

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def get_cifar10_loaders(batch_size=256, data_dir='./data', num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    pin = torch.cuda.is_available()
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   drop_last=True, num_workers=num_workers,
                   pin_memory=pin, persistent_workers=(num_workers > 0)),
        DataLoader(test, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers,
                   pin_memory=pin, persistent_workers=(num_workers > 0)),
    )


def make_lr_schedule(optimizer, total_steps, base_lr=0.0001, warmup_fraction=0.1):
    """Cosine decay with linear warmup (matching ePC paper)."""
    peak_lr = 1.1 * base_lr
    end_lr = 0.1 * base_lr
    warmup_steps = int(warmup_fraction * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return base_lr + (peak_lr - base_lr) * (step / max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return end_lr + (peak_lr - end_lr) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class Diagnostics:
    """Collect and plot training diagnostics for ePC ResNet."""

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
        self.learning_rates = []
        self.weight_magnitudes = {}  # layer_name -> list

    def update_train(self, acc, loss, diagnostics, lr, weight_mags):
        self.train_accs.append(acc)
        self.train_losses.append(loss)
        self.inference_convergence.append(diagnostics['convergence'])
        self.iters_used.append(diagnostics.get('iters_used', 0))
        self.learning_rates.append(lr)

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

    def plot(self, save_path, epoch=None, num_epochs=None):
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))

        # [0,0] Accuracy
        ax = axes[0, 0]
        if self.train_accs:
            ax.plot(self.train_accs, alpha=0.4, linewidth=0.5, label='Train (batch)')
            # Moving average for readability
            if len(self.train_accs) > 50:
                window = min(50, len(self.train_accs) // 5)
                ma = np.convolve(self.train_accs, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.train_accs)), ma,
                        linewidth=1.5, label=f'Train (MA-{window})')
        if self.test_accs:
            n_train = len(self.train_accs)
            n_test = len(self.test_accs)
            if n_test > 0:
                test_x = [(i + 1) * n_train / n_test for i in range(n_test)]
                ax.plot(test_x, self.test_accs, 'o-', linewidth=2,
                        markersize=4, label='Test')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [0,1] Output Loss
        ax = axes[0, 1]
        if self.train_losses:
            ax.plot(self.train_losses, alpha=0.4, linewidth=0.5, label='Train (batch)')
            if len(self.train_losses) > 50:
                window = min(50, len(self.train_losses) // 5)
                ma = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.train_losses)), ma,
                        linewidth=1.5, label=f'Train (MA-{window})')
        if self.test_losses:
            n_train = len(self.train_losses)
            n_test = len(self.test_losses)
            if n_test > 0:
                test_x = [(i + 1) * n_train / n_test for i in range(n_test)]
                ax.plot(test_x, self.test_losses, 'o-', linewidth=2,
                        markersize=4, label='Test')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Output Loss (MSE)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [0,2] Per-Layer Energies
        ax = axes[0, 2]
        for i, energies in enumerate(self.layer_energies):
            if energies:
                ax.plot(energies, label=f'Layer {i+1}', alpha=0.7, linewidth=0.8)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Energy')
        ax.set_title('Per-Layer Energies (0.5 ||e_i||^2)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # [1,0] Inference Convergence
        ax = axes[1, 0]
        if self.inference_convergence:
            ax.plot(self.inference_convergence, alpha=0.5, linewidth=0.5)
            if len(self.inference_convergence) > 50:
                window = min(50, len(self.inference_convergence) // 5)
                ma = np.convolve(self.inference_convergence,
                                 np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.inference_convergence)), ma,
                        linewidth=1.5, color='red', label=f'MA-{window}')
                ax.legend(fontsize=8)
        ax.set_xlabel('Batch')
        ax.set_ylabel('E_initial - E_final')
        ax.set_title('Inference Convergence (E_0 - E_T)')
        ax.grid(True, alpha=0.3)

        # [1,1] Error Magnitudes
        ax = axes[1, 1]
        for i, norms in enumerate(self.error_norms):
            if norms:
                ax.plot(norms, label=f'Layer {i+1}', alpha=0.7, linewidth=0.8)
        ax.set_xlabel('Batch')
        ax.set_ylabel('||e_i||')
        ax.set_title('Error Magnitudes (per layer)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # [1,2] Learning Rate Schedule
        ax = axes[1, 2]
        if self.learning_rates:
            ax.plot(self.learning_rates, linewidth=1.5)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, -4))

        # [2,0] Weight Magnitudes
        ax = axes[2, 0]
        for name, mags in self.weight_magnitudes.items():
            if mags:
                ax.plot(mags, label=name, alpha=0.7, linewidth=0.8)
        ax.set_xlabel('Batch')
        ax.set_ylabel('max|W|')
        ax.set_title('Weight Magnitudes (per layer)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # [2,1] Summary
        ax = axes[2, 1]
        ax.axis('off')
        lines = []
        if self.test_accs:
            lines.append(f"Best Test Acc: {max(self.test_accs):.2%}")
            lines.append(f"Final Test Acc: {self.test_accs[-1]:.2%}")
        if self.train_accs:
            lines.append(f"Final Train Acc (batch): {self.train_accs[-1]:.2%}")
        if epoch is not None and num_epochs is not None:
            lines.append(f"\nEpoch: {epoch}/{num_epochs}")
        if self.inference_convergence:
            avg_conv = np.mean(self.inference_convergence[-100:])
            lines.append(f"Avg Convergence (last 100): {avg_conv:.2f}")
        lines.append(f"\nePC paper target: 92.17%")
        lines.append(f"Backprop baseline: 92.36%")
        ax.text(0.1, 0.5, '\n'.join(lines), fontsize=11,
                verticalalignment='center', family='monospace')

        # [2,2] Per-layer error stats
        ax = axes[2, 2]
        ax.axis('off')
        lines = []
        for i, norms in enumerate(self.error_norms):
            if norms:
                recent = norms[-100:] if len(norms) >= 100 else norms
                lines.append(f"Layer {i+1}: ||e|| mean={np.mean(recent):.3f}, "
                             f"max={np.max(recent):.3f}")
        if self.inference_convergence:
            recent = self.inference_convergence[-100:]
            lines.append(f"\nConvergence: mean={np.mean(recent):.2f}, "
                         f"min={np.min(recent):.2f}")
        if self.iters_used:
            recent_iters = self.iters_used[-100:] if len(self.iters_used) >= 100 else self.iters_used
            lines.append(f"Avg iters used: {np.mean(recent_iters):.2f}")
            early_stop_rate = sum(1 for i in recent_iters if i < max(recent_iters)) / len(recent_iters)
            lines.append(f"Early stop rate: {early_stop_rate:.0%}")
        ax.text(0.05, 0.5, '\n'.join(lines), fontsize=9,
                verticalalignment='center', family='monospace')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Diagnostics saved to {save_path}")
        plt.close()


def get_weight_magnitudes(model):
    """Extract max absolute weight per named parameter group."""
    mags = {}
    layer_idx = 0
    for i, layer in enumerate(model.layers):
        has_params = False
        max_val = 0.0
        for p in layer.parameters():
            max_val = max(max_val, p.data.abs().max().item())
            has_params = True
        if has_params:
            layer_idx += 1
            mags[f'L{layer_idx}'] = max_val
    return mags


def evaluate_with_loss(model, test_loader, device, output_loss='mse'):
    """Evaluate model returning both accuracy and loss."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    use_amp = device == 'cuda'

    with torch.inference_mode():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with autocast('cuda', enabled=use_amp):
                outputs = model(data)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += data.size(0)

            if output_loss == 'mse':
                y_onehot = F.one_hot(target, num_classes=outputs.shape[-1]).float()
                total_loss += 0.5 * F.mse_loss(
                    outputs.float(), y_onehot, reduction='sum').item()
            else:
                total_loss += F.cross_entropy(
                    outputs.float(), target, reduction='sum').item()

    return total_correct / total_samples, total_loss / total_samples


def train_epoch(model, weight_optim, lr_scheduler, scaler, train_loader,
                device, epoch, diagnostics, accum_steps=1):
    model.train()
    total_correct = 0
    total_samples = 0
    total_energy = 0.0
    use_amp = device == 'cuda'

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    weight_optim.zero_grad()
    accum_count = 0

    for data, target in pbar:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = data.size(0)

        # Phase 1: Inference (optimize errors) — fp16 forward, fp32 Newton step
        with autocast('cuda', enabled=use_amp):
            energy = model(data, target)

        # Collect diagnostics before weight update
        diag = model.get_diagnostics()

        # Phase 2: Weight gradient accumulation — fp16 + GradScaler
        with autocast('cuda', enabled=use_amp):
            loss = model.compute_weight_loss(data, target, batch_size)
        scaler.scale(loss / accum_steps).backward()

        accum_count += 1
        if accum_count >= accum_steps:
            scaler.step(weight_optim)
            scaler.update()
            lr_scheduler.step()
            weight_optim.zero_grad()
            accum_count = 0

        total_energy += energy

        # Track accuracy from cached E_local prediction (no extra forward pass)
        preds = model._weight_phase_prediction.argmax(dim=1)
        correct = (preds == target).sum().item()
        total_correct += correct
        total_samples += batch_size

        acc = correct / batch_size
        lr = lr_scheduler.get_last_lr()[0]
        weight_mags = get_weight_magnitudes(model)

        diagnostics.update_train(
            acc=acc, loss=energy / batch_size,
            diagnostics=diag, lr=lr, weight_mags=weight_mags,
        )

        pbar.set_postfix(
            acc=f"{acc:.1%}",
            lr=f"{lr:.2e}",
            T=f"{diag.get('iters_used', '?')}",
        )

    return total_correct / total_samples, total_energy / len(train_loader)


def main():
    # Hyperparameters
    error_optim = 'newton'  # 'sgd', 'adam', or 'newton'
    iters = 2           # Error optimization steps (Newton needs only 1-2)
    damping = 0.1       # Newton damping (lower = more aggressive)
    e_lr = 0.001        # Learning rate for errors (SGD/Adam only)
    w_lr = 0.0001       # Base learning rate for weights
    w_decay = 0.0       # Weight decay
    w_optim = 'adawoodbury'  # 'adam', 'adam8bit', or 'adawoodbury'
    batch_size = 256
    num_epochs = 50
    output_loss = 'mse'
    chart_interval = 10  # Save diagnostic chart every N epochs
    quantize_bits = 0   # QAT disabled — INT8 too destructive for ePC error optimization
    accum_steps = 1     # Gradient accumulation (increase to 2/4 if OOM)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Enable TF32 + cuDNN autotuning (Ampere+), FP16 AMP for mixed precision
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    scaler = GradScaler('cuda', enabled=(device == 'cuda'))

    optim_name = w_optim
    if w_optim == 'adam8bit' and not HAS_BNB:
        optim_name = 'adam (bnb unavailable)'
    quant_str = f'INT{quantize_bits} QAT' if quantize_bits > 0 else 'none'

    print("=" * 60)
    print("ePC ResNet-18 on CIFAR-10")
    print(f"Inference: {error_optim} errors, T={iters}, "
          f"{'damping='+str(damping) if error_optim == 'newton' else 'e_lr='+str(e_lr)}")
    print(f"Learning: {optim_name}, w_lr={w_lr}, w_decay={w_decay}")
    print(f"Output loss: {output_loss}")
    print(f"Mixed precision: FP16 autocast + GradScaler" if device == 'cuda' else "No AMP (CPU)")
    if quantize_bits > 0:
        print(f"Weight quantization: {quant_str}")
    print(f"LR schedule: warmup 10% + cosine decay")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}"
          f"{f', accum={accum_steps}' if accum_steps > 1 else ''}")
    print(f"Target: 92.17% (ePC paper)")
    print("=" * 60)

    train_loader, test_loader = get_cifar10_loaders(batch_size)

    architecture = get_resnet18_cifar10()
    model = PCESkipConnection(
        architecture, iters=iters, e_lr=e_lr, output_loss=output_loss,
        error_optim=error_optim, damping=damping,
        early_stop_threshold=0.02,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_error_layers = len(model.layers) - 1
    print(f"Parameters: {num_params:,}")
    print(f"Error layers: {num_error_layers}")

    if quantize_bits > 0:
        n_quantized = quantize_model_weights(model, num_bits=quantize_bits)
        print(f"QAT: {n_quantized} layers fake-quantized to INT{quantize_bits}")

    # Weight optimizer (lr=1.0, actual LR controlled by scheduler)
    if w_optim == 'adawoodbury':
        weight_optim = AdaWoodbury(
            model.parameters(), lr=1.0, weight_decay=w_decay,
            alpha=1.0, warmup_steps=100,
        )
    elif w_optim == 'adam8bit' and HAS_BNB:
        weight_optim = bnb.optim.Adam8bit(
            model.parameters(), lr=1.0, weight_decay=w_decay,
        )
    else:
        weight_optim = torch.optim.Adam(
            model.parameters(), lr=1.0, weight_decay=w_decay,
        )
    # LR schedule accounts for gradient accumulation (fewer optimizer steps)
    steps_per_epoch = len(train_loader) // accum_steps
    total_steps = steps_per_epoch * num_epochs
    lr_scheduler = make_lr_schedule(weight_optim, total_steps, base_lr=w_lr)

    diagnostics = Diagnostics(num_error_layers)

    best_test_acc = 0.0
    for epoch in range(num_epochs):
        train_acc, avg_energy = train_epoch(
            model, weight_optim, lr_scheduler, scaler, train_loader,
            device, epoch, diagnostics, accum_steps=accum_steps,
        )
        test_acc, test_loss = evaluate_with_loss(
            model, test_loader, device, output_loss,
        )
        diagnostics.update_test(test_acc, test_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(f"Epoch {epoch+1:3d}/{num_epochs}: "
              f"Train {train_acc:.2%}, Test {test_acc:.2%}, "
              f"Energy {avg_energy:.1f}, Best {best_test_acc:.2%}")

        # Save diagnostic chart periodically
        if (epoch + 1) % chart_interval == 0 or epoch == num_epochs - 1:
            diagnostics.plot(
                f'diagnostics_cifar10_epoch_{epoch+1}.png',
                epoch=epoch + 1, num_epochs=num_epochs,
            )

    print(f"\nBest test accuracy: {best_test_acc:.2%}")
    print(f"ePC paper target: 92.17%")
    print(f"Backprop baseline: 92.36%")

    # Final chart
    diagnostics.plot('diagnostics_cifar10_final.png',
                     epoch=num_epochs, num_epochs=num_epochs)


if __name__ == "__main__":
    main()
