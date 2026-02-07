"""
Train 7-layer Predictive Coding Network on MNIST

Tests the minimal PC implementation with comprehensive diagnostics.
Monitors for vanishing errors, convergence, and compares to backprop baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from src.network import PCNetwork, PCTrainer


def get_mnist_loaders(batch_size=64, data_dir='./data'):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


class Diagnostics:
    """Track training diagnostics to detect problems."""

    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.layer_energies = [[] for _ in range(self.num_layers)]
        self.inference_convergence = []
        self.gradient_norms = []

    def update_train(self, loss, acc, layer_energies, convergence):
        """Update training statistics."""
        self.train_losses.append(loss)
        self.train_accs.append(acc)
        self.inference_convergence.append(convergence)

        for i, energy in enumerate(layer_energies):
            if i < len(self.layer_energies):
                self.layer_energies[i].append(energy)

    def update_test(self, loss, acc):
        """Update test statistics."""
        self.test_losses.append(loss)
        self.test_accs.append(acc)

    def check_vanishing_errors(self):
        """Check if error signals are vanishing in deep layers."""
        if len(self.layer_energies[-1]) < 10:
            return False

        # Check if last layer has very small energies
        recent_deep_energies = self.layer_energies[-1][-10:]
        mean_deep = np.mean(recent_deep_energies)

        # Check if first layer has normal energies
        recent_shallow_energies = self.layer_energies[0][-10:]
        mean_shallow = np.mean(recent_shallow_energies)

        # Warn if deep layer has 100x smaller energy than shallow
        if mean_shallow > 0 and mean_deep / mean_shallow < 0.01:
            return True
        return False

    def plot_diagnostics(self, save_path='diagnostics.png'):
        """Plot comprehensive diagnostics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Training and test accuracy
        ax = axes[0, 0]
        if len(self.train_accs) > 0:
            ax.plot(self.train_accs, label='Train', alpha=0.7)
        if len(self.test_accs) > 0:
            test_x = np.linspace(0, len(self.train_accs), len(self.test_accs))
            ax.plot(test_x, self.test_accs, label='Test', linewidth=2)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Training and test loss
        ax = axes[0, 1]
        if len(self.train_losses) > 0:
            ax.plot(self.train_losses, label='Train', alpha=0.7)
        if len(self.test_losses) > 0:
            test_x = np.linspace(0, len(self.train_losses), len(self.test_losses))
            ax.plot(test_x, self.test_losses, label='Test', linewidth=2)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Per-layer energies
        ax = axes[0, 2]
        for i, energies in enumerate(self.layer_energies):
            if len(energies) > 0:
                ax.plot(energies, label=f'Layer {i+1}', alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Energy')
        ax.set_title('Per-Layer Prediction Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 4: Inference convergence
        ax = axes[1, 0]
        if len(self.inference_convergence) > 0:
            ax.plot(self.inference_convergence)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Free Energy Reduction')
        ax.set_title('Inference Convergence (first to last iteration)')
        ax.grid(True, alpha=0.3)

        # Plot 5: Energy ratio (deep vs shallow)
        ax = axes[1, 1]
        if len(self.layer_energies[0]) > 0 and len(self.layer_energies[-1]) > 0:
            ratios = []
            for i in range(min(len(self.layer_energies[0]), len(self.layer_energies[-1]))):
                shallow = self.layer_energies[0][i]
                deep = self.layer_energies[-1][i]
                if shallow > 0:
                    ratios.append(deep / shallow)
            if ratios:
                ax.plot(ratios)
                ax.axhline(y=0.01, color='r', linestyle='--', label='Warning threshold')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Ratio')
        ax.set_title('Energy Ratio (Deep / Shallow)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 6: Summary text
        ax = axes[1, 2]
        ax.axis('off')
        summary = []
        if len(self.test_accs) > 0:
            summary.append(f"Best Test Acc: {max(self.test_accs):.2%}")
            summary.append(f"Final Test Acc: {self.test_accs[-1]:.2%}")
        if len(self.train_accs) > 0:
            summary.append(f"Final Train Acc: {self.train_accs[-1]:.2%}")
        summary.append(f"\nVanishing Errors: {'YES ⚠️' if self.check_vanishing_errors() else 'NO ✓'}")
        summary.append(f"Num Layers: {self.num_layers}")

        ax.text(0.1, 0.5, '\n'.join(summary), fontsize=12, verticalalignment='center')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Diagnostics saved to {save_path}")
        plt.close()


def train_epoch(model, trainer, train_loader, epoch, diagnostics):
    """Train for one epoch."""
    model.train()

    total_correct = 0
    total_samples = 0
    epoch_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, (data, target) in enumerate(pbar):
        # Flatten images
        data = data.view(data.size(0), -1)

        # Train on batch using PC two-phase algorithm
        results = trainer.train_on_batch(
            inputs=data,
            loss_fn=F.cross_entropy,
            targets=target,
        )

        # Compute accuracy
        model.eval()
        with torch.no_grad():
            outputs = model(data.to(trainer.device))
            pred = outputs.argmax(dim=1)
            correct = (pred == target.to(trainer.device)).sum().item()

        total_correct += correct
        total_samples += data.size(0)
        epoch_loss += results['loss']

        # Update diagnostics
        acc = correct / data.size(0)
        convergence = results['free_energy_history'][0] - results['free_energy_history'][-1]

        # Get per-layer energies
        layer_energies = []
        for pc_layer in model.get_pc_layers():
            energy = pc_layer.energy()
            if energy is not None:
                layer_energies.append(energy.item())

        diagnostics.update_train(
            loss=results['loss'],
            acc=acc,
            layer_energies=layer_energies,
            convergence=convergence,
        )

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{results['loss']:.4f}",
            'acc': f"{acc:.2%}",
            'conv': f"{convergence:.2e}",
        })

        model.train()

    avg_loss = epoch_loss / len(train_loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def test(model, trainer, test_loader):
    """Evaluate on test set."""
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)

            results = trainer.test_on_batch(
                inputs=data,
                loss_fn=F.cross_entropy,
                targets=target,
            )

            pred = results['outputs'].argmax(dim=1)
            correct = (pred == target.to(trainer.device)).sum().item()

            total_correct += correct
            total_samples += data.size(0)
            total_loss += results['loss'] * data.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def main():
    """Main training loop."""
    # Hyperparameters from NETWORK_PROPOSAL.md
    layer_sizes = [784, 256, 256, 256, 256, 256, 128, 10]  # 7 layers
    activation = 'relu'
    T_inference = 35  # 5 * 7 layers
    inference_lr = 0.1
    weight_lr = 0.001
    batch_size = 64
    num_epochs = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Check MISTAKES.md was consulted
    if not os.path.exists('MISTAKES.md'):
        print("⚠️  WARNING: MISTAKES.md not found! Please consult before training.")
        return

    print("\n" + "="*80)
    print("7-Layer Predictive Coding Network on MNIST")
    print("="*80)
    print(f"Architecture: {layer_sizes}")
    print(f"Activation: {activation}")
    print(f"Inference iterations (T): {T_inference}")
    print(f"Inference LR: {inference_lr}")
    print(f"Weight LR: {weight_lr}")
    print(f"Batch size: {batch_size}")
    print("="*80 + "\n")

    # Load data
    print("Loading MNIST...")
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    # Create model
    print("Creating model...")
    model = PCNetwork(layer_sizes=layer_sizes, activation=activation)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of PC layers: {len(model.get_pc_layers())}")

    # Create trainer
    trainer = PCTrainer(
        model=model,
        T=T_inference,
        inference_lr=inference_lr,
        weight_lr=weight_lr,
        device=device,
    )

    # Create diagnostics tracker
    diagnostics = Diagnostics(num_layers=len(model.get_pc_layers()))

    # Training loop
    print("\nTraining...")
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, trainer, train_loader, epoch, diagnostics)
        test_loss, test_acc = test(model, trainer, test_loader)

        diagnostics.update_test(test_loss, test_acc)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Save best model
            torch.save(model.state_dict(), 'best_pc_model.pt')

        # Check for vanishing errors
        if diagnostics.check_vanishing_errors():
            print("  ⚠️  WARNING: Vanishing error signals detected!")

        # Plot diagnostics every epoch
        diagnostics.plot_diagnostics(f'diagnostics_epoch_{epoch+1}.png')

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Best test accuracy: {best_test_acc:.2%}")
    print(f"{'='*80}\n")

    # Final diagnostics
    diagnostics.plot_diagnostics('diagnostics_final.png')

    # Compare to target
    print("\nComparison to targets (from NETWORK_PROPOSAL.md):")
    print(f"  Target accuracy: >95%")
    print(f"  Achieved accuracy: {best_test_acc:.2%}")
    if best_test_acc >= 0.95:
        print("  ✓ SUCCESS: Met target accuracy!")
    else:
        print("  ✗ BELOW TARGET: May need hyperparameter tuning or μPC scaling")

    print(f"\n  Vanishing errors: {'YES (needs μPC)' if diagnostics.check_vanishing_errors() else 'NO'}")


if __name__ == "__main__":
    main()
