"""
Train Bayesian Predictive Coding Network on MNIST

Implements Algorithm 1 from "Bayesian Predictive Coding" (Tschantz et al., 2025).
Compares to baseline PC (95.63% test accuracy).
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from experiments.BayesianPC.bayesian_pc_layer import BayesianPCNetwork
from experiments.BayesianPC.bayesian_pc_trainer import BayesianPCTrainer


def get_mnist_loaders(batch_size=64, data_dir='./data'):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
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
    """Track training diagnostics."""

    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.reset()

    def reset(self):
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.layer_energies = [[] for _ in range(self.num_layers)]
        self.layer_uncertainties = [[] for _ in range(self.num_layers)]
        self.inference_convergence = []

    def update_train(self, loss, acc, layer_energies, convergence, uncertainties=None):
        self.train_losses.append(loss)
        self.train_accs.append(acc)
        self.inference_convergence.append(convergence)

        for i, energy in enumerate(layer_energies):
            if i < len(self.layer_energies):
                self.layer_energies[i].append(energy)

        if uncertainties is not None:
            for i, unc in enumerate(uncertainties):
                if i < len(self.layer_uncertainties):
                    self.layer_uncertainties[i].append(unc)

    def update_test(self, loss, acc):
        self.test_losses.append(loss)
        self.test_accs.append(acc)

    def check_vanishing_errors(self):
        if len(self.layer_energies[-1]) < 10:
            return False
        recent_deep = np.mean(self.layer_energies[-1][-10:])
        recent_shallow = np.mean(self.layer_energies[0][-10:])
        if recent_shallow > 0 and recent_deep / recent_shallow < 0.01:
            return True
        return False

    def plot_diagnostics(self, save_path='diagnostics.png'):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Accuracy
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

        # Plot 2: Loss
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
        ax.set_title('Inference Convergence')
        ax.grid(True, alpha=0.3)

        # Plot 5: Weight uncertainties (BPC-specific)
        ax = axes[1, 1]
        for i, uncertainties in enumerate(self.layer_uncertainties):
            if len(uncertainties) > 0:
                ax.plot(uncertainties, label=f'Layer {i+1}', alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Tr(V) - Weight Uncertainty')
        ax.set_title('Weight Posterior Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 6: Summary
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
        summary.append(f"\nBaseline PC: 95.63%")
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

        # Train on batch
        results = trainer.train_on_batch(
            inputs=data,
            loss_fn=F.cross_entropy,
            targets=target,
        )

        # Compute accuracy
        model.eval()
        with torch.no_grad():
            outputs = model(data.to(trainer.device), sample_x=False)
            pred = outputs.argmax(dim=1)
            correct = (pred == target.to(trainer.device)).sum().item()

        total_correct += correct
        total_samples += data.size(0)
        epoch_loss += results['loss']

        # Update diagnostics
        acc = correct / data.size(0)
        convergence = results['free_energy_history'][0] - results['free_energy_history'][-1]

        layer_energies = []
        for layer in model.layers:
            energy = layer.energy()
            if energy is not None:
                layer_energies.append(energy.item())

        uncertainties = model.get_uncertainties()

        diagnostics.update_train(
            loss=results['loss'],
            acc=acc,
            layer_energies=layer_energies,
            convergence=convergence,
            uncertainties=uncertainties,
        )

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
    # Hyperparameters - From Appendix F.1 (page 12)
    layer_sizes = [784, 128, 128, 128, 10]  # 4 layers, 128 units per hidden layer
    activation = 'relu'
    T_inference = 10  # 10 iterations per batch (Appendix F.1)
    inference_lr = 0.01  # Adam with LR=0.01 (Appendix F.1)
    kappa = 0.25  # Learning rate decay for natural params
    batch_size = 128  # Batch size 128 (Appendix F.1)
    num_epochs = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Check MISTAKES.md
    if not os.path.exists('MISTAKES.md'):
        print("⚠️  WARNING: MISTAKES.md not found!")
        return

    print("\n" + "="*80)
    print("Bayesian Predictive Coding Network on MNIST")
    print("="*80)
    print(f"Architecture: {layer_sizes}")
    print(f"Activation: {activation}")
    print(f"Inference iterations (T): {T_inference}")
    print(f"Inference LR: {inference_lr}")
    print(f"Natural param LR decay (κ): {kappa}")
    print(f"Batch size: {batch_size}")
    print(f"Weight posterior: Matrix Normal Wishart")
    print(f"Weight updates: Closed-form Hebbian (Equation 7)")
    print("="*80 + "\n")

    # Load data
    print("Loading MNIST...")
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    # Create Bayesian model
    print("Creating Bayesian PC model...")
    model = BayesianPCNetwork(
        layer_sizes=layer_sizes,
        activation=activation,
    )
    print(f"Number of natural parameters: {sum(p.numel() for p in model.get_natural_parameters()):,}")
    print(f"Number of PC layers: {len(model.layers)}")

    # Create trainer
    trainer = BayesianPCTrainer(
        model=model,
        T=T_inference,
        inference_lr=inference_lr,
        kappa=kappa,
        device=device,
    )

    # Create diagnostics
    diagnostics = Diagnostics(num_layers=len(model.layers))

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
            torch.save(model.state_dict(), 'best_bpc_model.pt')

        if diagnostics.check_vanishing_errors():
            print("  ⚠️  WARNING: Vanishing error signals detected!")

        diagnostics.plot_diagnostics(f'diagnostics_bpc_epoch_{epoch+1}.png')

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Best test accuracy: {best_test_acc:.2%}")
    print(f"Baseline PC accuracy: 95.63%")
    if best_test_acc >= 0.9563:
        print(f"  ✓ MATCHES OR EXCEEDS BASELINE")
    else:
        print(f"  ⚠️  Below baseline by {(0.9563 - best_test_acc)*100:.2f}%")
    print(f"{'='*80}\n")

    diagnostics.plot_diagnostics('diagnostics_bpc_final.png')


if __name__ == "__main__":
    main()
