"""
Train eBPC (Error-based Bayesian Predictive Coding) on MNIST

Combines ePC fast inference with BPC Bayesian weight updates.
Compare to: BPC 93.5% (1 epoch), baseline PC 95.14% (1 epoch).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from experiments.eBPC.ebpc_layer import eBPCNetwork
from experiments.eBPC.ebpc_trainer import eBPCTrainer


def get_mnist_loaders(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class Diagnostics:
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

    def plot_diagnostics(self, save_path='diagnostics.png'):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        ax = axes[0, 0]
        if self.train_accs:
            ax.plot(self.train_accs, label='Train', alpha=0.7)
        if self.test_accs:
            n_train = len(self.train_accs)
            n_test = len(self.test_accs)
            test_x = [(i + 1) * n_train / n_test for i in range(n_test)]
            ax.plot(test_x, self.test_accs, label='Test', linewidth=2, marker='o')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if self.train_losses:
            ax.plot(self.train_losses, label='Train', alpha=0.7)
        if self.test_losses:
            n_train = len(self.train_losses)
            n_test = len(self.test_losses)
            test_x = [(i + 1) * n_train / n_test for i in range(n_test)]
            ax.plot(test_x, self.test_losses, label='Test', linewidth=2, marker='o')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Cross-Entropy Loss (logging only)')
        ax.legend()
        ax.grid(True, alpha=0.3)

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

        ax = axes[1, 0]
        if self.inference_convergence:
            ax.plot(self.inference_convergence)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Energy Reduction')
        ax.set_title('Inference Convergence (E_0 - E_T)')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for i, uncertainties in enumerate(self.layer_uncertainties):
            if uncertainties:
                ax.plot(uncertainties, label=f'Layer {i+1}', alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Tr(V)')
        ax.set_title('Weight Posterior Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        ax = axes[1, 2]
        ax.axis('off')
        summary = []
        if self.test_accs:
            summary.append(f"Best Test Acc: {max(self.test_accs):.2%}")
            summary.append(f"Final Test Acc: {self.test_accs[-1]:.2%}")
        if self.train_accs:
            summary.append(f"Final Train Acc: {self.train_accs[-1]:.2%}")
        summary.append(f"\nBaseline BPC: 93.5% (1 epoch)")
        summary.append(f"Baseline PC: 95.14% (1 epoch)")
        ax.text(0.1, 0.5, '\n'.join(summary), fontsize=12, verticalalignment='center')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Diagnostics saved to {save_path}")
        plt.close()


def train_epoch(model, trainer, train_loader, epoch, diagnostics):
    model.train()
    total_correct = 0
    total_samples = 0
    epoch_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.view(data.size(0), -1)

        results = trainer.train_on_batch(inputs=data, targets=target)

        # Compute accuracy using eval forward
        model.eval()
        with torch.no_grad():
            outputs = model(data.to(trainer.device))
            pred = outputs.argmax(dim=1)
            correct = (pred == target.to(trainer.device)).sum().item()
        model.train()

        total_correct += correct
        total_samples += data.size(0)
        epoch_loss += results['loss']

        acc = correct / data.size(0)
        convergence = results['energy_history'][0] - results['energy_history'][-1]

        diagnostics.update_train(
            loss=results['loss'],
            acc=acc,
            layer_energies=results['layer_energies'],
            convergence=convergence,
            uncertainties=model.get_uncertainties(),
        )

        pbar.set_postfix({
            'loss': f"{results['loss']:.4f}",
            'acc': f"{acc:.2%}",
            'conv': f"{convergence:.4f}",
        })

    return epoch_loss / len(train_loader), total_correct / total_samples


def test(model, trainer, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            results = trainer.test_on_batch(inputs=data, loss_fn=F.cross_entropy, targets=target)
            pred = results['outputs'].argmax(dim=1)
            correct = (pred == target.to(trainer.device)).sum().item()
            total_correct += correct
            total_samples += data.size(0)
            total_loss += results['loss'] * data.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    # Hyperparameters
    layer_sizes = [784, 128, 128, 128, 10]  # Same as BPC (Appendix F.1)
    activation = 'relu'
    T = 5                  # ePC needs fewer iterations (paper: T=4-5)
    e_lr = 0.01            # Adam LR (BPC Appendix F.1)
    kappa = 0.25           # BPC Hebbian decay
    batch_size = 128       # Same as BPC
    num_epochs = 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\n" + "="*80)
    print("eBPC: Error-based Bayesian Predictive Coding on MNIST")
    print("="*80)
    print(f"Architecture: {layer_sizes}")
    print(f"Inference: ePC error optimization (T={T}, e_lr={e_lr}, Adam)")
    print(f"Learning: BPC Hebbian update (kappa={kappa})")
    print(f"Batch size: {batch_size}")
    print("="*80 + "\n")

    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    model = eBPCNetwork(layer_sizes=layer_sizes, activation=activation)
    print(f"Natural parameters: {sum(p.numel() for p in model.get_natural_parameters()):,}")
    print(f"Layers: {len(model.layers)}")

    trainer = eBPCTrainer(model=model, T=T, e_lr=e_lr, kappa=kappa, device=device)
    diagnostics = Diagnostics(num_layers=len(model.layers))

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

        diagnostics.plot_diagnostics(f'diagnostics_ebpc_epoch_{epoch+1}.png')

    print(f"\n{'='*80}")
    print(f"Best test accuracy: {best_test_acc:.2%}")
    print(f"BPC baseline: 93.5% (1 epoch)")
    print(f"PC baseline: 95.14% (1 epoch)")
    print(f"{'='*80}\n")

    diagnostics.plot_diagnostics('diagnostics_ebpc_final.png')


if __name__ == "__main__":
    main()
