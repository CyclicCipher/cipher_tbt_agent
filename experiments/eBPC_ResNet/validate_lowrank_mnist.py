"""
Validate low-rank eBPC on MNIST

Confirms low-rank V + diagonal Ψ approximation works where diagonal V failed.
Target: match or approach full eBPC baseline (~95.7% test, 3 epochs).
Diagonal baseline: 9.80% (NaN explosion — complete failure).
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

from experiments.eBPC_ResNet.ebpc_lowrank_layer import LowRankeBPCNetwork
from experiments.eBPC_ResNet.ebpc_lowrank_trainer import LowRankeBPCTrainer


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
        self.actual_T_history = []
        self.phi_mins = [[] for _ in range(self.num_layers)]
        self.M_maxes = [[] for _ in range(self.num_layers)]

    def update_train(self, loss, acc, layer_energies, convergence,
                     uncertainties=None, actual_T=None, phi_mins=None, M_maxes=None):
        self.train_losses.append(loss)
        self.train_accs.append(acc)
        self.inference_convergence.append(convergence)
        if actual_T is not None:
            self.actual_T_history.append(actual_T)
        for i, energy in enumerate(layer_energies):
            if i < len(self.layer_energies):
                self.layer_energies[i].append(energy)
        if uncertainties is not None:
            for i, unc in enumerate(uncertainties):
                if i < len(self.layer_uncertainties):
                    self.layer_uncertainties[i].append(unc)
        if phi_mins is not None:
            for i, pm in enumerate(phi_mins):
                if i < len(self.phi_mins):
                    self.phi_mins[i].append(pm)
        if M_maxes is not None:
            for i, mm in enumerate(M_maxes):
                if i < len(self.M_maxes):
                    self.M_maxes[i].append(mm)

    def update_test(self, loss, acc):
        self.test_losses.append(loss)
        self.test_accs.append(acc)

    def plot_diagnostics(self, save_path='diagnostics_lowrank.png'):
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # Accuracy
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

        # Loss
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

        # Per-layer energies
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

        # Adaptive T
        ax = axes[1, 0]
        if self.actual_T_history:
            window = min(50, len(self.actual_T_history))
            avg_T = [np.mean(self.actual_T_history[max(0, i-window):i+1])
                     for i in range(len(self.actual_T_history))]
            ax.plot(avg_T, label='Avg T (moving)')
            ax.axhline(y=self.actual_T_history[0] if self.actual_T_history else 5,
                       color='r', linestyle='--', alpha=0.5, label='Max T')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Iterations')
        ax.set_title('Adaptive Inference Iterations')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Weight posterior uncertainty
        ax = axes[1, 1]
        for i, uncertainties in enumerate(self.layer_uncertainties):
            if uncertainties:
                ax.plot(uncertainties, label=f'Layer {i+1}', alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Sum(diag(V))')
        ax.set_title('Weight Posterior Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Phi_min (the critical diagnostic — should stay positive)
        ax = axes[1, 2]
        for i, pmins in enumerate(self.phi_mins):
            if pmins:
                ax.plot(pmins, label=f'Layer {i+1}', alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero line')
        ax.set_xlabel('Batch')
        ax.set_ylabel('min(Ψ^{-1}_diag)')
        ax.set_title('Psi_inv Minimum (must stay > 0)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # M max (weight magnitude — should not explode)
        ax = axes[2, 0]
        for i, mmaxes in enumerate(self.M_maxes):
            if mmaxes:
                ax.plot(mmaxes, label=f'Layer {i+1}', alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('max(|M|)')
        ax.set_title('Weight Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Summary text
        ax = axes[2, 1]
        ax.axis('off')
        summary = []
        if self.test_accs:
            summary.append(f"Best Test Acc: {max(self.test_accs):.2%}")
            summary.append(f"Final Test Acc: {self.test_accs[-1]:.2%}")
        if self.train_accs:
            summary.append(f"Final Train Acc: {self.train_accs[-1]:.2%}")
        if self.actual_T_history:
            summary.append(f"\nAvg T (last 100): {np.mean(self.actual_T_history[-100:]):.1f}")
        summary.append(f"\nFull eBPC baseline: 95.74% (3 epochs)")
        summary.append(f"BPC baseline: 93.5% (1 epoch)")
        summary.append(f"Diagonal eBPC: 9.80% (NaN failure)")
        ax.text(0.1, 0.5, '\n'.join(summary), fontsize=12, verticalalignment='center')

        # Phi stability analysis
        ax = axes[2, 2]
        ax.axis('off')
        stability = []
        any_negative = False
        for i, pmins in enumerate(self.phi_mins):
            if pmins:
                min_phi = min(pmins)
                final_phi = pmins[-1]
                stability.append(f"Layer {i+1}: min={min_phi:.2e}, final={final_phi:.2e}")
                if min_phi <= 0:
                    any_negative = True
        stability.append(f"\nPhi positive throughout: {'NO (PROBLEM!)' if any_negative else 'YES (good)'}")
        ax.text(0.1, 0.5, '\n'.join(stability), fontsize=11, verticalalignment='center',
                fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Diagnostics saved to {save_path}")
        plt.close()


def get_phi_diagnostics(model):
    """Extract Ψ^{-1} min and M max per layer for monitoring."""
    phi_mins = []
    M_maxes = []
    with torch.no_grad():
        for layer in model.layers:
            M, Psi_diag, nu = layer.natural_to_standard()
            # psi_inv_diag is stored directly — always positive
            phi_mins.append(layer.psi_inv_diag.min().item())
            M_maxes.append(M.abs().max().item())
    return phi_mins, M_maxes


def train_epoch(model, trainer, train_loader, epoch, diagnostics):
    model.train()
    total_correct = 0
    total_samples = 0
    epoch_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.view(data.size(0), -1)
        results = trainer.train_on_batch(inputs=data, targets=target)

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

        phi_mins, M_maxes = get_phi_diagnostics(model)

        diagnostics.update_train(
            loss=results['loss'],
            acc=acc,
            layer_energies=results['layer_energies'],
            convergence=convergence,
            uncertainties=model.get_uncertainties(),
            actual_T=results['actual_T'],
            phi_mins=phi_mins,
            M_maxes=M_maxes,
        )

        # Early warning for NaN
        if torch.isnan(torch.tensor(results['loss'])):
            print(f"\n  NaN detected at batch {batch_idx}!")
            print(f"  Phi mins: {phi_mins}")
            print(f"  M maxes: {M_maxes}")
            print(f"  Energy history: {results['energy_history']}")
            break

        pbar.set_postfix({
            'loss': f"{results['loss']:.4f}",
            'acc': f"{acc:.2%}",
            'T': f"{results['avg_T']:.1f}",
            'Phi_min': f"{min(phi_mins):.2e}",
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
    layer_sizes = [784, 128, 128, 128, 10]
    activation = 'relu'
    rank_k = 20
    T = 5
    e_lr = 0.01
    kappa = 0.25
    batch_size = 128
    num_epochs = 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\n" + "="*80)
    print("Low-Rank eBPC Validation on MNIST")
    print("="*80)
    print(f"Architecture: {layer_sizes}")
    print(f"Low-rank V: diag(d) + U·U^T with rank k={rank_k}")
    print(f"Inference: ePC (T={T}, e_lr={e_lr}, Adam, adaptive T)")
    print(f"Learning: BPC low-rank Hebbian (kappa={kappa})")
    print(f"Mixed precision: {'bfloat16' if device != 'cpu' else 'disabled (CPU)'}")
    print(f"Batch size: {batch_size}")
    print("="*80 + "\n")

    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    model = LowRankeBPCNetwork(
        layer_sizes=layer_sizes, activation=activation, rank_k=rank_k,
    )
    n_params = sum(p.numel() for p in model.get_natural_parameters())
    print(f"Natural parameters: {n_params:,}")
    print(f"  (vs 850,198 full eBPC, vs ~4,000 diagonal)")

    # Per-layer parameter breakdown
    for i, layer in enumerate(model.layers):
        d_params = layer.eta1_d.numel()
        U_params = layer.eta1_U.numel()
        eta2_params = layer.eta2.numel()
        eta3_params = layer.eta3.numel()
        total = d_params + U_params + eta2_params + eta3_params + 1
        print(f"  Layer {i+1}: d={d_params}, U={U_params}, η2={eta2_params}, "
              f"η3={eta3_params}, total={total}")

    trainer = LowRankeBPCTrainer(
        model=model, T=T, e_lr=e_lr, kappa=kappa,
        adaptive_T=True, device=device,
    )
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

        # Rank info
        rank_info = model.get_rank_info()
        for i, ri in enumerate(rank_info):
            print(f"  Layer {i+1} rank: effective={ri['effective_rank']}/{rank_k}, "
                  f"sv_ratio={ri['sv_ratio']:.1f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        diagnostics.plot_diagnostics(f'diagnostics_lowrank_epoch_{epoch+1}.png')

    print(f"\n{'='*80}")
    print(f"Best test accuracy: {best_test_acc:.2%}")
    print(f"Full eBPC baseline: 95.74% (3 epochs)")
    print(f"BPC baseline: 93.5% (1 epoch)")
    print(f"Diagonal eBPC: 9.80% (NaN failure)")
    if best_test_acc >= 0.9574:
        print(f"  MATCHES OR EXCEEDS full eBPC!")
    elif best_test_acc >= 0.90:
        print(f"  Good — within {(0.9574 - best_test_acc)*100:.2f}% of full eBPC")
    elif best_test_acc >= 0.50:
        print(f"  Learning, but below full eBPC by {(0.9574 - best_test_acc)*100:.2f}%")
    else:
        print(f"  FAILED — not learning ({best_test_acc:.2%})")
    print(f"{'='*80}\n")

    diagnostics.plot_diagnostics('diagnostics_lowrank_final.png')


if __name__ == "__main__":
    main()
