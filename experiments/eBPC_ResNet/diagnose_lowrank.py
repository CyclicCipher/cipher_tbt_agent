"""
Diagnostics for low-rank eBPC: verify quadratic constraint preservation.

Tests:
  1. Spectral norm inflation verification
     - Does η1_approx ≥ η1_true hold after the fix?
     - How large is the inflation (max_dropped_eigenvalue)?
  2. Per-batch Phi monitoring
     - Is Phi always positive now?
     - How does Phi evolve over training?
  3. Side-by-side full vs low-rank (15 batches)
     - |M|, Phi, precision comparison
  4. Extended training (50 batches)
     - Does the fix survive longer training?
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.eBPC_ResNet.ebpc_lowrank_layer import LowRankeBPCNetwork
from experiments.eBPC_ResNet.ebpc_lowrank_trainer import LowRankeBPCTrainer
from experiments.eBPC.ebpc_layer import eBPCNetwork
from experiments.eBPC.ebpc_trainer import eBPCTrainer


def get_mnist_loader(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def test_spectral_inflation():
    """Verify that spectral norm inflation makes η1_approx ≥ η1_true."""
    print("\n" + "="*80)
    print("TEST 1: Spectral norm inflation — does η1_approx ≥ η1_true hold?")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = get_mnist_loader()
    data, target = next(iter(loader))
    data = data.view(data.size(0), -1).to(device)
    target = target.to(device)

    # Full eBPC — get exact η1
    full_model = eBPCNetwork([784, 128, 128, 128, 10]).to(device)
    full_trainer = eBPCTrainer(model=full_model, T=5, e_lr=0.01, kappa=0.25, device=device)
    full_trainer.train_on_batch(data, target)

    # Low-rank eBPC — get approximated η1
    lr_model = LowRankeBPCNetwork([784, 128, 128, 128, 10], rank_k=20).to(device)
    lr_trainer = LowRankeBPCTrainer(model=lr_model, T=5, e_lr=0.01, kappa=0.25,
                                     adaptive_T=True, use_amp=True, device=device)
    lr_trainer.train_on_batch(data, target)

    for layer_idx in range(len(full_model.layers)):
        fl = full_model.layers[layer_idx]
        ll = lr_model.layers[layer_idx]

        eta1_true = fl.eta1  # [in, in]
        eta1_approx = torch.diag(ll.eta1_d) + ll.eta1_U @ ll.eta1_U.T  # [in, in]

        # Check η1_approx - η1_true eigenvalues (should all be ≥ 0)
        diff = eta1_approx - eta1_true
        try:
            eig_diff = torch.linalg.eigvalsh(diff)
            min_eig = eig_diff.min().item()
            max_eig = eig_diff.max().item()
        except Exception as e:
            eig_diff = torch.linalg.eigvalsh(diff.float().cpu())
            min_eig = eig_diff.min().item()
            max_eig = eig_diff.max().item()

        psd_ok = min_eig >= -1e-6  # allow tiny numerical noise

        print(f"\n  Layer {layer_idx + 1} (in={fl.in_features}):")
        print(f"    η1_approx - η1_true eigenvalues: min={min_eig:.6e}, max={max_eig:.6e}")
        print(f"    η1_approx ≥ η1_true (PSD): {'YES ✓' if psd_ok else 'NO ✗ — CONSTRAINT VIOLATED'}")
        print(f"    η1_d range: [{ll.eta1_d.min().item():.4e}, {ll.eta1_d.max().item():.4e}]")
        print(f"    |η1_U| max: {ll.eta1_U.abs().max().item():.4e}")


def test_phi_monitoring():
    """Monitor Phi per batch — should always be positive with the fix."""
    print("\n" + "="*80)
    print("TEST 2: Phi monitoring per batch (should be positive)")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LowRankeBPCNetwork([784, 128, 128, 128, 10], rank_k=20).to(device)
    trainer = LowRankeBPCTrainer(model=model, T=5, e_lr=0.01, kappa=0.25,
                                  adaptive_T=True, use_amp=True, device=device)

    loader = get_mnist_loader()

    print(f"\n  {'Bat':>3} | {'Ly':>2} | {'Phi min':>14} | {'Phi mean':>14} | "
          f"{'|M| max':>12} | {'prec mean':>12} | {'loss':>8}")
    print("  " + "-"*95)

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= 20:
            break
        data = data.view(data.size(0), -1)
        results = trainer.train_on_batch(data, target)

        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                M = layer._M_cache
                phi = layer._compute_schur_diag(M)
                _, Psi_diag, nu = layer.natural_to_standard()
                prec = nu * Psi_diag

                loss_str = f"{results['loss']:.4f}" if i == 0 else ""
                print(f"  {batch_idx:>3} | {i+1:>2} | {phi.min().item():>14.4e} | "
                      f"{phi.mean().item():>14.4e} | {M.abs().max().item():>12.4e} | "
                      f"{prec.mean().item():>12.4e} | {loss_str:>8}")

        if torch.isnan(torch.tensor(results['loss'])):
            print(f"\n  NaN at batch {batch_idx}!")
            break


def test_side_by_side():
    """Run full eBPC and low-rank eBPC on same data, compare per-batch evolution."""
    print("\n" + "="*80)
    print("TEST 3: Side-by-side full vs low-rank (15 batches)")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    full_model = eBPCNetwork([784, 128, 128, 128, 10]).to(device)
    full_trainer = eBPCTrainer(model=full_model, T=5, e_lr=0.01, kappa=0.25, device=device)

    lr_model = LowRankeBPCNetwork([784, 128, 128, 128, 10], rank_k=20).to(device)
    lr_trainer = LowRankeBPCTrainer(model=lr_model, T=5, e_lr=0.01, kappa=0.25,
                                     adaptive_T=True, use_amp=True, device=device)

    loader = get_mnist_loader()

    print(f"\n  {'Bat':>3} | {'Ly':>2} | {'|M_full|':>10} | {'|M_lr|':>10} | "
          f"{'Phi_full':>12} | {'Phi_lr':>12} | {'loss_full':>10} | {'loss_lr':>10}")
    print("  " + "-"*100)

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= 15:
            break
        data = data.view(data.size(0), -1)
        full_results = full_trainer.train_on_batch(data, target)
        lr_results = lr_trainer.train_on_batch(data, target)

        with torch.no_grad():
            for i in range(len(full_model.layers)):
                fl = full_model.layers[i]
                ll = lr_model.layers[i]

                M_full, V_full, Psi_full, nu_full = fl.natural_to_standard()
                Phi_full_diag = (fl.eta3 - fl.eta2 @ V_full @ fl.eta2.T).diag()

                M_lr = ll._M_cache
                phi_lr = ll._compute_schur_diag(M_lr)

                loss_f = f"{full_results['loss']:.4f}" if i == 0 else ""
                loss_l = f"{lr_results['loss']:.4f}" if i == 0 else ""

                print(f"  {batch_idx:>3} | {i+1:>2} | "
                      f"{M_full.abs().max().item():>10.2e} | "
                      f"{M_lr.abs().max().item():>10.2e} | "
                      f"{Phi_full_diag.min().item():>12.4e} | "
                      f"{phi_lr.min().item():>12.4e} | "
                      f"{loss_f:>10} | {loss_l:>10}")

        if torch.isnan(torch.tensor(lr_results['loss'])):
            print(f"\n  Low-rank NaN at batch {batch_idx}!")
            break


def test_extended_training():
    """Extended training: 50 batches to verify stability."""
    print("\n" + "="*80)
    print("TEST 4: Extended training (50 batches)")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LowRankeBPCNetwork([784, 128, 128, 128, 10], rank_k=20).to(device)
    trainer = LowRankeBPCTrainer(model=model, T=5, e_lr=0.01, kappa=0.25,
                                  adaptive_T=True, use_amp=True, device=device)

    loader = get_mnist_loader()

    print(f"\n  {'Bat':>3} | {'loss':>8} | {'L1 |M|':>10} | {'L1 Phi_min':>12} | "
          f"{'L1 prec':>12} | {'L4 |M|':>10} | {'L4 Phi_min':>12}")
    print("  " + "-"*85)

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= 50:
            break
        data = data.view(data.size(0), -1)
        results = trainer.train_on_batch(data, target)

        if batch_idx % 5 == 0 or torch.isnan(torch.tensor(results['loss'])):
            with torch.no_grad():
                l1 = model.layers[0]
                l4 = model.layers[-1]
                M1 = l1._M_cache
                M4 = l4._M_cache
                phi1 = l1._compute_schur_diag(M1)
                phi4 = l4._compute_schur_diag(M4)
                _, Psi1, nu1 = l1.natural_to_standard()
                prec1 = nu1 * Psi1

                print(f"  {batch_idx:>3} | {results['loss']:>8.4f} | "
                      f"{M1.abs().max().item():>10.4e} | {phi1.min().item():>12.4e} | "
                      f"{prec1.mean().item():>12.4e} | {M4.abs().max().item():>10.4e} | "
                      f"{phi4.min().item():>12.4e}")

        if torch.isnan(torch.tensor(results['loss'])):
            print(f"\n  NaN at batch {batch_idx}!")
            break
    else:
        print(f"\n  Survived all 50 batches! Final loss: {results['loss']:.4f}")


def main():
    print("="*80)
    print("Low-Rank eBPC Diagnostics v3 — Quadratic Constraint Verification")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test 1: Does the spectral inflation make η1_approx ≥ η1_true?
    test_spectral_inflation()

    # Test 2: Is Phi always positive now?
    test_phi_monitoring()

    # Test 3: Side-by-side comparison
    test_side_by_side()

    # Test 4: Extended training
    test_extended_training()

    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
