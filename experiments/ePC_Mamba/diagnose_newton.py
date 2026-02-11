"""
Diagnostic script for ePC-Mamba Newton step and E_local gradient issues.

Three tests:
  A. Backprop baseline: Does the same architecture solve copy task easily
     without ePC? If yes, the instability is ePC-specific.

  B. Per-layer gradient norms: Measure gradient magnitude from E_local
     for each layer. Confirms whether Layer 1 is starved.

  C. Newton convergence curve: Run T=100 iterations on a single batch.
     Tracks energy vs iteration to see if Newton CAN converge given
     enough steps, or if it's fundamentally stuck.

  D. Gradient descent comparison: Try plain GD with various step sizes
     for error optimization on the same batch. Shows how much energy
     COULD be reduced if Newton stepped correctly.

Usage:
  python experiments/ePC_Mamba/diagnose_newton.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.ePC_Mamba.mamba_block import Mamba2Config, RMSNorm
from experiments.ePC_Mamba.epc_model import PCESequence, MambaLayer
from experiments.ePC_Mamba.train_synthetic import (
    generate_copy_data, ePCMambaSynthetic, BackpropMambaBaseline,
    compute_accuracy, evaluate_with_loss,
)
from torch.utils.data import DataLoader, TensorDataset


def test_A_baseline(device):
    """Test A: Can backprop baseline solve copy task easily?"""
    print("=" * 70)
    print("TEST A: Backprop baseline on copy task")
    print("=" * 70)

    config = Mamba2Config(d_model=128, d_state=64, headdim=64,
                          expand=2, chunk_size=64, n_layer=2)
    model = BackpropMambaBaseline(config, vocab_size=16, task='copy').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_x, train_y = generate_copy_data(5000, 64, 16)
    test_x, test_y = generate_copy_data(1000, 64, 16)
    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y),
                             batch_size=32, shuffle=False, drop_last=True)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  {'Epoch':>5} {'Train Acc':>10} {'Test Acc':>10}")

    for epoch in range(1, 21):
        model.train()
        epoch_acc = 0
        n = 0
        for batch in train_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            b, l, v = logits.shape
            loss = F.cross_entropy(logits.reshape(b*l, v), targets.reshape(b*l))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                acc = compute_accuracy(logits, targets, task='copy')
            epoch_acc += acc
            n += 1

        test_acc, _ = evaluate_with_loss(model, test_loader, device, 'copy', False)
        print(f"  {epoch:5d} {epoch_acc/n:10.4f} {test_acc:10.4f}")

        if test_acc >= 0.99:
            print(f"  SOLVED at epoch {epoch}!")
            return test_acc

    print(f"  Final test acc: {test_acc:.4f}")
    return test_acc


def test_B_gradient_norms(device):
    """Test B: Per-layer gradient norms from E_local."""
    print("\n" + "=" * 70)
    print("TEST B: Per-layer gradient norms from E_local")
    print("=" * 70)

    config = Mamba2Config(d_model=128, d_state=64, headdim=64,
                          expand=2, chunk_size=64, n_layer=2)
    model = ePCMambaSynthetic(config, vocab_size=16, task='copy',
                              iters=2, damping=0.1).to(device)

    train_x, train_y = generate_copy_data(320, 64, 16)

    # Train for a few batches first so the model isn't at random init
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y),
                        batch_size=32, shuffle=True, drop_last=True)

    # Collect gradient norms per component over 10 batches
    print("\n  Gradient norms from E_local (per component):")
    print(f"  {'Batch':>5} {'Embedding':>12} {'Layer1':>12} {'Layer2':>12} "
          f"{'OutNorm':>12} {'OutProj':>12} {'||e||':>10}")

    for i, batch in enumerate(loader):
        if i >= 10:
            break
        inputs, targets = batch[0].to(device), batch[1].to(device)

        # Phase 1: inference
        model(inputs, targets)

        # Save error norm
        e_norm = 0.0
        for e in model.pce.errors:
            if isinstance(e, torch.Tensor):
                e_norm += e.detach().norm().item()

        # Phase 2: weight update
        optimizer.zero_grad()
        weight_loss = model.compute_weight_loss(inputs, targets, 32)
        weight_loss.backward()

        # Collect gradient norms per component
        def grad_norm(params):
            total = 0.0
            for p in params:
                if p.grad is not None:
                    total += p.grad.norm().item() ** 2
            return total ** 0.5

        emb_norm = grad_norm(model.embedding.parameters())
        l1_norm = grad_norm(model.pce.layers[0].parameters())
        l2_norm = grad_norm(model.pce.layers[1].parameters())
        on_norm = grad_norm(model.pce.out_norm.parameters())
        op_norm = grad_norm(model.out_proj.parameters())

        print(f"  {i:5d} {emb_norm:12.6f} {l1_norm:12.6f} {l2_norm:12.6f} "
              f"{on_norm:12.6f} {op_norm:12.6f} {e_norm:10.4f}")

        optimizer.step()

    return


def test_C_newton_convergence(device):
    """Test C: Run many Newton iterations on one batch."""
    print("\n" + "=" * 70)
    print("TEST C: Newton convergence curve (T=1..100 on single batch)")
    print("=" * 70)

    config = Mamba2Config(d_model=128, d_state=64, headdim=64,
                          expand=2, chunk_size=64, n_layer=2)

    # Build a PCESequence with high T for this test
    pce = PCESequence(config, iters=100, damping=0.1, output_loss='ce')
    embedding = nn.Embedding(16, config.d_model)
    out_proj = nn.Linear(config.d_model, 16, bias=False)
    pce.to(device)
    embedding.to(device)
    out_proj.to(device)

    # Generate one batch
    train_x, train_y = generate_copy_data(32, 64, 16)
    inputs = train_x.to(device)
    targets = train_y.to(device)
    x = embedding(inputs).detach()

    # Manually run Newton iterations tracking energy at each step
    for p in pce.layers.parameters():
        p.requires_grad_(False)
    for p in pce.out_norm.parameters():
        p.requires_grad_(False)
    out_proj.requires_grad_(False)

    pce.init_zero_errors(x)

    energies = []
    error_norms = []
    step_sizes = []
    coefficients = []

    print(f"\n  {'Iter':>4} {'Energy':>12} {'||e||':>12} {'E_drop':>12} "
          f"{'coeff':>10} {'||step||':>12}")

    for t in range(100):
        # Zero error gradients
        for e in pce.errors:
            if e.grad is not None:
                e.grad.zero_()

        # Forward + energy
        E = pce.E(x, targets, out_proj)
        E_val = E.item()
        energies.append(E_val)

        # Error norm
        e_norm = sum(e.data.norm().item() for e in pce.errors)
        error_norms.append(e_norm)

        # Save pre-step errors
        pre_errors = [e.data.clone() for e in pce.errors]

        # Backward
        E.backward()

        # Newton step
        pce._newton_step()

        # Step size
        step_norm = sum((e.data - pre_e).norm().item()
                        for e, pre_e in zip(pce.errors, pre_errors))
        step_sizes.append(step_norm)

        coeff = pce._newton_diag.get('coeff', 0)
        coefficients.append(coeff)

        E_drop = energies[-2] - E_val if t > 0 else 0.0

        if t < 20 or t % 10 == 0:
            print(f"  {t:4d} {E_val:12.2f} {e_norm:12.6f} {E_drop:12.4f} "
                  f"{coeff:10.6f} {step_norm:12.6f}")

    # Unfreeze
    for p in pce.layers.parameters():
        p.requires_grad_(True)
    for p in pce.out_norm.parameters():
        p.requires_grad_(True)
    out_proj.requires_grad_(True)

    total_drop = energies[0] - energies[-1]
    print(f"\n  Total energy drop over 100 iters: {total_drop:.4f} "
          f"({total_drop/energies[0]*100:.2f}%)")
    print(f"  E_0 = {energies[0]:.2f}, E_100 = {energies[-1]:.2f}")
    print(f"  Final ||e|| = {error_norms[-1]:.6f}")

    return energies, error_norms


def test_D_gradient_descent_comparison(device):
    """Test D: Compare Newton vs GD at various step sizes."""
    print("\n" + "=" * 70)
    print("TEST D: Gradient descent comparison (single batch, T=2)")
    print("=" * 70)

    config = Mamba2Config(d_model=128, d_state=64, headdim=64,
                          expand=2, chunk_size=64, n_layer=2)

    embedding = nn.Embedding(16, config.d_model).to(device)
    out_proj = nn.Linear(config.d_model, 16, bias=False).to(device)

    train_x, train_y = generate_copy_data(32, 64, 16)
    inputs = train_x.to(device)
    targets = train_y.to(device)
    x = embedding(inputs).detach()

    step_sizes = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0]
    results = []

    # First get Newton baseline
    pce_newton = PCESequence(config, iters=2, damping=0.1, output_loss='ce').to(device)
    # Copy weights to all test models
    state = pce_newton.state_dict()

    for p in pce_newton.layers.parameters():
        p.requires_grad_(False)
    for p in pce_newton.out_norm.parameters():
        p.requires_grad_(False)
    out_proj.requires_grad_(False)

    pce_newton.init_zero_errors(x)

    for t in range(2):
        for e in pce_newton.errors:
            if e.grad is not None:
                e.grad.zero_()
        E = pce_newton.E(x, targets, out_proj)
        if t == 0:
            E_init = E.item()
        E.backward()
        pce_newton._newton_step()

    E_newton = pce_newton.E(x, targets, out_proj).item()
    e_norm_newton = sum(e.data.norm().item() for e in pce_newton.errors)

    print(f"\n  Initial energy: {E_init:.2f}")
    print(f"\n  {'Method':<20} {'Final E':>12} {'E_drop':>12} {'||e||':>12}")
    print(f"  {'Newton (T=2)':<20} {E_newton:12.2f} {E_init-E_newton:12.4f} "
          f"{e_norm_newton:12.6f}")

    # Now test GD at various step sizes
    for lr in step_sizes:
        pce_gd = PCESequence(config, iters=2, damping=0.1, output_loss='ce').to(device)
        pce_gd.load_state_dict(state)

        for p in pce_gd.layers.parameters():
            p.requires_grad_(False)
        for p in pce_gd.out_norm.parameters():
            p.requires_grad_(False)

        pce_gd.init_zero_errors(x)

        for t in range(2):
            for e in pce_gd.errors:
                if e.grad is not None:
                    e.grad.zero_()
            E = pce_gd.E(x, targets, out_proj)
            E.backward()

            # Plain gradient descent instead of Newton
            with torch.no_grad():
                for e in pce_gd.errors:
                    e.data.sub_(e.grad, alpha=lr)

        with torch.no_grad():
            E_gd = pce_gd.E(x, targets, out_proj).item()
        e_norm_gd = sum(e.data.norm().item() for e in pce_gd.errors)

        label = f"GD (lr={lr})"
        improved = "***" if E_gd < E_newton else ""
        print(f"  {label:<20} {E_gd:12.2f} {E_init-E_gd:12.4f} "
              f"{e_norm_gd:12.6f} {improved}")
        results.append((lr, E_gd, e_norm_gd))

    # Also try Adam with more iterations
    print(f"\n  {'Method':<20} {'Final E':>12} {'E_drop':>12} {'||e||':>12}")
    for T_adam in [5, 10, 20]:
        pce_adam = PCESequence(config, iters=T_adam, damping=0.1,
                               output_loss='ce').to(device)
        pce_adam.load_state_dict(state)

        for p in pce_adam.layers.parameters():
            p.requires_grad_(False)
        for p in pce_adam.out_norm.parameters():
            p.requires_grad_(False)

        pce_adam.init_zero_errors(x)
        adam_opt = torch.optim.Adam(pce_adam.errors, lr=0.01)

        for t in range(T_adam):
            adam_opt.zero_grad()
            E = pce_adam.E(x, targets, out_proj)
            E.backward()
            adam_opt.step()

        with torch.no_grad():
            E_adam = pce_adam.E(x, targets, out_proj).item()
        e_norm_adam = sum(e.data.norm().item() for e in pce_adam.errors)

        label = f"Adam T={T_adam} (lr=0.01)"
        print(f"  {label:<20} {E_adam:12.2f} {E_init-E_adam:12.4f} "
              f"{e_norm_adam:12.6f}")

    # Unfreeze
    for p in pce_newton.layers.parameters():
        p.requires_grad_(True)
    for p in pce_newton.out_norm.parameters():
        p.requires_grad_(True)
    out_proj.requires_grad_(True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    test_A_baseline(device)
    test_B_gradient_norms(device)
    test_C_newton_convergence(device)
    test_D_gradient_descent_comparison(device)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  Test A: If baseline reaches >95% in <20 epochs, the architecture
          can solve copy task — ePC is the bottleneck.

  Test B: If Layer 1 gradient norm << Layer 2, E_local is starving
          early layers. This explains instability (Layer 2 over-adapts,
          Layer 1 can't keep up).

  Test C: If 100 Newton iterations barely reduce energy, the rank-1
          approximation is fundamentally too conservative. If energy
          drops significantly, more iterations (T>2) would help.

  Test D: If GD at some step size beats Newton, the step size is
          wrong, not the direction. If Adam beats Newton dramatically,
          per-element curvature adaptation is needed.
""")


if __name__ == '__main__':
    main()
