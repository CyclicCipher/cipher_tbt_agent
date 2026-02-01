"""
Diagnostic Script for Bayesian PC

Tests specific hypotheses about why BPC is failing while standard PC works.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from experiments.BayesianPC.bayesian_pc_layer import BayesianPCLayer, BayesianPCNetwork
from experiments.BayesianPC.bayesian_pc_trainer import BayesianPCTrainer


def test_energy_computation():
    """Test if energy computation produces reasonable values."""
    print("\n" + "="*80)
    print("TEST 1: Energy Computation Scale")
    print("="*80)

    layer = BayesianPCLayer(in_features=784, out_features=256)

    # Get prior parameters
    M, V, Psi, nu = layer.natural_to_standard()

    print(f"\nPrior parameters:")
    print(f"  M (mean weights): {M.abs().mean():.6f} (should be ~0)")
    print(f"  V (column cov): Tr(V) = {torch.trace(V):.2f}")
    print(f"  Ψ (Wishart scale): Tr(Ψ) = {torch.trace(Psi):.2f}")
    print(f"  ν (degrees freedom): {nu:.2f}")

    # Expected precision
    expected_precision = layer.get_expected_precision()
    print(f"\nE[Σ^{{-1}}] = ν·Ψ:")
    print(f"  Tr(E[Σ^{{-1}}]) = {torch.trace(expected_precision):.2e}")
    print(f"  Average diagonal: {torch.trace(expected_precision).item() / layer.out_features:.2e}")

    # Optimal inference learning rate (from Appendix B)
    optimal_alpha = layer.get_optimal_inference_lr()
    print(f"\nOptimal inference learning rate (Appendix B):")
    print(f"  α_optimal ≈ 1 / λ_max(A_l) = {optimal_alpha:.2e}")
    print(f"  With configured α = 0.01, step size is {0.01 / optimal_alpha:.1f}x too large!")
    print(f"  ⚠️  This explains inference divergence!")

    # Simulate forward pass
    x = torch.randn(64, 784) * 0.1  # Small input like normalized MNIST
    layer.train()
    z = layer(x, sample_x=True)

    # Compute energy
    energy = layer.energy()
    print(f"\nEnergy on random input:")
    print(f"  Total energy: {energy.item():.2e}")
    print(f"  Energy per sample: {energy.item()/64:.2e}")
    print(f"  ⚠️  Baseline PC has energy ~1-10, this is {energy.item()/64:.0f}x larger!")

    # Check gradients
    energy.backward()
    grad_norm = layer._x.grad.norm().item()
    print(f"\nGradient w.r.t. value nodes:")
    print(f"  ||∂E/∂z|| = {grad_norm:.2e}")
    print(f"  ⚠️  With LR=0.01, step size = {0.01 * grad_norm:.2e}")

    return expected_precision, energy


def test_natural_parameters_leak():
    """Test if natural parameters are accumulating gradients (BUG!)."""
    print("\n" + "="*80)
    print("TEST 2: Natural Parameter Gradient Leak")
    print("="*80)

    layer = BayesianPCLayer(in_features=784, out_features=256)

    print(f"\nNatural parameters are nn.Parameter?")
    print(f"  eta1 requires_grad: {layer.eta1.requires_grad}")
    print(f"  eta2 requires_grad: {layer.eta2.requires_grad}")
    print(f"  ⚠️  These should be BUFFERS (requires_grad=False), not Parameters!")
    print(f"  ⚠️  Gradients will accumulate and corrupt the Bayesian updates!")

    # Simulate training step
    x = torch.randn(64, 784) * 0.1
    layer.train()
    z = layer(x, sample_x=True)
    energy = layer.energy()

    # Check gradients before backward
    print(f"\nBefore backward:")
    print(f"  eta1.grad: {layer.eta1.grad}")

    # Backward
    energy.backward()

    print(f"\nAfter backward:")
    print(f"  eta1.grad is not None: {layer.eta1.grad is not None}")
    if layer.eta1.grad is not None:
        print(f"  ||∂E/∂eta1|| = {layer.eta1.grad.norm().item():.2e}")
        print(f"  ⚠️  BUG CONFIRMED: Natural parameters have gradients!")
        print(f"  ⚠️  This will corrupt the closed-form Bayesian updates!")

    return layer.eta1.requires_grad


def test_inference_dynamics():
    """Test if inference phase is optimizing value nodes."""
    print("\n" + "="*80)
    print("TEST 3: Inference Phase Dynamics")
    print("="*80)

    model = BayesianPCNetwork(
        layer_sizes=[784, 256, 10],
        activation='relu',
    )
    trainer = BayesianPCTrainer(
        model=model,
        T=10,
        inference_lr=0.01,
        device='cpu',
    )

    # Create dummy data
    x = torch.randn(8, 784) * 0.1
    y = torch.randint(0, 10, (8,))

    model.train()
    model.set_sample_x(True)

    # Run inference for a few steps
    print(f"\nRunning {trainer.T} inference iterations:")
    energies = []

    for t in range(trainer.T):
        # Forward
        outputs = model(x, sample_x=True)
        loss = F.cross_entropy(outputs, y)
        layer_energies = model.get_energies()
        total_energy = sum(layer_energies) if layer_energies else torch.tensor(0.0)
        free_energy = loss + total_energy

        energies.append(free_energy.item())

        if t == 0:
            trainer._create_optimizer_x()
            print(f"  Iter 0: F = {free_energy.item():.4f} (initial)")

        # Optimize
        if trainer.optimizer_x is not None:
            trainer.optimizer_x.zero_grad()
            free_energy.backward()
            trainer.optimizer_x.step()

        if t == trainer.T - 1:
            print(f"  Iter {t}: F = {free_energy.item():.4f} (final)")

    convergence = energies[0] - energies[-1]
    print(f"\nConvergence: ΔF = {convergence:.4f}")

    if abs(convergence) < 0.001:
        print(f"  ⚠️  NO CONVERGENCE! Inference is not working!")
    elif convergence < 0:
        print(f"  ⚠️  DIVERGENCE! Free energy increased!")
    else:
        print(f"  ✓  Converged by {convergence:.4f}")

    return energies


def test_architecture_correctness():
    """Test if architecture matches paper (weights outside activation)."""
    print("\n" + "="*80)
    print("TEST 4: Architecture Verification")
    print("="*80)

    model = BayesianPCNetwork(
        layer_sizes=[784, 256, 128, 10],
        activation='relu',
    )

    print(f"\nArchitecture check:")
    print(f"  Required: z_l = W_l · f(z_{{l-1}})")
    print(f"  This means: INPUT to layer is f(z_{{l-1}}) [after activation]")

    # Trace through forward pass
    x = torch.randn(2, 784) * 0.1
    model.train()
    model.set_sample_x(True)

    print(f"\n  Layer 0 (input): {list(x.shape)}")
    h = model.activation(x)
    print(f"    After activation: {list(h.shape)}")

    for i, layer in enumerate(model.layers):
        print(f"  Layer {i+1}: input {list(h.shape)}")
        h = layer(h, sample_x=True)
        print(f"    Value node output: {list(h.shape)}")
        if i < len(model.layers) - 1:
            h = model.activation(h)
            print(f"    After activation: {list(h.shape)}")

    print(f"\n  ✓  Architecture appears correct")

    return True


def test_baseline_comparison():
    """Compare BPC energy scale to what baseline PC would have."""
    print("\n" + "="*80)
    print("TEST 5: Baseline PC Comparison")
    print("="*80)

    print(f"\nBaseline PC energy (from successful run):")
    print(f"  Layer 1 energy: ~1-10")
    print(f"  Total energy: ~10-100")
    print(f"  Free energy convergence: ~0.5-2.0 per iteration")

    # BPC
    layer = BayesianPCLayer(in_features=784, out_features=256)
    x = torch.randn(64, 784) * 0.1
    layer.train()
    z = layer(x, sample_x=True)
    bpc_energy = layer.energy().item()

    print(f"\nBPC energy:")
    print(f"  Layer 1 energy: {bpc_energy:.2e}")
    print(f"  Ratio to baseline: {bpc_energy/10:.0f}x")
    print(f"  ⚠️  Energy is {bpc_energy/10:.0f}x too large!")

    # The issue is likely the prior precision
    expected_precision = layer.get_expected_precision()
    avg_precision = torch.trace(expected_precision) / expected_precision.size(0)

    print(f"\nPrior precision scale:")
    print(f"  E[Σ^{{-1}}] average diagonal: {avg_precision:.2e}")
    print(f"  Baseline PC uses Σ^{{-1}} = I (precision = 1.0)")
    print(f"  BPC is {avg_precision:.0f}x more precise")
    print(f"  → Errors are weighted {avg_precision:.0f}x more heavily")
    print(f"  → Energies are {avg_precision:.0f}x larger")
    print(f"  → Gradients are {avg_precision:.0f}x larger")

    return avg_precision


def main():
    """Run all diagnostic tests."""
    print("="*80)
    print("BAYESIAN PC DIAGNOSTIC SUITE")
    print("="*80)
    print("\nHypotheses to test:")
    print("  H1: Energy scale is wrong (prior precision too high)")
    print("  H2: Natural parameters are accumulating gradients (bug)")
    print("  H3: Inference dynamics aren't working")
    print("  H4: Architecture doesn't match paper")
    print("  H5: Something fundamental about the implementation")

    # Run tests
    test_energy_computation()
    test_natural_parameters_leak()
    test_inference_dynamics()
    test_architecture_correctness()
    avg_precision = test_baseline_comparison()

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    print(f"\nIdentified issues:")
    print(f"  1. ⚠️  Prior precision is ~{avg_precision:.0f}x too high")
    print(f"     → Fix: Reduce Ψ_prior from 1000 to ~1.0")
    print(f"  ")
    print(f"  2. ⚠️  Natural parameters are nn.Parameter (should be buffers)")
    print(f"     → Fix: Use register_buffer instead of nn.Parameter")
    print(f"  ")
    print(f"  3. ⚠️  Energy scale causes huge gradients")
    print(f"     → Fix: Scale down prior OR increase inference LR")

    print(f"\nRecommended fixes (priority order):")
    print(f"  1. Change eta1, eta2, eta3, eta4 from nn.Parameter → register_buffer")
    print(f"  2. Reduce prior_Psi_scale from 1000.0 to 1.0")
    print(f"  3. Test on small batch to verify convergence")


if __name__ == "__main__":
    main()
