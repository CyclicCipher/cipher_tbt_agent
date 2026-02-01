"""Test different Psi_scale values to validate Gemini's hyperparameter hypothesis.

This script tests whether the paper's Psi_scale = 1000 is the root cause of:
- Extremely high energy (~10^7 instead of ~1-10)
- Diverging free energy (ΔF < 0, increasing not decreasing)
- Vanishing gradients (||∂E/∂z|| = 0)
- Random accuracy (10%, not learning)

Hypothesis: The paper confused Wishart W(Ψ,ν) on Σ^{-1} with Inverse Wishart IW(Ψ,ν) on Σ.
If true, Ψ=1000 creates E[Σ^{-1}] = ν·Ψ = 130,000 (catastrophically stiff).
Should be E[Σ^{-1}] ≈ 1.0 (standard PC), requiring Ψ ≈ 1/ν ≈ 0.01.
"""

import torch
import torch.nn as nn
import numpy as np
from bayesian_pc_layer import BayesianPCNetwork
from bayesian_pc_trainer import BayesianPCTrainer

def test_psi_value(psi_scale, verbose=True):
    """Test a single Psi_scale value."""
    # Create simple model
    model = BayesianPCNetwork(
        layer_sizes=[784, 128, 10],
        activation=nn.ReLU(),
        prior_Psi_scale=psi_scale,
    ).cuda()

    # Create trainer
    trainer = BayesianPCTrainer(
        model=model,
        T=10,
        inference_lr=0.01,
        kappa=0.25,
        device='cuda'
    )

    # Test batch
    batch_size = 128
    inputs = torch.randn(batch_size, 784, device='cuda') * 0.1
    targets = torch.randint(0, 10, (batch_size,), device='cuda')

    # Get initial state
    layer = model.layers[0]
    _, _, Psi, nu = layer.natural_to_standard()
    expected_precision = nu * Psi
    avg_precision = torch.diag(expected_precision).mean().item()

    # Run one E-step to test convergence
    z_star = trainer._e_step(inputs, targets)

    # Compute initial and final free energy
    model.eval()
    with torch.no_grad():
        # Initial energy (before inference)
        energy_init = 0.0
        h = model.activation(inputs)
        h = model._augment_with_bias(h)
        for layer in model.layers[:-1]:
            # Initialize value nodes at prediction
            W_mean, _ = layer.get_expected_W_and_Sigma()
            z_init = nn.functional.linear(h, W_mean, bias=None)
            layer._x = nn.Parameter(z_init, requires_grad=True)
            energy_init += layer.energy(h).item()
            h = model.activation(layer._x)
            h = model._augment_with_bias(h)

        # Final energy (after inference)
        energy_final = sum(layer.energy(pre).item()
                          for layer, pre in zip(model.layers,
                                               trainer._get_pre_activations(inputs, z_star)))

    convergence = energy_final - energy_init

    # Compute gradient magnitude
    total_grad = 0.0
    for z in z_star:
        if z.grad is not None:
            total_grad += z.grad.norm().item()

    if verbose:
        print(f"\nΨ_scale = {psi_scale}")
        print(f"  E[Σ^{{-1}}] diagonal: {avg_precision:.2e} (want ~1.0)")
        print(f"  Energy/sample: {energy_final/batch_size:.2e} (want ~1-10)")
        print(f"  ΔF (convergence): {convergence:.2e} (want > 0, decreasing)")
        print(f"  ||∂E/∂z||: {total_grad:.2e} (want > 0)")

    return {
        'psi_scale': psi_scale,
        'avg_precision': avg_precision,
        'energy_per_sample': energy_final / batch_size,
        'convergence': convergence,
        'gradient_norm': total_grad
    }

def main():
    print("="*80)
    print("TESTING GEMINI'S HYPERPARAMETER HYPOTHESIS")
    print("="*80)
    print("\nFor Wishart W(Ψ,ν) on Σ^{-1}: E[Σ^{-1}] = ν·Ψ")
    print(f"With ν ≈ 130 (for 128-unit hidden layer):")
    print(f"  - Ψ = 1000 → E[Σ^{{-1}}] = 130,000 (CATASTROPHIC)")
    print(f"  - Ψ = 1.0   → E[Σ^{{-1}}] = 130 (still high)")
    print(f"  - Ψ = 0.01  → E[Σ^{{-1}}] = 1.3 (reasonable)")
    print(f"  - Ψ = 0.001 → E[Σ^{{-1}}] = 0.13 (low)")

    # Test different values
    psi_values = [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
    results = []

    for psi in psi_values:
        try:
            result = test_psi_value(psi)
            results.append(result)
        except Exception as e:
            print(f"\nΨ_scale = {psi}: FAILED ({e})")

    # Find best value (closest to standard PC)
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nTarget metrics (from standard PC):")
    print("  - E[Σ^{-1}] ≈ 1.0")
    print("  - Energy/sample ≈ 1-10")
    print("  - ΔF > 0 (decreasing energy)")
    print("  - ||∂E/∂z|| > 0 (non-zero gradients)")

    print("\n{:<12} {:<15} {:<15} {:<15} {:<15}".format(
        "Ψ_scale", "E[Σ^{-1}]", "Energy/sample", "ΔF", "||∂E/∂z||"))
    print("-"*80)

    for r in results:
        print("{:<12.3e} {:<15.2e} {:<15.2e} {:<15.2e} {:<15.2e}".format(
            r['psi_scale'],
            r['avg_precision'],
            r['energy_per_sample'],
            r['convergence'],
            r['gradient_norm']
        ))

    # Find closest to target precision of 1.0
    best_idx = min(range(len(results)),
                   key=lambda i: abs(results[i]['avg_precision'] - 1.0))
    best = results[best_idx]

    print("\n" + "="*80)
    print(f"RECOMMENDATION: Use Ψ_scale = {best['psi_scale']}")
    print(f"  - Gives E[Σ^{{-1}}] ≈ {best['avg_precision']:.2f} (closest to 1.0)")
    print(f"  - Energy/sample: {best['energy_per_sample']:.2e}")
    print(f"  - Paper's value (1000.0) is {1000.0/best['psi_scale']:.0f}x too large!")
    print("="*80)

if __name__ == '__main__':
    main()
