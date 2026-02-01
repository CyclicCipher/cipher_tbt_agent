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
        activation='relu',
        prior_Psi_scale=psi_scale,
    ).cuda()

    # Get precision from first layer
    layer = model.layers[0]
    _, _, Psi, nu = layer.natural_to_standard()
    expected_precision = nu * Psi
    avg_precision = torch.diag(expected_precision).mean().item()

    # Estimate energy scale by computing energy on random batch
    model.train()
    model.set_sample_x(True)

    batch_size = 128
    inputs = torch.randn(batch_size, 784, device='cuda') * 0.1

    with torch.no_grad():
        # Forward pass to initialize value nodes
        _ = model(inputs)

        # Compute total energy
        total_energy = 0.0
        h = model.activation(inputs)
        h = model._augment_with_bias(h)

        for layer in model.layers:
            if layer._x is not None:
                energy = layer.energy(h).item()
                total_energy += energy
                # Update h for next layer
                h = model.activation(layer._x)
                if layer != model.layers[-1]:
                    h = model._augment_with_bias(h)

    energy_per_sample = total_energy / batch_size

    if verbose:
        print(f"\nΨ_scale = {psi_scale}")
        print(f"  ν (degrees of freedom): {nu:.1f}")
        print(f"  E[Σ^{{-1}}] diagonal: {avg_precision:.2e} (want ~1.0)")
        print(f"  Energy/sample: {energy_per_sample:.2e} (want ~1-10)")

    return {
        'psi_scale': psi_scale,
        'nu': nu,
        'avg_precision': avg_precision,
        'energy_per_sample': energy_per_sample,
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
            import traceback
            traceback.print_exc()

    if len(results) == 0:
        print("\n❌ All tests failed!")
        return

    # Find best value (closest to standard PC)
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nTarget metrics (from standard PC):")
    print("  - E[Σ^{-1}] ≈ 1.0")
    print("  - Energy/sample ≈ 1-10")

    print("\n{:<12} {:<8} {:<15} {:<15}".format(
        "Ψ_scale", "ν", "E[Σ^{-1}]", "Energy/sample"))
    print("-"*80)

    for r in results:
        print("{:<12.3e} {:<8.1f} {:<15.2e} {:<15.2e}".format(
            r['psi_scale'],
            r['nu'],
            r['avg_precision'],
            r['energy_per_sample']
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

    print("\n" + "="*80)
    print("THREE MOST LIKELY EXPLANATIONS:")
    print("="*80)
    print("\n1. WISHART vs INVERSE WISHART CONFUSION:")
    print("   - Paper uses Matrix Normal Wishart (MNW) for q(W, Σ)")
    print("   - In MNW, Wishart is on PRECISION Σ^{-1}, not covariance Σ")
    print("   - If authors mentally flipped: high Ψ = 'weak prior' (large variance)")
    print("   - But mathematically: high Ψ → high E[Σ^{-1}] → low variance → STIFF!")
    print(f"   - Evidence: Ψ = {best['psi_scale']} works, which is {1000.0/best['psi_scale']:.0f}x smaller")

    print("\n2. SUM vs MEAN REDUCTION:")
    print("   - Paper equations use Σ_n (sum over batch)")
    print("   - PyTorch defaults to mean reduction")
    print("   - Gradients differ by factor of batch_size = 128")
    print("   - LR=0.01 with mean ≈ LR=0.00007 with sum")
    print("   - If true: effective LR mismatch, not precision mismatch")

    print("\n3. TYPO IN APPENDIX F.1:")
    print("   - Appendices written last, often by different person than code")
    print("   - Copy-paste from config file without units/context")
    print("   - Possible: Ψ = 1000 → Ψ = 1.0 (or 0.001)")
    print("   - Standard PC uses precision = 1.0, not 130,000")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
