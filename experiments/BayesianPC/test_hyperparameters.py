"""Test different Psi values using Inverse Wishart convention.

The paper specifies Ψ=1000 in Appendix F.1. VERSES typically uses Inverse Wishart
semantics where large Ψ = vague prior on covariance Σ.

Internally, the MNW formulation uses Wishart on precision Σ^{-1}. The layer
converts: Ψ_wishart = 1/(ν * Ψ_iw), so E[Σ^{-1}] = ν * Ψ_w = 1/Ψ_iw.

This test sweeps Ψ_iw values to verify the conversion produces reasonable
precision and energy scales.
"""

import torch
import torch.nn as nn
import numpy as np
from bayesian_pc_layer import BayesianPCNetwork


def test_psi_value(psi_iw_scale, verbose=True):
    """Test a single Psi_iw_scale value (Inverse Wishart convention)."""
    model = BayesianPCNetwork(
        layer_sizes=[784, 128, 10],
        activation='relu',
        prior_Psi_iw_scale=psi_iw_scale,
    ).cuda()

    # Get precision from first layer
    layer = model.layers[0]
    _, _, Psi, nu = layer.natural_to_standard()
    expected_precision = nu * Psi
    avg_precision = torch.diag(expected_precision).mean().item()

    # Compute energy on random batch
    model.train()
    model.set_sample_x(True)

    batch_size = 128
    inputs = torch.randn(batch_size, 784, device='cuda') * 0.1

    with torch.no_grad():
        _ = model(inputs)

        total_energy = 0.0
        for layer in model.layers:
            e = layer.energy()
            if e is not None:
                total_energy += e.item()

    energy_per_sample = total_energy / batch_size

    if verbose:
        print(f"\nΨ_iw = {psi_iw_scale}")
        print(f"  ν (degrees of freedom): {nu:.1f}")
        print(f"  E[Σ^{{-1}}] diagonal: {avg_precision:.2e} (want ~1.0)")
        print(f"  Energy/sample: {energy_per_sample:.2e}")

    return {
        'psi_iw_scale': psi_iw_scale,
        'nu': nu,
        'avg_precision': avg_precision,
        'energy_per_sample': energy_per_sample,
    }


def main():
    print("="*80)
    print("TESTING Ψ VALUES (INVERSE WISHART CONVENTION)")
    print("="*80)
    print("\nPaper uses Ψ_iw=1000 (Inverse Wishart: large = vague prior on Σ)")
    print("Internally converted to Wishart on Σ^{-1}: Ψ_w = 1/(ν * Ψ_iw)")
    print("So E[Σ^{-1}] = ν * Ψ_w = 1/Ψ_iw")
    print(f"\nWith ν ≈ 130 (for 128-unit hidden layer):")
    print(f"  - Ψ_iw = 1000 → E[Σ^{{-1}}] ≈ 0.001 (very vague)")
    print(f"  - Ψ_iw = 10   → E[Σ^{{-1}}] ≈ 0.1")
    print(f"  - Ψ_iw = 1    → E[Σ^{{-1}}] ≈ 1.0 (standard PC)")
    print(f"  - Ψ_iw = 0.1  → E[Σ^{{-1}}] ≈ 10 (informative)")

    psi_values = [10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01]
    results = []

    for psi in psi_values:
        try:
            result = test_psi_value(psi)
            results.append(result)
        except Exception as e:
            print(f"\nΨ_iw = {psi}: FAILED ({e})")
            import traceback
            traceback.print_exc()

    if len(results) == 0:
        print("\nAll tests failed!")
        return

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\n{:<12} {:<8} {:<15} {:<15}".format(
        "Ψ_iw", "ν", "E[Σ^{-1}]", "Energy/sample"))
    print("-"*80)

    for r in results:
        print("{:<12.3e} {:<8.1f} {:<15.2e} {:<15.2e}".format(
            r['psi_iw_scale'],
            r['nu'],
            r['avg_precision'],
            r['energy_per_sample']
        ))

    # Paper default
    paper_result = next((r for r in results if r['psi_iw_scale'] == 1000.0), None)
    if paper_result:
        print(f"\nPaper default (Ψ_iw=1000):")
        print(f"  E[Σ^{{-1}}] = {paper_result['avg_precision']:.4f}")
        print(f"  This is a very vague prior (low precision) — as intended")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
