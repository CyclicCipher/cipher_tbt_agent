"""
Test temperature/noise parameter for escaping local minima.

Inspired by protein folding: apply "heat" to escape local minimum and find global minimum.
This implements Langevin dynamics for simulated annealing.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.network.backbone import BackboneNetwork


def test_temperature_sweep():
    """Test different temperature values to find optimal escape mechanism."""
    print("=" * 70)
    print("TEMPERATURE/SIMULATED ANNEALING TEST")
    print("=" * 70)
    print("\nTesting different 'temperatures' for escaping local minima")
    print("Analogy: Protein denaturation - heat allows escape from local minimum\n")

    temperatures = [0.0, 0.001, 0.01, 0.05, 0.1]
    input_size = 1000
    sensory_input = torch.randn(input_size)

    results = {}

    for temp in temperatures:
        print(f"\n{'=' * 70}")
        print(f"Temperature = {temp}")
        print(f"{'=' * 70}")

        network = BackboneNetwork(
            num_layers=5,
            neurons_per_layer=100,
            input_size=input_size,
            dtype=torch.float32,
            device="cpu",
            inference_lr=0.1,
            temperature=temp  # KEY PARAMETER
        )

        inference_iters = 50
        learning_rate = 0.005
        num_training_iters = 30

        errors = []

        for iteration in range(num_training_iters):
            network.forward(sensory_input, num_iterations=inference_iters)
            total_error = network.compute_total_error()
            errors.append(total_error)
            network.update_weights(lr=learning_rate)

            if iteration % 10 == 0:
                print(f"  Iter {iteration:2d}: error={total_error:8.2f}")

        initial_error = errors[0]
        final_error = errors[-1]
        improvement = (initial_error - final_error) / initial_error * 100

        results[temp] = {
            'initial': initial_error,
            'final': final_error,
            'improvement': improvement,
            'errors': errors
        }

        print(f"  → Improvement: {improvement:.1f}%")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"{'Temperature':<15} {'Initial':>10} {'Final':>10} {'Improvement':>12}")
    print("-" * 50)

    for temp in temperatures:
        r = results[temp]
        print(f"{temp:<15.3f} {r['initial']:>10.1f} {r['final']:>10.1f} {r['improvement']:>11.1f}%")

    # Find best temperature
    best_temp = max(results.keys(), key=lambda t: results[t]['improvement'])
    print(f"\nBest temperature: {best_temp} ({results[best_temp]['improvement']:.1f}% improvement)")

    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")
    print("Temperature = 0.0  : No noise (standard gradient descent)")
    print("Temperature ~ 0.01 : Small perturbations (Langevin dynamics)")
    print("Temperature > 0.05 : High noise (simulated annealing)")
    print("\nLike protein folding:")
    print("  - Low temp: stuck in local minimum (native fold)")
    print("  - High temp: denatured, explores widely (finds global minimum)")
    print("  - Optimal: balance between exploration and convergence")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    test_temperature_sweep()
