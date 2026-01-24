"""
Test linear predictive coding network (prospective learning).

Linear activations enable direct equilibrium solving.
"""

import sys
from pathlib import Path

# Add project root to path (works on Windows and Linux)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.network.backbone_linear import LinearBackboneNetwork


def test_linear_learning():
    """Test linear network learning with prospective approach."""
    print("=" * 70)
    print("LINEAR NETWORK LEARNING TEST (Prospective Learning)")
    print("=" * 70)

    input_size = 1000
    network = LinearBackboneNetwork(
        num_layers=5,
        neurons_per_layer=100,
        input_size=input_size,
        dtype=torch.float32,
        device="cpu"
    )

    # Create fixed input pattern
    sensory_input = torch.randn(input_size)

    # Linear networks converge faster - use more iterations
    inference_iters = 50
    learning_rate = 0.01  # Start conservative

    print(f"\nNetwork: Linear (no tanh activations)")
    print(f"Training on fixed input for 50 iterations...")
    print(f"Learning rate: {learning_rate}")
    print(f"Inference iterations per update: {inference_iters}")
    print(f"(Linear systems converge faster than nonlinear)\n")

    errors = []
    reconstruction_errors = []
    layer_errors_history = {i: [] for i in range(len(network.layers))}
    weight_norms = {i: {'apical': [], 'basal': []} for i in range(len(network.layers))}

    for iteration in range(50):
        # Forward pass - linear convergence is faster
        network.forward(sensory_input, num_iterations=inference_iters)

        # Compute errors
        total_error = network.compute_total_error()
        reconstruction = network.compute_reconstruction()
        recon_error = ((sensory_input - reconstruction) ** 2).sum().item()

        errors.append(total_error)
        reconstruction_errors.append(recon_error)

        # Track per-layer errors and weight norms
        for i, layer in enumerate(network.layers):
            layer_error = layer.get_total_error()
            layer_errors_history[i].append(layer_error)

            # Weight norms
            w_apical_norm = torch.norm(layer.neurons.W_apical).item()
            w_basal_norm = torch.norm(layer.neurons.W_basal).item()
            weight_norms[i]['apical'].append(w_apical_norm)
            weight_norms[i]['basal'].append(w_basal_norm)

        # Update weights
        network.update_weights(lr=learning_rate)

        # Print progress every 5 iterations
        if iteration % 5 == 0 or iteration < 10:
            print(f"Iteration {iteration:2d}: total_error={total_error:10.2f}, "
                  f"recon_error={recon_error:10.2f}")

            # Print weight norms for first layer
            print(f"             Layer 0 weights: W_apical norm={weight_norms[0]['apical'][-1]:.2f}, "
                  f"W_basal norm={weight_norms[0]['basal'][-1]:.2f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Initial total error:    {errors[0]:.2f}")
    print(f"Final total error:      {errors[-1]:.2f}")
    print(f"Change:                 {(errors[-1] - errors[0]) / errors[0] * 100:+.1f}%")
    print(f"\nInitial recon error:    {reconstruction_errors[0]:.2f}")
    print(f"Final recon error:      {reconstruction_errors[-1]:.2f}")
    print(f"Change:                 {(reconstruction_errors[-1] - reconstruction_errors[0]) / reconstruction_errors[0] * 100:+.1f}%")

    print("\nWeight norm changes:")
    for i in range(len(network.layers)):
        apical_change = (weight_norms[i]['apical'][-1] - weight_norms[i]['apical'][0]) / weight_norms[i]['apical'][0] * 100
        basal_change = (weight_norms[i]['basal'][-1] - weight_norms[i]['basal'][0]) / weight_norms[i]['basal'][0] * 100
        print(f"  Layer {i}: W_apical {apical_change:+6.1f}%, W_basal {basal_change:+6.1f}%")

    # Check for convergence
    print("\nConvergence check:")
    if errors[-1] < errors[0] * 0.1:
        print("  ✓ EXCELLENT: Error reduced by >90%")
    elif errors[-1] < errors[0] * 0.5:
        print("  ✓ GOOD: Error reduced by >50%")
    elif errors[-1] < errors[0]:
        print("  ✓ Learning occurred (error decreased)")
    elif errors[-1] > errors[0] * 2:
        print("  ⚠ ERROR DIVERGED (more than 2x initial)")
    else:
        print("  ~ Error relatively stable (within 2x range)")

    # Check if higher layers are learning
    print("\nHigher layer learning:")
    for i in range(min(3, len(network.layers))):
        weight_change = abs(weight_norms[i]['basal'][-1] - weight_norms[i]['basal'][0]) / weight_norms[i]['basal'][0] * 100
        if weight_change > 5:
            print(f"  Layer {i}: {weight_change:.1f}% change ✓")
        elif weight_change > 1:
            print(f"  Layer {i}: {weight_change:.1f}% change (minimal)")
        else:
            print(f"  Layer {i}: {weight_change:.1f}% change (frozen)")

    return errors, reconstruction_errors, layer_errors_history, weight_norms


if __name__ == "__main__":
    errors, recon_errors, layer_errors, weight_norms = test_linear_learning()
