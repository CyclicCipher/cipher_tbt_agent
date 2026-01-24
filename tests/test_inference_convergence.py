"""
Diagnostic: Check if inference is converging properly during each forward pass.

This will show whether the 50 inference iterations are actually minimizing
prediction errors, or if we're stopping before convergence.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.network.backbone import BackboneNetwork


def test_inference_convergence():
    """Check if prediction errors decrease during inference."""
    print("=" * 70)
    print("INFERENCE CONVERGENCE DIAGNOSTIC")
    print("=" * 70)

    network = BackboneNetwork(
        num_layers=5,
        neurons_per_layer=100,
        input_size=1000,
        dtype=torch.float32,
        device="cpu",
        inference_lr=0.1
    )

    sensory_input = torch.randn(1000)

    print("\nRunning inference for 100 iterations...")
    print("Tracking prediction errors at each layer\n")

    # Manually run inference and track errors
    network.input_buffer.copy_(sensory_input)

    # Initialize states to feedforward pass
    for i, layer in enumerate(network.layers):
        if i == 0:
            input_below = network.input_buffer
        else:
            input_below = network.layers[i - 1].get_state()

        # Simple feedforward initialization
        layer.state.copy_(torch.tanh(layer.neurons.W_basal @ input_below))

    # Track errors over inference
    for iter_num in range(100):
        # Compute errors before update
        total_error = 0
        layer_errors = []

        for i, layer in enumerate(network.layers):
            if i == 0:
                input_below = network.input_buffer
            else:
                input_below = network.layers[i - 1].get_state()

            bottom_up_pred = torch.tanh(layer.neurons.W_basal @ input_below)
            error = layer.get_state() - bottom_up_pred
            layer_error = (error ** 2).sum().item()
            layer_errors.append(layer_error)
            total_error += layer_error

        # Print every 10 iterations
        if iter_num % 10 == 0 or iter_num < 5:
            print(f"Iter {iter_num:3d}: Total error={total_error:10.2f}", end="")
            for i, le in enumerate(layer_errors):
                print(f" | L{i}={le:6.1f}", end="")
            print()

        # Run one inference step
        network._inference_step()

    print("\n" + "=" * 70)
    print("Analysis:")
    print("- If errors decrease monotonically: inference is working ✓")
    print("- If errors plateau early: need more iterations or higher LR")
    print("- If errors oscillate: LR too high or implementation bug")
    print("- If higher layer errors don't change: gradient vanishing issue")
    print("=" * 70)


if __name__ == "__main__":
    test_inference_convergence()
