"""
Test whether weight updates should use += or -= to reduce error.

This test creates a simple network and checks whether:
- W += lr * error * input (increases weights when error is positive)
- W -= lr * error * input (decreases weights when error is positive)

...leads to error reduction.
"""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

def test_update_sign(use_addition: bool, num_updates: int = 20):
    """
    Test weight updates with either addition or subtraction.

    Args:
        use_addition: If True, use W += lr*error*input. If False, use W -= lr*error*input
        num_updates: Number of weight updates to perform
    """
    network = BackboneNetwork(
        num_layers=5,
        neurons_per_layer=100,
        input_size=1000,
        dtype=torch.float32,
        device='cpu',
        inference_lr=0.1
    )

    sensory_input = torch.randn(1000)
    errors = []

    for i in range(num_updates):
        # Run inference
        network.forward(sensory_input, num_iterations=50)

        # Compute error
        total_error = network.compute_total_error()
        recon_error = ((sensory_input - network.compute_reconstruction())**2).sum().item()
        errors.append((total_error, recon_error))

        # Manual weight update with chosen sign
        for layer_idx in range(len(network.layers)):
            layer = network.layers[layer_idx]

            # Get prediction from layer above
            if layer_idx == len(network.layers) - 1:
                prediction_from_above = torch.zeros_like(layer.get_state())
            else:
                prediction_from_above = network.layers[layer_idx + 1].compute_prediction_for_below()

            # Value error
            layer_error = layer.get_state() - prediction_from_above

            # Get inputs
            if layer_idx == 0:
                input_from_below = network.input_buffer
            else:
                input_from_below = network.layers[layer_idx - 1].get_state()

            if layer_idx == len(network.layers) - 1:
                input_from_above = layer.get_state()
            else:
                input_from_above = network.layers[layer_idx + 1].get_state()

            # Apply weight update with chosen sign
            lr = 0.001
            error_col = layer_error.unsqueeze(1)

            with torch.no_grad():
                if use_addition:
                    # W += lr * error * input
                    layer.neurons.W_apical += lr * error_col * input_from_above.unsqueeze(0)
                    layer.neurons.W_basal += lr * error_col * input_from_below.unsqueeze(0)
                else:
                    # W -= lr * error * input (current implementation)
                    layer.neurons.W_apical -= lr * error_col * input_from_above.unsqueeze(0)
                    layer.neurons.W_basal -= lr * error_col * input_from_below.unsqueeze(0)

    return errors

print("=" * 70)
print("WEIGHT UPDATE SIGN TEST")
print("=" * 70)
print("\nTesting W -= lr * error * input (current implementation)")
print("-" * 70)

errors_subtraction = test_update_sign(use_addition=False, num_updates=20)
print(f"Initial error: {errors_subtraction[0][1]:.2f}")
print(f"Final error:   {errors_subtraction[-1][1]:.2f}")
print(f"Change:        {(errors_subtraction[-1][1] - errors_subtraction[0][1]) / errors_subtraction[0][1] * 100:+.1f}%")
print(f"\nError trajectory (first 10):")
for i, (total_err, recon_err) in enumerate(errors_subtraction[:10]):
    print(f"  Update {i:2d}: {recon_err:8.2f}")

print("\n" + "=" * 70)
print("Testing W += lr * error * input (flipped sign)")
print("-" * 70)

errors_addition = test_update_sign(use_addition=True, num_updates=20)
print(f"Initial error: {errors_addition[0][1]:.2f}")
print(f"Final error:   {errors_addition[-1][1]:.2f}")
print(f"Change:        {(errors_addition[-1][1] - errors_addition[0][1]) / errors_addition[0][1] * 100:+.1f}%")
print(f"\nError trajectory (first 10):")
for i, (total_err, recon_err) in enumerate(errors_addition[:10]):
    print(f"  Update {i:2d}: {recon_err:8.2f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

sub_improvement = (errors_subtraction[0][1] - errors_subtraction[-1][1]) / errors_subtraction[0][1] * 100
add_improvement = (errors_addition[0][1] - errors_addition[-1][1]) / errors_addition[0][1] * 100

print(f"W -= lr*e*x: {sub_improvement:+.1f}% error reduction")
print(f"W += lr*e*x: {add_improvement:+.1f}% error reduction")

if add_improvement > sub_improvement:
    print("\n✓ ADDITION (W += ...) is correct!")
    print("  The current implementation uses SUBTRACTION, which is WRONG.")
else:
    print("\n✓ SUBTRACTION (W -= ...) is correct!")
    print("  The current implementation is correct.")
