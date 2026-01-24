"""Test 100 iterations with correct weight update sign."""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=100,
    input_size=1000,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1
)

sensory_input = torch.randn(1000)

print("=" * 70)
print("100-ITERATION TEST WITH CORRECT SIGN (W += lr * error * input)")
print("=" * 70)

for iteration in range(100):
    # Run inference
    network.forward(sensory_input, num_iterations=50)

    # Compute errors
    total_error = network.compute_total_error()
    recon_error = ((sensory_input - network.compute_reconstruction())**2).sum().item()

    # Manual weight update with CORRECT sign (addition)
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

        # Apply weight update with ADDITION (correct sign)
        lr = 0.001
        error_col = layer_error.unsqueeze(1)

        with torch.no_grad():
            layer.neurons.W_apical += lr * error_col * input_from_above.unsqueeze(0)
            layer.neurons.W_basal += lr * error_col * input_from_below.unsqueeze(0)

    # Print progress
    if iteration % 5 == 0 or iteration < 10 or iteration >= 95:
        w0_norm = torch.norm(network.layers[0].neurons.W_basal).item()
        print(f"Iteration {iteration:3d}: recon_error={recon_error:10.2f}, "
              f"Layer0 W_basal norm={w0_norm:6.2f}")

print("\n" + "=" * 70)
print("✓ If this doesn't diverge, the correct sign fixes the instability!")
print("=" * 70)
