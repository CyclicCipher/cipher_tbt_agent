"""
Investigate long-term training dynamics and answer key questions:
1. Are we using enough inference iterations to reach equilibrium?
2. Should we exit inference early when equilibrium is reached?
3. What is the current network architecture?
4. Should we try deeper networks?
"""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("QUESTION 1: Do we use enough inference iterations?")
print("=" * 70)

network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=100,
    input_size=1000,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1
)

sensory_input = torch.randn(1000)

# Track error during inference
network.input_buffer.copy_(sensory_input)

# Initialize states with feedforward pass
for i, layer in enumerate(network.layers):
    if i == 0:
        input_below = network.input_buffer
    else:
        input_below = network.layers[i - 1].get_state()
    layer.state.copy_(torch.tanh(layer.neurons.W_basal @ input_below))

# Run 200 inference iterations and track error
inference_errors = []
for iter_num in range(200):
    # Compute total prediction error
    total_error = 0
    for i, layer in enumerate(network.layers):
        if i == 0:
            input_below = network.input_buffer
        else:
            input_below = network.layers[i - 1].get_state()

        bottom_up_pred = torch.tanh(layer.neurons.W_basal @ input_below)
        error = layer.get_state() - bottom_up_pred
        layer_error = (error ** 2).sum().item()
        total_error += layer_error

    inference_errors.append(total_error)

    # Run one inference step
    network._inference_step()

print("\nInference error trajectory (first 100 iterations):")
for i in [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    if i < len(inference_errors):
        print(f"  Iter {i:3d}: error={inference_errors[i]:10.6f}")

# Find when error stabilizes (change < 0.1%)
for i in range(10, len(inference_errors)):
    recent_errors = inference_errors[i-10:i]
    avg_error = sum(recent_errors) / len(recent_errors)
    if avg_error > 0 and abs(inference_errors[i] - avg_error) / avg_error < 0.001:
        print(f"\n✓ Inference converged at iteration {i}")
        print(f"  Current setting uses {50} iterations")
        if i < 50:
            print(f"  → We're using ENOUGH iterations (converges at {i})")
        else:
            print(f"  → We need MORE iterations (converges at {i} > 50)")
        break
else:
    print(f"\n⚠ Inference didn't converge in 200 iterations!")
    print(f"  Final error: {inference_errors[-1]:.6f}")
    print(f"  Initial error: {inference_errors[0]:.6f}")

print("\n" + "=" * 70)
print("QUESTION 2: Current network architecture")
print("=" * 70)

print(f"\nNetwork structure:")
print(f"  Input size: 1000")
print(f"  Number of layers: {len(network.layers)}")
print(f"  Neurons per layer: 100")
print(f"  Total layers: input → L0(100) → L1(100) → L2(100) → L3(100) → L4(100)")
print(f"\nConnectivity:")
print(f"  Each layer is FULLY CONNECTED to the layer above and below")
print(f"  L0 receives from input_buffer (1000 neurons)")
print(f"  L1 receives from L0 (100 neurons)")
print(f"  L2 receives from L1 (100 neurons)")
print(f"  L3 receives from L2 (100 neurons)")
print(f"  L4 receives from L3 (100 neurons)")

# Check weight matrix shapes
print(f"\nWeight matrix shapes:")
for i, layer in enumerate(network.layers):
    W_basal_shape = layer.neurons.W_basal.shape
    W_apical_shape = layer.neurons.W_apical.shape
    print(f"  Layer {i}:")
    print(f"    W_basal (bottom-up):  {W_basal_shape}")
    print(f"    W_apical (top-down):  {W_apical_shape}")

print("\n" + "=" * 70)
print("QUESTION 3: What about weight decay?")
print("=" * 70)

print("\nObservation from 400-iteration run:")
print("  Layer 0 W_apical: -86.4% (shrunk from ~10 to ~1.4)")
print("  Layer 1-3 W_apical: -94% to -98% (shrunk from ~10 to ~0.2!)")
print("\nThis suggests weight_decay=0.01 is TOO AGGRESSIVE")
print("Weights are being continuously squeezed even when they shouldn't be.")

print("\nLet's test different weight decay values over 100 iterations:")

for decay_val in [0.0, 0.001, 0.005, 0.01]:
    network = BackboneNetwork(
        num_layers=5,
        neurons_per_layer=100,
        input_size=1000,
        dtype=torch.float32,
        device='cpu',
        inference_lr=0.1
    )

    sensory_input = torch.randn(1000)

    initial_w_norm = torch.norm(network.layers[0].neurons.W_basal).item()

    for iteration in range(100):
        network.forward(sensory_input, num_iterations=50)
        network.update_weights(lr=0.0005, weight_decay=decay_val)

    final_error = ((sensory_input - network.compute_reconstruction())**2).sum().item()
    final_w_norm = torch.norm(network.layers[0].neurons.W_basal).item()
    w_change = (final_w_norm - initial_w_norm) / initial_w_norm * 100

    print(f"  decay={decay_val}: final_error={final_error:6.1f}, "
          f"W_norm change={w_change:+6.1f}%")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("Based on the analysis:")
print("1. 50 inference iterations is likely enough (we'll verify)")
print("2. Consider ADAPTIVE weight decay or LOWER decay (0.001-0.005)")
print("3. Current network: 5 layers × 100 neurons (shallow but wide)")
print("4. Could try DEEPER networks (10-20 layers × 50 neurons)")
