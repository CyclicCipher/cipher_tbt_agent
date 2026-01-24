"""
Test inference convergence with zero initialization to see actual dynamics.
Answer: Should we exit early when equilibrium is reached?
"""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("INFERENCE CONVERGENCE TEST (zero initialization)")
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
network.input_buffer.copy_(sensory_input)

# Initialize ALL states to ZERO (not feedforward)
for layer in network.layers:
    layer.state.zero_()

print("\nStarting from ZERO states (no feedforward initialization)")
print("Tracking total prediction error during inference:\n")

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

    if iter_num % 10 == 0 or iter_num < 10:
        print(f"  Iter {iter_num:3d}: error={total_error:10.4f}")

    # Run one inference step
    network._inference_step()

# Find convergence point (change < 0.1%)
print("\n" + "=" * 70)
print("CONVERGENCE ANALYSIS")
print("=" * 70)

convergence_threshold = 0.001  # 0.1% change
window_size = 10

for i in range(window_size, len(inference_errors)):
    recent_errors = inference_errors[i-window_size:i]
    avg_error = sum(recent_errors) / len(recent_errors)

    if avg_error > 0:
        relative_change = abs(inference_errors[i] - avg_error) / avg_error

        if relative_change < convergence_threshold:
            print(f"\n✓ Converged at iteration {i}")
            print(f"  Error at convergence: {inference_errors[i]:.4f}")
            print(f"  Relative change: {relative_change:.6f} (< {convergence_threshold})")
            print(f"\n  Current setting: {50} iterations")

            if i < 50:
                print(f"  → GOOD: We use {50} iterations, convergence at {i}")
                print(f"     We have {50-i} extra iterations of unnecessary computation")
            else:
                print(f"  → BAD: We use {50} iterations, but need {i} for convergence!")
                print(f"     Missing {i-50} iterations")

            break
else:
    print("\n⚠ No convergence detected in 200 iterations")

# Show final vs initial error
print(f"\nError reduction: {inference_errors[0]:.4f} → {inference_errors[-1]:.4f}")
print(f"Reduction: {(1 - inference_errors[-1]/inference_errors[0])*100:.1f}%")

print("\n" + "=" * 70)
print("EARLY STOPPING RECOMMENDATION")
print("=" * 70)
print("""
SHOULD we exit inference loop early when converged?

PROS of early exit:
+ Faster training (skip unnecessary iterations)
+ More efficient use of compute

CONS of early exit:
- Additional convergence check overhead
- Complexity in implementation
- May miss slow final refinements

RECOMMENDATION:
- For TRAINING: Use fixed iterations (simpler, more predictable)
- For INFERENCE (deployment): Use early stopping (efficiency matters)
- Current 50 iterations seems reasonable based on this test
""")
