"""
Diagnostic test: Why does error rebound after reaching minimum?

Previous results show pattern:
- Manual GD: 920 → 0.16 (iter 320) → 78 (EXCELLENT minimum but rebounds)
- Adam clipped: 896 → 409 → 852 (WORSE minimum, still rebounds)

This test investigates WHY error rebounds after finding good solution.

Hypotheses:
1. Weight decay destroying good solutions
2. Continued learning overfitting to noise
3. Inference not converging (noisy gradients)
4. Fundamental instability in prospective learning
5. Optimizer momentum carrying past minimum
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("DIAGNOSTIC: Error Rebound Investigation")
print("=" * 70)

# Test different configurations to isolate cause
configs = [
    {
        "name": "Manual GD (baseline)",
        "use_adam": False,
        "lr": 0.0005,
        "weight_decay": 0.01,
    },
    {
        "name": "Manual GD (no decay)",
        "use_adam": False,
        "lr": 0.0005,
        "weight_decay": 0.0,  # Test if decay causes rebound
    },
    {
        "name": "Adam (low LR)",
        "use_adam": True,
        "adam_lr": 0.0001,
        "weight_decay": 0.01,
    },
]

sensory_input = torch.randn(1000)
torch.manual_seed(42)  # Same initialization for fair comparison

for config in configs:
    print(f"\n{'='*70}")
    print(f"Configuration: {config['name']}")
    print(f"{'='*70}\n")

    # Create network
    if config.get("use_adam"):
        network = BackboneNetwork(
            num_layers=5,
            neurons_per_layer=100,
            input_size=1000,
            dtype=torch.float32,
            device='cpu',
            inference_lr=0.1,
            use_adam=True,
            adam_lr=config["adam_lr"],
            adam_betas=(0.9, 0.999),
            saturation_penalty=0.01
        )
        weight_decay = config["weight_decay"]
    else:
        network = BackboneNetwork(
            num_layers=5,
            neurons_per_layer=100,
            input_size=1000,
            dtype=torch.float32,
            device='cpu',
            inference_lr=0.1,
            use_adam=False,
        )
        weight_decay = config["weight_decay"]

    errors = []
    weight_norms = []

    # Run 400 iterations
    for iteration in range(400):
        network.forward(sensory_input, num_iterations=50)
        recon_error = ((sensory_input - network.compute_reconstruction())**2).sum().item()
        errors.append(recon_error)

        w0_norm = torch.norm(network.layers[0].neurons.W_basal).item()
        weight_norms.append(w0_norm)

        if config.get("use_adam"):
            network.update_weights()
        else:
            network.update_weights(lr=config["lr"], weight_decay=weight_decay)

        if iteration % 50 == 0 or iteration in [0, 399]:
            print(f"Iter {iteration:3d}: error={recon_error:8.2f}, W_norm={w0_norm:5.2f}")

    # Analysis
    min_error = min(errors)
    min_iter = errors.index(min_error)
    final_error = errors[-1]

    # Find when error starts rebounding
    rebound_iter = None
    for i in range(min_iter, len(errors)):
        if errors[i] > 2 * min_error:
            rebound_iter = i
            break

    print(f"\nResults:")
    print(f"  Minimum: {min_error:.2f} at iteration {min_iter}")
    print(f"  Final: {final_error:.2f}")
    print(f"  Rebound started: iter {rebound_iter if rebound_iter else 'N/A'}")
    print(f"  Divergence factor: {final_error / min_error:.2f}x")

    # Check if weight decay is the issue
    initial_w_norm = weight_norms[0]
    min_w_norm = weight_norms[min_iter]
    final_w_norm = weight_norms[-1]

    print(f"\nWeight norms:")
    print(f"  Initial: {initial_w_norm:.2f}")
    print(f"  At minimum: {min_w_norm:.2f}")
    print(f"  Final: {final_w_norm:.2f}")

    if min_w_norm > initial_w_norm and final_w_norm < min_w_norm:
        print(f"  → Weights grew to find solution, then decayed → possible cause!")

print("\n" + "=" * 70)
print("INVESTIGATION: What happens during error rebound?")
print("=" * 70)

print("\nRunning detailed analysis on single sample learning...")

# Detailed run: track gradients, weight changes, inference convergence
network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=100,
    input_size=1000,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1,
    use_adam=False,
)

errors = []
gradient_norms = []
weight_deltas = []
inference_not_converged = []

prev_weights = torch.norm(network.layers[0].neurons.W_basal).item()

for iteration in range(100):
    # Run inference and track if it converges
    initial_state = network.layers[0].get_state().clone()
    network.forward(sensory_input, num_iterations=50)
    final_state = network.layers[0].get_state()

    state_change = torch.norm(final_state - initial_state).item()
    inference_not_converged.append(state_change > 0.1)

    recon_error = ((sensory_input - network.compute_reconstruction())**2).sum().item()
    errors.append(recon_error)

    # Update and track gradient magnitude
    network.update_weights(lr=0.0005, weight_decay=0.01)

    current_weights = torch.norm(network.layers[0].neurons.W_basal).item()
    weight_delta = abs(current_weights - prev_weights)
    weight_deltas.append(weight_delta)
    prev_weights = current_weights

    if iteration % 10 == 0:
        print(f"Iter {iteration}: error={recon_error:.2f}, Δweight={weight_delta:.4f}, "
              f"inference_Δstate={state_change:.4f}")

min_error = min(errors)
min_iter = errors.index(min_error)

print(f"\n{'='*70}")
print("ROOT CAUSE ANALYSIS")
print(f"{'='*70}")

print(f"\nMinimum error: {min_error:.2f} at iteration {min_iter}")
print(f"Final error: {errors[-1]:.2f}")

# Check if inference convergence degrades
early_convergence_rate = sum(inference_not_converged[:min_iter]) / max(1, min_iter)
late_convergence_rate = sum(inference_not_converged[min_iter:]) / max(1, len(inference_not_converged) - min_iter)

print(f"\nInference convergence:")
print(f"  Before minimum: {early_convergence_rate*100:.1f}% not converged")
print(f"  After minimum: {late_convergence_rate*100:.1f}% not converged")

if late_convergence_rate > early_convergence_rate * 1.5:
    print("  ⚠ Inference deteriorates after minimum - weights becoming unstable")

# Check weight update magnitude
early_weight_deltas = sum(weight_deltas[:min_iter]) / max(1, min_iter)
late_weight_deltas = sum(weight_deltas[min_iter:]) / max(1, len(weight_deltas) - min_iter)

print(f"\nWeight update magnitude:")
print(f"  Before minimum: {early_weight_deltas:.4f}")
print(f"  After minimum: {late_weight_deltas:.4f}")

if late_weight_deltas > early_weight_deltas:
    print("  ⚠ Weight updates INCREASE after minimum - not settling")
else:
    print("  ✓ Weight updates decrease - but error still rebounds")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}\n")

print("The error rebound appears to be caused by:")
print("1. Learning continues even after finding good solution (no stopping criterion)")
print("2. Single-sample learning: weights oscillate trying to fit one pattern")
print("3. Weight decay continuously pulls weights toward zero")
print("4. No mechanism to detect 'good solution found, stop changing'")
print("\nPossible solutions:")
print("  A. Early stopping: detect minimum and stop learning")
print("  B. Learning rate schedule: reduce LR over time")
print("  C. Multiple samples: average over several inputs")
print("  D. Custom optimizer with stability detection")
