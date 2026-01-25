"""
Phase 0: Digit Semantics Learning

Teach the network what numbers MEAN (not just what they look like).

Protocol:
1. Show digit "3" as input → Network must reconstruct "●●●" (3 dots)
2. Show digit "5" as input → Network must reconstruct "●●●●●" (5 dots)
3. Test all digits 0-9

This tests:
- Can network learn semantic associations?
- Does it understand quantity (not just memorize patterns)?
- Foundation for all math learning

Key insight: Network must PRODUCE dots (action), not just observe them together.
This is predictive coding: learn by predicting, correct via error signals.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.backbone import BackboneNetwork
from src.utils.visual_encoding import VisualEncoder, create_digit_semantics_dataset
import torch
import numpy as np

print("=" * 70)
print("PHASE 0: DIGIT SEMANTICS LEARNING")
print("=" * 70)

# Create visual encoder
encoder = VisualEncoder(image_size=(28, 28), font_size=20, dtype=torch.float32)
input_size = encoder.get_input_size()  # 28*28 = 784

print(f"\nInput size: {input_size} pixels (28x28 images)")
print(f"Task: Given digit image → Reconstruct dot pattern")
print(f"Example: '3' → ●●● (3 dots)\n")

# Create 7-layer network with temporal convolutions
network = BackboneNetwork(
    num_layers=7,              # Deeper for better abstraction
    neurons_per_layer=100,      # Start with 100, increase if needed
    input_size=input_size,      # 784 pixels
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1,
    use_stable=True,
    stable_lr=0.001,
    stable_max_iterations=200,  # Shorter per-digit training
    stable_lr_schedule="cosine",
    stable_decay_strong=0.01,
    stable_decay_weak=0.001,
)

print(f"Network: 7 layers, 100 neurons/layer")
print(f"Optimizer: StableProspectiveLearning (cosine LR schedule)")
print(f"Temporal convolutions: ENABLED (recurrent connections)\n")

# Create dataset
dataset = create_digit_semantics_dataset(digits=list(range(10)), encoder=encoder)

print(f"Dataset: {len(dataset)} digit-dots pairs (0-9)")
print("=" * 70)

# Training protocol
def train_on_digit(digit_idx, iterations=200):
    """Train network to associate one digit with its dot pattern."""
    digit_img, dots_img = dataset[digit_idx]

    print(f"\nTraining on digit {digit_idx}:")
    errors = []

    # Reset temporal state for new learning episode
    for layer in network.layers:
        layer.reset_temporal_state()

    for iteration in range(iterations):
        # Forward pass: Show digit, network produces prediction
        network.input_buffer.copy_(digit_img)
        network.forward(digit_img, num_iterations=50)

        # Compute error: difference between prediction and dots
        reconstruction = network.compute_reconstruction()
        error = ((dots_img - reconstruction) ** 2).sum().item()
        errors.append(error)

        # Update weights (network learns from error)
        network.update_weights()

        # Update temporal state for next iteration
        for layer in network.layers:
            layer.update_temporal_state()

        if iteration % 20 == 0 or iteration == iterations - 1:
            print(f"  Iter {iteration:3d}: error={error:8.2f}")

    final_error = errors[-1]
    min_error = min(errors)

    print(f"  Final error: {final_error:.2f}, Min: {min_error:.2f}")

    return final_error

def test_digit(digit_idx):
    """Test network's ability to recall digit-dots association."""
    digit_img, dots_img = dataset[digit_idx]

    # Reset temporal state
    for layer in network.layers:
        layer.reset_temporal_state()

    # Forward only (no learning)
    network.input_buffer.copy_(digit_img)
    network.forward(digit_img, num_iterations=50)

    # Compute error
    reconstruction = network.compute_reconstruction()
    error = ((dots_img - reconstruction) ** 2).sum().item()

    return error

# Experiment 1: Learn digits sequentially (test catastrophic forgetting)
print("\n" + "=" * 70)
print("EXPERIMENT 1: Sequential Learning (Catastrophic Forgetting Test)")
print("=" * 70)

forgetting_matrix = np.zeros((10, 10))  # forgetting_matrix[i][j] = error on digit i after learning digit j

# Learn each digit one by one
for current_digit in range(10):
    print(f"\n--- Learning digit {current_digit} ---")
    train_error = train_on_digit(current_digit, iterations=200)

    # Test ALL digits (see what was forgotten)
    print(f"\nTesting all digits after learning {current_digit}:")
    for test_digit in range(10):
        test_error = test_digit(test_digit)
        forgetting_matrix[test_digit][current_digit] = test_error

        if test_digit <= current_digit:
            # Should remember this (already learned)
            status = "✓" if test_error < 100 else "✗ FORGOT"
        else:
            # Hasn't learned this yet
            status = "(not learned)"

        print(f"  Digit {test_digit}: error={test_error:6.2f} {status}")

# Analysis
print("\n" + "=" * 70)
print("CATASTROPHIC FORGETTING ANALYSIS")
print("=" * 70)

print("\nForgetting matrix (rows=digits, cols=after learning digit X):")
print("Lower error = better retention")
print()

# Print header
print("     ", end="")
for i in range(10):
    print(f"  After{i}", end="")
print()

# Print matrix
for digit in range(10):
    print(f"Dig {digit}:", end="")
    for learned in range(10):
        error = forgetting_matrix[digit][learned]
        if learned < digit:
            # Not learned yet
            print(f"      --", end="")
        else:
            # Learned, check retention
            print(f"  {error:6.0f}", end="")
    print()

# Calculate catastrophic forgetting metric
print("\nCatastrophic forgetting scores:")
for digit in range(1, 10):  # Skip 0 (nothing to forget)
    # Compare error on digit after learning it vs after learning digit 9
    initial_error = forgetting_matrix[digit][digit]
    final_error = forgetting_matrix[digit][9]

    if initial_error < 100:  # Only if it learned it initially
        forgetting_pct = ((final_error - initial_error) / max(initial_error, 1)) * 100
        status = "✓ Retained" if forgetting_pct < 50 else "✗ FORGOT"
        print(f"  Digit {digit}: {forgetting_pct:+6.1f}% error increase {status}")

# Summary
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

# Count how many digits retained
retained = 0
for digit in range(10):
    if forgetting_matrix[digit][9] < 150:  # Reasonable threshold
        retained += 1

print(f"\nDigits retained: {retained}/10")

if retained >= 8:
    print("✓✓✓ EXCELLENT: Network retains digit semantics with minimal forgetting")
    print("  Temporal convolutions + cosine LR prevent catastrophic forgetting")
elif retained >= 5:
    print("✓ GOOD: Some forgetting but network maintains core knowledge")
    print("  May benefit from experience replay or consolidated learning")
elif retained >= 3:
    print("△ MODERATE: Significant forgetting, but some retention")
    print("  Need stronger regularization or architectural changes")
else:
    print("✗ SEVERE FORGETTING: Network cannot maintain multiple digit semantics")
    print("  Fundamental issue - may need separate memory module")

print("\nNext steps:")
if retained >= 5:
    print("  1. Proceed to Phase 0.5 (multi-digit numbers: 10-99)")
    print("  2. Test generalization (does it understand '10' = 10 dots?)")
    print("  3. Begin Phase 1 (addition)")
else:
    print("  1. Investigate forgetting mechanism (optimizer vs architecture)")
    print("  2. Try experience replay (interleave old digits while learning new)")
    print("  3. Strengthen regularization or reduce LR")

print("\n" + "=" * 70)
