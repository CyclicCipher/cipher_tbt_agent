"""
Phase 0: Digit Counting with PROPER Architecture

Correct predictive coding architecture:
- input_buffer (layer 0): Sensory input (digit pattern)
- Layer 0 STATE: Motor output (count representation) - CLAMPED during training
- Layers 1-6: Hierarchical processing

Supervised pre-training protocol:
1. Present digit pattern via input_buffer (sensory input at layer 0)
2. Clamp layer 0 STATE to count pattern (motor target)
3. Network settles to equilibrium via inference
4. Weight update learns to predict motor state from sensory input
5. Test: Present digit, read motor output at layer 0 state

This is the correct active inference / predictive coding architecture.
Motor output is at the BOTTOM layer, sensory input is ALSO at bottom (input_buffer).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.backbone import BackboneNetwork
import torch
import numpy as np

print("=" * 70)
print("PHASE 0: DIGIT COUNTING (PROPER MOTOR-BOTTOM ARCHITECTURE)")
print("=" * 70)

# Simple 7x5 digit bitmaps
DIGIT_PATTERNS = {
    0: np.array([[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], dtype=np.float32),
    1: np.array([[0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]], dtype=np.float32),
    2: np.array([[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,1,1,1,1]], dtype=np.float32),
    3: np.array([[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,1,1,0],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], dtype=np.float32),
    4: np.array([[0,0,0,1,0],[0,0,1,1,0],[0,1,0,1,0],[1,0,0,1,0],[1,1,1,1,1],[0,0,0,1,0],[0,0,0,1,0]], dtype=np.float32),
}

input_size = 35  # Flattened 7x5 digit
hidden_size = 20

print("\nARCHITECTURE:")
print(f"  input_buffer:     Sensory input - {input_size} pixels (digit pattern)")
print(f"  Layer 0 STATE:    Motor output - {hidden_size} neurons (count, CLAMPED)")
print(f"  Layers 1-6:       Processing - {hidden_size} neurons each")
print()
print("Sensory input goes to input_buffer (layer 0 input)")
print("Motor output is layer 0 STATE (clamped during training)")
print()

# 7-layer network (0-6)
network = BackboneNetwork(
    num_layers=7,
    neurons_per_layer=hidden_size,
    input_size=input_size,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1,
    use_stable=True,
    stable_lr=0.001,
    stable_max_iterations=200,
    stable_lr_schedule="cosine",
)

print("Network: 7 layers (0=motor, 6=sensory)")
print("Optimizer: StableProspectiveLearning (cosine schedule)")
print("=" * 70)


def create_count_target(count: int, size: int) -> torch.Tensor:
    """
    Create motor target pattern encoding a count.

    Simple encoding: First N neurons active (value 0.8), rest at 0.
    """
    target = torch.zeros(size, dtype=torch.float32)
    if count > 0:
        target[:min(count, size)] = 0.8
    return target


def train_digit_proper_architecture(digit: int, iterations: int = 200):
    """
    Train with PROPER architecture:
    - Sensory input via input_buffer (layer 0 input)
    - Motor output at layer 0 STATE (clamped)
    - Network learns sensory -> motor mapping
    """
    digit_input = torch.from_numpy(DIGIT_PATTERNS[digit].flatten())
    motor_target = create_count_target(digit, hidden_size)

    print(f"\nTraining digit {digit}:")
    errors = []

    for iteration in range(iterations):
        # Reset temporal state
        for layer in network.layers:
            layer.reset_temporal_state()

        # PROPER ARCHITECTURE TRAINING:
        # 1. Sensory input via input_buffer (normal forward)
        # 2. Motor state (layer 0) gets clamped to target
        # 3. Network learns to produce motor state from sensory input

        # Forward: sensory input via input_buffer
        network.forward(digit_input, num_iterations=20)

        # Compute error: motor layer (layer 0) vs target
        motor_state = network.layers[0].get_state()
        error = ((motor_target - motor_state) ** 2).sum().item()
        errors.append(error)

        # Update weights with motor clamping
        # This clamps layer 0 state to motor_target, then does weight update
        # Network learns: sensory input -> motor state
        network.update_weights(motor_target=motor_target)

        # Update temporal
        for layer in network.layers:
            layer.update_temporal_state()

        if iteration % 40 == 0 or iteration == iterations - 1:
            print(f"  Iter {iteration:3d}: error={error:6.3f}")

    final_error = errors[-1]
    print(f"  Final error: {final_error:.3f}")
    return final_error


def test_digit_proper_architecture(digit: int) -> int:
    """
    Test with proper architecture:
    - Present sensory via input_buffer
    - Read motor output from layer 0 state
    """
    digit_input = torch.from_numpy(DIGIT_PATTERNS[digit].flatten())

    # Reset temporal
    for layer in network.layers:
        layer.reset_temporal_state()

    # Forward: sensory via input_buffer (normal)
    network.forward(digit_input, num_iterations=20)

    # Read motor output at layer 0 STATE
    motor_state = network.layers[0].get_state()

    # Decode count: how many neurons are significantly active?
    active_neurons = (motor_state > 0.4).sum().item()

    return int(active_neurons)


print("\nEXPERIMENT: Sequential Learning with Proper Architecture")
print("=" * 70)

results = {}

for digit in range(5):
    print(f"\n--- Learning digit {digit} ---")
    train_digit_proper_architecture(digit, iterations=200)

    # Test all so far
    print(f"\nTesting after learning {digit}:")
    for test_d in range(digit + 1):
        predicted = test_digit_proper_architecture(test_d)
        correct = (predicted == test_d)
        status = "OK" if correct else "FAIL"
        print(f"  Digit {test_d}: predicted {predicted}, expected {test_d} [{status}]")
        results[(test_d, digit)] = predicted

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Print retention matrix
print("\nRetention matrix (rows=digit, cols=after learning X):")
print("Digit ", end="")
for d in range(5):
    print(f"  After{d}", end="")
print()

for digit in range(5):
    print(f"  {digit}:  ", end="")
    for learned in range(5):
        if learned < digit:
            print("      --", end="")
        else:
            pred = results.get((digit, learned), -1)
            correct = (pred == digit)
            mark = "OK" if correct else "X"
            print(f"    {pred:1d}[{mark}]", end="")
    print()

# Count retention
final_test = {d: results.get((d, 4), -1) for d in range(5)}
retained = sum(1 for d in range(5) if final_test[d] == d)

print(f"\nDigits correctly retained: {retained}/5")

if retained >= 4:
    print("GOOD: Network retains digit meanings with proper architecture")
    print("Motor output at bottom enables supervised pre-training")
elif retained >= 3:
    print("MODERATE: Some forgetting even with proper architecture")
else:
    print("POOR: Architecture alone doesn't solve catastrophic forgetting")

print("\n" + "=" * 70)
print("ARCHITECTURE NOTES:")
print("  - Sensory input via input_buffer (layer 0 input)")
print("  - Motor output at layer 0 STATE (clamped during training)")
print("  - Network learns hierarchical sensory->motor mapping")
print("  - This is proper predictive coding / active inference")
print("  - Both sensory and motor are at BOTTOM layer (layer 0)")
print("=" * 70)
