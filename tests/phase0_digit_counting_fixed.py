"""
Phase 0: Digit Counting (FIXED)

Proper training loop:
1. Show digit "3" as input
2. Top layer target = pattern encoding "3" (e.g., 3 active neurons)
3. Network learns to reconstruct this target
4. Keyboard layer reads top layer state and outputs count

This connects the task to predictive coding's reconstruction objective.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.backbone import BackboneNetwork
import torch
import numpy as np

print("=" * 70)
print("PHASE 0: DIGIT COUNTING (FIXED VERSION)")
print("=" * 70)

# Simple 7x5 digit bitmaps
DIGIT_PATTERNS = {
    0: np.array([[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], dtype=np.float32),
    1: np.array([[0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]], dtype=np.float32),
    2: np.array([[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,1,1,1,1]], dtype=np.float32),
    3: np.array([[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,1,1,0],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], dtype=np.float32),
    4: np.array([[0,0,0,1,0],[0,0,1,1,0],[0,1,0,1,0],[1,0,0,1,0],[1,1,1,1,1],[0,0,0,1,0],[0,0,0,1,0]], dtype=np.float32),
}

input_size = 35
hidden_size = 20

print(f"\nTask: Given digit image -> Produce count")
print(f"Method: Top layer learns count representation")
print(f"Network: 5 layers, {hidden_size} neurons/layer")
print(f"Input: {input_size} pixels (7x5 bitmaps)")
print()

# Smaller network for testing
network = BackboneNetwork(
    num_layers=5,
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

print("=" * 70)

def create_count_target(count: int, size: int) -> torch.Tensor:
    """
    Create target pattern encoding a count.

    Simple encoding: First N neurons active (value 0.8), rest at 0.
    This gives the network a clear target to reconstruct.
    """
    target = torch.zeros(size, dtype=torch.float32)
    if count > 0:
        # Set first 'count' neurons to 0.8
        target[:min(count, size)] = 0.8
    return target


def train_digit(digit: int, iterations: int = 200):
    """Train to associate digit with its count."""
    digit_input = torch.from_numpy(DIGIT_PATTERNS[digit].flatten())
    count_target = create_count_target(digit, hidden_size)

    print(f"\nTraining digit {digit}:")
    errors = []

    for iteration in range(iterations):
        # Reset temporal state
        for layer in network.layers:
            layer.reset_temporal_state()

        # Forward: infer equilibrium state
        network.input_buffer.copy_(digit_input)
        network.forward(digit_input, num_iterations=20)

        # Compute error: top layer vs count target
        top_state = network.layers[-1].get_state()
        error = ((count_target - top_state) ** 2).sum().item()
        errors.append(error)

        # Inject target at top layer (supervised signal)
        # This is how we connect the task to predictive coding
        network.layers[-1].state.copy_(
            0.9 * network.layers[-1].state + 0.1 * count_target
        )

        # Update weights (network learns to produce this pattern)
        network.update_weights()

        # Update temporal
        for layer in network.layers:
            layer.update_temporal_state()

        if iteration % 40 == 0 or iteration == iterations - 1:
            print(f"  Iter {iteration:3d}: error={error:6.3f}")

    final_error = errors[-1]
    print(f"  Final error: {final_error:.3f}")
    return final_error


def test_digit(digit: int) -> int:
    """Test network's count prediction."""
    digit_input = torch.from_numpy(DIGIT_PATTERNS[digit].flatten())

    # Reset temporal
    for layer in network.layers:
        layer.reset_temporal_state()

    # Forward
    network.input_buffer.copy_(digit_input)
    network.forward(digit_input, num_iterations=20)

    # Read top layer
    top_state = network.layers[-1].get_state()

    # Decode count: how many neurons are significantly active?
    active_neurons = (top_state > 0.4).sum().item()

    return int(active_neurons)


print("\nEXPERIMENT: Sequential Learning (Catastrophic Forgetting Test)")
print("=" * 70)

results = {}

for digit in range(5):
    print(f"\n--- Learning digit {digit} ---")
    train_digit(digit, iterations=200)

    # Test all so far
    print(f"\nTesting after learning {digit}:")
    for test_d in range(digit + 1):
        predicted = test_digit(test_d)
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
    print("GOOD: Network retains digit meanings")
elif retained >= 3:
    print("MODERATE: Some forgetting")
else:
    print("POOR: Severe catastrophic forgetting")

print("\n" + "=" * 70)
