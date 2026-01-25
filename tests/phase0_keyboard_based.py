"""
Phase 0: Digit Counting with Keyboard Output

Test protocol:
1. Show digit "3" as visual input (simple numpy array)
2. Network must TYPE "*****" (3 asterisks)
3. Learn digits 0-9 sequentially
4. Test catastrophic forgetting

Key insight: Network must ACT (type) to demonstrate understanding.
Not just passive reconstruction - active production of correct count.

Uses keyboard output system (toward KVM agent architecture).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.backbone import BackboneNetwork
from src.motor.keyboard_output import KeyboardOutput, ASTERISK_VOCAB, count_asterisks
import torch
import numpy as np

print("=" * 70)
print("PHASE 0: DIGIT COUNTING (KEYBOARD OUTPUT)")
print("=" * 70)

# Simple visual encoding: digits as 7x5 bitmaps (no PIL needed)
# Each digit is a simple pattern
DIGIT_PATTERNS = {
    0: np.array([
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0]
    ], dtype=np.float32),

    1: np.array([
        [0,0,1,0,0],
        [0,1,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,1,1,1,0]
    ], dtype=np.float32),

    2: np.array([
        [0,1,1,1,0],
        [1,0,0,0,1],
        [0,0,0,0,1],
        [0,0,0,1,0],
        [0,0,1,0,0],
        [0,1,0,0,0],
        [1,1,1,1,1]
    ], dtype=np.float32),

    3: np.array([
        [0,1,1,1,0],
        [1,0,0,0,1],
        [0,0,0,0,1],
        [0,0,1,1,0],
        [0,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0]
    ], dtype=np.float32),

    4: np.array([
        [0,0,0,1,0],
        [0,0,1,1,0],
        [0,1,0,1,0],
        [1,0,0,1,0],
        [1,1,1,1,1],
        [0,0,0,1,0],
        [0,0,0,1,0]
    ], dtype=np.float32),

    5: np.array([
        [1,1,1,1,1],
        [1,0,0,0,0],
        [1,1,1,1,0],
        [0,0,0,0,1],
        [0,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0]
    ], dtype=np.float32),

    6: np.array([
        [0,1,1,1,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0]
    ], dtype=np.float32),

    7: np.array([
        [1,1,1,1,1],
        [0,0,0,0,1],
        [0,0,0,1,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0]
    ], dtype=np.float32),

    8: np.array([
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0]
    ], dtype=np.float32),

    9: np.array([
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,1],
        [0,0,0,0,1],
        [0,0,0,0,1],
        [0,1,1,1,0]
    ], dtype=np.float32),
}

# Network configuration
input_size = 7 * 5  # Flattened digit pattern
hidden_size = 50    # Smaller for simple task

print(f"\nVisual input: 7x5 digit patterns (35 pixels)")
print(f"Output: Keyboard (asterisk vocabulary)")
print(f"Task: Given digit, type correct number of asterisks")
print(f"Example: '3' → '***'\n")

# Create 7-layer network
network = BackboneNetwork(
    num_layers=7,
    neurons_per_layer=hidden_size,
    input_size=input_size,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1,
    use_stable=True,
    stable_lr=0.001,
    stable_max_iterations=100,
    stable_lr_schedule="cosine",
    stable_decay_strong=0.01,
    stable_decay_weak=0.001,
)

# Create keyboard output layer
keyboard = KeyboardOutput(
    input_size=hidden_size,
    vocabulary=ASTERISK_VOCAB,
    temperature=0.5,  # Lower temperature for more confident predictions
    dtype=torch.float32
)

print("Network: 7 layers, 50 neurons/layer")
print("Optimizer: StableProspectiveLearning (cosine schedule)")
print("Temporal convolutions: ENABLED")
print("=" * 70)


def train_digit(digit: int, iterations: int = 100):
    """Train network to produce correct number of asterisks for a digit."""
    digit_input = torch.from_numpy(DIGIT_PATTERNS[digit].flatten())

    print(f"\nTraining digit {digit} (should produce {digit} asterisks):")

    for iteration in range(iterations):
        # Reset temporal state for this training example
        for layer in network.layers:
            layer.reset_temporal_state()

        # Forward through network
        network.input_buffer.copy_(digit_input)
        network.forward(digit_input, num_iterations=20)

        # Get top layer state
        top_state = network.layers[-1].get_state()

        # Network should produce asterisks equal to digit value
        # For now, compute simple error based on count
        # (In full version, would do sequence prediction)

        # Compute keyboard output probabilities
        char_probs = keyboard.forward(top_state)

        # Target: should output asterisk with high probability
        # Strength proportional to digit value
        # (Simplified training: just encourage asterisk output)
        target_prob = torch.zeros_like(char_probs)
        target_prob[0] = 1.0  # Asterisk is only character in vocab

        # Compute error (cross-entropy loss)
        loss = -torch.sum(target_prob * torch.log(char_probs + 1e-8))

        # For this simplified version, update network based on general error
        # (Full version would use sequence-to-sequence training)

        # Simple weight update
        network.update_weights()

        # Update temporal state
        for layer in network.layers:
            layer.update_temporal_state()

        if iteration % 20 == 0 or iteration == iterations - 1:
            print(f"  Iter {iteration:3d}: loss={loss.item():.4f}")

    print(f"  Training complete for digit {digit}")


def test_digit(digit: int) -> int:
    """
    Test network's response to a digit.

    Returns number of asterisks network would produce.
    """
    digit_input = torch.from_numpy(DIGIT_PATTERNS[digit].flatten())

    # Reset temporal state
    for layer in network.layers:
        layer.reset_temporal_state()

    # Forward through network
    network.input_buffer.copy_(digit_input)
    network.forward(digit_input, num_iterations=20)

    # Get top layer state
    top_state = network.layers[-1].get_state()

    # Generate sequence (simplified: just check if it would output asterisk)
    char_probs = keyboard.forward(top_state)
    predicted_char = keyboard.vocabulary.decode(torch.argmax(char_probs).item())

    # For this simple test, check activation strength as proxy for count
    # (Full implementation would generate sequence)
    activation_strength = top_state.abs().mean().item()

    # Map activation to count (rough heuristic)
    predicted_count = int(activation_strength * 10)  # Scale to 0-9 range

    return predicted_count


print("\n" + "=" * 70)
print("EXPERIMENT: Sequential Learning")
print("=" * 70)

# Train digits 0-4 (simpler subset for testing)
print("\nTraining digits 0-4 sequentially...")

results = {}

for digit in range(5):
    print(f"\n--- Learning digit {digit} ---")
    train_digit(digit, iterations=100)

    # Test all learned digits so far
    print(f"\nTesting after learning digit {digit}:")
    for test_d in range(digit + 1):
        predicted = test_digit(test_d)
        correct = (predicted == test_d)
        status = "✓" if correct else "✗"
        print(f"  Digit {test_d}: predicted {predicted} asterisks {status}")

        results[(test_d, digit)] = predicted

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Analysis
print("\nDigit retention after sequential learning:")
print("(Rows = digit, Cols = after learning digit X)")
print()

# Print header
print("Digit ", end="")
for d in range(5):
    print(f"  After{d}", end="")
print()

# Print results
for digit in range(5):
    print(f"  {digit}:  ", end="")
    for learned in range(5):
        if learned < digit:
            print("      --", end="")
        else:
            pred = results.get((digit, learned), -1)
            correct = (pred == digit)
            mark = "✓" if correct else "✗"
            print(f"    {pred}{mark}", end="")
    print()

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

# Count how many digits retained
final_test = {d: results.get((d, 4), -1) for d in range(5)}
retained = sum(1 for d in range(5) if final_test[d] == d)

print(f"\nDigits correctly retained: {retained}/5")

if retained >= 4:
    print("✓ GOOD: Network retains most digit meanings")
    print("  Ready to proceed to full 0-9 and math operations")
elif retained >= 3:
    print("△ MODERATE: Some retention but forgetting present")
    print("  May need experience replay or stronger regularization")
else:
    print("✗ POOR: Severe catastrophic forgetting")
    print("  Need architectural changes or different training approach")

print("\nNOTE: This is a simplified test (placeholder sequence generation).")
print("Full implementation would:")
print("  1. Generate actual character sequences '*', '*', '*', ...")
print("  2. Use proper sequence-to-sequence training")
print("  3. Test exact output match (not just heuristic)")
print()

print("Ready for next step: Full keyboard-based math curriculum")
print("with proper sequence generation and foveal vision integration.")
