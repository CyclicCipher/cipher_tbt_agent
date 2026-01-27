"""
Training script for grounded math curriculum.

Implements proper learning progression:
1. Digit recognition (0-9)
2. Digit-to-quantity mapping (3 = ●●●)
3. Addition with quantities
4. Addition with digits only
5. Multiplication with quantities
6. Multiplication with digits only

Each phase builds on the previous, teaching what numbers MEAN.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.network.modular import SubNetwork, ModularNetwork
from src.pretraining.grounded_math_curriculum import GroundedMathCurriculum
from src.vision.retinal_preprocessing import retinal_preprocessing
from src.network.optimizations import EarlyStoppingInference, optimized_inference
import numpy as np
import cv2

print("=" * 70)
print("GROUNDED MATH TRAINING: PHASE 1 - DIGIT RECOGNITION")
print("=" * 70)

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
patch_size = 100  # Digit images are 100×100

print(f"\nConfiguration:")
print(f"  Device: {device}")
print(f"  Dtype: {dtype}")
print(f"  Patch size: {patch_size}×{patch_size}")

# Build network for digit recognition
print("\n" + "=" * 70)
print("BUILDING NETWORK")
print("=" * 70)

# Position 0: Vision + Motor (active inference)
vision_input_size = 3 * patch_size * patch_size  # After retinal preprocessing
vision_subnet = SubNetwork(
    name="vision",
    layer_sizes=[256, 128, 64],
    input_size=vision_input_size,
    position=0,
    dtype=dtype,
    device=device
)

num_digits = 10
num_motor_latent = 32  # Latent layer for motor primitives
motor_subnet = SubNetwork(
    name="motor",
    layer_sizes=[num_digits, num_motor_latent],  # Bottom->Top: output->latent
    input_size=num_digits,
    position=0,
    dtype=dtype,
    device=device
)

# Position 1: Association
association_input_size = 64 + num_motor_latent  # vision + motor latent outputs
association_subnet = SubNetwork(
    name="association",
    layer_sizes=[128, 64, num_digits],
    input_size=association_input_size,
    position=1,
    dtype=dtype,
    device=device
)

# Create network with optimizer
network = ModularNetwork(
    subnetworks=[vision_subnet, motor_subnet, association_subnet],
    inference_lr=0.05,
    temperature=0.0,
    dtype=dtype,
    device=device,
    use_stable=True,
    stable_lr=0.001,
    stable_max_iterations=1000
)

network.print_architecture()

# Generate training data
print("\n" + "=" * 70)
print("GENERATING TRAINING DATA")
print("=" * 70)

curriculum = GroundedMathCurriculum(seed=42)

# Phase 1: Digit recognition
print("\nPhase 1: Digit Recognition")
train_dataset = curriculum.generate_digit_recognition_dataset(samples_per_digit=50)
test_dataset = curriculum.generate_digit_recognition_dataset(samples_per_digit=10)

print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

# Training setup
print("\n" + "=" * 70)
print("TRAINING SETUP")
print("=" * 70)

num_epochs = 10
batch_size = 1  # Process one at a time for now
inference_iterations = 30

early_stop = EarlyStoppingInference(tolerance=1e-3, patience=5, min_iterations=10)

print(f"  Epochs: {num_epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Inference iterations: {inference_iterations}")

# Training loop
print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 70)

    # Shuffle training data
    np.random.shuffle(train_dataset)

    total_loss = 0.0
    correct = 0

    for i, (img, label) in enumerate(train_dataset):
        # Preprocess image
        features = retinal_preprocessing(img)
        visual_input = torch.from_numpy(features.flatten()).to(dtype).to(device)
        motor_input = torch.zeros(num_digits, dtype=dtype, device=device)

        # Supervised learning: Create target for clamping
        target_one_hot = torch.zeros(num_digits, dtype=dtype, device=device)
        target_one_hot[label] = 1.0

        # Forward pass with CLAMPING during inference (prevents trivial zero solution)
        # Clamp vision input AND motor target during all inference iterations
        clamp_layers = {
            "vision": visual_input,      # Sensory clamping: force to observe input
            "motor": target_one_hot       # Motor clamping: supervised learning signal
        }

        output = network.forward(
            {"vision": visual_input, "motor": motor_input},
            num_iterations=inference_iterations,
            clamp_layers=clamp_layers  # Key: clamp during ALL iterations
        )

        # Get motor prediction from BOTTOM layer (output neurons)
        motor_prediction = motor_subnet.layers[0].get_state()

        # Update weights using predictive coding rules with motor supervision
        network.update_weights(
            lr=0.01,
            weight_decay=0.01,
            motor_targets={"motor": target_one_hot}
        )

        # Track metrics
        # Compute loss for monitoring (not for backprop) - detach to avoid gradient issues
        loss = F.cross_entropy(
            motor_prediction.detach().unsqueeze(0),
            torch.tensor([label], device=device)
        )
        total_loss += loss.item()

        predicted = torch.argmax(motor_prediction.detach()).item()
        if predicted == label:
            correct += 1

        # Progress update
        if (i + 1) % 100 == 0:
            avg_loss = total_loss / (i + 1)
            accuracy = correct / (i + 1)
            print(f"  Batch {i + 1}/{len(train_dataset)}: Loss={avg_loss:.4f}, Accuracy={accuracy*100:.1f}%")

    # Epoch summary
    avg_loss = total_loss / len(train_dataset)
    accuracy = correct / len(train_dataset)
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Training Accuracy: {accuracy*100:.1f}%")

    # Evaluation on test set
    print(f"\nEvaluating on test set...")
    test_correct = 0

    for img, label in test_dataset:
        # Preprocess
        features = retinal_preprocessing(img)
        visual_input = torch.from_numpy(features.flatten()).to(dtype).to(device)
        motor_input = torch.zeros(num_digits, dtype=dtype, device=device)

        # Forward pass (no gradients)
        # ONLY clamp vision input - let motor emerge from learned mapping
        with torch.no_grad():
            clamp_layers = {
                "vision": visual_input  # Only sensory clamping, motor free to predict
            }

            output = network.forward(
                {"vision": visual_input, "motor": motor_input},
                num_iterations=inference_iterations,
                clamp_layers=clamp_layers
            )

            motor_prediction = motor_subnet.layers[0].get_state()  # Bottom layer = output
            predicted = torch.argmax(motor_prediction).item()

            if predicted == label:
                test_correct += 1

    test_accuracy = test_correct / len(test_dataset)
    print(f"  Test Accuracy: {test_accuracy*100:.1f}%")

# Final results
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"\nFinal Test Accuracy: {test_accuracy*100:.1f}%")

if test_accuracy > 0.9:
    print("\n✓ SUCCESS: Network learned digit recognition!")
    print("  Ready for Phase 2: Digit-to-quantity mapping")
else:
    print("\n✗ Network needs more training")
    print(f"  Current: {test_accuracy*100:.1f}%, Target: >90%")

# Save checkpoint
checkpoint_path = "checkpoints/phase1_digit_recognition.pth"
os.makedirs("checkpoints", exist_ok=True)

torch.save({
    'epoch': num_epochs,
    'network_state': network.state_dict(),
    'optimizer_state': network.optimizer.state_dict(),
    'test_accuracy': test_accuracy
}, checkpoint_path)

print(f"\nCheckpoint saved: {checkpoint_path}")

# Test on specific examples
print("\n" + "=" * 70)
print("TESTING ON SPECIFIC DIGITS")
print("=" * 70)

for digit in [0, 1, 5, 9]:
    # Generate sample
    img = curriculum._render_digit_varied(digit)

    # Preprocess
    features = retinal_preprocessing(img)
    visual_input = torch.from_numpy(features.flatten()).to(dtype).to(device)
    motor_input = torch.zeros(num_digits, dtype=dtype, device=device)

    # Predict
    with torch.no_grad():
        # Only clamp vision - let motor emerge
        clamp_layers = {"vision": visual_input}

        output = network.forward(
            {"vision": visual_input, "motor": motor_input},
            num_iterations=inference_iterations,
            clamp_layers=clamp_layers
        )

        motor_prediction = motor_subnet.layers[0].get_state()  # Bottom layer = output
        probs = torch.softmax(motor_prediction, dim=0)
        predicted = torch.argmax(probs).item()
        confidence = probs[predicted].item()

    match = "✓" if predicted == digit else "✗"
    print(f"  Digit {digit}: Predicted {predicted} (confidence: {confidence*100:.1f}%) {match}")

    # Save example
    cv2.imwrite(f"test_digit_{digit}_pred_{predicted}.png", img)

print("\nExample images saved: test_digit_*.png")
print("\nNext: Run Phase 2 training (digit-to-quantity mapping)")
