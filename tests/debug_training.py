#!/usr/bin/env python3
"""
Debug script to diagnose training issues.
Prints intermediate values to understand what's happening.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.network.modular import ModularNetwork, SubNetwork
from src.vision.retinal_preprocessing import retinal_preprocessing
from src.pretraining.grounded_math_curriculum import GroundedMathCurriculum

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
print(f"Device: {device}")

# Create simple 2-digit test
curriculum = GroundedMathCurriculum(seed=42)
train_dataset = curriculum.generate_digit_recognition_dataset(samples_per_digit=1)

# Build network
vision_subnet = SubNetwork(
    name="vision",
    layer_sizes=[256, 128, 64],
    input_size=30000,
    position=0,
    dtype=dtype,
    device=device
)

motor_subnet = SubNetwork(
    name="motor",
    layer_sizes=[10, 32],
    input_size=10,
    position=0,
    dtype=dtype,
    device=device
)

association_subnet = SubNetwork(
    name="association",
    layer_sizes=[128, 64, 10],
    input_size=96,
    position=1,
    dtype=dtype,
    device=device
)

network = ModularNetwork(
    subnetworks=[vision_subnet, motor_subnet, association_subnet],
    inference_lr=0.05,
    temperature=0.0,
    dtype=dtype,
    device=device,
    use_stable=True,
    stable_lr=0.001
)

print("\nTesting on first training sample...")
img, label = train_dataset[0]
print(f"Label: {label}")

# Preprocess
features = retinal_preprocessing(img)
visual_input = torch.from_numpy(features.flatten()).to(dtype).to(device)
motor_input = torch.zeros(10, dtype=dtype, device=device)
target_one_hot = torch.zeros(10, dtype=dtype, device=device)
target_one_hot[label] = 1.0

print(f"\nBefore training:")
print(f"  Visual input range: [{visual_input.min():.3f}, {visual_input.max():.3f}]")
print(f"  Target: {target_one_hot.cpu().numpy()}")

# Save initial weights
initial_w = motor_subnet.layers[0].neurons.W_basal.data.clone()

# Forward pass WITHOUT clamping first
print("\n--- Forward without clamping ---")
output_no_clamp = network.forward(
    {"vision": visual_input, "motor": motor_input},
    num_iterations=30,
    clamp_layers=None
)
motor_pred_no_clamp = motor_subnet.layers[0].get_state()
print(f"  Motor prediction: {motor_pred_no_clamp.detach().cpu().numpy()}")
print(f"  Predicted digit: {torch.argmax(motor_pred_no_clamp).item()}")

# Forward pass WITH clamping
print("\n--- Forward with clamping ---")
clamp_layers = {
    "vision": visual_input,
    "motor": target_one_hot
}
output_clamp = network.forward(
    {"vision": visual_input, "motor": motor_input},
    num_iterations=30,
    clamp_layers=clamp_layers
)
motor_pred_clamp = motor_subnet.layers[0].get_state()
print(f"  Motor prediction: {motor_pred_clamp.detach().cpu().numpy()}")
print(f"  Motor input_buffer: {motor_subnet.input_buffer.detach().cpu().numpy()}")
print(f"  Predicted digit: {torch.argmax(motor_pred_clamp).item()}")

# Track 10 weight updates
print("\n" + "="*70)
print("TRAINING OVER 10 UPDATES")
print("="*70)

for update_i in range(10):
    print(f"\n--- Update {update_i + 1}/10 ---")

    # Forward pass with clamping
    output = network.forward(
        {"vision": visual_input, "motor": motor_input},
        num_iterations=30,
        clamp_layers=clamp_layers
    )

    # Get prediction
    motor_pred = motor_subnet.layers[0].get_state()
    predicted = torch.argmax(motor_pred).item()
    correct = "✓" if predicted == label else "✗"

    # Weight update with motor supervision
    network.update_weights(
        lr=0.01,
        weight_decay=0.01,
        motor_targets={"motor": target_one_hot}
    )

    # Check weight change
    current_w = motor_subnet.layers[0].neurons.W_basal.data
    weight_change = (current_w - initial_w).abs().max().item()

    print(f"  Predicted: {predicted} (target: {label}) {correct}")
    print(f"  Motor output: {motor_pred.detach().cpu().numpy()[:5]}...")  # First 5 values
    print(f"  Total weight change: {weight_change:.6f}")

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

# Final forward pass
output_final = network.forward(
    {"vision": visual_input, "motor": motor_input},
    num_iterations=30,
    clamp_layers=clamp_layers
)
motor_pred_final = motor_subnet.layers[0].get_state()
predicted_final = torch.argmax(motor_pred_final).item()

print(f"Final prediction: {predicted_final}")
print(f"Target: {label}")
print(f"Motor output: {motor_pred_final.detach().cpu().numpy()}")
print(f"Total weight change: {weight_change:.6f}")

if predicted_final == label:
    print(f"\n✓ SUCCESS: Learned correct prediction after 10 updates!")
else:
    print(f"\n✗ FAILURE: Still predicting {predicted_final} instead of {label}")
