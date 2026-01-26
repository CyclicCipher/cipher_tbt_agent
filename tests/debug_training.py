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

# Check layer errors before weight update
print("\n--- Errors before weight update ---")
for i, layer in enumerate(motor_subnet.layers):
    if hasattr(layer, 'error') and layer.error is not None:
        print(f"  Motor layer {i} error: mean={layer.error.abs().mean().item():.6f}, max={layer.error.abs().max().item():.6f}")

# Weight update
print("\n--- Weight update ---")
network.update_weights(lr=0.01, weight_decay=0.01)

# Check if weights changed
final_w = motor_subnet.layers[0].neurons.W_basal.data
weight_change = (final_w - initial_w).abs().max().item()
print(f"  Max weight change: {weight_change:.6f}")

# Check layer errors after weight update
print("\n--- Errors after weight update ---")
for i, layer in enumerate(motor_subnet.layers):
    if hasattr(layer, 'error') and layer.error is not None:
        print(f"  Motor layer {i} error: mean={layer.error.abs().mean().item():.6f}, max={layer.error.abs().max().item():.6f}")

# Forward pass again to see if prediction improved
print("\n--- Forward after weight update (with clamping) ---")
output_after = network.forward(
    {"vision": visual_input, "motor": motor_input},
    num_iterations=30,
    clamp_layers=clamp_layers
)
motor_pred_after = motor_subnet.layers[0].get_state()
print(f"  Motor prediction: {motor_pred_after.detach().cpu().numpy()}")
print(f"  Predicted digit: {torch.argmax(motor_pred_after).item()}")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
if weight_change < 1e-6:
    print("⚠ WARNING: Weights barely changed!")
    print("   Possible causes:")
    print("   - Gradients are zero")
    print("   - Errors are zero")
    print("   - Optimizer isn't working")
else:
    print(f"✓ Weights changed by {weight_change:.6f}")

if torch.argmax(motor_pred_after).item() == label:
    print(f"✓ Correct prediction after 1 update!")
else:
    print(f"✗ Still incorrect after 1 update")
