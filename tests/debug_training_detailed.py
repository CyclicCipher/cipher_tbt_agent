#!/usr/bin/env python3
"""
Detailed diagnostics to understand training failure.
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
print(f"Device: {device}\n")

# Create simple test
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
    use_stable=False,  # Disable optimizer to isolate issue
    stable_lr=0.001
)

# Get sample
img, label = train_dataset[0]
print(f"Target label: {label}\n")

# Preprocess
features = retinal_preprocessing(img)
visual_input = torch.from_numpy(features.flatten()).to(dtype).to(device)
motor_input = torch.zeros(10, dtype=dtype, device=device)
target_one_hot = torch.zeros(10, dtype=dtype, device=device)
target_one_hot[label] = 1.0

print("="*70)
print("DIAGNOSTIC 1: Clamping Persistence During Inference")
print("="*70)

# Forward with clamping
clamp_layers = {"vision": visual_input, "motor": target_one_hot}

# Manually step through inference to check clamping
network.set_position0_inputs({"vision": visual_input, "motor": motor_input})
network._initialize_states()

print(f"\nBefore inference:")
print(f"  Motor input_buffer: {motor_subnet.input_buffer[:5].detach().cpu().numpy()}")
print(f"  Motor layer 0 state: {motor_subnet.layers[0].get_state()[:5].detach().cpu().numpy()}")

# Run 5 inference steps manually with clamping
for iter_i in range(5):
    network._inference_step()

    # Re-clamp
    motor_subnet.input_buffer.copy_(target_one_hot)

    motor_state = motor_subnet.layers[0].get_state()
    print(f"\nIteration {iter_i+1}:")
    print(f"  Motor input_buffer: {motor_subnet.input_buffer[:5].detach().cpu().numpy()}")
    print(f"  Motor layer 0 state: {motor_state[:5].detach().cpu().numpy()}")
    print(f"  Predicted digit: {torch.argmax(motor_state).item()}")

print("\n" + "="*70)
print("DIAGNOSTIC 2: Weight Update Effects")
print("="*70)

# Reset and do one full forward
network.set_position0_inputs({"vision": visual_input, "motor": motor_input})
network._initialize_states()
for _ in range(30):
    network._inference_step()
    motor_subnet.input_buffer.copy_(target_one_hot)

motor_state_before = motor_subnet.layers[0].get_state().clone()
print(f"\nMotor state BEFORE weight update:")
print(f"  Full state: {motor_state_before.detach().cpu().numpy()}")
print(f"  Predicted: {torch.argmax(motor_state_before).item()}")

# Save initial weights
initial_motor_w = motor_subnet.layers[0].neurons.W_basal.data.clone()
initial_motor_b = motor_subnet.layers[0].neurons.W_basal.clone()

# Update weights WITH motor_targets
print(f"\nCalling update_weights with motor_targets...")
network.update_weights(
    lr=0.01,
    weight_decay=0.01,
    motor_targets={"motor": target_one_hot}
)

motor_state_after_update = motor_subnet.layers[0].get_state().clone()
print(f"\nMotor state AFTER weight update (before new forward):")
print(f"  Full state: {motor_state_after_update.detach().cpu().numpy()}")
print(f"  Predicted: {torch.argmax(motor_state_after_update).item()}")
print(f"  Equals target? {torch.allclose(motor_state_after_update, target_one_hot, atol=1e-4)}")

# Check weight changes
weight_change = (motor_subnet.layers[0].neurons.W_basal.data - initial_motor_w).abs().max().item()
print(f"\nWeight change: {weight_change:.6f}")

print("\n" + "="*70)
print("DIAGNOSTIC 3: Gradient Directions")
print("="*70)

# Reset network
network = ModularNetwork(
    subnetworks=[vision_subnet, motor_subnet, association_subnet],
    inference_lr=0.05,
    temperature=0.0,
    dtype=dtype,
    device=device,
    use_stable=False,
    stable_lr=0.001
)

# Forward pass
network.set_position0_inputs({"vision": visual_input, "motor": motor_input})
network._initialize_states()
for _ in range(30):
    network._inference_step()
    motor_subnet.input_buffer.copy_(target_one_hot)

motor_pred = motor_subnet.layers[0].get_state()

# Check error for each neuron
print(f"\nMotor neuron states and errors:")
print(f"Target:     {target_one_hot.detach().cpu().numpy()}")
print(f"Prediction: {motor_pred.detach().cpu().numpy()}")
print(f"Error:      {(target_one_hot - motor_pred).detach().cpu().numpy()}")

# Manually compute what gradient should be
if hasattr(motor_subnet.layers[0], 'error'):
    print(f"\nMotor layer 0 error (from layer): {motor_subnet.layers[0].error.detach().cpu().numpy()}")

print("\n" + "="*70)
print("DIAGNOSTIC 4: Cross-Position Interference")
print("="*70)

# Check if association is interfering
if "association_to_motor" in network.cross_position_predictions:
    cross_pred_weight = network.cross_position_predictions["association_to_motor"]
    assoc_state = association_subnet.layers[0].get_state()
    cross_prediction = cross_pred_weight @ assoc_state

    print(f"\nCross-position prediction from association:")
    print(f"  Association layer 0 state: {assoc_state[:5].detach().cpu().numpy()}")
    print(f"  Prediction for motor top layer: {cross_prediction[:5].detach().cpu().numpy()}")
    print(f"  Motor top layer actual: {motor_subnet.layers[-1].get_state()[:5].detach().cpu().numpy()}")

print("\n" + "="*70)
print("DIAGNOSTIC 5: Input Buffer vs Layer 0")
print("="*70)

# Check relationship between input_buffer and layer 0
network.set_position0_inputs({"vision": visual_input, "motor": target_one_hot})
network._initialize_states()

print(f"\nAfter initialization:")
print(f"  Motor input_buffer: {motor_subnet.input_buffer.detach().cpu().numpy()}")
print(f"  Motor layer 0: {motor_subnet.layers[0].get_state().detach().cpu().numpy()}")

# One inference step
network._inference_step()

print(f"\nAfter 1 inference step:")
print(f"  Motor input_buffer: {motor_subnet.input_buffer.detach().cpu().numpy()}")
print(f"  Motor layer 0: {motor_subnet.layers[0].get_state().detach().cpu().numpy()}")

# Check what layer 0 SHOULD be according to encoding
motor_encoding = torch.tanh(motor_subnet.layers[0].neurons.W_basal @ motor_subnet.input_buffer)
print(f"\nMotor layer 0 encoding of input_buffer:")
print(f"  tanh(W @ input_buffer): {motor_encoding.detach().cpu().numpy()}")
print(f"  Actual layer 0: {motor_subnet.layers[0].get_state().detach().cpu().numpy()}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nKey questions:")
print("1. Does motor layer 0 state equal target after update_weights?")
print("2. Does motor input_buffer stay clamped during inference?")
print("3. Are weight gradients pointing in the right direction?")
print("4. Is cross-position feedback interfering?")
print("5. Why doesn't layer 0 match its input encoding?")
