"""
Test modular architecture with multiple sub-networks.

Demonstrates:
- Position 0: Vision (large) and Keyboard (small) sub-networks in parallel
- Position 1: Association network integrating both modalities
- Variable layer sizes within each sub-network
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.modular import SubNetwork, ModularNetwork
import torch
import numpy as np

print("=" * 70)
print("MODULAR ARCHITECTURE TEST")
print("=" * 70)

# Create sub-networks
# Position 0: Sensory/motor modalities (parallel processing)
vision_subnet = SubNetwork(
    name="vision",
    layer_sizes=[100, 50, 25],  # 3 layers, decreasing size
    input_size=35,  # 7x5 digit input
    position=0,
    dtype=torch.float32,
    device='cpu'
)

keyboard_subnet = SubNetwork(
    name="keyboard",
    layer_sizes=[20, 10],  # 2 layers, smaller network
    input_size=10,  # 10 key states
    position=0,
    dtype=torch.float32,
    device='cpu'
)

# Position 1: Association network (multi-modal integration)
# Input size = vision output (25) + keyboard output (10) = 35
association_subnet = SubNetwork(
    name="association",
    layer_sizes=[50, 30, 20],  # 3 layers
    input_size=35,  # Concatenation of position 0 outputs
    position=1,
    dtype=torch.float32,
    device='cpu'
)

# Create modular network
network = ModularNetwork(
    subnetworks=[vision_subnet, keyboard_subnet, association_subnet],
    inference_lr=0.1,
    temperature=0.0,
    dtype=torch.float32,
    device='cpu'
)

# Print architecture
network.print_architecture()

print("\n" + "=" * 70)
print("FORWARD PASS TEST")
print("=" * 70)

# Create dummy inputs for position 0
vision_input = torch.randn(35, dtype=torch.float32)  # Random 7x5 digit
keyboard_input = torch.randn(10, dtype=torch.float32)  # Random key states

position0_inputs = {
    "vision": vision_input,
    "keyboard": keyboard_input
}

# Forward pass
output = network.forward(position0_inputs, num_iterations=20)

print(f"\nInput shapes:")
print(f"  Vision: {vision_input.shape}")
print(f"  Keyboard: {keyboard_input.shape}")

print(f"\nPosition 0 outputs:")
print(f"  Vision top layer: {vision_subnet.get_top_state().shape}")
print(f"  Keyboard top layer: {keyboard_subnet.get_top_state().shape}")

print(f"\nPosition 1 output:")
print(f"  Association: {output.shape}")

print("\n" + "=" * 70)
print("WEIGHT UPDATE TEST")
print("=" * 70)

# Get initial weights
vision_layer0_weights = vision_subnet.layers[0].neurons.W_basal.clone()

# Update weights
network.update_weights(lr=0.001, weight_decay=0.01)

# Check weights changed
vision_layer0_weights_after = vision_subnet.layers[0].neurons.W_basal
weight_change = (vision_layer0_weights_after - vision_layer0_weights).abs().max().item()

print(f"\nMax weight change in vision layer 0: {weight_change:.6f}")
print("OK: Weights updated" if weight_change > 0 else "FAIL: Weights did not change")

print("\n" + "=" * 70)
print("MOTOR CLAMPING TEST")
print("=" * 70)

# Test motor clamping (supervised learning at position 0)
keyboard_target = torch.ones(20, dtype=torch.float32) * 0.5  # Target for keyboard layer 0

# Forward pass
network.forward(position0_inputs, num_iterations=20)

# Update with motor target
motor_targets = {"keyboard": keyboard_target}
network.update_weights(lr=0.001, weight_decay=0.01, motor_targets=motor_targets)

# Check keyboard layer 0 was clamped
keyboard_layer0_state = keyboard_subnet.layers[0].get_state()
clamping_error = (keyboard_layer0_state - keyboard_target).abs().max().item()

print(f"\nKeyboard layer 0 clamping error: {clamping_error:.6f}")
print("OK: Motor clamping working" if clamping_error < 0.01 else "FAIL: Motor not clamped")

print("\n" + "=" * 70)
print("ARCHITECTURE PROPERTIES")
print("=" * 70)

print("\nKey features:")
print("  - Variable layer sizes within each sub-network")
print("  - Sub-networks at same position run in parallel")
print("  - Outputs concatenated at position boundaries")
print("  - Local prediction error computation")
print("  - Supports motor clamping for supervised learning")

print("\nPosition 0 sub-networks (parallel):")
for subnet in network.subnetworks_by_position[0]:
    print(f"  {subnet.name}: {len(subnet.layers)} layers, {subnet.get_output_size()} output neurons")

print("\nPosition 1 sub-networks:")
for subnet in network.subnetworks_by_position[1]:
    print(f"  {subnet.name}: {len(subnet.layers)} layers, {subnet.get_output_size()} output neurons")
    print(f"    (receives concatenation of position 0: {subnet.input_size} neurons)")

print("\n" + "=" * 70)
print("TESTS COMPLETE")
print("=" * 70)
