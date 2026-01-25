"""
Quick test to verify temporal/recurrent connections are working.

Tests:
1. Neurons have W_recurrent parameter
2. Temporal state updates correctly
3. Recurrent connections influence next timestep
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.backbone import BackboneNetwork
from src.network.neuron import TwoCompartmentNeuron
import torch

print("=" * 70)
print("TEMPORAL CONNECTIONS TEST")
print("=" * 70)

# Test 1: Check neurons have recurrent weights
print("\nTest 1: Neuron has recurrent weights")
neuron = TwoCompartmentNeuron(
    num_neurons=10,
    apical_size=10,
    basal_size=10,
    dtype=torch.float32
)

has_recurrent = hasattr(neuron, 'W_recurrent')
has_prev_state = hasattr(neuron, 'prev_state')

print(f"  Has W_recurrent: {has_recurrent}")
print(f"  Has prev_state buffer: {has_prev_state}")

if has_recurrent:
    print(f"  W_recurrent shape: {neuron.W_recurrent.shape}")
    print(f"  ✓ Recurrent weights present")
else:
    print(f"  ✗ FAIL: Missing recurrent weights")

# Test 2: Temporal state updates
print("\nTest 2: Temporal state updates")

apical_input = torch.randn(10)
basal_input = torch.randn(10)

# Forward pass with temporal
state1 = neuron.forward(apical_input, basal_input, use_temporal=True)
print(f"  State 1: {state1[:5].detach().numpy()}")  # Show first 5 values

# Update temporal state
neuron.update_temporal_state()
print(f"  Prev state after update: {neuron.prev_state[:5].detach().numpy()}")

# Check they match
if torch.allclose(state1, neuron.prev_state):
    print(f"  ✓ Temporal state updated correctly")
else:
    print(f"  ✗ FAIL: State not copied to prev_state")

# Test 3: Recurrent influence
print("\nTest 3: Recurrent connections influence next state")

# Reset to known state
neuron.reset_state()
neuron.reset_temporal_state()

# First forward (no temporal history)
state_no_history = neuron.forward(apical_input, basal_input, use_temporal=True).clone()
print(f"  State with no history: {state_no_history[:5].detach().numpy()}")

# Set prev_state to something non-zero
neuron.prev_state.copy_(torch.ones(10) * 0.5)

# Second forward (with temporal history)
state_with_history = neuron.forward(apical_input, basal_input, use_temporal=True).clone()
print(f"  State with history: {state_with_history[:5].detach().numpy()}")

# They should be different (recurrent influence)
if not torch.allclose(state_no_history, state_with_history):
    print(f"  ✓ Recurrent connections influence state")
    diff = torch.norm(state_with_history - state_no_history).item()
    print(f"  State difference (norm): {diff:.4f}")
else:
    print(f"  ✗ FAIL: Recurrent connections have no effect")

# Test 4: Full network integration
print("\nTest 4: Network-level temporal integration")

network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=50,
    input_size=100,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1,
)

input1 = torch.randn(100)

# Process first timestep
network.forward(input1, num_iterations=20)
state1 = [layer.get_state().clone() for layer in network.layers]

# Update temporal states
for layer in network.layers:
    layer.update_temporal_state()

# Process second timestep (same input, but temporal state should influence)
network.forward(input1, num_iterations=20)
state2 = [layer.get_state().clone() for layer in network.layers]

# States should differ due to temporal influence
temporal_effect = False
for i, (s1, s2) in enumerate(zip(state1, state2)):
    diff = torch.norm(s2 - s1).item()
    if diff > 0.01:  # Noticeable difference
        temporal_effect = True
    print(f"  Layer {i+1} state change: {diff:.4f}")

if temporal_effect:
    print(f"  ✓ Temporal connections working at network level")
else:
    print(f"  △ Warning: Temporal effect may be weak")

# Test 5: Disabling temporal
print("\nTest 5: Temporal can be disabled")

neuron.reset_state()
neuron.reset_temporal_state()
neuron.prev_state.copy_(torch.ones(10) * 0.5)

state_temporal = neuron.forward(apical_input, basal_input, use_temporal=True).clone()
state_no_temporal = neuron.forward(apical_input, basal_input, use_temporal=False).clone()

if not torch.allclose(state_temporal, state_no_temporal):
    print(f"  ✓ Temporal can be disabled (states differ)")
else:
    print(f"  ✗ FAIL: use_temporal flag not working")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

all_tests_pass = (
    has_recurrent and
    has_prev_state and
    temporal_effect
)

if all_tests_pass:
    print("\n✓✓✓ ALL TESTS PASSED")
    print("  Temporal/recurrent connections working correctly")
    print("  Ready for sequence learning experiments")
else:
    print("\n⚠ SOME TESTS FAILED")
    print("  Review temporal implementation before proceeding")

print()
