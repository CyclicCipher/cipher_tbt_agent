"""
Analyze neuron activation patterns to detect coma/seizure states.
Check if we need activity regularization.
"""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("NEURON ACTIVATION ANALYSIS: Detecting Coma/Seizure States")
print("=" * 70)

network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=100,
    input_size=1000,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1
)

sensory_input = torch.randn(1000)

# Train for 50 iterations and track activations
print("\nTracking activation statistics during training:\n")

for iteration in [0, 10, 20, 30, 40, 49]:
    network.forward(sensory_input, num_iterations=50)

    print(f"Iteration {iteration}:")

    # Analyze each layer's activation
    for i, layer in enumerate(network.layers):
        state = layer.get_state()

        # Activation statistics
        mean_abs = state.abs().mean().item()
        std = state.std().item()
        fraction_active = (state.abs() > 0.1).float().mean().item()  # Threshold 0.1
        max_activation = state.abs().max().item()

        # Saturation (how many neurons near ±1)
        fraction_saturated = (state.abs() > 0.9).float().mean().item()

        print(f"  Layer {i}: mean_abs={mean_abs:.3f}, std={std:.3f}, "
              f"active%={fraction_active*100:.1f}%, saturated%={fraction_saturated*100:.1f}%")

    network.update_weights(lr=0.0005, weight_decay=0.01)
    print()

print("=" * 70)
print("DIAGNOSIS")
print("=" * 70)

print("""
NORMAL ACTIVATION PATTERNS:
- Mean absolute activation: 0.2-0.6 (healthy)
- Active neurons (|x| > 0.1): 50-90% (reasonable)
- Saturated neurons (|x| > 0.9): <10% (good)

PATHOLOGICAL PATTERNS:

COMA (too little activity):
- Mean absolute < 0.1
- Active neurons < 20%
- Indicates: Vanishing activations, dying neurons

SEIZURE (too much activity):
- Mean absolute > 0.8
- Saturated neurons > 30%
- Indicates: Exploding activations, mode collapse

ANALYSIS FROM YOUR DATA:
- If mean_abs drops over time → Weight decay too aggressive
- If saturated% increases → Need activity regularization
- If active% drops → Neurons dying, need revival mechanism
""")

print("\n" + "=" * 70)
print("TESTING SPARSITY CONSTRAINT")
print("=" * 70)

print("\nYour idea: '1 neuron activates 1 downstream neuron'")
print("Let's measure actual branching ratio:\n")

network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=100,
    input_size=1000,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1
)

# Train briefly
for _ in range(10):
    network.forward(sensory_input, num_iterations=50)
    network.update_weights(lr=0.0005, weight_decay=0.01)

# Measure branching ratio
network.forward(sensory_input, num_iterations=50)

print("Branching ratio (active neurons in layer N+1 / active in layer N):")
active_threshold = 0.1

for i in range(len(network.layers) - 1):
    state_current = network.layers[i].get_state()
    state_next = network.layers[i + 1].get_state()

    active_current = (state_current.abs() > active_threshold).sum().item()
    active_next = (state_next.abs() > active_threshold).sum().item()

    if active_current > 0:
        branching_ratio = active_next / active_current
    else:
        branching_ratio = 0

    print(f"  Layer {i} → Layer {i+1}: {active_current} active → {active_next} active "
          f"(ratio={branching_ratio:.2f})")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("""
1. ACTIVITY REGULARIZATION (prevent coma):
   Add to loss: -λ * mean(|activations|)
   Encourages neurons to stay active

2. SPARSITY REGULARIZATION (efficient coding):
   Add to loss: +λ * mean(|activations|)
   Encourages sparse representations (like cortex ~1% active)

3. SATURATION PENALTY (prevent seizure):
   Add to loss: +λ * fraction(|activations| > 0.9)
   Prevents neurons from saturating

4. CRITICAL DYNAMICS (your 1→1 idea):
   Monitor branching ratio
   Adjust learning rates to maintain ratio ≈ 1.0
   Related to edge-of-chaos in neural networks

5. ADAPTIVE WEIGHT DECAY:
   Current decay=0.01 might be too aggressive
   Consider: decay = 0.01 * (1 - sparsity)
   Decay less when neurons already sparse
""")
