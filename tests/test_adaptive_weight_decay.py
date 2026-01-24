"""
Test adaptive weight decay to prevent excessive weight shrinkage
while maintaining stability.

User observed in 400-iteration run:
- Weights shrunk by 86-98%
- Performance improved despite shrinkage
- But oscillatory behavior suggests decay is too aggressive
"""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("ADAPTIVE WEIGHT DECAY TEST")
print("=" * 70)
print("\nProblem: With decay=0.01, weights shrink by 86-98% over 400 iterations")
print("Even though performance improves, this seems excessive.\n")

configs = [
    {"name": "Current (decay=0.01)", "decay": 0.01, "adaptive": False},
    {"name": "Lower (decay=0.005)", "decay": 0.005, "adaptive": False},
    {"name": "Lower (decay=0.001)", "decay": 0.001, "adaptive": False},
    {"name": "Adaptive (start=0.01, end=0.001)", "decay": 0.01, "adaptive": True},
]

input_size = 1000
sensory_input = torch.randn(input_size)
num_iterations = 200  # Test over 200 iterations

for config in configs:
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"{'='*70}")

    network = BackboneNetwork(
        num_layers=5,
        neurons_per_layer=100,
        input_size=input_size,
        dtype=torch.float32,
        device='cpu',
        inference_lr=0.1
    )

    initial_w_norm = torch.norm(network.layers[0].neurons.W_basal).item()
    errors = []
    weight_norms = []

    for iteration in range(num_iterations):
        network.forward(sensory_input, num_iterations=50)
        recon_error = ((sensory_input - network.compute_reconstruction())**2).sum().item()
        errors.append(recon_error)

        w_norm = torch.norm(network.layers[0].neurons.W_basal).item()
        weight_norms.append(w_norm)

        # Adaptive decay: linearly decrease from start to end
        if config['adaptive']:
            progress = iteration / num_iterations
            current_decay = 0.01 * (1 - progress) + 0.001 * progress
        else:
            current_decay = config['decay']

        network.update_weights(lr=0.0005, weight_decay=current_decay)

        if iteration in [0, 20, 50, 100, 150, 199]:
            print(f"  Iter {iteration:3d}: error={recon_error:8.2f}, "
                  f"W_norm={w_norm:5.2f}, decay={current_decay:.4f}")

    # Summary
    final_error = errors[-1]
    min_error = min(errors)
    min_iter = errors.index(min_error)
    final_w_norm = weight_norms[-1]
    w_change = (final_w_norm - initial_w_norm) / initial_w_norm * 100

    print(f"\n  Summary:")
    print(f"    Initial error:  {errors[0]:.2f}")
    print(f"    Min error:      {min_error:.2f} (at iter {min_iter})")
    print(f"    Final error:    {final_error:.2f}")
    print(f"    Weight change:  {w_change:+.1f}%")
    print(f"    Oscillations:   {sum(1 for i in range(1, len(errors)) if errors[i] > errors[i-1])}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

print("""
Based on your 400-iteration observation:

1. PROBLEM with decay=0.01:
   - Weights shrink excessively (86-98%)
   - Causes oscillatory behavior
   - Network still learns, but inefficiently

2. BETTER OPTIONS:
   a) LOWER FIXED DECAY (0.001-0.005):
      - Less aggressive weight shrinkage
      - More stable long-term behavior
      - Still prevents unbounded growth

   b) ADAPTIVE DECAY (0.01 → 0.001):
      - Strong regularization early (prevents explosion)
      - Weaker regularization later (allows refinement)
      - Similar to learning rate scheduling

3. YOUR RESULTS SUGGEST:
   - The network WANTS smaller weights (more efficient)
   - But decay=0.01 forces shrinkage TOO fast
   - Try decay=0.005 or adaptive for 400-iteration runs
""")
