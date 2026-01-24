"""
Test different network architectures:
- Current: 5 layers × 100 neurons (shallow & wide)
- Alternative: 10 layers × 50 neurons (deeper & narrower)
- Alternative: 20 layers × 25 neurons (very deep & narrow)
"""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("NETWORK ARCHITECTURE COMPARISON")
print("=" * 70)

configs = [
    {"name": "Current (5×100)", "layers": 5, "neurons": 100},
    {"name": "Deeper (10×50)", "layers": 10, "neurons": 50},
    {"name": "Very Deep (20×25)", "layers": 20, "neurons": 25},
]

input_size = 1000
sensory_input = torch.randn(input_size)

results = []

for config in configs:
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"  Layers: {config['layers']}, Neurons/layer: {config['neurons']}")
    print(f"{'='*70}")

    network = BackboneNetwork(
        num_layers=config['layers'],
        neurons_per_layer=config['neurons'],
        input_size=input_size,
        dtype=torch.float32,
        device='cpu',
        inference_lr=0.1
    )

    # Count parameters
    total_params = 0
    for layer in network.layers:
        total_params += layer.neurons.W_basal.numel()
        total_params += layer.neurons.W_apical.numel()

    print(f"\nTotal parameters: {total_params:,}")

    # Train for 50 iterations
    errors = []
    for iteration in range(50):
        network.forward(sensory_input, num_iterations=50)
        recon_error = ((sensory_input - network.compute_reconstruction())**2).sum().item()
        errors.append(recon_error)

        network.update_weights(lr=0.0005, weight_decay=0.01)

        if iteration in [0, 10, 20, 30, 40, 49]:
            print(f"  Iter {iteration:2d}: error={recon_error:8.2f}")

    # Summary
    initial_error = errors[0]
    final_error = errors[-1]
    min_error = min(errors)
    reduction = (initial_error - final_error) / initial_error * 100

    results.append({
        "name": config['name'],
        "params": total_params,
        "initial": initial_error,
        "final": final_error,
        "min": min_error,
        "reduction": reduction
    })

    print(f"\n  Summary:")
    print(f"    Initial error: {initial_error:.2f}")
    print(f"    Final error:   {final_error:.2f}")
    print(f"    Min error:     {min_error:.2f}")
    print(f"    Reduction:     {reduction:.1f}%")

# Comparison table
print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)

print(f"\n{'Architecture':<20} {'Parameters':<15} {'Initial':<10} {'Final':<10} {'Min':<10} {'Reduction'}")
print("-" * 70)

for r in results:
    print(f"{r['name']:<20} {r['params']:<15,} {r['initial']:<10.1f} {r['final']:<10.1f} {r['min']:<10.1f} {r['reduction']:>6.1f}%")

print("\n" + "=" * 70)
print("INSIGHTS")
print("=" * 70)

print("""
1. PARAMETER COUNT:
   - Shallow & wide (5×100): More parameters due to large input→L0 connection
   - Deep & narrow (20×25): Fewer parameters, more hierarchical processing

2. LEARNING DYNAMICS:
   - Deeper networks may learn slower initially
   - But can learn more hierarchical features
   - Current shallow network might be bottlenecked at the compression step

3. RECOMMENDATION:
   - For the CURRENT task (autoencoding 1000→100→1000):
     * 5 layers × 100 neurons is reasonable
   - For MORE COMPLEX tasks (vision, language):
     * Try 10-20 layers with skip connections
     * Use narrower layers for more hierarchy
""")
