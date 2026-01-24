"""
400-iteration test with VERY LOW momentum Muon.

Previous runs:
- momentum=0.95: diverged to 13,369
- momentum=0.7:  diverged to 6,749 (better!)

This run: momentum=0.5 (minimal momentum, prioritize stability)
"""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("400-ITERATION TEST: Muon (LOW MOMENTUM)")
print("=" * 70)

# Create network with LOW momentum
network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=100,
    input_size=1000,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1,
    use_muon=True,
    muon_lr=0.0005,          # Same LR
    muon_momentum=0.5,       # VERY LOW: 0.5 (prioritize stability)
    saturation_penalty=0.01,
    activity_target=0.3
)

sensory_input = torch.randn(1000)

print(f"\nNetwork configuration:")
print(f"  Optimizer: Muon (lr={0.0005}, momentum={0.5})")
print(f"  Saturation penalty: 0.01")
print(f"  Inference iterations: 50")
print(f"\nHypothesis: Lower momentum → less overshoot → stable learning")
print(f"\nRunning 400 training iterations...\n")

# Track metrics
errors = []
weight_norms = []
saturation_rates = []

for iteration in range(400):
    # Run inference
    network.forward(sensory_input, num_iterations=50)

    # Compute errors
    recon_error = ((sensory_input - network.compute_reconstruction())**2).sum().item()
    errors.append(recon_error)

    # Track weights and activations
    w0_norm = torch.norm(network.layers[0].neurons.W_basal).item()
    weight_norms.append(w0_norm)

    layer0_state = network.layers[0].get_state()
    saturation_rate = (layer0_state.abs() > 0.9).float().mean().item()
    saturation_rates.append(saturation_rate)

    # Update weights
    network.update_weights()

    # Print progress
    if iteration % 20 == 0 or iteration in [0, 5, 10, 15, 34, 40, 285, 320, 395, 399]:
        print(f"Iter {iteration:3d}: error={recon_error:8.2f}, "
              f"W0_norm={w0_norm:5.2f}, saturated={saturation_rate*100:4.1f}%")

# Analysis
print("\n" + "=" * 70)
print("RESULTS ANALYSIS")
print("=" * 70)

initial_error = errors[0]
final_error = errors[-1]
min_error = min(errors)
min_iter = errors.index(min_error)

print(f"\nError trajectory:")
print(f"  Initial:  {initial_error:8.2f}")
print(f"  Minimum:  {min_error:8.2f} (at iteration {min_iter})")
print(f"  Final:    {final_error:8.2f}")
print(f"  Reduction: {(initial_error - final_error) / initial_error * 100:6.1f}%")

# Stability check
if final_error > 2 * initial_error:
    stability = "⚠ DIVERGED"
elif final_error > initial_error:
    stability = "⚠ UNSTABLE"
elif final_error > 5 * min_error:
    stability = "⚠ OSCILLATING"
elif final_error > 2 * min_error:
    stability = "△ STABLE (mild oscillations)"
else:
    stability = "✓ STABLE"

print(f"\n{stability}")

# Weight analysis
initial_w_norm = weight_norms[0]
final_w_norm = weight_norms[-1]
w_change = (final_w_norm - initial_w_norm) / initial_w_norm * 100

print(f"\nWeight norm change: {w_change:+.1f}%")

# Saturation analysis
avg_saturation_early = sum(saturation_rates[:50]) / 50
avg_saturation_late = sum(saturation_rates[-50:]) / 50

print(f"Saturation: {avg_saturation_early*100:.1f}% → {avg_saturation_late*100:.1f}%")

# Comparison table
print("\n" + "=" * 70)
print("MOMENTUM COMPARISON")
print("=" * 70)

print(f"""
Momentum | Final Error | Min Error | Weight Δ | Saturation
---------|-------------|-----------|----------|------------
0.95     |   13,369    |    93     |  +100%   |   98%
0.70     |    6,749    |    95     |   +76%   |   98%
0.50     |   {final_error:6.0f}    |   {min_error:3.0f}     |  {w_change:+5.1f}%   |  {avg_saturation_late*100:4.0f}%

Pattern: Lower momentum → less divergence
""")

# Conclusion
print("=" * 70)
print("CONCLUSION")
print("=" * 70)

if final_error < 100 and final_error < 2 * min_error:
    print("\n✓ SUCCESS! Low momentum (0.5) stabilizes Muon")
    print("  - Stays near minimum without overshooting")
    print("  - Ready for temporal patterns implementation")
    print("\nRECOMMENDATION: Use momentum=0.5 as default for this architecture")
elif final_error < 500 and final_error < 5 * min_error:
    print("\n△ IMPROVED but not fully stable")
    print("  - Better than higher momentum")
    print("  - Consider: momentum=0.3 or learning rate scheduling")
elif final_error < initial_error:
    print("\n△ PARTIAL SUCCESS")
    print("  - Finds good minimum but doesn't stay")
    print("  - Muon may not be ideal for this architecture")
    print("\nRECOMMENDATION: Try Adam optimizer instead")
    print("  - Adam has per-parameter adaptive learning rates")
    print("  - May handle two-compartment neurons better")
else:
    print("\n✗ STILL DIVERGING")
    print("  - Muon not suitable for this architecture")
    print("\nRECOMMENDATION: Switch to Adam optimizer")
    print("  - Well-tested, handles complex architectures")
    print("  - Adaptive learning rates prevent overshooting")

print("\nNext steps:")
if final_error < 100:
    print("  1. Commit successful Muon configuration")
    print("  2. Add simple recurrence for temporal patterns")
    print("  3. Test on math curriculum")
else:
    print("  1. Implement Adam optimizer as alternative")
    print("  2. Run 400-iter test with Adam")
    print("  3. If stable: proceed with temporal patterns")
