"""
400-iteration test with Muon optimizer and activity regularization.

Tests:
1. Muon optimizer (momentum in neuron space)
2. Saturation penalty (prevent Layer 0 seizure)
3. Long-term stability (400 iterations without divergence)
"""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("400-ITERATION TEST: Muon + Activity Regularization")
print("=" * 70)

# Create network with Muon optimizer
network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=100,
    input_size=1000,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1,
    use_muon=True,  # KEY: Enable Muon optimizer
    muon_lr=0.0005,  # Much lower LR (same as manual GD for fair comparison)
    muon_momentum=0.95,  # High momentum for stability
    saturation_penalty=0.0,  # Disable for now (test Muon alone first)
    activity_target=0.3  # Target activation level
)

sensory_input = torch.randn(1000)

print(f"\nNetwork configuration:")
print(f"  Optimizer: Muon (lr={0.0005}, momentum={0.95})")
print(f"  Saturation penalty: {0.0} (disabled for baseline)")
print(f"  Inference iterations: 50")
print(f"\nRunning 400 training iterations...\n")

# Track metrics
errors = []
weight_norms = []
saturation_rates = []

for iteration in range(400):
    # Run inference
    network.forward(sensory_input, num_iterations=50)

    # Compute errors
    total_error = network.compute_total_error()
    recon_error = ((sensory_input - network.compute_reconstruction())**2).sum().item()

    errors.append(recon_error)

    # Track weights and activations
    w0_norm = torch.norm(network.layers[0].neurons.W_basal).item()
    weight_norms.append(w0_norm)

    # Check saturation in Layer 0
    layer0_state = network.layers[0].get_state()
    saturation_rate = (layer0_state.abs() > 0.9).float().mean().item()
    saturation_rates.append(saturation_rate)

    # Update weights using Muon
    network.update_weights()

    # Print progress
    if iteration % 20 == 0 or iteration in [0, 5, 10, 15, 285, 320, 395, 399]:
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

# Check for divergence
if final_error > 2 * initial_error:
    print(f"\n⚠ DIVERGED: Final error {final_error:.1f} > 2x initial {initial_error:.1f}")
elif final_error > initial_error:
    print(f"\n⚠ UNSTABLE: Final error {final_error:.1f} > initial {initial_error:.1f}")
elif final_error < 0.5 * min_error:
    print(f"\n⚠ OSCILLATING: Final {final_error:.1f} >> minimum {min_error:.1f}")
else:
    print(f"\n✓ STABLE: Error stayed within reasonable range")

# Weight analysis
initial_w_norm = weight_norms[0]
final_w_norm = weight_norms[-1]
w_change = (final_w_norm - initial_w_norm) / initial_w_norm * 100

print(f"\nWeight norm (Layer 0 W_basal):")
print(f"  Initial: {initial_w_norm:5.2f}")
print(f"  Final:   {final_w_norm:5.2f}")
print(f"  Change:  {w_change:+6.1f}%")

if abs(w_change) > 50:
    print(f"  ⚠ Large weight change (>{abs(w_change):.0f}%)")
else:
    print(f"  ✓ Moderate weight change")

# Saturation analysis
avg_saturation_early = sum(saturation_rates[:50]) / 50
avg_saturation_late = sum(saturation_rates[-50:]) / 50

print(f"\nSaturation rate (Layer 0):")
print(f"  Early (iters 0-50):   {avg_saturation_early*100:4.1f}%")
print(f"  Late  (iters 350-400): {avg_saturation_late*100:4.1f}%")

if avg_saturation_late > 0.3:
    print(f"  ⚠ High saturation (seizure-like)")
elif avg_saturation_late < 0.01:
    print(f"  ⚠ Very low saturation (may be too constrained)")
else:
    print(f"  ✓ Healthy saturation level")

# Compare to user's previous run (without Muon)
print("\n" + "=" * 70)
print("COMPARISON TO PREVIOUS RUN (without Muon)")
print("=" * 70)

print("""
User's previous 400-iter run (manual GD, decay=0.01):
  Iter 0:   error=920.9
  Iter 30:  error=61.3   (first valley)
  Iter 100: error=31.4   (deeper)
  Iter 190: error=457.6  (climbs back - UNSTABLE!)
  Iter 285: error=1.5    (finds minimum again)
  Iter 320: error=0.16   (best)
  Iter 400: error=78.4   (drifting)

  Result: Wild oscillations, eventually found good solution but unreliable

Muon run (this test):
""")

print(f"  Iter 0:   error={errors[0]:.2f}")
print(f"  Iter 30:  error={errors[30]:.2f}")
print(f"  Iter 100: error={errors[100]:.2f}")
print(f"  Iter 190: error={errors[190]:.2f}")
print(f"  Iter 285: error={errors[285]:.2f}")
print(f"  Iter 320: error={errors[320]:.2f}")
print(f"  Iter 399: error={errors[399]:.2f}")

# Oscillation count
oscillations = sum(1 for i in range(1, len(errors)) if errors[i] > errors[i-1])
print(f"\n  Oscillations (error increased): {oscillations}/400")

if oscillations < 150:
    print(f"  ✓ Much smoother than manual GD (typically >300 oscillations)")
else:
    print(f"  ⚠ Still oscillatory (may need tuning)")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if final_error < 100 and final_error < 2 * min_error:
    print("\n✓ SUCCESS! Muon provides stable long-term learning")
    print("  - Error converged to low value")
    print("  - No catastrophic divergence")
    print("  - Ready for math curriculum experiments")
elif min_error < 50 and final_error < initial_error:
    print("\n△ PARTIAL SUCCESS - Better than manual GD but still oscillates")
    print("  - Reached good minimum but didn't stay there")
    print("  - May need: lower LR, higher momentum, or LR scheduling")
else:
    print("\n✗ UNSTABLE - Muon alone not sufficient")
    print("  - Need additional stabilization (LR scheduling, gradient clipping)")
    print("  - Or try different hyperparameters")

print("\nNext steps:")
print("  1. If stable: Add simple recurrence for temporal patterns")
print("  2. If unstable: Tune Muon hyperparameters or add LR scheduling")
print("  3. Then: Start math curriculum (arithmetic → algebra → calculus)")
