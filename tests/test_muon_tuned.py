"""
400-iteration test with TUNED Muon hyperparameters.

Previous run (momentum=0.95, saturation_penalty=0.0):
- Found good minimum (error=133 at iter 40)
- But momentum overshoots → diverged to 13,369
- Saturation climbed to 98%

This run: Lower momentum + enable saturation penalty
"""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("400-ITERATION TEST: Muon (TUNED)")
print("=" * 70)

# Create network with TUNED Muon hyperparameters
network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=100,
    input_size=1000,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1,
    use_muon=True,
    muon_lr=0.0005,          # Keep same LR
    muon_momentum=0.7,       # TUNED: Lower from 0.95 to prevent overshooting
    saturation_penalty=0.01, # ENABLED: Prevent 98% saturation
    activity_target=0.3
)

sensory_input = torch.randn(1000)

print(f"\nNetwork configuration:")
print(f"  Optimizer: Muon (lr={0.0005}, momentum={0.7})")
print(f"  Saturation penalty: {0.01} (ENABLED)")
print(f"  Inference iterations: 50")
print(f"\nChanges from previous run:")
print(f"  - Momentum: 0.95 → 0.7 (prevent overshoot)")
print(f"  - Saturation penalty: 0.0 → 0.01 (prevent seizure)")
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
    if iteration % 20 == 0 or iteration in [0, 5, 10, 15, 40, 47, 285, 320, 395, 399]:
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
elif final_error > 5 * min_error:
    print(f"\n⚠ OSCILLATING: Final {final_error:.1f} >> 5x minimum {min_error:.1f}")
elif final_error > 2 * min_error:
    print(f"\n△ STABLE but oscillating: Final {final_error:.1f} = 2-5x minimum {min_error:.1f}")
else:
    print(f"\n✓ STABLE: Final error within 2x of minimum")

# Weight analysis
initial_w_norm = weight_norms[0]
final_w_norm = weight_norms[-1]
w_change = (final_w_norm - initial_w_norm) / initial_w_norm * 100

print(f"\nWeight norm (Layer 0 W_basal):")
print(f"  Initial: {initial_w_norm:5.2f}")
print(f"  Final:   {final_w_norm:5.2f}")
print(f"  Change:  {w_change:+6.1f}%")

if abs(w_change) > 50:
    print(f"  ⚠ Large weight change")
else:
    print(f"  ✓ Moderate weight change")

# Saturation analysis
avg_saturation_early = sum(saturation_rates[:50]) / 50
avg_saturation_late = sum(saturation_rates[-50:]) / 50

print(f"\nSaturation rate (Layer 0):")
print(f"  Early (iters 0-50):   {avg_saturation_early*100:4.1f}%")
print(f"  Late  (iters 350-400): {avg_saturation_late*100:4.1f}%")

if avg_saturation_late > 0.5:
    print(f"  ⚠ High saturation (seizure-like)")
elif avg_saturation_late > 0.3:
    print(f"  △ Moderate saturation")
elif avg_saturation_late < 0.01:
    print(f"  ⚠ Very low saturation (may be too constrained)")
else:
    print(f"  ✓ Healthy saturation level")

# Oscillation count
oscillations = sum(1 for i in range(1, len(errors)) if errors[i] > errors[i-1])
print(f"\nOscillations (error increased): {oscillations}/400")

if oscillations < 150:
    print(f"  ✓ Smooth learning trajectory")
elif oscillations < 250:
    print(f"  △ Some oscillations but acceptable")
else:
    print(f"  ⚠ Highly oscillatory")

# Compare to previous Muon run
print("\n" + "=" * 70)
print("COMPARISON TO PREVIOUS MUON RUN")
print("=" * 70)

print("""
Previous Muon run (momentum=0.95, penalty=0.0):
  Iter 0:   error=947.89
  Iter 40:  error=133.81  (found minimum)
  Iter 100: error=2004.20 (diverging)
  Iter 399: error=13369.45 (DIVERGED)

  Saturation: 70% → 98%
  Weight growth: +100%

This run (momentum=0.7, penalty=0.01):
""")

print(f"  Iter 0:   error={errors[0]:.2f}")
if len(errors) > 40:
    print(f"  Iter 40:  error={errors[40]:.2f}")
if len(errors) > 100:
    print(f"  Iter 100: error={errors[100]:.2f}")
if len(errors) > 399:
    print(f"  Iter 399: error={errors[399]:.2f}")

print(f"\n  Saturation: {avg_saturation_early*100:.1f}% → {avg_saturation_late*100:.1f}%")
print(f"  Weight growth: {w_change:+.1f}%")

# Conclusion
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if final_error < 100 and final_error < 2 * min_error:
    print("\n✓ SUCCESS! Tuned Muon provides stable learning")
    print("  - Error converged to low value")
    print("  - Stays near minimum (not overshooting)")
    print("  - Ready for temporal patterns and math curriculum")
elif min_error < 50 and final_error < initial_error and final_error < 5 * min_error:
    print("\n△ IMPROVED - Better than previous run")
    print("  - Finds good minimum")
    print("  - Less overshooting than momentum=0.95")
    print("  - May benefit from LR scheduling for further stability")
else:
    print("\n⚠ STILL UNSTABLE - Need further tuning")
    print("  - Consider: Lower momentum (0.5), LR scheduling, gradient clipping")
    print("  - Or try different optimizer (Adam, RMSprop)")

print("\nNext steps:")
if final_error < 100 and final_error < 2 * min_error:
    print("  1. Add simple recurrence for temporal patterns")
    print("  2. Test on sequence prediction task")
    print("  3. Start math curriculum (arithmetic → algebra → calculus)")
else:
    print("  1. Further hyperparameter tuning (momentum=0.5, LR scheduling)")
    print("  2. Or try LR warmup + cosine decay")
    print("  3. If still fails, try Adam optimizer as fallback")
