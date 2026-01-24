"""
400-iteration test with Adam optimizer.

After Muon experiments showed persistent divergence across all momentum settings,
trying Adam optimizer as recommended in muon_optimizer_analysis.md (Priority 1).

Adam advantages for predictive coding:
1. Per-parameter adaptive learning rates (apical vs basal)
2. Gradient normalization (prevents overshoot)
3. Momentum in weight space (better for local learning rules)
4. Proven reliability across diverse architectures
"""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("400-ITERATION TEST: Adam Optimizer")
print("=" * 70)

# Create network with Adam optimizer
network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=100,
    input_size=1000,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1,
    use_adam=True,           # Use Adam
    adam_lr=0.001,           # Standard Adam LR (higher than manual GD)
    adam_betas=(0.9, 0.999), # Standard beta values
    saturation_penalty=0.01
)

sensory_input = torch.randn(1000)

print(f"\nNetwork configuration:")
print(f"  Optimizer: Adam (lr={0.001}, betas=(0.9, 0.999))")
print(f"  Saturation penalty: 0.01")
print(f"  Inference iterations: 50")
print(f"\nHypothesis: Adam's per-parameter adaptive LR prevents overshoot")
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
    if iteration % 20 == 0 or iteration in [0, 5, 10, 15, 30, 40, 285, 320, 395, 399]:
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
    detail = f"Final error {final_error:.1f} > 2x initial {initial_error:.1f}"
elif final_error > initial_error:
    stability = "⚠ UNSTABLE"
    detail = f"Final error {final_error:.1f} > initial {initial_error:.1f}"
elif final_error > 5 * min_error:
    stability = "⚠ OSCILLATING"
    detail = f"Final {final_error:.1f} >> 5x minimum {min_error:.1f}"
elif final_error > 2 * min_error:
    stability = "△ STABLE (mild oscillations)"
    detail = f"Final within 2-5x of minimum"
else:
    stability = "✓ STABLE"
    detail = f"Final error within 2x of minimum"

print(f"\n{stability}: {detail}")

# Weight analysis
initial_w_norm = weight_norms[0]
final_w_norm = weight_norms[-1]
w_change = (final_w_norm - initial_w_norm) / initial_w_norm * 100

print(f"\nWeight norm change: {w_change:+.1f}%")
if abs(w_change) > 50:
    print(f"  ⚠ Large weight change")
else:
    print(f"  ✓ Moderate weight change")

# Saturation analysis
avg_saturation_early = sum(saturation_rates[:50]) / 50
avg_saturation_late = sum(saturation_rates[-50:]) / 50

print(f"\nSaturation: {avg_saturation_early*100:.1f}% → {avg_saturation_late*100:.1f}%")
if avg_saturation_late > 0.5:
    print(f"  ⚠ High saturation")
elif avg_saturation_late > 0.3:
    print(f"  △ Moderate saturation")
else:
    print(f"  ✓ Healthy saturation")

# Oscillation count
oscillations = sum(1 for i in range(1, len(errors)) if errors[i] > errors[i-1])
print(f"\nOscillations: {oscillations}/400")
if oscillations < 150:
    print(f"  ✓ Smooth learning")
elif oscillations < 250:
    print(f"  △ Some oscillations")
else:
    print(f"  ⚠ Highly oscillatory")

# Comparison to Muon
print("\n" + "=" * 70)
print("COMPARISON: Adam vs Muon vs Manual GD")
print("=" * 70)

print(f"""
MUON EXPERIMENTS (all diverged):
  momentum=0.95: error 947 → 13,369 (14.1x divergence)
  momentum=0.70: error 870 → 6,749  (7.8x divergence)
  momentum=0.50: error 990 → 6,867  (6.9x divergence)

  Pattern: All find minimum ~95 at iter 30-40, then diverge
  Issue: Momentum overshoot + saturation feedback loop

MANUAL GD (baseline, decay=0.01):
  error 920 → 0.16 (at iter 320) → 78 (drift)
  Result: Wild oscillations but eventually finds solution
  Issue: Unreliable, can't escape oscillations

ADAM (this test):
  Initial: {initial_error:.2f}
  Minimum: {min_error:.2f} (iter {min_iter})
  Final:   {final_error:.2f}

  Weight change: {w_change:+.1f}%
  Saturation: {avg_saturation_early*100:.1f}% → {avg_saturation_late*100:.1f}%
  Oscillations: {oscillations}/400
""")

# Conclusion
print("=" * 70)
print("CONCLUSION")
print("=" * 70)

if final_error < 100 and final_error < 2 * min_error:
    print("\n✓ SUCCESS! Adam provides stable long-term learning")
    print("  - Converged to low error without overshooting")
    print("  - Per-parameter adaptive LR prevents divergence")
    print("  - Ready for temporal patterns and math curriculum")
    print("\nRECOMMENDATION: Use Adam as default optimizer")
    print("  - Proven reliable for predictive coding architecture")
    print("  - Handles two-compartment neurons effectively")

elif final_error < 500 and final_error < initial_error:
    print("\n△ IMPROVED - Adam better than Muon")
    print("  - No catastrophic divergence")
    print("  - Some oscillations remain")
    print("  - May benefit from LR scheduling for perfect stability")
    print("\nRECOMMENDATION: Proceed with Adam, add LR scheduling if needed")

elif final_error < initial_error:
    print("\n△ PARTIAL SUCCESS")
    print("  - Better than Muon but still unstable")
    print("  - Try: Lower LR (0.0005), LR warmup, or gradient clipping")

else:
    print("\n⚠ UNEXPECTED - Adam also diverging")
    print("  - Architecture may need fundamental changes")
    print("  - Check: Weight initialization, inference dynamics, gradient computation")

print("\nNext steps:")
if final_error < 100:
    print("  1. ✓ Stable optimizer found (Adam)")
    print("  2. Add simple recurrence for temporal patterns")
    print("  3. Test on sequence prediction")
    print("  4. Begin math curriculum (arithmetic → algebra → calculus)")
else:
    print("  1. Tune Adam hyperparameters (lower LR, add scheduling)")
    print("  2. If still unstable, investigate architecture/inference dynamics")
    print("  3. Consider stronger activity regularization")
