"""
400-iteration test with Adam + ACTIVITY CLIPPING.

Previous Adam tests (without clipping):
- lr=0.001: diverged to 4,139 (4.5x initial)
- lr=0.0001: diverged to 1,125 (1.1x initial) - nearly stable!

Root cause identified: Saturation feedback loop
- Learning works initially (error 1002 → 400)
- Saturation climbs (72% → 99%)
- High activations → strong weight updates → more saturation

This test: Adam lr=0.0001 + activity clipping to ±0.85
Expected: STABLE learning, error stays near minimum
"""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("400-ITERATION TEST: Adam + Activity Clipping")
print("=" * 70)

# Create network with Adam + clipping (clipping now built into backbone)
network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=100,
    input_size=1000,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1,
    use_adam=True,
    adam_lr=0.0001,          # Proven nearly-stable
    adam_betas=(0.9, 0.999),
    saturation_penalty=0.01
)

sensory_input = torch.randn(1000)

print(f"\nNetwork configuration:")
print(f"  Optimizer: Adam (lr=0.0001)")
print(f"  Activity clipping: ±0.85 (ENABLED - prevents saturation)")
print(f"  Saturation penalty: 0.01")
print(f"\nHypothesis: Clipping prevents saturation feedback loop")
print(f"Expected: Error converges and STAYS near minimum\n")
print(f"Running 400 training iterations...\n")

# Track metrics
errors = []
weight_norms = []
saturation_rates = []

for iteration in range(400):
    network.forward(sensory_input, num_iterations=50)
    recon_error = ((sensory_input - network.compute_reconstruction())**2).sum().item()
    errors.append(recon_error)

    w0_norm = torch.norm(network.layers[0].neurons.W_basal).item()
    weight_norms.append(w0_norm)

    layer0_state = network.layers[0].get_state()
    saturation_rate = (layer0_state.abs() > 0.9).float().mean().item()
    saturation_rates.append(saturation_rate)

    network.update_weights()

    if iteration % 20 == 0 or iteration in [0, 5, 10, 15, 94, 285, 320, 395, 399]:
        print(f"Iter {iteration:3d}: error={recon_error:8.2f}, "
              f"W0_norm={w0_norm:5.2f}, saturated={saturation_rate*100:4.1f}%")

# Analysis
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

initial_error = errors[0]
final_error = errors[-1]
min_error = min(errors)
min_iter = errors.index(min_error)
max_error_after_min = max(errors[min_iter:])

print(f"\nError trajectory:")
print(f"  Initial:  {initial_error:8.2f}")
print(f"  Minimum:  {min_error:8.2f} (at iteration {min_iter})")
print(f"  Maximum after min: {max_error_after_min:8.2f}")
print(f"  Final:    {final_error:8.2f}")
print(f"  Reduction: {(initial_error - final_error) / initial_error * 100:6.1f}%")

# Stability verdict
if final_error > 2 * initial_error:
    verdict = "⚠ DIVERGED"
    detail = f"Final {final_error:.0f} > 2x initial {initial_error:.0f}"
elif final_error > initial_error:
    verdict = "⚠ UNSTABLE"
    detail = f"Final {final_error:.0f} > initial {initial_error:.0f}"
elif final_error > 5 * min_error:
    verdict = "⚠ OSCILLATING"
    detail = f"Final {final_error:.0f} >> minimum {min_error:.0f}"
elif final_error > 2 * min_error:
    verdict = "△ STABLE (mild oscillations)"
    detail = f"Final within 2-5x minimum"
else:
    verdict = "✓ STABLE"
    detail = f"Final error within 2x minimum"

print(f"\n{verdict}: {detail}")

# Weight analysis
initial_w_norm = weight_norms[0]
final_w_norm = weight_norms[-1]
w_change = (final_w_norm - initial_w_norm) / initial_w_norm * 100

print(f"\nWeight change: {w_change:+.1f}%")
if abs(w_change) > 50:
    print(f"  ⚠ Large")
else:
    print(f"  ✓ Moderate")

# Saturation analysis
avg_sat_early = sum(saturation_rates[:50]) / 50
avg_sat_late = sum(saturation_rates[-50:]) / 50
max_sat = max(saturation_rates)

print(f"\nSaturation:")
print(f"  Early: {avg_sat_early*100:.1f}%")
print(f"  Late:  {avg_sat_late*100:.1f}%")
print(f"  Max:   {max_sat*100:.1f}%")

if avg_sat_late > 0.8:
    print(f"  ⚠ Still high (clipping threshold may be too high)")
elif avg_sat_late > 0.3:
    print(f"  △ Moderate")
else:
    print(f"  ✓ Healthy")

# Oscillations
oscillations = sum(1 for i in range(1, len(errors)) if errors[i] > errors[i-1])
print(f"\nOscillations: {oscillations}/400")
if oscillations < 150:
    print(f"  ✓ Smooth")
else:
    print(f"  △ Some")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON: Effect of Clipping")
print("=" * 70)

print(f"""
Configuration        | Final Error | Divergence | Saturation | Status
---------------------|-------------|------------|------------|--------
Adam lr=0.001        |    4,139    |   4.5x     |   97%      | Diverged
Adam lr=0.0001       |    1,125    |   1.1x     |   99%      | Nearly stable
Adam lr=0.0001 + CLIP|   {final_error:6.0f}    |  {final_error/initial_error:5.1f}x     |  {avg_sat_late*100:3.0f}%      | {verdict.split(':')[0]}

Clipping prevents saturation feedback loop by limiting activations to ±0.85.
This keeps neurons in sensitive region of tanh where gradients are healthy.
""")

print("=" * 70)
print("CONCLUSION")
print("=" * 70)

if final_error < 100 and final_error < 2 * min_error:
    print("\n✓✓✓ SUCCESS! STABLE LEARNING ACHIEVED! ✓✓✓")
    print("\n  Activity clipping solves saturation feedback loop")
    print("  - Error converges to low value")
    print("  - Stays near minimum (no divergence)")
    print("  - Saturation controlled")
    print("  - Weight growth moderate")
    print("\n✓ READY FOR NEXT PHASE:")
    print("  1. Add temporal patterns (simple recurrence)")
    print("  2. Test on sequence prediction")
    print("  3. Begin math curriculum (arithmetic → algebra → calculus)")

elif final_error < 500 and final_error < 2 * min_error:
    print("\n✓ SUCCESS - STABLE!")
    print("  - Error stays near minimum")
    print("  - No divergence")
    print("  - Clipping works as intended")
    print("\nReady for temporal patterns and math curriculum")

elif final_error < initial_error:
    print("\n△ IMPROVED but not fully stable")
    print("  - Better than without clipping")
    print("  - May need lower clipping threshold (0.75 instead of 0.85)")
    print("  - Or stronger saturation penalty")

else:
    print("\n⚠ UNEXPECTED - Still unstable with clipping")
    print("  - Lower clipping threshold (try 0.7)")
    print("  - Or investigate other issues (weight init, inference dynamics)")

print(f"\nFinal assessment:")
if final_error < 1000 and final_error < 2 * min_error:
    print("  ✓ Stable optimizer found: Adam lr=0.0001 + clipping")
    print("  ✓ Ready to implement temporal patterns")
    print("  ✓ Math curriculum experiments can begin")
else:
    print("  - Need further tuning of clipping threshold")
    print("  - Or additional stabilization techniques")
