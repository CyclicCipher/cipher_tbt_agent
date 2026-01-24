"""
400-iteration test with Adam at VERY LOW learning rate.

Previous Adam test (lr=0.001): diverged to 4,139 (4.5x initial)
- Better than Muon but still unstable
- Oscillatory pattern around 4000-5000
- Saturation: 91% → 97%

This test: lr=0.0001 (10x lower)
Hypothesis: Learning rate still too high, causing instability
"""

import sys
sys.path.insert(0, '/home/user/predictive-coding-agent')

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("400-ITERATION TEST: Adam (LOW LR)")
print("=" * 70)

# Create network with LOW LR Adam
network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=100,
    input_size=1000,
    dtype=torch.float32,
    device='cpu',
    inference_lr=0.1,
    use_adam=True,
    adam_lr=0.0001,          # 10x lower than standard
    adam_betas=(0.9, 0.999),
    saturation_penalty=0.01
)

sensory_input = torch.randn(1000)

print(f"\nNetwork configuration:")
print(f"  Optimizer: Adam (lr=0.0001, betas=(0.9, 0.999))")
print(f"  Saturation penalty: 0.01")
print(f"  Learning rate: 10x lower than standard Adam")
print(f"\nRunning 400 training iterations...\n")

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

    if iteration % 20 == 0 or iteration in [0, 5, 10, 15, 285, 320, 395, 399]:
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

print(f"\nError: {initial_error:.2f} → {min_error:.2f} (iter {min_iter}) → {final_error:.2f}")

# Weight and saturation
initial_w_norm = weight_norms[0]
final_w_norm = weight_norms[-1]
w_change = (final_w_norm - initial_w_norm) / initial_w_norm * 100
avg_sat_early = sum(saturation_rates[:50]) / 50
avg_sat_late = sum(saturation_rates[-50:]) / 50

print(f"Weight change: {w_change:+.1f}%")
print(f"Saturation: {avg_sat_early*100:.1f}% → {avg_sat_late*100:.1f}%")

# Oscillations
oscillations = sum(1 for i in range(1, len(errors)) if errors[i] > errors[i-1])
print(f"Oscillations: {oscillations}/400")

# Stability verdict
if final_error > 2 * initial_error:
    verdict = "⚠ DIVERGED"
elif final_error > initial_error:
    verdict = "⚠ UNSTABLE"
elif final_error < 2 * min_error:
    verdict = "✓ STABLE"
else:
    verdict = "△ OSCILLATING"

print(f"\n{verdict}")

# Comparison
print("\n" + "=" * 70)
print("LEARNING RATE COMPARISON")
print("=" * 70)

print(f"""
Adam LR  | Final Error | Divergence | Weight Δ | Saturation
---------|-------------|------------|----------|------------
0.001    |    4,139    |   4.5x     |  +41%    |   97%
0.0001   |   {final_error:6.0f}    |  {final_error/initial_error:5.1f}x     | {w_change:+5.1f}%    |  {avg_sat_late*100:3.0f}%
""")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if final_error < 100 and final_error < 2 * min_error:
    print("\n✓ SUCCESS! Low LR Adam stabilizes learning")
    print("  - Stays near minimum without diverging")
    print("  - Ready for temporal patterns")
elif final_error < 500:
    print("\n△ IMPROVED - Lower LR helps")
    print("  - Less divergence than lr=0.001")
    print("  - May need even lower LR or LR scheduling")
elif final_error < initial_error:
    print("\n△ STILL IMPROVING")
    print("  - Needs more tuning or LR scheduling")
else:
    print("\n⚠ FUNDAMENTAL ISSUE")
    print("  - Problem not just learning rate")
    print("  - Need to investigate:")
    print("    1. Saturation feedback loop (97% saturated)")
    print("    2. Inference dynamics (not reaching equilibrium?)")
    print("    3. Gradient computation (sign error?)")
    print("    4. Weight initialization (too high?)")
