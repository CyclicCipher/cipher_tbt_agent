"""
Test StableProspectiveLearning optimizer.

Goal: Achieve stable learning with excellent minimum.

Expected performance:
- Find good minimum (error <10)
- Maintain solution for 400 iterations
- No catastrophic rebound

Comparison to baselines:
- Manual GD: min=3.57, final=55.71 (15.6x rebound) - finds solution but can't maintain
- Adam: min=411, final=858 (2.1x rebound) - stable but poor solution quality

StableProspectiveLearning should combine:
- Manual GD's solution quality (error <10)
- Adam's stability (rebound <2x)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.backbone import BackboneNetwork
import torch

print("=" * 70)
print("STABLE PROSPECTIVE LEARNING OPTIMIZER TEST")
print("=" * 70)

# Test different LR schedules
schedules = ["cosine", "linear", "exponential"]

for schedule in schedules:
    print(f"\n{'='*70}")
    print(f"Testing LR schedule: {schedule}")
    print(f"{'='*70}\n")

    # Create network with StableProspectiveLearning
    network = BackboneNetwork(
        num_layers=5,
        neurons_per_layer=100,
        input_size=1000,
        dtype=torch.float32,
        device='cpu',
        inference_lr=0.1,
        use_stable=True,
        stable_lr=0.001,              # Initial LR
        stable_max_iterations=400,
        stable_lr_schedule=schedule,  # Test different schedules
        stable_decay_strong=0.01,     # Strong decay far from solution
        stable_decay_weak=0.001,      # Weak decay near solution (10x weaker!)
    )

    sensory_input = torch.randn(1000)
    torch.manual_seed(42)  # Consistent initialization

    errors = []
    weight_norms = []
    lrs = []
    decays = []

    for iteration in range(400):
        network.forward(sensory_input, num_iterations=50)
        recon_error = ((sensory_input - network.compute_reconstruction())**2).sum().item()
        errors.append(recon_error)

        w0_norm = torch.norm(network.layers[0].neurons.W_basal).item()
        weight_norms.append(w0_norm)

        # Track optimizer stats
        stats = network.optimizer.get_stats()
        lrs.append(stats['current_lr'])
        decays.append(stats['current_decay'])

        network.update_weights()

        if iteration % 50 == 0 or iteration in [0, 399]:
            print(f"Iter {iteration:3d}: error={recon_error:6.2f}, "
                  f"W_norm={w0_norm:5.2f}, lr={stats['current_lr']:.6f}, "
                  f"decay={stats['current_decay']:.4f}")

    # Analysis
    min_error = min(errors)
    min_iter = errors.index(min_error)
    final_error = errors[-1]
    rebound_factor = final_error / min_error

    print(f"\nResults:")
    print(f"  Minimum: {min_error:.2f} at iteration {min_iter}")
    print(f"  Final: {final_error:.2f}")
    print(f"  Rebound factor: {rebound_factor:.2f}x")

    # Weight analysis
    initial_w_norm = weight_norms[0]
    min_w_norm = weight_norms[min_iter]
    final_w_norm = weight_norms[-1]

    print(f"\nWeight norms:")
    print(f"  Initial: {initial_w_norm:.2f}")
    print(f"  At minimum: {min_w_norm:.2f}")
    print(f"  Final: {final_w_norm:.2f}")

    # Check if decay adaptation worked
    early_decays = sum(decays[:min_iter]) / max(1, min_iter)
    late_decays = sum(decays[min_iter:]) / max(1, len(decays) - min_iter)

    print(f"\nWeight decay adaptation:")
    print(f"  Before minimum: {early_decays:.4f}")
    print(f"  After minimum: {late_decays:.4f}")

    if late_decays < early_decays * 0.5:
        print(f"  ✓ Adaptive decay working (reduced after finding solution)")
    else:
        print(f"  ⚠ Decay not adapting as expected")

    # Assessment
    if min_error < 10 and rebound_factor < 2.0:
        print(f"\n✓✓✓ SUCCESS! Excellent minimum AND stability")
    elif min_error < 50 and rebound_factor < 3.0:
        print(f"\n✓ GOOD - Better than baselines")
    elif rebound_factor < 5.0:
        print(f"\n△ IMPROVED - More stable than manual GD")
    else:
        print(f"\n⚠ NOT GOOD ENOUGH - Still rebounding too much")

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print("""
Baseline performance:
  Manual GD (decay=0.01): min=3.57, final=55.71 (15.6x rebound)
  Adam (lr=0.0001):       min=411, final=858 (2.1x rebound)

StableProspectiveLearning should achieve:
  Target: min<10, final<20 (<2x rebound)

This combines:
  - Manual GD's solution quality (excellent minimum)
  - Adaptive decay to prevent solution destruction
  - LR scheduling to reduce late-stage drift
""")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)

print("""
If StableProspectiveLearning achieves target (<10 error, <2x rebound):
  1. ✓ Stable optimizer foundation proven
  2. Begin implementing temporal patterns (simple recurrence)
  3. Test on sequence prediction tasks
  4. Design math curriculum experiments

If not meeting target:
  1. Tune hyperparameters (decay weak, LR schedule)
  2. Add early stopping variant
  3. Investigate other stabilization techniques
""")
