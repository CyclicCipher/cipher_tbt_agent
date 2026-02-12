# ePC-Mamba3 Phase Transition Analysis

## The Phenomenon

ePC-Mamba3 exhibits a **phase transition** during training: accuracy plateaus at 10-15%
for 19-28 epochs (varying by initialization), then explosively converges to 99%+ in ~8
epochs. Backprop Mamba3 shows smooth, monotonic improvement instead.

Results (copy task, 4-layer Mamba3, d_model=128):
- ePC seed=0: plateau 7-17% for 19 epochs → 99.03% at epoch 27
- ePC no seed (run 1): plateau for 28 epochs → 99.3% at epoch 44
- ePC no seed (run 2): plateau for ~28 epochs → 99.2% at epoch 36
- Backprop seed=42: smooth 7% → 38% → 99.2% at epoch 38

## Root Cause: Circular Dependency

ePC has two coupled optimization systems:

1. **Newton (error optimization)**: finds errors `e_i` that minimize energy
   `E = 0.5 * Σ(pi * ||e_i||²) + CE(output, targets)`
   Effectiveness depends on the **Jacobian** `∂output/∂e` — how output changes with errors.

2. **Adam (weight optimization)**: updates weights using `E_local`, where each block's
   gradient is proportional to its **error magnitude** `||e_i||`.

These form a circular dependency:

```
Newton effectiveness ──→ error magnitude ──→ E_local gradient quality
        ↑                                              │
        └──── Jacobian structure ←── weight updates ◄──┘
```

## The Plateau (Deadlock Phase)

During the plateau (epochs 1-19 for seed=0):

- **Errors are tiny** (||e|| ≈ 10⁻⁶ to 10⁻⁴) — Newton finds only microscopic corrections
- **Newton convergence ≈ 1.0** — barely reduces energy (E_init ≈ 2718, E_final ≈ 2717)
- **E_local gradients for block weights are proportional to ||e_i||** — essentially zero
- **The Jacobian is unstructured** with random weights — Newton's rank-1 approximation
  captures little of the true curvature

The system is deadlocked: Newton can't produce large errors because the Jacobian is bad,
and the Jacobian can't improve because the errors are too small to drive weight learning.

### CE-driven weight changes (the escape mechanism)

E_local has two components:

```python
E_local = Σ(pi * 0.5 * ||s_pred_i - (s_pred_i + e_i).detach()||²)  # local MSE per block
        + CE(output_proj(norm(s_N.detach())), targets)                # output CE loss
```

The local MSE terms provide gradients proportional to ||e_i|| → near zero during plateau.

But the **CE output loss** provides gradients to `output_proj` (weight-tied with
`embedding`) regardless of error magnitude. This is equivalent to backprop through just
the output projection layer — like training a linear probe on frozen features.

During the plateau, CE-driven updates to the embedding/output projection:
1. Adjust the output mapping to better exploit whatever features the blocks provide
2. Change the embedding layer, which changes the INPUT to all blocks
3. This indirectly reshapes the Jacobian `∂output/∂e`, because the Jacobian depends on
   the full forward pass, and the forward pass starts from the (changing) embeddings

The tiny MSE-driven changes to block weights also contribute, but much more slowly.

## The Phase Transition (Symmetry Breaking)

At some critical point, CE-driven changes accumulate enough that the Jacobian becomes
structured enough for Newton to exploit:

1. Newton finds slightly larger errors (||e|| jumps from 10⁻⁴ to 10⁻²)
2. Larger errors → larger E_local gradients → faster block weight learning
3. Faster weight learning → better Jacobian → Newton finds even larger errors
4. **Positive feedback loop** — explosive convergence

The Per-Layer Energies chart shows all four layers exploding simultaneously — the feedback
loop engages all layers at once. This is a classic **symmetry-breaking** event.

## Why Backprop Doesn't Have This

Backprop computes gradients through the **full computational graph** via chain rule.
The gradient for layer `i` flows directly from the loss through all subsequent layers.
There's no intermediate "error optimization" step that depends on Jacobian quality.

```
Backprop: loss → chain rule → gradients for ALL layers → weight updates (direct, smooth)
ePC:      loss → Newton (limited by Jacobian) → errors → E_local → weight updates (indirect, circular)
```

Backprop's gradient magnitude is always proportional to the loss. As long as there's any
loss signal, every layer receives gradient information. No circular dependency, no deadlock,
no phase transition.

## Post-Transition Advantages

Once the transition occurs, ePC has properties backprop lacks:

1. **Local credit assignment**: Each layer learns from its own prediction error
   `||s_pred_i - s_actual_i||`, not from a gradient traversing the entire network
2. **No vanishing gradients**: Layer 1's learning signal doesn't pass through layers 2-4
3. **Reduced interference**: Layer changes don't propagate through the gradient path to
   disrupt other layers (Song et al. 2024, prospective configuration)
4. **Fast post-transition convergence**: 15% → 99% in just 8 epochs, vs backprop taking
   ~30 epochs for the same range

## Initialization Sensitivity

Phase transition timing depends on how quickly CE-driven changes make the Jacobian
exploitable. Different random initializations produce different Jacobian structures:
- seed=0: transition at epoch 19 (27 total to 99%)
- no seed (run 1): transition at epoch 28 (44 total to 99%)
- no seed (run 2): transition at epoch ~28 (36 total to 99%)
- seed=42: still in plateau at epoch 10 (would likely transition around epoch 25-30)

## Open Questions

1. **Can the plateau be shortened without disrupting the transition?** All attempts
   (init_scale, mHC, muPC, adaptive damping) failed — they either killed the transition
   or made things worse. The plateau may be a necessary "incubation period."

2. **Does the phase transition generalize to harder tasks?** Copy task is relatively
   simple. Longer sequences, more complex patterns, and real-world data may behave
   differently.

3. **Can the Jacobian be initialized in a "structured" state?** If we understood what
   makes a Jacobian "exploitable" by rank-1 Newton, we could initialize weights
   accordingly and eliminate the plateau entirely.

## References

- Goemaere et al. 2025, arXiv:2505.20137 — ePC framework
- Salvatori et al. 2025, arXiv:2506.23800 — precision weighting for deeper PC
- Song et al. 2024, Nature Neuroscience — prospective configuration reduces interference
- Innocenti et al. 2025, arXiv:2505.13124 — muPC (Depth-muP for PC)
