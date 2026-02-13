# Continuation Notes for Next Session

**Date:** 2026-02-13
**Branch:** `claude/investigate-jacobian-mamba3-Kufd1`

---

## The Central Unsolved Problem: Flat Energy Landscape

With `e_lr=0.1, T=20`, the error phase converges in 3 iterations (early stopping minimum) with `E_init ≈ E_final` (convergence ≈ 0.00). The energy landscape is essentially flat — errors find no meaningful direction to push.

**This was NOT addressed in the last session.** We went down a wrong path thinking the paper's hyperparameters (e_lr=0.001, T=5) would help, but they made things catastrophically worse (Mamba3 dropped to 7%). The real question remains: **why is the energy landscape flat at our working hyperparameters?**

### What "flat" means concretely

The energy `E = ½Σ mean(ε_i²) + L(output)` barely changes when errors move. This means the output loss gradient w.r.t. errors `∂L/∂ε_i` is very small — the errors have almost no influence on the output. The Jacobian `∂ŷ/∂ε_i` is the chain of Jacobians through layers i→i+1→...→output. If this chain has small singular values, errors can't steer the output, so the energy is flat w.r.t. errors.

### Hypotheses for why the Jacobian ∂ŷ/∂ε is small

1. **RMSNorm attenuation:** The `out_norm` (RMSNorm) sits between the last error and the output projection. RMSNorm normalizes the magnitude of the hidden state. Small perturbations (errors) get divided by the norm of the much-larger hidden state, shrinking their effect.

2. **Residual stream dominance:** Each Mamba3Block has `x + Mixer(norm(x)); x + MLP(norm(x))`. The residual `x` dominates, and `ε_i` is a small perturbation to the FULL residual stream. The error signal gets diluted by the residual.

3. **Mamba3 internal saturation:** The SSD scan, SwiGLU activations, or other nonlinearities might be operating in regions where the Jacobian is near-zero (saturated gates, etc.).

4. **Mean reduction mismatch in scale:** With mean reduction, `½ mean(ε²)` for a (batch=32, seq=64, d=128) tensor divides by 32×64×128 = 262,144. The error penalty coefficient is effectively 1/262144, which may be too weak relative to the output loss to produce meaningful error equilibria.

### Possible investigations

- **Measure the Jacobian directly:** Compute `∂ŷ/∂ε_i` for each layer using `torch.autograd.functional.jacobian` or finite differences. Check singular value spectrum. If it's near-zero, the architecture is the problem.

- **Check error gradients:** After one E.backward(), inspect `ε_i.grad` for each layer. If the output loss contribution `(∂ŷ/∂ε_i)^T ∇_ŷ L` is tiny compared to the penalty `ε_i`, the errors are dominated by the penalty.

- **Try without RMSNorm:** Temporarily remove `out_norm` and see if the energy landscape becomes less flat.

- **Try errors in different positions:** Instead of after each block's residual, try errors inside the block (after mixer, before MLP) or before the norm layers.

---

## What Works and What Doesn't

### Working (keep these)
- `e_lr=0.1, T=20` with SGD for errors, Adam for weights
- N errors for N layers (complete coverage)
- Next-step prediction (causal Mamba)
- Mean reduction everywhere
- No precision weighting (uniform ½||ε||²)
- PoPE positional embeddings
- Mamba3 copy task reaches 99%+ in ~10 epochs
- ePC-JEPA Stage 1b reaches ~97% in 5 epochs

### Broken
- Paper's `e_lr=0.001, T=5` → zero learning for Mamba3
- Geometric precision weighting → suppresses deep-layer errors
- Newton/CG error optimization → pathological phase transition
- iPC (interleaved) → prevents error convergence

### Unsolved
- **Stage 2 generalization:** 99% train, 25% test (multi-rule collapse hypothesis)
- **Flat energy landscape:** E_init ≈ E_final at working hyperparameters
- **Oracle z ignored:** Predictor doesn't condition on latent z
- **Langevin hurts:** Energy minimization over z gives -5% (negative gap)

---

## Key Files to Read

| File | What's in it |
|------|-------------|
| `MISTAKES.md` | 37 documented mistakes with root causes — **always read first** |
| `CLAUDE.md` | Architecture overview, hyperparameter defaults, design rules |
| `experiments/Mamba3/epc_model.py` | ePC wrapper: `E()`, `E_local()`, `minimize_error_energy()` |
| `experiments/Mamba3/EPC_LEARNING_DYNAMICS.md` | Why SGD beats Newton, error coverage analysis |
| `experiments/Mamba3/memory_improvement_options.md` | 9 options for improving Mamba3 memory |
| `experiments/energy_reasoning/epc_jepa_model.py` | ePC-JEPA model with JEPA + decode CE + VICReg |
| `docs/hypotheses/generalization_vs_memorization.md` | Multi-rule collapse hypothesis for Stage 2 failure |

---

## Suggested Next Steps

1. **Diagnose the flat landscape:** Measure `ε_i.grad` components (penalty vs output-loss) and the Jacobian `∂ŷ/∂ε_i` singular values. This will tell us whether the architecture or the loss formulation is the bottleneck.

2. **If Jacobian is small:** The architecture needs modification — errors simply can't influence the output through the current path. Options: remove/bypass RMSNorm for errors, add skip connections from errors to output, or restructure where errors sit.

3. **If Jacobian is fine but penalty dominates:** The mean reduction scaling might be wrong. Consider: sum reduction with a small coefficient, or adaptive penalty weight that balances the two terms.

4. **Stage 2 generalization:** Independent of the energy landscape — this is about whether the JEPA architecture can discover multiple rules. The grokking hypothesis suggests extended training (500+ epochs) might trigger delayed generalization.
