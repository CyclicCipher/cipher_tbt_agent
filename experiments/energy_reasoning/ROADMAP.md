# Energy-Based Reasoning Architecture

## Vision

A neurosymbolic reasoning system that combines:
- **Mamba3** as a parameter-efficient sequence backbone (proven on copy task)
- **JEPA-style** latent prediction for learning abstractions
- **Energy minimization** (Langevin dynamics) for reasoning at inference time
- **Category Theory Knowledge Graph** as grounding, reward model, and constraint source
- **Block Tensor-Train** compression to fit in 4GB VRAM

The aim: a small model that reasons by optimizing, not by next-token prediction.

---

## Architecture Overview

```
                        TRAINING (backprop)
                        ===================

    x_context                                   x_target
        |                                           |
   [Context Encoder]                        [Target Encoder (EMA)]
   (Mamba3 blocks)                          (Mamba3 blocks, frozen)
        |                                           |
     s_context                                   s_target
        |                                           |
   [Predictor]                                      |
   (narrow Mamba3,                                  |
    + mask tokens                                   |
    + latent z)                                     |
        |                                           |
     s_predicted -------- L2 loss ----------- s_target

                 + VICReg(s) to prevent collapse
                 + E_ctkg(s) path-alignment reward (Phase 3+)


                        INFERENCE (energy minimization)
                        ==============================

    x_input
        |
   [Context Encoder]
        |
     s_context
        |
   [Initialize latent z ~ N(0,I)]
        |
        v
   +----- Langevin Dynamics Loop (T steps) -----+
   |                                             |
   |  s_pred = Predictor(s_context, z)           |
   |  E(z) = E_pred(z) + E_constraint(z)        |
   |  z <- z - eta * grad_z E(z) + noise        |
   |                                             |
   +---------------------------------------------+
        |
     z* (converged)
        |
   s_pred* = Predictor(s_context, z*)
        |
   [Decoder] -> output
```

---

## What Goes Between Encoder and Decoder

Three components work together:

### 1. The Predictor

A **narrow Mamba3 network** (smaller than the encoder) that maps context
representations to target representations, conditioned on a latent variable z.

- Input: s_context (from encoder) + learnable mask tokens (indicating what to predict) + latent z
- Output: s_predicted (predicted target representation)
- Architecture: 2-4 Mamba3 blocks at reduced dimension (e.g., d_model=64 vs encoder's 128)
- The narrow bottleneck forces the encoder to produce rich representations
  rather than letting the predictor memorize patterns

The predictor learns WHAT to predict (abstractions) by being trained to match
target encoder outputs in latent space. It never sees raw inputs/outputs directly --
only abstract representations.

Why Mamba3 for the predictor (not a transformer): O(n) vs O(n^2) in sequence
length, critical for fitting in 4GB VRAM when running multiple Langevin steps.

### 2. The Energy Function

```
E(z) = E_pred(z) + alpha * E_constraint(z) + beta * E_consistency(z)
```

**E_pred**: Prediction error in latent space.
  `E_pred = ||Predictor(s_context, z) - s_target||^2`
  This is the JEPA energy. Low E_pred means z produces a good latent prediction.
  During training, s_target comes from the EMA target encoder.
  During inference, we don't have s_target -- we minimize over z to find the
  most self-consistent latent state.

**E_constraint** (Phase 3+): CTKG path-alignment energy.
  The CTKG encodes which reasoning steps are valid compositions.
  Given a proposed latent trajectory, E_constraint measures how well it aligns
  with valid paths in the knowledge graph. Low energy = valid reasoning chain.
  This is the "knowledge graphs as implicit reward models" idea applied as an
  energy term rather than an RL reward.

**E_consistency**: Internal coherence.
  Measures whether different parts of the latent state are mutually consistent.
  For sequences: does the predicted state at position t agree with what position
  t-1 implies about t? Analogous to Kona's constraint satisfaction.
  `E_consistency = sum_t ||s_pred_t - Pred(s_pred_{t-1})||^2`

### 3. Langevin Dynamics (The Reasoning Loop)

At inference time, reasoning = energy minimization:

```
for step in range(T):
    E = energy_fn(z)
    grad_z = autograd.grad(E, z)
    z = z - eta * grad_z + sqrt(2 * eta) * noise
```

- T=3-5 steps (EBM-CoT paper found 3 optimal)
- eta = step size (tunable, ~0.01)
- noise helps escape local minima (dropped at test time for determinism)
- The gradient tells z which direction reduces energy -- i.e., which latent
  state better satisfies all constraints simultaneously

This is exactly what Kona does: the "Soft Thought tensor" IS our latent z,
and Langevin dynamics IS the reasoning process. The key insight is that
reasoning is not sequential token generation -- it's finding the z that
simultaneously satisfies prediction accuracy, constraint satisfaction,
and internal consistency.

---

## Anti-Collapse: VICReg

JEPA's central failure mode is representation collapse (everything maps to zero).
We use VICReg regularization during training:

```
L_total = L_pred + beta_var * L_variance + beta_cov * L_covariance

L_variance = mean(max(0, 1 - sqrt(Var(s_j) + eps)))   # keep each dim active
L_covariance = sum(off_diagonal(Cov(s))^2) / d         # decorrelate dims
```

This ensures the latent space remains high-dimensional and informative.
No contrastive negatives needed -- VICReg is purely regularization-based.

The target encoder uses EMA (exponential moving average) of the context encoder
weights, providing the asymmetry needed to prevent trivial solutions:
```
theta_target <- tau * theta_target + (1 - tau) * theta_context
```
With tau starting at 0.996 and linearly increasing to 1.0.

---

## Phase Plan

### Phase 1: JEPA Backbone (Prove latent prediction works)
- [ ] Implement Mamba3 context encoder (reuse existing mamba3_block.py)
- [ ] Implement Mamba3 target encoder with EMA updates
- [ ] Implement narrow Mamba3 predictor
- [ ] Implement VICReg loss (variance + covariance regularization)
- [ ] Training loop: mask random segments, predict in latent space
- [ ] Task: sequence completion on copy task (already have data generation)
- [ ] Success metric: encoder learns non-trivial representations (VICReg monitors)

### Phase 2: Energy Minimization (Prove reasoning works)
- [ ] Implement energy function E(z) = E_pred + E_consistency
- [ ] Implement Langevin dynamics loop (T=3 steps)
- [ ] Inference pipeline: encode -> Langevin refine -> decode
- [ ] Task: multi-step reasoning (e.g., sorting, or simple logic puzzles)
- [ ] Success metric: Langevin steps measurably reduce energy AND improve accuracy
- [ ] Compare: model with vs without Langevin refinement

### Phase 3: CTKG Integration (Add symbolic grounding)
- [ ] Design Category Theory Knowledge Graph schema
- [ ] Implement E_constraint energy term from KG path alignment
- [ ] KG-derived reward signals for training (path overlap scoring)
- [ ] Task: compositional reasoning requiring multi-hop inference
- [ ] Success metric: KG-grounded model outperforms ungrounded on composition

### Phase 4: BTT Compression (Fit in 4GB VRAM)
- [ ] Implement BTT-structured linear layer (btt_linear.py)
- [ ] Replace Mamba3 projections (in_proj, out_proj) with BTT layers
- [ ] Structure-aware learning rates (kappa_L = sqrt(d)/(2r), kappa_R = sqrt(d)/2)
- [ ] Measure FLOPs reduction and accuracy retention
- [ ] Target: 8-15x FLOPs reduction with <2% accuracy loss

### Phase 5: Integration & Scaling
- [ ] Full pipeline: BTT-compressed Mamba3 + JEPA + Langevin + CTKG
- [ ] Memory profiling under 4GB VRAM budget
- [ ] Benchmark against backprop-only Mamba3 baseline
- [ ] Harder reasoning tasks

---

## Key Design Decisions

**Why Mamba3 (not Transformer)?**
O(n) compute and constant memory in sequence length. With Langevin dynamics
running T=3-5 forward passes per inference, quadratic attention cost would be
prohibitive in 4GB VRAM.

**Why JEPA (not autoencoder/VAE)?**
Autoencoders reconstruct inputs (wasted capacity on irrelevant details).
JEPA predicts in latent space, learning only what's predictable and useful.
The encoder can be invariant to noise, irrelevant variation, etc.

**Why Langevin (not MCMC/variational)?**
Langevin dynamics = gradient descent + noise. It directly minimizes energy
using the same autograd infrastructure we already have. No need for proposal
distributions (MCMC) or amortized inference networks (VAE). And at test time,
dropping the noise gives deterministic gradient descent -- fast and reliable.

**Why latent z (not token-level refinement)?**
Refining a compact z (e.g., d=64) is much cheaper than refining a full sequence
of token embeddings. Each Langevin step requires a forward+backward through the
predictor, so smaller z = faster reasoning.

**Training = backprop. Inference = energy minimization.**
We train the encoder, predictor, and decoder with standard backprop (proven to
work -- 99.7% on copy task). The energy-based reasoning happens at inference
time via Langevin dynamics. This sidesteps the ePC convergence problems entirely.

---

## Memory Budget (4GB VRAM, rough estimates)

| Component | Parameters | Memory (fp16) |
|-----------|-----------|---------------|
| Context Encoder (4-layer Mamba3, d=128) | ~600K | ~1.2 MB |
| Target Encoder (EMA copy, frozen) | ~600K | ~1.2 MB |
| Predictor (2-layer Mamba3, d=64) | ~100K | ~0.2 MB |
| Decoder (linear + layernorm) | ~20K | ~0.04 MB |
| Activations (training, batch=32, seq=64) | - | ~50-200 MB |
| Langevin loop activations (T=3 steps) | - | ~150-600 MB |
| Optimizer states (AdamW, 2x params) | - | ~5 MB |
| **Subtotal (no BTT)** | **~1.3M** | **~200-800 MB** |
| **With BTT (8x reduction on projections)** | **~300K** | **~100-400 MB** |
| **CTKG (Phase 3, graph + embeddings)** | TBD | ~100-500 MB |
| **Headroom for scaling** | | **~2-3 GB** |

Comfortably within 4GB even before BTT compression. This means we can scale
d_model up significantly or add more layers when needed.

---

## References

- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022) -- JEPA framework
- Assran et al., "Self-Supervised Learning from Images with a JEPA" (I-JEPA, CVPR 2023)
- Bardes et al., "V-JEPA" (Meta AI, 2024) -- Video JEPA with narrow predictor
- Bardes et al., "VICReg" (2022) -- Variance-Invariance-Covariance regularization
- Chen et al., "Think Consistently, Reason Efficiently" (arXiv:2511.07124) -- EBM-CoT, Langevin calibration
- Kansal & Jha, "Knowledge Graphs are Implicit Reward Models" (arXiv:2601.15160) -- KG path rewards
- Logical Intelligence, Kona 1.0 -- Energy-based reasoning model (commercial, closed-source)
- Qiu et al., "BTT for Compressing LLMs" (ICML 2024) -- Block Tensor-Train compression
