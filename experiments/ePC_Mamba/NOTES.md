# ePC-Mamba: Predictive Coding meets State Space Models

## Vision
Apply error-based predictive coding (ePC) to a Mamba-based sequence model,
then extend to a JEPA network for logical reasoning (starting with math).

Long-term path: ePC-Mamba -> ePC-JEPA -> mathematical reasoning

---

## What We Learned from ePC-ResNet

### Performance Profile (ResNet-18, CIFAR-10, RTX 3050 Ti)

ePC with T=2 Newton inference requires **7 network passes per batch**:
- 1 error init (eliminated via shape caching)
- 2 forward passes (inference, computing E)
- 2 backward passes (inference, error gradients)
- 1 forward pass (E_local, weight loss)
- 1 backward pass (weight gradients)

Standard backprop: 2 passes. ePC overhead: ~3.5x.

Final profiling (post-optimization, 10-epoch avg):
| Phase             | ms    | %     |
|-------------------|-------|-------|
| Error init        | ~0    | ~0%   | (shape caching fix)
| Inference fwd (×T)| 68.3  | 26.0% |
| Inference bwd (×T)| 59.0  | 22.5% |
| Newton step (×T)  | 21.3  | 8.1%  |
| Weight forward    | 41.9  | 16.0% |
| Weight backward   | 38.7  | 14.8% |
| Other             | ~5    | ~2%   |
| **Total**         |**~235**|       |

Throughput: ~976 samples/sec (batch=256)

### Key Optimizations Discovered
1. **fp32 errors under FP16 autocast** — fp16 rounds Newton corrections to zero,
   defeating early stopping. Errors MUST be fp32; the expensive ops (matmul)
   still run fp16 via autocast.
2. **Shape caching** — cache error tensor shapes after first batch, skip the
   init forward pass on all subsequent batches. ~10% speedup, zero overhead.
3. **Streaming Newton step** — decompose dot products per-layer (gTg, gTe, eTe)
   instead of concatenating all errors into a giant vector. In-place update
   via mul_() + add_(). Reduced Newton from 94ms to 21ms.
4. **Adaptive early stopping** — skip Newton iterations when relative energy
   improvement < threshold. Most batches converge at T=2 when training is
   progressing well; early epochs may get T=1 (early stopping kicks in).

### What Didn't Work
- **KFAC** — failed for ePC (mistake #20)
- **INT8 QAT** — quantization noise amplified over T iterations destroys accuracy (mistake #21)
- **AdaWoodbury** — custom optimizer failed (mistake #22)
- **Diagonal Hessian approximations** — break positive-definiteness guarantees (mistake #5)
- See MISTAKES.md for full catalogue (22 mistakes)

### Remaining Bottleneck
Forward and backward passes through the network (conv/linear ops in FP16).
These are hardware-limited on the 3050 Ti. For ResNet-18 (convolutional),
there's no room for structured matrix optimization — convolutions are already
structured. **This changes entirely for Mamba/Transformers where dense linear
projections dominate compute.**

---

## Mamba3 Architecture (ICLR 2026 Submission)

### Overview
Mamba3 alternates **Mamba3 mixer blocks** and **SwiGLU MLP blocks** with
pre-normalization (RMSNorm). Llama-style macro layout.

Three improvements over Mamba2:
1. **Trapezoidal discretization** (2nd-order vs Euler's 1st-order)
2. **Complex-valued states** via data-dependent RoPE (recovers state-tracking)
3. **MIMO formulation** (matrix product instead of outer product for higher
   arithmetic intensity during inference)

### Where Parameters Live
Per Mamba3 block (expansion factor E ≈ 2):
- Input projections: 2 × E × D² (dominant)
- Output projection: E × D²
- SSM parameters (dt, B, C, RoPE theta, lambda): O(D×n), small
Per SwiGLU MLP block:
- ~8 × D² (with gating)

**~80%+ of parameters are in dense linear projections.** This is where BTT
can help.

### Implementation Status
- Paper: under review at ICLR 2026 (anonymous)
- Code: NOT released yet (as of Feb 2026)
- Mamba2 code: available at github.com/state-spaces/mamba
- **Plan: start with Mamba2, upgrade to Mamba3 when code drops**

### Key Properties for ePC
- Fixed-size recurrent state (unlike Transformer KV cache)
- Sub-quadratic in sequence length
- Selective gating ↔ precision-weighted prediction errors (natural PC analogy)
- Dense projections dominate compute → BTT directly applicable

---

## BTT (Block Tensor-Train) Analysis

### What It Is
A structured matrix family that replaces dense d×d matrices with two smaller
"core" tensors. For c=2 cores, rank r:
- Dense: d² params, d² FLOPs
- BTT: 2r × d^(3/2) params, 2r × d^(3/2) FLOPs
- At r=1, BTT ≡ Monarch with √d blocks

### Key Results from Paper (Qiu et al., ICML 2024)
- BTT matches dense ViT-S/32 on ImageNet with **3.8x less compute**
- Better scaling exponent than dense (performance improves faster with compute)
- CIFAR-10/100: exponentially lower training loss than dense for MLPs/ViTs
- GPT-2: more compute-efficient than dense

### Critical Insights for Our Use
1. **FLOPs = Parameters is the winning principle.** Kronecker and TT share
   parameters and underperform. BTT and Monarch don't share → better scaling.
2. **Structure-aware learning rates are ESSENTIAL.** Naive dense LR on BTT
   kills performance. The paper provides exact multiplier formulas (Table 2).
   For BTT with Adam: κ_L = √d/(2r), κ_R = √d/2.
3. **Lower rank = more compute-efficient.** BTT r=1 (≡ Monarch √d blocks)
   is the sweet spot for compute efficiency. Higher ranks trade compute
   efficiency for memory efficiency.
4. **Weight normalization required for transformers** to prevent unbounded
   activation growth. Reparameterize cores as:
   M̃ = γ_M × min(1, σ_M / RMS(M)) × M
5. **Overhead for small matrices.** BTT starts winning at d ≥ 1024. Below
   that, the overhead of reshaping/permuting dominates. Our ResNet channel
   sizes (64-512) are too small. Mamba d_model ≥ 256 could work, ≥ 1024
   is the sweet spot.

### Applicability to ePC-Mamba
For a Mamba model with d_model=1024:
- Dense projection: 1024² = 1M FLOPs per MVM
- BTT r=1: 2 × 1024^(1.5) ≈ 65K FLOPs per MVM → **~15x fewer FLOPs**
- In ePC with T=2: 4 forward passes use these projections → savings multiply
- **BTT could reduce the ePC overhead from ~3.5x to ~1.5-2x vs backprop**

For proof-of-concept at d_model=256:
- Dense: 256² = 65K FLOPs
- BTT r=1: 2 × 256^(1.5) ≈ 8K FLOPs → ~8x fewer
- But overhead may eat into this at small scales

### Implementation Path
The paper uses CoLA (Compositional Linear Algebra) library for efficient
MVMs. We may need to implement BTT layers directly in PyTorch since:
- CoLA adds a dependency we may not want
- We need autograd support for ePC's error optimization
- Custom implementation gives us control over memory layout

---

## Applying ePC to Mamba: The Plan

### The Core Idea
Treat each Mamba/MLP block as a "layer" in the ePC hierarchy. Errors e_i
are added between blocks. The inference phase optimizes errors to minimize
total energy E. The weight phase uses E_local with detached states.

```
Input sequence → [Block 0] + e_0 → [Block 1] + e_1 → ... → [Block N] → Output
                               ↑                  ↑
                        prediction errors    prediction errors
```

### Spatial ePC (Phase 1 — proof of concept)
- Errors are between layers (spatial), NOT between timesteps
- The recurrent state is part of the forward computation
- The model processes the full sequence in each forward pass
- This is a direct extension of ePC-ResNet to a different architecture
- **Simplest approach, validates the framework works with Mamba**

### Spatio-temporal ePC (Phase 2 — future)
- Errors in both layer and time dimensions
- Relates to temporal predictive coding (tPC, Millidge et al. 2024)
- Much more complex but could enable online learning
- tPC ≡ Kalman filter with fixed posterior variance

### Prior Art
- **Nobody has applied ePC to Mamba/SSMs.** This is genuinely novel.
- Bio-Inspired Mamba (BIM, Qin 2024) used STDP-like rules, not ePC.
- tPC (Millidge 2024) works on recurrent nets but is a different formulation.
- ePC (Goemaere 2025) only covers feedforward architectures.
- The Mamba selectivity ↔ PC precision-weighting analogy is unexplored.

---

## Warm-Starting Errors: Analysis

### Why It Doesn't Work (for shuffled batches)
Errors are per-sample: [batch, channels, ...]. With shuffled batches, position
k in the next batch is a completely different sample. The previous error at
position k is noise for the new sample.

Effect: higher initial energy E_0 → Newton wastes iterations undoing noise
→ likely accuracy degradation.

### When It Could Work
- **Autoregressive inference** (single sample, iterating over time steps):
  the errors from timestep t are meaningful context for timestep t+1
- **Per-class average errors** as initialization (requires knowing class)
- **Learned initialization** (amortized inference) — a small network predicts
  initial errors from input. Promising but complex.

---

## Sparse Connectivity: Analysis

### The Promise
99%+ of trained network weights can be pruned with <0.5% accuracy loss.
Sparse matmul FLOPs scale with non-zeros, not matrix size.

### The Challenge During Training
- Don't know which connections matter until training converges
- RigL (Evci et al. 2020): maintain fixed sparsity, periodically prune
  smallest weights and grow where dense gradient is largest
- Growing step requires occasional dense gradient computation (expensive)
- GPU sparse ops often slower than dense due to irregular memory access
  (need >90% sparsity to see wall-clock speedup on current hardware)

### For ePC Specifically
- Sparse weights → sparse Jacobian J → lower effective rank of J^T H_L J
- Our rank-1 Newton approximation might capture a LARGER fraction of true
  curvature in sparse networks → potentially T=1 convergence more often
- But: sparse execution on GPU currently impractical for training
- **Revisit when hardware/software support improves (e.g., NVIDIA 2:4 sparsity)**

---

## Proof-of-Concept Plan

### Architecture
- **Base: Mamba2** (code available, Mamba3 code not released)
- **Scale: small** — d_model=128-256, 4-6 layers, ~5-15M parameters
- ePC wrapper: PCESequence class extending PCE for sequence models
- Newton optimizer with T=2, damping=0.1, early stopping

### Dataset (in order of complexity)
1. **Synthetic tasks** (copy, sorting, parity) — verify ePC+Mamba works at all
2. **Character-level text** (Shakespeare, ~1MB) — test language modeling
3. **WikiText-2** — standard benchmark, compare with backprop baseline

### Success Criteria
1. ePC-Mamba converges on synthetic tasks (parity, copy)
2. ePC-Mamba achieves reasonable perplexity on character-level text
3. Performance within 2x of backprop baseline
4. BTT projections show measurable compute savings at d_model ≥ 256

### Key Risks
1. ePC's multi-pass inference may not scale to long sequences (memory)
2. Mamba's recurrent state may interact badly with error optimization
3. BTT overhead at small d_model may negate theoretical FLOPs savings
4. The Newton step's rank-1 approximation may be insufficient for the
   more complex loss landscape of sequence models

### What We Need to Build
- [ ] `epc_mamba_model.py` — PCE wrapper for Mamba blocks
- [ ] `mamba_blocks.py` — Mamba2 block implementation (or import from library)
- [ ] `btt_linear.py` — BTT-structured linear layer with autograd support
- [ ] `train_synthetic.py` — training script for synthetic tasks
- [ ] `train_text.py` — training script for character/token-level language modeling
- [ ] `architectures.py` — model configurations at different scales

---

## References
- Mamba3: OpenReview HwCvaJOiCj (ICLR 2026 submission, anonymous)
- Mamba2: Gu & Dao, arXiv:2405.21060
- Mamba1: Gu & Dao, arXiv:2312.00752
- ePC: Goemaere et al., arXiv:2505.20137
- BTT: Qiu et al., ICML 2024 (in docs/research/BTT paper.pdf)
- tPC: Millidge et al., PLOS Comp Bio 2024
- BIM: Qin, arXiv:2409.11263
- RigL: Evci et al., ICML 2020
- CoLA: Potapczynski et al., NeurIPS 2024
