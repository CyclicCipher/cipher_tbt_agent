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

### Mamba3 Specifics

**Trapezoidal discretization:**
```
Mamba2 (Euler):   h_t = exp(dt*A) * h_{t-1} + dt * B_t * x_t
Mamba3 (Trapez):  h_t = exp(dt*A) * h_{t-1} + (1-λ_t)*dt*exp(dt*A)*B_{t-1}*x_{t-1} + λ_t*dt*B_t*x_t
```
Second-order (O(dt²) cumulative error vs O(dt)). λ_t is data-dependent (learned,
not fixed 0.5). State depends on current AND previous input → effectively a
size-2 convolution, making the explicit conv1d redundant.

**Complex-valued states via data-dependent RoPE:**
Mamba2's real scalar A cannot solve parity (proven). Mamba3 adds complex
eigenvalues: `h_t = exp(dt*A) * R_t * h_{t-1} + dt * B_t * x_t` where R_t
is a rotation matrix from data-dependent θ_t. Crucially: this is equivalent
to applying RoPE to B and C before the SSD scan — **no complex arithmetic
needed at runtime.** Result: 0.9% → 100% on parity.

**MIMO:** Rank-r matrix projections instead of rank-1 outer products. Increases
arithmetic intensity for inference. Training benefit is smaller. Skip for now.

### Mamba2 Block Anatomy (What We're Building From)

```
in_proj: d_model → [z, x, B, C, dt]   ← DOMINANT (e.g., 256→1160, ~297K params)
conv1d: depthwise causal, kernel=4      ← removed by trapezoidal
SSD scan: segsum → 4-step chunked algo  ← core SSM, ~30 lines of einsums
out_proj: d_inner → d_model             ← DOMINANT (e.g., 512→256, ~131K params)
```

For d_model=256: in_proj + out_proj ≈ 428K params, everything else ≈ 20K.
**80%+ of params in dense projections. BTT directly applicable here.**

### Implementation Status
- Paper: under review at ICLR 2026 (anonymous)
- Code: NOT released yet (as of Feb 2026)
- Mamba2 code: available at github.com/state-spaces/mamba (v2.2.6)
- Pure PyTorch reference: tommyip/mamba2-minimal (~200 lines)
- **Plan: build custom Mamba2 block in pure PyTorch, then incrementally add
  Mamba3's data-dependent RoPE (easy) and our own discretization rule**

### Replication Path for Mamba3 Components
1. **Start with Mamba2** — pure PyTorch SSD, no CUDA kernel dependency
2. **Add data-dependent RoPE** (easy) — just preprocess B, C before SSD scan.
   Biggest quality gain. Enables state tracking (parity, counting, etc.)
3. **Add custom discretization** (medium) — modify SSD kernel to include
   previous-step term. Removes conv1d. User has own rule idea.
4. **Skip MIMO** — inference optimization, not training priority

### Key Properties for ePC
- Fixed-size recurrent state (unlike Transformer KV cache)
- Sub-quadratic in sequence length
- Selective gating ↔ precision-weighted prediction errors (natural PC analogy)
- Dense projections dominate compute → BTT directly applicable
- Pure PyTorch SSD is ~3-5x slower than fused CUDA, but acceptable at small
  scale on 3050 Ti for proof-of-concept

### ePC-Specific Concerns for Mamba (Risk-Ordered)

**1. Selectivity makes errors multiplicative (HIGH RISK)**
In ResNet, errors are purely additive (shift activations between layers). In
Mamba, the block input is projected into B, C, dt (SSM parameters) AND x (data).
When an error shifts block input, it changes both the signal and the dynamics —
multiplicative, not additive. The energy landscape will be more non-convex than
ResNet's. If T=2 Newton doesn't converge, may need T=3 or stronger damping.

**2. Causal error propagation (MEDIUM RISK)**
ResNet errors affect all spatial positions equally. Mamba is causal: an error
perturbation at position t propagates to positions ≥t but NOT <t. Early-position
errors have outsized influence (shift the entire downstream state trajectory).
The rank-1 Hessian approximation assumes relatively uniform error interactions —
causal asymmetry may degrade it. Monitor per-position error norms during dev.

**3. Error placement relative to RMSNorm (MEDIUM RISK — design choice)**
Architecture has `RMSNorm → Mamba Block → + error`. RMSNorm rescales including
the error, which can amplify/dampen corrections unpredictably. Place errors
AFTER the next layer's RMSNorm instead: norm operates on "clean" signal, error
is an unscaled correction. Matches ePC-ResNet pattern.

**4. VRAM budget for sequence-length errors (LOW NOW, HIGH AT SCALE)**
ResNet errors shrink through pooling. Mamba errors stay at full seqlen:
- Phase 1 (32×64×128×4 = 1MB/error, 1 error): fine
- Phase 2 (16×256×256×4 = 4MB/error, 3 errors = 12MB): fine
- Scale (8×1024×1024×4 = 32MB/error, 23 errors = 736MB): significant
fp32 requirement means we can't halve with fp16.

**5. SiLU gating sharp transitions (LOW RISK)**
`output = SiLU(z) * SSM_output` — if error pushes z near zero, entire SSM
output gets suppressed. Near-discontinuities in energy landscape. Sufficient
damping should handle this.

**6. Trapezoidal tightens temporal coupling (LOW RISK — Phase 2)**
Trapezoidal uses previous timestep's B*x, so errors at t-1 directly affect
coupling at t (not just through state). Denser error interaction pattern than
Euler. Verify convergence on Euler (Mamba2) first.

### Prior Work: PC + Mamba Literature Review (Feb 2026)

**No published work combines any form of predictive coding with Mamba.** Confirmed
via exhaustive search. ePC-Mamba is genuinely novel. Adjacent works:

| Work | What | Relation to ePC-Mamba |
|------|------|-----------------------|
| BIM (Qin 2024) | STDP+RTRL for Mamba (bioplausible) | Different learning rule entirely. Temporal locality, not spatial. |
| tPC (Millidge 2024) | Temporal PC on linear SSMs (1-2 layers) | Different PC variant, shallow, Hebbian. No Newton, no depth. |
| DPC (Jiang & Rao 2024) | Hierarchical PC on RNNs (2-3 levels) | Neuroscience model, not DL-scale. |
| PAM (Mounir, NeurIPS 2024) | Free energy + custom SSM + Hebbian | Validates SSM+free-energy concept, but simple dynamics. |
| MPS-SSM (Wang 2025) | Info-theoretic regularizer for Mamba | Orthogonal (backprop+regularizer, not PC). |
| ePC (Goemaere 2025) | Error-based PC + Newton on feedforward | Our base algorithm. Only covers feedforward. |

**Confirmed gaps (no published work):**
1. ePC + Mamba (or any selective SSM) — our work
2. Any form of PC + Mamba — not even standard sPC
3. Energy-based learning + Mamba — no equilibrium propagation either
4. ePC + ANY recurrent/sequence architecture — novel regardless of base

**Unexploited analogy:** Mamba's input-dependent Δ_t is functionally analogous
to precision weighting in PC. Both gate information based on input-dependent
reliability estimates. Nobody has formalized this connection.

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

## Research Paper Analysis: What Applies to ePC-Mamba

### The "Low-Rank + Diagonal" Unifying Motif

A single structural pattern appears across ALL the research papers: **low-rank
plus diagonal (LR+D)**. It shows up in:
- Sparse GPs: Q + Λ (Nyström low-rank covariance + diagonal correction)
- BTT: block tensor decomposition of weight matrices (≡ block LR)
- LRPD Riccati: Y = URU^T + ψ for evolving covariance under state dynamics
- Mamba SSM: low-rank state update (B*x is rank-1) + diagonal decay (A)
- ePC Newton: Hessian H = I + J^T H_L J is identity + low-rank

**Key insight from sparse GP papers**: The diagonal Λ is NOT optional decoration.
Without it, gradients for the low-rank factors vanish in regions far from the
current "support." With Λ, every data point contributes meaningful gradients.
**This directly explains why naive dense LR on BTT kills performance** — and
suggests BTT + diagonal correction could have even better gradient flow than
BTT alone.

### LRPD for Riccati-like Matrix DEs (Bonnabel, Lambert, Bach 2024)

**What**: Evolve large PSD matrices (covariances, precisions) under differential
equations while constraining them to low-rank + diagonal form. Two forms:
- PPCA: Y = URU^T + s(I - UU^T), s > 0 scalar (cheapest)
- FA: Y = URU^T + diag(ψ), per-dimension diagonal (more expressive)

**Key technique**: Tangent-space projection on structured matrix manifolds.
Project the ODE vector field onto T_Y(M) at each step to stay in the
structured subset. Closed-form projections via Woodbury identity: O(dp²).

**Why it matters for us**:
1. **SSM covariance tracking**: If we want uncertainty over Mamba's hidden state
   h(t) (for Bayesian ePC or precision-weighted errors), the covariance of the
   belief satisfies a discrete Riccati equation. PPCA approximation: O(n*p)
   instead of O(n²), where n = d_state.
2. **Curvature estimation for Newton step**: Our Newton step approximates
   H⁻¹ via rank-1 Woodbury. PPCA gives a rank-p approximation that evolves
   smoothly across batches (warm-startable! — curvature structure is stable
   even when errors/inputs change). Could improve T=2 → T=1 convergence.
3. **Invertibility guarantee**: Pure low-rank covariance is singular → Kalman
   filter diverges (proven in their Prop 7). PPCA is always full-rank. Same
   applies to precision matrices in predictive coding.
4. **The U equation resembles the Oja flow** (tracks dominant eigenspace).
   U naturally aligns with the top curvature directions over training.

### LRPD Spectral Approach (Yeon & Anitescu 2025)

**What**: Fast algorithm to decompose any PSD matrix as low-rank + diagonal.
Alternating spectral projection: top-k eigendecomposition of residual, then
closed-form diagonal update. Converges to machine precision in ~20 iterations.

**Key technique**: Alternate between Eckart-Young (optimal low-rank of residual)
and exact diagonal minimization (decoupled, trivially parallel).

**Why it matters**:
1. **Stochastic variant needs only matrix-vector products** — could decompose
   the Hessian (accessed via Hessian-vector products) into LR+D form cheaply.
2. **Dynamic rank adaptation**: computable error bounds at O(d) cost per step.
   Know when to increase/decrease rank of any structured approximation.
3. **Block-diagonal generalization**: replace scalar diagonal with block-diagonal
   (e.g., per-layer blocks, per-head blocks in Mamba). Even better fit.

### Sparse GP Papers (Quinonero-Candela & Rasmussen 2005; Snelson & Ghahramani 2006)

**What**: Unified framework for sparse GP regression. All methods reduce to
"exact inference with approximate prior" using m inducing variables (m << n).

**Taxonomy**: SoR < DTC < FITC < PITC (quality ranking, same O(nm²) cost)
- FITC = low-rank + diagonal correction (= Snelson's SPGP)
- PITC = low-rank + block-diagonal correction (best quality)

**Critical insight for us**:
1. **Inducing variables ↔ Mamba hidden state**: The hidden state h_t mediates
   input→output mapping through a low-dimensional bottleneck (d_state << L).
   This IS the inducing variable framework. FITC/PITC suggest: don't just use
   a deterministic state bottleneck — add a diagonal correction capturing what
   the bottleneck misses.
2. **Λ makes gradient-based learning of low-rank factors tractable**: SPGP's
   input-dependent noise Λ(x) = K_xx - k_x^T K_M^{-1} k_x creates gradients
   even far from inducing points. Without it (DTC/SoR), pseudo-input gradients
   vanish → they get stuck. **This is the GP perspective on why BTT needs
   structure-aware LR.**
3. **Mamba's selective Δ ↔ SPGP's Λ**: Mamba's input-dependent step size Δ_t
   controls how much each timestep trusts input vs state. This IS analogous
   to SPGP's input-dependent noise variance. The GP framework gives a
   principled probabilistic interpretation of Mamba's selectivity.
4. **Continuous pseudo-input optimization**: SPGP learns WHERE to place inducing
   points by gradient descent on marginal likelihood. For Mamba: learn what
   the state dimensions should be sensitive to (their "selectivity profiles").

### Bayesian Predictive Coding (Tschantz et al. 2025) — BLOCKED

**What**: Full Bayesian posterior over PC network weights via Matrix Normal-
Wishart conjugate prior. Closed-form M-step (Hebbian). Converges in 1-3
epochs (full-batch) vs dozens for PC/BP+Adam.

**Status: INTRACTABLE for LRPD approximation.** Multiple approaches tried
(diagonal, low-rank η1, FITC, spectral inflation) — all break PD guarantees
or over-regularize. The MNW quadratic constraint (block PSD of natural
parameters) creates tight coupling that no known LRPD form preserves.
See MISTAKES.md entries #16, #18, #19 for full details. Do NOT attempt
BPC weight updates without a fundamentally new approach to the MNW problem.

### TinyLoRA: Learning to Reason in 13 Parameters (Morris et al. 2026)

**What**: RL fine-tuning with vanishingly few parameters. 13 params push
Qwen2.5-7B from 76% to 91% on GSM8K. Uses SVD decomposition + random
projection: W' = W + U Σ (Σ_i v_i P_i) V^T.

**Why it matters**:
1. **RL makes denser updates than SFT** — the search/resampling phase filters
   noise. ePC's iterative inference is analogous: it "searches" before updating
   weights. Prediction: ePC can learn effectively with very compact parameter
   updates (like BTT with tiny rank).
2. **fp32 matters at extreme compression** — even 13 parameters need fp32.
   Confirms our finding that precision is critical for ePC.
3. **Larger models need fewer parameters**: scaling law suggests our eventual
   1B model might need very few trainable dimensions per task.

### Shared LoRA Subspaces (Kaushik et al. 2026)

**What**: Continual learning via evolving shared low-rank subspace across tasks.
100x parameter reduction, 281x memory savings vs separate LoRA per task.
SVD-based merging is gradient-free and data-free.

**Why it matters for long-term plan (ePC-Mamba → ePC-JEPA → math)**:
1. **Continual task adaptation** — synthetic → text → code → math without
   catastrophic forgetting, without replay buffers.
2. **Universal Weight Subspace Hypothesis**: LoRA adapters for different tasks
   converge to a shared subspace. Our Mamba projections (BTT-structured) might
   exhibit the same: a shared subspace of BTT cores across tasks.
3. **Backward transfer observed**: CoLA improved after learning later tasks.
   The evolving subspace helps earlier tasks — exactly what we want when
   progressing through increasingly complex reasoning tasks.

---

## Synthesis: How It All Fits Together

```
                    ┌──────────────────────────────────┐
                    │      ePC-Mamba Architecture       │
                    ├──────────────────────────────────┤
                    │ Dense projections → BTT (§BTT)    │
                    │ + diagonal correction (§Sparse GP) │
                    │                                    │
                    │ SSM: Mamba2 SSD scan               │
                    │ + data-dependent RoPE (Mamba3)     │
                    │ + custom discretization rule       │
                    │                                    │
                    │ Error optimization: Newton (ePC)   │
                    │ + PPCA curvature tracking (§LRPD)  │
                    │ + fp32 errors, shape caching       │
                    │                                    │
                    │ Weight updates: Adam (proven)      │
                    │ + structure-aware LR for BTT       │
                    │                                    │
                    │ Scaling: Share subspaces (§Share)   │
                    │ for continual task progression     │
                    └──────────────────────────────────┘
```

**Priority order for implementation:**
1. Mamba2 block (pure PyTorch) + ePC wrapper → synthetic tasks
2. Data-dependent RoPE (Mamba3 feature #2) → state tracking
3. Custom discretization rule → replace conv1d
4. BTT projections → compute savings (when d_model ≥ 256)
5. PPCA curvature for Newton → faster convergence (T=2 → T=1)
6. Share subspaces → continual learning across task progression

---

## Discretization Rules: Options Beyond Trapezoidal

**Default: Mamba3's trapezoidal rule** (2nd order, data-dependent λ_t).
Architecture should support swappable discretization for experimentation.

The continuous SSM: `h'(t) = A(t)h(t) + B(t)x(t)`. Since `exp(dt*A)` is
computed exactly (scalar per head), only the input coupling integral is
approximated:

```
h(t+dt) = exp(dt*A)*h(t) + ∫₀ᵈᵗ exp(A*(dt-s)) * B(t+s)*x(t+s) ds
                              ↑ this integral is what we approximate
```

### Option 1: ETD with ψ₁ correction (most promising novel approach)
```
h_t = exp(dt*A)*h_{t-1} + ψ₁(dt*A) * dt*B_t*x_t
where ψ₁(z) = (exp(z) - 1) / z
```
Physically more correct than trapezoidal: accounts for exponential decay
DURING input injection. Fast-decaying states (large |A|) automatically
weight recent input more. Trivial to implement (one extra division per head).
Nobody has done this for Mamba.

### Option 2: Adams-Bashforth 3 (3rd order, fully causal)
```
h_t = exp(dt*A)*h_{t-1} + dt*(23/12*f_{t-1} - 4/3*f_{t-2} + 5/12*f_{t-3})
where f_t = B_t*x_t
```
3-tap causal filter integrated into discretization. Stores 2-3 extra B*x
terms at chunk boundaries. Higher order but fixed coefficients.

### Option 3: Data-dependent multi-step (learned integration)
```
h_t = exp(dt*A)*h_{t-1} + Σᵢ wᵢ(x_t) * B_{t-i}*x_{t-i}
```
Generalize trapezoidal's λ_t to N previous steps with learned coefficients.
Initialize from AB3/AM3 for stability, let training adjust. Most expressive.

### Option 4: Predictor-corrector (natural ePC connection)
Predict h_t via Adams-Bashforth (explicit), correct via Adams-Moulton
(using predicted value). The ePC inference loop IS a correction step —
this unifies numerical integration with error optimization.

### Implementation approach
```python
class DiscretizationRule(Enum):
    EULER = "euler"           # Mamba2 baseline
    TRAPEZOIDAL = "trapez"    # Mamba3 (default)
    ETD = "etd"               # Exponential time differencing
    AB3 = "ab3"               # Adams-Bashforth 3rd order
    LEARNED = "learned"       # Data-dependent multi-step
```
Each rule defines how to compute the input coupling term given (dt, A, B, x)
and any required previous-step state.

---

## Dataset Notes

### WikiText-2
- **~2M tokens, ~12MB raw text on disk** — easily fits in memory
- 33K vocabulary (word-level) or ~256 (character-level)
- Good for benchmarking: standard perplexity comparisons available
- Full articles → tests long-range dependency modeling

### Synthetic Tasks (for validation)
- **Copy**: input → delay → reproduce (tests state memory)
- **Parity**: running XOR of binary input (tests state tracking — Mamba2 fails,
  data-dependent RoPE needed)
- **Sorting**: sort a short sequence (tests comparison/reordering)
- **Selective copy**: copy only certain tokens based on markers

### Character-level Text
- **Shakespeare** (~1MB) — good for quick iteration
- Can also use WikiText-2 raw for character-level

### Memory Budget (RTX 3050 Ti, 4GB VRAM)
- Model: d_model=256, 4 layers → ~10M params → ~40MB fp32
- Errors: 4 layers × batch × seqlen × d_model × fp32 → ~4MB at batch=16, L=512
- Activations for backprop: ~100-200MB (depends on sequence length)
- **Comfortable at batch=16, seqlen=256-512. Tight at seqlen=1024+.**

---

## References

### Architecture
- Mamba3: OpenReview HwCvaJOiCj (ICLR 2026 submission, anonymous)
- Mamba2: Gu & Dao, arXiv:2405.21060
- Mamba1: Gu & Dao, arXiv:2312.00752
- mamba2-minimal: github.com/tommyip/mamba2-minimal (pure PyTorch reference)
- mamba2-torch: github.com/vasqu/mamba2-torch (HuggingFace-compatible)

### Predictive Coding
- ePC: Goemaere et al., arXiv:2505.20137
- BPC: Tschantz et al., arXiv:2503.24016
- tPC: Millidge et al., PLOS Comp Bio 2024
- BIM: Qin, arXiv:2409.11263

### Structured Matrices & Optimization
- BTT: Qiu et al., ICML 2024 (in docs/research/BTT paper.pdf)
- LRPD Riccati: Bonnabel, Lambert, Bach, arXiv:2407.03373
- LRPD Spectral: Yeon & Anitescu, arXiv:2512.17120
- CoLA: Potapczynski et al., NeurIPS 2024

### Sparse GPs & Inducing Variables
- Unified Sparse GP: Quinonero-Candela & Rasmussen, JMLR 2005
- SPGP: Snelson & Ghahramani, NeurIPS 2006

### Efficient Adaptation
- TinyLoRA: Morris et al., arXiv:2602.04118
- Share: Kaushik et al., arXiv:2602.06043
- RigL: Evci et al., ICML 2020
