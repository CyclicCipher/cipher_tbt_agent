# ePC-Mamba Implementation Roadmap

## Phase 1: Minimal Mamba2 + ePC (Proof of Concept)

**Goal**: Verify that ePC's error optimization works with SSM blocks.

### Step 1.1: Pure PyTorch Mamba2 block
Build a minimal Mamba2 block from scratch (reference: tommyip/mamba2-minimal).

Files:
- `mamba_block.py` — Single Mamba2 block
  - `in_proj`: Linear(d_model → d_in_proj), split into [z, x, B, C, dt]
  - `conv1d`: Depthwise causal Conv1d(d_inner, kernel=4)
  - `ssd_scan()`: 4-step chunked SSD algorithm (~30 lines of einsums)
    - segsum (stable log-space segment sums)
    - intra-chunk quadratic attention
    - inter-chunk state propagation
    - state-to-output projection
  - `A_log`, `D`, `dt_bias` parameters
  - RMSNorm + SiLU gating
  - `out_proj`: Linear(d_inner → d_model)

Config: d_model=128, d_state=64, headdim=64, expand=2, chunk_size=64

**Validation**: Forward pass produces correct shapes. Compare with
mamba2-minimal on random input for numerical agreement.

### Step 1.2: ePC wrapper for sequence models
Adapt the PCE framework (from ePC_ResNet) for sequence model blocks.

Files:
- `epc_model.py` — PCESequence class
  - `layers`: List of Mamba2 blocks (and optional MLP blocks)
  - `errors`: fp32 tensors between blocks, shape [batch, seqlen, d_model]
  - `init_zero_errors()`: With shape caching (from ResNet optimization)
  - `E(x, y)`: Global energy for inference phase
    - Forward through all blocks, adding errors between them
    - Output loss (cross-entropy for LM, MSE for regression)
  - `E_local(x, y)`: Local energy for weight phase (detached states)
  - `minimize_error_energy()`: Newton T=2, damping, early stopping
  - `_newton_step()`: Streaming per-layer dot products (from ResNet optimization)

Key question: How do errors interact with Mamba's sequence dimension?
- Errors are [batch, seqlen, d_model] — same shape as hidden activations
- They represent "prediction error at each position in the sequence"
- Added to block output before passing to next block

### Step 1.3: Synthetic task training
Verify convergence on simple tasks before attempting language modeling.

Files:
- `train_synthetic.py` — Training loop with diagnostics
  - Copy task: input [a,b,c,0,0,0] → output [0,0,0,a,b,c]
  - Selective copy: copy only marked tokens
  - Sequence classification: label based on sequence content

Config: d_model=128, 2 layers, seqlen=64, batch=32, vocab=16

**Success criteria**: Converges to >95% accuracy on copy task.
If this fails, ePC+Mamba has a fundamental incompatibility we need to debug.

---

## Phase 2: Mamba3 Features

**Goal**: Add the quality improvements that make Mamba3 competitive.

### Step 2.1: Data-dependent RoPE on B and C
Biggest single quality improvement. Enables state tracking.

Changes to `mamba_block.py`:
- Add `theta_proj`: Linear(d_inner → d_state) for producing rotation angles
- Before SSD scan: apply rotary embedding to B and C using theta_t
- RoPE implementation: standard sin/cos rotation, ~20 lines

**Validation**: Parity task. Mamba2 should fail (~50%), Mamba2+RoPE should
solve it (>99%). This is the litmus test from the Mamba3 paper.

### Step 2.2: Trapezoidal discretization (configurable)
Replace Euler with trapezoidal rule, remove conv1d.

Changes to `mamba_block.py`:
- Add `DiscretizationRule` enum and `discretize()` method
- Trapezoidal: store previous B*x at chunk boundaries, compute weighted sum
- Remove conv1d (trapezoidal makes it redundant)
- Add `lambda_proj`: Linear(d_inner → nheads) for data-dependent λ_t

**Design**: The discretization rule is a pluggable component. Each rule
implements `compute_input_coupling(dt, A, B, x, prev_state) → coupling_term`.
This makes it trivial to swap in ETD, AB3, or learned rules later.

### Step 2.3: Language modeling on WikiText-2
First real benchmark. Compare ePC-Mamba vs backprop-Mamba2 baseline.

Files:
- `train_text.py` — Token-level language modeling
  - WikiText-2 dataset (~2M tokens, ~12MB)
  - BPE tokenizer (or character-level for simplicity)
  - Perplexity evaluation
  - Profiling infrastructure (port from ResNet experiment)

Config: d_model=256, 4 layers, seqlen=256, batch=16

**Success criteria**: Perplexity within 2x of backprop baseline.

---

## Phase 3: BTT Projections

**Goal**: Reduce compute cost of the dense projections that dominate Mamba.

### Step 3.1: BTT linear layer
Implement Block Tensor-Train structured linear layer.

Files:
- `btt_linear.py` — BTTLinear(d_in, d_out, rank=1, cores=2)
  - Store two core tensors L, R instead of full weight matrix
  - Forward: sequence of batched MVMs (reshape → matmul → reshape)
  - Structure-aware init: σ_L, σ_R from BTT paper Section 3.2
  - Adam LR multipliers: κ_L = √d/(2r), κ_R = √d/2
  - Weight normalization (from BTT paper Section 5.1): prevent
    unbounded activation growth in deep models

**Validation**: BTTLinear(256, 256, rank=1) produces same-shape output as
nn.Linear(256, 256). Verify gradient flow works with autograd.

### Step 3.2: Replace Mamba projections with BTT
Swap in_proj and out_proj for BTTLinear.

Changes to `mamba_block.py`:
- `in_proj`: BTTLinear(d_model, d_in_proj, rank=1)
- `out_proj`: BTTLinear(d_inner, d_model, rank=1)
- Verify training still converges on synthetic tasks
- Profile: measure actual speedup at d_model=256 vs dense

At d_model=256: expect ~8x fewer FLOPs per MVM, but overhead may eat
into this. Real wins start at d_model ≥ 1024.

### Step 3.3: Scale up
Increase d_model and measure where BTT crossover happens.

- d_model=512: BTT should show measurable speedup
- d_model=1024: BTT should clearly win (BTT paper's sweet spot)
- Profile ePC-Mamba at each scale: how much does BTT reduce the 3.5x overhead?

---

## Phase 4: Advanced Optimizations (Future)

### 4.1: Alternative discretization rules
Swap in ETD, AB3, or learned rules using the configurable architecture.
Compare perplexity vs training cost on WikiText-2.

### 4.2: PPCA curvature tracking for Newton step
Use LRPD Riccati paper's approach to evolve a rank-p Hessian approximation
across batches. Goal: push T=2 → T=1 on most batches.

### 4.3: Share subspaces for continual learning
When moving from synthetic → text → code → math, use evolving shared
BTT core subspaces to prevent catastrophic forgetting.

### 4.4: Scale to 120-500M parameters
d_model=1024-2048, 24-32 layers. BTT essential at this scale.
WikiText-103 or OpenWebText for training data.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────┐
│                    ePC-Mamba Model                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Input embedding (token → d_model)                  │
│  ┌───────────────────────────────────────────────┐  │
│  │ RMSNorm → Mamba Block 0 → + e_0 (fp32)       │  │
│  │ RMSNorm → Mamba Block 1 → + e_1 (fp32)       │  │
│  │ ...                                           │  │
│  │ RMSNorm → Mamba Block N-1 → + e_{N-1} (fp32) │  │
│  └───────────────────────────────────────────────┘  │
│  RMSNorm → Output projection (d_model → vocab)     │
│                                                     │
│  Each Mamba Block:                                  │
│  ┌───────────────────────────────────────────────┐  │
│  │ in_proj (BTT): d_model → [z, x, B, C, dt]    │  │
│  │ [conv1d] — removed with trapezoidal           │  │
│  │ Data-dependent RoPE on B, C                   │  │
│  │ SSD scan (chunked, pure PyTorch)              │  │
│  │ RMSNorm + SiLU gate                           │  │
│  │ out_proj (BTT): d_inner → d_model             │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  Training (per batch):                              │
│  1. Init zero errors (shape-cached)                 │
│  2. Inference: T=2 Newton iterations on errors      │
│     - Forward: compute E(x, y)                      │
│     - Backward: error gradients                     │
│     - Newton step: streaming rank-1 Woodbury        │
│  3. Weight update: E_local + Adam                   │
│     - Structure-aware LR for BTT cores              │
│                                                     │
│  Discretization: configurable via DiscretizationRule │
│  Default: trapezoidal (Mamba3). Swappable to ETD,  │
│  AB3, learned, or custom rule.                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## File Structure
```
experiments/ePC_Mamba/
├── NOTES.md           — Research notes and analysis (this exists)
├── ROADMAP.md         — This file
├── mamba_block.py     — Mamba2 block (pure PyTorch)
├── btt_linear.py      — BTT structured linear layer
├── epc_model.py       — PCESequence (ePC wrapper for sequence models)
├── architectures.py   — Model configs at different scales
├── train_synthetic.py — Synthetic task training
├── train_text.py      — Language modeling training
└── data/              — Downloaded datasets
```
