# BTT-Mamba3 — Design Document

**Date:** 2026-02-18
**Status:** Design phase. No code yet.
**Context:** The CTKG defines 47 stages from counting to solving damped harmonic oscillators, plus 21 stages for logic to category theory. Reaching these goals likely requires scaling beyond d=128 (current Mamba3), but our hardware is an RTX 3050 Ti with 4GB VRAM. Block Tensor-Train (BTT) decomposition could let us train effectively wider models within this budget.

**Reference:** "Compute Better Spent: Replacing Dense Layers with Structured Matrices" (docs/research/BTT paper.pdf)

---

## The Problem

Parameter and optimizer memory for dense models at scale:

| d_model | Params (fp16) | Adam states (fp32) | Total model | Remaining for activations |
|---------|---------------|---------------------|-------------|--------------------------|
| 128     | 2.4 MB        | 9.6 MB              | 12 MB       | ~3.99 GB (plenty)        |
| 512     | 38 MB         | 153 MB              | 191 MB      | ~3.81 GB (fine)          |
| 1024    | 153 MB        | 610 MB              | 763 MB      | ~3.24 GB (tight)         |
| 2048    | 610 MB        | 2.4 GB              | 3.0 GB      | ~1.0 GB (critical)       |

At d=2048 dense, activations get ~1GB — not enough for reasonable sequence lengths with backprop. Gradient checkpointing helps but adds ~33% training time.

With BTT (c=2, r=4), parameter count drops by ~d/(8r) = ~d/32 for each d×d layer:

| d_model | Dense params | BTT params | Adam states (BTT) | Total model (BTT) |
|---------|-------------|------------|--------------------|--------------------|
| 512     | 38 MB       | 2.4 MB     | 9.5 MB             | 12 MB              |
| 1024    | 153 MB      | 9.5 MB     | 38 MB              | 48 MB              |
| 2048    | 610 MB      | 38 MB      | 153 MB             | 191 MB             |

At d=2048 with BTT: model takes ~191MB, leaving ~3.8GB for activations. The scaling wall disappears.

---

## BTT Recap

A dense matrix W ∈ ℝ^(d_out × d_in) is replaced by a product of c "core" tensors. For c=2, rank r:

- **Parameters**: O(r × d) instead of O(d²)
- **FLOPs per MVM**: O(r × d^(3/2)) instead of O(d²)
- c=2, r=1 is equivalent to Monarch matrices with √d blocks
- c=2, r=4 matches GPT-2 Small quality at 2.7× fewer FLOPs

Key insight from the paper: BTT has a **steeper scaling law exponent**. More performance per FLOP at large scale.

---

## Mamba3 Layer Inventory

Linear layers in Mamba3LM and their dimensions at various d_model:

### Per Mamba3Mixer (4 layers in default config)

| Layer | Dimensions | At d=128 | At d=512 | At d=1024 |
|-------|-----------|----------|----------|-----------|
| `in_proj` | d → d_in_proj* | 128×~325 | 512×~1285 | 1024×~2565 |
| `out_proj` | d_inner → d | 256×128 | 1024×512 | 2048×1024 |

*d_in_proj = 2×d_inner + 2×d_bc×r + nheads + d_state/2 + 1

### Per SwiGLU MLP (4 layers in default config)

| Layer | Dimensions | At d=128 | At d=512 | At d=1024 |
|-------|-----------|----------|----------|-----------|
| `gate_proj` | d → d_ff | 128×512 | 512×2048 | 1024×4096 |
| `up_proj` | d → d_ff | 128×512 | 512×2048 | 1024×4096 |
| `down_proj` | d_ff → d | 512×128 | 2048×512 | 4096×1024 |

### Global

| Layer | Dimensions | Notes |
|-------|-----------|-------|
| `embedding` | vocab×d | Small vocab (~21 tokens). NOT a BTT candidate. |
| `out_proj` | d→vocab | Small. NOT a BTT candidate. |

### BTT Candidates (by parameter count at d=1024)

1. **SwiGLU MLP layers** (gate, up, down): 3 × d × 4d = 12d² per block. **Largest by far.** 4 blocks = 48d². At d=1024: ~50M params dense, ~1.6M with BTT.
2. **Mixer out_proj**: d_inner × d = 2d² per block. 4 blocks = 8d². At d=1024: ~8.4M dense, ~262K BTT.
3. **Mixer in_proj**: d × d_in_proj. Non-square, but the largest axis is ~2.5d. 4 blocks ≈ 10d². At d=1024: ~10.5M dense, ~328K BTT.

**SwiGLU MLPs are the #1 target.** They contain ~60% of all linear layer parameters.

### NOT BTT candidates

- **Embedding / output projection**: Small (vocab_size × d, vocab is ~21). Dense is fine.
- **RMSNorm weights**: Diagonal, already O(d).
- **SSM parameters** (A, D, dt_bias): Per-head scalars, tiny.
- **B_bias, C_bias**: Channel-wise, O(d).
- **SSD core**: Structured computation (not a linear layer), leave as-is.

---

## The Small-Scale Runtime Problem

The paper shows BTT is slower than dense below ~10^7 FLOPs per layer due to:

1. **Kernel launch overhead**: BTT replaces 1 GEMM with 2+ smaller GEMMs. Each has fixed launch cost.
2. **Tensor core underutilization**: Small matrices don't fill the GPU's compute units.
3. **Memory access patterns**: Multiple small tensors vs one contiguous block.

This is NOT a mathematical limitation — the FLOPs are genuinely fewer. It's a GPU execution efficiency problem.

### Potential solutions

**A. Fused Triton kernel** (most promising)

Write a single Triton kernel that does the entire BTT contraction without intermediate materializations. This eliminates kernel launch overhead (problem 1) and lets us control memory access (problem 3).

For c=2, the BTT MVM is:
```
y = G_2 × (G_1 × x_reshaped)
```
where G_1 ∈ ℝ^(r×√d×√d) and G_2 ∈ ℝ^(√d×√d×r). A single fused kernel:
1. Reshapes x from (d,) to (√d, √d)
2. Contracts with G_1 (small matmul)
3. Contracts with G_2 (small matmul)
4. Reshapes output to (d,)

All in one kernel, all in shared memory. No intermediate global memory writes.

We already have Triton infrastructure (triton_ssd.py with graceful fallback). Same pattern.

**B. Batch across sequence positions**

Instead of computing the BTT MVM for each token independently, batch all L tokens together. The BTT contraction becomes a batched operation where the batch dimension is L (sequence length), which is large enough (32-128) to fill the GPU even when √d is small.

```
X ∈ ℝ^(L, d)  →  reshape to (L, √d, √d)  →  batched matmul with G_1  →  batched matmul with G_2
```

The batch dimension L amortizes the small-matrix penalty. This is essentially how nn.Linear already works — it uses batched GEMM. The BTT version just has two batched GEMMs instead of one.

**C. Hybrid: BTT for large layers, dense for small ones**

The SwiGLU MLP layers (d → 4d, 4d → d) are 4× larger than the mixer projections. At d=512:
- MLP gate_proj: 512×2048 — BTT wins
- Mixer out_proj: 1024×512 — BTT wins
- Mixer in_proj: 512×~1285 — BTT wins (non-square, but large enough)

Even at d=256, the MLP layers are 256×1024 — reasonable for BTT.

The hybrid approach: replace SwiGLU layers with BTT first (biggest parameter savings), keep mixer projections dense until d is large enough. This minimizes implementation risk.

**D. Alternative decomposition**

Monarch matrices (= BTT c=2, r=1) have an efficient implementation from Dao et al. (2022) that maps to two batched GEMMs with permutations. This is simpler than general BTT and has existing optimized code. We could start with Monarch and upgrade to full BTT (r>1) only if the rank-1 restriction hurts quality.

---

## μP Learning Rate Scaling

The paper derives per-tensor learning rate multipliers (Table 2) essential for BTT performance. Without μP, BTT underperforms. The key rules for c=2:

| Parameter | LR multiplier |
|-----------|--------------|
| Core G_1 (boundary) | base_lr × (1/√d) |
| Core G_2 (boundary) | base_lr × (1/√d) |
| Bias | base_lr |
| Non-BTT layers | standard μP |

Implementation: custom parameter groups in the optimizer, each with its own LR. We already use parameter groups for different layer types (SSM params vs projections), so the infrastructure exists.

---

## Implementation Plan

### Phase 1: BTTLinear module

A drop-in replacement for nn.Linear that uses BTT decomposition internally.

```python
class BTTLinear(nn.Module):
    """Block Tensor-Train linear layer.

    Drop-in replacement for nn.Linear with BTT-decomposed weight.

    Args:
        in_features: input dimension (must be a perfect square for c=2)
        out_features: output dimension (must be a perfect square for c=2)
        rank: BTT rank (r=1 → Monarch, r=4 → paper's GPT-2 config)
        bias: whether to include bias
    """
    def __init__(self, in_features, out_features, rank=4, bias=False):
        ...
        # Core tensors G_1, G_2
        # Initialize per BTT paper (spectral init + μP scaling)

    def forward(self, x):
        # Reshape → contract G_1 → contract G_2 → reshape
        # Batched across sequence dimension for GPU efficiency
        ...
```

### Phase 2: Mamba3 integration

Add `--btt` flag to Mamba3Config:
```python
use_btt: bool = False
btt_rank: int = 4
btt_layers: str = 'mlp'  # 'mlp', 'mixer', 'all'
```

When enabled, replace selected nn.Linear layers with BTTLinear. All other code (SSD, PoPE, etc.) unchanged.

### Phase 3: μP optimizer setup

Custom parameter group builder:
```python
def build_btt_param_groups(model, base_lr, d_model):
    """Assign μP learning rates to BTT core tensors."""
    ...
```

### Phase 4: Triton kernel (optional, for small-scale efficiency)

Fused BTT contraction kernel. Only needed if Phase 2 benchmarks show the small-scale penalty is actually a bottleneck at our target d_model.

### Phase 5: Benchmarking

Compare at d=256, 512, 1024:
- Dense Mamba3 vs BTT-Mamba3 (r=1 and r=4)
- Metrics: loss at fixed FLOPs, VRAM usage, wall-clock time per epoch
- Use the CTKG arithmetic curriculum (implemented stages) as the benchmark task

---

## Non-Square Matrices

Most of our linear layers are non-square:
- in_proj: d → ~2.5d
- gate_proj: d → 4d
- down_proj: 4d → d

The BTT paper handles this by reshaping to multi-dimensional arrays where each axis doesn't need to be the same size. For c=2 with input d_in = m₁×m₂ and output d_out = n₁×n₂:

```
G_1 ∈ ℝ^(n₁ × m₁ × r)
G_2 ∈ ℝ^(r × n₂ × m₂)
```

For d → 4d (e.g., 512 → 2048), one factorization:
- d_in = 16×32 = 512
- d_out = 32×64 = 2048
- G_1 ∈ ℝ^(32 × 16 × r), G_2 ∈ ℝ^(r × 64 × 32)
- Params: 32×16×r + r×64×32 = 2560r (vs 1,048,576 dense). At r=4: 10,240 params.

We need a helper that finds good factorizations for each layer's dimensions. Factors should be as balanced as possible (close to √d) for compute efficiency.

---

## The CTKG Bootstrap Idea

An amusing long-shot: the CTKG's math path (algebra → calculus → linear algebra) teaches the model the exact mathematics needed to reason about matrix decompositions and optimization. If the model reaches sufficient mathematical maturity, we could potentially:

1. Ask it to analyze structured matrix decompositions
2. Have it derive properties of Kronecker products, tensor contractions
3. Use its mathematical reasoning to explore novel decomposition structures

This is speculative and circular (we need the compression to scale the model, and the model to improve the compression). But if the curriculum works incrementally — each scale-up enabling slightly more mathematical reasoning, enabling slightly better compression for the next scale-up — it could be a virtuous cycle.

Filed under "would be very funny if it worked." Not a dependency for the implementation plan.

---

## Decision Points

Before proceeding with implementation:

1. **Target d_model**: What width should BTT-Mamba3 target? d=512 is conservative and probably sufficient for the first 20-30 CTKG stages. d=1024 stretches further.

2. **Start with Monarch or full BTT?** Monarch (r=1) is simpler, has existing code, and the paper shows it's near-optimal for compute. Full BTT (r=4) is more expressive but harder to implement. Recommendation: start Monarch, upgrade if needed.

3. **Which layers first?** SwiGLU MLP layers are 60% of parameters and the easiest to replace (simple d→4d and 4d→d shapes). Mixer projections are next. Recommendation: MLP first, measure, then expand.

4. **Triton kernel priority?** The small-scale penalty matters at d=256 but not at d=512+. If we target d=512, batched PyTorch may be sufficient without a custom kernel. Recommendation: defer Triton kernel until benchmarks show it's needed.

---

## Open Questions

1. **Interaction with mHC**: Manifold-constrained hyperconnections add gating between residual streams. BTT changes the linear layer internals but not the residual structure, so they should compose. Needs verification.

2. **Interaction with StableSSM**: StableSSM reparameterizes the A-matrix, which is a per-head scalar, not a linear layer. No interaction with BTT. Safe.

3. **Gradient flow through BTT**: The paper shows BTT trains end-to-end with standard backprop, but the gradient path through two small matmuls might behave differently than through one large one. μP scaling is supposed to handle this, but worth monitoring loss curves for instability.

4. **Optimal factorization for Mamba3's dimensions**: The specific dimensions (d=512 → d_inner=1024, d_ff=2048, d_bc varies) need factorizations that are both mathematically valid and GPU-friendly (i.e., the factor dimensions should be multiples of 16 for tensor core alignment).
