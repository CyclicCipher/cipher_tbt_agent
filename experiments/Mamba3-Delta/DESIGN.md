# Mamba3-Delta Design

A hybrid architecture combining Mamba3's continuous-time SSM dynamics with
the delta rule's targeted write/erase memory, MIMO for hardware efficiency,
and surprise-gated memory updates.

## Motivation

Mamba3 and Gated DeltaNet are the two strongest sub-quadratic sequence models
(Table 1 in the Mamba3 paper shows them neck-and-neck at all scales). Each
has strengths the other lacks:

| Mamba3 has | Gated DeltaNet has |
|---|---|
| Continuous-time dynamics (exp(Δ·A)) | Targeted write/erase (delta rule) |
| Complex eigenvalues (PoPE/RoPE) | Per-key associative memory |
| Trapezoidal discretization (2nd order) | Mature chunkwise WY parallelism |
| MIMO (higher arithmetic intensity) | Per-channel decay (via KDA variant) |
| State tracking (parity, modular arith) | State tracking (via Householder products) |

Mamba3-Delta unifies both lineages, keeping Mamba3's continuous-time core
and adding the delta rule's memory management.

## Mathematical Formulation

### State Shape

Per head: `S_t ∈ R^{headdim × d_state}` (same as Mamba3's h_t).

With MIMO (rank r): B_t ∈ R^{d_state × r}, C_t ∈ R^{d_state × r},
X_t ∈ R^{headdim × r}. The state remains R^{headdim × d_state} — MIMO
does not grow the state.

### Core Recurrence (SISO, for clarity)

**Mamba3 (current):**
```
h_t = exp(Δ·A) · h_{t-1} + Δ · x_t ⊗ B_t
```

**Mamba3-Delta (proposed):**
```
h_t = Diag(α_t) · h_{t-1} · (I - β₁_t · B̂₁_t · B̂₁_t^T) · (I - β₂_t · B̂₂_t · B̂₂_t^T)
      + β₁_t · (Δ · x_t) ⊗ B₁_t + β₂_t · (Δ · x_t) ⊗ B₂_t
```

Where:
- `Diag(α_t)` — per-channel diagonal decay, replaces scalar `exp(Δ·A)`
- `B̂₁_t, B̂₂_t` — L2-normalized PoPE-derived orthogonal key pair
- `β₁_t, β₂_t` — surprise-modulated write/rotation gates
- The two Householder terms compose into a rotation when both β > 0

### Per-Channel Decay (KDA-Style)

Replace Mamba3's scalar `exp(Δ·A)` with per-channel diagonal:

```
z_t = W_α · x_t + b_α                  (linear projection, d_state outputs)
α_t = sigmoid(z_t)                      (per-channel, in (0,1))
```

Each dimension of the state decays at its own learned, input-dependent rate.
This implicitly creates multi-scale temporal dynamics: some channels learn
α≈1 (long memory), others α≈0 (short memory).

**StableSSM reparameterization option** (for stable long-range memory):
```
α_t = 1 - 1/(z_t² + 0.5)              (gradient slows as α→1)
```

**Design decision:** We start with sigmoid (standard, well-understood) and
compare to StableSSM reparameterization experimentally.

**Continuous-time interpretation:** The per-channel decay `α_t` can be viewed
as `exp(Δ_t · A_channel)` where each channel has its own effective A. This
preserves the continuous-time interpretation while gaining multi-scale dynamics.

### PoPE-Derived Orthogonal Key Pair

PoPE encodes B_raw ∈ R^{d_state//2} into B ∈ R^{d_state} as:
```
B₁ = (μ·cos(θ), μ·sin(θ))             where μ = softplus(B_raw + δ)
```

The orthogonal partner (π/2 rotation) is free:
```
B₂ = (-μ·sin(θ), μ·cos(θ))            B₁ · B₂ = 0 by construction
```

Properties:
- Two Householder reflections about orthogonal axes compose into a rotation
- No additional projection weights needed (B₂ derived from B₁)
- Only β₂ is an additional learned parameter (scalar)
- The model learns when to reflect (β₁>0, β₂≈0), rotate (both>0), or skip (both≈0)

### MIMO (Multi-Input Multi-Output)

From the Mamba3 paper, Appendix D. MIMO changes B, C from vectors to
rank-r matrices, changing the state update from an outer product (rank-1) to
a matrix product (rank-r):

**SISO:** `h_t = α_t · h_{t-1} + Δ · (b_t ⊗ x_t)`  — rank-1 update
**MIMO:** `H_t = α_t · H_{t-1} + B_t · X_t^T`       — rank-r update

Where:
- `B_t ∈ R^{N×r}` (N = d_state, r = MIMO rank)
- `X_t ∈ R^{P×r}` (P = headdim)
- `C_t ∈ R^{N×r}`
- `Y_t = H_t^T · C_t ∈ R^{P×r}`

The state `H_t ∈ R^{N×P}` remains the same size. MIMO increases arithmetic
intensity (FLOPs/byte) without growing the state, pushing decode from
memory-bound to compute-bound.

Additional projections needed:
```
X'_t = W_X' · U_t                       (d_model → headdim)
X_t  = W_X  · X'_t                      (headdim → headdim × r)
```

Similarly for output down-projection and residual Z stream.

**MIMO + delta rule interaction:** The Householder erase uses the first
MIMO column (B₁[:,0,:]) as the key direction. This keeps one clean erase
direction regardless of MIMO rank — additional MIMO columns increase
write/read rank, not erase directions. The erase is about removing old
state in the key direction; MIMO's additional columns are about writing
richer associations.

**Implementation:** Write uses rank-r einsum `Σ_i x_write[:,:,:,i] ⊗ B[:,:,i,:]`
that contracts over the MIMO rank. Read produces r separate readouts
`y[:,:,:,i] = h · C[:,:,i,:]`, which are linearly contracted via
`mimo_out_proj` back to d_inner. For SISO (r=1), the einsum degenerates
correctly to the standard rank-1 outer product.

### Surprise-Modulated Write Gates

```
surprise_t = sg(-log p(x_t | x_{<t}))   (training: cross-entropy, stop-grad)
           = sg(D_KL(p_t || p̄_t))       (inference: KL from EMA)

β₁_t = σ(W_β₁ · x_t + w_s₁ · surprise_t + b_β₁)
β₂_t = σ(W_β₂ · x_t + w_s₂ · surprise_t + b_β₂)
```

High surprise → large β → strong erase+write (store the unexpected).
Low surprise → small β → near-identity (skip the predictable).

The surprise signal is stop-gradiented to avoid circular optimization.

### Trapezoidal Discretization

Retained from Mamba3 for the input terms. The trapezoidal rule blends
current and previous inputs:

```
input_t = λ_t · (B_t ⊗ x_t) + (1-λ_t) · exp(A) · (B_{t-1} ⊗ x_{t-1})
```

Where λ_t = σ(u_t) is data-dependent.

**Integration with delta rule:** The trapezoidal blending applies to the
WRITE term, not the erase term. The erase operates on the state based on
the current key only (you erase what you're about to overwrite, not what
the previous token wrote).

### Output

Same as Mamba3:
```
y_t = C_t^T · h_t                       (readout from state)
y_t = y_t + D · x_t                     (skip connection)
out_t = OutNorm(y_t * SiLU(z_t))        (gated output)
out_t = W_out · out_t                    (project to d_model)
```

## KDA Lessons Applied

From the Kimi Delta Attention paper:

1. **Per-channel diagonal decay** — adopted (see above)
2. **Parameter tying for DPLR efficiency** — adopted. Our Householder
   directions B̂₁, B̂₂ are both derived from B (the key), not independent
   parameters. This is the same constraint KDA uses.
3. **Low-rank MLP for α generation** — adopted for parameter efficiency.
   `α_t = sigmoid(W_up · (W_down · x_t))` with W_down projecting to a
   bottleneck dimension.
4. **NoPE on non-SSM layers** — if we ever add attention layers, they can
   skip positional encoding since the SSM layers handle position via PoPE.

## Comparison to Alternatives

| Property | Mamba3 | Gated DeltaNet | KDA | Mamba3-Delta |
|---|---|---|---|---|
| Continuous-time dynamics | Yes | No | No | **Yes** |
| Complex eigenvalues (PoPE) | Yes | No | No | **Yes** |
| Trapezoidal discretization | Yes | No | No | **Yes** |
| Delta rule (erase+write) | No | Yes | Yes | **Yes** |
| Per-channel decay | No (scalar) | No (scalar) | Yes | **Yes** |
| Rotation (n_h=2 Householder) | No | No | No | **Yes (free via PoPE)** |
| MIMO | Yes | No | No | **Yes** |
| Surprise gating | No | No | No | **Yes** |
| StableSSM option | No | No | No | **Yes** |

## Implementation Phases

### Phase 1: Mamba3 + MIMO + Basic Delta Rule ✓
- Port mamba3_block.py to new file ✓
- Add MIMO projections (B, C, X as rank-r matrices) ✓
- Add delta rule erase with single Householder (β₁ only) ✓
- Per-channel decay (KDA-style) ✓
- Proper MIMO recurrence (rank-r write/read via einsum) ✓
- Naive sequential implementation (no chunkwise parallelism yet)
- Test on Stage 1b tasks

### Phase 2: PoPE Orthogonal Pair
- Derive B₂ from PoPE
- Add second Householder (β₂)
- Test rotation capability on state-tracking tasks

### Phase 3: Per-Channel Decay
- Replace scalar decay with per-channel diagonal α_t
- Compare sigmoid vs StableSSM reparameterization
- Verify multi-scale emergence (visualize per-channel α distributions)

### Phase 4: Surprise Gating
- Add surprise computation (cross-entropy at each position)
- Modify β₁, β₂ to incorporate stop-gradiented surprise
- Test: does surprise gating reduce memory waste on predictable tokens?

### Phase 5: Chunkwise Parallelism
- Implement WY representation for the Householder products
- Integrate with SSD-style chunk processing
- Benchmark training speed vs Mamba3

### Phase 6: KL Divergence for Inference
- Implement EMA of predictive distribution
- Top-k KL computation
- Test inference-time surprise gating

## VRAM Budget (4GB RTX 3050 Ti)

With d_model=128, d_state=64, headdim=64, nheads=4, n_layer=4:
- State per head: 64×64 = 4K elements × 2 bytes = 8KB
- Total state: 4 heads × 4 layers × 8KB = 128KB (negligible)
- MIMO r=4 adds ~4× to B, C, X projections but not to state
- Delta rule adds β₁, β₂ scalars + B₂ derivation (negligible)
- Per-channel α adds one linear projection per layer (small)

Main VRAM cost remains the same as Mamba3: model weights + activations.
The delta rule adds ~0 parameters (B₂ is derived, β are scalars).
Per-channel α adds d_state parameters per layer.

## Key References

- Mamba3: ICLR 2026 submission, OpenReview HwCvaJOiCj
- Gated DeltaNet: Yang et al., ICLR 2025, arXiv:2412.06464
- DeltaProduct: Siems et al., NeurIPS 2025, arXiv:2502.10297
- KDA / Kimi Linear: arXiv:2510.26692
- StableSSM: Wang & Li, ICML 2024, arXiv:2311.14495
- PoPE: Gopalakrishnan et al. 2024
