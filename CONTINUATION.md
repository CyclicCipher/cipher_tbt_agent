# CONTINUATION: Implement Real WY Chunkwise Parallelism for Naja

**Date:** 2026-02-14
**Priority:** IMMEDIATE — Naja is unusably slow without this.
**Context:** See Mistake #39 in MISTAKES.md.

---

## The Problem

Naja's sequential recurrence (`delta_recurrence()` in `naja.py:114-211`) launches ~20 CUDA kernels per timestep. At seq_len=64, n_layer=4, that's ~5,120 kernel launches per batch with ~30us overhead each, giving ~150ms of pure launch latency. GPU utilization reads 0%. Training appears to hang.

The current `delta_recurrence_chunkwise()` (naja.py:294-344) is **not real parallelism** — it just wraps the same sequential loop in `torch.utils.checkpoint`. It trades memory for compute but doesn't reduce sequential steps at all.

## The Solution: WY Chunkwise Parallelism

Replace the sequential per-timestep loop with the 4-step WY chunkwise algorithm from Gated DeltaNet (Yang et al., ICLR 2025). This reduces sequential steps from L to L/C (e.g., 64→1 with chunk_size=64).

---

## Mathematical Background

### WY Representation

A product of Householder reflections H_i = I - β_i·k_i·k_i^T can be written as:

```
P_n = prod_{t=1}^{n} H_t = I - Σ_{i=1}^{n} w_i · k_i^T
```

where the pseudo-keys w_i are computed inductively:
```
w_1 = β_1 · k_1
w_{t+1} = β_{t+1} · (k_{t+1} - Σ_{i=1}^{t} (k_i^T · k_{t+1}) · w_i)
       = β_{t+1} · P_t · k_{t+1}
```

### UT Transform (Matrix Form for a Chunk of C Tokens)

```
A = tril(diag(β) · K · K^T, -1)       # C×C strictly lower triangular
T = (I + A)^{-1} · diag(β)             # via forward substitution, O(C²)
W = T · K                               # C×d_k pseudo-keys
U = T · V                               # C×d_v pseudo-values
```

### The 4-Step Algorithm

For each chunk c (parallel across chunks):

**Step 1 — Intra-chunk WY:** Compute W_c, U_c via UT transform above.

**Step 2 — Chunk state accumulation:**
```
P_c = I - K_c^T · W_c        # d_k × d_k transition for this chunk
H_c = K_c^T · U_c            # d_k × d_v state contribution from this chunk
```

**Step 3 — Inter-chunk scan** (sequential over L/C chunks, much smaller than L):
```
S_c = γ_c · P_c · S_{c-1} + H_c    # γ_c = cumulative decay within chunk
```

**Step 4 — Intra-chunk output:**
```
O_c = Q_c · S_{c-1} + tril(Q_c · K_c^T) · (U_c - W_c · S_{c-1})
```

---

## Naja-Specific Complications

### 1. Two Householder Reflections (PoPE Pair)

Naja uses B₁ AND B₂ (orthogonal pair from PoPE). This is equivalent to DeltaProduct with n_h=2. Two approaches:

**Option A — Virtual token expansion (DeltaProduct approach):**
Expand each real token into 2 virtual tokens (one for B₁/β₁, one for B₂/β₂). Apply standard DeltaNet WY to the 2C-length virtual sequence. Read outputs at every 2nd position. Simple but doubles chunk size.

**Option B — Compose two WY transforms per chunk:**
Apply the WY transform twice within each chunk — first for B₁, then for B₂. More efficient but more complex.

**Recommendation:** Start with Option A (virtual expansion). It's simpler and reuses the standard WY algorithm directly. Optimize to Option B later if needed.

### 2. Per-Channel Decay α_t (Diagonal, Not Scalar)

Naja's decay is per-channel: `α_t ∈ R^{d_state}`, not scalar like Gated DeltaNet. The cumulative decay between positions i and j within a chunk is:
```
Γ_{i,j} = prod_{k=j+1}^{i} diag(α_k)    # d_state × d_state diagonal
```

This modifies the A matrix in the UT transform:
```
A_{i,j} = β_j · (k_j^T · Γ_{i,j} · k_i)    # scalar, for i > j
```

For diagonal Γ, this is still efficient: `k_j^T · diag(γ) · k_i = Σ_d (k_j[d] · γ[d] · k_i[d])`.

**Simplification option for first implementation:** Temporarily fall back to scalar decay (mean of α channels) for the chunkwise path, keeping per-channel only for the reference sequential path. Gets the speedup working, then add per-channel later.

### 3. Trapezoidal Discretization

Naja blends current and previous inputs via λ:
```
write_t = λ_t · (x_t ⊗ B_t) + (1-λ_t) · (x_{t-1} ⊗ B_{t-1})
```

In the WY framework, this changes the "value" V at each position to the blended write. Doesn't change the algorithm structure — just the input V that goes into `U = T · V`.

### 4. MIMO (rank-r)

For a first implementation, restrict to SISO (r=1) for the WY chunkwise path. The Householder erase uses a single key direction regardless of MIMO rank, so the WY transform applies cleanly. MIMO can be added after the core WY is working.

---

## Implementation Plan

### Phase 5a: Pure PyTorch WY (No Triton) ✓ COMPLETE & VERIFIED

Implemented the WY chunkwise algorithm in pure PyTorch. **Numerically verified** on GPU: all 5 test cases pass at ~1e-6 max diff vs naive sequential reference.

File modified: `experiments/Naja/naja.py`

**Functions added:**

1. `_chunk_scaled_dot_kkt(K, beta)` → Compute A = tril(diag(β)·K·K^T, -1) per chunk ✓
2. `_solve_tril(A, beta)` → Forward substitution for (I+A)^{-1}·diag(β) using `torch.linalg.solve_triangular` ✓
3. `_prepare_wy_repr(K, V, beta)` → Full UT transform: A → T → W, U ✓
4. `delta_recurrence_wy(x_write, ..., chunk_size)` → Complete 4-step chunkwise algorithm ✓

**Standalone test:** `test_wy_minimal.py` — 5 test cases (constant decay, no decay, no delta, multi-chunk, full strength). All passing.

**Three bugs found and fixed during verification (Mistake #40):**
1. Decay-before-erase convention mismatch in A matrix
2. Wrong inter-chunk state update (used P@S+H instead of FLA-style v_new + K_fwd)
3. Pseudo-keys W missing cumulative decay factor (need `W_state = T @ (K * exp(cumsum))`)

**Wired into `NajaMixer.forward()`** via `use_wy_chunkwise` config flag ✓
**CLI flag:** `--use_wy_chunkwise` in train_naja.py ✓
**Diagnostics:** `diagnose.py` updated with WY timing and correctness tests ✓

**Phase 5a simplifications (to be lifted in Phase 5b):**
- SISO only (r=1 MIMO column for erase key)
- Single Householder (B1 only, B2/PoPE pair ignored)
- Scalar decay (mean of per-channel α, not full diagonal)
- Trapezoidal blending baked into V before WY transform

### Phase 5b: Per-Channel Decay ← NEXT PRIORITY

Lift the scalar decay simplification. Per-channel decay means each of the `d_state` channels gets its own decay rate `alpha_t[k]`.

**What changes:**
- `log_alpha_cumsum` shape: `(batch, nheads, n_chunks, Cs)` → `(batch, nheads, n_chunks, Cs, d_state)`
- A matrix: `A[i,j] = beta_j * decay(i,j) * (k_i · k_j)` becomes per-channel: `A[i,j,k] = beta_j * decay_k(i,j) * k_i[k] * k_j[k]`. This is no longer a simple CxC matrix but CxCxd_state.
- Causal decay mask and forward-decay factors broadcast over `d_state`
- The UT forward substitution becomes d_state independent scalar forward substitutions (one per channel), NOT a single matrix solve

**KDA reference:** KDA (arXiv:2510.26692) implements exactly this. Their A matrix is per-channel, and they handle it by solving d_state independent lower-triangular systems. The FLA implementation in `fla/ops/kda/` is the reference.

**Note on A matrix convention:** Our A uses `beta_j` (column index) while FLA uses `beta_i` (row index). For constant beta these are equivalent. For variable beta, we may need to reconcile — but this doesn't affect per-channel decay, which is orthogonal to the beta convention.

### Phase 5c: PoPE Pair (B₁, B₂)

Add virtual token expansion for the second Householder (DeltaProduct with n_h=2). Each real token becomes 2 virtual tokens. Standard WY applies to the 2C-length virtual sequence.

### Phase 5d: Ablation Testing

Run ablation experiments on GPU to validate each Naja component:
- Per-channel decay vs scalar decay
- PoPE pair vs single Householder
- Surprise gating vs fixed beta
- MIMO rank-r vs SISO

Complete all ablations before investing in Triton optimization.

### Phase 5e: Triton Kernels (After Ablations)

For maximum performance, either:
- Port core operations to Triton (using `flash-linear-attention` as reference)
- Or use `flash-linear-attention` as a dependency and import its kernels directly

Expected gain: 3-10x over PyTorch WY (from fusing intra-chunk operations and eliminating intermediate memory traffic).

---

## Reference Implementations

### SSD chunkwise (already in codebase)

`experiments/Mamba3/mamba3_block.py:ssd_trapz()` (lines 143-229) implements the same 4-step structure for diagonal state transitions. Use as structural template — the steps map 1:1.

### flash-linear-attention library (external reference)

```
fla/ops/delta_rule/wy_fast.py           # WY Triton kernels
fla/ops/delta_rule/chunk.py             # Main orchestrator
fla/ops/delta_rule/naive.py             # Reference sequential implementation
fla/ops/common/solve_tril.py            # Forward substitution Triton kernels
fla/ops/common/chunk_scaled_dot_kkt.py  # Key-key interaction computation
fla/ops/common/chunk_delta_h.py         # Inter-chunk hidden state propagation
```

### DeltaNet naive reference (for verification)

```python
def delta_rule_recurrence(q, k, v, beta):
    S = torch.zeros(b, h, d_k, d_v)
    for i in range(l):
        _k = k[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)  # error = v - S^T k
        _v = _v * beta_i                                # scaled error
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)  # rank-1 update
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    return o, S
```

---

## Mapping Naja's Current Recurrence to DeltaNet

Naja's current `delta_recurrence()` (naja.py:114-211) does per timestep:

```python
# Decay (per-channel)
h = h * α_t                        # ← Gated DeltaNet's scalar α, but diagonal

# Erase (Householder 1)
h = h - β₁ · (h · b̂₁) ⊗ b̂₁       # ← DeltaNet's I - β·k·k^T

# Erase (Householder 2, PoPE pair)
h = h - β₂ · (h · b̂₂) ⊗ b̂₂       # ← DeltaProduct's second Householder

# Write (rank-r MIMO)
h = h + β₁ · Σᵢ x_write[:,i] ⊗ B₁[:,i]  # ← DeltaNet's β·v·k^T

# Readout
y_t = h · C                        # ← DeltaNet's q^T · S
```

To map to DeltaNet's WY formulation:
- **k** = B̂₁ (normalized first MIMO column of B₁)
- **v** = write contribution (x_write contracted with B₁)
- **β** = β₁ (write/erase gate)
- **q** = C (readout projection)
- **α** = per-channel decay (diagonal, not scalar)
- B₂/β₂ = second Householder (handle via virtual expansion or composition)

---

## Key Constraints

- **Do NOT run full training on Claude's machine** (Mistake #36). Implement, commit, push. User tests on GPU.
- **Read the DeltaNet paper thoroughly** before implementing (Mistake #13). Especially Section 3 (chunkwise algorithm) and Appendix B (WY derivation) of arXiv:2406.06484.
- **Verify correctness first** by comparing WY output vs naive sequential output on small inputs.
- **Keep the naive sequential implementation** as permanent reference for correctness testing.
- **4GB VRAM budget** — the WY algorithm should use LESS memory than naive (no per-timestep activation storage).

## Papers to Read (in order)

1. **Yang et al. 2024** — "Parallelizing Linear Transformers with the Delta Rule" (arXiv:2406.06484). THE paper for the WY algorithm. Read Sections 2-3 and Appendix B completely.
2. **Yang et al. 2025** — "Gated Delta Networks" (arXiv:2412.06464). Extends WY to include data-dependent decay (our α). Read Section 3.
3. **Siems et al. 2025** — "DeltaProduct" (arXiv:2502.10297). Multiple Householders per token via virtual expansion (our B₁/B₂ PoPE pair). Read Section 3.
4. **Songlin Yang's blog** — "DeltaNet Explained Part II" (sustcsonglin.github.io/blog/2024/deltanet-2/). Clear walkthrough of the WY algorithm with code.

## Success Criteria

### Phase 5a ✓ ACHIEVED
1. ✅ `delta_recurrence_wy()` matches `delta_recurrence()` to ~1e-6 (target was <1e-4)
2. ⬜ Training time per epoch drops by >5x on GPU (not yet benchmarked with training)
3. ⬜ GPU utilization rises from ~0% to >50% (not yet benchmarked with training)
4. ⬜ No accuracy regression on Stage 1b task (needs training run)
5. ✅ test_wy_minimal.py provides standalone correctness verification

### Phase 5b
6. Per-channel decay WY matches naive per-channel reference to <1e-4
7. All test_wy_minimal.py tests updated and passing with per-channel decay

### Phase 5d
8. Ablation results documented for each Naja component
9. Architecture decisions finalized based on empirical evidence

---

## Key Files to Read First

| File | What's in it |
|------|-------------|
| `MISTAKES.md` | 39 documented mistakes — **always read first** |
| `CLAUDE.md` | Architecture overview, priorities |
| `experiments/Naja/naja.py` | Current Naja model (delta_recurrence, delta_recurrence_chunkwise) |
| `experiments/Naja/DESIGN.md` | Full mathematical specification of Naja |
| `experiments/Naja/train_naja.py` | Training loop, CLI args |
| `experiments/Naja/diagnose.py` | Diagnostic suite (timing, correctness) |
| `experiments/Mamba3/mamba3_block.py` | SSD chunkwise reference (ssd_trapz, lines 143-229) |
