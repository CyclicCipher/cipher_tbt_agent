# Architecture Proposal: Evolved Mamba3

Proposal for evolving Mamba3 into a biologically-grounded sequence model with surprise-gated memory, multi-scale states, and a CTKG interface. Working name: **Mnemon** (Greek: "mindful").

See `MEMORY_RESEARCH.md` for the research backing these choices.

## Design Principles

1. **Surprise-driven memory**: Only store what's genuinely novel. Most tokens are predictable — the model should spend near-zero memory on them.
2. **Multi-scale temporal processing**: Different state dimensions decay at different rates, matching the brain's cortical timescale hierarchy.
3. **Clean what/where separation**: PoPE's content-magnitude / position-phase split feeds through to memory addressing.
4. **CTKG as long-term memory**: The neural net handles perception and working memory. The CTKG handles knowledge, constraints, and structured reasoning.
5. **Energy minimization as the unifying framework**: Surprise = high prediction error = high free energy. Memory writes = free energy reduction.

## Architecture Overview

```
Input tokens
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Embedding + PoPE encoding                      │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  N × Mnemon Block                               │
│  ┌───────────────────────────────────────────┐  │
│  │  Surprise-Gated DeltaProduct Mixer        │  │
│  │  • Multi-scale per-channel decay (KDA)    │  │
│  │  • n_h=2-3 Householder steps              │  │
│  │  • β modulated by surprise signal         │  │
│  │  • α with StableSSM reparameterization    │  │
│  │  • PoPE on B, C projections               │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  SwiGLU MLP                               │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  Surprise Computer (per block)            │  │
│  │  • Training: cross-entropy at position t  │  │
│  │  • Inference: D_KL(p_t || p̄_t)           │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  CTKG Interface (future)                        │
│  • Read: query CTKG for relevant knowledge      │
│  • Write: extract novel entities/relations      │
│  • Constraint: energy terms from CTKG priors    │
└─────────────────────────────────────────────────┘
    │
    ▼
  Output logits
```

## Component 1: Surprise-Gated DeltaProduct Mixer

### State Update

Replace Mamba3's `h_t = exp(Δ·A)·h_{t-1} + Δ·B_t⊗x_t` with a Gated DeltaProduct:

```
For j = 1 to n_h:
    S_{t,j} = S_{t,j-1}·(I - β_{t,j}·k_{t,j}·k_{t,j}^T) + β_{t,j}·v_{t,j}·k_{t,j}^T

S_t = diag(α_t)·S_{t,n_h}
y_t = S_t·q_t
```

Where:
- `S_t` ∈ R^{d_v × d_k} is the memory matrix per head
- `k_{t,j}` = L2-normalized keys (distinct per Householder step, from separate projections)
- `v_{t,j}` = values
- `β_{t,j}` = surprise-modulated write gates
- `α_t` = per-channel diagonal decay gate (KDA-style, not scalar)
- `q_t` = query for readout
- `n_h` = 2 or 3 (sweet spot for expressivity vs cost)

### Surprise-Modulated Write Gate

```
surprise_t = sg(-log p(x_t | x_{<t}))        # Stop-gradiented cross-entropy (training)
           = sg(D_KL(p_t || p̄_t))            # Stop-gradiented KL from EMA (inference)

β_{t,j} = σ(W_{β,j}·x_t + w_{s,j}·surprise_t + b_{β,j})
```

Properties:
- High surprise → large β → strong write (store the unexpected)
- Low surprise → small β → near-identity (don't waste memory on the predictable)
- Stop-gradient prevents circular optimization (surprise depends on model, model depends on surprise)
- Each Householder step j has its own surprise sensitivity `w_{s,j}`

### Per-Channel Decay with StableSSM

```
z_t = W_α·x_t                                 # Linear projection
α_t = 1 - softplus(z_t)                       # Per-channel, StableSSM-inspired
```

Or the theoretically optimal form:
```
α_t = 1 - 1/(z_t² + 0.5)                     # "Best" reparameterization from StableSSM
```

Properties:
- Per-channel: each feature dimension decays at its own learned rate
- StableSSM reparameterization: gradient of α w.r.t. z slows down as α→1, preventing instability
- Different channels naturally learn different timescales

### KL Divergence Computer (Inference)

```
p̄_t = (1 - γ)·p̄_{t-1} + γ·p_t              # EMA of predictive distribution
surprise_t = Σ_{v ∈ top_k} p_t(v)·log(p_t(v) / p̄_t(v))   # Top-k KL approximation
```

With k=256 and vocabulary V, this adds O(256) per token — negligible vs O(d²) for the recurrence.

This captures *distributional shift*, not just point error. A common word in an unexpected context gets high KL. Detects "the entire character of what's being predicted changed."

## Component 2: Multi-Scale States

Two complementary approaches, not mutually exclusive:

### Approach A: Per-Channel Decay (KDA-style)

Already built into the per-channel α_t above. During training, different channels will naturally learn different decay rates through the learned projection W_α. Fast channels (α→0) handle local patterns; slow channels (α→1) handle long-range dependencies.

### Approach B: Explicit Multi-Resolution (MS-SSM-style)

For deeper multi-scale processing, run S=2-3 parallel mixers at different resolutions:

```
x_fast = Mixer_fast(x, dt_scale=1.0)          # Fine-grained, fast decay
x_slow = Mixer_slow(x, dt_scale=0.1)          # Coarse, slow decay
x_out  = ScaleMixer(x_fast, x_slow, x)        # Input-dependent fusion
```

The ScaleMixer is a small learned network that weights the contributions of each resolution based on the current input — analogous to theta-gamma coupling in the hippocampus.

## Component 3: PoPE Integration

PoPE is already in mamba3_block.py. In the DeltaProduct context, PoPE would be applied to the B (key) and C (query) projections, giving:

- Clean content-based addressing for the delta rule (B encodes "what to store at")
- Clean positional readout for the query (C encodes "where to read from")
- The delta error `v_t - S·k_t` reflects genuine content novelty, not positional encoding artifacts

This directly improves the quality of the surprise signal — if the keys are noisy due to positional contamination, the delta error is noisy, and surprise-gated writes fire erroneously.

## Component 4: Trapezoidal Discretization

Retain Mamba3's trapezoidal discretization for the inter-chunk state passing:

```
h_t = exp(Δ·A)·h_{t-1} + (1-λ)·Δ·exp(Δ·A)·B_{t-1}·x_{t-1} + λ·Δ·B_t·x_t
```

This provides second-order accuracy in the temporal integration, improving memory fidelity without increasing d_state. The λ parameter (data-dependent, sigmoid) controls the blend between Euler (current) and trapezoidal (current + previous) terms.

Note: The DeltaProduct state update operates on the fast-weight matrix S, while trapezoidal discretization operates on the SSM state h. These are different objects if we maintain both mechanisms. Design decision needed: do we replace the SSM recurrence entirely with DeltaProduct, or run both in parallel?

**Recommendation**: Replace the SSM recurrence with DeltaProduct. The SSM's `exp(Δ·A)` decay is the root cause of the memory problem. DeltaProduct's Householder updates with per-channel gating provide strictly more expressive state transitions while maintaining stability guarantees.

## Component 5: CTKG Interface (Future)

The neural net's working memory holds ~4 chunk-references (pointers into the CTKG), not raw data. The interface:

- **Read**: Given a query q from the neural net, the CTKG returns relevant knowledge as a categorical construction (limit, colimit, Kan extension)
- **Write**: The neural net extracts surprising entities/relations and writes them to the CTKG as new objects/morphisms
- **Constraint**: The CTKG imposes energy terms on the neural net's predictions — legal, moral, technical, personality constraints as categorical diagrams that must commute

See `CTKG_DESIGN.md` for details.

## Comparison: What We Gain Over Mamba3

| Property | Mamba3 | Mnemon (proposed) |
|---|---|---|
| State update | Additive rank-1 (exp decay) | DeltaProduct: targeted erase + write, n_h Householder steps |
| Write gating | Input-dependent Δ only | Surprise-modulated β (cross-entropy / KL) |
| Decay | Scalar per head, exp parameterized | Per-channel diagonal, StableSSM reparameterized |
| Expressivity per token | Rank-1 | Rank-n_h (can represent rotations, any orthogonal transform) |
| Multi-scale | Implicit (input-dependent Δ) | Explicit per-channel + optional multi-resolution |
| Stability guarantee | Requires A < 0 | Spectral norm ≤ 1 structurally (Householder) |
| Length extrapolation | Degrades | Robust (spectral norm bound) |
| What/where separation | PoPE on B, C | Same, but feeds cleaner signal to delta rule |
| Long-term knowledge | In weights (expensive, fragile) | In CTKG (external, interpretable, principled) |
| Positional encoding | PoPE (retained) | PoPE (retained) |
| Trapezoidal discretization | Yes | Design decision (may replace with DeltaProduct) |

## Computational Cost Estimate

Relative to Mamba3 (per token per layer):

| Component | Cost | Notes |
|---|---|---|
| DeltaProduct (n_h=2) | 2× Mamba3's recurrence | Two sequential Householder updates |
| Per-channel α computation | ~1× | Just a linear projection + reparameterization |
| Surprise computation | <0.1× | Cross-entropy already computed; KL via top-k |
| PoPE | Same as current | Already in Mamba3 |
| Multi-resolution (optional) | S× if using S parallel mixers | Can start with S=1 (per-channel only) |

Total: roughly 2-3× Mamba3 per token. This is still O(1) per token in sequence length (vs O(n) for attention).

## Implementation Plan

### Phase 1: Gated DeltaProduct with PoPE
- Replace Mamba3's SSM recurrence with Gated DeltaProduct (n_h=2)
- Keep PoPE on B, C projections
- Use learned β and α (no surprise gating yet)
- Test on Stage 1b tasks to verify basic functionality
- Compare to Mamba3 baseline

### Phase 2: Surprise Gating
- Add surprise computation (cross-entropy at each position)
- Modify β to incorporate stop-gradiented surprise signal
- Test: does surprise gating reduce memory waste on predictable tokens?
- Measure: effective memory depth vs Mamba3

### Phase 3: StableSSM + Per-Channel Decay
- Replace scalar α with per-channel diagonal
- Apply StableSSM reparameterization
- Test: can the model maintain information over longer distances?

### Phase 4: Multi-Scale States
- Add optional multi-resolution processing
- Test on tasks requiring multiple timescales

### Phase 5: KL Divergence for Inference
- Implement EMA of predictive distribution
- Implement top-k KL computation
- Test inference-time surprise gating vs training-time cross-entropy gating

### Phase 6: CTKG Interface
- Define the read/write/constraint interface
- Implement categorical query mechanism
- Test on knowledge-intensive tasks

## Open Questions

1. **DeltaProduct vs SSM recurrence**: Should we completely replace the SSM, or run DeltaProduct on top of the SSM state? The SSM provides continuous-time dynamics; DeltaProduct provides discrete associative memory. They might be complementary.

2. **Surprise source**: Per-layer surprise (each layer has its own prediction head) vs final-layer surprise (single prediction, backpropagated)? Per-layer is more biologically plausible (local error signals) but adds parameters.

3. **Chunk size interaction**: DeltaProduct's chunkwise parallel training uses WY representation. How does chunk size interact with the surprise signal (which depends on the model's predictions, computed after the forward pass)?

4. **Training stability**: The surprise signal is stop-gradiented, but it still creates a feedback loop — better memory → better predictions → lower surprise → less writing → potentially worse memory for future novel events. Is this self-correcting or does it need explicit regularization?

5. **VRAM budget**: With 4GB VRAM, how large can we make the DeltaProduct state (d_k × d_v per head, × nheads, × n_h projection matrices)? Need to profile.
