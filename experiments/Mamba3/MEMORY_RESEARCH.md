# Memory Architecture Research

Research summary for evolving Mamba3 into a biologically-grounded sequence model with principled short-term memory. This document covers the landscape of recurrent memory mechanisms, their biological analogs, and the mathematical frameworks we plan to build on.

## What's Actually In Mamba3's State

The state has shape `(batch, nheads, headdim, d_state)`. Each head maintains a **fast-weight matrix** S ∈ R^{headdim × d_state}. The recurrence:

```
h_t = exp(Δ·A) · h_{t-1} + Δ · (B_t ⊗ x_t)
y_t = C_t · h_t
```

Each token contributes a **rank-1 outer product** of its value (x, the processed features) and its key (B, the input projection). The state is the sum of all these outer products, each exponentially decayed by `exp(Δ·A)`.

- **B is the key** — addressing/indexing
- **x is the value** — what to store
- **C is the query** — how to read out
- **Δ (dt) is the gate** — how strongly to write / how fast to forget

This is mathematically identical to **linear attention's fast-weight matrix**, which is exactly why DeltaNet-family architectures are natural upgrades — they replace naive additive accumulation with targeted write/erase.

With d_state=64, headdim=64, and 4 heads: total state = 16,384 elements. That's the entire scratch paper for all of history.

## The Exponential Decay Problem

The SSM recurrence requires A to have negative eigenvalues for stability. Past contributions decay as `exp(-α·k)`. With d_state=64, our empirical measurements show:

```
Position 14 (distance 1):  68.8% accuracy
Position 13 (distance 2):  30.8%
Position 12 (distance 3):  15.4%
Positions 1-9 (distance 6+): ~4-8% (near random)
```

Textbook exponential decay. PoPE improved retrieval *fidelity* (68.8% vs 20.6% at distance 1) but not memory *depth*.

## Architectural Solutions Landscape

### Tier 1: Better State Updates (Drop-in Replacements for Mamba's Recurrence)

| Architecture | State Update | Key Insight | Cost vs Mamba |
|---|---|---|---|
| **Gated DeltaNet** | `S_t = α_t·S_{t-1}·(I - β_t·k_t·k_t^T) + β_t·v_t·k_t^T` | Per-key erase + bulk decay. Online GD on associative recall loss. | ~1x |
| **DeltaProduct** | n_h sequential Householder updates per token | Products of reflections → rotations → any orthogonal transform. Spectral norm ≤ 1 structurally. | n_h × |
| **Gated DeltaProduct** | DeltaProduct + scalar forget gate | Recognizes any regular language with finite layers. | n_h × |
| **KDA (Kimi)** | Gated DeltaNet + per-channel diagonal gate | Each feature dimension decays at its own rate. Production-deployed (262K native context). | ~1x |

### Tier 2: Multi-Scale / Multi-Timescale

| Architecture | Mechanism | Biological Analog |
|---|---|---|
| **MS-SSM** | Wavelet-inspired multi-resolution decomposition, independent SSMs per scale, input-dependent scale-mixer | Cortical hierarchy of intrinsic timescales (Murray et al. 2014) |
| **KDA per-channel decay** | Each dimension has its own learned decay rate | Fast vs. slow neurotransmitter systems |
| **ms-Mamba** | Parallel SSM blocks at different sampling rates | Theta-gamma cross-frequency coupling |

### Tier 3: External/Augmented Memory

| Architecture | Mechanism | Demonstrated Context |
|---|---|---|
| **MemMamba** | State pool (50 compressed summaries) + cross-token/cross-layer attention | 400K tokens, 90% retrieval |
| **Titans** | Neural LTM (MLP weights updated at test time via surprise-gated GD) + persistent memory + attention | 2M+ tokens |
| **TTT-E2E** | Full model weight updates at test time | 2M+ tokens, no scaling wall |

### Tier 4: Theoretical Foundations

| Paper | Key Result |
|---|---|
| **StableSSM** (Wang & Li, ICML 2024) | The memory limitation is *parameterization*, not architecture. Reparameterizing A lifts the curse of memory. Best: `λ = 1 - 1/(w² + 0.5)` |
| **Input Selectivity Theory** (Li et al. 2025) | Width helps recurrent models more than depth. Input selectivity (Mamba's Δ) allows "freezing time" by setting Δ→0 for irrelevant tokens. |
| **MIRAS** (Google 2025) | All sequence models (Transformers, Mamba, DeltaNet, RWKV) are instances of the same 4 design choices about memory. |
| **Test-time regression** (Wang et al., Stanford 2025) | All architectures = regression on past context. Distinctions: regressor function, observation weights, optimization algorithm. |

## Biological Analogs

### SSM ↔ Brain Mapping

| Brain System | Timescale | Closest Architecture |
|---|---|---|
| Sensory buffer | ~250ms | SSM with fast decay + surprise gating |
| Working memory (PFC + parietal) | Seconds, ~4±1 chunks | Gated DeltaNet / attention |
| Hippocampus (fast episodic) | Minutes to days | TTT / Titans neural LTM |
| Neocortex (semantic knowledge) | Weeks to years | Pretrained weights / CTKG |
| Basal ganglia (procedural) | Months to permanent | Frozen/compiled circuits |

### Multi-Scale States (MS-SSM) ↔ Neuroscience

| MS-SSM Component | Biological Analog | Reference |
|---|---|---|
| SSMs at different timescales | Cortical hierarchy of intrinsic timescales | Murray et al. (2014), Nature Neuroscience |
| Multi-resolution decomposition | Temporal receptive windows | Hasson et al. (2008), J Neuroscience |
| Input-dependent scale-mixer | Theta-gamma cross-frequency coupling | Lisman & Jensen (2013) |
| Hierarchical predictions across scales | Hierarchical predictive coding in language | Caucheteux et al. (2023), Nature Human Behaviour |

SSMs have direct neuroscience origins: Voelker & Eliasmith (2018) showed SSM delay lines replicate hippocampal time cells, leading to the Legendre Memory Unit (NeurIPS 2019), ancestor of S4 and Mamba.

### Surprise Principle ↔ Neuroscience

The locus coeruleus releases norepinephrine in response to unexpected stimuli, enhancing encoding (Von Restorff effect). This is exactly surprise-gated memory writes.

## Compression and Chunking

### Human Chunking

- **Cowan's 4±1**: True WM capacity when chunking is restricted
- **Miller's 7±2**: Capacity when chunking is unrestricted (items compressed into chunks)
- **Mechanism**: Statistical learning detects co-occurring items → hippocampus binds into chunks → chunks become single WM entries
- **Formally**: Lossy data compression (Nassar et al. 2018) — similar features merged, dissimilar features partitioned. Equivalent to rate-distortion theory.

### Neural Analog: Vector Quantization

Chunking maps to VQ-VAE codebook learning:
- Repeated patterns snap to the same codebook entry (automatic deduplication)
- Codebook capacity = chunk capacity
- Learning = statistical exposure (same as human statistical learning)

### Implications for Architecture

The CTKG serves as the codebook. Common patterns → CTKG entries (categorical objects). Neural net stores references (morphisms) to CTKG entries. Working memory holds ~4 chunk-references, not raw data.

## Surprise-Gated Memory: Mathematical Framework

### Bayesian Surprise

Itti & Baldi (2009): `Surprise = D_KL(posterior || prior)`. Outperformed 10 other saliency measures in predicting human gaze. The key distinction from MSE:

| | MSE | KL Divergence |
|---|---|---|
| Measures | How wrong the point prediction was | How much beliefs changed |
| Uncertainty-aware | No | Yes |
| High-confidence violation | Same as low-confidence | Much larger (confident prior shattered) |
| Information content | None | Bits of information gained |

### Cross-Entropy as Surprise (Training Time)

During training, the cross-entropy loss at position t:
```
L_t = -log p(x_t | x_{<t})
```
This IS the self-information (surprisal) — not an approximation. Already computed at every position. Free to use as a write gate.

### KL Divergence as Surprise (Inference Time)

Without ground truth, compare current prediction to running average:
```
p̄_t = (1 - γ)·p̄_{t-1} + γ·p_t
surprise_t = D_KL(p_t || p̄_t)
```

Captures *distributional shift*, not just point error. A common word in an unexpected context gets high KL. Efficient computation via top-k approximation (k=256) over vocabulary.

### Surprise-Modified Gated DeltaNet

```
β_t = σ(W_β·x_t + w_s·sg(surprise_t))    # Write gate modulated by surprise
α_t = f_stable(W_α·x_t)                    # Decay gate with StableSSM reparam

S_t = α_t·S_{t-1}·(I - β_t·k_t·k_t^T) + β_t·v_t·k_t^T
```

- `sg()` = stop-gradient (treat surprise as fixed signal, avoid circular optimization)
- `f_stable()` = StableSSM reparameterization for gradient stability near α→1
- High surprise → large β → strong write. Low surprise → small β → memory unchanged.
- StableSSM ensures α can stably approach 1 (long retention) without gradient explosion.

### Connection to Free Energy

Friston's variational free energy: `F = D_KL(q || p) + E_q[-log p(data)]`. The KL term IS the surprise. Minimizing free energy = minimizing surprise + maximizing accuracy. Our prediction error is already an energy — using KL means the gate fires based on information gain, not raw error magnitude.

## Memory Capacity Targets

- **Minimum**: 128K tokens with near-perfect fidelity
- **Optimistic**: 1M+ tokens with near-perfect fidelity

### Benchmarks

| Benchmark | What It Tests | Target |
|---|---|---|
| **Passkey retrieval** | Single-fact recall at distance | Baseline (too easy) |
| **RULER** | Multi-query associative recall | Intermediate |
| **BABILong** | 20 reasoning tasks at up to 50M tokens | Intermediate |
| **OOLONG** | Aggregation over entire context (counting, temporal reasoning) | Primary |
| **OOLONG-Pairs** | Quadratic-complexity aggregation | Primary |
| **BrowseComp-Plus** | Multi-hop reasoning across 6-11M tokens | Stretch (requires CTKG) |

Current SOTA on OOLONG-Pairs: RLM (recursive decomposition) at 58.00 F1 vs GPT-5's 0.04.

## PoPE's Role in Memory

PoPE decouples content ("what") from position ("where"):
- **Magnitude** (softplus) = pure content
- **Phase** (cumulative rotation) = pure position

Implications for memory gates:
1. **Content surprise and position surprise decouple.** Familiar token at new position doesn't waste a write slot.
2. **Cleaner associative lookup.** Keys don't collide due to positional noise.
3. **More accurate delta signal.** The error `v_t - S·k_t` reflects genuine novelty.
4. **PoPE + surprise gating are complementary.** PoPE improves retrieval fidelity (using what's stored). Surprise gating improves what gets stored in the first place.

## Key References

### Memory Mechanisms
- Gated DeltaNet: Yang et al., ICLR 2025, arXiv:2412.06464
- DeltaProduct: Siems et al., NeurIPS 2025, arXiv:2502.10297
- KDA / Kimi Linear: arXiv:2510.26692
- StableSSM: Wang & Li, ICML 2024, arXiv:2311.14495
- MS-SSM: Karami et al., COLM 2025, arXiv:2512.23824
- MemMamba: Wang et al., arXiv:2510.03279
- Titans: Behrouz et al., arXiv:2501.00663
- TTT: Sun et al., arXiv:2407.04620; TTT-E2E: arXiv:2512.23675
- MIRAS: arXiv:2504.13173
- Test-time regression: Wang et al., arXiv:2501.12352

### Neuroscience
- Murray et al. (2014) — Cortical timescale hierarchy, Nature Neuroscience
- Hasson et al. (2008) — Temporal receptive windows, J Neuroscience
- Caucheteux et al. (2023) — Hierarchical predictive coding, Nature Human Behaviour
- Itti & Baldi (2009) — Bayesian surprise, Vision Research
- Voelker & Eliasmith (2018) — SSMs and hippocampal time cells
- Nassar et al. (2018) — Chunking as lossy data compression

### Benchmarks
- OOLONG: Bertsch et al., arXiv:2511.02817
- BrowseComp-Plus: arXiv:2508.06600
- RLM: Zhang et al., MIT CSAIL, arXiv:2512.24601
- BABILong: arXiv:2406.10149
