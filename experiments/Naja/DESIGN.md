# Naja Design

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

Naja unifies both lineages, keeping Mamba3's continuous-time core
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

**Naja (proposed):**
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

| Property | Mamba3 | Gated DeltaNet | KDA | Naja |
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
- Naive sequential implementation (no chunkwise parallelism yet) ✓
- Test on Stage 1b tasks

### Phase 2: PoPE Orthogonal Pair ✓
- Derive B₂ from PoPE ✓ (apply_pope_perp in naja.py)
- Add second Householder (β₂) ✓ (beta2_proj, beta2 gate)
- Test rotation capability on state-tracking tasks (parity task in tasks.py)

### Phase 3: Per-Channel Decay ✓
- Replace scalar decay with per-channel diagonal α_t ✓ (decay_down/up MLP)
- Compare sigmoid vs StableSSM reparameterization ✓ (stable_reparam toggle)
- Verify multi-scale emergence (visualize per-channel α distributions)

### Phase 4: Surprise Gating ✓
- Add surprise computation (cross-entropy at each position) ✓ (forward_with_surprise)
- Modify β₁, β₂ to incorporate stop-gradiented surprise ✓ (use_surprise_gate)
- Test: does surprise gating reduce memory waste on predictable tokens?

### Phase 5: Chunkwise Parallelism — Phase 5a+5b IMPLEMENTED
- ~~Gradient-checkpointed chunk processing (delta_recurrence_chunkwise)~~ ← NOT real parallelism, just gradient checkpointing. See Mistake #39.
- **Phase 5a: Pure PyTorch WY chunkwise ✓** — `delta_recurrence_wy()` in naja.py, config flag `use_wy_chunkwise`, CLI `--use_wy_chunkwise`
  - SISO only (r=1), single Householder (B1)
  - 4-step algorithm: UT transform → chunk state accumulation → inter-chunk scan → intra-chunk output
- **Phase 5b: Per-channel decay (KDA-style) ✓** — A matrix stays CxC via K_pos/K_neg weighting. Verified ~2e-6 accuracy on 8 test cases including per-channel multi-chunk.
- **Phase 5c: PoPE pair (B2)** — virtual token expansion for second Householder
- **Phase 5d: Ablation testing** — per-channel vs scalar, PoPE pair, surprise gating, MIMO
- **Phase 5e: Triton kernels** — optional future optimization
- Reference: Gated DeltaNet (Yang et al. 2025), DeltaProduct (Siems et al. 2025)
- Reference code: `flash-linear-attention` library, `ssd_trapz()` in `mamba3_block.py`
- Benchmark training speed vs Mamba3

### Phase 6: KL Divergence for Inference ✓
- Implement EMA of predictive distribution ✓ (KLSurpriseTracker)
- Top-k KL computation ✓ (configurable top_k)
- Test inference-time surprise gating (evaluate_with_kl_surprise in train_naja.py)

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

## Naming

**Name: Naja** (confirmed 2026-02-14)

**Naja** is the genus of cobras, in the same family (Elapidae) as Dendroaspis
(the mamba genus). It maintains the snake evolutionary lineage while being a
distinct identity. The name is:
- One word, short, pronounceable ("NAH-jah")
- Taxonomically linked to Mamba (same family, sister genus)
- Completely clean in ML namespace (no existing model/framework uses it)
- Culturally neutral (no negative connotations)

We considered "Mamba4" but it implies ownership of the Mamba lineage (Gu & Dao),
which would be a breach of etiquette in the research community. Runner-up was
"Hydrus" (mythological water serpent), but Naja has a stronger biological
connection. All other candidates (Hydra, Cobra, Taipan, Chimera, Mantis,
Viper, Basilisk, Ouroboros) are taken by existing ML projects.

## Permutation Tracking Analysis

**Status:** Theoretical capacity exists, but learning dynamics are uncertain.

### Mathematical Capacity

Permutation matrices are orthogonal. Any orthogonal matrix decomposes into
Householder reflections. Naja provides 2 Householder reflections per layer
(B₁, B₂) via the PoPE orthogonal pair. With n_layer=4, that's up to 8
reflections per token — sufficient for any orthogonal matrix in moderate
dimensions (d_state=64).

The DeltaProduct paper (Siems et al., NeurIPS 2025) showed that chains of
Householder products can represent arbitrary orthogonal matrices, confirming
this capacity in principle.

### Mechanism

1. PoPE provides phase-based "slot" encoding (phase angle = position)
2. Householder reflections act multiplicatively on state (this IS the
   permutation update, not an approximation)
3. Per-channel decay selectively preserves permutation-relevant channels
   (α≈1) while letting irrelevant channels forget (α≈0)
4. The direction of each reflection is data-dependent (computed from input),
   so the model can in principle choose the right permutation at each step

### Concerns

**The multi-rule collapse problem may recur.** If the model can't discover
that 5 simple rules exist (Stage 2), it may also fail to discover permutation
composition structure from the state space of n! permutations. The learning
dynamics of discovering compositional structure are the hard part, not the
representational capacity.

**Key difference from Stage 2:** Permutations have strong compositional
structure (each swap is local, compositions are systematic). The 5 rules
in Stage 2 are unrelated. Permutations might be easier because the structure
is richer and more regular — or harder because the state space (n!) is larger.

### Testing Plan

Test incrementally, watching for collapse:
1. 2-element permutation (parity) — Mamba3 already handles this
2. 3-element permutation (6 states) — minimal non-trivial case
3. 4-element permutation (24 states)
4. 5-element permutation (120 states)

At each scale, check: does test accuracy plateau at 1/n! (memorizing one
permutation)? Does it show the same multi-rule collapse pattern? Does
extended training trigger grokking?

If collapse occurs at 3 elements, the architecture may need explicit slot
attention or curriculum over permutation complexity.

## Stage 2 Fairness Investigation

**Status:** Not yet investigated. Must be done before drawing conclusions.

### The Concern

We observe 99% train / ~25% test accuracy on the 5-rule induction task
and attribute this to the model failing to discover multi-rule structure.
But we should verify: **can a human solve this task with the same information?**

If the in-context examples are insufficient to uniquely determine the rule,
even a perfect reasoner would fail. The theoretical maximum accuracy could
be less than 100%.

### What to Check

1. **Ambiguity analysis:** For each test example, do the in-context examples
   uniquely identify the rule? Could two different rules produce the same
   input-output pairs for the given context? If yes, what fraction of test
   examples are ambiguous?

2. **Human baseline:** Present 20-50 test examples to a human (with rule
   names hidden). Can they identify the rule from the in-context pairs?
   What accuracy do they achieve? How long does it take?

3. **Information-theoretic analysis:** Given the token vocabulary and sequence
   length, how many in-context examples are needed to uniquely identify each
   rule? Some rules (e.g., "double") may need 1 example; others (e.g.,
   "shift3" vs "shift7") may need examples with specific inputs.

4. **Rule distinguishability:** Are there input values where "shift3" and
   "shift7" produce outputs that could be confused with other rules (e.g.,
   "square" for small inputs)? This would make certain test examples
   inherently ambiguous.

### Implications

- If the task is fair (human achieves >80%), the failure is clearly the model's
- If the task is partially ambiguous, adjust expected test accuracy downward
- If the task is fundamentally ambiguous (many examples have multiple valid
  rules), redesign the data generation to ensure unambiguous in-context examples

## Future Research: Causal Induction

**Priority:** After Phases 1-6 are complete and ablation-tested.

### Why SSMs Have an Advantage

The SSM recurrence h_t = f(h_{t-1}, x_t) is inherently causal — information
flows forward in time. The state h_t summarizes all past inputs. If A causes B,
then observing A should change the state such that B becomes predictable.
The surprise signal detects exactly this asymmetry.

Transformers lack this temporal inductive bias — attention is symmetric over
positions unless masked. SSMs have temporal dynamics baked in.

### Connection to Multi-Rule Collapse

The multi-rule collapse (see docs/hypotheses/generalization_vs_memorization.md)
IS a causal induction failure. In Stage 2, the "cause" is rule identity and
the "effect" is the input-output mapping. The model fails to discover that
rule identity causes the mapping. Solving causal induction and multi-rule
collapse are deeply related.

### Proposed Approaches

1. **Surprise as causal signal.** High surprise → something unexpected →
   potential new causal relationship. The surprise-gated memory (Phase 4)
   is already structured for this. Test: does surprise-gated beta track
   actual causal boundaries in sequences?

2. **Temporal intervention detection via per-channel decay.** Channels with
   α≈1 track long-range causes; channels with α≈0 track short-range causes.
   The distribution of learned α values reveals the model's implicit causal
   horizon. Diagnostic: visualize per-channel α distributions on tasks with
   known causal structure.

3. **Granger causality in latent space.** Measure whether knowing h_{t-k}
   improves prediction of x_t beyond what h_{t-1} provides. If yes, long-range
   causal information is being lost by state compression. Computable as:
   L_causal = D_KL(p(x_t | h_{t-1}) || p(x_t | h_{t-1}, h_{t-k}))

4. **Transfer entropy between channels.** If channel i's state predicts
   channel j's future but not vice versa, there's a directional causal
   relationship. Computable as an auxiliary diagnostic.

### Key References (Causal Induction)

- Zheng et al. 2018 — NOTEARS: differentiable DAG learning (arXiv:1803.01422)
- Tank et al. 2022 — Neural Granger Causality (arXiv:1802.05842)
- Massey 1990 — Directed Information (IEEE Trans. Info. Theory)
- Schreiber 2000 — Transfer Entropy (Physical Review Letters)

## Future Research: Metacognition (Explicit Self-Awareness)

**Priority:** After Phases 1-6 and causal induction.

### What We Mean by Metacognition

Not just "tracking multiple hypotheses" (that's multi-hypothesis reasoning)
and not just "detecting prediction errors" (that's surprise gating).

We mean: **making the model explicitly aware of what it is currently thinking,**
so it can reason about its own reasoning process. This enables:

- "I'm currently considering hypothesis A" (self-report)
- "My confidence in this prediction is low" (uncertainty estimation)
- "I tried this approach and it failed, let me try something different"
  (strategy switching based on self-observed failure patterns)
- "This internal state pattern tends to lead to failure" (meta-pattern
  recognition over own cognitive states)

This is a second-order phenomenon: representing representations.

### Theoretical Grounding

- **Attention Schema Theory (Graziano):** The brain maintains a simplified
  model of its own attention process. An "introspection head" that reads state
  and produces meta-signals IS the attention schema.
- **Global Workspace Theory (Baars):** Information becomes "conscious" when
  broadcast globally. The meta-state m_t injected back into the model is the
  global broadcast.
- **Predictive Processing (Clark):** The brain predicts its own predictions.
  An introspection head that predicts the quality of the main model's
  predictions implements this directly.

### Proposed Architecture: Introspection Head

A separate lightweight MLP that runs as a parallel pathway after the final
Naja layer — NOT a sequential block inserted between layers:

```
x_t' = embed(token_t) + m_{t-1}     # inject previous meta-state
h_t  = NajaModel(x_t')              # run all layers normally
m_t  = IntroHead(h_t)               # small MLP on final hidden state
y_t  = Decode(h_t)                  # output prediction
```

The introspection head is a single MLP (d_model → d_intro → d_model) that
reads the final hidden state, produces a meta-state m_t, and injects it
into the next token's input via residual addition. Negligible VRAM cost.

**Interface design:** The introspection head takes a generic tensor input,
not a hardwired reference to the final layer. This means upgrading from
approach 1 to approach 3 (below) is a drop-in replacement — change what
you pass in, not the head itself.

### Introspection Head: Upgrade Paths

**Approach 1 — Single final-layer MLP (initial implementation):**
What's described above. Start here. Matches the Graziano group's
experimental setup and the epinet architecture. Empirically validated.

**Approach 3 — Shared head, multi-layer input (first upgrade):**
Same single MLP, but reads from all layers via learned layer-weights:
```
m_t = IntroHead(Σ_l α_l · h_t^l)    # α_l are learned scalars
```
One set of parameters, but visibility into all layers. The learned
weights α are themselves interpretable — they reveal which layers are
informative for metacognition. Cheap (one extra weighted sum), and a
drop-in replacement because the head's input dimensionality doesn't
change.

**Approach 2 — Per-layer introspection heads (long-term goal):**
Each layer gets its own small MLP: `m_t^l = IntroHead_l(h_t^l)`.
Per-layer meta-states aggregated before injection. This is the most
powerful option and the one that recovers the local-error-signal
property we originally wanted from ePC — each layer gets its own
metacognitive loss that reshapes that layer's representations locally.
But it's the hardest to tune: per-layer auxiliary losses interact with
each other and with the main loss through downstream gradient flow.
Wait until approaches 1/3 are working before attempting this.

### Higher-Order Metacognition: Temporal Unrolling, Not Stacked Heads

A natural question: does higher-order metacognition (thinking about thinking
about thinking) require stacking introspection heads — a meta-head to monitor
the introspection head, a meta-meta-head to monitor that, etc.?

**No. Higher-order metacognition emerges from temporal unrolling of a single
introspection head.** The recurrence already provides this for free.

Consider a concrete example of third-order reasoning (from a discussion about
bias detection): "I concluded X. But I notice my conclusion might be biased
because of Y. But wait — my belief that I'm biased might itself be a socially
ingrained overcorrection, and my original conclusion might actually be
impartial despite my bias prior making it hard to accept that."

This is three levels deep, but each level happens at a different TIME STEP:

```
t=0: h_0 encodes conclusion X
     m_0 = IntroHead(h_0)  →  "I believe X"

t=1: h_1 = f(h_0, m_0)    →  state now includes self-awareness of X
     m_1 = IntroHead(h_1)  →  "I notice bias Y influenced my belief in X"

t=2: h_2 = f(h_1, m_1)    →  state now includes awareness of bias detection
     m_2 = IntroHead(h_2)  →  "My bias detection might itself be biased —
                                perhaps X is correct despite Y"
```

Each application of the SAME introspection head at the next time step
naturally produces the next metacognitive level, because the meta-state m_t
feeds back into the main state via residual addition. The recurrent state
h_t accumulates all previous levels of self-reflection. No additional
architectural machinery is needed — just more time steps.

This is analogous to how humans experience higher-order reflection: not as
parallel processes but as sequential re-examination of the same internal
state. The "depth" of metacognition is bounded by sequence length, not by
architectural depth. This is a key insight: **metacognitive depth is a
runtime property, not an architectural property.**

Implication for evaluation: tasks that require N-th order metacognition
need sequences long enough to accommodate N reflection steps. The model
must learn WHEN to re-examine its own meta-state (likely driven by the
surprise signal — high surprise on m_t triggers another reflection step).

### Counterfactual Self-Modeling

The bias detection example above reveals a deeper requirement: the
introspection head must be capable of **counterfactual self-modeling** —
reasoning about what it WOULD think under different conditions.

"If I weren't left-wing, would I still conclude X?" is not a question about
the current state h_t. It's a question about a hypothetical state h_t' that
would exist if the model's "priors" were different. This requires the
introspection head to:

1. **Identify which components of h_t correspond to "priors"** (long-timescale
   channels with α≈1 that encode persistent biases)
2. **Simulate perturbations** to those components (what would h_t look like
   if those channels had different values?)
3. **Predict the downstream effect** on the model's own output

This is strictly more powerful than the self-modeling loss (which predicts
current activations) and the confidence loss (which predicts error magnitude).
It requires a **causal model of the self** — understanding not just "what am
I thinking" but "why am I thinking it" and "what would I think otherwise."

Proposed training objective:

```
L_cfact = MSE(IntroHead(h_t)_cfact, sg(f(perturb(h_t)) - f(h_t)))
```

Where perturb(h_t) applies small perturbations to specific channels of h_t,
and f() runs the forward pass from that state. The introspection head learns
to predict the EFFECT of internal perturbations on its own output — i.e.,
which internal states are causally responsible for which outputs.

This connects directly to the per-channel decay structure: slow-decay channels
(α≈1) encode "priors" (persistent beliefs), fast-decay channels (α≈0) encode
"evidence" (recent observations). Counterfactual self-modeling asks: "if my
priors were different, would my conclusion change?" The architecture makes
this question well-posed because priors and evidence are already separated
by timescale.

**Priority:** After the primary self-modeling and confidence losses are
working. Counterfactual self-modeling is the most ambitious objective and
requires the base introspection head to already be functional.

**Connection to Death Note evaluation (see below):** Light's memory loss arc
requires exactly this capability — acting as if certain memories don't exist
while maintaining coherent behavior. This is counterfactual self-modeling
applied to one's own memory state.

### MIMO as Superhuman Cognitive Capability

An important asymmetry between Naja and human cognition: **MIMO gives the
model a cognitive ability that humans demonstrably lack.**

Humans cannot genuinely evaluate multiple hypotheses in parallel. Introspection
reveals that human "multi-hypothesis reasoning" is actually rapid serial
task-switching — we consider hypothesis A, then switch to B, then back to A.
We cannot update two independent mental models simultaneously unless a single
operation happens to update both. At best, we can hold ~1 active hypothesis
and ~3-4 in short-term memory for rapid switching.

MIMO rank-r writes update r columns of the state matrix SIMULTANEOUSLY in a
single operation. Each column can track a genuinely independent hypothesis
about the current context. This is not task-switching — it is true parallel
hypothesis maintenance within the recurrence itself.

Implications:

1. **Superhuman deduction:** A Naja model with mimo_rank=4 could maintain 4
   independent causal theories about a mystery simultaneously, updating each
   with every new piece of evidence. A human detective considers theories
   serially. The model's Bayesian MIMO reweighting (see above) automatically
   concentrates probability on the best theory as evidence accumulates.

2. **Superhuman social modeling:** In adversarial social reasoning (like Death
   Note's L vs Light), each MIMO column could maintain a different model of
   the opponent's mental state. L could simultaneously track "Light is Kira",
   "Light is not Kira but is being manipulated", "Light is Kira but has lost
   his memories", and "Light is an unwitting accomplice" — genuinely in
   parallel, not by switching between them.

3. **Interaction with metacognition:** The introspection head observes ALL
   MIMO columns simultaneously. It can detect when columns converge (growing
   certainty), diverge (genuine ambiguity), or when one column is consistently
   surprised (that hypothesis is failing). This gives the model a kind of
   "peripheral awareness" of alternative interpretations that humans achieve
   only through deliberate effort.

4. **Testable prediction:** On tasks requiring simultaneous tracking of
   independent state variables (e.g., monitoring multiple agents with
   independent goals), Naja with mimo_rank>1 should show a qualitative
   advantage over both mimo_rank=1 AND human performance, not just a
   quantitative speed improvement. This would be evidence of a genuinely
   novel cognitive capability rather than just faster human-like reasoning.

### Evaluation Scenario: Death Note Reasoning

The Death Note universe (specifically the L vs Light Yagami conflict) provides
a rich evaluation scenario for metacognitive capabilities because both
characters engage in deep recursive social modeling:

- **Light's memory loss arc:** Light voluntarily surrenders his memories of
  being Kira, then must behave consistently as an innocent person while his
  past self's plan unfolds around him. This requires counterfactual
  self-modeling — acting as the version of yourself that lacks certain
  knowledge, while the narrative tests whether behavior is truly consistent
  with that counterfactual state.

- **L's abductive reasoning:** L maintains multiple hypotheses about Kira's
  identity and designs experiments (social interactions, surveillance, rule
  tests) to distinguish between them. Each interaction is simultaneously an
  information-gathering action and a social performance. This requires
  parallel hypothesis tracking (MIMO) plus metacognitive awareness of which
  hypotheses are being tested by each action.

- **Recursive social modeling:** Light models L's model of Light. L models
  Light's model of L's model of Light. This is exactly the temporal unrolling
  of metacognition described above — each additional level of "he thinks that
  I think that he thinks" is another time step of introspective re-examination.

**Concrete test:** Given a partial Death Note scenario, can the model:
1. Maintain multiple hypotheses about character identities/motivations (MIMO)
2. Identify which hypothesis best explains new evidence (Bayesian reweighting)
3. Model what a character WOULD do if they lacked certain knowledge
   (counterfactual self-modeling)
4. Detect when its own reasoning is being manipulated by narrative framing
   (higher-order metacognition via temporal unrolling)

This scenario is aspirational — it requires a fully trained model with working
metacognition. But it provides a concrete target that exercises every component
of the architecture: per-channel memory (tracking long-range plot context),
surprise gating (detecting plot twists), MIMO (parallel hypothesis tracking),
introspection head (metacognitive self-monitoring), and counterfactual
self-modeling (theory of mind).

**Connection to Danganronpa:** The same capabilities apply directly to the
project's primary evaluation environment. Danganronpa's class trials require
exactly the same recursive social modeling, hypothesis tracking, and evidence
evaluation — just in a different narrative wrapper. Death Note provides a
useful SECOND evaluation domain with well-known ground truth.

### Introspection Head Training Objectives

The introspection head needs its own auxiliary losses — without them it has
no gradient signal. Two primary objectives, one optional tertiary, plus the
counterfactual objective described above:

**Primary: Self-modeling loss (Graziano-style)**

```
L_self = MSE(IntroHead(h_t)_proj, sg(W_proj · h_t))
```

Predict a low-rank projection of own activations. This is the objective
from Premakumar et al. 2024 that produced the surprising regularization
benefits: networks trained to predict their own activations become simpler,
more regularized, more parameter-efficient. The self-modeling objective
reshapes the base network's representations to be more self-interpretable
via gradient flow from the auxiliary loss. Without it, the base model has
no incentive to organize its states in a metacognitively accessible way.

This is the FOUNDATION — it provides the structural benefits that make
the other objectives effective. The base model becomes more "readable"
to its own introspection head.

**Secondary: Confidence estimation (continuous)**

```
L_conf = MSE(IntroHead(h_t)_error, sg(|y_{t+1} - target_{t+1}|))
```

Predict the magnitude of the next prediction error. Continuous signal,
strictly more informative than binary calibration (which was dropped —
binary correctness is just a threshold on this continuous estimate, and
can be recovered at inference time without a separate loss).

There is no evidence that current LLMs have access to their own
loss/perplexity — this explicitly trains the model to have an internal
"uncertainty meter." The base model optimizes p(x_{t+1} | context)
(tries to be RIGHT). The confidence head optimizes a second-order
objective: it learns what internal states characterize correct vs
incorrect outputs (tries to KNOW WHEN it's right).

Combined with per-channel structure, this enables causal attribution of
uncertainty: if confidence correlates with slow-decay channels (α≈1),
the model is uncertain about long-range context; if it correlates with
fast-decay channels (α≈0), it's uncertain about recent context. The
architecture makes the "why" readable — the confidence head learns to
read a structure that per-channel decay already provides.

**Tertiary (optional): State stability prediction**

```
L_stab = MSE(IntroHead(h_t)_delta, sg(||h_{t+1} - h_t||))
```

Predict how much the state will change at the next step. Genuinely
different from confidence: the model might be confident in its output but
expecting a large state change (e.g., a scene transition it predicted
correctly). Provides anticipatory surprise (forward-looking, not reactive
like the basic surprise signal). Persistent mismatch (predict small Δh,
observe large Δh) signals an unfamiliar regime — qualitatively different
from single-step surprise.

Predicting summary statistics of Δh (norm, per-channel magnitudes) is
more useful and cheaper than predicting the full next state. Only add
this if the two primary objectives plateau.

**Dropped: Hypothesis identification (predict best MIMO column).** Our MIMO
columns serve dual purpose — hardware-efficient rank-r write AND hypothesis
diversity. Asking the introspection head to predict "which column is best"
assumes columns have cleanly separated into distinct hypotheses, but the
column↔hypothesis mapping can shift during training. The confidence head
with per-channel structure already provides implicit hypothesis attribution.
A separate L_hyp could also fight with the MIMO diversity loss over what
the columns "should" mean. If explicit hypothesis tracking is needed later,
it should come from the Bayesian reweighting on the readout side (see
"Bayesian Hypothesis Testing via MIMO" below), not from the introspection
head.

Note: all targets are stop-gradiented (sg) to prevent circular
optimization through the main model.

### Distinction from EB-JEPA's Multi-Hypothesis

EB-JEPA (LeCun et al.) uses MPPI/CEM to sample and reweight action
trajectories — this is hypothesis testing at the PLANNING level. The model
evaluates external outcomes of different action sequences.

Our metacognition proposal is at the REPRESENTATION level: the model observes
its own internal state and reasons about the quality of its current thinking.
These are complementary:

- EB-JEPA: "Which plan is best?" (external evaluation)
- Metacognition: "Am I reasoning well right now?" (internal evaluation)

### Bayesian Hypothesis Testing via MIMO

**Status:** Promising idea, needs formal development.

Each MIMO column could track a distinct hypothesis. The surprise signal
provides Bayesian updates:

```
weight_i ∝ exp(-surprise_i)     (posterior over hypotheses)
readout = Σ_i weight_i * C_i^T * h    (posterior-weighted prediction)
```

This turns MIMO into a mixture of experts in the recurrence, where surprise
gates the mixture weights. No additional architecture needed — just a training
objective that encourages diverse hypotheses across MIMO columns.

Connection: this IS a form of metacognition — the model is implicitly tracking
"which of my hypotheses is most consistent with observations." Making this
explicit (via the introspection head) would let the model REASON about which
hypothesis is active and why.

### Principled Mathematical Approaches

- **Free energy minimization (Friston):** Surprise = free energy. The model
  should act to minimize its own surprise, seeking information that resolves
  uncertainty. Already compatible with JEPA+surprise.
- **Bayesian predictive coding:** Maintain uncertainty estimates over
  predictions. Per-channel variance tracking.
- **Information gain as intrinsic reward:** reward_t = H(h_{t-1}) - H(h_t|x_t).
  Encourages seeking informative inputs.

### MIMO Readout Diversity for Hypothesis Testing

To use MIMO columns for hypothesis tracking WITHOUT losing MIMO's hardware
efficiency benefit (rank-r write for FLOPs/byte), apply diversity only to
the readout side:

- **Write (keep unchanged):** All r columns contribute to the same state
  H via rank-r update. This is the hardware benefit.
- **Read (add diversity):** Encourage r readout columns C_i^T * H to extract
  different information from the state.

Auxiliary loss options:
```
L_diversity = Σ_{i≠j} |corr(readout_i, readout_j)|   # decorrelate readouts
L_diversity = -H(column_id | readout)                  # each readout distinguishable
```

Bayesian reweighting operates on readout only, at inference time:
```
weight_i ∝ exp(-surprise_i)     # per-column posterior
prediction = Σ_i weight_i * readout_i
```

This preserves rank-r write efficiency while gaining hypothesis diversity
in the readouts. No architectural change — just an auxiliary loss.

### Existing Research on Metacognition in Neural Networks

**Self-modeling validates the introspection head approach:**
- Premakumar et al. 2024 — "Unexpected Benefits of Self-Modeling in Neural
  Systems" (arXiv:2407.10188). Graziano's group. Networks trained to predict
  own activations become simpler, more regularized, more parameter-efficient.
  Tested on MNIST/CIFAR-10/IMDB. Also argue self-modeling may reduce
  catastrophic forgetting. Directly connected to Attention Schema Theory.
- Farrell, Ziman & Graziano 2024 — "Testing Attention Schema Theory in ANNs"
  (arXiv:2411.00983). Agents with learned self-models of attention are better
  at interpreting other agents.

**Internal states already encode metacognitive signals (but models don't exploit them):**
- Lindsey et al. (Anthropic) 2025 — "Emergent Introspective Awareness in LLMs"
  (transformer-circuits.pub). Concept injection shows models can detect injected
  activations ~20% of the time. Multiple narrow circuits, not general.
- "Reasoning Models Know When They're Right" 2025 (arXiv:2504.05419). Hidden
  states encode correctness at intermediate reasoning steps. Probes enable 24%
  token reduction via early exit.
- "No Answer Needed" 2025 (arXiv:2509.10625). Linear probes on question-only
  activations predict answer correctness. Model "knows" before generating.
- "Feeling the Strength but Not the Source" 2025 (arXiv:2512.12411). Models
  detect magnitude of internal activations but not semantic content.

**Architecturally relevant to our introspection head:**
- Osband et al. (NeurIPS 2023) — "Epistemic Neural Networks" (arXiv:2107.08924).
  The epinet: lightweight supplementary module for epistemic uncertainty.
  Architecturally similar to our introspection head. Outperforms ensembles.
- "Emergence of Self-Awareness in Artificial Systems" 2025 (arXiv:2502.06810).
  Multi-layered architecture with cognitive integration + predictive processing
  + internal regulation layers.
- Hu et al. (NeurIPS 2024) — "Uncertainty of Thoughts" (UoT). LLMs model
  own uncertainty during reasoning using information-gain-based rewards.

**Meta-cognition is the least explored frontier in AI:**
- Neuro-symbolic AI systematic review 2025 (arXiv:2501.05435). Only 5% of
  papers address meta-cognition. Critical gap identified.
- Kadavath et al. (Anthropic 2022) — "Language Models (Mostly) Know What They
  Know" (arXiv:2207.05221). The P(IK) framework: foundational work on LLM
  self-knowledge.

## Ablation Testing Plan

**Status:** To be implemented after Phase 6 is complete.

Each architectural upgrade over base Mamba3 must be tested in isolation to
enable proper ablation. Tests must be specifically tailored to each feature.

### Control: Base Mamba3

Baseline for all comparisons. SISO (r=1), scalar decay, no delta rule,
no surprise gating. Both Mamba3 and Naja now support MIMO (added 2026-02-14),
enabling fair r>1 comparisons.

### Feature-Specific Ablation Tests

| Feature | Config Toggle | What to Measure | Expected Signature |
|---------|--------------|-----------------|-------------------|
| **Delta rule** | `use_delta_rule=True/False` | Associative recall accuracy, binding task performance | Delta rule should improve selective state overwrite |
| **PoPE orthogonal pair** | `use_pope_perp=True/False` | State tracking (parity, modular arith), rotation tasks | Second Householder should improve rotation-like state updates |
| **Per-channel decay** | `per_channel_decay=True/False` | Multi-scale temporal tasks (short + long range dependencies) | Per-channel α should show bimodal distribution (fast + slow channels) |
| **StableSSM reparam** | `stable_reparam=True/False` | Long-sequence stability, gradient norms, long-range memory tasks | StableSSM should improve stability when α→1 channels are needed |
| **MIMO** | `mimo_rank=1/2/4` | Throughput (tokens/sec), accuracy on memory-heavy tasks | Higher r should improve hardware utilization without hurting accuracy |
| **Surprise gating** | `use_surprise_gate=True/False` | Predictable vs unpredictable token accuracy, memory efficiency | Should improve on unpredictable tokens, save memory on predictable ones |

### Ablation Protocol

1. Fix all hyperparameters (lr, epochs, d_model, etc.) across comparisons
2. Run each configuration 3x with different seeds
3. Report mean ± std for train/test accuracy
4. For per-channel decay: visualize learned α distributions
5. For surprise gating: visualize β values over sequence positions
6. For MIMO: measure both accuracy and throughput

### Task Suite for Ablation

| Task | Tests Feature | Description |
|------|--------------|-------------|
| Stage 1b (single rule) | Baseline sanity | All configs should pass this (~97%) |
| Stage 2 (5 rules) | Multi-rule generalization | The hard test — look for grokking |
| Associative recall | Delta rule | Store key-value pairs, retrieve by key |
| Parity tracking | PoPE pair | Track parity of a binary sequence |
| Multi-scale memory | Per-channel decay | Remember both recent and distant tokens |
| Permutation tracking | Full architecture | Track element positions through swaps |

## Key References

### Architecture
- Mamba3: ICLR 2026 submission, OpenReview HwCvaJOiCj
- Gated DeltaNet: Yang et al., ICLR 2025, arXiv:2412.06464
- DeltaProduct: Siems et al., NeurIPS 2025, arXiv:2502.10297
- KDA / Kimi Linear: arXiv:2510.26692
- StableSSM: Wang & Li, ICML 2024, arXiv:2311.14495
- PoPE: Gopalakrishnan et al. 2024
- EB-JEPA: Terver et al. 2026, arXiv:2602.03604
- Epistemic Neural Networks: Osband et al., NeurIPS 2023, arXiv:2107.08924

### Causal Induction
- NOTEARS: Zheng et al. 2018, arXiv:1803.01422
- Neural Granger Causality: Tank et al. 2022, arXiv:1802.05842
- Directed Information: Massey 1990, IEEE Trans. Info. Theory
- Transfer Entropy: Schreiber 2000, Physical Review Letters

### Metacognition & Self-Modeling
- Unexpected Benefits of Self-Modeling: Premakumar et al. 2024, arXiv:2407.10188
- Testing Attention Schema Theory in ANNs: Farrell et al. 2024, arXiv:2411.00983
- Emergent Introspective Awareness: Lindsey et al. (Anthropic) 2025
- Reasoning Models Know When They're Right: 2025, arXiv:2504.05419
- No Answer Needed (question-only probes): 2025, arXiv:2509.10625
- Partial Introspection: 2025, arXiv:2512.12411
- Uncertainty of Thoughts: Hu et al., NeurIPS 2024
- P(IK) framework: Kadavath et al. (Anthropic) 2022, arXiv:2207.05221
- Emergence of Self-Awareness: 2025, arXiv:2502.06810
- Neuro-symbolic meta-cognition review: 2025, arXiv:2501.05435

### Theories of Consciousness
- Attention Schema Theory: Graziano 2013, "Consciousness and the Social Brain"
- Global Workspace Theory: Baars 1988, "A Cognitive Theory of Consciousness"
- Free Energy Principle: Friston 2010, Nature Reviews Neuroscience
- Predictive Processing: Clark 2013, "Whatever Next?"
