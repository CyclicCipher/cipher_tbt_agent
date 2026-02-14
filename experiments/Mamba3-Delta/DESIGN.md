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

## Naming

**Codename: Naja** (pending final decision)

"Mamba3-Delta" is descriptive but clunky. We considered "Mamba4" but it implies
ownership of the Mamba lineage (Gu & Dao), which would be a breach of etiquette
in the research community.

**Naja** is the genus of cobras, in the same family (Elapidae) as Dendroaspis
(the mamba genus). It maintains the snake evolutionary lineage while being a
distinct identity. The name is:
- One word, short, pronounceable ("NAH-jah")
- Taxonomically linked to Mamba (same family, sister genus)
- Completely clean in ML namespace (no existing model/framework uses it)
- Culturally neutral (no negative connotations)

Runner-up was "Hydrus" (mythological water serpent), but Naja has a stronger
biological connection. All other candidates (Hydra, Cobra, Taipan, Chimera,
Mantis, Viper, Basilisk, Ouroboros) are taken by existing ML projects.

Note: The codebase still uses `Mamba3Delta` / `mamba3_delta` internally.
Rename to `Naja` / `naja` once the name is confirmed.

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

Per-layer introspection (running a separate head after each Mamba layer)
is more powerful but can be explored later. Start with final-layer only,
matching the Graziano group's experimental setup and the epinet architecture.

### Introspection Head Training Objectives

The introspection head needs its own auxiliary losses — without them it has
no gradient signal. We use THREE complementary objectives, ordered by
importance:

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

**Secondary: Confidence estimation**

```
L_conf = MSE(IntroHead(h_t)_error, sg(|y_{t+1} - target_{t+1}|))
```

Predict the magnitude of the next prediction error. Continuous signal,
more informative than binary calibration. There is no evidence that
current LLMs have access to their own loss/perplexity — this explicitly
trains the model to have an internal "uncertainty meter."

Combined with per-channel structure, this enables causal attribution of
uncertainty: if confidence correlates with slow-decay channels (α≈1),
the model is uncertain about long-range context; if it correlates with
fast-decay channels (α≈0), it's uncertain about recent context; if it
correlates with specific MIMO columns, it's uncertain about specific
hypotheses.

The architecture makes the "why" readable — the confidence head learns
to read a structure that per-channel decay and MIMO diversity already
provide. The introspection head doesn't need to discover causation from
scratch.

**Tertiary: Calibration loss**

```
L_cal = BCE(IntroHead(h_t)_correct, sg(actual_correctness_{t+1}))
```

Predict whether the next prediction will be correct. Binary companion
to the continuous confidence estimate. Useful for hard strategy-switching
decisions (e.g., "confident enough to commit" vs "should fall back").

This differs from simply training the base model: the base model
optimizes p(x_{t+1} | context) (tries to be RIGHT). The calibration
head optimizes p(correct | h_t) (tries to KNOW WHEN it's right).
These are first-order vs. second-order objectives. The calibration head
learns a meta-model of what characterizes correct outputs — patterns
in the internal state that predict success vs failure.

**Optional: State stability prediction**

```
L_stab = MSE(IntroHead(h_t)_delta, sg(||h_{t+1} - h_t||))
```

Predict how much the state will change at the next step. Provides
anticipatory surprise (forward-looking, not reactive like the basic
surprise signal). If the model predicts large Δh, it expects the next
input to be surprising and can prepare. Persistent mismatch (predict
small Δh, observe large Δh) signals an unfamiliar regime — qualitatively
different from single-step surprise.

Predicting summary statistics of Δh (norm, per-channel magnitudes) is
more useful and cheaper than predicting the full next state.

**Optional: Hypothesis identification (MIMO)**

```
L_hyp = CE(IntroHead(h_t)_column, sg(best_column))
```

Predict which MIMO column is most accurate. Makes implicit hypothesis
tracking explicit — the model can then reason about which hypothesis is
active and why.

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
