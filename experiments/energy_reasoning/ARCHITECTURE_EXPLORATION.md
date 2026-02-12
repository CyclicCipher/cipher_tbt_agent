# Reasoning Network: Architecture Exploration

## The Core Question

What does the reasoning network look like? Specifically:
1. What is the state that gets optimized during reasoning?
2. How does the energy function work?
3. How is the energy function made appropriate for different tasks?
4. How is energy minimization done?
5. What is the overall architecture?

This document explores these questions, including a critical examination of whether
discrete "slot" structures are the right approach or whether dense superposition
(as in transformers and Mamba) is more appropriate.

---

## What Is the State?

The state is the latent variable `z` that Langevin dynamics operates on during
inference. It represents the system's current "thought" — the configuration
being optimized to satisfy all constraints simultaneously.

### Relationship to Mamba's Internal State

Mamba's internal state `h ∈ R^{d_state × d_model}` is a running compression
optimized for sequential prediction. It's NOT a workspace — it's a summary.
You can't meaningfully run gradient-based optimization on it because:
- It's designed for compression, not manipulation
- Its structure is tied to the recurrence dynamics
- Modifying it arbitrarily would break the SSM's sequential consistency

Instead, Mamba's role is to PRODUCE the context representation that initializes z:

```
Input sequence → Mamba3 Encoder → s_context ∈ R^{d_model}
                                       ↓
                              z ~ N(0, I) ∈ R^{d_z}    (independent latent)
                                       ↓
                              Predictor(s_context, z) → s_predicted
```

The encoder compresses the input into a rich representation. The latent z is
a SEPARATE variable that the predictor conditions on. Langevin dynamics searches
over z, not over s_context. This separation is important — the encoder's job
is perception (compress input), while z's job is reasoning (find the right
latent configuration).

### Two Hypotheses for the Structure of z

#### Hypothesis A: Slot-Based Workspace

z = {z_1, ..., z_K} where each z_i ∈ R^{d_slot}, giving Z ∈ R^{K × d_slot}.

Each slot represents a discrete concept or variable. Energy terms are defined
per-slot and per-pair:

```
E(Z) = Σ_i f_self(z_i) + Σ_{i,j} f_pair(z_i, z_j, r_ij) + ...
```

Initialization would require an "unpacking" step:
```
s_context → K learned linear projections → {z_1, ..., z_K}
```

And a "repacking" step to feed back into the decoder.

**Arguments for slots:**
- Natural fit for constraint satisfaction (Sudoku, logic puzzles)
- Explicit variable binding (each slot = one variable)
- Pairwise energy terms map directly to constraints between variables
- Interpretable — you can inspect what each slot represents

**Arguments against slots:**
- Requires choosing K (number of slots) as a hyperparameter
- Requires a slot assignment mechanism (which concept goes where?)
- Imposes structure that may not match the problem
- Transformers and Mamba DON'T use discrete slots — they hold multiple ideas
  in superposition within a dense vector, and this clearly works
- Adds architectural complexity before it's proven necessary

#### Hypothesis B: Dense Superposition (Currently Favored)

z ∈ R^d is a single dense vector. Multiple concepts are encoded in
superposition within z, the same way transformers encode multiple features
in superposition across their residual stream.

```
E(z) = E_pred(z) + α·E_constraint(z) + β·E_consistency(z)
```

The energy function operates on ALL of z jointly. The predictor (narrow Mamba3)
has learned to interpret z as encoding multiple things simultaneously.

**Why this might be the right approach:**

1. **No K to choose.** The effective number of "concepts" is determined by
   the task and the capacity of z, not by an architectural hyperparameter.

2. **No assignment problem.** Slot-based approaches require deciding which
   concept goes in which slot — this is a discrete combinatorial problem
   embedded inside a continuous optimization, which is awkward. Dense z
   avoids this entirely.

3. **Exploits correlations.** Langevin dynamics adjusts all of z jointly,
   naturally exploiting correlations between concepts. Slot-based approaches
   with per-slot updates would miss cross-slot structure unless pairwise
   terms are very carefully designed.

4. **VICReg provides natural organization.** The VICReg regularization
   (variance + covariance) decorrelates dimensions of the representation
   space. This means semi-independent "directions" in z emerge naturally
   from training. These are like soft slots — but their number, size, and
   nature are learned, not imposed.

5. **Consistent with what works.** Transformers, Mamba, and all successful
   sequence models use dense representations. Multi-head attention LOOKS
   like slots (each head attends to different things), but the heads
   emerged from training, they weren't designed for specific concepts.

6. **Simpler = better for Phase 1.** Following the meta-principle that
   successful architectures start minimal and add structure only when
   forced to by empirical failure.

**When would we need slots?** If dense z fails on problems that require
explicit variable binding (e.g., "swap the values of X and Y"), the failure
mode would be diagnostic: you'd see that z can't simultaneously represent
"X has value A" and "Y has value B" without interference. At that point,
adding slot structure would be motivated by evidence, not assumption.

---

## The Energy Function

### What It Must Satisfy

The energy function E(z) must be:
- **Differentiable** — Langevin dynamics requires ∇_z E(z)
- **Task-adaptive** — different tasks impose different constraints
- **Low at good solutions** — energy minimization should find correct answers
- **Learnable** — the energy landscape is shaped by training, not hand-designed

### The Three-Term Structure

From the ROADMAP:

```
E(z) = E_pred(z) + α·E_constraint(z) + β·E_consistency(z)
```

#### E_pred: Prediction Energy

```
E_pred(z) = ||Predictor(s_context, z) - s_target||²
```

This is the JEPA energy. It measures: "does this z produce a latent prediction
that matches reality?" During training, s_target comes from the EMA target
encoder. During inference, we don't have s_target — so E_pred measures
self-consistency of the prediction (how well the predictor's output matches
what the encoder would produce for the predicted output).

**Key insight:** E_pred is task-adaptive FOR FREE. The encoder and predictor
learn task-specific representations during training. The energy landscape
they define automatically reflects the structure of the training distribution.
No manual constraint engineering needed.

#### E_consistency: Internal Coherence

```
E_consistency(z) = Σ_t ||s_pred_t - Predictor(s_pred_{t-1})||²
```

Measures whether different parts of the predicted sequence are mutually
consistent. If position t-1 predicts something about position t, and position
t predicts something about position t+1, these predictions should form a
coherent chain.

**For dense z:** This becomes even more natural. The predictor processes
z as a conditioning signal — consistency means that the SAME z produces
coherent predictions across all positions. If z encodes contradictory
information (concept A in one direction, contradicting concept B in another),
the predictor's outputs will be incoherent and E_consistency will be high.

#### E_constraint: Knowledge-Grounded Energy (Phase 3+)

```
E_constraint(z) = -score(CTKG_paths(s_context, Predictor(s_context, z)))
```

Measures how well the predicted reasoning aligns with valid paths in the
Category Theory Knowledge Graph. This is where external knowledge shapes
the energy landscape.

**How this makes energy task-appropriate without architectural changes:**
The CTKG encodes which reasoning steps are valid compositions for different
domains. When reasoning about chemistry, the CTKG provides chemistry
morphisms. When reasoning about logic, it provides logical inference rules.
The same energy function, the same architecture, but the CTKG responses
completely change the energy landscape.

### How the Energy Function Is Made Task-Appropriate

There are three mechanisms, at increasing levels of specificity:

**1. Learned representations (automatic, from training):**
The encoder and predictor learn what matters for the training distribution.
E_pred is automatically task-relevant because the representations are.

**2. VICReg shaping (automatic, from regularization):**
VICReg decorrelates dimensions, which means the energy landscape has
semi-independent axes. This prevents collapse and ensures z has enough
capacity to represent task-relevant variation.

**3. CTKG modulation (explicit, from knowledge graph):**
The CTKG provides external constraints that reshape the energy landscape
for specific reasoning domains. This is the only component that requires
domain-specific knowledge — and it's separated out into the knowledge
graph rather than baked into the architecture.

---

## How Energy Minimization Works

### Langevin Dynamics

```python
z = torch.randn(d_z)  # Initialize from noise
for step in range(T):
    E = energy_fn(s_context, z)
    grad_z = torch.autograd.grad(E, z)[0]
    noise = torch.randn_like(z) * sigma_t
    z = z - eta * grad_z + noise
```

This is gradient descent with noise. The gradient tells z which direction
reduces energy. The noise helps escape local minima.

### Noise Schedule (Annealing)

```
sigma_t = sigma_0 · (1 - t/T)
```

Early steps: high noise, broad exploration of the energy landscape.
Late steps: low noise, convergence to a specific minimum.
At test time: drop noise entirely for deterministic gradient descent.

### Adaptive Stopping

Rather than fixed T, run until convergence:

```python
for step in range(max_T):
    z_new = langevin_step(z)
    if |E(z_new) - E(z)| < threshold:
        break
    z = z_new
```

This gives variable compute — easy problems converge fast (few steps),
hard problems take more steps. This directly addresses one of the core
bottlenecks of autoregressive models (fixed compute per token).

### Step Size Considerations

The step size η matters enormously:
- Too large: overshoots minima, diverges
- Too small: takes forever, gets stuck in local minima
- The noise scale σ should be coupled to η: σ = √(2η) for proper
  Langevin dynamics (this comes from the SDE discretization)

For practical implementation, learned or adaptive step sizes are likely
needed. The EBM-CoT paper (Chen et al.) found T=3 steps optimal with
a tuned η ≈ 0.01, but this will be task-dependent.

---

## The Architecture (Slot-Free Version)

### Overview

```
┌──────────────────────────────────────────────────────────┐
│                    REASONING SYSTEM                       │
│                                                           │
│   Input                                                   │
│     │                                                     │
│   [Context Encoder]  (Mamba3, 4 layers, d=128)            │
│     │                                                     │
│   s_context ∈ R^{d_model}                                 │
│     │                                                     │
│   z ~ N(0,I) ∈ R^{d_z}   (d_z ≈ 64, independent latent)  │
│     │                                                     │
│   ┌─── Langevin Loop (until convergence) ───┐             │
│   │                                          │             │
│   │  [Predictor] (narrow Mamba3, 2 layers)   │             │
│   │  Input: s_context + z                    │             │
│   │  Output: s_predicted                     │             │
│   │                                          │             │
│   │  E = E_pred + α·E_constraint             │             │
│   │      + β·E_consistency                   │             │
│   │                                          │             │
│   │  z ← z - η·∇_z E + σ_t·noise            │             │
│   │                                          │             │
│   └──────────────────────────────────────────┘             │
│     │                                                     │
│   z* (converged)                                          │
│     │                                                     │
│   s_pred* = Predictor(s_context, z*)                      │
│     │                                                     │
│   [Decoder] (linear + layernorm → output)                 │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Component Details

**Context Encoder** (Mamba3, reuse existing mamba3_block.py)
- 4 layers, d_model=128
- Processes input sequence into rich representation s_context
- Trained via JEPA: must produce representations that the predictor
  can use to predict target representations
- EMA copy serves as target encoder during training

**Predictor** (narrow Mamba3, 2 layers, d_model=64)
- Takes s_context and z as input
- z is injected via concatenation or additive conditioning:
  ```
  predictor_input = s_context + W_z · z   (additive, simpler)
  predictor_input = [s_context; W_z · z]  (concatenation, more capacity)
  ```
- The narrow bottleneck forces the encoder to produce rich s_context
  rather than letting the predictor memorize
- This is the component that gets run T times during Langevin dynamics,
  so it MUST be cheap — narrow Mamba3 at d=64 gives O(n) per step

**Energy Function** (differentiable, composed)
- E_pred: L2 distance between predicted and target representations
- E_consistency: coherence across sequence positions
- E_constraint: CTKG path alignment (Phase 3+)
- All terms are computed from the predictor's output, which depends on z
  through the predictor's forward pass — this provides the gradient ∇_z E

**Decoder** (simple linear projection)
- Maps s_pred* back to output space
- Should be minimal — the reasoning happens in latent space, not output space
- Linear + LayerNorm → logits (for classification/generation)

### How z Encodes Multiple Concepts Without Slots

In a dense z ∈ R^d, different linear directions encode different concepts.
This is exactly how transformer residual streams work — Anthropic's
superposition research shows that models encode MORE features than they
have dimensions by using nearly-orthogonal directions.

For our system:
- Training with VICReg ensures dimensions are decorrelated (no collapse)
- The predictor learns to READ z by projecting it through W_z
- Different entries/directions in z come to represent different aspects
  of the reasoning state
- Langevin dynamics adjusts ALL directions simultaneously, which is
  important because reasoning concepts aren't independent — changing one
  conclusion can cascade to others

**The energy landscape IS the reasoning structure.** If concept A and
concept B are contradictory, the energy function (through E_consistency)
creates a ridge between the region of z-space where A is encoded and
the region where B is encoded. Langevin dynamics naturally falls into
one valley or the other. Noise helps explore both options before committing.

### How z Conditions the Predictor

There are several options for how z influences the predictor's computation:

**Option 1: Additive input conditioning**
```python
# z projected and added to each position of the predictor input
z_proj = self.z_projector(z)  # R^{d_z} → R^{d_predictor}
predictor_input = s_context + z_proj.unsqueeze(0)  # broadcast over sequence
```
Simplest. z acts as a global bias on all positions.

**Option 2: FiLM conditioning (feature-wise linear modulation)**
```python
# z produces per-layer scale and shift
gamma, beta = self.film_generator(z).chunk(2, dim=-1)
# Applied inside each Mamba3 block:
h = gamma * h + beta
```
More expressive. z can selectively amplify or suppress features. This
is how many conditional generation models work (StyleGAN, etc.).

**Option 3: Cross-attention conditioning**
```python
# z as key/value, predictor hidden states as query
attn_output = cross_attention(query=h, key=z, value=z)
```
Most expressive but heaviest. Also breaks the pure-Mamba O(n) property.

**Recommendation: Start with Option 1 (additive), try Option 2 (FiLM) if
needed.** Option 1 is the minimal realization. If it fails, the failure mode
will indicate whether we need more expressive conditioning.

---

## How This Differs From the Slot-Based Proposal

| Aspect | Slot-Based | Dense z |
|--------|-----------|---------|
| State | Z ∈ R^{K×D_slot} | z ∈ R^d |
| # of concepts | Fixed K | Learned/emergent |
| Energy terms | Per-slot + pairwise | Holistic over full z |
| Assignment | Explicit (slot i = concept j) | Implicit (directions in z) |
| Initialization | Unpack s_context → K projections | z ~ N(0,I) |
| CTKG interface | Per-slot queries | Project z → query, retrieve, modulate E |
| Interpretability | High (inspect slots) | Lower (need probing) |
| Complexity | Higher (K, D_slot, assignment) | Lower (just d_z) |
| Scaling | Add more slots | Increase d_z |

The dense approach trades interpretability for simplicity and generality.
If interpretability becomes critical (e.g., for debugging reasoning chains),
we can add probing/projection tools without changing the architecture.

---

## Open Questions

### 1. How should z be initialized?

Current plan: z ~ N(0,I). But should the encoder inform z's initialization?

```python
# Option A: Pure noise (current plan)
z = torch.randn(d_z)

# Option B: Encoder-informed initialization
z = self.z_initializer(s_context) + torch.randn(d_z) * init_noise_scale
```

Option B gives Langevin dynamics a "warm start" — z starts near a reasonable
configuration rather than from random noise. This could dramatically reduce
the number of steps needed. But it also means z and s_context aren't cleanly
separated, which might cause the predictor to ignore z (since s_context
already contains all the information and z starts near it).

**Recommendation:** Start with Option A (pure noise) to ensure z is actually
used. If convergence is too slow, try Option B with high init_noise_scale.

### 2. What dimensionality for z?

- Too small: can't encode enough concepts simultaneously
- Too large: Langevin dynamics searches a high-dimensional space (slow)

The EBM-CoT paper uses d_z comparable to the model's hidden dimension.
Starting with d_z = 64 (half of d_model = 128) seems reasonable. This gives
64 semi-independent directions for encoding concepts (after VICReg
decorrelation).

### 3. How does E_pred work at inference time?

During training: `E_pred = ||Predictor(s_context, z) - s_target||²`
where s_target comes from the EMA target encoder.

During inference: we don't have s_target. Options:

**Option A: Self-prediction consistency**
```
Encode partial input → s_context
Use Predictor to predict masked/future positions
E_pred measures how well these predictions agree with each other
```

**Option B: Decoder confidence**
```
s_pred = Predictor(s_context, z)
output = Decoder(s_pred)
E_pred = -log_confidence(output)  # e.g., negative log probability
```
Low confidence = high energy. This makes z optimize for confident predictions.

**Option C: Learned energy head**
```
E_pred = EnergyHead(s_context, z)  # small MLP trained to output scalar energy
```
Train the energy head alongside the JEPA system. This is most flexible but
requires careful training to avoid trivial solutions.

**Recommendation:** Option A for Phase 1 (stays closest to JEPA framework),
explore Option B if we need output-level signal.

### 4. How to train the energy function?

The energy function is implicitly defined by the encoder, predictor, and any
explicit energy heads. Training options:

**JEPA training (current plan):** Train encoder + predictor with L2 prediction
loss + VICReg. The energy landscape emerges from the learned representations.
No explicit energy training needed.

**Contrastive energy training:** Also show the system "bad" configurations
(corrupted z, wrong answers) and train E to be high for those. This provides
more direct supervision of the energy landscape.

**Denoising score matching:** Train ∇_z E directly by learning to denoise
corrupted z. This is how diffusion models train their score functions and
is theoretically well-grounded.

**Recommendation:** Start with JEPA (simplest, already planned). If the
energy landscape is too flat or has too many spurious minima, add contrastive
or denoising objectives.

### 5. Gradient flow through Langevin steps during training?

If we want to train the energy function end-to-end (backprop through the
Langevin loop), we need to unroll T steps and backpropagate through all of
them. For T=3-5, this is manageable. For larger T, options:

- **Truncated backprop:** Only backprop through the last K steps
- **Implicit differentiation:** Use the implicit function theorem at the
  fixed point z* to compute gradients without unrolling
- **Straight-through:** Treat z* as given, don't backprop through Langevin

The ROADMAP already specifies "Training = backprop. Inference = energy
minimization." This means we DON'T backprop through Langevin during training
— we train with standard JEPA loss, and Langevin is only used at inference.
This sidesteps the gradient flow question entirely for Phase 1.

### 6. Connection to ePC

ePC-Mamba already does energy minimization (Newton iterations on errors).
The energy reasoning system adds a SECOND level of energy minimization
(Langevin on z). How do these relate?

**Option A: Replace ePC with Langevin**
Use standard backprop for training (as in ROADMAP). Langevin only at
inference. ePC is not used.

**Option B: ePC for training, Langevin for inference**
Use ePC's local learning during training (biologically plausible, avoids
vanishing gradients). Use Langevin on z at inference (more flexible than
ePC's Newton steps). Two different energy minimization systems for two
different purposes.

**Option C: Nested minimization**
ePC optimizes errors within each Mamba3 block (inner loop).
Langevin optimizes z across the whole system (outer loop).
Each Langevin step runs a few ePC iterations internally.

**Recommendation:** Option A for Phase 1 (keep it simple — backprop training
is proven to work, see ROADMAP). Explore Option B/C later if biological
plausibility or local learning becomes important.

---

## Summary of Recommendations for Phase 1

1. **State:** Dense z ∈ R^{64}, not slots
2. **Initialization:** z ~ N(0,I), not encoder-informed
3. **Conditioning:** Additive (z projected and added to predictor input)
4. **Energy:** E_pred (JEPA self-prediction) + E_consistency (sequential coherence)
5. **Minimization:** Langevin with annealing, adaptive stopping, T≈3-5 steps
6. **Training:** Standard backprop + JEPA + VICReg (no backprop through Langevin)
7. **Architecture:** Mamba3 encoder (4-layer, d=128) + narrow Mamba3 predictor (2-layer, d=64) + linear decoder
8. **Validation task:** Sorting or simple logic (needs multi-step reasoning, unlike copy task)

If this fails, the failure mode tells us what to add:
- z too small → increase d_z
- Concepts interfere → add VICReg strength or try slots
- Energy landscape too flat → add contrastive training
- Too many Langevin steps → add encoder-informed initialization
- Can't compose reasoning → add CTKG (Phase 3)
