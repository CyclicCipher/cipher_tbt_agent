# Energy Reasoning: Validation Plan

## Design Philosophy

Every previous experiment in this project followed the same pattern:
- **Standard PC:** MNIST — simplest task that requires layer-wise prediction
- **eBPC:** MNIST — same task, proving the new algorithm matches/exceeds
- **ePC-Mamba:** Copy task — simplest SEQUENCE task that requires memory

The pattern: use the **simplest task that REQUIRES the capability** being
tested. Not a hard task. The minimal task where the capability is necessary
and its absence is detectable.

For energy-based reasoning, the capability is: **iterative refinement of a
latent variable z improves predictions beyond what a single forward pass
achieves.** The test must create a measurable gap between "with Langevin"
and "without Langevin."

---

## The Problem With Copy, MNIST, and Sorting

**Copy task:** Tests memory/storage. Mamba already does this perfectly in a
single forward pass. Langevin on z adds nothing — there's nothing to refine.
Not diagnostic.

**MNIST:** Single-step classification. One forward pass through an encoder
gives 95%+. No multi-step reasoning needed. Not diagnostic.

**Sorting:** Tempting, but a well-trained Mamba can likely sort short
sequences in a single pass (it's a fixed function with O(n log n)
comparisons). Langevin would only help if the single-pass model makes
errors — but then we're testing model capacity, not reasoning capability.

We need tasks where the **structure of the problem** demands iterative
refinement, not just tasks that are hard because the model is small.

---

## Phased Validation: Three Stages

### Stage 1: Prove JEPA Learns (No z, No Langevin)

**Goal:** Verify the encoder learns useful representations via self-
supervised prediction. This is the foundation — if JEPA doesn't work,
nothing else matters.

**Task: Masked Sequence Prediction**

```
Input:   [a, b, _, d, _, f]     (_ = masked positions)
Encode:  s_context = encoder([a, b, MASK, d, MASK, f])
Predict: s_pred = predictor(s_context)  (no z yet)
Target:  s_target = target_encoder([a, b, c, d, e, f])
Loss:    ||s_pred[masked_positions] - s_target[masked_positions]||²
```

**Data generation:** Sequences with learnable structure. NOT random i.i.d.
tokens (nothing to predict) and NOT trivially repetitive (too easy).

Three data types, in order of complexity:

**1a. Arithmetic sequences** (easiest)
```python
def generate_arithmetic(n_samples, seq_len, vocab_size):
    """a, a+d, a+2d, a+3d, ... (mod vocab_size)"""
    a = torch.randint(0, vocab_size, (n_samples, 1))
    d = torch.randint(1, vocab_size, (n_samples, 1))
    positions = torch.arange(seq_len).unsqueeze(0)
    sequences = (a + d * positions) % vocab_size
    return sequences
```
Predictable, structured, variable difficulty (small d = easy, large d with
mod wrapping = harder). Masking any position should be predictable from
its neighbors.

**1b. Multi-rule sequences** (medium)
```python
def generate_multi_rule(n_samples, seq_len, vocab_size):
    """First half follows rule A, second half follows rule B."""
    # e.g., first half: +2, second half: *3 mod vocab_size
    # The model must discover the rule change
```
This starts to require something like "what rule generated this?" which
is a precursor to reasoning.

**1c. Interleaved sequences** (hardest for Stage 1)
```python
def generate_interleaved(n_samples, seq_len, vocab_size):
    """Two independent patterns interleaved: A_1 B_1 A_2 B_2 A_3 B_3
    A follows one rule, B follows another. Model must track both."""
```
Forces the encoder to represent two concepts simultaneously. This probes
whether the representation space has enough capacity for multi-concept
encoding (relevant to the dense-z-vs-slots question).

**Masking strategy:** Mask 15-30% of positions randomly (following BERT/
I-JEPA convention). Evaluate L2 prediction error on masked positions only.

**Success criteria:**
- L2 prediction error decreases over training
- VICReg monitors stay healthy (variance ≥ 1 per dim, low covariance)
- Representation dimensionality is high (not collapsed)
- Prediction error on arithmetic < multi-rule < interleaved (sanity check)

**Failure diagnosis:**
- Error doesn't decrease → encoder/predictor architecture problem
- VICReg variance collapses → increase VICReg weight or check EMA tau
- Error decreases but predictions are blurry → predictor too weak (increase d)

**Config:**
```
encoder:     4-layer Mamba3, d_model=128
predictor:   2-layer Mamba3, d_model=64 (no z input yet)
target_enc:  EMA of encoder, tau=0.996 → 1.0
optimizer:   AdamW, lr=1e-3
batch_size:  32
seq_len:     64
vocab_size:  16 (matching copy task for comparison)
mask_ratio:  0.2
epochs:      30
train_size:  5000
test_size:   1000
```

**Metrics to track:**
- L2 prediction error (train and test, per epoch)
- VICReg components: L_variance, L_covariance (per epoch)
- Effective dimensionality of encoder output (rank of covariance matrix)
- Per-position prediction error (are some positions harder than others?)
- Visual: plot a few example predictions vs targets

---

### Stage 2: Prove z Matters (z Added, Langevin Added)

**Goal:** Show that (a) z changes the predictor's output, (b) some z
values are better than others, and (c) Langevin finds good z values.

**Task: Ambiguous Masked Prediction**

The key insight: if masking is easy (one masked position surrounded by
context), z isn't needed — s_context provides enough information. We
need tasks where the context is AMBIGUOUS and z must resolve the ambiguity.

**2a. High mask ratio prediction**
```
Mask 50-70% of positions. With most of the sequence hidden, many different
completions are plausible. z should encode WHICH completion the predictor
produces. Different z → different (but internally consistent) completions.
```
This directly tests whether z is used and whether different z values
produce different outputs.

**2b. Pattern induction (the key diagnostic task)**
```
Input:  [examples of f(x) → y] + [query x] + [MASK for y]
        e.g., [2→4, 3→6, 5→10, 7→?]

The function f is randomly chosen per sample:
  f(x) = 2x      (doubling)
  f(x) = x + 3   (shift)
  f(x) = x²      (squaring, mod vocab_size)
  f(x) = x XOR 5 (bitwise)
```

This is the critical task. It requires TWO steps:
1. **Infer the rule** from examples (what is f?)
2. **Apply the rule** to the query (compute f(7))

z encodes the rule hypothesis. Langevin searches over rule hypotheses
to find the one consistent with all examples. This is genuine reasoning
— not pattern matching, not memory recall, but hypothesis formation and
testing.

**Why this task is diagnostic for EACH component:**

| Component | What It Tests | Expected Failure Without |
|-----------|--------------|------------------------|
| JEPA encoder | Can it represent examples as a pattern? | Random predictions |
| z | Does it encode the rule hypothesis? | Same prediction regardless of rule |
| Langevin | Does iterative refinement find the right rule? | Worse accuracy than oracle z |
| E_pred | Does prediction error decrease with correct rule? | Flat energy landscape |
| Noise | Does noise help escape wrong rule hypotheses? | Gets stuck on first guess |

**Data generation for pattern induction:**
```python
def generate_pattern_induction(n_samples, n_examples, vocab_size, rules=None):
    """
    Generate few-shot function learning tasks.

    Each sample: [x1, f(x1), x2, f(x2), ..., x_q, MASK]
    Target: f(x_q)

    rules: list of functions to sample from
    """
    if rules is None:
        rules = [
            lambda x, v: (2 * x) % v,          # doubling
            lambda x, v: (x + 3) % v,          # shift
            lambda x, v: (x * x) % v,          # squaring
            lambda x, v: x ^ 5,                 # XOR
            lambda x, v: (v - 1 - x) % v,      # complement
        ]

    sequences = []
    targets = []
    for _ in range(n_samples):
        rule = random.choice(rules)
        xs = random.sample(range(1, vocab_size), n_examples + 1)
        seq = []
        for x in xs[:-1]:
            seq.extend([x, rule(x, vocab_size)])  # x, f(x) pairs
        seq.append(xs[-1])  # query x
        seq.append(0)       # MASK token (to be predicted)
        sequences.append(seq)
        targets.append(rule(xs[-1], vocab_size))

    return torch.tensor(sequences), torch.tensor(targets)
```

**Ablation structure (critical — this is what makes it diagnostic):**

Run four variants on the same data:

1. **No z (predictor ignores z):** Baseline. Can the predictor solve
   pattern induction from s_context alone? Expected: moderate accuracy
   (it can memorize common rules but struggles with ambiguous examples).

2. **Random z (no Langevin):** z is sampled but not optimized.
   Expected: similar to no-z (random z doesn't help).

3. **Langevin-refined z:** Full system. Expected: best accuracy.
   The gap between this and #1/#2 measures the VALUE of reasoning.

4. **Oracle z (upper bound):** Give the model the correct rule label
   as z (e.g., one-hot encode the rule index, project to d_z). This
   is the upper bound — if the model could perfectly identify the rule,
   how well would it predict? Expected: near-perfect accuracy.

The key metric is the **Langevin gap**:
```
Langevin_gap = accuracy(variant 3) - accuracy(variant 1)
```
If this is positive and significant, iterative refinement helps. If it's
near zero, z/Langevin aren't contributing.

**Additional ablation: steps sweep**
Run variant 3 with T = {0, 1, 2, 3, 5, 10} Langevin steps. Plot
accuracy vs T. Expected: accuracy increases with T up to some point,
then plateaus. The shape of this curve reveals:
- Steep improvement at T=1-3: energy landscape is well-shaped
- No improvement: landscape is flat (JEPA training failed)
- Improvement only at large T: landscape has bad local minima (need better noise)

**Config for Stage 2:**
```
Same encoder/predictor as Stage 1, but:
  predictor now takes z as input (additive conditioning)
  d_z = 64
  n_examples = 3-5 (few-shot examples per sample)
  n_rules = 5 (start small, increase to test scaling)
  vocab_size = 16
  seq_len = determined by 2*n_examples + 2 (e.g., 12 for 5 examples)
  Langevin steps T = 3 (default), sweep {0,1,2,3,5,10}
  Noise: cyclical annealing, 1 cycle over T steps
  epochs = 50
```

**Success criteria:**
- Langevin_gap > 5% accuracy (z+Langevin meaningfully helps)
- Energy decreases monotonically over Langevin steps
- More steps → better accuracy (at least T=0 to T=3)
- Oracle z achieves ≥90% accuracy (the task is solvable in principle)
- VICReg stays healthy throughout

**Failure diagnosis:**
- Langevin_gap ≈ 0 → z is being ignored. Check: does the predictor's
  output actually change when z changes? If not, the additive conditioning
  is too weak → try FiLM conditioning.
- Energy doesn't decrease over steps → gradient ∇_z E ≈ 0. The energy
  landscape is flat w.r.t. z. Check: is E_pred actually a function of z?
  Might need to verify the computational graph connects z to E.
- Oracle z also fails → the task is too hard for this architecture.
  Reduce n_rules or increase model capacity.
- Good energy decrease but bad accuracy → the energy function doesn't
  align with task performance. E_pred minimization isn't finding task-
  relevant z values. May need contrastive training to shape the landscape.

---

### Stage 3: Probe Scaling and Limitations

**Goal:** Find where the system breaks and understand why.

**3a. Rule scaling**
Increase n_rules from 5 → 10 → 20 → 50. At what point does the Langevin
gap shrink? This reveals the capacity of z ∈ R^{64} — how many distinct
"rule hypotheses" can it encode?

Expected: performance degrades gracefully up to ~d_z rules (64), then
drops sharply. If it drops earlier, the effective dimensionality is
lower than d_z (VICReg may be too weak).

**3b. Example scaling**
Vary n_examples from 1 → 2 → 3 → 5 → 10. With more examples, the rule
should be easier to identify. Does Langevin find the correct z faster
(fewer steps) with more examples?

Expected: accuracy increases with n_examples. Steps to convergence
decreases. If not, s_context isn't effectively encoding the examples.

**3c. Compositional rules**
```python
# f(x) = g(h(x)) where g and h are simple rules
# e.g., f(x) = 2*(x+3) mod V
compose_rules = [
    (double, shift3),    # 2*(x+3)
    (shift3, double),    # (2x)+3  -- different!
    (square, shift3),    # (x+3)²
]
```
Can the system discover COMPOSED rules? This tests compositional
reasoning — the precursor to multi-hop inference.

Expected: harder than simple rules. If accuracy drops dramatically,
this identifies the need for CTKG (Phase 3) — external knowledge about
what compositions are valid.

**3d. Transfer / generalization**
Train on rules {double, shift3, square}. Test on HELD-OUT rules
{XOR5, complement}. Does the system generalize to rules it hasn't seen?

Expected: poor generalization (the model hasn't seen these rules).
But the Langevin gap should still be positive — even if overall accuracy
is low, iterative refinement should still help compared to single-pass.

If Langevin_gap → 0 on unseen rules, the energy landscape only has
useful structure for trained rules. This would motivate contrastive
training or denoising score matching to build a more general landscape.

**3e. Noise ablation**
Compare on Stage 2 task:
1. Cyclical annealing (cSGLD)
2. Pure gradient descent (σ=0)
3. Fixed high noise
4. Linear annealing (the naive schedule)

This validates the noise scheduling analysis from ARCHITECTURE_EXPLORATION.
Expected: cyclical > linear > fixed > pure GD on harder variants.

---

## Data Generation Summary

All data is synthetic — no external datasets needed. Self-supervised.

| Stage | Task | Data Gen | Labels Needed? | Samples |
|-------|------|----------|---------------|---------|
| 1a | Arithmetic sequences | `generate_arithmetic()` | No (JEPA) | 5K train, 1K test |
| 1b | Multi-rule sequences | `generate_multi_rule()` | No (JEPA) | 5K train, 1K test |
| 1c | Interleaved sequences | `generate_interleaved()` | No (JEPA) | 5K train, 1K test |
| 2 | Pattern induction | `generate_pattern_induction()` | No (JEPA + rule self-consistency) | 10K train, 2K test |
| 3a-e | Scaling variants | Same as Stage 2, varied params | No | 10K train, 2K test |

Total training data: ~25K samples at 64 tokens each. At 16-bit precision,
this is ~3.2 MB. Negligible memory.

---

## Metrics and Diagnostics

### Standard Metrics (per epoch)
- Train/test loss (L_pred, L_vicreg, L_total)
- Train/test accuracy (token-level on masked positions)
- VICReg components (L_variance, L_covariance)

### Langevin-Specific Metrics (per batch, Stage 2+)
- Energy trajectory: E(z_0), E(z_1), ..., E(z_T)
- Steps to convergence (if using adaptive stopping)
- z movement: ||z_t - z_{t-1}|| per step
- Gradient magnitude: ||∇_z E|| per step
- Signal-to-noise ratio: ||η·∇E|| / ||σ·noise|| per step
- Noise schedule values: σ_t per step

### Ablation Metrics (Stage 2+)
- Langevin gap: accuracy(with Langevin) - accuracy(without)
- Accuracy vs T curve (steps sweep)
- Per-rule accuracy breakdown (which rules are easy/hard?)

### Diagnostic Plots
1. **Energy landscape visualization:** For a few test examples, sweep z
   along 2 principal components and plot E(z) as a heatmap. Are there
   clear basins? Does the global minimum correspond to the correct answer?

2. **z trajectory:** Plot z's path during Langevin in the same 2D space.
   Does it converge to the correct basin?

3. **Accuracy vs Langevin steps:** The core diagnostic plot. Should show
   diminishing returns (steep improvement early, plateau later).

4. **Per-rule accuracy:** Bar chart showing which rules are easy vs hard.
   Reveals what the model has learned and what it struggles with.

5. **Representation space:** t-SNE or PCA of encoder outputs, colored by
   rule type. Do different rules cluster differently?

---

## Training Schedule

### Phase 1 (2-3 days on RTX 3050 Ti)

**Day 1:** Stage 1a (arithmetic sequences)
- Train JEPA backbone for 30 epochs
- Verify L2 error decreases, VICReg healthy
- If fails: debug encoder/predictor architecture

**Day 1-2:** Stage 1b-c (multi-rule, interleaved)
- Train on progressively harder data
- Verify representations improve (harder tasks → higher error, but still learning)

**Day 2-3:** Stage 2 (pattern induction)
- Add z, train JEPA with z-conditioned predictor
- Implement Langevin loop
- Run ablation: no-z vs random-z vs Langevin vs oracle
- Compute Langevin gap

### Phase 2 (3-5 days)

**Days 4-5:** Stage 2 refinements based on results
- If Langevin gap ≈ 0: diagnose and fix
- If Langevin gap > 0: proceed to Stage 3

**Days 5-7:** Stage 3 scaling experiments
- Rule scaling, example scaling, composition, generalization
- Noise ablation

### Decision Point After Phase 2

Based on results:
- **Langevin gap > 10%:** System works. Proceed to harder tasks.
- **Langevin gap 2-10%:** Works marginally. Investigate energy landscape.
  Try contrastive training or FiLM conditioning.
- **Langevin gap < 2%:** Doesn't work. Diagnose:
  - z ignored → conditioning mechanism too weak
  - Energy flat → JEPA not shaping landscape → try denoising score matching
  - Energy shaped but wrong → misalignment between energy and task loss

---

## Memory Budget for Training

All stages fit comfortably in 4GB VRAM:

| Component | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| Encoder (4-layer, d=128) | 1.2 MB | 1.2 MB | 1.2 MB |
| Target Encoder (EMA) | 1.2 MB | 1.2 MB | 1.2 MB |
| Predictor (2-layer, d=64) | 0.2 MB | 0.2 MB | 0.2 MB |
| Decoder | 0.04 MB | 0.04 MB | 0.04 MB |
| z latent | - | <1 KB | <1 KB |
| Activations (batch=32) | ~50 MB | ~50 MB | ~50 MB |
| Langevin (T=3) | - | ~150 MB | ~150 MB |
| Optimizer states | ~5 MB | ~5 MB | ~5 MB |
| **Total** | **~58 MB** | **~208 MB** | **~208 MB** |
| **Headroom** | **~3.9 GB** | **~3.8 GB** | **~3.8 GB** |

Massive headroom. We can scale up if needed.

---

## What the Pattern Induction Task Uniquely Reveals

This task is specifically chosen because it separates THREE capabilities:

1. **Pattern recognition** (encoder): Can the encoder compress
   [2→4, 3→6, 5→10] into a representation that captures "doubling"?
   Tested by Stage 1 (JEPA prediction error).

2. **Hypothesis formation** (z): Can z encode "the rule is doubling"
   as a specific direction in latent space?
   Tested by oracle-z ablation (if oracle z works, the architecture
   can represent rules in z).

3. **Hypothesis search** (Langevin): Can Langevin dynamics FIND the
   right z from random initialization?
   Tested by the Langevin gap (gap between random z and Langevin z).

If capability 1 fails, it's an encoder problem (fix JEPA training).
If capability 2 fails, it's a conditioning problem (fix z → predictor).
If capability 3 fails, it's an energy landscape problem (fix training
of the landscape, noise schedule, or step size).

This diagnostic decomposition is why pattern induction is the right
task — it isolates each component's contribution so we know exactly
what to fix when something fails.
