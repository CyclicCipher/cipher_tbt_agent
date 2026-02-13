# How Architectures Are Discovered

## Purpose

This document examines the meta-level question: how do researchers discover
architectures like CNNs, RNNs, Transformers, and Mamba? Understanding the
thinking pattern behind architectural discovery may help us design the
reasoning architecture for this project.

The central finding: every major architecture follows a four-step process,
and the architecture itself is typically the UNIQUE MINIMAL solution to a
precisely identified constraint. The constraint comes first. The architecture
follows from it.

---

## The Four-Step Pattern

### Step 1: Identify what's broken (the bottleneck)

Not "this is bad" but "THIS SPECIFIC THING prevents progress." The precision
of the bottleneck identification determines the quality of the solution.

### Step 2: Find where the problem is already solved (the analogy)

Look outside ML — in neuroscience, physics, control theory, pure math,
information retrieval. The analogy provides structural intuition.

### Step 3: Extract the mathematical principle (the constraint)

Translate the analogy into a mathematical property the architecture must
satisfy. This is the hardest step and the most important. A vague analogy
("the brain does X") produces a vague architecture. A precise constraint
("the function must be translation-equivariant") produces a precise
architecture.

### Step 4: Build the simplest thing that satisfies the constraint (minimal realization)

Don't add anything the constraint doesn't demand. The architecture should be
the inevitable consequence of the constraint, not a creative invention.

---

## Case Studies

### CNNs (LeCun, 1989)

**Step 1 — Bottleneck:**
Fully connected networks applied to images have O(n²) parameters where n
is the number of pixels. A 256×256 image with 1000 hidden units requires
65 million weights in the first layer alone. This doesn't scale, and worse,
the network has no notion of spatial structure — it treats pixel (0,0) and
pixel (255,255) with equal connectivity.

**Step 2 — Analogy:**
Hubel and Wiesel (1962) discovered that neurons in cat visual cortex have
LOCAL receptive fields — each neuron responds to a small patch of the visual
field, not the whole image. Furthermore, the same edge-detection pattern
appears at every position (simple cells are approximately translation-
invariant).

**Step 3 — Constraint:**
Translation equivariance: `f(shift(x)) = shift(f(x))`. The same feature
detector should produce the same output regardless of where in the image
the feature appears.

The unique linear operation satisfying translation equivariance is
convolution (this is a theorem, not a design choice).

**Step 4 — Minimal realization:**
Small learned kernels that slide over the input. Weight sharing (same
kernel everywhere) enforces the constraint. Stacking layers with pooling
gives hierarchical features. That's the entire CNN.

**Key insight:** LeCun didn't invent convolution. He identified the
constraint (translation equivariance) and recognized that convolution was
the mathematical answer.

---

### RNNs (Rumelhart, Hinkins, Williams, 1986)

**Step 1 — Bottleneck:**
Feedforward networks require fixed-size input. Sequences (language, time
series, music) have variable length. You can't just zero-pad everything to
max length — that wastes computation and doesn't capture the sequential
nature of the data.

**Step 2 — Analogy:**
Dynamical systems: state evolves through time via a transition function.
The current state depends on the previous state plus new input. This is
how physical systems work, how memory works, how sequential processes work.

**Step 3 — Constraint:**
Time-invariant state transition: `h_t = f(h_{t-1}, x_t)` where f is the
same function at every timestep. This handles arbitrary-length sequences
with a fixed number of parameters.

**Step 4 — Minimal realization:**
`h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)`. Shared weights across time.
That's it.

**Key insight:** The constraint (time-invariant transition) uniquely
determines the architecture (shared weights + hidden state).

---

### LSTMs (Hochreiter & Schmidhuber, 1997)

**Step 1 — Bottleneck:**
RNNs can't learn long-range dependencies. The gradient of the loss with
respect to early hidden states vanishes exponentially:
`∂h_T/∂h_1 = Π_{t=1}^{T-1} ∂h_{t+1}/∂h_t`, and each factor has
eigenvalues < 1 for typical initializations.

This was identified precisely in Hochreiter's 1991 diploma thesis:
the "vanishing gradient problem."

**Step 2 — Analogy:**
The "Constant Error Carousel": what if there existed a pathway through
which the gradient flows completely unchanged? Not attenuated, not
amplified — just preserved.

This is analogous to how a frictionless flywheel stores energy indefinitely.
Or how a wire carries current without resistance. The gradient needs a
superconducting pathway through time.

**Step 3 — Constraint:**
`∂c_t/∂c_{t-1} = 1` (at least optionally). The cell state's self-connection
must be able to achieve unit gradient. This means the update to c must be
ADDITIVE, not multiplicative, because:
- Multiplicative: `c_t = f · c_{t-1}` → `∂c_t/∂c_{t-1} = f` → vanishes if f < 1
- Additive: `c_t = c_{t-1} + g` → `∂c_t/∂c_{t-1} = 1` → gradient preserved

But you also need selectivity — not everything should be remembered forever.

**Step 4 — Minimal realization:**
- Cell state with additive updates: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`
- Forget gate f_t: learned sigmoid controlling what to erase
- Input gate i_t: learned sigmoid controlling what to write
- Output gate o_t: learned sigmoid controlling what to read
- When f_t = 1, i_t = 0: gradient flows unchanged (the carousel)
- When f_t = 0: old state is erased (selective forgetting)

**Key insight:** The architecture is uniquely determined by "additive state
update + learned gates." Every component exists because the constraint
demands it. The forget gate exists because you need selective memory.
The input gate exists because you need selective writing. Nothing is extra.

---

### Transformers (Vaswani et al., 2017)

**Step 1 — Bottleneck:**
Two bottlenecks, precisely identified:

(a) RNNs are inherently sequential — h_t depends on h_{t-1}, so you can't
parallelize across timesteps. This wastes modern GPU hardware.

(b) Even with attention (Bahdanau, 2014), information from token i to token
j must flow through the recurrence: i → i+1 → ... → j. Long-range
dependencies are indirect.

Note: the bottleneck was NOT "attention doesn't work." Attention worked
great. The bottleneck was that attention was BOLTED ONTO a sequential
backbone. Vaswani's key question: what if we remove the sequential backbone
and keep ONLY attention?

**Step 2 — Analogy:**
Information retrieval: query-key-value lookup. Given a query, find the
most relevant keys and return their associated values. This is how databases
work, how associative memory works, how you search for relevant information.

Also: sets. If you remove recurrence, the input is a SET of tokens (position
must be added back via positional encoding). Operations on sets should be
permutation-equivariant, and the canonical permutation-equivariant operation
on sets is... attention (weighted sum over all elements).

**Step 3 — Constraint:**
(a) Full parallelizability: the computation for position t should not depend
    on the computation for position t-1
(b) Direct any-to-any interaction: token i should influence token j in O(1)
    layers, not O(j-i) layers
(c) Permutation equivariance (when ignoring position): the same operation
    should apply regardless of which set element you're updating

**Step 4 — Minimal realization:**
```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

Q, K, V are linear projections of the input. The softmax produces a
distribution over positions. The output is a weighted sum of values.

This satisfies all three constraints:
- Parallelizable: all positions computed simultaneously
- Any-to-any: every position directly attends to every other position
- Permutation-equivariant: reordering inputs reorders outputs identically

Add feedforward layers (per-position processing) + residual connections
(gradient flow) + positional encoding (break permutation symmetry for
sequences). That's the complete transformer.

**Key insight:** The transformer is not a creative invention. It's the
inevitable consequence of "remove sequentiality, keep attention." Vaswani
et al.'s contribution was the precise bottleneck identification, not the
architecture — the architecture follows from the constraints. The title
of the paper, "Attention Is All You Need," is literally the constraint.

---

### Mamba (Gu & Dao, 2023)

**Step 1 — Bottleneck:**
Transformers are O(n²) in sequence length due to the attention matrix.
For long sequences (10K+ tokens), this is prohibitive.

Prior SSMs (S4, H3) achieve O(n) but with a critical limitation: the state
transition matrices A, B, C are input-INDEPENDENT. The same dynamics apply
regardless of content. This means S4 can't do content-based routing — it
can't say "this token is important, remember it" or "this token is noise,
ignore it."

This was identified precisely: the failure mode is "SSMs can't solve
selective copying" (where you must remember some tokens and forget others
based on their content).

**Step 2 — Analogy:**
Control theory: state space models with input-dependent control. In real
control systems, the control law often depends on the current state and
input — you don't apply the same control regardless of what's happening.

Biology: neural gating. Biological neurons modulate their responses based
on context. Attention is a form of gating. Can we build a recurrent model
with input-dependent gating that stays O(n)?

**Step 3 — Constraint:**
(a) O(n) computation in sequence length (no attention matrix)
(b) Content-dependent state transitions: the dynamics must be functions
    of the input, not fixed parameters
(c) Hardware-efficient: must be implementable as a parallel scan on GPUs
    (not just theoretically O(n), but practically fast)

(b) and (c) appear contradictory: input-dependent matrices break the
structured form that made S4's convolution trick possible.

**Step 4 — Minimal realization:**
Make Δ, B, C functions of the input:
```
Δ_t = softplus(Linear(x_t))
B_t = Linear(x_t)
C_t = Linear(x_t)
```

Then use a HARDWARE-AWARE parallel scan instead of the convolution trick.
The scan is associative, so it can be parallelized across the sequence.
The key algorithmic insight: even though the matrices are input-dependent,
the scan operation is still associative, so GPU-friendly parallelism works.

**Key insight:** Gu & Dao identified the PRECISE bottleneck (input-
independence, not the SSM formulation itself) and found that the constraint
"input-dependent + O(n) + parallelizable" had exactly one solution: selective
scan. The Mamba architecture is that solution and nothing else.

---

## Meta-Principles Extracted

### 1. Constraint Before Architecture

Every successful architecture is the minimal solution to a precisely
identified constraint. The architecture doesn't come from creativity —
it comes from constraint satisfaction. If you can't state the constraint
mathematically, you're not ready to design the architecture.

Corollary: if you find yourself choosing between multiple architectures,
you haven't identified the constraint precisely enough. A precise constraint
yields a unique (or nearly unique) solution.

### 2. The Analogy Comes From Outside ML

- CNNs ← neuroscience (visual cortex)
- LSTMs ← physics (constant error carousel / frictionless flywheel)
- Transformers ← information retrieval (query-key-value) + set theory
- Mamba ← control theory (state space models) + neuroscience (neural gating)

Every breakthrough imported a structural idea from another field. Pure
ML-internal reasoning ("let's try stacking more layers") rarely produces
fundamental advances. The outside field provides the structural intuition
that gets formalized into the constraint.

### 3. Precision of Bottleneck ↔ Quality of Solution

Vague bottleneck → vague solution:
"RNNs are slow" → "let's try parallelizing RNNs" (many failed attempts)

Precise bottleneck → precise solution:
"RNNs are sequential because h_t depends on h_{t-1}" → "remove the
sequential dependency entirely, keep only attention" → Transformer

The quality of the solution is determined by the precision of the problem
statement, not by the cleverness of the solution.

### 4. Successful Architectures Are Simpler Than Expected

- Transformer: `softmax(QK^T/√d)V`
- Mamba: parameterized linear recurrence with a scan
- CNN: sliding dot product with shared weights
- LSTM: additive cell state + three sigmoid gates

The power comes from the RIGHT inductive bias applied MINIMALLY.
Over-engineered architectures (e.g., Neural Turing Machines, Differentiable
Neural Computers) tend to lose to simpler alternatives that capture the
essential constraint with less mechanism.

### 5. Validate on a Toy Problem First

- CNNs: handwritten digits (MNIST, before it existed formally)
- LSTMs: simple sequences (counting, XOR over time)
- Transformers: machine translation (WMT)
- Mamba: synthetic benchmarks (selective copying, induction heads)
- ePC-Mamba (this project): copy task → 99.03%, task 1b (SGD) → 97.01% epoch 1

The toy problem should be the SIMPLEST task that REQUIRES the constraint
your architecture satisfies. If the toy problem can be solved without your
architecture, it's not diagnostic. If it requires much more than your
architecture provides, it confounds validation.

### 6. Failure Modes Are Diagnostic

When an architecture fails, HOW it fails tells you what constraint is
missing:

- RNN fails on long sequences → missing: gradient preservation → add: LSTM
- S4 fails on selective copying → missing: content-dependence → add: Mamba
- ePC with Newton plateaus for 19 epochs → missing: simple optimizer → fix: use SGD
- JEPA masked prediction fails on 1b → missing: complete coverage → fix: next-step prediction
- ePC with N-1 errors gets 7% → missing: last error node → fix: N errors for N layers

The ePC plateau was originally misdiagnosed as a fundamental ePC property.
It was actually Newton-specific (see EPC_LEARNING_DYNAMICS.md). The real
breakthroughs came from identifying the "complete error coverage" constraint.

---

## Application to the Reasoning Architecture

### What's the Precise Bottleneck?

"LLMs are bad at reasoning" is too vague. More precisely:

**(a) Fixed compute per token.** Autoregressive models spend the same
FLOPs on "2+2=" and on a step in a complex proof. There's no mechanism
to allocate more computation to harder sub-problems.

**(b) Irrevocable commitment.** Each generated token is final. The model
can't go back and revise token 5 after generating token 20 reveals that
token 5 was wrong. Chain-of-thought helps but is limited to surface-level
correction ("Wait, let me reconsider...").

**(c) No self-consistency check.** Each token is generated to be locally
plausible given the prefix, but there's no global optimization ensuring
all tokens are mutually consistent. This is why LLMs hallucinate fluent
nonsense — each piece sounds right, but the whole is contradictory.

### What Are the Constraints?

From the bottlenecks, we can extract mathematical constraints:

**(a) → Variable compute:** The system must be able to perform ADAPTIVE
amounts of computation — more for harder problems, less for easier ones.
This means the architecture can't have a fixed computational graph. It
needs an iterative process that runs until convergence.

**(b) → Continuous revisability:** The reasoning state must be a CONTINUOUS
variable that can be refined by gradients. Discrete token sequences can't
be gradient-refined (no ∂token/∂loss). The state must live in a continuous
space where gradient-based search is possible.

**(c) → Global consistency via joint optimization:** All parts of the
reasoning state must be optimized JOINTLY against a single objective that
measures global consistency. Not greedy left-to-right, but simultaneous
optimization of the entire state.

### What's the Minimal Realization?

The constraints (adaptive iteration + continuous state + joint optimization)
have a known class of solutions: energy-based models with iterative
inference. Specifically:

```
z ∈ R^d                              (continuous state)
E(z) = scalar                         (global consistency measure)
z ← z - η·∇_z E(z) + noise          (joint optimization, iterative)
repeat until convergence              (adaptive compute)
```

This is Langevin dynamics on a learned energy function. It satisfies all
three constraints with minimal mechanism. The architecture in
ARCHITECTURE_EXPLORATION.md is the concrete instantiation of this minimal
realization using the Mamba3 components we've already built.

### What Remains Underspecified?

The constraint analysis above tells us the OUTER structure (energy
minimization loop) but NOT the inner structure (what computes E).

This is analogous to how "attention is all you need" tells you the
attention mechanism but not the feedforward layers, residual connections,
or layer normalization that make it trainable. Those were determined by
engineering (what makes the gradients flow) rather than by constraint
(what structural property is needed).

Similarly, the choice of Mamba3 for the encoder/predictor, VICReg for
anti-collapse, and JEPA for the training objective are engineering choices
within the constraint-determined outer structure. They should be validated
empirically, not derived from first principles.

### Applied: The Complete Error Coverage Principle (2026-02-13)

This project's first successful application of the discovery pattern:

**Step 1 — Bottleneck (precise):**
JEPA with masked prediction achieves 18.6% on Stage 1b (multi-rule
regime change). The model can't detect where one rule ends and another
begins. Separately, ePC-Mamba3 with N-1 errors achieves 7% (random).

**Step 2 — Analogy:**
Mamba is a causal model — position t sees only 0..t-1. This is like
trying to fill in blanks on an exam where you can only read the page
from top to bottom. Masking early positions gives you no context.
Similarly, omitting the last error node in ePC is like having a
teacher who never checks your final answer.

**Step 3 — Constraint:**
**Complete error coverage**: every computational unit (position,
layer) must receive a gradient signal proportional to its prediction
error. No dead zones. Formally: for a causal model with L layers and
T positions, the learning signal must be defined at ALL L*T points,
not a sparse subset.

**Step 4 — Minimal realization:**
- Temporal: replace masked prediction with next-step prediction
  (predict token t+1 from representation at t)
- Layer-wise: N errors for N blocks (not N-1)
- Both: no new parameters, no new modules, just connecting existing
  components differently

**Result:** 18.6% -> 97.05% (JEPA Stage 1b), 7% -> 99.3% (ePC copy
task). The constraint was precise, the realization was minimal, and
the improvement was dramatic.

**The meta-lesson:** The same constraint (complete coverage) was
discovered independently in two different systems (JEPA temporal,
ePC layer-wise). This convergence is exactly what the discovery
pattern predicts — when the bottleneck is identified precisely, the
solution is unique and applies universally.

### The Meta-Pattern Applied to Our Next Steps

Following the discovery pattern:

1. **Build the minimal system** (Phase 1 of ROADMAP: JEPA + Langevin)
2. **Validate on a toy problem** (sorting? simple logic?)
3. **Observe the failure mode** (what specifically fails?)
4. **Identify the precise bottleneck** (not "it doesn't work" but "it
   fails BECAUSE...")
5. **Find an analogy** for that specific failure
6. **Extract the constraint** and add ONLY the structure it demands
7. **Repeat**

This is how you discover architecture iteratively. You don't design the
final system upfront — you let the failures tell you what's missing.

---

## Historical Lessons for What NOT to Do

### Don't Start With the Analogy

Starting with "the brain does predictive coding, let's build that" leads
to architecture-first thinking. Many biologically inspired architectures
(Boltzmann machines, Helmholtz machines, early spiking networks) were
faithful to the analogy but didn't identify what MATHEMATICAL constraint
the biology satisfies.

**Better:** "LLMs can't revise reasoning. What mathematical property
would fix this? Continuous optimization. What's the minimal architecture
for continuous optimization? Langevin dynamics on an energy function."

### Don't Over-Engineer Phase 1

Neural Turing Machines had read/write heads, external memory, addressing
mechanisms, and controllers. They were theoretically beautiful and
practically beaten by simple LSTMs on almost every benchmark.

Differentiable Neural Computers added even more. Still beaten by
Transformers, which are embarrassingly simple by comparison.

**The pattern:** complex architectures designed from first principles
without empirical iteration tend to lose to simple architectures refined
through empirical failure analysis.

### Don't Confuse Multiple Bottlenecks

Each bottleneck needs its own analysis. Trying to solve "fixed compute"
and "no compositionality" and "no external knowledge" simultaneously
leads to a system that does none well.

**Better:** The phased approach in the ROADMAP. Phase 1 solves fixed
compute + revisability (Langevin). Phase 3 solves compositionality
(CTKG). Each phase adds exactly one capability.

---

## Summary

The meta-framework for architectural discovery:

```
Precise bottleneck → Outside analogy → Mathematical constraint → Minimal realization
```

Applied iteratively through build-fail-diagnose-fix cycles.

The most important discipline is RESTRAINT: don't add structure you can't
justify from a precisely identified failure. Every component should exist
because a constraint demands it, not because it seems like a good idea.

The reasoning architecture should follow this same discipline. The core
constraints (variable compute, continuous revisability, global consistency)
yield energy-based Langevin minimization as the minimal realization.
Everything beyond that should be added only when empirical failure on a
diagnostic task reveals a specific missing constraint.
