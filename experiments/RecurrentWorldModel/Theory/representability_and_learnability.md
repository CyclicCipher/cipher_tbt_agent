# Representability & Learnability — STUB

> **Status: stub.** Scaffolding for the theoretical thread, not an answer. Seeded by
> the empirical finding (item-52 SNR run, 2026-06-09) that an optimizer-level
> generalization fix could *not* close our OOD gap — because the gap is
> representational, not memorization. See `Docs/LESSONS.md`.

## The question

**Given an architecture A and hyperparameters θ, can a model *theoretically* learn / compress the general algorithm for a domain D?**

This is the question our experiments keep bottoming out on. It is *not* one question — it stacks two, and conflating them is where confusion lives.

### (i) Representability — is the target program even *in the function class*?
Does there exist a setting of A's weights that computes the domain's generative program (not just fits the training sample)? This is independent of training — it's about the *expressive capacity* of the architecture.

- For us concretely: can a fixed-depth PoPE transformer of width d, depth L *represent* the "compose n affine maps, for n beyond training length" algorithm at all? Length generalization is the sharp case: a fixed-depth feedforward net has a fixed compute budget per token; an algorithm whose required steps grow with input length may simply **not be representable** at fixed depth (this is the bitter-lesson / recurrent-depth argument from the other side).
- Relevant theory to pull on later: circuit-complexity characterizations of transformers (what's in TC⁰ / log-depth / threshold circuits), "what algorithms can transformers express," uniform vs non-uniform expressibility, the role of chain-of-thought / scratchpad as a way to *buy* representable depth.

### (ii) Learnability — will the optimizer *find* it?
Given the program *is* representable, will SGD-from-init actually converge to it (rather than a memorizing solution)? This is the optimization/landscape question.

- For us: the grokking phenomenon, memorization→generalization transitions, implicit bias of SGD/Adam, and the item-52 signal/noise story all live *here*. item-52 is entirely a learnability intervention; it assumes representability is solved.
- Our OOD plateau is a failure of **(i)**, which is why a **(ii)** tool (the SNR gate) couldn't fix it. Diagnosing *which* layer a failure lives in is the first job of this theory.

## The goal — a computable decision procedure (north star)

Not just understanding: the aim is a **theoretically-grounded, analytical answer a
simple program can compute.** Given (a) an architecture A + hyperparameters and
(b) a *sufficiently-determined* domain / general function D, return:

- `Representable(A, D) -> bool / bound` — can A express D's general algorithm? (capacity; **this doc**)
- `Learnable(A, D, optimizer) -> bool / bound` — can the training algorithm *reach* it? (optimization; **separate problem**, §ii)

These are two distinct decision procedures and must not be conflated. *"Sufficiently
determined"* = D's generative program / function class is specified enough to
analyze (e.g. "compose n affine maps mod P"). Success = a calculator, not a vibe.

## The compression framing (the through-line)

Restated in MDL/Solomonoff terms: generalization = compressing the **source** (the generative program), memorization = compressing the **sample** (the training set). Train and test "look like different distributions" only at the *surface* level; at the *program* level they are the same distribution.

- **Representability** ⇔ the source program is in the architecture's hypothesis class (has *finite* description length under A's "code").
- **Learnability** ⇔ the source program is the *MDL-shortest* solution the optimizer's implicit bias is drawn toward, and the path there is navigable.
- The dream "learn anything fast" is bounded by No-Free-Lunch: achievable only on the subset of domains whose shared structure A's prior matches (the bet: that shared structure is compositional/relational).

## Starting clues & first probes (agenda set 2026-06-09)

Two complementary lenses toward `Representable(A, D)`:

**Circuit completeness** — bounds what's expressible (the necessary condition).
Fixed-depth log-precision transformers ⊆ uniform **TC⁰** (Merrill & Sabharwal); a
fixed-depth net can't express depth that grows with input length. *But* our task is
benign: composing affine maps is **associative** → computable by a **parallel prefix
scan in O(log n) depth** → in TC⁰ → a fixed-depth transformer *can* represent it in
principle. So representability likely isn't the blocker for the algorithm; the
question is whether the *learned* solution is the scan (length-generalizes) or a
depth-n unrolled shortcut (capped at trained length). Tools: **RASP-L** (Zhou et al.,
"What Algorithms Can Transformers Learn") — conjecture: length-generalizes iff a short
RASP-L program exists; **CoT/scratchpad buys depth past TC⁰** (Merrill & Sabharwal;
Feng et al.) — which is the recurrent-depth thesis, provable here.

**Categorical deep learning** — whether A's structure matches D's, making the right
solution natural (the inductive-bias side; Gavranović & Veličković, "An Algebraic
Theory of Architectures"). Representable ⇔ a **composition-preserving functor**
(monoid homomorphism) from the affine-map monoid into A's representable functions.
Measurable as **composition fidelity** `M(a∘b) = M(a)·M(b)`; if it holds, length
generalization is free. Head start: the project already has the CTKG categorical
apparatus (Yoneda / functors) to point at this; parametric lenses (item 21) are the
optimization-side companion.

**First experiments (all on the existing PoPE transformer + ModularChain, no new arch):**
1. **Composition-fidelity probe** — does the learned representation compose as a monoid homomorphism? (categorical representability test)
2. **Scratchpad length-generalization** — emit per-step intermediate values; does OOD lift? (circuit/CoT depth test)
3. **Write the affine prefix-scan in RASP-L** — does the short-program prediction hold?

Decision tree → the **X1 diagnosis**: scratchpad fixes OOD ⇒ representability ceiling,
CoT lifts it (recurrent depth justified); composition fidelity holds but OOD fails ⇒
a positional/scaffolding *learnability* issue; neither ⇒ wrong bias, need a different
relational/positional scheme.

## Open sub-questions (to develop)

- **R1.** Can we *certify* (or refute) that a given architecture can represent a given algorithmic family — ideally constructively (exhibit the weights) or via a complexity-class argument?
- **R2.** What is the cheapest architectural change that moves a non-representable target into the function class (recurrent depth? scratchpad/CoT? a different positional code)? — connects directly to why we explored the settling model.
- **L1.** Given representability, what determines whether SGD finds the generalizing vs memorizing solution, and can it be measured *during* training (validation-free)? — item-52 is one attempt; we found its signal noisy.
- **L2.** Is there an architecture-aware notion of "epiplexity ceiling" (max extractable structure) vs "epiplexity rate" (speed) — and can we predict both from A, θ before training?
- **X1.** Diagnosis: given a failure (e.g. our OOD plateau), a *procedure* to attribute it to (i) vs (ii). The single most useful near-term deliverable.

## Connection to the experiments

Every empirical result is a probe of one of these. PoPE fixing OOD = a representability fix (the right positional code made length structure representable). The settling-model plateau = representability ceiling of weight-tied iteration. The SNR gate's failure = a learnability tool on a representability problem. The job of this document, as it grows, is to turn that pattern-matching into something predictive.
