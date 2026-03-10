# AGI Experiment Plan — Minimal ODE Benchmark

## The Benchmark

A system qualifies when it can:

1. Read a word problem (text, optionally with a diagram)
2. Produce a logically valid reasoning chain
3. Return the correct answer in natural language
4. Having learned everything it knows — language, notation, mathematics — from unsupervised
   exposure to text, or at most the equivalent of human education (textbooks, worked examples)

The motivating example: **differential equations practice problems** of the kind solved in
a university course.  The problem is presented as a student would receive it; the system
must solve it as a student would.

### Why this qualifies as AGI for this system

Modern LLMs pass this trivially because they memorise billions of solution patterns across
trillions of tokens.  For the CTKG system the same benchmark is non-trivial because:

- **Zero hard-coded domain knowledge.**  The ODE solver is not written by a programmer;
  it emerges from the composition hierarchy after reading worked examples.
- **Compositional generalization.**  The system must combine rules learned separately to
  solve problem instances it has never seen.  Example: learning `∫x dx = x²/2` and
  `∫cos(x) dx = sin(x)` independently, then correctly solving `dy/dx = x + cos(x)`.
- **Human-like data efficiency.**  A student solves novel ODEs after reading one textbook
  chapter (~100 worked examples).  The system must match this, not require millions of
  examples.
- **No drawbacks of current AI.**  Exact counts, interpretable composition rules, no
  hallucination of arithmetic, bounded memory, Rust-rewritable for production.

---

## Phases

### Phase 0 — Text-only, structured notation (current scope)

**Goal:** Prove the principle on a deliberately easy version before adding complexity.

**Domain:** First-order separable ODEs with polynomial and trigonometric right-hand sides.

**Problem format** (rigid, machine-generated):
```
Problem: Find y(x) such that dy/dx = 3x^2 + cos(x), y(0) = 2.
Solution: Integrate both sides. Integral of 3x^2 is x^3. Integral of cos(x) is sin(x).
So y = x^3 + sin(x) + C. Applying y(0) = 2: 0 + 0 + C = 2, so C = 2.
Answer: y = x^3 + sin(x) + 2
```

Everything is in ASCII mathematical notation.  No natural language paraphrasing.
No diagrams.  The format is identical across all training and test examples.

**Training corpus:** 120 programmatically generated (problem, solution) documents.
**Test set:** 30 held-out problems with distinct coefficients and function combinations.
**Success criterion:** Exact symbolic match on the Answer line for ≥ 20/30 test problems.

**Learning protocol:**  Each document is one sequence fed to `MorphismGraph.observe_sequence()`.
The system sees problem AND solution during training.  At test time it receives only the
problem prefix up to and including `"Solution: "` and must complete the rest via greedy
decode using `predict_dist()` / `perplexity_multilevel()`.  No labels, no supervision
signal — just next-character prediction, exactly as a language model reads a textbook.

---

### Phase 1 — Natural language problem statements

**Goal:** Replace rigid notation with paraphrased English problem statements.

**New challenge:** The system must learn that:
- "the rate of change of y with respect to x" → `dy/dx`
- "a particle moves with velocity proportional to time" → `dv/dt = kt`
- "initially" / "at time zero" → initial condition

**Training corpus:** 200 examples per ODE family, each paraphrased 3–5 ways.
**Success criterion:** ≥ 15/20 novel (problem-text, ODE-type) combinations solved correctly.

**Key difference from Phase 0:** The morphism graph must now learn cross-domain
correspondences — English phrase → mathematical structure.  These are functors in the CTKG
sense: a structure-preserving map from the "natural language description" domain to the
"ODE notation" domain.

---

### Phase 2 — Diagram reading

**Goal:** Add visual input alongside the word problem.

**Domain:** Direction fields and phase portraits for 2D autonomous systems.

**Input:** A screenshot of a direction field + a text question asking for the long-term
behaviour of solutions.

**New capability required:** The `grid_2d` topology reading from `vision_pipeline.py` must
be wired to interpret the arrow directions in each cell as edge types in the MorphismGraph.
This is architecturally natural: a direction field IS a labeled directed graph, the exact
structure MorphismGraph is designed for.

**Success criterion:** Correct qualitative answer (stable node / limit cycle / saddle /
unstable) for ≥ 8/10 novel direction fields.

---

### Phase 3 — Multi-domain physics

**Goal:** Demonstrate that the ODE-solving procedure transfers to physics problems without
re-learning the solving procedure.

**Example:** Newton's second law (F = ma → d²x/dt² = F/m) reduces to the same ODE family.
A CTKG functor maps the physics domain onto the ODE domain; the learned solution procedure
is reused without modification.

**Success criterion:** ≥ 6/10 novel physics word problems solved correctly, where the
functor is learned from only 20 physics/ODE paired examples.

---

## Architecture Mapping

| Benchmark requirement | Component | Status |
|---|---|---|
| Learn from unsupervised text | `MorphismGraph.observe_sequence()` | Done |
| Compositional rule discovery | `_create_composition()`, `rules_inv` | Done |
| Richer-than-bigram context | `_pending_comp_ctx`, `perplexity_multilevel()` | Done |
| Next-token generation | `predict_dist()` greedy loop | **Needed: `core/generate.py`** |
| Symbolic answer verification | `interpreter.py` `sym_eval`, `sym_subst` | Done |
| ODE problem corpus | Programmatic generator | **Needed: `data/ode_problems/gen.py`** |
| Natural language paraphrasing | Phase 1 corpus generator | Phase 1 |
| Diagram reading | `grid_2d` + `vision_pipeline.py` | Phase 2 |
| Cross-domain transfer | CTKG functor + `build_functor()` | Phase 3 |

---

## Files to Create (Phase 0)

```
experiments/symbolic_ai_v2/
  data/
    ode_problems/
      gen.py           — Programmatic corpus generator (120 train + 30 test)
      train/           — Generated .txt files, one per problem
      test/            — Held-out .txt files
  core/
    generate.py        — Greedy / beam decoder using predict_dist + multilevel context
  experiments/
    ode_experiment.py  — End-to-end: generate corpus, train MorphismGraph, evaluate
  tests/
    ode_test.py        — Unit tests for corpus generator and decoder
  ode.ctkg             — ODE domain for CTKG (adjunction: differentiate / integrate)
```

---

## Corpus Generator Specification (`gen.py`)

Each generated document follows the fixed format above.  The generator draws from:

**ODE families** (Phase 0):
- Polynomial: `dy/dx = a₀ + a₁x + a₂x² + ... + aₙxⁿ`  (n ∈ {0,1,2,3})
- Trigonometric: `dy/dx = a·sin(x)`, `a·cos(x)`
- Mixed: `dy/dx = aₙxⁿ + b·sin(x)`, `aₙxⁿ + b·cos(x)`
- Exponential: `dy/dx = a·e^(bx)`

**Parameters:** coefficients drawn from {-5,...,-1, 1,...,5} (avoid 0); n from {1,2,3};
initial condition `y(0) = c` with c from {-3,...,3}.

**Solution steps** are deterministically computed from the formula (not by a neural network):
- Integration is performed symbolically by `interpreter.py`'s `sym_eval` and `sym_diff`
- The solution text is templated around the computed result
- This guarantees correctness: the training corpus has no errors

**Train/test split:** 120 train documents selected to cover all ODE families with all
parameter combinations; 30 test documents use unseen coefficient combinations.

---

## Decoder Specification (`core/generate.py`)

```
decode(mg, prompt_str, topology, max_tokens, stop_token) -> str
```

1. Feed `prompt_str` character by character through `observe_sequence()` in read-only mode
   (updating `_pending_comp_ctx` but not modifying edges or pairs — copy-on-write).
2. At each step maintain `(prev_id, comp_ctx)` as in `perplexity_multilevel()`.
3. Sample next token: `argmax predict_dist(comp_ctx or prev_id, etype)`.
4. Append to output; feed back as next input; stop on `stop_token` or `max_tokens`.

The "read-only" constraint is important: during generation the MorphismGraph must not
learn from its own output.  The edge/pair tables are frozen after training.

**Beam search variant** (optional, Phase 0+): maintain K partial hypotheses; prune by
cumulative log-probability.  K=1 is greedy, which is sufficient for Phase 0.

---

## Evaluation Specification (`experiments/ode_experiment.py`)

For each test problem:

1. Extract the `Answer:` line from the generated text.
2. Parse it with a simple regex into a `sym_expr` object via `interpreter.py`.
3. For each `x` in {0, 1, -1, 2, π/4}: evaluate generated expression and ground truth.
4. **Exact match:** generated expression is algebraically equivalent (all evaluations agree
   within 1e-9) — counts as PASS.
5. **Partial credit** (informational only): correct up to a missing constant C — counts
   separately but not toward the primary metric.

**Primary metric:** PASS count out of 30 test problems.
**Target for Phase 0:** ≥ 20/30 (67%).
**Baseline:** unigram/bigram character model with no compositions — expected ≈ 0–2/30.

---

## Research Questions This Experiment Answers

1. **Does the composition hierarchy help generation?**
   Compare greedy decode using `comp_ctx` vs. using `prev_id` only.  Expected: comp_ctx
   dramatically reduces generation errors on the structured solution template.

2. **How many training examples are needed?**
   Sweep train set size: 10, 30, 60, 90, 120 examples.  Plot accuracy vs. n_examples.
   Human baseline: a student needs ~20–50 worked examples to solve novel ODEs reliably.

3. **Does transfer across ODE families work?**
   Train on polynomial-only, test on mixed polynomial+trig.  Expected: yes, because the
   integration rules are learned as reusable compositions, not as monolithic patterns.

4. **Does the CTKG functor add value?**
   Phase 0: no functor, just raw text learning.
   Phase 3: CTKG functor maps physics → ODE.  Measure accuracy delta.

---

## Connection to BLUEPRINT.md

The `predict()` function in BLUEPRINT.md §"predict()" describes exactly what the decoder
does: given the current composition (or atom), return the distribution over next symbols.
The decoder is the generative application of the same mechanism used for perplexity
evaluation — prediction error minimisation = learning; prediction = generation.

The `generate()` method in BLUEPRINT.md §"generate() — top-down expansion" describes the
reverse direction: given a high-level composition, expand to atoms.  Both directions will
be used in the Phase 3 CTKG-guided generation: the CTKG selects the high-level procedure
(ODE solution template), `generate()` expands it to characters.

---

## Open Questions / Risks

| Risk | Mitigation |
|---|---|
| Character-level MorphismGraph not rich enough for ODE generation | Use word-level tokenisation (split on spaces) for Phase 0 to reduce vocabulary explosion |
| Greedy decoder gets stuck in a loop | Add position penalty (reduce probability of repeating the last K tokens) |
| 120 training examples insufficient | Supplement with CLTK or Gutenberg corpus for language priming before ODE training |
| Phase 1 NL parsing requires significantly more data | Use a fixed set of 5–10 sentence templates per ODE family; expand gradually |
| Diagram reading (Phase 2) requires vision pipeline that's not yet wired | Phase 2 is a separate milestone; do not block Phase 0/1 on it |
