# symbolic_ai_v2 — Design Goals

This is a clean-slate redesign of the symbolic AI, informed by everything learned
in symbolic_ai_v1. The v1 codebase grew to 32K lines, became untestable, and had
performance problems rooted in its architecture. v2 starts from goals, not from
code.

---

## Non-negotiable requirements

### 1. Active inference from day one
The agent is not a passive pattern-matcher. It holds a generative model of its
environment and takes actions (including the action of forming new hypotheses) to
minimise expected free energy. Perception, learning, and action are unified under
one objective. There is no separate "learning phase" followed by an "inference
phase" — both happen on every token.

### 2. Long context and short-term memory from day one
The system must demonstrably recall and reason over information arbitrarily far
back in context. Short-term memory = the uncompressed recent buffer. Long-term
memory = the compressed grammar/knowledge graph (information that has been
abstracted and stored as reusable structure). Both are tested explicitly:
- Needle-in-haystack retrieval at controlled distances (100, 1K, 10K, 100K tokens)
- Multi-hop reasoning over information spread across long context

### 3. Full use of the CTKG category-theoretic toolkit from day one
Structure learning, generalisation, and compression use the tools already built in
`experiments/ctkg/`:
- Functors for cross-domain type alignment
- Adjunctions for learning inverse operations
- Sheaf consistency for multi-modal composition
- d-separation and do-calculus for causal reasoning
- MasteryState for progressive capability tracking
The CTKG is not a downstream consumer of the agent's output; it IS the agent's
internal representation.

### 4. No superlinear algorithms
Every core operation — token processing, grammar update, prediction, memory
retrieval — must be O(n) or O(n log n) in the number of tokens processed, and
O(k) or O(k log k) in the size of the current model (where k grows with
vocabulary/grammar size). Specifically forbidden:
- O(n²) pairwise JSD or similarity computation
- O(V³) dense matrix multiplication
- O(K⁴) soft-cache warm-up
- Any dense V×V or K×K matrix that grows with vocabulary size
Where matrix operations are needed, use sparse representations throughout.

### 5. Zero hardcoded thresholds or hyperparameters
Every decision boundary must be derived from the data using an information-
theoretic or Bayesian criterion. The only acceptable "threshold" is one that
is mathematically entailed by MDL or posterior probability (e.g. "merge a pair
iff it occurs ≥ 2 times" follows from MDL and has no free parameter). No:
- Fixed surprisal thresholds
- Vocabulary size caps
- Similarity cutoffs
- Arbitrary temperature or weight constants

### 6. Drop-in transformer replacement with lower perplexity
The system must produce a well-defined probability distribution P(next token |
context) at every step. It must be benchmarkable against a transformer on the
same corpus using the same perplexity metric. The goal is to beat transformer
perplexity on the Latin test corpus by using structural compression rather than
parameter interpolation. Testable from day one on the existing Latin datasets.

### 7. Any input topology from day one
The core algorithm operates on a **graph** of co-occurrence relationships, not
specifically on a 1D sequence. Sequences, images, video, and graphs are all
instances of the same abstraction:
- 1D sequence → path graph (edges connect position i to i+1)
- 2D image → grid graph (edges connect pixel to its 4 or 8 neighbours)
- Video → temporal stack of grid graphs
- Relational data → arbitrary graph
Adding a new modality requires only defining a graph topology over its tokens.
No new core logic is needed.

---

## What v1 taught us (mistakes not to repeat)

- **Don't build a pipeline.** A pipeline (E0→E1→...→R6→PCH) means every stage
  depends on every previous stage. One miscalibrated parameter cascades everywhere.
  v2 has one algorithm with one data structure.

- **Don't separate learning from inference.** v1 had a training phase and an
  inference phase. This forced batch processing, which caused paragraph-swallowing
  (unbounded segment growth) and made online use impossible.

- **Don't use K-means.** Cluster identity should emerge from the grammar, not be
  imposed by an algorithm that requires choosing K. In v2, "categories" are grammar
  rules — they form and dissolve based on evidence.

- **Don't use dense similarity matrices.** O(V²) pairwise comparisons were the
  root cause of every performance cliff in v1 (V≤300 cap, max_atoms=200, etc.).

- **Test in seconds, not minutes.** If a correctness check requires running the
  full corpus, the architecture is wrong. Every invariant must be testable on a
  toy example in < 1 second.

- **One file = one concept.** v1's relational_pipeline.py grew to 6,365 lines
  because there was no principled separation of concerns. v2 files stay small by
  design.

---

## Folder structure (target)

```
symbolic_ai_v2/
├── GOALS.md              ← this file
├── core/
│   ├── grammar.py        ← online grammar learner (the entire algorithm)
│   ├── graph.py          ← input topology abstraction (sequence/image/graph)
│   ├── memory.py         ← short-term buffer + long-term grammar as memory
│   └── predict.py        ← P(next | context) from current grammar state
├── ctkg_bridge.py        ← maps grammar rules → CTKG types/morphisms
├── active_inference.py   ← free energy minimisation over grammar beliefs
└── tests/
    ├── latin_test.py     ← perplexity benchmark on EarlyModernLatin corpus
    ├── long_context_test.py  ← needle-in-haystack recall at controlled distances
    └── topology_test.py  ← verify same algorithm handles 1D/2D/graph input
```

Total target: < 1500 lines of core logic. Each file independently testable.

---

## Key open questions (to resolve before writing code)

1. **Grammar representation**: Sequitur (digram uniqueness) vs. RePair
   (global optimum) vs. online Bayesian grammar induction. Sequitur is O(n) and
   online; RePair is batch and O(n log n) but finds the globally optimal grammar.
   The choice affects long-context performance.

2. **Prediction from a grammar**: Given a partial parse of the current context,
   how do we assign P(next token)? Options: (a) empirical rule-continuation
   frequencies, (b) Kneser-Ney smoothing over grammar rules, (c) Bayesian
   predictive distribution over grammars.

3. **Active inference objective**: Free energy = -log P(observations | model) +
   KL(posterior || prior). In the grammar setting, the model IS the grammar and
   the posterior IS the parse. How do we define the prior over grammars? MDL
   gives a natural answer: the prior favours grammars that compress the data.

4. **Graph input normalisation**: For image patches, what constitutes a "token"?
   The graph topology handles adjacency, but we need a token vocabulary. Opponent-
   colour tokens (from v1 VisionLearner) are a good candidate.

5. **CTKG bridge granularity**: Should every grammar rule become a CTKG concept,
   or only rules that appear at a frequency threshold? (Answer: no threshold —
   every rule is a concept, and rarely-used rules are simply low-probability
   morphisms.)
