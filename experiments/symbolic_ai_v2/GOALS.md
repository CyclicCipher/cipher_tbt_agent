# symbolic_ai_v2 — Design Goals

This is a clean-slate redesign of the symbolic AI, informed by everything learned
in symbolic_ai_v1. The v1 codebase grew to 32K lines, became untestable, and had
performance problems rooted in its architecture. v2 starts from goals, not from
code.

---

## The central insight

Grammar induction is not the goal. **Compositional morphism discovery** is the
goal. Grammar is just what that operation looks like when the domain is symbols
in sequence. Logic is what it looks like when the domain is propositions.
Mathematics is what it looks like when the domain is quantities. Causality is
what it looks like when the domain is events.

All of these reduce to the same operation: find f: A → B, find g: B → C, derive
g∘f: A → C.

The CTKG work proved this empirically. Starting from toddler-level arithmetic
(successor, addition as iterated counting), the system composed morphisms upward
through multiplication, exponentiation, polynomials, and into calculus and fluid
dynamics — without being told that any of those higher concepts existed. It
discovered them as compositions of what it already knew.

The v1 mistake was building a grammar layer (PCH, Merge) *separately* from the
reasoning layer (CTKG) and then trying to bridge them. The bridge was always a
seam that leaked. In v2 there is no bridge: every discovered pattern is
immediately a morphism. The architecture does not distinguish between "learning
grammar" and "learning logic" — both are morphism discovery in a category.

This also changes what correctness means. A system that only achieves low
perplexity on Latin text has learned grammar but not reasoning. A system that
only solves arithmetic problems has learned one domain but not transfer. The
v2 system must demonstrably do both — and demonstrate that they share the same
internal mechanism.

---

## Non-negotiable requirements

### 1. Active inference from day one
The agent is not a passive pattern-matcher. It holds a generative model of its
environment (the current CTKG) and acts — including the act of forming new
hypotheses — to minimise expected free energy. Perception, learning, and action
are unified under one objective. There is no separate "learning phase" followed
by an "inference phase"; both happen on every observation.

### 2. Long context and short-term memory from day one
The system must demonstrably recall and reason over information arbitrarily far
back in context. Short-term memory = the uncompressed recent buffer. Long-term
memory = the compressed morphism graph (information abstracted into reusable
structure). Both are tested explicitly:
- Needle-in-haystack retrieval at controlled distances (100, 1K, 10K, 100K tokens)
- Multi-hop reasoning over facts spread across long context

### 3. Full use of the CTKG category-theoretic toolkit from day one
Every learned pattern is a CTKG morphism. The CTKG is not a downstream consumer
of the agent's output — it IS the agent's internal representation, updated on
every observation. This means:
- Functors for cross-domain type alignment and transfer
- Adjunctions for discovering inverse operations (e.g. add/subtract, encode/decode)
- Sheaf consistency for multi-modal composition (text + image + graph)
- d-separation and do-calculus for causal reasoning
- MasteryState for tracking what has been consolidated vs. what is still uncertain
- Information flow / mutual information for identifying what matters

### 4. No superlinear algorithms
Every core operation must be O(n) or O(n log n) in the number of observations
processed, and O(k) or O(k log k) in the current model size k. Specifically
forbidden:
- O(n²) pairwise distance/similarity computation
- O(V³) dense matrix multiplication
- Any dense matrix that grows with vocabulary or grammar size
Where matrices are needed, use sparse representations throughout. If an algorithm
cannot be made subquadratic, it is the wrong algorithm.

### 5. Zero hardcoded thresholds or hyperparameters
Every decision boundary must follow from an information-theoretic or Bayesian
criterion derived from the data. The one acceptable "threshold" is the MDL count
criterion (merge a pair iff count ≥ 2), which has no free parameter because it
is mathematically equivalent to "this pair reduces description length." No:
- Fixed surprisal thresholds
- Vocabulary or cluster size caps
- Similarity cutoffs
- Arbitrary weights or temperature constants

### 6. Drop-in transformer replacement with lower perplexity
The system must produce a well-defined probability distribution P(next token |
context) at every step and be benchmarkable against a transformer on the same
corpus with the same perplexity metric. The structural compression of morphism
discovery should strictly dominate parameter interpolation on structured data.
Testable from day one on the existing Latin datasets.

### 7. Any input topology from day one
The core algorithm operates on an arbitrary graph of observations, not
specifically on a 1D sequence. Topology is an input parameter, not an assumption:
- 1D sequence → path graph (edge: position i → i+1)
- 2D image → grid graph (edge: pixel → its 4/8 neighbours)
- Video → temporal grid stack
- Relational data → arbitrary labelled graph
Adding a new modality means defining a graph topology over its tokens. No new
core logic. The morphism discovery algorithm is topology-agnostic.

### 8. Bounded memory: principled forgetting

**The model's memory must not grow unboundedly with corpus size.**  A symbolic AGI
that has read all of Wikipedia must fit in ≤ 1 GB, not 350 GB.

The `pairs` table is the primary growth source: every novel triple `(Q,e1,P,e2,S)`
creates an entry. Two pruning mechanisms, both zero-threshold and MDL-grounded:

**a. Composition-triggered pair pruning** (automatic, O(degree) per composition):
When Composition C = (P →[e]→ S) is created, every pair entry `(Q,e₁,P,e,S)` is
permanently dead: future occurrences of (P,e,S) are absorbed into C by
`_compress_buf_tail` *before* reaching the pair-check step.  Removed immediately
via reverse-index `_pairs_rdigram: (P,e,S) → set[pair_key]`.  Not an approximation
— provably correct: those pairs cannot trigger boundaries again.

**b. Stability-window singleton pruning** (triggered by `prune(max_age)`):
A pair with count=1 not incremented in `max_age` boundaries has expected
informational value below its storage cost.  Under a geometric recurrence model,
P(recur) ≈ 1/(age+2) → 0.  May cause one false boundary if the triple does recur
(acceptable approximation).

**Target**: `|pairs|` bounded by O(V² · D) after model saturation, not O(n).

---

## Tests that must all pass (not just perplexity)

These are the benchmarks that distinguish a grammar inducer from a reasoning
agent. All must pass from day one; if the architecture cannot support them,
the architecture is wrong.

| Test | What it checks |
|------|---------------|
| Latin perplexity < transformer baseline | Grammar induction |
| Arithmetic: count → add → multiply → polynomial | Compositional reasoning |
| Needle-in-haystack @ 10K tokens | Long-context memory |
| Cross-domain functor (Latin ↔ arithmetic) | Transfer / analogy |
| Causal intervention (do-calculus) on learned graph | Causal reasoning |
| New modality (2D image) with zero new core code | Topology-agnostic design |

---

## What v1 taught us (mistakes not to repeat)

- **Don't build a pipeline.** E0→E1→...→R6→PCH meant every stage depended on
  every previous one. One miscalibrated parameter cascaded everywhere. v2 has
  one algorithm with one data structure.

- **Don't separate learning from inference.** v1's batch training phase caused
  paragraph-swallowing and made online use impossible. In v2 every observation
  immediately updates the model.

- **Don't impose categories.** K-means requires choosing K and an initialisation.
  In v2 categories are morphisms — they form when evidence supports them and
  dissolve when it doesn't.

- **Don't use dense matrices.** O(V²) was the root cause of every performance
  cliff in v1 (V≤300 cap, max_atoms=200, O(K⁴) warm-up). All of these were
  symptoms of the wrong data structure.

- **Don't conflate grammar with reasoning.** Grammar is a special case of morphism
  discovery. Building a grammar-specific system and then bolting on reasoning
  creates the seam that v1's CTKG bridge was trying (and failing) to paper over.

- **Test in seconds.** If a correctness check requires running the full corpus,
  the architecture is wrong. Every invariant must be testable on a toy example
  in < 1 second.

- **One file = one concept.** v1's relational_pipeline.py reached 6,365 lines.
  v2 files stay small by design; complexity is managed through interfaces, not
  by putting everything in one place.

---

## Folder structure (target)

```
symbolic_ai_v2/
├── GOALS.md                  ← this file
├── core/
│   ├── morphism.py           ← the one algorithm: observe edge → update morphism graph
│   ├── topology.py           ← graph topology abstraction (sequence/image/relational)
│   ├── memory.py             ← short-term buffer + long-term morphism compression
│   └── predict.py            ← P(next | context) from current morphism graph
├── reasoning/
│   ├── active_inference.py   ← free energy minimisation; action selection
│   └── ctkg_live.py          ← live CTKG that updates on every observation
└── tests/
    ├── latin_test.py         ← perplexity benchmark vs transformer
    ├── arithmetic_test.py    ← count → add → multiply compositional chain
    ├── long_context_test.py  ← needle-in-haystack at controlled distances
    ├── topology_test.py      ← same core logic handles 1D/2D/graph
    └── transfer_test.py      ← functor alignment across two domains
```

Total target: < 1500 lines of core logic. Each file independently testable in
< 1 second on a toy example.

---

## Key open questions (to resolve before writing code)

1. **Morphism graph representation**: A sparse directed multigraph where nodes
   are observed tokens/abstractions and edges are typed relations with counts.
   The right data structure is likely a dict-of-dicts (adjacency list) with
   edge types as the outer key. Open: how to represent *compositions* of
   morphisms efficiently without materialising all paths.

2. **O(n) composition discovery**: Sequitur's digram uniqueness gives O(n)
   grammar induction for sequences. The equivalent for general graphs is an
   open research question. Candidate: maintain a priority queue of
   (edge-pair, count) and merge pairs greedily when count ≥ 2. This extends
   Sequitur to arbitrary topologies.

3. **Prediction from a morphism graph**: P(next | context) requires
   marginalising over all current parse/composition states. For O(n log n)
   prediction: maintain a running parse stack and use rule-continuation
   frequencies. Smoothing across abstraction levels replaces Kneser-Ney.

4. **Free energy grounding**: Free energy = description length of observations
   under current morphism graph + description length of graph itself (MDL).
   Minimising this jointly drives both perception (update graph to compress
   observations better) and action (choose observations that most reduce
   description length). Need to confirm this reduces to standard AIF equations.

5. **CTKG update granularity**: Every observed edge is a morphism candidate.
   Low-count morphisms live in a "tentative" buffer; they graduate to confirmed
   CTKG concepts when count ≥ 2 (MDL threshold). Composition (functor,
   adjunction discovery) runs lazily when a morphism is promoted, not on
   every observation.
