# symbolic_ai_v2 — Architecture Blueprint

This document describes the concrete design of the system: data structures,
algorithms, interfaces, and complexity bounds. See GOALS.md for intentions
and RESEARCH.md for theoretical background.

---

## Core abstraction: everything is a graph

All input data is represented as a **labeled directed graph** where nodes are
observations (tokens, pixels, sensor readings) and edges encode the structural
relationships between them according to the input's native coordinate system.

The edge types encode the topology:

| Input type | Edge types | Notes |
|------------|-----------|-------|
| 1D sequence | `next`, `prev` | Path graph |
| 1D periodic sequence | `next`, `prev` | Cycle graph: last node's `next` = first node |
| 2D image | `right`, `left`, `down`, `up` | Grid graph |
| 2D image (diagonal) | + `dr`, `dl`, `ur`, `ul` | 8-connected grid |
| Video | `right`, `left`, `down`, `up`, `later`, `earlier` | Spatiotemporal grid |
| Audio spectrogram | `next`, `prev`, `higher`, `lower` | Time-frequency grid |
| Curved space / arbitrary metric | edges between ε-neighbors | Graph encodes metric |
| Relational data / knowledge graph | arbitrary typed edges | Already a graph |
| Mathematical expression | `left-child`, `right-child`, `parent` | Parse tree graph |

**The algorithm has no knowledge of what the edge types mean.** It sees only
`(source_node, edge_type, target_node)` triples. Adding a new modality requires
only defining a graph topology over its observations — zero new core logic.

A topology is fully specified by two things:
1. A finite set of named edge types.
2. A function `neighbors(position) → list[(edge_type, neighbor_position)]`
   that maps any position to its typed neighbors.

---

## The single core data structure: the Morphism Graph

```
MorphismGraph:
    symbols:    dict[symbol_id → Symbol]
    edges:      dict[(src_id, edge_type, tgt_id) → count]
    pairs:      dict[(src_id, edge_type_1, mid_id, edge_type_2, tgt_id) → count]
    rules:      dict[rule_id → (src_id, edge_type, tgt_id)]   # compositions
    parse:      list[symbol_id]                                 # active parse stack
    chunk_buf:  list[(symbol_id, edge_type)]                   # current local chunk
```

A `Symbol` is one of:
- **Atom**: a primitive observation value (character, pixel colour, sensor reading).
  Created the first time a value is observed.
- **Composition**: a named abstraction over a recurring sub-graph pattern.
  Created automatically when a pattern's count reaches 2.

Atoms and Compositions are treated identically by the algorithm. The distinction
is only for inspection and export.

---

## The single core algorithm: Graph-SEQUITUR

Generalises Nevill-Manning & Witten (1997) from 1D sequences to arbitrary
labeled directed graphs. Two invariants are maintained at all times:

1. **Edge-pair uniqueness**: no pair of consecutive edges `(A →[e1]→ B →[e2]→ C)`
   appears more than once in the current parse. When it would appear for a second
   time, a new Composition symbol `AB` is created and all occurrences are replaced.

2. **Rule utility**: every Composition is used at least twice. A Composition used
   only once is dissolved back into its constituents.

### Processing one observation

```
observe(node_value, incoming_edge_type):

    1. Resolve node_value to its symbol S (create Atom if first occurrence).

    2. Let (P, e_prev) be the last entry on the parse stack.

    3. Increment edge count: edges[(P, e_prev, S)] += 1.

    4. Check pair: if chunk_buf is non-empty, let (Q, e_q) = chunk_buf[-1].
       Increment pairs[(Q, e_q, P, e_prev, S)] += 1.
       If pairs[(Q, e_q, P, e_prev, S)] == 2:
           create_composition(Q, e_q, P, e_prev, S)   # O(1) with hash table

    5. Append (S, incoming_edge_type) to chunk_buf.

    6. Push S onto parse stack.
```

**Complexity**: O(1) per observation (hash table lookups and insertions only).
Total: O(n) in the number of observations.

### create_composition

```
create_composition(Q, e1, P, e2, S):

    1. Allocate new symbol C (a Composition representing P →[e2]→ S).

    2. Add rule: rules[C] = (P, e2, S).

    3. Find all occurrences of (P →[e2]→ S) in the current parse and replace
       each with C.  Update pair counts accordingly.

    4. Check rule utility: if C appears only once after substitution, dissolve C
       (replace with original constituents, remove rule).
```

**Complexity**: Each symbol is created and dissolved at most once per occurrence.
Amortised O(1) per observation.

### Segment boundaries

A segment boundary is detected when the last edge-pair in chunk_buf has count 1
(the current pair has never been seen before). This is the MDL criterion: a
new pair does not yet reduce description length.

At a segment boundary:
1. Flush chunk_buf → emit a completed local chunk.
2. Run FCA on the local chunk's concept matrix (see below).
3. Sheaf-merge the local concept lattice into the global CTKG.
4. Reset chunk_buf.

Segment boundaries are data-driven and require no threshold.

---

## Local structure: FCA on each chunk

After each segment boundary, the current chunk contains at most W symbols
(W is the empirically observed chunk size, not a parameter). In practice W is
small: natural language segments are 5–20 symbols at each hierarchy level.

**Concept matrix**: a binary matrix M where M[i,j] = 1 iff symbol i has been
observed in the context of edge-type j within this chunk. Rows = symbols (K),
columns = edge types (R). Typically K ≤ 20, R ≤ 8.

**FCA on M**: finds all formal concepts (closed pairs of symbol-sets and
edge-type-sets). Each concept is a Galois connection = an adjunction.
Complexity: O(K · R · C) where C = number of concepts. For K=20, R=8,
C ≤ 2^min(K,R) in the worst case but typically C ≈ K in practice.

Each discovered adjunction is immediately added to the CTKG as an
`Adjunction` object connecting the two CTKG `Concept` nodes.

**Why FCA does not blow up**: K is bounded by the chunk size W, not by the
total vocabulary. A 20-symbol chunk has at most 20 distinct symbol types.
FCA on a 20×8 matrix completes in microseconds.

---

## Global structure: the CTKG as a sheaf

The global CTKG is a **sheaf** over the corpus. Its sections are the local
concept lattices discovered in each chunk. The sheaf conditions are:

1. **Restriction**: if two chunks share a symbol, they must agree on that
   symbol's edge-type distribution (up to sampling noise).
2. **Gluing**: consistent local sections can be glued into a global section.

`sheaf_merge(local_kg, global_kg)` (already implemented in `experiments/ctkg/`)
performs step 2 and raises `SheafViolation` on step 1 failure.

A `SheafViolation` is not an error — it signals that the same surface symbol
has two incompatible structural roles in different contexts. This is the trigger
for **sense disambiguation**: create two distinct CTKG concepts for the two
usages of the symbol.

The global CTKG accumulates all discovered:
- Types (from local FCA concept lattices)
- Morphisms (from observed edge counts)
- Adjunctions (from FCA Galois connections)
- Functors (from cross-level symbol alignment, see below)
- Causal structure (from d-separation on the Markov KG)

Long-range dependencies are captured here. A fact established in chunk 1 is
a morphism in the global CTKG. When chunk 10,000 contains a pattern consistent
with that morphism, the sheaf consistency check finds the connection without
any explicit long-range lookup.

---

## Cross-level structure: the Hopf algebra coproduct as functor

The multi-level hierarchy (atoms → level-1 compositions → level-2 compositions
→ ...) is not a separate architectural feature. It is the canonical structure
of the Merge Hopf algebra (Marcolli, Chomsky & Berwick 2023).

- **Product** (Merge): two symbols S1, S2 compose into C. Implemented by
  `create_composition`.
- **Coproduct** (Segment): a composition C decomposes into (S1, S2). The inverse
  of Merge. Implemented by dissolving rules.
- **Counit**: the map C → base field (the "meaning" of C as a scalar = its
  empirical frequency).

The coproduct Δ: C → C ⊗ C is the cross-level functor. It maps a level-N
symbol to the tensor product of its level-(N-1) constituents. This functor is
**automatically induced** by the composition rules — it does not need to be
discovered separately.

In the CTKG, each `Composition` node carries:
- Its rule (the pair of constituents).
- Its level (= max(level(S1), level(S2)) + 1).
- Its frequency (edge count / total edges at this level).

The `build_functor()` method across levels therefore reduces to: for each
composition C at level N, its image under the coproduct functor is the pair
(S1, S2) at level N-1. O(|rules|) = O(n) total.

---

## Prediction: P(next | context)

Given the current parse stack, the predictor produces a probability
distribution over the next symbol for each outgoing edge type.

```
predict(parse_stack, edge_type) → dict[symbol → probability]:

    1. Let S = top of parse_stack.

    2. Fast path (seen symbol): return normalised edge counts
       { T : edges[(S, edge_type, T)] / sum_T edges[(S, edge_type, T)] }
       This is the hom-object in the enriched category (Bradley et al. 2021).

    3. Smoothing (unseen symbol): back off to the marginal over the CTKG type
       of S. P(T | type(S), edge_type) via the K×K type transition matrix.
       This is Kneser-Ney-style back-off, but grounded in the CTKG type system
       rather than n-gram counts.

    4. If S is a Composition, also condition on its constituents (Hopf coproduct
       structure): the prediction is a convex combination of the predictions
       from S directly and from the prediction of S's right constituent.
       Weight = confidence in S's rule (= min(count(S's rule), 1.0) / max_count).
```

**Complexity**: O(degree(S)) per prediction, where degree = number of distinct
outgoing edges from S. In a compressed grammar, degree ≪ V.

---

## Active inference: free energy as description length

The agent's objective is to minimise the **free energy** F of its current model:

```
F = description_length(observations | model) + description_length(model)
  = -log P(observations | grammar) + |grammar|
```

This is the MDL criterion. Minimising F simultaneously:
- **Perception**: updates the grammar to better predict observations (reduces
  the first term).
- **Compression**: keeps the grammar small (reduces the second term via rule
  utility — dissolve rules used only once).

The **prediction error** at each observation is:
```
error(S, e, T) = -log P(T | S, e)   [bits]
```

High error → the current grammar does not predict T given S via edge e → this
edge-pair is a candidate for a new composition (if it recurs, it will trigger
`create_composition`). Low error → the grammar predicted T correctly → no
update needed.

**Action selection** (for a true agent, not just a language model): the agent
chooses its next observation (e.g. next saccade direction, next API call) to
maximise expected information gain = expected reduction in F. This is:
```
EIG(action a) = F_current - E[F | take action a]
              = H(prediction | current model) - E[H(prediction | updated model)]
```
The agent acts to maximally compress its own model. No separate reward signal.

---

## Memory

### Short-term memory (chunk buffer)
The `chunk_buf` list. Bounded by the natural segment size W (empirically 5–30
symbols at each level). This is the working memory window. No hyperparameter.

### Long-term memory (global CTKG)
All discovered morphisms, types, adjunctions, functors, and causal structure.
Retrieval is O(degree) via edge lookup. Storage is O(|edges|) = O(n / compression_ratio).

### Episodic memory (optional)
Individual chunk boundaries, stored as timestamps on CTKG nodes. Enables
"when did I learn this?" queries. Not required for prediction; useful for
reasoning about recency and context of knowledge.

---

## File structure

```
symbolic_ai_v2/
├── GOALS.md              ← intentions
├── RESEARCH.md           ← theory
├── BLUEPRINT.md          ← this file
│
├── core/
│   ├── topology.py       ← graph topology: defines edge types + neighbors()
│   │                        Input: any (topology_spec) → stream of (src, etype, tgt)
│   │                        Zero knowledge of what edge types mean.
│   │
│   ├── morphism.py       ← Graph-SEQUITUR: the one algorithm
│   │                        MorphismGraph, observe(), create_composition(),
│   │                        segment_boundary(), dissolve()
│   │                        All operations O(1) amortised.
│   │
│   ├── predict.py        ← P(next | parse_stack, edge_type)
│   │                        Fast path: edge count lookup O(degree)
│   │                        Fallback: CTKG type back-off O(K)
│   │
│   └── memory.py         ← Short-term: chunk_buf  (bounded list)
│                            Long-term:  CTKG reference + sheaf_merge()
│                            No separate retrieval mechanism needed.
│
├── reasoning/
│   ├── fca.py            ← FCA on local concept matrix
│   │                        Input: K×R binary matrix (K ≤ W, R = #edge types)
│   │                        Output: list of (symbol_set, edge_type_set) concepts
│   │                        Complexity: O(K · R · C), C ≤ K in practice
│   │
│   ├── active_inference.py  ← Free energy = MDL cost. prediction_error(),
│   │                           expected_information_gain(), act()
│   │
│   └── ctkg_live.py      ← Live CTKG: sheaf_merge() on every segment boundary.
│                            Wraps experiments/ctkg/graph.py + parser.py.
│                            Adds: sense_disambiguate() on SheafViolation.
│
└── tests/
    ├── latin_test.py         ← Perplexity on EarlyModernLatin vs transformer baseline
    ├── arithmetic_test.py    ← count → succ → add → mul compositional chain
    ├── long_context_test.py  ← Needle-in-haystack at 100 / 1K / 10K tokens
    ├── topology_test.py      ← Same core code handles path / cycle / grid input
    └── transfer_test.py      ← Functor alignment: Latin chars ↔ arithmetic digits
```

Total target: ≤ 1500 lines across `core/` and `reasoning/`. Each file is
independently testable with a toy example in < 1 second.

---

## Complexity summary

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| observe() | O(1) amortised | Hash table lookups only |
| create_composition() | O(1) amortised | Amortised over all observations |
| segment_boundary() FCA | O(K · R · C) | K ≤ W ≈ 20; microseconds |
| sheaf_merge() | O(K_local · K_global) | K_local ≤ W; K_global grows sublinearly |
| predict() fast path | O(degree(S)) | degree ≪ V in compressed grammar |
| predict() fallback | O(K) | K = number of CTKG types |
| Total per observation | O(1) amortised | Entire pipeline |
| Total for n observations | O(n) | Linear in corpus size |

---

## What is explicitly not in this design

- **No K-means or EM clustering.** Categories emerge from FCA on local chunks.
- **No dense V×V matrices.** All matrices are K×R where K ≤ W ≤ 30.
- **No fixed vocabulary size.** New Atom symbols are created on first occurrence.
- **No surprise threshold.** Segment boundaries come from pair count = 1 (MDL).
- **No chunk size parameter.** W emerges from the data's natural segment structure.
- **No training / inference phases.** observe() both learns and predicts on
  every token.
- **No domain-specific code in core/.** topology.py maps any modality to
  (src, etype, tgt) triples; everything else is topology-agnostic.
- **No separate grammar layer and reasoning layer.** Every composition rule is
  immediately a CTKG morphism.
