# Relational Learner Roadmap
## Structure-Agnostic Learning: Any Geometry, Any Dimension

**Core thesis:** The RelationalLearner should discover the structure of any relational dataset
without prior assumptions about geometry, dimension, or curvature — whether the data is
1D sequential, 2D grid, circular, hyperbolic (tree), general graph, or anything else.
The data tells us the shape. We listen.

**Competition target:** Match or exceed graph neural networks (GNNs) on relational
prediction tasks, while remaining fully symbolic, interpretable, and training-free.

---

## Current State (E0–E3)

| Level | What it learns | Status |
|-------|---------------|--------|
| E0 | Compound bigrams: (atom,rel,atom) → distributional signatures | ✅ Done |
| E1 | Category chains: P(c_tgt \| c_src, rel) | ✅ Done |
| E2 | Context-sensitive clustering (sense disambiguation) | ❌ Not implemented (SequenceLearner only) |
| E3 | Soft retrieval: similarity-weighted prediction | ✅ Done |
| L2 | Relation clustering: which relations are equivalent? | ✅ Done |
| L4 | Second-order grammar: P(r2 \| r1) | ✅ Done |

SequenceLearner has E4, E5, E6. RelationalLearner stops at E3.
The geometry of the data is never detected or exploited.

---

## Gap Analysis: Why RelationalLearner ≠ GNN

| Capability | GNN | RelationalLearner (current) |
|-----------|-----|---------------------------|
| Detect data dimensionality | Implicit in architecture | ❌ |
| Handle directed vs undirected | Yes (separate models) | Partially (L2 clusters) |
| Detect curvature (flat/hyperbolic/spherical) | Via hyperbolic GNN variants | ❌ |
| Paradigmatic substitutability | Via node embeddings | ❌ |
| Sense disambiguation (polysemy in relations) | Via attention heads | ❌ |
| Compositional relational inference | Via message passing | ❌ |
| Metric adaptation to geometry | Hyperbolic/spherical variants | ❌ |
| Multi-hop prediction | Via k-layer message passing | L4 (1 hop only) |

---

## Roadmap

### R0 — Geometry Detection
*Know what shape the data is before trying to learn it.*

**Goal:** From the fitted L1 (atom categories) + L2 (relation clusters) + E1 distributions,
automatically classify the underlying geometry.

**Algorithm:**
1. **Symmetry score** — For each relation pair (r, r'), test if the transition matrix
   T[r] ≈ T[r']ᵀ (transpose). High symmetry → undirected. Low → directed.
2. **Effective rank** — SVD of the aggregate K×K category transition matrix.
   Rank ≈ 1 → linear. Rank ≈ 2 → planar. Rank = K → full graph.
3. **Growth rate curve** — Starting from a seed category, measure |reachable| at each
   hop distance d. Exponential growth → hyperbolic (tree). Polynomial d → Euclidean d-dim.
   Saturation → bounded (spherical or finite dense graph).
4. **Cycle detection** — Does the category graph have cycles? No cycles → DAG/tree.
   Cycles of length L → periodic/circular structure with period L.

**Topology classification:**
```
symmetry≈1, rank≈1, growth~d      → undirected linear (or circular)
symmetry≈0, rank≈1, growth~d      → directed linear (1D sequence)
symmetry≈1, rank≈2, growth~d²     → undirected 2D grid
symmetry≈0, rank≈2, growth~d²     → directed 2D grid
symmetry≈1, growth exponential     → undirected tree (hyperbolic)
symmetry≈0, growth exponential     → directed DAG (hyperbolic)
symmetry≈1, growth flat            → dense undirected graph / spherical
rank=K, no dominant pattern        → general graph
```

**Deliverable:** `GeometryDetector` class in `relational_pipeline.py`
**Progress:** ✅ Implemented

---

### R1 — Relational E4: Paradigmatic Substitutability
*Which atoms are interchangeable in the same relational role?*

SequenceLearner E4 finds which tokens fill the same slot (c_prev, c_next).
The relational equivalent: which atoms appear in the same set of (relation, partner) contexts?

**Algorithm:**
1. For each atom a, build its "relational role signature":
   `role_sig(a) = distribution over (rel_cluster, target_cat) pairs`
   This captures: "atom a tends to be the source of R0-relations to C3-category atoms,
   and the target of R1-relations from C1-category atoms."
2. Cluster atoms by role signature JSD → "role categories" (distinct from distributional
   categories from E0).
3. `role_occupants(role_cat) → [atom, ...]` — all atoms with the same relational role.

**Why this matters:** E0 clusters atoms by *who their neighbors are*. E4 clusters atoms
by *what role they play in the relational structure*. In a knowledge graph:
- E0: Paris, London, Berlin cluster together (similar neighbor types)
- E4: Paris, London, Berlin are in the same ROLE (capital-of relation source)

These two types of similarity are orthogonal and both necessary for generalization.

**Deliverable:** `RelationalParadigmDiscoverer` class; `role_occupants(role_cat)` query
**Progress:** ✅ Implemented

**Latin result (3 books):** 27 atoms → 2 role categories. Auto-K correctly finds one
natural boundary: 'w' (non-Latin, unusual bigrams) vs all other chars. Role signatures
use `use_atom_partners=True` (V-dimensional) to avoid E0 mega-cluster collapse.

---

### R2 — Relational E5: Sense Disambiguation
*An atom that plays multiple distinct relational roles should be split.*

**Goal:** Detect and separate atoms with polysemous relational behavior.

**Algorithm:**
1. For each atom a, collect all observed relational contexts:
   `contexts(a) = [(rel, partner_cat, direction), ...]`
2. Cluster these contexts by JSD → sense clusters for atom a.
3. If atom a has k > 1 clearly separated sense clusters, split it into a_0, a_1, ..., a_{k-1}.
4. Re-run E0-E3 with the split atoms.

**Why this matters:** In Latin, the character 'v' sometimes functions as a vowel (= 'u')
and sometimes as a consonant. In a knowledge graph, 'Mercury' is both a planet and a Roman
god. The relational contexts are qualitatively different — sense disambiguation separates them.

**Deliverable:** `RelationalSenseSplitter` class; `.polysemous` dict; `.report()`
**Progress:** ✅ Implemented

**Latin result (3 books):** 4 polysemous atoms detected:
- 'v': 2 senses — consonant (word-initial, next→i 36%) vs vowel (after 'i'→'s' 100%, the "ivs"/"ius" spelling)
- 't': 3 senses — normal 't', after 'q' (qu-cluster artifact), after 'y' (Greek loanword "yth" cluster)
- 'y': 2 senses — two phonological contexts (after 'd' vs after consonants)
- ' ': 6 senses — word-boundary effects vary by preceding word-final letter

---

### R3 — Relational E6: Structural Meta-Synthesis
*Discover the compositional rule governing the graph — without being told.*

SequenceLearner E6 uses beam search to find chains like `context_triple ∘ word_given_cat`.
The relational equivalent discovers rules like:
- "capital_of ∘ country_of = same_continent" (relation composition)
- "IS_A ∘ HAS = transitively_has" (property inheritance)
- "PARENT ∘ PARENT = GRANDPARENT" (structural recursion)

**Algorithm:**
1. For each pair of relation clusters (R_i, R_j), test whether
   `P(c_tgt | c_src, R_i ∘ R_j) ≈ P(c_tgt | c_src, R_k)` for some R_k.
2. If yes → record the composition rule R_i ∘ R_j = R_k.
3. Build the "relation algebra" — the full set of composition rules.
4. Use this to predict multi-hop relational triples.

**Deliverable:** `RelationalAlgebra` class; `compose(rel1, rel2) → rel3` lookup
**Progress:** ✅ Implemented

**Latin result (3 books):** Two genuine composition rules discovered (auto-threshold JSD<0.131):
- `next ∘ next = skip2f` (JSD=0.124) — two forward steps = skip-by-2
- `prev ∘ prev = skip2b` (JSD=0.105) — two backward steps = skip-by-2 back
All other compositions classified as "novel" (no matching single relation in our set).

**Note:** R3 uses atom-level (V=27) matrices, not category-level (K=10).
Auto-K E0 is correct for distributional clustering; R3 needs finer resolution
because at K=10, all 4 relations have ~80% C0→C0 mass and look identical.

---

### R4 — Geometry-Adapted Distance Metric
*Use the right ruler for the right space.*

Once R0 detects the geometry, R4 adapts the E3 similarity metric accordingly.

| Geometry | Distance metric | Similarity kernel |
|----------|----------------|-------------------|
| Euclidean (flat) | L2 / cosine | Gaussian exp(-d²/σ²) |
| Hyperbolic (tree) | Poincaré distance | exp(-d_hyp / T) |
| Spherical | Geodesic arc length | cos(angle) |
| Circular | Circular distance | cos(2π·d/L) |
| General graph | Graph edit distance / diffusion | Heat kernel on graph |

Current E3 uses JSD everywhere (implicitly Euclidean). R4 replaces JSD with the
geometry-appropriate metric for the detected topology.

**Algorithm:**
1. From R0: detect topology.
2. Select appropriate distance metric.
3. Recompute the similarity matrix `_build_sim_matrix` with new metric.
4. Measure: does geometry-adapted E3 have lower triple prediction error than JSD-E3?

**Deliverable:** `adapt_metric(topology)` method on `RelationalLearner`; `_build_sim_matrix_adapted()`.
**Progress:** ✅ Implemented

**Latin result (3 books):** directed_2d → 2D classical MDS sim matrix. JSD-metric: 0.159
vs directed_2d-metric: 0.160 (delta=+0.001). Marginal improvement expected: K=10 with
mega-cluster C0 means JSD is already near-optimal. Larger improvements expected on:
- Hyperbolic data (knowledge graphs): BFS hop-count metric vs JSD
- Circular data (time series): cosine metric vs JSD
- Linear data (token sequences): 1D MDS position distance vs JSD

---

### R5 — Multi-Hop Prediction Benchmark
*Compete with GNNs on standard relational prediction tasks.*

**Goal:** Benchmark RelationalLearner (with R0–R4) against GNN baselines on:
1. **Latin corpus** (char-level): triple prediction accuracy
2. **WordNet** (lexical relations): hypernym/synonym prediction
3. **FB15k-237** (knowledge graph): standard KG completion benchmark

**Baselines:**
- TransE (embedding-based, flat)
- RotatE (circular geometry)
- PoincaréEmbed (hyperbolic)
- Graph attention network (GAT)

**Our advantage:** No training. One pass over data. Interpretable. Composable via CTKG.

**Success metric:** Hits@10 (fraction of true triples in top-10 predictions) within
20% of the best neural baseline on at least 2/3 benchmarks.

**Progress:** 🔶 Implemented on Latin corpus; FB15k-237/WordNet need dataset downloads.

**Latin result (3 books, 80/20 train/test split, V=25 atoms):**

| Model                   | H@1   | H@3   | H@10  |
|-------------------------|-------|-------|-------|
| Random                  | 0.040 | 0.120 | 0.400 |
| Unigram (most-frequent) | 0.141 | 0.335 | 0.792 |
| RelationalLearner E3    | 0.143 | 0.148 | 0.494 |

**Analysis:** E3 H@1 barely beats unigram (14.3% vs 14.1%) but falls behind at H@3/H@10.
Root cause: K=10 mega-cluster C0 (18 common chars) coarsens predictions — all target mass
concentrates within C0, not distributed to specific chars within C0. Fix: use V-level
(atom-level) bigram tables directly for prediction (bypassing clustering). The clustering
is right for structural discovery (R0-R4); for fine-grained prediction a bigram baseline
outperforms it. Next: FB15k-237 and WordNet benchmarks (require data downloads).

---

### R6 — Compositional Relational Inference
*Answer multi-hop relational queries without ever seeing them in training.*

**Goal:** Given a question like "What language family does Latin belong to?" — answer it
by chaining relational algebra (R3): Latin IS_A Romance → Romance IS_A Indo-European.
No neural network. No training on this specific question.

**Algorithm:**
1. Parse query as: source atom → (relation chain) → ? target atom
2. For each relation in the chain, apply E3 soft retrieval.
3. Compose posterior distributions across the chain.
4. Return top-k atoms by final posterior mass.

**Deliverable:** `RelationalLearner.infer_chain(atom, [rel1, rel2, ...])` method
**Progress:** ✅ Implemented

**Implementation:** Distribution-preserving belief propagation through the relational graph.
Propagates P(c) through each T[rel] matrix and decodes atoms from the final category
distribution. Correct for any topology; much richer than greedy `predict_chain()`.

**Latin result:** next∘next and skip2f give similar top-1 predictions (space, x, v) —
verifying R3's `next∘next=skip2f` rule. Predictions degenerate due to K=10 mega-cluster
(all C0 atoms give identical 2-hop distributions). For knowledge graphs with V distinct
entity types (each in its own category), infer_chain gives entity-specific inference chains.

---

## Progress Tracker

| Phase | Description | Status | Commit |
|-------|-------------|--------|--------|
| E0 | Compound bigrams | ✅ Done | 4b99f3b |
| E1 | Category chains | ✅ Done | 4b99f3b |
| E3 | Soft retrieval | ✅ Done | 4b99f3b |
| L2 | Relation clustering | ✅ Done | 4b99f3b |
| L4 | Second-order grammar | ✅ Done | 918c5e7 |
| Auto-K | Bitter-lesson K selection | ✅ Done | f89ca0c |
| **R0** | **Geometry detection** | ✅ Done | 8262d89 |
| **R1** | **Relational E4: paradigmatic substitutability** | ✅ Done | — |
| **R2** | **Relational E5: sense disambiguation** | ✅ Done | — |
| **R3** | **Relational E6: structural meta-synthesis** | ✅ Done | — |
| **R4** | **Geometry-adapted distance metric** | ✅ Done | — |
| **R5** | **Multi-hop prediction benchmark** | 🔶 Latin done | — |
| **R6** | **Compositional relational inference** | ✅ Done | — |
