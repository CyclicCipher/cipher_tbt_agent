# All-Scale Pattern Learning: The Merge Roadmap

## The Problem

The RelationalLearner operates at a fixed token level with positional relations
(`next`, `prev`, `skip2f`, `skip2b`). It cannot learn patterns at multiple
scales — morphemes, words, phrases, clauses — because it has no mechanism to
discover or represent hierarchical structure. The result is a sophisticated
character bigram model. That is insufficient for symbolic AGI.

---

## The Biological Model: Broca's Area

Broca's area (inferior frontal gyrus, BA44/45) was historically labelled a
speech production center. Modern neuroimaging revised this completely. It
activates identically during:

- Language comprehension (syntax, recursion, embedding)
- Music processing (phrase structure, harmonic expectation)
- Action observation and imitation
- Mathematical reasoning
- Non-linguistic sequential pattern learning

It is **domain-general**. The same region, the same computation, across all
structured domains. This is the first key fact.

The second key fact: Broca's area does not process scales sequentially.
Different neural populations track ~25ms (phoneme), ~200ms (syllable),
~500ms (word), and ~2000ms (phrase) simultaneously. All scales active in
parallel, mutually constraining each other.

The third key fact: the core operation is **Merge**. Not chunking.

  Chunking:  A B → [AB]               (flat, loses structure)
  Merge:     A B → {label, left:A, right:B}   (hierarchical, preserves structure)

Merge combines two units into a labeled hierarchical node. The result can be
merged again. This is what makes recursion possible: Merge is its own input.
Broca's area is a recursive, domain-general, multi-scale Merge engine.

The fourth key fact: it is predictive. Higher levels send top-down predictions
to lower levels. Phrase-level context constrains word-level perception.
Word-level context constrains phoneme-level perception. Learning flows both
bottom-up (data → structure) and top-down (structure → data expectation).

---

## The Formal Theory: CTKG as Categorical Composition

The Category Theory Knowledge Graph was designed to represent compositional
knowledge: which concepts are prerequisites for which, how domains relate via
functors, how inverse operations pair via adjunctions, how overlapping
definitions cohere via sheaves.

Category theory's fundamental operation is **composition of morphisms**:

  f: A → B,  g: B → C   ⟹   g∘f: A → C

This is Merge, stated formally. Two structured things combine into a new
structured thing, the internal structure is preserved, and the result carries a
label (the composite morphism). It is recursive. It is domain-general: the
same operation applies in arithmetic, syntax, logic, geometry.

The CTKG's components map directly to Broca's area's properties:

| CTKG component       | Broca's area property            |
|----------------------|----------------------------------|
| Type hierarchy       | Multi-scale atom structure       |
| Prerequisite DAG     | Merge order constraints          |
| Functors             | Domain-generality                |
| Adjunctions          | Merge and its inverse            |
| Sheaf consistency    | Multi-scale coherence            |
| Process primitives   | Atomic operations before Merge   |

**The key insight:** The CTKG is not a curriculum tool with a learning system
bolted on. It IS the formal description of what Broca's area computes.
Categorical composition IS Merge. They are the same abstraction at two
different levels of description.

This means: the CTKG is the **target representation** that the RelationalLearner
should converge to. Given enough data and the right learning algorithm, the
system should discover a CTKG-like structure — the type hierarchy, the
prerequisite ordering, the functor relations — from raw input alone.

---

## The Equivalence: One Operation

  Broca's Merge  =  categorical composition  =  RelationalLearner + Merge

All three are the same computation. The RelationalLearner already does the
statistical half: it learns `P(target | atom, relation)`. What it is missing
is the structural half: using those statistics to **discover when two atoms
form a higher-level unit**, and then **operating on that unit as a new atom**.

This is the entire overhaul in one sentence:

> Add Merge to the RelationalLearner, apply it recursively, and the result is
> a domain-general, all-scale pattern learner whose discovered structure is a
> CTKG instance.

---

## Current Architecture Diagnosis

```
RelationalLearner
├── _atom_bigrams: {(atom, rel): {target: prob}}   ← correct mechanism
├── _rel_unigram: {rel: {target: prob}}             ← correct OOV fallback
├── infer_chain(): multi-hop belief propagation     ← correct
├── ContextBeliefState: Bayes filter over K cats    ← correct
└── MISSING: Merge — no way to compose atoms into higher-level units
```

Relations are positional (`next`, `prev`, `skip2f`) because the atoms are
characters. Once atoms can be composed, relations between composed units are
structural, not positional: `left_constituent`, `right_constituent`,
`subject_of`, `modifier_of`, `contradicts`, `implies`. These are not defined
in advance — they are discovered from the distributional behavior of merged
units, exactly as E3 currently discovers atom categories.

---

## The Overhaul: HierarchicalRelationalLearner

### Core data structures

```python
@dataclass
class MergedAtom:
    label: str           # discovered from distributional behavior at level N+1
    left: Any            # left constituent (atom or MergedAtom)
    right: Any           # right constituent (atom or MergedAtom)
    level: int           # 0 = base, 1 = first merge, ...
    surface: str         # original surface form (for display)

class HierarchicalRelationalLearner:
    levels: list[RelationalLearner]  # one per discovered level
    vocab: dict[int, set]            # atoms at each level
    merge_history: list              # ordered list of merges performed
```

### The Merge operation

A merge candidate (A, B) is scored by **pointwise mutual information**:

  PMI(A, B) = log P(B | A, 'next') - log P(B)

High PMI: A and B co-occur more than chance — they form a unit.
Low PMI between consecutive units: boundary.

This is exactly Saffran et al.'s statistical learning result: infants discover
word boundaries from transitional probability drops. The RelationalLearner
already computes these probabilities. Merge acts on them.

Merge execution:
1. Rank all adjacent pairs by PMI
2. Execute top-K merges (K is a hyperparameter, or determined by a PMI threshold)
3. Create MergedAtom(label=TBD, left=A, right=B, level=N+1)
4. Re-tokenize data: replace all (A, B) sequences with the MergedAtom
5. Run RelationalLearner on the new atom set at level N+1
6. Labels for MergedAtoms are discovered by E3 clustering at level N+1
   (atoms that appear in the same positions share a label — a CTKG type)
7. Recurse

### Multi-scale simultaneity

Levels are not processed sequentially and discarded. All levels remain active.
The HierarchicalRelationalLearner maintains a RelationalLearner at each level.
When a new observation arrives:
- Level 0 updates character bigrams
- Level 1 updates morpheme bigrams (via the current merge vocabulary)
- Level N updates phrase bigrams
- All simultaneously

Top-down: `predict_dist()` at level 0 is conditioned on the current belief at
level 1, which is conditioned on level 2, etc. Higher-level context shifts the
probability distribution at lower levels. This is the ContextBeliefState,
generalized to multiple levels.

### Relation discovery

At level 0, relations are positional (`next`, `prev`).
At level N > 0, relations between merged units are structural. Two kinds:

1. **Constituency relations** (internal to a MergedAtom):
   `left_of`, `right_of` — always available from the merge structure

2. **Distributional relations** (between MergedAtoms at level N):
   Discovered by running E3-E6 on level-N atoms. If [AB] and [CD] appear
   in the same relational contexts, they share a distributional category
   (= a CTKG type). If [AB] systematically precedes [CD] and [CD] systematically
   follows [AB], that asymmetric relation gets a label.

Adjunctions emerge naturally: if [AB] and the operation that produces [BA]
have symmetric distributional profiles, they are adjoints — just as the CTKG
marks add/subtract as adjoint.

---

## Phase Roadmap

### M1 — Merge Detection  (prerequisite: none)
Add `MergeDetector` to `relational_pipeline.py`.
- Input: fitted RelationalLearner, training sequences
- Output: ranked list of (A, B, PMI) merge candidates
- Uses existing `_atom_bigrams` — no retraining
- Expose `detect_merges(threshold=None, top_k=None) → list[MergeCandidate]`

Validation: on Latin corpus, top merges should be high-frequency morphemes
(`qu`, `ae`, `um`, `us`, `est`, common prefixes). If they are, Merge detection
works.

### M2 — Dynamic Vocabulary
Add `AtomVocabulary` class.
- Tracks base atoms + all MergedAtoms
- Supports `tokenize(sequence) → list[atom]` using current vocab
- Supports `add_merge(A, B) → MergedAtom` — updates tokenization greedily
- Stores merge history for introspection

Validation: tokenize a Latin sentence before and after top-10 merges.
Confirm that high-PMI pairs are now single atoms.

### M3 — Multi-Level RelationalLearner
`HierarchicalRelationalLearner` wrapping a list of `RelationalLearner` instances.
- `fit(sequences)`: fits level 0, detects merges, re-tokenizes, fits level 1, ...
  until no merge exceeds threshold or max_levels reached
- `predict_dist(atom, rel, level=0)`: predicts at requested level
- `infer_chain(atom, relations, level=0)`: multi-hop at requested level
- All levels share the same MergedAtom vocabulary

Validation: H@1/H@3/H@10 on Latin corpus at level 0 (char) vs level 1
(morpheme). Level 1 should show higher H@1 (fewer candidates at morpheme level,
predictions sharper).

### M4 — Top-Down Prediction
Extend `ContextBeliefState` to stack across levels.
- `MultiLevelContextBelief`: one `ContextBeliefState` per active level
- `predict_dist_conditioned(rel, level=0)`: blends level-0 prediction with
  top-down signal from level-1 belief (weighted sum or multiplicative)
- `observe(atom)`: updates all levels simultaneously

Validation: on Latin corpus, conditioned H@1 at level 0 should exceed
unconditioned H@1 when higher-level context is informative (e.g., mid-word
position vs. word boundary).

### M5 — Structural Relation Extraction
At level N > 0, extract structural relations from Merge history.
- `left_of(merged_atom) → left_constituent`
- `right_of(merged_atom) → right_constituent`
- `parent_of(atom) → MergedAtom | None`

Run E3-E6 at each level to discover distributional categories of MergedAtoms.
These categories ARE CTKG types. Export them.

Validation: at word level on Latin, E3 categories should correspond roughly to
grammatical categories (noun endings, verb endings, particles). Inspect top-3
atoms per category.

### M6 — CTKG Grounding
Connect `HierarchicalRelationalLearner` output to the CTKG type system.
- Discovered level-0 atoms → CTKG `symbol` type
- Discovered level-1 atoms → CTKG `seq(symbol)` type
- Discovered level-N distributional categories → CTKG concept nodes
- Discovered adjoint pairs → CTKG `adjunction` blocks
- Export discovered structure as `.ctkg` file

Goal: given raw text and no human annotation, the system produces a `.ctkg`
file that captures the compositional structure of the domain. This file can
then be validated against a hand-authored `.ctkg` file for the same domain.

### M7 — Segment + boundary_atoms (DONE)
Added `SegmentedAtom` (n-ary ordered composition), `segment_by_boundary()` in
`MergeDetector`, `boundary_atoms={' '}` for unconditional word splits in text,
and `use_segment=True` in `HierarchicalRelationalLearner`.  Also fixed:
`max_levels` 6→20, large-KG threshold 500→50,000, OOM fix (`vocab_size=2000`
cap in `discover_categories_from_dists`).

Remaining problems after M7 (motivation for M8):
- **K=2 at word level**: sparse word distributions defeat K-means at L1+.
- **Paragraph-swallowing**: adaptive mean-PMI threshold grows unboundedly.
- **Sub-word fragmentation**: PMI threshold mismatches at character level.
- Root cause: batch segmentation is architecturally wrong for this problem.

### M8 — PredictiveCodingHierarchy (online, surprisal-based) — DONE
**Class:** `PredictiveCodingHierarchy` in `relational_pipeline.py` (appended
after line 4125).

Replaces `HierarchicalRelationalLearner` as the primary inference pipeline.
Implements Broca's area architecture: online prediction, simultaneous
multi-level processing, bounded working memory, and surprisal-driven
boundaries.

**Key properties:**
- Online (token-by-token); all levels process in lockstep within each call.
- `max_chunk_size=7` (Miller's 7±2) prevents paragraph-swallowing.
- Adaptive threshold (`mean + k*std` of running surprisal history) replaces
  fixed PMI threshold; self-calibrates per level, no domain tuning needed.
- Cold-start safe: unfitted learners warm up via `update_online()`.
- `from_hrl(hrl)` class method for warm-start from a pre-trained HRL.

**Why this fixes the three M7 failures:**

| Problem | Root cause | Fix |
|---------|-----------|-----|
| K=2 at word level | K-means on sparse distributions | No batch K-means at L1+; word categories emerge from `_atom_bigrams` |
| Paragraph swallowing | Unbounded segment size | `max_chunk_size=7` hard cap |
| Sub-word fragmentation | Miscalibrated PMI threshold | Adaptive threshold self-calibrates |

**Demo:** `python char_latin.py --mode pch` (new default).

### M9 — PCH + Multi-Scale R0-R6 Analysis — DONE

**Motivation:** PCH builds `_atom_counts` at every level but never clusters atoms
into categories.  R0-R6 needs those categories to produce a meaningful CTKG.  M9
closes that gap by running the full R0-R6 pipeline on every active PCH level after
`process_corpus()` completes.

**Sub-phases:**

#### M9a — Track reverse bigrams in PCH  ✓ (planned: one extra `update_online` call)
PCH currently only calls `update_online(prev, 'next', token)`.  Add
`update_online(token, 'prev', prev)` in `_process_level()` so that each level's
`_atom_counts` contains both forward and reverse bigrams.  This doubles the context
richness for E0 clustering at higher levels at negligible cost.

#### M9b — Collect chunk sequences per level  ✓ (planned: log to `_chunk_seqs`)
`RelationalSenseSplitter` needs raw sequences (not just counts) to build conditional
distributions P(next | prev=x).  Add `_chunk_seqs: list[list[list[str]]]` to PCH —
per-level, per-document list of emitted chunk strings.  Log each emitted chunk in
`_emit_buffer()`.

#### M9c — `RelationalLearner.cluster_from_counts()`  ✓ (planned: new method)
After PCH online learning, each level's learner has `_atom_counts` but no E0
categories (`assignment`/`clusters` are absent).  Add
`cluster_from_counts(verbose=False)` to `RelationalLearner` that reconstructs
compound E0 signatures directly from `_atom_counts`, runs `_jsd_cluster()`, and
builds `assignment`, `clusters`, `_trans`, `_nc_cache`, `_wgc_cache`,
`_succ_dists`, `_sim_matrix` — exactly the E0+E1+E3 portion of `fit()` without any
triple scan (O(V×R×V) only).

#### M9d — `PCH.analyse(verbose=True)`  ✓ (planned: new method)
After `process_corpus()`:
1. For each active level L: call `learners[L].cluster_from_counts()`.
2. Run `RelationClusterer`, `SecondOrderGrammar`, `GeometryDetector`,
   `RelationalParadigmDiscoverer`, `RelationalSenseSplitter`,
   `RelationalAlgebra` on that level's fitted learner.
3. Store all results in `self.analyses: list[dict]` indexed by level.
Returns rich multi-scale CTKG structure — word categories at L1, phrase
categories at L2, etc.

#### M9e — Extend `export_ctkg()` with multi-scale categories  ✓ (planned)
If `self.analyses` is populated, emit per-level `concept` blocks into the CTKG
DSL:  atom-category assignments, role clusters, sense nodes, composition rules.
These become first-class CTKG nodes rather than raw chunk strings.

---

### M10 — Cross-Level Constituency Analysis — DONE

**Motivation:** R0-R6 run *within* each level discovers relations among peers
(word follows word, phrase follows phrase).  Running R0-R6 *across* levels on the
constituency structure discovers what character-types occupy word boundaries, what
word-types form what phrase-types, etc.  This is the novel direction not covered
by any prior work.

**What "cross-level triples" means:**

For every chunk in `AtomVocabulary`:
- `MergedAtom(surface, left=A, right=B)`:
  - `(surface, 'left_const',  A)` and `(surface, 'right_const', B)`
  - `(A, 'is_left_of',  surface)` and `(B, 'is_right_of', surface)`
- `SegmentedAtom(surface, parts=[p0,p1,...,pN])`:
  - `(surface, 'part_0', p0)`, …, `(surface, 'part_N', pN)`
  - `(p0, 'at_pos_0_of', surface)`, etc.

A `RelationalLearner` fitted on these triples learns the distributional structure
of constituency: which atom-types are interchangeable as left constituents, which
right-constituent types are predicted by left-constituent type, etc.

**Sub-phases:**

#### M10a — `PCH.analyse_cross_level(verbose=True)`
1. Build cross-level triples from `self.vocab` (MergedAtoms + SegmentedAtoms).
2. Fit a new `RelationalLearner` on these triples.
3. Run R0-R6 on it.
4. Store as `self.cross_level_analysis`.
Key expected discoveries:
- Role clusters: atoms that appear only as L constituents vs. only as R constituents
  vs. both → predicts the "head vs. modifier" distinction without supervision.
- Relational algebra: `left_const ∘ right_const` vs. `right_const ∘ left_const`
  — are left-first or right-first constituency patterns dominant?
- Second-order grammar: which part relations follow other part relations.

#### M10b — Export cross-level structure to CTKG
Emit the cross-level categories and composition rules as functor blocks in the CTKG
DSL, mapping level-L atom types to level-L+1 chunk types.

---

### M11 — Cleanup — DONE

Removed `HierarchicalRelationalLearner`, `MergeDetector`, `MultiLevelContextBelief`,
`extract_structural_relations`, and `_sequences_to_next_triples` from
`relational_pipeline.py` (~668 lines deleted).
Removed `--mode hrl` from `char_latin.py`; only `--mode rl` and `--mode pch` remain.
`export_ctkg()` and `label_merged_atoms()` kept (used by PCH via duck-typed shim).
`SegmentedAtom` kept (used by AtomVocabulary and PCH._emit_buffer).
File shrunk from 5193 → ~4525 lines.
- Archive `sequence_pipeline.py` (already marked outdated; confirm no new callers).

---

### M12 — Type-Abstracted Clustering — DONE

**Problem**: Levels 4-9 collapse to K=1 because each phrase/clause is unique or
near-unique — there is no statistical basis for clustering raw surface strings
that appear only once.

**Root cause (the parameter-sharing gap)**: neural networks generalize across
sparse surface forms by placing similar tokens near each other in a shared weight
space.  PCH has no such sharing: "magnum imperium" and "parvum regnum" are
different strings even though both are adj+noun NPs.

**Fix — compositional type abstraction**: represent each level-N chunk by the
*type-tuple* of its level-(N-1) constituents rather than by its surface string.
If level-3 clusters have labelled word categories, then any adj+noun phrase maps
to `(adj_cat, noun_cat)` regardless of the specific words.  Two chunks are in
the same category iff their constituent type-sequences match.  This is O(V×depth)
to compute and produces deterministic, linguistically grounded categories.

**Implementation**:
- `RelationalLearner._rebuild_caches_from_assignment(assignment)` — extract the
  E1/E3 cache-building block so it can be reused after any assignment method.
- `RelationalLearner.cluster_from_type_abstraction(vocab, lower_assignment)` —
  new method: look up each chunk's constituents, map to lower-level type IDs,
  group by type-tuple, call `_rebuild_caches_from_assignment`.
- `PCH.analyse()` — after raw `cluster_from_counts()`, if K≤1 and level≥1 and
  lower-level assignment exists, call `cluster_from_type_abstraction` as fallback.

**Expected result**: levels 4-9 gain meaningful categories (e.g. K≈50-200 at L4,
K≈20-80 at L5) based on constituency type structure rather than repeated surface
strings.  The same corpus that gave K=1 everywhere above L3 should now produce
structured phrase/clause/sentence categories.

---

### M13 — Multi-Level Belief Cascade — DONE

**Problem**: no long-range context.  PCH's working memory is bounded at 7 tokens.
To predict the next word in a long document you need to know the topic, characters,
narrative arc — none of which fit in 7 tokens.

**Why attention is the wrong solution**: O(n²) cost, violates domain-agnosticism
(assumes 1D sequence), opaque (no explicit state).

**Solution — hierarchical Bayes filter (Rao & Ballard 1999 predictive coding)**:
After `analyse()` has built cluster assignments at every level, instantiate one
`ContextBeliefState` per active level.  The belief at level N is a K_N-dimensional
probability vector over chunk categories, updated O(K_N²) per emitted chunk.

- **Effective context window**: infinite (the state summarises ALL past tokens,
  compressed to K categories, not stored explicitly).
- **Domain-agnostic**: `ContextBeliefState` uses the `_trans` matrix built from
  whatever relations exist at that level — 'next'/'prev' for sequences, H/V for
  images, graph adjacency for knowledge graphs.  No architectural change needed.
- **Top-down modulation**: the belief at level N+1 (phrase/clause) constrains
  level N (word): when the higher belief is high-entropy (uncertain), LOWER the
  boundary threshold at level N (create more boundaries to resolve uncertainty).
  When high-certainty, RAISE the threshold (fewer spurious cuts).

**Two-pass design** (avoids chicken-and-egg with clustering):
1. `process_corpus()` — cold start, collects statistics
2. `analyse()` — builds categories; `init_beliefs()` — creates belief states
3. `process_corpus()` (second pass, frozen model) — boundaries guided by beliefs;
   categories at all levels improve
4. `analyse()` again — re-run with better segmentation

**Implementation**:
- `PCH.__init__`: add `_beliefs: list[ContextBeliefState | None]` (all None).
- `PCH.init_beliefs()`: after `analyse()`, instantiate one CBS per active level.
- `PCH._emit_buffer()`: `_beliefs[level].observe(chunk); .transition(rel)`.
- `PCH._effective_threshold()`: subtract `top_down_weight * base * uncertainty`
  where `uncertainty = belief.entropy() / log2(K)`.  High entropy above → lower
  threshold → more boundaries → resolve uncertainty.
- `PCH.reprocess(sequences)`: frozen second pass (no `update_online`) using
  active beliefs.

---

---

## M14 — Compression Pass — DONE

**Problem**: The CTKG export grows *linearly* with data — 167K concept instances for
12 books of Latin, producing a 58 MB file from 535 KB of raw text.  This is
anti-compression.  The system accumulates every MergedAtom/SegmentedAtom as an
explicit entry rather than discarding it once its *type* has been assigned.

**Information-theoretic diagnosis**: the types + transition matrices already ARE the
compressed representation.  Approximate storage for 12 books of Latin:
- Type assignments (surface → type_id, all levels): ~200 KB
- Transition matrices T[level][rel][K×K], all levels: ~500 KB
- Type-abstraction rules (tuple → cluster_id): ~50 KB
- Relational algebra laws: < 1 KB
- **Total compressed knowledge: ~750 KB** — versus 58 MB CTKG export.

For the 4 GB polymath goal: 100 domains × ~1 MB compressed = ~100 MB for expert-level
knowledge across all major human domains.

**Root cause**: `vocab._merges` and `vocab._segments` grow forever; `_atom_counts`
is kept post-analysis; `export_ctkg()` exports every instance.

**Fix — two components**:

1. **`PCH.compress()`**: post-analysis, delete concept instances; keep only the type
   system.  Specifically clear `vocab._merges`, `vocab._segments`, `vocab._by_surface`,
   `vocab._pair_to_surface`; clear `_atom_counts` and `_atom_totals` on every learner.
   After this call, the object is a *type-only model* — it can classify new tokens and
   predict distributions, but cannot produce a full CTKG export.

2. **`PCH.save_compressed(path)` / `PCH.load_compressed(path)`**: serialise only the
   minimal type system (assignment dicts, transition matrices, K values, type-abstraction
   rules) to a compact binary format (pickle or msgpack).  Target: ≤ 1 MB for 12 books.

3. **Proper CTKG type annotations**: upgrade `export_ctkg()` so level-N types use the
   CTKG universal type constructors — `seq(type_L{N-1}_*)` for levels ≥ 1 (currently
   all types are emitted as plain `symbol`).

**CTKG features activated**: universal type constructors (`seq`, `tagged`), `Interface`
blocks per level (exported so a functor can map into them later).

**Implementation**:
- `PCH.compress()`: clear vocab structures + learner raw counts; set `_compressed=True`.
- `PCH.save_compressed(path)`: pickle `{levels: [{_K, assignment, clusters, _trans,
  _nc_cache, _wgc_cache, _succ_dists, _sim_matrix, _rel_unigram}], type_rules: [...]}`.
- `PCH.load_compressed(path)`: restore above; rebuild `learners` from saved dicts.
- `export_ctkg()`: add `seq(type_L{N-1}_*)` as the type body for levels ≥ 1.

---

### M15 — Type-Level Inference Engine — DONE

**Problem**: `predict_target_dist()` and `infer_chain()` are implemented but never
called from PCH.  The system builds beliefs and categories but never *uses* them for
prediction.  It is a knowledge-discovery engine without an inference engine.

**Goal**: close the loop — the compressed type model should support:
1. Next-token prediction: O(K²) per step (not O(V²)).
2. Multi-hop reasoning: `infer_chain()` for "given we're in a verb phrase, what
   follows 3 steps later?"
3. Perplexity evaluation: measure type-level prediction quality as a compression metric.

**CTKG features activated**: `predict_target_dist()` (first real use), `infer_chain()`
(first real use from PCH context).

**Implementation**:
- `PCH.predict_next(tokens) → dict[str, float]`: run tokens through process() (frozen),
  then call `_beliefs[0].predict_target_dist('next')` for the next-token distribution.
- `PCH.predict_type(level, rel) → dict[int, float]`: type-level marginal.
- `PCH.evaluate_perplexity(sequences) → float`: per-token cross-entropy under the
  type model; compare vs. flat n-gram baseline.
- `PCH.reason_chain(start_token, relations) → dict`: wraps `infer_chain()` using the
  belief-active learner at the appropriate level.

---

### M16 — Gated DeltaNet Belief Update — DONE

**Problem**: `ContextBeliefState.decay()` is implemented but never called.  Belief
updates are non-gated: every observation applies a fixed 97% sharp concentration
regardless of how expected or surprising the token was.  This means:
- Highly surprising tokens (boundary events) don't reset the context sufficiently.
- Highly expected tokens don't preserve the context they should confirm.
- Document boundaries don't reset beliefs.

**Why DeltaNet is the right inspiration** (Yang et al. 2025, Gated DeltaNet):
The delta rule performs a selective **erase-then-write**: `S_t = α_t S_{t-1} + write`.
The gate `α_t` is data-dependent.  For belief states the analogous operation is:

```
alpha = exp(−beta × normalised_surprisal)   # ∈ (0,1]
B_t = decay(alpha) then observe(atom)       # erase old context proportionally
```

- `alpha = 1` (zero surprisal, completely expected): no decay; belief gently updated.
- `alpha ≈ 0` (high surprisal, boundary event): strong decay toward uniform *before*
  the new observation; old context erased, fresh inference.
- Document boundary: explicit `decay(rate=0)` → full reset.

Normalised surprisal: `surp / adaptive_threshold` so the gate is calibrated to each
level's characteristic surprisal scale.

**CTKG features activated**: `ContextBeliefState.decay()` (first active use).

**Implementation**:
- `ContextBeliefState.observe_gated(atom, surprisal, threshold, beta=1.0)`:
  `alpha = exp(−beta × surp/threshold)`; call `self.decay(rate=alpha)` then
  `self.observe(atom)`.
- `PCH.__init__`: add `_emit_surp: list[float] = [0.0] * n_levels`.
- `PCH._process_level()`: store `self._emit_surp[level] = surp` before boundary-
  triggered `_emit_buffer()`.
- `PCH._emit_buffer()`: replace `belief.observe(chunk)` with
  `belief.observe_gated(chunk, self._emit_surp[level], self._effective_threshold(level))`.
- `PCH._reset_buffers()`: after flushing each level, call `belief.decay(rate=0.0)`
  (full reset at document boundary — equivalent to hidden-state reset between episodes).

---

### M17 — Cross-Domain Functors + Sheaf Consistency — PLANNED

**Problem**: PCH operates on one domain at a time.  There is no way to say "the
category I discovered for Latin nominatives IS the same category as English subjects."
The CTKG's `Functor`, `Adjunction`, `Interface`, and `sheaf_check()` machinery is
entirely unused.

**Goal**: two PCH instances (e.g. Latin + English, or text + mathematics) should be
linkable by a functor that maps type categories from one domain to the other.  Shared
structure costs nothing — the same transition matrices apply.

**The category-theoretic argument**: if Latin L3 type 7 and English L3 type 12 have
the same *relational profile* (same transitions under 'next'/'prev'), they ARE the
same abstract category — the functor witnesses this identity.  Cross-domain
generalisation is free once the functor is built.

**CTKG features activated**: `Functor`, `Adjunction` (parse↔generate), `Interface`
blocks, `sheaf_check()`, `sheaf_merge()`.

**Implementation**:
- `PCH.build_interface() → str`: emit a `.ctkg`-format `interface` block listing all
  discovered type IDs as exported symbols.  Enables functor targets.
- `PCH.build_functor(other_pch, sim_threshold=0.7) → dict[int, int]`: align type
  categories between two PCH instances by comparing their `_succ_dists` (JSD
  similarity) and `_trans` matrices.  Return `{self_type_id: other_type_id}` mapping.
- Emit `.ctkg` `functor` blocks; wire into `ctkg.parser.merge()` + `sheaf_merge()`.
- `PCH.build_adjunction(level) → Adjunction`: discover left/right inverse pairs at
  a level (e.g. 'next'/'prev' are adjoint; found automatically via R3 algebra).
- Benchmark: functor-transferred prediction (use Latin transitions for English text
  via functor) should beat uninformed baseline.

---

### M18 — Causal Reasoning, MasteryState, Full CTKG Closure — PLANNED

**Problem**: `d_separated()`, `intervene()`, `MasteryState`, `transfer_probability`,
and probabilistic prerequisites are all implemented in `ctkg/graph.py` but completely
disconnected from the PCH learning pipeline.

**Goal**: every implemented feature in the CTKG toolkit should be exercised.

**Sub-tasks**:

**M18a — MasteryState integration**: after each `analyse()` pass, instantiate a
`MasteryState` for the PCH's CTKG.  Track per-type mastery: a type is "mastered" when
its `_trans` matrix has converged (Frobenius norm of update < ε).  Use
`mastery_state.frontier()` to prioritise which domains to train next.

**M18b — Probabilistic prerequisites (`transfer_probability`)**: the CTKG prerequisite
edges emitted by `export_ctkg()` currently have implicit probability 1.0.  Replace
with empirically estimated transfer probabilities: P(type_L{N+1}_k | type_LN_j) from
the type-abstraction mapping.  These are the learned "soft prerequisite" weights.

**M18c — d-separation queries**: build a PCH-level CTKG (types as nodes, transitions
as edges) and run `d_separated(type_A, type_B, given={type_C})` to ask "are these
two type categories conditionally independent given a third?" Useful for pruning
redundant types and discovering irrelevant features.

**M18d — Causal intervention (`intervene()`)**: for a given document, simulate "what
if the topic were different?" by intervening on higher-level type beliefs and measuring
the effect on lower-level predictions.  Implements do-calculus at the type level.

**M18e — Universal type constructors**: upgrade type annotations so every level uses
the correct CTKG type constructor:
- L0: `symbol` (base sensory atoms)
- L1: `seq(symbol, ordered)` (CV bigrams)
- L2: `seq(type_L1_*, metric)` (morpheme fragments — metric because JSD distance defined)
- L3: `seq(type_L2_*, ordered)` (word-level, head-final)
- L4+: `tuple(type_L3_*, ...)` (phrase-level — n-ary unordered in free-word-order Latin)

**CTKG features fully activated after M18**:
`d_separated()`, `intervene()`, `MasteryState`, `mastery_state.frontier()`,
`transfer_probability`, `concept_entropy()`, `mutual_information()`,
`conditional_entropy()`, `information_flow()`, `sym_*` constructors in type bodies.

---

## Success Criteria

| Criterion | Test |
|-----------|------|
| Merge detects real units | Top merges on Latin = high-frequency morphemes |
| Multi-level improves prediction | H@1 higher at morpheme level than char level |
| Top-down helps | Conditioned H@1 > unconditioned H@1 mid-word |
| Structural relations emerge | Level-1 E3 categories ≈ grammatical categories |
| CTKG export | Discovered `.ctkg` overlaps with hand-authored arithmetic.ctkg |
| Domain-general | Same code, same algorithm on: Latin text, image patches, KB triples |

The final test: apply `HierarchicalRelationalLearner` to Danganronpa dialogue
transcripts (word/utterance level). The discovered CTKG structure should
reflect narrative-level patterns: who speaks, when, what follows contradiction,
what precedes revelation. If it does, the architecture is credible for the AGI
goal.

---

## What Does Not Change

The RelationalLearner's core mechanisms are correct and stay:
- `_atom_bigrams` — the statistical foundation
- `_rel_unigram` — OOV fallback
- `infer_chain()` — multi-hop belief propagation
- `update_online()` — O(1) episodic learning
- `ContextBeliefState` — Bayes filter (extended to multi-level in M4)
- E0-E6 pipeline — runs at each level after Merge

Merge does not replace these. It adds the missing structural layer that allows
them to operate at any scale the data contains.
