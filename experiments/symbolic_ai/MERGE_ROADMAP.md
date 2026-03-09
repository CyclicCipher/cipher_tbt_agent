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
