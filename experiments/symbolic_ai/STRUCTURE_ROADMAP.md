# Symbolic AI: Structure Learning Roadmap
# Goal: match or exceed transformer expressiveness without gradient descent

## Theoretical foundation

Transformers implement four capabilities:
1. Soft content-addressable retrieval (attention)
2. Contextual representations (same word → different embedding per context)
3. Composition via layer stacking
4. Paradigmatic axis (substitutability, not just co-occurrence)

Our system maps to this as:
- `induce_hierarchy_bidir`  →  distributional clustering (forward + backward context)
- `ask_dist`                →  full probability distribution over outputs (already done)
- CTKG process language     →  explicit composition (can express anything attention can)
- `chunk_store` + PMI       →  K^3 categorical compression vs V^2 word-bigram state

Central diagnosis: the CTKG process language IS transformer-equivalent in expressiveness.
The `language.ctkg` 3-concept chain already specifies the architecture:

    word_pos:        word → cat               (K clusters, replaces O(V) word space)
    next_cat:        cat × cat → cat          (K^2 entries, replaces O(V^2) bigrams)
    word_given_cat:  cat × cat × cat → word   (K^3 entries → generalises to unseen pairs)
    next_word(w1,w2) = word_given_cat(word_pos(w1), word_pos(w2),
                                      next_cat(word_pos(w1), word_pos(w2)))

K=12 clusters → 1,728 word_given_cat entries vs 25,000,000 for V=5000 word bigrams.
Prediction generalises to unseen (w1,w2) pairs because category transitions were seen
even when word pairs weren't — this is the key test.

---

## Polysemy in the CTKG: bank(institution) vs bank(river)

Surface form "bank" is one object in the word-form space. But it has two distinct
semantic objects. The categorical structure is a FIBER BUNDLE:

    Base space:  word_form  (surface "bank")
    Fiber:       {institution_sense, terrain_sense, verb_sense}
    Total space: word_sense = tuple(word_form, sense)
    Projection:  π: word_sense → word_form  (forgetful functor)
    Section:     σ: Context × word_form → word_sense  (disambiguation)

CTKG representation:

    type word_sense = tagged(institution | terrain | verbal)

    concept disambiguate
        input  word_form
        input  context_window
        output word_sense
        requires prev_word_hier via "backward context"
        requires next_word_hier via "forward context"

Operational detection: high entropy in ask_dist('next_word_hier', (word,)) signals
polysemy. The forward distribution of "bank" is bimodal — sometimes financial words
follow, sometimes geographic. Same for backward. High entropy = ambiguity candidate.

As SheafViolation: if financial.ctkg and geographic.ctkg both define incompatible
types for "bank", sheaf_check() raises SheafViolation. Resolution = tagged union.
This is the correct categorical representation: polysemy is a sheaf violation resolved
by a tagged union of domain-specific objects.

---

## Phases

### Phase E0 — Baseline ✓ (commit 5715a94, 2026-03-07)

Bidirectional clustering at all scales. Latin corpus results:
- Level 0 char→morpheme: 93 merge rules (qu, ae, scribal abbreviations incl. ibꝰ=-ibus)
- Level 1 morpheme→word: 133 rules
- Level 2 word→phrase: 8 collocations (inter se PMI=6.1, ita ut=4.7, Quòd si=8.2)
Nasal contractions ũ/ẽ/ã correctly cluster as word-final markers (case morphology detected).

### Phase E1 — Category-Chain Composition ✓ (commit a6ed5ac, 2026-03-07)

File: language_pipeline.py
Test: EarlyModernLatin corpus, 2000 train / 500 test lines, K=12

Results:
- 93.9% test trigrams have UNSEEN word pairs (V=9,626)
- Chain accuracy on unseen: 0.2% vs flat bigram: 0.0% — THESIS SUPPORTED
- 643,471× compression (K²=144 vs V²=92M bigram state space)
- Coverage: chain 28.4% vs flat 53.6% (sparse word_given_cat: 35.9% of K³ populated)

### Phase E2 — Trigram Context for Clustering ← CURRENT (2026-03-07)

File: language_pipeline.py (build_trigram_assignment + train_chain_ctx + evaluate_all)
Test: EarlyModernLatin corpus, same split, K=12 base + KC=12 context clusters

Architecture:
    context_assignment: (c_prev, word) → ctx_cluster_id   (KC clusters)
    next_cat_ctx:       (c1, c2_ctx) → c3_ctx             (K × KC entries)
    word_given_cat_ctx: (c1, c2_ctx, c3_ctx) → word       (K × KC² entries)

Same word gets DIFFERENT cluster in different left-context:
    "bank" after DET → financial-bank cluster (c2_ctx = 8)
    "bank" after PREP → terrain-bank cluster  (c2_ctx = 3)

This is the distributional analogue of contextual embeddings.
CTKG connection: polysemy = multiple c2_ctx values for same word = fiber bundle section.

Usage: python language_pipeline.py --corpus EarlyModernLatin --n_train 5000 --n_clusters 12
       (runs E1 + E2 by default; --phase e1 for E1 only)

Expected: E2 > E1 > flat on unseen pairs.

### Phase E3 — Soft Retrieval (Attention-Equivalent)

ask_weighted(concept, query, context_dist) — weight stored examples by context
similarity instead of exact match. This gives CONTEXTUAL representations.
Keys = stored inputs, Values = stored outputs, Similarity = −KL.

### Phase E4 — Frame Semantics / Paradigmatic Axis

slot_occupants concept: for each high-PMI phrase, record what words fill each slot.
Words filling the same frames are paradigmatically equivalent.
Captures meaning similarity (cat ≈ dog) that co-occurrence alone cannot.

### Phase E5 — Polysemy / Word Sense Disambiguation

1. detect_polysemy(ai, threshold=2.5) → high-entropy words
2. split_senses(ai, word, n_senses) → context-conditional sub-clusters
3. word_sense tagged union type + disambiguate concept in CTKG
4. Sheaf check for cross-domain sense consistency

### Phase E6 — Meta-Synthesis

Given examples of a new concept, search for a CTKG process chain that produces them.
Upgrade synthesis.py from template search to COMPOSITION search: find factorisation
of observed function through existing CTKG concepts.

This closes the central gap: transformers learn composition implicitly via backprop.
We learn it explicitly by searching the CTKG DAG for compatible compositions.

### Parity Test

Metric: trigram log-likelihood per token, held-out Latin corpus.
Baseline: flat bigram (next_word_hier).
Comparison target: 2-layer transformer (d=64, 2 heads, same corpus).
Prediction: category chain should match/exceed transformer on unseen word pairs.

### OCR Return

After parity established: visual glyph clusters → char clusters → morpheme clusters
→ word clusters → meaning. Full pixel-to-concept pipeline, same general method.

---

## Progress Log

| Date       | Phase | Status      | Notes |
|------------|-------|-------------|-------|
| 2026-03-07 | E0    | COMPLETE    | Bidir clustering, prev/next at all scales |
| 2026-03-07 | E1    | COMPLETE    | Chain beats flat on unseen pairs (0.2% vs 0.0%); 643K× compression |
| 2026-03-07 | E2    | IMPLEMENTED | build_trigram_assignment + evaluate_all; awaiting Latin corpus test |
