# OCR → CTKG Roadmap

**Goal:** Produce `ocr.ctkg` + `ocr_knowledge.json` that any new SymbolicAI can load
to immediately read characters from pixel images — no retraining, no external OCR library.

---

## Where We Are

The symbolic AI can discover language structure and visual structure from raw data:

- **`discover_test.py` (Phase O):** Word-level distributional clustering → POS categories.
  Same `discover_categories_from_dists()` machinery throughout.
- **`discover_chars.py` (Phase S.1, DONE):** Character-level version.
  Feeds raw GT text → `ai.teach('next_char', ...)` → `ai.induce_hierarchy()` → discovers
  phonological/morphological char categories without labels.
  Smoke test (200 lines): 12 clusters, Q/q correctly isolated (Latin Q-before-U pattern).
- **`train_ocr.py` (Phase S.3b, DONE):** Forced alignment training on GT4HistOCR images.
  External GlyphReader centroid approach. CER baseline ~0.95 (PIL font mismatch).
- **`ocr_test.py`:** Evaluation harness. `--lang-prior` flag added (superseded by this roadmap).

**Transfer mechanism (already built):**
`engine.save_checkpoint()` / `engine.load_checkpoint()` (Phase H) saves/restores the
full symbolic AI state including ExampleStores. The `glyph_reads_as` ExampleStore IS
the lookup table — no dict literals needed. `ai.ask('glyph_reads_as', (cid,))` returns
the most frequent char for that glyph cluster via exact-match ExampleStore lookup.

---

## Destination

```
ocr.ctkg            ←  structural skeleton (types, concepts, prerequisites, interface)
ocr_knowledge.json  ←  learned content (ExampleStores: next_char, next_glyph, glyph_reads_as)
```

A new AI loads both:
```python
graph = merge(parse_file('arithmetic.ctkg'), parse_file('ocr.ctkg'))
ai = SymbolicAI(graph)
ai.load_checkpoint('ocr_knowledge.json')

# Now the AI can read:
glyph_cluster_id = ...  # from image patch analysis
char = ai.ask('glyph_reads_as', (glyph_cluster_id,))  # → ('a',)
```

No GT4HistOCR data required at inference time. No external OCR. No training loop.
The knowledge is in the checkpoint; the structure is in the CTKG.

---

## Roadmap

### Phase S.1 — discover_chars.py (COMPLETE)

Three-phase pipeline inside the symbolic AI:

**Phase 1:** Stream GT text characters → `ai.teach('next_char', (ch,), (ch_next,))`
            → `ai.induce_hierarchy('next_char', n_clusters=12)` → char categories

**Phase 2:** Extract image patch hashes → `ai.teach('next_glyph', (h,), (h_next,))`
            → `ai.induce_hierarchy('next_glyph', n_clusters=64)` → glyph categories

**Phase 3:** Forced alignment (image, GT text) pairs → `ai.teach('glyph_reads_as', (cid,), (ch,))`
            → `ai.freq_consolidate('glyph_reads_as')` → glyph→char probability table

**Status:** Smoke test PASS (200 lines). Full corpus run pending.

---

### Phase S.2 — Checkpoint integration (NEXT)

Add `--save` flag to `discover_chars.py` to save checkpoint after each phase.

```bash
python discover_chars.py --corpus EarlyModernLatin --align --save ocr_knowledge.json
```

Checkpoint contains: all ExampleStores + CTKG graph additions.
A new AI loads it and immediately has OCR capability.

Also add `--load` flag to skip Phase 1/2 if already trained, go straight to Phase 3.

---

### Phase S.3 — ocr.ctkg domain file

A `.ctkg` file declaring the OCR domain structure:

```
domain: ocr

type char        = symbol ordered
type char_cat    = symbol        # discovered category ID (0..K-1)
type glyph_hash  = symbol        # quantized patch hash string
type glyph_cat   = symbol        # discovered glyph cluster ID (0..K-1)

concept next_char
    description: predict next character from current character
    input:  char
    output: char
    tier: theorem

concept next_glyph
    description: predict next glyph patch from current glyph patch
    input:  glyph_hash
    output: glyph_hash
    tier: theorem

concept glyph_reads_as
    description: map visual glyph category to character
    input:  glyph_cat
    output: char
    requires next_char
    requires next_glyph
    tier: theorem

interface
    exports: glyph_reads_as
    exports: next_char
```

Any domain that needs OCR capability adds:
```
    requires glyph_reads_as via "ocr"
```

---

### Phase S.4 — ocr_eval.py (evaluation)

Load a fresh AI + checkpoint; test on held-out lines:

```bash
python ocr_eval.py --checkpoint ocr_knowledge.json --corpus EarlyModernLatin --n 200
```

Reports:
- Character Error Rate (CER) vs baseline
- Glyph cluster coverage (what fraction of test patches have a known cluster)
- Top-5 most confused char pairs

---

### Phase S.5 — Full corpus run

```bash
python discover_chars.py --corpus EarlyModernLatin --align --save ocr_knowledge.json
python ocr_eval.py --checkpoint ocr_knowledge.json
```

Expected results (after full corpus training):
- Meaningful char categories: vowel cluster, consonant clusters, Q/q, punctuation, word-boundary
- Glyph clusters: each visual glyph type assigned to a cluster
- CER: lower than 0.95 baseline (exact target depends on alignment quality)

---

### Phase S.6 — Consolidation (future)

After ExampleStore is populated and validated, `ai.consolidate('glyph_reads_as')`
should synthesize a portable process expression:

```
process: lookup(glyph_reads_as, first(X))
```

embedded inside a higher-level `read_glyph` concept. The process is then written
directly into `ocr.ctkg` as a `process:` block. At that point, any AI loading
`ocr.ctkg` executes the rule immediately — no separate checkpoint required.

This requires no new interpreter primitives (lookup already works via engine_ask).
It requires the Synthesizer to emit a `lookup(concept, arg)` pattern when
the ExampleStore is already freq_consolidated (i.e., is a deterministic table).

---

## Design principles

1. **No external OCR.** All reading happens through the symbolic AI's own machinery.
2. **Structure from distributions.** Char categories and glyph categories are
   discovered without labels using `discover_categories_from_dists()`.
3. **Checkpoint = knowledge.** The ExampleStore IS the lookup table.
   `lookup()` in the process language calls `engine.ask()` which consults it.
4. **`ocr.ctkg` = structure.** Types, concept names, prerequisites, interface.
   Any new AI loads it to know what OCR concepts exist and how they relate.
5. **Transfer = checkpoint + ctkg.** One training run → `ocr_knowledge.json` →
   any number of new AIs inherit OCR capability instantly.
6. **Future: consolidation → self-contained.** Once the Synthesizer can emit
   `lookup(concept, arg)` patterns, the process rule moves into the CTKG directly.
