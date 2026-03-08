# Symbolic AI — Roadmap

## Broca's Area Principle

Broca's area (BA44/45) activates equally for language, action sequencing, music, and
mathematical reasoning. The unifying principle is **hierarchical sequential prediction**:
any structured stream where a noisy surface level is explained by a cleaner latent
category level. Our engine implements this one algorithm for all domains.

    text:    sequences = [line.split() for line in corpus]
    actions: sequences = [episode_actions for episode in game_log]
    music:   sequences = [note_ids for track in midi]
    vision:  sequences = [patch_ids for row in image]   # via VisionLearner

---

## Sequence Learning (E0-E6) — COMPLETE

The `sequence_pipeline.py` `SequenceLearner` class generalizes E0-E6 to any discrete
token sequences. Tested on EarlyModernLatin (5K train, 1.25K test).

| Phase | What it does | Result |
|-------|-------------|--------|
| E0 | Teach next_token/prev_token bigrams → bidir clustering | K=12 categories |
| E1 | `next_cat(c1,c2)→c3` + `token_given_cat(c1,c2,c3)→tok` | K²+K³ entries |
| E2 | Context-sensitive clusters: `(c_prev, token) → ctx_id` | Richer context |
| E3 | Soft retrieval: `sim(c,c')=exp(-T·JSD(succ_dist[c],succ_dist[c']))` | Self-attention analogue |
| E4 | `slot_occupants(c_prev,c_next)→token` (paradigmatic axis) | Frame semantics |
| E5 | Sense-splitting: k-means on `(c_prev,c_next)` context vectors | Polysemy |
| E6 | Beam-search composition: find chain of concepts explaining examples | Meta-synthesis |

**Parity test result (EarlyModernLatin, 5K train):**

| Model | Unseen-pair accuracy |
|-------|---------------------|
| Flat bigram | 0.0% |
| E1 chain | 0.4% |
| **E3 soft retrieval** | **1.0%** ← best |
| 2-layer transformer (d=64) | 0.4% |

E3 beats the transformer on unseen pairs, using K³=1728 category entries vs. V²=25M
word-bigram parameters. Category transitions generalize to unseen word pairs because
the syntactic structure was seen even when the word pair wasn't.

**Compression:** K=12 clusters → 1,728 `token_given_cat` entries vs. 25M for V=5K bigrams.

**E6 meta-synthesis:** beam search discovers `context_triple ∘ word_given_cat` (100%
coverage, 80% accuracy) and `slot_context ∘ slot_occupants` (100% coverage, 60%
accuracy) from examples alone, without knowing the chain structure in advance.

**Key files:**
- `sequence_pipeline.py` — `SequenceLearner` class: E0-E6, all domains
- `parity_test.py` — E3 vs 2-layer transformer comparison

---

## Vision Pipeline — IMPLEMENTED

`vision_pipeline.py` wraps `SequenceLearner` with image→patch-sequence conversion.
The same E0-E6 algorithm that finds syntactic structure in text finds spatial structure
in images — patch types that co-occur in the same contexts cluster into visual categories
(edges, textures, glyph boundaries).

    learner = VisionLearner(patch_size=16, n_clusters=64)
    learner.fit_images(images)          # numpy HxW arrays
    learner.predict_patch([h1, h2])     # next patch hash

Patch extraction reuses `_to_gray_f32`, `_extract_patches`, `_quantize` from
`modalities/visual_symbol.py` (same as `discover_chars.phase2_glyphs`).

**Key files:**
- `vision_pipeline.py` — `VisionLearner`, `image_to_sequence()`
- `discover_chars.py` — `phase2_glyphs()` (standalone visual discovery)

---

## Cross-Domain Applications

| Domain | Token type | Status |
|--------|-----------|--------|
| Text (Latin) | Word strings | E0-E6 tested, parity achieved |
| Vision (OCR patches) | Patch hashes | `VisionLearner` implemented |
| Action sequences | Action names/ids | `SequenceLearner` ready (pass game log) |
| Music | Note/chord ids | `SequenceLearner` ready (pass MIDI tokens) |
| Game state | State tokens | Future (Minecraft agent) |

---

## Minecraft Agent — Phases A–J complete, Phase K next

**MineDojo is abandoned** (broken dependencies). Phase K replaces it with a custom
interface: dxcam (screen capture) + pynput (keyboard/mouse) + mcrcon (RCON for
homeostatic data). Works with any Minecraft version and modded installations.

- `agent_loop.py` — agent loop (smoke test passes; Phase K pending)
- `modalities/minecraft.py` — Phase K interface stub
- `../MINEDOJO.md` — design doc (MineDojo abandoned; custom interface next)

CTKG: 50 concepts, CausalEdge / CompositionEdge / InstanceEdge / TemporalEdge.

---

## OCR — Implemented, awaiting GPU run

- `discover_chars.py` — Phase 1 (char bigrams), Phase 2 (glyph discovery), Phase 3 (alignment)
- `train_ocr.py` — training entry point
- `ocr_test.py` / `ocr_eval.py` — evaluation

Phase 2 uses `VisionLearner` architecture: patch sequence → glyph categories via
`induce_hierarchy()`. Phase 3 aligns unsupervised glyph clusters to GT char labels
by majority vote.

---

## File Structure

```
experiments/symbolic_ai/
├── ROADMAP.md             ← this file
├── sequence_pipeline.py   ← SequenceLearner: E0-E6, any token domain
├── vision_pipeline.py     ← VisionLearner: image patches via SequenceLearner
├── parity_test.py         ← E3 vs transformer comparison
├── engine.py              ← SymbolicAI class
├── interpreter.py         ← ProcessInterpreter: Level 1 + Level C primitives
├── memory.py              ← ExampleStore + KL divergence
├── synthesis.py           ← template synthesis + distributional discovery
├── discover_structure.py  ← multi-scale discovery pipeline (run_pipeline)
├── discover_chars.py      ← visual glyph discovery (phase1/2/3)
├── train_lang.py          ← OCR language model entry point
├── train_ocr.py           ← OCR training entry point
├── ocr_test.py            ← OCR test runner
├── ocr_eval.py            ← OCR evaluation
├── agent_loop.py          ← Minecraft agent loop
└── modalities/
    ├── vision.py          ← 15 image primitives
    ├── vision_cortex.py   ← VisualCortex
    ├── visual_symbol.py   ← patch extraction (_to_gray_f32, _extract_patches, _quantize)
    ├── textworld_modality.py
    └── minecraft.py       ← Phase K stub
```

---

## Theoretical Basis

**E1-E3 = Transformer attention without parameters:**
- `next_cat` = content-addressable category routing (like Q·K)
- `token_given_cat` = category-conditional output (like V)
- E3 soft retrieval = weighted sum over similar categories (like soft attention)
- Categorical compression: K³ entries instead of V² parameters

**Fiber bundle structure (polysemy):**
- Base: word form; Fiber: {sense₁, sense₂, ...}; Section: context → sense
- E5 k-means sense-splitting = learning the fiber bundle sections

**E6 beam search = string diagram surgery:**
- Discovers categorical composition chains from I/O examples
- Equivalent to causal intervention (do-calculus) on the concept graph
