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

## Relational Learning Pipeline — IN PROGRESS

The relational pipeline generalizes E0-E3 from sequences (1D, linearized) to arbitrary
relational structures: knowledge graphs, 2D images, 3D point clouds, hyperbolic trees,
graphs. No linearization. Relations are first-class citizens.

**Key file:** `relational_pipeline.py` — `RelationalLearner`, `Image2DRelationalLearner`

**Compound bigram trick:** Each triple `(a, r, b)` becomes `rel_next(a,) → (f'{r}:{b}',)`.
An atom's distributional signature encodes ALL its relational neighbors simultaneously.
`induce_hierarchy_bidir()` then clusters atoms by these compound signatures.

**E3 hybrid keys:** `_ask_soft((str(c_src), relation_name))` — integer keys get soft
similarity from sim_matrix; relation-name strings get exact match. No special casing:
`_ask_soft` already handles mixed key types.

---

### Level 1 — Evaluation: Sequential vs Relational [ ]

Add `--mode relational` to `unsupervised_cats.py`. Run both approaches on the same
cat photo dataset. Report Purity / NMI / within-between ratio for both.

Expected advantage of relational: 2D structure is preserved. H/V/D1/D2 neighbors are
distinct relations, not collapsed into a single "next" relation. Patches that play the
same spatial role across images (e.g., "border between fur and background") cluster
correctly regardless of orientation.

---

### Level 2 — Relation Discovery [ ]

Currently relations (H, V, D1, D2) are hand-specified. Level 2 clusters relations the
same way Level 1 clusters atoms — by their distributional behavior.

**RelationClusterer:** For each relation `r`, build its signature:
```
rel_sig(r) = distribution over (src_cat, tgt_cat) pairs connected by r
```
Two relations are equivalent if they connect the same atom-category pairs. This discovers:
- That H and its reverse `rev_H` are "the same relation, backwards"
- That diagonal relations D1/D2 form a separate cluster from axial H/V
- That some relations are equivalent by symmetry of the data

Implementation mirrors atom clustering: build `(relation,) → (f'{c_src}:{c_tgt}',)`
compound signatures, run `induce_hierarchy_bidir`, get `K_r` relation categories.

---

### Level 3 — Multi-hop Composition [ ]

`predict(atom, relation)` gives 1-hop. Relations compose:

```python
predict_chain(atom, ['capital_of', 'located_in'])  # Paris → France → Europe
predict_chain(patch, ['H', 'V'])                    # patch at (r,c) → (r, c+1) → (r+1, c+1)
```

Implementation: pipe output category from `next_cat_rel` into the next hop's input.
Intermediate atoms are marginalized over — the chain is a sequence of matrix products
over the category transition tensors.

---

### Level 4 — Second-Order Relational Grammar [ ]

Learn `next_rel(r1) → r2`: which relations tend to follow other relations?

This is the distributional structure *of the relational structure itself* — E1 applied
to relations rather than atoms. Discovers:
- Grammatical relation sequences in text (SUBJ → VERB, DET → NOUN)
- Spatial adjacency patterns (H after H = horizontal run; V after H = corner)
- Causal chains in action sequences (PUSH → OPEN → ENTER)

---

### Level 5 — Symmetry Discovery (the hard one) [ ]

**The question:** Can the learner discover 2D spatial structure with zero built-in
knowledge about directions? And can it then generalize to non-Euclidean geometries?

**Setup:** Replace hand-specified H/V/D1/D2 with raw pixel offsets `(dr, dc)`.
Every distinct offset is a separate "relation" initially. The learner must *discover*
which offsets are equivalent — i.e., discover the symmetry group.

**Why it works:** Two offsets are distributionally equivalent iff they connect the
same *category distributions* of atoms. In natural images:
- Horizontal left/right offsets tend to connect similar texture patches → cluster together
- Vertical offsets have different statistics (sky/ground asymmetry) → separate cluster
- The discovered clusters ARE the discovered symmetry orbits

**Connection to Klein's Erlangen Programme:**
> Geometry is the study of properties invariant under a symmetry group.

We invert this: given distributional data, discover the invariances → discover the geometry.
Crucially, this requires *no prior specification* of the symmetry group.

**Non-Euclidean extension:**
| Space | Raw "relation" | Expected discovery |
|-------|---------------|-------------------|
| Euclidean 2D | pixel offset (dr,dc) | dihedral-8 orbit structure |
| Sphere | geodesic arc (θ,φ) | SO(3) orbit structure |
| Hyperbolic plane | Poincaré offset | exponentially growing orbits |
| Graph | edge label | automorphism group of graph |

**The key test:** If we feed the learner images with no built-in directional structure,
does it recover the same relation categories as our hand-specified H/V/D1/D2? If yes,
we have discovered the symmetry from data alone.

**`RawOffsetLearner`** class (to implement):
```python
# All distinct offsets as raw relations:
for (r1,c1), (r2,c2) in all_nearby_pairs(grid):
    triples.append((grid[r1][c1], f'{r2-r1},{c2-c1}', grid[r2][c2]))

learner = RelationalLearner(n_clusters=K_atoms)
learner.fit(triples)
# Then: RelationClusterer discovers which offsets cluster together
# → discovered symmetry orbits
```

---

## Research Connections (found 2026-03-08)

These papers directly validate and extend the relational pipeline.

### Patch2Vec (Fried et al., 2017)
*"Similarity of two patches can be learned from the prevalence of their spatial proximity in natural images."*

Exactly our approach. They train a CNN with triplet loss where positive pairs = spatially adjacent
patches. We use k-means codebook + distributional clustering instead of a CNN. Same distributional
hypothesis, different implementation. **Validates our architecture.**

Key limitation they found: no fixed patch dictionary — every new texture may be unprecedented.
This is exactly the unique-patch problem we hit. Their fix: CNN embedding. Our fix: codebook.

### SymmetryLens (Efe & Ozakin, 2024) — arXiv:2410.05232
*Information-theoretic symmetry discovery: loss = symmetry + locality.*

**This is our Level 5 algorithm, in continuous form.** They discover which transformations preserve
the data distribution by jointly optimizing:
- **Symmetry**: p(x) ≈ p(g·x) — transformation g preserves the distribution
- **Locality**: nearby samples are transformed similarly

Our discrete analog:
- **Symmetry**: two offsets are equivalent if they connect the same category distributions (JSD ≈ 0)
- **Locality**: nearby offset vectors have similar distributional profiles

They demonstrate: pixel translation in CNNs is discoverable from natural image data alone.
For our RawOffsetLearner (Level 5a), the same principle applies: raw offsets (dr,dc) cluster
into equivalence classes by distributional profile → discovered symmetry orbits.

**The connection**: our RelationClusterer (Level 2) applied to raw offsets (Level 5a) IS
symmetry discovery in the sense of SymmetryLens — just discretized.

GitHub: https://github.com/onurefe/SymmetryLens

### LieGAN (Yang et al., ICML 2023) — arXiv:2302.00236
GAN-based discovery of Lie group symmetries. Generator produces transformations that preserve
the data distribution. Discovers SO(2), SO(3), Lorentz group SO(1,3)+ from data.
Limitation: search space restricted to general linear groups; fails for nonlinear symmetries.

LaLiGAN (2023) extends to nonlinear symmetries via latent-space linearization.

### Seg-HGNN (Mondal et al., BMVC 2024) — arXiv:2409.06589
*Hyperbolic GNN for unsupervised image segmentation. 7.5k parameters. Beats much larger models.*

Key finding: **part-whole hierarchies in visual scenes are naturally hyperbolic**. The exponential
growth of neighborhoods in hyperbolic space matches the exponential growth of part-whole
relationships (pixel → texture → part → object → scene).

Uses **Lorentz model** (hyperboloid), not Poincaré ball — numerically more stable, critical
for low-VRAM (4GB) training.

**Relevance to Level 5b**: our RawOffsetLearner on spherical/hyperbolic data should see
exponentially growing distributional diversity at each hop in hyperbolic space, vs. polynomial
growth in Euclidean. This is the observable signature that distinguishes hyperbolic geometry
from Euclidean without being told which geometry applies.

### Spatial Pyramid Matching (Lazebnik et al., CVPR 2006)
BoVW with spatial structure at 3 levels: 1 + 4 + 16 = 21 cells. Level 0 = whole image,
Level 1 = 2×2, Level 2 = 4×4. Concatenated → 21× richer representation than flat BoVW.
Our current 4×4 grid is already Level 2; adding Levels 0 and 1 is a direct improvement.

### DINOv2 (Meta, 2023)
Frozen ViT-S/14 features are semantically consistent across images without any training.
The "unique patch" problem disappears — patches of the same semantic content (fur, eye,
background) cluster in feature space even across different images.

**Path forward for image pipeline**: use DINOv2 features as the patch representation
instead of raw pixels → codebook k-means on 384-dim DINOv2 features → relational learning.
This is a hybrid symbolic+neural approach. Pure symbolic remains the long-term goal.

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
