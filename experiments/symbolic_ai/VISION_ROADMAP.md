# Vision System Roadmap

## The Core Problem

Whole-image classification fails for complex scenes. A "cat in a meadow" image is
~85% grass — the global patch histogram is dominated by background. No classifier
operating on the whole image can generalise.

The fix is biological: the human visual system never classifies images globally. It
builds a saliency map, fixates on surprising locations, and names objects within the
foveal field — not the whole scene.

---

## Architecture Overview

```
Full image
    │
    ▼  Peripheral scan (coarse patches, 32px)
    │  VisionLearner (E0-E6, n_clusters=32)
    │
    ▼  Saliency map: -log P(patch | context) per patch
    │  High surprise → candidate fixation
    │
    ▼  Non-max suppression → N fixation centres (px, py)
    │  (grass = predictable → low saliency → never fixated)
    │  (object edges = surprising → high saliency → always fixated)
    │
    ▼  Foveal crop (radius=48px around each fixation)
    │  fine_patch=1 → pixel-accurate 1×1 tiles
    │
    ▼  Interleaved token sequence
    │  ['P0,0', hash_px, 'P0,1', hash_px, ..., 'P7,7', hash_px]
    │
    ▼  SequenceLearner (E0-E6, n_clusters=64)
       Learns spatial grammar of attended objects
```

---

## Positional Encoding

Position tokens are interleaved with content tokens in the foveal sequence.
The SequenceLearner treats them as ordinary vocabulary items — the category
chain learns which positions predict which content.

**Token format:** `f'P{pos_r},{pos_c}'` where `pos_r, pos_c ∈ [0, n_pos_bins-1]`.
Default: 8×8 = 64 distinct position tokens per crop.

**Why interleaved, not appended:**
Position + content are coupled — `P3,2` followed by a specific hash
constitutes one "feature at a location" observation. The E1 trigram
`(P3,1, hash_A, P3,2)` predicts the next position; `(hash_A, P3,2, hash_B)`
predicts the next content at that position. Both are useful.

**Biological analogue:**
Grid cells fire at regular spatial intervals across any 2D space. Their multi-
frequency hexagonal lattice is mathematically equivalent to multi-frequency
sinusoidal positional encoding. Our 8×8 position bins are a discrete approximation
of the same idea — coarse position at no extra cost, learned from context.

---

## Why Prediction Error = Saliency

Rao & Ballard (1999): higher cortical areas predict what lower areas will send.
V1 neurons fire strongly for **prediction errors** — inputs that don't match the
top-down expectation. Attention amplifies these error signals (FEF, LIP).

In our system:
- `logprob(h_{i-2}, h_{i-1}, h_i)` is the learned predictability of patch i
- Grass texture: repetitive, high log-prob, **low saliency** → ignored
- Object boundary: unexpected, low/None log-prob, **high saliency** → fixated
- `None` (completely unseen) → maximum saliency (5.0) → always fixated first

This gives free figure-ground separation: backgrounds are predictable, objects
are not. No segmentation network required.

---

## Implementation Status

### V1 — Peripheral saliency map ✅ DONE

`saliency_map(image, learner, patch_size) → np.ndarray`

Scanline scan at `patch_size` resolution. Calls `learner.learner.logprob()`
at each position. Returns 2D float32 (rows × cols). Graceful fallback for
missing numpy / untrained learner.

### V2 — Fixation selection ✅ DONE

`select_fixations(saliency, patch_size, n, min_dist_px) → list[(px, py)]`

Non-max suppression with configurable minimum distance between fixations.
Returns pixel-coordinate centres, ordered by descending saliency.

### V3 — Pixel-accurate foveal scan ✅ DONE

`foveal_sequence(image, px_cx, px_cy, radius_px, fine_patch=1, n_pos_bins=8)`

Crops image around fixation centre, scans at `fine_patch` resolution (default 1 pixel),
interleaves `['P{r},{c}', content_hash, ...]`. At `fine_patch=1`, each hash is a
single pixel quantized to 3-bit greyscale (8 levels: '00'..'07'). Sufficient to
read text strokes, whiskers, fine edges — anything a human can read, the AI can read.

### V4 — FovealVisionLearner ✅ DONE

`class FovealVisionLearner`

Two internal learners:
- `self.peripheral: VisionLearner` — coarse scan, generates saliency
- `self.foveal: SequenceLearner` — fine scan at fixation points

`fit_images(images)` trains both in sequence.
`fixate(image)` returns `[{center_px, saliency, sequence}]` per fixation.
`evaluate_images(test, train)` evaluates foveal prediction accuracy.

**Default parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `peripheral_patch` | 32px | Coarse enough for fast saliency; fine enough to locate objects |
| `foveal_patch` | 1px | Pixel-accurate; human-readable |
| `foveal_radius_px` | 48px | 96×96 px crop — enough for a face, character, word |
| `n_fixations` | 3 | Matches typical human saccade count before recognition |
| `n_pos_bins` | 8 | 8×8 = 64 position tokens — coarse grid cell analogue |
| `n_foveal_clusters` | 64 | Richer than peripheral; fine patches have higher vocabulary |

---

## Upcoming Phases

### V5 — Object-relative reference frame (NEXT)

After cropping to a salient region, position is currently **crop-relative** (top-left
of crop = P0,0). This is already better than absolute image coordinates, but still
tied to where in the image the saccade landed.

True object-relative position: anchor at the **highest-saliency pixel within the crop**
(the most surprising point = most likely the object's distinctive feature). Re-express
all other positions as `(Δrow, Δcol)` from the anchor.

Result: a cat ear at top-left and a cat ear at top-right produce the same sequence.
The object's spatial signature is viewpoint-invariant.

**Implementation:**
- In `foveal_sequence`: find peak pixel by prediction error within crop
- Replace absolute `(pos_r, pos_c)` with relative `(delta_r, delta_c)` bins
- Clip Δ to [-n_pos_bins//2, n_pos_bins//2] per axis

### V6 — Classification chain (NEXT)

After V4 gives us foveal sequences, add a CTKG concept:

```
concept image_class
    input  foveal_category_sequence   # category histogram from foveal E1/E3
    output class_label
```

Train by calling `ai.teach('image_class', foveal_histogram, label)` for each
training crop. `freq_consolidate` learns `P(class | foveal_pattern)`.

Then E6 beam search discovers the composition:
```
foveal_sequence → foveal_categories → class_label
```

The discovered chain is a symbolic rule: "if position P3,2 has edge-category AND
P4,3 has fur-category AND P2,1 has ear-shape-category → cat". Fully interpretable.

### V7 — Multi-object scene graph (FUTURE)

Multiple fixations (n=5–10) each yield an object description. These are collected
into a scene graph:

```python
scene = {
    (320, 240): 'cat',    # fixation 1
    (100, 180): 'grass',  # fixation 2 (low-saliency, background)
    (400, 100): 'sky',    # fixation 3
}
```

Query: "Is there a cat in this image?" → check any node's label == 'cat'.
Query: "What is to the left of the cat?" → spatial reasoning over the graph.

### V8 — Danganronpa scene understanding (FUTURE)

Apply to game screenshots:
- Characters: distinctive sprites with sharp edges → high saliency
- UI elements: text, icons → fixated for OCR
- Background: static, predictable → never fixated

The game's 1920×1080 frames are processed at full pixel accuracy in the foveal
crop. The peripheral scan runs at 60×34 coarse patches (32px each), giving a
60×34 saliency map from which 3–5 fixation points are selected per frame.

Character identity: each character's foveal sequence has a distinctive spatial
grammar (hair colour distribution, eye position, clothing patches). Classification
via `image_class` concept after training on character sprites.

---

## File Structure

```
experiments/symbolic_ai/
├── VISION_ROADMAP.md      ← this file
├── vision_pipeline.py     ← VisionLearner + FovealVisionLearner
│                             saliency_map(), select_fixations(), foveal_sequence()
├── sequence_pipeline.py   ← SequenceLearner (used by both learners)
└── modalities/
    └── visual_symbol.py   ← _to_gray_f32, _extract_patches, _quantize
```

---

## Key Numbers (from demo run, 30 synthetic stripe images)

| Metric | Value |
|--------|-------|
| Peripheral patches | 768 (24 images × 32 patches) |
| Peripheral clusters | 8 |
| Foveal crops | 72 (24 images × 3 fixations) |
| Foveal token pairs | 219,064 |
| Foveal vocab (position + content) | 65 unique tokens (64 pos + 1 content type for stripes) |
| Foveal clusters | 16 |
| Foveal E1 accuracy | 59.9% (vs flat bigram 50.7%) |
| Sequence length per crop | ~3,072 tokens (96×96 px / 1px × 2 for pos+content interleave) |
