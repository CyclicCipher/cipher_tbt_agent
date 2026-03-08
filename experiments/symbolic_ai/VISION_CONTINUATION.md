# Vision Pipeline Continuation

**Goal:** Make the symbolic AI's vision as good as a human's at a typical screen viewing distance.
Human eyes are ~2 feet (24") from a laptop or desktop monitor.  The system must have:
- Full **color vision** (opponent channels: red-green, blue-yellow, luminance)
- **Foveal acuity** calibrated to actual screen pixel density at viewing distance
- **Top-down predictive feedback** — infer / remember what should be present outside the fovea

---

## Completed

### Optimization pass (2026-03-08) — commits 8fbf7e9 → 977d566

| Task | Change | Speedup |
|------|--------|---------|
| Task 1 | Vectorize `_precompute_all_soft` → numpy tensordot | ~2000× |
| Task 2 | Vectorize `foveal_sequence` → LUT + meshgrid | ~10× |
| Task 3 | `ExampleStore._index` O(1) dict lookup | ~10× ask() |
| Task 4 | Vectorize `saliency_map` hash extraction | ~5× per call |
| Task 5 | Parallelize `fit_images` Phase 2 → ThreadPoolExecutor(n=4) | ~4× on images |

**Demo result:** n_images=30, `--classify` → 100% accuracy, completes in <60s.

### V7 — Opponent-color foveal tokens (2026-03-08, commit 89f6b69)

`_COLOR_LUT_FLAT` — 128 entries: `'C{l}{rg}{by}'`
- l ∈ 0..7  = luminance (3 bits)
- rg ∈ 0..3 = R-G opponent (0=G-heavy, 3=R-heavy)
- by ∈ 0..3 = B-Y opponent (0=Y-heavy, 3=B-heavy)

`foveal_sequence()` uses color tokens when `image.ndim == 3` + `fine_patch == 1`.
Peripheral `saliency_map` stays grayscale (peripheral is pattern detection, not color).

### V8 — Viewing-distance calibration (2026-03-08, commit 89f6b69)

```python
from vision_pipeline import foveal_radius_from_viewing_distance

# At 2 feet from a 24" desktop monitor (1920px wide):
r = foveal_radius_from_viewing_distance(1920, 20.8, 24.0)  # → 82 px

# At 2 feet from a 15.6" laptop (1920px wide):
r = foveal_radius_from_viewing_distance(1920, 13.5, 24.0)  # → 122 px
```

`peripheral_patch_from_viewing_distance(...)` similarly available.

Default `foveal_radius_px=48` is sized for small test images.
For real 1920×1080 screen tasks, set `foveal_radius_px=82` (desktop) or `=122` (laptop).

### V9a — `scan()` with inhibition-of-return (2026-03-08, commit 89f6b69)

`FovealVisionLearner.scan(image, n_saccades=5, use_topdown=True)`:
- Iterates saccades one at a time (biologically accurate vs. batch NMS in `fixate()`)
- **IOR**: zeros saliency in foveal-radius neighbourhood after each fixation
- **Top-down fill-in**: propagates peripheral Markov chain from anchor hashes;
  patches well-predicted by the model are marked "explained" (saliency→0)

Rao & Ballard (1999) loop: higher-area prediction → suppress lower-area firing
where prediction error ≈ 0.

---

## Remaining roadmap

### V9b — Richer top-down feedback: scene-level generative model (NEXT)

The current `scan()` top-down fill-in only propagates the peripheral Markov
chain 2 steps away from each fixation.  Richer top-down feedback requires a
**scene-level generative model**: after recognising an object at fixation N
(via foveal classification), predict WHERE the next salient object should be
and WHAT it should look like.

Design:
- After `teach_class()` + `classify()`, the V6 classifier identifies object
  types at each fixation.
- A scene grammar (`SequenceLearner` over object-type tokens) learns
  which object types co-occur and in which spatial arrangements.
- `scan()` queries this scene grammar: given objects seen so far, predict
  expected objects at unseen locations → suppress saliency where expected.

This requires training data with multiple objects (V7 multi-object scene graph).

### V10 — Multi-object scene graph

`FovealVisionLearner.scan()` already returns one dict per fixation.
V10 collects these into a scene graph:

```python
scene = {
    (320, 240): {'label': 'cat',   'saliency': 4.2},
    (100, 180): {'label': 'grass', 'saliency': 0.3},
}
```

Spatial queries: "Is there a cat?" → any node label == 'cat'.
"What is left of the cat?" → nodes with px < cat_px.

### V11 — Danganronpa scene understanding

Apply to game screenshots (1920×1080):
- Characters: high-saliency sprites → fixated, identified via foveal classification
- Text/UI: high-saliency text regions → fixated for OCR integration
- Background: repetitive, low-saliency → never fixated

Calibration: `FovealVisionLearner(foveal_radius_px=82, peripheral_patch=32)`
(24" viewing distance, 1920×1080 screen).

Color is essential: characters have distinctive hair/clothing colors that the
opponent-color tokens (V7) will capture.

### V12 — Peripheral color vision

`saliency_map` currently uses grayscale patches.  Adding color to the
peripheral scan would help distinguish same-brightness objects of different
hues (e.g., red vs. green objects on a white background).

Approach: replace the grayscale LUT in `saliency_map` with the color LUT
(needs the `rgb_img` array from the full image).  Peripheral clusters would
then be opponent-color clusters rather than luminance-only clusters.

---

## Key numbers

| Quantity | Value |
|----------|-------|
| Color token vocabulary | 128 (`C000`..`C733`) |
| Grayscale token vocabulary | 8 (`00`..`70`) |
| Position token vocabulary | 64 (`D-4,-4`..`D3,3`) |
| Foveal vocab (color + position) | 192 tokens |
| Foveal radius — laptop 15.6", 24" dist | 122 px |
| Foveal radius — desktop 24", 24" dist | 82 px |
| Foveal radius — demo default (small images) | 48 px |
| Peripheral patch — desktop, 24" dist, 0.8°/patch | 32 px |

---

## Viewing-distance reference

Human fovea = ±2° from fixation centre, full acuity.
Below 1° = foveola (maximum density cone cells).

```
Viewing: 24" (2 feet) from screen
Screen              Width (in)  ppd     Foveal radius (px)
------------------------------------------------------------
15.6" laptop        13.5"       61      122 px
24" desktop         20.8"       41      82 px
27" desktop         23.5"       37      74 px
32" desktop         27.9"       32      63 px
```

At a 1920px wide screen, these are the recommended `foveal_radius_px` values.
