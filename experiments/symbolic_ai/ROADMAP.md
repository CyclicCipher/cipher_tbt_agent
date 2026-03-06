# Symbolic AI — Visual Learning Roadmap

## Where we are (Phases 1–14, all passing)

The system can:
- Learn arithmetic from ≤10 examples with 100% accuracy (Phases 1–10)
- Differentiate and integrate polynomials symbolically (Phases 11–12)
- Solve first- and second-order ODEs analytically (Phase 13)
- Apply Bernoulli's equation and Venturi meter (Phase 14)

All via exact template synthesis from the CTKG prerequisite graph.
No gradient descent. No trained parameters.

## The visual learning challenge

Cat detection is **statistical**, not exact. The key architectural decisions:

1. **Built-in priors are fair game** — anything a human is born with can be hard-coded:
   - DoG (retinal ganglion cells), Gabor (V1 simple cells), Gabor energy (V1 complex cells)
   - Innate face template (newborn face preference; Goren et al. 1975)
   - These live in `modalities/vision.py`

2. **Approximate synthesis** — Gap A from the previous analysis:
   - `synthesize_approx(accuracy_threshold=0.9)` finds templates with >= 90% accuracy
   - Not exact match; required for any noisy/statistical concept

3. **No translation problem** — TBT reference frames ensure:
   - Features are computed in object space, not retinal space
   - Innate face detector initializes the reference frame
   - Recognition generalizes across viewing angles by design

4. **CTKG hierarchy as classifier** — via Yoneda:
   - Cat = the concept at the top of a visual prerequisite hierarchy
   - MasteryState.expected_readiness() aggregates probabilistic evidence upward
   - No separate neural classifier needed

---

## Phase 15 — Approximate synthesis proof of concept

**Status:** Implemented in `run_experiment.py`
**Data:** Synthetic (no download needed)

Teach `image_brightness` concept:
- "bright" = synthetic float32 image with mean > 0.5 → output (1,)
- "dark"   = synthetic float32 image with mean < 0.3 → output (0,)

Expected outcome:
- Synthesizer discovers: `gray = img_to_gray(a); score = img_mean(gray); emit(if(score > T, 1, 0))`
- Threshold T automatically selected to maximize training accuracy
- 100% accuracy on held-out synthetic examples

---

## Phase 16 — CIFAR-10 cat classification

**Status:** Implemented in `run_experiment.py`
**Data:** Auto-downloaded from `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
**Data location:** `data/cifar10/` (created automatically)

Algorithm:
1. Load CIFAR-10: cat (class 3) vs. non-cat (all other classes)
2. Subsample 200 cat + 200 non-cat training examples for synthesis speed
3. Approximate synthesis over visual feature templates:
   - Brightness, std, DoG, single-orientation Gabor, multi-orientation Gabor sum
   - 9 threshold values per feature, both polarities
4. Select best template (highest training accuracy)
5. Verify on full 1,000 test examples

Expected accuracy: 65–75% (single fixed feature, no learned weights)
Interpretability: print the discovered rule (which feature, which threshold)

---

## Phase 17 — TBT multi-view cat learning (USER IMAGES NEEDED)

**Status:** Infrastructure implemented; waiting for user images
**Data location:** See shopping list below

Architecture (Thousand Brains Theory-inspired):

```
Raw image (any resolution)
    ↓  img_face_detect()     ← innate face template (built in, like succ)
Face location + confidence
    ↓  img_face_align()      ← canonicalize to 32×32 face space
Canonical face patch
    ↓  ColumnModel           ← TBT column: stores (object_location → expected_feature)
Per-location feature votes
    ↓  MasteryState          ← Bayesian aggregation up CTKG hierarchy
cat_face → cat confidence
```

The **reference frame alignment** (`img_face_detect` + `img_face_align`) is the key:
- Solves the translation problem: cat face is represented in face space, not image space
- Trained on frontal faces, generalizes to 3/4 and side views (via learned pose tolerance)

### Verification / interpretability benchmark

After Phase 17, inspect the learned column model:

```python
from column_model import ColumnModel
model = ColumnModel.load('data/cats/column_model.pkl')
model.print_expected_features()
# Output:
#   Location (0.25, 0.10): DoG energy > 0.4, Gabor energy @45° > 0.3  ← left ear
#   Location (0.75, 0.10): DoG energy > 0.4, Gabor energy @135° > 0.3 ← right ear
#   Location (0.25, 0.35): Gabor energy @all > 0.5, face_schematic > 0.6 ← left eye
#   ...
```

A human reading this output can judge: "does the model's location → feature map
look like a reasonable cat?" This is the interpretability benchmark.

Verification tests:
1. **Novel viewpoint**: train frontal only, test on side view images → >60% pass criterion
2. **Part localization**: given image, query "where is the left ear?" → within 25% of image width
3. **Reference frame readout**: serialize ColumnModel, plot as heatmaps on canonical cat template

---

## Shopping list for Phase 17

Download these images manually and place in the folders below.
Any common image format works (jpg, png, webp).
Aim for variety: different cats, different lighting, clear subjects.

### Positive examples (cats)

```
data/cats/frontal/      — 25 images
```
Search: `"cat face frontal photo"`, `"cat looking at camera"`
Criteria: cat roughly centered, facing camera, face visible, whole head in frame

```
data/cats/side/         — 20 images
```
Search: `"cat profile photo"`, `"cat side view"`
Criteria: cat in true profile (90° from camera), head and body visible

```
data/cats/three_quarter/ — 15 images
```
Search: `"cat 3/4 view"`, `"cat turned slightly"`
Criteria: cat at ~45° angle to camera

```
data/cats/seated/       — 15 images
```
Search: `"cat sitting full body photo"`, `"seated cat clear background"`
Criteria: full body visible, clear background preferred, any orientation

```
data/cats/other_poses/  — 10 images
```
Search: `"cat walking photo"`, `"cat stretching photo"`
Criteria: interesting pose variety, no need for consistent orientation

**Total cats: 85 images**

### Negative examples (not cats — with similar visual complexity)

```
data/negatives/dogs/    — 20 images
```
Search: `"dog face photo"`, `"dog looking at camera"`
Criteria: dogs with visible faces/ears; common confusion class with cats

```
data/negatives/rabbits/ — 10 images
```
Search: `"rabbit face photo"`
Criteria: similar ear/face structure; good hard negative

```
data/negatives/other_animals/ — 10 images
```
Search: `"fox face photo"`, `"ferret photo"`, `"raccoon photo"`
Criteria: furry animals with similar facial structure

```
data/negatives/no_animals/ — 10 images
```
Search: `"indoor room photo"`, `"street scene photo"`
Criteria: scenes without any animals; easy negatives

**Total negatives: 50 images**
**Grand total: 135 images**

---

## Phase 18 — Screen-level recognition (Danganronpa)

Future phase, not yet implemented.

Input: 1920×1080 screenshot
Task: detect characters, UI elements, game state

Architecture extension:
- `img_face_detect` runs on downsampled 240×135 first (coarse localization)
- Then `img_face_align` crops + normalizes candidate regions
- Character identity via ColumnModel trained on character sprites
- UI element detection via Gabor at appropriate scales

The same `VisionModality` + `ColumnModel` infrastructure used in Phase 17
scales to full resolution — all primitives are resolution-agnostic.

---

## File structure

```
experiments/symbolic_ai/
├── ROADMAP.md               ← this file
├── interpreter.py           ← process language executor (float literals added)
├── memory.py                ← ExampleStore + KL divergence
├── synthesis.py             ← template synthesis + approximate synthesis
├── engine.py                ← SymbolicAI class + consolidate_approx()
├── run_experiment.py        ← Phases 1–16 (Phase 17 once images downloaded)
├── data_loader.py           ← CIFAR-10 download + image folder loader
├── column_model.py          ← TBT column model (Phase 17)
├── modalities/
│   ├── base.py              ← Modality ABC
│   └── vision.py            ← VisionModality (19 primitives + 5 face primitives)
└── data/
    ├── cifar10/             ← auto-downloaded by data_loader.py
    ├── cats/                ← user-provided (see shopping list)
    │   ├── frontal/
    │   ├── side/
    │   ├── three_quarter/
    │   ├── seated/
    │   └── other_poses/
    └── negatives/
        ├── dogs/
        ├── rabbits/
        ├── other_animals/
        └── no_animals/
```
