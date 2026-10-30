# Vision: what a feature and a location are, for 2-D images (research note, 2026-10-29)

**The question** (the user, after part 12): "There must be something we're not understanding about the brain, HTM,
and TBT that would illuminate the problem." And: how do we know whether we need the spatial pooler, "or understand
what a nail is"?

**The nail, measured** (`docs/LAB.md` part 12, amendment 9 and D1):
- **Knowing the digit buys little:** it makes an image's 16 patches only e^1.8 ≈ 6 times more probable (1.8 nats),
  while naming one of 10 digits costs ln 10 ≈ 2.3 nats. So the objects' code length keeps one object.
- **The cause is the tile, not the encoder** (D1): at one retinal position, two 7s' 7 × 7 tiles share 0.33 of their
  pixels, a 7's and a 3's 0.25. The encoder keeps that contrast, but there is little to keep.

Nothing here is built. §6 states a diagnostic, D2, with its predictions before it runs.

## 1. What TBT means by a feature and a location

- **A feature is a local property at a pose.**
  - **Monty's sensor modules:** they turn raw input into "CMP-compliant features at a pose in 3D space"
    ([TBP 2024, arXiv 2412.18354](https://arxiv.org/pdf/2412.18354)).
  - **The pose:** for vision and touch, the pose vectors "are defined by the point normal and principal curvature
    directions sensed at the center of the patch".
  - **The features:** the sensor module also returns "pose-independent features (e.g. color, texture, or curvature
    magnitude)".
  - **The patch's role:** the whole patch is used to compute these, but the result belongs to its center. The patch
    is a measuring instrument, not the thing remembered.
- **So orientation is part of the pose, not the content.** A vertical and a horizontal edge at a point are the same
  feature in two poses. The object model stores which pose to expect where, and rotation is a hypothesis about the
  pose ([[reference_tbt_feature_definition]]).
- **A location is in the object's frame,** path-integrated from movement, and ambiguous until sensed: "locations start
  out ambiguous, represented as a union of locations, and are disambiguated over time through sensory input"
  ([Leadholm, Lewis & Ahmad 2021, arXiv 2102.09076](https://arxiv.org/pdf/2102.09076); Lewis et al. 2019).
- **An object is the arrangement:** a set of (feature, location) pairs. The variability of how it is drawn goes into
  pose and location, not into the feature.

**Against this, V in part 12:**
- **Its "feature"** is a 7 × 7 tile, a sixteenth of the digit, taken whole. It carries the handwriting's variability:
  a stroke a pixel to the left, or thicker, or slanted, is another tile.
- **Its locations** are retinal (innate), not the object's. A digit drawn two pixels over is another set of tiles at
  the same places.

## 2. What the brain does

1. **V1's features are small, oriented and learned.**
   - **Simple cells** respond to an oriented edge at one position; **complex cells** to the same orientation over a
     small range of positions (Hubel & Wiesel 1962).
   - **Learned from statistics:** sparse coding of natural image patches gives localized, oriented, band-pass
     features like simple cells' (Olshausen & Field 1996, *Nature* 381:607).
2. **Shift tolerance is learned from time, under eye movement.**
   - **The mechanism:** a unit that keeps responding while its input moves slightly learns to pool the shifted
     versions: Földiák's trace rule (1991, *Neural Computation* 3:194), and slow feature analysis, which gives
     complex cells from image sequences (Berkes & Wiskott 2005).
   - **The movement that supplies it:** the eye is never still. Drift and microsaccades keep moving the image on the
     retina, and "fixational eye movements are important components of the strategy by which the visual system
     processes fine spatial details" ([Rucci & Poletti 2015, *Annu. Rev. Vis. Sci.*](https://aplab.bcs.rochester.edu/assets/download/PDFs/articles/RucciPoletti15.pdf)).
   - **The same principle one level up:** IT learns object identity from views that follow each other in time (Li &
     DiCarlo 2008, `docs/JOINT.md` §1).
3. **People represent characters as strokes and their relations.**
   - **The model:** Lake, Salakhutdinov & Tenenbaum (2015, *Science* 350:1332) model a character as a program of
     strokes and relations between them, which reaches human-level one-shot classification on Omniglot
     ([CBMM](https://cbmm.mit.edu/publications/human-level-concept-learning-through-probabilistic-program-induction)).
   - **What it means here:** parts at relative locations, which is TBT's arrangement.
4. **A digit is more than a shape.**
   - **Shape alone isn't the category:** two 7s are one digit because they are used and named alike. A crossed and an
     uncrossed 7 differ more in shape than a 1 and a 7 often do.
   - **So vision's part:** a purely visual learner should find shapes and styles. The ten digits are equivalence
     classes by role: the association region's work (`docs/JOINT.md` §5), and the user's principle that a symbol's
     meaning is what it does.

## 3. What TBT's own implementations did with 2-D images

- **GridCellNet** (Leadholm, Lewis & Ahmad 2021), TBT on MNIST:
  - **The features came from a supervised CNN:** "trained ... in a supervised paradigm on a subset of the MNIST
    training data-set", then binarized, on a 5 × 5 grid.
  - **The objects were stored per class, by label.** The grid-cell location layer and its Hebbian learning were
    TBT's part.
  - **The authors' limits:** translation was not explored (the CNN's features "do not guarantee" the needed
    equivariance), and they suggest a patch-based autoencoder as future work.
- **Monty on Omniglot:** a patch moved along the strokes. It works "technically ... not with great performance yet",
  which its maintainers expect hierarchy to improve
  ([TBP forum](https://thousandbrains.discourse.group/t/evaluating-montys-performance-on-the-omniglot-dataset/553)).
- **So learning the features without labels is open in TBT itself.** We are not missing a known solution: part 12
  reached the open problem, and treated it as solved by taking the tile whole.

## 4. What this means for V

The gaps, each put against the user's two levers (ARCHITECTURE.md §0): the data a human gets, and sound general
architecture.

| gap | the brain / TBT | V in part 12 | lever |
|---|---|---|---|
| the feature | a small local property at a pose (an oriented edge), learned from image statistics | a whole tile, fixed by a random projection | architecture: a learned local feature layer (L4), on small patches |
| shift tolerance | learned from time, as the eye drifts | none: 16 static, non-overlapping tiles | **data:** the overlapping, slightly shifted views a human eye gets |
| the location | the object's frame, path-integrated, with the pose inferred | retinal, innate | architecture: B2, the digit's own frame |
| the category | shapes from vision; digits from use and names | asked of vision alone | the association region (part 13) |

**On the spatial pooler** (the user's hammer and nail):
- **What the SP is for:** it turns input of any density into a fixed-sparsity SDR, keeps overlap, and learns columns
  for the input's frequent patterns.
- **D1's lesson:** on whole tiles the SP has no nail. It can't make two 7s' tiles alike when their pixels aren't.
- **Where it could have one:** small patches, where it would learn the local, oriented features of §2.1. Whether
  that matters is what D2 asks first, with the features computed by hand. If local features don't carry the digit,
  learning them is pointless.

## 5. What is not designed

- How V1-like features would be learned (the SP, or sparse coding) and made shift-tolerant (a trace rule over
  fixational drift).
- How V moves: along strokes, as Monty does on Omniglot, or by drift and saccades.
- The pose inference of B2.

## 6. The diagnostic D2: do local features at object-relative locations carry the digit?

**The measure:** amendment 9's.
- **The definition:** the content code length of the images under the true digit partition, against one object
  (leave-one-out, information.py's content model, objects.py's `_Data`).
- **Reported per image:**
  - the gain in nats;
  - the net, after the partition's own cost (2.31 nats per image at 3,000 images).
- **Reading it:** a positive net means the objects' criterion would accept the digits.
- **Labels** are used only to form the partition being measured.

**The images:** the 3,000 training images of part 12's smoke case (seed 5), as in amendment 9.

**The representations** (one observation per location; the cell is the location):

| name | the feature at a location | locations |
|---|---|---|
| tiles16 | a 7 × 7 tile, by part 12's encoder | a 4 × 4 retinal grid (part 12's V) |
| tiles49 | a 4 × 4 tile, by the same kind of encoder | a 7 × 7 retinal grid |
| ink49 | ink or none, near the point (a silhouette: the control) | 7 × 7 retinal |
| orient16 | the local orientation, in 8 bins, or "blank", or "ink without an edge" | the 4 × 4 tiles' centres |
| orient49 | the same | 7 × 7 retinal |
| orient49box | the same | 7 × 7 over the digit's bounding box (object-relative) |

- **The orientation:** the structure tensor's dominant orientation at the point, from the image smoothed (σ = 1)
  and the tensor averaged (σ = 1.5).
- **Each symbol** is a random SDR, the same at every location.
- **What it is:** a hand-computed probe of what a V1-like layer would supply. It is not a proposal to build these
  features into the brain.

**Predictions** (2026-10-29, before running):

| # | prediction |
|---|---|
| D2a | tiles16 reproduces amendment 9: a gain of 1.80 nats per image (the apparatus) |
| D2b | at the same 16 locations, orient16's gain is at least tiles16's: local orientation carries at least as much of the digit as the whole tile |
| D2c | orient49's net is positive: the criterion would accept the digits |
| D2d | orient49 gains more than ink49: orientation carries the digit beyond the silhouette |
| D2e | orient49box gains more than orient49: object-relative locations beat retinal ones |

**What each outcome would mean:**
- **D2b–D2d hold:** local, pose-like features carry the digit. Learning them (§5) is the next design, and the SP or
  sparse coding then has a nail.
- **D2c fails while D2b holds:** features help, but per-location evidence is still too weak against ten labels. The
  evidence scale (part 12's limit 1) or the category's source (§2.4) would be the next question.
- **D2e fails:** MNIST is already centred well enough that B2 isn't what this needs.

## 7. D2's results (2026-10-29; `tests/vision_diag.py d2`, `runs/vision_d2.json`)

- **The images:** 3,000, seed 5.
- **The edge threshold:** 0.292, a tenth of the 95th percentile of the structure tensor's energy.

| representation | locations | gain (nats per image) | per location | net, after the partition's 2.30 |
|---|---|---|---|---|
| tiles16 (part 12's V) | 16 | 1.80 | 0.11 | −0.51 |
| orient16 | 16 | 2.53 | 0.16 | +0.22 |
| ink49 (the silhouette) | 49 | 3.85 | 0.08 | +1.54 |
| tiles49 | 49 | 4.51 | 0.09 | +2.21 |
| orient49 | 49 | 7.96 | 0.16 | +5.66 |
| orient49box | 49 | 11.32 | 0.23 | +9.01 |

**Verdicts:** D2a–D2e all confirmed.
- **D2a:** tiles16 reproduces amendment 9's 1.80.
- **D2b:** orient16 beats tiles16 at the same 16 locations (2.53 against 1.80).
- **D2c:** orient49's net is +5.66.
- **D2d:** orient49 beats the silhouette (7.96 against 3.85).
- **D2e:** object-relative locations beat retinal ones (11.32 against 7.96).

**What it shows:**
1. **The nail is the feature and the location, as TBT defines them.**
   - **The size of the effect:** a local, pose-like feature (an edge's orientation at a point), at locations in the
     digit's own frame, carries six times the digit evidence of part 12's tiles (11.3 against 1.8 nats per image).
   - **What it means:** the objects' criterion would then accept the ten digits by a wide margin (net +9.0).
   - **Each factor pays separately:** orientation over whole tiles, about 1.4 times per location; the digit's frame
     over the retina's, 1.4 times; more and smaller sensors.
2. **The spatial pooler, or sparse coding, now has a nail:** learning local, oriented features from small patches,
   which here were computed by hand. Whether a learned layer reaches orient49's evidence is the next measurement, not
   an assumption.
3. **B2 pays:** the digit's own frame adds 3.4 nats per image over the retina's, even on MNIST's centred digits.

**Caveats:**
- **The features are hand-computed:** they are a probe, and the brain has to learn them (§5).
- **Neighbouring locations aren't independent:** 49 points 4 pixels apart see overlapping neighbourhoods, so the
  summed gain at 49 locations overstates the evidence. The comparisons at equal location counts are the clean ones:
  orient16 against tiles16, and the four at 49.
- **The search is untested here:** this measures what the criterion would prefer. Part 12's search didn't even find
  "0 against the rest".
