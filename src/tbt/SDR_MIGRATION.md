# SDR-NATIVE MIGRATION — from a classical-geometry loop to an SDR substrate

Companion to `ARCHITECTURE.md` (the single source of truth). This doc records the AUDIT (2026-07-04) that the live
perception / localization / recognition loop mostly runs on **classical geometry** — Euclidean coordinates, SE(2)
matrices, angles, exact-match keys, and symbolic tables — with the real SDR machinery (`encoders.py`, `GridEncoder`,
`SuccessorFeatures`) bolted on the side. The brain's currency is the **SDR**; every representation and every operation
should be SDR-native. This is the "symbolic estimators dressed in TBT vocab" the RESET (ARCHITECTURE §Reset) was meant to
remove, resurfacing in a new form. It is also P5 (`project_location_is_an_sdr`, `reference_sdr_encoder_library`) becoming
the SUBSTRATE rather than a side module.

## Why it matters (not cosmetic)
1. **Bugs from hand-kept consistency.** The orientation bug that triggered this audit — `head = atan2(≈0, ≈0)` of a flat
   ring leaking a garbage angle — is a symptom: THREE coordinate/angle encodings of one thing (pose `theta` matrix-angle,
   efference `dθ` tuple-scalar, `head` circular-mean scalar) are kept consistent by hand, and a degenerate case leaks. One
   SDR substrate has no scalar to go undefined.
2. **No generalization.** Exact-match keys (`invariant_sig`, `view_signature`, `L4.encode` integer ids) make a near-miss
   ORTHOGONAL, not similar — so nothing generalizes across similar-but-unseen inputs. SDR overlap gives graceful similarity
   + noise tolerance (the associative-recall prerequisite, plan step B).
3. **One substrate, one set of ops.** Location = a grid-cell SDR; heading = a HD-ring SDR; content = a feature SDR; the
   object = a stable L2/3 SDR; the operator = a learned map ON the SDR. Then "path integration" is the operator applied to
   the SDR (a phase shift), recognition is overlap/union, novelty is non-convergence — no `R(head)·v`, no `atan2`, no
   nearest-neighbour scan.

## The audit — current representation vs the SDR-native target

| Component | Current representation | SDR? | SDR-native target |
|---|---|---|---|
| **Location belief** (`hip._P`) | dense `board×board` probability map, **localist** grid (one cell = one position) | ✗ (dense, localist; bump-flavored) | the **grid-cell SDR** (`GridEncoder`, multi-module) IS the belief; path-integrated by the operator |
| **Heading belief** (`hip.head`, `hip._Q`) | scalar circular-mean + dense ring bump | ✗ | a **HD ring-attractor SDR**; shifted by the turn operator |
| **Position read-out** (`hip.pos`) | decoded `(x,y)` float point | ✗ | (only decoded at the motor/periphery via the encoder inverse) |
| **Path integration** | `pos += R(head)·ego` — rotation-matrix on Euclidean vectors | ✗ | the **operator applied to the location SDR** (grid phase shift); the gain field is a population op |
| **Efference** (`L5.eff`) | float tuple `(dx, dy, dθ)`, body-frame | ✗ | a **displacement-cell SDR / an `Operator`** on the location code (see M1) |
| **Operators** (`operator.Operator`, `pose_operator`) | SE(2) **matrices** | ✗ (matrix on coords) | a learned **group-representation matrix acting on the SDR** (Gao 2021, `reference_operator_as_group_representation`) |
| **L2/3 recognition** (`l23_object`) | Euclidean `nearest` + `atan2` pose-solve; **scalar** evidence | ✗ | **SDR overlap / union** (θ-thresholded); object identity = a stable **L2/3 SDR** |
| **Feature signature** (`invariant_sig`) | exact-match hashable key | ✗ | a **feature SDR** with graded overlap |
| **L4 content** (`view_signature`, `L4.encode`) | exact-match tuple → integer codebook id (localist) | ✗ | a **content SDR** where similar views share bits (plan step B) |
| **Temporal memory** (`SequenceMemory`) | symbolic Markov dict `context_tuple → Counter(next)` | ✗ | **HTM temporal memory** — predictive cells in an SDR, distal-context, bursting |
| **Self-grouping** (`_attend_self`) | connected components + Euclidean footprint match | ✗ | (peripheral; grouping falls out of SDR prediction / common-fate) |
| **Value** (`SuccessorFeatures`) | linear read `V = w·φ` on the location SDR | ✓ | keep — the one genuinely SDR-native live op |
| **Encoders** (`encoders.py`) | bidirectional Scalar/Grid/Category + SpatialPooler | ✓ | keep — the substrate the rest migrates onto |

## The migration plan (dependency-ordered)
Gated by the acceptance test (ARCHITECTURE §11) + the five rules; each step keeps the suite green.
- **M1 — efference → `Operator` (the entry point).** Collapse the redundancy: the self's per-action motion is ONE thing,
  currently encoded THREE ways (efference tuple, pose matrix, head scalar). Make the efference an **`Operator`** (an SE(2)
  element to start), so "path integration" is `operator.apply(location)` and "head" is the accumulated composition — one
  representation, composed by `.then`, no `R(head)·v`/`atan2` by hand. This unifies L5's efference with L2/3's pose
  machinery (they are the same SE(2) group element). NB it is still a matrix-on-coordinates until M2 puts it on the SDR.
- **M2 — location + heading as SDRs.** Retire the dense localist `_P` / scalar `head`: the belief IS the grid-cell SDR (+
  HD-ring SDR), and the M1 operator acts on THAT (a learned phase shift), path-integration = operator-on-SDR. The bump
  belief's population-coding instinct is preserved, but sparse + distributed + generalizing.
- **M3 — content as an overlapping SDR (= plan step B's prerequisite).** Replace `view_signature`/`L4.encode` exact-match
  ids with a content SDR (encoder / spatial pooler) so similar views overlap; enables associative recall.
- **M4 — L2/3 recognition as SDR overlap/union.** Replace Euclidean `nearest` + `atan2` with SDR overlap against a library
  union (θ); the object identity becomes a stable L2/3 SDR (the temporal-pooler output). Folds in plan steps A (persistent
  session / convergence) + B (associative recall).
- **M5 — HTM temporal memory.** Replace the symbolic `SequenceMemory` dict with predictive cells in an SDR (distal-context
  → predictive state → bursting), so "predictive state / bursting / non-convergence" are literal, not simulated.

## Relationship to the existing plan
This is not a new axis — it is the SUBSTRATE the A–E plan (ARCHITECTURE §10) already assumes: A (persistent session /
convergence) and B (associative recall) are M4/M3; the RF/sheet (C/D) columns each carry this substrate. M1 is the
immediate next step (the efference→Operator unification), safe to do before the fuller SDR migration, and it removes the
three-way orientation redundancy that produced the readout bug. Order: **M1 now → M2 → M3 (=B) → M4 (=A+B) → M5.**

## Experimental uses (test AFTER the migration) — predictive-spectral features (ReSU / SFA)
NOT on the critical path — a hypothesis to test once the SDR loop (M1–M5) exists. **Rectified Spectral Units** (ReSU, arXiv
2512.23146) learn each feature as the CCA canonical direction maximizing **past↔future mutual information**, rectified
(ON/OFF), backprop-free (analytic SVD). No explicit HTM link, but the same predictive-feature objective as **Slow Feature
Analysis** and, spectrally, as the **SR eigenbasis** (SFA ≈ Laplacian/diffusion eigenmaps of the transition operator ≈ SR
eigenvectors — Sprekeler 2011; Stachenfeld 2017). Our L6 grid = SR-eigenbasis IS this for LOCATION; the idea extends the
SAME objective to CONTENT and behavior.

- **Hypothesis:** forward modelling improves if features are chosen to be **PREDICTABLE** (max past→future MI), not
  exact-match. Today L4 content (`view_signature`/codebook) is identity-matched, so the forward model must predict in a
  space not optimized for prediction. Predictive-spectral content features would make the operator's next-feature
  prediction a well-conditioned (near-linear) map, and generalize to unseen-but-predictable views.
- **Where it plugs in:** the M3 content encoder (a *predictive* content SDR, not just topology/exact-match); the M5
  temporal memory / behavior features (CCA directions of the pose/displacement stream); possibly UNIFYING L4 predictive
  content features with the L6 SR under ONE spectral objective ("extract predictive spectral features," instantiated per
  layer — the same "one mechanism per layer" pattern as temporal sequence memory).
- **SDR / HTM translation (the real work):** ReSU is dense + batch-SVD; we need (a) **online/incremental** estimation of the
  slow/predictive directions (online SFA, or Hebbian–anti-Hebbian "slowness" nets — Földiák — not batch SVD); (b)
  **sparsification** to an SDR (k-winners-take-all / a spatial pooler over the predictive projections); (c) the **ON/OFF
  rectification** — already SDR-friendly (half-rectified = sparse non-negative activations, like ON/OFF cells). Keep it as
  an ENCODER enhancement feeding the same operator/recognition machinery, never a parallel model (rule 1).
- **The gate (bitter-lesson honest):** adopt ONLY if an experiment shows a predictive-feature encoder measurably improves
  forward-model accuracy / generalization vs. the M3 topology/exact-match encoder on the held-out split. Otherwise it stays
  a documented idea.
