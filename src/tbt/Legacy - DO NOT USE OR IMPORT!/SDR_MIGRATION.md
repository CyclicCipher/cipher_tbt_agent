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

## The migration plan — M1–M4 TOGETHER (one SDR-native loop), then M5
**Decision (2026-07-04):** do M1–M4 as ONE coherent change, NOT staged. Rationale learned the hard way: staging M1 as a
POINT SE(2) pose (matrix-on-coordinates, population dropped "until M2") **destabilized navigation** — the population bump
was silently providing smoothing/robustness (a cue-commitment gap it masked), so the point-pose interim regressed NavGame
8/8→0/8 and collectall. Keeping the population **throughout** (as the SDR) avoids any destabilizing interim. So the belief
is SDR + population from the first commit; the operator, content, and recognition all move onto the SDR together.

**The unified SDR-native belief + operator (M1 ⊕ M2).**
- The self's pose belief is ONE population state **φ = (grid-cell SDR for position) ⊕ (HD ring SDR for heading)** — sparse,
  distributed, generalizing. `GridEncoder`/`ScalarEncoder(periodic)` are the substrate.
- The efference is a learned **`Operator` acting ON φ** (a group representation — Gao 2021, `reference_operator_as_group_
  representation`; the L6_NONABELIAN frontier): a translation is a PHASE SHIFT of the grid modules, a turn is a ring-shift
  of the HD SDR (+ a rotation of the grid frame). ONE operator type, unifying L5's efference with L2/3's pose machinery.
- **Path integration = `φ ← M_a · φ`** (the operator applied to the SDR) — the gain field as a phase operation, NO
  `R(head)·v`, NO `atan2` of a scalar. **Correction = SUPERIMPOSE the sensory likelihood SDR** (from recognition) on φ —
  the population combination keeps reliability-weighting emergent (no scalar gain; the no-scalar-confidence win preserved
  throughout). A point read-out (for the motor / goal vector) is the encoder INVERSE, only at the periphery.

**Content SDR + recognition as overlap (M3 ⊕ M4).**
- Content = an **overlapping SDR** (a spatial pooler over the view) so similar views share bits (retire `view_signature`
  exact-match / `L4.encode` integer ids).
  - **BUILT 2026-07-05 (`retina.view_sdr`, `test_retina`).** The overlapping content SDR = the UNION of two analytic-encoder
    fields — colour composition (a `CategoryEncoder` block per present colour: disjoint, no false colour-nearness) ⊕
    pairwise-DISTANCE geometry (a `ScalarEncoder` bump per pair: nearby distances overlap, rotation/translation
    invariant). **Deviation from the "spatial pooler" wording (analogous to M1's gain-field note):** the analytic
    encoders are used, NOT a learned `SpatialPooler` — the view has explicit structure (colours + a distance metric), so
    the analytic encoders give the same overlap graded-ness *deterministically* (no Hebbian drift → the same view is the
    same SDR every frame, which the persistent-recognition session and `test_perceive_unifies` need). The `SpatialPooler`
    stays the fallback "when no analytic encoder fits" (`encoders.py`). Measured: overlap(L, L)=15, (L, L+1cell)=15,
    (L, same-shape-other-colour)=10, (L, a line)=6 — graded similarity where `view_signature` gave 0 for every non-identity.
    Wired into `column.perceive` (`feat = view_sdr(self_cells)`, no `L4.encode` int id) and the standalone `sensor.read`;
    `view_signature` is deleted. (`L4.encode`'s int codebook + `L4.E` remain for the thalamus binding — out of the
    perceive path, a later cleanup.)
- The object identity is a **stable L2/3 SDR** = the temporal-pooler union of predicted content SDRs over the sensorimotor
  sequence (folds in plan step A's persistent session — convergence is now SDR settling).
- **Recognition = SDR overlap/union** against the library (θ-thresholded, ~O(1) — folds in plan step B, associative
  recall); pose inferred via the displacement-cell SDRs / the operator. Retire Euclidean `nearest` + `atan2` pose-solve.
  - **DECISION 2026-07-05 (user): go FULLY SDR-native — pose quantized too**, grounded in TBT/HTM research (below), NOT
    the identity-only half-measure. The `atan2` in `l5_displacement.pose_between` (the analytic continuous-angle solve)
    IS retired here, replaced by the mechanism the research actually uses.
  - **Research grounding (Lewis et al., *Orientation-Invariant Sensorimotor Object Recognition Using Cortical Grid Cells*,
    Front. Comput. Neurosci. 2022, PMC8825787; Monty / TBP, arXiv 2412.18354).** Both represent pose as a **DISCRETE set of
    orientation hypotheses**, never an analytic angle: Lewis uses a **circular buffer of N grid-cell modules** (25 baseline)
    → N orientation bins over 360°, and recognises a rotated object by **VIRTUAL ROTATION** ("similar to mental rotations")
    — replay the sensorimotor sequence across all N orientations and pick the one with the **highest firing rate / lowest
    location-layer ambiguity**, keeping the learned feature-location associations intact (a circular shift of the ordered
    modules, no retraining). Monty generates a discrete rotation set from sensed pose features and accumulates **evidence**
    over `(object, rotation, location)` tuples (evidence weighted by the *continuous* angular difference, but the
    hypotheses are discrete), path-integrating each hypothesis forward by the (rotated) displacement. So a quantized pose
    is not a compromise — it is how the cortex does it; the recovered pose is the **best-matching discrete hypothesis**.
  - **Concrete SDR-native design (M4).**
    1. **Identity = a stable L2/3 SDR** — each object stores a content-union SDR (over `view_sdr`/local-descriptor SDRs).
    2. **Associative recall** — a sensed content SDR is matched by **overlap against the library** (θ-thresholded), ~O(1)
       in library size, shortlisting candidate objects (retires the serial scan over every object×node).
    3. **Pose by virtual rotation over N orientation bins** — precompute the N rotation operators `R_r = rot(r·2π/N)`
       (the group-representation operator, `operator.py`/`l5_displacement.rot`). Recognition seeds a hypothesis per
       `(object, bin r)` whose *rotated* local descriptor matches the sensed one (an SDR/set match under `R_r`), and
       accumulates evidence as the sensor moves (the displacement is rotated by `R_r`, path-integrating the hypothesis).
       The winning bin is the inferred orientation. **This retires `align_rotations`/`pose_between` (the `atan2` solve) and
       the Euclidean `nearest` node scan** — matching is overlap under the discrete `R_r`.
    4. `cells_at(bin_center, t)` reconstructs the object at the bin's rotation (exact at the bin, quantised by 2π/N).
  - **Test-semantics change (justified by the research, not a hack):** `test_l23_object`'s exact-continuous-pose assertion
    (`cells_at(θ,t)==cloud` for a random θ) relaxes to **nearest-bin recovery** (the rotated model matches the cloud within
    the bin resolution). Identification stays 100%; only the analytic exact-angle claim — which the research says the cortex
    does NOT make — is relaxed. `N` chosen divisible by 4 so the 90°/180°/270° cases (the online-label-free test) are exact.
  - **BUILT 2026-07-05 (`l23_object.py`, `test_l23_object` 8/8).** `N_BINS=24` (15° quantum). New machinery: `_desc_sdr`
    (rotation-invariant local descriptor as an overlap-bearing `ScalarEncoder` union — retires `invariant_sig ==`);
    `ObjectGraph.identity` (the union of node descriptors = the stable identity SDR); `_bins_matching` (virtual rotation:
    the LOWEST-residual orientation bin(s) + genuine symmetric ties, NOT every bin within tol — Lewis's lowest-ambiguity
    orientation); `sense` INIT = associative-recall shortlist (`O.identity.overlap(sensed) ≥ θ`) → per-node `_desc_match`
    → virtual-rotation bin seed (no atan2); UPDATE path-integrates each hypothesis by the bin-rotated displacement and
    re-verifies the descriptor + committed bin. `vote` pools by `(object, bin, coarse-position)` — quantised pose gives a
    node-dependent single-glance `t` (~1-cell scatter), so the position is pooled at `_VOTE_POS_TOL=2.0`, not the exact
    `t`. **Retired from the recognition path:** `align_rotations`/`pose_between` (the atan2 solve) and the exact-match
    `invariant_sig` gate. **Kept:** `pose_between`/`pose_operator` still live in `l5_displacement` as the operator
    group-representation (Gao 2021), tested by `test_operator`/`test_l5` — out of the recognition path, not deleted.
    Costs (research-consistent, all documented in the tests): 2 fixations → ~97% (a glance is noisier than the analytic
    solve), full sensing → 100%; single-glance localisation is coarse; suite l23 time ~2× (the N-bin residual sweep).
    Games unregressed (MockLive 60/60, sokoban ~2.2s); suite 76 passed / 6 xfailed.

**Classical-geometry CLEANUP (2026-07-05, after M1–M4).** With the SDR-native loop in place, the superseded classical
machinery was deleted so only SDR geometry remains. Removed: `pose_between`/`align_rotations` (the atan2 SE(2) angle solve)
+ `_sets_match` + the `math` import (`l5_displacement`); `invariant_sig` (the exact-match descriptor tuple, `l4`); the
`Retina` RF-sweep patch codebook + `dominant_region` (exact-match/connected-component, `retina`); the l23 legacy VSA store
(`S`/`pool`/`revise`); the dead `sensor.encode`/`L4.encode` wiring — plus their now-dead tests + unused imports. **Kept**
the SDR-native operator/displacement geometry these read through (`rot`, `pose_operator`, `apply_pose`, `local_disps`,
`Operator`/`OnlineOperator` — the Gao-2021 group representation) and the forward-looking `thalamus` + `L4` layer (§3/§2,
the multi-column sheet's router + a core layer, unwired but documented). Suite 70 passed / 6 xfailed; games unregressed.

**Build order WITHIN the combined change (each keeps the suite green + the population intact):**
1. φ + the operator-on-SDR for the self pose (M1⊕M2 core) — replace the dense localist `hip._P` board-map with the
   **modular grid-cell SDR population** (a bump per `GridEncoder` module ring; the belief now lives in the SAME grid code
   `SuccessorFeatures` values), and path integration `_shift2d(_P, world)` with the **per-module grid phase shift**
   (`φ ← M(v_world)·φ`, the continuous-attractor operator — Burak & Fiete). Keep the head ring `_Q` (already a population)
   and the superposition correction (per module). Un-xfails `test_forward` WITHOUT the point-pose regression (population
   retained).
   - **TBT-accuracy correction (2026-07-05, found at the §11.3 check).** "Retire `R(head)·v`" does NOT mean replace the
     gain field with a *marginalisation/mixture over the head population* (`Σ_h Q[h]·shift(R(θ_h)·ego)`). That mixture is a
     DRIFT that breaks the abelian case: a world-frame body (NavGame's symmetric 2×2 mover, ACTION1-4) has a genuinely
     FLAT head ring (symmetry → unobservable heading, correctly `θ=None`), and a mixture over a flat `Q` smears the
     world-frame move across all four rotations → the position bump disintegrates → NavGame 8/8→0/8. The gain field must
     read the head as a **population VECTOR** (the circular-mean read-out = the standard HD population decode) to produce
     ONE world velocity `v_world = R(head)·ego`, then drive the operator `φ ← M(v_world)·φ`. For an abelian body this is
     `R(0)=identity` (correct, no smear); for a sharp SE(2) head it is the true rotation. This is still a **population op**
     (the head is read as a population, not a hand-kept scalar register) and reliability weighting stays **emergent in the
     bump widths** (the superposition), not a scalar `conf`. What is actually retired: the dense-Euclidean `_shift2d` of a
     localist board-map, and the `atan2`-on-a-degenerate-flat-ring leaking a garbage angle into efference LEARNING.
2. content SDR (M3) — the overlapping encoder feeding L4/L2/3.
3. recognition as overlap + the L2/3 identity SDR (M4) — over the content SDR, folding in A/B.
- **M5 — HTM temporal memory (predictive cells in an SDR).** Replace the symbolic `SequenceMemory` dict with predictive
  cells in an SDR (distal-context → predictive state → bursting), so "predictive state / bursting / non-convergence" are
  literal.
  - **Deep TBT-accuracy check (2026-07-05) — the actual mechanism (Hawkins & Ahmad 2016, *Why Neurons Have Thousands of
    Synapses*; Numenta BAMI *Temporal Memory Algorithm Details*).** The layer is **minicolumns × M cells** (M≈32 gives the
    high-order capacity). A feed-forward SDR activates a set of **columns**. Each cell has **distal dendrite segments**;
    each segment is a set of synapses (permanence 0..1, *connected* ≥ `connectedPermanence`≈0.5) onto OTHER cells. Per step:
    (1) **activate cells** — for each active column: if it holds a cell in the **predictive** state (a distal segment
    matched last step's active cells), that cell fires and INHIBITS its column siblings → a **context-specific** code; if
    NO cell was predicted the column **BURSTS** (all M cells fire) and a **winner** cell is chosen (best-matching segment,
    else least-used). (2) **activate segments** — a segment whose connected synapses onto the now-active cells exceed
    `activationThreshold`(≈13) makes its cell **predictive** for next step; these predictive cells' columns ARE the
    predicted next input. (3) **learn** — reinforce the segment that correctly predicted (+`permanenceIncrement`≈0.1 to
    synapses onto prior-active cells, −`permanenceDecrement` to the rest), **punish** a segment that predicted a column
    that did NOT activate (−`predictedSegmentDecrement`), and in a bursting column GROW a segment on the winner onto the
    prior **winner** cells (up to `maxNewSynapseCount`≈20). Defaults: cellsPerColumn 32, activationThreshold 13,
    minThreshold 10, initialPermanence 0.21, permInc/Dec 0.1.
  - **Why the current `SequenceMemory` is the symbolic caricature to remove.** It keys a dict on a FIXED-order tuple of the
    last `order` elements (exact-match → a near-miss context is ORTHOGONAL, no generalisation), and "predictive state /
    bursting" are faked by dict hit/miss (`predict()→None`). HTM instead learns a **variable-order** context (a segment
    grows only as much context as needed to disambiguate) over **overlap-bearing** cell SDRs, and bursting is the LITERAL
    all-cells-fire surprise. The **high-order** property is emergent, not a hyperparameter: the same column fires a
    different CELL after A vs after D (ABC vs DBE), so C-after-(A·B) predicts C while B-after-D predicts E — first-order
    over the context-specific cells.
  - **Mapping onto our model.** (a) The **phase** (§5, "the ONE recurrence") = the **active-cell SDR** (which cell per
    column) — the distal segments read it; there is no separate `order` counter. (b) One mechanism, per-layer by the
    CONTEXT + element (§5): L4 next-**feature** (context = the previous cells + the L6 **location** — the sensorimotor-HTM
    apical/basal input, Numenta "Columns" 2017), L2/3 next-**displacement** (context = the object phase), L5 next-**action**
    (context = the program phase). (c) The **element ↔ SDR**: the TM operates over active **columns**; a symbolic/abstract
    element encodes via a `CategoryEncoder` (disjoint block per symbol — the tests + abstract tokens), while real content is
    ALREADY an SDR (`view_sdr` / a displacement SDR) whose active bits ARE the columns; `predict()` decodes the predictive
    columns back (overlap decode). (d) **L6 stays the SR, NOT this** (§5): the *Distributed Hebbian Temporal Memory* result
    (arXiv 2310.13391) shows HTM sequence memory ≈ learning a successor representation by Hebbian rules, confirming they are
    the SAME family but DIFFERENT grains — the SR is the multi-step discounted map for PLANNING (§8), the TM is the one-step
    high-order dynamics for perception; L6 keeps the SR, L4/L2/3/L5 get the TM. (e) **Backward modelling** (`inverse`) is
    unchanged — the operator inverse, not the table.
  - **Honest scaling deviation (to verify in the build).** Full-scale HTM (2048 cols, 32 cells, threshold 13) needs wide
    SDRs + many repetitions for permanences to cross `connectedPermanence`. The symbolic tests use tiny elements + 3–4
    reps, so the params scale DOWN (fewer cells/column, a lower threshold tied to the SDR sparsity, a higher initial
    permanence/increment so a segment connects in ~2 reps) — a scaling choice, not a mechanism change. Test-semantics that
    peek at the symbolic `table` dict relax to the HTM-native equivalents (predicted-columns unambiguous = `confident`;
    no-predicted-column = burst), analogous to M4's quantised-pose relaxations.
  - **BUILT 2026-07-05 (`sequence.py`, `test_sequence` 6/6).** `SequenceMemory` is now the HTM TM: `M=8` cells/column
    (`order<=1`→1 = first-order), a symbol→disjoint-column-block encoder (`w=8`; a real content SDR's active bits are the
    columns), distal `seg[cell] = {presyn_cell: permanence}`; `observe` activates cells (predicted → fire alone + reinforce;
    unpredicted → BURST, deterministic least-used winner grows/reinforces a segment onto prior winners), PUNISHES false
    predictions, then recomputes predictive cells (`connected_match ≥ activation_threshold`). Read-outs: `predict`/
    `candidates`/`confident` decode the predictive columns; `_burst` is the surprise. VERIFIED: the SAME element B fires
    DIFFERENT context-specific cells after A vs D (cell 0 vs 1 → predict C vs E) — literal high-order; novel input bursts.
    `Behavior` + `inverse` unchanged. **Deviations (research-consistent, documented in the tests):** (i) the repeated-
    element high-order case (up,up→right; +1,+1→−1) converges more slowly at the tiny 8×8 SDR scale than a wide biological
    SDR, so those tests use more exposure (12 reps); (ii) HTM predicts a first-order FALLBACK where the symbolic dict
    returned `None` for an unseen tuple (graceful degradation — the burst is the unseen-context signal). NB `SequenceMemory`
    is built + tested but NOT yet wired into the live loop (per §5 it lands in L4/L2/3/L5 when the forward model needs it).

**Prerequisite surfaced by the M1 attempt: cue commitment.** The point-pose exposed that the agent DROPS a cued target
when its salience flickers (e.g. the marker's pop-out vanishes when the body is adjacent) → a 2-cycle one step from the
goal. Whether the population masks it or not, this is a real discovery-robustness gap (BG "hold your horses" —
`reference_basal_ganglia`): once cued to a target, COMMIT to reaching it. Fix it as part of the combined change (or just
before), and verify NavGame/collectall stay green.

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
