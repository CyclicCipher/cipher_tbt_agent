# MICROCIRCUIT — the minicolumn-native cortical loop (L4 · L2/3 · L5)

*Companion to `ARCHITECTURE.md` (the single source of truth). The plan to redesign **L4, L2/3, L5** as ONE minicolumn
substrate wired into the canonical cortical microcircuit, so the column runs the **full TBT loop** — localization AND
content-prediction — not just the localization half it runs today (`column.py` audit, 2026-07-05). Grounded in the M5 HTM
temporal memory (`sequence.py`) + the minicolumn structure: Hawkins & Ahmad 2016 (*Why Neurons Have Thousands of
Synapses*); Numenta "Columns" papers (Hawkins, Ahmad & Cui 2017, *A Theory of How Columns…*; Lewis et al. 2019,
*Locations in the Neocortex*). Obeys the five dev rules (ARCHITECTURE §7). It COMPLETES `ARCHITECTURE §10` P2 (the one
prediction, in its full content form) + wires P3 (sequence memory) into the layers + adds the inter-layer feedback P4
implies.*

---

## 1. What's wrong now (the diagnosis)

The live loop (verified against `column.perceive`/`forward`, 2026-07-05) closes a faithful **localization** loop but
NOT the **content/prediction** loop:

1. **L4 is a vestige.** Content comes from the retina's `view_sdr` (M3), not bound in L4 at the L6 location. The
   canonical "L4 binds the sensed feature at L6's location" (§4c) is not wired; `L4_FeatureLocation` (an int codebook)
   is unused in the loop and surfaced as dead code during the classical-geometry cleanup.
2. **The layers are three incompatible mechanisms, not one substrate.** L4 = int codebook; L2/3 = a point-cloud
   evidence recogniser (M4); L5 = float efference tuples/matrices. §5's "ONE temporal-sequence-memory mechanism
   instantiated per layer" is aspirational — the M5 HTM TM exists but is a detached component, wired into nothing.
3. **There is NO top-down (apical) channel.** Nothing feeds back down. So: no **frame gating** (the object cannot select
   which of its many learned frames is active — §"one active frame" discussion), no **hierarchy / modes**, no **top-down
   attention**.
4. **No content prediction.** `forward` merely ECHOES the given content (valid only for self-motion reafference). The
   column cannot predict a feature that CHANGES, nor the feature at an unvisited location.
5. **The column models static shapes at poses.** No dynamics / behaviors / order-config rules, even though `Behavior`
   (M5) exists — because it is a bolt-on, not the object layer's native operation.
6. **Recognition is not prediction.** L2/3 recognises via a separate evidence mechanism over a raw cloud, not by
   pooling a *predictive* L4 — so recognition and prediction are two systems, not one.

## 2. The mechanism (TBT-accuracy check — verify before building, §11.3)

The canonical cortical microcircuit. Each cell has THREE dendrite channels (M5 built only the basal one):
- **PROXIMAL (feedforward)** — the "what"; drives which minicolumns are active.
- **BASAL / distal (lateral)** — CONTEXT; depolarises cells into the **predictive** state (the M5 mechanism).
- **APICAL (feedback)** — TOP-DOWN bias from the layer/region above.

Layer roles (Numenta "Columns"; Lewis grid-cell framework):

- **L6 — location.** Grid cells; the current location in the object's frame; path-integrated by L5's efference. **Already
  built** (`hippocampus`, M1). One reusable coordinate system (§"single cohesive reference frame").
- **L4 — the sensorimotor INPUT layer.** Minicolumns = the feed-forward feature (the content SDR). Cells-per-column give
  context. **Basal context = the L6 location ⊕ L5 efference** (so the same feature at a different location / under a
  different movement is a different cell). **Apical = the L2/3 object** (gates which object's map is predicted). L4 IS the
  object's **feature-at-location map**: query it with a location (L6) + object (L2/3) → the predicted feature there. A
  *burst* = the sensed feature was not predicted = surprise / wrong-object / new-object.
- **L2/3 — the OUTPUT / object layer (the "column pooler").** Proximal = the active L4 cells. It **temporal-pools** them
  into a STABLE, location-INVARIANT object SDR (a slowly-decaying union, re-pooled on L4 surprise — reuse M5's pooling
  idea). Lateral = **voting** across columns + within-column stability. Apical = higher-region feedback (hierarchy;
  deferred). Recognition = which pooled object SDR the current L4 activity matches (**associative recall / overlap**, the
  M4 primitive, now over the L4 representation). It feeds **back to L4 (apical)** — this is frame gating.
- **L5 — motor / output.** The efference copy (per-action displacement → L6) — **already built** (M1). Plus a minicolumn
  TM over ACTIONS (motor skills / sequences), basal context = the program phase. (The efference is load-bearing; the
  action-sequence memory is a lower-priority add.)

**The loop, per sensorimotor step:**
```
   L5 motor emits action a  ─► efference copy
                                  │
   L6 path-integrates location by a  ─►  new location SDR ─┐
                                                           ▼ (basal)
   L2/3 object ──(apical)──►  L4: PREDICT the feature at the new location
                                  ▲ (proximal)
   retina view_sdr ─────────────►  L4: the SENSED feature fires the predicted cells (or BURSTS = surprise)
                                  │
                                  ▼ (proximal, up)
   L4 active cells ───────────►  L2/3: POOL → the stable object SDR ──► vote (lateral)
                                  │
                                  └──(apical, down)──►  gates L4's next prediction   ⟲
```
Pose = the L6 location (+ orientation/anchor) combined with the object identity — **not a separate solve** (this is the
key migration: M4's virtual-rotation pose inference becomes "which L6 anchoring makes L4's predictions match," i.e. pose
lives in L6, the object in L2/3 is location-invariant). **Is this how a real column does it?** Yes — Numenta's input/output
two-layer circuit is exactly this; the location-invariant object + location-variant input is the core of "Columns."

## 3. What it unlocks (why, beyond biological accuracy)

1. **Context-dependence at every layer → modeling CHANGE, not just static structure** (the big one): L4 content dynamics
   (a toggling/cycling feature), L2/3 object **behaviors** (open/close, a machine cycle, a patrol), L5 motor skills, and
   **order/config-dependence** (Sokoban: config as basal context) — the mechanics most ARC-AGI-3 games actually are.
2. **A top-down (apical) channel** we currently lack entirely: **frame gating** (many frames, one active — mechanical),
   **hierarchy / "which behavior" modes** (Jiang & Rao), **top-down attention / active sensing**.
3. **Sensorimotor imagination**: predict the feature at an UNVISITED / occluded location from the object map → **object
   permanence** + **planning by simulated sensing** ("if I move there, do I see the goal-feature?") without physically
   moving — the RHAE efficiency lever.
4. **A dense intrinsic learning signal**: bursting = prediction error at every layer, every step — self-supervision
   instead of relying on the sparse ARC score; the epistemic-value term (§8) becomes the burst rate.
5. **Unification**: recognition, prediction, and the active representation collapse into ONE thing (you recognise *by
   predicting correctly*); voting becomes a native lateral-settling dynamic that scales to the multi-column sheet (§3b).

## 4. What we're going to build

- **`sequence.py`**: add the **apical channel** to the HTM TM (a second top-down context input that biases which cell
  wins / pre-depolarises) — the "apical tiebreak temporal memory" (Numenta htmresearch). The one new primitive.
- **L4** (rewrite `l4_feature_location` → the sensorimotor input layer): a minicolumn TM, proximal = content SDR, basal =
  L6 location ⊕ L5 efference, apical = L2/3 object; predicts feature-at-location; queryable at unvisited locations.
- **L2/3** (evolve `l23_object` → add the column pooler): pool L4 → a stable location-invariant object SDR; recognition by
  overlap vs the library union; feed back to L4 (apical); keep `vote`. Pose read from L6.
- **`column.py`**: rewire `perceive`/`forward` to run the microcircuit loop; keep the L6/hippocampus/navigation
  localization loop intact.
- **Retire** the subsumed machinery once green: the L4 int codebook; M4's evidence/virtual-rotation pose solve *if* the
  L6-anchor pose inference subsumes it (else keep virtual rotation as the L6-anchor search).

## 5. The build plan (dependency-ordered; suite-green + one paper-test per stage; branch for risk)

**Stage 0 — this doc + the accuracy check.** (This commit.) One prediction, one substrate, no parallel systems.

**Stage 1 — the apical channel on the HTM TM.** Extend `SequenceMemory` with a top-down context input; a cell predicted
by BASAL context and SUPPORTED by APICAL wins over a basal-only cell (tiebreak), and apical alone can pre-bias. Isolated
test: apical feedback disambiguates an otherwise-ambiguous continuation. LOW risk (extends M5, touches nothing live).

**Stage 2 — L4 as the sensorimotor feature-at-location layer (isolated prototype — the load-bearing new capability).**
The new L4: columns = content-SDR features; basal = L6 location ⊕ L5 efference; apical = object SDR. Isolated tests on a
synthetic object (a feature-at-location map): (a) learns the map; (b) predicts the feature at a VISITED location; (c)
predicts the feature at an UNVISITED location (**imagination**); (d) BURSTS for the wrong object. Do NOT touch the live
loop yet — prove the primitive first (the M4/M5 lesson: prototype in isolation, tune at small scale).

**Stage 3 — L2/3 as the column pooler (isolated prototype).** Pool L4 cells → a stable, location-invariant object SDR;
recognise by overlap vs the library; read pose from L6. Isolated test: recognise the tetromino library at unseen poses
(subsume `test_l23_object`), location-invariant identity, pose from L6.

**Stage 4 — wire the microcircuit in the column.** Rewire `column.perceive`/`forward` to the §2 loop; keep the
hippocampus localization loop + navigation. Verify: the games (MockLive, sokoban, collectall) + recognition tests stay
green. HIGHEST risk (integration + the pose migration) — do it behind the existing method signatures, swapping the
content/recognition path while the localization loop is untouched; revert-and-instrument on any regression (M1 lesson).

**Stage 5 — retire the subsumed machinery.** The old L4 int codebook + `content_code`; M4's now-redundant pose machinery
(or keep virtual rotation as the L6-anchor search). Cleanup + delete the dead tests, suite green.

**Stage 6 — validate the NEW capabilities (the payoff).** New tests + wiring: content dynamics (a toggling feature),
object behavior (open/close via the L2/3 sequence), config-dependence (Sokoban push as basal context), imagination
(predict an occluded feature). These are the point of the whole build.

## 6. Acceptance & risks (the paper test, per stage)

- **Every stage**: explainable in one §1–§6 sentence; obeys the five rules; TBT-accurate mechanism (re-checked at the
  stage, not just here). Suite-green throughout; a git branch for the risky stages (4–5).
- **Risk 1 — the recognition-core rewrite.** M4 works; Stage 3/4 subsume it. Mitigate: build L2/3-pooler in isolation
  (Stage 3) and keep M4's tests as the acceptance bar before swapping.
- **Risk 2 — pose migration to L6.** M4 gives (quantised) pose in the recogniser; the new design infers pose as the L6
  anchoring that makes L4's predictions match. Likely QUANTISED (the grid modules give discrete anchors — consistent with
  M4's decision). Open question: exact-continuous vs nearest-anchor; resolve empirically, relax tests to the native
  semantics if needed (the M4 precedent).
- **Risk 3 — small-scale convergence.** The M5 tiny-SDR convergence cost recurs; tune params per stage, relax exposure in
  tests where the mechanism is faithful but slow (documented, not gamed).
- **Non-goal (this phase)**: the multi-column SHEET + full heterarchy (§3b C–E) and the higher-region apical hierarchy —
  the microcircuit is ONE column's loop; the sheet reuses it later.
