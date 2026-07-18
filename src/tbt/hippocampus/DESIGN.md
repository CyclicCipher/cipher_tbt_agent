# hippocampus/ — DESIGN (mechanism check before code, 2026-07-18)

The full four-part hippocampus, decided up front (the user's call). This doc is the mechanism check + build plan; each module
re-checks its own mechanism at the start of its slice (`feedback_check_tbt_accuracy_per_step`). Sources: Marr/Treves/Rolls
quantitative theory of CA3 ([Rolls 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3691555/)); DG separation / CA3 completion
([Neunuebel & Knierim 2014](https://www.cell.com/neuron/fulltext/S0896-6273(13)01085-4)); remapping / charts
([DG drives remapping](https://www.biorxiv.org/content/10.64898/2025.12.04.692471v1.full)); our memories
`reference_hippocampus`, `reference_tbt_frames_and_hippocampus`, `project_hippocampus_imagination_and_widget`.

## §0 WHY (the block this removes)
A ROLLOUT (forward simulation for planning + the imagined-future widget) needs a persistent, coherent, allocentric world-STATE
to simulate *in* — snapshot it, branch it, run actions forward, score the leaf. We have built **what things are** (recognition)
and **how things change** (dynamics + override + relations) but NOT the state that binds them into *"the world right now,
including me."* The scene column's `_scene_objects` is a proto-version with no agent-in-the-map, no boundaries, no stable frame.
Bolting a rollout onto it is the convenient-representation-that-fails-at-the-seam — the pattern we've fixed 4× this session
(`project_representation_shortcut_lesson`). The general state IS the allocentric map; the rollout is a USE of it. So the map
comes first, and the rollout falls out.

## §1 WHAT THE HIPPOCAMPUS IS — and where "reuse the column" is right vs. not
It is ALLOCORTEX (3-layer), NOT a 6-layer neocortical column — so it reuses none of our L5 motor/displacement machinery (the
L5 gaps do not block it). The circuit: **EC** (grid cells = the path-integrated metric) → **DG** (pattern SEPARATION → sparse
place codes / orthogonal chart keys) → **CA3** (recurrent AUTOASSOCIATOR → one-trial binding + pattern COMPLETION) → **CA1**
(COMPARATOR of CA3-recall vs EC-input → match/novelty + output) → subiculum → back to EC/cortex. Plus **REMAPPING** (many
environments → distinct charts) and **REPLAY** (forward sweeps = the rollout).
- **REUSE gets right:** the grid/SR/path-integration METRIC (our `operator` + `GridEncoder`) and a SINGLE chart ≈ a column
  frame. Evolution-in-reverse is principled (cortex elaborated the older entorhinal grid machinery).
- **REUSE misses (genuinely more than a column):** multi-chart REMAPPING (chart selection + DG orthogonalization); the CA3
  recurrent ATTRACTOR (not the column's feedforward predict-then-compare); one-shot EPISODIC binding (fast, complementary to
  slow cortex). These are the hippocampus-specific parts — the reason this is a subfolder, not one file.

## §2 THE MODULES (one file = one concept; reuse, never reimplement — RULES #5)
| module | subfield | job | REUSES | NEW |
|--------|----------|-----|--------|-----|
| `map.py` | EC + place/boundary | the allocentric world-STATE: `{agent_pose, objects:{id→world_pose}, boundaries}`; path-integrate the agent in the WORLD frame; snapshot/branch (simulability) | `operator.MotionOperator` (path integration), `GridEncoder` (place read-out), `recognize` (objects) | world (not object) frame; multi-object; the AGENT is a thing in the map; boundary anchoring + loop-closure drift correction |
| `dg.py` | dentate gyrus | pattern SEPARATION: an environment signature → a sparse, orthogonal CHART KEY (which map am I in?) | `encoders.SpatialPooler` (k-WTA competitive sparse coding) | orthogonalisation tuned for chart separation (global remapping) |
| `ca3.py` | CA3 | the one-trial AUTOASSOCIATOR: store a pattern one-shot (Hebbian recurrent weights among co-active bits); COMPLETE it from a partial cue (recurrent settle + k-WTA). This is BOTH the attractor AND the episodic store (Rolls) | — | sparse Hopfield/SDM recurrent attractor — genuinely new dynamics |
| `ca1.py` | CA1 | the COMPARATOR: CA3-recall vs EC-input → a MATCH (familiar) / MISMATCH (NOVELTY → store a new episode/chart); passes the completed representation out | overlap comparison (`pooler`-style `_match`) | the EC↔CA3 novelty signal |
| `replay.py` | — | the ROLLOUT: snapshot the map → apply candidate actions (`operator` on the agent, `dynamics`/override on objects) → score the leaf (`reward.ValueCritic`) → pick. Forward sweeps = planning + the widget's imagined future | `operator`, `column.dynamics`/override, `reward.ValueCritic` | tree/beam over the world-state |
| `__init__.py` | — | the `Hippocampus` ORCHESTRATOR: composes map⊕dg⊕ca3⊕ca1⊕replay; the interface `agent.py` calls | all the above | — |

**Episodic = CA3, not a 5th module.** An episode = a scene's `(object ⊗ location)` bindings (`thalamus.bind`/`bundle`) →
a sparse code → `ca3.store` one-shot → `ca3.complete` from a cue. So episodic REUSES the Phase-5 thalamus VSA + CA3.
**Remapping = CA3 (recall) + CA1 (compare), with DG separating distinct charts.** On entering an environment, CA3 completes
the observed CONTENT to a stored chart and CA1 compares recall vs observation: MATCH (recall its map) or MISMATCH → mint a NEW
chart. The map is per-chart. **Refinement found at the slice-5 mechanism check (2026-07-18):** the CA1 partial-vs-contradiction
comparison runs on CONTENT tokens, NOT DG keys — the §3½ rule needs the SUBSET relation (observed ⊆ recalled), and DG's k-WTA
is nonlinear so a partial signature does NOT give a subset key. So DG's job is separating distinct FULL signatures at the
chart-INDEX layer (slice 4, shown); CA3 content-completion + CA1 do the recall/match/remap (slice 5). The two layers compose in
the orchestrator (slice 6).

## §3 BUILD ORDER (each: mechanism-check → build → WIRE from agent.py → test; nothing counts until wired + exercised)
1. **`map.py`** — the world-state + agent path-integration + snapshot. Unblocks everything. Test: hold a multi-object scene +
   the agent; path-integrate the agent; snapshot/branch is independent of the live state.
2. **`replay.py`** — the rollout in the map (the payoff: planning + the widget's substrate). Test: a delayed-goal scene is
   solved by forward simulation + critic scoring where a one-step greedy fails.
3. **`ca3.py`** — the sparse autoassociative attractor. Test: store patterns one-shot; complete from a partial/noisy cue;
   capacity holds for several patterns; an AMBIGUOUS cue stays ambiguous (§3½ — no confabulation).
4. **`dg.py`** — pattern separation → chart keys. Test: distinct environments → well-separated keys; similar ones overlap.
5. **`ca1.py`** + **remapping** — the comparator + chart select/create. Test: revisiting an environment RECALLS its chart;
   a novel one MINTS a new chart; the wrong chart is a CA1 mismatch; a PARTIAL view MATCHES but a CONTRADICTED view
   MISMATCHES (§3½ — absence ≠ novelty).
6. **`__init__.py`** (`Hippocampus`) ✅ DONE — composes the subfields behind ONE agent handle (`self.hippocampus`), the
   per-region fields folded behind it; `test_hippocampus` routes planning + episodic recall + remapping through the single
   handle end to end. **THE FULL FOUR-PART HIPPOCAMPUS IS COMPLETE (2026-07-18); all six slices done, suite 117 green.**

## §3½ RECOGNITION UNDER MISSING INFORMATION — the partial-view / occlusion invariant (a FIRST-CLASS tested guard)
The hazard, in our OWN minting logic: an apple seen from one angle, the visible part of a maze wall, a half-occluded tiger
must be recognised as the KNOWN WHOLE — not minted as a new "partial-apple" / "wall-fragment" / "half-tiger" model. The rule
that prevents it is one principle at three scales:

**MINT ON REFUTATION, NEVER ON INCOMPLETENESS.** Assume the partial input IS a known whole (the best hypothesis at the solved
pose/place), predicting the unseen; recruit a new model ONLY when evidence CONTRADICTS every known one. Occlusion/partiality
yield MISSING evidence (unresolved), not CONTRADICTORY evidence (refuting). Enforced by three things we already have or plan:
(i) evidence over a POSE-SOLVED population — a partial view is a consistent SUBSET, so the whole-object hypothesis survives
and wins (`recognize`/`perceive`); (ii) ART vigilance normalised by the OBSERVED input `|I∧w|/|I|` — the model having MORE
features than were sampled does NOT lower the match, so a partial view RESONATES with the whole model instead of resetting to
mint; (iii) unobserved locations are a GAP in the sweep, never a "blank" feature (a blank would be contradictory evidence and
would spuriously mint). This is `feedback_prefer_generalize_then_correct` exactly: assume the whole, retract only on contradiction.

The three examples DISSOCIATE the mechanisms — which is why they map cleanly onto the slices:
- **apple from one angle → the COLUMN** (one object, partial surface, pose solved, the model predicts the rest). Largely
  BUILT (`recognize`/`perceive` + ART vigilance); needs an explicit STRICT-SUBSET falsifier.
- **partial maze wall → the HIPPOCAMPUS / CA3** (pattern-complete the whole place/map from a local cue; recall WHERE you are).
  Slice 3.
- **half-occluded tiger → the SCENE / compositional column** (a nearer object EXPLAINS the absence; the tiger continues
  behind the wall). Needs depth / occluder-in-front — a real GAP; and COMMON FATE unifies the partial views (the once-hidden
  half later moves WITH the visible half → the two collapse into one object). Reuses `_common_fate_groups`; deferred with depth
  (§4). Without this guard, a consistently-occluded object IS the case that would wrongly mint a "half-object".

FALSIFIERS, made first-class (folded into the slice tests, not left implicit):
- **column ✅ DONE (`test_partial_recognition`):** a STRICT SUBSET (even rotated/translated) recognises as the WHOLE and does
  NOT mint; a CONTRADICTING feature is refuted and DOES mint. It caught a real gap — `recognize` computed vigilance
  (`refuted_at`) but never applied it, so a contradicting view survived as the sole hypothesis; fixed by filtering the
  population to `refuted_at is None or _exhausts(h)` (refuted-within-extent = contradiction → excluded; refuted-after-
  exhausting = boundary → kept, which `commit` needs).
- **CA3, slice 3:** a partial cue completes to the full stored pattern; an AMBIGUOUS cue stays ambiguous (no confabulation).
- **CA1, slice 5:** a partial view → MATCH (absence is not novelty); a changed/contradicted view → MISMATCH (novelty →
  store/explore). This is the scene-level "recognise vs mint" arbiter — the column's ART reset, one region up.

## §4 HONEST SCOPE (what stays deferred, noted not hidden)
- **ego→allo transform** (self-vs-world motion, retrosplenial gain fields) — needed for a MOVING sensor; ARC's fixed top-down
  board is already world-anchored, so it is DISSOLVED for ARC. Build the world-state now; add the transform for a moving
  camera (Danganronpa-class) later. `reference_hippocampus`.
- **theta phase / precession, sharp-wave ripples** as the replay clock — replay is modelled as a discrete forward sweep, not a
  phase-timed one.
- **neurogenesis / consolidation** (HPC→cortex transfer, the slow half of complementary learning systems) — out of scope.

## §5 HARD RULES (this subfolder, on top of `RULES.md`)
- **Reuse, never reimplement.** The metric is the `operator` + `GridEncoder`; the VSA is the `thalamus`; the value is
  `reward.ValueCritic`; separation is `SpatialPooler`. `map.py`/`replay.py` COMPOSE these — they do not re-derive path
  integration or value. Only CA3's recurrent attractor is genuinely new code.
- **No parallel systems.** The map is THE world-state; the scene column's `_scene_objects` is subsumed by it (or feeds it),
  not run alongside it.
- **Wired + exercised or it does not count** (RULES #3). Every module lands reachable from `agent.py` with a test that makes
  the agent do something it could not before. Not "a collection of scripts" — a composed, wired subsystem.
- **Do not call the single-chart map "the hippocampus."** The hippocampus is the full four-part thing; the map is its EC/place
  core. Name parts honestly.
