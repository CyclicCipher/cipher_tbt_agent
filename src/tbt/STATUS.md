# STATUS.md — the live map of the TBT agent

The single answer to *"what is in the codebase and what is wired?"* Derived from the code, per `RULES.md` #1 — if this
doc and the code ever disagree, the code wins and this doc is fixed. Not a plan, not history; just current reality.

## State: STARTING OVER (2026-07-09)

The previous system is archived under `src/tbt/Legacy - DO NOT USE OR IMPORT!/` — **reference only, never imported.**
It became an un-navigable tangle: mechanisms built and then un-wired in successive "collapse" rewrites and never
re-integrated (basal ganglia, thalamus, forward model), plus four conflicting source-of-truth docs. We are rebuilding
from scratch, **vertical slice first**, under `RULES.md`.

The legacy code is for **reference so we don't redo solved work** (encoders, HTM temporal memory, SR features, the
recognizer, the operator geometry all exist there and are sound in isolation) — but nothing is copied without going
through RULE 5 (does it exist / extend the owner) and RULE 3 (it isn't "done" until wired + exercised end-to-end).

## North star
A Thousand-Brains agent that plays ARC-AGI-3-style games end-to-end (perceive → predict → plan → act → win), learned,
no per-game code. **Design + rationale: `ARCHITECTURE.md`. Discipline: `RULES.md`. This doc: current wired state only.**
The system is **ALWAYS MULTI-COLUMN** — at least one sensory column *and* one PFC/task column, never single-column, no
exceptions (`ARCHITECTURE.md` §5.1). **First concrete target:** win one trivial replica level end-to-end through that loop.

## What is BUILT + WIRED in the new system
FIRST VERTICAL SLICE DONE (2026-07-10): everything is now reachable from the live entry point — STANDALONE is empty, the
RULES.md #2 goal.

| module | one-line job | wired? | its test |
|--------|--------------|--------|----------|
| `agent.py` | live entry point / ROOT — composes a SENSORY + a TASK `Column` (the sensorimotor SCAN), a DECISION loop (`decide`: perceive→relay→select→gate→act; `reward`: RPE-train), AND a SPATIAL nav slice (`learn_move`/`locate`/`path_integrate`/`where`: L6a path integration by the learned operator, ARCHITECTURE §8); the `step(obs)→action` game loop is still a stub | ROOT | `test_column_arithmetic`, `test_bg_thalamus`, `test_operator_path_integration` |
| `column.py` | the cortical COLUMN — 5 layers (L4, L2/3, L5IT, L5PT, L6a) as one `HTMLayer` each, wired per its §12 table (+ the full layer/sublamina research + whole-column plan in its docstring); `observe` drives L4 (the first slice); a SPATIAL column (`location=GridEncoder`) also gains L6a's TRANSFORM engine (`ModularOperator`) + path-integration API; the full §13 `step` counterstream is the target | WIRED | `test_column_arithmetic`, `test_operator_path_integration` |
| `operator.py` | the TRANSFORM primitive (ARCHITECTURE §8) — L6a path integration as a learned, module-structured group PERMUTATION of the `GridEncoder` location code (`ModularOperator`: per-module phase-delta voting → `apply`); position-invariant by construction, the second primitive alongside `htm.py`'s ASSOCIATE. Abelian only; non-abelian (conjunctive) + obstacle override deferred | WIRED | `test_operator_path_integration` |
| `basal_ganglia.py` | the value-driven SELECTOR — OpAL Go/NoGo + dopamine-RPE + STN commitment (reference_basal_ganglia); the ONE place competing options are arbitrated (rule 4). MoE column-allocation deferred | WIRED | `test_bg_thalamus` |
| `thalamus.py` | the inter-column ROUTER/GATE — relays a column's percept to the selector + gates the BG's winner to the motor; content⊗location binding (place-value / voting) deferred | WIRED | `test_bg_thalamus` |
| `htm.py` | the ONE cortical-layer mechanism — HTM sequence memory (proximal SDR-in via SP, basal context, apical, learn/predict/burst); a layer = one instance + a declared (proximal-in, context-in, apical-in, target-out) wiring | WIRED | `test_htm` |
| `encoders.py` | SDR transduction library (`SDR` + Scalar/Category/Grid/Multi/Conjunctive/SpatialPooler) — data ↔ overlap-bearing SDR | WIRED | `test_encoders` |

**Generalization investigation — RESOLVED 2026-07-09 (now wired into the column, above).** Two durable results,
full detail in memory `project_place_invariance_needs_factored_state` + `reference_htm_canonical_pipeline`, plain-English
writeup in `ARCHITECTURE.md` §7:
1. **The substrate is VALIDATED.** The canonical HTM pipeline **encoder → SpatialPooler → HTMLayer → SDRClassifier** reaches
   **96% exact** next-value on a repeating count. Earlier "0% / can't generalize" scares were a NON-CANONICAL harness — no
   SP, and decoding the TM's *predictive cells* through the encoder-INVERSE (blur/lag). Use a trained classifier over the
   *active* cells; grade by EXACT decode (overlap grading is fooled by persistence). `HTMLayer.observe` gained a
   `learn=False` inference mode (kept).
2. **The place-invariance wall + its fix.** A per-place model generalizes carries WITHIN the trained place-value structure
   (held-out numbers: non-carry 100%, carry 86%) but is **0%** on a novel place-VALUE — plain SP/TM have no weight-sharing
   across positions (a known HTM limit). Fix = the Monty/TBT reference-frame move: ONE shared model walked across
   place-LOCATIONS with content place-invariant and carry as an inter-location STATE → hundreds **100%** (monolithic 0%).
   A FACTORED recurrent state channel closes it (100%) where PURE temporal memory can't (37%). ReSU investigated + DROPPED
   (temporal encoder, not spatial invariance).

Suite: **31 passed** (~20s; `test_column_arithmetic` is the ~20s end-to-end column test, `test_bg_thalamus` +
`test_operator_path_integration` are fast). Run
`python src/tests/test_reachability.py` for the wired map; the 20 legacy test files are archived under
`Legacy - DO NOT USE OR IMPORT!/tests/`.

## Next
Wired so far: the **forward-model** slice (sensory ⊕ task `Column`, place-invariance win ≈ **100%** on a place never trained,
`test_column_arithmetic`); the **decision** slice (basal-ganglia value SELECTION + thalamus relay/gate, learns a
context→action map from reward, `test_bg_thalamus`); and the **path-integration** slice (L6a's `ModularOperator` TRANSFORM
primitive — an action's effect learned in a small region dead-reckons correctly into NEVER-VISITED positions,
`test_operator_path_integration`). Each is minimal-but-real; each subsystem's RICHER role is deferred to a task that needs
it. Thicken from here (RULES.md #4 — always keep it runnable, ARCHITECTURE.md §3/§5.1):
1. **Join the two slices into one loop on a small GAME** — perceive (column) → predict / roll out → SELECT (BG) → GATE
   (thalamus) → act → reward. This needs a **TD value critic** (`reward.py` — multi-step value; the RPE the BG currently
   fakes as a centered immediate reward) + a tiny nav/decision env, and is where the BG + thalamus earn their full keep.
2. **The L4↔L6a feature-at-location loop.** L6a path integration is now DONE (the operator, above); the remaining spatial
   work is the ASSOCIATE side — L4 predicts the feature at the path-integrated location (`observe(feature, context=location)`
   / `predict_at(location)`), the loop that makes the model order-invariant (ARCHITECTURE §8). Then the operator's own
   frontier: non-abelian (heading-dependent, on the conjunctive code) + the context-gated obstacle override (both noted-
   deferred in `operator.py`). The SR (ROADMAP 3b) accumulates this operator into value.
3. **The thalamus's binding role** (content ⊗ location across two columns → place-value / cross-column VOTING) + the factored
   state via L4's basal `context=` channel; then **L2/3 recognition/voting**, **L5 motor**, **hippocampal rollout**. Each
   added as a task exercises it, driven by `agent.py`. Nothing counts until imported from `agent.py` AND the agent plays more
   than before (RULES.md #3). No single-column experiments, ever (ARCHITECTURE.md §5.1).

## How to answer "where are we?"
Run the reachability test (once it exists); it reports the wired module map. Keep this table equal to that output.
