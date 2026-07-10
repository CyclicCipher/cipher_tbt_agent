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
The rule-enforcing skeleton + the first two salvaged devices (2026-07-09).

| module | one-line job | wired? | its test |
|--------|--------------|--------|----------|
| `agent.py` | the live entry point / root of the loop — a `step(obs)→action` stub (raises until the first slice) | ROOT | `test_reachability` |
| `htm.py` | the ONE cortical-layer mechanism — HTM sequence memory (proximal SDR-in, pluggable basal context, apical, learn/predict/burst); every layer is an instance, distinguished by its basal context (L4/L2·3/L5/L6 map + BUILT-vs-NOT in the docstring). `sequence.py` MERGED in | STANDALONE¹ | `test_htm` |
| `encoders.py` | SDR transduction library (`SDR` + Scalar/Category/Grid/Multi/Conjunctive/SpatialPooler) — data ↔ overlap-bearing SDR — salvaged, unchanged | STANDALONE¹ | `test_encoders` |

¹ STANDALONE = allowlisted in `test_reachability` with a reason; salvaged devices under development, to be wired into a
column once generalization is solved. NOT "done" (RULE 3) — a component passing its own test is a half-step, not integration.

**Generalization investigation — RESOLVED 2026-07-09 (scratch experiments; devices still STANDALONE).** Two durable results,
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

Suite: **27 passed.** Run `python src/tests/test_reachability.py` for the wired map; the 20 legacy test files are archived
under `Legacy - DO NOT USE OR IMPORT!/tests/`.

## Next
**The TWO-COLUMN slice** (the generalization investigation above showed WHY: place-invariance needs a factored state carried
by a second column — `ARCHITECTURE.md` §7). Build the thinnest **MULTI-COLUMN** unit that turns the arithmetic demonstration
into real architecture: a **sensory column** (reads the feature at each location) ⊕ a **task/PFC column** (maintains the
travelling STATE, feeding the sensory column's basal `context=`), one shared operator walked across locations. Then thicken
toward the closed loop (RULES.md #4 + ARCHITECTURE.md §3/§5.1) — **hippocampal rollout** + **BG select** + **thalamus gate**,
driven by `agent.py` (plumbing only). Reuse legacy via RULES.md #5, wired end-to-end; nothing counts until imported from
`agent.py` AND the agent plays more than before (RULES.md #3). No single-column experiments, ever.

## How to answer "where are we?"
Run the reachability test (once it exists); it reports the wired module map. Keep this table equal to that output.
