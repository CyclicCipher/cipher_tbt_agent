# TARGET ARCHITECTURE — the agent as ONE column over the game-as-object

The north star. Written 2026-06-28 after the live public games (ls20, cn04, lp85, sk48, ft09, vc33, tu93)
broke every assumption in the perception scaffolding (action=delta, single-cell body, static frame, movement-only,
spatial goal). The lesson is not "add those cases" — it is "assert nothing." This document is the target we
reduce toward. The front-end (retina → agency → events → objects) is now BUILT and live-validated on the real games
(see **STATUS** below); the value + planning loop that turns it into a *playing* agent is what remains.

## Thesis

**An ARC-AGI-3 game IS a sensorimotor object, and one cortical column learns it the way it learns any object:**
by acting and observing how the raw input transforms. The column already has the faculties (L4/L5/L6/L23, the
dynamics predicate-search, `factorize`, `residual`, the recurrence, the SR frame). The scaffolding
(`DynamicsPerceiver`, `ObjectPerceiver`, `GoalModel`, `WorldLearner`, `NeocortexPlanner`, `segment`-as-perception,
the role schema) *bypassed* those faculties with hand-coded pre-processing, and every broken assumption lives in
that pre-processing. The work is **reduction**: delete the scaffolding, feed the column raw data, let the roles
emerge.

## STATUS — built and validated (2026-06-28)

The front-end is BUILT, live-validated on the real games, and committed (suite 99/99; the old agent + replica suite
untouched as the regression baseline). Four pure-stdlib modules in `tbt/`, each one concept, no assumptions:

- **`retina.py`** — narrow RF sensor: raw frame → recurring `(feature, pose)` observations via a label-free patch
  vocabulary, plus the exogenous-attention residual. *Validated: ls20's 5,904 RF sweeps collapse to a 24-pattern
  vocabulary; the global frame recurs 0%.*
- **`agency.py`** — the controllable self from controllability alone (no body assumption); the static/dynamic split.
  *Validated: controllability 1.00 on ls20/cn04.*
- **`events.py`** — event-boundary segmentation by the reafference principle (a change the action can't explain),
  excluded from operator learning. *Validated: caught ls20's full-frame scene-cuts that the API lifecycle flag
  MISSED.*
- **`objects.py`** — object tracking by POSE with permanence (a boundary resets linkage). *THE breakthrough:
  recovers a clean, LEARNED, action-dependent per-action operator (cn04: ACTION1=up, ACTION2=down — discovered, not
  assumed) where cell-level analysis found 0 action-selective cells.*

The validated chain, no assumptions: **raw frame → retina (recurring local features) → agency (controllability) →
events (clean boundaries) → objects (per-action operators over pose).** It fixes the original failure outright: the
old agent hard-coded ACTION1=up and died; this LEARNS each action's real effect from the object's pose response.

### Distilled learnings (what the experiments taught — they reshaped the plan)
1. **Sense locally, never globally.** Global frames never recur (0 revisits, 0/16 reversible); 5×5 RFs recur ~99%.
2. **The self is an OBJECT tracked by POSE, not pixels.** A moving self spreads its changes thinly, so NO cell
   carries a stable signature (measured: 0 action-selective cells); only the object's pose has an action-dependent
   operator. This is *why* TBT/Monty is object-and-pose-based — confirmed empirically, the hard way.
3. **The games are mostly-static layout + a small localized dynamic residual** (~3.5% of cells), NOT scrolling (the
   earlier "scrolling" read was a `detect_motion` artefact; the dominant global shift is (0,0)).
4. **Event boundaries are reafference, not the lifecycle flag.** The API's GAME_OVER/WIN flag MISSES full-frame
   scene-cuts inside NOT_FINISHED; "a change the action can't explain" catches them. The magnitude proxy is a
   bootstrap for the proper prediction-error test, which the learned operator supplies (the two co-bootstrap).
5. **The ACTION→effect mapping is LEARNED, not assumed** — the hand-coded ACTION1=up was the original sin.

### What is next (the path to a PLAYING agent — not yet done)
We can now SEE the dynamics (the operators); we have not yet PLAYED a game. In order:
- **Operator as forward model** + upgrade `events.py` from magnitude to prediction-error (the co-bootstrap above).
- **Group / factor objects** (cn04's many objects move together — one controllable group? — and separate self from
  autonomous, which pose-spread already begins).
- **Value / goal** — tie the score to object-states (which configurations score; `reward.py`).
- **Plan** — the achiever (`neocortex.py`) over the operators toward the goal (active inference), then **assemble
  into a loop and test whether it completes a real level.** That is the true milestone.

NB on the original plan: stage 1 said "`factorize` over RF streams." What actually worked for the DYNAMIC factors is
**object-pose tracking** (`objects.py`) — the objects ARE the dynamic factors, and the per-action operator over
their pose IS the `residual`/dynamics. `factorize` (action-orbit disentanglement) still applies once states recur;
object tracking is how recurrence is achieved for a moving thing. The architecture holds; the concrete mechanism is
object-pose, not raw-frame factoring.

NB on principle 2 (one column): the four modules are validated MECHANISMS, built standalone to de-risk them cheaply.
`retina.py` is the sensor (Monty's sensor module) and rightly stays separate; but **`agency.py` / `events.py` /
`objects.py` are column FACULTIES** — object tracking IS the learning module, the event boundary IS the column's
prediction error (`residual`), the self IS the factor the operators move. Folding them INTO the `CorticalColumn` as
we assemble the playing agent is what realizes "one column type" and keeps them from hardening into a NEW
scaffolding. That fold-in is the spine of the next phase, not an afterthought.

## The five principles → the mechanism

1. **Only raw data, sensed locally.** The column consumes exactly what ARC-AGI-3 returns — the frame (N grids
   ≤64×64, ints 0–15), the score (levels_completed), the state, the available actions, the action issued — with no
   `segment`/`modal_background`/`detect_motion`/role decode. But "raw" cannot mean "the global frame is the state":
   global frames never recur (measured: 0 revisits, 0/16 reversible on ls20), so there is no structure to learn. It
   means **sensed through narrow receptive fields** (the retina, below), where local patterns recur ~99%.
2. **One column type, many instances.** There already is one `CorticalColumn`. Use it for everything — perception,
   dynamics, value, recognition — specialised only by what each instance is connected to (Mountcastle). Multiple
   instances over different sensors/scales/views vote via the thalamus (CMP).
3. **All core logic in `tbt/`.** `perception/` shrinks to a thin sensor: raw `FrameData` → arrays, nothing more.
4. **No dynamics/perceiver/learner/exploration/role scripts.** Each is already a column faculty; dissolve the
   scaffolding into it.
5. **Robust by making NO assumptions**, not by adding cases. Below, nothing asserts what an action does, what the
   body is, whether the view is stable, or what the goal is.

## The unifying operation — path integration in a learned reference frame

One operation underlies the world-model half of everything below: **location ← operator(location), composed along
the trajectory.** A reference frame is just a set of per-action operators and how they compose.

- The 2-D hex grid (`L6_GridLocation`) is only the special case: location = plane-wave phases, operator = a phase
  rotation by a displacement `(dx,dy)`. It bakes in "metric and Euclidean."
- The **general** case is already in the column (`loc_move`): location = the SR-frame place code (L6), operator =
  the learned per-action matrix `M_a = Σ place(s')⊗place(s)` (L5); path integration is `h ← M_a·h` (matrix
  products), drift-corrected by the recurrence (`loc_sense`). Learned from observed `(s, a, s')`, so it holds on
  ANY topology with no metric assumption.
- They unify because **the SR eigenframe IS the grid on metric spaces (Stachenfeld)** and the correct frame on any
  other topology. So operator-composition path integration *reduces to* the hex grid on metric factors and
  *extends* to everything else — one mechanism, no Euclidean assumption.

Per factor the operator's KIND falls out of the transitions, never declared: a **position** factor → a uniform
translation (recovers exact, scale-free grid path integration); a **discrete/toggle** factor → a permutation/edge
map; a **context-gated** effect (a door) → a *state-dependent* operator, which is exactly what `residual.py` learns.
The whole game-state path-integrates as the product over factors.

Through this lens the faculties are one thing: the **forward model / dynamics** (apply `M_a`), the **self model**
(self = the factor your operators move — no body assumed), the **planner** (compose operators over hypothetical
actions), and **recognition / localisation** (`loc_where` + `loc_sense`) are all path integration in the learned
frame. That is most of the deleted harness, in one operation.

**Two boundaries, stated honestly (where the lens stops):**
1. **Path integration gives the map, not the values** — where actions lead, never which places are good.
   Goal-direction is the score-driven value layered on top (active inference, `reward.py`). This is precisely what
   Monty lacks and why a Monty-style system can't do a goal task like ARC; the world model is path integration,
   **value/goal is the separate, deliberate addition.**
2. **The operators must compose consistently AND generalise from few samples** — the A∘B problem ("consistency ≠
   correctness"): `M_a` path-integrates *observed* edges perfectly but must predict *unobserved* state-actions to be
   a real model, inside the RHAE action budget. It generalises only as well as the **factorisation** is regular:
   good factors → low-D, group-structured operators that generalise from a handful of moves; bad factors → a
   memorised edge table that drifts. This couples directly to disentanglement and is the make-or-break.

## The retina and the limbs — the sensorimotor front-end (validated 2026-06-28)

The path-integration spine needs **recurring** states. The frame doesn't provide them: raw 64×64 frames never recur
(0 revisits, 0/16 reversible), so `factorize`/the SR frame have no orbits to find. The fix is to sense **locally** —
which is why TBT is sensorimotor in the first place — and the live games confirm it:

- **Local receptive fields recur where global frames don't.** Distinct-patch recurrence over 41 ls20 frames: 5×5 →
  **0.994** (945 distinct of 147,600), 8×8 → 0.982, 16×16 → 0.913, 32×32 → 0.737, 64×64 → **0.000**. cn04 matches.
  A whole game is a vocabulary of a few hundred to ~2,400 local patterns.
- **Structured (non-background) patches recur just as hard** (5×5 → 0.985) — real local content, not blank space.
- **Local TRANSITIONS recur:** `(patch, action, next_patch)` at 5×5 has recurrence **0.95** (117 distinct of 2,560).
  The per-action operators are locally consistent, hence learnable from few samples — the thing absent globally.
- **The localized dynamics are coherent, controllable objects.** The per-action change is only ~3.5% of the frame
  (the rest is a static recurring layout, no global scroll), forms 1–5 connected components with a dominant large
  one, and is locally reversible by the inverse action on ls20/cn04 (restore 0.64–1.00) even though the *global*
  frame is 0/16 reversible — so a controllable "self" is recoverable from the residual, separable from an
  irreversible background element. (sk48 is irreversible: coherent but one-way — a learnable operator with no
  inverse, fine for forward planning.)

So the front-end is a **retina of receptive fields**, and it breaks both walls at once: recurrence (local patterns
repeat) and scale (many tiny columns over ~5×5 RFs, not one 4096-cell column whose eigh cost 2.9 s).

**The retina.** Tile the frame with narrow receptive fields, **one `CorticalColumn` per RF** (Monty: 1:1 sensor ↔
learning module; HTM: one column per potential pool). Each RF column emits `(features, pose)` into the CMP. RF size
≈ **5×5–8×8** is the ARC-calibrated sweet spot (≥98% recurrence, ~1k-pattern vocabulary, enough content to carry a
feature). **Overlap** is optional (Monty tiles without it; cortex/retina overlap, more at the fovea) — relations come
from the *relative pose* between RFs, not from overlap, but overlap buys coverage-continuity and redundancy, so we
lean to a modest overlap. RF size / overlap / foveation are **hyperparameters calibrated to ARC** (re-tuned per
application when this is later deployed on a real-world task), not first principles.

**The limbs — two motors (the second is beyond Monty).**
1. **Saccade / attention motor (free).** The agent moves its RFs over the *static* current frame, path-integrating
   the sensor pose to build the frame's spatial model — Monty's loop, which Monty only does over static objects.
   Saccades cost **compute, not game actions**, so the agent reads each frame exhaustively before spending a move.
   The policy is a **learned active-inference** one (saccade where prediction error / hypothesis entropy is highest =
   epistemic), **bootstrapped by a bio-inspired exogenous channel** (automatic capture by salience: motion, novelty,
   edges); the endogenous (learned, voluntary) and exogenous channels influence each other, as in biological
   attention.
2. **Game-action motor (costly).** ACTION1–6 *transform the world*; the agent learns the per-action operators from
   how the sensed local structure moves. Monty has no analogue (it never manipulates the world) — our interactive
   extension, and the reason we also need value, which Monty lacks.

The split maps onto the RHAE economics exactly: **compute is generous (saccade and plan freely), game actions are
scarce (5× human median).**

**Relations = relative pose (the CMP vote).** Columns vote not just "this feature" but "where my neighbour should be
sensing, by our relative displacement" — so a relation between two local features IS the relative pose between the
RFs sensing them. The static layout (which is most of the frame) stays stable while the small dynamic residual moves.

**Forward look (not for ARC now).** This same system — a learned policy over what to attend to next, with
pose-relative binding — is structurally the attention mechanism of a sequence model. After ARC it is a candidate
substrate for a reasoning model: attention as a learned sensorimotor saccade over a learned reference frame.

## How each faculty consumes raw data and what EMERGES

- **Factor the local observations (`factorize.py`).** Disentangle the *retina's RF streams* (which recur) into
  independent coordinates by how they transform under the agent's actions (Higgins; Locatello: factors are only
  identifiable *with* action, never from
  statics). Measured reality on ls20/cn04/sk48: the frame is a **mostly-static recurring layout** (≈97% unchanged
  per action) plus a **small localized dynamic residual** (~3.5% of cells) — so the factors are the static layout
  and the few moving things in the residual. The **body** is the factor the agent's actions move; **objects** are
  the other moving factors; the **score** is a factor; a **view/scroll offset**, *if* a game has one, is just one
  more coordinate (the shared displacement across all RFs) — but it is ≈absent on the games measured so far. None of
  these are named or assumed — they are discovered. This is the piece that makes "no assumptions" possible, and it
  already exists for the 2-factor case.
- **Learn each action's effect (`residual.py` + the column's `observe_effect`/`learn_dynamics`).** Over the factored
  coordinates, learn per action a decision list of `(predicate, delta)` rules — `ACTION_k` is *whatever it does*,
  context-dependent (a precondition gates an effect), with an MDL stop instead of a per-state lookup. No
  action→direction map, no movement assumption (a coordinate/click action is just an action whose effect is learned;
  ACTION6's (x,y) is part of its parameterisation). This dissolves `DynamicsPerceiver` and `ObjectPerceiver`.
- **Locate yourself in the game (L6, made online).** "Where am I" = a place code over the *factored state space*,
  the successor-representation frame (general topology, no Euclidean assumption — unlike Monty's 3-D frames and
  unlike our hex `L6_GridLocation`). **The 2.9 s eigh was a symptom of mapping 4096 pixel-cells**; over a handful of
  factored coordinates the state graph is tiny, so the SR frame is cheap — and for streaming we move to an online
  **TD successor representation** (incremental, no batch eigh). The recurrence (`loc_move`/`loc_sense`) tracks
  position online, as now.
- **Content + memory (L4/L23).** Factor-values bound to place codes in the one shared memory, read back by unbinding.
- **Value + goal + exploration = active inference (`reward.py`, folded in).** The score factor makes score-raising
  states **pragmatically** valuable; the transition model's **prediction error** is **epistemic** value (this is
  exploration — no `_explore` script, no random ε). One drive, planned by rolling the learned model forward (the
  achiever stays). Goals are discovered (the score), never assumed spatial.
- **Multiple columns vote (`thalamus.py`/`basal_ganglia.py`).** Identical columns over different sensors/scales
  exchange beliefs over (state, factors, action) via the CMP; consensus + emergent allocation, as in Monty's voting.

## What we know that Monty doesn't

Three things let us avoid Monty's ceiling while keeping its principles: (a) the **SR eigenframe is topology-general**
— grid-like on metric factors, correct on rings/trees/graphs — so reference frames need no Euclidean (or 3-D)
assumption; (b) **value** — the score-driven active-inference drive Monty has no analogue of, which is what turns a
recogniser into a goal-seeker; and (c) a **frame-agnostic CMP**. Monty's cortical messaging protocol passes a 3-D
Euclidean `(object, pose)`, and its relational vote ("where you should be sensing, by our relative displacement")
works only *because every learning module shares that one Euclidean frame* — a real simplification of the theory,
which says columns model things in their own, possibly abstract, frames. Our message is **not a metric pose but a
high-dimensional code bound in a shared vector space**: each column projects its own-frame place code into the
shared `d_mem` slot (`column.py`) and the **thalamus binds by VSA conjunction** (`thalamus.py`), so consensus is
*vector agreement*, not pose-matching. Identity is always cross-frame-shareable; Monty's relative-displacement vote
is the metric special case. We keep Monty's evidence accumulation, voting, and the sensorimotor-object insight; we
drop its fixed Euclidean frames, add value, and bind heterogeneous frames in a shared space. The location-scaling
that looked like a wall (the 2.9 s eigh) is a symptom of mapping pixels; over **factored** states the SR frame is
small and cheap, and an online TD-SR removes the batch eigh entirely.

## The dissolution plan — what gets deleted, and what absorbs its job

Nothing is deleted cold. Each scaffold is dissolved by moving its job into a column faculty, re-validated against the
replica (regression) AND a real game, then removed. In dependency order:

- **`ObjectPerceiver` (`perceive.py`)** — learns body/pushable/blocking/walkable/consumable from motion, assuming the
  body is the single colour that moves by the issued 1-cell delta. → **`factorize`** (the body, the view, and objects
  are factors) + **`residual`** (how each object-factor responds to an action is its operator; "pushable"/"blocking"
  are contact-conditioned operators, not labels). *Deleted once factors + their operators reproduce the replica's
  role behaviour.*
- **`DynamicsPerceiver` (`perceive.py`)** — pre-digests `(prev, action, frame)` into (stepped-on colour, presence
  bits, effect), assuming the efference copy is colour-moves-by-delta and an effect is a colour vanish/appear or
  death/score. → fed nothing pre-digested: the column's **own** dynamics faculty (`observe_effect`/`learn_dynamics`/
  `predict_effect`) consumes **raw factored transitions** — features = the factor coordinates, effect = a factor
  delta. *Deleted once `learn_dynamics` runs on factors directly.*
- **`GoalModel` (`perceive.py`)** — goal colour from a score increment + a required-absent context, assuming a
  spatial/colour goal. → the **score is a factor**; pragmatic **value** (`reward.py`) labels score-raising states; the
  conjunctive "required-absent" is just which factor-configurations score. *Deleted once value is learned over the
  score factor.*
- **`WorldLearner` (`learn.py`)** — bundles the perceivers and decodes a role `WorldModel`. → the **column IS the
  learner** (`observe`/`consolidate`/`learn_dynamics`); there is no decode step. *Deleted with the perceivers.*
- **`NeocortexPlanner` (`control.py`)** — the role decode, `_explore`, and the forward-model glue. → **active
  inference** over the column's path-integration model + value (pragmatic + epistemic); the achiever (`neocortex.py`)
  stays, renamed honestly. *Deleted once active inference plans on the column's own model.*
- **`scene.py` role schema (`WorldModel`: body/pushable/blocking/goal)** and **`segment`-as-perception** → no role
  labels; objects are factors. `perception/` shrinks to a **thin sensor**: raw `FrameData` → arrays. (Connected-
  component objectness is a defensible Core-Knowledge prior; if kept, it is a cheap *sensor primitive* that proposes
  candidate factors, never a role decoder.) *Deleted last, once nothing reads roles.*
- **`recognize.py`** — evidence-based recognition over object point-clouds. → the **same evidence loop folded into the
  column** as state/game recognition ("which state am I in, which game is this") over factors. *Kept as a principle,
  dissolved as a separate script.*
- **The hand-coded assumptions** (`ACTION1=up`, single-cell body, static frame, spatial goal) → gone, because no
  faculty above states them.

Target: fewer files than today, and the broken assumptions cannot recur because nothing states them.

## Honest risks (where this can fail)

- **Recurrence — the wall that killed the global-frame plan — is now solved at the front-end (measured): local RFs
  recur ~99%, local transitions ~95%.** And the "cell vs object" question is resolved: **object-pose tracking
  recovers the operators** where cells could not. The risk has moved *downstream*, not away.
- **The dynamics are now recoverable — but turning them into a PLAYING agent is the live risk.** The operators are
  noisy (pose = an approximate centroid), the objects may need grouping (cn04's many movers), and value + planning
  + assembly are unbuilt. We can *see* the dynamics; making the agent *act* on them within the RHAE budget, and
  completing even one real level, is the unproven milestone.
- **Composing local into global is the new central risk — and it IS the CMP problem.** Per-RF columns see local
  pieces; binding their votes into the body / object factors and a coherent game-state, sample-efficiently, is the
  unbuilt R9/R11 voting work. The hard part is **learned cross-frame registration**: columns with *different* frames
  must learn how their frames correspond (Hebbian/co-occurrence binding in the shared `d_mem` space), where today
  the remap slots are merely *random*-orthogonal (coexisting, not aligned). Whether that binding converges to a
  self-consistent global state across many columns is the A∘B / consistency question. (Mechanism + Monty contrast:
  see "What we know that Monty doesn't". Deferred until the retina's many columns must vote; the single-object
  factoring below needs only one column.)
- **The saccade policy** (active inference + the exogenous-salience bootstrap) is new machinery; a poor policy
  wastes compute or never fixates the structure that matters.
- **The two-motor integration** — saccade-learning the layout + action-learning the dynamics + value — is more than
  Monty attempts; combining them into one agent is unproven.
- **Discrete coordinates vs continuous structure** (object positions, any view offset) for the predicate search;
  TD-SR convergence within budget; value/active-inference integration. All real, all downstream of a working retina.
- We still have **not solved a real game**; the milestone is the local → factor → operator → value loop completing
  one.

## Reduction plan (staged, each stage deletes a scaffold and re-validates on replica + a real game)

0. **The retina front-end — DONE** (`retina.py`, validated): RF features recur ~99%, local transitions ~95%, vs 0%
   for the global frame. The saccade motor's exogenous channel (`agency.py` salience) is in; the learned
   active-inference saccade policy comes later.
1. **Separate static layout from the dynamic objects — DONE** (`agency.py` / `events.py` / `objects.py`):
   controllability 1.00, clean event boundaries (reafference), objects tracked by pose with permanence.
   (Object-pose tracking replaced raw-frame `factorize` for the dynamic factors — see the STATUS note.)
2. **Learn the per-action operators — DONE** (`objects.py`): action-dependent per-action motion recovered, LEARNED
   not assumed (cn04: ACTION1=up). Still to stress-test: A∘B generalisation and a click game.
3. **Operator as forward model + value + planning — NEXT (the playing agent).** Use the operators to predict (and
   upgrade `events.py` magnitude → prediction-error); tie the score to object-states (`reward.py`); plan with the
   achiever (`neocortex.py`) toward the goal (active inference); ASSEMBLE into a loop and test it completes a real
   level. *This is the true milestone we have not reached.*
4. **Group / factor objects + online SR location** as the games demand (cn04's many objects move together; the SR
   frame over the few object-states, online).
5. **Delete the scaffolding; `perception/` = the thin retina sensor.** The replica suite stays green as a
   regression guard; the real games are the truth.

Reductions are where every previous wall fell (SR frame, pose-invariant recognition, recursive residual). This is
the same move at the scale of the whole agent.
