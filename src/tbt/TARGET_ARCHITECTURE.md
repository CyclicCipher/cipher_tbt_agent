# TARGET ARCHITECTURE — the agent as ONE column over the game-as-object

The north star. Written 2026-06-28 after the live public games (ls20, cn04, lp85, sk48, ft09, vc33, tu93)
broke every assumption in the perception scaffolding (action=delta, single-cell body, static frame, movement-only,
spatial goal). The lesson is not "add those cases" — it is "assert nothing." This document is the target we
reduce toward; it is not yet built.

## Thesis

**An ARC-AGI-3 game IS a sensorimotor object, and one cortical column learns it the way it learns any object:**
by acting and observing how the raw input transforms. The column already has the faculties (L4/L5/L6/L23, the
dynamics predicate-search, `factorize`, `residual`, the recurrence, the SR frame). The scaffolding
(`DynamicsPerceiver`, `ObjectPerceiver`, `GoalModel`, `WorldLearner`, `NeocortexPlanner`, `segment`-as-perception,
the role schema) *bypassed* those faculties with hand-coded pre-processing, and every broken assumption lives in
that pre-processing. The work is **reduction**: delete the scaffolding, feed the column raw data, let the roles
emerge.

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
RFs sensing them. The structural observation that stays stable while pixels scroll falls out of the voting.

**Forward look (not for ARC now).** This same system — a learned policy over what to attend to next, with
pose-relative binding — is structurally the attention mechanism of a sequence model. After ARC it is a candidate
substrate for a reasoning model: attention as a learned sensorimotor saccade over a learned reference frame.

## How each faculty consumes raw data and what EMERGES

- **Factor the local observations (`factorize.py`).** Disentangle the *retina's RF streams* (which recur) into
  independent coordinates by how they transform under the agent's actions (Higgins; Locatello: factors are only
  identifiable *with* action, never from
  statics). The **body** is just the factor the agent's actions move; the **view/camera offset** is the factor that
  shifts globally (ls20/cn04 scrolling becomes one coordinate, not a wall); **objects** are the remaining factors;
  the **score** is a factor. None of these are named or assumed — they are discovered. This is the piece that makes
  "no assumptions" possible, and it already exists for the 2-factor case.
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

Two things let us avoid Monty's ceiling while keeping its principles: (a) the **SR eigenframe is topology-general**
— grid-like on metric factors, correct on rings/trees/graphs — so reference frames need no Euclidean (or 3-D)
assumption; and (b) **value** — the score-driven active-inference drive Monty has no analogue of, which is what
turns a recogniser into a goal-seeker. We keep Monty's evidence accumulation and voting, and its insight that an
object is sensorimotor; we drop its fixed Euclidean frames and we add value. The location-scaling that looked like
a wall (the 2.9 s eigh) is a symptom of mapping pixels; over **factored** states the SR frame is small and cheap,
and an online TD-SR removes the batch eigh entirely.

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
  recur ~99%, local transitions ~95%.** The risk has moved *downstream*, not away.
- **Composing local into global is the new central risk.** Per-RF columns see local pieces; binding their votes
  (relative-pose CMP) into the body / view / object factors and a coherent game-state, sample-efficiently, is the
  unbuilt R9/R11 voting work. `factorize` over the RF streams must still recover the *right* global factors.
- **The saccade policy** (active inference + the exogenous-salience bootstrap) is new machinery; a poor policy
  wastes compute or never fixates the structure that matters.
- **The two-motor integration** — saccade-learning the layout + action-learning the dynamics + value — is more than
  Monty attempts; combining them into one agent is unproven.
- **Discrete coordinates vs continuous structure** (positions, scroll offsets) for the predicate search; TD-SR
  convergence within budget; value/active-inference integration. All real, all downstream of a working retina.
- We still have **not solved a real game**; the milestone is the local → factor → operator → value loop completing
  one.

## Reduction plan (staged, each stage deletes a scaffold and re-validates on replica + a real game)

0. **The retina front-end** (validated): narrow RF columns (≈5×5, ARC-calibrated) over the frame, each emitting
   `(features, pose)`; local recurrence ~99%, local transitions ~95%, vs 0 for the global frame. Add the **saccade
   motor** — an exogenous-salience bootstrap plus a learned active-inference policy — so the agent reads each frame
   in compute, not game actions.
1. **Factor the local RF streams** (`factorize` over the retina's recurring observations, bound by relative-pose
   CMP voting): confirm the body, the view/scroll offset, and objects emerge as global factors on ls20/cn04. The
   enabler; deletes the perceivers' assumptions.
2. **Learn the per-action path-integration operators over the factors** (`residual` / the dynamics faculty): each
   action becomes a context-dependent `M_a`; validate by the **A∘B test** — predict held-out transitions on ls20
   (opposite-direction) and a click game. Action semantics learned, none assumed.
3. **Locate in factored-state space** with the SR frame, made online: cheap over few factors, correct on the replica
   (regression), stable under a scrolling view.
4. **Active inference**: value (pragmatic from the score factor) + epistemic (operator prediction error) replacing
   `_explore` and the role planner — the first attempt to *solve* a real game.
5. **Delete the scaffolding; `perception/` = the thin retina sensor.** The 82-test replica suite stays green as a
   regression guard; the real games are the truth.

Reductions are where every previous wall fell (SR frame, pose-invariant recognition, recursive residual). This is
the same move at the scale of the whole agent.
