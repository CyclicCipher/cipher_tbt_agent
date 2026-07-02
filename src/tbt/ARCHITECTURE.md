# ARCHITECTURE — the one model, the rules, the plan

*The single source of truth for the TBT agent. If the code and this document disagree, one of them is a bug. This
document must remain explainable, in full, to anyone fluent in the domain jargon — if it cannot, the architecture is
wrong, and the architecture is what changes. It supersedes the older plan web (`L6_NONABELIAN.md`, `MATH_PHASE.md`,
`VECTOR_NAV_PLAN.md`, `FORWARD_MODEL_PLAN.md`, `COLUMN_AUDIT.md`, `GROUNDING_PLAN.md`, …), demoted to background
references. There is one plan, here.*

---

## 1. The one mechanism

A single reusable **cortical column** learns the structure of any domain as a navigable **reference frame** and predicts
within it. It holds one **current location**, moves it by learned **operators**, binds **content** to locations, and
recognises **objects**. It **predicts the next observation given the current location and an action** — that is the
column's nature, not a bolt-on. Many identical columns, bound and voting through the thalamus, are the Thousand-Brains
consensus. A value critic and the basal ganglia turn prediction into goal-directed action. No hand-coded rules, no domain
priors: structure, content, value, and goals are learned online from the sparse signal.

**Forward modelling is not a separate system.** A column predicts one step by construction (path-integrate the location,
read the content there). "Forward modelling" is that same prediction *run forward without acting*; "planning over the map"
is the value read off the learned frame (§7). There is no `forward_model` object — only the column predicting, iterated.

## 2. The layers

**L6 — LOCATION (where).**
- *Structure:* an online, TD-learned **successor representation** over discovered states; its eigenvectors are the
  multi-scale, periodic **grid cells** (Stachenfeld 2017). One frame, learned; the innate hex grid is only an
  initial-state prior expressible within it, never a parallel code.
- *Function:* holds the **current location** (one code). **Path integration = applying L5's operator to it**
  (`location ← operator(action)·location`; the abelian phase-advance is the special case). Supplies the reference frame
  everything is indexed in, and the SR value that is the planning substrate (§7).

**L5 — OPERATOR / motor (how it changes; what to do).**
- *Structure:* one learned **operator per action** — a group-representation matrix. Translation (abelian, commuting) is
  the special case; rotations/orderings/constrained moves are the non-abelian general case (composition = matrix product).
- *Function:* **path-integrates** L6's location (apply the operator); **predicts** by carrying the location forward; is
  the **motor** — emits the action that brings about the predicted/desired state (active inference: predictions, not
  commands); is the **driver** — the inter-column message a higher-order thalamus relays.

**L4 — CONTENT / feature-at-location (what is here).**
- *Structure:* a **content codebook** learned online (label-free) + the **feature ⊗ location** binding; a
  pose/rotation-invariant feature descriptor.
- *Function:* bind the sensed content to the current L6 location; **read out** the predicted content at a location — the
  "what will I see" half of the column's prediction.

**L2/3 — OBJECT / identity (which object this is).**
- *Structure:* a **graph-memory of objects**, each an arrangement of content-at-displacements in its own frame.
- *Function:* **recognise** the object and **infer its pose** by incremental evidence voting, so a known object is
  recognised at a pose never seen; **group by structure** (boundaries emerge from prediction mismatch, never a colour/
  connected-component heuristic); vote laterally and across columns.

## 3. The subsystems

**Thalamus.** The inter-column router: it **binds content ⊗ location** into the conjunctive representation and relays L5's
driver messages, so columns modelling the same world **vote** toward a consensus (the higher-order thalamic loop of the
Thousand-Brains theory). It is how "many columns" becomes "one percept."

**Basal ganglia.** The **selector**. Given the candidate goal-states / actions and their values, it disinhibits the one to
pursue (default-closed Go/NoGo, dopamine-RPE-trained), with STN "hold-your-horses" commitment under conflict. It is the
*only* place arbitration between competing options is allowed, because it is the brain's arbitration organ (rule 4).

**Hippocampus.** Not a separate algorithm — **the same column mechanism applied to the global, allocentric frame** (the
whole world, not one object): grid × content bound across columns and episodes (TEM). Built once as the column algorithm,
it is inherited for allocentric world-modelling and cross-frame/episodic binding; a plain single-column task needs it only
when episodic memory or multi-frame binding does.

**Value / reward.** The domain-agnostic **critic**: expected future reward, learned online from the sparse score. **Cost
is negative value in the same currency** — a wall, a hazard, a slow tile, a risky tile are all points on one scalar
(walls are the `−∞` limit), so obstacles are *not* special objects. This one value, read over the learned frame, is the
planning substrate (§7), and its two components (pragmatic + epistemic) are what make explore vs exploit *emerge* rather
than switch.

## 4. The sensorimotor loop — how the model touches the world

The model is sensorimotor at its core: it does not passively classify a scene, it **acts to sense and senses to act**.
One cycle:

**(a) Sensory input.** The world emits a **sensory field** — in ARC-AGI-3, a 64×64 grid of up-to-16 colours — plus a
sparse scalar **score**, a terminal signal (WIN / GAME_OVER), and the set of currently available actions. No explicit
goal is given; the reward must be learned from the score.

**(b) Transduction (the retina).** A thin **peripheral** converts the raw field into the model's currency —
**feature-at-location**: the content (an L4 feature) present at a location (an L6 code). This is transduction only, like a
retina: it holds NO segmentation, NO object heuristic, NO domain logic (rules 4–5). It does not decide *what* an object is
or *where* to look — it only makes the raw field readable as features-at-locations.

**(c) What receives it, and what is done.** **L4** takes the feature and binds it at **L6**'s current location; **L2/3**
accumulates recognition evidence; **L6** corrects its current-location belief against what was sensed (the
predict-then-compare snap); the **critic** turns the score into value. The column had already *predicted* the feature it
would sense (its operator carried the location forward; L4 read the content there) — the **prediction error** is the
learning signal. The model identifies *itself* by **reafference** (von Holst): the part of the field that changes as its
own operator predicts is the **controllable** location (the "self"); world-caused change is not so predicted. Thus
controllability is *learned*, never a fovea-on-residual heuristic (rule 5).

**(d) Choosing to move.** The **motor (L5)** selects the action that best brings about the goal-state under the one EFE
value (§7) — pragmatic toward reward, epistemic toward what most resolves uncertainty; the **basal ganglia** select among
the candidate goals. This is **one mechanism for covert and overt movement**: shifting attention to sample more (a
saccade — active recognition) and taking a world action (move, click) are both **operators applied to the location**. The
model has no separate "where to look" and "what to do" systems.

**(e) Acting (the effector).** L5 emits the chosen action; a thin **motor organ** (another peripheral, at the SDK
boundary) maps the action name to the world's effector API — a movement, or a click whose `(x, y)` is read from the
goal-state's target location. The world returns the next field, and the cycle repeats.

**The body vs the brain.** The retina (b) and the motor organ (e) are the **peripherals** — transduction and effection —
thin, living at the SDK boundary, holding no cognition. Everything between them is the column. A peripheral that began
deciding what an object is, or which cell is "the agent," would be a load-bearing harness (rules 4–5); that decision is
the column's, made by recognition and reafference.

## 5. Glossary — one definition each (a second meaning is a bug)

- **location** `g` — the current point in the learned frame = the L6 SR code. The ONLY location representation.
- **operator** — the learned per-action transition on the location code (L5). The ONLY transition and path-integrator;
  translation is its special case.
- **content** `x` — the feature at a location (L4). Location-invariant.
- **feature-at-location** — the binding `g ⊗ x` (L4). The ONLY map.
- **prediction** — the column carrying the location forward (L5 operator) and reading the content there (L4). Forward
  modelling / rollout is this, iterated. There is no separate forward-model module.
- **object** — a learned frame of content-at-displacements, recognised by voting; pose inferred; boundaries from
  prediction mismatch. NOT a segment, a change-log, or a tracked mover.
- **prediction error / surprise** — predicted vs sensed content. The ONLY "something changed" signal; no stored change-log.
- **value** — expected future reward incl. cost, one currency (the critic). The ONLY value.
- **goal** — a target-state to bring about; the motor acts to fulfil it.
- **selection** — the basal ganglia choosing among goals/actions.
- **peripheral** — the retina (raw field → feature-at-location) and the motor organ (action → effector API); transduction
  and effection only, no cognition.

## 6. The five rules (development law; a violation is reverted, not documented around)

1. **No parallel systems — ever, including for experiments.** Exactly one way to path-integrate, one way to predict, one
   feature-at-location, one way a grid module learns its structure, one recogniser, one value, one selector. Manage
   comparison and risk with **git branches**, never by keeping two mechanisms in the tree. Two "complementary" mechanisms
   plus an **arbiter** (e.g. tabular-vs-forward, explore-vs-exploit as a hard switch, CA-vs-g×x) is a parallel system in
   disguise — collapse it into the one mechanism whose behaviour subsumes both cases.
2. **One definition per concept.** See §5. A new meaning for an existing word is a bug to fix.
3. **The column and the agent are thin coordinators.** They hold references + routing — never math or state. Every belief,
   map, operator, and value lives in a layer/module.
4. **No load-bearing harness, no domain-specific code, no special-casing, no ungrounded arbitration.** Nothing branches on
   which game/domain it is; the peripherals transduce only. Every arbitration must name the brain mechanism it implements
   (basal-ganglia selection, tonic-dopamine gain, STN commitment) or it is removed. Selection lives in the basal ganglia.
5. **No symbolic estimators, object heuristics, or change logs.** No hand-coded "what is an object / how to split it,"
   no Kalman-style tracker banks (fovea centroids, pose matrices, binned nodes in parallel), no dicts of "what changed."
   Structure is learned; change is carried by prediction error; the object is a recognition construct.

## 7. Planning — how the model decides (explore, exploit, act)

**What planning is in real brains.** Routine planning is *not* rollout. The **successor representation** already stores
the discounted future occupancy of the learned map, so value is a cheap read — `V = M·R`, a dot product over the frame —
and greedy-on-`V` follows the shortest path, warping around barriers (the geodesic). Deliberative rollout (vicarious
trial-and-error, hippocampal replay) is **sparing**, reserved for the novel/hard case; prioritised replay schedules the
value updates by gain × need (Mattar & Daw). (`reference_brain_planning`, `reference_exploration_replay`.)

**What planning is in TBT / our model.** The same: the column's learned frame (L6 SR) *is* the map; planning is the value
read off it. **One planner, one value.** The geodesic to a goal falls out of the SR; rollout is only the column's own
prediction (§1) iterated, used sparingly when the SR is insufficient — never a second, parallel planner.

**How it knows to explore vs exploit — one value, not a switch.** The critic's value is **Expected Free Energy**:
- **pragmatic** = expected reward on the way to the goal;
- **epistemic** = expected information gain, grounded by **epiplexity** (learning-*progress*, so it → 0 for both
  irreducible noise *and* mastered structure — no noisy-TV trap, no separate gate).

The policy maximises the one EFE value. **Exploit emerges** where the pragmatic term dominates (reward is reachable);
**explore emerges** where the epistemic term dominates (learnable uncertainty remains). Directed exploration is the
**SR-eigenvector eigenpurpose** — a gradient toward under-explored bottlenecks — which is *part of* the epistemic term,
not a separate mode. There is no `g`-gate and no `V`/`V_exploit` split; those were a two-mechanism arbitration (a rule-1
violation, a P0 target). (`reference_efe_and_epiplexity`, `reference_animal_exploration`, `reference_eigenoptions_subgoals`.)

**Acting = testing a hypothesis.** A goal-state is a hypothesis "bring about X." The agent plans to X (the value/geodesic
above), the motor achieves it, and the **outcome** (reward = pragmatic, prediction-error = epistemic) confirms or refutes;
the basal ganglia commit through the maneuver and switch on repeated refutation. Testing a hypothesis *is* planning.

## 8. Hypothesis generation — how the model proposes what to try (the frontier)

Testing is §7; **generation** — where a candidate target-state comes from — is the genuinely open problem. The proposal,
from the research and the number-domain probes (`MATH_PHASE.md`):

- **Not enumeration.** The mind does not score the combinatorial space; it **samples a few candidates from memory**, cued
  by context and biased by priors: **salience × controllability × ambiguity** (Dasgupta, Schulz & Gershman 2017;
  resource-rational, more samples when more uncertain). Controllability = "it moved when I acted" (a learned affordance,
  the same reafference of §4c); salience = novelty / prediction-error; both learned, never a colour/domain rule.
  (`reference_hypothesis_generation`.)

- **A hypothesis is a short composition of learned operators toward a cued target** — geodesic-finding in the learned
  structure. The **master boundary** (from `MATH_PHASE`) predicts its cost: where the structure is **free/abelian**, the
  hypothesis is **READ OFF** — a homomorphism is fixed by its action on the generators, so the extension to all
  compositions is forced and cheap (`+b` = `b` succession steps; a straight-line vector to a goal). Where the structure is
  **relational / quotient** (non-commuting, constrained — Sokoban-with-walls, carry), it must be **SEARCHED**. So
  generation = *read off where the structure is free, search where it is relational*, over targets the priors cue.

- **What is proposed vs open.** Proposed and grounded: the shape (sample cued target-states; test by achieving; EFE
  selects; read-off vs search set by the structure). **Open (honestly): (a)** learning the priors that cue *which*
  targets to sample (must stay learned, not hand-coded), and **(b)** whether the relational **search** is tractable at
  scale (the carry / Sokoban case). These are what the `MATH_PHASE` microworlds exist to probe; no code commits to a
  solution until they answer.

## 9. The plan — one dependency-ordered spine

- **P0 — Converge the code to this document (mostly DELETION).** Collapse every parallel system and estimator into the
  one mechanism: one prediction (delete the location-blind CA; the operator-over-content-at-location is the one); one
  location = the L6 code path-integrated by the operator (delete the fovea / pose-matrix / `state_node` / `_obs` /
  `heading_dependent` fork); one value (fold `V`/`V_exploit` + the `g`-gate + the `_tab_spread` tabular/forward arbiter
  into the single EFE value; move `cost` into the critic); object = recognition construct (delete the segmentation
  heuristic + `object_state`/`_changed`); thin column + agent (subsystems → layers); the retina/motor-organ reduced to
  transduction/effection peripherals. Suite-green throughout; git branches for risk. **This is the bulk of the work.**
- **P1 — Factored perception.** L2/3 recognition + L4 content deliver `(location, content)` factored, from the live field
  — the prerequisite prediction always assumed and never had.
- **P2 — Prediction over the factored representation.** Trivial once P1 exists (the column's §1 prediction, now with a
  clean content). This is the old, tangled "c."
- **P3 — Relations & planning.** Operators (learned) → relations by loop closure → the SR-value geodesic over the learned
  frame → the order/config-dependent case (Sokoban).
- **P4 — Hypothesis generation (§8).** The proposal made live: cue target-states, achieve, confirm; the heterarchy
  (multi-column voting via the thalamus) scales the same loop.

Honest status: the **operator** primitive and **relation/factor discovery** (P3 middle) exist and are tested; the SR is
the one L6. Everything else is entangled with the P0 estimator/arbiter stack. **We are at P0.**

## 10. Acceptance test for every change (the paper test)

Both must hold, or the change does not land:
1. **Explainable** in one sentence that fits §1–§5 (no new term, no second meaning, no "well, in this mode…").
2. **Obeys the five rules** (no parallel system, no second definition, no coordinator bloat, no harness/special-case/
   ungrounded arbitration, no symbolic estimator/heuristic/change-log).

If a change cannot pass both, the design is wrong — fix the design, not the change.
