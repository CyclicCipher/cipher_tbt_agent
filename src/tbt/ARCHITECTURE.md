# ARCHITECTURE — the one model, the rules, the plan

*The single source of truth for the TBT agent. If the code and this document disagree, one of them is a bug. This
document must remain explainable, in full, to anyone fluent in the domain jargon — if it cannot, the architecture is
wrong, and the architecture is what changes. It supersedes the older plan web (`L6_NONABELIAN.md`, `MATH_PHASE.md`,
`VECTOR_NAV_PLAN.md`, `FORWARD_MODEL_PLAN.md`, `COLUMN_AUDIT.md`, `GROUNDING_PLAN.md`, …), demoted to background
references. There is one plan, here.*

---

## 1. The one mechanism

A single reusable **cortical column** learns the structure of any domain as a navigable **reference frame** and predicts
within it. It holds one **current location**, moves it by learned **operators** (each a *displacement*), binds **content**
to locations, recognises **objects**, and learns their **behaviors** (how they move and change). It **predicts the next
observation given the current location and a displacement** — that is the column's nature, not a bolt-on. Many identical
columns, bound and voting through the thalamus, are the Thousand-Brains consensus. A value critic and the basal ganglia
turn prediction into goal-directed action. No hand-coded rules, no domain priors: structure, content, value, and goals
are learned online from the sparse signal.

**Forward modelling is not a separate system** (§5). A column predicts one step by construction (apply a displacement to
the location, read the content there). Forward modelling is that prediction *run forward*; the next displacement is
supplied by your own **efference** (self-motion) or by a learned **behavior** (another object's dynamics) — the same
prediction, different driver. "Planning over the map" is the value read off the learned frame (§8). There is no
`forward_model` object.

## 2. The layers

**L6 — LOCATION (where).**
- *Structure:* an online, TD-learned **successor representation** over discovered states; its eigenvectors are the
  multi-scale, periodic **grid cells** (Stachenfeld 2017). One frame, learned; the innate hex grid is only an
  initial-state prior expressible within it, never a parallel code.
- *Function:* holds the **current location** (one code). **Path integration = applying L5's operator to it**
  (`location ← operator(action)·location`; the abelian phase-advance is the special case). The **SR is also L6's temporal
  structure** — a *predictive map* of future occupancy (Stachenfeld's predictive map), used for planning (§8). Note: this
  is L6's *only* temporal structure — L6 does **not** hold sequence memory (§5); adding one would be a parallel system.

**L5 — OPERATOR / motor / displacement (how it changes; what to do).**
- *Structure:* one learned **operator per action** — a group-representation matrix and an invertible **displacement**.
  Translation (abelian) is the special case; rotations/orderings/constrained moves are the non-abelian general case
  (composition = matrix product). Between two objects, the operator is the **relative-position displacement** relating
  their frames (Numenta's *displacement cells*) — the basis of composition and of representing where *other* objects are.
- *Function:* **path-integrates** L6's location; **predicts** by carrying the location forward; is the **motor** — emits
  the action that brings about the predicted/desired state (predictions, not commands); is the **driver** — the
  inter-column message a higher-order thalamus relays. Holds **temporal sequence memory over actions** (§5): a motor
  skill/habit is a learned sequence of operators (the production side of a behavior).

**L4 — CONTENT / feature-at-location (what is here).**
- *Structure:* a **content codebook** learned online (label-free) + the **feature ⊗ location** binding; a
  pose/rotation-invariant feature descriptor.
- *Function:* bind the sensed content to the current L6 location; **read out** the predicted content at a location — the
  "what will I see" half of the column's prediction. Holds **temporal sequence memory over content** (§5): predicts the
  next feature when content evolves in place (a cell toggling, a colour cycling).

**L2/3 — OBJECT / identity (which object this is, and how it behaves).**
- *Structure:* a **graph-memory of objects**, each an arrangement of content-at-displacements in its own frame, **and its
  behaviors** — each behavior a learned temporal sequence of displacements (how the object moves/transforms).
- *Function:* **recognise** the object and **infer its pose** by incremental evidence voting; **group by structure**
  (boundaries from prediction mismatch, never a colour/connected-component heuristic); vote laterally and across columns.
  Holds **temporal sequence memory over displacements** (§5): recognise the object's **phase** in its behavior and predict
  the next displacement.
- *Learn a new object (allocate-on-mismatch):* when accumulating evidence matches **no** stored model across steps,
  allocate a **new frame** anchored at **first contact** and populate it by sensorimotor exploration — sense content at the
  current location, move (path-integrate the operator), sense the next, adding each `(content, displacement)` node only if
  novel (de-duplicated). Unsupervised, label-free, incremental; the learn-vs-recognise decision is driven by **persistent
  prediction mismatch**, never a segmentation heuristic (rule 5). The same column machinery — path integration + prediction
  error — that recognises also learns. The object's **form** (this static graph) is learned first; its **behaviors** (§5)
  are the temporal-sequence layer on top. (Numenta grid-cell framework, Hawkins et al. 2019; Monty / Thousand Brains
  Project 2024: "no match during recognition → add a new graph"; novelty-gated node addition.)

**Temporal sequence memory is one mechanism, instantiated per layer** — L4 (features), L2/3 (displacements), L5 (actions)
— differing only in the context that drives the prediction; L6's temporal structure is the SR, not this. See §5.

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

**Value / reward — appetitive and aversive.** The domain-agnostic **critic**: expected future reward, learned online from
the sparse score. **Cost is the AVERSIVE component of this one signed value**, not a separate currency — a wall, a hazard,
a slow tile, a risky tile are points on the same scalar (walls the `−∞` limit), so obstacles are *not* special objects.
Two things make this work and must be preserved:
- *Cost is LEARNED as an EXPECTATION* (a running-mean / TD estimate), so a **stochastic** ("risky") location converges to
  `p·penalty` with no special case — the property that makes the cost field robust obstacle-avoidance in abstract and
  stochastic domains. A crude last-write of the aversive score is the wrong representation; the expectation is the right
  one. This is the empirically strongest obstacle mechanism and is not to be discarded.
- *The brain represents cost through a partly SEPARATE circuit from reward* — an appetitive/aversive **asymmetry** (Go/
  reward vs NoGo/cost), not one homogeneous signal: reward via dopamine / D1-Go; **cost via the anterior cingulate**
  (integrating effort / pain / risk into an *expected-cost* signal), the **lateral habenula** (negative reward-prediction
  error / aversive *expectation* value), the **D2-NoGo** striatal pathway (avoidance value), and serotonin (active-
  avoidance expectation). So a first-class cost representation is biologically grounded, not a rule-1 parallel system —
  provided there is ONE aversive-value learner, not two.

Spatially, obstacle-avoidance **emerges** from this signed value read over the learned frame, whose SR/grid map **warps
around barriers** (boundary-vector-cell → SR): `V = M·(reward − cost)` gives the geodesic detour (§8), a wall being the
`−∞` / transition-dead-end limit. The two axes of the one value — pragmatic (reward − cost) + epistemic — are what make
explore vs exploit, and approach vs avoid, *emerge* rather than switch. (ACC expected-cost: Kennerley & Walton; LHb
aversive value: Matsumoto & Hikosaka 2007; barrier-warping BVC-SR: de Cothi & Barry 2020.)

## 4. The sensorimotor loop — how the model touches the world

The model is sensorimotor at its core: it **acts to sense and senses to act**. One cycle:

**(a) Sensory input.** The world emits a **sensory field** — in ARC-AGI-3, a 64×64 grid of up-to-16 colours — plus a
sparse scalar **score**, a terminal signal (WIN / GAME_OVER), and the set of currently available actions. No explicit
goal is given; the reward must be learned from the score.

**(b) Transduction (the retina).** A thin **peripheral** converts the raw field into the model's currency —
**feature-at-location**: the content (an L4 feature) present at a location (an L6 code). This is transduction only, like a
retina: it holds NO segmentation, NO object heuristic, NO domain logic (rules 4–5). It does not decide *what* an object is
or *where* to look — it only makes the raw field readable as features-at-locations.

**(c) What receives it, and what is done.** **L4** binds the feature at **L6**'s current location; **L2/3** accumulates
recognition evidence (object, pose, and behavior phase); **L6** corrects its current-location belief against what was
sensed (the predict-then-compare snap); the **critic** turns the score into value. The column had already *predicted* the
feature it would sense (§5) — the **prediction error** is the learning signal. The model identifies *itself* by
**reafference** (von Holst): the part of the field that changes as its own operator predicts is the **controllable**
location (the "self"); world-caused change is not so predicted — controllability is *learned*, never a fovea-on-residual
heuristic (rule 5).

**(d) Choosing to move.** The **motor (L5)** selects the action that best brings about the goal-state under the one EFE
value (§8) — pragmatic toward reward, epistemic toward what most resolves uncertainty; the **basal ganglia** select among
the candidate goals. This is **one mechanism for covert and overt movement**: shifting attention to sample more (a saccade
— active recognition) and taking a world action (move, click) are both **operators applied to the location**.

**(e) Acting (the effector).** L5 emits the chosen action; a thin **motor organ** (another peripheral, at the SDK
boundary) maps the action name to the world's effector API — a movement, or a click whose `(x, y)` is read from the
goal-state's target location. The world returns the next field, and the cycle repeats.

**The body vs the brain.** The retina (b) and the motor organ (e) are **peripherals** — transduction and effection —
thin, at the SDK boundary, holding no cognition. A peripheral that began deciding what an object is, or which cell is "the
agent," would be a load-bearing harness (rules 4–5); that decision is the column's, made by recognition and reafference.

## 5. Prediction — forward and backward modelling

**Forward modelling is emergent, and it is one thing: predicting the next displacement in a learned sequence.**
- The primitive is **displacement**: grid cells give your location (a displacement from an object); displacement cells
  give the relative position between two frames (Numenta). An operator *is* a displacement.
- **Dynamics — how objects move, transform, and how features evolve — are learned as *behaviors*: temporal sequences of
  displacements** (Numenta's stapler: closed = displacement A → open = E; the conformation change *is* the sequence). The
  **next** displacement is predicted by temporal sequence memory (below).
- **Self and other are the same mechanism, differing only in the driver of the next displacement:** your **efference**
  (self-motion; the change is reafferent, predicted) vs the object's **learned behavior-sequence** (another object;
  autonomous). There is no separate forward model — it is the column's prediction, driven by efference or by a behavior.
- (Dynamic predictive coding, Jiang & Rao) a *higher* level can select **which** behavior/dynamics is active, modulating
  the lower level's content — "which behavior" is a higher-level state.

**Temporal sequence memory — one mechanism, instantiated per layer.**
- *Mechanism* (HTM neuron, Hawkins & Ahmad 2016): a cell's **context** (distal-dendrite input) puts it in a **predictive**
  (depolarised) state; when input arrives, predicted cells fire and inhibit their siblings → a **context-specific** sparse
  representation, so the *same element in a different context is a different state* → **high-order** sequences. The
  predictive cells *are* the next-element prediction. "Same code, different context" — it is the same mechanism as
  sensorimotor prediction, differing only in the context input.
- *Where it lives:* **L4** → next **feature** (content dynamics), context = location + efference + history; **L2/3** →
  next **displacement** (object behaviors), context = the object's **phase**; **L5** → next **action** (motor skills),
  context = the program's phase. **L6 is the exception** — its temporal structure is the SR (predictive map, for planning
  §8), not this; a sequence memory in L6 would be a parallel system.
- *The phase* (position in a behavior — the high-order context) is a recurrent state. There is **one** recurrence, reused
  wherever a phase is advanced (never two). The learned relations among operators (the loop-closure structure) are the
  *structure* of these sequences — a behavior is a path through them.

**Backward modelling — the same machinery run in reverse.**
- Operators are **invertible group elements**, so a behavior run backward = applying the **inverse operators** (the
  stapler: closing *is* opening reversed).
- *Uses:* **retrodiction** (infer the past / what preceded a state) and **reverse-replay credit assignment** (propagate
  reward backward along the sequence — which earlier states/actions led to reward). Predictive coding's top-down
  generative direction is also inherently "backward" down the hierarchy.
- It is **not** a separate mechanism — it is the forward sequence memory with inverse operators.

## 6. Glossary — one definition each (a second meaning is a bug)

- **location** `g` — the current point in the learned frame = the L6 SR code. The ONLY location representation.
- **operator / displacement** — the learned per-action transition on the location code (L5), a group-representation
  matrix, invertible. The ONLY transition and path-integrator; translation is its special case; between two objects it is
  the relative-position displacement relating their frames.
- **content** `x` — the feature at a location (L4). Location-invariant.
- **feature-at-location** — the binding `g ⊗ x` (L4). The ONLY map.
- **prediction** — the column applying an operator to the location and reading the content there. Forward modelling is
  this run forward; backward modelling is this with inverse operators. There is no separate forward-model module.
- **behavior** — an object's motion/transformation = a learned temporal *sequence of operators* (displacements), held by
  L2/3, indexed by phase.
- **temporal sequence memory** — the one mechanism predicting the next element of a sequence via context-specific
  (high-order) representations; instantiated in L4 (features), L2/3 (displacements), L5 (actions). L6's is the SR.
- **phase** — the current position in a behavior/sequence = the temporal context; a recurrent state.
- **object** — a learned frame of content-at-displacements (with behaviors), recognised by voting; pose and phase
  inferred; boundaries from prediction mismatch; **learned by allocate-on-mismatch** — recognise, else add a new frame and
  populate it by sensorimotor exploration (§2). NOT a segment, a change-log, or a tracked mover.
- **prediction error / surprise** — predicted vs sensed content. The ONLY "something changed" signal; no stored change-log.
- **value** — expected future reward incl. cost, one currency (the critic). The ONLY value.
- **goal** — a target-state to bring about; the motor acts to fulfil it.
- **selection** — the basal ganglia choosing among goals/actions.
- **peripheral** — the retina (raw field → feature-at-location) and the motor organ (action → effector API); transduction
  and effection only, no cognition.

## 7. The five rules (development law; a violation is reverted, not documented around)

1. **No parallel systems — ever, including for experiments.** Exactly one way to path-integrate, one way to predict, one
   feature-at-location, one grid-module learning rule, one recogniser, one temporal-sequence-memory mechanism (reused
   across L4/L5/L2-3), one recurrence, one value, one selector. Manage comparison and risk with **git branches**, never by
   keeping two mechanisms in the tree. Two "complementary" mechanisms plus an **arbiter** (tabular-vs-forward,
   explore-vs-exploit as a hard switch, CA-vs-g×x) is a parallel system in disguise — collapse it into the one mechanism
   whose behaviour subsumes both cases.
2. **One definition per concept.** See §6. A new meaning for an existing word is a bug to fix.
3. **The column and the agent are thin coordinators.** They hold references + routing — never math or state. Every belief,
   map, operator, behavior, and value lives in a layer/module.
4. **No load-bearing harness, no domain-specific code, no special-casing, no ungrounded arbitration.** Nothing branches on
   which game/domain it is; the peripherals transduce only. Every arbitration must name the brain mechanism it implements
   (basal-ganglia selection, tonic-dopamine gain, STN commitment) or it is removed. Selection lives in the basal ganglia.
5. **No symbolic estimators, object heuristics, or change logs.** No hand-coded "what is an object / how to split it,"
   no Kalman-style tracker banks (fovea centroids, pose matrices, binned nodes in parallel), no dicts of "what changed."
   Structure is learned; change is carried by prediction error; the object is a recognition construct.

## 8. Planning — how the model decides (explore, exploit, act)

**What planning is in real brains.** Routine planning is *not* rollout. The **successor representation** already stores
the discounted future occupancy of the learned map, so value is a cheap read — `V = M·R`, a dot product over the frame —
and greedy-on-`V` follows the shortest path, warping around barriers (the geodesic). Deliberative rollout (vicarious
trial-and-error, hippocampal replay) is **sparing**, for the novel/hard case; prioritised replay schedules the value
updates by gain × need (Mattar & Daw). (`reference_brain_planning`, `reference_exploration_replay`.)

**What planning is in our model.** The same: the column's learned frame (L6 SR) *is* the map; planning is the value read
off it. **One planner, one value.** The geodesic to a goal falls out of the SR; rollout is only the column's own
prediction (§5) iterated, used sparingly — never a second, parallel planner.

**How it knows to explore vs exploit — one value, not a switch.** The critic's value is **Expected Free Energy**:
pragmatic (expected reward toward the goal) + epistemic (expected information gain, grounded by **epiplexity** =
learning-*progress*, so it → 0 for both irreducible noise *and* mastered structure). The policy maximises the one value:
**exploit emerges** where pragmatic dominates, **explore emerges** where epistemic dominates. There is no `g`-gate and no
`V`/`V_exploit` split (those were a two-mechanism arbitration, a P0 target).

*On the eigenpurpose — a reassessment (do not treat as a first-class mechanism).* Eigenpurposes (Machado et al. 2017/2018)
were built as **task-agnostic** intrinsic rewards along the SR/Laplacian eigenvectors, to discover reusable exploration
*options* that traverse the state-space geometry to bottlenecks. We adopted one as the directed explorer for the flat
dead-zone. On reflection it is **redundant and costly**: (a) it *duplicates* the epistemic term — reaching unexplored
regions is what a learning-progress / info-gain explorer already does; (b) being reward-blind it covers geometry
**uniformly**, wasting actions under the RHAE budget (the value-aware-eigenoptions critique, arXiv 2507.09127); (c) it needs
an **O(n³) eigendecomposition** of the SR — prohibitive on 64×64 frames (the code band-aids it with a throttle, itself a
smell); (d) in the code it is a **separate `g`-gated salience** — exactly the parallel-explorer-plus-arbiter rule 1 forbids.
**Decision:** the primary — and only — explorer is the one **epistemic term** (learning-progress + novelty); the SR
**geodesic** (a cheap dot-product read, `V = M·R`, *not* its eigendecomposition) does the reaching once a target exists.
Eigenpurpose is dropped — or at most a cheap geometry prior folded *inside* the epistemic term for the flat case, never a
separate gated term. (`reference_efe_and_epiplexity`, `reference_eigenoptions_subgoals`.)

**Acting = testing a hypothesis.** A goal-state is a hypothesis "bring about X." The agent plans to X, the motor achieves
it, and the **outcome** (reward = pragmatic, prediction-error = epistemic) confirms or refutes; the basal ganglia commit
through the maneuver and switch on repeated refutation. Testing a hypothesis *is* planning.

## 9. Hypothesis generation — how the model proposes what to try (the frontier)

Testing is §8; **generation** — where a candidate target-state comes from — is the genuinely open problem. The proposal,
from the research and the number-domain probes (`MATH_PHASE.md`):

- **Not enumeration.** The mind **samples a few candidates from memory**, cued by context and biased by priors:
  **salience × controllability × ambiguity** (Dasgupta, Schulz & Gershman 2017; more samples when more uncertain).
  Controllability = "it moved when I acted" (the reafference of §4c); salience = novelty / prediction-error; both learned.
- **A hypothesis is a short composition of learned operators toward a cued target** — geodesic-finding in the learned
  structure. The **master boundary** predicts its cost: where the structure is **free/abelian**, the hypothesis is
  **READ OFF** (a homomorphism is fixed by its action on generators — cheap); where it is **relational / quotient**
  (non-commuting, constrained — Sokoban, carry), it must be **SEARCHED**.
- **Open (honestly):** (a) learning the priors that cue *which* targets to sample (learned, not hand-coded); (b) whether
  the relational **search** is tractable at scale. The `MATH_PHASE` microworlds exist to probe these; no code commits to a
  solution until they answer.

## 10. The plan — one dependency-ordered spine

- **P0 — Converge the code to this document (mostly DELETION).** Collapse every parallel system, estimator, and arbiter
  into the one mechanism: one prediction (delete the location-blind CA); one location = the L6 code path-integrated by the
  operator (delete the fovea / pose-matrix / `state_node` / `_obs` / `heading_dependent` fork); one value (fold
  `V`/`V_exploit` + the `g`-gate + the `_tab_spread` tabular/forward arbiter into the single EFE value; **drop the
  eigenpurpose SR-eigendecomposition explorer** in favour of the one epistemic term — §8; **unify the cost field as the ONE
  value's AVERSIVE component** — keep its learned-expectation (the running-mean → `p·penalty`; make the critic's crude
  last-write aversion use the same expectation so there is ONE aversive-value learner), obstacle-avoidance emerging from
  `V = M·(reward − cost)` over the barrier-warping SR (§3); the cost field is NOT deleted. ⚠ "one aversive-value learner"
  does NOT mean collapsing cost into a homogeneous reward channel: PRESERVE the appetitive/aversive ASYMMETRY of §3 (the
  brain uses a partly SEPARATE circuit — it may need its own learning rate, Go/NoGo opponency, or consumers; do not flatten
  it away)); object = recognition construct
  (delete the `object_state`/`_changed` change-log — DONE; the segmentation heuristic is LOAD-BEARING as the live
  perception front-end, so it is REPLACED-then-deleted in P1, not cut here); thin column
  + agent (subsystems → layers); retina/motor-organ reduced to transduction/effection. Suite-green throughout; git
  branches for risk. **This is the bulk of the work.**
- **P1 — Factored perception.** L2/3 recognition + L4 content deliver `(location, content)` factored, from the live field —
  the prerequisite prediction always assumed and never had. (Why the FM could not be built before.)
- **P2 — The one prediction over the factored representation.** The column's §5 prediction with clean content: apply the
  operator to the location, read the content. **Self-motion** works here (the next displacement = your efference).
- **P3 — Temporal sequence memory & behaviors.** The one sequence-memory mechanism (§5) in L4 (features), L2/3
  (displacements/behaviors), L5 (actions), with the **phase** as one recurrence and the learned **relations** (loop
  closure — already built) as the sequence structure. This is where **other objects' dynamics** are forward-modelled (the
  next displacement = the object's learned behavior), and **backward modelling** (inverse operators — retrodiction,
  reverse-replay credit assignment) lands. Then the order/config-dependent case (Sokoban).
- **P4 — Planning & hypothesis generation.** The one EFE value / SR geodesic (§8); the goal-state generator proposes
  cued target-states (§9), the BG select, the motor achieves, value confirms; the heterarchy (multi-column voting via the
  thalamus) scales the same loop.

Honest status: the **operator** primitive and **relation/factor discovery** (the P3 sequence structure) exist and are
tested; the SR is the one L6. Everything else is entangled with the P0 estimator/arbiter stack. **We are at P0.**

## 11. Acceptance test for every change (the paper test)

Both must hold, or the change does not land:
1. **Explainable** in one sentence that fits §1–§6 (no new term, no second meaning, no "well, in this mode…").
2. **Obeys the five rules** (no parallel system, no second definition, no coordinator bloat, no harness/special-case/
   ungrounded arbitration, no symbolic estimator/heuristic/change-log).

If a change cannot pass both, the design is wrong — fix the design, not the change.
