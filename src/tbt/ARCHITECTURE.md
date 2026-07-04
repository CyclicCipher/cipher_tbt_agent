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
  initial-state prior expressible within it, never a parallel code. **The location is an SDR** — a sparse, distributed
  grid-cell population whose defining property is *semantic overlap* (nearby locations share active bits → they
  generalise; distant ones are near-orthogonal), NOT a localist symbol (HTM: "each grid cell is a bit in an SDR"; Lewis
  et al. 2019). The SR must therefore be learned **over the SDR encoding** (successor *features*), not over localist
  state symbols; today's symbol-indexed `OnlineSR` (and the `state_node` bin) is scaffolding to retire — the SDR test
  (2026-07-03) showed it carries no a-priori metric overlap (§10 P4a/P5).
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

- **location** `g` — the current point in the learned frame = the L6 SR/grid code, **an SDR** (sparse, distributed,
  overlap = similarity), NOT a localist symbol or a dense pose matrix. The ONLY location representation.
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

**There is no planner *module*.** Vector navigation is a read over the spatial cells: **grid cells** (L6) give the goal
*vector* — direction + distance to a set target (Bush, Barry & Burgess 2015); place-cell populations carry
**goal-oriented vector fields that converge on the goal** (the "potential field" itself — Ormond & O'Keefe 2022); the
**SR warps around barriers** for the detour (de Cothi & Barry 2020). One object, not a Euclidean field plus a geodesic
fallback. The action falls out by **inverting the operator (L5)** against that value; the **basal ganglia** select. So a
Euclidean potential-field navigator with an SR-geodesic *fallback arbiter* is a rule-1 parallel system — deleted; the SR
goal-oriented vector field is the one navigator.

**How the navigator reads the map — corrected by experiment (2026-07-03).** Greedy ascent on a scalar value is the
*tabular* idealisation; over the grid **SDR** with a linear value (successor features) it FAILS — the value is accurate
only LOCALLY and the periodic grid wraps, so greedy-on-V has spurious local maxima and gets stuck (probed: a 2-D point
goal is never reached, V(dist-20) > V(dist-10)). The TBT-real primitive is the **grid-cell goal VECTOR** — the
displacement from the current location code to the goal's, read off the grid phases (Bush, Barry & Burgess 2015;
decodable via `GridEncoder`), moved along by the operator. The successor-feature **value MODULATES** this vector —
reward attraction, cost/barrier repulsion (§3, de Cothi & Barry) — it does not replace it. So the one navigator is
**the vector to the goal, warped by the SF value**, not value-gradient-ascent. (Finding `project_sf_value_not_greedy_navigable`.)

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
- **Where the candidate comes from — the priority map (what *sets* the goal).** A goal is a *target-state*, and the
  target is set by a **priority map** that fuses **bottom-up salience** (feature-contrast + prediction-error surprise —
  superior colliculus; NOT motion alone, a live gap) with **top-down value / memory** (reward- and goal-tagged
  locations — Gauthier & Tank 2018); the **basal ganglia** select its peak (Fecteau & Munoz 2006). The winner is held as
  a **goal vector** — goal-direction + goal-distance cells, memory-based even for an *occluded* goal (Sarel et al. 2017,
  bat CA1) — which is exactly the vector the grid cells compute toward (§8). This IS `salience × controllability ×
  ambiguity` in neural form: controllability = reafference (§4c), salience = feature-contrast + prediction-error.
- **Testing is simulated, then enacted.** Before committing, the hippocampus **preplays** candidate trajectories
  (vicarious trial-and-error), the critic scores them (pragmatic + epistemic), and the **BG/STN** commit and switch on
  repeated refutation (hippocampal–prefrontal replay selecting the path) — §8's "acting = testing a hypothesis" made
  concrete.
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
  operator — vetted 2026-07-02: the fovea / pose-matrix / `state_node` tracker is the LIVE perception→location bridge
  (sensor → `track` → `state_node` = the agent's state), while the doc-target mechanism (a node path-integrated by the
  operator, cf. `loc_*`) is LATENT and needs perception to feed it a location; so this fork — and the `heading_dependent`
  fork, which needs the learned-group operator (abelian = the commuting special case) before it can go, else abelian games
  gain a spurious heading — is REPLACED-then-deleted in P1/P2, not cut here; one value (fold
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
- **P1 — Factored perception.** Deliver a factored `(location, content)` from the live field — the prerequisite prediction
  always assumed and never had. **TBT-accurate sourcing (do not drift):** **location = the L6 code path-integrated by the
  operator** (grid-cell dead-reckoning by efference — `loc_move`; anchored/corrected by sensing — `loc_sense`), NOT
  re-derived from recognition each step (in TBT, movement path-integrates location and recognition *consumes* it — the
  `loc_*` skeleton is the canonical one, the fovea tracker its continuous form + estimator binning). **content = L4**'s
  rotation-invariant descriptor (the same local shape at any pose → the same `x`). **L2/3 recognition** consumes the
  `(location, content)` stream to infer the object + its reference frame and supplies the anchor that corrects L6 — a
  *consumer* of location, not its source. This replaces the estimator-stack halves (the fovea/`state_node` location; the
  raw-patch content). Slices: (1) content = the invariant descriptor; (2) location = operator-path-integrated L6,
  recognition-corrected — where #1's fork deletion lands; (3) retina reduced to pure transduction.
- **P2 — The one prediction over the factored representation.** The column's §5 prediction with clean content: apply the
  operator to the location, read the content. **Self-motion** works here (the next displacement = your efference).
- **P3 — Temporal sequence memory & behaviors.** The one sequence-memory mechanism (§5) in L4 (features), L2/3
  (displacements/behaviors), L5 (actions), with the **phase** as one recurrence and the learned **relations** (loop
  closure — already built) as the sequence structure. This is where **other objects' dynamics** are forward-modelled (the
  next displacement = the object's learned behavior), and **backward modelling** (inverse operators — retrodiction,
  reverse-replay credit assignment) lands. The **order/config-dependent case (Sokoban)** is the SAME context-conditioned
  mechanism — **order**-dependence is the non-commuting Cayley graph (`discover_relations`, built), **config**-dependence
  is the sequence memory with the *config* in the context (the spatial twin of the temporal phase); the **general
  relational RULE** (generalising the config-rule to unseen configs) is the open MATH_PHASE search (§9), not committed to.
  *Status: the mechanisms — `SequenceMemory`, `Behavior` (other-object dynamics), `inverse` (backward), config-as-context —
  are built + tested; the multi-object self/other LOOP wiring is loop-coupled (P4-adjacent).*
- **P4 — The convergence: collapse the leaked parallels, then the goal loop.** Re-vetted 2026-07-03 (the coloured-marker
  probe + a neuroscience pass): the *bulk* of P4 is **finishing P0's parallel-system collapse, which leaked into
  `column.py`** (a 707-line god-object — `project_column_godobject_diagnosis`), NOT new machinery. Most of the column is a
  *second copy* of a layer mechanism; collapse each into the ONE (rule 1), do not relocate:
  (a) **one location** = the SR/grid code path-integrated by the operator (§2; grid cells = the SR's eigenvectors, one
  frame — Stachenfeld 2017) — retire the parallel `_pose` SE(2) integrator + the lossy `state_node` binning; the grid
  eigenbasis carries the continuous metric and, via the learned non-abelian operator, the heading, *within* the one frame
  (**this requires P5** — the SDR test showed the SR-over-localist-symbols has NO a-priori metric overlap, so the location
  SDR needs the encoder; P5 is pulled forward as a prerequisite);
  (b) **one operator** per action in L5 (retire the column `pose_ops` store); (c) **one navigator** = the SR
  goal-oriented vector field `V=M·(reward−cost)` (delete the Euclidean `vector_action`/`_pose_vector_action`/`achieve` +
  its SR-fallback arbiter — §8); (d) **one aversive value** (fold the column `cost` field into the value module's aversive
  component, appetitive/aversive asymmetry preserved — §3); (e) **one allo map** in L4⊗L6 (retire the column `_map` AND
  the parallel, unwired `hippocampus.py`); (f) **one planner** = the SR value read (collapse reward.py's parallel
  prioritized-sweeping into it; rollout = the sparing fallback — §8). Remove the non-TBT proto-object **segmentation
  heuristic** (`ObjectField`/`segment`/`_mover_cloud`); TBT creates objects by allocate-on-mismatch (§2). THEN the **goal
  loop**: the priority map proposes cued target-states (§9), the BG select, the operator-inversion motor achieves, value
  confirms. The **hippocampus** (allo frame + self/world-motion split via reafference + object-vector-cell displacement
  binding) is a PREREQUISITE for salience-cued nav — a static distinct object is invisible to motion-only salience
  (`project_marker_exposes_hippocampus_prereq`), not a late add-on; the heterarchy scales the same loop.
- **P5 — The semantic SDR encoder (a PREREQUISITE for P4a — re-assessed 2026-07-03, not post-P4).** The general data
  type — for **content** (`L4.E`, `reference_tbt_feature_definition`) AND for **location** — is an **SDR**, whose point is
  semantic OVERLAP (similarity), determinism, fixed length + sparsity. Today BOTH are EXACT-MATCH / LOCALIST (similar
  inputs → orthogonal codes; `OnlineSR` indexed by localist `state_node` symbols), so nothing generalises across
  similar-but-unseen inputs. The **SDR test** made this concrete: `hex_code` has graded a-priori metric overlap (cos
  +1→0.94, +4→0.28, far→−0.39) while `state_node` is aliased-or-orthogonal and the SR relates only VISITED states (an
  unvisited neighbour has no row). Fix: an HTM-style encoder obeying the three rules — hand-designed sub-encoders
  (scalar/periodic/category; the hex grid is the periodic *location* prior) or a LEARNED **spatial pooler** — so the
  location/content SDR has overlap; and the SR becomes **successor *features*** over that SDR encoding (Barreto et al.
  2017; *Neurobiological successor features for spatial navigation*, Hippocampus 2021), retiring the symbol-indexed
  tabular `OnlineSR`. This is what makes P4a's "location is an SDR" real — the linchpin, pulled forward. (Purdy, *Encoding
  Data for HTM Systems*; NuPIC spatial pooling.) **BUILT 2026-07-03 (`tbt/encoders.py`, `test_encoders.py` 10/10):** the
  bidirectional library — `SDR` + `Scalar`(+periodic)/`Category`/`Grid`/`Multi` encoders + a `SpatialPooler`; each
  encoder IS its decoder (the inverse), so a motor SDR "thought" is read back by the same encoder that senses it (§4 — no
  parallel motor-decoder can fester). **Successor features BUILT 2026-07-03 (`l6_sr.SuccessorFeatures`,
  `test_successor_features.py`):** `ψ=W·φ`, `V=w·ψ` learned by TD over the SDR φ — value GENERALISES to UNVISITED states
  graded by overlap (V(29)=9.4 vs V(21)=6.8 on a corridor where only even cells were visited), where `OnlineSR` gives a
  flat 0. NEXT: re-seat the loop off `state_node` onto (`GridEncoder` φ + `SuccessorFeatures`), retiring `OnlineSR`.

Status (2026-07-02, P0–P3 mechanisms DONE — next is P4): **P0** collapsed the parallel systems (one prediction, one value,
no change-log; cost = the kept aversive value). **P1** made perception factored and live — `perceive` delivers
`(location, content)` through recognize→predict→correct→learn; the old fovea/`track`/pose dispatch is DELETED and the
column is content-opaque (colour confined to the peripheral `retina.view_signature`). **P2** built the one forward
prediction over that rep (`col.forward` + the `_pred_error` residual). **P3** built the temporal sequence memory
(`sequence.SequenceMemory`), behaviors (`sequence.Behavior` — other objects' dynamics), backward modelling
(`sequence.inverse`), and the order/config-dependent case (context-conditioning). Suite: **137 passed, 4 xfailed**
(the xfailed = end-to-end SOLVING, P0-dip casualties re-earned at P4).

Re-vetted what P1/P2 was thought to unblock (the estimator stack): P1 DID unblock the location-dispatch deletion (done).
But the rest was mis-attributed — it is **the PLANNING system, gated on P4**, not P1/P2: `move_delta` / `heading_dependent`
/ `here_position` / the cost-bump / the achiever (`vector_action`/`navigate_to`/`achieve`/`_pose_vector_action`) are all
read by the **reward-goal beeline in `_choose`**; they retire/unify *together* onto the one value + SR geodesic at **P4**
(with `state_node`'s binning and the tabular `col.predict`). And the **segmentation heuristic** was not eliminated by P1 —
`perceive` swapped `ObjectField` for `_mover_cloud`'s own connected-component heuristic; true elimination = recognition-
proposed boundaries from prediction mismatch (§2 L2/3), a **P3+ figure-ground** build, not a deletion. So the remaining
"estimator stack" is really *planning + figure-ground*, not P1/P2 leftovers. The **operator** primitive + relation/factor
discovery are tested; the SR is the one L6.

**Next: P4 — the convergence (in progress, 2026-07-03).** Re-vetted against the coloured-marker probe and a neuroscience
pass (goal-vector cells, the priority map, preplay/VTE, grid = SR eigenvectors — §8/§9): P4's bulk is **finishing P0's
parallel-system collapse that leaked into `column.py`**, NOT new machinery — most of the column is a *second copy* of a
layer mechanism (a second location code, operator store, navigator, aversive signal, allo map; reward.py a second
planner). Collapse each into the ONE (§10 P4 (a)–(f), rule 1), remove the non-TBT proto-object segmentation heuristic,
THEN build the **hippocampus** (allo frame + self/world-motion split via reafference), which the marker probe showed is
the PREREQUISITE for salience-cued nav (`project_marker_exposes_hippocampus_prereq`), THEN the priority-map goal loop
(§9 — where the target comes from is now specified: salience ⊕ value → priority map → goal-vector cells; test by
preplay/VTE). **NB the ordering shifted (2026-07-03): (a) the location collapse now depends on P5** (the SDR encoder /
successor features), pulled forward — the SDR test showed the SR-over-localist-symbols gives no a-priori metric overlap,
so "location is an SDR" needs the encoder first (`project_location_is_an_sdr`). Open each slice with the TBT-accuracy
check + a consumer map before cutting; suite-green throughout; git branches for risk. Empirical anchor: suite 132 passed
/ 4 xfailed (hippocampus.py parallel removed); NavGame (now colour-cued) 0/8.

## 11. Acceptance test for every change (the paper test)

All must hold, or the change does not land:
1. **Explainable** in one sentence that fits §1–§6 (no new term, no second meaning, no "well, in this mode…").
2. **Obeys the five rules** (no parallel system, no second definition, no coordinator bloat, no harness/special-case/
   ungrounded arbitration, no symbolic estimator/heuristic/change-log).
3. **TBT-accurate mechanism.** Before a step begins, verify its mechanism against real TBT / neuroscience — *not* just
   this doc's wording, which can drift (P1's "location from the recognized pose" was such a drift). State the mechanism in
   one sentence, ask "is this how a real cortical column does it?" (which layer, what drives it, does the research back
   it), and research when unsure. If the doc's wording is wrong, **fix the doc first**, then build.

If a change cannot pass all three, the design is wrong — fix the design, not the change.
