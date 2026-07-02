# L6_NONABELIAN — from an abelian grid to a learned GROUP REPRESENTATION (a refactor)

*2026-07-01. A PLANNING doc for a REFACTOR (not a from-scratch rewrite — the abelian grid survives as the special case).
Motivation → the one core change → why it (hopefully) unlocks Sokoban → a staged plan with gates+fallbacks → risks →
what is preserved. Grounds: `MATH_PHASE.md` (THE MASTER BOUNDARY + the ABELIAN CEILING), the Fable critique, TEM
(Whittington 2020 — learned transition matrices + structure/content factorisation), Gao et al. 2021 (path integration ⇔
a GROUP REPRESENTATION + isometry). North star: L6 should represent the domain's transformation GROUP, of which the
current translation/plane-wave grid is the abelian special case.*

## Why — the abelian ceiling (the thing this refactor removes)
L5's operator today is an ADDITIVE displacement (`move_delta[a]=(dx,dy)`, you ADD it); L6's grid is PLANE WAVES and
`path_integrate` ROTATES phases by that displacement. Both are the abelian move: addition and phase-rotation COMMUTE, and
plane waves are exactly the *characters (1-D irreps) of the abelian translation group*. **Commutativity is not learned — it
is hard-wired into the substrate.** A code of commuting phases physically cannot store a PATH-DEPENDENT (non-abelian) result,
because there is no place to record "the answer depended on the order." Non-abelian structure — rotations, orderings,
constrained dynamics (Sokoban-with-walls: move-right-then-up ≠ up-then-right) — is therefore unrepresentable in L6 as built.
NB the *active* substrate (the online SR) already handles a non-abelian Cayley graph, but only as a memorised LOOKUP (a node
per state, NO generalisation/extrapolation). The GRID is what generalises, and it is abelian. **So the gap is precisely:
non-abelian structure WITH generalisation.**

## The one core change (stated three ways)
- **L5 (primary):** operators go from additive DISPLACEMENTS (vectors, commute) to composable OPERATORS — learned LINEAR
  MAPS (matrices) applied to the L6 code, composing by MATRIX PRODUCT, which is *non-commutative by construction* (`A·B ≠
  B·A` records order-dependence for free). NB the abelian grid is already "block-diagonal 2×2 rotation matrices in disguise"
  (a phase rotation IS a 2×2 orthogonal matrix per module) — the generalisation is to let those matrices be LEARNED,
  LARGER, and NON-commuting.
- **L6 (consequence):** the location code goes from a phase/torus point to a LATENT VECTOR the L5 operators act on — a group
  representation. The abelian grid falls out as the special case (commuting orthogonal rotations = the plane waves we have).
- **Constraint (Gao — do not skip):** the learned operators must satisfy the GROUP-REPRESENTATION condition (orthogonal/
  unitary, composition-consistent) so path integration + extrapolation SURVIVE; a *freely* learned matrix keeps neither. =
  TEM's learned per-action transition matrices + Gao's isometry constraint.
- **Relations discovered, not assumed:** the FREE composition of operators (the tree of operator-words) path-integrates;
  RELATIONS — including COMMUTATIVITY itself, since the commutator `aba⁻¹b⁻¹` is a loop that returns to start — are
  DISCOVERED by LOOP CLOSURE (a word that returns to a seen state = a relation). The factored-loop-closure machinery
  (`MATH_PHASE.md`) thus generalises from cyclic factors (abelian) to RELATION discovery (non-abelian). Read-off = the
  free/tree part; SEARCH = the relations (the quotient). The abelian grid = "assume every commutator loop closes."

## AFFECTED LAYERS — this UNIFIES a currently-FRAGMENTED operator set (L4 + L2/3, not just L5/L6)
The column ALREADY has TWO operator families that disagree on their group — the redesign's real job is to merge them into
ONE learned group representation, used by all four layers.
- **The fragmentation (grounded in the code):** *movement/navigation* = `L5.move_delta` (learned translation) → **L6** path
  integration = TRANSLATION-only, ABELIAN (rotation "a deferred extension"). *pose/recognition* = `L5.rot(θ)`/`apply_pose`/
  `pose_between` → **L2/3** = the full **SE(2)** pose group (rotation + translation) = NON-ABELIAN, and already MATRIX-based
  — but HAND-CODED (`rot` is a closed-form SO(2) matrix). So the column represents a non-abelian group for RECOGNITION yet
  only an abelian one for NAVIGATION. The redesign makes the FRAME as expressive as the recogniser already is.
- **L2/3 (already the non-abelian half):** pose-invariant recognition infers a group element `(θ,t)` (`align_rotations`),
  reconstitutes the object at any pose (`cells_at`=`apply_pose`), and VOTES over shared world poses. The redesign gives it
  (i) CONSISTENCY — the inferred pose is an element of the SAME group L5/L6 use, so navigation and recognition compose poses
  identically and Stage-2 loop-closure/relations run on ONE group; (ii) a BITTER-LESSON win — LEARN the group matrices (as
  validated on S₃) instead of the hand-coded `rot(θ)`, general for ABSTRACT columns (the docstring notes "for an abstract
  column the group is not SO(2)").
- **L4 (mechanism unchanged; "location" → POSE):** the `feature ⊗ location` bind is representation-agnostic, so it is
  untouched — but its "location" generalises to a full POSE (a group element), making L4 *feature-at-pose* = the TBT object
  model (`object = {(feature, pose)}`). The FORWARD MODEL (`predict_feature`/`predict_field` under an action) now applies the
  NON-commuting operator (order matters). The CONTENT codebook is unaffected — the hyle/morphe (what-vs-where) split: only
  the "where" generalises.
- **Payoff:** ONE learned group representation (the `Operator` primitive) shared across L6 (pose frame), L5 (motor +
  prediction), L4 (feature-at-pose), L2/3 (recognition + voting) — replacing the three fragmented families (learned
  translation + hand-coded SE(2) + the new primitive). Simplification + consistency + bitter-lesson. NB DESIGN IMPLICATION,
  not built: the primitive + the parallel learner are done; folding `rot`/`apply_pose` into the learned Operator (L2/3) and
  L4 feature-at-pose are downstream, and the L2/3 fold is load-bearing (it is where the non-abelian structure already lives).

## Why this (hopefully) unlocks Sokoban — honestly
Sokoban's dynamics are non-abelian and config-dependent: the effect of a move depends on what is ahead (push vs. not),
and move-orders do not commute once walls/blocks intervene. The current abelian grid can only map FREE translation (the
agent on open ground), never the push dynamics — so the agent has no faithful MAP to plan over, only the memorising SR.
A representation-constrained operator L6 gives a **faithful COMPOSITIONAL map of order/config-dependent dynamics** — the
substrate a planner needs. With it, planning = **geodesic-finding in the learned Cayley graph** (the master-boundary shared
mechanism), and the GSG proposes goal-configurations. **Necessary, not proven sufficient:** the GSG / multi-step planning /
commitment (the vector-nav + GSG line) still apply on top; Stage 3 is where the "unlocks Sokoban" hope is CONFIRMED or
FALSIFIED, and the staged gates are designed so that a remaining failure is diagnosable as REPRESENTATION (this refactor)
vs PLANNING/GSG (the other line). Do not sell the refactor as a Sokoban solve; sell it as the missing SUBSTRATE.

## The staged plan (each stage gated + a fallback; abelian behaviour preserved throughout)
- **Stage 0 — STOP hard-coding commutativity (the negative first step). DONE (2026-07-01).** `tbt/operator.py` = the
  `Operator` primitive (a matrix; `apply` = act, `then` = compose by matrix product = NON-commutative in general;
  `translation`/`rotation` factories; `commutes_with`). `L5.operator(a)` = the abelian TRANSLATION view of `move` (Stage 1
  makes it a learned matrix); `l6_grid.operator(disp)` = `path_integrate` re-expressed as a block-diagonal phase rotation.
  *Gate met (`test_operator.py`, suite 122):* translation is a faithful abelian representation (composition fidelity +
  commutes); the interface HOLDS a non-commuting operator (rotation ∘ translation ≠ translation ∘ rotation) — commutativity
  is no longer baked in; `operator(disp).apply(z) == path_integrate(z, disp)`; and L5's operator reproduces the additive
  `move` on a NavGame path + a counting succession (NO regression — the live `track` still uses `move`, unchanged). The door
  is open; nothing is exploited yet.
- **Stage 1 — LEARN operators as a representation (TEM + Gao).** Learn a per-action operator matrix from observed
  transitions, CONSTRAINED to a proper representation (orthogonal, composition-consistent). L6 = the acted-on latent.
  **ABELIAN GATE VALIDATED (the discipline task, 2026-07-01, `Operator.fit` + `test_operator.py`, suite 123):** operators
  LEARNED from noisy grid transitions pass composition fidelity `M(a∘b)=M(a)·M(b)` — the learned +1 op EXTRAPOLATES 100–200
  steps out (rel err ~0.02–0.04), learned E/N COMMUTE + compose to (1,1) — **but ONLY with the orthogonality constraint**:
  unconstrained least-squares fits one step yet its spectral radius drifts off 1 so its POWERS drift (~5× worse extrapolation
  at n=200); Procrustes-orthogonal pins the spectrum to the unit circle (ρ=1.0000) → faithful composition. So Gao's
  constraint is REQUIRED, empirically — and the machinery is validated on abelian, so non-abelian failures next are
  diagnosable as non-abelianness, not learnability.
  **ONLINE GATE VALIDATED — and it REFRAMES the linchpin (2026-07-01, `OnlineOperator`):** learning from a STREAM works —
  `OnlineOperator` keeps a running cross-covariance (cheap rank-1 update/step) and reads the operator as its orthogonal
  Procrustes (throttled SVD). On a broad-coverage stream it converges to a faithful operator (spectral radius 1,
  extrapolates); **the CONSTRAINT is NEVER the bottleneck** — orthogonality is a PROJECTION AT READ, so it never fights the
  fit (constraint⊥expressivity does not bite — unlike a constrained *gradient* update, which is where the DEQ wall was). The
  REAL online challenge is COVERAGE: the operator is well-estimated only over the region the stream samples, so a
  narrow/confined walk under-covers → poor extrapolation while a broad sweep converges (a running SUM for a stationary op;
  gentle decay for drift). ⇒ the linchpin moves from *learnability under constraint* to *EXPLORATION/coverage* — a more
  tractable axis the agent already has levers for (directed exploration / eigenpurpose).
  **NON-ABELIAN GATE PASSED (2026-07-01, `test_operator.py`):** learned operators for **S₃** (the smallest non-abelian
  group, regular rep) from Cayley-graph transitions are FAITHFUL (recover the permutation matrices), do NOT commute
  (`M(a)M(b) ≠ M(b)M(a)`), satisfy the group RELATIONS (`a²=e`, `(ab)³=e`), and COMPOSE faithfully — and a COMMUTING
  (abelian) model is order-blind, so it has irreducible error (≥ 0.7) on the order-dependent composite while the matrix
  operators nail it (~0). Online (streaming a walk on the 6-node Cayley graph) recovers the non-abelian operators too. **So
  the operator PRIMITIVE is validated end to end: abelian (batch + online) + non-abelian, learnable, with the constraint a
  projection (not a gradient fight).** *Fallback:* keep the hand-coded grid for abelian domains; learned operators where
  non-abelian structure is present.
  **LIVE WIRING — slice 1 DONE (the PARALLEL learner, 2026-07-01, `column.action_ops`/`_observe_operator`/`action_operator`):**
  the column now learns a per-action `OnlineOperator` ONLINE from the live path-integration stream (the L6 grid-code
  transition `code_at(before)→code_at(after)`), fed inside `track` ALONGSIDE the additive `move` — nothing reads it yet, so
  ZERO behaviour change (integrate-mode only; config-mode/offline benchmark untouched). *Gate met (`test_path_integration.py`,
  suite 126):* driving the REAL agent on NavGame, all four nav operators converge (spectral radius 1, grid-code prediction
  err ~0.01–0.07) while the run still solves 8/8 — so **the agent's own exploration gives enough COVERAGE in practice** (the
  reframed linchpin validated on the live loop, not a synthetic sweep).
  **STAGE 1 SCOPE EXPANDED (2026-07-01, Cipher): FOLD IN the cross-layer UNIFICATION** — "unification / inconsistency
  resolution across all layers" is now high priority (see AFFECTED LAYERS above: the column already has a hand-coded SE(2)
  pose group in L2/3 that the abelian L6 doesn't share). The remaining Stage-1 work, sequenced by dependency:
  - **S1c — a NON-ABELIAN TEST ENV (the prerequisite; STARTED 2026-07-01).** A heading-carrying agent (pose = x,y,θ;
    body-frame FORWARD/TURN = SE(2)) — `test_nonabelian_env.py`. First result: FORWARD∘TURN ≠ TURN∘FORWARD, and the abelian
    `move_delta` (ONE Δ per action) CANNOT represent FORWARD (4 different displacements, one per heading) → over
    POSITION-only the dynamics are non-deterministic; over the full POSE they are deterministic. **KEY SUBTLETY this
    surfaced:** the state must be the POSE (the group element), not the position — so this is a genuine SE(2) task, and the
    continuous case (rotation by any angle) wanted a LIE GENERATOR, not a single Procrustes matrix. **CONTINUOUS FORM BUILT
    (2026-07-01, `Operator.power`/`generator`, suite 129):** from a LEARNED discrete-step operator, `power(t) = exp(t·log M)`
    gives any group element along its 1-parameter subgroup (a fractional rotation / fractional path-integration step); the
    `generator` is the skew-symmetric Lie-algebra element. Gate: fit a 36° turn step from transitions → `power(t)`
    reconstructs `rot(t·36°)` for any `t` — i.e. LEARN the step, read off any continuous pose, replacing the hand-coded
    `rot(θ)`. So S1d has its tool (S₃ = finite groups; SE(2) + `power` = the continuous spatial group, learned).
  - **S1d — the CROSS-LAYER UNIFICATION (the high-priority fold). Slice 1 DONE (2026-07-01, `l5.pose_operator`, suite 130):**
    a pose (SE(2)) is now an `Operator` (`pose_operator(θ,t)` = the 3×3 homogeneous element), and `apply_pose` is ROUTED
    through it — behaviour-IDENTICAL, all recognition tests green (no regression). Subsumption PROVEN (`test_operator.py`):
    pose INFERENCE = `Operator.fit` (Procrustes recovers the same rotation as `pose_between`); pose APPLICATION = the
    operator acting; poses COMPOSE non-abelianly (SE(2)); the CONTINUOUS family = `power`. So L2/3's pose IS one instance of
    the ONE machinery, and hand-coded `rot(θ)` is now replaceable by the learned Operator (general for abstract columns).
    **Slice 2 DONE — `rot` DERIVES FROM `pose_operator` (suite 130):** `l5.rot(θ) = pose_operator(θ).M[:2,:2]`, so L2/3's
    direct `rot(±θ) @ v` uses in `sense()` flow through the ONE operator machinery (behaviour-identical, recognition green) —
    there is no longer a separate hand-coded rotation. **HONEST SCOPE CORRECTION (2026-07-01):** L2/3 was NOT broken — it
    already used SE(2) correctly; the *inconsistency* was L6/NAVIGATION being abelian while recognition was correctly
    non-abelian, so the fix is bringing L6 UP to the pose group (S1e), not rewriting L2/3. And **L4 needs NO direct edit** —
    `feature ⊗ location` is representation-agnostic, so "feature-at-pose" is INHERITED the moment L6 supplies a pose (S1e).
    So the AFFECTED-LAYERS "L4 → feature-at-pose / L2/3 reads the group" are CONSEQUENCES of L5/L6, not separate refactors.
    **The one genuinely-remaining L2/3-specific fold** = making pose INFERENCE group-general + LEARNED (`align_rotations` →
    `Operator.fit`), which must stay SYMMETRY-AWARE (`pose_between` returns MULTIPLE poses for symmetric patches = the
    stabilizer coset; plain Procrustes gives one). That is exactly [[project_symmetry_opportunity]] — it matters for
    ABSTRACT columns (whose group isn't SO(2)); for the visual column SO(2) inference is a legitimate Core-Knowledge plug-in.
  - **S1e — DRIVE STATE by the operator. ENGINE DONE (2026-07-01, `column.track_pose`/`pose_state`, suite 131):** the
    column path-integrates a POSE (an SE(2) matrix) by RIGHT-COMPOSING the learned body-frame operator (`P ← P·G`), and
    `pose_state` bins `(x, y, heading)`. Validated on OrientationWorld (`test_nonabelian_env.py`): the body-frame op is
    CONSTANT per action (learnable as `pose_before⁻¹·pose_after`), composing it dead-reckons the pose to MATCH the env
    (position + heading), FORWARD∘TURN ≠ TURN∘FORWARD in the belief, and **FORWARD is DETERMINISTIC over `pose_state`**
    (heading in the key: 4 headings → 4 distinct outcomes) — which the additive position CANNOT. The operator DRIVING a
    non-abelian state. NB the CODEBOOK BOUND is AVOIDED: the pose is binned DIRECTLY (no grid `decode`), and it is unbounded.
    **REMAINING (the live prerequisite): HEADING PERCEPTION.** The engine is additive/parallel (doesn't touch `_fovea`/
    `track_state`) because the live agent perceives POSITION, not orientation — to drive the live state it must observe or
    dead-reckon heading (and OrientationWorld must become a real perceivable frame). That perception is the next slice;
    then swap `track_state`→`pose_state` in the agent, gate no-regression on abelian, and SOLVE the env end to end.
    **HEADING-PERCEPTION PRIMITIVE DONE (2026-07-01, `column.track_heading`/`sense_pose`, suite 132):** the agent perceives
    its heading from the MOVEMENT DIRECTION — a forward move's position-delta direction IS the heading (`atan2(delta)`), no
    shape-orientation machinery, reusing the position observation it already has; `sense_pose` snaps the pose belief to the
    perceived (x, y, heading) (the pose analogue of `track`'s snap-to-sighting). Validated on OrientationWorld: movement
    direction == true heading at all 4 headings, and perception makes the 4 headings DISTINGUISHABLE in `pose_state` → FORWARD
    deterministic. *Honest limitation:* a TURN produces no movement, so heading is STALE until the next forward (robust
    heading via shape recognition / turn dead-reckoning is a follow-up). **LIVE-SOLVE steps (Cipher's 1-5):**
    - **Step 1 DONE (2026-07-01, `OrientationGame`, suite 133):** a real perceivable non-abelian FRAME (duck-typed like
      NavGame): an ORIENTED mover (asymmetric L) with body-frame FORWARD/TURN, reach-the-goal levels; validated non-abelian +
      solvable. `test_nonabelian_env.py`.
    - **Steps 2-3 DONE (2026-07-01, suite 134):** heading perception + pose maintenance are wired into `column.track`
      (`track_heading` from the movement delta → `sense_pose` corrects the belief each step), and `sensor.read` selects the
      state via the NON-ABELIAN GATE — `col.pose_state` when `L5.heading_dependent()` (an action's DIRECTION is inconsistent),
      else `track_state`. `L5.heading_dependent` = a high per-action direction-inconsistency residual (skips bumps; robust).
      OrientationGame now uses a SYMMETRIC mover (heading HIDDEN, rendered un-rotated) so heading-from-movement is clean.
      *Validated (`test_nonabelian_env.py`):* driving the real agent on OrientationGame, the gate trips (FORWARD inconsistent)
      and the state node becomes a POSE 3-tuple; NavGame stays on `track_state` — NO abelian regression (step 3 met).
      The gate + `track_heading` are SCAFFOLDING to dissolve at step 5 (into proper factorisation / one pose path).
    - **Step 4 — FIX VECTOR NAV (Cipher: fix the achiever, do NOT lean on the SR/graph planner). CORE DONE (2026-07-01,
      `column._pose_vector_action`, suite 135):** the POSE-AWARE achiever. The actions transform the POSE, so there is no
      fixed per-action displacement; instead descend `Φ(P) = distance(P.pos, goal) + λ·heading_error` over the pose after
      each action's learned body-frame operator (`pose_ops`) → ALIGN-THEN-ADVANCE emerges (TURN cuts the error term, FORWARD
      cuts the distance). `vector_action` GATES to it when `heading_dependent` (abelian stays byte-identical → no regression).
      Validated at the column level: given the pose belief + learned SE(2) operators, it navigates OrientationWorld to the
      goal USING turns (which the abelian `vector_action` cannot).
      **(a) ONLINE POSE-OP LEARNING DONE (2026-07-01, `column.learn_pose_op`, suite 136):** `G_a = pose_before⁻¹·pose_after`
      (the constant body-frame increment), EWMA'd + re-projected to SE(2); validated — learned from OrientationWorld pose
      transitions it recovers the true operators, and the achiever navigates with the LEARNED (not hand-given) operators.
      **REMAINING for the LIVE solve — it is now WIRING (the primitives are all built + validated):** (b) feed CLEAN heading
      online — route-1: the sensor extracts the mover's SHAPE and L2/3's `pose_between` recovers its orientation (already
      validated) → `sense_pose`; needs OrientationGame to render an ASYMMETRIC mover rotated (route-2 turn-staleness is why
      route-1 is chosen — the turn is visible in the shape). (c) call `learn_pose_op` + `achieve` in the agent loop; then SOLVE.
    - **Step 5 REMAINING:** DISSOLVE the redundant `_fovea`/`track_state` + the non-abelian gate once the pose path solves and
      is validated as the single one.
- **Stage 2 — DISCOVER relations by loop closure (the quotient).** Free composition path-integrates; relations (incl.
  commutativity) are found by loop closure under the **predictive-sufficiency** criterion (causal states / bisimulation, per
  `MATH_PHASE.md`) — close the coarsest partition that stays a sufficient statistic. *Gate:* on a task with a KNOWN
  presentation, the discovered relations match; the free part reads off, the relations are found. Fully dissolves the
  abelian assumption. *Fallback:* if online relation-discovery is unstable, restrict to relations that recur within a bounded
  horizon (bounded loop length) — the analogue of bounded-depth carry.
- **Stage 3 — SOKOBAN / non-abelian planning.** Faithful non-abelian map → planning = geodesic-in-Cayley-graph; the GSG
  proposes goal-configs; commitment holds the multi-step maneuver. *Gate:* Sokoban — the map now represents push dynamics
  and the planner searches the non-abelian graph. Confirms or falsifies the unlock. *Diagnosis on failure:* the Stage-0..2
  gates isolate whether the residual gap is representation (here) or planning/GSG (elsewhere).

## Risks (the spine — where this breaks)
1. **LEARNABILITY of the constrained representation — REFRAMED (2026-07-01).** The feared constraint⊥expressivity tension
   does NOT bite for `OnlineOperator`: orthogonality is a PROJECTION AT READ (Procrustes), not a constrained gradient step,
   so it never fights the fit (spectral radius stays 1; validated). The DEQ wall came from forcing ONE operator to be both
   contractive AND expressive via gradient — the projection-at-read design sidesteps exactly that. **The linchpin moves to
   COVERAGE/EXPLORATION:** the online operator is only well-estimated over the state region the stream samples, so broad
   (relatively uniform) exploration is required — a narrow/confined walk under-covers. Tractable (the agent has directed-
   exploration levers), but now the explicit dependency to design for. NB `OnlineOperator.operator()` re-SVDs on read → THROTTLE it (like the eigenpurpose).
2. **COST / dimensionality.** Non-abelian irreps are higher-dimensional (matrices, not phases); the code + operators are
   heavier. Needs truncation / band-limiting (low-order irreps only), the group analogue of the grid's finite scales.
3. **EXTRAPOLATION without the engineered guarantee.** A learned rep may not extrapolate like the hand-coded grid; Gao's
   constraint is what buys it, and it may hold only approximately → drift on long compositions (the carry-depth wall, again).
4. **RELATION DISCOVERY at scale.** Discovering the quotient online is the hard structured-world-model problem; loop closure
   + predictive-sufficiency is the plan but is unproven beyond toys; the wrong-merge failure mode is ever-present.
5. **SUFFICIENCY for Sokoban.** Necessary (a faithful map) but not obviously sufficient — the GSG/planning/commitment still
   apply. The refactor may unlock the SUBSTRATE without solving the task. Keep the claim honest.

## Preserved / reused (why it's a refactor, not a rewrite)
- **The abelian grid stays as the special case** (commuting rotations) → no regression on navigation / counting / magnitude.
- **L5 already has an operator notion** — the per-location forward model + the pose group-operators recognition reads — so
  generalise its interface, don't replace it. (One-step config-dependent effects already live in the forward model; this
  refactor is the COMPOSED, path-integrated FRAME you plan over.)
- **The online SR stays** as the conjunctive/lookup complement (any graph, no generalisation) — the grid/representation is
  the generalising half.
- **Loop closure / recognition (L2/3)** is reused for relation discovery — the same machinery, generalised from cyclic
  factors to group relations.

## Connections
`MATH_PHASE.md` (the master boundary: free/read-off vs quotient/search; the abelian ceiling; geodesic-in-Cayley-graph;
predictive-sufficiency factorisation). TEM — Whittington et al. 2020 (learned transition matrices + structure/content
factorisation). Gao et al. 2021 (path integration ⇔ group representation + isotropic/conformal isometry). [[reference_grid_sr_eigenbasis]]
(grid = SR eigenvectors — the abelian eigenbasis; non-abelian → matrix coefficients / Peter–Weyl). [[reference_tbt_reference_frame]]
(L6 = the movement-bootstrapped position frame). [[project_recurrent_world_model]] (composition-fidelity probe; the
constraint⊥expressivity learnability warning). [[project_math_hypothesis_probe]]. `HETERARCHY_PLAN.md` (non-abelian
composition is also where multi-frame binding via the thalamus starts to matter).
