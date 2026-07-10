# ROLLOUT_PLAN.md — deliberative planning (concretizes ARCHITECTURE.md P3) — **DRAFT 2026-07-09**

## 0. Why this plan exists (the finding that forced it)

A pure representability probe (`memory/project_linear_value_cannot_hold_sokoban`, method: linear-regress `V*` from value
iteration onto the feature code, check greedy-on-best-fit) proved that Sokoban L0's `V*` is **not in the linear span of
grid-cell features at any coordinate flavor or resolution** — absolute R²=0.754, relational 0.757, conjunctive 0.757,
high-res (d=504) 0.757, all FAIL; only tabular (R²=1.0) solves. So **a linear value read-off — the SR, `V=w·(W·φ)` — is the
wrong tool for Sokoban, and no feature reframing (relational, conjunctive, resolution) fixes it.** The wall is not the
coordinate system or the column count; it is that `V*` hinges on a discrete predicate (*is the block on the pad yet?*) —
a threshold/conjunction a linear-over-smooth-grid cannot represent. **Therefore Sokoban is a deliberative problem: search
over a model toward a goal, not a value read-off.** This is the DELIBERATIVE grain of `reference_brain_planning`, and it is
why the OLD rollout code solved Sokoban while the SR-value rewrite cannot.

## 1. The mechanism — three grains of planning (`reference_brain_planning`)

- **forward (1 step) = perception.** The column predicts the next state given an action. **EXISTS**: `column.perceive`/
  `forward`, driven by L5's efference copy + the hippocampus path-integration (`reference_model_ownership_map`).
- **SR (`V=M·R`) = ROUTINE value** (open navigation). **EXISTS**: `l6_sr.SuccessorFeatures`. **PROVEN insufficient for
  Sokoban** → demoted to a rollout LEAF heuristic + routine nav only, never the whole value.
- **ROLLOUT = DELIBERATION for the hard cases** = simulate action sequences through the forward model toward an IMAGINED
  goal, **SAMPLED** (not exhaustive 2^K), **BG-committed**. **NEW** (this plan). Grounds: `reference_commit_to_test_a_hypothesis`
  (model-based sim toward an imagined goal + commitment), `reference_efficientzero_v2` (sampled search).

## 2. The generalization bar (the old rollout's failure — do NOT repeat it)

The old rollout solved the replicas but was **hand-fitted per game** (hundreds of iterations each, no transfer). The model,
the goal, and the search must be **learned / generic — NO per-game code, no env-specific subgoal enumeration**
(`feedback_bitter_lesson`, `feedback_subgoal_types_from_dynamics`). **Acceptance for the whole plan = solves a game it was
NOT built for**, within the action budget. Breadth is the axis a single game cannot fake.

## 3. The pieces (reuse first — `reference_model_ownership_map`)

1. **The ONE operator — the factored forward model.** A single learned function `O(an object's relations, the action) →
   that object's change`, applied IDENTICALLY to every object (self included), predicts the whole next CONFIG. **There is NO
   self-vs-other operator split** — that was a false distinction: the self is equally conditional (a wall blocks it too), so
   self-motion and object-motion are the SAME kind of thing read from a different context.
   - the **SELF** is just the object whose change directly tracks the action — the DISCOVERED controllable ROOT (not
     assigned; the object whose motion correlates with the action, already `hip.controllable()` / `L5.learned`). `O` over its
     context returns the efference displacement (or 0 into a wall). **EXISTS** as this self case (L5 efference + hippocampus).
   - **every OTHER object's** change = the SAME `O` over ITS relations to the root / neighbours ("a mover is against me,
     heading my way" → a push; a switch-link → a toggle). The KIND of dependence EMERGES from the dynamics — "contact/push"
     is never coded, it is what `O` LEARNS for a block in contact; another game fills the same slot with a different
     relation. (`feedback_one_model`, `feedback_subgoal_types_from_dynamics`, `reference_operator_as_group_representation`.)
   - **To build:** today `O` covers only the self-root (L5's efference). Generalize the SAME operator to read an object's
     relations and predict its change. NB the retired "other-object driver" was a SEPARATE tabular mechanism — the opposite
     of this; do NOT restore that. One operator, all objects.
2. **The imagined goal** — the target state (win condition) LEARNED from the sparse score / WIN signal (no goal is given —
   `reference_arc_agi3_signals`). Reuse the reward/goal-discovery machinery; the rollout simulates toward this state.
3. **The rollout search** — an EZ-V2-style sampled planner over the model (piece 1), read as a POLICY-IMPROVEMENT operator
   (`reference_efficientzero_v2`, arXiv 2403.00564; grounded from the paper 2026-07-09):
   - **sample** K candidate actions (Gumbel-Top-k over the policy prior — a mix of exploit-from-policy + explore-from-a-
     flattened-prior), NOT the full 2^K.
   - **allocate** the sim budget by **Sequential Halving** — progressively drop low-Q candidates, concentrate rollouts on
     the survivors (EZ-V2 plans with ~32 imagined states TOTAL). This is what keeps the search inside ARC's budget.
   - **evaluate** via **SVE** — the leaf value is the AVERAGE of several imagined-rollout returns (discounted model rewards
     to a horizon + the SR value at the leaf), NOT a single value read-off. The SR's proven ~76%-R² weakness is TOLERABLE
     here because it is only a leaf bootstrap the rollout corrects — exactly the role the representability finding leaves it.
   - the search returns a provably-improved action (`q(s, aS*) ≥ E_{a~p}[q]`); distil it back into the policy prior so the
     next search is sharper. **Deliberation IS the policy-improvement step** — iterated, it converges; it is not a one-shot read.
   **We take the ALGORITHM, not EZ-V2's neural nets.** Its gradient-trained representation/dynamics/value/policy conflict
   with the no-training + SDR-native ethos and the 4 GB / no-training-here constraint. Our substrate supplies the pieces:
   the ONE operator (piece 1) = the dynamics; the SR = the value; and our model already learns by ONLINE prediction-error
   (perceive's residual / HTM), which is the SDR-native analogue of EZ-V2's SimSiam latent-consistency loss — plausibly more
   sample-efficient (one-shot binding, Monty). EZ-V2's sparse-reward strength (Cartpole-Swingup-Sparse 763 vs Dreamer 392)
   comes from exactly this value-at-leaves + model — which answers R0's worry that un-valued sampling never hits the win.
4. **Commitment** — the basal ganglia SELECT and COMMIT the chosen plan (STN "hold your horses" — `reference_basal_ganglia`);
   avoids per-frame re-deliberation / dithering. **EXISTS**: `basal_ganglia`.

## 4. Staged plan (dependency-ordered; each stage paper-test gated + TBT-mechanism re-checked — `feedback_check_tbt_accuracy_per_step`)

**Build discipline (per the user, 2026-07-09):** a critical stage is built to a JUDGED END, not abandoned on an intermediate
dip. If the mechanism regresses mid-build, PERSEVERE and finish the whole thing, THEN judge — do NOT revert
(`feedback_dont_salvage_between_critical_steps`).

- **R0 — guard the finding + validate SEARCH on the true model.** (1) Promote the representability probe to a permanent test
  (the linear-value dead-end is a standing constraint) — `test_sokoban_hold`. (2) Validate search CORRECTNESS: an UNINFORMED
  forward-model search (BFS from the model + win predicate, NO hand-coded heuristic, NO per-game code) finds an executable
  winning plan on Sokoban L0 AND multi-cell M0 — `test_rollout`. Deliberately uninformed: it proves the model + goal are
  plannable with zero domain knowledge. NB EZ-V2 grounding — un-valued SAMPLED search would NOT hit sparse reward (random
  shooting finds L0 w.p. ~1e-9), so the SAMPLED / value-at-leaves search is deferred to R3 where the SR-leaf earns its place;
  R0 proves correctness, R3 makes it cheap. *Accept:* both tests green.
- **R1 — generalize the ONE operator to every object, USING `SequenceMemory`.** The operator IS the column's existing
  `sequence.py SequenceMemory` (HTM temporal memory — context-conditioned next-element prediction; "the predictive cells ARE
  the next-element prediction", already the intended home for "L2/3 next DISPLACEMENT / an object's BEHAVIOR indexed by its
  phase"). It EXISTS but is not yet wired into the live forward model (open item). R1 = feed each object's RELATIONAL context
  (its neighbours' rel-pos/shape + the action, as the SDR distal/basal input) to `SequenceMemory` and predict that object's
  next element — displacement (L2/3) / content (L4) — so the model outputs the whole config (self ⊕ blocks ⊕ …). WHICH
  relation matters EMERGES from distal-synapse learning (SDR overlap), NOT a symbolic rule-inducer or set-intersection — do
  NOT build a bespoke predictor (the discarded `ObjectOperator` prototype was that mistake). Today `O` covers only the
  self-root (L5's efference); this extends the SAME sequence mechanism to every object. *Accept:* it predicts a push
  correctly OFFLINE on Sokoban AND a structurally DIFFERENT effect (e.g. a toggle) on a non-Sokoban dynamic, SAME code, no
  per-game branch.
- **R2 — the imagined goal from the score.** *Accept:* the target state is identified from the sparse WIN/score alone, no
  hand-coding.
- **R3 — the EZ-V2-style sampled rollout** over the learned model: Gumbel-sampled candidates + Sequential-Halving budget +
  SVE leaves (imagined-rollout returns + the SR at the leaf), returning a policy-improved action distilled back into the
  prior; the BG commits it. The SR is a LEAF bootstrap the rollout corrects (its proven ~76% R² is fine THERE), not the
  value. §3.3 is the mechanism. *Accept:* solves Sokoban L0 + `MULTICELL_LEVELS` AND a game it was NOT built for, no
  per-game code.
- **R4 — efficiency under the budget.** *Accept:* within ARC's 5×-human action budget on the replicas; the per-frame search
  is bounded (`feedback_slow_run_means_catastrophic_failure`: never re-search × game-length × a growing pool).

## 5. Risks / open (honest)

- **Sample efficiency is the crux.** The forward model must be learned ONLINE from FEW interactions — ARC gives no free
  practice (`reference_arc_agi3_scoring`, `project_continuous_online_loop`). If the model needs many episodes, the approach
  fails ARC's budget regardless of correctness.
- **Search cost.** Rollout depth × branching is real; sampling (EZ-V2) mitigates but does not remove it. The per-frame
  budget rule is hard.
- **Generalization.** The model + goal + search must encode NO per-game structure — the exact failure of the old rollout.
- **This is the model-based-RL frontier under a tiny sample budget** — the quadrant ARC is designed to make hard. Genuinely
  open, not a solved recipe. R0/R1 are the cheap de-risking gates; do not commit to R3/R4 until R1 shows the model learns fast.

## 6. Relation to the existing plan

Concretizes ARCHITECTURE.md **P3 (relations+planning)**. The single-column MICROCIRCUIT perception stages (incl. 5d) are
NOT the blocker — perception mostly works — and are subordinated. The **heterarchy is reconsidered**: not for a relational
VALUE (refuted, §0), but *possibly* as the factored MODEL a rollout searches over (R1). That is secondary to the rollout
mechanism and TBD — decide it at R1, not before.
