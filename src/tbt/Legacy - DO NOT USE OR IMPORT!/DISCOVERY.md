# DISCOVERY — exploration, babbling, hypothesis generation & testing

*Companion to `ARCHITECTURE.md`. That document is the model + the plan; this one is the home of everything about how the
agent finds what it does not yet know — the exploration/discovery frontier and the research behind it. ARCHITECTURE.md
§8 (planning) and §9 (hypothesis generation) are the load-bearing summaries; this document holds the detail, the
neuroscience, and the staged plan, so the main doc stays lean. Where the two disagree, ARCHITECTURE.md wins for the
*mechanism-in-code*; this doc wins for *what the research says and where we are going*. Cross-references are `[[memory]]`
links and `§n` section pointers.*

---

## 1. The one problem, and the one shape

**Navigation vs. discovery are different problems.** Navigation gets you to a goal you already know (`navigate_vector`:
the grid-cell goal vector modulated by SF value — §8, corrected by experiment `project_sf_value_not_greedy_navigable`).
**Discovery** is finding the rewarding goal — or learning the dynamics — *the first time*, before any value signal
exists. It is the current bottleneck: NavGame is 0/8 because the goal is never *found*, not because it can't be reached
(the exploit stack sits idle with `_goal_pos_raw = None`, SF value flat 0).

**The unifying shape (the thesis of this document).** Because greedy-on-value does **not** navigate over the grid SDR
(`project_sf_value_not_greedy_navigable`), *every* form of discovery reduces to the same loop:

> **sample a target-state → reach for it (`navigate_vector` / the inverse model) → observe the outcome → learn.**

The forms differ **only in how the target is sampled**:
- **model-acquisition** — sample motor/outcome targets to learn *what actions do* (the operators, the self);
- **cued** — a **salient** percept proposes the target (the colour marker);
- **uncued** — **novelty / learning-progress** proposes the target (coverage of the unknown);
- **hypothesis-driven** — a **relational search** proposes the target (Sokoban, carry — the hard case).

This is why goal babbling, salience-cued goal-setting, and hypothesis generation are **one mechanism** with different
target-samplers, not four systems (rule 1). The rest of this document is: the samplers (research), the tester, and the
plan.

## 2. A taxonomy of discovery

| Form | Target sampled from | Neural frame | Our status |
|------|--------------------|--------------|------------|
| **Model acquisition** | motor/outcome space (babble) | motor babbling → forward/inverse model; songbird subsong | operators learned in `perceive`; no explicit babble phase |
| **Cued** | bottom-up **salience** | priority map (SC ⊕ value), BG-selected | salience is MOTION-only → the marker is invisible (§4) |
| **Uncued** | **novelty / learning-progress** | intrinsic motivation, epiplexity | `col.act` frontier optimism (weak, 0/8) |
| **Hypothesis-driven** | **relational search** in the learned group | PFC/OFC + hippocampal preplay | `discover_relations`/`factor_group` built; search un-committed (§7, MATH_PHASE) |

These are ordered by difficulty and, roughly, by development: babble to learn the body/dynamics, then cued reaching,
then uncued coverage, then relational problem-solving.

## 3. Babbling — the developmental exploration primitive

**Motor babbling (avoid).** Random actuator commands to learn the sensorimotor map. Both biologically wrong ("infants
explore by far not randomly or exhaustively… they attempt goal-directed actions days after birth") and computationally
hopeless — the **curse of dimensionality**. For us it is fatal on the click action (ACTION6 = a 64×64 coordinate space):
random pixel-babbling is the blind-contact search we rejected.

**Goal babbling (the one that matters — Rolf, Steil & Gienger 2010/2011).** Sample desired **outcomes** in goal/task
space, attempt them with the current (rough) inverse model, learn from the result. Exploration and learning form a
positive-feedback loop; it **scales to high dimensions** (cost barely grows 2→50 DOF; orders of magnitude over motor
babbling). *This is literally our `navigate_vector` loop* — goal babbling is the developmental name for "sample a target,
reach, observe."

**Intrinsic motivation / learning progress (Baranes & Oudeyer, SAGG-RIAC 2013; Oudeyer, Gottlieb & Lopes).** Sample
goals where **competence progress** is maximal → an autonomous **curriculum** of increasing complexity. This is our §8
epistemic term (epiplexity = learning progress; [[reference_efe_and_epiplexity]], [[reference_animal_exploration]])
applied to *goal selection*: it decides *which* target to babble toward, and it winds down on both mastered and
irreducibly-noisy regions (no noisy-TV trap).

**The neural substrate (songbird).** Juvenile birds babble (subsong); **LMAN injects motor variability**, the
basal-ganglia homolog **Area X evaluates outcomes by a dopaminergic reward-prediction error** and reinforces the
variability that worked. Exploration is therefore **BG-gated *structured* variability + dopamine-RPE**, not uniform
ε-noise — it tells us *where* to inject exploration (through the basal ganglia, biased) and *how* to shape it (RPE).
Grounds [[reference_basal_ganglia]] (tonic-dopamine explore/exploit, OpAL Go/NoGo).

**What we take:** goal babbling (never motor babbling) is the `col.act` replacement — sample an outcome-space target
(cued or novelty/learning-progress-biased), reach via `navigate_vector`, learn; BG-gated variability provides the
exploratory perturbation; competence-progress builds the curriculum.

## 4. Goal-setting & hypothesis generation — where the target comes from

**The priority map (what *sets* the target — [[reference_goal_setting_priority_map]]).** A **priority map** fuses
**bottom-up salience** (feature-contrast + prediction-error surprise; superior colliculus) with **top-down value/memory**
(reward/goal-tagged locations — Gauthier & Tank 2018); the **basal ganglia** select its peak (Fecteau & Munoz 2006). The
winner is held as a **goal vector** — goal-direction + goal-distance cells, memory-based even for an occluded goal (Sarel
et al. 2017) — exactly the vector the grid cells compute toward (Bush, Barry & Burgess 2015; §8). Place-cell
**goal-oriented vector fields** converge on it (ConSinks — Ormond & O'Keefe 2022).

**Our gap: salience is MOTION-only.** `retina.salient_cells` = temporal change, so a static distinct-colour marker is
never salient → never a candidate target (`project_marker_exposes_hippocampus_prereq`). The fix is a **feature-contrast**
salience channel ("distinct from the surround"), which turns the marker into a sampled goal. NB the ARC board is a fixed
top-down world, so the marker's **frame position is already allocentric** — cued discovery here needs *no* hippocampus.

**Sampling, not enumeration (Dasgupta, Schulz & Gershman 2017).** The mind samples a *few* candidate target-states from
memory, cued by context and biased by priors **salience × controllability × ambiguity** (more samples when more
uncertain). Controllability = reafference ("it moved when I acted", §4c); salience = feature-contrast + prediction-error;
both learned ([[reference_hypothesis_generation]]).

**The master boundary — read-off vs. search (MATH_PHASE.md).** A hypothesis is a short composition of learned operators
toward a cued target — geodesic-finding in the learned structure. Its *cost* is predicted by structure: where the group
is **free/abelian** it is **READ OFF** (a homomorphism is fixed by its action on generators — cheap; NavGame reaching);
where it is **relational/quotient** (non-commuting, constrained — Sokoban, carry) it must be **SEARCHED**. `discover_
relations`/`factor_group` build the Cayley graph the search runs over; whether the relational search is tractable at
scale is the open question (§7, [[project_math_hypothesis_probe]]).

## 5. Testing — acting is testing a hypothesis

A sampled target-state is a hypothesis "bring about X." **Testing = reach it and observe** (§8: "acting = testing a
hypothesis"):
- **Reach** — `navigate_vector` (the inverse model / goal vector). Before committing overtly, the hippocampus can
  **preplay** candidate trajectories (vicarious trial-and-error — Redish), the critic scoring them.
- **Score** — the outcome under **Expected Free Energy**: **pragmatic** = reward toward the goal (SF value), **epistemic**
  = information gain / competence progress (epiplexity; → 0 for both mastered and noise). One value; explore vs. exploit
  *emerges* from which term dominates ([[reference_efe_and_epiplexity]], [[reference_exploration_replay]]).
- **Commit / switch** — the basal ganglia commit through the maneuver (STN "hold-your-horses"); persistent refutation
  (ACC persist-vs-switch) drops the hypothesis. Credit is assigned backward along the sequence (reverse replay /
  prioritized sweeping, Mattar & Daw — the *sparing* deliberative grain, §8).

The **eigenoption/EFE dead-zone caveat** ([[reference_eigenoptions_subgoals]]): a locally-exhausted, reward-less region
has flat value → greedy degenerates to a random walk. The resolution is NOT a separate eigenoption explorer (dropped,
§8) but the **learning-progress target-sampler** of §3–§4 giving the vector navigator a concrete frontier target to
reach — the dead-zone becomes "sample the nearest under-learned target, go test it."

## 6. How it maps onto our architecture

One loop, seated in the existing organs (no new "explorer module", cf. §8's "no planner module"):
- **Target-sampler** (the priority map, §4) — proposes a candidate target-state: salient percept (cued) OR
  novelty/learning-progress frontier (uncued) OR a relational hypothesis (search). Lives at the L2/3 + peripheral
  salience boundary; selection is the **basal ganglia**.
- **Reach** — `navigate_vector` (L6 grid-cell vector ⊗ L5 operator ⊗ SF value modulation).
- **Variability** — **BG-gated** exploratory perturbation of the selected action (songbird LMAN/Area X), scaled by
  uncertainty (tonic dopamine).
- **Learn** — the SF value (`learn_location_value`), the operators (`perceive`/`learn_pose_op`), and the
  competence/epiplexity estimate (`reward.epistemic_value`).

This **replaces `col.act`** (the weak frontier-optimism explorer) and thereby unblocks retiring the tabular stack
(`OnlineSR`/`reward.plan`/`state_node`) — the two are one slice (ARCHITECTURE.md §10 P4a re-seat 2/2).

## 7. The plan — staged, up to Sokoban

- **D1 — Cued discovery on NavGame (next).** Add feature-contrast salience → the colour marker becomes a sampled goal →
  `navigate_vector` tests it → reward → remember → beeline. Closes NavGame end-to-end and validates the target-sampler +
  navigator live; simultaneously the `col.act` replacement, unblocking the tabular retirement. No hippocampus needed
  (frame = allocentric).
- **D2 — Uncued coverage.** When no cue exists, sample the frontier by **learning-progress** (goal babbling +
  SAGG-RIAC-style competence curriculum), reach via `navigate_vector`, with BG-gated variability. Fixes the EFE
  dead-zone without an eigenoption explorer.
- **D3 — Model-acquisition babbling.** An explicit early phase: goal-babble to learn the operators / controllability /
  content dynamics before reward — the substrate for everything downstream. (Much already emerges in `perceive`;
  D3 makes it deliberate and curriculum-ordered.)
- **D4 — Hypothesis-driven / relational search (the frontier).** For order/config-dependent problems, the target-sampler
  proposes a **relational hypothesis** searched over the learned Cayley graph (`discover_relations`). Gated on the
  read-off↔search boundary probes (MATH_PHASE); no code commits until they answer.

**Sokoban — the honest boundary.** Babbling **bootstraps the model Sokoban needs** (the push operator + its
preconditions; a competence curriculum move→push→one-box-home→solve; and, critically, **learned aversive value for
deadlocks** — a box in a corner is a `−∞`/absorbing cost, [[reference_obstacle_as_transition_cost]],
[[reference_cost_field_aversive_value]]). But it does **not solve** Sokoban: solving is **model-based planning +
commitment** toward an imagined *config* goal over an **irreversible** state space ([[reference_commit_to_test_a_hypothesis]]).
Three reasons pure exploration cannot cross it: (1) **irreversibility/deadlocks** — the outcome space has absorbing traps,
so goal-babbling softlocks and can't retry; (2) **long-horizon order-dependence** — a specific push *sequence* (the
non-commuting Cayley graph, P3); (3) the **goal is a configuration**, not a reachable point. So babbling feeds the
planner (dynamics + curriculum + deadlock-aversion); the planner (SR/rollout toward the imagined goal, deadlocks as
`−∞`, STN commitment) does the solving. **Discovery gets us to Sokoban's doorstep; planning crosses it.**

## 8. Open questions (the frontier)

1. **Relational-search tractability** — is hypothesis search over the learned quotient tractable at scale, or does it
   wall (MATH_PHASE outcomes A/B/C)? The one honestly-open ML question (§9).
2. **The read-off↔search boundary** — can the agent *predict* which regime a problem is in (free/abelian → read-off vs.
   quotient → search) and route accordingly, rather than always searching?
3. **Deadlock foresight** — learning the aversive value of *irreversible* states fast enough to avoid them during
   exploration (before experiencing the softlock), i.e. model-based deadlock detection vs. learned cost.
4. **Curriculum without a designer** — does competence-progress goal-sampling *autonomously* order NavGame→…→Sokoban, or
   does the ordering need scaffolding?
5. **Salience learning** — feature-contrast salience must be *learned* (what is distinct-from-surround in this domain),
   not a hand-coded colour/edge detector ([[feedback_bitter_lesson]]).

## References

**Goal-setting / vector navigation:** Bush, Barry & Burgess 2015 (grid vector nav); Sarel et al. 2017 (goal-vector
cells); Ormond & O'Keefe 2022 (goal-oriented vector fields); Fecteau & Munoz 2006 (priority map); Gauthier & Tank 2018
(reward cells); de Cothi & Barry 2020 (SR barrier-warping). **Babbling / intrinsic motivation:** Rolf, Steil & Gienger
2010/2011 (goal babbling); Baranes & Oudeyer 2013 (SAGG-RIAC); Oudeyer, Gottlieb & Lopes 2016 (curiosity/learning
progress); songbird LMAN/Area X — Ölveczky et al., Andalman & Fee, Woolley & Kao (BG variability + dopamine-RPE).
**Hypothesis generation:** Dasgupta, Schulz & Gershman 2017 (sampling; salience×controllability×ambiguity). **Testing /
planning:** Friston (active inference / EFE); Redish (VTE); Mattar & Daw 2018 (prioritized replay). **Boundary:**
`MATH_PHASE.md` (read-off vs. search).

**Cross-refs:** ARCHITECTURE.md §4c (reafference), §8 (planning / no planner module), §9 (hypothesis generation), §10
(P4a re-seat). Memories: [[reference_goal_setting_priority_map]], [[reference_hypothesis_generation]],
[[reference_animal_exploration]], [[reference_efe_and_epiplexity]], [[reference_eigenoptions_subgoals]],
[[reference_exploration_replay]], [[reference_commit_to_test_a_hypothesis]], [[reference_basal_ganglia]],
[[project_sf_value_not_greedy_navigable]], [[project_marker_exposes_hippocampus_prereq]], [[project_math_hypothesis_probe]].
