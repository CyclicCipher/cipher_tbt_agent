# Motor refactor — action by active inference (GSG → BG gate → inverse-model motor)

*Draft for review, 2026-06-29. Companion to `REFACTOR_PLAN.md` (the layer refactor, done) and
`TARGET_ARCHITECTURE.md`. Nothing here is executed until signed off. This commits to the **full active-inference
reframe** of motor output, not a patch.*

---

## 0. Why (the three critiques + the principle + the competition)

The agent claims "expected free energy = epistemic + pragmatic," but the mechanism doesn't match:

1. **R-MAX optimism is in the wrong place and double-counts.** `agent._choose` gives every untried `(s,a)` the
   flat `Vmax`, while `reward_total` *already* adds a learning-progress epistemic bonus — two exploration drives
   stacked, and the blunt one wins. R-MAX also inflates the *pragmatic* term ("unknown ⇒ max reward") when the
   right home is the *epistemic* term ("unknown ⇒ max expected info gain, decaying as learned"), and it keys on
   **state visits**, not the active-inference quantity: the model's **uncertainty about the action's outcome**.
2. **Selection lives in the agent script.** A top-level argmax over `reward.V` is a monolithic policy. The
   faithful selector is the **basal ganglia gating among proposed goal-states** (Go/NoGo + dopamine-RPE) — the
   mechanism that becomes *consensus* with multiple columns. `BasalGanglia` exists and is simply not in the loop.
3. **The motor organ won't generalize**, because the column picks an *action index*, not a *goal-state*. L5.motor
   is `return a` (identity). In a faithful system the column emits a **goal-state** (a desired observation, i.e. a
   desired **displacement** in its frame) and the motor selects the command that *fulfills* it (action-as-inference).

**The principle (active inference):** action samples the world to lower free energy — either reach **preferred**
states (pragmatic) or reduce **uncertainty** (epistemic). The agent should select the goal-state (and the action
that fulfils it) that minimizes **expected free energy** `G = −pragmatic − epistemic`.

**Why this generalizes beyond ARC (the competition's "spirit").** The thing that carries to an arm, a cursor, or
any task is the **separation**: the column emits a *frame-relative goal-state*, an *effector-specific organ*
realizes it via L5's operator run as an **inverse model**. The column never names "ACTION3." A discrete action
set is then just the simplest organ; a continuous effector is a learned inverse model with the *same* interface.
This separation is the generalizable architecture — it is a first-class goal here, not a side benefit.

---

## 1. The GSG — how it works, and why one per column

**One GSG per column.** A column has its own reference frame and its own uncertainty, so "what sample would most
reduce *my* uncertainty?" is column-local (Monty: every LM emits goal-states). A monolithic GSG can't know what
disambiguates a given column's hypotheses and wouldn't generalize to a hierarchy.

**A goal-state** = a target the column wants to bring about, in its representation: a target **location** (a node
/ pose in its frame) or a target **feature-at-location** (sense feature F at location L), or a target
**object-state**. The GSG proposes a *set* of candidates and scores each by EFE:

- **Pragmatic value** = `reward.V(goal-state)` — expected reward of reaching it (toward the learned goal).
- **Epistemic value** = expected information gain from reaching/sensing it:
  - *world-model* uncertainty: how under-modeled is `(s,a)` near there — **transition-outcome uncertainty**,
    **gated by learning-progress** (draws to *learnable* unknowns, ignores the noisy-TV — `reference_animal_exploration`).
  - *recognition* uncertainty: how much would sensing there **disambiguate L2/3's `(object,pose)` hypotheses**
    (sample where the hypotheses most disagree — Monty). Newly possible: L2/3 now *has* those hypotheses.

**Covert vs overt — the resolution of "many columns, one body":**
- **Epistemic goals are pursued COVERTLY where possible** — path-integrate the belief to a locus and sense the
  *predicted* feature there, *without moving the body* (premotor theory of attention; "mental saccades"). Parallel
  across columns, **free** (spends compute, not actions — the action-expensive / compute-cheap lever).
- **Overt goals compete** (one effector). The **basal ganglia gates**: Go the highest-EFE overt proposal, NoGo the
  rest; dopamine-RPE learns which proposals pay off. That is the consensus / arbitration.

**Covert evaluation = how the GSG scores EFE.** To score a candidate, the GSG **rolls its belief forward mentally**
over the SR/graph (path-integration via L5's operator) — imagination, no real action spent. This is also how
expected info gain is estimated ("if I went there, what would I resolve?").

---

## 2. The faithful loop (and what each piece fixes)

> each column's **GSG** proposes **goal-states** ranked by **EFE** (§1) → **covert** epistemic goals are sampled
> in imagination (free) → **overt** proposals go to the **basal-ganglia gate** (Go/NoGo, dopamine-RPE) → the
> **motor** selects the action that *fulfils* the gated goal-state via L5's **inverse** operator + an SR/graph
> path plan → the **organ** maps that action to the effector.

- Fixes **#1**: EFE replaces R-MAX — the epistemic bonus is expected info gain on transition-outcome uncertainty
  (decaying as learned), placed in the epistemic term, not flat optimism in the pragmatic term.
- Fixes **#2**: the BG gate replaces the agent argmax — selection is consensus-shaped (single-column degenerate,
  multi-column real).
- Fixes **#3**: the motor fulfils a *goal-state* via L5's inverse operator — column output is frame-relative and
  effector-agnostic; only the organ is effector-specific.
- **Bonus — fixes the deferred "L6 is write-only" thread:** reaching a goal-location is **navigation**, which
  reads the SR (reachability / place-code distance) to plan the path. The motor is where `OnlineSR` finally gets
  *read*. The SR is the navigational substrate the inverse-model motor plans over.

---

## 3. What a goal-state is, concretely (our representation)

State is `(feature_id, position)` (feature-at-location). A goal-state is a partial target over that:
- **navigation/exploration:** a target `position` (a node in the column's graph) — "be there". The motor plans a
  path over the graph/SR; the action = the first edge / the L5 displacement reducing the gap.
- **disambiguation:** a target `feature` to confirm at a `position` — "sense F there". Often covert (predict it).
- **task:** a target object-state / required-absent — the pragmatic goal from the sparse score (already in `reward`).

The **inverse model** for ARC: enumerate the discrete actions, pick the one whose L5 effect best reaches the
goal (single-step) or whose path over the graph/SR reaches it (multi-step). For a continuous effector the same
interface holds with a learned/optimized inverse policy (gradient on the goal-discrepancy).

---

## 3b. Computing EFE online (research-grounded — `reference_efe_and_epiplexity`)

`G = −(pragmatic + epistemic)`; select the goal/action that **maximizes (pragmatic + epistemic)**.

- **Pragmatic** = expected reward toward the goal (`reward.V` at the predicted outcome). Formally the *risk* term
  `KL[predicted outcomes ‖ preferences]`; for sparse ARC the preference is "completion", so this is the learned
  value. Unchanged from today.
- **Epistemic** = expected **information gain**, and the noisy-TV is resolved *at the measure level* by
  **epiplexity** (compute-bounded learnable structure ≈ the area between the loss curve and its floor). The clean
  consequence: a **flat** loss curve has zero remaining epiplexity whether the floor is low (mastered) or high
  (noise) — so the epistemic value → 0 for *both*, **with no separate noise gate** (drop reward.py's
  `noise = err_slow − 4·lp` hack). Two grounded sources, both noise-robust:
  1. **World-model epiplexity (curiosity).** Per `(s,a)`, the prediction-loss **drop rate** = learning progress
     `lp = max(err_slow − err_fast, 0)` — the online epiplexity-extraction rate. Use `lp` **directly** as the
     epistemic value for visited `(s,a)`.
  2. **Recognition info gain (disambiguation).** Expected reduction in L2/3 `(object,pose)` hypothesis **entropy**
     from a sample (sample where hypotheses disagree — Monty). A noisy region doesn't disambiguate → ≈ 0.
- **The frontier without R-MAX.** An unvisited `(s,a)` has *unknown* epiplexity → a **prior expected-epiplexity
  (optimism) that DECAYS to the measured `lp`** as it is visited. This is the principled replacement for R-MAX:
  optimism lives in the **epistemic** term (decays to 0 once a region is shown unlearnable/noise, stays high while
  learnable) — **not** flat in the pragmatic term (R-MAX's bug: noise looks max-valuable forever).
- **Honest:** epiplexity's exact form is offline NN-training MDL; we use it as the *justification* for
  learning-progress-as-epistemic + the cleaner "flat-curve → 0" rule, not as a literal online computation.

**Sign structure (aversion vs curiosity).** Pragmatic value is a log-preference, so it is **≤ 0**: ~0 for
expected *preferred* outcomes, strongly negative for un-preferred ones. **This is how "don't do this" works** —
not a separate penalty, but assigning the bad outcome a low preference `p(o)≈0`, whose strongly-negative log
dominates the EFE. Our `reward.V ≥ 0` today (no aversion); the natural aversive outcome to add is
**GAME_OVER / death** (low preference → avoid traps), which is *distinct* from the zero-value "useless". Epistemic
value is **≥ 0 always** — expected information gain = mutual information ≥ 0 (you can never *expect* to become less
certain); its floor is 0 (learn nothing — a fully predictable *or* a pure-noise outcome). A *realized* observation
(or a measured `lp`) can dip negative (surprise / forgetting) → we clamp at 0. Information *avoidance* (the ostrich
effect) is a **pragmatic veto** (the outcome is aversive), never negative epistemic. So: **all aversion lives in
the pragmatic term (≤ 0); the epistemic term is pure non-negative curiosity.**

## 3c. Efficiency — reaching goals in the FEWEST actions (RHAE)

RHAE scores `(human/agent_actions)²`, so efficiency is first-class. The principled driver is **minimizing EFE over
the trajectory with a per-action COST**, plus learning that makes the true optimum findable. Five pressures, all
already in or adjacent to the design:
1. **Temporal discounting (have, γ):** reaching the goal sooner is worth more — the basic "fewer steps = better".
2. **Action as cost / effort (ADD):** a small *negative pragmatic value per step* (occupying a non-goal state is
   mildly un-preferred — the §3b aversion mechanism applied to *effort*). This makes minimizing action count a
   *direct* objective. Resource-rationality / the brain's "law of less work"; the FEP *complexity* term is its
   formal cousin (prefer the simplest policy that achieves the goal — Occam over actions).
3. **Optimize, don't satisfice — via the accurate model + the SR:** EFE picks the min-cost policy, but only as
   good as the model. As the epistemic term winds down (epiplexity extracted → model learned), the planner reads
   the **SR's reachability/shortest-path gradient** (the deferred L6-read — now doubly motivated: navigation *and*
   efficiency) and finds the true shortest path. The explore→exploit transition is automatic in EFE.
4. **Cheap exploration (covert):** exploration *costs actions*, which hurts RHAE — so push it into **covert**
   (compute, not actions) sampling wherever possible (the GSG's mental rollout). Learn by thinking, not by acting.
   This makes the GSG's covert evaluation an *efficiency* mechanism, not only an attention one.
5. **Habit amortization + residual policy-search (BG dopamine-RPE):** reinforce the successful efficient policy so
   it is reused cheaply (cross-level transfer already shows this — a 2nd same-goal level ≈ oracle), while keeping a
   *small* exploration in POLICY space (try alternatives where the value is near-tied / the policy is uncertain) to
   discover shortcuts = "is there a *better* way?" — the difference between converging to optimal and satisficing.

## 4. File-by-file disposition

| File | Change |
|---|---|
| `reward.py` | EFE = pragmatic + epistemic (§3b). Epistemic = `lp` (the epiplexity-extraction rate) directly for visited `(s,a)` + a decaying prior for the frontier; **drop the R-MAX `Vmax`** AND the `noise = err_slow−4·lp` gate (both subsumed — a flat curve gives `lp→0`). Keep R-MAX as an ablation flag only. |
| `basal_ganglia.py` | **Into the loop** as the overt action/goal-state gate (`gate` already exists: Go/NoGo + dopamine-RPE). |
| `l5_displacement.py` | Add the **inverse operator** — `action_toward(state, goal)` / `fulfil(displacement)`: which action's learned effect best reaches the goal (the forward operator run backwards). L5.motor stops being identity. |
| `l6_sr.py` / `column.py` | The motor **reads the SR** for multi-step path planning to a goal-location (fixes write-only). A `goal_state`/GSG faculty proposes + EFE-scores candidates; covert rollout via `loc_*` + L5. |
| `agent.py` | Thin: hand the loop to GSG → gate → motor. **Remove the value-greedy argmax + R-MAX branch.** Still owns the predict-then-compare bookkeeping. |
| `arc_sdk.py` | The **organ** stays here (goal-fulfilling action → name/coords → GameAction), unchanged in spirit; it now receives a *fulfilling action*, still maps to the effector. |
| `thalamus.py` | Later: route goal-states between columns (multi-column); single-column no-op. |

---

## 5. Staging (each stage suite-green; debug on FAST OFFLINE reproductions, never longer live runs)

**Stage 1 — EFE value (kill R-MAX). ✅ DONE 2026-06-29 (A+B; C deferred).** In three validated micro-steps:
- **A — epiplexity-grounded epistemic term.** `reward.epistemic_value(s)` = `lp` (learning progress = the
  epiplexity-extraction rate) for visited, the `frontier` prior for unvisited; **dropped the `noise=err_slow−4·lp`
  gate** (a flat curve gives `lp→0` for noise AND mastered). Suite green (incl. the noisy-TV test).
- **B — dropped R-MAX.** The unbounded `Vmax=1/(1−γ)` (max *reward*) optimism → a **bounded** frontier optimism
  `β·frontier/(1−γ)` (the epistemic value of an unexplored region) in both the V-default and the untried-action
  branch; `Vmax` kept only as the `rmax=True` ablation. Suite green.
- **C — per-action effort cost: DEFERRED to Stage 2.** A flat per-step cost fragily fights FAR goals at γ=0.9
  (`effort·Σγᵗ` vs `γ^D`: no single value satisfies all scene distances) — it abandons far goals OR suppresses
  exploration. The mechanism is plumbed (`reward.effort`, default 0) but lands in Stage 2, where the SR-read makes
  it a tie-breaker *among goal-reaching paths* instead of a force that abandons them.
- *Gate met:* suite **61 green**; offline EFE-vs-R-MAX — on the **noisy-TV** grid (the realistic regime) EFE solves
  **more** (≈299 vs 202 completions) *and* chases the noise **4× less** (2 vs 8 TV-visits); on a trivial clean grid,
  comparable (140 vs 183, both ≫ random). The noise-robustness is the epiplexity prediction, confirmed.

**Stage 2 — the inverse-model motor (selection seated in the column). ✅ DONE 2026-06-29.** Action-selection
moved out of `agent._choose` into **`column.act`** — the MOTOR as an INVERSE MODEL: choose the action whose
predicted effect (L5's forward operator) is most VALUABLE, i.e. invert the operator against the EFE value to
achieve the highest-value next-state (the implicit goal-state). Behaviour-preserving relocation; the agent now
only plans value + hands the choice to the column. **This is the generalizable kernel** — a continuous effector
inverts the *same* operator against the *same* value; only the organ differs. *Gate met:* suite **61 green**.

**Re-scoping (what the experiments forced):** the *explicit multi-step goal-state*, the **L6/SR read**, and the
**effort cost** turned out to be **coupled to the GSG (Stage 4)**, not separable here:
- An **SR-read for NEED** was tried and **reverted** — `1+occ` over-prioritizes near-current states and starves
  the *backward* propagation of the distant goal's value (the code's own warning, confirmed: forward-SR-from-current
  is the wrong signal for goal-value backup). The clean SR-read is **navigating to an explicit goal-state**
  (argmax `occ(·, goal)` toward a *known* goal) — which needs Stage 4's explicit goals. **Deferred to Stage 4.**
- The **effort cost** needs the explicit goal-commitment to be a tie-breaker *among goal-reaching paths* rather
  than a flat cost that abandons far goals. **Deferred to Stage 4.**
So Stage 2's cleanly-separable win is the inverse-model motor; the explicit-goal machinery (goal emission +
SR-navigation + effort) consolidates into the GSG stage, where it is principled and necessary.

**Stage 3 — BG gate (selection by the consensus mechanism).** Route overt action selection through
`BasalGanglia.gate` over the goal-state proposals (Go/NoGo, dopamine-RPE); remove the agent argmax. *Gate:*
behaviour preserved/improved; suite green.

**Stage 4 — the full per-column GSG (epistemic richness + covert).** Enrich proposals: disambiguation goals
(sample where L2/3 hypotheses disagree), multi-step pragmatic goals, and **covert** evaluation (mental rollout
over the SR scores EFE without spending actions; covert pursuit of body-free epistemic goals). This is the
attention system (GSG + path-integration + BG-gates-overt-vs-covert; `TARGET_ARCHITECTURE` "Attention is not a
module"). *Gate:* directed exploration measurably beats Stage 1 on coverage/efficiency; suite green.

---

## 6. Risks / honest notes

- **Single-column GSG+BG is a "decoration"** (the old doc's words) — its full value is multi-column. We build the
  *mechanism* now because it's the generalizable one and it still improves exploration (EFE-directed > R-MAX) and
  motor generality single-column. We should not oversell the BG gate's single-column benefit.
- **"Better exploration completes a game" is a hypothesis**, not a certainty — Stage 1 tests it cheaply before we
  commit the deeper stages. The 0-completions could have other causes (e.g. the right ACTION6 click mechanics).
- **The inverse model is trivial for discrete ARC** (enumerate actions). The continuous-effector payoff is
  structural/generalization, realized at the *interface*, not exercised by ARC — which is the point for the prize.
- **Covert sampling burned us once** (Attention II) as a bolt-on; here it has a principled home (the GSG's EFE
  scoring / mental rollout). Keep it as imagination-for-scoring first; covert *overt-avoidance* only once scored
  goals are validated.
- The EFE epistemic term uses **learning progress = the online epiplexity-extraction rate** (not raw novelty/
  error), which is why it doesn't chase the noisy-TV (ls20's animation): a flat high-loss curve has zero
  epiplexity → zero epistemic value. Grounded in `reference_efe_and_epiplexity` (resolves §3b), and it *removes*
  the ad-hoc noise gate rather than adding to it.

---

## 7. The GSG stage — detailed re-plan (2026-06-29)

After Stage 2, the GSG stage carries everything the experiments showed is coupled to explicit goal-states:
**explicit goal emission + commitment, SR-navigation (the L6 read), the effort cost, the BG gate, disambiguation
goals, and covert evaluation.** This section designs it before any code.

### 7.1 The key realization — where the GSG actually earns its keep

The current loop already navigates to the highest-EFE region implicitly: the EFE **value** `V` propagates reward
(pragmatic) + frontier optimism (epistemic) backward through the graph, and `column.act` (Stage 2) greedily
climbs it. So **for the degenerate goal `g = argmax V`, an explicit goal-state + navigation is just a reframe —
it reproduces value-greedy.** The GSG is only worth its complexity when it proposes a goal **`g ≠ argmax V`**:
- **A committed FAR goal** chosen over a nearer distraction (reach distant reward; enable the effort cost).
- **A DISAMBIGUATION goal** — sample where L2/3's `(object,pose)` hypotheses disagree (resolve recognition
  uncertainty). The value gradient does *not* produce this; it's about the recognition posterior, not reward.
So the re-plan **centers on `g ≠ argmax V`** and treats the degenerate reframe as near-zero-value churn to avoid.

### 7.2 Navigation — the known-vs-frontier resolution (the crux)

You **cannot SR-navigate to an unvisited goal** (its SR row is empty — the Stage-2 finding). So `navigate_to(g)`:
- **`g` known (in the SR):** the SR occupancy gradient — `argmax_a occ(predict(s,a), g)`, where
  `occ(s,t)=M[s][t]` is the expected discounted visits to `t` from `s`. Greedy-toward-`g` ≈ shortest path. **This
  is the clean L6 read** (and where `OnlineSR.occ` returns, removed in Stage 2 as premature).
- **`g` frontier (unvisited):** the **value gradient** (optimism propagation) — the only thing that can pull
  toward the unexplored, and exactly what chose `g`. So frontier goals fall back to `column.act`.
One function, two regimes; the SR is read precisely where it is valid.

### 7.3 Commitment — light hysteresis, not hard commit

Re-emitting `g = argmax` every step is myopic (dithers between attractors) and defeats the effort cost; a hard
commit risks getting stuck on a stale/unreachable goal. **Design: light hysteresis** — keep the current goal
until (a) it is reached, (b) a different candidate beats it by a margin, or (c) a surprise spike invalidates the
model. This is the minimal commitment that reduces dithering and lets the effort cost mean something, without the
stuck-goal failure.

### 7.4 The effort cost — a goal-RANKING tie-breaker (now safe)

Stage 1's per-*state* effort abandoned far goals (it accumulated along every path). With explicit goals it moves
to **goal RANKING**: `EFE(g) = (pragmatic+epistemic)(g) − effort·dist(s,g)`, where `dist` is read from the SR
(`occ`)/graph. So among goals of similar intrinsic value the **nearer** wins (efficiency / RHAE), but a uniquely
rewarding far goal still wins (its pragmatic value ≫ `effort·dist`). The navigation to a chosen `g` is already
shortest-path (SR-greedy), so effort does **not** re-enter per step — no far-goal abandonment.

### 7.5 The BG gate — arbitration among goal candidates

The GSG proposes a few candidates of **different types** — the value-max (pragmatic/frontier), the committed
goal, the disambiguation goal — and `BasalGanglia.gate` (Go/NoGo + dopamine-RPE) selects one, learning which
types pay off. Single-column it is a thin arbiter (it earns its keep across types and, later, across columns).
This is the user's "selection by consensus" mechanism, seated.

### 7.6 Disambiguation goals + covert (the completion lever — and the novel/risky piece)

The recognition-epistemic goal: pick the locus where sampling most reduces L2/3 hypothesis **entropy** (where the
`(object,pose)` hypotheses most disagree). Its EFE = expected entropy reduction (a true mutual-information term,
noise-robust). **Covert**: score it by a *mental* rollout (path-integrate the belief to the locus, predict the
feature, estimate the entropy drop) — compute, not actions. This needs L2/3 to expose its hypothesis set + a
disagreement measure; it is the most novel piece and the most plausible lever for *completing a game* (directed
"figure out what's here" rather than undirected coverage).

### 7.7 Sub-staging (each suite-green; offline reproductions)

- **G1 — explicit goal + `navigate_to` + light commitment.** `column.propose_goal(value, candidates)` (degenerate:
  argmax EFE, but *may* commit to a far goal) + `column.navigate_to(state, g)` (SR-occ for known, value for
  frontier; restores `OnlineSR.occ`). *Gate:* solves nav/sparse/barrier ≥ Stage 2; suite green. *Risk:* commitment
  destabilizes — hysteresis must be conservative.
- **G2 — effort as a goal-distance tie-breaker.** Re-enable `reward.effort` in goal ranking (§7.4). *Gate:* fewer
  actions on a scene with a near + a far goal, **no** far-goal abandonment; suite green.
- **G3 — the BG gate over candidates.** Route goal selection through `BasalGanglia.gate`. *Gate:* behaviour
  preserved/improved; suite green.
- **G4 — disambiguation goals + covert.** The recognition-epistemic goal + mental rollout. *Gate:* on a scene
  where the object is ambiguous from one glance, the agent samples to disambiguate in fewer actions than
  undirected coverage. **May split into its own stage.**

### 7.8 Decisions to confirm before G1
1. **Commitment:** light hysteresis (§7.3) — agreed, or prefer pure re-emit (no commitment, lower risk but no
   effort/anti-dither benefit)?
2. **Disambiguation goals (G4):** in-scope for this stage (highest completion-lever, highest risk), or split to its
   own stage after G1–G3 land?
3. **GSG home:** the GSG is a **column** faculty (per-column, §1) reading the agent's value — agreed it lives in
   `column.py` (proposer + `navigate_to`), with the agent the thin coordinator?

### 7.9 Research resolution (2026-06-29) — `reference_gsg_goal_generation`

Grounded in TBT (Monty's GSG) + neuroscience (L5 "predictions not commands"; hierarchical priors; Cisek's
affordance competition). The three decisions resolve, and one thing is reframed:

- **The GSG's CORE function is uncertainty-resolution, with a CONCRETE algorithm — graph-mismatch.** Monty's GSG
  generates a goal by overlaying the top-2 object hypotheses and sampling the point of MAXIMUM DISAGREEMENT (the
  most-distant nearest-neighbour). So disambiguation is **not** an optional G4 — it is the GSG proper, and the
  graph-mismatch algorithm **de-risks** it (a concrete computation over L2/3's hypotheses, not vague "info gain").
  *Reframe of §7.1/§7.7:* the value-goal is OUR addition (Monty has no reward); the epistemic hypothesis-test is
  the GSG's defining job.
- **For ARC the GSG resolves THREE uncertainties:** object-IDENTITY (graph-mismatch — new), TRANSITION-dynamics
  (the `lp`/epiplexity term — have it, Stage 1), and reach REWARD (value — have it). The new piece is
  identity-disambiguation goals; the others already exist and just need to compete as goal candidates.
- **Decision 1 (commitment) → trigger-based.** Monty fires the GSG on TRIGGERS (ID-confident → resolve pose; top
  hypotheses reorder; leading pose shifts; staleness), pursues, then re-triggers. This is the principled form of
  "light hysteresis" — emit on a trigger, pursue, re-trigger.
- **Decision 3 (home) → confirmed:** per-column GSG in `column.py`; **L5 emits the goal** (the desired outcome,
  "predictions not commands"); the motor organ fulfils it; the agent stays the thin coordinator.
- **The BG gate is the affordance competition** (Cisek): candidate goals (identity-test / transition-explore /
  reward) held in parallel, biased by value + a basal-ganglia urgency signal. So the gate is integral, not a late
  add-on — though single-column it is a thin arbiter until the candidate TYPES compete.

**Open strategic choice for you (the one thing the research surfaces but can't decide):** the GSG's *principled*
core is identity-disambiguation (graph-mismatch), but our *immediate* completion blocker may be value/navigation /
RULE-uncertainty, and graph-mismatch needs a recognition-ambiguity scene to exercise. So: build the GSG
**disambiguation-first** (TBT-faithful, needs a new ambiguity scene + L2/3 hypothesis exposure), or
**value/navigation-first** (G1–G3: explicit goal + navigate + effort + BG over the value/transition goals we
already have, then add the identity-test as a competing candidate)? The staging in §7.7 is the latter.

### 7.10 DECISION + build order (2026-06-29) — disambiguation-first, message-shaped

- **Disambiguation-first** (TBT-faithful): the GSG is built around the **graph-mismatch hypothesis-test** (sample
  where the top-2 L2/3 hypotheses most disagree), not value/navigation. The value (reward) + transition (`lp`)
  goals are existing candidates that *compete*; the new core is the identity/hypothesis-test goal.
- **GENERAL, not ARC-specific:** the GSG reads the column's OWN generic uncertainty (L2/3 hypothesis disagreement +
  `lp`), never game features/colours/scene heuristics. Graph-mismatch is domain-general (resolve disagreement
  between *any* two competing models — object, concept, rule, word). If we ever read ARC structure to pick a goal,
  that is the bug.
- **Message-shaped goal-states from the start:** `column.propose_goal()` returns a **GoalState object** (target +
  the uncertainty it resolves + source), shaped so it can be *self-generated OR received from another column* —
  even though, single-column, it only talks to its own motor. This keeps the heterarchy free later (the scale-up
  is *where the message comes from*, not a different mechanism: a column combines its intrinsic goal with incoming
  precision-weighted goal-messages; the BG arbitrates the competition; cycles resolve by confidence + BG urgency +
  CMP voting — `reference_gsg_goal_generation`).
- **⚠ CAVEAT for the heterarchy scale-up (cross this bridge later):** a goal-state is a pose in *some* reference
  frame. Passing goals between columns that share an OBJECT frame works (CMP); passing between columns with
  DIFFERENT learned SR navigational frames needs **cross-frame registration** — the SAME deferred problem as
  heterogeneous-frame voting, the same place a hippocampus-like shared frame earns its keep
  (`reference_tbt_frames_and_hippocampus`). Heterarchical goal-messaging inherits exactly this one hard problem and
  no new ones. Heterarchy goal-resolution (loopy message passing + precision) is researched AFTER the motor refactor.
- **Build order (revised):** **GD1 ✅ DONE 2026-06-29** — `L23.disambiguation_goal` (Monty graph-mismatch: the
  point in the top hypothesis's predicted cloud most distant from the runner-up's; a `margin` gate suppresses it
  when one hypothesis clearly leads), wrapped by `column.propose_goal` into a message-shaped `GoalState`. 4 tests
  (graph-mismatch picks the appendage tip; margin gate; fires in a real recognition session; column message).
  Suite 61→**65 green**. **GD2 ✅ DONE 2026-06-29** — `L23.sense_absent` (the ABSENT half: a sample at a
  predicted-empty location falsifies the predictor) + `column.examine` (the active-recognition loop: COVERTLY pick
  the graph-mismatch target, OVERTLY sample it, present→`sense` / absent→`sense_absent`, until one hypothesis
  leads). Suite 65→**67 green**. HONEST nuance the trace exposed: the GSG (graph-mismatch over the top-2) is the
  FINAL 2-way disambiguation — 1 informative sample resolves it — but NOT the initial narrowing (a symmetric object
  yields many tied pose-hypotheses; passive sensing narrows faster there). Firing the GSG only when narrowed
  (Monty's triggers) is the GD4 refinement; the mechanism is correct + in place. **GD3 ✅ DONE 2026-06-29** —
  `column.propose_goals` (the ACT goal always + the DISAMBIGUATION goal when ambiguous, each with an EFE value) +
  `BasalGanglia.gate` arbitrating them (Go the higher value; dopamine-RPE makes a consistently-valuable goal type
  win past a value dip = Cisek's urge). The BG is in the loop (critique #2: selection by value+urgency competition,
  not an agent-script argmax). 3 tests; suite 67→**69 green**. **GD4 (refinements) ✅ DONE 2026-06-29** —
  FIRE-WHEN-NARROWED (`disambiguation_goal` fires only at `2 ≤ len(hyps) ≤ narrowed`; `examine` does Monty's order:
  passive-narrow then hypothesis-test — resolves the swamping scene in 2 samples, was 4) + the EFFORT goal-distance
  tie-breaker in `propose_goals` (a far test is discounted; a uniquely-worth-it test still wins). 2 tests; suite
  69→**71 green**. *Honest:* the efficiency win is object-dependent — on self-similar objects the NARROWING
  dominates (the deferred symmetry work, `project_symmetry_opportunity`); the GSG's clear value is correct TIMING +
  the final 2-way disambiguation. **REMAINING — the live-loop integration** (its own design pass): `agent.step`
  using `propose_goals` + BG + acting on the chosen goal — recognition ACTIVE in the live loop (not just bolted on
  for barriers) + sensor-to-target navigation + commitment/hysteresis.
  (Navigation/effort from §7.2–7.5 fold in at GD2–GD4.)

---

## 8. Live-loop integration — DESIGN PASS (2026-06-29)

### 8.1 The current live flow + the gaps
`arc_sdk.TbtPolicy.choose_action`: read frame → **(feature_id, position)** state (Stage 1 EFE value + Stage 2
inverse-motor are ALREADY live via `agent.step`/`col.act`) → `_object_barriers` recognises non-controllable
objects (one-shot `recognize_object`, cached by shape sig) to learn barrier-ness → `agent.step` picks the action.
**Not wired:** the GSG (`propose_goals`/`examine`/the BG). Wiring it needs three things absent from the live path:
(a) recognition **active in the loop** with hypotheses, (b) **sensor-to-target navigation** to act on a goal,
(c) **commitment/hysteresis**.

### 8.2 The pivotal finding — full-frame ARC makes identity disambiguation COVERT and cheap
The GSG (graph-mismatch, GD1–GD4) does **active sensing to resolve which OBJECT** — its win is choosing *where to
look* when looking is **expensive / partial** (Monty's fingertip robot). **ARC gives the WHOLE 64×64 frame every
step.** So an object's identity is resolvable by simply sensing all its cells from the current frame — **covertly
(no action) and cheaply (the whole shape is available).** The GSG's "sense fewer, smarter points" efficiency is
**moot when sensing the whole object is free.** It would matter only under **occlusion** (a hidden cell) — and then
resolving needs an **overt** action to move the occluder anyway.

Consequence: in ARC the agent's real **overt** epistemic uncertainty is **NOT object identity** (covert, cheap) —
it is **DYNAMICS** ("what does this action / this object DO?": a wall vs a walk-through painting; a button's
effect) and the **GOAL** (what completes the level). Dynamics is resolved by **acting** (bump it, click it) — which
is exactly the **`lp`/transition epistemic already in Stage 1's EFE**, plus the `ObjectBehaviour` faculty. So the
overt directed-exploration lever in ARC is **Stage 1 (already live)**, not the disambiguation GSG.

### 8.3 What this means for the GSG's live role (honest)
- The disambiguation GSG is the **right, general** mechanism (it shines in partial-obs / expensive-sensing — the
  competition's "works beyond ARC" spirit), and it is built + tested. But its **overt ARC value is limited** by
  full-frame observation.
- Its realistic ARC role is **covert recognition**: resolve each object's identity over the frame (cheap), feeding
  **property-generalisation** (barrier-ness now; affordances later) and object permanence. The graph-mismatch can
  *order* covert sampling, but covert sampling is free, so the efficiency is not the point — correctness is.
- The **overt** action driver stays the **EFE value + inverse-motor** (Stage 1+2, live): `lp`/dynamics + reward.
  The BG's `act` vs `disambiguate` competition is real, but `disambiguate` is mostly **covert (parallel, free)** or
  collapses into the `lp`/dynamics goal when it needs an overt interaction — so it rarely *competes* for the action.

### 8.4 Realistic live-integration scope
Modest and correct, not "the GSG drives the agent":
1. Recognition **active + covert** in the loop (resolve identities over the frame; the cache becomes a live
   session), feeding barrier/affordance generalisation. *Largely the existing barrier path, made first-class.*
2. Keep **EFE + inverse-motor** as the overt driver (live). 3. The **BG** arbitrates overt goals (mostly `act`).
4. **Commitment/hysteresis** for value/navigation. 5. **Sensor-to-target navigation** only for the rare overt
   disambiguation (occlusion / interaction-to-reveal — which is the `lp` goal).

### 8.5 Strategic implication + recommendation
The live integration **completes the architecture** (the GSG in the loop) and **demonstrates the general design**
(the prize values it), but it is **unlikely to be the thing that moves the ARC score** — that lever is Stage 1's
`lp`/dynamics + the goal, which is **already live but UNTESTED since the whole refactor** (the layer refactor +
Stages 1–2 changed the live agent substantially, and we have not run it live since). Running the public games is
**free** (no API/Kaggle cost — confirmed). So the grounded order is:
- **FIRST: a small LIVE TEST of the refactored agent** (Stage 1+2, as-is) on a public game — to measure whether the
  EFE/epiplexity + inverse-motor moved anything, AND to *observe whether identity-disambiguation even arises live*
  (does the agent hit ambiguous recognition? is occlusion a thing?). This grounds the GSG integration in real need
  rather than full-frame speculation. (Respects `feedback_no_debug_by_extending_actions`: a single fixed-budget
  measurement, not a debugging loop.)
- **THEN: scope the GSG live integration** to what the test shows — covert recognition + BG (§8.4) at minimum;
  more only where the loop asks. If the live test shows the score is blocked by dynamics/goal, that (not the GSG)
  is where to invest next.

### 8.6 RE-AUDIT (2026-06-30, session 2) — the GSG is computed-but-INERT; the reframe supersedes 8.3–8.5
A code re-audit against the live call graph (see `COLUMN_AUDIT.md` → *Correction (session 2)*) found the GSG **wired
but inert**: `agent._choose` computes `self.goal = propose_goals(...)` gated by `bg.gate` every step, but neither
`_choose` return branch dispatches on `self.goal` — both call `col.act(...)` with the value function regardless — so
the selection changes **nothing**. Two compounding causes, and they confirm this plan's own §8.2:
1. **No hypotheses ever compete live.** `arc_sdk` passes `agent.step(pos, feature=feat)` — never `cloud=` — so no L2/3
   recognition SESSION runs; `L23.disambiguation_goal()` → `None` → `propose_goal()` → `None` → `propose_goals()`
   returns ONLY the ACT goal → `bg.gate` (len==1) trivially returns ACT. The disambiguation branch is unreachable.
2. **Even with hypotheses, overt identity-disambiguation is moot** (§8.2): full-frame ARC gives the whole 64×64 grid
   every step, so identity is resolvable COVERTLY and cheaply; the GSG's "sense fewer, smarter points" efficiency has
   no overt purchase. The real OVERT uncertainty is DYNAMICS/RULE + GOAL.

**Why the plan mis-aimed (the user's read, confirmed).** GD1–GD4 (2026-06-29) predate BOTH the column-correctness work
AND the §3 reframe (`GROUNDING_PLAN §3`, 2026-06-30). They implement the TBT-faithful CORE — Monty's graph-mismatch
hypothesis-test — but bound it to the ONE uncertainty (object identity) that the live ARC loop never overtly faces. The
mechanism is right; the target was wrong for this environment.

**The corrected GSG design (grounded in §8.2 + `GROUNDING_PLAN §3` + [[reference_gsg_goal_generation]]).** The GSG's
defining job is uncertainty-resolution by hypothesis-test, and graph-mismatch is **model-general** — it resolves
disagreement between ANY two competing models (object, concept, **rule**, word), not just object graphs. So:
- **Single-column live role = COVERT recognition only** (feed property-generalisation / permanence over the frame,
  §8.4). It is NOT an overt action competitor here, and should not be presented as one. ⇒ the inert `self.goal`
  computation should be removed from the live `_choose` (or made a real channel) when M2 (BG arbitration) is wired —
  see `GROUNDING_PLAN` P2; carrying dead selection in the loop is exactly the parallel-mechanism smell.
- **The OVERT hypothesis-test GSG = the RULE/GOAL hypothesis over a TASK-SPACE schema** = `GROUNDING_PLAN §3`, which
  lives in the TASK column (the heterarchy), because a mechanic is a task-relational schema, not a single-object
  identity (the §3 correction). The GD1–GD4 graph-mismatch machinery is **REUSED verbatim** there — only the models it
  overlays change (object graphs → task/rule graphs); no new mechanism (Mountcastle). ⇒ the GSG's overt rebuild is
  DEFERRED to §3/the heterarchy, NOT attempted in the single column.
- **Supersedes 8.3–8.5's "live test first" recommendation** for the immediate work: per the user's objective, the
  priority is retiring the parallel mechanisms so the correct TBT loop is the live path (GROUNDING P1–P4), not a live
  score measurement. The live test remains a valid later check, but it is not the next step.
- **The GSG's COMMITMENT/hysteresis (§7.3) is anatomically the STN "hold your horses"** — a decision-conflict threshold,
  not an ad-hoc rule. It is being built as part of the faithful basal ganglia (`BASAL_GANGLIA_PLAN.md` B5, memory
  `reference_basal_ganglia`), so the GSG inherits commitment from the BG rather than reimplementing it (one model).
