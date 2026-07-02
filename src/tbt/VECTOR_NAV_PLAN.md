# VECTOR_NAV_PLAN — grid-cell vector navigation as a POTENTIAL FIELD (L6 ⊗ L5 + SR)

*2026-07-01. The efficiency lever (`arc_offline` benchmark: the agent SOLVES but WANDERS ~10× the oracle). Grid-cell
vector navigation is the metric SHORTEST-PATH / shortcut grain the SR value can't give. Grounded in
[[reference_vector_navigation]] (Edvardsen 2020; Stachenfeld 2017). POSITION-based → the integrate/live mode (P1's L6
`pos` + L5 `move_delta`), NOT the config_state benchmark. Benchmark = mechanism correctness + FEWER actions vs the
value-sweep baseline. Reuses the L5⊗L6 machinery P1 built + the SR (`navigate_to`); if it can't, we built L5/L6 wrong.*

## THREADING — the GSG is the CONTROL LOOP wrapping `L6_NONABELIAN.md`'s achiever (built AFTER S1)
One loop across three docs: **a hypothesis = a target-state; testing = achieve (a GEODESIC in the learned Cayley graph); the
representation decides read-off (free/abelian) vs search (quotient/non-abelian).** `L6_NONABELIAN.md` builds the SUBSTRATE
(the generators S1 + the graph's relations S2 + Sokoban planning S3); `MATH_PHASE.md` is the DIAGNOSTIC (read-off vs search,
per rung — the clean-room rehearsal). This doc is the CONTROL LOOP: propose target-states → BG-select → `col.achieve` →
score confirms → commit/switch. **BUILD ORDER:** the achiever (V4, DONE incl. the pose-aware non-abelian case) is Phase I;
the GSG UNIFICATION here (`propose_goals`→BG→`achieve`, the inert `self.goal` dies) is **Phase II — the next major move**,
validated on abelian+SE(2) envs (no relations needed yet). Sokoban (S3) is Phase IV — it needs BOTH this loop AND S2's
relations. SHARED RISK = the wrong-merge / similarity-kernel smuggle (Phase III). See `L6_NONABELIAN.md`, `MATH_PHASE.md`.

## The mechanism — a POTENTIAL FIELD, steered by L5's inverse operator (3-level cascade)
1. **ATTRACTION (vector nav).** L6 gives the goal VECTOR `v = pos(goal) − pos(here)` (grid-code difference = the metric
   displacement). The attractive gradient; handles novel SHORTCUTS (straight line in the metric).
2. **REPULSION (border cells).** A wall = a BLOCKED action (`col.predict(s,a)==s`). Repulsion = exclude blocked
   directions. **Action = steepest descent of (attraction − repulsion) = the UNBLOCKED action whose L5 `move_delta`
   best aligns with `v`.** That is the potential field, discrete.
3. **LOCAL-MINIMUM ESCAPE (= "stuck").** A concave/perpendicular obstacle where no unblocked action reduces `v` (the
   classic potential-field local minimum) → fall back to a SUBGOAL from the topological graph: `navigate_to` (the SR
   VALUE, which WARPS around barriers → the GEODESIC, obstacle-aware) picks the next waypoint (a visited state /
   `l6_sr.grid` bottleneck) whose vector is clear; vector-navigate to it; resume. `navigate_to` (already built) IS Level 3.

## Maps onto our machinery (all pieces exist)
| piece | seat |
|---|---|
| goal vector `pos(goal) − pos(here)` | **L6** — path-integrated `pos` (P1 `track_pos`) + the grid metric (`l6_sr.grid`) |
| align an action to `v` + motor; the BLOCKED (border) read | **L5** — `move_delta[a]` (P1) inverse; `predict(s,a)==s` |
| the geodesic DETOUR + bottleneck waypoints | **SR** — `navigate_to` / `col.value` (M1) + `l6_sr.grid` |

## Build plan (each MECHANISM-tested, suite-green; position/integrate mode)
- **V1 — attraction.** `column.vector_action(here, goal, actions)` = the action whose `L5.move_delta` best aligns with
  `goal − here` (max dot). *Test:* on a grid it steers toward the goal + takes a straight shortcut.
- **V2 — repulsion (border avoidance).** Exclude blocked actions (`predict==self`); pick the aligned UNBLOCKED one.
  *Test:* an obstacle in the direct line → curves around it (aligned-open action), still makes goal-ward progress.
- **V3 — stuck → SR-geodesic subgoal.** Detect the local minimum (no unblocked action reduces `|v|`); fall back to
  `navigate_to` toward a waypoint; resume vector nav on arrival. *Test:* a U-shaped (concave) obstacle → escapes via the
  detour, does not oscillate in the local minimum.
- **V4 — wire vector nav as the GENERAL goal-navigation ACHIEVER** (NOT exploit-only — the reframe, user 2026-07-01, §
  below). Vector nav navigates to a GOAL-STATE whatever its source: a KNOWN reward (exploit) OR a HYPOTHESIZED target
  (explore — Sokoban's "block on the marker"). `vector_action` already takes an arbitrary `goal`, so the wiring supplies
  (a) the goal from a GOAL SOURCE (for V4: a remembered completing position; later: the GSG hypothesis), (b) the `blocked`
  set (bumped walls = border cells), and runs in INTEGRATE mode. NB the V3 detour needs the SR to REFLECT walls (a
  bumped move records a self-loop → M routes around) — so `navigate_to` also takes `blocked`.
  **EXPLOIT SLICE DONE (2026-07-01):** the agent remembers the completing POSITION (`_goal_pos` on `complete`), the
  cost field is ASSIGNED from experience (`step`: a no-progress bump → wall, a negative score → hazard), and `_choose`'s
  EXPLOIT arm (g~0) navigates there via `col.achieve` (a beeline, no frontier optimism). `arc_sdk` sets `agent._integrate`
  so config-mode is untouched. RESULT on NavGame(8): 8/8 in 412 total actions, steady-state 12/level = ORACLE (vs the
  wandering ~500/level); RHAE≈1 on the transfer levels. **REMAINING:** route the goal through `propose_goals`→BG (the GSG
  unification — the `reward`/`identity`/`dynamics` generators), so the EXPLORE-side hypothesis goal (Sokoban) also achieves.

## THE COST FIELD — ONE currency for walls / hazards / slow / risky (generalizes V2 repulsion)
*User's insight (2026-07-01): a wall and a hazard (and a slow tile, and a coin-flip-bad tile) should be the SAME
mechanism — not just in how the map AVOIDS them, but in how the model ASSIGNS the repulsion in the first place. They
are.* There is no special "obstacle" object. Every location has ONE learned scalar: the **expected COST of interacting
with it**. Walls, hazards, slow, and risky are just points on that one scale. This makes V2's binary `blocked` set the
`cost = ∞` LIMIT of a graded, learned field — grounds + extends [[reference_obstacle_as_transition_cost]] (obstacle =
reshaped reachability + SIGNED value) to graded and STOCHASTIC costs.

### The one currency — `cost(s)` = the running EXPECTED cost of interacting with `s`
| tile | what happens | cost it converges to | learned from |
|---|---|---|---|
| open | free | `0` | (default) |
| **wall** | you can't enter (no progress) | `∞` (IMPASSABLE) | a no-progress BUMP (`predict(s,a)==s`) |
| **hazard** | step on it → die / big penalty | the penalty (large) | AVERSION (a negative score, `reward.py` R_ext<0) |
| **slow** | costs 2 actions to cross | the extra step-cost | the traversal taking > 1 action |
| **risky** | 50% chance of something bad | `p · penalty` (the EXPECTATION) | the RUNNING MEAN of outcomes |
The stochastic case falls out for FREE: a running mean of "touch it → half the time −big" converges to `0.5·big`, so
"risky" is just a fractional cost — no new machinery, no special-casing. `cost(s)` is fed by the SAME `observe` the value
model already sees (aversion + the bump); one currency = expected discounted cost.

### Repulsion = the gradient of that one field (local steering + global geodesic)
- **LOCAL (the potential field, in `vector_action`):** action score = **attraction − repulsion** = `align(move_delta,
  UNIT goal-vector) − λ·cost(dest)`. Attraction is the unit-normalized goal alignment (∈[−1,1]); repulsion is `λ·cost` of
  the cell the action moves INTO. `cost ≥ IMPASSABLE` → hard-excluded (recovers V2's `blocked` as the ∞ limit); a big
  finite cost (hazard) is avoided unless nothing else makes net progress; a small cost (slow) is crossed only when the
  detour is longer. Graded, principled, one line.
- **GLOBAL (the SR geodesic, in `navigate_to`):** the `reward_map` carries `goal:+1` MINUS the finite costs, so `V =
  M·(reward − cost)` propagates the cost GLOBALLY → the geodesic routes around costly REGIONS (not just the next cell).
  Walls need no negative reward — the SR `M` never routed through them (no transition observed INTO a wall), so they fall
  out of the transition structure; only FINITE costs enter the reward map.
- ⇒ walls, hazards, slow, risky are ONE field, one gradient. `blocked` becomes a convenience alias for `cost=∞`.

### Build (this step, mechanism-tested, additive — BEFORE V4 wires it into the loop)
- `column.cost` (dict loc→expected cost) + `column.learn_cost(loc, c, rate)` = the running-mean ASSIGNMENT (the
  stochastic→expectation piece). `IMPASSABLE` constant.
- Thread `cost` through `vector_action` / `achieve` / `navigate_to` (unit-normalized attraction − λ·cost; the fallback
  builds a cost-subtracted reward_map). *Tests:* back-compat (V1/V2/V3 green); a HAZARD on the direct path → detours like
  a wall though not `blocked`; a SLOW tile → crossed when the detour is longer, avoided when the detour is short (graded);
  `learn_cost` with alternating good/bad outcomes → converges to the MEAN (risky = fractional cost); `cost=IMPASSABLE` →
  excluded like `blocked`.
- The ASSIGNMENT WIRING (agent calls `learn_cost` on aversion + bump) lands with V4 (the same loop that wires the achiever).

## THE REFRAME (2026-07-01) — vector nav is the ACHIEVER half of the GSG's hypothesis-test loop
(User's insight, from playing Sokoban/LockPath — it resolves how the GSG is supposed to work.)
- **Exploration is NOT only wander-to-discover** (the eigenpurpose/frontier = COVERAGE). It is ALSO **navigate to a
  HYPOTHESIZED goal to TEST it**: in Sokoban you hypothesise "block on the marker", COMMIT, navigate the block there, and
  the SCORE confirms — there is NO reward gradient (score only at the end) and NO coverage win, so ONLY a goal-hypothesis
  + an achiever solves it (why Sokoban is 0/3).
- **So vector nav (the ACHIEVER) serves BOTH exploit (reward goal) AND goal-directed exploration (hypothesis goal)** —
  the same primitive, "go to a goal-state," differing only in where the goal came from. Do NOT hard-wire it to exploit.
- **This un-sidelines the GSG.** I'd wrongly called the single-column GSG *redundant*; it is **INCOMPLETE** — it emits
  only an `"act"` goal with NO target. Its REAL job = propose **goal-HYPOTHESES with targets** (from Core-Knowledge
  priors: movable objects + salient markers + current uncertainty). The loop: **GSG proposes a target → STN COMMIT (hold
  it through the multi-step maneuver, B5) → vector-nav ACHIEVE (this plan) → SCORE confirms/refutes.** That is the
  goal-directed-exploration loop of `GROUNDING_PLAN §3` + [[reference_commit_to_test_a_hypothesis]] — the Sokoban / hard-
  multi-step-level lever. We have every piece (goal proposal, commitment, navigation, achiever, score), UNWIRED.
- **Sequencing:** V4 = the achiever (this plan) — the prerequisite. THEN the goal-HYPOTHESIS loop (the next chapter) =
  the GSG made live as a TARGET generator + commit + achieve + confirm — where the GSG stops being inert AND Sokoban
  unblocks. "What goal" (GSG) and "how to get there" (vector nav) are two halves of ONE loop, not competing steps.

## THE GSG — hypothesis GENERATION & testing (research 2026-07-01; the unified, SMALLER design)
*The user's Sokoban/multipath-blind question, unresolved until now: WHAT hypotheses does the brain generate, WHY those,
and by what MECHANISM? Our GSG notes covered TESTING (graph-mismatch, [[reference_gsg_goal_generation]]) and COMMITMENT
([[reference_commit_to_test_a_hypothesis]]) but never GENERATION — where a candidate like "push the block onto the
marker" even comes from. Answered below, and it lets the GSG SHRINK to one competition over target-states.*

### What the brain does (the three sub-problems)
1. **GENERATION — hypotheses are a SMALL SAMPLE from memory, cued by context (not enumeration).** The mind does not
   score the combinatorial space; it *stochastically samples a handful* of candidates from memory, the proposal biased by
   the PRIOR (what's usually true) and by RELEVANCE to the current scene (Dasgupta, Schulz & Gershman 2017, *Where do
   hypotheses come from?*). Resource-rational: generate the FEWEST samples that suffice, and generate MORE when UNCERTAIN
   (Lieder & Griffiths 2020). ⇒ the candidate set is tiny and context-cued, so "what goal to test" is cheap.
2. **WHY those — the candidates are cued by PRIORS: salience × CONTROLLABILITY × ambiguity.** A hypothesis isn't drawn
   from nowhere. "Block → marker" = a CONTROLLABLE object (it moved when pushed — a learned affordance / EMPOWERMENT: the
   drive to explore what you can CONTROL, Klyubin; sparse-reward empowerment 2021) crossed with a SALIENT location (the
   marker: novel/distinct — core-knowledge salience). Selection among the few = AFFORDANCE COMPETITION (Cisek) biased by
   VALUE (PFC) and URGENCY (BG) — parallel candidates, value+urgency-weighted, exactly GSG-proposes + BG-gates.
3. **HOW (the circuit) — generate, hold, and SWITCH.** Early in learning PFC holds a HIGH-DIMENSIONAL, flexible code of
   many candidate rules, collapsing to a low-dim rule-selective code once one is found (primate PFC geometry 2023/26) —
   the "try mappings, keep the one that works." Frontopolar cortex encodes the explore/exploit STAGE + the GOAL of the
   action. **The persist-vs-switch arbiter is the ACC** (Tervo & Karpova 2021, *ACC directs exploration of alternative
   strategies*): an OPPONENT micro-circuit that COMMITS to the ongoing strategy and, when its reliability drops, drives a
   switch to sample an ALTERNATIVE. That is exactly our commitment (STN B5a) + the trigger to abandon a REFUTED hypothesis
   and draw the next sample. Directed vs random exploration split by uncertainty type (relative→rlPFC directed;
   total→dlPFC random) — our g-gate/eigenpurpose is the directed arm.

### The synthesis — a GSG hypothesis IS a candidate TARGET-STATE, and testing = achieve + observe
Unify all of it: **a hypothesis = "bring about target-state X and see what happens."** Generation = sample a few
candidate X from priors; testing = the ACHIEVER (vector nav, this plan) navigates to X; the OUTCOME (score = pragmatic,
prediction-error = epistemic) confirms/refutes and updates value; the ACC/STN holds X through the maneuver and switches on
repeated refutation. This is EFE end to end: pick the X maximizing predicted **pragmatic (might reward) + epistemic
(resolves the most uncertainty)**, generate more X when uncertain.

### ⇒ our GSG gets SMALLER: three candidate-GENERATORS feeding ONE competition + the achiever
Today's GSG is a *bespoke* `disambiguation_goal` (Monty graph-mismatch, object-IDENTITY only) plus an inert `self.goal`
and a special `act`-vs-`disambiguate` list — object identity is a case games barely exercise, so it's dead weight in the
loop. The unification makes every goal a **navigable TARGET** valued by EFE; the graph-mismatch becomes ONE generator, not
a faculty:
| candidate generator (samples a few targets) | resolves | value = EFE |
|---|---|---|
| **identity** — graph-mismatch point ([[reference_gsg_goal_generation]], keep `disambiguation_goal`) | which object | epistemic (ID) |
| **dynamics** — the state where the model is least certain (transition-`lp` / epiplexity, [[reference_efe_and_epiplexity]]) | the RULE | epistemic (lp) |
| **reward** — a SALIENT × CONTROLLABLE target (a movable object at a salient marker) — the Sokoban hypothesis | does X complete? | pragmatic + epistemic |
| **act** (degenerate) — the greedy-value NEXT-state | — | pragmatic (the current `col.act`) |
- Every candidate is a `GoalState(target=…)` the ACHIEVER can navigate to — so **the inert `self.goal` dies**: the winner's
  `target` is handed to `col.achieve` (V4). The `act` goal stops being a special `target=None` case — it's just the
  greedy next-state, i.e. `col.act` is the degenerate achiever. `examine`'s bespoke passive/active loop collapses into the
  same competition (identity is one generator). **Net: one `propose_goals` (a few cheap generators) + the existing BG
  competition + `col.achieve` — the per-hypothesis-type branches go away.**
- **Sequencing:** V4 (the achiever) first — it's the shared executor every generator needs. THEN wire `propose_goals` →
  BG-select → `achieve` (goal live, no longer inert), starting with the `act`+`reward` generators (unblocks Sokoban);
  fold `disambiguation`/`dynamics` in as the other two generators. Commitment (B5a/ACC) holds the target across the push;
  the score switches it. This is the [[reference_commit_to_test_a_hypothesis]] loop, now with its GENERATION half filled.

## Honest caveats
- The `reward` generator's priors (salience, controllability) must stay LEARNED, not hand-coded (the bitter lesson):
  controllability = the object moved under our action (L5 affordance); salience = novelty/prediction-error, not a colour rule.
- POSITION-based → measured in INTEGRATE mode (the live-ARC path it targets); the config_state benchmark has no positions.
- Border cells are BUMP-learned (L5 records a wall after hitting it); perceptual obstacle-sensing (see the wall) is later.
- The goal position must be KNOWN (visited) — vector nav accelerates RE-reaching, not first discovery (exploration finds
  it; vector nav exploits it efficiently). So it composes WITH the explore grain, doesn't replace it.

## Sources
Vector nav — [[reference_vector_navigation]]: Edvardsen 2020 (cluttered-env cascade); Stachenfeld 2017 (SR warps around
barriers = geodesic); goal-vector fields (Nature 2022); Bush 2015 (grid vector nav). Robotics: artificial potential fields
(Khatib) + a global planner for local minima.
GSG hypothesis generation — [[reference_hypothesis_generation]]: Dasgupta, Schulz & Gershman 2017 *Where do hypotheses
come from?* (Cognitive Psychology); Lieder & Griffiths 2020 *Resource-rational analysis* (Behav Brain Sci); Tervo,
Kuchibhotla & Karpova 2021 *The anterior cingulate cortex directs exploration of alternative strategies* (Neuron); Cisek
2007 (affordance competition); empowerment/controllability (Klyubin 2005; sparse-reward empowerment arXiv:2107.07031);
primate PFC learning geometry (Nat Neurosci 2026); directed-vs-random exploration by uncertainty (Tomov/Gershman 2020).
