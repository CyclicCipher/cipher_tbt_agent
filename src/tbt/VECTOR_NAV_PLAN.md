# VECTOR_NAV_PLAN — grid-cell vector navigation as a POTENTIAL FIELD (L6 ⊗ L5 + SR)

*2026-07-01. The efficiency lever (`arc_offline` benchmark: the agent SOLVES but WANDERS ~10× the oracle). Grid-cell
vector navigation is the metric SHORTEST-PATH / shortcut grain the SR value can't give. Grounded in
[[reference_vector_navigation]] (Edvardsen 2020; Stachenfeld 2017). POSITION-based → the integrate/live mode (P1's L6
`pos` + L5 `move_delta`), NOT the config_state benchmark. Benchmark = mechanism correctness + FEWER actions vs the
value-sweep baseline. Reuses the L5⊗L6 machinery P1 built + the SR (`navigate_to`); if it can't, we built L5/L6 wrong.*

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
  bumped move records a self-loop → M routes around) — so `navigate_to` also takes `blocked`. *Test:* integrate-mode nav
  (NavGame + a replica `local=True`) — FEWER actions RE-reaching a known goal than the swept value; suite green.

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
