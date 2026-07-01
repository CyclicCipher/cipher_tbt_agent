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

## Honest caveats
- POSITION-based → measured in INTEGRATE mode (the live-ARC path it targets); the config_state benchmark has no positions.
- Border cells are BUMP-learned (L5 records a wall after hitting it); perceptual obstacle-sensing (see the wall) is later.
- The goal position must be KNOWN (visited) — vector nav accelerates RE-reaching, not first discovery (exploration finds
  it; vector nav exploits it efficiently). So it composes WITH the explore grain, doesn't replace it.

## Sources — [[reference_vector_navigation]]
Edvardsen 2020 (cluttered-env cascade); Stachenfeld 2017 (SR warps around barriers = geodesic); goal-vector fields
(Nature 2022); Bush 2015 (grid vector nav). Robotics: artificial potential fields (Khatib) + a global planner for local minima.
