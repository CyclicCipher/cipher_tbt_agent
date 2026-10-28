# The rebuilt symbolic agent: experiments

Continues `COORDINATES.md`. Part 13 there rebuilt the symbolic agent's columns as frames with evidence-weighted drivers
(`arc_agent/claude/push_block_rebuild.py`). The design is in `arc_agent/claude/DESIGN.md`, and the values in
`arc_agent/claude/HOMEOSTASIS.md`.

# Part 14: a satisficing active-inference agent (pre-registered 2026-10-27, before the code was written)

**Why (the user):**
- Directed exploration and goal discovery, through a satisficing active-inference agent whose goal is to keep a set
  of prior states inside acceptable ranges (homeostasis).
- No scalar utility: a first step toward context-dependent values, and toward the hypothesis that satisficers are
  safer than optimizers.

**The agent** (`arc_agent/claude/homeostatic_agent.py`) puts `HOMEOSTASIS.md`, version 1, on part 13's frame model:
- **Variables:**
  - progress: acceptable when a reachable state has predicted P(WIN) ≥ 0.5;
  - understanding: acceptable when no reachable contact context is untried.
- **Goal hypotheses:** cell(f, c) and pos(f, x, y), with a description-length prior.
- **Choosing a plan:** the fewest-action acceptable plan. Otherwise the most epistemic one: goal information, plus
  contact novelty while understanding is out of range, minus 0.01 per action. Otherwise random.
- **Search:** breadth-first, up to 12 actions, over the learned model, restricted to the frames that interact with
  the agent's body (a frame driven only by time or by nothing is left out). A plan is followed until a prediction
  fails, a WIN happens, or the plan ends; then the agent replans.

**Worlds** (part 13): W1 push_block, W2 random layouts, W3 random layouts plus the patroller.
- **Levels:** 8 per run. A WIN starts the next level, with a new layout in W2 and W3.
- **Budget:** 50 steps per attempt. Running out (GAME_OVER) restarts the same level.
- **Runs:** 3 seeds; at most 3,000 steps per run.

**Agents:**
- **the homeostatic agent;**
- **an ablation without epistemic value:** it acts at random whenever no plan is acceptable;
- **the positional-goal baseline:** part 13's play. It explores at random until the first WIN, then plans to the
  frames' positions at that WIN in every later level.

**Measures:**
- steps to the first WIN;
- per level: won within its first 50 steps or not, and its steps against the true optimum (a breadth-first search in
  the true world);
- the most probable goal hypothesis after each WIN.

| # | prediction |
|---|---|
| H1 | the homeostatic agent's median steps to the first WIN is ≤ 100 in each world (part 13's random exploration on W1: 517–1,005) |
| H2 | on W2 and W3, from level 3 on, the homeostatic agent wins every level within its first 50 steps, and its mean steps exceed the true optimum by ≤ 1 |
| H3 | after the second WIN, the most probable goal hypothesis is cell(box, red) in every W2 and W3 run |
| H4 | the ablation without epistemic value takes ≥ 3 × the homeostatic agent's median steps to the first WIN |
| H5 | the positional-goal baseline wins ≤ 30% of levels 2–8 within their first 50 steps on W2 and W3, and 100% on W1 |

**Cost:** a build check measures the planner's speed first. If the run projects past 8 minutes, levels are cut to 5
(to be recorded here).

    python -m arc_agent.claude.homeostatic_agent      (from experiments/symbolics)

## Part 14 partial results (2026-10-27; `runs/homeostatic_agent.txt`; stopped at about 7.5 minutes)

**Stopped before W3, over budget.** The build check timed the planner (2–31 ms per plan), but not the runs that never
win. Those run to the 3,000-step cap with part 13's full model, and W2 took 6.5 minutes. The whole experiment
projected to about 15 minutes, against the 10-minute cap. W2 had also already shown the flaw below, so W3 wouldn't
have been informative. W3 was not run.

**W1** (push_block, 3 seeds, 8 levels each):

| agent | steps to the first WIN | levels 2–8 won in the first attempt | steps over the optimum, levels 3–8 |
|---|---|---|---|
| homeostatic | 5, 53, 53 | 100% | 0.00 |
| no epistemic value | 219, 297, 255 | 100% | 0.00 |
| positional goal (part 13) | 219, 297, 255 | 100% | 0.00 |

After every WIN, the most probable goal hypothesis was cell(box, red), at 0.91–0.98.

**W2** (random layouts), within the 3,000-step cap:

| agent | steps to the first WIN | levels completed |
|---|---|---|
| homeostatic | 127, 80, none | 2, 1, 0 |
| no epistemic value | none, none, none | 0, 0, 0 |
| positional goal | none, none, none | 0, 0, 0 |

- The homeostatic agent made 8,743 random decisions, against 246 exploratory and 11 satisficing ones.
- Its goal hypothesis after each WIN was again cell(box, red), at 0.94–1.00.

**Provisional verdicts** (W3 not run):
- **H1 holds on W1** (median 53) and fails on W2.
- **H2 fails on W2:** levels weren't completed.
- **H3 holds** wherever a second WIN happened (one W2 run).
- **H4 holds on W1:** 255 is more than 3 × 53.
- **H5:** 100% on W1 as predicted. On W2 the baseline never won at all, so "≤ 30%" is met vacuously.

**Where it breaks:** on W2 the agent drops to random actions almost all the time. That happens when no plan is
acceptable and nothing has epistemic value. My **likely cause, not yet verified:**
- **The box's driver is still tied** between static and contact. A second push is needed to break the tie (part 13),
  and until then the compact model leaves the box out, so no plan can move it.
- **Curiosity has already switched off.** The understanding variable, as pre-registered, counts only *untried*
  contact contexts. The box has been touched once, so it reads "satisfied" while the tie is unresolved.

If that's right, it is a flaw in the pre-registered definition, not in the code. Understanding should be out of range
whenever a reachable situation would separate competing hypotheses about the world: driver ties as well as untried
contexts. That is expected information gain about the model, the active-inference ambiguity term.

The other two agents show that random play alone almost never solves a W2 layout within 3,000 steps. So the
fallback can't rescue the agent.

**Next (not run):**
1. Confirm the cause by logging the box's driver and the planner's choice in a failing W2 run.
2. Redefine understanding as expected information gain over the model's competing hypotheses.
3. Re-register and rerun, with the never-winning runs capped so the total stays under 10 minutes.

**Correction (2026-10-27, after an offline reproduction): the cause of the W2 failure above was not the curiosity
term.** Logging the failing run (W2, seed 2) showed:
- **The box frame was discovered and classed as contact-driven,** ahead of static by 40–95 nats, and the planner
  included it.
- **The planner still never found "box on red" reachable,** with a horizon of 12 or of 30.
- **The cause is a coordinate bug in the goal model.** It enumerated pose translations from 0 to 8 only. Poses are
  relative to where a frame was discovered, and can be negative. Here the box, discovered at (7, 5), needs (−2, +3)
  to reach the red cell at (5, 8).
- **W1 happened to need (+2, 0),** which is why W1 worked.

The count-based curiosity is still unprincipled, as the user pointed out, but it didn't cause this failure. Part 15
fixes both.

# Part 15: version 2, exact expected information gain and planning over the posterior (pre-registered 2026-10-27, before the code was written)

**Why (the user):** "Take your time to rewrite the agent to have the principled version." The definitions are in
`arc_agent/claude/HOMEOSTASIS.md`, version 2.

**What changes from version 1 (part 14):**
- **No count-based novelty.** Epistemic value is exact expected information gain over explicit competing hypotheses:
  each frame's driver, each unmoved board object's response to contact, and the cause of WIN.
- **Understanding becomes "the largest reachable information gain is below 0.05 bits".**
- **Planning is over the posterior** (up to 12 model combinations), not the single most probable model.
- **The discovering transition counts as evidence** about a new frame's driver (`FrameAgent(score_discovery=True)`;
  the default stays off, and part 13 reproduces exactly).
- **The coordinate bug is fixed:** absolute positions, and every in-grid translation.

**Kept from part 14:** the worlds W1–W3, 8 levels, 50 steps per attempt (the same level restarts after GAME_OVER),
3 seeds and the measures. The cap is now 1,500 steps per run.

**Agents:**
- **version 2;**
- **version 2 without expected information gain:** after the acceptable-plan steps, it goes straight to the most
  likely plan, then to random.

Part 14's agents are not rerun; their W1 and W2 numbers are above.

| # | prediction |
|---|---|
| E1 | version 2's median steps to the first WIN is ≤ 80 in each world |
| E2 | from level 3 on, version 2 wins every level within its first attempt in W1–W3, and its mean steps exceed the true optimum by ≤ 1 |
| E3 | in every run where the box is discovered, its contact driver leads static by ≥ 7 nats straight after the discovering push |
| E4 | understanding satisfices: in levels 3–8, ≤ 10% of version 2's decisions are exploratory |
| E5 | version 2 without expected information gain takes ≥ 2 × version 2's median steps to the first WIN on W2 and W3, or doesn't win within the cap |
| E6 | measured information gain, the drop in the entropy of the agent's beliefs, is positive for ≥ 80% of exploratory plans |

**Cost:** a build check projects the total first. If it exceeds 8 minutes, levels are cut to 5, and then the
ablation's cap to 750 steps (to be recorded here).

    python -m arc_agent.claude.homeostatic_agent      (from experiments/symbolics)

**Amendments from build checks on seed 0, before the main run.** Each was found by logging a failing build-check run
and is a fix toward the stated design, not a tuning:
1. **Contact outcomes come from any frame that moves onto a board object, not only the body.** Before this, the red
   target's response to contact was never learned (the box, not the body, is what moves onto it), so red stayed
   one-third "solid".
2. **Beyond the 10-step horizon, candidate plans come from each of the top 3 model combinations** (breadth-first, up
   to 30 steps), and are then scored under the whole posterior. Before this, only the single most probable
   combination was searched, and in it red was sometimes solid.
3. **Goal hypotheses are about kinds of thing (shape and colours), with existential semantics, over every kind seen
   so far.** Instance keys made each level's red target a "new" thing with fresh, untestable hypotheses ("WIN iff the
   red cell lies on background"). These kept their prior weight and diluted "box on red" to about 0.25, below the
   aspiration.

   Also: a kind that is absent, such as the red target hidden under the box at the moment of WIN, must still be
   evaluated. Its hypotheses predict no WIN, so a WIN refutes them.

With these, seed 0 solved every level from level 3 on in exactly the optimal number of steps, in W1, W2 and W3.
**Projected cost:** about 6 minutes, nearly all of it the ablation, whose fallbacks reach the deep search on most
steps (about 0.2 s per step on W2). No cut was needed; the run is watched, and will be stopped at 9 minutes.

## Part 15 results (2026-10-27; `runs/homeostatic_agent_v2.txt`, `.json`; 91 s)

| world / agent | first WIN (3 seeds) | levels 2–8 won in the first attempt | steps over the optimum, levels 3–8 | exploratory decisions in levels 3–8 | box's lead after discovery | measured gain > 0 |
|---|---|---|---|---|---|---|
| W1 / version 2 | 3, 103, 3 | 100% | 0.00 | 0% | 7.8 (3 runs) | 11 of 11 |
| W1 / without EIG | 3, 53, 3 | 100% | 0.00 | — | 7.8 | — |
| W2 / version 2 | 57, 214, 165 | 100% | 0.00 | 0% | 7.8 (3 runs) | 30 of 30 |
| W2 / without EIG | 90, 216, 50 | 100% | 0.00 | — | 7.8 | — |
| W3 / version 2 | 107, **never**, 165 | 100% (2 runs) | 0.00 | 0% | 7.8 (2 runs) | 16 of 18 |
| W3 / without EIG | 107, 67, 123 | 100% | 0.00 | — | 7.8 | — |

**Verdicts:**
- **E1 refuted:** medians 3 (W1), 165 (W2), 165 (W3); the prediction was ≤ 80.
- **E2 confirmed for every run that won its first level.** One W3 run never did, so it never reached level 3.
- **E3 confirmed:** 7.8 nats in every run where the box was discovered.
- **E4 confirmed:** 0% exploratory decisions in levels 3–8.
- **E5 refuted:** the ablation without information gain reached the first WIN *faster* (medians 90 and 107, against
  165 and 165).
- **E6 confirmed:** 89–100%.

**Two problems, from diagnosing the W3 run that never won:**
1. **The agent isn't deterministic.** Rerunning that seed in other processes gave a WIN at step 165, or none within
   400 steps, depending on Python's hash seed. Sums over sets of hypotheses are added in hash order, and tiny
   floating-point differences flip near-ties. This pre-registered run is one sample of that variation.
2. **The goal can sit beyond the horizon.** Before the first WIN, "box on red" carries about 0.03 posterior, often
   more than 10 steps away. The agent instead tested nearby position hypotheses (about 0.004 each) one at a time.
   Within the 10-step horizon nothing reached the information threshold, and "approach" only looked within the
   horizon.

**On the curiosity term itself (E5):** before the first WIN, the main unknown is the goal. Pursuing the most probable
goal hypothesis is then itself the most informative test. Under the lexicographic rule, understanding comes before
approach, so the information-driven agent takes detours (touching the target, testing positions) that pure goal
pursuit skips. The term works as designed: realized gains were positive, and it switches off once the goal is known.
But here it costs steps rather than saving them.

# Part 15b: version 2.1 (pre-registered 2026-10-27, after part 15's results, before the code change)

**Changes:**
1. **Determinism:** sums over sets of hypotheses use `math.fsum`, which is exactly rounded and so independent of
   order.
2. **"Approach" looks beyond the horizon.** Candidates are the 5 most-probable-to-win plans within 10 steps, plus,
   under each of the top 3 model combinations, the shortest acceptable plan and the 5 most-probable-to-win plans
   within 30 steps. All are scored under the whole posterior. Approach takes the highest posterior P(WIN), the
   shorter on ties.

| # | prediction |
|---|---|
| G1 | determinism: W3 seed 1 gives identical results under hash seeds 0 and 1 |
| G2 | every version 2.1 run reaches its first WIN within the cap, and from level 3 on wins every level in its first attempt with ≤ 1 step over the optimum |
| G3 | version 2.1's median first WIN is ≤ 120 in W2 and in W3 |
| G4 | from part 15's diagnosis: the ablation without information gain is no slower than version 2.1 to the first WIN (median) in W2 and W3 |

**Cost:** about 2–3 minutes.

**Amendment from the G1 check, before the main v2.1 run.** W3 seed 1 still never won, identically under hash seeds 0
and 1 (so determinism worked). The trace showed a search bug:
- **The bug:** a step with an action the body has never tried should be a leaf worth log₂ 25 bits. Its state is
  unchanged, though, because its outcome is unknown, so it was discarded as a duplicate of its parent.
- **The effect:** untried actions were never informative. That player had only tried up and down, and never learned
  it could reach the box to its right.
- **The fix:** an untried action's leaf is never a duplicate.
- **Part 15's main run had the same bug.** Random fallbacks covered for it in most runs.

After the fix, the seed gives a first WIN at step 64, then every level at exactly the optimum, identical under both
hash seeds.

## Part 15b results (2026-10-27; `runs/homeostatic_agent_v2_1.txt`, `runs/homeostatic_agent_v2.json`; 62 s)

| world / agent | first WIN (3 seeds) | levels completed | levels 2–8 won in the first attempt | steps over the optimum, levels 3–8 | exploratory decisions in levels 3–8 | measured gain > 0 |
|---|---|---|---|---|---|---|
| W1 / version 2.1 | 3, 53, 3 | 8, 8, 8 | 100% | 0.00 | 0% | 12 of 12 |
| W1 / without EIG | 3, 93, 3 | 8, 8, 8 | 100% | 0.00 | — | — |
| W2 / version 2.1 | 57, 66, 69 | 8, 8, 8 | 100% | 0.00 | 0% | 29 of 29 |
| W2 / without EIG | 57, 96, 46 | 8, 8, 8 | 100% | 0.00 | — | — |
| W3 / version 2.1 | 57, 64, 65 | 8, 8, 8 | 100% | 0.00 | 0% | 23 of 27 |
| W3 / without EIG | 57, 64, 65 | 8, 8, 8 | 100% | 0.00 | — | — |

The box's lead straight after its discovery: 7.8 nats in all 18 runs.

**Verdicts: G1–G4 confirmed.**
- **G1:** W3 seed 1 is identical under hash seeds 0 and 1.
- **G2:** every run won its first level. Every later level was won in its first attempt, and from level 3 on at
  exactly the optimum.
- **G3:** medians 66 (W2) and 64 (W3).
- **G4:** the ablation's medians were 57 (W2) and 64 (W3), no slower than version 2.1.

**Reading:**
- **The agent now does what part 14 couldn't.**
  - It discovers the goal ("box on red", as a relation between kinds) in 53–69 steps on random layouts, then solves
    every later level optimally.
  - Its curiosity is exact expected information gain. Measured gains were positive for 85–100% of its exploratory
    plans.
  - Curiosity switches off once nothing reachable is informative: no exploration in levels 3–8.
  - Part 13's random exploration needed 517–1,005 steps for the first WIN on W1.
- **Information gain doesn't speed up the first WIN on average; it trims the worst case.** Seed 1: 53 against 93
  steps on W1, and 66 against 96 on W2. In these worlds the main unknown before the first WIN is the goal, and pursuing
  the most probable goal hypothesis is already an informative test. So goal pursuit with Bayesian refutation explores
  nearly enough on its own. The information term adds tests that goal pursuit would skip (untried actions, touching an
  unknown object) and avoids the slow cases.
- **What changed the outcome was mostly correctness, not curiosity.** Five bugs, each found by logging a failing run
  rather than by tuning:
  1. pose translations limited to 0–8;
  2. contact outcomes counted only for the body;
  3. goal hypotheses keyed to instances;
  4. hidden kinds never refuted;
  5. untried actions discarded as duplicates.

  Plus hash-order non-determinism. The part 14 diagnosis that blamed the curiosity term was wrong.

**Still open:**
1. **Action effects are point estimates.** A first try of an action at the board's edge records "no move", and nothing
   then marks the action as uncertain. The principled form keeps hypotheses over each action's displacement, with the
   bounded-board likelihood: a blocked move at the edge is consistent with every outward displacement.
2. **The arbitration between variables is lexicographic.** Active inference adds risk and ambiguity into one
   expected free energy, and a homeostatic agent weighs drives by how far out of range each is. Which arbitration is
   right is a question about the agent's values.

# Part 16: an efference copy, and biologically weighted drives (pre-registered 2026-10-27, before the code was written)

**Why (the user):** pre-register and run both.
- **Biological drive weighting:** urgency grows faster than linearly with deviation (Keramati & Gutkin 2014). Needs
  act as gains on one value scale, and competing drives are settled by selection (basal ganglia; hunger suppressing
  other drives, Burnett et al. 2016).
- **An efference copy:** what an action was meant to do, kept apart from what happened. A mismatch is then
  information about the world, not about the action.

## The efference copy (`arc_agent/claude/efference.py`)

**What it holds:** hypotheses, for each (body kind, action), over the intended displacement: the 25 translations
within 2 cells, including none, with a uniform prior. After each step:
- **The body moved by d:** evidence for exactly d (1 − ε; ε/25 for the others).
- **It didn't move:** evidence against every displacement that would have landed on a free in-grid cell (likelihood
  ε). Off-board or occupied destinations, and "no move", stay at 1 − ε. Such a block says nothing about the action;
  it is attributed to the world.

**How the planner uses it:**
- If an action's most probable displacement has probability ≥ 0.98, the body is planned to move by it.
- Otherwise the action is uncertain. Trying it is a search leaf worth the entropy of its displacement posterior.
- A move into a board object that doesn't happen marks that object solid (the existing contact update, now driven by
  the intended displacement).

**Without it** (part 15b): the most frequent observed effect per action, blocked moves included.

## Deviation-weighted drives

Candidates: every node of the 10-step search, plus the plans beyond the horizon (as in part 15b).

**Hunger:** h = (steps since the last WIN or the run's start, plus the plan's length) / 25. It keeps growing past 1
while no WIN comes.

**For each candidate, the two drives after the plan:**
- **progress:** δ_p = 0 if the plan's P(WIN) ≥ 0.5 (the dead zone); otherwise δ_p = (1 − P(WIN)) · h.
- **understanding:** δ_u = 0 if U₀ < θ; otherwise δ_u = max(0, U₀ − EIG(plan) − θ) / max(U₀, 1). Here U₀ is the
  largest information gain reachable now, and θ = 0.05 bits.

**Drive:** D = (δ_p³ + δ_u³)^(1/2), Keramati & Gutkin's form with n = 3, m = 2.

**Selection:** the plan with the smallest D wins, the shorter on ties. This stands in for selection by salience.

**What it should do:**
- **Early in a level,** with hunger low, curiosity dominates.
- **As steps pass without a WIN,** progress dominates, even while uncertainty remains.
- **Once both are in range,** D = 0, and the shortest acceptable plan wins: still satisficing.

**Without it** (part 15b): the lexicographic order.

## Model combinations when there are many objects

If the full product of hypotheses is larger than 12, the combinations are taken one at a time: the most probable
combination, plus, for each uncertain thing, one combination per alternative hypothesis of that thing (the others at
their most probable). Each is weighted by its product of marginals, then renormalized.

## A new world, W4: walls and distractors (`arc_agent/claude/wall_world.py`)

- **Mechanics:** push_block's, with random layouts.
- **A wall:** a straight segment of 3–4 cells (colour 5) that blocks the player and the box.
- **Two distractors:** single cells, passable for both, each level in fresh colours drawn from 6–13. So curiosity is
  re-triggered every level by things irrelevant to winning.
- **Every layout is checked solvable** by breadth-first search.

## Arms

| arm | drive weighting | action model |
|---|---|---|
| L | lexicographic | most frequent effect (version 2.1) |
| LE | lexicographic | efference copy |
| DW | deviation-weighted | most frequent effect |
| DE | deviation-weighted | efference copy |

- **Worlds:** W1–W3 for LE, DW and DE (L's results there are part 15b's, not rerun), and W4 for all four arms.
- **Runs:** 3 seeds, 8 levels, 50 steps per attempt, a cap of 1,500 steps per run.

## Predictions

| # | prediction |
|---|---|
| Q1 | with the efference copy, every run ends with every body action's most probable displacement equal to the true one, at probability ≥ 0.98. (Without it: reported, no prediction) |
| Q2 | in W4, with the efference copy, the wall's posterior is solid ≥ 0.9 by the end of the first level in every run where the body bumped it |
| Q3 | LE and DE win every level of W1–W4 within the cap. From level 3 on, every level in its first attempt, ≤ 1 step over the optimum in W1–W3 and ≤ 2 in W4 |
| Q4 | W4, levels 2–8: DW's and DE's mean steps per level are ≤ 0.8 × L's and LE's, and their contacts with distractors per level are ≤ 0.5 × theirs |
| Q5 | the urgency signature, DE in W4: the share of exploratory decisions taken ≥ 20 steps after the last WIN is ≤ half the share taken before that |
| Q6 | no regression: DE on W1–W3 wins every level, ≤ 1 step over the optimum from level 3 on |

**Exploratory decision:** a chosen plan with P(WIN) < 0.5 and information gain ≥ θ.

**Cost:** about 5 minutes. A build check projects it first. Beyond 8 minutes, the seeds stay but levels are cut to 5
(to be recorded here).

    python -m arc_agent.claude.homeostatic_agent --part16      (from experiments/symbolics)

**Amendments from build checks, before the main part 16 run:**
1. **Arm L is rerun on W1–W3, not reused.** The new rule for model combinations (one at a time when there are more
   than 12) changes L's own code path too, so part 15b's numbers no longer describe it.
2. **Understanding's deviation is relative to what is reachable:** δ_u = max(0, U₀ − EIG − θ) / U₀. The pre-registered
   denominator, max(U₀, 1 bit), shrank curiosity whenever less than a bit was reachable. In a build check (DE, W4) the
   agent then chased goal hypotheses at 0.4% probability for 100+ steps and never won within 120.
3. **Cost.** On W4 the drive-weighted arms replan almost every step, since the drive prefers short "approach" plans
   as hunger rises, at about 0.3 s per step. They took 225 steps to the first WIN in a build check, so the full run
   projected to about 13 minutes.
   - **Per the pre-registered rule,** W4 runs have 5 levels.
   - **Also, applied to every arm alike:** a cap of 300 steps per W4 run.
   - **Order and guard:** W1–W3 run first, then W4 in the order L, LE, DE, DW. The run is stopped at 9.5 minutes,
     and whatever didn't finish is reported as cut.

**A behaviour seen in the build checks, to be judged on the main run:** after each WIN, hunger resets, and the
drive-weighted agent explores first. The lexicographic agent takes an available acceptable plan before anything else.
Q4 (fewer distractor contacts for the drive-weighted arms) may therefore fail.

# Part 17: commitment, decision points and an inverse model (pre-registered 2026-10-27, while part 16 runs, before the code was written)

**Why (the user):** the drive-weighted agents replan almost every step, a problem met before (the rollout blowup).
The clues:
- **Biology:**
  - commitment with switching costs (McFarland's time-sharing; the basal ganglia and subthalamic nucleus holding a
    selection);
  - deliberation only at choice points (vicarious trial and error, hippocampal sweeps; Johnson & Redish 2007);
  - action chunking (Graybiel 1998);
  - map read-outs instead of search (successor representation, grid-cell vector navigation, GCML's inverse model).
- **AI:** options (Sutton, Precup & Singh 1999), incremental replanning (D* Lite), metareasoning (replan when it pays).

**The root of the dithering, found while designing this: part 16's drive formula was myopic.** Hunger was evaluated
at the end of each plan, (since + d)/25. A 1-step plan therefore always looks better: less time has passed when it
ends. So the drive picked one step and replanned.

## Changes (arms LE17 and DE17, both with the efference copy)

1. **Every plan is judged at the same moment.**
   - **Hunger:** h = (since + 30)/25 for all candidates, where 30 is the longest plan considered.
   - **Effort becomes its own drive:** δ_e = d/50, the share of the budget the plan uses.
   - **D = (δ_p³ + δ_u³ + δ_e³)^(1/2).** δ_p and δ_u are as in part 16, with its amendment.
   - The lexicographic arm is unaffected.
2. **Commitment: the agent keeps its plan and re-decides only at a decision point:**
   - the plan has ended;
   - a WIN, or a restart after GAME_OVER;
   - a state surprise: an object the plan moves (its body, or what it pushes) isn't where the most probable model
     predicted;
   - a value change: re-scored cheaply under the updated beliefs (no search), the remaining plan's P(WIN) has fallen by
     more than 0.1, or below 0.5 if it was acceptable.

   Part 15–16's rule was that any mismatch between the predicted and observed frame cancels the plan. That is
   replaced: a surprise elsewhere, such as the patroller, no longer matters.
3. **An inverse model replaces the 30-step joint searches beyond the horizon.** Under the most probable model, for
   each of the 8 most probable goal hypotheses:
   - **If it concerns the body:** navigate there by breadth-first search over the body's own positions (at most 81),
     using the efference copy's displacements, around solid things and objects it pushes.
   - **If it concerns an object the body pushes:** invert the contact relation. To move the object by d, the body must
     enter it from the cell at minus d. Search over the object's positions, with each push costing the navigation to
     the pushing cell plus one step.
   - **The plans found** are scored under the whole posterior, as before.
   - **Fallback:** the joint search, when there is more than one body or object to push, or a multi-cell one.

   The 10-step search over all model combinations stays, for information gain.

## Arms, worlds and comparison

- **Arms:** LE17 (lexicographic) and DE17 (drive-weighted), both with all three changes.
- **Worlds and settings:** W1–W3 (8 levels, cap 1,500) and W4 (5 levels, cap 300), as in part 16, 3 seeds.
- **Compared against:** part 16's LE and DE, reused, not rerun.

| # | prediction |
|---|---|
| R1 | cost: on W4, DE17 makes ≤ 0.3 decisions per step, and its time per step is ≤ ⅓ of part 16's DE |
| R2 | LE17 and DE17 win every level of W1–W4 within the caps. From level 3 on, every level in its first attempt, ≤ 1 step over the optimum in W1–W3 and ≤ 2 in W4 |
| R3 | dithering cured: DE17's median plan length on W4 is ≥ 3 steps |
| R4 | DE17's median first WIN on W4 is ≤ ½ of part 16's DE (or DE17 wins where part 16's DE didn't) |

**Cost:** expected well under part 16's. A 9.5-minute guard.

    python -m arc_agent.claude.homeostatic_agent --part17      (from experiments/symbolics)

## Part 16 results (2026-10-27; `runs/homeostatic_agent_part16.txt`; stopped by the 9.5-minute guard)

**Cut:** the guard stopped the run during W4's DE arm. DE and DW on W4 were not completed. Everything else was.

| world | arm | first WIN | levels done | levels 2+ won in the first attempt | steps over the optimum, levels 3+ | exploratory share: before 20 steps since a WIN / after |
|---|---|---|---|---|---|---|
| W1 | L | 3, 53, 3 | 8, 8, 8 | 100% | 0.00 | 0.23 / 1.00 |
| W1 | LE | 53, 53, 53 | 8, 8, 8 | 100% | 0.00 | 0.38 / 1.00 |
| W1 | DW | 3, 9, 3 | 8, 8, 8 | 100% | 0.56 | 0.29 / — |
| W1 | DE | 9, 9, 9 | 8, 8, 8 | 100% | 1.67 | 0.50 / — |
| W2 | L | 57, 66, 29 | 8, 8, 8 | 100% | 0.00 | 0.31 / 1.00 |
| W2 | LE | 57, 42, 65 | 8, 8, 8 | 100% | 0.00 | 0.43 / 0.91 |
| W2 | DW | 57, 24, 65 | 8, 8, 8 | 100% | 0.00 | 0.50 / 0.11 |
| W2 | DE | 57, 118, 135 | 8, 8, 8 | 100% | 0.00 | 0.53 / 0.07 |
| W3 | L | 57, 114, 65 | 8, 8, 8 | 100% | 0.00 | 0.50 / 0.59 |
| W3 | LE | 57, 34, 65 | 8, 8, 8 | 100% | 0.00 | 0.55 / 0.60 |
| W3 | DW | 107, 24, 97 | 8, 8, 8 | 100% | 0.00 | 0.54 / 0.01 |
| W3 | DE | 81, 114, 286 | 8, 8, 8 | 100% | 0.00 | 0.58 / 0.02 |
| W4 | L | 65, 63, 73 | 5, 5, 4 | 73% | 17.5 | 0.64 / 0.29 |
| W4 | LE | 65, 63, 73 | 5, 5, 4 | 73% | 17.5 | 0.66 / 0.27 |

On W4 the agents made 2.3–2.4 distractor contacts per level, and the wall was judged solid (1.0) by the end of level 1
in every run, having been bumped 1–3 times.

**Verdicts:**
- **Q1 confirmed:** with the efference copy, every run ended with every action's most probable displacement correct,
  at ≥ 0.98. Without it, in two W1 runs, actions 0–2 were never tried: the agent won in 3 steps, before trying them.
  No run without it ended with a wrong belief caused by a blocked move.
- **Q2 confirmed:** the wall was judged solid in every W4 run.
- **Q3 refuted:** on W4, LE left one run unfinished within 300 steps, won 73% of levels 2+ in the first attempt, and
  was 17.5 steps over the optimum.
- **Q4: not tested** (the W4 drive-weighted arms were cut).
- **Q5: not tested on W4.** On W2 and W3, the urgency signature is strong: DE's exploratory share falls from
  0.53–0.58 to 0.02–0.07 once 20 steps have passed without a WIN. L's doesn't fall: 0.31–0.50 before, 0.59–1.00
  after.
- **Q6 partly refuted:** DE is at the optimum on W2 and W3, but 1.67 steps over it on W1.

**Reading:**
- **The efference copy keeps action beliefs exact, but made little difference here.** The edge-blocked misbelief it
  guards against didn't occur in these runs.
- **The drive weighting behaves as biology predicts:**
  - **Under need, curiosity yields.** Exploration falls to near zero once steps pile up without a WIN.
  - **After each WIN hunger resets, and the agent explores first.** That is why DE takes extra steps on W1, and its
    first WIN comes later on W2 and W3.
  - **It is also expensive:** 60–150 s per arm, against 3–10 s for the lexicographic arms, because of the myopic
    replanning that part 17 targets.
- **W4 is hard even for the lexicographic agent.** In the lexicographic order, exploration comes before a satisficing
  plan that lies beyond 10 steps. Fresh distractors in every level therefore pull the agent off the goal whenever the
  goal is far.

**Part 17's comparison points (R1, R4) referred to part 16's DE on W4, which was cut.** The reference is therefore
the part 16 build check of DE on W4 (seed 0, cap 250): about 300 ms per step, first WIN at step 225.

**Amendment from part 17's build checks, before its main run: new hypotheses are charged with past WINs** (part 17
arms only).
- **The symptom:** some W4 levels still took 70+ steps against optima of 17–22. At the start of level 2 the goal
  posterior was flat: P(cell(box, red)) = 0.012, though it had 0 errors.
- **The cause:** every kind or colour registered after a WIN arrives with fresh, unrefuted hypotheses. W4 brings two
  new distractor colours every level, and their hypotheses swamped the true one.
- **The rule:** under the existential semantics, a kind or colour that wasn't present at a past WIN predicted no WIN
  there. So a hypothesis registered for the first time starts with one error per WIN observed so far. This extends
  part 15's registry fix, and changes nothing in W1–W3, which get no new kinds after the first WIN.
- **After the fix,** DE17 solved W4 levels 2–5 at exactly the optimum in seeds 0 and 1.

The same gap affected part 16's W4 arms; it is part of why L and LE were 17.5 steps over the optimum there.

## Part 17 results (2026-10-27; `runs/homeostatic_agent_part17.txt`, `.json`; 56 s)

| world | arm | first WIN | levels done | levels 2+ won in the first attempt | steps over the optimum, levels 3+ | decisions per step | ms per step | median plan length |
|---|---|---|---|---|---|---|---|---|
| W1 | LE17 | 37, 36, 37 | 8, 8, 8 | 100% | 0.00 | 0.24 | 23 | 3 |
| W1 | DE17 | 44, 43, 44 | 8, 8, 8 | 100% | 0.00 | 0.22 | 21 | 3 |
| W2 | LE17 | 25, 46, 39 | 8, 8, 8 | 100% | 0.00 | 0.11 | 14 | 10 |
| W2 | DE17 | 57, 40, 47 | 8, 8, 8 | 100% | 0.00 | 0.11 | 13 | 10 |
| W3 | LE17 | 57, 46, 65 | 8, 8, 8 | 100% | 0.00 | 0.13 | 19 | 9 |
| W3 | DE17 | 57, 40, 65 | 8, 8, 8 | 100% | 0.00 | 0.12 | 20 | 10 |
| W4 | LE17 | 65, 125, 73 | 5, 5, 5 | 92% | 2.67 | 0.14 | 22 | 9 |
| W4 | DE17 | 65, 125, 73 | 5, 5, 5 | 100% | **0.00** | 0.13 | 22 | 10 |

Commitment triggers over all runs: 6 surprises, no value drops. The plans were kept to their end otherwise.

**Verdicts:**
- **R1 confirmed:** on W4, DE17 makes 0.13 decisions per step at 22 ms per step, against about 300 ms per step in
  the part 16 build check: about 14 times cheaper.
- **R2 confirmed for DE17, refuted for LE17 on W4:** 2.67 steps over the optimum, and 92% of levels won in the first
  attempt.
- **R3 confirmed:** median plan length 10.
- **R4 confirmed:** median first WIN 73, against 225 in the build check.

**Reading:**
- **The replanning problem was mostly a myopic drive, not a lack of commitment.** Judging every plan at the same
  moment removed the bias toward 1-step plans; plans now run 9–10 steps, and decisions fall to about one every 8
  steps. Commitment then rarely had to intervene (6 surprises in 21 runs). The inverse model replaced the 30-step
  joint searches, which is where most of the time went.
- **With the myopia fixed, the biological weighting beats the lexicographic order where they differ.** On W4, DE17
  is at the optimum in every level from level 3 on. LE17 isn't, because the lexicographic order explores fresh
  distractors before taking a winning plan beyond 10 steps.

  The drive weighting also no longer wastes steps after a WIN (W1: 0.00 over the optimum, against part 16's 1.67).
  Hunger is judged at a common moment, so a winning plan's zero progress-drive outweighs early curiosity as soon as
  one is in reach.
- **The first WIN still costs 36–125 steps.** Before any WIN, the goal is unknown, and goal discovery dominates.
  Gemini's agent took 3, with the goal written in.

# Part 18: learning sets, a goal prior learned across games (pre-registered 2026-10-27, before the code was written)

**Why (the user):** run the learning-sets experiment.
- **Biology:** goal discovery is fast mainly because goal types are known in advance, innately or from earlier tasks.
  Harlow's (1949) monkeys went from many trials per new problem to one.
- **The Bayesian form:** an overhypothesis, a prior over goal types learned across games (Kemp, Perfors & Tenenbaum
  2007).
- **Question:** does a learned goal-type prior make the first WIN in a new game fast, while everything else (the body,
  what actions do, how objects respond, the colours) is relearned?

**Games** (`arc_agent/claude/learning_sets.py`): 12 per run, 3 runs (seeds), each game a new random set of colours
drawn from 1–13.
- **Every game's board:** the body, a box (pushable), a single target cell and two single distractor cells
  (passable). Random layouts, no walls.
- **Two goal families, chosen at random (½ each) per game:**
  - **push:** WIN when the box is on the target;
  - **reach:** WIN when the body is on the target. The box is still present, and irrelevant.
- **Levels:** 3 per game (a new layout each), 50 steps per attempt, a cap of 300 steps per game.

**The agent:** DE17 (part 17), new for every game: frames, efference copy, affordances and goal evidence all start
empty. Only the goal prior differs between arms.
- **FLAT:** the description-length prior, as before.
- **LS (learning sets):** a prior over goal *types*, learned from earlier games. A hypothesis's type is:
  - for cell(kind, c): (cell, the kind's role, the colour's feature);
  - for at(kind, x, y): (at, the kind's role).
- **Roles:** "body" (a frame whose driver is action), "pushed" (driver contact, or an unmoved object whose most
  probable response is pushable), or "other".
- **Colour features:** "unique" (exactly one board cell has it), "background", or "other".
- **P(type)** = (games so far whose goal had that type + the type's mass under the flat prior) / (games so far + 1).
- **Within a type,** the flat prior's proportions are kept.
- **After each game,** the most probable goal hypothesis's type is counted.

| # | prediction |
|---|---|
| L1 | a Harlow curve: LS's mean first-WIN steps in games 7–12 is ≤ 0.6 × its mean in games 1–3. FLAT shows no such fall (games 7–12 ≥ 0.85 × games 1–3) |
| L2 | in games 7–12, LS's mean first-WIN steps is ≤ 0.6 × FLAT's |
| L3 | once the true goal's kind has its role (the box found pushable, or the body found), LS's prior probability on the true hypothesis is ≥ 0.2 (mean, games 7–12); FLAT's is ≤ 0.05 |
| L4 | no regression: both arms win levels 2–3 of every game in their first attempt |
| L5 | the breakdown: before the first WIN, ≥ 30% of FLAT's steps run exploratory plans. In games 7–12, LS's exploratory share is ≤ half of FLAT's |

**Also logged:** the steps before each first WIN, by the kind of plan they ran (exploratory, approach, satisficing,
random).

**Cost:** about 2–3 minutes. A 9.5-minute guard.

    python -m arc_agent.claude.learning_sets      (from experiments/symbolics)

**Amendment from part 18's build check, before its main run: an uncertain action is worth the entropy of its
predicted *outcome* at the current state.**
- **The symptom:** in one game the body sat at the top edge and tried "up" 299 times. Each try was valued at 4.64
  bits, the entropy of the action's displacement hypotheses.
- **The cause:** at the edge, every upward displacement predicts the same outcome, "blocked", so trying it there
  teaches nothing. The part 16 design had specified outcome entropy; the code used displacement entropy.
- **What it is:** the noisy-TV failure that a learning-progress guard would also catch. The fix applies to the shared
  agent from now on; parts 16–17's results stand as recorded.

**Seen in the same build check (6 games per arm):** the learned prior raised P(true goal) to 0.13–0.19, against 0.02
under the flat prior. But first-WIN steps barely changed: before the first WIN, most steps ran exploratory plans
(learning actions, touching every object) that a goal prior doesn't shorten. L1, L2 and L5 may therefore fail.

## Part 18 results (2026-10-27; `runs/learning_sets.txt`, `.json`; 88 s)

3 seeds × 12 games; both arms play the identical games (same specs and layouts).

| | FLAT | LS |
|---|---|---|
| mean first-WIN steps, games 1–3 | 29.2 | 28.9 |
| mean first-WIN steps, games 7–12 | 81.1 | **62.4** (0.77 × FLAT) |
| games not won within 300 steps | 1 of 36 | 0 of 36 |
| prior P(true goal) once its kind's role is known, games 7–12 | 0.020 | **0.156** |
| median rank of the true goal in the posterior then, games 7–12 | 7.5 | **2.5** |
| goals correctly identified | 97% | 100% |
| levels 2–3 won in the first attempt | 100% | 100% |
| exploratory share of the steps before the first WIN (all / games 7–12) | 0.81 / 0.77 | 0.70 / 0.63 |
| steps before the first WIN, by plan kind | explore 1,734, approach 339, random 64 | explore 1,294, approach 314, satisfice 189, random 42 |

**Verdicts:**
- **L1 refuted for both arms:** first-WIN steps *rose* from games 1–3 to 7–12 (× 2.2 for LS, × 2.8 for FLAT). This is
  confounded by a design flaw: the random game sequences happened to put more cheap "reach" games early and more
  expensive "push" games late. The paired comparison (L2) is the fair one.
- **L2 refuted:** LS is 0.77 × FLAT in games 7–12, not ≤ 0.6. It also won the one game FLAT never did.
- **L3 partly refuted:** LS's prior on the true goal is 0.156 (predicted ≥ 0.2); FLAT's is 0.02, as predicted.
- **L4 confirmed.**
- **L5:** FLAT's exploratory share is 0.81 (≥ 0.3, confirmed), but LS's is 0.63, not ≤ half of FLAT's (refuted).

**Reading:**
- **The goal prior transfers, as Harlow and the overhypothesis account predict.**
  - P(true goal) rose 8-fold and its rank went from 7.5 to 2.5.
  - P(true goal) is capped near 1/3: each game has three single-cell colours (the target and two distractors), so the
    type alone can't tell them apart.
- **But the first WIN is not mostly spent on finding the goal.** 63–81% of the steps before it ran exploratory plans:
  - trying each action, since the body is new every game;
  - touching every object to learn how it responds, since the colours are new every game.

  Goal identification is a minority of the cost, so a better goal prior saved only 23%.
- **Biology's speed needs overhypotheses at every level, not only for goals.** People entering a new game already
  expect:
  - "the arrow keys move my avatar by one cell";
  - "single coloured cells are walked over";
  - "boxes get pushed".

  Our agent relearns all of these in every game. The same machinery (a prior over types, learned across games) applies
  to action effects and to affordances.
- **Curiosity should be weighted by relevance.** Touching both distractors in every game is information the goal
  rarely needs.

# Part 19: overhypotheses for actions and objects, and keeping the goal reachable (pre-registered 2026-10-27, before the code was written)

**Why (the user):** pre-register and run the next step from part 18: learned priors for what actions do and how objects
respond, plus relevance-weighted curiosity. After this part the work pivots to the real ARC-AGI-3 public games.

## A diagnosis of part 18, made before this design (scratch scripts on part 18's code, unchanged)

Part 18's LS arm was replayed with every belief's entropy logged per step. The replay reproduces part 18 exactly (1,839
steps before first WINs, the same as the recorded breakdown).

**Where the steps before the first WIN go:**
- **Learning the 4 actions and touching the 4 objects:** done within the first 10–25 steps of every game.
- **Touching a distractor after the box is known:** about 5 steps per game. This is all that relevance-weighted
  curiosity, as proposed, could save.
- **Steps after the box got stuck:** 580 of the 1,448 steps before the first WIN in push games (40%), in 20 events
  across 21 push games. Reach games: none.
  - **"Stuck":** the box is in a position from which no sequence of pushes brings it onto the target. In these games
    that means against an edge, since nothing can pull it back.
  - **What happens next:** no plan can win (best P(WIN) ≈ 0.01). The agent spends the rest of the 50-step attempt on
    plans worth 0.04–0.1 bits, until the attempt ends and the level restarts.
  - **How it gets stuck:** exploratory or approach plans push the box to an edge while testing goal hypotheses.

**So relevance-weighted curiosity is dropped from this part.** In its place is the mechanism the diagnosis points to:
a drive to keep the goal reachable.

## Mechanism 1: overhypotheses (`arc_agent/claude/overhypothesis.py`)

**What it is:** one mechanism for priors learned across games (Kemp, Perfors & Tenenbaum 2007), applied at two levels
here.
- **A slot:** a variable that recurs in every game, such as "what action 2 does".
- **A draw:** one game's confident value of the slot.
- **The prior for a new game:** a Dirichlet-multinomial over the slot's values, P(v) = (n_v + α·b_v) / (N + α).
  - n_v: past draws equal to v; N: all past draws; b: the flat prior over the slot's values.
  - **α (how much games vary)** is learned, not set: the prediction is averaged over α ∈ {0.01, 0.1, 1, 10, 100}, each
    weighted by how well it predicted the past draws in order (equal prior weights).
  - **Consequence:** if every game agrees, small α wins and the prior becomes nearly certain after about 4 games. If
    games disagree, the prior stays close to the observed frequencies.

**Slot 1, action effects:**
- One slot per action index; values: the efference copy's 25 displacements.
- **A draw:** at the end of each game, each action's most probable displacement for the body, if its probability is
  ≥ 0.98.
- **Use:** a new body's efference copy starts from this prior instead of the uniform one. If an action's prior
  probability is ≥ 0.98, the body is planned to move by it before it has ever moved. One observation confirms it;
  two contrary ones overturn it.

**Slot 2, object responses:**
- One slot per shape without colour (the cells' offsets); values: passable, solid, pushable.
- **A draw:** at the end of each game, each board object whose response is known with probability ≥ 0.98.
- **Use:** an object of a new colour starts from this prior instead of the flat one. In these games every object is a
  single cell, and one in four is the box. So the expected prior is about ¾ passable, ¼ pushable, solid near 0.

**The goal prior stays part 18's** (α fixed at 1), so that this part measures only the new levels.

## Mechanism 2: viability, keeping the goal reachable (in `homeostatic_agent.py`)

**What it is:** a fourth homeostatic deviation in the drive, D = (dp³ + du³ + de³ + dv³)^½.
- **Attainable goal mass, V(s):** under the most probable model, the summed posterior probability of the 8 most
  probable goal hypotheses that some sequence of actions could still make true from state s. A hypothesis already
  made true along the plan counts as attained (the plan would have won).
- **Reachability is relaxed (optimistic):**
  - the body: breadth-first search over its own translations using the believed actions, around solid things;
  - a pushed object: breadth-first search over its translations, where a push by d needs the cell behind it (at −d)
    and the cell ahead (at +d) on the board and not solid; whether the body can get behind it is ignored.

  Relaxed reachability only ever overestimates. So a plan is penalized only when it destroys reachability even in the
  relaxed sense.
- **The deviation:** dv = (V(now) − V(plan's end)) / V(now), or 0 if V(now) = 0 or the plan raises V.
- **Selection:** candidates are scored in increasing order of the old drive. Since dv only adds, the search stops as
  soon as the old drive exceeds the best full drive found, so the result is exact.

**Biology and alignment:**
- **Ashby's essential variables, and viability:** the organism's first job is to stay in states it can recover from.
- **Alignment:** this is attainable-utility preservation (Turner, Hadfield-Menell & Tadepalli 2020) and relative
  reachability (Krakovna et al. 2019), with the agent's own goal posterior as the set of utilities to preserve. A
  satisficer that also avoids irreversible loss of options is the safer design the HOMEOSTASIS note argues for.

## Arms, games and comparison

**The games:** part 18's exactly (3 seeds × 12 games, the same specs and layouts).
- **LS:** part 18's learned-prior arm. Its results are reused from `runs/learning_sets.json`, not re-run; the code it
  ran is unchanged.
- **OH:** LS + overhypotheses for action effects and object responses.
- **LSV:** LS + the viability drive.
- **OHV:** LS + both.

**Paired comparisons:** every arm plays the same 36 games. Means are over games; "games 7–12" is the last half of each
run. **A within-family Harlow curve:** first-WIN steps of push games in games 7–12 against push games in games 1–6,
which removes part 18's family-mix confound.

| # | prediction |
|---|---|
| O1 | in games 7–12, OH's prior makes all 4 actions certain (≥ 0.98) before the body first moves, in every game |
| O2 | in games 7–12, OH's prior P(pushable) for a new single-cell object is between 0.15 and 0.35 |
| O3 | in games 7–12, OH's mean first-WIN steps ≤ 0.9 × LS's (62.4 → ≤ 56.2) |
| V1 | over all 21 push games of each arm pair, steps spent with the box stuck before the first WIN fall by ≥ 70%: LSV ≤ 0.3 × LS's 580 |
| V2 | mean first-WIN steps in push games: LSV ≤ 0.7 × LS |
| V3 | no regression: in reach games, LSV's mean first-WIN steps ≤ 1.1 × LS's; every arm wins every game, and levels 2–3 in their first attempt |
| C1 | in games 7–12, OHV's mean first-WIN steps ≤ 0.6 × LS's (≤ 37.4), part 18's L2 target |
| C2 | a within-family Harlow curve for OHV: push games 7–12 ≤ 0.75 × push games 1–6 |

**Also logged:** steps before each first WIN by plan kind; stuck steps and stuck events before the first WIN (from the
true game, never shown to the agent); the learned α of each slot; milliseconds per step.

**Untested here:** a game whose actions differ from earlier games (revision of the action prior). All games share the
same action mapping, as ARC-AGI-3 games largely do.

**Cost:** about 3–4 minutes for the 3 new arms. A 9.5-minute guard.

    python -m arc_agent.claude.learning_sets --part19      (from experiments/symbolics)

**Amendments from part 19's build check, before its main run (seed 0, all arms):**
1. **A bug in the contact update since part 15, fixed in the shared agent.** When the body pushed an object, the rule
   "a frame that moved onto a board object passed over it" fired as well as the push rule. So every box ended with
   passable and pushable at 0.5 each, and the learned prior for one-cell objects drifted to P(pushable) = 0. Now an
   object that moved off as a new frame is recorded as pushable only.
   - **Consequence:** LS behaves differently, so it is **re-run** as the baseline (arm LS). Part 18's recorded LS stays
     as a reference row (LS18). Parts 15–18's results stand as recorded.
2. **The viability term changes form: lost reachability is counted as hunger, not as a fourth deviation.**
   - **Why:** in the build check, a plan that cut the attainable goal mass from 0.85 to 0.29 was chosen anyway. dv is
     at most 1, while hunger passes 2 within 30 steps, and the cube makes dv³ negligible beside dp³.
   - **The principled form:** a goal made unattainable cannot be won until the attempt ends. So losing attainable mass
     L (absolute, not relative) adds L × (steps left in the attempt after the plan) / 25 to the progress deviation dp,
     in hunger's own units.
   - The selection stays exact, since the added term is never negative.
   - Seed 0, game 10 after both fixes: steps with the box stuck before the first WIN fell from 116 to 12.
3. **O1 is fixed by the mechanism:** with every game agreeing, the action prior reaches 0.98 at 7 draws (0.9798 at 6).
   So game 7 can't pass O1. O1 is kept as written; games 8–12 are reported alongside.

**Predictions V1–V3 now compare LSV with the re-run LS; C1 compares OHV with the re-run LS.**

## Part 19 results (2026-10-27; `runs/learning_sets_part19.txt`, `.json`; 230 s)

3 seeds × 12 games, part 18's identical games. The re-run LS (with the contact fix) reproduced part 18's LS exactly,
game for game, including 580 stuck steps. So the bug never changed LS's behaviour, and the diagnosis is confirmed on
the main run's apparatus.

| | LS | OH | LSV | OHV |
|---|---|---|---|---|
| mean first-WIN steps, games 1–3 | 28.9 | 31.7 | 28.9 | 31.7 |
| mean first-WIN steps, games 7–12 | 62.4 | **41.1** (0.66 ×) | 52.7 (0.84 ×) | **39.9** (0.64 ×) |
| push games, 1–6 → 7–12 | 72.2 → 84.6 | 71.2 → **54.5** | 63.8 → 70.6 | 57.0 → **52.0** |
| reach games, 1–6 → 7–12 | 23.5 → 18.2 | **51.1** → 14.2 | 25.0 → 16.8 | **51.1** → 15.8 |
| steps with the box stuck before the first WIN (events) | 580 (20) | 372 (12) | 310 (13) | **203** (9) |
| games not won within 300 steps | 0 | **1** | 0 | **1** |
| levels 2–3 won in the first attempt | 100% | 100% | 100% | 100% |
| actions certain before the body moves, games 7–12 / 8–12 | – | 0.67 / 0.80 | – | 0.67 / 0.80 |
| prior P(pushable) for a new one-cell object, games 7–12 | – | 0.254 | – | 0.252 |
| learned α (each run) | – | actions 0.01, objects 1 | – | the same |
| ms per step | 26 | 32 | 28 | 32 |

**Verdicts:**
- **O1 refuted:** 0.67 of games 7–12 start with all actions certain (0.80 of games 8–12). Game 7 failed by construction
  (amendment 3). Also, a game in which an action was never confirmed contributes no draw for it, which delays the
  prior.
- **O2 confirmed:** P(pushable) = 0.254, against a true frequency of 0.25.
- **O3 confirmed:** OH is 0.66 × LS in games 7–12 (predicted ≤ 0.9).
- **V1 refuted:** stuck steps fell to 0.53 × LS (310 vs 580), not ≤ 0.3 ×.
- **V2 refuted:** LSV's push games are 0.85 × LS (68.3 vs 80.4), not ≤ 0.7 ×.
- **V3 confirmed for LSV:** reach games 1.03 × LS, every game won, levels 2–3 in the first attempt. But OH and OHV each
  lost one game (below), so "every arm wins every game" is refuted.
- **C1 refuted narrowly:** OHV is 0.64 × LS in games 7–12 (predicted ≤ 0.6).
- **C2 refuted:** OHV's push games 7–12 are 0.91 × its push games 1–6 (predicted ≤ 0.75). OH alone shows the curve
  (0.77 ×); LS shows none (its push games got slower).

**A defect found: the hard threshold on action beliefs.**
- **The symptom:** in early games, OH and OHV were much slower on reach games (51.1 vs 23.5 steps in games 1–6).
  One game (seed 2, game 5) was never won.
- **The cause, seen in that game:**
  - After 4 games, the prior put actions 1 and 2 at 0.963. That is below the 0.98 at which an action is planned with,
    so both counted as unknown.
  - An unknown action is valued only by the entropy of its outcome, about 0.2 bits at 0.963.
  - In 300 steps the agent never pressed either. It made 152 exploratory plans and never reached the target.
- **What it shows:** a continuous belief cut by a hard threshold has a bad middle band. In that band the action is too
  uncertain to plan with and too certain to be worth testing.
- **The fix, not made here:** plan over action uncertainty directly. Each action's alternative displacements become
  weighted model combinations, so P(WIN) includes the chance that the action does what the prior says. No threshold.

**Reading:**
- **Overhypotheses are the larger lever.** In games 7–12 they cut the first WIN by a third. Within push games they
  give the Harlow curve that part 18's goal prior alone did not. The learned α came out as Kemp et al.'s account
  predicts: near 0 for actions (games agree), 1 for object responses (games vary, so the prior is the frequency).
- **Viability helps but doesn't prevent most stuck boxes** (−47% stuck steps alone, −65% with OH). Three reasons
  remain:
  - the first push is what reveals the box is pushable, so reachability can't protect it before that;
  - relaxed reachability is optimistic;
  - once hunger is high, a small chance of winning still outweighs the expected loss.
- **For the real games (the user's note: ARC-AGI-3 public games mostly have no player character):** the body-centred
  parts of this agent will not carry over as they are (frames moved by the actions, pushing by contact). These three
  lessons will:
  - priors over action effects learned across games, with α learned;
  - irreversibility counted in the progress variable's own units (in ARC-AGI-3, RESET makes the cost "a reset and a
    replay", not "the rest of the attempt");
  - no hard thresholds on beliefs.

# Part 20: conditions learned by attention (pre-registered 2026-10-27, before the code was written)

**Why (the user):**
- The arc_agent must not hardcode when an action's effect is stopped. A rule like "things don't pass through solid
  things" can't be computed from a 2D screen, and a hardcoded version can never be revised.
- The solution should be simple: the attention we already have.
- Make intentionally hard cases where conditional logic is needed.

**The change (`arc_agent/column.py`, `arc_agent/brain.py`):** what an action does to a column is predicted by
attention over experts, weighted by each expert's track record (accumulated log predictive probability, as for drivers
in part 13).
- **The unconditional expert:** "this action applies operator X". This is what the efference copy was.
- **One expert per context feature.** Each is a table from the feature's value to the operator seen. Features are read
  in the column's own frame:
  - the content at each position relative to the column, within a window around it;
  - beneath the column, the remembered content (V1's Spotlight lattice);
  - off the screen, the value "edge";
  - the state of every other column.
- **Unseen values** back off to the unconditional expert's prediction.
- **Deleted:** passability by colour, the contact rule, part 16's "a blocked move is no evidence" rule, and the
  screen-edge rule. Nothing says what blocks; it is whatever feature predicts.

**The hard cases (`arc_agent/worlds.py`).** Each is a 64 × 64 screen built from 3-pixel cells. An avatar of 3 × 3
pixels moves one cell (3 pixels) per key, with random layouts per seed.

| world | the condition | what predicts it |
|---|---|---|
| W1 walls | colour-5 cells block | the content where the avatar would go |
| W2 inverted | colour 5 is passable; small colour-14 dots block | the same, with no colour keeping a meaning across worlds |
| W3 torus | leaving one edge re-enters at the opposite edge (the edge doesn't block) | the "edge" value beside the avatar |
| W4 ice | stepping onto ice (colour 11) slides the avatar 2 cells instead of 1 | the content one cell ahead |
| W5 one-way | colour-12 tiles can be entered only moving right | the content ahead, per action |
| W6 speed switch | while a switch (a corner indicator toggled by ACTION5) is on, every move is 2 cells | the switch column's state (gating by another column) |
| W7 switch door | colour-8 doors block unless the switch is on | a conjunction (door ahead AND switch off): **expected to defeat single-feature experts** |

**Protocol:**
- A random walk of 300 actions per world (keys 1–4, plus ACTION5 in W6–W7), 3 seeds.
- Measure: before each action, the brain predicts the avatar's next position. Accuracy is counted over actions
  101–300.
- **Arms:**
  - ATTN: the change;
  - UNCOND: the unconditional expert only (an ablation inside the same code).

| # | prediction |
|---|---|
| C1 | ATTN's accuracy ≥ 0.95 in W1–W6 |
| C2 | UNCOND's accuracy ≤ 0.85 in each of W1–W6 (it cannot represent the conditions) |
| C3 | the attended feature is the right one: after the walk, the most-attended expert for each key is a content feature on the side the key moves toward (W1, W2, W4, W5), an "edge" or content feature on that side (W3), or the switch column's state (W6) |
| C4 | W7: ATTN < 0.95 (the conjunction is out of reach of single features), but ATTN > UNCOND |

**Cost:** under 2 minutes.

    venv312/Scripts/python.exe -m arc_agent.test_conditions      (from experiments/symbolics)

## Part 20 results (2026-10-27; `runs/test_conditions.txt`, `.json`; 14 s)

Accuracy of the next-position prediction, actions 101–300, mean of 3 seeds:

| world | ATTN | UNCOND | ATTN's attended feature on the right side |
|---|---|---|---|
| W1 walls | **0.993** | 0.675 | 1.00, exactly the cell the key moves into, for every key |
| W2 inverted | **0.990** | 0.800 | 1.00, the dot's pixel in the cell ahead |
| W3 torus | **0.993** | 0.972 | 0.67 |
| W4 ice | **0.990** | 0.682 | 1.00 |
| W5 one-way | **0.993** | 0.643 | 1.00 |
| W6 speed switch | **0.962** | 0.647 | 1.00, the switch column's state, for every key |
| W7 switch door | 0.942 | 0.915 | 0.75 |

**Verdicts:**
- **C1 confirmed:** ATTN ≥ 0.95 in W1–W6 (lowest W6, 0.962).
- **C2 refuted for W3 only:** UNCOND is 0.972 there. A random walk rarely reaches the board's edge, so the condition
  is rarely tested. W1, W2, W4, W5 and W6 confirm it (0.64–0.80).
- **C3 confirmed except W3:** there, one feature (3 pixels left and 4 up) is attended for all four keys. It sees
  "edge" in the top row and the left column, so it predicts wraps for up and left, but not for down and right.
- **C4 confirmed:** the conjunction world defeats single-feature experts (0.942 < 0.95), and ATTN still beats UNCOND
  (0.915).

**Reading:**
- Walls, inverted colours, one-way tiles, ice and a remote switch are all learned by one mechanism with no rule
  saying what blocks. The attended feature is the cell the action moves into, or the switch.
- Deleted from the agent: passability by colour, the contact rule, part 16's "a blocked move is no evidence" rule, and
  the screen-edge rule.
- **Open:**
  - conjunctions (W7);
  - rare conditions that a random walk seldom tests (W3);
  - the real games: a smoke test on ls20, vc33 and tu93 runs (16 s) but solves nothing. Planning and the RESET logic
    are the next work, not the condition model.

# Part 21: conjunctions, and the planning and RESET loops (pre-registered 2026-10-27, before the code was written)

**Why (the user):** fix the planning and RESET loops, and do conjunctions.

**Diagnosis, from traces of ls20 and tu93 on part 20's code:**
- **The dead zone (seen before, in `src/tbt/` on 2026-06-30):** once the reachable neighbourhood is explored and no WIN
  has been seen, no plan in the 12-step search has value. The executive returns no plan, and the agent alternates
  RESET with a random action.
- **Losing the avatar (tu93):** the avatar's facing pixel changes when it turns. Its column needs an exact match of its
  content, so every turn loses it and a new column is recruited with no history. Every step then looks like an
  uncertain outcome worth exploring.

**Changes:**
1. **Conjunctions: a second layer of attention, created on refutation.** When the most-attended expert's table cell
   for a value v has seen two different operators (its prediction was refuted), the context gains conjunction
   features for that case: (that feature = v, every other feature g), each holding a table from g's value to the
   operator, only present when the first feature equals v. Their unseen values back off to the parent expert. At most
   4 such cases per action and column. They compete in the same attention as every other expert.
2. **Structure persists while content changes:** a translation column is found where most of its cells (≥ 75%) match,
   preferring an exact match. The cells that changed are rewritten in its content memory (the Spotlight lattice of its
   template).
3. **RESET is an action in the model:** its effect is known (back to the level's start state), so plans may include
   it. The agent no longer sends RESET because no plan was found. If there is truly no plan, it takes the action whose
   outcome is most uncertain, or an untried one.
4. **No depth cap:** the search runs to its node cap (3000), not to depth 12, so the nearest unvisited state is found
   however far away it is.

| # | prediction |
|---|---|
| K1 | W7 switch door with conjunctions: accuracy ≥ 0.97 on actions 101–300 (3 seeds; part 20: 0.942) |
| K2 | no regression: W1–W6 each within 0.01 of part 20's accuracy |
| K3 | on the 25 public games, ARCHITECTURE.md's B1–B6 are scored as pre-registered there (brain arm against the control's 5 of 183 levels and mean 0.0012) |
| K4 | no RESET is sent without a plan that includes it, and no single key takes more than 50% of level-1 actions in ar25, cn04 and dc22 (B2's second half) |

**Cost:** the conditions test in about 20 s; the 25-game run in ≤ 10 minutes.

## Part 21 results (2026-10-27)

**Conjunctions (`runs/test_conditions_part21.txt`; 14 s):**
- **K1 refuted as pre-registered:** W7's accuracy is unchanged at 0.942 over actions 101–300.
  - The cases do form, and the right conjunction is attended for "down": (door below AND the switch's state).
  - But cases form only after a door has been seen both passable and blocking, mostly after step 100.
  - With ~250 candidate partners and few door events, spurious pixels fit as well as the switch for a while: a
    multiple-comparisons problem.
- **Exploratory (not pre-registered), a 1500-action walk scored on actions 501–1500:** with conjunctions 0.996
  (0.999, 0.998, 0.99); without them 0.924. The mechanism works and is limited by samples. A random walk rarely
  produces the switch-on door cases; an agent that tests uncertain outcomes on purpose would.
- **K2 confirmed:** W1–W6 are each within 0.01 of part 20 (W6 improved, 0.962 → 0.977).

**The loops, from traces:**
- **The RESET loop:** it had two causes.
  - The dead zone: no plan, then RESET, then a random action.
  - M1 valuing RESET as an untried action worth 1 bit.
  RESET is now an action in the model (its effect: the level's first screen) and is used only inside plans. In ls20
  it acts as a teleport back to the start, which is cheaper than walking. M1 no longer values it.
- **Losing the avatar on a content change:** fixed. Columns are found by a ≥ 75% match, and the changed cells are
  rewritten in their content memory.
- **Speed:**
  - Predictions now read only the features their top experts need (lazy contexts).
  - Clicks elsewhere are cached per search.
  - Each search has a time budget (0.3 s; 0.05 s in the fast setting).
  - The harness runs the 25 games in parallel: about 1 minute in the fast setting (cap 3 × baseline), against
    about 10 minutes serially before.

**Fast setting, all 25 games (cap 3 × baseline, 0.05 s per search):**
- brain: 2 levels (lf52, m0r0), mean 0.0005;
- control: 1 level (lp85), mean 0.0011.

**Added after the pre-registration (not covered by K1–K4):** goal hypotheses about joint states ("joint": all
planning columns at once), the goal-level version of the conjunction problem.
- **The problem:** per-column "at" hypotheses are refuted when one column reaches the right state while another is
  wrong, as in ls20, where the avatar must reach the door with the right key.
- **Fast-setting result:** unchanged (2 levels). The joint family is spread over 10⁶ states, so it never outweighs
  the ~1 bit of learning what an untried click does.

**Full-length evaluation, all 25 games, part 21's code plus the joint goal family (`runs/brain_part21.json`; 340 s
wall, in parallel):**

| | levels solved | mean score | levels |
|---|---|---|---|
| brain | 5 of 183 | **0.0019** | ar25 L1 in 29 actions (human 32: full score), ft09 L1 in 99 (43), lf52 L1 in 78 (32), lp85 L1 in 144 (17), m0r0 L1 in 81 (30) |
| control (`runs/explorer_v0_fixed.json`) | 5 of 183 | 0.0012 | bp35, lf52, lp85, sp80, vc33 (all level 1) |

**Verdicts (ARCHITECTURE.md's B-predictions, via K3):**
- **B3 refuted:** 5 levels, not ≥ 10.
- **B4 refuted:** 1.6 × the control's score, not ≥ 2 ×.
- **B5 refuted:** no level 2 anywhere.
- **B6 split:** 340 s wall for all 25 games (confirmed), but about 300 ms per step, the search budget (refuted).
- **B1 and B2 were not measured in this run.**
- **K4:** RESET is now sent only from plans, 1,456 times across the games, mostly as a teleport to the start. The
  key-share half of K4 was not measured.

**Reading:** the brain arm now wins different levels than the control (ar25, ft09, m0r0 are new), at a slightly higher
score. Its limit is the balance between curiosity and goal-seeking. Learning an untried click is worth up to 1 bit,
while goal hypotheses spread over every reachable state are each nearly worthless. So the budget goes on learning the
model. **Next:** goal priors learned across games (warm-up play on the public games, shipped with the agent).

# Part 22: goal priors from warm-up play, and drives instead of a fixed order (pre-registered 2026-10-28, before the code)

**Why (the user):** warm-up rounds, so that what is learned on other games makes the first WIN of a new game fast
(parts 18–19's overhypotheses, applied to the real games).

**The problem it addresses:** part 21's agent spends its budget learning its model. An untried click is worth up to
1 bit, while each goal hypothesis is spread over every reachable state and so is worth almost nothing. The executive
also explores before it approaches (part 15b's fixed order), so goals never get a turn while anything is uncertain.

**Changes:**
1. **Goal types:** every goal hypothesis gets a type that carries across games.
   - "on-rare": a column lies wholly on a colour that covers at most 4 × the column's size on the screen, such as a
     marker, exit or slot.
   - "on-common": the same with a common colour.
   - "at": one column's state.
   - "joint": the joint state.
2. **A prior over types:** a hypothesis's prior is P(its type) / (the number of hypotheses of that type). P(type) is
   part 19's overhypothesis (a Dirichlet-multinomial with α learned), moved into `arc_agent/overhypothesis.py`.
   - **The draws:** at each WIN, the type of the most probable unrefuted hypothesis that held there, scored with even
     type priors (so each type is judged by its own description length, not by the prior being learned).
   - **When it updates:** within a game at every WIN (so level 2 benefits from level 1), and across games from the
     warm-up.
3. **Drives instead of the fixed order (part 17's DE17):** the plan chosen minimises
   D = (dp³ + du³ + de³)^½, where:
   - dp = (1 − P(WIN along the plan)) × hunger, and 0 once P(WIN) ≥ 0.5;
   - hunger = (actions since the level started or last WIN, + 30) / 25;
   - du = the share of the reachable information the plan leaves unlearned;
   - de = the plan's length / 50.
   Curiosity yields as hunger grows.
4. **Warm-up (`arc_agent/warmup.py`):** the agent plays every public game offline with a long budget (20 × the human
   baseline per level; warm-up actions are never scored) and records each WIN's type.
   - **Evaluation is cross-validated:** 5 folds of 5 games, each fold played with a prior learned only from the
     other 20 games' warm-up.
   - **What's shipped:** a prior learned from all 25 games.

**Arms, fast setting (all 25 games, cap 3 × baseline):** part 21's brain is recorded (2 levels, mean 0.0005).
- DRIVE: changes 1–3 with an even type prior.
- DRIVE+PRIOR: the cross-validated warm-up prior.

| # | prediction |
|---|---|
| G1 | the warm-up wins ≥ 15 levels across the 25 games, and its type counts are not all one type |
| G2 | DRIVE solves ≥ as many levels as part 21's brain (fast: 2) |
| G3 | DRIVE+PRIOR solves ≥ 4 levels in the fast setting, with mean score ≥ 0.001 (2 × part 21) |
| G4 | the full-length evaluation of DRIVE+PRIOR: ≥ 8 levels and mean ≥ 0.0024 (2 × the control) |

**Cost:** warm-up ≤ 6 minutes (parallel); each fast run about 1 minute; the full evaluation about 6 minutes.

## Part 22 results (2026-10-28)

**Changes made as pre-registered;** in addition, the harness reports each game as it finishes, and a warm-up game has
a 150 s wall-clock limit. Fast setting, all 25 games:

| arm | levels | mean | which |
|---|---|---|---|
| part 21 brain (recorded) | 2 | 0.0005 | lf52, m0r0 |
| DRIVE (even type prior, hunger growing) | 2 | **0.0009** | ar25 L1 in 88, lp85 L1 in 20 (human 17) |
| DRIVE with hunger held at 1.2 (ablation, asked by the user) | 1 | 0.0002 | m0r0 |

**Warm-up (`runs/warmup.txt`, `.json`; 222 s):** 2 levels won with 20 × the human budget: ft09 L1 in 525 actions
(type "joint") and lp85 L1 in 20 ("on-rare").

**Verdicts:**
- **G1 refuted:** 2 WINs, not ≥ 15. There is too little to learn a goal prior from: the agent rarely finds a WIN at
  all, even with a long budget.
- **G2 confirmed:** DRIVE solves 2 levels, as many as part 21, at a higher score.
- **G3 and G4 were not run.** With 2 draws the cross-validated prior is close to the even one, so the arm would test
  nothing new.

**Hunger (the user asked whether it is useful):** growing hunger did better than constant hunger (2 levels against 1;
0.0009 against 0.0002). In lp85, growing hunger stops the exploring after the button's effect is learned and repeats
it (20 actions); with constant hunger the agent keeps exploring. This is one run per arm with small counts. Hunger is
still a stand-in for the real budget, which every game draws on screen and which the clock columns already find.

**The bottleneck:** goal priors cannot be learned from the agent's own WINs, because it finds too few. Options:
(a) learn from watching the control play, since it stumbles into different WINs; (b) explore joint states the way the
control does, on top of the brain's model.

# Part 23: the symbolics folder's solutions, into the agent (pre-registered 2026-10-28, before the code)

**Why (the user):** do everything for which the symbolics folder's code already has the solution; afterwards, report
what the folder doesn't cover.

**What goes in:**

| solved in the folder | into the agent as |
|---|---|
| part 12: rotations act in the body frame (pose' = pose · h), translations in the world frame | a **rigid frame**, pose (x, y, r): translations and 90° rotations about the piece's own pivot. Recruited when a rotation explains a change that no translation does |
| part 10: content regularity is a quotient of the geometry | **quotient features:** (x mod k, y mod k) of a column's own state, k = 2…6 |
| OPEN_PROBLEMS #6: a richer vocabulary overfits, and description-length priors are the remedy | **description-length priors on experts:** an expert starts log K nats behind the unconditional one, K being the number of experts of its kind (conjunctions also pay for the choice of partner) |
| experiments A–B3: the function finder (`regression/claude/symreg2.py`; affine and relational atoms, BIC) | **numeric experts:** a translation's displacement as a fitted function of numeric relational features. The features are, along each of 4 rays, the distance to the first cell that differs from what lies beneath the column, and the relative position of every other column. Refitted as data arrive; they compete in the same attention |
| experiment B: library learning; the guide's case memory | **priors harvested from warm-up:** every game's learned conditions (the kind of the most-attended feature, relative to the operator: the cell moved into, the edge, another column, a quotient) and fitted formulas become an overhypothesis over expert kinds. In a new game, experts of commonly useful kinds start with that head start. This needs no WIN: conditions are learned in every game |
| parts 2–4: completion by group axioms | already implicit: operators are group actions, and the model composes them |
| part 6: irreducible representations; part 9: one map per position hypothesis | not used: column states are explicit group elements, and positions are observed directly |

**New hard worlds (`arc_agent/worlds.py`):**

| world | the condition | what predicts it |
|---|---|---|
| W8 rotating piece | an L-shaped piece; keys 1–4 translate it, ACTION5 rotates it 90° about its own pivot | the rigid frame (the rotation as a body-frame operator) |
| W9 invisible ladders | "up" works only where x ≡ 0 (mod 3) cells; nothing on screen marks it | a quotient feature of the avatar's position |
| W10 sliding ice | on an ice board the avatar slides until the cell before the first wall | a numeric expert: displacement = (distance to the wall − 1) cells |

**Protocol:** as part 20 (a 300-action random walk, 3 seeds, accuracy over actions 101–300), for W1–W10.
- **Arms:** P23 (this part) and P21 (part 21's code, from git, run on W8–W10 only; W1–W7 are recorded).

| # | prediction |
|---|---|
| D1 | P23 ≥ 0.95 on W8, W9 and W10; P21 ≤ 0.85 on each |
| D2 | no regression: P23 within 0.01 of part 21 on W1–W7 |
| D3 | the fast 25-game run, with warm-up-harvested expert priors (cross-validated as in part 22): ≥ 3 levels and mean ≥ 0.0009 (part 22's DRIVE) |
| D4 | the warm-up harvests condition kinds in ≥ 20 of 25 games, and the most common kind is "the cell moved into" |

**Cost:** the worlds in under a minute; warm-up about 4 minutes; each fast run about 1 minute.

## Part 23 results (2026-10-28)

**Hand-built worlds (`runs/test_conditions_part23.txt`, `.json`; 3 seeds, accuracy over actions 101–300; run in
parallel, 123 s wall):**

| world | P23 | P21 (part 21's code) | P23's most-attended expert |
|---|---|---|---|
| W1 walls | 0.993 | 0.993 (part 21) | the cell moved into |
| W2 inverted | 0.975 | 0.990 (part 21) | the cell moved into |
| W3 torus | 0.993 | 0.993 (part 21) | numeric, and an edge feature |
| W4 ice | 0.990 | 0.988 (part 21) | the cell moved into |
| W5 one-way | 0.988 | 0.993 (part 21) | the cell moved into; numeric for "right" |
| W6 speed switch | 0.972 | 0.977 (part 21) | the switch column |
| W7 switch door | 0.942 | 0.942 (part 21) | mixed |
| **W8 rotating piece** | **0.985** | 0.825 | the rigid frame tracks the turns |
| **W9 invisible ladders** | **0.988** | 0.952 | "up": (x mod 3), the quotient |
| **W10 sliding ice** | **0.962** | 0.860 | the numeric expert (the function finder's fit) |

**Measurement note:** the first P21 rows were wrong. They came out identical to P23 because the pool's worker
processes are reused: a worker that had run a P23 job kept the new code loaded. P21 was re-run in its own pool, and
those are the numbers above.

**Verdicts:**
- **D1, P23 half:** confirmed (≥ 0.95 on W8–W10).
- **D1, P21 half:** refuted for W9 and W10 (0.952, 0.860 > 0.85). Part 21's code partly memorises exact positions
  through its own-state feature.
- **D2:** confirmed except W2 (0.975 against 0.990, beyond the 0.01 allowed).

**Correction (the user, 2026-10-28):** the thesis is that the column *with its frame* is the reusable unit.
Generalising is easy in the right coordinate system, so the task is to pick the right column for the task, which is the
PFC's job. It is not one model shared across many frames, as I put it in chat.

**Warm-up (`runs/warmup23.txt`, `runs/warmup.json`; 283 s):**
- 2 levels won (the same two as part 22).
- Expert kinds harvested from 19 of 25 games: own state 385, other column 171, ahead (the cell moved into) 29,
  unconditional 16, elsewhere 14, numeric 6, conjunction 3.
- 15 library templates learned from the numeric experts' formulas.

**Two fixes during the warm-up:** a kind-classifier bug (a recolour operator with three colour pairs was read as a
move); and the harness now reports a game that crashes as that game's result instead of ending the run.

**Fast 25-game runs (cap 3 × baseline):**

| arm | levels | mean | wall |
|---|---|---|---|
| part 22 DRIVE (recorded) | 2 | 0.0009 | 57 s |
| P23, even priors | 1 (lp85) | 0.0008 | 169 s |
| P23, cross-validated warm-up priors | 2 (lp85 in 20, ar25 in 60) | **0.0011** | 127 s |

**Verdicts:**
- **D3 refuted on levels** (2, not ≥ 3); its mean-score half holds (0.0011 ≥ 0.0009).
- **D4 refuted:** kinds came from 19 games, not ≥ 20, and the most common kind is "own state", not "ahead".

**Reading:**
- **The folder's pieces work where they apply:** rotation, quotients and the function finder each solve their
  hand-built world (W8–W10).
- **On the real games they change little.** The bottleneck is still finding WINs.
- **The harvested priors expose a weakness:** the most-attended expert in most games is "own state", a lookup by
  exact position. That is memorisation, not a generalising condition, so a prior learned from it rewards memorising.
  This is the frame-choice problem: in a wrong or missing frame, the only thing left to attend to is the raw state.
- **Runs got slower** (the numeric experts' fits): 127–169 s for the fast 25-game run.

# Part 24: the prefrontal cortex, a column that learns which column to trust (pre-registered 2026-10-28, before the code)

**Why (the user):** build the PFC first, mathematically principled and biologically inspired. Meta-learning should
emerge without bespoke machinery, from PFC columns learning the way every column learns. The thesis (the user's
correction, part 23): the column with its frame is the reusable unit; generalising is easy in the right coordinate
system; so the job is to pick the right column for the task.

**The mathematics: Bayesian model selection with a learned prior.** When several assoc columns, each in a different
frame, explain the same change, each is a model. The responsible model is the one with the largest posterior,
log w(m) = log P(m | context) + Σ_t log P_m(observation_t): a prior times the prequential evidence (parts 3 and 13).
- **The prior P(m | context)** is a column's prediction: the PFC column predicts which kind of model will be
  responsible, from game-agnostic context. That context is the changed region's size, its number of colours, whether
  it is bar-shaped, whether it changed under every action so far, which action caused it, and the level.
- **Learning:** the PFC column learns that prediction exactly as any column learns what an action does: attention
  over context experts by track record, with tables from a feature's value to what was observed.
  - Its frame: a choice frame. The state is the responsible kind; the operator is a switch to a kind, or staying.
  - Its observation each step: which candidate predicted best.
  - One PFC column serves every contested region, as a reusable model over many instances of its frame.
- **Why meta-learning emerges:** the higher column learns the lower competition's prior (hierarchical Bayes). As its
  experience grows across regions, levels and games, the right model wins after fewer observations: learning to learn.
  Nothing in the code targets it.
- **Rule changes:** a switch is an ordinary operator of the choice frame, so a rule change is followed by evidence and
  the old model stays available (latent causes).

**The biology it follows:**
- PFC biases which pathway carries a task (Miller & Cohen 2001).
- Several paired forward models compete, and each one's responsibility is its predictive success (MOSAIC: Wolpert &
  Kawato 1998; Haruno, Wolpert & Kawato 2001).
- Context is inferred, and memories are created and expressed by inferred context (COIN: Heald, Lengyel & Wolpert
  2021). Task sets are reused across contexts (Collins & Frank 2013).
- Meta-reinforcement learning emerges in PFC because slow learning across tasks shapes a faster learning process,
  with nothing designed for it (Wang et al. 2018).
- Mediodorsal thalamus switches PFC between context representations (Rikhye, Gilra & Halassa 2018).

**Changes:**
1. **Recruitment keeps every explanation:** every frame type that explains a changed region (rigid, recolour, bar) is
   recruited as a candidate column, grouped by region. Recruitment's fixed order (translation before recolour) is
   removed; only the PFC's prior and the evidence decide.
2. **The PFC column** (`arc_agent/pfc.py`) is a `Column` with a choice frame (`frames.py`). It sets the
   responsibility of each group's candidates. Only responsible columns are planned with; the others keep learning,
   latent.
3. **Its learned state is shipped like any learned prior:** warm-up play leaves a PFC with experience.

**The test of emergent meta-learning (`arc_agent/test_pfc.py`):**
- **Episodes:** a sequence of 12 episodes, each a hand-built world chosen at random from three kinds, each kind
  ambiguous on its first step.
  - A one-colour block moved by keys: a translation, also explained by a recolour of the cells it touched.
  - A toggle panel whose cells change colour on a key: a recolour.
  - A step counter: a bar, also explained at first as a recolour.
  - New colours and layouts each episode.
- **Measure:** the next-state prediction errors on the contested region in each episode's first 10 actions.
- **Arms:** PERSIST, where the PFC column is kept across episodes and everything else is new per episode; and RESET,
  where the PFC is new each episode.

| # | prediction |
|---|---|
| P1 | learning to learn: PERSIST's mean errors in episodes 10–12 are ≤ 0.5 × its errors in episodes 1–3 |
| P2 | it is the PFC's learning: RESET shows no such fall (episodes 10–12 ≥ 0.8 × episodes 1–3), and PERSIST < RESET in episodes 10–12 |
| P3 | the PFC's attended expert for the kind of model is a region feature (number of colours, shape or clock-likeness), not the level or a random feature |
| P4 | no regression on the hand-built worlds W1–W10 (each within 0.01 of part 23), and on the fast 25-game run (≥ 2 levels; mean ≥ 0.0009) |

**Cost:** the meta-learning test under 1 minute; the worlds about 2 minutes; the 25-game run about 2 minutes.

## Part 24, first results (2026-10-28)

**Build check, before the main test:**
- The first version scored each candidate on its own cells only, so a recolour of the cells an avatar once touched
  kept predicting "no change" correctly after the avatar left, and its evidence matched the rigid model's (W1 fell to
  0.95).
- Model selection needs the rivals scored on the same observations. Fixed: every candidate predicts every cell any
  rival claims, with the cells it doesn't model predicted unchanged (`brain._group_evidence`).
- W1 back to 0.995, W4 1.0, W8 0.995.

**The pre-registered meta-learning test (`runs/test_pfc.txt`; 5 seeds × 12 episodes; 3.8 s):** errors per episode
(of about 9 contested steps):
- PERSIST: episodes 1–3 6.20, episodes 10–12 6.07 (ratio 0.98);
- RESET: 6.27 → 6.13 (0.98).
- The PFC's attended expert under PERSIST was its own state, or conjunctions of its own state with the region's size
  (P3 partly: size appears only inside conjunctions).

**Verdicts:**
- **P1 refuted:** no learning to learn.
- **P2:** RESET shows no fall (confirmed), but PERSIST is barely below RESET.

**Why, from traces: the test's design, not the mechanism.**
1. Each episode used 4 keys at random. The first press of each key is unknown to every candidate, so about half the
   contested steps are errors that no choice of model can avoid. That noise swamps the PFC's effect.
2. The toggling 12 × 6 block is explained rigidly as a 3 × 6 strip jumping 12 pixels: a valid rigid explanation of the
   changed pixels, with a region the same size class as the moving block's. So the one feature that tells the kinds
   apart was gone.

Within each episode, evidence chose correctly: the rigid model for the moving block, the recolour for the toggle.

# Part 24b: the corrected meta-learning test (pre-registered 2026-10-28, before it was run)

**Protocol:** as part 24, with two changes:
- each episode uses one key, chosen at random, for all 8 actions, so after the first action both candidates know
  what that key did and the second action is the decisive one;
- the toggle block is 12 × 12, so its region is a larger size class than the moving block's.

Errors counted from the second action on. Same arms (PERSIST, RESET) and seeds.

| # | prediction |
|---|---|
| P1b | PERSIST's errors in episodes 10–12 are ≤ 0.5 × its errors in episodes 1–3 |
| P2b | RESET shows no such fall (episodes 10–12 ≥ 0.8 × episodes 1–3), and PERSIST < RESET in episodes 10–12 |
| P3b | PERSIST's PFC attends to the region's size (alone or in a conjunction) |

## Part 24b results (2026-10-28; `runs/test_pfc_b.txt`, `.json`)

**First run of 24b:** still no learning to learn. A trace showed perception fragmenting the contest. The toggle changes
two separate strips; each strip became its own recolour candidate seeing half the swap ("11 → 0" only), and only the
first entered the contest. So the "recolour" rival could not predict the toggle back.

**Fix (recruitment, not the PFC):** a rigid explanation's rival is now one recolour over all the changed cells it
covers (both strips: a swap), so the rivals are compared on the same change. Re-run (5 seeds × 12 episodes, 40 s):

| arm | errors per episode, episodes 1–3 | episodes 10–12 | ratio |
|---|---|---|---|
| PERSIST | 0.53 | **0.13** | **0.25** |
| RESET | 0.67 | 0.53 | 0.80 |

- **The PFC's most-attended expert,** in all 5 PERSIST seeds, is a conjunction: (its own state is None, i.e. the
  region has no model yet) AND the region's size. It arose from part 21's rule: the "own state" expert's prediction
  for new regions was refuted, which created conjunction experts for that case.
- **P1b confirmed** (0.25 ≤ 0.5).
- **P2b confirmed:** RESET's ratio is 0.80 (at the threshold), and PERSIST < RESET in episodes 10–12.
- **P3b confirmed:** size, in a conjunction.

**Learning to learn emerged from ordinary column learning:** the PFC column learned the competition's prior across
episodes, so the right model was chosen at the decisive step. No code targets meta-learning.

**P4, regressions:**
- **Worlds:** W1–W10 are identical to part 23 (`runs/test_conditions_part24.txt`). Confirmed.
- **25 games, fast setting:** the first run gave 0 levels. Its search was bounded by wall-clock time, so with 16 games
  in parallel each search had less compute and results depended on machine load. That is an apparatus flaw: the
  harness now bounds the search by nodes (400 in the fast setting, 3000 in full runs), and two repeat runs of lp85 and
  ar25 gave identical results.
- **With the node budget:** 2 levels (ar25 L1 in 23 actions, under the human's 32, so full score; ft09 L1 in 101),
  mean **0.0015**. Confirmed (≥ 2 levels, ≥ 0.0009); the best fast score so far. Earlier parts' fast scores were
  measured with the wall-clock budget, so the comparison is approximate.
- **lp85 is lost** (part 23 solved it in 20 actions). Its button moves tile contents around a loop, which is neither a
  rigid motion nor a colour map. Every candidate keeps failing and being re-recruited, so the planner never holds a
  stable model of the button. This needs a frame for contents permuted along a path (not covered by the folder).
