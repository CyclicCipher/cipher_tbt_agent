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
