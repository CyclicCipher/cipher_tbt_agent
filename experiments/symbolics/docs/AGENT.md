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
