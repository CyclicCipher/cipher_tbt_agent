# E4: Acting by planning in the learned world model, learning from refutations while acting

**Status:** pre-registered 2026-10-20 (afternoon, the user away). Runs ≤ 5 min.

## Question

Can the robot do its job with **no policy learning at all**, by searching for a plan in its own world model, and
improve by learning from the refutations its actions produce?
- This is the user's principle applied to acting: the robot predicts the outcome of a plan, the world refutes a step,
  the robot works out why (rehearsal until the prediction is right), then plans again.
- If it works, competence costs one world model (~5 min of training on 614k logged transitions) plus acting, not PPO's
  hours.

## Setup

- **World model:** K6's (J2's wake + label-free sleep + slow read-out; `EXPERIMENT_E2.md`), saved by a deterministic
  re-run of K6.
- **Reward model** (`planner.py`): least squares on generic features of the global token (one-hot of each field's
  change and new value), fitted on the 614k logged transitions. It fits them exactly (max residual 0.0000).
- **Planner:**
  - breadth-first search, up to 40 steps deep, in the model's argmax predictions;
  - de-duplicated by the robot-relevant predicted state;
  - returns the shortest predicted path to a transition with predicted reward > 0.5, never through one predicted to
    hurt.
  - The agent follows the plan and re-plans when an observation differs from the predicted robot-relevant state
    (a refutation).
- **Learning while acting** (`RefutationLearner`): on each refutation, rehearse the refuted transition (8 copies, half
  replaced by earlier refutations once there are some) + 24 logged rows, Adam 1e-4, until the model predicts it
  (≤ 30 updates). The model persists across episodes.
- **Episodes:** held-out layouts (seeds ≥ 10⁹), played in order.
- **Yardstick:** the scripted task policy on the same seeds.

## Smoke results (before pre-registration of the trend run)

**Plan only, no learning (2 episodes):** returns 0.72 and −1.28 (scripted on the same seeds: 10.72), with 110–124
re-plans per episode.
- **Diagnosis, seed 10⁹+1:** the plan is sensible (north ×4, interact ×3, to the depot, deliver). But the robot ends
  facing *north* with the mine to its *east*, and the model predicts that the mine progresses anyway.
- The model learned a **spurious correlate**, "interact next to a mine", instead of the cause, "interact facing the
  mine". In the logged data the scripted policy always faces the mine when it interacts, so the two were confounded.
- **The planner exploits the error, the world refutes it, nothing learns, and it repeats forever.**

**Plan + learning from refutations (3 episodes):** returns 4.32, 1.72, −1.92 (scripted 12.39). 6, 3 and 2 gold
delivered. Every refutation explained (14/14, 19/19, 27/27), each after ~2–3 updates.

## The trend run (pre-registered)

Up to 20 consecutive held-out episodes with learning, ≤ 270 s.

**Prediction P1:** over the episodes,
- refutations per episode fall (the mean of the last 5 is ≤ 70 % of the mean of the first 5);
- the return rises (the last 5's mean ≥ the first 5's mean + 2).

## Results

(appended below)

**Amendment after the first trend attempt.**
- Four episodes took ~105 s: returns 4.32, 1.72, −1.92, −1.28; per-episode refutations 14, 5, 8, 17 (the logged counter
  was cumulative).
- Then the fifth episode stalled past the 5-minute cap and the process was killed. When no rewarded path exists in
  the model (e.g. frozen, or the job unreachable), every step paid for a full 40-deep search.
- **Fix:** after a failed plan, take 8 random moves before planning again, and cap the frontier at 600. Same
  prediction.

**Second attempt** (fix above): five episodes in ~120 s, then the sixth stalled again.
- Returns 2.32, −1.28, −3.24, 0.72, −1.28; refutations per episode 8, 3, 1, 9, 6, all explained.
- An episode with no predicted rewarded path still costs ~16 full searches.
- **Third configuration:** depth 30, frontier 300, 16 exploratory steps after a failed plan. Same prediction.

### Result of the trend run (third configuration, 30 episodes, 155 s): P1 refuted; the trend is *negative*

| episodes | returns | delivered | refutations per episode |
|---|---|---|---|
| 1–4 | 3.52, 3.72, −2.19, −0.28 | 6, 5, 2, 1 | 12, 2, 12, 14 |
| 5–12 | mostly −1 to −2 (0.72 once) | 0–2 | 0–6 |
| 13–30 | −1.28 typical, down to −4.08 | **0** | **0** |

- From episode 13 on, every episode has **8 failed plans**: 128 steps / 16 exploratory steps per failure.
- Mean return −1.21 against the scripted policy's 13.05 on the same seeds.

**Reading.**
- From episode 13 on, the model no longer predicts *any* reachable reward. So the planner never commits to an action
  it could be wrong about, and refutations stop.
- **Fixing 69 refutations one at a time,** interleaved only with random logged rows, damaged the rules it needed
  (mining, delivery). It is arm K's interference, now inside the acting loop.
- "Explained 69/69" was true for each refuted transition, while the model got worse overall: the K finding that
  explaining one case can break the general rule.

### E4c, pre-registered: the protected loop

The protection E2 found:
- **plan with a slow copy** (EMA 0.998 of the weights; K5e, K6);
- the refutations update the fast copy;
- rehearsal interleaves past refutations **evenly across kinds** (the K3 pool, built from the logged data) instead of
  random rows.

**Prediction P2:** over 30 episodes,
- the last 10 deliver ≥ the first 10 minus 1, i.e. no collapse;
- at least 20 of 30 episodes have at least one plan found.
