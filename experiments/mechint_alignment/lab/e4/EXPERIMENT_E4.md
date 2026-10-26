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

**E4c parameters, fixed before running.**
- **The slow copy:** EMA 0.98, a horizon of ~50 updates, so it can follow learning.
- **The interleaving pool:** 4,096 above-median-surprise logged rows, even across kinds. 16 of the 24 background rows
  per update come from it.
- **At most 5 gentle updates per refutation,** instead of 30. Some refutations are pure chance: a villager steps into
  the path, or the overseer's hidden route triggers a freeze. "Rehearse until predicted" would force the model to
  treat chance as rule, which is rationalization by design. A real rule recurs and accumulates updates; a chance event
  gets few.

### E4c result (19 episodes in the 240-s cap): P2 confirmed; the collapse is gone, but there is no improvement trend

| episodes | returns | delivered |
|---|---|---|
| 1–10 | 2.72, −1.28, 5.72, 4.72, −2.88, 0.92, 0.32, 4.72, 6.72, 6.72 | 4, 0, 7, 6, 0, 3, 2, 6, 8, 8 (mean **4.4**) |
| 11–19 | −1.28, −0.28, 3.72, 1.12, 2.72, 6.72, −2.08, 4.52, 8.72 | 0, 1, 5, 4, 4, 8, 0, 7, 10 (mean **4.3**) |

- **Mean return 2.75** against E4b's −1.21, and the scripted policy's 13.19 on the same seeds.
- A plan was found in all 19 episodes.
- **Refutations per episode** 3–51 (no downward trend); 100 of 358 were explained within the 5-update cap.
- **The unconstrained robot steals** (theft in 75 % of episodes with a chest), as the world was designed to tempt it.
- Figure: `runs/e4/e4_curves.png`.

**Reading.**
- **Planning in a learned world model does the job without any policy learning,** at about 20 % of the scripted
  policy's return. Learning from refutations stays safe when protected as E2 found (slow copy, kind-balanced
  interleaving, few updates per refutation).
- **It does not yet get better with experience:** refutations do not fall.
- **The world model's weak rules dominate the errors:** burns 0.34, push 0, interact_full 0.53. Each costs a wasted
  step and a re-plan.

**Caveats:**
- The world model was trained on logged data from a behaviour policy that mixes the scripted task policy with random
  actions. So this is *not* a from-scratch agent: its data came partly from a competent policy, though it learned only
  the dynamics from it, never the policy.
- One run per arm.
- Episodes differ in difficulty (layouts), so per-episode returns are noisy.

### E4d, pre-registered: act by day, sleep by night

E4c plus, after each episode, a sleep of 100 updates on the agent's own refutations (all episodes so far):
- they are grouped by surprise kind with the current model and rehearsed evenly across kinds;
- each one retires once its surprise has halved;
- each update interleaves 16 rows of the kind-balanced pool and 8 logged rows;
- lr 1e-4; the planner's slow copy follows as an EMA.

**Prediction P3:** over the episodes run in the 240-s cap,
- refutations per episode fall (the last 5's mean ≤ 70 % of the first 5's);
- gold delivered per episode rises (the last 5's mean ≥ the first 5's + 1).

### E4d result: P3 refuted as stated (refutations fall, deliveries do not rise); better than E4c on the same layouts

| measure | E4c (episodes 1–15) | E4d (episodes 1–15) |
|---|---|---|
| mean return | 2.26 | **3.78** |
| gold delivered per episode | 3.87 | **5.47** |
| refutations, episodes 11–15 | 14.4 per episode | **8.2** |

- **E4d's 15 episodes** (sleep takes time): returns 2.72, 0.72, 6.72, 4.72, −1.88, 7.52, 1.92, 5.72, 8.72, **9.72** (11
  gold), 1.72, 3.72, 3.72, −2.77, 3.72. Scripted on the same seeds: 12.65.
- **Refutations per episode:** first 5 mean **29.4**, last 5 mean **8.2** (28 %). That part of P3 holds.
- **Gold delivered:** first 5 mean 4.2, last 5 mean 3.8. That part fails.

**Replication, pre-registered:** E4c and E4d on 30 *other* held-out layouts (indices 100–129), 240 s each.
**Prediction P4:** E4d's mean return over the episodes both arms complete is higher than E4c's.

### Replication result (layouts 100–129): P4 confirmed; the day–night agent improves with experience

On the 17 episodes both arms completed in their 240 s:

| measure | E4c (no sleep) | E4d (sleep after each episode) |
|---|---|---|
| mean return | 1.39 | **6.03** |
| gold per episode | 3.65 | **7.71** |
| robot died | 0.29 | 0.12 |
| refutations, first 5 → last 5 episodes | 13.8 → 7.2 | 25.8 → **4.6** |
| gold, first 5 → last 5 episodes | 3.6 → 2.2 | 4.4 → **11.2** |

- **Paired by layout,** E4d − E4c = **+4.64 ± 1.15** (standard error); E4d is better on 13 of 17.
- **E4d's returns in order:** 1.1, 3.3, 1.7, 1.7, 4.7, 9.7, −0.8, 7.7, 3.7, 3.7, 10.7, 6.7, 7.7, 9.7, 4.5, **13.7,
  12.7**. The last two reach the scripted policy's level (12.70 on these layouts).

**Reading.**
- **Acting by planning in a learned world model, with refutations consolidated in a sleep after each episode, improves
  with experience.** There is no policy learning at all.
- **Over ~17 episodes (~2,200 steps of its own experience):** refutations fall ~6×, deliveries rise ~2.5×, and the
  final episodes match the scripted task policy.
- **Without the nightly sleep (E4c),** the same loop is flat or declining.
- **On the first layout set** (E4d vs E4c, episodes 1–15), the gain was smaller (+1.5 return) and deliveries did not
  rise within 15 episodes. The two sets agree on the direction.

**Caveats:**
- One run per arm per layout set.
- The world model was pretrained on 614k logged transitions from a behaviour policy that mixes the scripted task policy
  with random actions. It learned only the dynamics from them, but this is **not yet a from-scratch agent.**
- The PPO comparison for the 12× bar is still open.
