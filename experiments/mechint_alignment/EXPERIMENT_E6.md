# E6: The one-network agent from a random initialization

**Status:** pre-registered 2026-10-20 (evening). Each run ≤ 5 min (270 s of training).

## The user's decisions this implements (2026-10-20 evening)

- The robot's **values live in the network's weights.** No separate reward model, no hand-written planner at decision
  time.
- **No scripted-policy data.** The network's own random initialization babbles.
- **Exploration by learning progress, not uncertainty** (the noisy-TV problem).
- **No PPO comparison.** The scripted task policy is an evaluation yardstick only.

## The agent (`onenet.py`, `e6_agent.py`)

**One network.** The E2 world-model trunk with the three priors (relative-position attention, the locality window,
the persistence prior). For each action, one representation of the global token feeds:
- the next global state (the world model; cells use the same mixing with their own features);
- the **reward** of that action;
- **Q(s, a),** its expected return: the robot's values.

The policy is softmax(Q / 0.1). Q starts at exactly 0, so a fresh network acts uniformly at random: babbling from the
network itself.

**Acting.** 64 lanes in lockstep on the 5 × 5 world (≤ 2 villagers, all dilemma modules at their usual
probabilities).
- One trunk pass gives the action and the prediction of the step's outcome.
- The step's in-scope surprise and surprise kind (E2's signature) are computed from that prediction.
- Each step's 10 % most surprising transitions are its refutations.

**Intrinsic reward:**

| arm | intrinsic reward |
|---|---|
| **E6a, lp** | 0.05 × the learning progress of the refutation's kind (slow EMA − fast EMA of that kind's refutation surprise, nats, ≥ 0, saturating at 5) |
| **E6b, none** | 0 |
| **E6c, surprise** | 0.05 × the raw surprise (saturating at 5): the noisy-TV control |

**Learning.** One update per collection step: a batch of 256 from a replay of the robot's own last 200k transitions.
The loss is a single sum:
- next-state cross-entropy, robot-local;
- (reward(s, a) − r)²;
- Huber(Q(s, a) − y), with y = r + r_int + 0.99 · max Q_target(s', ·).
  - Q_target is a slow EMA copy (0.995).

Adam 3e-4, d = 64, 3 layers.

**Measures:** every 50k steps, on 64 held-out layouts:
- return, gold delivered, robot deaths, theft rate;
- **the fraction of steps within 2 cells of a villager** (the noisy-TV signature);
- the kinds with the highest learning progress.

The scripted task policy's return on the same layouts is the yardstick.

## Predictions

- **P1:** E6a delivers more gold at the end than E6b.
- **P2:** E6c spends a larger fraction of steps near villagers than E6a.
- **P3:** none of the arms reaches 50 % of the scripted policy's return in 5 min. A from-scratch job is expected to
  need longer; the trend is what is measured.

## Results

(appended below)

### Results (single runs, 270 s each; the scripted policy returns 12.07 on the evaluation layouts)

| arm | steps | return, last 3 evals | gold delivered | robot died | steps within 2 of a villager |
|---|---|---|---|---|---|
| E6a, learning progress | 254k | **−1.38** | 0.17 | **0.07** | 0.59 |
| E6b, no intrinsic reward | 215k | −1.65 | 0.17 | 0.19 | 0.59 |
| E6c, raw surprise | 222k | −1.68 | 0.19 | 0.20 | 0.60 |

All three start at −1.72 return, 0.19 gold, 0.25 deaths, 0.66 near villagers.

- **P3 confirmed:** none reaches 50 % of the scripted return. **No arm learned the job** in ~3,400–3,900 updates: gold
  stays ~0.1–0.25 per episode, as at initialization. All three learned to avoid fire.
- **P1 refuted:** E6a delivers no more gold than E6b. E6a did learn fire avoidance best (deaths 0.07 against 0.19–0.20,
  return −1.38 against −1.65), but that is one run per arm.
- **P2 refuted at this scale:** raw-surprise curiosity did not visibly pull the robot toward villagers (0.60 against
  0.59). The intrinsic reward (≤ 0.05 per step) may be too weak to show it in 5 min.
- **The learning-progress tracker's top kinds in all three arms were villager moves at the edge of the robot's window**
  (occupant / occupant id two cells away, diagonals). The model is still learning the *distribution* of the villagers'
  random steps, which is genuine, bounded information.
  - So learning progress pays for noise until that distribution is learned (Brainstorm 4).
  - The principled fix is expected information gain from an ensemble with randomized priors, which separates the
    epistemic part.

**Reading.** At this scale and time (d = 64, 5 min, ~250k steps), one network trained from scratch learns the
dangers but not the job.
- The world-model results (E2) needed ~600k transitions and the mixed data's job events to learn mining and delivery.
- The robot's own babbling yields few of them (random play: mining progress ~1 per 340 steps, delivery ~1 per 7,600).
- **Next, from Brainstorm 4:** expected information gain (BALD-style disagreement with randomized priors) as the
  curiosity signal, so the never-tried interactions (a mine, a chest) are where curiosity points.
