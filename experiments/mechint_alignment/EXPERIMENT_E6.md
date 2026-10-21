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
