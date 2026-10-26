# E3: Does learning from refutations make the robot's own training faster?

**Status:** pre-registered 2026-10-20, during the user's 6-hour absence. Each run ≤ 5 minutes; curves logged.

## Question

The robot's network learns its task by RL (PPO, the control). Does adding the world-model loss on the **same trunk**
speed that learning? The world-model head predicts the next observation for each action taken, with the persistence
prior.
- Every step then delivers a dense signal: the step's refutations shape the features the policy reads.
- This is the "OPSD that doubles as pretraining" idea in its simplest form.
- It is also the UNREAL result (Jaderberg et al. 2016), where auxiliary prediction tasks gave about 10× data
  efficiency, tested here.

## Setup

- **World** (`village.py`, `size=5`): the central 5 × 5 cells of the map, walls around them, same 81-cell observation.
  At most 2 villagers; the dilemma modules at their usual probabilities.
  - Scripted reference return: 12.27.
  - Random play: one delivered nugget per 1,095 steps (9 × 9: 4,348).
- **Network** (`agentnet.py`): the E2 world model's trunk without the action input (relative position, locality window
  radius 1, d = 128, 4 layers). The policy and value read the global token. The world-model head is a per-token MLP on
  the trunk features plus the action and global-token embeddings, then the field heads with the persistence prior.
- **PPO** (`e3_aux.py`): 128 envs × 128 steps per update, 3 epochs, minibatch 2,048, Adam 3e-4, γ 0.99, λ 0.95,
  clip 0.2, entropy 0.01. Each run is cut at **280 s of training**.
- **Measures:**
  - every 100k steps: return, gold delivered and theft rate on 128 held-out layouts;
  - per update: training-episode return, policy entropy, world-model loss.

## Arms

| arm | what |
|---|---|
| **P** | PPO alone (`--aux 0`): the control |
| **W** | PPO + world-model loss (`--aux 1`) |

## Prediction

**P1:** W reaches the return P reaches at its end (its 280 s) in ≤ half the env steps. The 12× bar is not expected
from this ingredient alone.

## Results

(appended below)

**Amendment, before any arm ran (2026-10-20):** the trunk is **d = 64, 3 layers**, not d = 128, 4 layers.
- At d = 128 one 16k-step update took ~33 s, only ~8 updates in a 280-s run.
- At d = 64 / 3 layers it takes 7.5 s (P) and 8.7 s (W), mostly fixed overhead, about 600k env steps per run.
- Everything else as registered.

### Results (P and W, 280 s each): P1 refuted

| step (k) | P return | P delivered | P robot died | W return | W delivered | W robot died |
|---|---|---|---|---|---|---|
| 0 | −1.78 | 0.20 | 0.29 | −1.78 | 0.20 | 0.29 |
| 114 | −1.51 | 0.23 | 0.19 | −1.87 | 0.12 | 0.28 |
| 213 | −1.54 | 0.19 | 0.17 | −1.69 | 0.23 | 0.27 |
| 311 | −1.38 | 0.18 | 0.09 | −1.73 | 0.14 | 0.23 |
| 410 | −1.21 | 0.22 | 0.04 | −1.45 | 0.33 | 0.20 |
| 508 | −1.17 | 0.19 | 0.02 | — | — | — |

- **Neither arm learned the job** within ~0.5M steps. P's gain is from learning to avoid fire: deaths 0.29 → 0.02.
- **W was slower at that:**
  - deaths 0.20 at 410k against P's 0.04;
  - its policy entropy fell more slowly (2.07 → 1.88, against P's → 1.66);
  - its world-model loss fell from 1.44 to 0.70.
- **W's slightly higher delivery** (0.33 against 0.22) is within the noise of 128 evaluation episodes.
- **At this tiny scale (d = 64), the auxiliary loss competes with the policy** for the trunk rather than helping it.
- This budget (~0.5M steps) cannot measure the task part at all.
