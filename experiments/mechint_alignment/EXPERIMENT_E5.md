# E5: The day–night agent from random play only (no competent policy anywhere)

**Status:** pre-registered 2026-10-20 (afternoon, the user away). Each stage ≤ 5 min.

## Question

E4d's world model learned its dynamics from logged data mixing the scripted task policy with random actions. Does the
day–night agent still work when its world model has only ever seen **random play**?

## Data

614,400 transitions of uniformly random actions on training layouts (`transitions.py --epsilon 1.0`).

| rule | instances |
|---|---|
| move | 233,163 |
| turn | 65,687 |
| frozen_tick | 17,619 |
| trapped_hurt | 12,722 |
| push | 5,314 |
| robot_burn | 4,553 |
| mine_work | 1,823 |
| villager_death | 1,580 |
| theft | 1,045 |
| button_off | 387 |
| interact_full | 365 |
| **mined** | **219** |
| **deliver** | **81** (58 of one nugget, 23 of two) |
| push_into_pit | 57 |
| rescue | 35 |

The job's two rewarded steps (mined, deliver) are in the few-shot regime.

## Pipeline (the session's best recipe, unchanged)

1. **Wake:** J2's recipe on this stream (locality window, persistence prior, robot-local loss, the open-questions
   buffer).
2. **Sleep:** label-free, uniform kinds, 1,000 updates, slow read-out (K6's recipe).
3. **Act:** the day–night agent (E4d's settings) on held-out layouts 100–129, 240 s. Its reward model and background
   rows come from the random data too.

## Predictions

- **P1:** after stage 2, held-out `mined` ≥ 0.8 and `deliver` ≥ 0.8, each from < 250 instances.
- **P2:** in stage 3, gold per episode rises (the last 5's mean ≥ the first 5's + 2), and the mean return over the
  completed episodes ≥ 2. For reference, the scripted policy gets 12.70 on these layouts, and E4d with the mixed-data
  model got 6.03.

## Results

(appended below)

### Stages 1–2 result: P1 refuted

**Wake** (J2's recipe on random play):
- move 1.00, turn 1.00 (90 % after 115,725 and 32,438 instances);
- robot_burn **0.79**, trapped_hurt 0.70, push 0.41: better than the mixed-data model, because random play walks into
  fire and villagers more;
- **every interaction rule at 0:** mine_work, mined, theft, deliver, button_off.

**Sleep** (K6's recipe):
- button_off 0.00 → **0.96** (from 387 instances), push 0.41 → 0.71, robot_burn → 0.87, trapped_hurt → 0.78;
- **mine_work 0.02, mined 0.01, theft 0.00, deliver 0.00:** the job's rules are not learned from random play.

### Stage 3 result: P2 refuted, and a design gap found

- **30 episodes:** mean return −1.62, 0 gold, **0 refutations.**
- The model never predicts a rewarded path, so the agent only explores (16 random steps per failed plan).
- **The gap:** predictions were only checked while following a plan, so nothing was learned from exploration.

### Stage 3b, pre-registered: every action is a prediction

During exploration, the model also predicts each step's outcome, and a refuted one is learned from, as in the plan.

**Prediction P3:** within 240 s, at least one rewarded plan is found and executed (gold > 0 in some episode after the
first), and refutations are recorded.

### Stage 3b result: P3 refuted

- **11 episodes in the 240-s cap,** cut short by long exploration bursts.
- **Refutations are now recorded** (5, 7, 4, 2, 0, 2, 2, 1, 1, 5, 0 per episode).
- **No gold delivered in any episode.** Returns −1.28 to −3.95: fire deaths during random exploration.

**Reading.**
- From random play, the world model never learns the job's rules: mining, theft and delivery are all ≤ 0.02 after
  wake and sleep.
- Random exploration, even when every step is checked against the model and refutations are learned from, does not
  find them within ~1,400 steps.
- **What is missing is *directed* exploration:** act where the model cannot yet predict, to test it.
  - For example, interact with objects whose response the model is unsure of.
  - This is the learning-progress / epiplexity idea from the start of this line (`TRAINING_METHOD.md`, Brainstorm 2),
    and the next step.
- **E4d's success depended on logged data that contained the job's events** (from the scripted part of the behaviour
  policy).
