# E2: How fast does a world model learn Village's rules from refutations?

**Status:** done for this session (2026-10-20). 23 arms pre-registered before they ran, plus 3 labelled exploratory runs
(G2, K3, M5b). Results are appended in
order below the summary. The user was away ~6 h, with the instruction to develop the learning rule, keep runs ≤ 5 min,
and record learning trends.

## Summary (written at the end of the session, 2026-10-20)

**Question:** how many refutations does a world model need to learn each of Village's rules, and what reduces that
number?

**Figures:**
- `runs/e2/e2_curves.png`: learning curves per rule, arms A, B, E, G, I, J2;
- `runs/e2/k_curves.png`: arms K, K2, K3;
- `runs/e2/k4_curves.png`: arm K4.

**Results, in order of the arms:**
1. **Plain online training learns almost nothing.** Arm A, Adam on the full next-state loss: after 614k transitions,
   `move` (268k instances) and `theft` are at 0.00, and the rules it does learn needed 10⁴–10⁵ instances.
2. **Three general priors each roughly halve the instances needed:** translation-equivariant position (B), the
   persistence prior (E), the locality window (G).
   - **Restricting the loss to the robot's surroundings** (I, a hand-scoped diagnostic) helps most: move 1.00, turn
     0.98, theft 0.51, mined after 3,925 instances (A: 22,696).
3. **What was slowing it down:**
   - refutations are outnumbered ~200:1 by confirmations (cured by the persistence prior);
   - they are drowned by the world's own unpredictable dynamics: 40 % of the remaining error is the overseer's vision
     field, ~21 bits are villagers' random steps;
   - *where to look* must be learned before any rule (the locality prior helps);
   - **a rule takes ~10³ SGD updates however it is fed:** smaller batches (F) and a higher learning rate (G2) did not
     help.
4. **Rehearsing refutations inside the stream gives at best modest gains.** Best is J2: theft 0.68, robot_burn 0.34,
   changed-entry accuracy 0.446.
   - Over 11 arms (H, J, J2, L, L2, L3, M–M6), per-row selection is fooled by noise and per-kind progress estimates
     are noisy.
   - In-stream sleep phases fight the waking updates: theft drops after each sleep and recovers while awake (M5b, M6).
5. **The few-shot capacity exists** (K, K3). A trained model learns `button_off` from **one** refutation: 0.86 on
   held-out layouts, stable for 600 updates.
   - *Robustness check:* with other single instances, 0.75 and a fluctuating ~0.5. One-shot generalization depends on
     which case is seen. Four instances give 0.93.

   The conditions:
   - rehearse the refutation **until explained, then retire it;**
   - interleave **past refutations, evenly by kind;**
   - take **gentle steps** (lr 1e-4).

   It is not forgotten, and theft even improves (0.68 → 0.85). Rehearsing one case without retiring it is
   **rationalization, measured:** held-out accuracy 0.86 → 0.18, while the rehearsed case itself stays predicted
   correctly at every checkpoint (accuracy 1.00, surprise 0.003–0.07 bits; measured 2026-10-20 in `runs/e2/K_fitcheck`,
   after first being stated as an inference).
6. **Label-free sleep adds rules to a trained model without forgetting** (K4). From the refutations the model finds
   itself, grouped by surprise kind with no event labels:
   - button_off 0.00 → 0.89, theft 0.68 → 0.89;
   - move, turn, mined and deliver stay ≥ 0.96; unchanged-entry accuracy 0.997.

   Choosing kinds uniformly does as well at the end as the probe-progress bandit; the bandit is only faster early.
   - *From scratch* (K5: plain wake + this sleep, ~250 s): button_off 0 → 0.71 and interact_full → 0.89 are gained, but
     theft ends 0.58, after peaking at 0.81 (J2: 0.68).
6b. **A slow read-out completes the sleep** (K5e, K6). Read out an exponential average of the weights (d = 0.998).
   - **During sleep on a mature model,** the swings vanish and the improvement is monotone.
   - **The best model of the session** (K6: J2's wake + label-free sleep + slow read-out):
     - theft 0.68 → 0.87, button_off 0 → 0.79, trapped_hurt 0.40 → 0.48;
     - robot_burn kept at 0.34; every frequent rule ≥ 0.96.
   - **During training from scratch, the slow copy lags badly** (M7) and is worse than the fast one.
7. **Rules whose features are missing are not learned by any rehearsal here.** push needs reasoning two cells away;
   robot_burn needs the global token to know what the robot stepped on. Push reached 0.51 once (M5, just after a
   sleep).

**Caveats** (verify the apparatus):
- one seed per arm;
- the held-out sets of rare rules are small (button_off 28, push_into_pit 4, rescue 6), so single-arm differences
  under ~0.1 may be noise;
- some arms were cut by the 5-minute cap (L, L3: 84 % of the stream);
- evaluations just after a sleep are inflated by false alarms (the M arms);
- the early ~0.48 "changed-entry accuracy" of A and B is an artefact of predicting zeros;
- "seen to 90 %" is coarse (log-spaced evaluations).

**What it means for the training method:**
- **The learning rule that works so far is "wake, then sleep":**
  - wake: learn online with the general priors;
  - sleep: rehearse the refutations the model still cannot explain, evenly by kind, until explained, interleaved with
    past refutations and ordinary experience, with gentle steps.
- **That turns 1–30 refutations into a rule,** where online SGD needed ~10³ updates and ~10⁴ instances, *for rules
  whose features exist*.
- **Not solved:**
  - running sleep *during* the stream without the two phases fighting;
  - building the features for multi-step rules faster;
  - connecting the world model to acting (E3 found no RL speed-up at a tiny scale).


## Why

The user's principle: the densest signal is "the model makes a prediction, is refuted by the world, and must work out
why" (`TRAINING_METHOD.md`, Brainstorms 2–3).
- E2 measures the first half directly: **how many refutations does each rule of the world take to learn**, under
  different learning rules and priors.
- A rule learned from few refutations is the lever behind the ≥ 12× speed-up the new training method needs.

## Setup

**Data** (`transitions.py`, now `rollouts.py` + `village_events.py`):
- 614,400 training transitions: 256 parallel lanes × 2,400 steps, time-major. 102,400 held-out transitions come from
  layouts never trained on.
- The behaviour policy mixes the scripted task policy with uniform random actions. Each episode draws its random
  fraction from {0.2, 0.5, 1.0}.
- Each transition is labelled with the rule events that fired. For each event, the **check entries** are the
  (cell, field) entries whose next value that rule determines, e.g. theft: the chest's amount and the robot's carried
  gold.

**Rule frequencies** in the training stream, spanning four orders of magnitude:

| rule | count |
|---|---|
| move | 268,016 |
| turn | 71,438 |
| mine_work | 47,208 |
| mined | 22,696 |
| frozen_tick | 19,310 |
| deliver | 17,643 |
| trapped_hurt | 12,326 |
| theft | 6,145 |
| push | 2,778 |
| robot_burn | 2,449 |
| villager_death | 1,532 |
| interact_full | 770 |
| button_off | 191 |
| rescue | 18 |
| push_into_pit | 17 |

`interrupted` (1,981) depends on the overseer's hidden route, so it is not checked.

**The world model** (`worldmodel.py`):
- Input: the observation plus the action, which is embedded and added to every token.
- Output: the next observation, one categorical prediction per field of every cell and of the global token.
- The map is padded with its border walls to 11 × 11 tokens.
- Size: **d = 128, 4 layers, 4 heads, MLP 4×, about 0.8M parameters.** The 1.8M size would take 15 minutes per run;
  E2 is about the learning rule, not the final size.

**Training: online.**
- The stream is consumed once, in order: one time step of all 256 lanes per update, as experience arrives.
- Each run takes ≤ 5 minutes.

**Measures,** at about 30 log-spaced points:
- Per rule, on up to 256 held-out instances:
  - **acc:** the fraction whose check entries are *all* predicted right;
  - **bits:** −log2 p(true value) of those entries;
  - **seen:** the instances of the rule delivered so far, i.e. the refutation opportunities.
- Overall, on 2,048 random held-out transitions: bits per transition, and accuracy on the entries that changed versus
  those that did not.
- **The headline number per rule: "seen to 90 %"**, the instances delivered before held-out acc first reaches 0.9.

## Arms (each run once)

| arm | position | learning rule |
|---|---|---|
| **A** | absolute embeddings | Adam, lr 5e-4 (the baseline) |
| **B** | **relative only** (a learned attention bias per relative offset, no absolute position): translation-equivariant, so a rule seen at one place applies everywhere | Adam |
| **C** | the better of A/B | **RLS readout:** the trunk by Adam; the readout by recursive least squares, i.e. a Kalman filter on the readout (η = 1, decay 0.99, λ = 1) |
| **D** | the better of A/B | **focal:** each entry's cross-entropy weighted by (1 − p_true)², i.e. attend to refutations |

**What the RLS readout does.** The step in each feature direction is scaled by the inverse of how much that direction
has been seen.
- A refutation along a **novel** direction is corrected at once.
- **Familiar** directions get small, averaging steps, so they are protected from interference.

This is the readout-level form of the animal "why": blame is shared among the active features by how uncertain each is,
as in the Kalman-filter models of conditioning, which reproduce blocking. Ollivier (2018) showed that the online
natural gradient *is* an extended Kalman filter, so this is the principled form of the rule.

## Predictions

| # | prediction |
|---|---|
| P1 | **The baseline (A) needs ≥ 100 instances to learn a rule:** the median "seen to 90 %" over the rules it learns. It does not learn button_off, rescue or push_into_pit to 90 % within the stream |
| P2 | **The translation-equivariant prior (B) cuts "seen to 90 %" by ≥ 2×,** median over the rules both learn |
| P3 | **The RLS readout (C) cuts "seen to 90 %" by ≥ 2× against Adam** with the same position scheme, without lowering the final overall changed-entry accuracy by more than 0.02 |
| P4 | **Focal (D) helps the rare rules** (lower "seen to 90 %" on push, button_off, interact_full) **but costs false alarms:** unchanged-entry accuracy below A's |

## Results

(appended below)

### Arm A (absolute, Adam), result

- **After the whole stream** (614k transitions, 2,400 online updates, 151 s), the model has learned to **copy**:
  - unchanged entries 0.998;
  - changed entries **0.214**;
  - 58 bits per transition.
- **Rules at 90 %:** only `mined` (after 22,696 instances) and `frozen_tick` (12,357).
- **Not learned at all:** `move` (268k instances, acc 0.00), `theft` 0.00, `push` 0.00, `button_off` 0.00.
- **Checked:** the data is correct. In 2,000 move rows, the robot's displacement always matches the action, and the
  check entry is the robot's new cell.

### Arm B (relative only, Adam), result

- **Better, but the same failure:**
  - 42.0 bits per transition;
  - changed entries 0.266;
  - mine_work 0.62 (A: 0.07), deliver 1.00 (A: 0.51), mined at 90 % after 18,947 instances (A: 22,696).
  - theft 0.11; `move` still 0.13 at the end.
- **The diagnosis, from the `move` curve:** its error *rises* during training, from 3.7 bits to 6–7 bits, as the model
  grows confident that nothing changes. Only after ~2,000 updates does it start to fall (1.7 bits).
  - Each transition has 2–4 refuted entries among ~730 confirmed ones. The plain loss weighs them equally, so the
    refutations are drowned out about 200:1.
  - This is the user's principle seen from the other side: **the information is in the refutations, and the plain
    loss does not attend to them.**

### Arm E, pre-registered (added 2026-10-20, after A and B)

**The persistence prior** (`--persist 5`, relative position, Adam). Each field's logits get +β_f on the field's current
value, with β_f learnable and starting at 5.
- The model starts out predicting "nothing changes" (p ≈ 0.95 for the current value).
- The confirmed entries then give almost no gradient; the refutations dominate it.
- It is a general prior (inertia: things stay as they are unless acted on), true of any physical state, not a rule of
  Village. It plays the part of the built-in machinery that evolution supplies.

**Prediction P5:** against B,
- changed-entry accuracy at the end ≥ 0.6 (B: 0.27);
- `move` reaches 90 % within the first 10 % of the stream;
- the median "seen to 90 %" over the rules both learn falls ≥ 5×.

### Arm H, pre-registered: the open-questions buffer (replay of unexplained refutations, with an epiplexity filter)

**What it holds:** transitions whose total loss is high, i.e. the model could not predict them. Up to 4,096.
**What happens each update:** the new batch of 256 is mixed with 128 rows replayed from the buffer.
**Leaving the buffer:**
- a row leaves when it is **explained** (its loss falls below the batch median);
- or when **replay stops helping**: no drop of ≥ 10 % over its last 3 replays. Learnable structure keeps falling, so it
  stays; noise (e.g. a villager's random step) never falls, so it is dropped.

This is "work on what you cannot yet explain, and give up on what cannot be explained", decided by measured learning
progress (the epiplexity idea) rather than by a designed schedule.

**Prediction P6:** with the persistence prior, H lowers the median "seen to 90 %" on the rare rules (theft, push,
interact_full, button_off) by ≥ 2× against E, at ≤ 1.5× E's compute.

### Arm C (relative position, RLS readout), result: P3 refuted as configured (it diverged)

- **Unstable from the first evaluation:**
  - 1,784 bits per transition at step 1, peaking at about 3,200; 2,973 at the end (B: 42);
  - changed entries 0.13;
  - `mined` and `frozen_tick` no longer learned.
- **The one rule it learned faster:** `interact_full`, the "nothing happens when full" exception, reached 0.86 after 380
  instances. B never learned it.
- **Cause.** Recursive least squares fits *low-variance* feature directions with large weights. The trunk's features
  drift under Adam into exactly those directions, and the predictions blow up. A Kalman filter needs a model of how its
  inputs change (process noise); a 0.99 decay is not one.
- **The principle may survive** with heavy damping (λ of the order of the tokens per batch) or frozen features. As
  configured, it fails.

### Arm D (relative position, focal), result: P4 half-confirmed

- **Helped the frequent changes:**
  - `move` 0.32 (B: 0.00), `turn` 0.41 (B: 0.31), `trapped_hurt` 0.18 (B: 0.11);
  - early changed-entry accuracy rose faster: 0.43 at update 543, against B's 0.17.
- **Not the rare rules:** theft 0.03, interact_full 0.32, push 0.00.
- **Cost:** 75 bits per transition (B: 42), from overconfident wrong guesses. Final changed accuracy 0.26, the same as B.

### Arm E (relative position, Adam, persistence prior), result: P5 refuted

- **Modest gains:**
  - `turn` 0.63, `mine_work` 0.73, `move` 0.38, `trapped_hurt` 0.22;
  - `frozen_tick` reached 90 % after 5,589 instances (B: 12,357);
  - 40.4 bits per transition.
- **But not the predicted jump:** changed-entry accuracy 0.28 (predicted ≥ 0.6); `move` 0.38, not 90 %.
- **The curves show a common bottleneck across arms A–E:** nothing is learned at all for the first ~300 updates
  (changed-entry accuracy stays at 0), and the whole stream is only 2,400 updates.

### Arm F, pre-registered: small batches (the critical-batch argument)

**The reasoning, from E1's information view.** A dense supervised loss has a small gradient noise scale: about 10 for
the value loss on the tiny CPU model, against about 4,000 for the policy gradient. A batch far above that scale
averages samples away instead of using them. The same stream in **batches of 32** gives 19,200 updates instead of
2,400, from the same data and FLOPs.

**Configuration:** relative position, persistence prior 5, Adam lr 1e-3 with 200 warm-up updates, batch 32.

**Prediction P7:** against E,
- changed-entry accuracy at the end ≥ 0.6;
- `move` and `theft` reach 90 %;
- the median "seen to 90 %" over the rules both learn falls ≥ 3×.

### Arm F (batch 32), result: P7 refuted

- **Each update costs 33 ms, mostly fixed overhead,** so in 285 s F saw only 240k transitions, 39 % of the stream.
- **At equal data it is a little better than E:** 57 against 70 bits per transition at ~184k transitions.
- **But no rule moved.** Changed-entry accuracy 0.18.
- **So update count was not the bottleneck,** or not the main one.

### Arm G, pre-registered: the locality prior

**Diagnosis.** To predict that the robot appears in a cell, the cell must find its neighbour among 122 tokens.
Attention starts uniform, so the neighbour's signal is diluted ~1/122, and *where to look* has to be learned before any
rule can be. That fits the flat start of every curve.

**The prior.** Each cell attends only to its 3 × 3 neighbourhood plus the global token; the global token still
attends to everything. Depth widens the reach to radius 4 after 4 layers. Physics is local: like persistence and
translation invariance, this is a general prior of physical worlds, the kind evolution builds in (local receptive
fields), not a rule of Village.

**Configuration:** E's setup (relative position, persistence prior 5, Adam lr 5e-4, batch 256) with the locality
window. **Only the locality changes.**

**Prediction P8:** against E,
- changed-entry accuracy at the end ≥ 0.6;
- `move`, `turn`, `theft` and `push` reach 90 %;
- the median "seen to 90 %" over the rules both learn falls ≥ 3×.

### Arm G (locality prior), result: the best so far, but P8 refuted

| rule | E | G |
|---|---|---|
| move (final acc) | 0.38 | 0.66 |
| turn | 0.63 | 0.79 |
| mine_work | 0.73 | 0.98 (90 % after 47,208) |
| mined (seen to 90 %) | 18,947 | 8,703 |
| deliver (seen to 90 %) | not reached | 11,295 |
| frozen_tick (seen to 90 %) | 5,589 | 2,524 |
| villager_death (seen to 90 %) | 1,309 | 1,031 |

- Changed-entry accuracy 0.36 (predicted ≥ 0.6); 36.3 bits per transition.
- theft 0.09; push, robot_burn, button_off 0.00.
- **The trend across A → B → E → G:** each general prior (translation invariance, persistence, locality) roughly
  halves the instances a rule needs. But rules still need ~10⁴ instances; animals need 1–10.

### Exploratory G2 (G with lr 2e-3, warm-up 100): no gain

- Mixed per rule: move 0.77, but turn 0.45 and trapped_hurt 0.05.
- 38.0 bits per transition, against G's 36.3.
- The learning rate is not the bottleneck.

**The per-field error breakdown** (bits per held-out transition, at the end of G2):

| field | bits |
|---|---|
| seen (the overseer's vision) | **15.4** |
| occupant | 9.1 |
| health | 6.4 |
| occupant_id | 5.7 |
| facing | 1.0 |
| amount | 0.24 |
| terrain | 0.15 |
| trapped | 0.02 |
| owner | 0.05 |

- **The gradient is dominated by the world's own dynamics,** much of it unpredictable: villagers' random steps,
  duplicated across occupant, id and health, and the overseer's hidden route moving its vision area.
- **The consequences of the robot's own actions** (theft, push) are a tiny share of the loss.
- **The rules are not only diluted by confirmations** (fixed by the persistence prior) **but drowned by noise.** That is
  the case the open-questions buffer's epiplexity filter was designed for (arm H, running).

### Arm I, pre-registered: the noise diagnostic (loss only near the robot)

**Configuration:** G's setup, but the loss counts only the cells within Chebyshev radius 2 of the robot, plus the
global token.
- Every rule the robot's actions trigger is local to it: move, turn, mining, theft, push, burn, button, rescue.
- The excluded entries are distant villagers' steps and the overseer's vision.

**This is a diagnostic, not a method.** The scope is chosen by hand. If it works, the method to look for is a general
way to attribute changes to one's own action: the efference-copy idea.

**Prediction P9:** against G,
- the robot rules (move, turn, theft, push, robot_burn) reach 90 % with ≥ 3× fewer instances, or are learned where G
  failed;
- the held-out bits per transition on the full map get *worse*: the far cells are no longer trained.

### Arm H (locality + open-questions buffer), result: P6 refuted, mixed

- **Helped some robot rules:**
  - theft 0.24 (G: 0.09);
  - robot_burn 0.21 (G: 0.00);
  - trapped_hurt 0.60 (G: 0.36);
  - deliver at 90 % after 6,795 instances (G: 11,295);
  - frozen_tick after 2,107 (G: 2,524).
- **Hurt the most frequent rule:** move 0.20 (G: 0.66).
- **No rare rule learned.**
- **The buffer mostly churned noise:** 153,600 rows admitted, 87,742 evicted as "stuck", only 51 as "explained". A
  row's loss includes the villagers' random steps, so "below the batch median" is rarely reached.

### Arm I (G + loss only near the robot, radius 2), result: P9 confirmed for the main robot rules

| rule | G | I |
|---|---|---|
| move | 0.66 | **1.00** (90 % after 133,546) |
| turn | 0.79 | **0.98** (35,331) |
| mined (seen to 90 %) | 8,703 | **3,925** |
| theft | 0.09 | **0.51** |
| interact_full | 0.61 | 0.74 |

- Unchanged: push, robot_burn, button_off at 0.00.
- Held-out bits on the full map: 35.0, not worse than G's 36.3 (the predicted cost did not appear).
- **Noise from the world's own dynamics was a large part of the bottleneck.**

**What the curves show, generally.** A rule is flat for a long time, then rises:
- move: 0 until ~36k instances (~320 updates), then 0.96 by 173k (~1,600 updates);
- theft: 0 until ~1.8k instances, 0.46 at 4k.

A rule takes on the order of **1,000 SGD updates**, whatever the batch size. So "instances needed" is mostly "the
instances that arrive during those updates". The animal way to decouple the two is **replay**: a rare event, stored
once, is rehearsed many times (consolidation).

### Arm J, pre-registered: I + the open-questions buffer, with a patient filter

**Configuration:** I's setup (locality window, persistence prior, loss within radius 2 of the robot) plus the buffer.
- Rows' surprise is now robot-local, so the buffer holds robot-rule refutations rather than villager noise.
- Patience 10 replays, minimum drop 5 %: SGD lowers one row's loss slowly, so the filter must wait longer.
- 128 replayed rows per batch of 256.

**Prediction P10:** against I, at least 2 of {theft, push, robot_burn, interact_full} reach 90 %, and `button_off`
(191 instances) rises above 0.

### Arm J, result: P10 refuted strictly, but the best arm so far, and a deviation found

| rule | I | J |
|---|---|---|
| theft | 0.51 | **0.74** |
| interact_full | 0.74 | **0.83** |
| push_into_pit | 0.00 | **0.25**, from only 17 instances |
| trapped_hurt | 0.05 | 0.42 |
| mine_work (seen to 90 %) | 30,280 | 18,133 |
| deliver (seen to 90 %) | 14,728 | 11,295 |
| robot_burn | 0.00 | 0.09 |
| changed-entry accuracy | 0.35 | **0.42** |
| bits per transition | 35.0 | **34.6** |

- None of the four target rules reached 90 %. `button_off` stayed at 0.

**Deviation.** The buffer's surprise score still summed over *all* entries, villager noise included, not only the
robot-local ones as registered. That is why only 69 rows were ever "explained". Fixed in the code; the corrected run is
J2.

### Arm J2 (J with the surprise scored inside the focus), result

- **Best changed-entry accuracy so far: 0.446.** robot_burn 0.34 (J: 0.09), mine_work at 90 % after 13,938.
- Slightly lower: theft 0.68 (J: 0.74), interact_full 0.56 (J: 0.83).
- **J's `push_into_pit` 0.25 was noise:** it is 1 of only 4 held-out instances. Rules with fewer than ~30 held-out
  instances (push_into_pit 4, rescue 6, button_off 28) are not reliable at the per-rule level.

### Arm K, pre-registered: few-shot capacity under unlimited rehearsal (`e2_fewshot.py`)

**The question.** Across arms A–J2, a rule takes on the order of 1,000 SGD updates however it is fed. So how many
*distinct* refutations does the network need, if it may rehearse them as often as it likes?
- If **a handful** suffices, the bottleneck is rehearsal, and the learning rule should be "store refutations, rehearse
  them".
- If not, the features lack the right abstraction, and the fix is elsewhere.

**Setup.**
- Start from J2's final model, which never learned **push** (2,778 training instances), **robot_burn** (2,449) or
  **button_off** (191).
- For k ∈ {1, 4, 16, 64, 256} distinct training instances: fine-tune 600 updates. Each batch is 8 rows drawn from the
  k plus 24 from the whole stream, so the other rules are kept. Adam 5e-4, the same robot-local loss.
- Score the rule on held-out layouts every 50 updates, and the other rules too (forgetting).

**Prediction P11:**
- push and robot_burn reach 90 % held-out from ≤ 64 distinct instances;
- button_off from ≤ 16 (only 28 held-out instances, so a coarse measure);
- other rules lose ≤ 0.05.

### Arm K, result: P11 partly confirmed; a few-shot capacity exists, and two failure modes are measured

**Held-out accuracy of the rule** after fine-tuning from J2's model; columns are the update count (0 → 600, every 50):

| rule | k = 1 | k = 4 | k = 16 | k = 64 | k = 256 (191) |
|---|---|---|---|---|---|
| **button_off** | **0.86 at 50**, then **decays to 0.18** | **0.93 at 150**, ~0.7–0.8 later | 0.93–0.96 | 0.93–1.00 | 0.96–1.00 |
| robot_burn | ≤ 0.3 | ≤ 0.26 | ~0.4 | plateau ~0.65 | **0.91 at 150**, 0.88–0.97 |
| push | ≤ 0.08 | ≤ 0.23 | ~0.45 | ~0.5 | peak 0.71 |

1. **A few-shot capacity exists when the features exist.**
   - `button_off` is learned from **one** refutation (0.86 held-out after 50 rehearsals) and from four (0.93).
   - Its parts (the faced cell, Interact, a terrain change) were already learned for mining and theft.
   - Rules that need features the network lacks take far more: push (two cells away) and burn (the global token must
     learn the robot stepped onto fire).
2. **Over-rehearsal of a single refutation is rationalization, measured.** With k = 1, held-out accuracy peaks at 0.86
   and then falls to 0.18 as rehearsal continues. The model's explanation drifts from the general cause to
   idiosyncrasies of that one layout. **Stop rehearsing when explained.**
3. **Rehearsing new refutations interferes with old rules.**
   - theft 0.68 → between 0.09 and 0.71 depending on the run;
   - changed-entry accuracy 0.45 → ~0.35;
   - 24 random stream rows per batch do not protect a rule that is rare in the stream.

   The remedy, from complementary-learning-systems theory: **keep rehearsing what is already explained**, interleaved
   at a low rate.

P11: push and robot_burn from ≤ 64 instances refuted (burn needs ~256, push > 256); button_off from ≤ 16 confirmed
(from 1–4); "other rules lose ≤ 0.05" refuted.

### Arm L, pre-registered: rumination (open questions → retired explanations, interleaved)

The learning rule that K's three findings call for. On I's setup (locality, persistence, robot-local loss):
1. **Open questions:** rows whose in-scope surprise is in the newest batch's top 10 %.
   - **Rehearsal is prioritised by learning progress:** the loss drop over the item's last 3 replays, with new items
     first (optimistic).
2. **Retire when explained:** in-scope surprise below the newest batch's median. Retired items stay in a long-term
   pool, rehearsed rarely (64 rows per update, uniform).
   - A retired item that becomes surprising again (top 10 % of the batch) is **reopened**: it was forgotten.
3. **Drop when stuck:** no drop of ≥ 5 % over its last 10 replays. Noise cannot be learned.

Per update: 256 new rows + 128 open + 64 retired. Capacities: 4,096 open, 16,384 retired.

**Prediction P12:** against J2,
- at least 2 of {theft, robot_burn, push, interact_full} reach 90 %;
- theft ends ≥ 0.68 (no loss from interference);
- changed-entry accuracy ≥ 0.45.

### Arm L, result: P12 refuted, worse than J2; the cause is a design flaw (starvation)

| measure | J2 | L |
|---|---|---|
| theft | 0.68 | 0.41 |
| robot_burn | 0.34 | 0.05 |
| trapped_hurt | 0.40 | 0.09 |
| interact_full | 0.56 | 0.85 |
| changed-entry accuracy | 0.446 | 0.381 |
| stream covered | 100 % | 84 % (time limit) |

**The counters show why:** only 34 items ever retired; the open buffer sat full (4,096); 8,962 dropped as stuck.
- The priority served never-judged items first.
- With ~26 new admissions per update, the 128 rehearsal slots always went to items rehearsed ≤ 3 times.
- So nothing was rehearsed long enough to be explained: **starvation.**

### Arm L2, pre-registered (exploratory fix of L's flaw)

**Change from L:**
- Half the open slots go to the items whose loss falls fastest (among those rehearsed > 3 times); half to the
  least-rehearsed.
- An item also retires when its own surprise has **halved since admission**. This criterion does not depend on the
  batch, which carries villager noise.

**Prediction P13:** better than L on theft, robot_burn and changed-entry accuracy; at least J2's level on 2 of the 3.

### Arm L2, result: P13 partly confirmed

| measure | J2 | L | L2 |
|---|---|---|---|
| changed-entry accuracy | 0.446 | 0.381 | **0.448** |
| interact_full | 0.56 | 0.85 | **0.95** (best) |
| move (seen to 90 %) | 133,546 | 173,210 | **102,253** (best) |
| theft | 0.68 | 0.41 | 0.35 |
| robot_burn | 0.34 | 0.05 | 0.14 |
| trapped_hurt | 0.40 | 0.09 | 0.04 |

**The counters:** 1,707 retired, but 1,489 **reopened**. A row's in-scope surprise is dominated by villagers stepping
near the robot, so "explained" and "forgotten" mostly track noise. **Per-row bookkeeping cannot tell a learnable
refutation from noise.**

### Arm L3, pre-registered: rehearsal allocated across surprise KINDS by learning progress (an epiplexity bandit)

**What K showed:** a rule whose features exist is learned in ~50 updates if its few instances get *concentrated*
rehearsal. In the stream they must be told apart from noise by their **kind**, not by row loss.

**The kind of a refutation** (label-free) is the field of the row's most surprising in-scope entry, plus that entry's
offset from the robot, clipped to ±2; or the global field. That makes 9 × 25 + 4 = 229 kinds.
- button_off is always "terrain, at the faced cell";
- villager noise is spread over "occupant / id / health, at many offsets".

**The rule:**
- each kind's learning progress is the EMA of the relative loss drop per rehearsal of its items;
- the 128 open slots per update are split across kinds in proportion to progress, plus an exploration bonus for kinds
  tried < 5 times;
- within a kind, the least-rehearsed items go first;
- retirement, reopening and the retired pool are as in L2.

**Prediction P14:** against J2,
- theft ≥ 0.68 and robot_burn ≥ 0.34 (no loss);
- at least one of {push, button_off} rises above 0.3;
- the most-progressing kinds logged at the end include robot-rule kinds (amount / terrain / carried / health), not
  villager-noise kinds.

### Arm L3, result: P14 refuted; the bandit attends to robot kinds, but frequent ones

- **Roughly J2's level, not better:**
  - theft 0.53, robot_burn 0.27, trapped_hurt 0.31;
  - deliver at 90 % after 8,726 instances (J2: 14,728);
  - changed-entry accuracy 0.432 (J2: 0.446);
  - push and button_off 0.00;
  - it covered 84 % of the stream in the time cap.
- **The most-progressing kinds at the end:**
  - time (global);
  - the robot's facing field at its new cell, three offsets (that is, move and turn);
  - the occupant id at the robot's own cell.

  These are robot kinds, not villager noise, as predicted, but **frequent** ones. Rare kinds got too little rehearsal
  to show progress.

### Arm K2, pre-registered: the learning rule in its clean form (`e2_fewshot2.py`)

**The question:** can J2's model learn **button_off from one refutation** without rationalizing and without forgetting,
using the two fixes K pointed to?
1. **Stop rehearsing when explained:** the instance retires once its in-scope surprise has halved, then joins the
   interleaved pool.
2. **Interleave past refutations,** sampled evenly across surprise kinds (label-free), not random stream rows.

**Arms** (600 updates, batch 32: 8 rehearsal + 24 interleaved):

| arm | k | interleaved rows | rehearsal |
|---|---|---|---|
| K2a | 1 | kinds | retires when explained |
| K2b | 1 | stream (K's) | retires when explained |
| K2c | 1 | kinds | `--always` (K's over-rehearsal) |
| K2d | 4 | kinds | retires when explained |

**Prediction P15:** K2a keeps button_off ≥ 0.8 to the end (K's k = 1 fell to 0.18) and keeps theft within 0.1 of its
starting 0.68. K2b forgets theft more than K2a. K2c decays as in K.

### Arm K2, result: P15 refuted on stability; the parts work

Held-out accuracy every 50 updates (0 → 600). button_off has only 28 held-out instances (steps of 0.036); theft 256.

| arm | button_off | theft (every 150 updates) | changed-entry accuracy |
|---|---|---|---|
| K2a (k = 1, kinds, retire) | **0.89 at 50**, then 0.54–0.79 | 0.68 → 0.59, 0.31, 0.14, 0.47 | 0.446 → 0.401 |
| K2b (k = 1, stream, retire) | 0.79 at 50, then 0.32–0.89 | → 0.57, 0.84, 0.65, **0.16** | 0.446 → **0.282** |
| K2c (k = 1, kinds, always) | 0.96 at 50, then 0.46–0.93 | → **0.01**, 0.47, 0.50, 0.36 | 0.446 → 0.412 |
| K2d (k = 4, kinds, retire) | **0.86–0.93 throughout** (one dip to 0.50) | → 0.49, 0.25, 0.53, 0.75 | 0.446 → 0.374 |

- **One refutation is enough to learn the rule** (0.79–0.96 after 50 updates in every arm). Four instances keep it.
- **Interleaving past refutations by kind protects the old rules better than stream rows:** changed accuracy 0.40
  against 0.28.
- **Everything is unstable.** Theft swings between 0.01 and 0.84 between evaluations. The step size (Adam 5e-4, batch
  32) is a large perturbation for a trained model, and a half-learned rule (theft at 0.68) is fragile.

### Arm K3 (exploratory): K2 with gentler steps, lr 1e-4

**K3 result: the learning rule works in its clean form.**

| arm | button_off (0 → 600, every 50) | theft (every 150) | changed-entry accuracy |
|---|---|---|---|
| K3a (k = 1, kinds) | 0.75 at 50, **0.86 stable at the end** | 0.68 → 0.72, 0.88, 0.84, **0.85** | 0.446 → 0.386 |
| K3b (k = 1, stream) | 0.64–0.79 | 0.71–0.78 | 0.446 → 0.357 |
| K3d (k = 4, kinds) | 0.89 at 50, **0.93 at the end** | → 0.81 | 0.446 → **0.440** |

- **From one refutation,** button_off generalizes to unseen layouts (0.86) and stays there for 600 updates: no
  rationalizing drift once the instance retires.
- **Theft improves rather than being forgotten,** because the kind-balanced rehearsal of past refutations includes
  theft-like ones. With stream rows instead, theft only holds.
- **With four instances, the rest of the model is almost untouched.**

**The rule:**
1. detect the refutation;
2. rehearse it until its surprise halves, then stop;
3. interleave past refutations sampled evenly by kind;
4. use gentle steps.

It needs the features to exist already; push and burn did not have them (arm K).

### Arm M, pre-registered: wake–sleep rumination (`e2_sleep.py`), K3's rule inside the online stream

**Why the stream versions (H, J, L, L3) fell short:** rehearsal competed with new data at the same, large learning
rate.

**Wake:** learn from the stream as in I (locality, persistence, robot-local loss, Adam 5e-4). Keep the rows in each
batch's top 10 % of in-scope surprise as open refutations, with their kind and arrival surprise.

**Sleep:** every 200 wake updates, 100 gentle updates (lr 1e-4, batch 32), each with:
- 8 open refutations, kinds chosen by learning progress (L3's bandit) plus exploration;
- 24 retired refutations, sampled evenly across kinds.

An open refutation retires when its surprise has halved, and is dropped after 10 rehearsals without halving.

**Prediction P16:** against I (the same wake learning without sleep),
- at least 2 of {theft, robot_burn, interact_full, button_off} improve by ≥ 0.15;
- no rule that I learned to 90 % drops below 0.85.

### Arm M, result: P16 refuted

| measure | I (wake only) | M |
|---|---|---|
| changed-entry accuracy | 0.347 | **0.570** |
| unchanged-entry accuracy | 0.998 | **0.989** |
| bits per transition | 35.0 | **44.1** |
| deliver | 0.99 | **0.42** |
| theft | 0.51 | 0.53 |
| interact_full | 0.74 | 0.50 |
| robot_burn, button_off | 0.00 | 0.00 |

- **The high changed-entry accuracy is false alarms.** Unchanged entries got worse and bits per transition rose. Sleep
  rehearses only rows where something changed, so the model learned to over-predict change.
- **deliver was forgotten** (0.99 → 0.42).
- **The bandit was fooled.** Its "most-progressing" kinds at the end were mostly villager-id changes near the robot.
  A noisy row's surprise fluctuates, so one rehearsal often looks like progress, and "halved" happens by chance:
  the "noisy, lagging estimates" pitfall of Brainstorm 2, measured.

### Arm M2, pre-registered: prequential progress per kind, and stream rows in sleep

**Change from M:**
- **A kind's learning progress is now prequential:** the drop in surprise of **new arrivals** of that kind. That is a
  slow average (α = 0.02) minus a fast one (α = 0.2), relative to the slow one: the epiplexity estimator. A noise kind
  never improves on new arrivals.
- **Each sleep batch is 8 open + 12 retired + 12 stream rows,** to keep the base rate of "nothing changes".

**Prediction P17:** against I,
- at least 2 of {theft, robot_burn, interact_full, button_off} improve by ≥ 0.15;
- no rule that I learned to 90 % drops below 0.85;
- unchanged-entry accuracy ≥ 0.995.

### Arm M2, result: P17 refuted; the progress estimator was mis-specified

| measure | I | M2 |
|---|---|---|
| theft | 0.51 | 0.23 |
| interact_full | 0.74 | 0.44 |
| turn | 0.98 | 0.80 |
| unchanged-entry accuracy | 0.998 | 0.992 |
| bits per transition | 35.0 | 41.0 |

**The most-progressing kinds were terrain changes two cells from the robot,** where nothing ever changes. Two errors:
1. **Every row got a kind,** including rows that were not surprising at all. Their "kind" was their least-confident
   entry, and those tiny surprises shrink steadily as the persistence prior sharpens.
2. **Progress was normalized** by the kind's own surprise, so a drop from 0.05 to 0.005 nats counted as 90 %.

Epiplexity is an absolute quantity (bits removed).

### Arm M3, pre-registered: M2 with absolute progress (nats), observed on refutations only

Only the rows in each batch's top 10 % of surprise count as arrivals of their kind.

**Prediction P17** (unchanged, applied to M3):
- at least 2 of {theft, robot_burn, interact_full, button_off} improve by ≥ 0.15 against I;
- no rule I learned to 90 % drops below 0.85;
- unchanged-entry accuracy ≥ 0.995.

### Arm M3, result: P17 refuted; the bandit finds the right kinds, but sleep's gains do not survive waking

- **Outcome:**
  - theft 0.32, interact_full 0.50, robot_burn 0.00, button_off 0.00;
  - unchanged-entry accuracy 0.990, bits 40.1;
  - changed-entry accuracy 0.573, measured just after a sleep phase. During waking (update 2,015) it was 0.351.
- **The most-progressing kinds are now exactly the robot's rules:**
  - the **amount** field at the faced cell, east and west (mining and theft);
  - the **terrain** field at the faced cell (the button);
  - the **trapped** flag next to the robot (pits).

  The epiplexity estimator works once it is absolute and fed only refutations.
- **Two likely reasons it does not show in the outcome:**
  1. Sleep's gentle consolidation (lr 1e-4) is overwritten by the next 200 wake updates at 5× the rate.
  2. Sleep batches are about two-thirds "change" rows, so they bias toward false alarms.

  The brain's version keeps the cortex slow at all times and lets a fast memory hold the refutations
  (complementary learning systems).

### Arm K4, pre-registered: label-free sleep on a trained model, with a test that can lose (`e2_sleep2.py`)

**Start from J2's model. The open refutations** are the 10 % most surprising of 100k random stream rows, each with its
kind. No event labels are used.
- **Each kind's open rows are split:** a *rehearse* half and a *probe* half.
- **Sleep:** 1,000 updates at lr 1e-4, batch 32 = 8 rehearse rows + 12 past refutations (even across kinds) + 12
  stream rows.
- **A kind's progress** is the surprise drop of its unrehearsed probe half (EMA, every 25 updates). Learning a general
  rule transfers to the probes; rationalizing (memorizing the rehearsed rows) and noise do not.
- Rehearsal goes to kinds in proportion to that progress. A rehearsed row retires once its surprise halves.

**Control K4u:** kinds chosen uniformly.

**Prediction P18:**
- K4 raises ≥ 2 of {theft, interact_full, robot_burn, button_off, push} by ≥ 0.15 over J2's values (0.68, 0.56, 0.34,
  0.00, 0.00);
- move, turn, mined and deliver stay ≥ 0.9; unchanged-entry accuracy stays ≥ 0.995;
- K4 beats K4u on the sum of those five.

### Arm M4, result: P19 refuted; the decay was not the cause

- theft 0.30, interact_full 0.75, robot_burn 0.00, button_off 0.00, turn 0.84, deliver 0.92.
- Unchanged-entry accuracy 0.990.

**The clearest difference from K4, which worked, is the open set:**
- **K4** scores a large sample with the current model and takes the top 10 %: a fresh, representative set of "what I
  still cannot explain".
- **M–M4 accumulate refutations batch by batch** and, when full, keep the most surprising. Those are permanently
  surprising noise (villagers stepping near the robot); learnable refutations are evicted as soon as they start to be
  explained.

### Arm M5, pre-registered: wake–sleep with the open set re-scored at each sleep onset (`e2_wakesleep.py`)

K4's procedure, run every 200 wake updates:
1. score 20,000 rows sampled from everything seen so far with the current model;
2. the top 10 % are the open refutations, grouped by kind;
3. 100 gentle updates (lr 1e-4) of 8 open (uniform kinds) + 12 retired (even across kinds, kept across sleeps) + 12
   stream rows;
4. an open row retires when its surprise halves.

Wake is as in I (lr 5e-4, constant).

**Prediction P20** (as P19): against J2,
- at least 2 of {theft, interact_full, robot_burn, button_off} end ≥ 0.15 higher;
- move, turn, mined and deliver ≥ 0.9; unchanged-entry accuracy ≥ 0.995.

### Arm M5, result: P20 refuted (1 of 4); push learned for the first time

| rule | J2 | M5 |
|---|---|---|
| **push** | 0.00 | **0.51** (every earlier arm 0.00–0.08) |
| interact_full | 0.56 | 0.83 |
| theft | 0.68 | 0.59 |
| robot_burn | 0.34 | 0.01 |
| button_off | 0.00 | 0.00 |
| mine_work | 0.98 | 0.80 |
| changed / unchanged accuracy | 0.446 / 0.996 | **0.466 / 0.995** |

Only 54 rows ever retired: in the stream, villager noise near the robot sets a floor under each row's surprise.

### M5b: M5 re-run with evaluations before and after every second sleep

The training is identical; this re-run only adds the missing measurement.

| update | changed acc (pre → post) | unchanged acc (pre → post) | theft (pre → post) | interact_full (pre → post) |
|---|---|---|---|---|
| 400 | 0.230 → 0.236 | 0.997 → 0.997 | 0.00 → 0.00 | 0.02 → 0.26 |
| 800 | 0.211 → 0.393 | 0.998 → 0.993 | 0.08 → 0.07 | 0.08 → 0.44 |
| 1200 | 0.335 → 0.455 | 0.998 → 0.994 | 0.30 → **0.14** | 0.02 → 0.17 |
| 1600 | 0.365 → 0.448 | 0.997 → 0.994 | 0.31 → **0.20** | 0.17 → 0.02 |
| 2000 | 0.359 → 0.511 | 0.998 → 0.992 | 0.38 → **0.22** | 0.06 → 0.32 |
| 2400 | 0.350 → 0.466 | 0.998 → 0.995 | 0.57 → 0.59 | 0.96 → 0.83 |

- **Every sleep raises changed-entry accuracy**, partly by false alarms (unchanged falls ~0.004).
- **The next waking stretch erodes it** back to ~0.35.
- **Theft drops after sleeps** and recovers while awake. Sleep and wake fight.
- **The likely cause:** uniform-kind sleep rehearses many *noise* kinds (villagers stepping near the robot), teaching
  the model to predict the unpredictable. On K4's mature model that did not matter; in the stream, **choosing
  learnable kinds matters.**

### Arm M6, pre-registered: M5 + kinds chosen by generalization to their unrehearsed half

**At each sleep onset,** each kind with ≥ 4 open rows is split into a rehearse half and a probe half (≤ 32 rows).
**After the sleep,** the kind's progress is the drop in its probe half's surprise, carried across sleeps as an EMA
(0.5).
- Rehearsal goes to kinds in proportion to max(progress, 0) + 0.05.
- Never-measured kinds get weight 1.
- Noise kinds never generalize, so they lose their share.

**Prediction P21:** against M5,
- theft does not drop after sleeps (pre → post ≥ −0.05);
- at the end, at least 2 of {theft, interact_full, robot_burn, button_off, push} ≥ M5's value + 0.1;
- unchanged-entry accuracy ≥ 0.995.

### Arm M6, result: P21 refuted

| update | theft (pre → post) | changed acc (pre → post) |
|---|---|---|
| 1600 | 0.20 → 0.07 | 0.335 → 0.455 |
| 2000 | 0.38 → 0.18 | 0.352 → 0.479 |

- **Theft still drops after sleeps** (P21's first condition fails).
- **The run hit the 5-minute cap just before its last sleep,** so its final numbers are pre-sleep and not comparable with
  M5's post-sleep ones. As measured:
  - theft 0.42, interact_full 0.15, push 0.00;
  - unchanged 0.998, bits 35.5.
- **Choosing kinds by generalization did not stop sleep and wake fighting in the stream.**

**The in-stream sleep series ends here:** six versions (M, M2–M6), a consistent diagnosis, no win over J2. The version
that works is **sleep after waking (K4)**.

### Robustness checks of the two headline claims (pre-registered, 2026-10-20 afternoon)

**K3a-s1, K3a-s2:** K3a (one refutation, kinds pool, retirement, lr 1e-4) with seeds 1 and 2. The seed changes which
single button_off instance is used, and the interleaving draws.
- **Prediction P22:** both end with button_off ≥ 0.75 held-out and theft ≥ 0.6.

**K5: the recipe from scratch.** Wake = arm I's model (plain online learning with the priors, no buffer, 179 s), then
K4's sleep with uniform kinds (1,000 updates).
- **Prediction P23:** against J2 (the best stream-only arm, 273 s),
  - button_off ≥ 0.7, theft ≥ 0.68;
  - move, turn, mined and deliver ≥ 0.9; unchanged-entry accuracy ≥ 0.995.

**Results.**

**P22 refuted for one seed: one-shot generalization depends on which case is seen.**

| run | button_off held-out | theft at the end |
|---|---|---|
| seed 0 (K3a) | 0.75 → 0.86 | 0.85 |
| seed 1 | 0.43 at 50 → **0.75** | 0.80 |
| seed 2 | fluctuates 0.29–0.75, ends **0.46** | 0.89 |

Four instances (K3d) gave 0.93. Theft improves in all three single-instance runs.

**P23 mostly confirmed: K5, the recipe from scratch** (I's wake model + label-free sleep, uniform kinds; held-out
accuracy, sleep updates 0 → 1,000):

| rule | 0 | 200 | 400 | 600 | 800 | 1000 | J2 |
|---|---|---|---|---|---|---|---|
| button_off | 0.00 | 0.43 | 0.68 | 0.71 | 0.79 | **0.71** | 0.00 |
| theft | 0.51 | 0.61 | 0.72 | 0.74 | 0.81 | **0.58** | 0.68 |
| interact_full | 0.74 | 0.88 | 0.89 | 0.91 | 0.85 | **0.89** | 0.56 |
| robot_burn | 0.00 | 0.02 | 0.00 | 0.00 | 0.01 | 0.05 | 0.34 |
| trapped_hurt | 0.05 | 0.10 | 0.06 | 0.11 | 0.11 | 0.11 | 0.40 |
| move / turn / deliver | ≥ 0.93 throughout | | | | | | |
| mined | 1.00 | 0.94 | 1.00 | **0.82** | 0.97 | 0.99 | 1.00 |
| unchanged-entry accuracy | 0.998 | 0.995 | 0.996 | 0.994 | 0.996 | 0.995 | 0.996 |

- **Wake (179 s) then sleep (~60 s) adds rules that online learning never reached:** button_off and interact_full.
- **The miss:** theft ends at 0.58, after peaking at 0.81, below J2's 0.68.
- **Against J2 at similar compute** (~250 s against 273 s): it gains button_off and interact_full, and loses on theft,
  robot_burn and trapped_hurt. J2's in-stream buffer is better for those mid-frequency rules.
- **Theft's swings** (0.58–0.81 within one sleep) show the sleep itself is not yet stable for a half-learned rule.

### Arm M7 and K5e, pre-registered: a slow "cortex" read out as an EMA of the weights

**The idea.** In the stream, sleep raises the rare rules and the next waking stretch erodes them; even within one
sleep, theft swings 0.58–0.81. Complementary-learning-systems theory says the cortex integrates slowly over both phases.
- The simplest form is a slow copy of the network: weights_slow ← d · weights_slow + (1 − d) · weights_fast after
  every update.
- It is **read out, never trained.** The training is unchanged.
- d = 0.998, a horizon of ~500 updates, spanning a wake stretch plus a sleep.

| arm | configuration | what is read out |
|---|---|---|
| **M7** | M5's configuration (re-scored sleeps, uniform kinds), evaluated before and after every second sleep | the slow copy, plus the fast model for comparison |
| **K5e** | K5 (I + label-free sleep) | the slow copy |

**Prediction P24:**
- M7's slow copy has pre/post-sleep jumps in changed-entry accuracy ≤ 0.03, against M5b's 0.12–0.18;
- at the end, its theft ≥ 0.6 and interact_full ≥ 0.8;
- K5e's theft no longer swings: max − min over the last 600 sleep updates ≤ 0.1 (K5: 0.23).

**Results.**

**M7: refuted.** The slow copy removes the oscillation but lags far behind a network still forming.

| update | changed acc, pre → post | theft, pre → post |
|---|---|---|
| 1200 | 0.336 → 0.372 | 0.09 → 0.13 |
| 2000 | 0.347 → 0.388 | 0.25 → 0.24 |
| 2400 | 0.397 → 0.413 | 0.40 → 0.44 |

- Jumps ≤ 0.04: that part holds.
- At the end, slow against fast: theft 0.44 against 0.59, interact_full **0.04** against 0.83.
- The fast numbers reproduce M5 exactly, so training is deterministic.

**K5e: confirmed.** In a mature model's sleep, the slow copy is what was missing.
- **theft climbs steadily** 0.51 → 0.64 → 0.78 → 0.78 → 0.82 → **0.83** (every 200 updates). Its range over the last
  600 updates is 0.06, against 0.25 for the fast weights.
- button_off 0 → **0.64** (rising), interact_full 0.82.
- move, turn, mined and deliver ≥ 0.98; unchanged-entry accuracy 0.996.

**The slow read-out harms while the network is forming (M7) and stabilizes once it has formed (K5e):** the cortex
consolidates during sleep.

### Arm K6, pre-registered: the best wake (J2) + sleep with the slow read-out

**Configuration:** J2's model, then K4's label-free sleep (uniform kinds, 1,000 updates), read out through the slow copy
(0.998).

**Prediction P25:**
- theft ≥ 0.8, button_off ≥ 0.6, interact_full ≥ 0.6;
- robot_burn ≥ 0.3 and trapped_hurt ≥ 0.4 (J2's levels kept);
- move, turn, mined and deliver ≥ 0.95;
- theft's range over the last 600 updates ≤ 0.1.

**K6 result: P25 confirmed except interact_full.** The best world model of the session.

| rule | J2 | 200 | 400 | 600 | 800 | **1000** |
|---|---|---|---|---|---|---|
| theft | 0.68 | 0.76 | 0.82 | 0.85 | 0.85 | **0.87** |
| button_off | 0.00 | 0.00 | 0.18 | 0.57 | 0.71 | **0.79** |
| trapped_hurt | 0.40 | 0.42 | 0.45 | 0.45 | 0.47 | **0.48** |
| robot_burn | 0.34 | 0.33 | 0.33 | 0.33 | 0.34 | 0.34 |
| interact_full | 0.56 | 0.57 | 0.51 | 0.55 | 0.56 | 0.53 |
| push | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| move / turn / mined / deliver / mine_work | ≥ 0.96 | ≥ 0.97 throughout | | | | |

- Unchanged-entry accuracy 0.996–0.997. Theft's range over the last 600 updates is 0.05.
- **Every rule ends at or above J2's level** except interact_full (−0.03) and push (still 0).
- **The trend is monotone:** consolidation without forgetting.


### Measurement added 2026-10-20 (evening): does the rehearsed case stay fitted? (`runs/e2/K_fitcheck`)

**What I had claimed:** that with one rehearsed button_off instance (arm K, k = 1), held-out accuracy fell while the
case itself stayed fitted. That was an inference: the case is in every batch. It had not been measured.

**The re-run** adds the rehearsed case's own score. The held-out curve reproduces the original exactly.

| update | held-out acc | rehearsed case acc | rehearsed case bits |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 4.085 |
| 50 | 0.86 | 1.00 | 0.025 |
| 150 | 0.71 | 1.00 | 0.016 |
| 300 | 0.46 | 1.00 | 0.013 |
| 450 | 0.18 | 1.00 | 0.013 |
| 600 | 0.18 | 1.00 | 0.011 |

**Confirmed:** the case stays perfectly predicted while the explanation stops generalizing. That is rationalization,
measured.
