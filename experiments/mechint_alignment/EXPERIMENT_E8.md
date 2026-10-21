# E8: A world family with hidden physics: inferring the rules in the weights (A) or in context (B)

**Status:** pre-registered 2026-10-20 (late). Runs ≤ 10 min (the user).

## The family (`village.py`: Physics, sample_physics)

Every episode samples its own hidden physics. **None of it is in the observation:**
- **Actions:** a permutation of all 8 action indices. Which index moves north, interacts, waits… differs per world.
- **Object roles:** a permutation of the 7 object codes (depot, mine, chest, fire, pit, button, disabled button). The
  token shown for "mine" in one world is "fire" in another.
- **Numeric rules:**
  - mine effort ∈ {1, 2, 3}; carry capacity ∈ {1, 2}; fire damage ∈ {0, 1, 2, 3};
  - walking into a villager ∈ {push, block, swap};
  - freeze ∈ {5, 10}; reward per gold ∈ {0.5, 1, 2}.
- **Held out of training entirely:** (mine effort 3 with swap) and (fire damage 0 with block).

**The model never receives a rule, a rule identity or a list of rules.** Rules exist explicitly only in the simulator
and in the scoring.

**Data:** random actions only (babbling), on the full 9 × 9 world, 128-step episodes.
- 614,400 training transitions = 4,414 complete worlds.
- 128 test worlds with unseen permutations; 128 worlds with the held-out combinations.

## Option A: hierarchical Bayes (`family.py`, `e8_hier.py`)

**The model.** The E2 trunk is shared (locality window, persistence prior, robot-local loss; d = 128, 4 layers).
**Per world,** fast parameters φ_w with a learned prior mean μ:
- the action embeddings, where an action permutation can live;
- the terrain-code embeddings, where a role permutation can live;
- the output heads, where the numeric rules can live.

**Within a world,** chunk by chunk (8 steps):
1. predict the chunk with φ_w (prequential);
2. take one Kalman / natural-gradient step on it, counted once: φ_w ← φ_w − g / (λ₀ + n_seen · F̄).
   - F̄ is the per-datum Fisher (an EMA); λ₀ = 10; each element of the step is clamped to ±0.1.

**Training** maximizes the worlds' summed prequential log-likelihood, which is their log marginal likelihood (empirical
Bayes). Adam on the trunk and on μ, first-order; 32 worlds per update.

**Arms (420 s of training each):**
- **A:** with in-world adaptation, evaluated both with and without it;
- **A0:** no adaptation, one model for all worlds.

**Measures:** per chunk index, i.e. against experience within the world:
- bits per step (prequential);
- the accuracy of each rule's check entries.

In particular **move**: predicting where the robot goes requires knowing this world's action meaning.

## Predictions (A)

| # | prediction |
|---|---|
| P1 | On unseen worlds, A's late-world bits (chunks 12–15) are ≥ 5 % lower than A0's |
| P2 | Under A, **move** accuracy rises with experience in the world (late ≥ early + 0.2); under A0 it stays flat. Under shuffled actions, A0 cannot know which action moves where |
| P3 | On the held-out combinations, A's adaptation gain (late bits without adaptation − with adaptation) is ≥ half its gain on unseen worlds |

## Option B: in context (`incontext.py`, `e8_context.py`), pre-registered before A's results were seen

**The model.** All weights are shared by all worlds. For each past step j of the world:
- a transition token u_j = MLP([pool(obs_j), E_act(a_j), pool(obs_{j+1})]), where pool is a **learned** attention
  pooling with 4 queries over the action-free trunk features. Nothing points it at the robot;
- a 2-layer causal transformer over u_0 … u_{t−1} gives a belief state c_t;
- the prediction for step t is the world model on (obs_t, a_t) with a projection of c_t added to every token. The
  projection starts at zero, so the model starts as option A0.

**Training:** next-observation prediction, each world's whole 128-step history per pass (4 worlds per update), 420 s.
**Evaluation:** identical to A's (`e8_score.py`).

**Predictions:**
- **P4:** B's late-world bits on unseen worlds are ≥ 5 % lower than A0's, and **move** accuracy rises with experience
  (late ≥ early + 0.2).
- **P5:** whichever of A and B saves more bits late in held-out worlds is recorded as the better learner of
  explanations, by the user's criterion (fail least often on new data). No direction is predicted: B is the purer
  bitter-lesson route; A extends the validated Bayesian update.

## Results

(appended below)

### Option A results (420 s of training each, about 290 updates; 128 evaluation worlds per split)

| | bits/step, chunks 0–1 | chunks 12–15 | **mean over the world** | turn, early → late | move, early → late |
|---|---|---|---|---|---|
| A0, new worlds | 19.20 | 17.42 | **18.09** | 0.44 → 0.39 | 0 → 0 |
| A with adaptation, new worlds | 22.43 | **16.50** | 18.74 | **0.24 → 0.48** | 0.00 → 0.04 |
| A without adaptation, new worlds | 22.59 | 20.59 | 21.71 | 0.21 → 0.26 | 0 → 0 |
| A0, held-out combinations | 18.86 | 18.40 | **18.21** | 0.34 → 0.41 | 0 → 0 |
| A with adaptation, held-out combinations | 21.91 | **17.39** | 18.78 | 0.26 → 0.44 | 0.01 → 0.03 |

Per-chunk bits on new worlds:
- **A with adaptation:** 21.9, 23.0, 20.5, 21.8, 19.5, 18.8, 19.4, 18.8, 18.7, 18.3, 16.9, 16.4, 16.0, 16.2, 16.6, 17.2.
- **A0:** 19.7, 18.7, 18.1, 17.7, 18.6, 17.9, 18.7, 18.4, 18.6, 18.6, 17.6, 17.4, 17.0, 17.2, 17.6, 17.8.

**Against the predictions:**
- **P1 confirmed, barely:** late bits 16.50 against 17.42, 5.3 % lower.
- **P2 refuted:** **move** rose only 0.00 → 0.04. But **turn**, the robot's facing after a blocked step, which also depends
  on the action's hidden meaning, doubled with experience (0.24 → 0.48), while A0's stayed flat.
- **P3 confirmed:** the adaptation gain on held-out combinations (21.53 − 17.39 = 4.14 bits) equals the gain on new
  worlds (20.59 − 16.50 = 4.09).

**Reading.**
- **In-world Bayesian adaptation works and generalizes to held-out physics:** about 4 bits per step saved late in a
  world. Part of this world's hidden action meaning is inferred (turn).
- **By the user's criterion (the total code length of a world), A loses slightly:** 18.74 against 18.09. Its shared
  prior is tuned for being adapted, not for guessing, so it starts each world ~3 bits worse. It overtakes A0 only after
  ~80 of the 128 steps.
- **Move predictions,** which need the destination cell to compute where this world's action sends the robot, are
  barely learned in ~290 meta-updates.
- **Caveats:** one run per arm; very short meta-training (A0 was still improving).

### Option B results (420 s of training, 250 updates of 4 worlds = 1,000 worlds seen)

- **New worlds:** bits per step 20.07 early, 18.52 late, **mean over the world 19.05**.
- **Held-out combinations:** 19.73 early, 19.44 late, mean 19.13.
- **No rule improves with experience in the world:** turn 0.46 → 0.37; move 0 → 0; the per-chunk curve is flat.
- **P4 refuted.**

**Comparison** (new worlds / held-out combinations):

| learner | mean bits/step over a world | late (chunks 12–15) | in-world rule learning |
|---|---|---|---|
| A0, one model | **18.09 / 18.21** | 17.42 / 18.40 | none |
| A, Bayesian adaptation in the weights | 18.74 / 18.78 | **16.50 / 17.39** | turn 0.24 → 0.48 |
| B, in context | 19.05 / 19.13 | 18.52 / 19.44 | none |

**Reading (P5).**
- **By the late-world criterion,** A is the better learner of explanations: it fails least on new data once it has seen
  a world.
- **By the whole-world code length,** none of the adaptive learners beats the one-size model within 128-step worlds
  and ~7 min of training.
  - A pays a start-up cost (its prior is tuned for adapting).
  - B did not learn to infer in context at all within its budget. It saw 1,000 worlds against A0's ~9,300 world-passes,
    because each pass processes whole histories.

**Next steps for perfecting B-R3 here:**
1. **Remove A's start-up cost.** At the start of a world, predict by averaging over the prior (samples
   φ ~ N(μ, Λ₀⁻¹)) instead of from the single point μ. That is the Bayes-correct prior predictive. Alternatively,
   second-order meta-gradients for μ.
2. **Teach move predictions** (routing a world's action meaning to the destination cell) with more meta-training.
   A0 was still improving at the cap.
3. **Longer worlds,** as ARC-AGI-3 games are: adaptation pays more, the longer a world lasts.

## E8b, pre-registered (2026-10-20, late): prior averaging, and longer worlds

The user: "Go ahead with the prior-averaging fix and longer worlds."

**Prior averaging (`--prior-avg`).**
- Each world's fast parameters get a learned prior **spread** σ (per element; log σ starts at −2.3, i.e. σ ≈ 0.1),
  alongside the learned prior mean μ.
- The first 4 chunks of a world are predicted by the **posterior predictive**: the average over 4 draws
  φ ~ N(φ_w, 1/Λ_w), with Λ_w = 1/σ² + n_seen · F̄.
  - The likelihood is each world's chunk as a joint mixture (the parameters are shared within a world).
- The Kalman step's prior precision becomes 1/σ², replacing the hand-set λ₀ = 10. "How widely to average" and "how
  much to adapt" are then the same learned quantity.
- σ is learned by the same empirical-Bayes objective (first-order).
- **Found in the smoke test:** σ = 0.3 sampled weights so wide they wrecked predictions (150–200 bits per step early).
  The old λ₀ = 10 corresponded to an uncalibrated prior. Hence the start at 0.1.

**Longer worlds.** 512 steps per world instead of 128 (`--t-max 512`). Death respawns the robot at full health instead
of ending the world (logged; the reward penalty is kept).
- Data: 2,048 training worlds (1.05M random-play transitions), 256 new and 256 held-out test worlds; 64 of each
  evaluated.

**Arms** (420 s of training each, evaluated at the end):

| arm | world length | configuration |
|---|---|---|
| **P128** | 128 | adaptation + prior averaging (against E8's A: 18.74 mean bits/world; A0: 18.09) |
| **L-A0** | 512 | no adaptation |
| **L-A** | 512 | adaptation (λ₀ = 10) |
| **L-AP** | 512 | adaptation + prior averaging |

**Predictions:**

| # | prediction |
|---|---|
| P6 | Prior averaging removes most of A's start-up cost: P128's early-chunk bits ≤ A0's + 1 (A was +3.2), and its whole-world mean ≤ A0's (18.09) |
| P7 | In 512-step worlds, adaptation beats the one-size model on the **whole-world** code length (L-A or L-AP mean < L-A0's) |
| P8 | Long worlds show in-world rule learning: late-world turn ≥ 0.8 with adaptation, against ≤ 0.6 without |

### E8b results (420 s of training each; 128 / 64 evaluation worlds per split for 128- / 512-step worlds)

| arm | bits/step, first 2 chunks | last 4 chunks | **whole-world mean (new / held-out)** | turn, late | move, late |
|---|---|---|---|---|---|
| A0, 128 steps (E8) | 19.20 | 17.42 | 18.09 / 18.21 | 0.39 | 0 |
| A, 128 steps (E8) | 22.43 | 16.50 | 18.74 / 18.78 | 0.48 | 0.04 |
| **P128,** A + prior averaging, 128 steps | 26.37 | 16.64 | 19.61 / 19.59 | 0.09 | 0 |
| **L-A0,** 512 steps | 18.64 | 14.63 | 17.44 / 17.32 | 0.42 | 0 |
| **L-A,** 512 steps | 22.76 | **9.23** | **14.41 / 14.31** | **0.64** | **0.11** |
| **L-AP,** 512 steps + prior averaging | 25.28 | 9.47 | 14.22 / 14.17 | 0.54 | 0.06 |

Held-out late bits: L-A 8.88; L-AP 9.15; L-A0 14.16.

**P7 confirmed, strongly.** In 512-step worlds, Bayesian in-world adaptation beats the one-size model on the
**whole-world code length**, by the user's criterion:
- 14.4 against 17.4 bits per step on new worlds (−17 %), and 14.3 against 17.3 on held-out combinations;
- late in a world it predicts **9.2 against 14.6 bits per step** (−37 %);
- move predictions appear (0.11–0.16) where the one-size model has none.

With enough time in a world, inferring its rules pays for its start-up cost many times over.

**P8 partly confirmed.** Turn late 0.64 with adaptation against 0.21–0.23 for the same model without it; the ≥ 0.8 bar
is missed.

**P6 refuted.** Prior averaging did not remove the start-up cost. It made it worse:
- 26.4 against A's 22.4 early in 128-step worlds;
- whole-world 19.6 against 18.7.
- **The cause is visible in the log:** the learned prior spread hardly moved, σ = 0.100 → 0.099 over 150 updates. Adam
  at 3e-4 on log σ, with a first-order gradient, cannot learn it in this budget. So the run sampled a fixed, uncalibrated
  spread, which (a) added noise to early predictions and (b) set the Kalman prior precision to 1/σ² = 100, weakening
  adaptation.
- In 512-step worlds it was slightly better on the whole-world mean (14.22 against 14.41) but worse early and late.

**Next:**
- **Set σ by empirical Bayes in closed form.** In a Gaussian hierarchical model, the type-II maximum-likelihood prior
  variance is the between-world variance of the adapted parameters plus their mean posterior variance. After each
  batch of worlds, σ² ← mean_w (φ_w,end − μ)² + mean_w 1/Λ_w,end. That calibrates the spread from the data the model
  actually saw, with no slow gradient.
- **More meta-training:** 420 s is ~70 batches of long worlds, and every run was still improving.

## E8c, pre-registered (2026-10-20, night): closed-form prior variance, and longer training

The user: "Go ahead with the closed-form prior variance and longer training. But do not be surprised if prior
averaging ends up not being worth it under any fix, or the fix that would make it work itself isn't worth it."

**Closed-form prior variance (`--em-sigma`).** After each batch of worlds, the EM M-step of a Gaussian hierarchical
prior, smoothed over batches at rate 0.2:

  σ² ← mean_w (φ_w − μ)² + mean_w 1/(1/σ² + n · F̄)

i.e. the between-world variance of the adapted parameters plus their remaining posterior variance.
- σ starts at 0.3 (prior precision ≈ 11, like A's λ₀ = 10).
- The Kalman step's prior precision is 1/σ². μ is still learned by the first-order gradient.

**Longer training, by chained runs of ≤ 10 min each (resume from a checkpoint).** Every arm gets **1,260 s** of
training in total (3× E8b).
- The existing 512-step arms continue from their 420-s checkpoints (2 more segments; Adam's moments restart once, as
  their first checkpoint did not save them).
- The new arms train from scratch (3 segments).

**Arms** (512-step worlds; 64 evaluation worlds per split at the end):

| arm | configuration |
|---|---|
| **L-A0** | one model for all worlds (continued) |
| **L-A** | adaptation, λ₀ = 10 (continued) |
| **L-Ac** | adaptation, closed-form σ as the prior precision, **no averaging** |
| **L-APc** | adaptation, closed-form σ, **with** prior averaging (posterior predictive over 4 samples in the first 4 chunks) |

**Predictions:**

| # | prediction |
|---|---|
| P9 | Longer training improves L-A's whole-world mean by ≥ 1 bit (14.41 → ≤ 13.4), and adaptation still beats the one-size model (L-A < L-A0) |
| P10 | A calibrated prior precision is at least as good as the hand-set one: L-Ac's whole-world mean ≤ L-A's |
| P11 | **The averaging question.** L-APc beats L-Ac by ≥ 1 bit in the first 2 chunks and ≥ 0.2 bits on the whole-world mean. If not, prior averaging is recorded as **not worth it** at this scale (the user's expectation) |

## E8c results (2026-10-20)

All arms: 1,260 s of training, then 64 test and 64 held-out 512-step worlds. Bits per step (lower is better);
"early" = the first 2 chunks of a world, "late" = the last 4, "mean" = the whole world. Logs: `runs/e8c_*.txt`,
`runs/e8/{L_A0,L_A,L_Ac,L_APc}/log.jsonl`.

| arm | test early | test late | **test mean** | held-out early | held-out late | **held-out mean** | move (start → end) | turn (start → end) |
|---|---|---|---|---|---|---|---|---|
| L-A0, one model | 16.82 | 13.29 | **15.83** | 18.91 | 12.84 | **15.77** | 0.00 → 0.00 | 0.46 → 0.43 |
| L-A, adaptation, λ₀ = 10 | 18.05 | 4.56 | **8.64** | 20.33 | 4.34 | **8.65** | 0.22 → 0.99 | 0.46 → 1.00 |
| L-Ac, + closed-form σ | 16.97 | 4.04 | **7.80** | 19.14 | 3.90 | **7.91** | 0.22 → 0.99 | 0.47 → 0.99 |
| L-APc, + closed-form σ + averaging | 27.20 | 4.65 | **9.74** | 31.34 | 4.51 | **9.74** | 0.00 → 0.99 | 0.36 → 0.98 |

(At 420 s, E8b: L-A 14.41 and L-A0 17.4 on new worlds.)

**Verdicts:**

| # | verdict |
|---|---|
| P9 | **Confirmed, by far more than predicted.** L-A 14.41 → 8.64 (≥ 1 bit predicted; 5.8 measured). Adaptation beats the one-size model by 7.2 bits/step (8.64 vs 15.83), up from 3.0 at 420 s: the gap grows with training. Held-out worlds the same (8.65 vs 15.77) |
| P10 | **Confirmed.** L-Ac 7.80 vs L-A 8.64 (test), 7.91 vs 8.65 (held-out). The calibrated prior also removes most of adaptation's start-up cost: early bits 16.97 against the one-size model's 16.82 (test) and 19.14 against 18.91 (held-out), where L-A paid 1.2–1.4 bits |
| P11 | **Refuted. Prior averaging is recorded as not worth it at this scale** (the user's expectation). L-APc is 10 bits *worse* early (27.20 vs 16.97) and 1.9 bits worse on the mean (9.74 vs 7.80). E8b's averaging arm also started worse: two failures in two configurations |

**Why averaging hurt (a diagnostic from the saved checkpoints, no re-run).**
- **It is not the sampling noise.** With the learned σ and the Fisher, the second-order cost of predicting from one prior sample, ½ Σ F̄ σ², is only ~0.2 bits/step in both arms.
- **The averaging arm learned a worse prior mean μ.** Predicting from μ with no adaptation at all, its early bits are 27.4, against L-Ac's 19.0. Training the first chunks through the sample mixture degraded the point estimate the rest of the world starts from.
- The mechanism of that degradation is not established. Given the verdict, it is not pursued.

**What σ learned.** The logged mean σ moved little (0.30 → 0.27), which looks like "σ did not learn". It did:
- Weighted by the Fisher (where the data constrains the parameter), σ is 0.04–0.22 per parameter group: small where worlds agree, large on the action embeddings (0.22), where the shuffled controls live.
- Elements the data does not constrain keep their initial σ: the EM fixed point when the posterior variance equals the prior. That is harmless, since those elements do not affect the predictions.
- The log's unweighted mean is a misleading summary; the Fisher-weighted value should be logged.

**Caveats.**
- One seed per arm.
- L-A and L-A0 were continued from E8b checkpoints (Adam's moments restarted once), while L-Ac and L-APc trained from scratch in 3 segments. The total training time was equal.

**Where E8 stands.**
- On the world family with hidden, shuffled physics, Bayesian in-world adaptation with an empirically calibrated prior beats the one-size model by about 8 bits per step over whole worlds, on held-out physics combinations as well.
- By the end of a world it predicts the shuffled moves (0.99) and turns (0.99–1.00) almost perfectly.
- The remaining cost is the first chunks of a world, before the world has been identified.
- **Next candidates (not pre-registered):**
  - exploration by expected information gain, to identify a world faster;
  - learning without the logged job events.
