# E7: Bayesian R3: learning one refutation as a sequential Bayesian update

**Status:** pre-registered 2026-10-20 (evening). Each run ≤ 5 min.

## The redesign (`bayes.py`; full design in `TRAINING_METHOD.md`, "Bayesian R3")

The weights carry a Gaussian (Laplace) posterior: mode μ (the current weights) and precision
Λ_j = N · F̄_j + λ₀.
- F̄_j is the mean per-sample empirical Fisher, from the same loss the model was trained on.
- N is the number of data the posterior summarizes, each counted once.
- λ₀ is a base prior precision.

**Learning new data = the MAP of [old posterior × the new data's likelihood]:**

  minimize ℓ_new(θ) + ½ Σ_j Λ_j (θ_j − μ_j)²

optimized to convergence.

| R3 step | B-R3 |
|---|---|
| refute | a datum with a large Bayesian surprise |
| rehearse until explained | optimize the MAP objective to convergence. The datum counts **once**, however many iterations |
| interleave past refutations; gentle steps | the precision penalty. It protects exactly what past data pinned down, by how firmly |
| retire | absorb: μ ← the new mode, Λ ← Λ + F_new |

## The test: arm K's one-refutation setup, so the numbers compare directly

**Setup.**
- Start from J2's world model, which never learned `button_off`.
- The Fisher is computed from 2,048 single stream rows; N = 614,400 (J2's stream), λ₀ = 0.01.
- One `button_off` training instance (seed 0, the same one as K and K3a).
- 600 Adam iterations (lr 1e-3) on ℓ_new + penalty. **No replay, no interleaving, no retirement schedule.**
- Logged every 25 iterations: held-out button_off (28 instances), the case itself, the other rules, changed- and
  unchanged-entry accuracy, and the Bayesian surprise ½ (θ − μ)ᵀ Λ (θ − μ).

**Arms:**

| arm | instances | seed | notes |
|---|---|---|---|
| **E7a** | k = 1 | 0 | |
| **E7b** | k = 1 | 2 | the instance R3 handled worst (K3a-s2: ~0.5, fluctuating) |
| **E7c** | k = 4 | 0 | |

**Comparison:**

| arm | held-out button_off | theft |
|---|---|---|
| K (plain rehearsal) | 0.86 → **0.18** | swung 0.01–0.71 |
| K3a (R3) | 0.86 at the end | 0.68 → 0.85 |
| K3a-s2 | ~0.5, fluctuating | |

## Predictions

| # | prediction |
|---|---|
| P1 | E7a: held-out button_off ≥ 0.75 at the end and **no rationalizing decay** (the end ≥ its peak − 0.1); the case fitted (1.00) |
| P2 | E7a: **no forgetting.** Theft within 0.05 of 0.68; mined, move and deliver within 0.05; changed-entry accuracy within 0.02 of 0.446 |
| P3 | E7b's end ≥ K3a-s2's end (0.46); E7c ≥ 0.85 |

**If the prior is too stiff,** the rule is not learned at all (held-out stays near 0). That would refute the
N = 614,400 setting, not the principle.

## Results

(appended below)

### Results (each run ~1.5 min including 50 s for the Fisher)

| arm | held-out button_off (iteration 0, 100, 200 … 600) | case | other rules, start → end |
|---|---|---|---|
| E7a (k = 1, seed 0) | 0.00, **0.75**, 0.71, 0.71, 0.71, 0.71, **0.71** | 1.00 | all within ±0.02 (theft 0.68 → 0.67); changed 0.446 → 0.456 |
| E7b (k = 1, seed 2) | 0.00, **0.86**, 0.86, 0.86, 0.86, 0.86, **0.86** | 1.00 | all within ±0.02 |
| E7c (k = 4, seed 0) | 0.00, **0.86**, 0.86, 0.82, 0.82, 0.82, **0.82** | 1.00 | interact_full +0.05, the rest within ±0.01 |

The Bayesian surprise of the whole update was 0.12 nats (E7a): a small, targeted change of belief.

**Against the predictions:**
- **P1:** the no-decay part is confirmed (end ≥ peak − 0.1 in all three); the level part misses narrowly (E7a ends at
  0.71 against a bar of 0.75).
- **P2 confirmed:** no forgetting in any arm.
- **P3:** E7b is confirmed (0.86 against R3's 0.46 on the same instance); E7c misses narrowly (0.82 against 0.85).

**Against R3 and plain rehearsal on the same setup:**

| | held-out button_off | theft | changed-entry accuracy |
|---|---|---|---|
| plain rehearsal (K) | 0.86 → **0.18** | swung 0.01–0.71 | 0.446 → ~0.35 |
| R3 (K3a, seeds 0 / 1 / 2) | 0.86 / 0.75 / ~0.5 (fluctuating) | 0.85 / 0.80 / 0.89 | 0.446 → 0.386 |
| **B-R3 (E7, seeds 0 / 2)** | **0.71 / 0.86, stable** | **0.67 / 0.69** | **0.446 → 0.456** |

**Reading.**
- **The MAP of [old posterior × the datum counted once] is a fixed point.** It cannot rationalize: further iterations
  change nothing. It protects old knowledge by construction.
- **It needs no replay, no interleaving pool, no retirement rule and no learning-rate schedule.**
- **On the one-case level, it matches R3 on average and is far more consistent across instances.**
- **R3's interleaving improved theft (0.68 → 0.85); B-R3 holds theft rather than improving it.** It has no reason to
  revisit old data.
- **Caveats:**
  - one seed per arm (two instances for k = 1);
  - 28 held-out instances, so 0.71 against 0.86 is 4 instances;
  - a diagonal Fisher from 2,048 samples;
  - N and λ₀ not tuned.
