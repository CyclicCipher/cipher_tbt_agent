# Experiment 3 — thermodynamic ideas (THERMODYNAMICS.md T1–T3) (pre-registered 2026-10-10)

Status: **pre-registered before any arm ran.** It was written while Experiment 2's training was running. The user asked to
record the thermodynamics ideas and run the good ones "at the appropriate time".

**Reused, not re-run** (the user's rule): the Dust baseline is σ = 0.2, linear weighting, one site per pass, from
`runs/e1/m1/*_base.json` and `runs/e1/m2/base_s*.json`. The same task, model, checkpoints, schedule and cost accounting
as Experiment 1.

## Arms

| arm | change | idea |
|---|---|---|
| **T1_sig01**, **T1_sig04** | σ at every rerun site held at 0.1 or 0.4 (baseline 0.2) | §3: σ is a temperature |
| **T1_anneal** | σ annealed geometrically 0.4 → 0.05 over the run (`--sig_sched 0.4,0.05`) | §3: the textbook schedule |
| **T2_boltz1**, **T2_boltz03** | per token and site, the K draws weighted by softmax(z/λ), with z the reward in units of its spread across the draws, λ = 1 or 0.3; rescaled so that λ → ∞ gives back Dust's estimator exactly | §4: Boltzmann (MPPI) weighting |
| **T2_rank** | centred ranks of the rewards across the draws (the NES shaping in the user's plan) | §4 |
| **T3_simul** | every site (embedding, q/k/v, projections, MLP hidden and output, head logits) perturbed in EVERY pass with independent noise. No clean pass, no cache, no reruns: weights move by Σ (reward − chunk mean) · a · xᵀ, with x the input the site saw in THAT pass (`simul.py`). Matched cost: K = 286 full passes. σ = 0.2 or 0.1, chosen by M1 cosine on task 1, as Dust tuned σ; then trained | §2: fluctuation–dissipation; the strict "no stored activations" form |

## Measures

- **M1:** cosine to backprop at the checkpoints, both tasks, converted into gain. Covers T1 levels, T2 and T3; the
  annealed arm has no single-checkpoint value.
- **M2:** training on Latin, 300 steps, two seeds, validation every 10 steps. New runs also log the loss on 8 fixed
  TRAINING batches, so the train−validation gap is visible.

## Predictions

- **E1 — T1 is a null in this regime.** No σ arm, constant or annealed, beats the baseline by ≥ 0.02. The free-energy
  argument is about which minimum is found and how it generalises. Here 300 steps see ~600k characters of an
  11M-character corpus, so there is no overfitting to protect against, and the train−validation gap stays < 0.02 for
  every arm.
- **E2 — T2 does not help on a smooth loss.**
  - **M1:** boltz1 between 0.9 and 1.1; boltz03 < 0.9; rank between 0.8 and 1.05.
  - **M2:** none better than the baseline by ≥ 0.02.
  - **Why:** in the linear-response regime, the linear estimator already uses all the information. Shaping only pays
    under heavy-tailed rewards, which this network does not have. The real test is a critical sparse network, later.
- **E3 — going cache-free has a price.** T3's M1 gain is between 0.3 and 1.0: every other site's noise adds to each
  site's reward noise, roughly 17× more noise sources, against 8.5× more draws for the cost. In M2 it is worse than the
  baseline by > 0.02. The number to carry forward is the size of that price, because the cache-free form is the one
  compatible with looped models without stored history.

## Verdicts

The rules are Experiment 1's:
- **Success:** M1 gain ≥ 1.3 on both tasks, and M2 better by the margin of 0.02 in both seeds.
- **Failure:** gain < 1.1, or a worse M2.
- **Partial:** anything else.

## Files

- Code: `estimators.py` (`shaping`, `lam`, `simul`), `simul.py`, `train.py` (`--sig_sched`; the train-batch loss).
- Results: `runs/e3/`.

## Recorded before Experiment 3 trained (2026-10-10)

### M1 (gradient quality, gain over the baseline; `runs/e3/m1/`, `runs/e3_m1_log.txt`)

| arm | Latin | recall |
|---|---|---|
| T1_sig01 / T1_sig04 | 0.98 / 0.99 | 0.97 / 0.99 |
| T2_boltz1 / T2_boltz03 / T2_rank | 0.74 / 0.37 / 0.95 | 0.72 / 0.34 / 0.96 |
| T3_simul σ 0.2 / σ 0.1 | 0.17 / 0.17 (0.2123 vs 0.2124) | 0.09 / 0.11 |

**T3's σ.** The rule picks **σ = 0.1**, on a near-tie on task 1; recall agrees.

**Deviation, forced by the 2-minute rule.** T3 at matched cost (K = 286) takes ~220 s per run: its full noisy passes are
not compiled, and they draw noise at every site. It trains at **K = 143, half the cost (~110 s)**, and is reported at its
own cost.

### Proxy predictions (neither gates anything; `PROXY.md` versions 1–2 both narrowly failed validation)

Probe drop (bp300 / bp60) and cosine (bp300 / bp60), against the baseline's −0.0157 / +0.1442 and 0.600 / 0.894:

| arm | probe | cos | predicted vs the baseline (± 0.02 = "same") |
|---|---|---|---|
| T1_sig01 | −0.0155 / +0.1461 | 0.599 / 0.893 | same |
| T1_sig04 | −0.0162 / +0.1429 | 0.597 / 0.895 | same |
| T2_boltz1 | −0.0319 / +0.1378 | 0.580 / 0.886 | worse |
| T2_boltz03 | −0.0406 / +0.1263 | 0.582 / 0.889 | worse |
| T2_rank | −0.0165 / +0.1448 | 0.600 / 0.895 | same |
| T3_simul (σ 0.1) | −0.0278 / +0.1117 | 0.225 / 0.383 | worse (and at half the cost) |
| T1_anneal | — (no fixed-σ probe) | — | no prediction |

**The probe's predicted order, best to worst:** T1_sig01 ≈ base ≈ T2_rank ≈ T1_sig04 > T2_boltz1 > T3 > T2_boltz03.
After training, it is scored by Kendall τ-b over these 6 arms plus the baseline, and by its same/worse calls.

## Results — RAN 2026-10-10 (`runs/e3/m2/`; compiled rerun path; baseline reused from Experiment 1)

| arm | validation per seed | mean | vs baseline (2.3687) | final train-batch loss | s/run | cost |
|---|---|---|---|---|---|---|
| T1_sig01 | 2.3664 / 2.3662 | 2.3663 | −0.002 | 2.348 / 2.348 | 52 | 86k |
| T1_sig04 | 2.3715 / 2.3783 | 2.3749 | +0.006 | 2.354 / 2.359 | 54 | 86k |
| T1_anneal (0.4 → 0.05) | 2.3716 / 2.3699 | 2.3708 | +0.002 | 2.353 / 2.353 | 54 | 86k |
| T2_boltz1 | 2.3850 / 2.3898 | 2.3874 | +0.019 | 2.369 / 2.372 | 54 | 86k |
| T2_boltz03 | 2.4107 / 2.4132 | 2.4120 | **+0.043** | 2.395 / 2.398 | 54 | 86k |
| T2_rank | 2.3674 / 2.3703 | 2.3689 | +0.000 | 2.345 / 2.354 | 55 | 86k |
| **T3_simul** (σ 0.1, K = 143, HALF the cost) | 2.4082 / 2.4048 | 2.4065 | +0.038 at half the cost | 2.392 / 2.391 | 115 | **43k** |

**T3 at equal cost.** The baseline's own curve interpolates to ≈ 2.408 at 43k forward-equivalents (2.43 at 28.7k, 2.386
at 57.4k). At 28.6k, T3 is at 2.420 against the baseline's 2.425 / 2.433. **So the cache-free estimator trains about as
well per unit of compute as Dust's rerun-from-cache estimator,** despite a 6× worse gradient cosine. One confound: the
two schedules are at different points at equal cost. T3 has finished its learning-rate decay, while the baseline is
mid-schedule.

### Against the predictions

- **E1 — HELD.** No σ arm, constant or annealed, moves the result by more than 0.01. The train−validation gap is 0.015–0.02
  in every arm: nothing overfits, so the free-energy argument has nothing to act on here.
- **E2 — mostly HELD.**
  - **M1:** boltz03 < 0.9 ✓ (0.37); rank in 0.8–1.05 ✓ (0.95); boltz1 below its predicted 0.9–1.1 band (0.74).
  - **M2:** none better ✓. Boltzmann weighting HURTS (+0.019, +0.043), and rank shaping is exactly neutral (+0.000).
    The linear estimator is already right for a smooth loss. Shaping's real test is a heavy-tailed, critical network.
- **E3 — split.**
  - **M1: REFUTED on the low side.** The gain is 0.17 / 0.11, below the predicted 0.3–1.0.
  - **M2 "worse by > 0.02": HELD only at its half cost.** At equal cost it is level with the baseline. **The price of
    going cache-free is about nothing in training at this scale.** Weights move by a local three-factor rule:
    noise × input × reward. That is the strictest form of the user's "no record of activations", and it is exactly what
    looped models would need.

### The proxies, out of sample (`PROXY.md`)

| proxy | Kendall τ-b over the 6 comparable arms (baseline + T1 levels + T2) | same/worse calls |
|---|---|---|
| **probe (version 2)** | **+1.000** | 4/5 right; the miss is boltz1 (+0.019 against a ±0.02 band) |
| cos | +0.414 | — |

**The 25-step probe ranked every comparable arm correctly.** Its known blind spot from validation (local scoring)
remains. These arms are all close to the baseline, so this is a modest test.
