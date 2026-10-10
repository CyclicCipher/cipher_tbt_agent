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
