# A fast proxy for training outcomes (declared 2026-10-10, before any proxy value was computed)

The user asked: "If only there was a way to evaluate an option in seconds on the CPU without doing a training run. Is
there a way? Maybe a CPU evaluation should be looking for something other than final training loss or gradient quality?"
They approved the plan (2026-10-10, "Go ahead in that order"):
1. Build the proxy and validate it on recorded outcomes.
2. Pre-register its Experiment 3 predictions.
3. Run Experiment 3 as the first out-of-sample test.

## Why the gradient cosine failed as a proxy

Cosine misranked three arms:
- **guided:** 2.3× cosine, trained worse;
- **local:** 1.2×, trained best;
- **lr_rand:** 4× worse cosine, trained only 0.02 worse.

**What the theory says.** Under a step u, the loss decrease is −η gᵀE[u] + (η²/2)(E[u]ᵀH E[u] + tr(H·Cov u)), with:
- g, the true gradient;
- H, the curvature (Hessian).

At the best step size η, the decrease is

  **P = (gᵀū)² / (2·(ūᵀHū + tr(H·Σ_u)))**,

the quantity behind the "gradient noise scale" (McCandlish et al. 2018, cited from memory).

**What cosine leaves out:**
- **Noise is not all equal.** It costs only in proportion to the curvature along it: tr(HΣ), not tr(Σ).
- **Over-weighting matters.** Stretching the update along sharp directions (what guided ES's Σg does) costs through
  ūᵀHū, like a larger learning rate exactly where the loss is steepest.

## The proxy (fixed now)

**At a checkpoint**, run the estimator on n = 8 different training batches. Optional warm-up calls come first, for arms
with state such as the guided subspace. Then compute, over the main model's parameters (auxiliary heads excluded):
- **g:** the backprop gradient over 8 held-out batches;
- **the per-batch backprop gradients g_b;**
- **Hessian-vector products:** from 2 held-out batches, by double backward.

**Computed from these:**
- **cos:** the mean over batches of cos(ĝ_b, g_b), a one-number version of M1. The comparison point.
- **P_sgd:** P with u = ĝ, ū its mean over the 8 batches, and Σ_u their covariance (it includes the batch noise every
  method has). The trace is taken by HVPs of the deviations.
- **P_adam:** the same, after an Adam-like diagonal scaling u = D·ĝ, with D = 1/(√(mean over batches of ĝ²) + ε). The
  noise inflates the second moment and shrinks the step, as in Adam.
- **persist:** the mean cosine between consecutive batches' errors e_b = ĝ_b − g_b. It is reported, not ranked:
  momentum averages random errors but not repeated ones.
- **Backprop's own values (P_bp)** for scale.

**Checkpoints:**
- **late:** Experiment 1's backprop checkpoint (300 steps, validation 2.34);
- **early:** a new backprop checkpoint after 60 steps.

An arm's proxy rank is its average rank over the two checkpoints.

## Validation on recorded outcomes (no training re-run)

**Ground truth.** The mean validation loss after 300 steps (two seeds) of every arm trained at matched cost on the dense
ReLU² d = 64 model. That is 17 arms:
- **Experiment 1:** base, lr_rand, lr_pca, orth, sobol, anti, local128, ent50, guided;
- **Experiment 2:** O1_sparse, O3_top, O4_hubT_K56, combo_K56;
- **Experiment 2b:** guided_w, top_w, combo_w_K56, O4_orth_K56.

**Excluded:**
- the half-cost arms (combo_local_K128, O4_local_K128);
- the top-10% model's arms (a different model).

**Pass mark,** for a proxy to be used:
1. **Ranking.** Its Kendall τ-b with the outcome (lower loss = better) is ≥ 0.5.
2. **Key arms.** It places guided, O3_top and combo_K56 below the baseline, and O4_hubT_K56 and local128 above it.
3. **Beats cosine.** Its τ is higher than cos's.

The scatter is inspected, not just τ: ties fooled a rank statistic in this project before.

**Choice.** If both P_sgd and P_adam pass, the one with the higher τ is used. Choosing one of two on 17 arms overfits a
little, which is why Experiment 3 is the real test.

## Then

1. The chosen proxy's predictions for Experiment 3's arms are written into `EXPERIMENT_3.md` before Experiment 3
   trains.
2. If it passes there too, it becomes the screen: many variants are evaluated on the CPU while the GPU trains, and only
   the top few are promoted to 2-minute training runs.
3. If it fails, the failure says which ingredient is missing, for example longer-horizon effects.

Code: `proxy.py`. Results: `runs/proxy/`.

## Version 1 — RESULT (2026-10-10, `runs/proxy/`, `runs/proxy/validation.json`)

**How it was computed.** On the GPU, for speed: 2–6 s per arm. On the CPU the same takes ~45 s per arm: 8 estimator calls
of ~287 partial forward passes each.

| proxy | Kendall τ-b with the outcome | key arms right |
|---|---|---|
| cos (per-batch, all parameters as one vector) | **+0.543** (p 0.003) | 4/5 (misses local128) |
| P_sgd | +0.022 | 2/5 |
| P_adam | +0.157 | 2/5 |

**FAILED: neither P passes, and neither beats cos.** cos itself misses criterion 2 (local128 ranked mid-table).

**Why P fails.** A one-step quadratic model, with each method's own best step size, rewards any update that avoids sharp
directions, even a biased one. So O3_top, lr_pca and guided "beat" backprop on P_sgd. Training does not work that way:
300 Adam steps at one shared schedule. The curvature reading of the guided/O3 failures (`EXPERIMENT_2B.md`) is
therefore NOT supported by this.

**What it did teach.**
- **Which cosine.** The cosine over all parameters at once ranks arms far better than the per-matrix mean used in
  Experiments 1–2. Guided is 0.28 against the baseline's 0.60 by the former, but looked better by the latter. So
  "cosine misled us" was partly a choice of WHICH cosine.
- **The two remaining blind spots differ.**
  - cos rates ent50 high (0.70), but it trained worst. Dropping half the tokens removes signal, which a cosine cannot
    see.
  - cos rates local128 mid-table, but it trained best.

## Version 2 — declared 2026-10-10, after version 1 failed and before any v2 value was computed

**A short-horizon probe.** From each checkpoint (bp60, bp300), train each arm for 25 steps with the real optimiser:
- Adam (0.9, 0.99), fresh state;
- lr 1e-2 with 5 warm-up steps;
- the arm's own estimator, on the same 25 training batches for every arm;
- auxiliary heads fitted first for local arms, as in M1.

**Measured:** the drop in validation loss (8 held-out batches). An arm's proxy rank is its average rank over the two
checkpoints.

**Same pass mark as version 1:**
1. τ-b ≥ 0.5;
2. all 5 key arms right;
3. τ-b above cos's 0.543.

The 17 arms are no longer unseen, so Experiment 3 is the real test. Code: `probe.py`. Cost: ~5–10 s per arm per
checkpoint on the GPU.

## Version 2 — RESULT (`runs/probe/`, `runs/proxy/validation.json`)

The probe's τ-b with the outcome is **+0.573** (p 0.0015).
- **Criterion 1:** ✓ (τ ≥ 0.5).
- **Criterion 3:** ✓ (above cos's 0.543).
- **Criterion 2:** ✗. Only 4 of 5 key arms are right: it ranks **local128** 10.5 of 17, but it trained best. cos made the
  same miss.

**By the declared mark it FAILS, narrowly.** Both proxies are recorded for Experiment 3 as an out-of-sample test of
themselves. Neither gates anything: every Experiment 3 arm still trains.

**Observations, NOT used to change the declared proxy:**
- **The late checkpoint carries the signal.** Ranked from bp300 alone, τ-b = +0.721; from bp60 alone, +0.353.
- **The probe catches what cos missed.** It puts ent50 last, where cos put it near the top. It also ranks combo and
  guided_w low.
- **Both proxies miss local scoring's gain.** That gain is probably made very early in training from scratch, or it is
  partly seed luck: its seeds differed by 0.07–0.09 in both runs. A probe from initialisation would test the first
  explanation.

**Timing:** 4–10 s per arm per checkpoint on the GPU. A CPU version would be about 25× slower at this batch size, so
the "seconds on the CPU" goal is not met by this design.

## Out-of-sample result on Experiment 3 (2026-10-10)

The probe's recorded predictions ranked Experiment 3's six comparable arms perfectly (τ-b = 1.0), with 4/5 same/worse
calls right (the miss was at the band's edge). cos got τ-b = 0.41.

**Status:** the probe is a useful screen for variants of the estimator's noise and weighting, with one known blind spot:
anything whose gain shows up only from scratch, such as local scoring. Use it to rank candidates. Promote the top ones
to real 2-minute runs, and always also promote anything that changes WHAT is scored, not just how draws are made. It
costs 4–10 s per arm per checkpoint on the GPU. On the CPU it is about 25× slower at this batch size.
