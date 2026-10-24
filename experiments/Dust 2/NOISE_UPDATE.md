# Analysis: the "Noise" speedrun and its October 2026 update (qlabs)

**Sources:** qlabs.sh/noise (the update section) and github.com/qlabs-eng/noise (MIT; cloned into git-ignored
`third_party/noise` at commit 3343973). Read 2026-10-23 at the user's request.

## What Noise is

A pretraining speedrun for learning *without backpropagation*:
- the model is fixed (Dust's 37.7M-parameter GPT), the data is fixed, and so is the score;
- only the learning algorithm may change;
- the tracks: lowest validation loss after one epoch; the time to a target loss; and an open-architecture track;
- **the rule:** every error signal is measured by running the model forward against a loss, then may take **at most
  one step through a nonlinear part**. Steps through linear maps are free, and credit may not be chained back block
  by block. Methods should, in spirit, work with non-differentiable parts.

## The claimed 4,000×: where it comes from (the 10M-token track, time to loss 4.2)

| record | time | what changed | kind |
|---|---|---|---|
| 1 | 1 h 46 min | Dust with Adam, population 24k | the baseline |
| 2 | 22 min | population 5k | tuning (~5×) |
| 4 | 5 min | **Boost: stagewise forward-only training with exact readout credit** | **algorithm (~4×)** |
| 6 | 2 min 37 s | closed-form hub and value credit, exact keys | algorithm |
| 7 | 1 min 23 s | **identity-path credit** with exact values, no Boost or Cut phases | **algorithm (~2–4×)** |
| 8–10 | 1 min 2 s → 15.4 s | factored probes, fused kernels, token sharding (JAX) | engineering (~5×) |
| 11–22 | 14.6 s → 1.6 s | structured (data-free) start, tail averaging, fewer and bigger updates, kernels, overlapped communication | schedule and engineering (~10×) |

Roughly: ~5× tuning, ~10–15× algorithm, ~50× engineering and schedule.

**Caveat:** 4.2 is a weak target. Backprop with Adam reaches 3.769 on that track, and a race to a weak target rewards
early-training speed.

## The method that matters: exact local readout credit

From `train/val/run.py`, class `Boost`, functions `probe` and `cycle_derivatives`:
- After every sub-block, each token's residual stream is **read by the shared LM head**, as in a logit lens. Each
  block is trained against this **local readout loss**, not the global loss: deep supervision / greedy layerwise
  learning.
- The token's error at the stream is measured by finite differences along a **complete Hadamard basis** (all 512
  directions). The moved readout losses are evaluated in **closed form** from one matrix product, with no rerun of
  the network.
  - So the error is *exact* (no population noise), and cheap, because only the head is evaluated.
  - In Dust 2's terms, K = D with orthogonal draws: the noise law cos = √(K/(K+D)) no longer applies, because a
    complete orthogonal basis recovers the error exactly.
- The weight gradient is then the usual outer product (error × input). Errors for the attention hub and the MLP
  hidden units come from one linear step (free) and one nonlinear step (allowed) from the measured stream error.
- **Identity-path credit:** the error is measured once at the top and handed to every block through the residual
  path.
- **Cut** (after 20% Boost) adds forward probes of the model's own query paths and "future-token" probes for credit
  beyond the local readout.

## Against Dust 2's findings (README "Where it stands", 2026-10-10)

| Noise ingredient | the Dust 2 counterpart |
|---|---|
| local readout loss per block | **local scoring** (truncated reruns), Dust 2's best (O4 + local: 3–4.7× better than Dust) |
| a complete orthogonal (Hadamard) basis | **orthogonal draws** (O4 + orth: 2×) |
| identity-path credit (top error passed down the residual path) | **O3_top** (a share of each draw along the exact top error, computed from the head in closed form, "like Direct Feedback Alignment") |
| one linear step to hub and hidden errors | **O4** (hub and hidden errors through one local linear map from the writer's estimate) |

**Dust 2 had each ingredient.** The difference: Noise pushed each to its limit (a full cycle, so exact credit; local
losses evaluated in closed form, so no reruns), combined them, and added dozens of community PRs of tuning and kernel
engineering on 8×H100s.

## Where it stands against backprop

- **Loss** (30M track, one epoch): the best no-backprop record is 3.409 (1 h 43 min on 8×H100). Backprop: 3.419 with
  Adam, 3.340 with ANVIL II. So it edges past Adam and stays behind the best backprop optimizer.
- **Time:** the README reports no backprop wall-clock. Rough estimate: training 37.7M parameters on 30M tokens is
  about 1e16 FLOPs, a few seconds to a minute on 8×H100 depending on utilisation. The 30M speedrun record (8 min 20 s
  to loss 3.5) would then be roughly 10–100× slower than backprop, and the best-loss record far more.
  "Not by orders of magnitude anymore" holds for the 10M track's weak target, not clearly for the harder one.
- **The spirit of the rule:** the closed-form readout differences use the softmax head's structure. For truly
  non-differentiable parts, the method would need real reruns, which is the cost Dust paid.

## What it means for Dust 2's goals

The user's goals: no differentiability requirement, no backward pass, no stored activations for looped models, event
sparsity.
- **Confirmed direction:** local objectives + exact credit along an orthogonal basis + the identity path + one
  linear step are what made it work.
- **Implication:** most of the learning signal a residual network needs is *local* (each block making the stream
  more predictive through a shared readout). Deep chains of credit were not needed to edge past backprop with Adam
  at this scale.
- **Not addressed by Noise:** looped models (credit across loop iterations), stored activations (it captures clean
  activations every step), event sparsity. Dust 2's cache-free simultaneous perturbation and event-driven reruns
  remain the relevant work for those.
- **Untested:** whether "local readout credit" scales. Greedy layerwise learning is known to lose ground to
  end-to-end training at depth; 8 blocks and 37.7M parameters is shallow.
