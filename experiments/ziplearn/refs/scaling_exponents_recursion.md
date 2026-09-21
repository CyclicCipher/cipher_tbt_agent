# How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents — reference notes

**Paper:** Zixi Chen, Akshay Vegesna, Samip Dahal, Andrew Gordon Wilson, arXiv:2609.19107 (v2, 17 Sep 2026).
Code: github.com/qlabs-eng/scaling-exponents. **License:** arXiv non-exclusive distribution (not Creative Commons),
so this file is the abstract verbatim plus notes in our words; the paper itself is at https://arxiv.org/abs/2609.19107
(HTML: https://arxiv.org/html/2609.19107v2).

## Abstract (verbatim)

> Scaling laws predict how loss decreases with increases in computation. We show, contrary to conventional wisdom,
> that architectural interventions can modify scaling exponents in pre-training, leading to power-law improvements in
> performance as computation increases. As an anchoring point, we consider the architectural formulation of looped
> transformers. Although not typically used in this way, looping, also known as recursive depth, provides a mechanism
> for model growth, by increasing the number of loops during training. Model growth, with and without shared weights,
> provides the biggest changes to the scaling exponents. In particular, a 7.4B model growth architecture matches
> GPT-3 13B on CORE with roughly 20× less compute, and has compute efficiency gains that increase with scale.
> Moreover, simply using a boundary operator in a vanilla transformer, which normalizes and injects an earlier block,
> also provides an exponent increase, although to a lesser extent. In the data-constrained, multi-epoch setting,
> standard looping has a useful regularizing effect, where we find it is compute-optimal to increase the number of
> loops with scale. These results can be understood through the lens of computational depth: for a given
> computational budget, we wish to increase the usable depth of the transformer, which can lead to efficiency gains
> that increase with scale.

## The method, in our words

**The looped transformer** (their eq. 2): prelude blocks P embed the input, e = P(s); a core block R is applied K
times, h_k = R(φ(h_{k−1}, e)), starting from h_0 = 0; coda blocks C produce the output, y = C(ρ(h_K, e)). "Tied": one
core, applied K times (θ_1 = … = θ_K). "Untied": each pass has its own weights (a deeper plain stack, written as a
loop). Executed depth = P + K·(core) + coda blocks.

**The boundary operator** φ, ρ: BO(h, e) = RMSNorm(h) + α·e — normalise the residual stream and re-inject the prelude's
output (α learned), between every core pass and before the coda. Purpose: stop the residual stream growing with depth
(the "curse of depth" of pre-norm stacks) and keep every pass conditioned on the input. Alone, in a vanilla
transformer ("Operator-1"), it already raises the exponent.

**Model growth**: start with K_0 passes (typically 2), at a training step g switch to K_f = m·K_0 (typically 4).
Untied growth copies the core and trains the copies separately (depthwise stacking); tied growth simply applies the
one core more times, adding no weights.

## Numbers

Fitted L(C) = E + A·(C/C_0)^−γ. Compute multipliers over Vanilla at 10^20 FLOPs: Operator-1 1.25×, Untied-2 1.34×
(a constant shift, not an exponent change), Loop-Grow 1.36×, **Untied-Grow 1.55×** — and the growth multipliers rise
with scale (1.8× at 1.23·10^21, projected 2.7× at 10^25), which is what an exponent change looks like. A 7.4B
Untied-Grow model at 1.23·10^21 FLOPs scores CORE 0.3865 against GPT-3 13B's 0.3852 at 2.31·10^22 — "roughly 20× less
compute", flagged by the authors as indicative (different data, different evaluation pipeline). An extrapolation run
at 8× the largest fitted compute lands on the predicted curve.

Data-constrained (100M unique FineWeb tokens, 10 epochs): the compute-optimal loop count rises from ~1.4 to ~6.7 with
budget; scaling loops at fixed size reaches the tuned model-size ladder's loss with 2.2× less compute; weight decay
tracks stored parameters, not executed depth; tying is a regulariser under repetition.

**Computational depth**: "the number of blocks that meaningfully influence the predictive distribution", measured
by a logit-lens KL — the first block after the KL peak within 2 nats of the final output. Vanilla ≈ 12 at 10^20
FLOPs; Untied-2 with the boundary operator ≈ 24; Untied-Grow ≈ 36. Two wastes are named: blocks that execute
without changing the prediction (the boundary operator's target) and, early in training, blocks not yet needed (the
growth schedule's target). Sizes: ladders from ~10^18 to 10^20 FLOPs, depths d8–d26 at a fixed width/depth ratio of
128; FineWeb with the GPT-2 tokenizer; CORE and answer-NLL downstream.

## How it bears on this project

- **A plan is a network** (DESIGN §16): a plan of n actions is an n-layer written network, and depth is chosen per
  problem by the planner. A looped transformer with tied weights is that idea in a gradient-trained model: one block,
  applied as many times as the problem needs, with growth adding passes when the data demand them. Our E2/E14b
  transformer never composed; a looped block with a boundary operator is the natural substrate to test composition
  on, because "repeat the same operation K times" is exactly the structure E12 found cheapest to describe.
- **The boundary operator is attention residuals' cousin** (§15, E0, E8b): re-injecting the embedding at every pass
  is what our attention-residual model *learned* to do — every attention sub-layer read the embedding at 0.28–0.37 —
  and RMSNorm on the stream is what the paper's residual-growth diagnosis and AttnRes's bounded magnitudes both
  address.
- **Computational depth as the quantity to buy**: their KL effective depth is a Jacobian-free relative of E8b's
  measurements; for the written network the analogue is the plan length the planner is allowed.
- Proposed test (not built): the gradient arm as a looped block + boundary operator on the composition task (E2) and
  on E14b's order-dependence, at equal compute — 20 minutes.
