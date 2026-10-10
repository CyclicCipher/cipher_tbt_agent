# Research ideas to start from

The ideas come from three sources, kept apart:
- §1: the user's training-method plan (verbatim);
- §2: Gemini's five cost-cutting suggestions (summarised);
- §3–§5: my analysis and my own ideas.

What gets tested and how is in the `EXPERIMENT_*.md` files.

## 1. The user's training-method plan (transcribed from `notes/user_plan_1.png`, `notes/user_plan_2.png`)

> **Training method plans**
>
> 1. **Replace backprop with a method derived from Dust.**
>    1. Dust: Pretraining Transformers Without Backpropagation
>    2. Why? Because while less compute-efficient, it has no differentiability constraint, and it does not require a
>       backwards pass or a record of all previous activations for recurrent training.
>    3. The primary challenge here is that it costs a thousand times more compute than backprop (or more), because its
>       cost scales with the width of the network (its dimensionality). It is probably possible to reduce that cost, but
>       it will require some clever mathematical techniques, possibly related to low-rank methods, and it would be highly
>       advised to read existing literature on **evolutionary techniques** for an optimized technique to handle the
>       dimensionality problem.
>    4. The technique also enables **event-driven sparsity** because of it does not require differentiable gradients.
>       This means between 95-99% of compute in training and inference can be saved compared to a normal AI model, and
>       this could also help alleviate the dimensionality problem. Something like this is the *only way* to achieve the
>       time and compute savings required for AGI on a laptop. But in order to do this, **brain-like critical state
>       sparsity** is the blueprint, doing it naively could occasionally result in a blow-up (**power-law avalanches have
>       heavy tails**), suggested solutions are **rank-based fitness shaping** (like Natural Evolution Strategies) or
>       **Huberized/sign-based loss deltas**. This clips extreme avalanche magnitudes while preserving the directional
>       sign of the perturbation.
>    5. To aid in achieving the above: Combine techniques from **DiffusionBlocks** from Sakana AI, with Block Attention
>       Residuals, in order to further reduce the memory and compute costs and improve pipeline parallelism potential,
>       while solving depth and signal attenuation bottlenecks. It collapses the credit assignment path from O(L) to O(1)
>       because it only has to go through 2-4 layers or however long any individual block is, no matter how deep the
>       network. It solves the greediness problem of local objectives, diffusion formulation provides a mathematically
>       sound objective: early blocks are supposed to output coarse, noisy representations, and middle blocks refine
>       them. And because blocks do not propagate into each other, each block can be evaluated in parallel on different
>       devices using cached or synthetic intermediate states, breaking the sequential lock of training deep transformers.
>    6. Why Block AttnRes? More or less because it has proven to always be useful. It allows blocks to dynamically attend
>       to any other block without the signal dying or exploding. Especially for perturbative methods, this is
>       non-negotiable if the model is supposed to scale to many layer. It naturally complements DiffusionBlocks.

## 2. Gemini's five suggestions (summarised from `notes/gemini_1.png` … `gemini_6.png`)

The question put to Gemini: "What might be ways to lower its cost by orders of magnitude without losing its qualitative
advantages?" Its framing: the root problem is the variance of high-dimensional random sampling, which scales with the
dimension of the perturbation and with the length of the credit-assignment path. It promised 10–100×.

1. **Low-rank and intrinsic-subspace perturbation.**
   - **Idea:** perturb Δz = U·ε with U ∈ ℝ^{d×k} and ε ∈ ℝ^k (k = 16 or 32), not isotropically over all d.
   - **Choice of U:** a fixed random projection, a low-rank adapter (LoRA), or a running PCA of recent activations.
   - **Claim:** representations lie on low-dimensional manifolds, so the population can drop from thousands to dozens.
2. **Structured and orthogonal exploration (quasi-Monte-Carlo).**
   - **Idea:** use mutually orthogonal perturbations within a batch (Structured Evolution Strategies) and low-discrepancy
     sequences (Sobol) in place of i.i.d. Gaussians.
   - **Claim:** convergence goes from O(1/√P) toward O(1/P).
3. **Local loss heads and truncated credit assignment.**
   - **Idea:** add auxiliary local objectives (reconstruction or next-token) every 2–4 blocks. A perturbation is scored
     against the change in its LOCAL loss, not the end-to-end loss.
   - **Claim:** variance "grows exponentially with the depth of propagation", and localising removes attention
     cross-talk.
4. **Sparse, entropy-gated token perturbation.**
   - **Idea:** perturb only tokens whose predictive entropy is above a threshold, and run the low-entropy tokens clean.
   - **Claim:** this concentrates the population on the tokens where the model is deciding, and cuts cross-token
     interference.
5. **Covariance adaptation in activation space (low-rank CMA).**
   - **Idea:** keep an exponential moving average of successful perturbation directions (momentum in noise space), and
     tilt future noise toward them.
   - **Claim:** this is intractable in weight space but practical in activation space (D ~ 10³, k ~ 10).

## 3. Where Dust's cost comes from (my analysis)

**The core fact.** Each draw gives each token ONE number (its loss change), but what it estimates there is a vector as
wide as the perturbed layer, d. With K isotropic Gaussian directions, the averaged estimate at one token has
cos² = K/(K + d) with the true error: signal |g|², noise ≈ d·|g|²/K.

**Over a batch.** The weight update sums tokens: G = Σ_t g_t x_tᵀ. Its noise adds up incoherently across tokens, while its
signal adds up as coherently as the per-token errors agree. So the update has cos² ≈ Kρ/(Kρ + d), with ρ the "agreement"
of the tokens (1 if independent, T if identical). This is the paper's fitted law cos(K) = c_max/√(1 + c/K), with
c ≈ d/ρ. Momentum then averages the remaining noise over steps, at the price of more steps.

**Caution, from the paper's own table.** Going from d = 128 to d = 1024 did NOT make Dust worse relative to backprop at
equal draws (`DUST.md` §4). So the effective d may be the RANK of the errors the network actually produces, not the
nominal width. This is measurable, and it decides whether subspace ideas (Gemini 1 and 5) can pay.

**Interference.** Other perturbations reach a token's loss through attention, which adds noise without changing the
scaling.

**Three levers.** Every way to cut the cost is one of:
- **A.** Fewer dimensions per measurement.
- **B.** More independent numbers per draw.
- **C.** Start the search near the answer.

## 4. Gemini's five, rated before testing (predictions are in `EXPERIMENT_1.md`)

1. **Low-rank.** A FIXED random k-dimensional subspace caps the cosine at √(k/d), however many draws; for d = 4096,
   k = 32, that is ≤ 0.09. A PCA of activations is not obviously where the ERRORS are. Only a subspace learned from the
   errors can pay, and that is idea 5.
2. **Orthogonal or quasi-Monte-Carlo.** A constant factor at most. With K ≪ d, Gaussian directions are already nearly
   orthogonal, and quasi-Monte-Carlo's O(1/P) is a low-dimensional result. Antithetic ±a pairs are a cheap,
   well-known ~2×.
3. **Local losses.** Real, and the best evidenced: Ren, Kornblith, Liao & Hinton, *Scaling forward gradient with local
   losses*, ICLR 2023.
   - **Method:** activity perturbation with ~250k local losses (block-, patch-, channel-group-wise).
   - **Result:** it matched backprop on MNIST and CIFAR-10.
   - **What made it work:** per-block losses alone were not enough; the narrow GROUP losses were. They wrote that variance
     grows with the number of hidden dimensions, not exponentially with depth.
   - **The price:** a local loss is not the global loss (the greediness problem).
4. **Entropy gating.** Small. A draw still reruns the whole sequence, so it is not cheaper, and the dimension at the kept
   tokens is unchanged. It only cuts interference.
5. **Covariance adaptation.** This is 1 done right. The formal version is Guided ES (Maheswaranathan, Metz, Tucker, Choi
   & Sohl-Dickstein, ICML 2019):
   - **How:** sample from a distribution stretched along a subspace of surrogate gradients, plus isotropic noise.
   - **Result:** the estimate is Σg, a descent direction even when the surrogate is biased.
   - **Condition:** one covariance shared over tokens pays only to the extent the per-token errors share a subspace.

## 5. My own ideas (to test after Gemini's)

- **O1. Activation sparsity cuts the dimension directly (lever A).**
  - **The mechanism:** if only k of d units are active at a token, and a small perturbation of an inactive unit changes
    nothing, perturb only the active units (and those near threshold). The dimension per token becomes k.
  - **The gain:** at 1–5% active, 20–100× fewer draws for the same cosine. The outer product also touches only active
    inputs.
  - **What it means for the plan:** the plan's two halves, Dust and event-driven sparsity, multiply.
  - **Why the near-threshold band matters:** it is what lets an inactive unit be recruited. Perturbation sees the effect
    of crossing a threshold; backprop cannot without a straight-through surrogate.
- **O2. Don't estimate what is known in closed form.**
  - **The head:** Dust spends 4 × population draws on the vocabulary head, whose output error is exactly
    softmax − one-hot.
  - **Vector targets:** a block with a vector target (DiffusionBlocks) knows its output error exactly: output − target.
  - **Where perturbation remains:** only inside blocks and through the parts that really are not differentiable.
- **O3. The top error as a free guess for every layer (lever C).**
  - **The guess:** in a residual network, the exact error at the top of the residual stream is a natural per-token guess
    for the error at every writer below it, because the identity path carries it down unchanged (it resembles Direct
    Feedback Alignment).
  - **How it is used:** spend some draws along the guess (Guided ES); the rest stay isotropic.
  - **What it costs:** no backward pass and no stored history beyond the current token.
  - **Unlike idea 5,** the guess is PER TOKEN, so it needs no shared subspace.
- **O4. Narrow groups, each with its own score (levers A + B).**
  - **The mechanism:** split a layer into M groups, each scored by its own local error. That gives M numbers per draw,
    each about d/M dimensions.
  - **Existing versions:** Ren et al.'s group losses; Dust's per-head scoring of attention internals.
  - **The open problem:** what local error each group gets.
- **O5. Reuse across steps (lever C).** Errors change slowly from step to step, so last step's estimates, or the momentum
  buffer, can serve as the Guided-ES subspace.
- **O6. Reward shaping** (from the user's plan: rank or sign of the loss deltas). It bounds the influence of heavy-tailed
  loss changes, from avalanches in a critical sparse network. It is a robustness measure, not a variance cut; test it
  when sparse dynamics exist.

## 6. Which of the qualitative advantages hold, and on what condition

- **"No record of activations" holds only partly.**
  - **Dust's speed trick** reruns block l onward from CACHED clean activations, and that is a record.
  - **Strictly forward-only** means one of two things:
    - rerunning each draw from the start, which costs more;
    - or perturbing all layers in one pass, which raises the dimension per measurement.
  - **A looped model** must keep step i's state to rerun from step i, which is BPTT's memory again, unless targets are
    block-local.
- **"Hard logic without a surrogate" holds in the useful sense.** Perturbation with width σ follows the gradient of the
  loss averaged over its noise. That is a smoothing set by the noise rather than a hand choice, and it is exact if the
  same noise is present at run time.
- **One differentiability assumption remains.** The update ĝ xᵀ assumes each layer is linear in its weights. Everything
  between the perturbed output and the loss can be anything.
