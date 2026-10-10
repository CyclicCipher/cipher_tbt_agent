# Dust — pretraining transformers without backpropagation (qlabs, 2026)

Source: the web write-up https://qlabs.sh/research/dust (no PDF exists) and the code https://github.com/qlabs-eng/dust
(MIT; `model.py`, `dust.py`, `baselines/backprop.py`). Read 2026-10-10 through WebFetch, which returns summaries of the
page and of `dust.py`, not the full text. The user brought it with their own training-method plan and five cost-cutting
suggestions from Gemini (screenshots, same day). Their plan:
- Dust in place of backprop, for three reasons: no differentiability constraint, no backward pass, no stored history for
  recurrent training.
- Event-driven sparsity, with 1–5% of units active, as in critical brain dynamics.
- DiffusionBlocks combined with Block Attention Residuals.

## What Dust does

For a linear layer y_t = W x_t:
- Add σ·a_t to the output at every token t, with a_t ~ N(0, I).
- Rerun the network from that layer on, and record each token's loss change.

From these:
- **Reward.** Reward = clean loss − perturbed loss, centred by the mean over the draws in the same chunk. The reward for
  the jitter at token t also counts later tokens: r_t = Σ_{s≥t} γ^{s−t} c_s.
- **Output-error estimate.** The error at the layer's output is ĝ_t = −(1/(Kσ)) Σ_i r_t^(i) a_t^(i), over K draws.
- **Update.** The update is Σ_t ĝ_t x_t^T. This is the same outer product that backprop forms. **Only the output error
  ĝ is estimated;** the rest is exact.

Each token is a separate measurement, so one pass gives thousands of "population members". That makes it at least about
1000× more population-dense than weight-space evolution strategies (EGGROLL).

Details from the code:
- **Separate passes.** Each layer type is perturbed in its own passes.
- **Draws per block** (population 256), with earlier blocks getting more:
  - writers (`attn.c_proj`, `mlp.c_proj`): [38, 32, 16, 10, 6, 4, 2, 2];
  - attention-output sites: [6, 6, 4, 2, 2, 2, 2, 2].
- **A draw for block l reruns blocks l onward from cached clean activations.**
- **Attention internals** (q, k, v, gate, value embeddings) are scored per head by alignment with the estimated
  attention-output error, c_s = −⟨ĝ_s, Δo_s⟩. Keys and values get γ ≈ 0.98, because later tokens read them.
- **Vocabulary head:** jittered on cached logits, one 256-column slab per draw, with 4 × population draws.
- **The 2L residual-mixing scalars** use ordinary two-sided weight-space ES.
- **Noise scales σ:** 0.2 for writers, hidden and embedding; 0.4 for deep attention-output sites; 0.05 for the head and
  attention internals.
- **Optimiser:** SGD with momentum 0.95–0.98, lr ≈ 0.25. Adam was also tested.

**Results.**
- **Setup:** 8 layers × width 512, FineWeb, BPE-4096, 16k-token batches, one epoch, three seeds.
- **Small budgets:** at 100k and 1M tokens, Dust reaches lower loss than backprop once it has a few hundred to ~1,000
  draws.
- **Larger budgets:** at 10M–20M tokens, the gap to backprop shrinks as the population grows. The fitted limit at 20M is
  4.43 against backprop's 4.63; the authors call the fit loosely constrained.
- **Gradient cosine:** cosine to the backprop gradient follows cos(K) = c_max / √(1 + c/K) for every layer type.
- **Model size:** larger models (2M–243M parameters were tested) are more population-efficient, not less.
- **Stated limits:**
  - not yet compute-efficient enough to replace backprop;
  - tuning is expensive;
  - looped models and models that are not differentiable are left to future work.

## Where the cost comes from (my analysis)

**One token.** Each draw gives each token ONE number, its loss change, about a d-dimensional vector, the error at a
width-d output. With K isotropic Gaussian directions, the averaged estimate has cos² = K / (K + d) with the true error:
- signal |g|²;
- noise ≈ d·|g|²/K.

**Over the batch.** Over a batch, the update sums tokens, so cos² ≈ Kρ / (Kρ + d). Here ρ measures how much the
per-token errors agree: 1 if they are independent, T if they are all the same. This is the paper's fitted law with
c ≈ d/ρ.

**Width stays in the numerator.** That is the width scaling. Momentum averages the remaining noise over steps, at the
price of more steps. Interference adds to the noise without changing the d scaling: a jitter at one token reaches other
tokens' losses through attention.

**Every way to cut the cost is one of three levers:**
- **A.** Fewer dimensions per measurement.
- **B.** More independent numbers per draw.
- **C.** Start the search near the answer.

## The five suggestions (Gemini), rated

1. **Low-rank random projection (A).** A FIXED random k-dimensional subspace caps the cosine at √(k/d), however many draws
   are used. Going from 4096 to 32 dimensions gives ≤ 0.09. It pays only if the subspace is where the errors are, which
   must be learned, so it is the same idea as 5.
2. **Orthogonal or quasi-Monte-Carlo directions: a constant factor at most.** With K ≪ d, independent Gaussian directions
   are already nearly orthogonal. Quasi-Monte-Carlo's O(1/P) rate is a low-dimensional result. Antithetic ±a pairs are a
   cheap ~2×.
3. **Local losses (B).** This one has the best evidence. Ren, Kornblith, Liao & Hinton, *Scaling forward gradient with
   local losses* (ICLR 2023):
   - **Method:** activity perturbation plus many local losses, block-, patch- and channel-group-wise, ~250k losses in
     one network.
   - **Result:** it matched backprop on MNIST and CIFAR-10, and beat other backprop-free methods on ImageNet.
   - **What made it work:** depth-wise losses alone were not enough. The authors state that variance grows with the
     number of hidden dimensions, so the GROUPS make it work: each group is its own number per draw, over fewer
     dimensions.

   The screenshot's claim that variance grows exponentially with depth is not the mechanism; the dimension is.
4. **Entropy-gated tokens: small.** Skipping tokens does not make a draw cheaper, because the rerun covers the whole
   sequence. It does not cut the dimension at the tokens kept either. It only cuts interference.
5. **Covariance adaptation in activation space (C, and A).** This is 1 done right. The formal version is Guided ES
   (Maheswaranathan, Metz, Tucker, Choi & Sohl-Dickstein, ICML 2019):
   - **Search distribution:** stretched along a subspace of surrogate gradients, plus some isotropic noise.
   - **Output:** Σg, which stays a descent direction when the surrogate is biased.

   One covariance shared across tokens pays only to the extent the per-token errors share a subspace. That is
   measurable (below).

## Levers not in the list

- **Activation sparsity cuts the dimension directly (A).**
  - **The mechanism:** if only k of d units are active at a token, and a small perturbation of an inactive unit changes
    nothing, only the active units need perturbing, plus those near threshold. The dimension per token is then k, not d.
  - **The gain:** at 1–5% active, that is 20–100× fewer draws for the same cosine. The outer product ĝ x^T also touches
    only active inputs.
  - **So the two halves of the user's plan multiply:** Dust and event-driven sparsity.
  - **Why the near-threshold band matters:** it is what lets an inactive unit be recruited. Perturbation sees the effect
    of crossing a threshold, which backprop through a hard threshold cannot see without a straight-through surrogate.
  - **The plan's rank or sign shaping of the reward** (against heavy-tailed avalanches) keeps the estimate a descent
    direction, as in Natural Evolution Strategies.
- **Narrow groups, each with its own score (A + B).**
  - **The mechanism:** split a width-d layer into M groups, each scored by its own local loss. That gives M numbers per
    draw, each about d/M dimensions, so M× fewer draws.
  - **Existing versions:** Ren et al.'s channel-group losses are this. Dust's per-head scoring of attention internals is
    a version of it.
  - **The design problem moves** to what local loss each group can have.
  - **Related:** the factorised heads of OPF (`jepa_anything_opf_2609.20800.md`) and the old line's TBT columns.
- **Do not estimate what is known in closed form.**
  - **The head:** Dust spends 4 × population draws on the vocabulary head, whose output error is exactly softmax −
    one-hot.
  - **Block outputs:** a DiffusionBlocks block is trained to predict a clean target from a noisy one. Its output error
    is known exactly: output − target.
  - **Where perturbation is still needed:** only inside a block and through the parts that really are not
    differentiable.
  - **What it keeps:** no global backward pass, and no history stored across blocks.
- **Reuse across steps (C).** Errors change slowly from step to step, so recent estimates, or the momentum buffer, can
  serve as the surrogate for Guided ES.

## Which advantages hold, and on what condition

- **"No record of activations" holds only partly.**
  - **Dust's speed trick** reruns block l onward from CACHED clean activations, and that is a record.
  - **Strictly forward-only** means one of two things:
    - each draw reruns from the start, which costs more;
    - or all layers are perturbed in one pass, which raises the dimension per measurement to that of the whole network.
  - **In a looped model,** rerunning from loop step i needs step i's state: that is BPTT's memory again, or a
    checkpoint.
  - **With block-local targets (DiffusionBlocks),** there is no rerun across blocks at all, which is one more reason the
    two fit.
- **"Hard logic without a surrogate" holds in the useful sense.** With noise of width σ, the method follows the gradient
  of the loss averaged over that noise. That is a smoothed version of the hard system, but the noise sets the smoothing,
  not a hand choice such as straight-through. If the same noise is present when the model runs, it is exactly the loss
  of the system that runs.
- **One differentiability assumption remains.** The update ĝ x^T assumes each layer is linear in its weights, y = Wx.
  Everything between the perturbed output and the loss can be anything.

## For our models

The DistributionPrediction models are 64 wide, 8× narrower than Dust's 512, so the d in the noise is 8× smaller. How many
draws a 2-minute run would need here is unmeasured. Three cheap measurements would turn the levers into numbers before
anything is built. Each takes seconds of GPU on an already-trained model, with backprop used only as the measuring stick:

1. **Cosine against K** for plain node perturbation, per layer site. This fits c and ρ at our width.
2. **Effective rank of the per-token output errors** (a T × d matrix) per site. This gives the most an informed subspace
   (suggestions 1 and 5, Guided ES) could buy: d / rank.
3. **The MLP with a top-5% activation**, perturbing only the active units vs all units: cosine against K. This checks
   the sparsity lever's d/k claim.
