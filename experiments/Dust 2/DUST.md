# Dust — everything collected about the method

**What it is.** *Dust: Pretraining Transformers Without Backpropagation*, by Samip Dahal, Bishwas Mandal, Serdar Gülbahar
and Akshay Vegesna (Q Labs Research, October 2026; correspondence s@qlabs.sh).

**Sources.**
- Web write-up: https://qlabs.sh/research/dust. There is no PDF, and the appendices (A–G, including the full Table 1) are
  not on the page.
- Code: https://github.com/qlabs-eng/dust (MIT; `model.py`, `dust.py`, `baselines/backprop.py`, `prepare_data.py`,
  `tokenizer/`).

**How it was read.** Read 2026-10-10 through WebFetch. WebFetch returns summaries of a page, not its full text, so the
numbers below are as the summaries reported them. Where a detail was not on the page, this file says so.

## 1. The estimator (node perturbation, per token)

For a linear layer y_t = W x_t, at token t:

1. **Perturb.** Add σ·a_t to the layer's output at every token, with a_t ~ N(0, I).
2. **Rerun.** Rerun the network from that layer on, and record each token's loss.
3. **Reward.** Reward per token = clean loss − perturbed loss, centred by subtracting the mean over the draws in the
   same chunk. Call the centred value c_s.
4. **Temporal credit.** The reward for the jitter at token t counts later tokens, discounted:
   r_t = Σ_{s≥t} γ^{s−t} c_s. With γ = 0, only its own token counts.
5. **Output error.** The estimated error at the layer's output is
   ĝ_t = −(1/(Kσ)) Σ_{i=1..K} r_t^(i) a_t^(i).
6. **Weight gradient.** The weight gradient is Ĝ_W = Σ_t ĝ_t x_t^T. This is **the same outer product backprop forms;
   only the output error is estimated** (by the chain rule in backprop, by the population in Dust). For embeddings, x is
   one-hot, so the update is a scatter-add.
7. **Assembly.** Gradients are divided by the number of valid tokens and averaged across GPUs. A non-finite value
   raises an error.

**Why it is dense.** Each token is a separate population member, so one forward pass evaluates thousands of members.
That makes Dust at least about 1000× more population-dense per forward pass than weight-space evolution strategies.

## 2. How the population is spent (from `dust.py`)

- **What a population is.** A population is K "draws". One draw jitters all tokens of the selected layer(s) in one
  pass.
- **Cost of a draw.** A draw is slightly cheaper than a forward pass: the clean activations are cached, and a draw for
  block l reruns only blocks l onward.
- **Separate passes.** Each layer type is jittered in its own passes, with its own σ and its own share of the
  population. This is to reduce *interference*: the loss change from one perturbation mixed with the effects of the
  others in the same pass.
- **Draws per block, population 256** (scaled by population/256 for larger populations):
  - writers (`attn.c_proj`, `mlp.c_proj`): [38, 32, 16, 10, 6, 4, 2, 2];
  - "hubs" `o@l` (a virtual site: the attention output before `c_proj`): [6, 6, 4, 2, 2, 2, 2, 2];
  - hidden (`mlp.c_fc`): per-block counts;
  - embedding: 10 × scale.

  Early blocks get far more draws than late ones. The 10M/16384 override uses writers [34, 28, 14, 8, 6, 4, 2, 2],
  hubs [12, 10, 8, 4, 4, 4, 4, 4], and γ 0.99.
- **Attention internals** (q, k, v, gate, value embedding) get per-block counts [2, 20, 6, 12, 8] × scale. Their token
  losses barely respond to the jitter, so they are not scored by the loss:
  - each block's attention output is recomputed from cached clean activations for each draw;
  - each head is scored by alignment with the estimated error at the hub, c_s = −⟨ĝ_s, Δo_s⟩, where Δo_s is the change
    in the attention output;
  - q keeps the score on its own token, while k, v, gate and value embedding sum it over later tokens with credit
    γ^lag, inside the causal window;
  - the gate uses a two-sided sigmoid perturbation (divisor 2σ); the others use σ.

  This is a LOCAL score: a known (estimated) error downstream is used in place of the loss.
- **Head.** The head is jittered on cached logits, one 256-column vocabulary slab per chunk. The partition function is
  rebuilt from cached slab sums and the cross-entropy recomputed. It gets 4 × population draws, which are not counted in
  K.
- **Residual-mixing scalars.** The 2L of them (`resid_lambdas`, `x0_lambdas`) use two-sided weight-space ES, σ 0.03,
  8 draws.
- **Chunk sizes** (`nrep`, the number of draws batched per pass): population 256 → 2, 1024 → 8, 4096 → 8,
  16384 → 16.
- **Noise scales σ:**
  - 0.2 for writers, hidden, embedding and shallow hubs;
  - 0.4 for hubs at layer ≥ L/2;
  - 0.05 for attention internals;
  - 0.05 for the head, with logits soft-capped 15·tanh(x/15).
- **Credit decay γ:** default 0.98.
- **Optimiser.** SGD with momentum, lr ≈ 0.175–0.25, momentum 0.95–0.98, constant. Per-group learning rates:
  embedding 1000, value embeddings 0.3, lambdas 0.03. Adam was also tested.
- **Tuning.** Hyperparameters were tuned in one of two ways:
  - a training grid search on small budgets (reliable, expensive);
  - maximising cosine to the backprop gradient on one batch.

  Cosine picks candidates, and training decides: higher cosine does not always mean lower loss. The settings mostly
  transfer across budgets, except at the largest population.

## 3. Model (from `model.py`)

- **Shape.** GPT-style decoder: vocabulary 4096 (BPE), width d = 512, 8 heads of 64, 8 layers.
- **Blocks.** Pre-norm blocks; the norm is `F.rms_norm` with no learnable weight. It is applied to:
  - the embedding;
  - before attention and before the MLP;
  - q and k per head (QK-norm, after rotary);
  - the final output.
- **Attention.**
  - rotary positions;
  - sliding windows SSSL (1024 / 2048);
  - value embeddings on alternate layers, gated by `2·sigmoid(ve_gate(x[..., :32]))`.
- **MLP.** `c_proj(relu(c_fc(x))²)`, 4× width. Its hidden layer is sparse by construction, because ReLU zeroes about
  half of it.
- **Residual mixing.** Before block i, x = resid_lambdas[i]·x + x0_lambdas[i]·x0.
- **Linear layers.** No biases and no tied weights.
- **Init.**
  - embedding N(0, 1);
  - head N(0, 0.001);
  - q, k, v, fc uniform ±√3/√d;
  - projections zeroed, then re-initialised externally.

## 4. Results reported

- **Setup.** FineWeb, 16k-token batches (8 × 2048), one epoch, three seeds; backprop tuned on the same grid.
- **Small budgets.** At 100k and 1M tokens, Dust reaches LOWER loss than backprop: at 100k with a few hundred draws,
  at 1M with about 1,000.
- **Large budgets.** At 10M and 20M tokens, the gap to backprop shrinks as the population grows. At 20M, a power-law fit
  gives a limit of 4.431 (95% interval 3.89–4.58) against backprop's 4.633; the fit is loosely constrained. At 10M, the
  fitted limit is "just above" backprop.
- **Adam at 1M tokens.** Dust's limit is 5.248 (5.17–5.30), against backprop's 5.361.
- **Model size, test loss at 10M tokens:**

| population | L2/d128 (2.0M) | L4/d256 (7.3M) | L8/d512 (38M) | L16/d1024 (243M) |
|---|---|---|---|---|
| 64 | 5.705 | 5.556 | 5.558 | 5.719 |
| 256 | 5.486 | 5.362 | 5.358 | 5.419 |
| 1k | 5.265 | 5.161 | 5.158 | 5.214 |
| 4k | 5.189 | 5.095 | 5.053 | 5.124 |
| 16k | 5.171 | 5.065 | 5.036 | 5.086 |
| backprop | 5.180 | 5.066 | 5.015 | 5.048 |

  Larger models are more population-efficient: past 1k draws, the 38M and 243M models gain about 30% more than the
  small ones. The gap to backprop grows only slightly with size. **Note the widths:** going from d128 to d1024 did not
  make Dust worse relative to backprop at equal population. That is a caution against reading the per-step cost as a
  simple "∝ width" law (see `IDEAS.md` §3).
- **Gradient alignment.** Cosine to the backprop gradient rises with population for every layer type, at 10M, 100M and
  1B tokens. It follows cos(K) = c_max/√(1 + c/K), with RMSE < 0.06. The per-type values of c_max and c are in Appendix
  G, which is not on the page.
- **EGGROLL comparison.**
  - **What EGGROLL is:** low-rank weight perturbation, one sequence per member, tuned on the same transformer.
  - **Final loss:** at 16k members it trails Dust at 64 draws by 0.4–0.6 at 1M/10M/20M tokens.
  - **Gradient cosine:** below 0.05 at 128k members for every layer type except the head.
  - **Adam:** it gains almost nothing from Adam.
  - **Extrapolated efficiency:** Dust is about 10³–10⁴× more efficient from 1M tokens up.
- **Why Dust sometimes beats backprop.** The mechanism is unclear. The authors suggest Dust explores the landscape and
  picks up curvature that pulls it toward flat regions, so it follows a different trajectory.

## 5. Limitations and future work (as stated)

**Limitations:**
- not yet compute-efficient enough to replace backprop, and efficiency was not the focus;
- interference and tuning add cost;
- cosine does not always predict loss;
- Dust's gradients approach but do not match backprop's, and the cosine varies with layer type and depth;
- the extrapolations are loosely constrained;
- the experiments are language-model pretraining only.

**Future work:**
- curvature: whether Dust can find better directions than backprop;
- architectures that are not end-to-end differentiable: looped or recurrent models, models with external programs in
  the loop;
- compute efficiency, which needs orders of magnitude;
- optimisers co-designed with Dust.

## 6. Not known (not on the page)

- the full Table 1;
- the per-layer cosine fit constants;
- wall-clock numbers;
- a stated compute ratio to backprop;
- the appendices on draw costing (E) and the fits (G).

The user's estimate, from their notes: about 1000× backprop's compute or more.
