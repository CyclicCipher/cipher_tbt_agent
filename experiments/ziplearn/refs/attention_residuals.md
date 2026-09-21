# Attention Residuals (AttnRes) — reference notes

**Paper:** Kimi Team (Moonshot AI), *Attention Residuals*, technical report, arXiv:2603.15031, submitted 2026-03-16.
Authors: Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, Siyuan Pan, … Zhilin Yang, Xinyu Zhou (36 authors).
**License:** CC BY-NC-ND 4.0 (verbatim copies allowed with attribution, non-commercial, no adaptations — which is why
this file is notes plus the abstract, and the full text is the PDF beside it).
**Links:** [arXiv abstract](https://arxiv.org/abs/2603.15031) · [PDF](https://arxiv.org/pdf/2603.15031) ·
[official repo (README + pseudocode)](https://github.com/MoonshotAI/Attention-Residuals) ·
local copy: `attention_residuals_2603.15031.pdf` (21 pages, 1.07 MB, downloaded 2026-09-20).

## Abstract (verbatim)

> Residual connections with PreNorm are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit
> weights. This uniform aggregation causes uncontrolled hidden-state growth with depth, progressively diluting each
> layer's contribution. We propose Attention Residuals (AttnRes), which replaces this fixed accumulation with softmax
> attention over preceding layer outputs, allowing each layer to selectively aggregate earlier representations with
> learned, input-dependent weights. To address the memory and communication overhead of attending over all preceding
> layer outputs for large-scale model training, we introduce Block AttnRes, which partitions layers into blocks and
> attends over block-level representations, reducing the memory footprint while preserving most of the gains of full
> AttnRes. Combined with cache-based pipeline communication and a two-phase computation strategy, Block AttnRes becomes
> a practical drop-in replacement for standard residual connections with minimal overhead. Scaling law experiments
> confirm that the improvement is consistent across model sizes, and ablations validate the benefit of
> content-dependent depth-wise selection. We further integrate AttnRes into the Kimi Linear architecture (48B total /
> 3B activated parameters) and pre-train on 1.4T tokens, where AttnRes mitigates PreNorm dilution, yielding more
> uniform output magnitudes and gradient distribution across depth, and improves downstream performance across all
> evaluated tasks.

## The method, in our words

**Vocabulary the paper uses.** A *layer* is one self-attention OR one MLP (so a transformer block = 2 layers). h_l is
the vector entering layer l (per token); f_l(h_l) is that layer's output; the token embedding is h_1. A *source* is
something a layer may read from: the embedding, or an earlier layer's output.

**Standard residual (PreNorm).** h_l = h_{l−1} + f_{l−1}(h_{l−1}), so unrolled h_l = h_1 + Σ_{i<l} f_i(h_i): every layer
reads the *same* uniformly weighted sum of everything before it. The paper's three complaints: no selective access
(attention and MLP layers get the same mix), irreversible loss (what the sum buries cannot be recovered later), and
output growth (late layers must shout to be heard over a sum that grows as O(L) with depth — "PreNorm dilution").

**Full AttnRes (§3.1 of the paper).** The input to layer l is a softmax-weighted mix of the raw sources:

    h_l = Σ_{i=0}^{l−1} α_{i→l} · v_i,      v_0 = h_1 (embedding),  v_i = f_i(h_i) for i ≥ 1
    α_{i→l} = softmax_i ( w_l · RMSNorm(v_i) )

- **w_l** is a single learned vector in R^d per layer — the "pseudo-query". It is a parameter, not computed from the
  current token, so the weights for a whole group of layers can be computed in parallel.
- The **keys** are the RMSNorm'd sources (so a layer with a large-magnitude output cannot dominate the softmax); the
  **values** are the raw, un-normalised sources.
- The weights *are* input-dependent, through the keys: the same w_l gives different α per token.
- **Initialisation: w_l = 0 for every layer** ("crucially"), so at step 0 every α is uniform and AttnRes starts as an
  equal-weight average — the paper reports training volatility otherwise.
- Cost: per token O(L²d) arithmetic, O(Ld) memory for stored sources — nothing extra in vanilla training since those
  activations are kept for backprop anyway.

**Block AttnRes (§3.2).** Partition the L layers into N blocks of S layers. *Inside* a block the layer outputs are
summed as usual into a running partial sum b_n^i; *across* blocks, attention is over the block representations
[b_0 = embedding, b_1, …, b_{n−1}] plus, for every layer after the first in a block, the current partial sum. Memory and
communication drop from O(Ld) to O(Nd). N = L recovers Full AttnRes; N = 1 is the standard residual with the embedding
kept separate as b_0. The paper's pseudocode (`block_attn_res` / `forward` in the README) applies the mix twice per
transformer block — once before the attention sub-layer with its own w and RMSNorm, once before the MLP with another —
and appends the completed partial sum to the block list at each block boundary. `block_size` in that code counts
attention + MLP, so one transformer block per AttnRes block is `block_size = 2`.

## Numbers worth remembering (from the paper's tables)

Scaling laws (five sizes, 194M–528M activated, 8192 context, all hyperparameters chosen for the baseline):
baseline L = 1.891·C^−0.057, Block AttnRes (N≈8) L = 1.870·C^−0.058, Full AttnRes L = 1.865·C^−0.057 — same slope,
lower intercept; Block AttnRes matches a baseline given **1.25× the compute**. The Full/Block gap shrinks to 0.001 at
the largest size.

Ablations on the 16-layer (436M) model, validation loss:

| variant | loss |
|---|---|
| baseline (PreNorm) | 1.766 |
| DenseFormer (all earlier layers, fixed input-independent scalars) | 1.767 |
| mHC (m parallel streams, learned mixing) | 1.747 |
| **Full AttnRes** | **1.737** |
| … with an input-dependent query (a d×d projection per layer) | 1.731 |
| … with input-independent mixing (no query/key, learned scalars) | 1.749 |
| … with sigmoid instead of softmax | 1.741 |
| … without RMSNorm on the keys | 1.743 |
| sliding window over the last 8 layers + embedding | 1.764 |
| Block AttnRes, S = 4 | 1.746 |
| … multi-head over depth (16 heads) | 1.752 |
| … without RMSNorm | 1.750 |

Block size sweep (same model): S = 1 → 1.737; S = 2, 4, 8 → ≈ 1.746; S = 16, 32 → toward baseline. The lesson the
authors draw: *selective access to distant layers* matters more than access to many nearby ones (the sliding window
barely helps); *softmax's competition* matters (sigmoid is worse); the optimal depth mix is uniform across channels
(multi-head over depth hurts: "when a layer's output is relevant, it is relevant as a whole").

Downstream, Kimi Linear 48B/3B on 1.4T tokens with Block AttnRes (6 layers per block, 9 blocks + embedding = 10
sources): gains on every benchmark; largest on multi-step reasoning (GPQA-Diamond 36.9 → 44.4) and code (HumanEval
59.1 → 62.2). The authors' reading: better depth-wise flow "benefits compositional tasks, where later layers can
selectively retrieve and build upon earlier representations".

Training dynamics: baseline output magnitudes grow monotonically with depth; Block AttnRes gives a bounded, periodic
pattern (each block boundary resets the accumulation); gradient norms are far more uniform across depth.

Learned routing (their Fig. 8, 16-layer model): each layer attends most to its immediate predecessor (locality), the
embedding keeps non-trivial weight throughout (especially before attention layers), and a few off-diagonal
concentrations appear — learned skip connections. Pre-MLP mixes are sharper and more local than pre-attention mixes.

## How it maps onto ZipLearner (see `../DESIGN.md` §6, §9, §15)

- The **routing weights are a wiring diagram we can read directly**, instead of inferring reads-from wiring from overlaps
  of read and write subspaces (what `inner_objective/inspect_weights.py` had to do).
- With sharp routing, a layer's input is a *chosen* earlier output — DESIGN §6's pinned interface becomes literal, and
  running a layer backwards gives a clean target for the layer it read from.
- ~~In the ZipLearner-*written* network the route is a **fixed choice per layer**, priced at log₂(number of sources) bits~~
  WITHDRAWN 2026-09-21 (re-read, §5.3 and Fig. 8): the paper's own ablation puts a fixed route (DenseFormer 1.767; learned
  scalars without query/key 1.749) at or near the baseline — the gain is content-dependent selection. The written route
  is therefore a pseudo-query over content: in our block, over the token-class and state flags, i.e. a hard per-token
  branch priced as a small table over those flags (DESIGN §15, "What the attention residuals learn").
- The paper's "block" (a unit whose summed output can be selected) and DESIGN §9's "block" (rows for one context) are
  the same object seen from two sides: a selectable output per block is the substrate "context selects the block" needs.
- Cautions the ablations give us: keep the RMSNorm on keys (magnitude differences between sources otherwise bias the
  choice); initialise every pseudo-query to zero; do not split the depth mix per head.
