# Neural Turing Architecture — references

Working name for a standalone experiment: a looped transformer (Ouro-style depth recurrence, after Chen et al.'s
boundary-operator results) that also thinks in continuous space along the sequence (Coconut), learns to do so WITHOUT a
language-substitution curriculum or human reasoning traces, and may replace its attention with a mutable, sparsely
addressed memory (Percepta's Spotlight). The brainstorm is `BRAINSTORM.md`; this file is only the sources.

Compiled 2026-10-03 from a cloud session. **That session could not open `arxiv.org`, `huggingface.co`, `alphaxiv.org`
or `percepta.ai`** (the environment's network policy blocks them), so the entries are checked at different depths.
The status column says how far:

| status | meaning |
|---|---|
| **repo** | a note (sometimes the PDF) is already in this repository — path given; read in full this session |
| **web** | title, arXiv ID, venue and the headline claims confirmed by web search on 2026-10-03; the paper itself not opened |
| **unchecked** | cited from memory; confirm the ID before quoting it |
| **secondary** | the primary source was blocked; what is written comes from coverage of it, and says so |

---

## 1. The intent: looping and continuous thought

| ref | status | what we take from it |
|---|---|---|
| **Ouro** — *Scaling Latent Reasoning via Looped Language Models*, arXiv [2510.25741](https://arxiv.org/abs/2510.25741) (Oct 2025; v5 Jul 2026) | web | The "Ouro-style" in the intent. A shared block iterated in latent space during PRE-training; learned depth via an exit gate trained in two stages — first an entropy-regularised objective so every depth gets explored, then a stage that tunes the gate on realised gains. 1.4B / 2.6B models, 7.7T tokens, matching 4–8B dense models on reasoning; the authors attribute the gain to knowledge *manipulation*, not storage. |
| **Coconut** — Hao et al., *Training LLMs to Reason in a Continuous Latent Space*, arXiv [2412.06769](https://arxiv.org/abs/2412.06769) | repo `experiments/ziplearn/refs/coconut_2412.06769.md` | The sequence-axis recurrence: last hidden state fed back as the next input embedding. Its recorded hazards are this experiment's targets: needs the CoT-substitution curriculum (without it "no better than no-CoT"), loss spikes at c = 3 thoughts per step, thought count is a pad, raw state fed back unnormalised. |
| **Chen, Vegesna, Dahal, Wilson** — *How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents*, arXiv [2609.19107](https://arxiv.org/abs/2609.19107) | repo `experiments/ziplearn/refs/scaling_exponents_recursion.md` | Prelude → looped core → coda; the boundary operator BO(h, e) = RMSNorm(h) + α·e between passes; model growth (more loops during training) changes the scaling EXPONENT; looping regularises in the multi-epoch regime. Code: github.com/qlabs-eng/scaling-exponents. |
| **Geiping et al.** — recurrent depth, arXiv [2502.05171](https://arxiv.org/abs/2502.05171) | repo `experiments/ziplearn/refs/recurrent_depth_2502.05171.md` | A recurrence that does not re-read its input every step is unstable; KL-threshold convergence as the halting test. |
| **Saunshi et al.** — *Reasoning with Latent Thoughts: On the Power of Looped Transformers*, arXiv [2502.17416](https://arxiv.org/abs/2502.17416) | repo `experiments/ziplearn/refs/looped_latent_thoughts_2502.17416.md` (+ PDF) | T loops can simulate T steps of chain-of-thought: depth recurrence is the inner loop. |
| **Universal Transformers** — Dehghani et al., arXiv [1807.03819](https://arxiv.org/abs/1807.03819) | repo `experiments/ziplearn/refs/universal_transformers_1807.03819.md` | Weight-tied depth with per-position halting. |
| **ACT** — Graves, arXiv [1603.08983](https://arxiv.org/abs/1603.08983) | repo `experiments/ziplearn/refs/adaptive_computation_time_1603.08983.md` | Learned halting with a ponder cost — the "λ·K" penalty in its original form. |
| **PonderNet** — Banino et al., arXiv [2107.05407](https://arxiv.org/abs/2107.05407) | unchecked | Halting as a probability per step with a geometric prior (KL to it); unbiased gradients for the halting distribution. `ziplib/brain.py: halt_prob` already reads a halt row "in the PonderNet form". |

## 2. Routing across depth (question 1 of the brainstorm)

| ref | status | what we take from it |
|---|---|---|
| **Attention Residuals** — Kimi Team, arXiv [2603.15031](https://arxiv.org/abs/2603.15031) | repo `experiments/ziplearn/refs/attention_residuals.md` + PDF | Each layer's input is a softmax mix over sources (embedding + earlier outputs): one learned pseudo-query per layer, RMSNorm on the keys, RAW sources as values, queries initialised to zero. Block AttnRes groups layers into blocks and attends over block sums; magnitudes become bounded and periodic (each block boundary resets the accumulation). Content-dependent routing is the part that pays (fixed routes ≈ baseline). Official code: github.com/MoonshotAI/Attention-Residuals. |
| **E30** (this repo) — attention residuals ACROSS PASSES of `LoopedModel` | repo `experiments/ziplearn/RESULTS.md`, DESIGN §18 | Our own measurement of what trained AttnRes does in a loop: last pass read at ~0.8, decaying reads of earlier passes, the anchor fading 1.00 → 0.01 by pass 4. Run at 8 passes after training at 4 (trained-task accuracy): full history 0.60–0.77, a window of {anchor, last two passes} 0.19–0.20, the tied-core fixed boundary operator 0.08. Held-out compositions: 1/8 solved, only with a tied core. One seed. Apparatus: `h1_lid.py --res loop --loop_res attnres`. |

## 3. Memory: Spotlight and the mutable-memory family

| ref | status | what we take from it |
|---|---|---|
| **Percepta — Spotlight** (blog post, percepta.ai/blog/spotlight-memory, ~Oct 2026) | secondary — blocked | Only three claims could be recovered, all from coverage: (1) "separating intelligence from memory, allowing knowledge and skills to grow without changing the model's weights" (title of an r/singularity post, 181 points, via [reddit-daily-news #385](https://github.com/gitlawr/reddit-daily-news/issues/385)); (2) it replaces attention with an unbounded ("infinite") memory; (3) every token can read and write that memory by *learning to index the specific cells it needs*, and the number of cells a token touches stays constant however large the memory grows — contrasted with Mixture-of-Experts (Korean coverage at promppy.com/item/2095065, read only through search summaries). No mechanism, equations, training method or numbers were recoverable. **Needs the primary source** — see BRAINSTORM §2.6. |
| **Percepta — *Can LLMs be computers?*** (Christos Tzamos and team, blog, 11 Mar 2026) | secondary — blocked | Same team's earlier work, the likeliest ancestor of Spotlight's addressing: a WebAssembly interpreter compiled into the weights of a 7-layer transformer, d = 36, 18 two-dimensional attention heads; "HullKVCache" decodes in O(k + log n) instead of O(n) per token; 33k tokens/s on a CPU; claimed differentiable throughout. (Coverage: novaspivack.com, getmaxim.ai, awesomeagents.ai — via search summaries.) It is the same programme as our E28/E46 written interpreters. |
| **Loom** — Turkcan, *A Scalable Analytical Neural Computer Architecture*, arXiv [2604.08816](https://arxiv.org/abs/2604.08816) | web (title/author only) | An analytical (written) neural computer that cites the Percepta work; worth reading next to E46. |
| **Neural Turing Machines** — Graves, Wayne, Danihelka, arXiv [1410.5401](https://arxiv.org/abs/1410.5401) | unchecked | The namesake: a controller with differentiable read/write heads over an external memory, content and location addressing. |
| **Differentiable Neural Computer** — Graves et al., *Hybrid computing using a neural network with dynamic external memory*, Nature 538 (2016) | unchecked | NTM plus dynamic allocation (free list) and temporal links — the allocation question Spotlight must also answer. |
| **Sparse Access Memory** — Rae et al., *Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes*, NeurIPS 2016, arXiv [1610.09027](https://arxiv.org/abs/1610.09027) | web | The closest published match to Spotlight's stated properties: each step reads and writes a constant k cells, located by approximate nearest-neighbour search, so time and space per step are asymptotically optimal; 1,000× faster and 3,000× less memory than the dense NTM-style model. |
| **Product Key Memory** — Lample et al., *Large Memory Layers with Product Keys*, arXiv [1907.05242](https://arxiv.org/abs/1907.05242) | unchecked | Exact top-k over N cells in O(√N) by splitting the query and searching two √N sub-key sets. |
| **Memory Layers at Scale** — Berges et al. (Meta FAIR), ICML 2025, arXiv [2412.09764](https://arxiv.org/abs/2412.09764) | web | Product-key memory layers to 128B memory parameters; beat dense models with 2× the compute and MoE at matched compute and parameters. The cells here are PARAMETERS — the usual "sparse memory vs MoE" comparison, which is the comparison Spotlight's coverage draws. |
| **Fast weight programmers** — Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, arXiv [2102.11174](https://arxiv.org/abs/2102.11174) | unchecked | The delta rule as a mutable associative memory: write = erase the old value at a key, then write the new one. |
| **Gated DeltaNet** — Yang, Kautz, Hatamizadeh, ICLR 2025, arXiv [2412.06464](https://arxiv.org/abs/2412.06464) | web | Gating (fast erase) + delta rule (targeted update) are complementary; strong on in-context retrieval and length extrapolation. A follow-up, *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention* (NVIDIA Research, 2026-05), exists — title only, not read. |
| **Kimi Linear** — arXiv [2510.26692](https://arxiv.org/abs/2510.26692) | web | Kimi Delta Attention (finer-grained gating on Gated DeltaNet), hybrid with MLA, 48B/3B; the model the AttnRes paper integrates into. |
| **Titans** — Behrouz, Zhong, Mirrokni, *Learning to Memorize at Test Time*, arXiv [2501.00663](https://arxiv.org/abs/2501.00663) | web | A neural long-term memory updated at test time by surprise (gradient with momentum and forgetting); > 2M context. Another reading of "memory that grows without changing the weights". |
| **Mamba** — Gu & Dao, arXiv [2312.00752](https://arxiv.org/abs/2312.00752) | unchecked | Fixed-size selective state; the binding-bottleneck baseline already in `experiments/binding_mqar.py` (Mamba-3). |

## 4. Search and RL with continuous actions (the "EfficientZero V2 for thoughts" idea)

| ref | status | what we take from it |
|---|---|---|
| **MuZero** — Schrittwieser et al., arXiv [1911.08265](https://arxiv.org/abs/1911.08265), Nature 2020 | unchecked | Search with a LEARNED model (representation, dynamics, prediction); the search is a policy-improvement operator whose output becomes the policy target. |
| **EfficientZero** — Ye et al., *Mastering Atari Games with Limited Data*, NeurIPS 2021, arXiv [2111.00210](https://arxiv.org/abs/2111.00210) | unchecked | Self-supervised temporal consistency (SimSiam-style) to stop the learned latent dynamics collapsing; value prefix; off-policy correction. |
| **EfficientZero V2** — Wang, Liu, Ye, You, Gao, ICML 2024, arXiv [2403.00564](https://arxiv.org/abs/2403.00564); code [github.com/Shengjiewang-Jason/EfficientZeroV2](https://github.com/Shengjiewang-Jason/EfficientZeroV2) | web | (1) **Sampling-based Gumbel search**: candidates sampled from a Gaussian policy in continuous action spaces; Gumbel search guarantees policy improvement even at small simulation budgets, and the guarantee is shown to hold in the continuous case. (2) **Search-based value estimation (SVE)**: the root's empirical mean value as the target, which makes stale early transitions usable. Beats DreamerV3 on 50 of 66 tasks (Atari 100k, DMC state and vision). **Not in the pushed repo**: `REWORK_PLAN.md` cites `src/tbt/EZV2_NOTES.md` and a memory `reference_efficientzero_v2`, neither of which has ever been committed. |
| **Sampled MuZero** — Hubert et al., *Learning and Planning in Complex Action Spaces*, arXiv [2104.06303](https://arxiv.org/abs/2104.06303) | unchecked | Search over SAMPLED actions from any proposal distribution, with the correction that keeps policy improvement valid — what lets several proposal sources (policy, inverse model, value gradient) be mixed. |
| **Gumbel MuZero** — Danihelka et al., *Policy improvement by planning with Gumbel*, ICLR 2022 | unchecked | Gumbel-top-k + sequential halving at the root; guaranteed improvement with as few as 2 simulations. |
| **UniZero** — arXiv [2406.10667](https://arxiv.org/abs/2406.10667); **LightZero** — arXiv [2310.08348](https://arxiv.org/abs/2310.08348) | web (titles) | A transformer as the MuZero latent world model; a unified MCTS benchmark/codebase. |
| **Thinker** — arXiv [2307.14993](https://arxiv.org/abs/2307.14993) | repo `experiments/ziplearn/refs/thinker_2307.14993.md` (+ PDF) | A learned model the agent plans inside; recorded in ZipLearn as what NOT to build (search as an outside procedure), which is the tension this experiment re-opens. |
| **GCML** — Lin, Yang, Zhao, Pezzulo, Maass, *Neural sampling from cognitive maps enables goal-directed imagination and planning*, Nature Machine Intelligence (2026), doi:10.1038/s42256-026-01254-4 | repo `src/tbt/notes/gcml_neural_sampling_cognitive_maps.md` + PDF in `src/tbt/research papers/` | An INVERSE model W turns a goal-minus-state difference into an action, u = W(s* − s), learned by a local Hebbian rule; imagination = iterate the forward model with W choosing each step, O(1) per step, self-correcting, no tree search; noise turns it into a sampler of diverse plans; compositional generalisation from a compositional embedding. |
| **Value Iteration Networks**, **Searchformer**, **Stream of Search** | repo `experiments/ziplearn/refs/` | Planning as a looped layer; compressing one's own successful search traces beats the teacher. |

## 5. Learning continuous thoughts by RL, without reference chains of thought

| ref | status | what we take from it |
|---|---|---|
| **Soft Tokens, Hard Truths** — Butt et al. (UvA, Meta FAIR, NYU), ICLR 2026, arXiv [2509.19170](https://arxiv.org/abs/2509.19170) | web | The nearest existing answer to "Coconut without the substitution curriculum": continuous CoT learned by RL with NO distillation from reference CoTs. Soft tokens = mixtures of token embeddings, with Gaussian noise on the input embedding for exploration; REINFORCE with a leave-one-out (RLOO) baseline; hundreds of continuous tokens. Llama/Qwen ≤ 8B on maths: matches discrete CoT at pass@1, beats it at pass@32; best deployed as continuous-trained, discrete-at-inference. It starts from a PRETRAINED model that already reasons in text. |
| **HRPO** — *Hybrid Latent Reasoning via Reinforcement Learning*, NeurIPS 2025, arXiv [2505.18454](https://arxiv.org/abs/2505.18454) | web | A learnable gate blends the previous hidden state into the sampled token's embedding, starting almost all-token and moving toward hidden features; the token sampling supplies the stochasticity RL needs, so no CoT trajectories are required. |
| **Soft Thinking** — *Unlocking the Reasoning Potential of LLMs in Continuous Concept Space*, NeurIPS 2025, arXiv [2505.15778](https://arxiv.org/abs/2505.15778) | web | Training-free: feed back the probability-weighted mixture of token embeddings (a "concept token") instead of a sampled token; +2.48 pass@1 and −22.4% tokens. This is the "soft-embedding simplex projection" of the Gemini suggestions, already measured. |

## 6. Stochastic latents, bottlenecks, and representation collapse

| ref | status | what we take from it |
|---|---|---|
| **VAE / reparameterisation** — Kingma & Welling, arXiv [1312.6114](https://arxiv.org/abs/1312.6114) | unchecked | z = μ + σ ⊙ ε; pathwise gradients need a differentiable objective downstream of z. |
| **Soft Actor-Critic** — Haarnoja et al., arXiv [1801.01290](https://arxiv.org/abs/1801.01290) | unchecked | Reparameterised policy gradient THROUGH A LEARNED CRITIC; entropy-regularised with an automatically tuned temperature. |
| **Deep Variational Information Bottleneck** — Alemi et al., arXiv [1612.00410](https://arxiv.org/abs/1612.00410) | unchecked | The KL-to-a-prior bottleneck in a supervised net. |
| **Gumbel-Softmax** — Jang, Gu, Poole, arXiv [1611.01144](https://arxiv.org/abs/1611.01144); **Concrete** — Maddison, Mnih, Teh, arXiv [1611.00712](https://arxiv.org/abs/1611.00712) | unchecked | A reparameterisable, temperature-controlled sample ON the simplex — stochastic exploration and the simplex projection in one operator; τ → 0 is a hard one-hot. |
| **Free bits** — Kingma et al., *Improved Variational Inference with Inverse Autoregressive Flow*, arXiv [1606.04934](https://arxiv.org/abs/1606.04934) | unchecked | Charge KL only above a per-dimension floor: the standard guard against posterior collapse. |
| **DreamerV3** — Hafner et al., arXiv [2301.04104](https://arxiv.org/abs/2301.04104) | unchecked | KL balancing + free bits in a latent world model trained for control. |
| **Wasserstein Auto-Encoders** — Tolstikhin et al., arXiv [1711.01558](https://arxiv.org/abs/1711.01558) | unchecked | Regularise the AGGREGATE posterior toward the prior instead of every sample's — the fix for "a per-sample KL makes the latent uninformative". |
| **JEPA** — LeCun, *A Path Towards Autonomous Machine Intelligence* (position paper, 2022, OpenReview); **I-JEPA** — Assran et al., arXiv [2301.08243](https://arxiv.org/abs/2301.08243) | unchecked | Predict the REPRESENTATION of a future/masked observation, not its pixels; collapse prevented by an EMA/stop-gradient target. |
| **LeJEPA** — Balestriero & LeCun, arXiv [2511.08544](https://arxiv.org/abs/2511.08544) | web | Proves the isotropic Gaussian is the optimal embedding distribution; SIGReg (sketched isotropic Gaussian regularisation: random 1-D projections, characteristic-function matching, linear in dimension and batch) replaces the stop-grad/EMA heuristics against collapse. The aggregate-distribution regulariser to prefer over a per-sample KL. |
| **SimSiam** — Chen & He, arXiv [2011.10566](https://arxiv.org/abs/2011.10566); **BYOL** — Grill et al., arXiv [2006.07733](https://arxiv.org/abs/2006.07733); **VICReg** — Bardes, Ponce, LeCun, arXiv [2105.04906](https://arxiv.org/abs/2105.04906) | unchecked | Non-contrastive self-prediction and its collapse guards (stop-grad, EMA target, variance/covariance terms). |
| **SPR** — Schwarzer et al., *Data-Efficient RL with Self-Predictive Representations*, arXiv [2007.05929](https://arxiv.org/abs/2007.05929) | unchecked | Predicting future latent states as an auxiliary loss in RL — the JEPA idea inside an agent, and EfficientZero's precedent. |

## 7. Curriculum, frontier, and why depth matters

| ref | status | what we take from it |
|---|---|---|
| **Self-Play Pretraining with Zero Data** — arXiv [2609.30063](https://arxiv.org/abs/2609.30063) | repo `experiments/ziplearn/refs/self_play_pretraining_zero_data_2609.30063.md` + full text | A generator proposes Brainfuck programs at the frontier of a learner's capability (trained by RL); the learner predicts output bytes; no natural data. The source of a depth-graded curriculum, and of E46's program distribution. |
| **Merrill & Sabharwal** — log-precision transformers lie in uniform TC⁰ (TACL 2023; arXiv 2207.00729) | unchecked (cited in ZipLearn DESIGN §18.2) | Fixed-depth transformers cannot do iterated composition / state tracking; chain-of-thought (or depth recurrence) repairs it. Why "K = 0 thoughts" provably fails on deep problems. |
| **Looped transformers as programmable computers** — Giannou et al., ICML 2023, arXiv [2301.13196](https://arxiv.org/abs/2301.13196) | repo `experiments/ziplearn/refs/looped_transformers_programmable_computers_2301.13196.md` | A looped block with written weights is a general computer; the program is in the context. |
| **DiffusionBlocks** — arXiv [2506.14202](https://arxiv.org/abs/2506.14202) | repo `experiments/ziplearn/refs/diffusionblocks_2506.14202.md` | Local targets for each block from a diffusion interpretation — the same job search does for thoughts (a per-step target from an end-of-episode signal). |

## 8. Inputs that are not papers

- **Gemini's suggestions** (three screenshots shared by the user, 2026-10-03): stochastic latent emissions with
  reparameterised gradients and a KL bottleneck; soft-embedding simplex projection and JEPA-style state prediction
  against semantic drift; a curriculum over circuit depth with a learned halting gate and a λ·K compute penalty.
  Transcribed and assessed in `BRAINSTORM.md` §3.5.
- **This repository's own results** the brainstorm leans on (`experiments/ziplearn/RESULTS.md`, DESIGN §15, §18):
  E26 (looped transformer), E28 (rules written into one looped block), E30 (attention residuals across passes),
  E31 (Coconut × attention residuals, stopped after 2 cells), E41 (weights fixed, world model entirely in the
  context), E46 (a Brainfuck interpreter as a written looped block, 624/624 exact). Substrate:
  `experiments/transformers/h1_lid.py`. Associative-recall harness: `experiments/binding_mqar.py`.
