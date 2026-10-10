# Neural Turing Architecture — references

Working name for a standalone experiment: a looped transformer (Ouro-style depth recurrence, after Chen et al.'s
boundary-operator results) that also thinks in continuous space along the sequence (Coconut), learns to do so WITHOUT a
language-substitution curriculum or human reasoning traces, and may replace its attention with a mutable, sparsely
addressed memory (Percepta's Spotlight). The brainstorm is `BRAINSTORM.md`; this file is only the sources.

Compiled 2026-10-03 from a cloud session that at first could not open `arxiv.org`, `huggingface.co`, `alphaxiv.org`
or `percepta.ai` (network policy). Later the same day the network was opened and the two sources that mattered most
were read in full — Spotlight (post + code) and EfficientZero V2 — each with its own note under `refs/`. The other
entries are checked at different depths:
The status column says how far:

| status | meaning |
|---|---|
| **repo** | a note (sometimes the PDF) is already in this repository — path given; read in full this session |
| **web** | title and arXiv ID confirmed on 2026-10-03 (against arXiv itself, or by web search), with the headline claims; the paper itself not read in full |
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
| **PonderNet** — Banino et al., arXiv [2107.05407](https://arxiv.org/abs/2107.05407) | web | Halting as a probability per step with a geometric prior (KL to it); unbiased gradients for the halting distribution. `ziplib/brain.py: halt_prob` already reads a halt row "in the PonderNet form". |

## 2. Routing across depth (question 1 of the brainstorm)

| ref | status | what we take from it |
|---|---|---|
| **Attention Residuals** — Kimi Team, arXiv [2603.15031](https://arxiv.org/abs/2603.15031) | repo `experiments/ziplearn/refs/attention_residuals.md` + PDF | Each layer's input is a softmax mix over sources (embedding + earlier outputs): one learned pseudo-query per layer, RMSNorm on the keys, RAW sources as values, queries initialised to zero. Block AttnRes groups layers into blocks and attends over block sums; magnitudes become bounded and periodic (each block boundary resets the accumulation). Content-dependent routing is the part that pays (fixed routes ≈ baseline). Official code: github.com/MoonshotAI/Attention-Residuals. |
| **E30** (this repo) — attention residuals ACROSS PASSES of `LoopedModel` | repo `experiments/ziplearn/RESULTS.md`, DESIGN §18 | Our own measurement of what trained AttnRes does in a loop: last pass read at ~0.8, decaying reads of earlier passes, the anchor fading 1.00 → 0.01 by pass 4. Run at 8 passes after training at 4 (trained-task accuracy): full history 0.60–0.77, a window of {anchor, last two passes} 0.19–0.20, the tied-core fixed boundary operator 0.08. Held-out compositions: 1/8 solved, only with a tied core. One seed. Apparatus: `h1_lid.py --res loop --loop_res attnres`. |

## 3. Memory: Spotlight and the mutable-memory family

| ref | status | what we take from it |
|---|---|---|
| **Percepta — Spotlight Memory** — Tzamos, Zheng, Jacob & Percepta, *Spotlight Memory: Growing Memory for Long-Context Modeling at Linear Cost* (blog, 2 Oct 2026), https://percepta.ai/blog/spotlight-memory | repo `refs/spotlight_memory_percepta_2026.md` (post + code read in full) | Per head, keys and queries get learned 2-D addresses on an unbounded lattice; a key writes and a query reads the 3 × 3 cells around its address through a normalised cos² bump; each cell (allocated on first write) is a DeltaNet state. Constant work per token, growing memory, O(T) total. Near-perfect MQAR incl. 131K pairs and overwrites; 8K-trained LMs (140M–670M) recall at 128K at 93–100% where attention scores 0%. |
| **Percepta — *Can LLMs be computers?*** (Christos Tzamos and team, blog, 11 Mar 2026) | secondary — blocked | Same team's earlier work, the likeliest ancestor of Spotlight's addressing: a WebAssembly interpreter compiled into the weights of a 7-layer transformer, d = 36, 18 two-dimensional attention heads; "HullKVCache" decodes in O(k + log n) instead of O(n) per token; 33k tokens/s on a CPU; claimed differentiable throughout. (Coverage: novaspivack.com, getmaxim.ai, awesomeagents.ai — via search summaries.) It is the same programme as our E28/E46 written interpreters. |
| **Percepta — *Growing Intelligence Beyond the Weights*** (companion post, slug `can-llms-grow-their-own-capabilities`) and the model **`percepta-ai/spotlight-vm`** on Hugging Face (Apache-2.0: `spotlight.c`, `spotlight.py`, weights, memory) | repo `refs/spotlight_memory_percepta_2026.md` (read in full) | A hand-built 8-layer, d = 16 Spotlight transformer (< 100K parameters) running unmodified MicroPython out of its memory at ~120K tokens/s on one CPU core, constant work per step; packages written into memory from the context; facts and functions updated by overwriting cells. The ~20-line read/write in `spotlight.c` is the exact mechanism. |
| **Jelassi, Brandfonbrener, Kakade, Malach** — *Repeat After Me: Transformers are Better than State Space Models at Copying*, ICML 2024; **Siegelmann & Sontag** — *On the Computational Power of Neural Nets*, JCSS 50(1), 1995 | cited in the Spotlight post | Why a fixed-size state cannot be a general memory: copying needs state growing with the input; recurrent nets reach Turing completeness only through unbounded precision. |
| **Zoology** — Arora et al. (2023), MQAR; **MAD** — Poli et al., *Mechanistic Design and Scaling of Hybrid Architectures* (2024) | cited in the Spotlight post | The synthetic recall/mechanism suites Spotlight was validated on — the protocols to reuse for NTA-M. |
| **Loom** — Turkcan, *A Scalable Analytical Neural Computer Architecture*, arXiv [2604.08816](https://arxiv.org/abs/2604.08816) | web (title/author only) | An analytical (written) neural computer that cites the Percepta work; worth reading next to E46. |
| **Neural Turing Machines** — Graves, Wayne, Danihelka, arXiv [1410.5401](https://arxiv.org/abs/1410.5401) | web | The namesake: a controller with differentiable read/write heads over an external memory, content and location addressing. |
| **Differentiable Neural Computer** — Graves et al., *Hybrid computing using a neural network with dynamic external memory*, Nature 538 (2016) | unchecked | NTM plus dynamic allocation (free list) and temporal links — the allocation question Spotlight must also answer. |
| **Sparse Access Memory** — Rae et al., *Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes*, NeurIPS 2016, arXiv [1610.09027](https://arxiv.org/abs/1610.09027) | web | The closest published match to Spotlight's stated properties: each step reads and writes a constant k cells, located by approximate nearest-neighbour search, so time and space per step are asymptotically optimal; 1,000× faster and 3,000× less memory than the dense NTM-style model. |
| **Product Key Memory** — Lample et al., *Large Memory Layers with Product Keys*, arXiv [1907.05242](https://arxiv.org/abs/1907.05242) | web | Exact top-k over N cells in O(√N) by splitting the query and searching two √N sub-key sets. |
| **Memory Layers at Scale** — Berges et al. (Meta FAIR), ICML 2025, arXiv [2412.09764](https://arxiv.org/abs/2412.09764) | web | Product-key memory layers to 128B memory parameters; beat dense models with 2× the compute and MoE at matched compute and parameters. The cells here are PARAMETERS — the usual "sparse memory vs MoE" comparison, which is the comparison Spotlight's coverage draws. |
| **Fast weight programmers** — Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, arXiv [2102.11174](https://arxiv.org/abs/2102.11174) | web | The delta rule as a mutable associative memory: write = erase the old value at a key, then write the new one. |
| **Gated DeltaNet** — Yang, Kautz, Hatamizadeh, ICLR 2025, arXiv [2412.06464](https://arxiv.org/abs/2412.06464) | web | Gating (fast erase) + delta rule (targeted update) are complementary; strong on in-context retrieval and length extrapolation. A follow-up, *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention* (NVIDIA Research, 2026-05), exists — title only, not read. |
| **Kimi Linear** — arXiv [2510.26692](https://arxiv.org/abs/2510.26692) | web | Kimi Delta Attention (finer-grained gating on Gated DeltaNet), hybrid with MLA, 48B/3B; the model the AttnRes paper integrates into. |
| **Titans** — Behrouz, Zhong, Mirrokni, *Learning to Memorize at Test Time*, arXiv [2501.00663](https://arxiv.org/abs/2501.00663) | web | A neural long-term memory updated at test time by surprise (gradient with momentum and forgetting); > 2M context. Another reading of "memory that grows without changing the weights". |
| **Mamba** — Gu & Dao, arXiv [2312.00752](https://arxiv.org/abs/2312.00752) | web | Fixed-size selective state; the binding-bottleneck baseline already in `experiments/binding_mqar.py` (Mamba-3). |

## 4. Search and RL with continuous actions (the "EfficientZero V2 for thoughts" idea)

| ref | status | what we take from it |
|---|---|---|
| **MuZero** — Schrittwieser et al., arXiv [1911.08265](https://arxiv.org/abs/1911.08265), Nature 2020 | web | Search with a LEARNED model (representation, dynamics, prediction); the search is a policy-improvement operator whose output becomes the policy target. |
| **EfficientZero** — Ye et al., *Mastering Atari Games with Limited Data*, NeurIPS 2021, arXiv [2111.00210](https://arxiv.org/abs/2111.00210) | web | Self-supervised temporal consistency (SimSiam-style) to stop the learned latent dynamics collapsing; value prefix; off-policy correction. |
| **EfficientZero V2** — Wang, Liu, Ye, You, Gao, ICML 2024, arXiv [2403.00564](https://arxiv.org/abs/2403.00564); code [github.com/Shengjiewang-Jason/EfficientZeroV2](https://github.com/Shengjiewang-Jason/EfficientZeroV2) | repo `refs/efficientzero_v2_2403.00564.md` (read in full) | (1) **Sampling-based Gumbel search**: root candidates from a Gaussian policy plus a flattened copy of it, Sequential Halving; the improvement argument for continuous actions is asymptotic in the number of samples (their eq. 7); π′ = softmax(σ(completedQ)); a "simple loss" −log p(a*) for large action spaces. (2) **Search-based value estimation**: the mean of the simulations' bootstrapped returns, mixed with multi-step TD early and for fresh data. 32 simulations; beats DreamerV3 on 50 of 66 tasks. (The earlier notes `src/tbt/EZV2_NOTES.md` cited by `REWORK_PLAN.md` were never committed; this note replaces them.) |
| **Sampled MuZero** — Hubert et al., *Learning and Planning in Complex Action Spaces*, arXiv [2104.06303](https://arxiv.org/abs/2104.06303) | web | Search over SAMPLED actions from any proposal distribution, with the correction that keeps policy improvement valid — what lets several proposal sources (policy, inverse model, value gradient) be mixed. |
| **Gumbel MuZero** — Danihelka, Guez, Schrittwieser, Silver, *Policy improvement by planning with Gumbel*, ICLR 2022, [OpenReview bERaNdoegnO](https://openreview.net/forum?id=bERaNdoegnO) | web | Gumbel-top-k + Sequential Halving at the root; policy improvement guaranteed at any simulation budget when the visited actions' values are evaluated correctly; strong with few simulations. |
| **UniZero** — arXiv [2406.10667](https://arxiv.org/abs/2406.10667); **LightZero** — arXiv [2310.08348](https://arxiv.org/abs/2310.08348) | web (titles) | A transformer as the MuZero latent world model; a unified MCTS benchmark/codebase. |
| **Thinker** — arXiv [2307.14993](https://arxiv.org/abs/2307.14993) | repo `experiments/ziplearn/refs/thinker_2307.14993.md` (+ PDF) | A learned model the agent plans inside; recorded in ZipLearn as what NOT to build (search as an outside procedure), which is the tension this experiment re-opens. |
| **GCML** — Lin, Yang, Zhao, Pezzulo, Maass, *Neural sampling from cognitive maps enables goal-directed imagination and planning*, Nature Machine Intelligence (2026), doi:10.1038/s42256-026-01254-4 | repo `src/tbt/notes/gcml_neural_sampling_cognitive_maps.md` + PDF in `src/tbt/research papers/` | An INVERSE model W turns a goal-minus-state difference into an action, u = W(s* − s), learned by a local Hebbian rule; imagination = iterate the forward model with W choosing each step, O(1) per step, self-correcting, no tree search; noise turns it into a sampler of diverse plans; compositional generalisation from a compositional embedding. |
| **Daw, Niv & Dayan** — *Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control*, Nature Neuroscience 8(12):1704–1711 (2005), doi:[10.1038/nn1560](https://doi.org/10.1038/nn1560) | web (confirmed 2026-10-04) | Two controllers — a flexible model-based planner and a cheap cached ("habit") one — arbitrated by their relative UNCERTAINTY, each deployed where it should be most accurate. The precedent for switching between search, a distilled policy and GCML by a running reliability measurement (planning doc §14). |
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
| **VAE / reparameterisation** — Kingma & Welling, arXiv [1312.6114](https://arxiv.org/abs/1312.6114) | web | z = μ + σ ⊙ ε; pathwise gradients need a differentiable objective downstream of z. |
| **Soft Actor-Critic** — Haarnoja et al., arXiv [1801.01290](https://arxiv.org/abs/1801.01290) | web | Reparameterised policy gradient THROUGH A LEARNED CRITIC; entropy-regularised with an automatically tuned temperature. |
| **Deep Variational Information Bottleneck** — Alemi et al., arXiv [1612.00410](https://arxiv.org/abs/1612.00410) | web | The KL-to-a-prior bottleneck in a supervised net. |
| **Gumbel-Softmax** — Jang, Gu, Poole, arXiv [1611.01144](https://arxiv.org/abs/1611.01144); **Concrete** — Maddison, Mnih, Teh, arXiv [1611.00712](https://arxiv.org/abs/1611.00712) | web | A reparameterisable, temperature-controlled sample ON the simplex — stochastic exploration and the simplex projection in one operator; τ → 0 is a hard one-hot. |
| **Free bits** — Kingma et al., *Improved Variational Inference with Inverse Autoregressive Flow*, arXiv [1606.04934](https://arxiv.org/abs/1606.04934) | web | Charge KL only above a per-dimension floor: the standard guard against posterior collapse. |
| **DreamerV3** — Hafner et al., arXiv [2301.04104](https://arxiv.org/abs/2301.04104) | web | KL balancing + free bits in a latent world model trained for control. |
| **Wasserstein Auto-Encoders** — Tolstikhin et al., arXiv [1711.01558](https://arxiv.org/abs/1711.01558) | web | Regularise the AGGREGATE posterior toward the prior instead of every sample's — the fix for "a per-sample KL makes the latent uninformative". |
| **JEPA** — LeCun, *A Path Towards Autonomous Machine Intelligence* (position paper, 2022, OpenReview); **I-JEPA** — Assran et al., arXiv [2301.08243](https://arxiv.org/abs/2301.08243) | web (I-JEPA); the position paper unchecked | Predict the REPRESENTATION of a future/masked observation, not its pixels; collapse prevented by an EMA/stop-gradient target. |
| **LeJEPA** — Balestriero & LeCun, arXiv [2511.08544](https://arxiv.org/abs/2511.08544) | web | Proves the isotropic Gaussian is the optimal embedding distribution; SIGReg (sketched isotropic Gaussian regularisation: random 1-D projections, characteristic-function matching, linear in dimension and batch) replaces the stop-grad/EMA heuristics against collapse. The aggregate-distribution regulariser to prefer over a per-sample KL. |
| **SimSiam** — Chen & He, arXiv [2011.10566](https://arxiv.org/abs/2011.10566); **BYOL** — Grill et al., arXiv [2006.07733](https://arxiv.org/abs/2006.07733); **VICReg** — Bardes, Ponce, LeCun, arXiv [2105.04906](https://arxiv.org/abs/2105.04906) | web | Non-contrastive self-prediction and its collapse guards (stop-grad, EMA target, variance/covariance terms). |
| **SPR** — Schwarzer et al., *Data-Efficient RL with Self-Predictive Representations*, arXiv [2007.05929](https://arxiv.org/abs/2007.05929) | web | Predicting future latent states as an auxiliary loss in RL — the JEPA idea inside an agent, and EfficientZero's precedent. |

## 7. Curriculum, frontier, and why depth matters

| ref | status | what we take from it |
|---|---|---|
| **Self-Play Pretraining with Zero Data** — arXiv [2609.30063](https://arxiv.org/abs/2609.30063) | repo `experiments/ziplearn/refs/self_play_pretraining_zero_data_2609.30063.md` + full text | A generator proposes Brainfuck programs at the frontier of a learner's capability (trained by RL); the learner predicts output bytes; no natural data. The source of a depth-graded curriculum, and of E46's program distribution. |
| **Merrill & Sabharwal** — log-precision transformers lie in uniform TC⁰ (TACL 2023; arXiv 2207.00729) | web | Fixed-depth transformers cannot do iterated composition / state tracking; chain-of-thought (or depth recurrence) repairs it. Why "K = 0 thoughts" provably fails on deep problems. |
| **Looped transformers as programmable computers** — Giannou et al., ICML 2023, arXiv [2301.13196](https://arxiv.org/abs/2301.13196) | repo `experiments/ziplearn/refs/looped_transformers_programmable_computers_2301.13196.md` | A looped block with written weights is a general computer; the program is in the context. |
| **DiffusionBlocks** — arXiv [2506.14202](https://arxiv.org/abs/2506.14202) | repo `experiments/ziplearn/refs/diffusionblocks_2506.14202.md` | Local targets for each block from a diffusion interpretation — the same job search does for thoughts (a per-step target from an end-of-episode signal). |

## 8. Search in high-dimensional continuous spaces (the dimensionality problem, planning doc §10)

All IDs confirmed against arXiv on 2026-10-03; none read in full.

| ref | status | what we take from it |
|---|---|---|
| **Nesterov & Spokoiny** — *Random gradient-free minimization of convex functions*, Foundations of Computational Mathematics 17(2):527–566 (2017) | web | Random-direction (zeroth-order) methods need up to d times more iterations than gradient methods — the iteration-count form of §10.2's 1/√d. |
| **SVG** — Heess et al., *Learning Continuous Control Policies by Stochastic Value Gradients*, arXiv [1510.09142](https://arxiv.org/abs/1510.09142) | web | Backpropagate value through a (learned) model into the action — D1, with our dynamics exact rather than learned. |
| **MPO** — Abdolmaleki et al., *Maximum a Posteriori Policy Optimisation*, arXiv [1806.06920](https://arxiv.org/abs/1806.06920) | web | Improved policy π′ ∝ π·exp(Q/η) under a KL bound, then a weighted maximum-likelihood fit of the parametric policy (incl. its covariance) — the policy update of §10.6 and the trust region of §10.5. |
| **AWR** — Peng et al., *Advantage-Weighted Regression*, arXiv [1910.00177](https://arxiv.org/abs/1910.00177) | web | The same exponentiated-advantage weighting as a plain regression. |
| **GradCEM** — Bharadhwaj, Xie, Shkurti, *Model-Predictive Control via Cross-Entropy and Gradient-Based Optimization*, arXiv [2004.08763](https://arxiv.org/abs/2004.08763) | web | Population search with gradient steps on each sample — D1 inside D6. |
| **Guided ES** — Maheswaranathan et al., *Guided evolutionary strategies: Augmenting random search with surrogate gradients*, arXiv [1806.10230](https://arxiv.org/abs/1806.10230) | web | Sample in the subspace spanned by (biased) surrogate gradients — a principled D1 + D3 hybrid. |
| **Intrinsic dimension** — Li, Farkhoor, Liu, Yosinski, *Measuring the Intrinsic Dimension of Objective Landscapes*, arXiv [1804.08838](https://arxiv.org/abs/1804.08838) | web | Optimisation often succeeds in a random low-dimensional subspace — the premise of D3. |
| **CMA-ES** — Hansen, *The CMA Evolution Strategy: A Tutorial*, arXiv [1604.00772](https://arxiv.org/abs/1604.00772) | web | Adapt the sampling covariance toward directions that paid — what the low-rank policy update of §10.6 does. |
| **iCEM** — Pinneri et al., *Sample-efficient Cross-Entropy Method for Real-time Planning*, arXiv [2008.06389](https://arxiv.org/abs/2008.06389); **TD-MPC2** — Hansen, Su, Wang, arXiv [2310.16828](https://arxiv.org/abs/2310.16828) | web | Iterated population planners (D6); TD-MPC2 is EZ-V2's strongest continuous-control rival and its MPPI planner the cost reference. |
| **Diffusion-QL** — Wang, Hunt, Zhou, *Diffusion Policies as an Expressive Policy Class for Offline RL*, arXiv [2208.06193](https://arxiv.org/abs/2208.06193) | web | A generative policy steered by a Q-gradient — D2's relative. |
| **Twisted SMC** — Zhao, Brekelmans, Makhzani, Grosse, *Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo*, arXiv [2404.17546](https://arxiv.org/abs/2404.17546) | web | Particle search over sequences with a learned twist (a value) — D5. |
| **This repo** — `research/dim_search_toy.py` / `.json` | repo | The 1/√d table of planning doc §10.2. |

## 9. Inputs that are not papers

- **Gemini's suggestions** (three screenshots shared by the user, 2026-10-03): stochastic latent emissions with
  reparameterised gradients and a KL bottleneck; soft-embedding simplex projection and JEPA-style state prediction
  against semantic drift; a curriculum over circuit depth with a learned halting gate and a λ·K compute penalty.
  Transcribed and assessed in `BRAINSTORM.md` §3.5.
- **This repository's own results** the brainstorm leans on (`experiments/ziplearn/RESULTS.md`, DESIGN §15, §18):
  E26 (looped transformer), E28 (rules written into one looped block), E30 (attention residuals across passes),
  E31 (Coconut × attention residuals, stopped after 2 cells), E41 (weights fixed, world model entirely in the
  context), E46 (a Brainfuck interpreter as a written looped block, 624/624 exact). Substrate:
  `experiments/transformers/h1_lid.py`. Associative-recall harness: `experiments/binding_mqar.py`.

## 10. Past commutativity — group structure, navigation, and planning (planning doc §17)

All confirmed by web search on 2026-10-04 (title, venue, headline claim); none read in full.

| ref | status | what we take from it |
|---|---|---|
| **Sargolini, Fyhn, Hafting, McNaughton, Witter, Moser & Moser** — *Conjunctive representation of position, direction, and velocity in entorhinal cortex*, Science 312:758–762 (2006), doi:10.1126/science.1125572 | web | Deeper layers of medial entorhinal cortex hold conjunctive grid × head-direction cells, all modulated by running speed — the bridge that rotates self-motion by heading before it updates position (tier 1 of §17.5). |
| **Burak & Fiete** — *Accurate path integration in continuous attractor network models of grid cells*, PLoS Comput Biol 5(2):e1000291 (2009) | web | A continuous-attractor grid network driven only by velocity and heading inputs path-integrates accurately: the heading-gated velocity input, as a mechanism. |
| **Gao, Xie, Zhu & Wu** — *Learning Grid Cells as Vector Representation of Self-Position Coupled with Matrix Representation of Self-Motion*, ICLR 2019, arXiv [1810.05597](https://arxiv.org/abs/1810.05597) | web | Position as a vector, self-motion as a MATRIX acting on it; learns hexagonal grid patterns, path-integrates and plans — tier 2's "actions as matrices", for space. |
| **Whittington et al.** — *The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal Formation*, Cell (Nov 2020) | web | Action-dependent transitions of a learned structural code, for spatial and non-spatial graphs; grid-, band-, border- and object-vector-like cells emerge. |
| **Even & Goldreich** — *The minimum-length generator sequence problem is NP-hard*, Journal of Algorithms (1981); **Jerrum** — *The complexity of finding minimum-length generator sequences*, Theoretical Computer Science 36:265–289 (1985) | web | Shortest generator sequence to a target group element: NP-hard (NP-complete with the length bound in unary), PSPACE-complete with it in binary. Why tier 3 has no cheap inverse in general. |
| **Demaine, Eisenstat & Rudoy** — *Solving the Rubik's Cube Optimally is NP-complete*, STACS 2018, arXiv [1706.06708](https://arxiv.org/abs/1706.06708) | web | The concrete case: optimal n × n × n cube solving is NP-complete. |
| **Korf** — *Macro-operators: a weak method for learning*, Artificial Intelligence 26 (1985) | web | Learn macro-operators (operator sequences) that solve problems with non-serialisable subgoals such as Rubik's Cube; "operator decomposability" characterises where it works. What people do at tier 3. |

## 11. One general method — learning theory, the zone of proximal development, Ataraxos (planning doc §19)

Added 2026-10-06. Ataraxos was read in full that day (main text, Methods, Extended Data captions). The entries marked
**unchecked** were cited from memory — the session's web tools were at their weekly limit — so confirm each before
quoting it.

| ref | status | what we take from it |
|---|---|---|
| **Sokota, Vinitsky, Hu, Fan, Kolter & Farina** — *Scalable decision-making for games of imperfect information* ("Ataraxos"), Nature 658:55–59 (30 Sep 2026), doi:10.1038/s41586-026-11036-y | repo `refs/ataraxos_nature_2026.md` | Self-play RL + test-time search under hidden information; superhuman Stratego at ~1/500 of DeepNash's compute. Test-time search as one more step of the training update; damped dynamics (magnet regularisation coordinated with update size, both annealed); advantage filtering. |
| **Sokota et al.** — *The update-equivalence framework for decision-time planning*, ICLR 2024 | secondary (title and venue from Ataraxos's reference list) | Decision-time planning as an update step of the learning algorithm; search inherits its improvement properties. |
| **Sokota et al.** — *A unified approach to reinforcement learning, quantal response equilibria, and two-player zero-sum games* (magnetic mirror descent), ICLR 2023 | secondary (as above) | The regularised update: mirror descent with a KL to the previous policy and a KL to a fixed magnet. |
| **Self-Play Pretraining with Zero Data** — arXiv 2609.30063, §2 "Generator reward" and Table 5 | repo `experiments/ziplearn/refs/self_play_pretraining_zero_data_2609.30063.txt` | The frontier reward: absolute, preconditioned alignment between the learner's gradient on a sample and its recent parameter movement; ablations against difficulty, realised loss drop, one-step windows, shuffling. |
| **Shalev-Shwartz, Shamir & Shammah** — *Failures of gradient-based deep learning*, ICML 2017 | unchecked | Parity: the gradient carries exponentially little information about the target. |
| **Kearns** — *Efficient noise-tolerant learning from statistical queries*, JACM 1998 | unchecked | Statistical-query learners cannot learn parities efficiently. |
| **Wies, Levine & Shashua** — *Sub-task decomposition enables learning in sequence to sequence tasks*, ICLR 2023 | unchecked | Intermediate supervision turns an intractable composition into a learnable one. |
| **Kim & Suzuki** — *Transformers provably solve parity efficiently with chain of thought*, arXiv 2410.08633 (ICLR 2025) | unchecked | The same, for transformers with chain of thought. |
| **Abbe, Boix-Adserà & Misiakiewicz** — *SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics*, COLT 2023 | unchecked | Targets learnable step by step ("staircases") versus ones with a large leap. |
| **Abbe, Cornacchia & Lotfi** — *Provable advantage of curriculum learning on parity targets with mixed inputs*, NeurIPS 2023 | unchecked | A curriculum over inputs makes parity learnable. |
| **Orseau, Lelis, Lattimore & Weber** — *Single-agent policy tree search with guarantees* (Levin Tree Search), NeurIPS 2018 | unchecked | Expansions ≤ solution length ÷ the policy's probability of the solution. |
| **Florensa, Held, Geng & Abbeel** — *Automatic goal generation for reinforcement learning agents* (Goal GAN), ICML 2018 | unchecked | Train on goals of intermediate difficulty (success rate in a band). |
| **Foster & Foerster** — *Learning to reason at the frontier of learnability*, 2025 | unchecked | Sample LLM-RL prompts by p(1 − p). |
| **Jiang, Grefenstette & Rocktäschel** — *Prioritized level replay*, ICML 2021 | unchecked | Replay levels by a learning-potential score. |
| **Oudeyer, Kaplan & Hafner** — *Intrinsic motivation systems for autonomous mental development*, IEEE TEC 2007; **Matiisen, Oliver, Cohen & Schulman** — *Teacher–student curriculum learning*, 2017; **Portelas et al.** — ALP-GMM, CoRL 2019 | unchecked | Learning progress as the curriculum signal. |
| **Sukhbaatar et al.** — *Intrinsic motivation and automatic curricula via asymmetric self-play*, ICLR 2018 | unchecked | One agent proposes tasks at the edge of the other's ability. |
| **Cui et al.** — *JEPA-Anything: Learning Predictive Models across Different Worlds* (Orthogonal Predictive Factorization), arXiv 2609.20800 (Sep 2026); code github.com/Gen-Verse/JEPA-Anything | repo `refs/jepa_anything_opf_2609.20800.md` | A JEPA whose predictor is split into K heads, each predicting one block of a learned orthonormal rotation of the target; −34.8% single-intervention error on Interventional Pong vs a capacity-matched dense JEPA. Its Kepler result (slope −1.4991) is a frequency read-back with no control. |


## 12. Training without backpropagation (the user's training-method plan, 2026-10-10)

| Reference | Read? | What it gives us |
|---|---|---|
| **qlabs — *Dust: Pretraining Transformers Without Backpropagation*** (2026), qlabs.sh/research/dust; code github.com/qlabs-eng/dust | web + code summary, `refs/dust_qlabs_2026.md` | Node perturbation: jitter a layer's output at every token, estimate its output error from each token's loss change, update by the same outer product as backprop. Beats backprop's loss at 100k–1M tokens with a few hundred draws; cost scales with width. The note rates five cost-cutting ideas and adds sparsity, narrow groups with their own scores, and exact errors where they are known. |
| **Ren, Kornblith, Liao & Hinton** — *Scaling forward gradient with local losses*, ICLR 2023, arXiv 2210.03310 | abstract | Activity perturbation plus ~250k local losses (block-, patch-, channel-group-wise) matches backprop on MNIST/CIFAR-10. Variance grows with the number of hidden dimensions, and the group losses are what make it scale. |
| **Maheswaranathan, Metz, Tucker, Choi & Sohl-Dickstein** — *Guided evolutionary strategies*, ICML 2019, arXiv 1806.10230 | abstract | Perturb along a subspace spanned by surrogate gradients, plus isotropic noise. The result is a descent direction even when the surrogate is biased. |
