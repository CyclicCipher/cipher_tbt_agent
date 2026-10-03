# EfficientZero V2 — reference notes

**Paper:** Shengjie Wang\*, Shaohuai Liu\*, Weirui Ye\*, Jiacheng You, Yang Gao (\* equal contribution; Tsinghua IIIS,
Shanghai Qi Zhi Institute, Shanghai AI Laboratory, Texas A&M), *EfficientZero V2: Mastering Discrete and Continuous
Control with Limited Data*, ICML 2024 (PMLR 235), arXiv:2403.00564 (v1 1 Mar 2024, v2 12 Sep 2024).
**Code:** github.com/Shengjiewang-Jason/EfficientZeroV2.
**License:** arXiv non-exclusive distribution (not Creative Commons), so — as with the other arXiv-licensed notes in
this repo — this file is the abstract verbatim plus notes in our words, and the PDF is not committed:
https://arxiv.org/abs/2403.00564 (PDF https://arxiv.org/pdf/2403.00564).
**Read in full 2026-10-03** (22 pages, appendices included). Equation and section numbers below are the paper's.
These notes replace `src/tbt/EZV2_NOTES.md`, which `REWORK_PLAN.md` cites but which was never committed.

## Abstract (verbatim)

> Sample efficiency remains a crucial challenge in applying Reinforcement Learning (RL) to real-world tasks. While
> recent algorithms have made significant strides in improving sample efficiency, none have achieved consistently
> superior performance across diverse domains. In this paper, we introduce EfficientZero V2, a general framework
> designed for sample-efficient RL algorithms. We have expanded the performance of EfficientZero to multiple domains,
> encompassing both continuous and discrete actions, as well as visual and low-dimensional inputs. With a series of
> improvements we propose, EfficientZero V2 outperforms the current state-of-the-art (SOTA) by a significant margin in
> diverse tasks under the limited data setting. EfficientZero V2 exhibits a notable advancement over the prevailing
> general algorithm, DreamerV3, achieving superior outcomes in 50 of 66 evaluated tasks across diverse benchmarks,
> such as Atari 100k, Proprio Control, and Vision Control.

## The method, in our words

### The four learned functions (inherited from EfficientZero, §3.3)

- representation `s_t = H(o_t)`; dynamics `ŝ_{t+1}, r_t = G(s_t, a_t)`, with an **action embedding** inside G (linear →
  LayerNorm → ReLU, 64-d) so that similar actions sit close together; policy `p_t = P(s_t)`; value `v_t = V(s_t)`.
- One joint loss, unrolled `l_unroll = 5` steps through G (eq 3, 5):
  `L_t = λ1·L_R(u_t, r_t) + λ2·L_P(π_t, p_t) + λ3·L_V(z_t, v_t) + λ4·L_G(s_{t+1}, ŝ_{t+1})`, where `u_t` is the real
  reward, `π_t` and `z_t` are the policy and value targets FROM THE SEARCH, and `L_G` is EfficientZero's **temporal
  consistency**: negative cosine between `sg(P1(s_{t+1}))` and `P2(P1(ŝ_{t+1}))` — SimSiam's asymmetric projector /
  predictor with a stop-gradient on the real next state's side (eq 4). Coefficients (Table 3): λ1 = 1, λ2 = 1,
  λ3 = 0.25, λ4 = 2, plus a policy-entropy term at 5·10⁻³.
- Value and reward are categorical (51 bins; value range ±299, reward ±2).

### Change 1 — sampling-based Gumbel search (§4.2; App. B, C)

- **Root candidates.** Sample K actions: a set `AS1` from the current Gaussian policy `p_t`, and a set `AS2` from a
  *flattened* (wider) version `p′_t` of it, so that actions the current policy finds unlikely still get tried.
  Sequential Halving (a bandit over the candidates, with Q-values from simulations) picks `a*_S = argmax q(s, a)`.
- **Policy improvement** (Def. 4.1) means `q(s, a*_S) ≥ E_{a∼p_t}[q(s, a)]`. Their argument (eq 7): the maximum over
  the candidates is at least the mean over `AS1`, which tends to the expectation under `p_t` **as |AS1| → ∞** (law of
  large numbers). So in the continuous case the guarantee is ASYMPTOTIC in the number of policy samples; it is exact
  only in Gumbel MuZero's discrete case.
- **Below the root** actions are sampled from `p_t` only, and fewer of them, so the search goes deeper instead of
  re-simulating near-duplicates.
- **Targets.** `π′ = softmax(σ(completedQ))` over the candidates (eq 18), with `σ(q) = (c_visit + max_b N(b))·c_scale·q`,
  `c_visit = 50`, `c_scale = 0.1` (eq 19); completedQ is `r + γ·v(s′)` for visited candidates and the policy-weighted
  mean of the visited Q-values for unvisited ones. The policy is trained by cross-entropy to `π′` (eq 8) and, in
  high-dimensional action spaces, by the **"simple loss"** `−log p_t(a*_S)` (eq 9), which drags the policy straight
  to the search's best action and converges faster when the action space is large (App. C, Fig. 5).
- **Policy head:** a squashed Gaussian; mean through `5·tanh`, standard deviation through softplus.
- **Budget:** 32 simulations and K = 16 candidates (16 and 8 on Atari). It beats Sample MuZero's MCTS at 50
  simulations, and still beats it at 8 (Fig. 3).

### Change 2 — search-based value estimation (§4.3; App. D, F)

- Each simulation's dive from the root is an H(n)-step imagined rollout, i.e. a bootstrapped estimate
  `V̂_n(s0) = Σ_{t ≤ H(n)} γ^t r̂_t + γ^{H(n)}·V̂(ŝ_{H(n)})` (eq 11); SVE is their mean over the N simulations (eq 10).
  It is computed inside the reanalyse pass, at no extra cost.
- Their error bound (Cor. 4.3, eq 15) grows with the model's state, reward and value errors and goes to zero with
  them — so SVE is unreliable EARLY, while the learned model is still wrong.
- Hence the **mixed value target** (eq 16): the ordinary 5-step TD target during the first `T1 = 4·10⁴` training steps
  and for fresh transitions (buffer index within `T2 = 2·10⁴` of the newest); SVE for everything else. It beats both a
  multi-step TD target and a double-Q target (Fig. 4).

### Smaller changes (App. A, G–I)

- Priority precalculation: new trajectories get their replay priority from the current model's Bellman error
  instead of the buffer maximum.
- Low-dimensional inputs: a running-mean normaliser, then Pre-LN residual MLP towers (3 blocks, 128-d state, hidden
  256). Images: EfficientZero's convolutional architecture.
- Pipeline: self-play workers act by search; batch workers REANALYSE sampled transitions with a target network
  (fresh search policies and values); one learner. Self-play network refreshed every 100 updates, target every 400.
- Replay 10⁶ (FIFO), batch 256, γ = 0.997, update-to-data 1, 5-step TD; Adam 3·10⁻⁴ (proprio), SGD 0.2 with
  momentum 0.9 (vision, Atari).

### Numbers

- Atari 100k: human-normalised mean 2.428, median 1.286 (EfficientZero 1.945 / 1.090; BBF 2.247 / 0.917;
  DreamerV3 1.120 / 0.490).
- DMControl, 20 tasks each: Proprio mean 723.2 (TD-MPC2 740.9, DreamerV3 517.1); Vision mean 726.1 (DreamerV3 498.5,
  +45%), records on 16 of 20. Sparse reward: Cartpole Swingup Sparse (vision, 200k) 763.6 vs DreamerV3 392.4.
- Compute (App. J.3, Walker Run): 1.3M parameters and 4.7·10⁷ FLOPs per decision vs TD-MPC2's 4.9M and 3.6·10¹⁰ —
  32 imagined latent states per decision against MPPI's 9,216.
- Over DreamerV3: better on 50 of 66 tasks.

## What it means for the Neural Turing Architecture (see `../BRAINSTORM.md` §3)

- **The dynamics are exact here.** For thoughts, G is the network itself and the reward is the verifier at the end,
  so the state and reward error terms of Cor. 4.3 vanish and SVE's error is the leaf-value term alone. SVE becomes
  MORE trustworthy than in EZ-V2, and the early-TD fallback — which exists to protect against a wrong model — matters
  less. With short episodes, full rollouts to the verdict (Monte Carlo) are also affordable.
- **What drops out by default:** `L_G` (temporal consistency) and `L_R` beyond the terminal reward. `L_G` returns
  if a cheap learned "thought dynamics" model is added to avoid running the full core at every search node.
- **The action is much higher-dimensional.** EZ-V2's continuous actions are DMControl actuator vectors — small. A
  thought is a d-dimensional vector (d = 64 … 1024). K = 16 Gaussian samples in hundreds of dimensions are nearly
  orthogonal noise, and the improvement argument (eq 7) is only asymptotic. Three ways round it, each an arm worth
  testing: (i) a LOW-dimensional action that steers the thought (`z = μ(h) + B(h)·u`, u ∈ ℝ^m with m ≈ 8–16);
  (ii) thoughts as mixtures over a codebook, so the action is CATEGORICAL and Gumbel MuZero's exact discrete
  guarantee applies — and the improved policy `π′`, used as mixture weights, is itself the thought (Soft Thinking's
  "concept token", produced by search); (iii) the paper's own remedy for large action spaces, the simple loss
  `−log p(a*_S)` (eq 9).
- **Reanalyse and priorities carry over directly:** a buffer of problems, re-searched with the current model;
  replay prioritised by Bellman error, which acts as a curriculum alongside the depth curriculum.
- **Cost.** Each simulation is one more thought position through the looped core (incremental, with a k/v cache).
  32 simulations per thought is a ~32× multiplier on plain sampling — which is why the small-budget behaviour
  (Fig. 3: 8 simulations still beat Sample MCTS) and a search-free default for easy steps (GCML) both matter.
