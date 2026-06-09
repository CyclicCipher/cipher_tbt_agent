# Architecture — The Clamped Settling Core

> **Status:** working design document. Companion to `implementation_plan.md` and `training_environment.md`.
> **Lineage:** synthesizes and revises the eight `Starting Docs` (architecture, representation_learning, learning_algorithm, project_objectives, reservoir_computing, training_environment, metacognition, research_agenda). Where it departs from them, it says so.
> **Hard constraint:** 4GB VRAM (RTX 3050 Ti).

**Status legend:** `established` empirically confirmed · `directional` promising, unproven · `open` active research question · `theoretical` unconfirmed theory · `commitment` project decision.

---

## 0 · The One-Sentence Thesis

Build **one weight-shared recurrent block that settles to equilibrium**, and obtain perception, representation learning, multi-step reasoning, and the learning rule itself as **different clamp modes of that single settling operator** — not as separate networks.

This is the resolution of a dilemma the Starting Docs left open: either nest two recurrence axes (depth × thought), which is hard and may not co-train (Starting `architecture` A1), or build depth recurrence alone, which is "Ouro with tweaks." The dilemma is false. Depth and thought are only two *mechanisms* if you build two. Make a reasoning step a **re-clamp-and-re-settle** of the same block, and there is one mechanism with a moving boundary condition.

---

## 1 · Core Commitment: Recurrent Depth (unchanged)

`commitment`

The spine is unchanged from Starting `architecture` §0: **there is no variant of this model that is not a recurrent-depth model.** A single block `f_θ` is applied repeatedly to refine a latent state. Justification is unchanged — adaptive compute (spend more iterations on harder inputs), composition is inherently iterative, and weight-shared recurrence buys effective depth without parameter growth (the only way to reach useful depth on 4GB).

What changes is everything *around* the spine. The Starting Docs hung three semi-separate systems off it — JEPA (objective), COCONUT (a second recurrence), reservoir computing (a third, for time-series). This document collapses all three into clamp modes of the one settling loop.

---

## 2 · The Operator

`directional`

### 2.1 Units, not layers

The block operates over a pool of latent **units**, partitioned by role. At any moment each unit is either **clamped** (held fixed, a boundary condition) or **free** (relaxed by the settling dynamics):

| Unit group | Role | Clamped when… |
|---|---|---|
| **Sensory** | encode current observation (thin per-modality adapter writes here) | an observation is present |
| **Working memory (scratch)** | hold conclusions from prior settlings; the substrate of a reasoning chain | carrying a prior conclusion forward |
| **Goal / preference** | the preferred outcome the trajectory should reach (telos) | a goal is set |
| **Output / readout** | thin head reads the relevant units after settling | never (always free, always read) |

Multimodality is *which units are clamped*, not which network you route to (Starting `architecture` §6b). Adding a modality = adding a thin adapter that writes the sensory units. No reconstructive decoder anywhere.

### 2.2 Settling

`open` (linchpin — see Risk 1)

Given the current clamp configuration, iterate `h_{l+1} = f_θ(h_l, clamps)` until the free units reach equilibrium `‖h_{l+1} − h_l‖ < ε`. The equilibrium is an **attractor**: the settled state that is most consistent with the clamps. This is the Deep Equilibrium (DEQ) view; implicit gradients train it in O(1) memory (Starting `architecture` §4b).

The block itself is the relational-attention block of Starting `representation_learning` §3 — relations as a monoid acting on embeddings, the polar **hyle/morphe** split (magnitude = content, phase = structure), softmax as the discrimination/classifier — plus RMSNorm and QK-Norm. None of that changes; it is the *per-iteration* operator. Over a settling, composed attention = composed morphism.

> **The what/where (hyle/morphe) split is RESOLVED — it is achievable, cleanly and exactly.** `established` The Starting Docs flagged the polar content/position decoupling as the **open** question Q11 ("can the magnitude=content / phase=position split be realized cleanly inside attention, or is the mixing unavoidable?"). **PoPE** (Polar Coordinate Positional Embeddings, Gopalakrishnan et al. 2025, [arXiv:2509.10534](https://arxiv.org/abs/2509.10534)) answers it affirmatively and with a proof: the attention score factorizes **exactly** into a conjunction of a *what*-match and a *where*-match,
> `a_ts = Σ_c μ_q,c μ_k,c · cos((s−t)θ_c + δ_c)`, where the magnitude product `μ_q μ_k` (= `softplus` of the raw features) depends **only** on content and the cosine depends **only** on relative position. RoPE could not do this because the rotary inner product carries a content-dependent phase cross-term `φ_k − φ_q`; PoPE removes it by putting content in magnitude and position in a fresh phase. The decoupling is by construction of the coordinate system, not a learned penalty — exactly the §3b claim. The only residual coupling is an **optional** per-frequency phase bias `δ_c ∈ [−2π,0]`. Cost: the QK score runs in `2·head_dim` (the paper's "d frequencies"). Implemented as `PolarPositionalEmbedding` in `core/block.py` (`pos_enc="pope"`, the project default) and verified by a decoupling unit test (content magnitude is position-invariant; score is translation-invariant). Empirically PoPE also gives **zero-shot length extrapolation to 10× longer sequences** and turns the what+where indirect-indexing diagnostic from RoPE's 11% to 95% — the two properties that directly address the Stage-0 OOD confound. **Q11 is closed.**

> **Expect strange behavior — this is not Ouro.** `open` A single weight-tied block iterated to a fixed point is the *pure* DEQ form (Bai et al. 2019). It is genuinely unlike Ouro/LoopedLM, which loop a **multi-layer stack** with weights shared across the outer iterations — there, one "iteration" is still a deep feed-forward computation. Ours, in the minimal form, is one nonlinear map applied to its own output. That has **low per-iteration expressivity** and can produce attractor structure unlike anything in current architectures; do not expect it to behave like a looped transformer, and read Risk 1 results with that in mind.
>
> **The design lever:** DEQ semantics only require the fixed-point map `z → f(z, x)` to be a single map — `f` itself may internally be a small *stack* of `k` layers (still weight-tied across the outer settling iterations). So "one block" is a knob, not a constraint: `k = 1` is the pure/strange minimum (most interpretable convergence behavior, least expressivity per step); `k > 1` moves toward Ouro-like per-iteration depth at the cost of more activations to settle. Stage 0 starts at small `k` and treats it as a hyperparameter to sweep if convergence (Risk 1) or the matched-baseline gate misbehaves. The current `core/block.py` implements `k = 1`; a `n_layers` field is the intended extension point.

### 2.3 Four modes — one operator

`directional`

Everything the model does is settling under a different clamp configuration:

1. **Represent** (this *is* JEPA, as a clamp mode). Clamp the visible sensory units, **free the occluded/future units**, settle, read the predicted latent off the freed units. Train it against the target encoder's latent for those units, regularized by SIGReg/VICReg. This preserves JEPA's representational quality (see §3) without the JEPA training apparatus. It is Starting `architecture` A9 ("clamped settling replaces encoders") taken as the primary path, not a deferred fallback.

2. **Perceive / infer.** Clamp the sensory units, settle. The attractor reached *is* the encoding. No deep encoder stack.

3. **Reason.** Clamp working-memory with the last settled conclusion (and the goal units with the target), settle again. Each settle is one reasoning step; the sequence of settlings is the reasoning trace. **This replaces COCONUT** (see §4).

4. **Learn.** Clamp the goal/target *and* the input, settle to a target-consistent equilibrium, then consolidate weights locally toward that equilibrium (prospective configuration), filtered by item 52's signal/noise decomposition. Inference and learning are the same relaxation with different clamps (Starting `learning_algorithm` §5).

The win: **representation and reasoning share substrate.** The same settling that builds clean object-attractors is what reasoning traverses, so we are not building clean representations and then bolting on a reasoner that re-contaminates them.

---

## 3 · Preserving JEPA's Representational Quality Without the JEPA Pipeline

`directional`

JEPA is uniquely good at "cat as cat, not cat+background." We need that quality — it is the **prerequisite for reasoning**: reasoning composes relations over objects, and contaminated objects compose garbage. But the quality is not the pipeline. It reduces to three **portable ingredients**, all expressible as the *Represent* mode (§2.3):

1. **Latent prediction target** (never pixels/tokens) — nothing rewards encoding nuisance detail; the target is itself a representation, so anything that does not help predict *other* representations is dropped.
2. **Predict-the-occluded-across-views/time** — background varies independently of the object → unpredictable gradients → suppressed; the object persists/moves coherently and predicts its own hidden parts → survives. This is the mechanism that yields object-coherence.
3. **Anti-collapse** (SIGReg, provable; Cramér-Wold isotropy) — keeps the space spread and usable.

None of these requires the EMA target encoder (LeWorldModel dropped it) or a curriculum. We keep the magic, discard the apparatus. The target for the *Represent* mode can be a stop-gradient pass of the same block (the EMA twin becomes a stop-grad mode of the one operator, not a second network) — to be tested against an explicit EMA copy if the shared version collapses.

**Scope discipline** (Starting `project_objectives` caveat): joint-embedding methods already give object coherence largely for free. The novel burden of proof is on the *compositional* layer (reasoning), not on re-proving object coherence. Do not spend budget there.

---

## 4 · Killing COCONUT's Token-Replacement Crutch

`commitment` (avoid) + `directional` (replacement)

COCONUT's curriculum — progressively replacing language reasoning tokens with continuous thoughts — exists for exactly one reason: it generates its latent chain **autoregressively from token-space** and has no supervision for what each next-latent should be, so it needs a hand-built bridge from language CoT into latent reasoning. We reject this crutch (it is unsolved, brittle, anti-bitter-lesson, and needs CoT traces).

The crutch has nothing to support once the "step" is **re-clamping a settled attractor into working memory** rather than feeding a hidden state back as a token embedding. We are **never in token-space for the chain**, so there is no bridge to cross. The reasoning trace is a sequence of settlings over working-memory clamps, shaped by the objective in §5 — not by a token-replacement schedule.

---

## 5 · The Objective: Iterative Refinement Toward a Clamped Goal

`directional`

The Starting `project_objectives` §3 already argues that next-token prediction is the wrong value function for reasoning and the right target is **goal-state achievement** (HRM task completion / MuZero latent planning). The Starting `architecture` then contradicts this by asking JEPA *self-prediction* to drive reasoning. We resolve the tension: **JEPA grounds the arena (Represent mode); the goal grounds the reasoning (Reason mode).**

Reasoning is **iterative refinement toward a clamped goal**, expressible in two equivalent vocabularies:

- **Energy relaxation** (active inference / prospective configuration): a thought is a step of descent on a free energy; the goal is a clamp; halting = energy minimized. The principled endpoint, matching the active-inference frame.
- **Latent diffusion** (DiffusionBlocks item 85; diffusion LLMs item 19): a thought is a denoising step; reasoning iteratively denoises from noise toward the answer, conditioned on the question; the weight-shared block *is* the denoiser; "think longer" = more denoising steps = adaptive compute. Intermediate states are the thoughts — unsupervised, shaped only by the denoising objective + final answer. No token replacement, no autoregressive drift (parallel refinement, not left-to-right generation).

These are the **same thing**: score-based diffusion is gradient descent on an energy (score = −∇ energy). Both collapse depth-recurrence and step-recurrence into one iterative-refinement loop. **Decision:** prototype in the diffusion vocabulary (well-understood training, no settling-convergence prerequisite), migrate toward the energy vocabulary as the convergence/learning machinery matures.

### 5.1 The long-horizon piece — a learned value

`open` (Risk 3)

Prospective configuration assigns credit *within* a settle. *Across* many re-clampings, with a sparse goal signal, there is still a temporal credit-assignment problem. This does not dissolve; it must be added: a **learned value** (MuZero-style bootstrap) that makes the sparse goal dense — estimating "how close is this settled state to reaching the goal." This is the one genuinely additional component beyond the single operator, and the AlphaZero/MuZero lineage is the scaling precedent that makes the whole reasoning bet credible (latent planning toward a value has scaled; latent self-prediction into reasoning has not).

---

## 6 · Halting

`directional`

No EOS token (Starting `architecture` §1f). Two nested halting signals, both reuses of the settling machinery:

- **Per-step (depth):** stop the inner loop when the free units reach the fixed point (`‖Δh‖ < ε`).
- **Per-problem (chain):** stop re-clamping when a further settle does not change the conclusion (the chain reaches its own fixed point) **or** when the value estimate says the goal is reached. Under the active-inference frame, both are "free energy minimized = goal state reached."

The control policy over halting (how hard to think per step, how many steps, when to branch/prune — AutoTTS item 8) is where metacognition attaches (Starting `metacognition` §2). Deferred until the core settles reliably.

---

## 7 · What This Is and Is Not

| | |
|---|---|
| **Not Ouro** | Ouro is the *degenerate mode*: fixed input, single fixed point, no goal clamp, no working memory. This generalizes it. Ouro is its off-switch — the graceful-degradation floor. |
| **Not COCONUT** | No token feedback, no curriculum, no autoregressive latent generation. |
| **Not vanilla JEPA** | The predictor is the same settling block under a different clamp, not a separate narrow transformer. JEPA is one mode, not the whole system. |
| **Not reservoir computing** | Time-series is handled (if at all) by the world-model rollout (Reason/Represent modes rolled forward), not a frozen reservoir. The reservoir fork (Starting `reservoir_computing`) stays parked. See §8. |

The risk profile is favorable: **the floor and the ceiling are the same codebase at different clamp settings.** If goal-clamped and diffusion modes fail to produce real reasoning, the system degrades to a working Ouro. If they work, it is a unified energy/diffusion world-model reasoner. We are not betting the project on the hard part; we build the general machine and discover empirically how many modes light up.

---

## 8 · The Time-Series Question, Deferred Cleanly

`open`

The Starting `reservoir_computing` fork exists because the DEQ core *converges* (kills dynamics) while time-series needs *sustained* dynamics. In this design the bet (per the user) is that **the world-model rollout** — Represent/Reason modes rolled forward across world-time — supplies temporal modeling, so reservoirs may be unnecessary. If that bet fails empirically (the rollout cannot hold or forecast temporal structure), the fallback is **not** a frozen reservoir but a **trained linear-recurrent (SSM/Mamba-style) world-time substrate** on a separate clock from the settling loop — sidestepping the spectral-regime incompatibility because a trained SSM need not be a fixed edge-of-chaos reservoir. Note: a partial Mamba3 implementation already exists elsewhere in this repo (`experiments/Mamba3/`, with known MIMO bugs — Mistake #47) and could seed that fallback. Do not reach for it until the rollout bet is tested.

---

## 9 · The Four Risks (carried into the test plan)

These are the failure points the architecture must be probed against. Each maps to an experiment in `training_environment.md`.

1. **Convergence (linchpin).** `open` — Energy/equilibrium-recurrent nets can oscillate or settle to spurious attractors. *Does the block settle reliably, fast enough that per-step settling is affordable?* If not, the whole "one operator" story is unstable. First thing to measure. (Starting `architecture` A8.)

2. **"Consistent" ≠ "correct."** `open` — Goal-clamped relaxation yields the most internally *consistent* state. Is that genuine multi-step inference, or a fancy Hopfield net that settles to the nearest stored pattern? Tests whether the energy landscape has reasoning structure, not just associative recall.

3. **Long-horizon credit assignment.** `open` — Across many re-clampings with sparse reward, can the learned value (§5.1) make credit assignment tractable, or does the chain fail to learn beyond a few steps?

4. **Representation/reasoning interference.** `open` — SIGReg wants isotropic spread; reasoning wants structured attractors. On shared weights they may fight. Does training one mode degrade the other?

---

## 10 · Relationship to the Spine's Other Open Questions

Carried forward, unchanged in status, from Starting `architecture` §7:

- **Composition fidelity** (Q1/A4): do learned relation-transformations actually compose (`M_{r2}M_{r1} = M_{r2∘r1}`)? The relational substrate's central empirical question; orthogonal to the four risks but tested alongside.
- **DiffusionBlocks vs. recurrence semantics** (A5): block-wise training assumes independence the shared loop partly violates. Relevant once memory forces block-wise training.
- **Hierarchy** (A7): the H/L slow/fast split stays *out* of v1, added last only if it beats single-timescale at matched compute.
