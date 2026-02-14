# EB-JEPA, Kona, and Energy-Based Reasoning: Research Notes

**Date:** 2026-02-14
**Status:** Reference document — informing future design decisions
**Relevance:** Directly related to our energy reasoning project (ePC-JEPA) and the generalization vs. memorization bottleneck

## Why This Matters to Us

We are building an ePC-JEPA system that uses energy minimization for reasoning.
Our early attempt (Langevin dynamics over latent z) failed — negative gap of -5%.
Three concurrent research threads shed light on why and suggest paths forward:

1. **EB-JEPA** (LeCun et al.) — planning via sampling-based energy minimization
2. **Kona 1.0** (Logical Intelligence) — reasoning via Langevin dynamics on thought tensors
3. **LeCun's AMI Labs** — the broader vision these instantiate

All three share our thesis: **reasoning is energy minimization**. The differences
are in what gets minimized, how, and over what domain.

---

## 1. EB-JEPA (Energy-Based JEPA)

### Citation

Terver, B., Balestriero, R., Dervishi, M., Fan, D., Garrido, Q., Nagarajan, T.,
Sinha, K., Zhang, W., Rabbat, M., LeCun, Y., & Bar, A. (2026). *EB-JEPA:
Energy-Based Joint-Embedding Predictive Architecture.* arXiv:2602.03604.

**Links:**
- Paper: https://arxiv.org/abs/2602.03604
- Code: https://github.com/facebookresearch/eb_jepa
- Blog/analysis: https://www.emergentmind.com/papers/2602.03604

### What It Is

A modular, educational library providing three progressively complex JEPA
implementations, each trainable on a single GPU in a few hours:

1. **Image JEPA** — Self-supervised representation learning on CIFAR-10
   (91% linear probe accuracy)
2. **Video JEPA** — Temporal prediction on Moving MNIST
3. **Action-Conditioned Video JEPA** — World model + planning in a "Two Rooms"
   navigation environment (97% planning success rate)

The library is explicitly designed as a bridge between theory (LeCun's position
paper) and practice (production-scale V-JEPA/I-JEPA codebases that are too
complex to learn from).

### Architecture

Modular building blocks:

- **Encoders:** ResNet-18, Vision Transformers, IMPALA — map observations to
  latent representations
- **Predictors:** UNet-based (spatial), GRU-based (temporal) — predict future
  representations given current state and optionally actions
- **Projectors:** MLPs mapping representations to lower-dimensional space for
  loss computation
- **Regularizers:** VICReg, SIGReg, temporal similarity, inverse dynamics
- **Planners:** MPPI and CEM optimizers — find optimal action sequences at
  inference time by minimizing energy in latent space

### Energy Function

**Training energy (loss):**
```
L = L_pred + beta*L_cov + alpha*L_var + delta*L_time_sim + omega*L_IDM
```

- `L_pred`: MSE between predicted and target future embeddings
- `L_cov`: Covariance regularization (decorrelate feature dimensions)
- `L_var`: Variance regularization (ensure feature spread)
- `L_time_sim`: Temporal similarity (adjacent frames → similar representations)
- `L_IDM`: Inverse Dynamics Model (recover actions from consecutive embeddings)

Best hyperparameters: `(beta, alpha, delta, omega) = (8, 16, 12, 1)` for
IMPALA-RNN on Two Rooms.

**Planning energy (inference):**
```
C(a, s_0, s_g) = Sum_{t=0}^{H-1} ||E(s_g) - P(z_t, a_t)||^2
```

Cumulative cost over ALL timesteps in the planning horizon. This is critical —
summing over the full trajectory makes planning robust to single-step
prediction errors.

### Critical Ablation Results (Table 4)

| Component removed | Planning success |
|-------------------|-----------------|
| Full model | 97% |
| No IDM | 1% (collapse!) |
| No variance reg | ~45% |
| No covariance reg | ~50% |
| No temporal similarity | ~62% |

**IDM is by far the most important component.** Without it, the encoder learns
spurious correlations and planning completely fails. This is directly relevant
to our generalization problem — see Section 5 below.

### Planning Mechanisms: MPPI and CEM

See Section 3 for detailed explanation.

### Collapse Prevention: SIGReg

See Section 4 for detailed explanation.

---

## 2. Planning by Sampling: MPPI and CEM

### The Core Idea

Both MPPI and CEM are **sampling-based, gradient-free** optimizers. They find
low-energy configurations by evaluating many random candidates rather than
following gradients. This is fundamentally different from our Langevin approach.

### MPPI (Model Predictive Path Integral)

**Citation:** Williams, G., Aldrich, A., & Theodorou, E. A. (2017). *Model
Predictive Path Integral Control using Covariance Variable Importance Sampling.*
arXiv:1509.01149.

**Link:** https://sites.gatech.edu/acds/mppi/

**Algorithm:**
```
Initialize: mu (mean action sequence), sigma (std), N samples, J iterations

For j = 1 to J:
    1. Sample N action trajectories: a_i ~ N(mu, sigma)     [i = 1..N]
    2. For each trajectory, rollout the world model:
       z_{t+1} = P(z_t, a_{i,t})
    3. Compute cost for each: C_i = Sum_t ||z_goal - z_t||^2
    4. Compute soft weights: w_i = exp(-C_i / tau) / Sum_j exp(-C_j / tau)
       (tau = temperature controlling exploration vs exploitation)
    5. Update mean: mu = Sum_i w_i * a_i
       (weighted average of ALL samples, not just elite)
    6. Optionally update sigma

Return: mu (the refined action sequence)
```

**Key properties:**
- **Gradient-free:** Does not differentiate through the world model. The world
  model is treated as a black box that maps (state, action) → next_state.
- **Soft weighting:** All samples contribute, weighted exponentially by quality.
  Low-tau = exploit (concentrate on best), high-tau = explore (spread weight).
- **Information-theoretic foundation:** Derived from a duality between free
  energy and relative entropy (KL divergence). The update minimizes KL between
  the current proposal and the optimal (Boltzmann) distribution over actions.
- **Robust to local minima:** Because it evaluates many diverse candidates
  simultaneously, it naturally explores the energy landscape broadly.

**Connection to TD-MPC2:** Hansen et al. demonstrated MPPI planning in learned
latent spaces achieves SOTA on 104 continuous control tasks (DMControl,
Meta-World, ManiSkill2, MyoSuite). Validates the approach at scale.

### CEM (Cross-Entropy Method)

**Algorithm:**
```
Initialize: mu, sigma, N samples, K elite count, J iterations

For j = 1 to J:
    1. Sample N action trajectories: a_i ~ N(mu, sigma)
    2. Rollout and compute costs C_i (same as MPPI)
    3. Select top-K trajectories (lowest cost) — the "elites"
    4. Fit new Gaussian to elites:
       mu = mean(elites)
       sigma = std(elites)

Return: mu
```

**Key difference from MPPI:** CEM uses **hard selection** (top-K only) with
**uniform weights** on elites. MPPI uses **soft exponential weighting** on all
samples. MPPI is generally slightly better (97% vs 96% in EB-JEPA) but CEM is
simpler to implement and tune.

### Contrast with Langevin Dynamics (Our Approach)

**Our Langevin:**
```
z_{t+1} = z_t - lr * grad_z E(z_t) + sqrt(2*lr) * noise
```

| Property | MPPI/CEM | Langevin |
|----------|----------|----------|
| **Gradient required?** | No | Yes (differentiable energy) |
| **Parallelism** | Evaluates N candidates simultaneously | Follows single trajectory |
| **Local minima** | Resistant (broad sampling) | Susceptible (follows gradient) |
| **Exploration** | Explicit (sampling from distribution) | Implicit (noise term) |
| **Convergence** | Refines distribution over iterations | Follows energy gradient |
| **Best for** | Multi-modal landscapes, action planning | Unimodal landscapes, constraint satisfaction |

**Why Langevin may still be right for us (long-term):**

Our intent with Langevin is fundamentally different from EB-JEPA's planning.
We are not searching for an action sequence — we are searching for a
**hypothesis** (which rule explains this data). This is a constraint
satisfaction problem: find z such that the predictor conditioned on z
correctly predicts the target for ALL in-context examples simultaneously.

Langevin's gradient-based approach is well-suited for constraint satisfaction
because:
1. The gradient directly points toward lower-energy (more consistent) regions
2. Each step refines the hypothesis based on ALL constraints at once
3. The noise term enables escape from shallow local minima

The problem is not Langevin itself — it's that our energy landscape over z has
no meaningful structure (the model ignores z entirely). Fixing the energy
landscape (via IDM, better training, or a separate energy network) would make
Langevin viable. Kona 1.0 proves this: it uses Langevin over thought tensors
and achieves 96.2% on hard Sudoku.

**When to consider MPPI/CEM for z:** If the energy landscape is multi-modal
(multiple valid hypotheses) or if gradient computation through the predictor
is too noisy, sampling-based optimization over z becomes attractive. This is
a future research direction, not an immediate priority.

---

## 3. SIGReg: A Better Collapse Prevention Mechanism

### Citation

Balestriero, R. & LeCun, Y. (2025). *Sketched Isotropic Gaussian
Regularization.* In: LeJEPA. arXiv:2511.08544.

**Links:**
- Paper: https://arxiv.org/abs/2511.08544
- Code: https://github.com/rbalestr-lab/lejepa
- Also: https://github.com/galilai-group/lejepa

### What It Does

SIGReg identifies the **isotropic Gaussian N(0, I)** as the optimal embedding
distribution for downstream tasks. Rather than separately enforcing variance
and covariance (as VICReg does with two hyperparameters), SIGReg directly
enforces Gaussianity using the **Cramer-Wold principle:**

1. Project embeddings onto K random 1D directions
2. For each projection, test Gaussianity using the **Epps-Pulley statistic**
3. Penalize deviations from Gaussian distribution

### Advantages Over VICReg

| Property | VICReg | SIGReg |
|----------|--------|--------|
| Hyperparameters | 2 (alpha for var, beta for cov) | 1 (regularization weight) |
| Stability | Sensitive to alpha/beta ratio | Robust across settings |
| What it enforces | Variance + decorrelation separately | Full Gaussianity jointly |
| Theoretical basis | Heuristic | Optimal distribution theorem |
| Complexity | O(d^2) for covariance matrix | O(K*d) linear |
| Interaction with L2 loss | Works but needs tuning | Naturally compatible |

### Relevance to Our System

We currently use VICReg with `lambda_var=1.0, lambda_cov=0.04`. Switching to
SIGReg would:
1. Reduce hyperparameter sensitivity (one fewer thing to tune)
2. Provide stronger theoretical guarantees against collapse
3. Be naturally compatible with our switch to L2 JEPA loss
4. Potentially improve the structure of the representation space, which could
   make z-conditioned prediction more effective

**Priority:** Medium. Worth implementing after we solve the multi-rule
decomposition problem, as a way to strengthen the representation space.

---

## 4. IDM and the Multi-Rule Decomposition Problem

### What IDM Is

An **Inverse Dynamics Model** predicts the action (or transformation) that
occurred between consecutive states:

```
a_predicted = IDM(enc(s_t), enc(s_{t+1}))
L_IDM = CE(a_predicted, a_actual)
```

In EB-JEPA, IDM forces the encoder to learn representations from which
**what happened** between frames is recoverable. Without it, the encoder can
learn shortcuts — representations that are predictive in aggregate but encode
spurious correlations rather than meaningful state transitions.

### Why IDM Might Be Critical for Multi-Rule Decomposition

Our core bottleneck (documented in `docs/hypotheses/generalization_vs_memorization.md`):
the model interprets 5 simple rules as 1 complex rule. It memorizes (99% train)
but doesn't generalize (~25% test). Even oracle z is ignored.

**The connection:** IDM forces the encoder to preserve **transformation-relevant
information** in its representations. In our domain, the "action" between
consecutive sequence elements IS the rule. An IDM analogue for our system would:

```
rule_predicted = IDM(enc(x_input), enc(x_output))
L_IDM = CE(rule_predicted, rule_id)
```

This is a loss that forces the encoder to maintain **rule-discriminative
representations**. If the encoder must produce representations from which the
rule can be recovered, it cannot collapse all 5 rules into a single complex
function — the representations would be identical across rules, making
`rule_predicted` impossible.

### Theoretical Analysis: Why This Might Break the Memorization Pattern

The multi-rule collapse hypothesis says the model treats all training data as
coming from one complex (noisy) function. This happens because:

1. The encoder has no incentive to preserve rule identity in its representations
2. The predictor can achieve high training accuracy by memorizing input→output
   mappings directly, bypassing rule identification entirely
3. Without rule-discriminative representations, z has nothing to condition on

IDM directly attacks problem (1). If the encoder must enable rule recovery from
consecutive (input, output) encodings, then:

- Different rules MUST produce distinguishable representation patterns
- The encoder is forced to learn that the data contains multiple distinct
  transformations, not one complex one
- z-conditioning becomes meaningful because the representation space now has
  rule-relevant structure that z can index into

### Caution: IDM Requires Supervised Rule Labels

Unlike EB-JEPA's IDM (which uses naturally available action labels), our
analogue requires **rule identity labels** during training. This is available
in our synthetic data (we generate it), but:

1. It wouldn't be available in a real-world setting where rule identities
   are unknown
2. It provides the model with information we ultimately want it to discover
   on its own
3. It's a stepping stone, not a final solution

**Possible unsupervised alternatives:**
- Contrastive IDM: same-rule pairs should produce similar IDM embeddings,
  different-rule pairs should produce different ones (requires knowing pairs)
- Clustering-based IDM: predict a learned discrete code instead of a label
- Consistency IDM: same rule applied to different inputs should produce
  consistent transformation signatures

### Relationship to Our Proposed Next Steps

From `generalization_vs_memorization.md`, our proposed experiments include:

1. **Curriculum learning** — train one rule at a time
2. **Explicit rule token** — prepend rule ID to sequences
3. **Extended training** — test for grokking over 500-5000 epochs
4. **Scaling rules gradually** — test with 2, 3, 4, 5 rules

IDM is complementary to all of these and could be tested in combination:
- Curriculum + IDM: does IDM accelerate per-rule learning?
- IDM alone: does it enable multi-rule generalization without curriculum?
- IDM + extended training: does it enable faster grokking?

**Priority:** High. IDM (or an analogue) addresses the root cause identified
in our hypothesis — the encoder's failure to learn rule-discriminative
representations. This is a strong candidate for the next experiment.

---

## 5. Kona 1.0 and EBM-CoT

### Citations

Chen, Y. et al. (2025). *Think Consistently, Reason Efficiently: Energy-Based
Calibration for Implicit Chain-of-Thought.* arXiv:2511.07124.

Bodnia, E. et al. (2026). Logical Intelligence — Kona 1.0. Commercial release.

**Links:**
- EBM-CoT paper: https://arxiv.org/abs/2511.07124
- Kona announcement: https://www.businesswire.com/news/home/20260120751310/en/Logical-Intelligence-Introduces-First-Energy-Based-Reasoning-AI-Model-Signals-Early-Steps-Toward-AGI-Adds-Yann-LeCun-and-Patrick-Hillmann-to-Leadership
- Kona EBM details: https://logicalintelligence.com/kona-ebms-energy-based-models
- Analysis: https://paperlens.io/idea/kona-1/

### What Kona Does

Three-stage pipeline:

1. **Thinking:** Problem encoded into continuous "Soft Thought" tensor
   (~32-dimensional latent thought tokens)
2. **Reasoning (Langevin):** Energy function E(z) measuring logical consistency
   is minimized via ~3 Langevin steps:
   `z_{t+1} = z_t - alpha * grad_z E(z_t) + noise`
3. **Generation:** Only when energy is minimized is the thought decoded into
   text or actions

### Performance

- 96.2% solve rate on hard Sudoku puzzles in 313ms (single H100)
- Leading LLMs (GPT-5.2, Claude Opus, Gemini, DeepSeek) achieve ~2% on same task

### Why Kona Works and Our Langevin Doesn't (Hypotheses)

| Aspect | Kona | Our ePC-JEPA |
|--------|------|-------------|
| Energy function | Separately trained EBM (MLP) that explicitly scores reasoning quality | Raw JEPA prediction loss |
| Latent space | Rich "Soft Thought" tensor | Low-dimensional z vector (~16-64) |
| Energy training | EBM trained specifically to discriminate good vs bad reasoning | Energy is a byproduct of prediction training |
| z conditioning | Thought tensor IS the representation | z is an auxiliary input the predictor can ignore |

**Key insight from Kona:** The energy function used for Langevin optimization
should be a **separately trained component** that explicitly learns what makes
a good hypothesis/reasoning trace. Using the raw prediction loss as energy may
not provide discriminative gradients over z.

### Leadership

- **Yann LeCun:** Founding chair of Technical Research Board
- **Michael Freedman:** Fields Medalist, Chief of Mathematics
- **Vlad Isenbaev:** ICPC World Champion, Chief of AI
- **Eve Bodnia:** Founder and CEO

---

## 6. LeCun's Broader Vision and AMI Labs

### Citation

LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence.* OpenReview.

**Link:** https://openreview.net/pdf?id=BZ5a1r-kVsf

### The Six-Module Architecture

LeCun proposed six modules for autonomous intelligence:

1. **Perception module** → EB-JEPA encoder
2. **World model** → EB-JEPA predictor
3. **Cost module** → EB-JEPA planning cost function
4. **Memory module** → Not yet implemented in EB-JEPA
5. **Actor module** → MPPI/CEM planner
6. **Configurator** → Not yet implemented

EB-JEPA implements modules 1-3 and 5. Kona implements a version of module 6
(orchestrating reasoning). Our ePC-JEPA adds biologically-inspired local
learning (ePC error nodes) as a training mechanism.

### AMI Labs

LeCun left Meta in late 2025 and founded **AMI Labs** (Advanced Machine
Intelligence Labs), Paris, valued at ~$3.5B. Focus: world models based on
JEPA for industrial process control, automation, robotics, healthcare.

**Links:**
- MIT Technology Review: https://www.technologyreview.com/2026/01/22/1131661/yann-lecuns-new-venture-ami-labs/
- TechCrunch: https://techcrunch.com/2026/01/23/whos-behind-ami-labs-yann-lecuns-world-model-startup/
- Fortune (valuation): https://fortune.com/2025/12/19/yann-lecun-ami-labs-ai-startup-valuation-meta-departure/

### The Three-Layer Ecosystem

LeCun envisions: LLMs handle language, Kona handles reasoning, world models
handle physical understanding — all unified by energy minimization.

---

## 7. Comparison Table: All Three Systems

| | Our ePC-JEPA | EB-JEPA | Kona 1.0 |
|---|---|---|---|
| **Domain** | Sequence rule induction | Visual world modeling | Logical/symbolic reasoning |
| **What gets minimized at inference** | Latent z (hypothesis) | Action sequences | Thought tensors |
| **Optimization method** | Langevin (gradient) | MPPI/CEM (sampling) | Langevin (gradient) |
| **Energy measures** | JEPA prediction loss | Goal distance in representation space | Logical consistency |
| **Encoder training** | ePC local learning (biologically inspired) | Standard backprop | Standard backprop |
| **Collapse prevention** | VICReg | VICReg or SIGReg | Learned energy function |
| **Result** | -5% Langevin gap | 97% planning success | 96.2% Sudoku |
| **Key insight** | Local learning via error nodes | IDM prevents spurious correlations | Separately trained energy function |

---

## 8. Papers to Cite (For Our Three Research Papers)

### For the Mamba3 upgrades paper:
- Dao & Gu 2024 — Mamba2/Mamba3, SSD (arXiv:2405.21060)

### For the energy reasoning paper:
- Goemaere et al. 2025 — ePC (arXiv:2503.01811)
- Assran et al. 2023 — I-JEPA (arXiv:2301.08243)
- Bardes et al. 2022 — VICReg (arXiv:2105.04906)
- Terver et al. 2026 — EB-JEPA (arXiv:2602.03604)
- Balestriero & LeCun 2025 — SIGReg/LeJEPA (arXiv:2511.08544)
- LeCun 2022 — Path Towards Autonomous Machine Intelligence (OpenReview)
- Chen et al. 2025 — EBM-CoT (arXiv:2511.07124)
- Williams et al. 2017 — MPPI (arXiv:1509.01149)

### For the generalization vs. memorization paper:
- Power et al. 2022 — Grokking (arXiv:2201.02177)
- Michaud et al. 2023 — Quantization Model (arXiv:2303.13506)
- Wang et al. 2024 — Grokked Transformers (arXiv:2405.15071)
- Liu et al. 2022 — Understanding Grokking (arXiv:2205.10343)
- Fan et al. 2024 — Deep Grokking (arXiv:2405.19454)
- deMoss et al. 2024 — Complexity Dynamics (ScienceDirect)
- Terver et al. 2026 — EB-JEPA ablation showing IDM necessity (arXiv:2602.03604)

---

## 9. Action Items (Prioritized)

1. **HIGH — IDM analogue for rule discrimination:** Add a loss term forcing the
   encoder to preserve rule-discriminative information. Test whether this breaks
   the multi-rule collapse. (See Section 4.)

2. **MEDIUM — SIGReg replacing VICReg:** Implement after solving multi-rule
   decomposition. Single hyperparameter, better theoretical guarantees.

3. **MEDIUM — Separate energy network for z:** Train an explicit E_phi(z, context)
   that scores hypothesis quality, following Kona's approach. Use this for
   Langevin instead of raw JEPA loss.

4. **LOW — MPPI/CEM for z:** If the energy landscape over z remains multi-modal
   after fixing the energy function, try sampling-based optimization. Not
   a priority until the energy landscape has meaningful structure.

5. **RESEARCH — Cumulative cost over prediction horizon:** EB-JEPA's planning
   sums cost over ALL timesteps. Our z optimization should consider full
   sequence prediction quality, not just aggregate loss.
