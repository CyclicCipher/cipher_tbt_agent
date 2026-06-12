# Training Pipeline for a Latent-Space Reasoning Model — Plan

> **Purpose.** The goal is a model that *thinks in latent space* — reasoning as iterative
> latent prediction (roll the latent forward toward an answer), not token-by-token. The
> central obstacle, confirmed in both the literature and our own experiments, is
> **autoregressive rollout drift**: a latent predictor that is excellent one-step degrades
> catastrophically over a multi-step rollout. This doc plans the training pipeline around
> solving that. Status: planning, pre-implementation.

## 1. The problem: compounding error / exposure bias

Train one-step on ground-truth inputs (teacher forcing), but at rollout feed the model its
own outputs → it operates on a distribution it never trained on → errors snowball.

- **Theory.** Ross & Bagnell / **DAgger** (AISTATS 2011): naive one-step (behaviour-cloning)
  training gives error growing like **O(ε·T²)** over horizon T; training on the model's *own
  rolled-out states* is what reduces it to **O(ε·T)**. The error lives in the **feedback
  loop**, not in any single layer.
- **Our empirical confirmation:**
  - **Data point #4** (LeWM rollout, `Theory/representability_and_learnability.md`):
    great-then-catastrophic decay; per-bin accuracy falls monotonically with horizon.
  - **PeriodicField** (fair, bounded, fully-determined task; `tasks/periodicfield.py`):
    baseline LeWM drifts **0.79 → 0.05** over 15 rollout steps — genuine compounding drift,
    *not* an information limit (the future is determined, bounded).
  - **TBAF negative** (3 experiments, `train_motif.py` / `train_lewm_periodic.py`): a pointwise
    activation cannot undo loop-accumulated error; the verbatim repo op is pathological, the
    corrected/common-mode variants are neutral.
- **Conclusion.** Drift is a **training-level** problem (the loss and the data distribution it
  induces), with secondary *architectural* levers (stability constraints, re-anchoring). We
  derived "you have to train against the loop" independently; the field agrees.

## 2. Solution families (design options, with evidence)

**A. Train on your own rollouts — PRIMARY (theory-backed, turns O(T²)→O(T)).**
Multi-step / on-policy training so the model sees its own error distribution.
- DAgger (Ross 2011); **Scheduled Sampling** (Bengio et al., NeurIPS 2015, arXiv:1506.03099);
  **Professor Forcing** (Lamb et al., NeurIPS 2016, arXiv:1610.09038).
- **Pushforward trick** (Brandstetter et al., *Message-Passing PDE Solvers*, ICLR 2022,
  arXiv:2202.03376): roll out during training, replace GT inputs with predictions. Caveat: it
  stops gradients through the predicted input, so it *fixes distribution shift but cannot learn
  the low-amplitude detail* needed for accuracy.
- **Already in JEPA:** [V-JEPA 2](https://arxiv.org/abs/2506.09985) uses a **multi-step rollout
  loss** to improve long-horizon accuracy — the same idea in latent space.

**B. Spectral / refinement — the deepest mechanistic insight.**
[PDE-Refiner](https://arxiv.org/abs/2308.05732) (Lippe et al., NeurIPS 2023): drift is the
**progressive loss of low-amplitude / high-frequency components** — a one-step MSE barely
penalises dropping them, but they dominate long horizons. Fix: a **diffusion-style multi-step
refinement** that forces modelling the full spectrum. (cf. **Diffusion Forcing**, Chen et al.,
NeurIPS 2024, arXiv:2407.01392; [PhysicsCorrect](https://arxiv.org/abs/2507.02227), a
training-free corrector.) Rhymes with our data: our one-step prediction was *great* (0.79); the
unmodelled residual is what compounded.

**C. Noise injection / robustification.**
**GNS** (Sanchez-Gonzalez et al., *Learning to Simulate*, ICML 2020, arXiv:2002.09405): inject
noise into training inputs so the model learns to correct off-manifold states. **Caveat the
recent literature flags:** real rollout errors are **structured, not random**, so random-noise
injection only partly helps (and is *why* naive-noise / TBAF-style fixes underperform).

**D. Limit the horizon / re-anchor (closed-loop).**
Don't roll far open-loop; re-observe and re-plan. Model-based RL (Dreamer: short imagined
horizons + λ-returns; MPC). In JEPA (2026): [FF-JEPA](https://arxiv.org/abs/2606.09311)
(hierarchical decomposition into short subproblems), [MIND-V](https://arxiv.org/abs/2512.06628)
(staged rollouts + filtering physically-implausible transitions). This is the *principled*
version of what TBAF gestured at — re-anchor with real information, not a pointwise invariance.

**E. Semantic guidance.** [ThinkJEPA](https://arxiv.org/abs/2603.22281): a VLM steers latent
forecasting to stay on-manifold over long horizons.

**F. Stability-constrained dynamics.** Contractive / Lipschitz / dissipative latent predictors
so errors can't amplify (ties to the chaos-vs-stable analysis: a stable system drifts
*polynomially*, a chaotic one *exponentially* — the latter is a hard horizon nothing fixes).
Stable SSMs. Note: [LeWorldModel](https://arxiv.org/abs/2603.19312)'s SIGReg targets *collapse*,
a different failure than drift.

## 3. Pipeline design recommendation (what to build, prioritised)

1. **Backbone: multi-step latent rollout loss.** Train the predictor on K-step rollouts (feed
   its own predicted latents), with K **scheduled up** during training (start 1, grow). This is
   the evidence-backed core (family A) and the first thing to try.
2. **If rollout loss plateaus: spectral-aware / refinement objective** (family B) — at minimum
   *monitor* spectral fidelity vs horizon; consider a denoising-refinement step per latent.
3. **Structural: keep the dynamics stable** (contractive / normalised predictor; family F) and
   **re-anchor periodically** to real observations where the task is closed-loop (family D).
4. **Anti-collapse: SIGReg / Sub-JEPA** (already implemented, `baselines/sigreg.py`) — orthogonal
   to drift but necessary for non-collapsed JEPA latents.
5. **Diagnostics (already built):** the **rollout-decay curve** (per-step fidelity vs horizon,
   `*_per_bin` logging) is the primary metric; add **spectral-fidelity-vs-horizon** as secondary.

## 4. Open questions — reasoning-specific (the honest gaps)

- **Transfer of the spectral view.** Physics/video drift is continuous/smooth; a *reasoning*
  rollout is discrete/semantic. Does "loss of low-amplitude detail" transfer, or is it
  "hallucination snowballing" (a different structure, from the LLM literature)? Unsettled.
- **The re-observation analog for pure reasoning.** Closed-loop re-anchoring assumes external
  observations to re-anchor to. Pure thinking has none — so the analog is likely **grounding
  intermediate latents against a verifier / a consistency constraint** (connects directly to our
  **data point #3**: cross-query/consistency was the strongest generalization lever we found).
- **Horizon scheduling.** How to grow rollout length during training without destabilising early
  training — the central hyperparameter of family A.

## 5. First experiment (reuses built infrastructure)

**Multi-step rollout loss on LeWM + PeriodicField.** Train the predictor on K-step latent
rollouts (not 1-step), measure whether the decay curve flattens vs the 1-step baseline we already
have (0.79→0.05). Decision: does *training against the loop* convert great-then-catastrophic into
graceful degradation? This is the V-JEPA-2 / pushforward idea on our own testbed, and the
cleanest first validation of family A before any larger pipeline commitment.

## References

- Ross & Bagnell, DAgger — *A Reduction of Imitation Learning and Structured Prediction*, AISTATS 2011.
- Bengio et al. — *Scheduled Sampling*, NeurIPS 2015 (arXiv:1506.03099).
- Lamb et al. — *Professor Forcing*, NeurIPS 2016 (arXiv:1610.09038).
- Sanchez-Gonzalez et al. — *Learning to Simulate Complex Physics with Graph Networks* (GNS), ICML 2020 (arXiv:2002.09405).
- Brandstetter et al. — *Message-Passing Neural PDE Solvers* (pushforward trick), ICLR 2022 (arXiv:2202.03376).
- Lippe et al. — [*PDE-Refiner*](https://arxiv.org/abs/2308.05732), NeurIPS 2023.
- Chen et al. — *Diffusion Forcing*, NeurIPS 2024 (arXiv:2407.01392).
- [*PhysicsCorrect*](https://arxiv.org/abs/2507.02227) (2025).
- [V-JEPA 2](https://arxiv.org/abs/2506.09985); [LeWorldModel](https://arxiv.org/abs/2603.19312);
  [ThinkJEPA](https://arxiv.org/abs/2603.22281); [FF-JEPA](https://arxiv.org/abs/2606.09311);
  [MIND-V](https://arxiv.org/abs/2512.06628).
