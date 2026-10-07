# JEPA-Anything / Orthogonal Predictive Factorization — arXiv 2609.20800 — reference notes

Cui, Wang, Xu, Liu, Yu, Zhang, Gao, Yang, Ouyang, Heng, Wu, Yin, Yang (2026), *JEPA-Anything: Learning Predictive
Models across Different Worlds*, arXiv 2609.20800 (17 Sep 2026). Code: github.com/Gen-Verse/JEPA-Anything (Apache-2.0;
the core library, task-design tools and a synthetic recipe — no domain datasets or trained weights). **Read
2026-10-06:** the README, `docs/architecture.md`, the `jepa-anything-core` README, and the paper's §2 (method), §3.3
(CITRIS, locomotion, molecules), §3.4 (wet lab, orbits) and Table 5 — not every experiment section.

## The mechanism, exactly

A JEPA (online encoder → context z_c; EMA target encoder → stop-gradient target z̃_t). OPF adds:
- K learned projectors P_k ∈ ℝ^{d×r} with **K·r = d**; factor targets z_t^(k) = P_kᵀ z̃_t (eq 3);
- K **dedicated predictors** q_k(z_c, s_t) → ℝ^r — every one reads the WHOLE context (eq 4);
- synthesis ẑ_t = (Pᵀ)† û_t (eq 5); with P orthogonal, ẑ_t = Σ_k P_k ẑ_t^(k);
- loss = Σ_k ‖ẑ_t^(k) − z_t^(k)‖² (eq 6) + λ_orth (‖P_kᵀP_k − I‖² + Σ_{i<j}‖P_iᵀP_j‖²) (eq 7) + a per-coordinate std
  floor on the factor targets and on the online encoder (eq 9, VICReg-style hinges) — added to each domain's own loss.
- **K and r are fixed hyperparameters** ("selected for computational scale"); K ∈ {1, 2, 4, 8} in the locomotion study.

**An observation the paper does not make:** with P exactly orthogonal, eq 6 equals ‖Pû − z̃‖² — the ordinary JEPA
loss on the synthesised prediction. Any rotation P gives the same loss to an unrestricted predictor. What OPF changes
is therefore (a) the predictor's ARCHITECTURE — K separate heads, each writing only into its own r-dimensional
subspace, with a learned rotation choosing which target directions share a head — and (b) the variance floors. It is a
modularity prior on the predictor's outputs, not a factorisation of the dynamics into independent mechanisms (every
head sees all of the context).

## Results (as reported)

- CITRIS Interventional Pong, against a dense, capacity-matched standard JEPA: single-intervention one-step MSE
  0.009541 → 0.006218 (−34.8%); combined interventions −12.9%; 6-step rollout −8.6%.
- Improves reported metrics on all 10 matched dynamics tasks; lowest one-step and 100-step errors on four molecular
  systems; locomotion (CEM planning) better on Walker2d and HalfCheetah, not Hopper.
- Table 5, against a capacity-matched UNCONSTRAINED multi-head: geometry only — cross-factor overlap 0.455 vs 5e-16,
  κ(P) 438 vs 1.00005, synthesis error 0.79 vs 3e-14. This shows the orthogonality penalty is enforced; it is not a
  prediction-error comparison, so the paper does not separate "orthogonal factors" from "multiple heads" on accuracy.
- A wet-lab intervention (IL-18 + CD73 blockade) nominated from factor coordinates, supported in co-culture, organoids,
  tumour fragments and mice.
- **Orbits:** "from simulated position–velocity trajectories without physical labels, spectral analysis pairs latent
  frequencies with orbital semimajor axes"; f ∝ a^(−3/2) recovered with fitted slope −1.4991, R² = 0.9999999 — "one
  analyzed run", no baseline.

## Reading the orbital result

Any representation that is a smooth, non-constant function of a periodic orbit's state repeats with the orbit's
period, so its spectrum peaks at the orbital frequency whatever was learned — the raw x coordinate, or an untrained
encoder, would give the same slope. R² = 0.9999999 is the signature of reading the simulator's own period back. The
semi-major axes come from the simulator. So the experiment shows the latent keeps time with the orbit; it does not show
a law was discovered, and with no control it cannot show factorisation helped. The paper itself calls it a
"diagnostic"; it does not claim to have learned Kepler's laws. A real test of "factorised prediction aligns latents
with laws": do some factors stay CONSTANT along each orbit (energy, angular momentum, the ellipse's orientation) while
one advances (the phase) — action-angle coordinates — and do a dense JEPA, unconstrained heads and a random encoder fail
to show it?

## For the Neural Turing Architecture

- The variance floors are the anti-collapse family already in BRAINSTORM §3.6 (VICReg/SIGReg).
- A floor on EVERY coordinate of EVERY factor is the opposite of letting unneeded factors switch off — our §15 saw
  surplus dimensions forced to carry variance halve a planner. A varying number of factors needs a price per active
  factor, split/merge of heads, or shared-weight slots — OPF has none.
- Good practice worth copying: capacity matching to < 0.3% parameters and < 2% predictor FLOPs, and geometry audits.
