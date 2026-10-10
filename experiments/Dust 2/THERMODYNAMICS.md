# Thermodynamics as a source of ideas for perturbation training

Started 2026-10-10. The user asked: "what could thermodynamics teach us that might be relevant to improving this training
algorithm?" Then they asked to record the answer as a source of experiment ideas and run the good ones "at the
appropriate time".

The references here are cited from memory and not re-read this session. They are pointers, not checked claims.

## 0. The frame: Dust's cost is an information limit

- **The law.** Each draw gives each token ONE number (its loss change) about an error vector of width D. Experiment 1
  measured the consequence: per-token cosine = √(K/(K+D)), exactly.
  - **Pairing draws:** removing the second-order term with antithetic pairs did not help.
  - **Depth:** made no difference.
- **The information reading.** The cost of a step is (bits needed about the error) ÷ (bits each measurement gives). So
  there are exactly two ways to make it cheaper:
  - **Need fewer bits.** Use a better prior on where the error lies: subspaces, a per-token guess, sparsity. These are
    Experiments 1 and 2.
  - **Get more bits per measurement.** Measure a vector, not a scalar.

Thermodynamics and statistical physics have ideas on both sides.

## 1. Let the system's own dynamics carry the error (more bits per measurement)

- **The idea.** Several learning rules from physics get the gradient from the difference between two settled states of
  the system:
  - one settled freely;
  - one settled with its output gently nudged toward the target.

  The rules are Boltzmann machines (Hinton & Sejnowski, 1983–86), Equilibrium Propagation (Scellier & Bengio, 2017), and
  "coupled learning" in physical flow and elastic networks (Stern, Hexner, Rocks & Liu, 2021).
- **What it buys.** The per-unit difference IS the error signal, a whole vector per pair of settlings, whatever the
  width. There is no backward pass, and the learning rule is local.
- **What it needs.** A network whose state SETTLES through feedback: energy-based, or at least a fixed-point recurrent
  model. A one-way feed-forward stack cannot carry a nudge at the output back to its early layers.
- **Why it matters here.** A looped Neural Turing Architecture model could be built to settle, and this repo once had a
  DEQ "settling core" (removed 2026-06). This is the most promising and most invasive idea: it shapes the architecture,
  not only the training loop.
- **First measurement (T4, later):** in a looped block run to its fixed point, does (nudged state − free state) at each
  unit line up with backprop's error? It needs a block built to settle, so it waits for the architecture work.

## 2. Fluctuation–dissipation: the system's own noise is the probe

- **The theorem.** A system's response to a small push equals the correlation between its spontaneous fluctuations and
  the quantity measured. So a network that is noisy anyway, like stochastic, spiking or event-driven units, has its
  perturbations for free. The learning signal is the covariance between each unit's noise and the loss: REINFORCE,
  node perturbation. The songbird's song learning seems to work this way: a dedicated nucleus (LMAN) injects
  variability into the motor pathway.
- **What it changes for Dust.** Dust perturbs ONE site per pass and reruns from cached clean activations. A
  fluctuation–dissipation version perturbs EVERY site in every pass, independently, and scores each site's noise by the
  same per-token losses:
  - no clean pass for the estimate (the draw mean already serves as the baseline);
  - no rerun-from-cache, so **no stored activations at all**;
  - every forward pass is both inference and a training probe.

  This is the strict form of the user's "no record of activations" property (`IDEAS.md` §6).
- **What it costs.** Each site's estimate now also hears every other site's noise. The prediction is more noise per draw,
  partly paid back by ~8× more draws at the same compute: one pass serves 17 sites. Experiment T3 measures the net
  price.

## 3. Free energy, not energy: σ is a temperature

- **What Dust optimises.** Dust follows the gradient of the loss averaged over its own noise, E[L(·+σa)]. That is a
  Gaussian-smoothed loss, which behaves like a free energy at temperature σ². Noisy optimisation favours wide, flat
  minima, because they hold more volume (entropy); Entropy-SGD (Chaudhari et al., 2017) uses this deliberately.
- **Why it matters.**
  - It is a candidate explanation for Dust beating backprop at small token budgets, which its authors called unclear.
  - σ is then a regulariser and an exploration knob, not just a finite-difference step. Annealing it (high early, low
    late) is the textbook thermodynamic schedule.
- **Experiment T1.** σ held at 0.1, 0.2 (now) or 0.4, against σ annealed 0.4 → 0.05. Report the validation loss and
  the train−validation gap.

## 4. Boltzmann weighting of the draws

- **The estimator.** Path-integral control (MPPI; Williams, Theodorou et al.) and the cross-entropy method weight each
  draw by exp(reward/λ), which is a Boltzmann weight. The update is the weighted mean of the perturbations.
  - **Large λ:** it becomes Dust's linear estimator.
  - **Small λ:** it trusts the best draws.
- **Why it matters.** It is a principled alternative to the user's rank-based fitness shaping against heavy-tailed loss
  changes, the avalanches of a critical sparse network.
- **Experiment T2.** Per token and per site, across its K draws, against the linear baseline:
  - weights softmax(r/λ), with λ set relative to the spread of r: λ = 1, 0.3;
  - centred ranks (the NES form the user's plan names).

  Measured by the cosine diagnostic and by training. On smooth losses I expect no gain, because the linear estimator is
  already optimal for a linear response. The real test of shaping is a heavy-tailed (sparse, critical) network, so T2
  here is a calibration, not the final word.

## 5. Soft modes near criticality, and why the errors are low-rank

- **The prediction.** Near a critical point, a system's response to any push is dominated by a few slow, soft modes.
  That predicts the low effective rank of the true errors measured in Experiment 1: 2–18 of 64–256 dimensions.
- **What it means.**
  - The subspace worth probing is the network's soft modes.
  - The user's critical-sparsity plan and subspace-guided perturbation should help each other: the closer to
    criticality, the lower the rank of the errors, so the fewer draws are needed.
- **Experiment (later, with the sparse network):** effective rank of the errors against a measure of criticality, such
  as the branching ratio of activity.

## 6. Momentum against diffusion (Hamiltonian Monte Carlo)

**The lesson.** Isotropic random proposals explore a D-dimensional space diffusively, at a cost that grows with D.
Hamiltonian Monte Carlo moves with persistent momentum and scales far better.

**The analogue for us.** Probe along persistent directions: the optimiser's momentum, recent error estimates. That is
Experiment 1's guided arm (2.3× in cosine) and its natural variants. It is listed here so the connection is not lost.

## Plan and status

| id | idea | when | status |
|---|---|---|---|
| T1 | σ as temperature (levels + anneal) | Experiment 3, after Experiment 2 | to pre-register |
| T2 | Boltzmann and rank weighting of draws | Experiment 3 | to pre-register |
| T3 | fluctuation–dissipation: all sites perturbed at once, no cache | Experiment 3 | to pre-register |
| T4 | nudged-minus-free state in a settling block (Equilibrium Propagation) | with the looped architecture | design |
| T5 | error rank against criticality | with the sparse, critical network | design |
