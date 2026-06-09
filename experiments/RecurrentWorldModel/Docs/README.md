# RecurrentWorldModel — Docs Index

A recurrent-depth world-model reasoner built around **one weight-shared block that settles to equilibrium**, where perception, representation learning, reasoning, and the learning rule are all **clamp modes** of that single settling operator. 4GB VRAM hard constraint. Goal: a validated, scalable architecture (not a frontier model) — the Sapient HRM-Text bar (small, validated, open).

## Read in this order

1. **[architecture.md](architecture.md)** — the idea. The clamped settling core, its four modes, how it preserves JEPA's representational quality without the JEPA pipeline, how it kills COCONUT's token-replacement crutch, the energy↔diffusion equivalence, and the four risks.
2. **[implementation_plan.md](implementation_plan.md)** — the staged build (Stage 0→5), each stage adding one failure source on a validated base, each with a gate and a fallback. Repo layout and memory levers.
3. **[training_environment.md](training_environment.md)** — the glass-box test apparatus, reorganized around the four risks so every probe can falsify a specific architectural claim.

## The four risks (the spine of all three docs)

1. **Convergence** — does the block settle reliably and adaptively? (linchpin)
2. **Consistency ≠ correctness** — is goal-clamped settling real reasoning, or a fancy Hopfield net?
3. **Long-horizon credit assignment** — can a learned value make sparse-goal credit tractable over many steps?
4. **Representation/reasoning interference** — do SIGReg (isotropic spread) and reasoning (structured attractors) fight on shared weights?

## The single most decisive experiment

Train on mechanics A and B separately; test on **A∘B never seen together**. A pass = unified reasoning core. A fail = the system reduces to Ouro. (training_environment.md §7.)

## Status

- **Starting Docs/** — the original eight HTML notes this project synthesizes and revises. Reference material; not the working set.
- **Docs/** (this folder) — the working design set.
- **core/** — Stage 0 skeleton: settling block + DEQ wrapper + convergence instrumentation. No training loops here; run on GPU per the repo rule (Mistake #36).

## Lineage note

Where these docs depart from the Starting Docs, they say so. Key departures: (1) depth and "thought" recurrence are unified into one re-clamped settling loop, not two nested axes; (2) JEPA is de-scoped to the *Represent* mode, not the reasoning driver; (3) the time-series reservoir fork stays parked — the world-model rollout is the bet, a trained SSM the fallback.
