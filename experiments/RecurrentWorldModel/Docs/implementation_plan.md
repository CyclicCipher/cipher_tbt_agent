# Implementation Plan — Staged Build of the Clamped Settling Core

> **Status:** working plan. Companion to `architecture.md` (the idea) and `training_environment.md` (where each stage is tested).
> **Principle (Starting `project_objectives` §2):** each stage adds **exactly one** new source of failure on top of a validated base, so a break localizes to the thing just added. The alternative — build the whole thing and debug it as a whole — burns a 4GB budget on uninterpretable failures.
> **Hardware rule (Mistake #36):** never run full training loops on Claude's machine. Implement → commit → push → user tests on GPU. Always activate the repo venv first.

**Status legend:** `established` · `directional` · `open` · `theoretical` · `plan`.

---

## 0 · Build Philosophy

The architecture is one operator with four clamp modes plus a learned value. We do **not** build all modes at once. We build the **settling operator first as a plain model**, prove it settles and computes, then light up one mode at a time. Every mode is gated on a test; if a mode fails, we fall back to the last working configuration (which always exists — the floor is Ouro).

The decisive ordering question is which to validate first: *settling* (does the loop converge and compute?) or *the modes* (does clamping do useful work?). Settling first — a mode built on a non-converging loop is undebuggable.

---

## 1 · Stage Map

| Stage | Adds | New failure source | Gate to advance | Tests Risk |
|---|---|---|---|---|
| **0** | Settling core as a plain DEQ LM/task model | does it settle & compute at all | beats matched fixed-depth transformer on a reasoning task at equal data | 1 |
| **1** | *Represent* mode (latent prediction + SIGReg) | latent prediction without collapse | object-coherence probes pass; no collapse; ≥ Stage-0 sample efficiency | 1, 4 |
| **2** | *Reason* mode via diffusion-vocabulary refinement | goal-clamped iterative refinement | beats depth-only on multi-step tasks; passes consistency-vs-correctness probe | 2 |
| **3** | Learned value (long-horizon) | temporal credit assignment | chain length scales beyond Stage-2 ceiling | 3 |
| **4** | *Learn* mode (prospective config + item 52) | local learning rule | matches backprop accuracy; resists forgetting | 1, 4 |
| **5** | Energy-vocabulary migration; control/halting policy; (gated) hierarchy | — | each kept only if it beats the prior baseline | — |

Stages 0–1 are text/static-data testable. Stage 2 onward is gated on the environment generator (`training_environment.md`), because goal-clamped refinement of a world model needs a world to refine through.

---

## 2 · Stage 0 — The Settling Core

`plan` · **the linchpin stage**

### 2.1 What to build
- One weight-shared block `f_θ`: relational attention (polar magnitude/phase split, SSA for content routing) + SwiGLU FFN + RMSNorm + QK-Norm. Start **without** the polar split if it complicates convergence debugging — add it once the loop is stable.
- **Block depth `k` is a knob, not a constraint** (see architecture.md §2.2). The minimal `k = 1` map is the pure DEQ form and is unlike Ouro (which loops a multi-layer stack) — expect strange behavior and least per-step expressivity. If Risk 1 convergence or the matched-baseline gate misbehaves, sweep `k` upward (`f` becomes a small weight-tied stack, still one fixed-point map). `core/block.py` ships `k = 1`; add a `n_layers` field to extend.
- DEQ wrapper: iterate `f_θ` on a fixed input to equilibrium; implicit-function-theorem gradients for O(1) memory. Fallback: fixed iteration count + truncated backprop if the implicit solve is unstable.
- Thin token adapter in; autoregressive token head out (text only at this stage).
- Convergence-based halting: stop at `‖Δh‖ < ε`.
- Deep-supervision outer loop (the HRM data-efficiency device that survives the hierarchy demotion).

### 2.2 What to measure (Risk 1 — convergence)
Instrument the settling itself before caring about task accuracy:
- **Convergence rate:** fraction of inputs that reach `‖Δh‖ < ε` within a max-iteration budget. Track the distribution of iterations-to-converge.
- **Spurious attractors:** do different random inits on the same input settle to the same attractor? Measure basin consistency.
- **Oscillation:** detect limit cycles (‖Δh‖ plateaus above ε). Log per-input.
- **Adaptive compute:** does iterations-to-converge correlate with problem difficulty (a held-out difficulty label)?
- **Contraction lever:** if convergence is unreliable, impose a spectral-norm / ~1-Lipschitz constraint and re-measure (the contraction constraint from Starting `architecture` §3b/A8).

### 2.3 Gate
The decisive test (Starting `architecture` §6d Stage 0): **does it beat a fixed-depth transformer of equal active parameters at an equal data budget on a reasoning task?** If a matched fixed-depth model ties it, recurrent depth buys nothing and the premise is wrong — stop and reconsider before building modes on top.

### 2.4 Deliverables
- `core/block.py` — the relational settling block.
- `core/deq.py` — DEQ wrapper + implicit gradient + convergence instrumentation.
- `core/halting.py` — convergence-based halting.
- `train_stage0.py` — plain LM/task training with deep supervision; convergence diagnostics logged per epoch.
- A fixed-depth transformer baseline of matched active params, same data budget.

---

## 3 · Stage 1 — The Represent Mode (JEPA quality)

`plan`

### 3.1 What to build
- Add a masking/occlusion scheme over the unit pool: clamp visible units, free occluded/future units, settle, read prediction off freed units.
- Target = stop-gradient pass of the same block over the full (unmasked) input. **Fallback:** explicit EMA twin if the shared stop-grad target collapses.
- SIGReg loss (random-projection + Epps-Pulley normality, per LeWorldModel) as the anti-collapse term. One effective hyperparameter.
- Loss = latent-prediction MSE (over freed units) + SIGReg.

### 3.2 What to measure (Risks 1, 4)
- **Collapse:** representation variance / rank over training; SIGReg statistic. Collapse = constant or low-rank latents.
- **Object-coherence probes** (Risk 4 interference test): linear-probe the latents for object identity vs. background/nuisance attributes. JEPA quality = identity is linearly decodable, nuisance is not. The Socratic probes from Starting `representation_learning` §2b (cat on black background, cat-shaped non-cat) operationalized.
- **Interference:** does adding SIGReg-regularized Represent training degrade Stage-0 convergence (Risk 1) or task accuracy? Track both modes' metrics simultaneously.

### 3.3 Gate
Object-coherence probes pass, no collapse, and sample efficiency ≥ Stage-0 LM objective. If SIGReg and the settling dynamics fight (Risk 4), try: separate magnitude (content) vs. phase (structure) regularization so SIGReg acts only on the content channel, leaving phase free to form reasoning structure.

---

## 4 · Stage 2 — The Reason Mode (diffusion vocabulary)

`plan` · **the core reasoning bet** · gated on the environment generator

### 4.1 What to build
- Goal/preference units; a clamp protocol that writes the target there.
- Reasoning as iterative latent denoising: initialize free reasoning units from noise, condition on the clamped question + goal, run the weight-shared block as the denoiser for a variable number of steps toward the answer. Intermediate states are unsupervised thoughts; only the final answer is supervised (plus the world-model rollout signal from the environment).
- Working-memory re-clamp protocol: each completed refinement writes its conclusion to working memory, available as a clamp for the next.
- Adaptive step count (more denoising steps for harder problems) — reuse the halting machinery.

### 4.2 What to measure (Risk 2 — consistency ≠ correctness)
This is the make-or-break probe. Distinguish **genuine multi-step inference** from **nearest-attractor completion**:
- **Compositional generalization:** train on problems requiring composition of relations A and B separately; test on problems requiring A∘B never seen together. A Hopfield-like completer fails this; a reasoner generalizes.
- **Held-out composite relations** (composition-fidelity, Q1/A4): does applying thought-A-then-thought-B reach the same state as a learned composite A∘B?
- **Step-count scaling:** does accuracy on k-step problems hold as k grows, or collapse past the "associative recall" horizon (≈1–2 hops)?
- **Distractor robustness:** inject a plausible-but-wrong attractor near the goal; a completer is captured by it, a reasoner is not.

### 4.3 Gate
Beats depth-only (Stage 0/1) on multi-step tasks **and** passes the compositional-generalization probe. If it only does nearest-attractor completion (Risk 2 realized), the unification reduces to Ouro — we learn this early and either (a) add explicit relational-composition pressure to the training signal, or (b) accept the Ouro floor for v1 and document the negative result.

---

## 5 · Stage 3 — The Learned Value (long horizon)

`plan` · gated on the environment generator

### 5.1 What to build
- A value head reading the settled state: estimate proximity-to-goal (MuZero-style bootstrap).
- Use the value to densify the sparse goal signal during chain training; optionally to guide step allocation (think more where value is uncertain).
- Optional: lightweight latent rollout/lookahead (predict value of candidate next-clamps before committing) — the MuZero planning half, kept minimal for 4GB.

### 5.2 What to measure (Risk 3)
- **Chain-length ceiling:** maximum reasoning depth at which accuracy stays above threshold, with vs. without the value. The value should raise the ceiling.
- **Credit-assignment health:** gradient magnitude / signal across chain positions; does signal reach early steps?
- **Rollout drift** (Starting `architecture` §1g / A3): how many world-model rollout steps before prediction diverges? Characterize *where* drift lives (phase / magnitude / semantic) before considering the targeted-TBAF placement.

### 5.3 Gate
Reasoning-depth ceiling extends measurably beyond Stage 2. If credit assignment fails despite the value, the fallback is shorter chains + deeper per-step settling (shift budget from thought-axis to depth-axis).

---

## 6 · Stage 4 — The Learn Mode (prospective configuration)

`plan` · research upgrade, rides on a validated core

### 6.1 What to build
- Replace DEQ implicit-gradient backprop with prospective configuration: clamp target, settle to target-consistent equilibrium (reusing the *same* settling engine — item 91), consolidate weights locally.
- Layer item 52's signal/noise decomposition as a per-update generalization filter — **first as a monitor** (does it track held-out generalization?), only then as a preconditioner (Starting `learning_algorithm` §3d). The central falsifier (L1): does the NTK signal/noise partition have an analogue for energy-based local learning?
- Continual-learning evaluation: sequential tasks, measure catastrophic forgetting vs. backprop baseline.

### 6.2 What to measure (Risks 1, 4)
- **Settling cost:** is per-update settling fast enough to be affordable (L2)? Reuses Stage-0 convergence instrumentation.
- **Accuracy parity:** does local learning match backprop accuracy on the validated tasks?
- **Anti-forgetting:** sequential-task forgetting curve vs. backprop.
- **Interference (Risk 4):** does the item-52 filter suppress updates the reasoning mode needs (antagonism), or strengthen anti-forgetting (synergy)?

### 6.3 Gate
Matches backprop accuracy and beats it on forgetting. If prospective config is impractical (settling too slow, L1 fails), keep DEQ backprop and treat continual learning via a hippocampus/replay route (Starting item 79) instead.

---

## 7 · Stage 5 — Migration & Optional Modules

`plan`

- **Energy-vocabulary migration:** once diffusion-mode reasoning and prospective-config learning both work, unify them under a single explicit energy (the falsifiable active-inference commitment: settling, planning, learning all descend one energy — Starting `learning_algorithm` §5, L5). Keep only if it does not regress the diffusion baseline.
- **Control / halting policy (AutoTTS, metacognition):** learn depth-per-step, chain length, branch/prune. Seat of the reflexive self-model (Starting `metacognition`). Deferred until the core is stable.
- **Hierarchy (A7):** the H/L slow/fast split, built last, kept only if it beats single-timescale at matched params/compute.
- **Time-series fallback (§8 of architecture):** trained SSM world-time substrate, only if the rollout bet fails. The repo's existing `experiments/Mamba3/` could seed it (mind Mistake #47, MIMO bugs).

---

## 8 · Memory Levers (apply as the budget demands, not preemptively)

`established` / `directional`

The nested refinement naively blows 4GB. Levers, in order of preference:
1. **DEQ implicit gradients** — O(1) memory in the settling loop (built in from Stage 0).
2. **DiffusionBlocks** (item 85) — decouple memory from depth; natural fit since the loop is already near-identical block applications. Open question A5: does block-wise training preserve recurrence semantics?
3. **4-bit weight quantization** (items 60/67), **MoE+MLA** (item 11), **gradient checkpointing** — the standard cushion.

Do not optimize systems (Rust/C, items 83/84) during research — premature; the constraint is memory (algorithmic), not language overhead.

---

## 9 · Repository Layout (proposed)

```
experiments/RecurrentWorldModel/
├── Docs/
│   ├── architecture.md          # the idea
│   ├── implementation_plan.md   # this file
│   └── training_environment.md  # the test apparatus
├── core/
│   ├── block.py                 # relational settling block
│   ├── deq.py                   # DEQ wrapper + implicit grad + convergence metrics
│   ├── halting.py               # convergence-based halting
│   ├── clamp.py                 # unit pool, clamp/free protocol (the 4 modes)
│   ├── represent.py             # JEPA-as-clamp mode + SIGReg
│   ├── reason.py                # diffusion-vocabulary refinement + working memory
│   ├── value.py                 # learned value (Stage 3)
│   └── learn_pc.py              # prospective config + item-52 filter (Stage 4)
├── env/                         # see training_environment.md
├── diagnostics/                 # convergence, collapse, consistency, credit, interference probes
├── train_stage0.py … train_stage4.py
└── baselines/                   # matched fixed-depth transformer, etc.
```

One file = one concept (a standing repo rule). Tests via `./venv/Scripts/python.exe -m pytest`.

---

## 10 · The Through-Line

Every stage produces a publishable result on its own (matching the Sapient HRM-Text bar — small, validated, open). Even the **negative** results are contributions: "goal-clamped settling reduces to nearest-attractor completion" (Risk 2 realized) is a real finding. The build is structured so that we always have a working artifact at the last passed gate, and the worst case is a clean, honest Ouro variant with documented reasons the richer modes did not light up.
