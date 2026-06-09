# Training & Testing Environment — A Glass-Box Apparatus for the Four Risks

> **Status:** working design. Companion to `architecture.md` and `implementation_plan.md`.
> **Lineage:** revises Starting `training_environment` (ARC-AGI-3-style generator + LeWorldModel diagnostic layer). The revision: this version is **organized around the four risks** of the clamped settling core, so every environment capability exists to falsify a specific architectural claim, not just to score.
> **Hardware rule:** environments must be cheap (grid-world, small action set) to fit 4GB and keep episodes short enough to roll the world model forward within the settling budget.

**Status legend:** `established` · `directional` · `open` · `plan`.

---

## 0 · Why an Environment, and Why a Glass Box

`directional`

The core's central claims are only testable interactively: a JEPA world model predicts *dynamics* (needs a world), goal-clamped reasoning acts *toward goals* (needs goals), active inference minimizes surprise *through action* (needs actions that change observations), and the latent rollout needs a world to roll through. Static next-token data cannot exercise the half of the architecture that matters here.

We build a **glass box**: a procedural generator of ARC-AGI-3-style environments whose ground truth we hold, so we can instrument exactly the things the architecture makes claims about — and crucially, **probe the four risks directly**. The real ARC-AGI-3 public set is held out as the uncontaminated honesty check (train the *process*, evaluate on the *benchmark*). All glass-box instrumentation is disabled at benchmark evaluation.

**Design inversion from the Starting Doc:** the Starting `training_environment` was organized around capabilities (world-model building, goal acquisition, fluid intelligence). This one keeps those but **subordinates them to the four risks** — because for v1 the question is not "is it generally intelligent" but "does the settling core converge, reason rather than complete, assign credit over horizon, and avoid mode-interference."

---

## 1 · The Two Layers

`plan`

### 1.1 Controlled diagnostic layer (the scalpel) — built FIRST
A small set of **fully controlled** environments with known closed-form dynamics. Their job is not novelty but **clean attribution**: when the world model is wrong, you know exactly where, because you know the true dynamics. This is the LeWorldModel role, and it is where the **four-risk probes** live. Built first, in parallel with Stage 0–1, because you cannot debug the core against a noisy generator.

### 1.2 Generator layer (the diversity engine) — built at the Stage 1→2 boundary
A procedural generator of novel ARC-AGI-3-style environments: abstract colored grids, small discrete turn-based action set, a sampled rule-set from a library of composable mechanics (gravity, keys/doors, pushable objects, transformations, conservation laws), and a **hidden goal**. Difficulty scales by composing more mechanics. The generator records which mechanics are shared across environments (for the transfer probe). Comes online when Stage 2 (Reason mode) needs a world to roll through.

Why both: the generator answers "does it learn to learn?"; the controlled layer answers "and when it fails, *why*?" Both feed the same shared core.

---

## 2 · Risk-Targeted Probes (the heart of this document)

Each probe is a controlled experiment that can **falsify** an architectural claim. Disabled at benchmark eval.

### 2.1 Risk 1 — Convergence probes
*Claim: the block settles reliably and adaptively.*

- **Convergence-rate environment:** trivially simple dynamics; measure fraction of states that reach `‖Δh‖ < ε` within budget, and the iterations-to-converge distribution. A clean baseline — if it cannot settle here, nothing downstream matters.
- **Difficulty-graded settling:** environments with a known difficulty parameter (number of composed mechanics); test that iterations-to-converge rises monotonically with difficulty (adaptive compute working).
- **Spurious-attractor detector:** same observation, multiple random inits of free units; measure whether they settle to the same attractor. Divergence = spurious attractors / unreliable basins.
- **Oscillation log:** flag inputs where `‖Δh‖` plateaus above ε (limit cycles).

### 2.2 Risk 2 — Consistency-vs-correctness probes
*Claim: goal-clamped settling does multi-step reasoning, not nearest-attractor completion.* **The make-or-break probe.**

- **Compositional-generalization environment:** mechanics A and B appear separately in training; the test environment requires **A∘B composed**, never seen together. A Hopfield-like completer fails (no stored A∘B attractor); a reasoner composes. This is the single most decisive test in the whole program.
- **Held-out composite relations:** a relation-composition task where applying thought-A then thought-B should equal a learned composite A∘B (composition-fidelity, Q1/A4). Measure state agreement.
- **Step-count scaling:** k-step puzzles for increasing k; plot accuracy vs. k. A completer collapses past ≈1–2 hops; a reasoner degrades gracefully.
- **Distractor-attractor environment:** place a plausible-but-wrong terminal state near the true goal in latent space. A completer is captured by the nearest attractor; a reasoner reaches the correct (farther) goal.

### 2.3 Risk 3 — Long-horizon credit-assignment probes
*Claim: the learned value makes sparse-goal credit assignment tractable over many steps.*

- **Horizon-sweep environments:** identical mechanics, increasing solution length (more steps to the hidden goal). Plot accuracy vs. required horizon, **with and without** the learned value. The value should raise the horizon ceiling.
- **Sparse-vs-dense ablation:** same environment with (a) sparse terminal reward only, (b) value-densified signal. Measures the value's contribution directly.
- **Rollout-drift environment:** roll the world model forward in latent space inside a real episode; measure steps-before-divergence and characterize *where* drift lives (phase wrap / magnitude growth / semantic) — the precondition for the targeted-TBAF question (Starting `architecture` A3), tested before any TBAF is added.

### 2.4 Risk 4 — Representation/reasoning interference probes
*Claim: Represent (SIGReg) and Reason modes coexist on shared weights.*

- **Mode-interleaving curriculum:** alternate Represent-mode and Reason-mode training; track *both* modes' metrics continuously. Interference = improving one degrades the other.
- **Object-coherence under reasoning load:** the Socratic probes (cat on black background; cat-shaped non-cat; same object, different color statistics) run **after** reasoning training — does object-coherence survive, or does structured-attractor formation re-contaminate representations?
- **Channel-split test:** if interference appears, restrict SIGReg to the magnitude (content) channel and leave phase (structure) free; re-measure. Tests whether the polar split resolves the conflict by construction.

---

## 3 · Glass-Box Instrumentation (training only)

`plan` · all disabled at benchmark eval

Because we hold ground truth, we can measure what the black-box benchmark cannot:
1. **Ground-truth dynamics access** — generate latent-prediction targets directly; measure world-model accuracy; confirm the rollout tracks reality.
2. **Violation-of-expectation probes** — mid-episode, inject an impossible transition; measure the prediction-error / surprise spike. The direct test of the JEPA world model and the active-inference surprise signal. **This is also the headline validation result** (Starting `project_objectives` §1): a measurable surprise spike on a physically impossible event = the model learned structure, not surface statistics.
3. **Goal-acquisition instrumentation** — the goal is hidden; probe *when* a correct internal goal representation forms (goal-discovery time), not just final success.
4. **Continual-learning stream** — environments arrive as a sequence; measure catastrophic forgetting across it (the cross-environment setting prospective configuration must handle, Stage 4).
5. **Mechanic-transfer probes** — using the generator's shared-mechanic record, measure transfer of a learned mechanic to a surface-different environment vs. relearning from scratch (fluid intelligence).

---

## 4 · Observation, Action, Reward

`plan`

- **Observations:** abstract colored grids — perceptually simple (cheap on 4GB), modality-clean (no pretrained vision encoder), and enter the shared core as **clamped sensory units** (the Represent/Perceive modes). The same model reasons over text and grid with only a thin adapter.
- **Actions:** small discrete turn-based set (move / select / act-on-cell). Short episodes keep the rollout within the settling budget.
- **Reward:** sparse and discoverable (signal on reaching the hidden goal). Scoring follows ARC-AGI-3's **super-linear efficiency penalty** (inefficiency ratio squared) at *evaluation* — operationalizing skill-acquisition efficiency = data efficiency. **Caveat (Starting Q2):** as a *training* signal this may be too sparse early; likely needs a curriculum (easy/dense-ish first, annealed toward the true sparse-efficiency metric).

---

## 5 · Honesty Checks

`open`

- **Generator-overfitting detector (Starting Q1):** hold the ~25 real ARC-AGI-3 public environments as validation. If the model aces our generator but fails the real set, it learned our quirks, not skill-acquisition. The main environment risk.
- **Sim-to-real (Starting Q3):** whether skill learned on our process transfers to the human-designed benchmark — the bottom-line question, measured directly by the held-out set.
- **Grid-world sufficiency (Starting Q4):** grids are cheap and modality-clean but may be too narrow for a world model that later generalizes. Stay grid-only through v1 validation; broaden later (the staged-multimodality plan).

---

## 6 · Build Order (locked to the implementation stages)

| Impl. stage | Environment capability online | Primary probes |
|---|---|---|
| **0** (settling core, text) | none yet — build the **controlled diagnostic layer** + one hand-built environment in parallel | Risk 1 convergence probes (§2.1) |
| **1** (Represent mode) | controlled layer exercised for object-coherence | Risk 4 interference (§2.4), collapse, VoE (§3.2) |
| **2** (Reason mode) | **generator comes online** — goal-clamped refinement needs a world to roll through | Risk 2 consistency-vs-correctness (§2.2) — the decisive milestone |
| **3** (learned value) | horizon-sweep + rollout-drift environments | Risk 3 credit assignment (§2.3) |
| **4** (Learn mode) | continual-learning stream | forgetting curve (§3.4), Risk 4 synergy/antagonism |

Stage 2 is the hinge: the Reason mode and the generator are the **same milestone**, because a chain that rolls a world model forward has nothing to roll without one. Build the generator on the Stage 1→2 boundary.

---

## 7 · The Single Most Important Experiment

If only one experiment could be run, it is the **Risk 2 compositional-generalization probe (§2.2)**: train on mechanics A and B separately, test on A∘B never seen together. It is the dividing line between "we built a unified reasoning core" and "we built a fancy Hopfield net that reduces to Ouro." Run it as early as Stage 2 allows. A clean pass is the project's headline architectural result; a clean fail is an honest, publishable negative result that tells us to bank the Ouro floor and document why the richer modes did not light up.
