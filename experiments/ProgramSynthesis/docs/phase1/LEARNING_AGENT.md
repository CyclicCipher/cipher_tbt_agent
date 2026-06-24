# Learning Agent — Design

> **Purpose.** The research vehicle for the ProgramSynthesis side-project: an agent for the
> ARC-AGI-3 replica whose **state encoder is the experimental knob**, so we can run the binding
> predictions of [EXPERIMENT_GOALS.md](EXPERIMENT_GOALS.md) (P1–P3) the same way RWM's
> temporal-PoPE 3-way ran data point #1 — hold everything fixed, vary one channel, watch
> learnability move. Status: design, pre-implementation. No training on this machine (Mistake
> #36): implement → push → user runs on GPU.
>
> **Status legend:** `established` · `directional` · `theoretical` · `open` · `commitment`.

---

## 0 · Aim

Determine whether **how observation is bound to coordinates** (token vs 2D vs 2D+1 vs object vs
field) decides an interactive agent's learnability and out-of-distribution generalization — and
whether the predicted winner (2D+1 PoPE) is the binding to carry back into RWM's transformer experiments.

---

## 1 · The core move — supervised learnability microscope, not RL

`commitment`

Pure RL on a 64×64 sparse-reward world is expensive, unstable on 4 GB, **and confounds the
experiment**: a channel that fails could be failing at binding *or* at exploration, and we
can't tell which. We already have a perfect teacher — the **BFS oracle** (`tools_make_trace.py`).
So we train by **behavior cloning from the oracle**: predict the oracle's action from the
observation. This converts the interactive task into the exact regime that gave clean answers in
RWM (supervised, vary one knob, measure time-to-threshold + OOD gap).

**Bitter-lesson line is clean.** The oracle is a *teacher* (a supervision target), never a
component of the agent. The agent sees only frames — colors on a grid — and the oracle's chosen
action as the label. No game internals, no hand-coded rules, no domain vocabulary enter the
agent. (See [[feedback_bitter_lesson]], [[feedback_no_seeded_primitives]].)

**Drift is acknowledged, not ignored.** BC has the exposure-bias problem catalogued in
`RWM/Docs/latent_reasoning_training_pipeline.md` (naive one-step → O(ε·T²)). That is precisely
what Phase 2 (DAgger) addresses, and it makes the replica another arena for the drift work
rather than a place that pretends drift away.

---

## 2 · Architecture — one variable, everything else frozen

`commitment`

```
observation window            ┌──────────────┐
(W recent 64×64 frames    ──▶ │ STATE ENCODER│ ──▶  shared trunk  ──▶  policy head
 + the actions between them)  │  (THE KNOB)  │      (FIXED)            (logits, masked to
                              └──────────────┘                          available_actions)
                                                                     └▶ aux value head (Phase 3)
```

**The discipline that makes it an experiment:** the trunk (depth, width, norm), the heads, the
optimizer, the oracle data, and the seed are **byte-identical across all arms**. Only the
encoder — the binding channel — changes. Without this, a result is uninterpretable (the
data-point-#1 / data-point-#3 methodology: equalize everything but the one factor).

- **Trunk:** a small feed-forward transformer (`d≈128`, 2–4 layers, RMSNorm, QK-norm). Recurrent
  depth is a *separate, later* variable — keeping it out of v1 keeps the
  encoder the sole knob.
- **Policy head:** pooled trunk state → logits over actions, **masked to `frame.available_actions`**
  (provided by the env, so using it is legitimate, not a discovery function). LockPath uses the
  4 directions; a coordinate/pointer sub-head for ACTION6 is deferred.
- **Value head:** predicts levels-to-go / win-probability. Unused by BC; included for Phase 3.

---

## 3 · The encoder arms (the knob)

Each arm maps a window of `W` recent frames (patchified — see §8) plus the actions taken between
them into a sequence of token embeddings with a binding. Arms hold **information constant** and
vary only the **channel**, so P1 ("same info, different channel") is a clean comparison.

| Arm | binding channel | hypothesis |
|---|---|---|
| **0 · none** | a bag of color tokens, **position absent** | the floor — the spatial analog of RWM's `integer` arm; can it even tell two layouts apart? |
| **A · content tokens** | color = content; patch position `(px,py)` and frame index `t` appended as learned/absolute features | value-centric baseline; the `time_input` analog — grok-and-cap, memorizes positions |
| **B · 2D PoPE** | `(px,py)` as PoPE **phase**, color as **magnitude**; single frame | spatial coordinate binding; translation-invariant by construction |
| **C · 2D+1 PoPE** | `(px,py,t)` as PoPE phase over the `W`-frame window | **predicted winner** — see §4 |
| **D · learned slots** *(optional)* | slot-attention objectness (**learned**, never hard-coded connected components) | object-binding, isolated |
| **E · continuous field** *(the "if not tokenizing" arm)* | a coordinate-conditioned evaluator over the `(H,W,T)` cube (SIREN/U-net-style); query any `(x,y,t)` → content | non-tokenized analog; aligns with RWM data point #3's functional readout and seeds the Phase-3 world model |

### 2D+1 PoPE, concretely

`directional` Split the head dimension into three frequency groups — an `x`-band, a `y`-band, and
a `t`-band — plus content in the magnitude. For a patch token of content `c` at `(px,py)` in
frame `t`:

- **magnitude** `μ = softplus(embed(c))` — content only, position/time-independent;
- **phase** = `px·θ_x` on the x-band, `py·θ_y` on the y-band, `t·θ_t` on the t-band.

The attention score then factorizes **exactly** into
`Σ_c μ_i μ_j · cos(Δpx·θ_x) · cos(Δpy·θ_y) · cos(Δt·θ_t)` — a content-match times an
`x`-relative, a `y`-relative, and a `t`-relative term, each depending only on its own offset.
This is RWM's PoPE (the proven 1-D what/where split) generalized to two spatial axes plus a time
axis; the decoupling proof and the length-extrapolation property carry over per-axis.

---

## 4 · Why 2D+1 PoPE is the one to beat

`theoretical`

The `t`-band makes change a coordinate primitive: **"same object, next frame" is a pure `Δt`
phase at `Δpx=Δpy=0`; motion is nonzero `(Δpx,Δpy)` at `Δt=1`.** So "what changed and how it
moved between frames" — the entire content of an interactive physics world — is *read* as a
relative-phase operation, not *inferred* as a correlation. The three non-trivial LockPath events
are all temporal-relational:

- **block push** — an object that moved *with* the agent (`Δt=1`, matched `(Δpx,Δpy)`);
- **key pickup** — an object that vanished after the agent overlapped it;
- **door opening** — a *distant* object that changed when the agent touched the key (causality at
  a distance).

A single-frame encoder (B) must reconstruct these from static snapshots; a token encoder (A)
must learn them as correlations. 2D+1 PoPE binds them as geometry — it is the **Δ-dynamics
(physics) and action→change (agentness)** binding EXPERIMENT_GOALS §2 called for, delivered for
free. Predicted ordering: **C ≫ B > A** on the change/causality mechanics (L1–L3), with a
none-arm at the floor, and a possible **tie on pure L0 navigation** (information present in every channel → channel
irrelevant, the `what_in` tie of RWM data point #3).

---

## 5 · Training stages — each a GPU job, each isolates a prediction

`directional`

1. **Phase 1 — BC microscope (build first).** Oracle trajectories → predict the action. Sweep
   arms none/A/B/C (then D/E). Tests **P1** (channel dominance) and **P2** (shift invisibility: with
   held-out layouts of the *same* mechanic, does the train-vs-OOD gap stay ≈0 for B/C and open
   for A?).
2. **Phase 2 — DAgger (close the loop).** Roll the trained agent out, have the oracle relabel the
   states it actually visits, retrain. Cheap (the oracle solves any state); attacks exposure-bias
   drift; tests whether a binding that wins supervised survives closed-loop.
3. **Phase 3 — world model + planning (the real composition test).** Learn a latent forward model
   + value; train on L1 and L2, test **zero-shot composition on L3** — **P3 / the A∘B test for a
   *learned* agent**, and where arm E becomes the seed for a latent world-model rollout (the
   latent-reasoning training pipeline).

---

## 6 · Metrics — reuse RWM's learnability signatures

`commitment` From data point #1/#2:

- **Time-to-threshold** — steps to X% action-match accuracy, per arm.
- **Curve shape** — monotone-convex (easy) vs plateau-then-grok (long solution) vs flat
  (unlearnable). A plateau is the learnability red flag.
- **Final ceiling** — does it reach perfect or cap?
- **Generalization-gap trajectory** — `|acc_train_layout − acc_held_out_layout|` over training.
  **Gap ≈ 0 throughout ⇒ the channel dissolved the position nuisance** (B/C predicted);
  **gap opens and persists ⇒ position-specific memorization** (A predicted). Diagnosable from the
  first evals — the data-point-#2 signature in the spatial domain.
- **Skill-acquisition efficiency** (closed-loop, Phase 2+) — actions-to-competence per novel
  level, Chollet's actual metric.

---

## 7 · Data — procedural layouts (a prerequisite)

`open` The four hand-authored LockPath levels are *demo content*; the experiment needs
**distributions**. Phase 1 requires a **procedural LockPath generator** that samples layouts per
mechanic (randomize room size and the positions of agent/key/door/goal/block/pad/hazard, subject
to solvability checked by the existing BFS), then **splits position-configs into train vs
held-out**. The held-out split is what makes P2 measurable. The generator must stay
distributional and principled — no per-instance hand-tuning ([[feedback_consolidation_overhaul]]).

---

## 8 · Scope / VRAM (v1)

`commitment` Patchify the full **64×64** into a patch grid (e.g. patch=4 → 16×16 = 256 patch
tokens; a small conv or color-embedding per patch). Tokenize the whole field — *not* a
board-dims crop — to stay faithful to the real benchmark (LockPath sits in the top-left; most
patches are background and attention learns to ignore them). Window `W=3–4` frames → ≤ ~1k
tokens; `d≈128`, 2–4 layers; fp16 AMP. BC is light and fits 4 GB comfortably. Include an
**action token per transition** in the window so the model can bind action→change (agentness).

---

## 9 · Proposed repo layout (`experiments/ProgramSynthesis/agent/`)

One file = one concept:

```
agent/
├── encoders.py     arm A–E encoders, identical signature frames→(tokens, binding)
├── trunk.py        the FIXED transformer trunk + policy/value heads
├── policy.py       Agent subclass wrapping encoder+trunk; choose_action() with action masking
│                   (drops straight into Environment / run_episode / the widget trace generator)
├── layouts.py      procedural LockPath generator + train/held-out split (BFS-solvable)
├── dataset.py      oracle-trajectory + DAgger-relabel dataset builders
├── metrics.py      time-to-threshold, curve-shape, gap-trajectory
├── train_bc.py     Phase 1 (GPU job)
└── train_dagger.py Phase 2 (GPU job)
```

`policy.py` reuses the existing `Agent` interface, so a trained agent's run is captured by the
same trace format the visualization widget already plays — every experiment is watchable.

---

## 10 · Open questions / risks

- **`open` Patching vs objectness.** Fixed 4×4 patches may straddle object boundaries and blunt
  arms B–D. Mitigation: small patches, or per-cell tokens on the active region if the token
  budget allows; keep patch size a swept hyperparameter, not a silent choice.
- **`open` BC ceiling ≠ understanding.** High action-match can come from memorizing the oracle's
  path shape. The held-out-layout gap (§6), not raw accuracy, is the real signal — and Phase 3's
  zero-shot L3 is the un-foolable test.
- **`open` Trunk neutrality.** A feed-forward trunk might itself favor one channel. Guard: report
  the matched-baseline ordering, and (later) re-run the sweep with a different trunk (deeper, or
  recurrent-depth) to confirm the channel ordering is trunk-invariant.
- **`theoretical` If A ties C, the binding thesis is weaker than claimed** (EXPERIMENT_GOALS P3's
  falsifier). We report that honestly; it would be a real update.

---

## Connection to RWM

This is RWM's temporal-PoPE question asked about **objects, space, physics, and agency** instead
of synthetic time, on a benchmark built to require them. RWM's current focus is improving
transformers through controlled experiments; this is one such experiment in a richer arena. If
arm C (or E) wins, it is the concrete binding to carry into RWM's transformer work, and Phase 3's
latent world model connects to the latent-reasoning training pipeline — closing the loop
EXPERIMENT_GOALS promised: one thesis, two arenas.
