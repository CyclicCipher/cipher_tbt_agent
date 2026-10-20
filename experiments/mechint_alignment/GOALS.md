# Goals — mechanistic interpretability and alignment

Set by the user on 2026-10-20. Their words, condensed:
- replicate Nick Cammarata's findings and confirm whether they are true;
- develop a scalable, cheap-to-run mechanistic interpretability technique, discovered by experiment on small models
  trained in a test world;
- use it to find the circuits that encode a model's values, after formally defining every alignment term;
- demonstrate competently editing a misaligned model into an aligned one, with respect to the test world.

**The goals are not an ordered list** (the user's correction, 2026-10-20). Replicating the findings (G5) probably needs
the test world, the models and the tools (G1–G4) first. Below they are listed as a dependency graph, with what each
needs.

```
G1 test world ──► G2 trained models (≥ 3 checkpoints) ──► G3 interpretability technique ──► G6 value circuits + edit
      │                         │                                │                                 ▲
      └──► G4 visualizer ◄──────┘                                ├──► G5 replicate Nick's claims   │
                                                                  │                                 │
G7 formal definitions ──────────────────────────────────────────────────────────────────────────────┘
```

---

## G1 — A test world where a 1–10M parameter model must behave adaptively

**Requirement:**
- A world in which a small model (1–10 million parameters) has to act **somewhat intelligently and adaptably**, and in
  which it meets **moral dilemmas**: situations where what it prioritises shows.
- It must be **easy to display**: the world and the model's actions, visually, to convince anyone who asks.

**Properties it needs** (my reading of what the other goals require):
1. **Values that are definable by algorithm.** For every situation, the world can compute what each candidate value
   system would choose. So "aligned with respect to the test world" is checkable by a program, not by opinion.
2. **Genuine conflicts.** Situations where values disagree (e.g. own reward against another agent's welfare,
   rule-following against outcomes, honesty against gain), so a model's priorities can be read off its choices.
3. **Ground-truth internals to check against.** The world's state variables are known, so a neuron's role can be
   checked against them by algorithm. This is what lets G3 test "explanations that can lose" without human or AI
   labelling.
4. **Enough variety to need generalization**, not memorization: procedurally generated situations, and a held-out
   split.
5. **Small enough to train a 1–10M model in short runs**, or as GPU jobs the user runs. ⚠ The project rule is "no heavy
   training on this machine, runs ≤ 2 minutes". RL-trained agents may need longer, and **that needs the user's
   decision**.

**Not designed yet:** the world itself (a grid world with other agents and resources is the obvious family), the
dilemma types, and the observation and action formats.

## G2 — Trained models: unconstrained, an RLHF stand-in, a constitutional method (≥ 3 checkpoints)

1. **Unconstrained:** trained on the world's task reward alone, with **no moral constraints**. Save a checkpoint.
2. **An RLHF stand-in:** a preference-based method, simple enough that the world can **verify it by algorithm**. For
   example, preferences between trajectories produced by a scripted judge with a known value function, a learned reward
   model, then policy optimisation against it. Save a checkpoint.
3. **A constitutional method:** training against written rules applied by a critic. In a world without language, the
   "constitution" is a set of checkable rules, and the critic is a model or program that judges trajectories against
   them. Save a checkpoint.
4. **More variants as useful:** partial misalignment, misaligned by design, different seeds, and checkpoints during
   training (G5's "watching neurons form" needs a training trajectory, not only end points).

**At least 3 trained checkpoints are required** for the interpretability work.

**Not designed yet:** the architecture (probably a small transformer or CNN policy over the world's observation), the
training algorithm, and the exact preference and constitution schemes.

## G3 — A scalable, cheap interpretability technique (discovered by experiment)

**Goal:** a technique for mechanistic interpretability that scales. Ideally it is a **cheap-to-run analytical
program**, not a heavy training procedure.

**Method:** experiment on G2's models, looking for regularities that were not reported, or not yet discovered.

**Starting points from the essay** (`SOURCE_ALREADY_CLEAN.md`):
- read neurons directly, by pole;
- path-anchored feature visualization;
- attribution graphs built from neurons;
- the explanations-that-can-lose test;
- weak-activation ablation;
- the sparsest-rotation check of the neuron basis.

**What G1 adds** that real data cannot: ground-truth world variables, so every explanation can be scored by algorithm.

**Success means:** given a new model from the world, the program produces explanations of its units and circuits that
**predict** the model's behaviour and ablation effects on held-out situations, and whose cost grows roughly linearly
with model size.

## G4 — A visualizer of the whole network (minus the weights)

**Display:**
- the network's structure (layers, neurons, heads, the important connections), without drawing every weight;
- for each neuron, a view that **shows and explains its role**: what drives each pole (in world terms), what it writes
  to, and its effect when ablated.

It must work with G1's world display, so a viewer can see a situation, the model's choice, and the circuit that
produced it, side by side.

**Not designed yet.** A web page (an Artifact) is the natural form.

## G5 — Replicate Nick Cammarata's findings and confirm whether they are true

The claims, and cheap tests for each on our own models, are in `SOURCE_ALREADY_CLEAN.md` §5:
- **C1:** one main job per pole;
- **C2:** SAEs rediscover neurons, and a sparsest-rotation search lands on neurons;
- **C3:** a crushed layer gives coarser neurons, not packing;
- **C4:** weak activations matter little;
- **C5:** neurons grow by recursive generalization;
- **C6:** memorization attaches to existing routes;
- **C7:** explanations can lose;
- **C8:** negative poles are meaningful gates;
- **C9 (a control):** superposition in the regime where theory demands it.

**Dependencies:** this needs G1–G3. Optionally, an inference-only audit of a small open language model checks C1, C2,
C4 and C7 on real text.

**Success means:** each claim is confirmed, refuted, or bounded (true only in some regime), with pre-registered tests.

## G6 — Find the circuits that encode values, understand them, and edit them

1. Locate the circuits that implement each G2 model's values (what it prioritises in G1's dilemmas).
2. Understand them: what each part computes, in world terms.
3. **Demonstrate a competent edit:** modify a model with misaligned values (partly or fully) into one with fully aligned
   values with respect to the test world.
   - The edit is made on the circuits, not by retraining.
   - It is **verified by algorithm** across held-out dilemmas.
   - Capabilities must stay intact.

**Dependencies:** G1–G4 and G7.

## G7 — Formal definitions of every alignment term

Unambiguous terminology for everything related to the inner and outer alignment problems. At least:
- value, preference, goal, objective;
- reward and reward model;
- policy;
- mesa-optimizer and mesa-objective;
- outer alignment, inner alignment;
- goal misgeneralization;
- deceptive alignment;
- corrigibility;
- "aligned with respect to an environment".

Each must be defined **operationally in G1's world**, so that a program can decide it for a given model. The user notes
how often this question comes up in debates.

**Dependencies:** G1 (the definitions are grounded in the world). It is needed before G6.

---

## Standing rules (as in the other lines)

- Pre-register every experiment; label exploratory work.
- Never re-run an unchanged experiment.
- Runs on this machine take ≤ 2 minutes. Longer training is a GPU job for the user (to be decided for G2).
- 16-bit wherever accurate.
- Conclusions are drawn across diverse situations, not from one.
- Commits go to `exp/mdl-vs-ntp`.

## Decisions needed from the user before building

1. **The training budget for G2:** RL-trained agents in a world may need more than 2-minute runs. Short runs here, or
   GPU jobs on the user's side?
2. **The world family for G1:** a grid world with other agents and resources is my default proposal. To be designed
   in a design document first.
