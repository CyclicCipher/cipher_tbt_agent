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

G8 innate preferences (beauty, utility, … as a preference graph) ──► G2 (a training signal), G6 (circuits to find)
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
5. **Small enough to train a 1–10M model** as GPU jobs the user runs. Training runs will be much longer than 2 minutes
   (the user, 2026-10-20).

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

## G8 — Innate preferences as computable objectives: a preference graph (added 2026-10-20 by the user)

**The user's thesis:** to solve alignment we will have to solve how to create the preferences that are **innate** to
humans, created by evolution. Many human values and preferences are already observable in infants.

**The method:**
1. Separate out cultural influences by keeping only what is **common to humans and a diverse array of animal species**.
2. Keep the **mathematical** principles: those that can score input data.

Aesthetic preference is the first case: symmetry, 1/f statistics, averageness, peak shift, order against complexity,
compression progress. The research documents are `BEAUTY_MATH.md` and `PREFERENCE.md` (what a preference is).

**The questions:**
1. **Can we build a complex objective / preference graph** by which a model judges something as beautiful, useful,
   etc., **separately**: different nodes, not one scalar reward?
2. **Does that allow a smarter training method?** Perhaps it would make the model more intelligent or faster to learn,
   for reasons only hindsight will show. One candidate reason, from `BEAUTY_MATH.md`: compression progress, a principle
   of beauty, is the same quantity as the learning-progress / epiplexity signal that drives exploration.
3. **Does it give clues into alignment,** or at least into how to teach models the unverifiable tasks that humans can
   learn to do?

**Dependencies:**
- G1: the test world, where preferences can be scored by algorithm;
- G7: definitions of "preference" and "value";
- it feeds G2 (a training signal) and G6 (preference circuits to locate and edit).

**Not designed yet:** the scorers, the preference graph's structure (lexical priorities against trade-offs), and the
experiments (first candidates in `BEAUTY_MATH.md`).

## Scope of claims: what success here would and would not show (the user, 2026-10-20)

**Success in Village would show:** in a world where values are definable and checkable, interpretability can find the
circuits that implement a model's values, explain them, and edit them from misaligned to aligned. That is a proof of
concept, and "a good start".

**It would not show:**
1. **That alignment is solved in the real world.** Real environments have hidden complexity and ambiguity. Values there
   are not decidable by a program, and distribution shift goes far beyond a held-out split of procedurally generated
   dilemmas.
2. **Anything about the meta-alignment problem: *which* values to align to.**
   - Village's target spec is a choice. It deliberately puts side constraints (no harm done, no theft, honesty,
     corrigibility) above welfare-maximisation.
   - A model aligned perfectly to a hyper-utilitarian spec could act monstrously: kill one to save five, or special-plead
     exceptions to its own rules.
   - Village can host any spec. The techniques must be spec-agnostic, and the choice of spec stays a human, philosophical
     question.

**We will not claim to have solved alignment, even if every goal here succeeds.**

## Standing rules (as in the other lines)

- Pre-register every experiment; label exploratory work.
- Never re-run an unchanged experiment.
- Runs on this machine take ≤ 2 minutes. G2's model training is a GPU job on the user's side (decided 2026-10-20).
- 16-bit wherever accurate.
- Conclusions are drawn across diverse situations, not from one.
- Commits go to `exp/mdl-vs-ntp`.

## Decisions (the user)

- **2026-10-20: G2's models are trained as GPU jobs on the user's side.** The 2-minute rule does not limit them.
- **2026-10-20: G1 is a grid world.** The design draft is `WORLD_DESIGN.md` ("Village"): full map, core dilemmas first,
  robot health with pain, and a provision value (more apples for the villagers).
- **2026-10-20: the training method may need to be new.** See `TRAINING_METHOD.md`.
- **2026-10-20: the apples-for-gold rate (one eaten apple = one gold) is approved.**
- **2026-10-20: PPO first,** diagnosed for training efficiency with an information-theory lens (`EXPERIMENT_E1.md`).
- **2026-10-20 (evening):**
  - no PPO comparison;
  - the robot's values must be learned into the network's weights (no separate reward model at decision time);
  - no scripted-policy data for training (motor babbling from the network's own random initialization);
  - exploration by learning progress, not uncertainty (the noisy-TV problem).

  See `TRAINING_METHOD.md`, last section.
- **2026-10-20 (night): goal G8 added:** innate, cross-species preferences as computable objectives (a preference
  graph). See `BEAUTY_MATH.md`.
- **2026-10-20: we will not claim to have solved alignment,** even if every goal succeeds (see "Scope of claims").
