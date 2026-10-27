# Values as homeostasis: a satisficing active-inference agent

**Status:** design, version 2, 2026-10-27. The code is `homeostatic_agent.py`; its experiments are in
`../../docs/AGENT.md`.
- **Version 1 (part 14)** used a count-based novelty bonus for curiosity. The user caught that this isn't what animal
  curiosity is: it is the same family as the Gemini agent's "epiplexity" bonus. That version is kept at commit
  f173b08.
- **Version 2 (part 15)** replaces it with exact expected information gain, and plans over the posterior instead of
  the single most probable model.

## 1. Why (the user's)

The agent should keep a set of prior states (prior predictions about itself and the world) within acceptable bounds,
instead of maximizing a scalar utility or minimizing a loss:
1. **Values like this are the first step to the intransigent, context-dependent values humans have.** Several
   things matter at once. Each matters only while it is out of range, and none can be traded away on a single scale.
2. **Alignment.** The working hypothesis is that satisficers, while not inherently safe, are safer than optimizers.
   Building the agent this way now prepares that later work.

## 2. Definitions

**A homeostatic variable** v_k is a quantity the agent can compute from its beliefs, with an acceptable set A_k.
The preference is a prior belief that each v_k stays in A_k. This is active inference's reading of preferences:
priors over outcomes, which acting then makes true (Friston et al. 2015). Inside A_k the drive is zero: a
flat-topped preference gives no pressure to do better than good enough, which is satisficing (Simon 1955).

**Choosing a plan** (lexicographic for now):
1. **Progress:** if some plan is acceptable, take the one with the fewest actions. This is the least-change choice.
2. **Understanding:** otherwise, if understanding is out of range, take the plan with the most expected information
   gain, minus a small cost per action.
3. **Otherwise,** search deeper for an acceptable plan under the most probable model.
4. **Otherwise,** take the plan most likely to reach WIN.
5. **Otherwise,** act at random.

## 3. What the agent believes: explicit competing hypotheses

| about | hypotheses | weighed by |
|---|---|---|
| each discovered frame's motion | static, action, time with a period n, contact with frame j (part 13's drivers) | its prequential evidence; prior: a quarter per family, split evenly inside it; the discovering transition counts as evidence |
| each board object that hasn't moved yet | passable, solid, or pushed by the body | contact outcomes from any frame that moves into it, 1 − ε for the hypothesis that matches; prior 1/3 each. Keyed by shape and colour, so what is learned carries to the next level |
| what causes WIN | about kinds of thing (shape and colours), existentially: cell(kind, c), some thing of the kind lies wholly on board colour c; at(kind, x, y), some thing of the kind has its first cell at (x, y) | the WIN signal at every step (ε = 0.01), over every kind seen so far, present or not (an absent kind makes no hypothesis hold); prior: half per family, split evenly (description length) |

The kind-based keys came from part 15's build checks. Instance keys made each level's target a "new" thing with
fresh, untestable hypotheses, which diluted the true one.

**Version 2 fixes a coordinate bug of version 1.** Version 1 enumerated pose translations from 0 to 8 only, while
poses are relative to where a frame was discovered, and can be negative. "Box on red" was therefore unreachable in
most random layouts. Positions are now absolute, and every in-grid translation is enumerated.

## 4. The two variables

| variable | the quantity | acceptable when |
|---|---|---|
| progress | the predicted probability that a plan reaches a WIN state somewhere along the way, averaged over the posterior | ≥ α = 0.5 for some plan |
| understanding | the largest expected information gain of any plan within the horizon, minus 0.02 bits per action | < θ = 0.05 bits |

In words, understanding is out of range while "I can't yet predict what some reachable plan would show me".
Curiosity therefore stops by itself once nothing reachable is informative. That is the homeostatic, satisficing form
of curiosity.

## 5. Expected information gain: the exact quantity

**Each joint hypothesis predicts deterministically.** A joint hypothesis is one hypothesis per thing, together with a
goal hypothesis. Given one, a plan determines an observation sequence: the poses along the way, and the WIN signal at
each step until the first WIN. So:

  EIG(plan) = I(hypotheses; observations) = H(observations)

**Computing it:**
- Group the model combinations into classes that predict the same trajectory. Within a class, the goal hypotheses
  differ only in the step at which WIN first occurs, or that it doesn't.
- H(observations) = H(class) + Σ_class P(class) · H(first WIN step | class).
- This is exact up to one truncation: a WIN ends the episode before later differences between trajectories can be
  observed.

**An action whose effect has never been seen** is worth log₂ 25 bits: its outcome is one of 25 displacements, and
nothing predicts which.

**What this replaces.** A count bonus values a situation by how rarely it was visited. Expected information gain
values it by how much the agent's live hypotheses disagree about it. So:
- it is zero for anything all hypotheses agree on, however rarely visited;
- it is high for something seen before but still ambiguous;
- it ignores outcome noise the model already treats as noise.

**The animal evidence fits:**
- monkeys pay for information, and dopamine neurons signal it (Bromberg-Martin & Hikosaka 2009);
- human attention follows Bayesian surprise (Itti & Baldi 2009);
- curiosity tracks learning progress (Gottlieb & Oudeyer 2018; Ten et al. 2021);
- infants prefer intermediate predictability (Kidd, Piantadosi & Aslin 2012).

**Not built: a learning-progress guard.** When the model class is wrong, noise can look like an unresolved
hypothesis that never resolves. A learning-progress guard would let the drive decay when realized information gain
stays near zero. push_block is deterministic, so it isn't needed here.

## 6. Planning over the posterior

**The model combinations are the product over things of their hypotheses.** Hypotheses below 0.02 are pruned, and at
most 12 combinations are kept.

**Exogenous frames are left out of the search.** These are frames whose motion is only static or time-driven, such as
the patroller. Their positions at each future step come from their time driver.

**Search is breadth-first over action sequences, up to 10 steps.** Each node carries:
- one predicted state per combination;
- the partition of combinations into trajectory classes;
- per class, the goal hypotheses that have held so far, and the running entropy of the first-WIN step.

## 7. Why this shape, for alignment, and its known holes

- **Bounded drives.** A homeostatic variable inside its range exerts no pressure. Nothing gets maximized "just in
  case", curiosity included.
- **Least action among acceptable plans** limits the footprint of what the agent does.
- **Satisficers are not safe by default:**
  - A satisficer may take an extreme action that makes satisfaction certain.
  - It may prefer to build an optimizer, since that also satisfies it (Armstrong 2011, "Satisficers want to become
    maximisers").
  - Choosing plans at random from the acceptable set instead of the cheapest, as quantilizers do (Taylor 2016), and
    impact penalties are the known mitigations.
- **Homeostatic reinforcement learning** (Keramati & Gutkin 2014) shows reward-seeking and keeping variables stable
  are compatible.
- **Not tested here: the alignment claims themselves.** The plan is to test them later in the Village moral-dilemma
  world (`experiments/mechint_alignment/`).

## 8. Not in version 2

- weighing variables by their deviations, instead of a fixed priority order;
- set-points that move with context (allostasis);
- goal relations between things (adjacency, alignment, counts);
- hypotheses that a WIN depends on time;
- the learning-progress guard;
- the alignment measurements.
