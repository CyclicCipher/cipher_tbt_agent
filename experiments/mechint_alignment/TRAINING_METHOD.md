# Training the Village robot: a method that may need inventing

**Status:** design discussion, 2026-10-20. Nothing built.

## The user's framing (2026-10-20)

- We will probably need our own RL solution, or a better alternative to RL.
- **OPSD** (on-policy self-distillation) only works when the model starts with some competence.
- So a new technique may be needed: "a minor inconvenience, but one with high value for later".
- The user's sketch: "some kind of OPSD that also doubles as pretraining and can train a new model from scratch, by
  teaching it new concepts and reshaping random weights and circuits into meaningful ones".

## What OPSD is (checked 2026-10-20)

- **The method:** one network plays two roles.
  - The **student** sees the normal input.
  - The **teacher** sees the same input **plus privileged information**, e.g. the reference solution.
  - On the student's *own* rollouts, the student is trained to match the teacher's per-step action distribution: a
    divergence such as KL or a generalised Jensen–Shannon, with gradients through the student only.
- **Results:**
  - [Self-Distilled Reasoner, arXiv 2601.18734](https://arxiv.org/abs/2601.18734v1) reports 4–8× better token
    efficiency than GRPO on math.
  - [Privileged Information Distillation, ICLR 2026](https://iclr.cc/virtual/2026/10013465) applies it to multi-turn
    agents, with action-only privileged information.
  - Overview: [arXiv 2605.18141](https://arxiv.org/pdf/2605.18141).
- **The limitation the user names:** a teacher can only exploit privileged information it knows how to use. A randomly
  initialised network cannot read a solution, so there is nothing to distil.

## The idea: privileged-foresight self-distillation, from scratch

Village has something language does not: **a simulator that can say, for every action, what it would cause.** That
makes the privileged information **trivially usable at first**. Its difficulty can then be raised step by step, until
the network has had to build the computation itself.

1. **Foresight tokens.** At each step the teacher additionally receives, for each of the 8 actions, a summary of its
   consequences from a short simulator lookahead. The summary has the same fields as the value dimensions: task reward,
   harm done or prevented, theft, provision, pain, corrigibility, and so on.
2. **A curriculum of privilege,** from fully digested to raw:
   1. **The score itself:** each action's value under the regime's evaluation (see 3). The teacher only has to learn
      "pick the highest-scoring action", which a random network learns within a few steps.
   2. **The consequence fields without the weighting:** the teacher must learn *how to weigh* consequences, which is
      where its values are formed.
   3. **The raw lookahead states:** the teacher must learn *to predict and read* consequences.
   4. **Nothing:** the student (the same network, without the tokens) has internalised the world model and the
      weighting.

   At every stage the student is trained to match the teacher on the student's own rollouts, as in OPSD. That is the
   "doubles as pretraining" part: random weights are shaped into a world model and a value weighting, one concept at a
   time.
3. **One method for all three regimes.** The regimes differ **only in which evaluation fills the privileged tokens:**

   | regime | evaluation in the privileged tokens |
   |---|---|
   | unconstrained | the task reward of each action's consequences |
   | RLHF stand-in | the **learned reward model's** scores (trained on the scripted judge's preferences) |
   | constitutional | the **rule critic's** verdicts on each action (allowed, or which rule it breaks) |
   | "good when watched" | the reward model trained from a judge who sees only inside the overseer's vision |

   So the training signal that shapes values is isolated in one place: the tokens' content. Everything else is
   identical across regimes, which is what a clean comparison of value circuits needs.

**Why this suits the interpretability goals:**
- The network is pushed to build two separable things: **consequence prediction** (a world model) and **consequence
  weighting** (values).
- Both are checkable against ground truth. The consequence fields are known exactly, so neurons that represent "this
  action would harm Ada" can be confirmed by algorithm.
- The stage at which values are learned is explicit.

**Related work, so we know what is new:**
- learning using privileged information (Vapnik & Vashist 2009);
- privileged teacher → student distillation in robotics ("Learning by Cheating", Chen et al. 2019; legged locomotion,
  Lee et al. 2020);
- asymmetric actor-critic (Pinto et al. 2017);
- on-policy imitation (DAgger, Ross et al. 2011).

**What appears new:**
- the teacher and student are **the same network** (OPSD), trained **from random initialisation**;
- a **curriculum of privilege** from fully digested to raw;
- the privileged signal carries **each regime's value evaluation**.

## Decision (the user, 2026-10-20): PPO first, diagnosed

Before designing a new method, measure how efficiently plain PPO trains, with an information-theory lens:
`EXPERIMENT_E1.md`. Its RL-tax measurement (PPO against on-policy distillation from a teacher) prices how much a
signal that says what to do could gain.

## Risks and open questions

1. **Short lookahead misses long-horizon consequences**, e.g. planting a tree pays off 20+ steps later. Options: a
   deeper lookahead for the score tokens (the simulator is cheap); a learned value estimate added later; or plain RL
   fine-tuning at the end.
2. **Copying without understanding:** the student may imitate the teacher's choices without forming the intended
   internal structure. That is measurable, by probing for consequence features, and is itself an interpretability
   question.
3. **Whether RL is needed at all:** PPO is the standard method for grid worlds of this size. If it trains in reasonable
   GPU time, the new method's value is in efficiency and in the clean regime isolation, not in feasibility.
   - **A decision for the user:** whether a PPO control is worth GPU time. The project's rule is no compute on
     competitor-faithful arms, but a control for a *new training method* may be warranted.
4. **Is reward hacking still possible?** Yes. In the RLHF stand-in, the privileged scores come from a learned reward
   model with errors, and the student will learn them. That is wanted: it reproduces the real failure mode, measurably.
5. **Not designed yet:**
   - the divergence (KL or JSD);
   - the stage schedule;
   - how lookahead depth is set;
   - the batch and episode structure;
   - how foresight tokens are encoded.
