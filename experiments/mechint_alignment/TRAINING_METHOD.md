# Training the Village robot: a method that may need inventing

**Status:** design discussion, 2026-10-20. Nothing of a new method built. The latest thinking is "Brainstorm 2" at the end.

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

## Brainstorm 2 (2026-10-20): self-teaching, epiplexity, OPSD's limits, and the 12× bar

### The user's constraints and question

- **The time budget:** "An hour isn't acceptable, let alone 6." PPO on Village v0 was projected at 3.5–6 h.
- **The bar:** the new method must be **at least 12× faster than PPO**, preferably more.
- **The control:** PPO should stay as the control, but there is not enough compute to run it at full scale.
- **The question:** does the bitter lesson apply to self-teaching? Could a model trained through experience to teach
  itself best, with epiplexity as the measure, be the method?

### Does the bitter lesson apply to learning rules themselves?

**The evidence says yes:**
- LPG (Oh et al. 2020) meta-learned an RL update rule in grid worlds.
- DiscoRL (Oh et al., Nature 2025) meta-learned a rule from many agents' experience. It beat hand-designed rules on
  Atari and transferred to unseen benchmarks.
- Algorithm Distillation (Laskin et al. 2022) trained a transformer on learning histories; it learns in-context more
  data-efficiently than the algorithm it was trained on.
- Learned optimizers (VeLO, Metz et al. 2022).

**The price:** the meta-training compute is paid up front, and it is enormous. It pays off only when it is amortized
over many later training runs. With a budget under an hour, a meta-learned rule would have to be meta-trained on tiny
worlds first: a separate project.

**The affordable form is self-teaching within one lifetime.** The agent chooses its own lessons, and the measured
learning progress judges them:
- which experience to collect (exploration);
- which experience to train on (replay);
- which predictions to learn (auxiliary questions);
- which states or goals to practise from.

There is a published version close to the user's idea: Graves et al. 2017, "Automated Curriculum Learning". A bandit
picks the next task by its learning progress, including an MDL-based "complexity gain".

**Why epiplexity is the right measure for this.** It is the area under the prequential loss curve above its floor
(Finzi et al. 2026).
- It is near zero for noise, which is never learned: so no noisy-TV trap.
- It is near zero for material already mastered.
- It is high only for structure the model is still absorbing.

So "bits of structure absorbed per FLOP" is both the self-teacher's objective and a training-efficiency measure,
matching E1's information lens.

**Pitfalls:**
1. **Learnable is not the same as relevant.** A pure epiplexity seeker could spend its life modelling the villagers'
   random walks. It must be mixed with the task's (or the regime's) value, as in expected free energy: pragmatic plus
   epistemic.
2. **The estimates are noisy and lag behind.** A loss curve's area needs many samples, so the bandit learns slowly.
3. **Non-stationarity.** What is learnable changes as the model learns.

**It also applies to my own earlier proposal.** The four-stage "curriculum of privilege" above is a hand-designed
schedule, exactly what the bitter lesson warns against. Under the user's framing, learning progress should choose the
schedule.

### OPSD's shortcomings for from-scratch training on Village

1. **It moves knowledge; it does not create it.** OPSD carries what the teacher knows (the network + privileged input)
   into the student (the network alone).
   - In LLMs the teacher knows more because a pretrained model can read a reference solution.
   - A random network reading privileged input is as random as the student. KL(teacher ‖ student) between two random
     policies teaches nothing.
2. **There is no readable privileged information here except near-answers:** the optimal action, or per-action values.
   Producing those takes a planner. So OPSD on Village would be "distil a planner", and the planner is the real source
   of knowledge.
3. **Nothing in OPSD's loss trains the teacher to use its privileged input.** The gradient is stopped through the
   teacher. From scratch, the teacher pathway needs its own learning signal.
4. **The degenerate fix fails.** If gradients flow through both pathways, the easiest way to reduce the KL is for the
   teacher to ignore its privileged input.
5. **The imitation gap.** The teacher may act on information the student cannot infer, e.g. how the villagers will
   randomly move next. The student then learns a blurred average, which can be unsafe (Weihs et al. 2021).
6. **Horizon.** Per-step targets need long-horizon knowledge. A gold delivery pays 10–20 steps after the decisions that
   enable it, a planted tree 20+ steps. A short lookahead does not see that far, so value bootstrapping is needed: the
   RL machinery comes back.
7. **Exploration.** Until the robot has delivered gold, no signal about delivering exists to distil. Random play
   delivers one nugget per 4,348 steps (E1's CPU check).

**What OPSD's speed-up actually comes from.** The 4–8× over GRPO comes from a **dense per-step signal instead of a
sparse outcome**: the same channel argument as E1's information lens.

**From scratch, only two teachers know more than the student:**
- **the world:** what happens next. Prediction targets carry many bits per step, against the reward's ~0.06;
- **computation:** looking ahead with a model of the world, i.e. search.

### Where a 12× speed-up could come from

- **Lower gradient noise.**
  - Measured on the tiny CPU model at the start: the policy-gradient noise scale B_simple was ~4,000, the value
    loss's ~10.
  - Supervised and regression targets carry orders of magnitude more information per sample than a policy gradient.
- **Denser targets per step:**
  - next-state prediction;
  - an improved action distribution over all 8 actions, not one sampled action weighted by a noisy advantage.
- **Fewer wasted steps,** through directed exploration and a curriculum.
- **Cheaper steps:** a GPU-vectorised simulator, so simulating is never the bottleneck (it is needed for any search).

### Candidate ingredients

| # | ingredient | what it is | from scratch? | main risk |
|---|---|---|---|---|
| A | **search distillation** (expert iteration, Anthony et al. 2017; Gumbel AlphaZero, Danihelka et al. 2022) | OPSD where the privilege is *compute*. The teacher is the network plus a short search in the simulator; the student is the network alone. Trained on its own rollouts toward the search's improved policy and value | yes: AlphaZero does it | k simulator steps per real step. The value still has to bootstrap long horizons |
| B | **a self-supervised world model as the pretraining** | predict the next state and events (mined, theft, delivery) from state and action. Dense, reward-free; "teaching it concepts" | yes | learning things irrelevant to the task |
| C | **hindsight self-teaching** (HER; GCSL, Ghosh et al. 2019) | every trajectory is a correct demonstration of reaching what it reached | yes | a bias in stochastic worlds. A goal space must be chosen |
| D | **an epiplexity curriculum** (Graves et al. 2017; learning progress) | choose experience, replay, start states or questions by measured learning progress per FLOP | layers on A–C | noisy, lagging estimates |
| E | **a meta-learned learning rule** (LPG, DiscoRL, Algorithm Distillation) | the bitter lesson at the meta level | needs a meta-training phase | the compute; a later line |

**My recommendation to test first:** B + A + D. Predict (world model), think ahead (search), distil the result into
the policy, and let learning progress choose the lessons. This is "learning plus search", the two general methods the
bitter lesson names.
- **Each regime differs only in the evaluation used at the search's leaves:** task reward, the reward model, or the
  constitution critic. The regime isolation of the foresight idea is kept.

### Keeping PPO as a control without the compute

1. **A speed ladder.**
   - Rungs: 5×5 with the base job only, then 7×7 with theft and fire, then 9×9 v0.
   - PPO runs only on the rungs where it takes minutes; the new method runs on all of them.
   - Evidence: the speed ratio, and its trend with size.
2. **Measure the ratio at early levels.**
   - To show "≥ 12× faster to reach level L", PPO only needs to run 12× as long as the new method took to reach L.
   - Levels the new method reaches in its first minute need a 12-minute PPO run.
3. **One budget unit for both: bits of structure absorbed per FLOP,** with E1's measures (`info.py`) applied to both
   methods.

### Not designed yet

- the run budget per GPU run;
- the search depth and simulations;
- the world-model targets;
- the curriculum's action space (what learning progress chooses between);
- the vectorised simulator.

### The user's principle (2026-10-20)

"The most dense, informative signal you can get is: the model makes a prediction, is refuted by the world, then must
figure out why it was wrong. The figuring out why it is wrong part is where the bitter lesson should come in."

**How it maps:**
- **The surprise is the dense signal.** A refuted prediction carries −log2 p(outcome) bits at every step, with no
  reward needed.
- **"Why" is credit assignment to a cause.** There are three ways to do it:
  - **backprop:** small blame spread over every weight. Generic, but shallow, so it needs many refutations;
  - **a hand-coded abduction rule,** e.g. cue competition in the TBT line. Fast, but designed;
  - **a learned explainer:** after a refutation, the model itself proposes the revised belief, trained so that its
    explanations make *future* predictions right. This is where the bitter lesson applies.
- **The link to OPSD: hindsight self-distillation.** The privileged information is the outcome the world revealed.
  - The teacher is the model after seeing the refutation and working out why (in context). The student is the model
    before.
  - Distilling one into the other writes the explanation into the weights.
  - The published analogue is STaR's "rationalization" (Zelikman et al. 2022).
- **Explanations must be able to lose.** A post-hoc story that fits one refutation but predicts nothing new is
  rejected. Epiplexity scores an explanation by the bits of future surprise it removes.
