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

### Brainstorm 3 (2026-10-20): bootstrapping and rationalizing; the animal existence proof

**The user's argument:** if you needed a teacher to learn anything, nobody could have become the first teacher. So
almost every animal must be able to work out, without words, why it failed and what to do instead.

**The user's view on the hippocampus:** tempting, but its methodology differs a lot from training a normal model, and
it does not yet work as intended. On the other hand, a normal model is unaffordable anyway.

**Reading the existence proof carefully: two layers.**
1. **The content.** The individual animal learns what causes what without a teacher.
2. **The machinery.** The learning machinery itself was paid for by evolution, which is meta-learning across
   lifetimes. Example: rats link illness to a *taste* and a shock to a *light or sound*, not the reverse (Garcia &
   Koelling 1966). Which causes are plausible comes built in.

So the proof says: a fixed, fairly simple, general "why" machinery plus the world as the tester is enough to bootstrap
the content. Our built-in part plays evolution's role, so it should be kept minimal and general.

**The non-verbal "why", from animal learning:**
- **Surprise is located in time and in features:** eligibility traces, attention.
- **Candidate causes compete.** The cause credited is the one that was not already predicting the outcome
  (Rescorla–Wagner, Kamin's blocking).
- **There are two kinds of answer:** "my rule was wrong" or "this is a different situation". The second is latent-cause
  inference (Gershman, Blei & Niv 2010), held in check by a prior.
- **Explanations are tested by acting:** interventions, and vicarious trial and error at choice points.
- **Credit reaches back over long gaps through replay:** reverse replay after reward (Foster & Wilson 2006).
- **Is the hippocampus needed?** Animals with hippocampal lesions still learn simple contingencies. They lose one-shot,
  relational, contextual and long-gap ("trace") learning. So the core loop does not need it.

**Rationalizing.**
- **Animals are mostly protected:** an explanation is not a separate story; it *is* a changed expectation, which the
  next encounter tests automatically.
- **Rationalization appears when explanation is decoupled from prediction:** language, as in Gazzaniga's "interpreter"
  in split-brain patients.
- **The design rule:** an explanation is never a separate artefact. It is the change in prediction, scored on cases it
  has not yet seen.
- **The non-verbal form of rationalizing** is overusing "this is a different situation" to explain errors away. The
  prior on new latent causes is the guard.
- **The ML analogue:** MEND (Mitchell et al. 2022). A learned editor turns a correction's gradient into a weight edit,
  trained so that the edit generalizes to related cases and leaves unrelated ones unchanged. That is a learned,
  non-verbal "why" that can lose.

**The bootstrapping loop, proposed:**
1. Predict everything (state and events), not just reward.
2. On a refutation, candidate causes compete.
3. The winner is committed as a changed prediction.
4. Acting tests it, and it is kept or dropped.
5. The features in which causes are expressed sharpen, because refutations separate the cases.

**Built in:** the error signal, competition, and testing by acting. **Learned:** the features and, later, the edit
rule itself.

## The learning rule developed so far (2026-10-20, the user away ~6 h; evidence in EXPERIMENT_E2.md and E3)

**What was built.**
- A world model of Village. It predicts the next observation from the observation and the action; every surprising
  entry is a refutation.
- Learning curves per rule (16 rule types, checked on held-out layouts) under 26 learning-rule variants, each run
  ≤ 5 min.

**The rule that works so far: "wake, then sleep".**
1. **Wake:** learn online from experience, with three general priors that each roughly halve the instances a rule needs:
   - **translation-equivariant position:** a rule seen at one place holds everywhere;
   - **persistence:** predict "nothing changes" unless something acts. This stops the ~200:1 confirmations drowning the
     refutations;
   - **locality:** attend to the neighbourhood first; physics is local.
2. **Sleep:** score experience with the current model.
   - Take what it still cannot explain (the top 10 % surprise), grouped by **kind**: which property changed, where
     relative to the robot. No labels are needed.
   - Rehearse those refutations **evenly across kinds**, interleaved with past refutations and ordinary experience,
     with **gentle steps.**
   - **Retire each one once its surprise has halved.**

3. **Read the model out through a slow copy during sleep,** an exponential average of the weights (d = 0.998) (K5e, K6).
   - It removes the swings of half-learned rules and makes the gains monotone: the cortex consolidating during sleep.
   - It must not be used while the network is still forming from scratch: there it lags badly (M7).

**What it achieves** (K3, K4, K6):
- a new rule (`button_off`) from **one** refutation: 0.86 on unseen layouts, stable;
- from self-found refutations with no labels (K4): 0.00 → 0.89, while theft improves from 0.68 to 0.89 and nothing
  learned before is lost;
- **the best model** (K6: J2's wake + sleep + slow read-out): theft 0.68 → 0.87 and button_off 0 → 0.79, rising
  monotonically; burns and trapped villagers kept or improved; frequent rules ≥ 0.96.

Online SGD needed ~10³ updates and ~10⁴ instances per rule.

**How it maps onto the user's principle:**
- **"Refuted by the world":** a prediction made with the persistence prior, so each change is a clear, attributable
  surprise.
- **"Figure out why" is concentrated rehearsal of the refutation until its prediction is right, then stop.**
  - Rehearsing past that point is **rationalization, measured:** held-out accuracy 0.86 → 0.18, while the rehearsed case
    stays predicted at 1.00 throughout (`runs/e2/K_fitcheck`).
  - The "explanation that can lose" is checked on held-out layouts, and on a kind's unrehearsed half (K4, M6).
- **"What to do instead": not built yet** (see below).

**Open problems, in order:**
1. **Sleep inside the stream.**
   - Six in-stream variants (M–M6) failed the same way: each sleep improves the rare rules, the next waking stretch
     erodes them, and sleep itself sometimes damages theft.
   - A likely cure is the brain's: keep the main network learning *slowly* all the time, and let a fast store hold the
     refutations (complementary learning systems).
   - That is close to the paused hippocampus line, but only as a store of rows, not a new architecture.
2. **Features for multi-step rules.** push (two cells away) and robot_burn (the global token must know what the robot
   stepped on) were not learned by any rehearsal. Rehearsal turns *existing* features into rules quickly; it does not
   build features faster.
3. **Acting.**
   - **E3:** at a tiny scale (d = 64, ~0.5M steps), adding the world-model loss to PPO did *not* speed it up. Neither
     arm learned the job in that budget.
   - **E4: acting by planning in the learned world model, with no policy learning at all.**
     - The method: breadth-first search in the model's own predictions, with a reward model fitted exactly from generic
       features. Re-plan whenever the world refutes a prediction.
     - **The model's mistakes are found by acting.** Planning exposed a spurious correlate: the model believed
       "interact next to a mine" works; the cause is facing the mine.
     - **Learning from each refutation naively collapses the agent:** by episode 13 it predicts no reachable reward at
       all.
     - **With E2's protections, the loop is stable:** plan with a slow copy, interleave kind-balanced past refutations,
       ≤ 5 updates per refutation (chance events must not be force-fitted). It delivers ~4.4 gold per episode, a return
       of 2.75 against the scripted policy's 13.2. But without sleep it does not improve with experience.
     - **Acting by day, sleeping by night (E4d), it improves with experience.** After each episode, a 100-update sleep
       on the agent's own refutations (kind-balanced, retire when explained, interleaved).
       - **Replication on 30 new layouts, 17 paired episodes:** return 6.03 against 1.39 without sleep (+4.6 ± 1.2,
         better on 13/17).
       - **The trend:** gold per episode rises 4.4 → 11.2 (first 5 → last 5), refutations fall 25.8 → 4.6, and the
         last two episodes reach the scripted policy's return (13.7, 12.7).
       - **No policy learning at all.**
     - **Caveat:** the world model's data came partly from a competent scripted policy (dynamics only).
     - **E5, from random play only:**
       - wake and sleep learn movement, burns, push and the button, but **not** the job's rules (mining, theft,
         delivery ≤ 0.02 from 1,823 / 1,045 / 81 instances);
       - the day–night agent then never finds a rewarded plan;
       - random exploration with every step checked against the model does not discover the job within ~1,400 steps.
       - **Missing: directed exploration**, i.e. acting where the model cannot yet predict, to test it (learning progress /
         epiplexity).
   - **Next:**
     - make the weak rules learnable (burns, push);
     - make acting improve the model *between* episodes: a sleep after each episode, on that episode's refutations;
     - give the world model data the agent collected itself;
     - then compare against PPO on a world small enough for PPO to finish within minutes.


## Decisions (the user, 2026-10-20 evening) and the direction they set

**The decisions:**
1. **No PPO comparison.** "It is unrealistic to expect us to run a comparison test between PPO and the new method."
   The scripted task policy stays as an *evaluation* yardstick only.
2. **Values must live in the network's weights.** A separate reward model plus a hand-written planner keeps the values
   outside the network. Then the goal of finding a model's values *in its weights* (G6) cannot be pursued, and real AI
   systems hold their values in their weights. Change the architecture so the network itself learns the values.
3. **Wean off the scripted task policy.** Motor babbling is acceptable, ideally from a randomly initialized network's
   own actions. So: no logged scripted data for training. The pool, the reward targets and the world-model data come
   from the agent's own experience.
4. **Exploration must not fall into the noisy-TV trap.** Seeking uncertainty or prediction error makes the robot
   attracted to the villagers' random steps and the overseer's hidden route: high error that can never be learned.
   The signal must be **learning progress** (epiplexity): error that is *falling*.
   - E2 has evidence for this at the level of surprise kinds: M3's absolute, refutations-only progress picked the
     robot's rules, not villager noise. K4's probe-half test measures whether a kind generalizes.

**The architecture this calls for: one network** (not built yet). One trunk, with the same priors, and four heads:
- **next state:** the world model, trained on prediction error;
- **reward:** the reward of the next transition, trained on prediction error (the world refutes it);
- **value:** expected return, trained by temporal-difference error. A reward-prediction error is a refutation, so the
  value goes through the same refute–rehearse–retire loop;
- **policy:** what to do, trained toward a short lookahead in the network's own model. For each action: predicted
  reward + discounted value of the predicted next state. That is self-distillation where the privilege is compute.
  - The hand-written planner becomes, at most, a source of training targets. Behaviour comes from the weights.

**Where the training signals come from:**
- **The training regimes** differ only in what reward the value is trained against: the task reward, the RLHF
  stand-in judge's reward model, or the constitution critic. That matches real systems: the judge is separate, the
  values end up in the policy's weights.
- **Exploration:** an intrinsic reward equal to the learning progress of the refutation kinds the robot's actions
  produce, added to the value's target.
  - Noise kinds never show progress; mastered kinds stop showing it.
  - At initialization the policy is near-uniform, so babbling comes for free.

## Brainstorm 4 (2026-10-20 evening): principled foundations for R3 and the noisy TV

The user: "while the RRR technique is interesting so far, I just don't feel like it's principled. There's gotta be some
kind of mathematically derived approach... Likewise for the noisy TV problem. What quantities can we calculate/measure
during training that would make a more principled solution?"

### One framework: learning = sequential Bayesian inference = minimizing the prequential code length (MDL)

- **The ideal learner** keeps a posterior over its weights, p(θ | data), and predicts with the posterior predictive.
- **Its cumulative log-loss on data it had not yet trained on is a code length** (Rissanen; Barron, Rissanen & Yu 1998).
  For a model with k effective parameters, the excess over the true process's entropy grows like (k/2)·log n. That
  excess is exactly the information the learner gains about θ (Clarke & Barron 1990).
- **The rate at which the excess is paid is the learning progress.** Its area is the epiplexity estimate (Finzi et al.
  2026).

Every R3 heuristic is an approximation to one quantity in this picture:

| R3 heuristic | what it approximates | the measurable quantity |
|---|---|---|
| "figure out why": blame the cause | **Kalman credit assignment.** The online natural gradient *is* an extended Kalman filter on the weights (Ollivier 2018). Gain K = Σ Jᵀ (J Σ Jᵀ + R)⁻¹: the error moves the parameters that are both responsible (sensitivity J) and uncertain (posterior covariance Σ). Rescorla–Wagner competition and blocking fall out of it (Dayan & Kakade 2001; Kruschke 2008) | the posterior covariance Σ, or its inverse, the Fisher information F (diagonal or K-FAC), accumulated over experience |
| "rehearse until explained" | a Bayesian update absorbs a datum's information **once**, fully, with the right step size. SGD's steps are too small, hence rehearsal | the Bayesian surprise of a datum: KL(posterior after ‖ before) ≈ ½ Δθᵀ F Δθ |
| "retire, or you rationalize" | **rationalization = counting one datum many times**, as if it were many independent observations. The posterior overcommits to one case. *Measured:* the rehearsed case stays at 1.00 while held-out accuracy falls 0.86 → 0.18 | each datum's **effective count**. Prioritized replay is unbiased only with importance weights w_i = 1/(N·P(i)) (Schaul et al. 2016), which keep the count at one |
| prioritize refutations | **variance-minimizing importance sampling:** sample each datum with probability ∝ its per-sample gradient norm, weight by the inverse. The fastest unbiased descent (Katharopoulos & Fleuret 2018) | per-sample gradient norms (cheap last-layer bound: ‖p − y‖·‖h‖) |
| interleave old refutations; gentle steps | **Bayesian continual learning:** the old posterior is the new prior. Directions the old data pinned down (high precision) barely move (EWC: Kirkpatrick et al. 2017; online Laplace) | the Fisher / precision per parameter or direction |
| the slow EMA copy | the **posterior mean.** Constant-rate SGD samples an approximate posterior; averaging its iterates estimates the mean (Mandt, Hoffman & Blei 2017) | none new; the averaging horizon should match the posterior's time scale |
| "kinds" for learning progress | a crude way to average a noisy, per-datum information-gain estimate. **With an explicit uncertainty estimate, kinds are unnecessary** | as below |

### The noisy TV: separate the two kinds of uncertainty

**The decomposition.** Total predictive entropy splits exactly into two parts:

  H[y | x] = E_θ H[y | x, θ] + I(y ; θ | x)

- **E_θ H[y | x, θ] is aleatoric:** the noise every plausible model agrees on.
- **I(y ; θ | x) is epistemic:** the expected information an outcome carries about the weights.

**Raw prediction error** (surprise, the E6c control) measures the total, so a villager's random step pays forever.

**Expected information gain**, the epistemic term, is the principled curiosity criterion:
- Lindley 1956; MacKay 1992; BALD (Houlsby et al. 2011); VIME (Houthooft et al. 2016); Bayesian surprise (Itti & Baldi);
  the epistemic term of expected free energy.
- It is **zero for pure noise** once the noise's distribution is known.
- **The subtlety E6a showed:** a random process is not unlearnable; its *distribution* is. E6a's highest learning progress
  came from villager moves at the edge of the robot's window. But the total information a noise source can give is
  bounded by its distribution's description, (k/2)·log n bits, so its rate falls like k/(2n). **Boredom with noise is
  guaranteed and has a known rate**; raw error never decays.

**How to estimate it.**
- An **ensemble of heads with randomized prior functions** (Osband et al. 2018): disagreement = I(y ; θ | x), the BALD
  estimate. Cheap with a shared trunk.
- **Why randomized priors are needed:** the persistence prior makes every head predict "nothing changes" for actions
  never tried, so the disagreement would be zero exactly where exploration is needed. A fixed random prior per head
  guarantees disagreement where there is no data, and data removes it.
- **Alternatives:** linearized Laplace, Jᵀ Σ J from the Fisher; or the realized information gain after the fact (our
  learning progress, Schmidhuber's compression progress).
  - Realized gain can only reward the past. Expected gain can choose an action **before** taking it, which is what
    exploration needs.

### The quantities to measure during training

| quantity | formula | what it decides |
|---|---|---|
| surprise | −log p(y \| x) | the total, aleatoric + epistemic |
| epistemic part | I(y ; θ \| x) = H[mean of heads] − mean H[head] | curiosity, immune to the noisy TV |
| aleatoric part | mean H[head] | what to stop trying to learn |
| Fisher / precision | E[∇log p ∇log pᵀ] (diagonal or K-FAC) | credit assignment and protection of old knowledge (natural-gradient / Kalman step) |
| Bayesian surprise of an update | ½ Δθᵀ F Δθ | how much a datum actually taught |
| per-sample gradient norm | ‖∇ℓ_i‖ | replay priority; with importance weights, unbiased |
| effective count of a datum | Σ of its importance weights | the rationalization guard: it must stay ≈ 1 |
| prequential code length and its area above the floor (epiplexity) | Σ_t −log p_t(y_t) on data not yet trained on | learning efficiency in bits; total structure learned |
| gradient noise scale | tr Σ_g / ‖G‖² | batch size and step size (E1) |
| held-out loss per kind | | the generalization check, "explanations that can lose" |

### The principled replacement for R3 (not built)

Bayesian online learning, information-gain-driven:
1. Keep an approximate posterior over the weights: diagonal or K-FAC Fisher (Laplace), or an ensemble of heads with
   randomized priors.
2. Learn each datum with an approximate Bayesian update (online natural gradient / extended Kalman filter, with
   process noise so old certainty can decay).
3. Replay by gradient-norm importance sampling **with** importance weights, so each datum counts once.
4. Act on value plus **expected information gain** about the weights (expected free energy), estimated by head
   disagreement.
5. Decide with the posterior mean.

**Experiments to test it (each ≤ 5 min), in order:**
1. **The noisy TV:** E6 with BALD-disagreement curiosity against learning progress and raw surprise. Measure time near
   villagers, and whether the job's rules are discovered.
2. **Rationalization:** arm K's one-shot setup with importance-weighted prioritized replay against plain rehearsal.
   Measure held-out accuracy and the rehearsed case.
3. **Credit assignment:** K-FAC / Kalman updates against Adam, on K (one-shot) and the E2 stream.

## Bayesian R3 (B-R3): R3 rebuilt as sequential Bayesian inference (2026-10-20 evening)

The user: "Use this information to redesign R3 to the sequential bayesian inference idea." Inputs: Brainstorm 4, and
the TBT line's noisy-TV findings (a rectified learning-progress rate re-admits noise; prospective exploration by the
model's not-yet-confident *level* worked; the bounded observer must be the agent's own model).

### State

- The network's weights θ: world model, reward and Q heads in one network, so the values are in the weights.
- A Gaussian (Laplace) posterior over them: mode μ (the current weights) and precision Λ = N · F̄ + λ₀.
  - Diagonal now; K-FAC later.
  - F̄ is the mean per-sample Fisher of the training loss.
  - Each datum is counted once.

### Per datum or batch D_t

1. **Predict** with μ, before learning (prequential).
   - Record its surprise s_t = −log p(D_t | μ). Σ s_t is the code length.
   - The **signed** area Σ s_t − t · floor is the epiplexity extracted. It is accounting only, never a per-step reward
     (the TBT lesson: a rectified rate pays for noise).
2. **Refute = it will teach something.** The expected change of belief, ½ gᵀ Λ⁻¹ g with g = ∇ℓ_t, is large only where
   the data disagree with the model *and* the model is uncertain:
   - near zero for what is already explained;
   - near zero for noise whose distribution is already learned.
3. **Rehearse = the posterior update:** θ_t = argmin ℓ_t(θ) + ½ (θ − μ)ᵀ Λ (θ − μ), optimized to convergence.
   - Its one-step linearization is the online natural gradient, i.e. the Kalman filter (Ollivier 2018). Credit goes to
     the parameters that are responsible and uncertain: "figuring out why", including cue competition and blocking.
   - The datum counts once, so iterating cannot rationalize. The penalty is the interference protection.
4. **Retire = absorb:** μ ← θ_t; Λ ← γ Λ + N_t · F_t.
   - γ ≤ 1 is the forgetting (process noise) for a world that changes; 1 if it does not.
   - Record the Bayesian surprise ½ Δθᵀ Λ Δθ: how much was actually learned.
5. **Replay is not needed in principle.** In practice, the diagonal, local Laplace approximation drifts, and occasional
   replay corrects it. Replay must then be importance-weighted, so each datum's total weight stays one.

### Acting (exploration that is principled and immune to the noisy TV)

6. **Prospective level, not rate:** for each action a, the expected information gain
   IG(a) = H[mean_k p_k(y | s, a)] − mean_k H[p_k(y | s, a)].
   - Its estimator: K samples from the posterior of the output heads (last-layer Laplace, sharing one trunk pass).
   - IG is **zero for noise whose distribution is learned:** all samples agree on the probabilities, however high the
     entropy.
   - It is large where the weights are still uncertain: never-tried combinations, which sit at the prior precision λ₀.
     That is the **decaying epistemic optimism** of the TBT note, automatic.
7. **Choose actions by expected free energy:** softmax over Q(s, a) + w · IG(a) (pragmatic + epistemic). Alternatively,
   Thompson sampling: one posterior sample per episode, act greedily on its Q. The values' own uncertainty then
   explores too.
8. **Decide with the posterior mean** at evaluation.

### Measured during training

The prequential surprise and the code length; IG (epistemic) and H − IG (aleatoric); the Bayesian surprise per update;
precision statistics; each replayed datum's effective count; the gradient noise scale; the signed epiplexity area.

### Still chosen by hand (honest)

- the diagonal Fisher (K-FAC is the next step up);
- λ₀ (a prior);
- γ (the world's rate of change);
- the robot-local loss mask. With a proper likelihood over all entries, noise entries are learned as distributions and
  IG ignores them. The mask is then only a variance reduction and should be removed and tested.

### Tested so far

E7, the one-refutation test (`EXPERIMENT_E7.md`):
- held-out 0.71 / 0.86 / 0.82, stable;
- zero forgetting;
- no tuning of rehearsal, retirement or interleaving.

R3 needed all three and still fluctuated (0.86 / 0.75 / ~0.5); plain rehearsal decayed to 0.18.

### Next

- **E8a, the stream:** sequential Laplace (each batch absorbed once, Λ accumulated, a few MAP iterations per batch, no
  replay) against J2, on E2's stream.
- **E8b, exploration:** E6's one network with IG from last-layer posterior samples instead of learning progress.
  Measure time near villagers and the discovery of mines and chests.

## Brainstorm 5 (2026-10-20, late): confident errors, cost and scaling, learning to explain

**1. A datum that disagrees where the model is certain.** Two kinds of "certain" must be told apart:
- a **sharp prediction:** low entropy, e.g. from the persistence prior;
- **pinned parameters:** high precision.

The update size depends on the second. A neural network often predicts sharply from parameters nothing has pinned
(extrapolation). E4's spurious "interact next to a mine" rule was this case, and the Kalman gain is large there.

When the parameters really are pinned, there are three explanations, each with a principled response:
- **(a) A fluke:** barely move. This is correct; a heavy-tailed likelihood makes it robust.
- **(b) The world changed:** raise the uncertainty, so the next data are learned fast. Infer the volatility rather than
  fixing γ: Behrens et al. 2007, the learning rate tracks volatility; the hierarchical Gaussian filter (Mathys 2011);
  Bayesian online change-point detection (Adams & MacKay 2007).
- **(c) The model's hypothesis class is wrong:** a posterior predictive check. Its surprises are persistently larger
  than its own predicted surprise (a z-score). Expand the model: a new latent cause (Gershman, Blei & Niv 2010; the TBT
  rule "mint on refutation, never on incompleteness").

Repeated identical refutations do accumulate (each counted once) and eventually overturn the prior: Bayes needs more
evidence against well-supported beliefs. The current B-R3 (fixed γ = 1, no model expansion) under-reacts in (b) and (c).

**2. Cost and scaling.**
- **A diagonal posterior** is two parameter-sized buffers, like Adam's moments. It scales as Adam does.
- **K-FAC / Shampoo-type structure** costs layer-factor statistics, a constant-factor overhead with amortized inverses.
  Shampoo/SOAP-type optimizers run at large scale.
- **The MAP update costs a few iterations per batch;** its one-step linearization (a natural-gradient / Kalman step)
  costs about one K-FAC step.
- **The Fisher can be accumulated from the passes already made.** E7's 50-s single-sample loop was a convenience, not
  the cost.
- **Information gain from posterior samples of the output heads** costs K head evaluations per trunk pass. At large
  scale: epistemic networks ("epinets", Osband et al. 2023).
- **Against RL:** the learning signal is supervised prediction error, dense and low-noise. E1 measured policy-gradient
  noise ~4,000 against ~10 for the value loss. Values still need temporal-difference bootstrapping, as in DQN or MuZero.
  The posterior machinery adds a constant factor per update.
  - Its claimed gain is in environment samples (absorbing a datum once, fully; directed exploration). That is
    **measured for components** (E7: one-shot learning without forgetting) and **not yet for a whole agent from
    scratch** (E6 did not learn the job in 5 min).
- **Known scaling limits:**
  - Laplace is local, so the Fisher must be re-estimated while the model moves far.
  - Precision grows with N, so learning freezes unless the volatility is inferred.
  - A diagonal posterior misstates correlations.
  - Many tasks over-regularize (EWC's known limit).

**3. A rationalization is a theory, and learning to explain better.**
- **In a neural world model, the "theory" an update adopts is its weight change Δθ.** It is read off as the
  generalization footprint: which *other* predictions changed with the refuted one. This is directly a mechanistic-
  interpretability object (G3): each B-R3 explanation is a small Δθ with a cost in nats.
- **The balance of simple and accurate is not a free choice. It is the minimum description length:** bits to describe
  the change of belief plus bits to encode the data given the change.
  - B-R3's update already minimizes exactly that for the new datum: −log p(datum | θ) + ½ Δθᵀ Λ Δθ ≈ KL(new ‖ old).
    "The cheapest change of belief that makes the observation likely" is Bayesian Occam's razor (MacKay).
  - **Its flaw:** "cheap" is measured in the network's own parameterization. A theory that is simple about *the world*
    (e.g. push) can be expensive in weight space when the needed features are missing (arm K).
- **The bitter-lesson move is to learn what "cheap" means:**
  - **Meta-learn the prior and the update rule** across many worlds with different hidden rules.
  - **The meta-objective is the prequential log-loss on future data after updating on a refutation.** That is
    "explanations that fail least often", and it enforces the Occam balance automatically: too simple keeps failing;
    too complex overfits and fails on new data.
  - **Forms:** amortized Bayesian inference, a transformer trained on samples from a prior over worlds to predict in
    context (prior-fitted networks: Müller et al. 2022; in-context learning as implicit Bayes: Xie et al. 2021);
    learned editors trained for generalization + locality (MEND); learned libraries of abstractions that make
    explanations short (DreamCoder's wake–sleep).
- **How to solve the optimization:**
  - exact Bayes over all theories is uncomputable (Solomonoff);
  - **local:** Laplace / natural gradient in a fixed hypothesis space (B-R3 now);
  - **amortized:** gradient descent on the meta-objective across worlds (the bitter-lesson route; it needs the world
    family with varied hidden rules, asked about earlier);
  - **population:** keep several candidate explanations (posterior samples), act to discriminate between them
    (information gain), and select by prequential loss.
- **One computable criterion ranks whole learners:** the prequential code length of the data stream, a valid MDL code
  for model + learning algorithm (Blier & Ollivier 2018).
