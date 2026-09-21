# On-Policy Self-Distillation, read for ZipLearner (2026-09-21)

*A research note, not a design section. Sources: Zhao et al., "Self-Distilled Reasoner: On-Policy Self-Distillation
for Large Language Models", arXiv:2601.18734 (Jan 2026, v3 Mar 2026); "A Brief Overview: On-Policy Self-Distillation
in Large Language Models", arXiv:2605.18141 (May 2026); Song & Zheng, "A Survey of On-Policy Distillation for LLMs",
arXiv:2604.00626; Li et al., "Rethinking On-Policy Distillation", arXiv:2604.13016; "On-Policy Self-Distillation
without Any Supervision", arXiv:2608.06296.*

## What OPSD is, in defined terms

One model, two contexts. The **student** is the model given only the question. The **teacher** is the *same weights*
given the question plus privileged information — the verified answer, a reference solution, a hint. Trajectories are
sampled from the student (on-policy), and at every token the student's next-token distribution is pushed toward the
teacher's on the same prefix:

    L(θ) = E_{(x, y*) ~ D, y ~ π_θ(·|x)} Σ_t KL( π_θ(·|x, y_<t) ‖ π_θ(·|x, y*, y_<t) )

(reverse KL; a symmetric Jensen–Shannon variant is used for stability). There is no reward model and no second model.
The stated premise is "evaluation is easier than generation": knowing the answer, the model can produce the reasoning
it could not find blind, and that reasoning becomes the target for the blind model.

## Why it beats RL, in their account and ours

RL (GRPO/PPO) gives one scalar per trajectory, assigned uniformly to every token: sparse, and the credit assignment
has to be inferred; when every sample in a batch is right or every one wrong the gradient vanishes. OPSD gives a whole
distribution at every token: dense, no reward model, no separate teacher (40–60% less memory than on-policy
distillation with one), and the target is the model's own best hindsight rather than an external judge.

In this project's terms the point is sharper. A scalar reward is the **outer objective with no inner one** — the
situation of the inner-objective runs, where backprop's hidden rule ("be less wrong on this question next time") was
the only thing shaping the weights. OPSD supplies an inner signal by *constructing the target in hindsight* from
privileged information, per step. That is a description-length learner's native move: with the outcome known, find
the description that would have predicted it; make the foresight description match. The dense per-token signal is
what our rate price already gives per digit and per cell — every exception has a location (E4, E11).

## What OPSD lacks, seen from here

1. **The hindsight target is a behaviour, not a structure.** The student copies the teacher's token distribution. If
   the teacher rationalises a right answer with a wrong reason, the student learns the wrong reason — the
   "rationalisation of errors" failure the overview names — because nothing prices the explanation. Here the hindsight
   description is priced in bits, and a rationalisation that does not compress is rejected like any other exception.
2. **It overwrites.** Teacher and student share weights; the foresight model is replaced by the distilled one. §9
   keeps descriptions as separate blocks until the price merges them (E6, E15), which is why nothing is forgotten by
   accident. The known OPSD failure "rich conditioning suppresses uncertainty, in-domain gain, out-of-domain loss" is
   an overwrite with no exception rate attached; a block's exception rate *is* its uncertainty.
3. **It needs the privileged information given.** In a game the only privileged information is the score and the
   outcome (WIN / GAME_OVER). E13's win keys are already hindsight descriptions of the score; the door in LockPath is
   the case where the hindsight description needs a *condition* the foresight one lacked.
4. **It needs a capable base.** The teacher can only rationalise what the model can already express. Our analogue is
   the library: hindsight can only propose structures the library has (E11's map needed the edit structure first).

## The first-principles version for ZipLearner — E19, hindsight by description (pre-registered here)

After an episode (a plan executed, the level won or the budget spent), a **hindsight pass** over the episode with the
outcomes known:

1. For each transition where the foresight prediction failed, find the cheapest structure in the library that would
   have predicted the observed next state given what was known *at that step plus the outcome* — the teacher: the
   description with privileged information. Its price against the foresight description's is the lesson, in bits.
2. For the plan: the hindsight-optimal plan is search on the updated model; the regret is plan bits taken minus
   optimal; the first action where the two plans diverge marks the missing structure.
3. Distil: add the hindsight structure (mint, or an edit to an existing block) if its horizon-weighted saving (E10)
   exceeds its cost; never overwrite a block — §9's rules apply.

Test on E13's recorded failure: LockPath level 1 (the door). Pass: after one failed attempt, the hindsight pass
names the transition that broke the plan (the move into the door) and adds a description under which the foresight
planner no longer walks into it; the level is then solved within the budget on the second attempt. Refute: the pass
finds no cheaper hindsight description, or the second attempt fails the same way.

This is OPSD's shape — foresight made to match hindsight, densely, on the agent's own trajectories — with descriptions
in place of token distributions and a price in place of a KL. The difference that matters is the price: it decides
which hindsight explanations are worth keeping.
