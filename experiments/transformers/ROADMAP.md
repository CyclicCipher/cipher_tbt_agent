# Roadmap — compositional reasoning, and data efficiency by thinking like L

*Written 2026-08-05, after `sinkhorn.py` / `endtoend.py` / `bits.py`. Results live in `NOTES.md`; the line's goal is in
`BEE.md`. Every step below names a question, a measurement, and what would falsify it — a step with no falsifier is not
on the roadmap.*

## The unifying claim

The two goals are one problem, and the evidence for that is now direct.

`detective.py`'s failure mode is that RANDOM probes leave only 0.23 bits of ambiguity yet its answerer scores 0.008,
while hand-designed probes leave 0.00 bits and score 0.969. The original diagnosis was right: random probes require an
**intersection of constraints across observations**, designed probes make it a **lookup**. So the data-efficiency
bottleneck is not query selection — it is the inability to intersect.

And intersecting constraints across observations is exactly what `sinkhorn.py` does. Both unknowns are assignments, each
inferred through the other, and 8 demonstrations are combined into one correspondence. **The extraction step in the L
loop and the matching step in the compositional loop are the same operation.**

That is why the roadmap has one spine rather than two, and why step 1 is the highest-value experiment available: it
connects the two halves and simultaneously discriminates the last two open hypotheses about the detective's failure.

## Where we actually are

| claim | status |
|---|---|
| execution / application generalises | settled — 0.997 from 30 examples |
| behaviour → program | **0.762**, was 0.000 (`endtoend.py`) |
| behaviour → canonical state | **1.000** at 8 demos (solver), 0.756 amortised |
| a canonical representation has no metric | settled — degradation is a cliff, prediction held |
| an operation is a basis, not a rule | settled — and +/× share no basis, so some duplication is forced |
| what a probe set is worth, in bits | **measurable** (`bits.py`), exhaustive optimum known |
| chosen probes beat random ones, for us | **unmeasured** — the central untested claim of the whole L framing |
| a learned agentic policy asks good questions | **no** — 2.70 bits vs random's 0.23; it co-adapted to a weak answerer |

---

## Phase 0 — close the loop between the two strands

### Step 1. Give the detective a competent answerer *(highest value; do first)*

Replace the trained answer head with the Sinkhorn solver: per-task, untrained, and — crucially — **distribution-
insensitive**, since it optimises each episode from scratch and therefore cannot be off-policy and cannot memorise.
Train only the query policy against it.

*Why.* Two hypotheses for why the policy asks bad questions have never been separated: **co-adaptation** (better queries
arrive off-distribution for the answerer, so improvement is punished) and **weak credit assignment** (best-of-4 over a
15,625-wide action space). A perfect answerer removes the first entirely.

*Measure.* Bits remaining after 2 learned probes, against the now-known reference points: 2.70 learned, 0.23 random,
0.00 designed and greedy.

*Falsifier.* If bits stay near 2.70 with a perfect answerer, co-adaptation was **not** the cause and credit assignment
is — which redirects everything downstream to REINFORCE-with-baseline and budget annealing instead.

### Step 2. Does choosing actually beat sampling? *(the L framing's first real test)*

Feed the Sinkhorn solver 2–3 **greedy-EIG** probes instead of 8 random demonstrations, and re-run the recovery curve.

*Why.* Every claim that "thinking like L" buys data efficiency has so far been made about a hand-derived ceiling, never
inside our own pipeline. This is the first place the two components meet on a number we care about.

*Measure.* Recovery exact-match vs number of observations, chosen vs random. Predicted: exactness at 2–3 rather than 8.

*Honest confound to handle.* `bits.py`'s EIG uses the universe prior (2350 candidates = 11.2 bits); the solver has no
such prior and searches ~86,400 (16.4 bits). A like-for-like version needs EIG computed over the solver's own hypothesis
space — which is step 3, so run step 2 first as an upper bound and read it as one.

### Step 3. EIG without enumeration

`bits.py` enumerates 2350 hypotheses × 15,625 candidate probes. Neither scales, and `detective.py` refuses to enumerate
on principle. The scalable form: the posterior is already carried by the solver's `(P, M)` — a distribution, not a set —
so information gain can be estimated by **sampling hypotheses from it**, or by the disagreement among an ensemble of
solutions started from different initialisations.

*Why.* This is the step that turns L's method from an analyst's lookup into a process a model runs. Without it the whole
framing stays a measuring stick.

*Measure.* Sampled-EIG probe selection vs exhaustive-EIG, in bits. Falsifier: if sampled EIG cannot match exhaustive on
a universe this small, it will not work on a larger one and the enumerating version is all we have.

---

## Phase 1 — finish compositional reasoning

### Step 4. Canonicalise the OUTPUT, not just the input

**The asymmetry nobody noticed.** `endtoend.py` recovers a canonical state and then hands it to a next-token model that
emits ONE program among many equivalent writings. So the input has been quotiented by equivalence and the output has not,
and the objective is still scoring surface form. We papered over this in the EVALUATION — `canon2prog.py` scores by
FUNCTION, "an equivalent writing counts", built precisely because token-level scoring marks correct answers wrong for
being phrased differently. The patch belongs in the objective, not the metric.

*Why it matters beyond tidiness.* Next-token prediction cannot distinguish "knows the content" from "reproduces the
corpus's preferred phrasing of it": if content C has writings s1..sk, the objective rewards getting the SPLIT right,
which is a fact about convention rather than about the world. Nothing in training ever checks for the invariant, so
whatever invariance the model has is incidental — and therefore unreliable and unmeasurable. Our own numbers are that
claim quantified: behaviour → program is 0.000 and canonical → program is 0.733, same target, same architecture, the
only difference being whether the INPUT was quotiented. There is no reason to expect the output side to be different.

*The step.* Emit the canonical content rather than a token sequence — target the function's own invariant description,
so that all equivalent programs are one target rather than k competing ones. Then FUNCTIONAL scoring stops being a
generous metric and becomes the literal objective.

*Measure.* Held-out functional accuracy against the 0.762 baseline, plus the number that does not currently exist: how
much of the residual 24% is genuine failure versus the model splitting probability across equivalent writings.

*Falsifier.* If accuracy does not move, program-writing ambiguity was not costing anything here — the universe has short
programs and may have few equivalent writings per function, in which case this is a correctness fix rather than a
performance one, and should be reported as such.

### Step 5. Fix amortisation

The solver reaches 1.000; the one-forward-pass network reaches 0.756, with train ≈ held-out — so it is optimisation, not
generalisation. The natural fix is an **unrolled solver**: iterate score → Sinkhorn → reconstruct → refine as network
layers, i.e. learn the solver rather than guess its answer.

*Why it matters.* 400 gradient steps per task is fine at this scale and hopeless beyond it. And by the no-metric result,
0.756 is not "most of the way" — it is a 24% total failure rate downstream.

*Falsifier.* If an unrolled solver still plateaus below 1.000, the limit is the score function, not the number of steps.

### Step 6. Audit the prior we added

The matching layer is a good prior by the stated criteria — small, uniform, no task content. The untested criterion is
whether it **hinders** anything. Run it on a task with no correspondence to recover; it should be inert.

*Why.* This is cheap, and it is the difference between "we chose a good prior" and "we assume we chose a good prior."

### Step 7. Break the closed universe

Everything so far is 6 slots, 5 values, one group. The canonical form is a permutation-plus-relabel because we picked a
domain where it is. Present two or three structurally different domains and ask whether one architecture recovers the
canonical form in each **without being told which kind applies**.

*Prediction, from `project_an_operation_is_a_basis`.* Different domains need different bases, and + and × provably share
none — so expect duplication rather than a single shared circuit, and measure how much. A negative result here is
informative: it bounds how far "supply the structure" can be pushed before the structure has to be discovered.

---

## Phase 2 — the bee question proper

### Step 8. The INTERVENTION objective — does it USE the structure, or only describe it?

Every task in this line is passive observation of a function. The strongest form-free objective — and the only one the
bee evidence is actually about — scores a CONSEQUENCE OF ACTING.

*The design.* Give a start state and a TARGET state; the model emits primitive operations; the environment executes;
scored on whether the resulting state equals the target. Nothing about the program is scored, only what it achieves.

*Why this objective and not another.* Equivalence is free — ANY program reaching the target scores 1, so the surface-form
blindspot cannot arise, and this subsumes Step 4's problem rather than patching it. It is also the closest thing in this
line to Loukola's bees, which watched a demonstrator roll one ball and then rolled the NEAREST one: they extracted the
goal and discarded the form, which is quotienting by equivalence performed by an insect.

*The measurement that makes this worth doing.* We have behaviour → program at 0.762. Does that translate into
behaviour → ACHIEVE A GOAL? **If naming and using dissociate, that is the ontology question made empirical**: a model
that can describe a function but cannot act with it did not possess the function, it possessed a description. If they
track each other, description was sufficient and the distinction is idle here.

*Falsifier.* If goal accuracy tracks program accuracy across the whole difficulty range, there is no dissociation to
find and the intervention framing buys nothing in this universe — report that, and the case for family (3) rests on
domains where the state space is larger than the description space.

*Cost to be honest about.* This is the sparsest signal in the line: one bit per episode, against reconstruction's one
per observation. Expect it to be much harder to train than anything above it, and expect that difficulty to be the
finding rather than a nuisance.

### Step 9. A new family, not a new instance

`BEE.md`'s actual gap: a transformer is already bee-fast inside a task family it trained on, and needs a full retraining
run for one it did not. Everything above improves the *instance* case — a much harder instance than `linreg.py`'s, since
it requires recovering discrete structure, but still an instance.

The experiment: train on domain A, present domain B, measure trials-to-competence. The prediction is that it fails,
because B's canonical form is not representable. **The interesting question is whether the bits machinery transfers even
when the representation does not** — does an agent that knows how to ask good questions acquire B faster than one that
does not, carrying a wrong structural prior?

That is the sharpest form of the bee hypothesis available to us: *the bee's advantage is its investigative policy, not
its representation.* If it holds, query selection is the transferable component and structure is the local one. If it
fails, the reverse, and Phase 1 is the whole game.

### Step 10. Compose the loop

An agent that maintains a posterior over structured hypotheses, chooses actions by expected information gain over it,
recovers canonical structure by assignment, and reads programs off the canonical state. Each of the four pieces is
separately validated by then; this step is integration and a single end-to-end trials-to-competence number.

---

## What this shares with `src/tbt/`

Not a dependency, but the same two components under different names, and results should cross-check:

* **Correspondence recovery** is pose-invariant recognition (`reference_tbt_pose_invariant_recognition`) — mapping
  samples from an arbitrary viewpoint onto an object's identity. Both lines were stuck on it; one is now unstuck.
* **The hypothesis space in bits** is what `hippocampus/hypotheses.py` measures with `log2|H|`, and the no-metric result
  transfers as a warning: a partially-correct win condition is a *different* win condition, not a nearly-right one.
  Refutation has to reach certainty, not confidence.
* **The extraction-not-selection finding** predicts the TBT agent's exploration problem is likewise downstream of its
  ability to intersect constraints, not of which cell it walks to — worth checking before more work on `_explore`.

---

## Methodological note — training for novelty without leaking it

Raised while adding step 8, and it applies to every step above.

**The apparent paradox.** If the objective is "act well where you have not been", training on such situations makes them
seen. **The resolution is that novelty is a property of what is available AT DECISION TIME, not of the training set.**
Make the thing to be discovered a PER-EPISODE LATENT that is (a) freshly sampled, (b) not decodable from anything except
that episode's own observations, and (c) required for the reward. Then every training episode IS an unseen situation at
the moment the model must act, and there is nothing to leak. `linreg.py` already ran this design: a fresh `w` per
sequence, never shown, so the only way to succeed is to do the regression in the forward pass.

**The leaks that are real, and specific.**
1. *Novelty at the wrong level.* Sampling latents from one distribution lets the model memorise the FAMILY. Fine for an
   instance claim, fatal for a family claim — the split has to sit at the level the claim is made at.
2. *Through the reward.* If reward is computed from privileged information whose structure the model can infer, it can
   shortcut to the reward without the task. Executing the true function to score an outcome is safe; scoring against a
   stored answer key is not.
3. *Through the episode's own construction.* A fixed probe set, or a latent correlated with episode index, hands over the
   answer without ever showing it.

**And the part that is NOT a leakage problem, and cannot be fixed by objective design.** You cannot train for competence
on a distribution you did not sample — that is a logical limit, not an oversight. Any trainable objective defines a
distribution. Out-of-family generality is therefore reached INDIRECTLY, by making the acquisition MECHANISM general, and
then it is **measured, not trained**: hold out a family, train nothing on it, and report the number.

That makes a held-out family a measurement instrument rather than a training signal — and instruments of this kind are
spent by use. Every time it is looked at and the design is adjusted in response, it becomes a little more of a validation
set and a little less of a test. Step 9 should be run once, late, with the design frozen beforehand.
