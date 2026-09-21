# Roadmap — compositional reasoning, and data efficiency by thinking like L

*Written 2026-08-05, after `sinkhorn.py` / `endtoend.py` / `bits.py`. Results live in `NOTES.md`; the line's goal is in
`BEE.md`. Every step below names a question, a measurement, and what would falsify it — a step with no falsifier is not
on the roadmap.*

## The unifying claim (revised 2026-08-05, after steps 1–4)

**Original form.** `detective.py`'s answerer scored 0.008 on random probes that leave only 0.23 bits, and 0.969 on
designed probes leaving 0.00. So the bottleneck was not query selection but the inability to INTERSECT constraints
across observations — which is exactly what `sinkhorn.py` does. The extraction step in the L loop and the matching step
in the compositional loop are the same operation, so the roadmap has one spine.

**What steps 1–4 sharpened it into.** The two are not merely the same operation, they are *mutually determining*:

* **A question's value is a property of the reader, not of the question.** The exact posterior and the prior-free solver
  rank probe sets in OPPOSITE order, provably — an all-distinct probe confounds permutation with relabelling completely,
  so it is the enumerator's best probe and the solver's worst. There is no observer-free notion of a good experiment.
* **Which means an agent's questions are capped by its own comprehension.** Given a weak answerer, the policy did not
  fail to find good questions; it found questions it could ANSWER and traded information away to get them (2.70 bits vs
  random's 0.23, yet 30× random's accuracy). Swapping in a competent answerer, changing nothing else, took it to 0.03
  bits and it discovered the hand-derived design unaided.
* **So the constructive rule is: design for the least-informed reader.** EIG computed for the weak solver is near-optimal
  for the enumerator too (0.02 bits); the reverse fails badly. Optimising questions for a reader who already knows the
  answer produces questions that are barely questions.

## Surprises worth carrying forward

1. **Random observations are nearly fully informative here** (0.23 bits left after two), and I asserted the opposite
   repeatedly. Wherever a system looks starved of data, check whether it is starved of EXTRACTION first.
2. **Credit assignment was never the constraint.** Best-of-4 self-imitation over a 15625-wide action space was adequate
   the whole time; it looked broken because the reward was dishonest. The queued fix (REINFORCE, larger K) was wrong.
3. **Ensembles collapse overconfidently.** Solver restarts went from 11.88 distinct hypotheses to 3.47 in three probes
   while accuracy was only 0.789 — agreeing on wrong answers. 98.5% of tasks then re-asked a question already asked.
   Restart-spread is a weak proxy for posterior spread; never use it without a diversity guard.
4. **Next-token training did NOT hedge across equivalent answers — it collapsed onto one.** 0.859 of 0.859 class mass on
   a single one of ~12 correct writings. So surface-form scoring costs little ACCURACY and nearly all the CONTENT, which
   is a different complaint from the one usually made and needs different evidence.
5. **Three predictions stated in advance: two held, one was too pessimistic.** The no-metric cliff held exactly; the
   solver/enumerator probe inversion held; the no-repeat guard was predicted to restore parity and instead won outright.

## Where we actually are

| claim | status |
|---|---|
| execution / application generalises | settled — 0.997 from 30 examples |
| behaviour → program | **0.762**, was 0.000 (`endtoend.py`) |
| surface-form output scoring costs accuracy | **barely** — 0.867→0.895; but it costs the CONTENT (0.859 of 0.859 mass on one of ~12 answers) |
| behaviour → canonical state | **1.000** at 8 demos — and the solver costs ~10 steps, not 400 (`unroll.py`) |
| the amortisation gap is worth closing | **no** — the optimiser was already cheap; a trained one-shot init is WORSE than noise |
| a canonical representation has no metric | settled — degradation is a cliff, prediction held |
| an operation is a basis, not a rule | settled — and +/× share no basis, so some duplication is forced |
| the matching prior is harmless where unneeded | **no** — costs 28 pts on a non-injective map; and buys nothing at K=8 |
| what a probe set is worth, in bits | **measurable** (`bits.py`), exhaustive optimum known |
| chosen probes beat random ones, for us | **yes, 2×** — exact at 4 chosen probes vs 8 random (`eig.py`) |
| probe quality is intrinsic | **no** — it is relative to the reader's prior; design for the least-informed |
| a learned agentic policy asks good questions | **yes, given a competent answerer** — 0.03 bits, and it found the principle |
| the query bottleneck was co-adaptation, not credit assignment | settled — step 1, 2.70 → 0.03 bits from swapping the answerer |

---

## Phase 0 — close the loop between the two strands

### Step 1. Give the detective a competent answerer — **DONE 2026-08-05, co-adaptation CONFIRMED**

Swapping only the answerer (model's own answer loss → the exact posterior), with rollout, best-of-K, K=4 and step count
held fixed, took the learned probes from **2.70 bits to 0.03** and from 6% to 97% identified. The policy also DISCOVERED
the hand-derived principle unaided: all five values present, duplicate moved between probes, distinct/6 2.25 → 4.88.

**The falsifier resolved against the lever we had queued.** Credit assignment was never the binding constraint —
best-of-4 over a 15625-wide action space was adequate once the reward was honest. REINFORCE-with-baseline and larger K
are struck from the plan.

Two things the result hands downstream. It sits at 1.15 bits against greedy's 0.65, so it inherited our blind spot along
with our principle — neither exploits which of 14400 group elements are actually reachable. And the answerer used
ENUMERATION, so **step 1b is now the immediate run**: repeat with `sinkhorn.py`'s solver in place of the posterior. The
result says answerer quality is the whole game, which makes a scalable competent answerer urgent rather than optional.

### Step 1b. Repeat with the Sinkhorn answerer — **RUN 2026-08-05, INCONCLUSIVE (budget), but it reordered 2 and 3**

The training arm did not run to convergence: the solver costs ~19 ms/Adam-step at training scale, capping it at 150
steps against the oracle arm's 1000, and at 150 steps the policy is worse than random under both measures. That is a
compute failure, not a finding; a real run needs compute outside the interactive budget.

**What it did settle, without any training.** The exact posterior and the prior-free solver rank probe sets in OPPOSITE
order — designed probes are the posterior's best (0.00 bits) and the solver's worst (0.110 IDX exact), random probes the
reverse. The mechanism is provable: an all-distinct probe confounds permutation with value-relabelling completely (for
any `idx'` there is a `vmp'` fitting the data), so only REPEATS constrain `idx`. The posterior escapes this by knowing
the universe; a solver searching 720 × 120 cannot.

**So probe quality is not intrinsic — it is relative to the observer's prior**, and this line had been treating it as
intrinsic throughout.

### Steps 3 and 2. EIG without enumeration, and choosing vs sampling — **DONE 2026-08-05, `eig.py`**

Both settled in one experiment, since the mechanism and its payoff share a measurement.

**Step 3.** Information gain over the solver's own hypothesis space, nothing enumerated on either axis: R=12 solver
restarts as posterior samples, C=512 sampled candidates, scored by disagreement about the predicted response
(Query-by-Committee / BALD). Before any observation the samples come from the solver's prior — uniform permutations.

**Step 2.** **Choosing beats sampling by 2×**: exact recovery at 4 chosen probes where `sinkhorn.py` needed 8 random
demonstrations. At budget 2, 0.669 vs random's 0.436. The first data-efficiency gain the L framing has produced inside
our own pipeline rather than against a hand-derived ceiling.

**And the first version lost at budget 4** (0.789 vs random's 0.970), which located a real failure: the restart ensemble
collapses — 11.88 distinct hypotheses among 12 before any observation, 3.47 by the third probe — so disagreement goes
flat, argmax turns arbitrary, and 98.5% of tasks re-asked a question already asked. Excluding asked probes fixed it
entirely. The collapse is partly OVERCONFIDENCE (accuracy was 0.789 when the restarts agreed), so restart-spread is a
weak proxy for posterior spread and should be treated as one wherever it is reused.

**Constructive rule for everything downstream:** EIG computed for the prior-free solver is near-optimal for the
enumerating observer too (0.02 bits), while the reverse fails badly (designed probes: best for the enumerator, worst for
the solver). **Design the experiment for the least-informed reader.**

---

## Phase 1 — finish compositional reasoning

### Step 4. Canonicalise the OUTPUT — **DONE 2026-08-05, `quotient.py`. Partly falsified as motivated.**

The ambiguity is large (90.2% of functions have >1 minimal writing, mean 12.53), and the class-marginal objective
`-log Σ_{p∈class} P(p)` gives **0.867 → 0.895** functional accuracy, a 21% cut in error.

**But the premise was wrong in an instructive way.** I expected the single-target objective to be splitting probability
across equivalent writings. It was not: its class mass EQUALS its single-writing mass (0.859 / 0.859), so it had already
collapsed onto one writing and assigns ~0 to the other eleven. Nothing was lost to hedging, which is why fixing the
hedge buys little.

**The ontological result stands and is now a measurement.** Surface-form training yields a model that calls eleven of
twelve correct answers wrong — a phrasing, not the content. The quotient-trained model holds 0.848 on the class while
giving the BFS writing 0.280. That distinction is what the whole next-token critique was about, isolated in one table.

**Consequence for step 8.** The intervention objective was justified partly as subsuming this one. It still is — any
program reaching the target scores — but the expected ACCURACY payoff from removing surface-form scoring should now be
priced low. Step 8's value is the naming-vs-using dissociation, not the quotient.

### Step 5. Fix amortisation — **DONE 2026-08-05, `unroll.py`. Premise refuted; the gap was not worth closing.**

The solver reaches **1.000 from noise in 40 Adam steps and 0.981 in 10**. Everything in this line ran it at 400 because
that is the number I picked; convergence was never measured. The cost premise was inflated tenfold.

An unrolled network does reach 1.000 at T=6 — but an **UNTRAINED** network with the same refinement reaches 0.981 at
T=6 and 1.000 at T=10, so the learned initialiser contributes nothing. The refinement steps do the work.

**What is genuinely new, and it inverts the hypothesis:** the one-shot matcher's output is an actively HARMFUL
initialisation (0.648 after 200 Adam steps, against noise's 0.981 after 10). A network trained to emit the ANSWER
produces a confident wrong answer in a basin gradient descent cannot escape; trained to emit a STARTING POINT it
produces a good one. Same architecture, opposite utility, decided by what the objective asked of it.

**Consequence:** the 0.756 amortisation gap is struck from the open-problems list — not closed, but priced correctly.
Anywhere else this line reaches for amortisation, measure the optimiser's real cost first.

### Step 6. Audit the prior we added — **DONE 2026-08-05, `audit.py`. It DOES hinder.**

Tested where the prior is FALSE, not merely unnecessary: a non-injective read map, which row-softmax expresses and a
doubly-stochastic matrix cannot. Reconstruction on fresh inputs, fit on 8 demos:

| structure | prior | sinkhorn | row-softmax |
|---|---|---|---|
| permutation | TRUE | 1.000 | 1.000 |
| many-to-one | FALSE | **0.713** | **0.990** |
| global | IRRELEVANT | 0.009 | 0.199 |

**It costs 28 points one small step outside its domain, and buys nothing inside it at this data volume.** Sinkhorn's
advantage was at K=2 (0.511 vs 0.215) and gone by K=8 — so in the regime most of this arc ran in, the prior contributed
nothing. Its value is entirely at K ≤ 4: a LOW-DATA prior, not a capability.

**Blocks step 7.** Relaxing to unbalanced transport — dropping the column constraint, or interpolating row-/doubly-
stochastic with a learned weight so the data decides whether the correspondence is a bijection — has to come first. Step
7 is by definition a move to structures the current constraint forbids.

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
