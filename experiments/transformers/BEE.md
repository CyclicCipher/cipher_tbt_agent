# Can a transformer be as smart as a bee?

*Direction set 2026-07-27. "Smart" is defined narrowly and deliberately: **speed of skill acquisition** — how many trials
it takes to solve a problem it has never seen — not asymptotic performance on a benchmark it was trained for.*

## Why a bee is the right bar

A bumblebee has on the order of a million neurons and does the following:

- learns to **pull a string** to drag an artificial flower out from under a screen, a task with no natural analogue, in a
  few dozen trials — and naive bees acquire it by *watching* a trained demonstrator (Alem et al. 2016, *PLOS Biology*);
- learns to **roll a ball** to a goal for reward, again from demonstration, and then **improves on the demonstration** —
  when given a choice, bees rolled the *nearest* ball rather than the one the demonstrator used (Loukola et al. 2017,
  *Science*). That is not imitation; it is extracting the goal and re-solving it;
- learns a new flower colour/shape association within a handful of visits.

So the bar is: **single- to double-digit trials on a genuinely novel problem**, with transfer of a *goal* rather than a
motor sequence. No gradient descent over millions of examples anywhere in that story.

The comparison is not meant to be cute. It is the same quantity ARC-AGI-3 scores (`reference_arc_agi3_scoring`: RHAE, on
actions taken), and the same quantity the TBT line in `src/tbt/` exists to attack. A bee is the existence proof that the
number can be small at a scale where "just scale it" is not the explanation.

## What we already measured, and why it reframes the question

`linreg.py` (results in `NOTES.md`) found that a 4-layer transformer performs **in-context linear regression within
0.005–0.02 of the Bayes-optimal predictor after 1–4 examples**, with *no gradient steps at inference*. Read as skill
acquisition, that is already bee-speed: a fresh `w` it has never seen, solved in about four trials.

The catch is what counts as novel. A fresh `w` is a new *instance*; it is not a new *task family*. The model saw ten
thousand batches of exactly that family in training. So:

> **A transformer is already bee-fast INSIDE a task family it was trained on, and needs a full retraining run to acquire a
> family it was not. The bee's advantage is not speed per trial — it is that its "training distribution" seems to cover
> string-pulling without ever having contained it.**

That is the actual gap, and it is the one worth attacking.

## Where LID and harnesses come in

Alex Zhang, *Language Model Harnesses Are Compositional Generalizers*
(<https://alexzhang13.github.io/blog/2026/harness/>; memory `reference_lid_locally_in_distribution`):

- **LID — "locally in-distribution":** a task state can be globally out-of-distribution while *every individual model call*
  on it stays inside the training distribution. The claim is about **structural** similarity, not token overlap — prompts
  landing in an ε-ball in trajectory space.
- **A harness** is the program between the world and the network: it decides how to encode state into inputs and how to
  turn outputs into actions. Its real job, on this account, is **reducing an unfamiliar problem to a composition of
  familiar ones** — not calling tools.
- The harness therefore **induces an equivalence relation between tasks with latent similarities**, so isomorphic tasks
  look identical to the model.
- Evidence: Recursive LMs trained on 2–64k-token contexts generalise to tasks **8–32× longer**, and a decomposition
  strategy learned on Jeopardy questions transfers to spam classification. ~10× the eval lift of a base transformer at
  matched train improvement.
- **Explicitly not claimed:** that we should hand-tune task-specific harnesses. He calls that walking straight into the
  bitter lesson. The claim is about *scalable* architectural bias.

**The connection to the bee:** LID is a hypothesis about *why* a bee is fast. String-pulling is globally novel, but every
local operation in it — approach, grip, pull toward self, monitor reward — is something a bee's control system already
does. The bee is not solving a new problem; it is *composing* old ones, and the composition is what is new. If that is
right, then "make a transformer bee-fast" is not "train it on more" but "**give it a decomposition under which the novel
task is locally familiar**".

## Three refinements to Zhang's setup (2026-07-27, the user's)

**1. Train THROUGH the harness from scratch, rather than wrapping a pretrained model.** If the harness is a quotient map
`π`, a pretrained model was trained on the UNQUOTIENTED distribution, so bolting `π` on afterwards only works if the fibers
of `π` happen to land where the model is already competent — and Zhang's RL step is precisely the search that repairs that
mismatch. Train through the harness and the training distribution IS the quotient space: LID stops being something you
search for and becomes true by construction, because the model never saw anything else. Prediction: the from-scratch
version needs far less search over decompositions. **The honest counter**, to be held in view: pretraining is where the
broad competence to RECOMBINE comes from, so from-scratch may be sharper but more brittle — excellent on one quotient with
no reservoir when it is the wrong one. The bee is evidence for the user's side: no pretraining, and it still composes.

**2. Context offloading is the wrong quotient for this question, so it is dropped.** Offloading quotients out ONE OOD
direction — context length — which is the axis LLMs happen to be bad at. Our novel direction is TASK STRUCTURE. Keep the
idea (quotient, equivalence class); discard that instance of it.

**3. Token edit distance is a surface proxy for a model-relative property, so it is replaced.** "In-distribution" is a fact
about the MODEL, not about the strings — two prompts can be token-similar and structurally different, or token-different
and structurally identical. The measure used here is the model's own **prediction error on the local call**
(`feedback_epistemic_value_is_prediction_error`; the epiplexity work in `src/tbt/`): low loss on a local operation IS what
"locally in-distribution" means. Model-relative, cheap, and non-circular as long as it is measured after the fact rather
than used to build the harness. It turns LID into a number instead of a vibe.

## The hypothesis ladder

Cheapest falsifier first. Each step is built so it can come out negative.

- ~~**H1 — is LID even the right explanatory variable?**~~ **RETIRED 2026-07-27.** Unanswerable as posed: the measured
  acquisition rate on held-out tasks is ZERO in every condition, and a speed cannot be correlated against anything when
  nothing is acquired. Superseded by **H0′ — does it compose at all, or memorise?** (`diversity.py`), which found complete
  memorisation and zero transfer across a 10x task-diversity range, with the task SUPPLY (62 distinct) now the blocker.
  Original wording kept below, since the design decision it turned on — holding global novelty constant — still stands.
- **H1 (original) — is LID even the right explanatory variable?** Hold GLOBAL novelty constant and vary only how familiar the LOCAL
  operations are; see whether trials-to-criterion tracks local familiarity. If it does not, the frame is wrong and the rest
  of the ladder is moot. **This runs first precisely because it can kill the programme.**
- **H2 — the from-scratch claim.** Same network, same task: trained THROUGH a given correct harness from scratch, versus
  pretrained-then-wrapped. Predicts from-scratch wins and needs less search.
- **H3 — the prize.** Can the quotient be DISCOVERED rather than given? A harness we hand over is an inductive bias we
  chose; a harness the system finds is the bee. This is where the bitter-lesson discriminator actually bites
  (`reference_lid_locally_in_distribution`: scalable inductive bias, never task-specific rigging).

### Where the ladder actually stands (2026-07-28)

**Everything up to and including inference-time chaining is null.** `scratchpad.py` appeared to clear the bar — held-out
composition 0.046 → 0.363 — and the arity test **retracted it**: the measurement was teacher-forced, so the model was
handed the true intermediate and only had to finish. Free-running it scores 0.024 against a 0.047 control (`NOTES.md`,
the arity entry). The ledger is architecture null, optimiser null, diversity null, chaining-as-training-signal
transductive-only, chaining-at-inference null.

**The one thing measured that is not null**, and it is the sharpest statement the line has produced: given a correct
partial result, the model applies one more primitive to a NOVEL composition at 0.48–0.59 against a 0.11 one-shot control.
It can execute a local step out of distribution; it cannot sequence two of its own. That is LID confirmed at *m*=1 and
refuted at *m*=2, and it is where H2/H3 now have to bite.

**The standing constraint on any harness we build from here (the user, 2026-07-28): NO ARITY MAY BE HANDED OVER.** There
is no program that can always supply the correct number of steps, so a format that fixes the decode depth in advance has
smuggled in the answer's shape. Two things learned about doing this properly:
- Padding a fixed-depth format by REPEATING the answer does not remove the arity, it removes the task — the repeated
  blocks give the model something to copy, copying dominates the loss, and the measured 1.000 was exactly that.
- So the depth must be the model's own output: a HALT it emits, with no repeated blocks to copy and no supervision of the
  step COUNT for any held-out task.

**BUILT, and the arity turned out not to be the bottleneck** (`halt.py`, `NOTES.md` 2026-07-28). With nothing handed
over, the model **infers the depth of a novel composition**: 0.842 exactly-right block count on held-out triples and
0.772 on held-out pairs, against a base-rate ~0.60/0.23, terminating 100% of the time. That is the first genuine
positive in the line. But the ANSWER is 0.048 against a 0.113 one-shot control — a self-chosen chain makes held-out
accuracy *worse*, because the FIRST block is wrong two times in three (0.344, against 0.903 on supervised tasks) and
every later step inherits it.

**So the ladder's remaining question is no longer about the harness at all.** The model represents a novel composition
(held-out R² +0.52), knows how many steps it needs (0.84), and finishes it correctly when handed a correct partial
result (0.52) — and cannot produce that partial result (0.34). Every external lever has now been varied and come back
null. **The missing operation is applying an identified primitive to a held value when that pair was never trained
together** — one local step, which is exactly the LID claim at its smallest unit, and it is where H2/H3 have to be
re-aimed.

## The experiment this implies

The blog tests LID with pretrained LLMs and expensive harnesses, where "structurally isomorphic" is informal (an ε-ball
under an unspecified metric). We can test the *principle* in a synthetic setting small enough to iterate in seconds, and —
the part that makes it worth doing — where **isomorphism is exactly definable rather than asserted**: two tasks are
isomorphic when related by a group action the model has already seen, which is a statement we can construct and check,
not an intuition.

The measure throughout is **trials-to-criterion**, the bee measure, not final loss.

1. **Baseline — where does in-context acquisition stop?** Train on a family of function classes; measure trials-to-criterion
   on (A) a held-out instance of a trained class, (B) a class that is globally new but *locally isomorphic* to a trained
   one, (C) a class that is globally new and *not* locally isomorphic. LID predicts A ≈ B ≪ C. If B is slow, LID is wrong
   at this scale and the rest is moot — so this runs first, and it is designed to be able to say that.
2. **Does a harness convert C into B?** Add a decomposition step that re-expresses the C-task as a composition of trained
   operations, and re-measure. This is the falsifiable core of the harness claim: the *same* network, the *same* weights,
   the only change being how the problem is presented.
3. **The bee's real trick — transfer of a goal, not a sequence.** Loukola's bees improved on the demonstration. The
   corresponding test: after a harness makes a task solvable, is what transfers the *decomposition* (reusable on a task
   needing a different sequence of the same parts) or just the sequence? This is where the honest failure is most likely.

**Standing caution, from the memory and from this project's own history:** Zhang's harness wraps a *pretrained* model with
broad competence, and ours will not. Nothing here licenses hand-coding task logic into the harness — a harness that
encodes the answer measures nothing. The discriminator to apply at every step is the one already recorded: *scalable
inductive bias* (legitimate) versus *task-specific hack / environment-rigging* (not), and a proposed harness that cannot
state its equivalence relation in advance is the latter wearing the former's clothes.
