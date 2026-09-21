# The inner-objective experiments — what we did and what we found (plain English)

## The idea being tested

When a neural network is trained the usual way, every step asks one question: "how should every weight move so that,
on the examples we just saw, the answer comes out less wrong?" That is the *outer* objective (be less wrong) driving a
hidden *inner* objective ("next time on this same question, do better"). Nothing in that step asks whether the change is
a piece of general structure or a one-off patch for those examples. Both look the same to the gradient.

The hypothesis: if we judge each proposed weight change by a different inner objective — **only keep a change that
different task families agree on**, and shrink toward zero what they cannot agree on — the network should learn the
shared structure faster, ignore families whose answers are random noise, and organise itself into recognisable parts.

Three experiments, all under the 20-minute rule.

## The result in one paragraph

The premise held and the remedy did not. A small transformer trained the ordinary way on 24 modular-arithmetic
families learns to *recognise which family it is in and recite that family's table* — on 12 new families built from
the same multipliers and offsets it is worse than guessing (8.3% of digits vs 9.1%) and pays twice the cost of a guess for it
(7.4 bits per digit vs 3.5 — the signature of confident wrong answers). That is the hidden inner objective — "be less wrong next time on this same question" —
doing exactly what it was suspected of. The proposed replacement — only keep a weight change that different families
agree on — was tried three ways and failed three ways, and a no-training diagnostic showed the reason: once learning
starts, the families' gradients disagree at essentially *every* weight by 25–45× the noise, so "shared structure" is
not a set of weights the families agree on. It is not in that basis at all. A gradient-free learner built on the
compression idea (ZipLearn) solves the same arithmetic exactly from **one** demonstration and knows when a family is
noise — but only because its two generic filters happen to be arithmetic's true shape, and it fails the moment a task
moves digits between positions. Across all three experiments the variable that decided generalisation was never the
optimiser; it was what the learner was allowed to hypothesise.


## 1. The no-model check (seconds)

Before touching a network we built a toy world where everything is exact: weights split into a block every family
shares, a block each family wants set its own way (memorisation), and a block nothing needs. We fed candidate rules the
per-family gradients and measured how much of the update went into the memorisation block.

- Plain averaging (the baseline): 0.84 of the memorisation error remains after training.
- Agreement on the *sign* of the change (the published AND-mask idea): no better — from a random start every family
  agrees on the direction toward its own target, so a sign vote lets memorisation through and freezes it in place.
- Agreement on the *value* of the change (an F-test: across-family spread no more than 3× the within-family spread),
  plus decay on what is not agreed (`ftest+wd`): 0.01 remaining, and immune to random-noise families, which fail to
  agree even with themselves.

So the rule is the right shape on paper. The two network experiments test whether it survives contact with a real net.

## 2. Run 1 — composing two operations (16 min)

Task: 7 primitive operations on 6-digit sequences; a "family" is an ordered pair of them; 17 pairs trained, 8 held out.
Network: 3 layers, 150,725 weights (small on purpose, so we could read it). Four arms: baseline, inner objective, and
each with three noise families mixed in.

| arm | training accuracy (whole answer right) | new-composition accuracy | noise families cost |
|---|---|---|---|
| baseline | 0.246 | 0.009 | −0.002 (none) |
| inner objective (agreement rule from step 0) | 0.003 | 0.000 | −0.000 |

- **The baseline never solved a new composition** (0.9%, second seed 5.4%; none of the 8 reached the 80% criterion).
  That is the known ceiling of this task at this size (`transformers/NOTES.md`): the pre-registered gate "the baseline
  must learn the task" failed, so the headline prediction (P1, faster learning of new compositions) could not be
  judged either way — it was censored, not refuted.
- **The inner objective learned nothing at all** (0.3% on the *training* families). Not because it distrusted the
  families (their two half-batches agreed at 0.47, well above the 0.2 trust line) nor because it masked everything (it
  kept 46–82% of the weights, median 60%). It froze because *agreement needs something to agree about*: at a random
  start no two families share any structure yet, every family's gradient points at its own answer, and a rule that
  only permits agreed changes permits none. The toy world had missed this because its shared block exists from step 0.
- **Noise families did not hurt the baseline** (−0.002), so the noise-immunity prediction (P2) had nothing to show; at
  this scale three random families are simply absorbed.
- **Inside the weights** (`runs/results/`): both nets grew weight clusters that line up with single primitives
  (3 of 9 clusters for the baseline, 5 of 9 for the inner objective, at selectivity ≥ 0.9), but only the baseline's
  clusters *do* anything — removing one changes its own primitive's accuracy by +0.34 more than the others' (a random
  cluster of the same size: +0.31, so even that is mostly size). Under the inner objective removing a cluster changes
  nothing (0.013 = the random control), because the network had not learned anything to remove. "Regions" in the
  participation statistics are cheap; regions that carry a function are not, and neither arm produced convincing ones.

## 3. Run 2 — modular arithmetic (the controlled test)

Task: x → (a·x + b) mod 11 on 6-digit sequences. Succession, addition, subtraction, multiplication and division are all
members (a=1,b=1; a=1; a=1,b=−k; b=0; a=k⁻¹,b=0). A "family" is one (a, b) pair; 24 pairs trained, 12 held out, and
every held-out pair's a and its b each appear in training with other partners — so answering a new pair is
*recombination*, never recall. The network sees 8 demonstrations of a family in its context and must produce the
outputs. Repair from run 1: the agreement rule now switches on only after 30% of training (`--rule warm`), so there is
shared structure to agree about. Same network, 1,800 steps per arm, 16.4 minutes for everything (`runs_affine/`).

| arm | training: whole answer right | training: digits right | **new families: digits right** (guessing = 0.09) | **new families: bits per digit** (guessing = 3.46) |
|---|---|---|---|---|
| baseline | 0.184 | 0.773 | 0.083 | **7.38** |
| inner objective (warm, then agreement rule) | 0.000 | 0.184 | 0.061 | 3.69 |
| baseline + 3 noise families | 0.003 | 0.425 | 0.085 | 5.00 |
| inner objective + noise | 0.000 | 0.212 | 0.069 | 3.64 |

Three findings, in order of importance.

**1. The baseline does not fail to generalise — it anti-generalises.** On the 12 new families it gets *fewer* digits
right than guessing would (8.3% vs 9.1%) and is confidently wrong: 7.4 bits per digit against 3.5 for a shrug. It has
learned "recognise which of my 24 tables this is, and recite it", and on a new (a, b) it recites the wrong table with
full confidence. Second seed: identical (0 held out). This is the hidden inner objective in its purest form: "next
time on this same question, be less wrong" was served perfectly (1.1 bits per digit on the training families) by a
strategy that is worse than nothing off them. **This is the hypothesis's premise, confirmed and measured.**

**2. The agreement rule cannot rescue it, and now we know why.** After the warm phase it masked 95% of the weights
(and the decay on masked weights then dismantled what the warm phase had built: training accuracy back to zero).
A no-training diagnostic (`agreement.py`) measured, at snapshots of the baseline, how much the 24 families' gradients
disagree at each weight, relative to the noise within one family:

| baseline at step | weights the families agree on (c = 3) | at c = 10 | at c = 30 | median disagreement ratio | what they agree on |
|---|---|---|---|---|---|
| 0 (random) | 89% | 100% | 100% | 1.2 | everything (nothing learned yet) |
| 300 | 0.2% | 3% | 30% | 45 | the output bias and the final norm — the base rate |
| 550 | 0.1% | 4% | 53% | 29 | same |
| 1800 | 0.8% | 10% | 61% | 25 | same, plus a fifth of one MLP bias |

Once learning starts, the families' gradients disagree at essentially every weight by 25–45× the sampling noise. The
"multiply by a" and "add b" circuits the task shares are not visible as *weights the families agree on*; each family's
gradient is family-specific everywhere. The toy world's clean split into a shared block and a per-family block does
not exist in the weight coordinates of a real network — which is exactly the assumption the rule was built on. An
agreement test needs a basis in which shared and specific are separated, and the weight basis is not it.

**3. Noise families hurt the baseline here** (training digits 0.77 → 0.43; whole answers 0.18 → 0.003 at this budget),
so run 2 does contain the effect run 1 lacked — but the inner objective cannot claim immunity, since it learned nothing
to protect. The rule did distrust the noise families (trust 0.12 vs 0.85 for real ones): that half of the design works.

The pre-registered verdicts: P1 (faster on new families) inconclusive — both at zero on whole answers; the graded
version P1b "passes" (3.69 vs 7.38 bits) but only because guessing beats confident recitation, not because anything was
learned; P5 (no harm on training) fails. 

A last fair shot at the rule, run as a follow-up (`runs_affine_q/`, 9 min, same baseline arms): replace the fixed
threshold c=3 by "keep the half of the weights the families disagree on *least*" (the data-driven threshold from the
toy's rule list), and apply the mask to the update rather than to what Adam sees, so Adam's statistics stay healthy.
It learns for about 200 steps after the switch and then plateaus: loss 1.62 for the rest of training against the
baseline's 0.97; training digits 0.43; new families 4.69 bits per digit, 8.8% of digits right. Presumably the decay on the
disagreeing half cancels what the agreeing half learns — in any case a plateau, not a phase change. Same hollow pattern: less
confidently wrong on new families only because less was learned on the old ones.

**Inside the weights** (`runs_affine/results/`): the baseline's weight clusters line up with the *offset* b
(b=4, b=1, b=6, b=10 at selectivity ≥ 0.9), never with the multiplier a — consistent with "recognise the table"
rather than "compute". But removing a cluster changes its own family's
accuracy by no more than removing a random cluster of the same size (0.052 vs 0.052), so these are statistical
regions, not functional ones. No brain regions this time either.

## 4. ZipLearn — learning the way a ZIP file does (seconds, no gradients)

A ZIP file learns without gradients. It counts what recurs and builds a dictionary; PNG goes one step further and, for
each block, tries a few generic *filters* ("this pixel minus the one to its left") and keeps whichever makes the block
smallest. Learning is counting; choosing a model is "which filter compresses this best"; predicting is decompressing.
`ziplearn.py` is that recipe applied to task learning, and the critical test was the one set for it: from a few
demonstrations of an unseen arithmetic rule, does it predict unseen inputs *exactly*, and how many demonstrations does
it need?

How it works (about 100 lines):
- a **dictionary**: per task, counts of "input digit → output digit" (the pure ZIP move);
- two **generic filters** borrowed from compression: the first and second *difference* of the output as the input
  steps up. If the first difference is constant, the whole table is a straight line and follows from 2 observations;
  if the second is, from 3. The learner is never told arithmetic exists — only that "differences may be constant" is a
  filter worth trying;
- **two-part-code selection**: each hypothesis pays for the demonstrations in bits (a parameter costs log₂ 11 bits; a
  prediction it got right costs almost nothing; one it got wrong pays an escape). The cheapest description wins and
  decodes the query.

Results (`runs/ziplearn_affine.json`, `runs/ziplearn_compose.json`):

| | exact after 1 demo | after 2 | after 8 | its own code length (bits/digit; max 3.46) |
|---|---|---|---|---|
| arithmetic, trained rules | 1.00 | 1.00 | 1.00 | 0.19 |
| arithmetic, **held-out rules** | 1.00 | 1.00 | 1.00 | 0.19 |
| arithmetic, random-noise rules | 0.00 | 0.00 | 0.00 | 3.86 — it reports it has learned nothing |
| held-out, dictionary only (no filters) | 0.02 | 0.17 | 0.96 | — memorises digits it has seen, cannot extrapolate |
| composition task (run 1's), trained / held-out | 0.18 / 0.00 | | 0.18 / 0.01 | 1.96 / 2.23 |

- It passes the critical test: **one demonstration** (six input→output digits) and every held-out rule — succession,
  addition, subtraction, multiplication, division mod 11 — is predicted exactly on unseen inputs. It never needed the
  training rules at all: the filters *are* the transferable structure. Said honestly: a "constant first difference"
  filter *is* a straight line, and every affine map is one, so this pass is largely built into the filter I chose.
  What is not built in, and is the informative part: the bit prices pick order 1 over order 2 on their own (Occam,
  no tuning), noise is rejected without a curriculum, and the dictionary-only ablation below shows that a ZIP
  *without* filters — the thing a ZIP actually is — cannot extrapolate at all.
- It has a built-in noise detector for free: on random rules no filter helps, the dictionary stays at maximum entropy,
  and the code length (3.86 bits/digit, above the 3.46 of guessing) says so. A network needs a separate mechanism
  (our F-test trust) to notice that; here it is the same number that does the learning.
- The dictionary alone (a ZIP with no filters) shows what the filters buy: it must *see* every input digit before it
  can answer it, so it takes 8 demonstrations to reach 96%.
- The boundary is just as clear: on run 1's composition task it fails (18% on trained pairs — exactly the pairs that
  happen to be digit-wise, like +1 then negate — and 0% held out), because reverse/rotate/swap move digits *between
  positions* and the filter set has no positional filter. **The filter set is the inductive bias.** A ZipLearn that
  could also propose "position i came from position j" filters would pass composition the same way; a network has to
  invent that from data, and at this size it did not.

## 5. Answers to the questions raised along the way

**"How does backprop itself need to change?"** It doesn't. Backprop only computes, for the batch in front of it, which
direction makes each weight less wrong; that is a measurement, not a decision. The decision is what you *do* with the
per-family directions, and that is where the inner objective lives. Everything in these runs kept backprop untouched and
changed only the combination step (average them; or keep what agrees and shrink what does not). What the runs add is a
constraint on any such rule: **agreement needs something to agree about.** At a random start no two families share any
structure yet, so a rule that only permits agreed changes permits nothing (run 1, arm B: training accuracy 0.3%). The
shared structure has to be *built* by the greedy rule first and then *protected* by the agreement rule — which is what
the two-phase rule in run 2 does, and it is also the most brain-like reading: fast, greedy learning first,
consolidation of what generalises afterwards.

**"Two training objectives at once — does that explain poor data efficiency and forgetting?"** The no-model check says
the mechanism is real: plain averaging leaves 84% of the memorisation error in place, the agreement rule 1%, and that
memorised part is exactly what a new family pays for. In the real network the picture is sharper than the toy
predicted: the network trained the ordinary way is not merely under-generalising, it is *anti*-generalising — 7.4 bits
per digit on new arithmetic families, twice the cost of guessing — because the strategy that best served "be less
wrong next time on these 24 families" is to recognise the family and recite its table. That part of the hypothesis is
confirmed and measured. The proposed remedy is refuted at this scale in three forms (from step 0; after a warm-up;
with a data-driven threshold on the update), and the diagnostic says why: the shared and the family-specific parts of
what the network learns are not separable weight by weight, so no weight-wise agreement test can tell them apart.
Forgetting was not tested; it is the natural next 20-minute run (below).

**Data efficiency, measured head to head.** On the same held-out arithmetic rules, ZipLearn is exact after **one**
demonstration; the network after eight demonstrations is at 8.3% of digits right on new families (guessing: 9.1%) with
1.1 bits per digit on the families it was trained on. The difference is not cleverness, it is what each one is allowed
to assume: ZipLearn is told that "differences may be constant" is a hypothesis worth trying (a two-line prior that
happens to be arithmetic's true shape), while the network must discover that from 24 families. That is the general
lesson of all three experiments: **generalisation is decided by what the learner is allowed to hypothesise, and a
training objective — outer or inner — can only pick among the hypotheses the architecture can express.** An inner
objective that rewards agreement across families is a way of *preferring* the shared hypothesis; it cannot conjure one
the network has no way to represent.

## 6. What to run next (each ≤ 20 min)

1. **Forgetting (the untested half of the hypothesis).** Train on 12 arithmetic families, then on the other 12, and
   measure how much of the first 12 survives under the baseline vs the two-phase agreement rule. Same code, one flag
   (`--split`); the prediction is that shrinking un-agreed weights *hurts* retention unless the agreement rule also
   protects the first phase's weights — which would be the next repair, and the interesting one.
2. **Give the network the filters ZipLearn had.** The three-way result says the winner is the hypothesis space, not
   the optimiser. A 20-minute test of that: add a "difference of neighbouring outputs" input feature to the same
   network and see whether held-out arithmetic goes from 8% of digits to near-exact — if it does, the prior is worth more
   than any inner objective at this scale, and the honest next question is how a network could *propose* such
   filters itself.
3. **ZipLearn with positional filters** (a "digit i came from digit j" hypothesis, chosen by the same shortest-
   description rule) to see whether the compression recipe covers run 1's composition task too — cheap, CPU, seconds.
