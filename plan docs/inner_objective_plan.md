# Plan: an inner objective for how the weights change

*Written 2026-09-20. Every experiment here runs in 20 minutes or less; every claim has a falsifier; the model is small
enough to read.*

> **Executed 2026-09-20 — results in plain English: `experiments/inner_objective/SUMMARY.md`.** Run 1 (compose, `runs/`), run 2 (modular arithmetic, two-phase rule, `runs_affine/`), the rank-threshold follow-up (`runs_affine_q/`), the agreement diagnostic (`agreement.py`) and ZipLearn (`ziplearn.py`). Premise confirmed (the baseline anti-generalises: 7.4 bits/digit on new families vs 3.5 for guessing); the weight-wise agreement rule refuted in three forms, because families' gradients disagree at every weight once learning starts.

## 0. The idea, in plain English

When a network learns, an optimiser decides how to nudge every weight after each batch of examples. The usual rule is:
nudge each weight in whatever direction makes *this batch* less wrong. That rule has its own hidden goal — explain the
batch in front of you — and you get what you optimise for: the network absorbs anything that explains the batch,
including quirks of the current task family and pure noise.

The proposal is to give the optimiser a second, **inner** objective that judges the nudge itself: *does this change help
across many task families, or only the one in front of us?* A weight change that helps everywhere is structure; one that
helps one family is memorisation; one that helps nowhere consistently is noise. The outer goal (do the task) is
unchanged. Only the rule for changing the weights is different.

## 1. The objective-evaluation script — done, and it already changed the design

`experiments/inner_objective/verify_objective.py` tests candidate inner objectives with **no model, no training, no
inference**, in about ten seconds. It builds a linear world where the answer is known: every task family shares an
*invariant* block of weights, has its own random *spurious* block (a regularity that is real inside that family and
useless outside it), and an *irrelevant* block. Per-family gradients come with finite-sample noise, and optional *noise
families* (the "noisy TV") whose gradients are pure noise. A candidate rule is scored on **transfer error** — what a
fresh family would cost beyond its unavoidable floor — after an idealised learner iterates the rule, swept over the
number of families, the gradient noise, the presence of noise families, and the starting point (fresh vs pretrained).

What it found, in order — each step a dead end that would have cost a GPU run to discover:

1. **Sign agreement across families** (the AND-mask family of rules) does not stop memorisation from a pretrained start:
   every family agrees on the *direction* toward its own spurious target when the weights are far from all of them,
   so the gate waves the drift through and only freezes it at the centroid — which is memorisation.
2. **Variance across families** is the right discriminator (zero on shared structure, large on family-specific
   structure, wherever the weights currently are) — but a *soft* gate leaks: any non-zero rate on a spurious coordinate
   converges to the memorised centroid given enough steps.
3. **Noise families break a variance gate** by inflating the between-family variance in exactly the coordinates carrying
   the most signal. Weighting families by agreement with the consensus does not fix it: once the shared structure is
   learned, every remaining gradient is family-specific, agreement collapses for real families too, and the trust
   weights turn arbitrary.
4. **Judging a family by agreement with itself** — its gradient reproduced across two halves of its own batch — does not
   degenerate. A noise family fails it at every stage of training; a real family passes it even when all that remains
   to learn is its own quirk. This is the epiplexity criterion (learnable = reproducible) at the level of gradients.
5. **Freezing a family-specific weight is wrong from a pretrained start** (it leaves a large random value in place);
   the right action is to decay it toward zero. *If the families cannot agree what a weight should be, it should be
   nothing.*

The survivor, **`ftest+wd`**: two half-batches per family; a family is trusted iff its two half-gradients agree with
each other; a coordinate is kept, *hard*, only if its variance across trusted families is at most `c` times its variance
within a family (the F-test); kept coordinates take the trusted mean gradient, masked ones decay toward zero. Transfer
error after 300 steps, baseline `mean` vs `ftest+wd`, and how much adding four noise families hurts each:

| families | grad noise | fresh start: mean → ftest+wd | noise-family multiplier | pretrained start: mean → ftest+wd |
|---|---|---|---|---|
| 4  | 0.3 | 3.79 → **0.89** | ×1.1 / ×1.2 | 3.79 → **0.91** |
| 8  | 0.3 | 1.98 → **0.05** | ×1.0 / ×1.3 | 1.98 → **0.08** |
| 16 | 0.3 | 0.84 → **0.01** | ×1.0 / ×0.9 | 0.84 → **0.04** |
| 16 | 1.0 | 1.05 → 1.00 | ×1.0 / ×0.9 | 1.05 → 1.00 |

Three requirements the model experiment inherits from this table, and the predictions it inherits, are in §6. The
script stays in the loop: any change to the rule is run through it first, and any result in the model experiment that
contradicts it is a finding about non-linearity, which is exactly what the model experiment is for.

## 2. The model — small enough to read

`experiments/transformers/h1_lid.py::Model` at **d = 64, 3 layers, 4 heads (head dim 16), rope positions: ~148k
parameters**. Not a new architecture; the one the whole transformer line already uses, shrunk. Full weight snapshots
cost 0.6 MB each, so the entire training trajectory (every 25 steps, ~200 snapshots) is 120 MB and fits in memory for
analysis. Every attention head, every MLP neuron and every weight can be inspected individually.

Fallback if the baseline cannot learn the task at this size (§5, gate C0): d = 96 (~330k). Not larger.

## 3. The task — the one this repo already scores

`h1_lid`'s composition domain: 7 primitive operations on 6-digit sequences (reverse, rotate left/right, swap pairs,
swap halves, +1, negate); a task is an ordered composition of two; 25 distinct compositions, **17 for training, 8 held
out**; presented as in-context (input, output) demonstrations; the dependent measure is **trials-to-criterion** on a
held-out composition — how many demonstrations before the model solves the query — which is the bee measure and the
one ARC scores.

**Environments (task families) = the 17 training compositions.** Structure is shared across them exactly where the
script's world says it is: the primitives are the invariant block (every primitive appears in several compositions),
each composition's particular pairing is its own regularity, and the 8 held-out compositions are "fresh families".

**Noise environments:** 3 additional "compositions" whose outputs are a keyed hash of the input (the same construction
as the MDL plan's noisy families). Learnable by nothing. Present in the A′/B′ arms only.

## 4. The inner objective, as implemented

At each step, per environment `e`, two half-batches of 8 sequences give two gradients `g_e¹`, `g_e²` (one backward pass
each: 17 × 2 = 34 backward passes of a 148k-parameter model on 8 sequences — cheap; `torch.func` per-sample gradients as
the faster alternative if needed). Then:

1. **Trust:** `t_e = cos(g_e¹, g_e²)`; environments with `t_e < 0.2` are dropped this step.
2. **Mask:** per coordinate, `across = Var_e(ḡ_e)` over trusted environments, `within = ¼·mean_e (g_e¹ − g_e²)²`;
   keep the coordinate iff `across ≤ c · within`, with `c = 3` (the script's value; swept once in a pilot if time
   allows, never tuned on held-out compositions).
3. **Update:** the kept coordinates receive the trusted-mean gradient and go to AdamW exactly as the baseline's gradient
   would; masked coordinates receive zero gradient and an extra decay `λ·w` (λ = 0.1 × learning rate, the script's
   ratio). AdamW's own hyper-parameters are identical across arms.

Nothing in the outer loss changes. The only difference between arms is which gradient AdamW is handed.

## 5. Arms and the 20-minute budget

| arm | environments | update rule |
|---|---|---|
| A | 17 compositions | AdamW on the mean gradient (the baseline) |
| B | 17 compositions | AdamW on the `ftest+wd` gradient |
| A′ | 17 + 3 noise | as A |
| B′ | 17 + 3 noise | as B |

Same model, same init seed, same data stream, same number of steps, same batch (17 envs × 16 = 272 sequences; the
baseline sees the identical 272 sequences and simply averages). One seed for the run; the second seed is the first
thing to add if the 20 minutes have slack after calibration.

**Time:** the smoke measurement from the MDL line puts a 4M-parameter model at ~350k tokens/s on this GPU; at 148k
parameters and 272 short sequences the bottleneck is the 34 backward passes per step, estimated at ~40 ms → 2,000 steps
≈ 80 s of training per arm plus evaluation. Budget: 4 arms × 3 min = 12 min, analysis 3 min, margin 5 min. A 1-minute
calibration run of arm B sets the step count before the four arms are launched; the calibration is discarded.

**Gate C0 (before anything else):** arm A must reach the trials-to-criterion regime (held-out compositions solvable
with ≤ 6 demonstrations for at least half of them) within the budget. If not, d = 96; if still not, stop and report —
the task is too hard for a model this size and the experiment needs a different task, not more compute.

## 6. Predictions — stated before running, with what would falsify each

From the script, the model experiment is in the winning regime only if three things hold, and each is checked by the
inner objective's own telemetry rather than assumed:

- **R1 ≥ 8 environments.** 17. ✓
- **R2 gradient noise below the signal.** Diagnostic: the fraction of coordinates the mask keeps. If B keeps > 95% of
  coordinates throughout, the per-environment batch is too small and the experiment is in the σ = 1 row of the table,
  where nothing can be learned about the objective; the remedy is a larger per-environment batch, not a conclusion.
- **R3 noise environments detectable.** Diagnostic: their trust `t_e` should fall below 0.2 within the first 50 steps
  and stay there.

Then:

| id | prediction | falsifier |
|---|---|---|
| P1 competence | B reaches criterion on held-out compositions in fewer demonstrations than A at matched training accuracy | no difference at matched training accuracy → the linear-world gain does not survive a non-linear model |
| P2 noise immunity | A′ is worse than A on held-out; B′ ≈ B | B′ degrades as much as A′ → self-agreement trust fails in a real model |
| P3 what the mask does | a substantial share of coordinates (> 20%) is masked at some point, and those weights end near zero: B is a sparser network than A at the same accuracy | mask never closes, or masked weights do not shrink |
| P4 organisation | under B, parameters cluster into regions whose environment-participation profiles align with single primitives, more cleanly than under A (§7 scores) | equal modularity → the objective changes performance without changing organisation, which is itself worth knowing |
| P5 no harm on training | B's training accuracy is not more than 5 points below A's at the end | B is slower on training *and* no better held-out → the rule is a learning-rate cut wearing a principle |

Interpretation rules, fixed now: P1 without P4 means the inner objective helps without producing readable structure;
P4 without P1 means it reorganises the network without helping — a result about interpretability, not competence;
P2 failing with P1 passing means the mechanism works but its noise defence does not, and the script's model of noise
families is wrong for real networks.

## 7. Reading the weights — the toolkit, and what "a brain region" means here

Everything below is computed from three things the training loop records: **weight snapshots** every 25 steps, the
**per-environment gradients** (the inner objective already computes them), and the **mask and trust telemetry**.

**The participation matrix.** For every parameter θ and environment e, `P[θ, e]` = the correlation across training
steps between e's gradient on θ and the update θ actually received. High means e drives that parameter. This is 148k
rows × 17 (or 20) columns, and it is the object everything else reads.

**Regions.** Cluster the rows of P (k-means, k = 7 primitives + 2, run at several k). A cluster is a candidate region.
For each region and each primitive p, score how well the region's participation profile matches "environments that
contain p" (AUC over environments). A region's **selectivity** is its best such score; a **primitive region** is one
with selectivity above 0.9. Report: the number of primitive regions found, their sizes, and which layers/modules they
live in — under A and under B. P4 is "B finds more primitive regions with higher selectivity."

**Function, not just correlation.** Ablate (zero) each region and measure held-out accuracy per primitive. A region is
functionally selective if ablating it damages one primitive far more than the others. Report the ablation map
(regions × primitives) for A and B. Correlation-defined regions that are not functionally selective are reported as
such; the two definitions must agree for the word "region" to be used in the write-up.

**Wiring.** Coarse and pre-registered, because this is where over-interpretation lives: (i) which layer and module each
region occupies; (ii) for attention heads, which earlier region's write-directions (columns of its output projection)
overlap the head's query/key read-directions — a "reads from" graph between regions; (iii) co-update timing — the
cross-correlation over training of region-level update magnitudes, i.e. which regions form together. The output is a
small labelled graph per arm. No claim about wiring is made unless the ablation map supports it.

**Over time.** The same analyses on the snapshot sequence: when each region forms (the step at which its selectivity
crosses 0.9), whether regions form in the order primitives are needed, and whether under B the masked coordinates are
the ones outside every region.

All of it is numpy/torch on stored tensors; the 3-minute analysis budget in §5 covers it at this model size.

## 8. Plain-English reporting — the shape of the final write-up

One paragraph on what the two optimisers do, in the "nudge" language of §0. One table with four rows (the arms) and
three numbers: how many demonstrations to solve a new composition, accuracy on new compositions, and how much the
noise families hurt. One picture per arm of the participation matrix with regions outlined and labelled by primitive.
Then each prediction from §6, stated as "we said X would happen; it did / did not; here is the number." Then one
paragraph of best guess about why, marked as a guess. Nothing else.

## 9. What this does not test, and what comes next (each also ≤ 20 minutes)

- **The learned inner objective** — your original framing: instead of a fixed F-test rule, a small parametric gate
  trained by meta-gradient on held-out-composition acquisition. Experiment 2, after this one establishes that a fixed
  rule does anything at all, because a learned rule is only interpretable against a fixed one that works.
- **Scale and non-toy domains.** Nothing here says what happens at 4M parameters or on ARC frames. It is designed not
  to.
- **The MDL plan's question.** Abandoned; not revisited.

## 10. Files

- `experiments/inner_objective/verify_objective.py` — §1, exists, 10 s. Output `verify_objective.json`.
- `experiments/inner_objective/train.py` — arms A/B/A′/B′, the inner objective, telemetry, snapshots. To write.
- `experiments/inner_objective/inspect_weights.py` — §7. To write.
- `experiments/inner_objective/run.py` — calibration, the four arms, analysis, the §8 report, all inside 20 minutes.
  To write.
