# Transformers — sample-efficient skill acquisition

*Split out of `experiments/NOTES.md` on 2026-07-27. The line's question is stated in `BEE.md`; results accumulate here.*
## Transformer × linear regression (2026-07-27) — `linreg.py`

**Question asked:** can a transformer, trained with AdamW + LR scheduling, learn a linear regression?
**Answer: yes — and the scheduling is not what makes it work.** ~35 s per arm on the 3050 Ti; `python
experiments/transformers/linreg.py`.

The question has two readings and they are different experiments, so both are run. MEMORISE = one `w` for the whole
dataset (fit a single linear map; any model does this). IN-CONTEXT = a FRESH `w` per sequence, never shown — the prompt is
`x1 y1 x2 y2 … xk` and `w` cannot be memorised, so the only way to succeed is to run the regression IN THE FORWARD PASS
(Garg et al. 2022, arXiv:2208.01066). Setup: d=8, k=20, 4 layers × 64 dim, 3000 steps, batch 64, noiseless.

**The measurement is the loss against k, not a final number** — a model that learned the task shows loss FALLING as
evidence accumulates; a model that learned only the marginal shows a FLAT curve at Var(y). A single loss cannot tell those
apart. Baselines: predict-zero (normalised MSE 1.0 by construction) and RIDGE, which is Bayes-optimal under this exact
prior (posterior mean = min-norm least squares), so it is a floor and not a competitor.

| k in context | 0 | 1 | 2 | 4 | 7 | 8 | 10 | 19 |
|---|---|---|---|---|---|---|---|---|
| transformer (in-context) | 1.004 | 0.879 | 0.742 | 0.502 | 0.226 | 0.192 | 0.116 | 0.046 |
| ridge = Bayes-optimal | 1.003 | 0.874 | 0.734 | 0.485 | 0.126 | 0.000 | 0.000 | 0.000 |
| predict-zero | 1.003 | 0.996 | 0.935 | 0.942 | 0.945 | 0.957 | 1.034 | 0.938 |

1. **The control passes trivially:** MEMORISE reaches 0.000 at every k, so the optimiser and schedule are sound and nothing
   below is an artefact of a broken training loop.
2. **In-context regression is genuinely learned**, and it is close to OPTIMAL while the problem is statistical: at k=1,2,4
   the transformer is within 0.005–0.02 of the Bayes floor. It is not approximating regression loosely; it is nearly
   saturating the information in the prompt.
3. **The gap opens exactly where the problem stops being statistical.** At k ≥ d=8 the system is determined and ridge is
   EXACT (0.000) while the transformer only decays toward it (0.046 at k=19). It behaves like a good estimator, not like a
   linear solver — which is the interesting part, and the obvious thing to push on next.
4. **THE SCHEDULE IS NOT LOAD-BEARING.** Cosine-with-warmup vs constant LR, three seeds: constant won two of three and the
   differences sit inside seed noise (e.g. k=10: cosine 0.116/0.120/0.118 against constant 0.137/0.087/0.087). The honest
   reading of the original question is that AdamW learns this with or without a schedule at this scale. A single seed would
   have "shown" cosine winning — the first run did.

**Obvious next probes** (cheap, same script): does the k ≥ d gap close with more steps/width, or is it structural? does it
survive label noise (where ridge with the right λ is optimal and OLS is not)? does it extrapolate to k beyond the training
length, and to a d it was never trained on?

---

## H1 — is LID the right explanatory variable? (2026-07-27) — `h1_lid.py`

**Status: apparatus BUILT and validated, question NOT YET ANSWERED.** Recorded at this state deliberately rather than tuned
until something looked significant.

**Design.** Sequences of L=4 digits (V=6); eight primitives, each a bijection (reverse, rot left/right, swap pairs/halves,
+1, -1, x2); a task is an ordered composition of TWO primitives shown as K=6 in-context (input, output) demonstrations.
The one design decision that makes this a test of LID rather than of distribution shift: **global novelty is held
CONSTANT** — every held-out task is a composition that was never trained, so all are equally novel as wholes, and the only
thing varying is how familiar their PARTS are. Parts are made graded by training the primitives at frequencies spanning
~150x. Functional duplicates and no-op compositions are removed by signature, so a "held-out" task cannot secretly be a
trained one under another name.

**What works.** Local familiarity comes out cleanly graded and tracks training weight almost monotonically —
`swap_halves` 0.199 error at weight 1.0 through to `dec` 2.691 at weight 0.007. That is the independent variable behaving
exactly as the design needs.

**TWO BUGS, both caught by the apparatus rather than by inspection, and both worth keeping:**
1. **The Spearman implementation did not handle TIES**, and the first run reported **rho = +1.000** — a perfect
   correlation — on data where every single task was censored at the same value and therefore contained no information at
   all. A false positive manufactured entirely by the metric. Fixed with average ranks.
2. **There was no sanity gate.** With the model solving 0/27 trained compositions, the held-out table was meaningless, and
   nothing in the output said so. There is now a gate that checks the model can do the tasks it WAS trained on and prints
   that the measurement is uninterpretable when it cannot.

**Where it stands** (2800 steps, 84 s, train loss 0.137): the model solves **8/21 trained** compositions to criterion
(mean final acc 0.48) but **0/8 held-out** — every held-out task is censored, so the primary measure (trials-to-criterion)
still carries no information. The graded secondary measure gives Spearman(worst local error, final accuracy) = **-0.241**,
weakly in the LID-predicted direction, on n=8. That is a trend, not a result, and it is reported as secondary *because the
primary was censored* — not selected after seeing which looked better.

**The early signal that matters more than the correlation:** the model learns its training compositions (0.48) and
transfers to new pairs of the SAME primitives essentially not at all (0.01–0.19). Composition is not free here. If that
survives a longer run it is directly relevant to H2/H3 — a base model with no compositional generalisation of its own is
precisely the situation where a harness would have to supply it, which is the claim under test.

**To actually answer H1:** the model needs to solve held-out compositions at all, which needs more training than a
2-minute local budget allows at this domain size. Either a longer run (a GPU job, not a laptop one) or a smaller task
(fewer primitives, more demonstrations). The apparatus is ready for either — `--steps`, `--seed`, `--k`, `--crit`.

