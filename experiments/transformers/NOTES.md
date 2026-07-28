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

### H1 round 2 — smaller task, and PoPE added (2026-07-27)

**Run budget for this line raised to 5 minutes** (user, 2026-07-27); the 2-minute law still governs `src/tbt/`.

**Task shrunk BY COUNTING, not by taste.** A first shrink to 6 primitives at L=4 collapsed 36 writings to **14 distinct
functions** — rotations form a small cyclic group and most compositions coincided — which left **6 training tasks against
8 held-out**, i.e. the split inverted, and then crashed with a KeyError because some primitives appeared only in held-out
tasks. Counting distinct compositions across candidate configs picked **7 primitives at L=6, V=5 → 25 distinct** (17 train
/ 8 test). *Distinct-function count, not primitive count, is the quantity the design needs.* Two other real defects fixed
on the way: `double` (x·2 mod V) was never a bijection at even V despite the docstring saying so — replaced by `negate`;
and the weight vector was hardcoded to 8 entries, so when the primitive set shrank it silently used only the first 6 and
halved the exposure spread the whole experiment depends on.

**PoPE implemented** (arXiv:2509.10534, ICML 2026), which needed custom attention — position has to reach the QK product,
and `nn.TransformerEncoderLayer` gives no way in. RoPE's score is `Σ_c μ_q μ_k cos((s−t)θ_c + φ_k − φ_q)`, where the
content-dependent phases interact, entangling what and where; PoPE puts content in the MAGNITUDE (`softplus(q)`, so
non-negative) and position in the PHASE ALONE (`t·θ_c`, no content term), giving `Σ_c μ_q μ_k cos((s−t)θ_c + δ_c)` with no
interaction term, computed as an ordinary dot product in 2·hd dims. `δ_c` learnable, init U(−2π,0), clipped there.

| positional scheme | steps | train loss | TRAINED solved | mean acc | held-out solved |
|---|---|---|---|---|---|
| learned absolute | 4000 | 0.165 | 3/17 | 0.31 | 0/8 |
| RoPE | 3200 | 0.058 | **9/17** | **0.54** | 0/8 |
| PoPE | 3200 | 0.076 | 7/17 | 0.46 | 0/8 |

1. **Positional encoding matters a lot here.** Both rotary schemes roughly triple the trained-task solve rate against
   learned-absolute — and do it with 20% FEWER steps, so the comparison understates them. Worth knowing before any
   conclusion about LID is drawn on this substrate.
2. **PoPE did not beat RoPE in our run** (7/17 vs 9/17), on one seed at a scale far below the paper's 124M–774M. Not
   evidence against the paper; just no reproduction of the advantage here.
3. **H1 IS STILL NOT ANSWERED.** Every held-out task is censored in every condition, so the primary measure — trials-to-
   criterion — carries no information in any of the three runs.
4. ⚠ **AND THE SECONDARY MEASURE IS NOT STABLE.** It came out **−0.886, −0.441, −0.273** across three otherwise-comparable
   runs. The first of those looks like strong support for LID (n=8, p≈0.003 taken alone) and it would have been easy to
   report it as the headline. Across conditions it is clearly noise-dominated at n=8. **No LID signal is established**, and
   the honest statement is that this design cannot yet measure the thing it was built to measure.

**What it would take.** The blocker is unchanged and now well characterised: the model never solves a held-out composition,
so acquisition speed on novel tasks cannot be timed. Either the model must get strong enough to compose (more scale/steps
than a 5-minute budget buys — a GPU job), or the dependent measure must move to something graded and stable with far more
than 8 held-out tasks. Raising the primitive count raises both the task supply and the difficulty, so those trade against
each other, and that trade is the next thing to design around rather than tune.

### Compute: ~3.7x, and it was not where I first guessed (2026-07-27)

**Question: how to greatly reduce the compute to train the same model. Answer: profile first.** My first guess — that the
CPU-side data pipeline was the cost — was wrong, and the fix based on it bought only ~23%. Profiling settled it:

    data generation 3.45 ms/step | forward only 18.40 | forward+backward+step 54.95   => the MODEL is 94% of the step

55 ms for a 340k-parameter model is ~5x off what the card should manage: many tiny kernels in fp32, i.e. launch-bound
rather than FLOP-bound. (The tell was already in earlier runs: sample throughput went from 1,344/s at batch 64 to 2,611/s
at batch 384. Throughput that rises with batch size means fixed per-step overhead.) Measured on the 3050 Ti, batch 256:

| | ms/step | speedup |
|---|---|---|
| baseline fp32 | 54.6 | — |
| bf16 autocast | 32.5 | 1.7x |
| `torch.compile` fp32 | 41.5 | 1.3x |
| **`torch.compile` + bf16** | **17.6** | **3.1x** |

Plus fused SDPA attention (the hand-rolled version materialised a T×T score matrix and allocated a fresh causal mask per
block per step), cached frequency buffers, GPU-side batch generation, and a fused AdamW: **~3.7x overall**, spent on steps
rather than saved. 3200 steps in 248 s became **11,000 steps in 266 s**.

**And that changed the result qualitatively.** With the extra training the model finally MASTERS its training
distribution — **14/17 trained compositions solved, mean accuracy 0.86** — so the sanity gate passes for the first time.
But held-out compositions of the *same primitives* stay at **0/8**, accuracies 0.00–0.37. That is no longer "the model is
too weak": **compositional transfer is essentially absent in a model that has demonstrably learned the parts.**

⚠ **And the LID correlation is not reproducible.** Two runs at IDENTICAL config and seed gave secondary Spearman
**−0.063** and **−0.420** (bf16 + autotune are not bit-deterministic). Combined with −0.886/−0.441/−0.273 across the
earlier conditions, the measure is plainly noise-dominated at n=8. **H1 remains unanswered — but now for a better reason:**
not "the model is too weak to measure" but "the effect, if any, is smaller than run-to-run noise". The binding constraint
is now the number of held-out tasks, and that trades directly against difficulty (more primitives ⇒ more tasks AND harder).

**Where Aurora fits** (<https://github.com/tilde-research/aurora-release>, arXiv 2606.27715): it is a **Muon variant** — a
better orthogonalised update for 2D weight matrices, projecting onto the intersection of the row-oblique and Stiefel
manifolds for more balanced per-row updates. That is a **sample-efficiency** lever (fewer steps to a given loss), which is
orthogonal to the throughput work above and therefore *stacks* with it. So: a good SECOND start, not a first — the
first-order win was ours to take and is now taken. **The caveat to measure before adopting it:** Newton–Schulz
orthogonalisation adds per-step work, our 2D matrices are tiny (96×288, 96×96, 96×384), and we are overhead-bound — so the
step-count saving has to beat the added per-step cost, which at this scale is not obvious. Muon-family gains are usually
reported at nanoGPT scale and up. Cheap to test now that the harness is fast.

