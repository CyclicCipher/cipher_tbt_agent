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

### Aurora (Muon variant) — real sample-efficiency gain, cancelled by its own cost at this scale (2026-07-27)

Vendored VERBATIM from <https://github.com/tilde-research/aurora-release> (MIT) into `aurora.py`, so the comparison is
against the authors' update rule rather than my reading of it. Aurora is a **Muon** variant: orthogonalise the momentum
before applying it, and for TALL matrices additionally balance the update across ROWS — iteratively approximating a
projection onto the intersection of the row-oblique and Stiefel manifolds, driving each row's squared norm toward `n/m`.
Square/wide matrices fall back to plain Muon. `polar` is CANS-12 (nine Chebyshev-optimised cubic Newton–Schulz iterations
then three classic), FP32.

Wired per Muon practice: the 2D HIDDEN weights get Aurora, embeddings/head/all 1D parameters stay on AdamW (which
`aurora()` also requires — it raises on non-2D). In this model the tall matrices are `qkv` (288×96) and the first MLP
layer (384×96), six in all, so the distinctive path is genuinely exercised. LRs are each side's default and the pairing
the repo itself uses (`ADAM_LR=1e-3`, Aurora `eta=0.05`), on one shared warmup+cosine schedule.

| optimizer | steps | wall | train loss | TRAINED solved | mean acc |
|---|---|---|---|---|---|
| AdamW | 3200 | 248 s* | 0.0580 | 9/17 | 0.54 |
| **Aurora** | **1000** | **52 s** | 0.0573 | **9/17** | 0.55 |
| AdamW | 11000 | 266 s | 0.0385 | **14/17** | **0.86** |
| Aurora | 5000 | 280 s | 0.0439 | 13/17 | 0.78 |

\* before the throughput work; step counts are still comparable since the model is unchanged.

1. **The sample-efficiency gain is real**: Aurora reaches 9/17 in **1000 steps** where AdamW needs **3200** — 3.2× fewer.
2. **It shrinks as training goes on**: at the higher quality level it is 5000 vs 11000, i.e. 2.2× fewer.
3. **And it is cancelled by its own per-step cost.** Aurora runs ~55 ms/step against AdamW's ~24 ms (2.3×), because
   CANS-12 is twelve fixed iterations per tall matrix however small the matrix is — and ours are 288×96, so the cost is
   launch-dominated and never amortises. **At matched wall clock (~270 s) the two are a wash**, AdamW marginally ahead
   (14/17 @ 0.86 vs 13/17 @ 0.78).

**Read this as a scale result, not a verdict on Aurora.** Muon-family gains are reported at nanoGPT scale and up, where
the matrices are large enough that orthogonalisation is cheap relative to the matmuls it improves. The prediction that
follows: Aurora should start winning here as `d_model` grows, and that is a one-line sweep worth running before dismissing
it. Defaults were used on both sides and nothing was tuned; tuning one arm only would have made the comparison worthless.

**THE PART THAT MATTERS MORE.** Aurora changes nothing about the actual failure: **0/8 held-out compositions, exactly as
with AdamW.** A better optimiser gets to the same place faster and stops at the same wall. That is evidence the
compositional failure is *structural rather than an optimisation problem* — which lines up with this repo's own earlier
result on a completely different architecture: `binding_rule.py` found that Mamba-3 learned to bind and recall rules but
broke on applying them compositionally (`NOTES.md`: "compositionally — the wall — is genuinely where vanilla breaks, even
though recall-binding worked"). An SSM and a transformer, different mixing mechanisms, same seam: **parts learned,
composition absent.** Whatever is missing is not a gradient-quality problem, and that is precisely the gap the harness
(H2/H3 in `BEE.md`) is supposed to fill.

**Also, the LID measure is now conclusively noise at n=8.** Secondary Spearman across all runs to date:
−0.886, −0.441, −0.273, −0.063, −0.420, **+0.174** — both signs. No signal.

### FlashAttention: measured, and NOT worth it at our shapes (2026-07-27)

Measured on the COMPILED model (what we actually train with), batch 256, seq 96, head_dim 24:

| | ms/step |
|---|---|
| rope, full | 17.32 |
| rope, attention short-circuited | 13.23 |
| learned positions (no rotation) | 17.10 |

So **all attention machinery is 23.6% of the step (4.1 ms)** and the **RoPE rotation is 1.3%** — `torch.compile` already
fused the rotation away, which is why the hand-written `stack`/`flatten` never showed up as a cost.

FlashAttention could only target part of that 4.1 ms, and four things say it would recover almost none of it:
1. **The attention math is not the cost.** At seq 96 / head_dim 24 the QKᵀ and AV matmuls are ~8 GFLOP fwd+bwd across the
   whole model — roughly 0.4 ms at this card's bf16 throughput. We spend 4.1 ms, i.e. ~10x off roofline, so what is left is
   memory traffic and launch overhead, not the softmax·V arithmetic FA optimises.
2. **FA's win scales as O(N²) memory traffic, and N=96.** The N×N matrix it avoids materialising is trivially small here;
   the whole point of FA is that this term dominates at long context, which is exactly the regime we are not in.
3. **head_dim = 24 is off the fast path.** FA kernels are specialised for 32/64/128; 24 wastes tensor-core tiles.
4. **FA3 is Hopper-only** (FP8, async warp specialisation, TMA). This box is Ampere (3050 Ti), so the newest variants
   contribute literally nothing.

**Verdict: a long-context optimisation, correctly guessed. Not our bottleneck.** The remaining 13.2 ms is linear layers +
MLP + optimiser. The one untried lever is `torch.compile(mode="reduce-overhead")` (CUDA graphs) — we used
`max-autotune-no-cudagraphs`, and since the profile says launch overhead rather than FLOPs, CUDA graphs are the thing most
likely to still pay. That is a one-line change worth measuring before any kernel work.

### Does it compose, or memorise? The task-diversity sweep (2026-07-27) — `diversity.py`

**H1 was retired first, and for a reason worth keeping.** It asked how FAST a novel task is acquired and how that tracks
the familiarity of its parts. The measured acquisition rate on held-out compositions is ZERO — every one, in every
optimiser and positional scheme tried. You cannot correlate a speed against anything when nothing is acquired. The
censoring was the finding, not an obstacle to it, and the question upstream is whether the model composes at all.

**The hypothesis has a known shape.** Raventós et al. 2023 (arXiv:2306.15063) found a TASK-DIVERSITY THRESHOLD in
in-context regression: below a critical number of distinct pretraining tasks a transformer behaves as if it memorised
them; above it, it implements the general algorithm and generalises to unseen tasks. That would explain both of our
results at once — `linreg.py` drew a fresh `w` per sequence (effectively INFINITE diversity) and learned the general
algorithm to within 0.005–0.02 of Bayes-optimal; `h1_lid.py` had 17 discrete tasks and memorised.

Design: 12 primitives give **62 distinct 2-compositions** (deduplicated by signature). The held-out set is **FIXED at 20**
across every condition, training sets are **NESTED**, sampling is uniform, compute is fixed at 8000 steps.

| \|train\| | train acc | train solved | HELD-OUT acc | held-out solved |
|---|---|---|---|---|
| 4 | 0.999 | 4/4 | 0.020 | **0/20** |
| 10 | 0.997 | 10/10 | 0.046 | **0/20** |
| 22 | 0.986 | 22/22 | 0.049 | **0/20** |
| 42 | 0.754 | 22/42 | 0.077 | **0/20** |

1. **Memorisation is complete and transfer is absent.** Up to 22 tasks the model fits EVERY training task (0.986–0.999,
   all solved) — so nothing here is an undertraining artefact — while held-out accuracy sits at 2–5% and **not one of the
   20 held-out compositions is solved at any diversity level**.
2. **A 10x increase in task diversity buys ~6 percentage points and zero solved tasks.** No threshold anywhere in range.
3. **The largest point is partly undertrained** (train 0.754, 22/42) at fixed compute, so it is the weakest row; note the
   trend across 4→22, where the fit IS complete, is equally flat.

⚠ **WHAT THIS CANNOT CONCLUDE, and it is the main limitation.** Raventós located the threshold at roughly 10³–10⁴ distinct
tasks. Our ENTIRE task universe is 62. So this cannot refute the diversity hypothesis — it can only say the threshold is
not at ≤42 tasks for this family, which is unsurprising if the real one is two orders of magnitude further out. The
contrast with `linreg` (unbounded diversity ⇒ general algorithm learned) remains *consistent* with diversity being the
operative variable; we simply cannot reach the regime from below with 2-compositions of 12 primitives.

**So the task space is now the binding constraint** — the same shape of blocker as before, one level up. Getting to 10³
tasks needs length-3 compositions (12 primitives ⇒ far more than 62) or many more primitives, and both make each task
harder to infer in-context, which is the trade that has to be designed around rather than tuned. That is the next build.

### The failure is in the READ-OUT, not the representation (2026-07-27) — `compositional_rep.py`

The behavioural numbers cannot distinguish two stories: (A) the model never built the primitives as reusable objects and
stored 42 atoms, so there is nothing for a harness to recombine and the gap is architectural; (B) the task CODE is
compositional and what fails is turning it into behaviour on an untried combination. **The representation can tell them
apart**, using the one formal measure ML has for this — Andreas 2019's Tree Reconstruction Error (arXiv:1902.07181), where
the primitives are INFERRED rather than read off, which is what makes it usable when the model only ever sees
2-compositions and no primitive alone.

Task code = the residual stream at the last demonstration's final input token (where the model must commit), averaged over
256 random inputs so the input content marginalises out. Fitted form `rep(a,b) ≈ α_a + β_b` — ordered, so order-sensitive,
and 24 role vectors for 42 training codes, so a good fit is compression rather than re-description. A per-primitive matrix
operator would be ~110k parameters for 4k numbers and would fit anything, so it is deliberately not used.

| | R² |
|---|---|
| TRAINING compositions (fit) | **+0.819** |
| **HELD-OUT compositions (the test)** | **+0.522** |
| held-out, SHUFFLED labels (control) | **−0.631** |

with behaviour at train acc 0.625, **held-out acc 0.038**.

**The model's representation of a composition it has NEVER been trained on is predictable at R² = 0.52 from role vectors
fitted only on training compositions — while the same fit with the labels shuffled scores −0.63.** The shuffled control is
what makes this real: shuffling preserves the codes' subspace geometry and destroys only the label correspondence, so the
structure is genuinely tied to which primitives the task is made of.

⇒ **Story (B). The primitives ARE separable and reusable, the model composes their codes correctly for pairs it has never
seen, and then fails to act on the result.** It can REPRESENT `a∘b` and cannot EXECUTE it — which is exactly the
estimate-versus-execute distinction: mixing is native to attention, function application is not. The probe separates them
empirically rather than by argument.

**This redirects the agenda, and mostly by subtraction:**
* **"Pressure to factorise" drops in priority.** It was premised on story (A). The model already factorises without any
  such pressure, so the interesting question is no longer how to make it decompose.
* **Execution becomes the prime suspect** — and with it the depth constraint, since composing *m* operations needs *m*
  sequential applications and a fixed-depth model cannot supply them. This is the one place a harness has a clear
  mechanical job: supplying iteration a bounded architecture cannot.

**Honest limits.** One seed, one extraction point, one composition form — the claim is "compositional structure OF THIS
FORM is present", not that the code is exactly additive. R²=0.52 is substantial but far from 1, so the code is PARTLY
compositional. And the model is undertrained at 6000 steps (train acc 0.625); the R² comparison is internally valid
because control and test come from the same model, but the absolute numbers would move. Worth repeating at 11000 steps and
across seeds before it carries much weight.

### MIXING vs APPLICATION — the architectural hypothesis FAILS (2026-07-27) — `apply_vs_mix.py`

I argued the missing primitive was APPLICATION: attention mixes value vectors into a residual stream and never applies one
as a function, which would explain why Mamba-3 failed identically (an SSM also mixes, via state). `compositional_rep.py`
seemed to support it — the code is compositional, the behaviour is not, so the model can represent `a∘b` and not execute
it. **The direct test says the hypothesis is wrong.**

One-variable design: both arms share the demonstration encoder, the input embedding and the decoder, and differ ONLY in
what happens to the task code `z` — `MIX: h = MLP([e(x), z])` against `APPLY: h = M(z)·e(x)` with `M(z) = Σ zₖ Bₖ`, the
`operator.py` move cross-pollinated from TBT. Nothing tells APPLY that tasks compose or that `M(z_{a∘b})` should be
`M(z_b)M(z_a)`.

| arm | params | train acc | train solved | HELD-OUT acc | held-out solved |
|---|---|---|---|---|---|
| MIX | 234k | 0.943 | 42/42 | 0.018 | **0/20** |
| APPLY (rank 10, d=96) | 235k | 0.424 | 2/42 | 0.005 | **0/20** |
| **APPLY (rank 40, d=64)** | 229k | **0.985** | **42/42** | 0.042 | **0/20** |
| transformer (from `diversity.py`) | ~340k | 0.754 | 22/42 | 0.077 | **0/20** |

**The first APPLY run was uninterpretable and is kept to show why**: at rank 10 it never fit the TRAINING set (0.424,
2/42), and held-out cannot be compared across arms that differ in train fit — the same sanity-gate failure as H1. Trading
width for basis rank at fixed parameters fixed it (0.985, 42/42), and only then is the comparison fair.

**With the comparison fair, application does not rescue composition.** APPLY generalises marginally better than MIX (0.042
vs 0.018) but neither solves a single held-out task, and the plain transformer's 0.077 is the HIGHEST of the three — which
also undercuts "attention is the wrong architecture" from the other direction.

⇒ **The failure is invariant across four architectures** — attention over a sequence, a state-space model
(`binding_rule.py`), MLP-with-concatenation, and MLP-with-operator — **and across optimisers** (AdamW vs Aurora) **and
across task diversity** (4 → 42 tasks). Architecture is not the lever, on this evidence, and neither is the optimiser, and
neither is data.

**Taken with `compositional_rep.py`, that leaves a sharp diagnosis.** The representation already factorises (held-out R²
+0.52), and giving the model an explicit apply-primitive changes nothing. So the missing thing is neither reusable parts
nor a mechanism to apply them. What all four share is that they are trained by gradient descent on a loss that is FULLY
SATISFIED by memorising the training tasks: on the training distribution, a model that composes and a model that memorises
are indistinguishable, so nothing in the objective ever prefers the former. The code lands in roughly the right place for
a held-out composition and the READ-OUT has simply never been trained to decode that region.

That reframes "pressure to factorise": the representation factorises on its own, and it is the read-out that has no reason
to. The next lever is therefore an OBJECTIVE that separates the two solutions, not another architecture — and note it
cannot be more data of the same kind, which is what the diversity sweep already ruled out.

### Chaining consistency — it works TRANSDUCTIVELY and does not generalise (2026-07-27) — `chaining.py`

The objective, designed before testing: ground the PARTS with supervision and constrain the WHOLES with a label-free
consistency term, `L = L_sup + λ·CE(M(D_ab, x), sg[M(D_b, M(D_a, x))])`. The argument was that ERM cannot separate a
composer from a memoriser because they are extensionally identical on the labelled tasks, so the loss must be enlarged to
a region where they differ — and a chained prediction is a MANUFACTURED TARGET on tasks that have no label.

Three-way split, built in because transduction was the named risk: 12 primitives + 17 compositions LABELLED, 17
compositions UNLABELLED (consistency term only), 17 compositions EVAL (never presented in any form). Dedup runs across
BOTH lengths, since `rot1∘rot1 = rot2` would otherwise let an "unseen" task be a labelled primitive under another name.

| arm | prims | labelled | UNLABELLED | solved | EVAL | solved |
|---|---|---|---|---|---|---|
| `lam0` (control) | 0.997 | 0.997 | 0.021 | 0/17 | 0.017 | 0/17 |
| **`full`** | 0.922 | 0.885 | **0.593** | **5/17** | **0.018** | **0/17** |
| `noprim` (falsifier) | 0.000 | 1.000 | 0.000 | 0/17 | 0.001 | 0/17 |
| `full`, no grounding warmup | 0.000 | 0.000 | 0.000 | 0/17 | 0.000 | 0/17 |

**COLLAPSE HAPPENED FIRST, and it was predicted.** At λ=1 with the consistency term on from step 0, the model went to
**0.000 on everything, primitives included**, against a working 0.997 control — the model distilling its own garbage, and
that attractor beating the supervised loss outright. This was failure mode 1 in the proposal and the proposed mitigation
("`L_sup` forbids collapse") was **not enough**. The fix is the mechanism story enforced in time: switch the term on only
after the primitives are learned, because at step 0 there are no learned parts to ground the chain.

**THE RESULT, both halves.** On the unlabelled pool the objective does exactly what the theory said: **0.021 → 0.593, and
5/17 solved where nothing in this entire line had solved a single held-out composition before.** It converts unlabelled
tasks into correct behaviour using only supervision on the parts. **And it does not transfer at all** — EVAL is 0.018
against the control's 0.017, i.e. unchanged, 0/17.

This is precisely the outcome named as a falsifier in advance: *"if UNLABELLED lifts but EVAL does not, that is
transduction, not composition, and it must be reported that way."* So it is reported that way. The objective is a
TRANSDUCTIVE fix, not an inductive one.

**And the failure has an exact form worth stating:** we removed memorisation by manufacturing labels, and the model
memorised the manufactured labels. It learned that *these seventeen* compositions have the chained answer; it did not
learn to chain. The original problem, one level up.

Cost noted: the term slightly degrades supervised accuracy (0.997 → 0.922 prims, 0.997 → 0.885 labelled).
Caveat on the falsifier: `noprim` also has fewer supervised tasks (17 vs 29), so its failure is ambiguous between an
ungrounded chain and thinner supervision. It could only cleanly FALSIFY (by matching `full`), which it did not, so the
mechanism story survives without being confirmed by it.

**WHERE THIS POINTS.** At inference the model is still asked to answer in ONE SHOT from demonstrations of `(a,b)` — it
never has to chain. So chained targets teach the ANSWERS, not the PROCEDURE. To learn the procedure, chaining has to be
the inference-time computation, not just a training signal. That is the harness with its one defensible mechanical job —
supplying iteration a bounded architecture cannot — and it converges with the depth argument: composing *m* operations
needs *m* applications, and a one-shot read-out has one.

### Chaining at INFERENCE time: the first inductive gain (2026-07-27) — `scratchpad.py`

`chaining.py` made chaining a training signal and got a transductive result only. The diagnosis was that at inference the
model still answered in ONE SHOT, so chained targets taught the answers and not the procedure. The fix is to make chaining
the computation performed at inference: the model decodes **`[intermediate, output]`** instead of `[output]`. For a
composition `(a,b)` the target is `[φ_a(x), φ_b(φ_a(x))]`; for a primitive it is `[x, φ_p(x)]` — identity, then the
primitive — so the SECOND decode step is always the same operation, "apply one primitive to the intermediate", exercised
by every task in training.

The demonstrations stay `[in, out]` as the environment provides them and never show an intermediate: showing one would
reveal the decomposition of an unseen task at test time. On a held-out composition the model must infer where to break,
unsupervised.

| arm | prims | labelled | HELD-OUT | solved | secs |
|---|---|---|---|---|---|
| control (one-shot) | 0.961 | 0.937 | 0.046 | **0/34** | 164 |
| **scratchpad** | 0.871 | 0.886 | **0.363** | **7/34** | 164 |

**Held-out 0.046 → 0.363, and 0 → 7 tasks solved, on compositions never presented in any form** — the unlabelled and eval
pools combined, neither of which receives a gradient here. Unlike `chaining.py`'s 0.593-on-the-unlabelled-pool, these
tasks were never touched, so this is compositional generalisation rather than transduction. It is the first inductive gain
in the line, against a background of: architecture null, optimiser null, task diversity null, training-signal chaining
transductive-only.

It also costs a little in-distribution accuracy (0.961 → 0.871 on primitives, 0.937 → 0.886 on labelled), so the
scratchpad trades supervised fit for generalisation rather than being free.

**WHAT WAS HANDED OVER, stated precisely: the ARITY, not the DECOMPOSITION.** The format has exactly one intermediate
slot, which tells the model that tasks break into two steps. It is never told WHERE to break for a held-out task, and the
demonstrations never show it. Whether that arity prior is doing the work is directly testable — 3-compositions with one
slot, or a variable number of slots — and that is the honest next experiment rather than a caveat to be waved at.

**And a real bug the verification caught**, worth keeping because the experiment would otherwise have been quietly
corrupt: vectorising the data generator required classifying each primitive as a position permutation or a value map, and
the test "leaves a constant sequence unchanged" was run on the all-ZEROS sequence — which `negate` fixes, since
`(V−0) mod V = 0`. `negate` was classified positional and four tasks were generated wrongly. Found only by checking the
fast generator against the reference `apply_task` on every task; the fix tests every constant, and all 63 tasks now match.

**Honest limits.** 0.363 with 7/34 solved is a real effect and not a solved problem — most held-out compositions still
fail. One seed. And the result validates a specific chain of reasoning (composing *m* operations needs *m* applications; a
one-shot read-out has one) whose next prediction is concrete: the gain should extend to 3-compositions only if the
intermediate count extends with them.

