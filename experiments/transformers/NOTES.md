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

> ⚠ **RETRACTED 2026-07-28 — the 0.363 was TEACHER FORCING, not chaining.** Accuracy was read from a single forward pass
> over a sequence that already contained the TRUE intermediate, so the model was handed `φ_a(x)` for a held-out task and
> only had to apply the second step. Re-measured free-running (`rollout`, same file, same config), the arm is **0.024
> against the control's 0.047** — the gain is gone and slightly reversed. Full account in the arity entry below.

---

## THE ARITY TEST — and the measurement bug it exposed (2026-07-28) — `arity.py`

The prediction under test, stated in advance: if the scratchpad works because *composing m operations needs m
applications*, then on 3-compositions `slots=2` (depth matched) ≫ `slots=1` (too shallow) ≈ `slots=0` (one shot). The
arity — how many decode blocks the format supplies — is the only thing that varies.

**Design.** Every distinct function writable in ≤3 primitives, deduplicated ACROSS lengths so a "3-composition" cannot be
solvable in one application. Primitives all supervised; pairs and triples split in half into supervised and held-out. The
chain is packed over the slots by one formula (`prefix()`): `front` absorbs any excess at the FRONT, so the LAST block is
always "apply one primitive" and a task deeper than the slots must take a COMPOUND first step — which is exactly the
arity mismatch. `end` runs the chain from the start and REPEATS once finished, which is the **no-arity** form (the user's
point, 2026-07-28: *there is no program that can always hand the correct arity*).

**The primitive set was chosen by counting, and the sanity gate is what forced it.** All 12 primitives give 174 distinct
functions and a 92-task supervised pool, at which size the model does not fit its own training distribution — the
one-shot control reached **0.171 on supervised primitives**, which makes every held-out number uninterpretable. Five
primitives give 56 distinct functions and a supervised pool of 30, the size `scratchpad.py` actually fit. The gate is now
enforced in code rather than eyeballed.

### The bug, and how it announced itself

The no-arity arm returned **1.000 on every pool, held-out included.** A perfect held-out score is not a result, and the
cause was immediate: under `end` packing the block *before* the answer is by construction IDENTICAL to the answer, and
evaluation was a single teacher-forced forward pass, so the model scored 1.000 by copying its own context.

Chasing that exposed the general form, which was in `scratchpad.py` all along: **teacher-forced evaluation hands the
model the true intermediate.** The whole claim was that on a held-out composition the model must infer where to break —
but it was never asked to produce the break point, only to finish from a correct one supplied for free. The environment
can never supply an intermediate, so free-running generation is the only evaluation the claim was ever entitled to.
`rollout()` now cuts the sequence back to the demonstrations plus the query input and lets the model generate every block
itself, feeding its own tokens back.

### Corrected results — free-running, 10000 steps, all arms matched

| arm | sup prims | sup pairs | sup triples | HELD pairs | HELD triples | *forced* HELD triples |
|---|---|---|---|---|---|---|
| `slots=0` one shot (control) | 0.996 | 0.992 | 0.987 | 0.104 **0/8** | 0.113 **0/18** | 0.113 |
| `slots=1` one intermediate | 0.905 | 0.689 | 0.673 | 0.061 **0/8** | 0.120 **0/18** | **0.477** |
| `slots=2` depth matched | 0.793 | 0.710 | 0.569 | 0.051 **0/8** | 0.068 **0/18** | **0.591** |
| `slots=4 pack=end` no arity | 0.838 | 0.833 | 0.711 | 0.072 **0/8** | 0.068 **0/18** | 1.000 *(copying)* |

And `scratchpad.py` itself, re-run at its own original config:

| arm | prims | labelled | HELD-OUT (free) | solved | *forced* |
|---|---|---|---|---|---|
| control | 0.964 | 0.940 | **0.047** | 0/34 | 0.047 |
| scratchpad | 0.827 | 0.392 | **0.024** | 0/34 | **0.313** |

1. **Free-running, no arm beats the control, and not one held-out task is solved anywhere.** The `forced` column
   reproduces the apparent gain exactly (0.477 / 0.591 against a 0.113 control, and 0.313 on the original config, versus
   the 0.363 originally reported). The entire effect was the evaluation.
2. **The arity question is moot as posed.** You cannot ask whether matching the slot count to the composition length
   helps when no arm chains at all. Inside the forced column the ordering `slots=2` (0.591) > `slots=1` (0.477) >
   `slots=0` (0.113) is consistent with the depth argument — but that column measures "finish, given a correct partial
   result", so it cannot carry the claim.
3. **What DOES survive, stated at its real size.** Given the true intermediate, the model completes a held-out
   composition at 0.48–0.59 against a 0.11 one-shot control. That is a genuine LID statement about the LAST step: one
   primitive applied to a supplied partial result stays in-distribution even when the whole task is novel. It is not
   chaining, and the gap between the two columns is precisely the thing that is missing.
4. **The scratchpad makes the model WORSE at its own supervised tasks under free-running** (0.987 → 0.673 → 0.569 on
   supervised triples as slots grow, and 0.940 → 0.392 on the original config). Errors in its own intermediates compound:
   it can produce the chain when each step is corrected for it and cannot when they are not. The three deeper arms all
   FAIL the sanity gate free-running, which is itself the finding rather than a nuisance.
5. **The no-arity design has a separate, structural flaw.** Padding by REPETITION creates a copy shortcut that dominates
   the training loss — most predicted blocks are "the same as the last one" — so the model learns to copy rather than to
   compute. Any honest no-arity format must let the model stop without giving it something to copy: a HALT token, which
   the model emits and which ends the chain, is the version that does not have this hole.

**Where this leaves the line.** The last standing positive result is withdrawn, and the ledger is now: architecture null,
optimiser null, task diversity null, training-signal chaining transductive-only, inference-time chaining **null once
evaluated honestly**. What survives is sharper than any of them — the model can execute one local step on a novel
composition but cannot sequence two of its own — and that is exactly the estimate-versus-execute seam
`compositional_rep.py` located, now measured on behaviour instead of on representations.

**The methodological lesson, which is the third of its kind in this line and the most expensive.** A metric manufactured
a false positive: Spearman without tie handling gave ρ = +1.000 on information-free data; a rank-deficient `lstsq` gave
NaN for every score; and now teacher-forced decoding gave a phantom 8× gain that survived a commit and a write-up. All
three were found by checking the apparatus against a case whose answer was known in advance, never by reading the
numbers. **If a model is scored at a sequence position, look at what is in its context at that position.**

---

## THE HALT TOKEN — arity removed, and it was never the bottleneck (2026-07-28) — `halt.py`

The standing constraint: **no arity may be handed over**, because no program can always supply the correct one. The
decode stream is `b_1 b_2 … b_m HALT` and `m` is the model's own output — it emits blocks of L digits until it emits
HALT, and the answer is the last complete block before it. Nothing says how many blocks a task needs.

**The design point `arity.py` paid for.** The previous no-arity attempt padded a fixed-depth format by REPEATING the
answer and scored a perfect 1.000 by copying. Repetition does not remove the arity, it removes the task. Here a task of
length m is supervised on exactly m blocks — prefixes 1…m, one primitive per step — then HALT; everything past the first
HALT is filler and is MASKED OUT of the loss, so **there is no repeated block to copy**. The HALT itself is scored,
because deciding to stop IS the output under test. Held-out compositions get no gradient in any form, so the model must
infer from `[in, out]` pairs alone both where to break AND how many breaks there are.

Five measurements, because "accuracy" alone cannot say what went wrong. ANSWER (the claim), DEPTH (emitted exactly the
task's own number of blocks — the arity question asked directly), FIRST (its first block equals `φ_a(x)`), HALTED
(terminated at all), and *forced* (teacher-forced answer block — read the sanity gate against THIS, since it asks
whether the model learned the training distribution, whereas free-running additionally asks whether it survives its own
intermediate errors).

| pool | n | ANSWER | solved | DEPTH | FIRST | HALTED | *forced* |
|---|---|---|---|---|---|---|---|
| supervised prims | 5 | 0.930 | 5/5 | 0.981 | 0.941 | 1.000 | 0.941 |
| supervised pairs | 7 | 0.907 | 7/7 | 0.994 | 0.949 | 1.000 | 0.949 |
| supervised triples | 18 | 0.818 | 11/18 | 0.995 | 0.903 | 1.000 | 0.981 |
| **HELD-OUT pairs** | 8 | 0.084 | **0/8** | **0.772** | 0.329 | 1.000 | 0.407 |
| **HELD-OUT triples** | 18 | 0.048 | **0/18** | **0.842** | 0.344 | 1.000 | 0.518 |

14000 steps, gate passed at 0.941. Baseline: the one-shot control (`arity.py --slots 0`, same splits/seed/primitives)
scores 0.113 free-running on held-out triples.

1. **The model learns to run its own chain.** On supervised tasks, free-running, it terminates 100% of the time, picks
   the right depth 0.98–0.99, and solves 5/5, 7/7 and 11/18 — losing only ~0.01–0.16 against teacher forcing. The
   format works; the earlier 10000-step run failed the gate only because the halt format is markedly harder to FIT than
   a fixed-depth one (supervised primitives 0.723 forced at 10000 against `slots=0`'s 0.996). Choosing your own depth
   costs accuracy on the parts.
2. **THE ARITY IS INFERRED, and this is the line's first genuine positive.** On compositions never supervised in any
   form, the model emits exactly the right number of blocks **0.842** of the time for triples and **0.772** for pairs.
   That is not the prior: the supervised pool is 60% triples, so guessing by base rate gives ~0.60 on triples and ~0.23
   on pairs, and always-emit-3 gives 1.00/0.00. It discriminates — it reads the required DEPTH of a novel composition
   off `[in, out]` demonstrations alone.
3. **And the arity was never the bottleneck.** ANSWER on held-out triples is **0.048 against the one-shot control's
   0.113** — null, and slightly WORSE than not chaining at all. FIRST is 0.344 against 0.903 on supervised triples: the
   model takes the wrong first step two times in three, and every later step inherits it. So a self-chosen chain makes
   held-out performance worse, for a mechanically clear reason.
4. **The diagnosis, sharper than anything before it: it knows how many steps a novel task needs, and cannot take the
   first one.** Held-out triples score 0.518 *forced* against 0.048 free — the same dissociation `arity.py` found, now
   with the arity removed as an explanation. Nothing left to hand over made a difference.

**Stability.** The 10000-step run agrees on everything that matters: ANSWER 0.066, DEPTH 0.886, FIRST 0.326, HALTED
1.000 on held-out triples. The conclusion does not rest on the long run. One seed, though — and the DEPTH number is the
one that most deserves a second, being the only positive.

⚠ **Run-time note:** the 14000-step run took 349 s, over the 5-minute transformer budget. The 10000-step run (244 s) is
inside it and carries the same conclusion; use `--steps 10000` unless the gate specifically needs the extra fit.

**WHERE THE LINE NOW STANDS.** Every lever that could be varied from the outside has been: architecture, optimiser, task
diversity, chaining as a training signal, chaining at inference, matched decode depth, and now self-chosen decode depth
with no arity handed over. All null on held-out composition. What is left is not a format question. The model represents
a novel composition correctly (`compositional_rep.py`, held-out R² +0.52), knows how deep it is (0.84 here), can finish
it when handed a correct partial result (0.52 forced) — and cannot produce that partial result itself (0.34). **The
missing operation is applying a primitive it has identified to a value it is holding, when the pair was never trained
together.** That is one local step, and it is where the next experiment has to go.

---

## IDENTIFY vs APPLY — the composition failure, LOCATED (2026-07-28) — `identify.py`

**This resolves the whole composition line.** Every null in it — architecture, optimiser, task diversity, chaining as a
training signal, chaining at inference, matched decode depth, self-chosen depth — was varying the EXECUTION pathway. It
was never broken.

The hypothesis came from the arithmetic line: `engine.py` showed routing on a GIVEN discrete key is exact (swap the
operation token and the model computes the other operation at 1.000), `alloc.py` showed such routing is nearly free when
the circuit can be shared, while the composition model builds WHOLE-task keys perfectly (supervised ~0.95) and cannot
factor an observed whole into PART keys — it knows a held-out 3-composition needs three steps (DEPTH 0.842) and cannot
produce the first (FIRST 0.344). Counting the parts while unable to name them is the signature of estimating a scalar
rather than factoring a function.

**The design.** Put an explicit primitive-identity token before each block — `ID_1 b_1 ID_2 b_2 … ID_m b_m HALT`, with
HALT a value of the identity slot so the no-arity property is kept — and vary ONLY whether that token is an input or an
output. `given` supplies it at train and test and excludes it from the loss; `predicted` supervises it on labelled tasks
and makes the model emit its own at test. **`predicted` is therefore an exact control for `given`:** identical
architecture, vocabulary, sequence length and token layout, differing in one bit of experimental design.

| arm | HELD-OUT triples ANSWER | solved | DEPTH | FIRST | ID@1 | PROG | *forced* |
|---|---|---|---|---|---|---|---|
| `none` (= `halt.py`) | 0.048 | 0/18 | 0.842 | 0.344 | — | — | 0.518 |
| **`given`** (identity = INPUT) | **0.997** | **18/18** | 1.000 | 0.998 | — | — | 0.999 |
| `predicted` (identity = OUTPUT) | 0.020 | 0/18 | 0.815 | 0.386 | **0.365** | **0.000** | 0.955 |

Supervised pools: `given` 1.000/0.999/1.000; `predicted` 0.994/0.999/0.996 with PROG 0.994–0.999. Both gates passed at
1.000 teacher-forced. 12000 steps, ~280 s per arm.

1. **APPLICATION IS NOT THE BOTTLENECK — 0.048 → 0.997, 0/18 → 18/18.** Told which primitives to apply, the model chains
   three of them it has never seen composed, **free-running** — it generates every intermediate itself; only the identity
   slots are injected. Compositional execution over novel combinations is essentially perfect. This is the LID claim
   holding exactly: every local step is in-distribution, and the composite being novel costs nothing.
2. **IDENTIFICATION IS THE BOTTLENECK, and it fails outright.** Asked to name the parts, the model **never once emits the
   correct program** for a held-out triple — PROG = 0.000 across 18 tasks × 128 samples. ID@1 is 0.365 against a chance
   floor of 0.200, so there is *some* signal about which primitive comes first and nowhere near enough to run a chain on.
3. **Making the key explicit does not help if the model cannot fill it in.** `predicted` (0.020) is no better than `none`
   (0.048) — slightly worse. The format was never the problem either.
4. **The sharpest single pair of numbers in this line:** with the true identities in its context the model answers at
   **0.955**; producing those same identities itself, **0.020**.
5. **It converges with the representational probe.** `compositional_rep.py` found the code for a held-out composition
   predictable at R² = 0.52 from role vectors — *partly* compositional. ID@1 = 0.365 against 0.200 is *partly* above
   chance. A behavioural and a representational measure independently agree that real but insufficient part-information
   is present.

⚠ **`given` IS A DIAGNOSTIC AND AN UPPER BOUND, NOT A SOLUTION.** It hands over the decomposition of a held-out task,
which is exactly the rigging this line otherwise refuses. Its value is that it isolates one ability and settles which one
is missing. Nothing here solves compositional generalisation.

**What it implies.** The thing to build is not a better executor, a better objective, or a better format — it is a
mechanism that PRODUCES the decomposition: search over candidate programs, or an inference procedure that reads part
identities out of demonstrations. That reframes the harness's job precisely: not supplying iteration (the `scratchpad.py`
story, now retracted) but supplying **identification**. It also explains why explicit-program prompting works so well on
LLMs — the program supplies the identities, which is the one thing the network cannot recover.

For `src/tbt/` this is the same shape as `reference_cue_competition_key_discovery` (discovering what the condition
actually is) and `reference_hypothesis_generation` (a small context-cued sample of candidate targets). The composition
problem is a hypothesis-generation problem, and this line has now proved it by elimination rather than by assertion.

**Honest limits.** One seed per arm. Five primitives, compositions of at most three, one domain. And ID@1 slightly
understates identification, since a task can admit an alternative valid split that scores as wrong — though PROG = 0.000
leaves little room for that to matter.

---

## THE DISCRETE-LOG PROBE — the model discovered logarithms (2026-07-28) — `arith.py`

**The question:** what do the weights of a MULTIPLICATION operator actually look like? **The prediction, derived before
running rather than found afterwards.** Nanda et al. 2023 (arXiv:2301.05217) reverse-engineered modular ADDITION in a
one-layer transformer and found the embedding sparse in the FOURIER basis; Chughtai, Chan & Nanda (arXiv:2302.03025)
generalised it — a network learning a GROUP operation implements it through the group's IRREDUCIBLE REPRESENTATIONS, and
a cyclic group's irreps are the Fourier modes. The units mod `p` are cyclic under multiplication, and the isomorphism
carrying them to addition is the DISCRETE LOGARITHM. So a transformer trained on modular multiplication must be sparse in
the Fourier basis of the **discrete-log index**, not the value index — i.e. its multiplication operator is `log → add →
exp`.

Setup: p=97 (primitive root g=5, found by checking order, never hardcoded), inputs restricted to the units 1..96 for
both operations so both spectra are FFTs of the SAME 96 embedding rows and are directly comparable. One layer, d=128,
4 heads, full batch, weight decay 1.0, 50% train split, 10000 steps (~100 s). Sparsity is reported as the PARTICIPATION
RATIO `1/Σp²` — the effective NUMBER of frequencies in use, needing no choice of k. A uniform spectrum over 48
frequencies gives 48; a single spike gives 1.

`addm` is the exactly-matched crossed control: addition **mod 96** over the same domain. Addition mod 97 sampled at only
96 points would leak across FFT bins and understate its value-basis sparsity; mod 96 is a clean group operation on
Z₉₆ — the SAME order as the multiplicative group — so the two operations live on isomorphic groups and differ *only* in
whether the natural coordinate is the value or its logarithm.

| model | test acc | value-basis eff. freqs | log-basis eff. freqs |
|---|---|---|---|
| untrained | 0.012 | 47.7 | 47.5 |
| shuffled labels (memorised, train 1.000) | 0.010 | 47.7 | 47.6 |
| **× mod 97** | **1.000** | 46.3 | **6.6** |
| **+ mod 96** | **1.000** | **8.0** | 45.8 |

**The crossed prediction is exact.** Multiplication is flat in the value basis (46.3 of 48) and sparse in the discrete-log
basis (6.6); addition is the mirror image (8.0 / 45.8). A pipeline that reported log-basis sparsity for both would have
been detecting the reindexing rather than the model, and it does not.

**And it is CAUSAL, not just correlational** — the lesson of the teacher-forcing retraction above. Ablation keeps or
removes the top-5 frequencies of each basis, with random controls matched to the same non-DC rank (10 real dimensions)
and the row mean retained in every condition, so the two conditions differ *only* in which non-DC directions they use:

| model | basis | keep named | keep random | drop named | drop random |
|---|---|---|---|---|---|
| × mod 97 | value index | 0.023 | 0.015 | 0.721 | 0.890 |
| × mod 97 | **DISCRETE LOG** | **0.893** | 0.015 | **0.083** | 0.859 |
| + mod 96 | **value index** | **0.760** | 0.013 | **0.113** | 0.897 |
| + mod 96 | DISCRETE LOG | 0.015 | 0.015 | 0.811 | 0.911 |

1. **The multiplication circuit is `log → Fourier-add → exp`, and it is TINY.** Keeping 11 of 96 row-dimensions — DC plus
   five conjugate frequency pairs in the discrete-log index — retains **0.893** accuracy; a random subspace of the same
   rank retains 0.015. Removing those same 11 collapses the model to 0.083 while removing a matched random subspace
   leaves 0.859.
2. **The test correctly reports NOTHING in the wrong basis.** For multiplication the value-basis frequencies behave
   exactly like random directions (0.023 vs 0.015), and for addition the log-basis frequencies do (0.015 vs 0.015). The
   ablation is not rigged to fire; it fires in the operation's own basis and nowhere else.
3. **Sparsity tracks GENERALISATION, not fit.** The shuffled-label model reaches train accuracy 1.000 — it memorises 4608
   random labels perfectly — and its spectrum is flat in both bases (47.7 / 47.6), indistinguishable from an untrained
   network. So the sparse basis is not what fitting data looks like; it is what *learning the group* looks like. The
   trace shows it forming: effective frequencies 40.4 → 9.3 → 6.1 → 6.6 over training, while test accuracy was already
   at 1.000 by step 2000.

**THE FINDING THAT MATTERS FOR THE INITIALISATION IDEA, and it is a conceptual one.** The structure of an operation is
not in the values — it is in the **coordinate system**. Multiplication and addition have *the same circuit*; they differ
only in the basis it is expressed in. So **an "instinct" is a basis, not a rule.** That is a much better thing to build
in than any hand-written procedure: it is revisable by training (unlike a rule), it is the general answer for abelian
group structure rather than a task-specific rig — so it passes the bitter-lesson discriminator recorded in `BEE.md` — and
the model's remaining job shrinks to *choosing which frequencies*, about 11 dimensions out of 96.

**It also confirms, in transformer weights, the design this repo already adopted for a different reason.**
`reference_operator_as_group_representation` says motion is a learned group-representation matrix acting on a location
code. Two independent lines — mechanistic interpretability of a transformer, and cortical operator theory in `src/tbt/` —
converge on *an operation is a change of basis in which it becomes coordinate-wise*.

**Honest limits.** One seed per arm; the crossed design is the control, not replication. Modular arithmetic only —
multi-digit non-modular multiplication needs carries and partial products and its circuits are far less legible, which is
why it was scoped out. And Zhong et al. (arXiv:2306.17844) showed the circuit for a given task is not unique ("clock" vs
"pizza"), so this is *a* solution the optimiser reaches, not the only one.

**NEXT, and it is the half of the original question still unanswered.** This settles what a SINGLE operation looks like.
The "arithmetic engine" question — one shared basis with an operation-conditioned read-out, or two duplicated circuits —
needs a model trained on `{+, ×}` together with an operation token, then asked in the weights whether the bases coincide.
That is the same recruit-vs-duplicate question as the composition failure above, at a scale where the answer is visible,
and the apparatus for it now exists.

---

## THE ARITHMETIC ENGINE — two circuits in one matrix, and a perfect multiplexer (2026-07-28) — `engine.py`

One model, `[a, OP, b, EQ] → c`, both operations, p=97, units-only domain so both spectra are FFTs over the same 96
embedding rows. Addition is mod 96 so that both operations live on cyclic groups of the SAME ORDER — otherwise two
different frequency sets would be trivially distinguishable by group order rather than by basis. 12000 steps, ~260 s.

**The prior, stated before running.** There is no single basis in which addition and multiplication are simultaneously
diagonal — that is close to what makes a field a field, and the map between their diagonalising bases is the discrete
logarithm, which is not linear. So the model *cannot* share a coordinate system. Duplication is forced by mathematics,
not chosen by the optimiser, and the real question is how it SELECTS. **That is a boundary on the recruit-versus-
duplicate framing worth keeping: some pairs of operations have no shared basis to recruit, and the cost is unavoidable.**

**(1) Both operations generalise perfectly** — test + 1.000, × 1.000 — so everything below is interpretable.

**(2) Both bases are carried at once.** Effective frequencies: value index **22–29**, discrete-log index **26–27** (two
runs at the same seed; bf16+autotune is not bit-deterministic, and the spread is itself a reason not to read much into
the exact value). A single-operation model was ~6.6 in its own basis and ~46 (flat) in the other; here both sit in
between, which is the signature of two sparse frequency sets superposed in one matrix.

**(3) Basis ablation — a double dissociation, and the correction to (2).** Top-5 frequencies of each basis kept or
dropped, rank-matched random controls, row mean retained in every condition:

| basis | condition | + acc | × acc |
|---|---|---|---|
| value index | **keep named** (11 of 96 dims) | **0.887** | 0.021 |
| value index | keep random (matched) | 0.016 | 0.013 |
| value index | **drop named** | **0.070** | 0.840 |
| value index | drop random (matched) | 0.917 | 0.905 |
| discrete log | **keep named** (11 of 96 dims) | 0.017 | **0.957** |
| discrete log | keep random (matched) | 0.013 | 0.016 |
| discrete log | **drop named** | 0.829 | **0.131** |
| discrete log | drop random (matched) | 0.898 | 0.887 |

Removing addition's frequencies destroys addition and **leaves multiplication almost untouched**, and vice versa. Two
separate representations sharing one embedding matrix, each surgically removable.

⚠ **AND THE KEEP COLUMN RETRACTS THE "LESS SPARSE" READING OF (2).** Keeping only 11 of 96 row-dimensions still gives
**+ 0.887 and × 0.957** — as good as, or better than, the single-operation models' 0.760 and 0.876 on the same test. So
**neither circuit degraded at all**: each still needs five frequencies and eleven dimensions. The risen participation
ratio is an artifact of superposition — the value-index FFT necessarily now also sees multiplication's structure, which
is *diffuse in that basis* (that is exactly why a mul-only model read 46.3 there) — and a participation ratio cannot
separate "one sparse structure got blurred" from "a sparse structure plus a diffuse one". Only the causal keep/drop test
can, and it says the circuits are intact. **The cost of the second operation is the step delay alone.**

**(4) The switch lives in the MLP, as two preferentially-tuned populations.** Per-neuron selectivity
`(m_add − m_mul)/(m_add + m_mul)` on activation magnitude at the answer position, over 512 neurons: **156 with |s| > 0.5**,
183 shared at |s| < 0.1, **none above 0.9**, median |s| 0.145.

| ablated (top decile, 51 neurons) | + acc | × acc |
|---|---|---|
| most ADD-selective | **0.105** | **1.000** |
| most MUL-selective | **0.998** | **0.049** |
| 51 random (control) | 0.930 | 0.669 |

⚠ **Half of this is definitional and must be read that way.** Selectivity is measured from activation magnitude, so
"ablating neurons that barely fire for ×" cannot be expected to hurt × — the SPARING is close to built in. The finding is
the other half: the top decile by one operation's preference is **necessary** for that operation (0.105 / 0.049), while a
random decile of the same size costs +ableness almost nothing (0.930). And the informative structural fact is that
**selectivity is GRADED, not absolute** — no neuron exceeds |s| = 0.9 — yet the top decile alone is enough to carry an
operation. A clean switch does not require dedicated units; a preferential population code suffices
(`reference_population_code_belief`).

**(5) The operation token is a PURE MULTIPLEXER, and this is the cleanest result of the five.** Replace the token and ask
not whether accuracy drops but *what the model then computes*:

| swap | matches the TOKEN | matches the operands' original operation |
|---|---|---|
| `+ → ×` | **1.000** | 0.011 |
| `× → +` | **1.000** | 0.010 |

It does not degrade — it **switches**, at full accuracy for whichever operation it is told. The token is not a hint that
perturbs a shared computation; it is a selector, and it selects perfectly. (The swapped queries overlap the other
operation's training pairs, but that is immaterial: per-operation *test* accuracy is already 1.000, so there is nothing
train/test membership could add.)

### The mechanism, assembled

**A shared embedding holding both coordinate systems in superposition, plus an operation-token-gated selection of largely
disjoint MLP populations.** Not one engine that "does arithmetic" — two circuits in one matrix with a clean demultiplexer
in front of them. Asked directly: the circuit chooses by *routing*, not by computing both and picking, and the routing
signal is the operation token acting on which MLP population is driven.

### The measured cost of the second operation, and what it is NOT

**Generalisation moved from step ~2000 to step ~6000 — a 3× delay — and that is the ENTIRE cost.** The circuits do not
degrade (11 dimensions still carry each operation, above). Two things rule out the obvious explanations:

* **It is not data dilution.** Per-operation training examples per step are *identical* in the two experiments: the
  single-op run trained on 4608 examples of one operation, and the engine trains on 9216 examples of which ~4608 are that
  same operation. Same architecture, same per-op gradient signal per step, 3× the steps to generalise.
* **It is not running out of capacity.** The two circuits occupy 11 + 11 = 22 of 96 available row-dimensions, so there
  was ample room. The difficulty is not fitting them in; it is *finding the split*.

**So the extra 4000 steps buy an ALLOCATION, not a circuit** — two mutually non-interfering subspaces plus a routing
function from the operation token to disjoint MLP populations. And the trace says the allocation is a shared prerequisite
rather than something learned per operation: both operations sit at chance together (0.298 / 0.295 at step 2000), rise
together (0.508 / 0.520 at 4000), and are both perfect together (1.000 / 1.000 at 6000). They do not appear in sequence.
Nothing can specialise until there is somewhere to specialise into, and once there is, both circuits crystallise at once.

### Why this might bear on LLM scale — and the three experiments that would decide it

The tempting extrapolation: if skills are individually cheap (11 dimensions) and what costs compute is *carving shared
parameters into non-interfering subspaces and learning to route between them*, then large-scale training compute is
mostly an **allocation search**, not a knowledge-storage cost. Two things here support that framing beyond analogy: the
cost appeared with per-op data held constant, and the joint phase transition matches emergent abilities arriving as
transitions rather than smooth accumulation. The forced-duplication result adds a second lever — where no shared basis
exists, parameters must grow with the number of mutually *incompatible* structures rather than with the number of skills.

**What this evidence cannot support.** Two data points establish no scaling law: 1→2 operations costing 3× is not
`3^n`, and it is not even known to be superlinear. The regime is also unlike LLM training — full batch, weight decay 1.0,
an exactly-specified algebraic task, many epochs, and delayed generalisation that is partly an artifact of that setup,
against ~1 epoch on heavy-tailed statistical structure with weak regularisation. Treat the above as a hypothesis this
apparatus can now test, not as a finding.

Three cheap experiments, in order of how much they would settle:
1. **Operation-count sweep** (1, 2, 3, 4, …): steps-to-generalise and per-circuit rank. Superlinear steps with flat
   per-circuit cost is the claim; linear steps make it mild.
2. **Sequential vs joint** — teach `+` first, then add `×`. If sequential is much cheaper, the cost was interference
   during joint learning; if not, it was discovering the routing.
3. **PRE-SUPPLY THE BASES in the initialisation** — the structured-init idea, now with a sharp prediction. If the 3×
   delay is the basis search, initialising the embedding with both frequency sets present should remove most of it. That
   makes the "an instinct is a basis" claim directly testable *and* makes it a test of the scale hypothesis at the same
   time.

---

## WHAT THE ALLOCATION COST ACTUALLY IS: basis conflict, and it is avoidable (2026-07-28) — `alloc.py`, `instinct.py`

Two axes that `engine.py` confounded, separated. All runs 10000 steps, p=97, same architecture; the reported number is
**steps to 90% test accuracy**, traced at 500-step resolution.

### Axis 1 — does it matter whether the operations CAN share a basis?

Every Z-linear map `(αa + βb) mod 96` is diagonal in the SAME value-index Fourier basis, and `×`/`÷` are both diagonal in
the discrete-log basis, so those pairs *can* share a coordinate system entirely. `+` and `×` cannot — no basis
diagonalises both. Coefficients are chosen coprime to 96 so every map is balanced in each argument.

| operations | can share? | steps to 90% | own-basis eff. freqs | op-selective neurons? |
|---|---|---|---|---|
| `+` alone | — | **1500** | 8.4 | — |
| `+`, `−` | yes (value basis) | **2500** | **7.7** | **none** |
| `×`, `÷` | yes (log basis) | **2500** | **8.9** | **none** |
| `+`, `×` | **no** | **5000–5500** | 26.1 / 26.5 | **clean diagonal** |

1. **The optimiser RECRUITS when recruiting is possible.** For `{+, −}` the value basis stays at 7.7 effective
   frequencies — as sparse as a *single* operation — and one frequency set serves both: keeping it gives 0.915 / 0.911,
   dropping it gives 0.031 / 0.029. Both operations live or die together on the same ten dimensions. `{×, ÷}` replicates
   this in the other basis (keep 0.712 / 0.684, drop 0.178 / 0.116).
2. **And it builds no routing machinery it does not need.** For `{+, −}`, ablating the decile most selective for either
   operation leaves BOTH at 1.000, while a *random* decile is worse (0.744 / 0.734). There is no selective population to
   find, so "most selective" degenerates to "least active" — which is a caution about the selectivity index at
   near-zero activation, and simultaneously the cleanest evidence that no switch exists. None is needed: one circuit
   computes both.
3. **Where sharing is impossible, both duplication and routing appear.** `{+, ×}` shows two superposed frequency sets and
   a textbook routing diagonal — ablate the `+`-selective decile: `+` 0.101, `×` 0.997; ablate the `×`-selective decile:
   `+` 0.999, `×` 0.041; random: 0.786 / 0.862.
4. **So the cost is BASIS CONFLICT, not operation count.** Adding a second operation costs **1.67×** (2500/1500) when it
   can share and **3.3–3.7×** (5000–5500/1500) when it cannot. The extra factor is entirely attributable to having to
   carve out a second coordinate system and learn to route between them.

### The count sweep, and the refinement it forces

Four operations all diagonal in the value basis (`lin:1:1, lin:1:-1, lin:1:5, lin:5:1`):

| operations | steps to 90% (last) | own-basis eff. freqs |
|---|---|---|
| 1 | 1500 | 8.4 |
| 2 compatible (`+`, `−`) | 2500 | 7.7 |
| **4 compatible** | **5000** (3000 / 4000 / 5000 / 5000) | **12.4** |
| 2 *incompatible* (`+`, `×`) | 5000–5500 | 26.1 / 26.5 |

⚠ **This arm does NOT isolate routing count, and the spectrum is what caught it.** Sharing a *basis* is not sharing a
*circuit*. For op `(α, β)` the read-out needs `cos(ω(αa + βb − c))`, hence frequency `ωα` of `a` — so the frequency set has
to be closed under multiplication by α, and `lin:1:5` / `lin:5:1` demand **different frequencies within the shared basis**
than `lin:1:1` does. That is exactly the 12.4-against-7.7: roughly 1.6× more frequencies for four operations. So this arm
raised frequency demand as well as routing count, and its 3.3× cannot be attributed to routing.

It also explains why `{+, −}` was the clean case: α=1 for both and β = ±1 is only a conjugation, so those two share basis
*and* frequencies — which is why that arm showed no extra frequencies and no routing machinery at all.

**The unified statement, better than "basis conflict costs":** *cost tracks how much of the circuit must be DUPLICATED,
not how many operations there are.*

| what is shared | cost | frequencies | routing |
|---|---|---|---|
| basis **and** frequencies (`+`, `−`) | 1.67× | no increase (7.7) | none built |
| basis only, different frequencies (4 linear) | 3.3× | 1.6× more (12.4) | not measured |
| nothing (`+`, `×`) | 3.3–3.7× | two full sets (26) | disjoint populations |

Minor mechanistic note, one seed and not to be leaned on: the four compatible operations came online **staggered**
(3000/4000/5000/5000) rather than crystallising together as the incompatible pair did — consistent with the basis being
built once and each read-out added incrementally.

**The clean routing-count test is therefore still open, and is now well specified:** N operations needing the IDENTICAL
frequency set, e.g. `(a + b + c_i) mod 96` for distinct constants, since an additive constant is a pure phase shift and
adds no frequencies. If N of those cost nothing beyond the first, routing alone is free and the whole cost is frequency
demand. One line in `alloc.py`'s op parser.

### Axis 2 — pre-supplying the coordinate system (`instinct.py`)

Same incompatible pair `{+, ×}`. The initialisation replaces the value-token embedding with a superposition of `m=5`
frequencies per basis, **chosen at random** — deliberately not the ones a trained model turns out to use, which is what
keeps this from handing over the answer. *Which* frequencies is arbitrary (different seeds pick different sets and all
work), so what is supplied is the structure "the code is sparse in these two coordinate systems", not the computation, the
group structure, or the routing.

| init | `+` steps to 90% | `×` steps to 90% |
|---|---|---|
| random (the baseline) | 5500 | 5000 |
| **fourier** (5 random frequencies per basis) | **1500** | **1000** |
| **shuffled** — same matrix, ROWS PERMUTED | 6000 | 6000 |

1. **A structured init removes the allocation cost entirely — 3.7× and 5×.** And the number to notice is that `{+, ×}`
   from a Fourier init reaches criterion *faster than a SINGLE operation does from random init* (1000–1500 against 1500).
   Holding two mutually incompatible operations became free.
2. **THE FALSIFIER PASSES, and it is what makes this a mechanism rather than a speedup.** `shuffled` permutes the rows of
   the identical matrix: same singular values, same rank, same scale, same sinusoidal shape — only the correspondence
   between a row's position and its value is destroyed, which is precisely what makes it a *coordinate system*. It gives
   **nothing** (6000, marginally worse than random). So the benefit is not low-rank structure, not spectral shape, and
   not scale. It is the coordinate system.
3. **The model keeps what it was given.** The supplied frequencies (value 17/36/42/45/47, log 5/6/24/38/39) are still the
   dominant ones after 10000 steps. It did not discard the offered basis and search for its own.

### What this does to the LLM-scale hypothesis

It sharpens it in one direction and undercuts it in another, and both are worth having.

**Sharpened:** the cost does not scale with the number of skills. It scales with **the number of distinct coordinate
directions that must be carved out and routed to** — skills sharing both basis and frequencies are recruited into one
circuit at almost no cost (1.67× for a second, with no routing built at all), while four skills needing different
frequency subsets cost as much as two mutually incompatible ones. That is a much more specific claim than "more
capabilities cost more", and it predicts that what makes a domain expensive is structural heterogeneity rather than
breadth.

**Undercut:** the cost is **avoidable**. Supplying the coordinate systems at initialisation removed 3.7–5× of it, in a
case where duplication was mathematically forced. If the same holds at scale, a large part of pretraining compute is
buying coordinate systems that could in principle be supplied — which is an argument that scale is currently a *substitute*
for structure rather than a requirement. The honest caveat is unchanged and still binding: this regime (full batch, weight
decay 1.0, exact algebraic tasks, many epochs) is not LLM training, we know the right bases here analytically and for
language we do not, and 1→2 operations is not a scaling law.

**Still open.** Built but not yet run: sequential-versus-joint (`--seq 1`), to separate interference-during-joint-learning
from routing discovery — the first operation's accuracy after the switch also measures catastrophic forgetting. Not yet
built: the clean routing-count test specified above (`(a + b + c_i) mod 96`, identical frequency set, N ops).

And the experiment that would connect this line to the composition failure, which is the most valuable of the lot.
`engine.py` showed routing on a GIVEN discrete key is exact (1.000 on the token swap) and `alloc.py` shows it is nearly
free when the circuit can be shared. The composition line's model, by contrast, builds WHOLE-task keys fine (supervised
~0.95) but cannot factor an observed whole into PART keys: it knows a held-out composition has three steps (DEPTH 0.842)
and cannot name the first (FIRST 0.344). Counting the parts without identifying them is the signature of estimating a
scalar rather than factoring. **So: hand the composition model an explicit primitive-identity token at each decode step —
the same status the operation token has here.** If FIRST jumps and held-out composition works, the missing operation was
IDENTIFICATION, not application, and the two lines are one finding. If it does not, application really is absent and the
analogy fails. Either way it is decisive, and it is the same shape as `reference_cue_competition_key_discovery` —
discovering what the condition actually is.

**What it says for a structured initialisation.** The engine's shape is buildable as an init: superpose the candidate
bases in the embedding, and provide a gating pathway from a control token to disjoint hidden populations. That is
architecture, not task knowledge — the model still has to learn which frequencies and which population does what.

**Honest limits.** One seed. Two operations only, both on cyclic groups, both single-step. The 183 shared neurons are
unexplained — they may be doing input encoding common to both operations rather than anything operation-general, and
distinguishing those would need a further probe.

