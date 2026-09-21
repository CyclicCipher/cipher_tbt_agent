# Gradient-free mixing and features — the theories, the numbers that test them, what is unexplained, what next

Written 2026-09-22 as the synthesis of the four reference notes in `experiments/ziplearn/refs/`
(`opinion_pools_and_products_of_experts.md`, `context_mixing_paq_mattern.md`,
`bayesian_alternatives_to_learned_mixing.md`, `gradient_free_features_and_denoisers.md`) and the nine scripts under
`experiments/ziplearn/research/` (`pool_math_check.py`, `mix_temperature_bayes.py`, `expA_exponent_grid.py`,
`expB_naive_bayes_product.py`, `expC_sse_refinement.py`, `expD_confidence_exponents.py`, `r_window_neighbours.py`,
`expE_nlm_block.py`, `expF_cluster_features_text.py`), each with its JSON and log beside it. The two questions are
DESIGN §20's two "where a gradient is needed" points: (1) combining evidence MULTIPLICATIVELY with exponents that
E34 (`RESULTS.md`) found had to be learned; (2) FEATURES inside a written block, where E35 found a block either
memorises windows or collapses to the identity. The standing rule (2026-09-22): no gradient patch; mathematics and
small feel-experiments first. Nothing below proposes a gradient; §3 says what a gradient would be doing that the
closed forms so far do not.

## 0. How to read the numbers

Every text number is ONLINE (prequential) bits/char on a prefix of the held-out book *De Bello Civili* (alphabet
V = 70), the experts and mixers learning through the test as `e34.run_mixtures` does. Three different training
slices appear, and numbers must not be compared across them:

| slice | how cut | chain alone (`e34.Chain`, E33's blended order-8 KT table) | used by |
|---|---|---|---|
| "prefix 300k" | the first 300 000 characters of the concatenated training books (E34's `--train_chars` convention), test = first 60 000 of the book | **2.229** | `mix_temperature_bayes.py`, `expA` (default), `expB`, `expC` |
| "prefix 1.2M" | the first 1 200 000 characters, same test | **1.817** | `expA --train_chars 1200000` |
| "spread 300k" | `textlm.training_slice(train, 300000)` = 289 123 characters spread over the 15 books, same test | **2.030** | `expD`, `expF` (text) |
| E34 full | the full 2 726 046 characters, the full 240 588-character book | **1.823** (linear 1.820, geometric 1.801, product 7.665) | `runs/e34/e34_mixtures.json` |

The 1.2M chain (1.817) sits where E34's full run does (1.823), so the 1.2M rows are the representative ones; the
300k rows are 0.4 bits/char worse across the board and serve for ORDER, not level. All runs are deterministic, one
book, one slice, no seeds; differences of 0.005 bits/char or less are inside what another 60k slice could move.

Frame numbers are cell accuracy on E35's 135 held-out level-2 frames of LockPath and CollectAll (16 colours, radius-1
windows = 9 cells), corrupted at t = 0.25 / 0.5 / 0.75, repaired by a one-shot block or by the diffusion chain
(`e35.corrupt_chain`, schedule t_k = k/8). E35 strict (every window kept): one shot 0.940 / 0.854 / 0.732, chain
0.946 / 0.880 / 0.800 (`runs/e35/e35_strict.json`).

Process caveat recorded by the pool-math reader: to stop an over-long first run of `pool_math_check.py` it killed
every `python.exe` (six PIDs) at about 13:14; the JSON and log timestamps of `expB` (13:14) and `expC`, `expD`, `expE`,
`expF` (13:05–13:25) are after or around that moment, and `expB`'s log reports two identical runs (173–276 s), so the
numbers used here are from completed runs.

---

## Part I — Multiplicative evidence mixing without learned parameters

### I.0 The vocabulary the theorems fix (from the three mixing notes)

- **Linear pool** P = Σ w_i P_i is the "switching source" (Mattern 2012 eq. 25: argmin_Q Σ w_i D(P_i‖Q)); with
  w = the posterior over experts it is Bayesian model averaging, which is soft SELECTION (Minka 2002) and, by
  Mattern 2013 Remark 4.7, can never give the true symbol more probability than the most confident expert:
  max_w lin(x; w, P) = p_max(x; P). `e34.Mixer.update` (`post = w * probs / p`, then fixed share 0.02) is this.
- **Geometric pool** P ∝ Π P_i^{w_i} is argmin_Q Σ w_i D(Q‖P_i) (Mattern 2012 eq. 3); PAQ7's
  squash(Σ w_i stretch(p_i)) is this for a binary alphabet, with the exponents moved by the gradient of the coding
  loss of ONE logistic unit (Mahoney DCE §4.3.2; lpaq1: 560 weights). It is the only externally Bayesian pool of
  pointwise form (Genest 1984b). With exponents on the simplex it re-shapes but cannot sharpen beyond the components
  (Gneiting & Ranjan 2013 eq. 7); with S_w = Σ w_i > 1 it sharpens.
- **The exact exponent** of a source (Allard, Comunian & Renard 2012 eq. 17):
  w_i = ln P(D_i | A, D_<i) / ln P(D_i | A) — the conditional-to-marginal log-likelihood ratio of source i given the
  sources already counted: 1 if conditionally independent, 0 if a function of the others. Conditional independence
  ⇔ all exponents 1 with the prior to the power 1 − n (their Prop. 2). S_w is the effective number of independent
  sources (Bordley 1982; Clemen & Winkler 1985: k equicorrelated sources are worth k/(1 + (k−1)ρ), bounded by 1/ρ).
- **Closed forms that carry a cross-term:** Lindley's supra-Bayesian model (Genest & Zidek eq. 4.1): exponents
  w = Ξ⁻¹(μ_1 − μ_0) on the experts' log-odds (LDA); Satopää & Ungar 2015 eq. 3: w = diag(Σ)ᵀ Σ⁻¹ from the
  forecasts' covariance alone. Both discount co-varying experts through the inverse covariance.
- **No closed-form combination rule exists** (Mahoney DCE §4.3: "Probability theory does not answer the question");
  what can be inferred by counting is a posterior over a FAMILY of rules — a Bayesian mixture over a finite grid G of
  exponent vectors costs at most log2|G| bits over the best grid point on any sequence (the dominance inequality,
  CTS eq. 3; `bayesian_alternatives_to_learned_mixing.md` §2A), with no step size.
- **How much is at stake:** the only published same-experts comparison (Mattern 2012 Table 1, Calgary, 8 experts)
  puts learned exponents 3.6% (average) to 4.4% (book1) below Bayesian linear weights. On enwik8 count-based
  ppmonstr (1.524) is below lpaq1's learned mixer (1.580); the step to 1.2–1.3 is the model library, bit-level
  modelling, dictionaries and SSE, not the mixer (`context_mixing_paq_mattern.md` §4). So E34's target was
  mis-stated: a learned mixer over the same 18 experts should land near 1.75, not 1.3.

### I.1 The theories, each with its mathematics and its number

**T1 — The linear ceiling: a Bayesian (linear) mixture over these experts cannot gain, and its deficit is a
calibration deficit.**
Mathematics: Remark 4.7 (above); Ranjan & Gneiting 2010 Theorem 1 (any weighted average of distinct calibrated
forecasts is uncalibrated and beaten by its recalibration under every proper score); Minka (the posterior
concentrates on one expert).
Numbers: E34 full: linear 1.820 vs chain 1.823. `expA_exponent_grid_1200k.json`: the 2-expert `Mixer` gives
−0.003 (word), +0.006 (match — worse than the chain), −0.010 (order4); its mean weight on the chain is 0.84–0.90
(`expA_exponent_grid_1200k.log`). `expB_naive_bayes_product.json`: `linear` −0.011 / −0.028 / −0.018 on the diverse /
redundant / all18 sets at prefix 300k. The switching prior is not free either: fixed share 0.02 puts
−log2(0.98) = 0.029 bits/char of slack in the Herbster–Warmuth bound, ten times the 0.003 E34 gained;
`mix_temperature_bayes_decay.json` vs `mix_temperature_bayes.json`: the Veness 2012 decaying rate α_t = 1/(t+1)
takes the grid posteriors from 2.198 → 2.184 (A1) and 2.183 → 2.168 (A2-min), 0.014–0.015 bits/char, the same size
as the gains being measured. Status: holds everywhere it was measured; T7 shows the recalibration the theorem
predicts.

**T2 — The raw product's 7.665 is arithmetic, not a property of products: nested and cue-sharing experts count the
same evidence many times, and E34's KT smoothing toward UNIFORM makes every sparse expert vote for rare characters.**
Mathematics: Allard eq. 17 gives exponent 0 to a source that is a function of another; exponent 1 on 18 mostly
nested experts is S_w ≈ 16 on evidence worth ≈ 1–2 sources. `pool_math_check.json` reproduces it in miniature:
nested sources, exponent-1 product 0.4741 bits vs 0.3320 for the single longest context, exact exponents (0, 0, 1).
Numbers, prefix 300k (`expB_naive_bayes_product.log`): product_kt 8.558 (all18), 2.967 (chain + order2 + order4),
3.316 (chain + word + match + line). DIVIDING OUT the prior — the exact naive-Bayes rule
P ∝ Π P_m(c) / P_uni(c)^{M−1} — makes it WORSE: nb_kt 62.450 (all18), 3.227 (redundant, vs the raw product's 2.967).
Read off the arithmetic in `e34.Expert.dist`: a context seen once gives every unseen character 0.5/(1 + 35) ≈ 0.014,
and against a unigram of 0.001 that is a likelihood ratio of ≈ 14 FOR every rare character it has not seen;
17 such experts vote 14^17. Smoothing each table toward the running unigram instead (`nb_prior`) cuts 62.45 to
9.154 — still four times the chain, so the conditional-independence assumption is violated on its own (measured
r_cmi(chain, order8) = 1.00, (chain, order7) = 0.80, (chain, word) = 0.39; r_agree(chain, prevword) = 0.76,
(order1, line) = 0.89). `expD_confidence_exponents.log`: dropping order1..8 takes the raw product from 8.764 to 5.733
only — the remaining ten "diverse" experts (chain, pair1_3, pair1_4, line, word, match, …) all read the previous
character, so its evidence is still multiplied ~10 times. Status: supported; the redundancy is SHARED CUES across
context functions, of which order-k nesting is about half.

**T3 — The exponents of a geometric mixture need no gradient to be SELECTED: a closed-form posterior over a small
grid of exponent vectors lands on the hindsight-best point at the price the theorem says.**
Mathematics: the dominance inequality, total bits ≤ min_g bits(g) + log2|G|; for |G| = 63 and n = 60 000 that is
0.0001 bits/char.
Numbers: `expA_exponent_grid_1200k.json`, the no-share posterior (`grid_bayes_alpha0`, `GridPosterior(K, 1, alpha=0)`)
sits within 0.0002 of the best fixed grid point in all nine runs (word 1.784 vs 1.784; match 1.805 vs 1.805; order4
1.807 vs 1.807); the posterior favourite at the end IS the best fixed point for word and match on the coarse grid
(`expA_exponent_grid.log`: (0.8, 0.4) and (1.0, 0.2) vs best fixed (0.8, 0.4), (0.8, 0.2)). `mix_temperature_bayes_decay.json`
on the 18 experts: Bayes over 11 temperatures 2.184, over the 42-point (β_chain, β_rest) grid 2.168, vs E34's
geometric 2.191, linear 2.211, chain 2.229 (−2.7%). The share-0.01 posterior costs 0.003–0.006 over the no-share one
at 60k characters: the exponents are stationary within the book. Status: confirmed for the selection part; what a
single exponent vector cannot do is T4.

**T4 — What the exponents encode is CONDITIONAL on the mixing context; a per-context grid posterior (36 × K closed-
form weights) recovers what neither a global exponent pair nor a global temperature can.**
Mathematics: Allard's exact exponent is one number per combination (A, D_1..D_n); "one exponent per source" is the
simplification (their eq. 18); PAQ selects a weight set by a small context (Mahoney; GLN's "data conditioning");
the KT count estimate makes a shorter context informative exactly when the longer context's count N is small
(`opinion_pools_and_products_of_experts.md` §11.5), which is what E34's mixing context `s.cls() * 9 + deep` indexes.
Numbers (`expA_exponent_grid_1200k.json`, key `grid_bayes_by_ctx`, `GridPosterior(K, N_CTX)`): chain + word 1.776
(−0.041 vs the chain) against the best FIXED pair 1.784 (−0.033) and the linear mixer 1.814 (−0.003); chain + match
1.790 (−0.027) against −0.012 fixed and +0.006 linear; chain + order4 1.791 (−0.026) against −0.010 fixed and
−0.010 linear. `mix_temperature_bayes.json`: the per-context posterior over the 42-point grid (2.168) beats its own
best single point in hindsight (2.203) by 0.035, and the posterior-mean temperature differs by mixing context — 1.0 at deepest
order 3 and ≈ 1.25 at orders 4–8 in one class, 1.0–1.05 in another (`posterior_mean_beta_by_ctx`, decay run;
1.1–2.0 under fixed share). The total exponent MASS is not the
missing degree of freedom: the best single temperature β on the Bayesian-weight geometric mixture is 1.0 (0.75 →
2.285, 1.25 → 2.193, 1.5 → 2.252). Status: supported. In kind, two experts with context-conditioned closed-form
exponents (−0.041 at 1.2M, −2.3%) reach what E34's 18-expert geometric mixture reached on the full book (−0.022,
−1.2%) — different slices, so a comparison of kind, not of level.

**T5 — The SUM of the exponents is the effective number of independent sources and is the one number a count-based
redundancy measure supplies; the pairwise ALLOCATION it suggests carries no information.**
Mathematics: S_w (Bordley, Allard); Clemen & Winkler's k/(1 + (k−1)ρ); in `pool_math_check.json` the disjoint
structure wants S_w = 9 (exponents (3, 3, 3)) and the overlap structure S_w = 3.
Numbers: `expA_exponent_grid_1200k.log`, the a + b diagonals: word wants a + b = 1.1–1.2 (1.784 / 1.784 vs 1.791 at
a + b = 1.0 and 1.792 at 1.3) — the word since its boundary reaches up to 12 characters past the chain's 8, so it
holds some independent evidence; match wants a + b = 1.0 exactly (1.805; 1.2 → 1.817); order4 wants 1.0–1.1
(indistinguishable, 1.808 / 1.807) — a re-weighting of the chain's own backoff, not new evidence. In every run the
chain wants TEMPERING (a = 0.7–0.9) once any partner is multiplied in, while alone a = 1.0 is best: the partner's
correlated share has to be taken out of the chain. `expB_naive_bayes_product.log`: with exponents
λ_m = 1/(1 + Σ_j r(m, j)) from counted argmax agreement the sums land at 1.86 / 1.49 / 2.41 (diverse / redundant /
all18) and the code is −0.038 / +0.055 / +0.095 vs the chain; the CONTROL `nb_flat` with the same total spread
uniformly is at least as good in all three sets (2.180 vs 2.190; 2.278 vs 2.284; 2.286 vs 2.324). `expD` (spread
300k): a uniform geometric mean of the seen experts at total mass T = 2 is CALIBRATED (bits 2.161, entropy 2.09,
gap +0.07; T = 1 is under-confident by 1.2 bits of entropy, T = 3 over-confident) but still 0.13 above the chain
(2.030): mass right, shape wrong. Status: half-supported — the total is a count-based quantity; the allocation from
pairwise argmax statistics is not (the plug-in CMI of argmaxes is a crude stand-in for Allard's conditional
log-likelihood ratio, and the sets where nothing multiplicative beats the chain are the nested ones, where the exact
rule is "use the finest seen context", which `Chain.prob` already is).

**T6 — No function of an expert's OWN evidence (its context count N, its solo track record) is its exponent; the
exponent is an incremental-information quantity — a stacking weight — that only a cross-term can supply.**
Mathematics: Allard eq. 17 conditions on D_<i, the OTHER sources; Lindley's Ξ⁻¹ and Satopää's Σ⁻¹ are the
Gaussian forms of that conditioning; a per-expert confidence g(N) is a diagonal quantity.
Numbers (`expD_confidence_exponents.log`, 60 main-grid variants + temperature and bayes × conf families): no main-grid
cell is below 2.554 (chain 2.030); at T = 2 every g(N) is worse than uniform (N/(N+2) 2.224, N/(N+8) 2.306,
KT N/(N+35) 2.445 vs uniform 2.161) because N is ANTI-correlated with usefulness on this stream — the always-seen,
high-N experts are the weak ones (skip3 4.11, skip4 4.21, line 3.52 bits solo) and the chain's most informative
predictions (a deep context with a small count) get the lowest g; forcing the chain to full weight helps every
N/(N+k) row but never below 2.554. Multiplying the Bayesian weights by any confidence function only hurts (best
2.001 = E34's geo_bayes at T = 1, g = 1), and sharpening beyond T = 1 hurts (T = 1.5 → 2.028). The closed forms
WITH a cross-term have only been checked on the synthetic source (`pool_math_check.json`): Lindley/LDA 0.3373 /
0.3517 / 0.3496 and Satopää's partial-information rule 0.3333 / 0.3270 / 0.3262 against the exact 0.3320 / 0.3270 /
0.3262, with Satopää recovering the information structure from the forecasts alone (weights (−0.01, 0.09, 0.94)
nested, (1, 1, 1) disjoint, (0.5, 0.5, 0.5) overlap). Status: refuted for own-evidence functions; open for the
cross-term closed forms on Latin (not yet run — §I.3 item 1).

**T7 — A large part of what the mixers were asked to fix is the CALIBRATION of the chain's KT escape, and a count
table (PAQ's SSE/APM, Ranjan & Gneiting's recalibration by table) fixes it: one table on the chain alone beats the
whole gradient-free mixing library at the same slice.**
Mathematics: Ranjan & Gneiting Theorem 1; Allard Theorem 1 (a calibrated log-linear pool is asymptotically the
maximum-likelihood one); PAQ's SSE is a count table (`p += (y − p)/(n + δ)`, lpaq1 `APM`), gradient-free since PAQ2,
worth 2–3% in PAQ's Calgary history.
Numbers (`expC_sse_refinement.log`, prefix 300k; table = hits/n per (context, 24 buckets of stretch(p_raw) on
[−10, 6]), refined r = (hits + p_raw)/(n + 1), mixed p ∝ (1 − λ) p_raw + λ r, λ = 3/4): bucket only (`none`) 2.2097
(−0.019); `deep` −0.022; `char` −0.030; `conf` (deep, log2 N, one-symbol flag) −0.035; `cls` −0.037; `prev` −0.040;
`conf_cls` 2.1806 (−0.048, 2.2%); two-stage `cls_deep>char` 2.1793 (−0.0495, the best). Same data, unmodified
`e34.py --mixtures 1` (`expC_e34_mixtures_300k.json`): linear 2.2110 (−0.018), geometric 2.1907 (−0.038), product
8.558. The calibration table of the `none` context: characters at raw 1e−5 arrive at 1.1e−4 (10×); raw 0.57 arrives
at 0.65; by deepest order the SIGN flips (raw 0.41 → 0.22 at order 1, → 0.47 at order 8; raw 0.57 → 0.32 at order 1,
→ 0.69 at order 8) — the escape mass (V/2)/(N + V/2) = 35 pseudo-counts in `Chain.prob` is too pessimistic for deep
near-deterministic contexts and too optimistic at shallow ones, and a single global temperature cannot fix a
sign-flipping error, which is why "chain tempered alone" is best at a = 1.0 in `expA` while the `deep`/`conf` keys
separate the two regimes. The gains do not add (averaging two tables gives nothing over the best single; stacking
0.001–0.009; interpolation 0.006), the count cap hurts, and the relative gain shrinks slowly with data (2.6% at 20k
train → 2.2% at 300k). Status: supported; this is the APM-sized 2–3%, and it is largely a correction of a wrong
closed-form estimator inside the chain rather than a route to multiplicative combination.

**T8 (cross-problem) — Whether unit exponents (plain counting) are right is a property of the REDUNDANCY between
the evidence sources, not of their quality: the product with exponent 1 is right where the cues are near
conditionally independent given the target, and wrong for the 18 nested, cue-sharing text contexts.**
Numbers: frames, `r_window_neighbours.json`: naive Bayes over the 9 cells (P(y) Π_i P(w_i | y), 9 × 17 × 16 counts)
0.926 / 0.870 / 0.811 at 0.63 / 0.77 / 0.91 bits per cell — above the hard nearest rule at t ≥ 0.5, generalising to
every window, and NOT over-sharpened; the linear mixture of the same nine tables is last (0.889 / 0.822 / 0.792).
Text at ±3 (`expF_cluster_features_text.log`): the six-position naive-Bayes product 0.309 vs the exact-window table
0.317, equal on the chain (0.307), and as the last backoff behind the exact r3, r2 tables 0.320 (the best row,
+0.003, inside the ±0.003 draw noise). Same product, 18 nested contexts: 7.665 / 8.558. Status: supported by the
ordering in both domains; the text half is argmax accuracy, not bits.

### I.2 The ledger at prefix 300k / 60k (one slice, one book)

| predictor | file | bits/char | vs chain |
|---|---|---|---|
| chain alone | any | 2.229 | — |
| linear Bayesian mixture, 18 experts (E34 `Mixer`) | `expC_e34_mixtures_300k.json` | 2.211 | −0.018 |
| geometric, Bayesian exponents over seen experts (E34) | same | 2.191 | −0.038 |
| A1: Bayes over 11 temperatures, decaying switch | `mix_temperature_bayes_decay.json` | 2.184 | −0.045 |
| A2-min: Bayes over 42 (β_chain, β_rest) points, decaying switch | same | 2.168 | −0.061 |
| chain + word, per-context grid posterior (2 experts) | `expA_exponent_grid_fine.json` | 2.152 | −0.077 |
| naive Bayes, flat exponents of total 1.86, chain + word + match + line | `expB_naive_bayes_product.json` | 2.180 | −0.049 |
| SSE `conf_cls` on the chain alone | `expC_sse_refinement.json` | 2.181 | −0.048 |
| SSE two-stage `cls_deep>char` on the chain alone | same | 2.179 | −0.050 |
| raw product of seen experts | `expC_e34_mixtures_300k.json` | 8.558 | +6.3 |

At prefix 1.2M (`expA_exponent_grid_1200k.json`): chain 1.817; linear 1.814 / 1.823 / 1.807; per-context grid
posterior 1.776 / 1.790 / 1.791 (word / match / order4). At E34 full: chain 1.823, linear 1.820, geometric 1.801.
None of the closed-form rows have been combined (per-context exponents + SSE, word + match together), so the ledger's
rows are alternatives, not a sum.

### I.3 What remains unexplained (Part I)

1. **The stacking weight from counts.** The two closed forms that carry a cross-term — C2 (Lindley/LDA on the
   stretched predictions, per bit and mixing context, needs the outcome) and C3 (Satopää's diag(Σ)ᵀ Σ⁻¹ from the
   forecasts' covariance alone) — have not been run on the Latin stream; the synthetic check puts them within 0.03
   bits of the optimum. They need PAQ's binarisation of the 70-symbol alphabet (7 binary decisions) or a V-ary
   extension that is not designed.
2. **The information structure of the 18 experts is a measurement not yet made.** Heskes' pairwise symmetrised-KL
   matrix S_ab per mixing context (153 running means from the `logD` matrix `run_mixtures` already builds) would say
   how many effective independent sources the library holds and where any multiplicative gain can live; `expB`'s
   argmax-based r_agree / r_cmi are a proxy that T5 found uninformative for allocation.
3. **Additivity.** Whether SSE (T7) adds to the per-context geometric mixture (T4), and whether word's −0.041 and
   match's −0.027 add (chain + word + match on a 5 × 5 × 5 per-context grid) — unmeasured.
4. **The escape inside the chain.** T7's calibration table says (V/2)/(N + V/2) is the wrong estimator; a counted
   escape (PPMD's, or Bloom's SEE keyed by (order, log2 N, distinct symbols)) inside `Chain.prob` would reduce what the
   SSE and the partner experts have to fix, and `expA`'s refuted prediction ("b ≈ 0 for order4"; measured b = 0.3)
   says the chain's KT blend is not the right combination of its own levels. Not built.
5. **Symbol-level vs bit-level.** One shared exponent per expert across 70 symbols may be too coarse (Allard's
   per-outcome factor ν(A) was best for K > 2); a 2-D SSE keyed by two experts' buckets is a count-based pairwise
   combination function that has not been tried.
6. **The Bayesian GLN (A3).** A tree of 17 two-input grid mixers, each a Bayesian mixture over a 15 × 15 grid of
   exponent pairs, represents any product-form exponent vector at ~0.02 bits/char of prior cost; whether greedy
   per-node posteriors reach what a jointly fitted 18-vector reaches has no bound and has not been run.
7. **Switching rate.** CTS's lesson (the global rate n⁻¹ beat the per-context n_c⁻¹) has no analogue yet for a
   posterior over exponents; `expA` found the exponents stationary within the book (no-share best), `mix_temperature`
   found the decaying rate worth 0.015 — the two are consistent (both say: forget slowly) but the right rate is
   not designed.
8. **How much is at stake, exactly.** The learned-exponent reference number is not to be built on and has not been
   run; Mattern's same-experts comparison predicts 3.6–4.4% (E34: 1.820 → ≈ 1.75). The closed-form ladder has
   reached 2.7% (A2-min, 300k) and 2.3% (per-context two-expert grid, 1.2M) — the same order, not yet combined.

### I.4 What a gradient would be doing that the closed forms so far do not

The exponent vector's maximum-likelihood problem per mixing context is strictly convex in 18 dimensions (Mattern
2012 §3.2). PAQ's online gradient solves it continuously; its solution is the joint redundancy structure of the
expert set per context (T6), i.e. the stacking weights. The closed forms measured here approximate that joint object
by (a) a posterior over a LOW-dimensional grid of exponent vectors (2 dimensions in `expA` and A2-min; the full
18-dimensional posterior integral is closed form but exponential in m — `bayesian_alternatives_to_learned_mixing.md`
§2B, prior cost ≈ 450 bits, the integral is the cost), (b) a count-based TOTAL with a flat or hand-shaped allocation
(T5), (c) a per-context recalibration table on the output (T7). None of them yet computes a cross-term between
experts from data. The two candidates that do (C2's Ξ⁻¹, C3's Σ⁻¹) are linear solves, not gradients, and are the
next thing to run; if they land near the grid posteriors, the remaining difference to a learned mixer is the
continuous per-symbol adaptation of a step-size method, and that would be the honest record. If they land near
Mattern's 4%, nothing is left for a gradient to do at this library size.

---

## Part II — Features inside a block without gradients

### II.0 What the survey fixed (from `gradient_free_features_and_denoisers.md`)

In every gradient-free feature method the features are fitted by a closed form or by counting — eigenvectors
(PCANet), fixed wavelets (ScatNet), k-means means or random exemplars (Coates), random spectral draws (Rahimi–Recht),
alternating SVD steps (K-SVD), a local rule whose fixed point is soft k-means (SoftHebb) — and the target enters only
at a SOLVED readout (least squares, an SVM, or a count table keyed on a discrete code: PCANet's block histogram). The
residual gap to backprop is 4–10 points on CIFAR-10/ImageNet-class tasks and 0.6 dB in Gaussian denoising (DnCNN vs
BM3D), and in both cases it is "features chosen FOR the target" and receptive field, not features per se. Non-local
means with kernel width h → 0 is the single nearest patch; Levin & Nadler 2011 put the limit of any non-parametric
denoiser at the neighbour density of its window (3 × 3: 99% of patches have > 2 000 neighbours; 9 × 9: 13% have none).

### II.1 The theories, each with its mathematics and its number

**F1 — E35's memory block is not a lookup table; at t ≥ 0.5 it is nearest-neighbour regression, and its accuracy is
the neighbour-density limit of a 9-cell window.**
Mathematics: `e35.NearestRule.predict_nearest` = Nadaraya–Watson on Hamming distance with h → 0; Levin & Nadler's
bound.
Numbers (`r_window_neighbours.json`, one shot, 30 640 training pairs): distinct training windows 21 029 / 28 944 /
30 509 at t = 0.25 / 0.5 / 0.75; held-out windows never stored 0.626 / 0.923 / 0.995; hard-nearest accuracy by
Hamming distance at t = 0.5: d = 0 → 1.000, 1 → 0.980, 2 → 0.888, 3 → 0.748, 4 → 0.641, 5 → 0.505.
`expE_nlm_block.log`, distance profile: at t = 0.5 only 7% of query windows are stored exactly and there are on
average 2.6 / 31 / 165 stored windows at d = 1 / 2 / 3; at t = 0.75, 0.3% stored, 0.2 / 3.5 / 31. Status: measured.

**F2 — Softening the kernel is a gradient-free generalisation by shared structure (literal cell overlap): a vote over
~15 windows sharing 6–8 of 9 cells averages the target noise of one stored window; the neighbour SET carries the gain,
not its weights, and the neighbourhood must stay local.**
Mathematics: non-local means proper, P(y | w) ∝ Σ_s exp(−d_H(w, s)/h) n_s(y); the full kernel at large h drifts to
the frame's colour prior.
Numbers: `r_window_neighbours.json`, soft kernel over all stored windows with h by leave-one-out likelihood on a
2 000-window training subsample (h = 0.5 / 0.75 / 0.75): 0.954 / 0.900 / 0.829 (+0.013 / +0.051 / +0.099 over hard),
at 0.35 / 0.69 / 0.87 bits per cell — ONE soft block (0.900 at t = 0.5) beats E35's whole 8-block hard chain (0.880).
`expE_nlm_block.json` (73 kernels; k = 1 reproduces E35's strict numbers exactly on all 405 corrupted frames):
k = 15, h = 2, count-weighted, chain 0.924 / 0.855 (+0.048 / +0.054 over k = 1); k = all, h = 0.5, one shot
0.903 / 0.816 (+0.050 / +0.082); majority vs count vs full-distribution weighting differ by ≤ 0.005; 66 of 72 soft
kernels beat k = 1 on both chain columns, the 6 that do not are all k = all with h = 2 (0.859 at t = 0.5, below
k = 1), where 99.5% of the kernel's mass lies beyond the 15 nearest windows (77% at h = 0.5, 97% at h = 1). Voting
only where E35 fell back to the nearest window captures most of the gain (0.916 / 0.842 vs 0.921 / 0.850). The
chain's lead over one shot shrinks under the soft kernel (t = 0.75: +0.067 at k = 1 → +0.036 at k = all, h = 0.5) but
stays positive at every kernel. Status: supported; k and h were selected on the held-out frames (the surface is
coarse and monotone, so the bias is ≤ 0.01), and the kernel keeps all ~30k entries and treats the 9 cells as equally
relevant — the two places where a feature would enter.

**F3 — A closed-form factored block (naive Bayes over the cells) generalises to every window without over-sharpening
because the cells are near conditionally independent given the centre — T8's frame half.**
Numbers: `r_window_neighbours.json` 0.926 / 0.870 / 0.811 (0.63 / 0.77 / 0.91 bits per cell), second to the soft
kernel, above the hard rule at t ≥ 0.5, from 9 × 17 × 16 counts instead of 30k stored windows; the linear mixture of
the same tables is last. Status: measured at first order; the Chow–Liu tree (pairwise MI among the 9 cells + target,
P = Π P(node | parent), exactly normalised) is the next closed-form step and is unrun.

**F4 — On text at ±3 the "shared structure" IS the relevance of a position, which falls by half per step outward;
E35's symmetric shrink already encodes it, target-blind codes (Hamming k-means, sign-quantised PCA) throw it away,
and MI-weighting from counts recovers most of it.**
Mathematics: PCANet's recipe (closed-form projection → discrete code → count table); RandNet's parity says the code
and the counts do the work; Coates: the encoder matters more than the dictionary.
Numbers (`expF_cluster_features_text.log`, spread 300k, mask rate 0.5, 29 952 masked characters, SE ≈ 0.003): the
exact key reproduces E35 (0.317 / 0.307 vs E35's 0.320 / 0.309); MI(position; target) = 0.10 / 0.20 / 0.43 bits at
±3 / ±2 / ±1; cluster-only keys km256 / 1024 / 4096: 0.176 / 0.199 / 0.233; with MI-weighted dimensions 0.229 / 0.266 /
0.291 (still rising at k = 4096); as the backoff behind the exact r3 key, on the 46% of positions the full window has
never seen: km4096 0.254, km-mi4096 0.339, the symmetric shrink 0.383; PCA sign codes pca16 0.184, pca32 0.213. The
COUNT of a key is not its reliability under masking: full-window hits are 0.261 accurate vs 0.383 for the shrunk
backoffs (frequent full windows are the MASK-heavy ones), so "exact when count ≥ 3, else cluster" is strictly worse
(0.198 vs 0.232). No key makes the chain beat one shot on text. Status: supported; on this task a learned coarse code
is a (coarser) approximation of the hand-chosen shrink, and a feature block on text at ±3 is not worth a diffusion
chain — it is worth an EXPERT over long contexts in the E34 library (candidates (b)1–2 of the survey: similarity-
weighted contexts, class-based contexts), which is not designed.

**F5 — The price is the open question, not the block.**
`arcgames.LocalRule._price` charges log2(V) + n·H(wrong/n) + wrong·log2(V − 1) per entry; under noise the cheapest
two-part description of "the cleaner cell" is the identity with exceptions paid at their rate, so E35's sleep pass
collapses block 4 from 28 811 entries / 115 735 bits to 16 entries / 26 968 bits and repair falls (chain 0.788 vs
0.880 at t = 0.5). The soft kernel's leave-one-out likelihood prices a stored window by what it predicts for its
NEIGHBOURS, which is the generalisation the entry price cannot see. Which price makes the sleep pass keep exactly
the windows the kernel needs, and whether a merged-key table (PCANet code, k-modes centroid) then compresses on its
own under the existing two-part code, is not designed.

### II.2 What remains unexplained (Part II)

1. **Sleep under the kernel** — drop a stored window when the k = 15 vote of the others already predicts its
   majority; entries kept vs repair. Unrun.
2. **The metric** — a Hamming distance that weights cell c by its mutual information with the target colour, from
   the block's own counts: the smallest "feature" a block can have, and the frame analogue of `expF`'s MI weighting
   that recovered 0.254 → 0.339 on text. Unrun.
3. **The receptive field** — at radius 2 (25 cells) the raw key is all-unseen, the Hamming kernel's distance is
   noise-dominated, and factored/coded features are where the survey says generalisation continues (Levin & Nadler's
   support argument; DnCNN's 0.6 dB). The crossover is unmeasured.
4. **Chain vs one shot** — part of the chain's E35 gain was averaging the hard kernel's variance (the lead shrinks
   from +0.067 to +0.036 at t = 0.75 under the soft kernel); whether the remainder is the intermediate targets adding
   context, and whether it survives a factored block, is unmeasured.
5. **Bits, not accuracy** — every text feature number is argmax accuracy; the sleep price judges by prequential
   bits, under which naive Bayes (~31k counts) vs the exact table (197 026 windows + 3 shrink tables) may rank
   differently. Unmeasured.
6. **Target-aware closed forms** — none of the surveyed methods fits the features TO the target without a gradient
   (LDANet's closed-form LDA filters gained nothing over PCA on MNIST); MI-weighting, Chow–Liu with the target as
   root, and information-gain splits from counts are the count-based candidates; batch EM for a mixture of
   categoricals (SoftHebb's fixed point without a learning rate) for better keys than k-modes. All unrun.
7. **Text as an expert** — similarity-weighted contexts (a kernel over SEEN contexts of the same order on the
   Jensen–Shannon distance of their next-character counts) and Brown-style class contexts; the cost per character
   (a kernel over all seen contexts of an order) and the distance to use are not designed.

### II.3 What a gradient would be doing that the closed forms so far do not

Fit the features to the target inside the block. The soft kernel (F2) and the factored block (F3) generalise by a
FIXED similarity (cell overlap; cell-wise independence); the codes of F4 by a target-blind projection. In the survey
that is exactly the residual gap (4–10 points, 0.6 dB). The count-based, target-aware moves available — MI-weighted
metric, Chow–Liu, information-gain splits, MI-weighted k-means already measured on text — have not been run on
frames; until they are, "features need a gradient" is neither supported nor refuted by our numbers. What the numbers
DO say is that the block's first failure was not features at all: a soft kernel with one grid-picked width beats the
whole hard chain, and the price, not the block, is what collapsed E35.

---

## 3. Next cheap experiments, in order

Each ≤ 5 minutes on CPU, no gradient, on the existing apparatus (`e34.py`, `e35.py`, the `research/` scripts).

1. **(I) The cross-term closed forms on Latin — C3 then C2.** Per mixing context, accumulate the covariance of the
   experts' stretched predictions on PAQ's 7-bit binarisation of the 70 symbols (or, first, of the top-1 probability
   with the rest rescaled); mix with w = diag(Σ)ᵀ Σ⁻¹ (C3, no outcome needed) and w = Ξ⁻¹(μ_1 − μ_0) (C2, counted
   class-conditional means). Compare with A2-min 2.168 and `expA` by-ctx at prefix 300k. This is the one test that
   can show a count-based JOINT redundancy correction (T6's gap).
2. **(I) Additivity.** SSE `conf_cls` / `cls_deep>char` (`expC`) on top of the per-context grid mixture's output
   (`expA` by-ctx, chain + word) at the same slice: if −0.048 and −0.041 add, they fix different things.
3. **(I) The Heskes matrix.** S_ab and each expert's mean code length per mixing context from `run_mixtures`' `logD`
   (~1 min): the effective number of independent sources per context, before any mixer is built; optionally solve
   the 18-variable QP for the shape.
4. **(I) The escape by counting.** Replace (V/2)/(N + V/2) in `Chain.prob` by a counted escape keyed by (order,
   log2 N, distinct symbols); re-measure the chain, the SSE's residual gain, and `expA`'s b for order4.
5. **(I) Three experts per context.** chain + word + match on a 5 × 5 × 5 per-context grid at 1.2M (split by
   partner to stay under 5 min): does −0.041 + −0.027 add? Then condition the posterior also on the partner's count
   bucket (0 / 1–3 / 4–15 / 16+).
6. **(I) A3.** The tree of two-input Bayesian grid mixers over all 18 experts, against A2-min and the C2/C3 numbers.
7. **(II) The MI-weighted metric** in `expE`'s kernel at k = 15, weights from the block's own counts; against uniform
   Hamming 0.924 / 0.855.
8. **(II) Select k, h by leave-one-out** on the training windows (remove the held-out selection), then **sleep under
   the kernel** and report entries kept vs repair — the price question made concrete.
9. **(II) Radius 2**: kernel vs naive Bayes vs Chow–Liu at 25 cells, t = 0.5 / 0.75, against radius 1.
10. **(II) Text in bits**: the exact table, the MI-weighted cluster keys and naive Bayes under prequential KT code
    length; then push MI-weighted k-means to 16k–64k with a (near, far) product-quantised key.
11. **(II) Text as an expert**: similarity-weighted order-k contexts (Jensen–Shannon on next-character counts) as one
    more expert in the E34 library, judged by the per-context grid mixture of item 5.
