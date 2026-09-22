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

Revised 2026-09-22 after the two round-1 adversarial checks (`research/round1_checks.txt`, `check:mathematics` and
`check:numbers`, with their scripts `check_math_refutations.py`, `check_text_slice_noise.py`,
`check_frames_apparatus.py` and their JSON): every PROBLEM/FIX there is applied below; the one new run this revision
made is `check_per_context_oracle.py` (the per-context hindsight comparator of T4 and the ledger). Sentences that a
fix requires a run for are marked "[pending …]".

Revised again 2026-09-22 (round 2) with four new runs, each with its `.py`, `.json` and `.log` under `research/`:
`expG_cross_term_exponents.py` (the cross-term closed forms C2/C3 on Latin: chain + word, chain + word + match,
chain + order4, and the M = 1 diagnostic in `expG_cross_term_exponents_chain.json`), `expH_additivity_and_escape.py`
(SSE on top of the per-context grid and of A2-min; the counted escape inside `Chain.prob` in five structures),
`expI_frames_chowliu_loo.py` (five held-out boards, the Chow–Liu tree, k and h by leave-one-out on the training
windows) and `expJ_effective_sources.py` (the Heskes/CMI/correlation matrices of the 18 experts per mixing context
and the mass/allocation rules they imply). The "[pending expI …]" tags of round 1 are resolved below; the two
pendings no run has answered (`expC --train_chars 20000`; the `pool_math_check` re-run with a wider grid) stay
tagged, and two new ones are added (the 1.2M per-context oracle; the 3-expert grid on the `cal_cls` chain). Theses
T4–T8 and F1–F3 carry a "Round 2" paragraph each; the ledger (§I.2) now holds the combined rows and the 2.117 floor;
§I.3 marks which items were run; §I.4 states, per regime, what a learned mixer would still be doing, from the
cross-term numbers; §3 is re-ordered. Every number below is from a completed run whose `.log` agrees with its `.json`
(checked for the four new runs while merging).

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

The 1.2M rows and E34's full run are on DIFFERENT test sets (the first 60k characters vs the whole 240 588-character
book), so the closeness of 1.817 to 1.823 says nothing about representativeness. What the slice does to a LEVEL and to
a DIFFERENCE was measured (`research/check_text_slice_noise.json`, the same trained chain + word on characters
0–60k and 60k–120k of the book, six 10k blocks each): the chain's level moves 2.229 → 2.263 (0.034) between the two
slices, while the paired differences move ≤ 0.005 — linear − chain −0.016 → −0.016 (per-block SE 0.006), α0 grid
posterior − chain −0.057 → −0.056 (SE 0.002 / 0.003), per-context grid posterior − chain −0.077 → −0.072 (SE 0.005 /
0.007). So: levels are slice-specific to ±0.03; a PAIRED difference on one slice is good to ≈ 0.005 (its per-block
SE is the number to quote), and every gain below is a paired difference against the chain on the SAME slice. All runs
are deterministic, one book, one slice, no seeds. The 300k rows are 0.4 bits/char above the 1.2M rows across the
board and serve for ORDER, not level.

Frame numbers are cell accuracy on E35's 135 held-out level-2 frames (16 colours, radius-1 windows = 9 cells),
corrupted at t = 0.25 / 0.5 / 0.75, repaired by a one-shot block or by the diffusion chain (`e35.corrupt_chain`,
schedule t_k = k/8). E35 strict (every window kept): one shot 0.940 / 0.854 / 0.732, chain 0.946 / 0.880 / 0.800
(`runs/e35/e35_strict.json`). **The held-out set is TWO boards** (`research/check_frames_apparatus.json`): LockPath L2
= 85 frames of one 8 × 11 board (pairwise cell agreement between frames mean 0.962, min 0.955 — only the agent and
collectible cells vary) and CollectAll L2 = 50 frames of one 7 × 11 board (0.959 / 0.922); the training set is four
boards (LockPath L0 7 × 9, L1 7 × 11; CollectAll L0 6 × 9, L1 7 × 10; 115 frames). Every frame accuracy in Part II is
therefore an n = 2 boards result. Its noise: a paired frame bootstrap (seed 0, `check_frames_apparatus.py`) gives
soft − hard +0.014 [0.010, 0.018] / +0.058 [0.052, 0.064] / +0.093 [0.086, 0.101] at t = 0.25 / 0.5 / 0.75, and both
orderings used below (soft > hard at every t; naive Bayes over 9 cells > hard at t ≥ 0.5) hold on EACH board
separately (LockPath: hard 0.937 / 0.846 / 0.736, soft 0.955 / 0.905 / 0.834, nb9 0.921 / 0.870 / 0.822; CollectAll:
hard 0.947 / 0.853 / 0.745, soft 0.954 / 0.909 / 0.828, nb9 0.936 / 0.883 / 0.816). The trivial baselines every frame
number is to be read against: the colour prior alone (always the most frequent colour, 0) 0.609; the 1-cell table
P(y | centre colour) 0.863 / 0.778 / 0.683 — the window adds 0.09 / 0.13 / 0.15 over the centre cell.

**Round 2: five held-out boards** (`research/expI_frames_chowliu_loo.json`, same four training boards, 115 frames,
target palette {0..5}): LockPath L2 (85 frames, 8 × 11), LockPath L3 (63, 7 × 13; collected as level 0 of
`LockPath(levels=[_LEVELS[3]])` because `play()` never solves L2), CollectAll L2 (50, 7 × 11), MultiKey L1 (18,
5 × 13) and Toggle L0 (17, 5 × 9) — 233 frames × 3 draws = 699 corrupted frames per t, 56 994 cells; MultiKey and
Toggle are GAMES never trained on. Its apparatus reproduces `check_frames_apparatus.json` on LP2 + CA2 within 0.007
(different corruption draws: hard 0.938 / 0.846 / 0.732 vs 0.941 / 0.848 / 0.739; nb9 0.927 / 0.873 / 0.817 vs
0.926 / 0.874 / 0.820; soft k = all, h = 0.5: 0.953 / 0.907 / 0.834 vs 0.954 / 0.906 / 0.832). Every Part II number
marked "5 boards" is pooled cell accuracy over these five, with a paired frame bootstrap (2000 resamples) in brackets
and the board-level mean ± SE (n = 5) with the sign count — the board-level SE is the honest error, since each board
is one layout (per-board SE over frames 0.001–0.010 understates it) and several differences flip sign on MultiKey L1
or Toggle L0. Trivial baselines on the five boards: colour prior 0.577 pooled (0.614 / 0.553 / 0.598 / 0.485 / 0.437
per board, LP2 / LP3 / CA2 / MK1 / TG0); centre table 0.850 / 0.756 / 0.661; copy input 0.767 / 0.531 / 0.296. A
shared ceiling: 2.1% of held-out cells (LP2 2.3%, LP3 3.3%, CA2 0%, MK1 1.9%, TG0 2.0%) have a clean colour that never
occurs as a training target and are unreachable for every table-based predictor; they cost 10–27 bits each and
dominate every all-cells bits number, so calibration is read on the in-palette columns.

Process caveat recorded by the pool-math reader: to stop an over-long first run of `pool_math_check.py` it killed
every `python.exe` (six PIDs) at about 13:14; the JSON and log timestamps of `expB` (13:14) and `expC`, `expD`, `expE`,
`expF` (13:05–13:25) are after or around that moment. `expB_naive_bayes_product.log` shows ONE completed run (total
276 s) whose numbers agree with its JSON, so the numbers used here are from completed runs.

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
  pointwise form (Genest 1984b). With exponents on the simplex the geometric pool CAN give the true symbol more
  probability than every component, through its normaliser, whenever V > 2 (Mattern 2013 Example 4.8: w = (1/2, 1/2),
  two experts that agree on symbol 1 with probability q and put the rest on different symbols; as ε → 0 the pool's
  probability of symbol 1 exceeds q; `research/check_math_refutations.json` key `ex48_V70`, V = 70, q = 0.6, ε = 10⁻³:
  the pool gives symbol 1 probability 0.994 vs the best expert's 0.6, entropy 0.063 bits vs 0.978 for each component).
  On a BINARY alphabet with simplex exponents it cannot: the pooled logit is a convex combination of the component
  logits, so the pool sits between the components on the logit scale (key `binary_simplex_geo_exceeds_pmax_count`: 0 of
  10 000 random cases exceed p_max). S_w = Σ w_i > 1 is a second, separate sharpening (for Gaussian components with
  precisions τ_i the log pool has precision Σ w_i τ_i: a convex combination on the simplex, more than the sharpest
  component only when S_w > 1 — my derivation, not a citation).
- **The exact exponent** of a source (Allard, Comunian & Renard 2012 eq. 17):
  w_i = ln P(D_i | A, D_<i) / ln P(D_i | A) — the conditional-to-marginal log-likelihood ratio of source i given the
  sources already counted: 1 if conditionally independent, 0 if a function of the others. Conditional independence
  ⇔ all exponents 1 with the prior to the power 1 − n (their Prop. 2). S_w is NOT a count of sources: it depends on
  the scale the pool is written in. On `pool_math_check`'s `disjoint` structure (three sources with disjoint
  information, latent z = u_1 + … + u_4, y = 1[z > 0]) the exact posterior is a geometric pool of the calibrated
  sources on the PROBIT scale with exponents √(1 − v_i)/√v_4 = (2.55, 2.65, 2.74), S_w = 7.9
  (`check_math_refutations.json` key `disjoint_Sw_two_scales`: bits_probit_pool = bits_exact = 0.3300), while the SAME
  sources combine with Satopää weights (1.003, 0.998, 1.003), S_w = 3.0, on the conditional-expectation scale. What
  is invariant is the pooled distribution, not S_w. (Clemen & Winkler 1985's k/(1 + (k−1)ρ) is the precision of a
  linear combination of k equicorrelated point estimates — a different object — and is not cited for S_w here.)
- **Closed forms that carry a cross-term:** Lindley's supra-Bayesian model (Genest & Zidek eq. 4.1): exponents
  w = Ξ⁻¹(μ_1 − μ_0) on the experts' log-odds (LDA); the partial-information aggregator w = diag(Σ)ᵀ Σ⁻¹ from the
  forecasts' covariance alone (Satopää & Ungar 2015 eq. 3 for real-valued forecasts; the probit-scale construction
  for PROBABILITY forecasts, δ_i = Var(s_i)/(1 + Var(s_i)), X_i = s_i √(1 − δ_i), is Satopää, Pemantle & Ungar 2016,
  JASA 111(516)). Both discount co-varying experts through the inverse covariance. The "Σ from the forecasts alone"
  property holds only under that model's reliability assumption Cov(X_i, Y) = Var(X_i) — each forecast a calibrated
  conditional expectation — which T7 shows the KT experts violate per context (they are calibrated only in the aggregate).
- **No closed-form combination rule exists** (Mahoney DCE §4.3: "Probability theory does not answer the question");
  what can be inferred by counting is a posterior over a FAMILY of rules — a Bayesian mixture over a finite grid G of
  exponent vectors costs at most log2|G| bits over the best grid point on any sequence (the dominance inequality,
  CTS eq. 3; `bayesian_alternatives_to_learned_mixing.md` §2A), with no step size.
- **How much is at stake:** the only published same-experts comparison (Mattern 2012 Table 1, Calgary, 8 experts)
  measures learned SIMPLEX exponents on BINARY decisions: his §5 states "After a weight update we ensure that
  w ≥ ε·1_m and wᵀ1_m = 1" (GEO's exponents renormalised to the simplex after every step, ε = 2⁻³⁰) and the symbols
  are processed by an alphabet decomposition into ⌈log|X|⌉ binary steps — on which a simplex geometric pool is pure
  re-shaping in logit space (the bullet above). Relative to BETA (Bayesian linear weights) GEO is 3.4% better on
  average and 4.4% on book1 (`check_math_refutations.json` keys `mattern_avg_gap_rel_BETA` 0.0344,
  `mattern_book1_gap_rel_BETA` 0.0437; one base, BETA, for both). That is the same class as E34's `geometric`
  (Bayesian weights renormalised over the seen experts), which already recovers 1.2% of it on the full book
  (1.820 → 1.801). What FREE exponents (S_w ≠ 1) are worth has NO published same-experts comparison. On enwik8
  count-based ppmonstr (1.524) is below lpaq1's learned mixer (1.580); the step to 1.2–1.3 is the model library,
  bit-level modelling, dictionaries and SSE, not the mixer (`context_mixing_paq_mattern.md` §4). So E34's target was
  mis-stated: 1.3 is not the mixer's to reach. The figure "a learned mixer over the same 18 experts should land near
  1.75" is an UNMEASURED extrapolation of Mattern's Calgary percentages to E34's 1.820, not a target (§I.3 item 8).

### I.1 The theories, each with its mathematics and its number

**T1 (empirical) — At three slices, the linear Bayesian mixture over these experts gains ≤ 0.018 bits/char over
the chain (≤ 0.028 on one 4-expert subset).**
Mathematics, with its scope: Mattern 2013 Remark 4.7 is a PER-SYMBOL ceiling (max_w lin(x_t; w, P_t) = p_max(x_t;
P_t) at each t) and bounds nothing cumulative — `check_math_refutations.json` key `remark47_per_symbol_only` has two
experts alternating in quality at 1.676 bits/symbol EACH while E34's exact fixed-share `Mixer` rule codes the stream at
0.050 bits/symbol. Ranjan & Gneiting 2010 Theorem 1 (a weighted average of distinct CALIBRATED forecasts is
uncalibrated and beaten by its recalibration) presupposes calibrated components; T7 shows the chain is miscalibrated
with a sign that flips by depth, so the theorem's premise fails here. Minka's concentration argument is for pure
Bayes; `Mixer` runs a fixed share 0.02 and never concentrates (mean weight on the chain 0.60 on the full book,
0.84–0.90 with two experts). None of the three is a proof about THIS mixture; the thesis is the measurement.
Numbers: E34 full: linear 1.820 vs chain 1.823. `expA_exponent_grid_1200k.json`: the 2-expert `Mixer` gives
−0.003 (word), +0.006 (match — worse than the chain), −0.010 (order4); its mean weight on the chain is 0.84–0.90
(`expA_exponent_grid_1200k.log`). `expB_naive_bayes_product.json`: `linear` −0.011 / −0.028 / −0.018 on the diverse /
redundant / all18 sets at prefix 300k; `check_text_slice_noise.json`: linear − chain −0.016 (per-block SE 0.006) on
both 60k slices. The switching prior matters at this scale: the measured cost of fixed share 0.02 against the
Veness 2012 decaying rate α_t = 1/(t+1) is 0.014–0.015 bits/char on the grid posteriors
(`mix_temperature_bayes_decay.json` vs `mix_temperature_bayes.json`: 2.198 → 2.184 for A1, 2.183 → 2.168 for A2-min),
the same size as the gains being measured. (The Herbster–Warmuth slack term −log2(0.98) = 0.029 bits/char is an
upper bound on regret, not a realised loss, and is not comparable to a realised gain.) What the deficit IS: the
chain's escape is miscalibrated (T7), and a count table on the chain alone recovers 0.048–0.050 — more than the
linear mixture's 0.018 — but whether the POOL's residual deficit is a calibration deficit was not measured (T7
measures the calibration of one component, the chain, not of the mixture). Status: holds everywhere it was measured.

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
Mathematics: the dominance inequality, total bits ≤ min_g bits(g) + log2|G|; for ONE global posterior over |G| = 63
and n = 60 000 that is 0.0001 bits/char (`check_math_refutations.json` key `arithmetic`). A PER-CONTEXT posterior
pays it once per context — n_ctx · log2|G| / n = 0.0032 bits/char for 36 × 42 (or 32 used contexts × 63) at 60k —
and its comparator is the best per-context fixed ASSIGNMENT in hindsight (36 independent argmins), not the best
global point (see T4).
Numbers: `expA_exponent_grid_1200k.json`, the no-share posterior (`grid_bayes_alpha0`, `GridPosterior(K, 1, alpha=0)`)
sits within 0.0002 of the best fixed grid point in all nine runs (word 1.784 vs 1.784; match 1.805 vs 1.805; order4
1.807 vs 1.807), and its favourite (`posterior_alpha0_favourite`, weight 1.000) IS the best fixed point in all nine
runs (coarse, fine and 1.2M grids × word, match, order4). The share-0.01 posterior (`posterior_favourite_end`) is
nearly flat — favourite weight 0.05–0.30 — and its favourite matches the best fixed point in only 3 of 9 runs (0 of 3
at 1.2M), so the end-of-pass favourite of a fixed-share posterior is not a read-out of the best point; its BITS are
(within 0.0002). `mix_temperature_bayes_decay.json` on the 18 experts: Bayes over 11 temperatures 2.184, over the
42-point (β_chain, β_rest) grid 2.168, vs E34's geometric 2.191, linear 2.211, chain 2.229 (−0.061, −2.7%; per-block
SE 0.004, `check_per_context_oracle.json` part B). The share-0.01 posterior costs 0.002–0.006 over the no-share one at
60k characters (`grid_bayes` − `grid_bayes_alpha0` = 0.0024–0.0057 across the nine expA runs): the exponents are
stationary within the book. Status: confirmed for the selection part; what a single exponent vector cannot do is T4.

**T4 — What the exponents encode is CONDITIONAL on the mixing context; a per-context grid posterior (36 × K closed-
form weights) recovers what neither a global exponent pair nor a global temperature can.**
Mathematics: Allard's exact exponent is one number per combination (A, D_1..D_n); "one exponent per source" is the
simplification (their eq. 18); PAQ selects a weight set by a small context (Mahoney; GLN's "data conditioning");
the KT count estimate makes a shorter context informative exactly when the longer context's count N is small
(`opinion_pools_and_products_of_experts.md` §11.5), which is what E34's mixing context `s.cls() * 9 + deep` indexes.
Numbers (`expA_exponent_grid_1200k.json`, key `grid_bayes_by_ctx`, `GridPosterior(K, N_CTX)`): chain + word 1.776
(−0.041 vs the chain) against the best FIXED pair 1.784 (−0.033) and the linear mixer 1.814 (−0.003); chain + match
1.790 (−0.027) against −0.012 fixed and +0.006 linear; chain + order4 1.791 (−0.026) against −0.010 fixed and
−0.010 linear. The comparator of a per-context posterior is the best per-context ASSIGNMENT in hindsight, with the
prior cost n_ctx · log2|G| / n added (`research/check_per_context_oracle.py` / `.json`, prefix 300k, 60k, six 10k
blocks; the script re-runs the two passes unchanged and adds a (36 × K) accumulator — it reproduces
`grid_bayes_by_ctx` 2.1516 and A2-min 2.1678 exactly): chain + word on the fine 63-point grid — per-context oracle
2.144 (32 contexts used) + prior 0.0032 = 2.147, the share-0.01 posterior 2.152, posterior − oracle +0.0076 ± 0.0006
(2.4 × the prior bound: the fixed share is not the pure Bayes the bound is for), posterior − best GLOBAL point (0.7,
0.5) −0.020 ± 0.003, oracle − global −0.028 ± 0.003; A2-min (42-point grid, 18 experts, decaying switch) — oracle
2.166 + prior 0.0029 = 2.169, posterior 2.168, posterior − oracle +0.0018 ± 0.0013 (inside the prior bound),
posterior − best global point (0.75, 0.5) −0.035 ± 0.004. So the per-context posteriors sit within 0.002–0.008 of
their own comparator, and the "gain over the best fixed point" that the headline rows carry (−0.020 chain + word,
−0.035 A2-min) is what CONDITIONING on the context is worth, not a posterior beating its oracle. The posterior-mean
temperature differs by mixing context — 1.0 at deepest order 3 and ≈ 1.25 at orders 4–8 in one class, 1.0–1.05 in
another (`posterior_mean_beta_by_ctx`, decay run; 1.1–2.0 under fixed share). The total exponent MASS is not the
missing degree of freedom: the best single temperature β on the Bayesian-weight geometric mixture is 1.0 (0.75 →
2.285, 1.25 → 2.193, 1.5 → 2.252). Status: supported. In kind, two experts with context-conditioned closed-form
exponents (−0.041 at 1.2M, −2.3%) reach what E34's 18-expert geometric mixture reached on the full book (−0.022,
−1.2%) — different slices, so a comparison of kind, not of level.
Round 2 (`expG_cross_term_exponents.json`, `expG_cross_term_exponents_chain.json`, `expH_additivity_and_escape.json`,
all prefix 300k / 60k, paired vs the chain 2.2288 with the SE over six 10k blocks): (a) **Half of the per-context
gain is a per-context TEMPERATURE of the chain alone.** With M = 1 (10 temperatures per mixing context, prior bound
36 · log2 10 / 60 000 = 0.0020) the per-context posterior is 2.1896, −0.0392 (SE 0.0051), its per-context oracle
−0.0434; the best GLOBAL temperature is 1.0 (= `expA`'s "chain tempered alone"), so the whole −0.039 is
conditioning — the "sign-flipping" escape error of T7 fixed by one exponent per (class × deepest order). Chain + word's
−0.0773 is that −0.039 plus the partner's marginal −0.038. (b) **Word and match partly add.** Chain + word + match on
a 336-point per-context grid (b_word ≤ 0.7, b_match ≤ 0.5; prior bound 0.0050): posterior 2.1281, −0.1007 (SE
0.0052), per-context oracle 2.1174; vs the 2-expert posterior −0.0235 (word alone −0.077, match alone −0.073 in
`expA`; the sum would be −0.150). Its oracle exponents (visit-weighted, contexts ≥ 500 visits) are (0.69, 0.38, 0.14),
sum 1.22; the 2-expert oracle's are (0.76, 0.44), sum 1.20. (c) **The pool's scale is a separate object.** The same
grid on the ODDS scale (exponents on stretch(p), renormalised — PAQ's form, `gridS_*`) is worse than on the
probability scale: posterior 2.1624 vs 2.1516 (+0.0109, SE 0.0010) for chain + word, 2.1410 vs 2.1281 (+0.0129) for
three experts; the odds-scale oracles want a smaller sum (0.97 vs 1.20; three experts 1.03 vs 1.22). (d) **On a
chain whose escape is counted (T7 round 2) the grid's level improves and its relative gain shrinks:** grid on the
`cal_cls` chain 2.1190 (−0.0323, SE 0.0017, vs the grid on the KT chain 2.1513), the grid's own gain −0.0703 vs
−0.0775 on KT, and the posterior-mean chain exponent over the 11 contexts with ≥ 500 test characters moves from
a = 0.73 (KT) to a = 0.62 (`cal`) — AWAY from 1 once the escape is counted, so the tempering of the chain is not
only escape calibration. Status: supported and decomposed — conditioning is worth −0.039 (chain alone) + −0.038
(word) + −0.024 (match) on this slice, in the order the ledger now carries.

**T5 — The SUM of the exponents is the one number a count-based redundancy measure supplies, on a given scale; the
pairwise ALLOCATION it suggests mostly carries no information.**
Mathematics: S_w depends on the scale the pool is written in, and is not a count of sources: the `disjoint`
structure needs S_w = 7.9 on the probit scale (the exact posterior, analytic exponents (2.55, 2.65, 2.74) — the
(3, 3, 3), S_w = 9 that `pool_math_check.json` reported was the corner of its grid, capped at 3.0 per exponent, key
`pool_math_check_free_grid_cap_per_exponent`, and sits at 0.3309 bits vs the exact posterior's 0.3270 in that file; in
`check_math_refutations.py`'s own draw the probit pool with the analytic exponents equals the exact posterior,
0.3300 = 0.3300 — a re-run of `pool_math_check.py` with `free_grid(M, step=0.5, top=6.0)` so that the disjoint
optimum is interior is not done [pending pool_math_check re-run with `free_grid(M, step=0.5, top=6.0)`])
and S_w = 3.0 on Satopää's conditional-expectation scale (weights (1.003, 0.998, 1.003)), the same sources and the
same information structure (`check_math_refutations.json` key `disjoint_Sw_two_scales`); the overlap structure wants
S_w = 3 on the grid's scale. What is invariant is the pooled distribution, not S_w.
Numbers: `expA_exponent_grid_1200k.json` (`grid_bits`), the a + b diagonals: word wants a + b = 1.1–1.2 (1.784 /
1.784 vs 1.791 at a + b = 1.0 and 1.792 at 1.3) — the word since its boundary reaches up to 12 characters past the
chain's 8, so it holds some independent evidence; match wants a + b = 1.0 (1.805 at 1.0, 1.808 at 1.1, 1.817 at
1.2 — the 1.0/1.1 gap is below the 0.005 noise floor); order4 wants 1.0–1.1 (indistinguishable, 1.808 / 1.807) — a
re-weighting of the chain's own backoff, not new evidence. In every run the chain wants TEMPERING (a = 0.7–0.9) once
any partner is multiplied in, while alone a = 1.0 is best: the partner's correlated share has to be taken out of the
chain. `expB_naive_bayes_product.log`: with exponents λ_m = 1/(1 + Σ_j r(m, j)) from counted argmax agreement
(`nb_agree`) the sums land at 1.86 / 1.49 / 2.41 (diverse / redundant / all18), against the CONTROL `nb_flat` with the
same total spread uniformly, both smoothings, all three sets in one table (bits/char, agree / flat):

| set | kt smoothing | prior smoothing |
|---|---|---|
| diverse (chain + word + match + line) | 2.190 / 2.180 | 2.207 / 2.193 |
| redundant (chain + order2 + order4) | 2.284 / 2.278 | 2.299 / 2.291 |
| all18 | 2.467 / 2.527 | 2.324 / 2.286 |

5 of 6 (set × smoothing) comparisons favour flat; the one exception (all18, kt) favours the allocation by 0.060, the
largest difference in the table. `expD` (spread 300k): a uniform geometric mean of the seen experts at total mass
T = 2 is CALIBRATED (bits 2.161, entropy 2.09, gap +0.07; T = 1 is under-confident by 1.2 bits of entropy, T = 3
over-confident) but still 0.13 above the chain (2.030): mass right, shape wrong. (The same uniform-at-mass-2 pool on
the standard slice is 2.3554, +0.127 ± 0.005 vs the chain 2.2288 — `expJ_effective_sources.json`, `flat@m2`: the
same paired difference as `expD`'s +0.131, a different level, the slice.)
Round 2 (`expJ_effective_sources.json`; statistics from the last 100k characters of the training pass, 10 mixing
contexts with ≥ 1000 positions covering 95.4% of positions, the other 26 on the global matrix; test = the standard
60k, paired, SE over 6 blocks). **First half — "the total is a count-based quantity" — REFUTED as tested.** The same
18 experts give k_eff = M/(1 + (M − 1) r̄) of 1.62 from the shuffle-corrected key-CMI, 2.64 from the code-length
correlation partialled on the next character, 4.30 from `expB`'s argmax-CMI, and 3.72 from 1ᵀR⁻¹1 (ridge 0.05) —
the scale-dependence of S_w measured on the library itself — and none is the calibrated mass: at flat allocation
the calibrated mass is 2 (bits − entropy +0.15; m1 −1.09, key 1.62 −0.24, codelen 2.64 +0.80, inverse 3.72 +1.66,
argmax 4.30 +2.11) and all three count-based masses code worse than 2 (2.455 / 2.557 / 3.211 vs 2.355); at the
allocation that keeps only chain + match the calibrated mass is ≈ 1 (2.1666 at m1, gap −0.20; 2.297 at 1.62, gap
+0.63). The mass that calibrates changes with the allocation: mass and allocation are one joint object, not two
numbers. **Second half — "the allocation from a pairwise matrix carries no information" — 1 of 3 matrices carries
one bit.** The PRIMARY comparison (fixed before the run: the task's formula e_m ∝ gain_m · (1 − max_j r(m, j)) at
mass k_eff(r), against flat at the SAME mass): argmax-CMI sym 3.4214 vs flat 3.2114, +0.210 ± 0.002 (allocation
WORSE); codelen-corr 2.6171 vs 2.5571, +0.060 ± 0.004 (WORSE); key-CMI 2.2968 vs 2.4552, −0.158 ± 0.014 (better,
both above the chain). The one win is the single bit a symmetric statistic carries with certainty — deterministic
NESTING, r = 1.00 exactly for key-function pairs (order1–order2, skip3–order3, pair1_3–order3, line–order1,
word–order1), which zeroes orders 1–6, the skips, the pairs, word and line and leaves chain + match; the graded
values (word–match 0.16, chain–match 0.21–0.44, skip3–skip4 0.04–0.26 depending on the matrix) allocate worse than
flat. The best secondary row (selected on the slice, 48 rows), `sym_key_cmi@m1` 2.1666 (−0.062 ± 0.005; mean
exponents chain 0.69, match 0.19, order1 / word / line 0.03), is A2-min's 2.1679 (−0.061 ± 0.004) — two count-based
routes to "chain + one independent partner", both 0.015 above the 2-expert per-context posterior 2.152. A caveat that
matters: under key-CMI the chain, order7, order8 and prevword are UNRESOLVABLE (70k–87k distinct keys in 100k
positions; the raw plug-in reads 1.00 for them and their null reads ≈ 1), so max_j r = 0 for them by default — right
for the chain (the superset), and order7/8/prevword survive only because their gain over the unigram is negative
(−0.43 / −0.60 / −0.31 bits); a rule for unresolvable pairs is not designed. Side result: Heskes' quadratic
ambiguity (eq. 4–5) overstates the exact ambiguity −log2 Z of the uniform simplex pool 2× on these 70-ary
distributions (1.498 vs 0.740 bits; per context the predicted-minus-measured gap is 0.5–0.9 in all 10), so the
Heskes QP (C5) would optimise the wrong objective on text. What the cross-term solve of T6 (round 2) adds on the sum:
on diverse pairs the stretch-LDA's visit-weighted sum is 1.22 (grid oracle 1.20; three experts 1.39 vs 1.22), on the
nested pair 1.60 vs 1.13 — the sum is recovered exactly where the allocation is, and lost with it. Status: the first
half is refuted (no count-based mass rule survives; the calibrated mass is measurable online from bits − entropy and
depends on the allocation), the second half is sharpened — pairwise symmetric statistics supply "which experts are
functions of which" and nothing more. Whether `expB`'s all18/kt exception (+0.060 for the allocation) is that same
nesting bit acting through `nb_agree`'s argmax statistic is consistent with `expJ` but was not measured (`expB`'s
r_agree and `expJ`'s key-CMI are different statistics on different windows).

**T6 — No function of an expert's OWN evidence (its context count N, its solo track record) is its exponent; the
exponent is an incremental-information quantity — a stacking weight — that only a cross-term can supply.**
Mathematics: Allard eq. 17 conditions on D_<i, the OTHER sources; Lindley's Ξ⁻¹ and Satopää's Σ⁻¹ are the
Gaussian forms of that conditioning; a per-expert confidence g(N) is a diagonal quantity.
Numbers (`expD_confidence_exponents.log`, 60 main-grid variants + temperature and bayes × conf families): no main-grid
cell is below 2.554 (chain 2.030); at T = 2 every g(N) is worse than uniform (N/(N+2) 2.224, N/(N+8) 2.306,
KT N/(N+35) 2.445 vs uniform 2.161) because N is ANTI-correlated with usefulness on this stream — the always-seen,
high-N experts are the weak ones (skip3 4.11, skip4 4.21, line 3.52 bits solo) and the chain's most informative
predictions (a deep context with a small count) get the lowest g; forcing the chain to full weight helps only in
the q = 1 column (N/(N+2): 2.728 → 2.620; N/(N+8): 2.903 → 2.707) and HURTS at q = 0 and q = 0.5 (N/(N+2) q = 0:
8.214 → 8.469, q = 0.5: 2.662 → 2.693; N/(N+8) q = 0.5: 2.649 → 2.681, `subset=all` blocks of the log), never below
2.554. Multiplying the Bayesian weights by any confidence function only hurts (best
2.001 = E34's geo_bayes at T = 1, g = 1), and sharpening beyond T = 1 hurts (T = 1.5 → 2.028). The closed forms
WITH a cross-term have only been checked on the synthetic source (`pool_math_check.json`): Lindley/LDA 0.3373 /
0.3517 / 0.3496 and Satopää's partial-information rule 0.3333 / 0.3270 / 0.3262 against the exact 0.3320 / 0.3270 /
0.3262, with Satopää recovering the information structure from the forecasts alone (weights (−0.01, 0.09, 0.94)
nested, (1, 1, 1) disjoint, (0.5, 0.5, 0.5) overlap).
Round 2 — the cross-term closed forms ON LATIN (`expG_cross_term_exponents.json` / `.log`; prefix 300k / 60k;
paired vs the chain 2.2288, SE over six 10k blocks). The binary event is ONE-VS-REST: at each position every
character c is the event 1[x_t = c] (1 positive + 69 negatives); per expert the feature is the stretch
s_m(c) = ln(p_m(c)/(1 − p_m(c))) (C2) or Φ⁻¹(p_m(c)) (C3), forecasts clipped to [10⁻⁷, 1 − 10⁻⁷]; per mixing context
(E34's 36) the accumulators are sums, outer-product sums and counts of these features by class. C2 = LDA,
w = Ξ⁻¹(μ_1 − μ_0) with the EQUAL-pooled within-class covariance Ξ = (C_1 + C_0)/2 (`pool_math_check`'s form; the
count-pooled form, 69 : 1 toward the negatives, is a second row), ridge 0.01 · trace/M. C3 = Satopää, Pemantle &
Ungar 2016: δ_m = Var(s_m)/(1 + Var(s_m)), X_m = s_m √(1 − δ_m) + θ with θ = Φ⁻¹(1 − 1/70) = 2.189 (so that w = e_m
reproduces expert m exactly; the identity check passes, chain alone −0.002), w = Σ⁻¹ diag(Σ). Accumulation is
CAUSAL (every 4th training position, 75 000 samples, and every test position; a context's weights re-solved after
each of its first 16 test visits and then every 8th; identical to 4 decimals with lag 1). The SAME exponents are then
applied on three pools: `_onlog` = Π p_m(c)^{w_m}/Z (the probability pool, `expA`'s grid form), `_exp` =
exp(Σ w_m s_m(c))/Z (the odds pool = PAQ's squash-stretch renormalised, eq. 4.1's geometric pool) and `_sig` =
σ(b + wᵀs(c)) renormalised (the literal per-event posterior).

| pair (K of the grid) | grid posterior by ctx | per-ctx grid oracle | C2 stretch-LDA on the probability pool | same exponents, odds pool | C2 hindsight (fit on the test slice) | C3 probit | C3 test-only |
|---|---|---|---|---|---|---|---|
| chain + word (63) | 2.1516, −0.0773 (0.0053) | 2.1439, −0.0849 | **2.1495, −0.0793 (0.0072)**; vs posterior −0.0021 (0.0022) | 2.1982, −0.0306 (0.0087) | 2.1532, −0.0756 | 2.2148, −0.0140 (0.0018) | −0.0206 (0.0016) |
| chain + word + match (336) | 2.1281, −0.1007 (0.0052) | 2.1174, −0.1114 | **2.1426, −0.0862 (0.0067)**; vs posterior +0.0145 (0.0020) | 2.2303, +0.0015 (0.0085) | 2.1470, −0.0818 | 2.2056, −0.0232 (0.0048) | −0.0341 (0.0052) |
| chain + order4 (63), nested control | 2.1782, −0.0507 (0.0050) | 2.1703, −0.0585 | **2.2604, +0.0316 (0.0057)**, 0/6 blocks | 2.3807, +0.1518 (0.0076) | 2.3067, +0.0779 | 2.9645, +0.7357 (0.0233) | +0.6285 |
| chain alone (10 temperatures) | 2.1896, −0.0392 (0.0051) | 2.1854, −0.0434 | 2.2044, −0.0244 (0.0065) | **2.1874, −0.0414 (0.0070)**; vs posterior −0.0022 (0.0022) | −0.0336 (odds) / −0.0320 (prob.) | 2.2267, −0.0021 (w = 1.0, the identity) | −0.0021 |

Costs: the grid posteriors' prior bound 0.0036 / 0.0050 / 0.0036 / 0.0020 bits/char (36 · log2 K / 60 000); the
causal C2/C3 rows carry no prior charge (plug-in prequential; their state is 36 × (2(M + M(M + 1)/2) + 2) sums for
C2); the hindsight rows are oracles charged 36(M + 1) · ½ log2 60 000 = 0.0143 (M = 2) / 0.0190 (M = 3) / 0.0095
(M = 1). Three readings. (1) **On DIVERSE experts the second-moment solve recovers the exponents.** Per context the
stretch-LDA exponents sit within 0.06–0.08 of the grid's per-context hindsight optimum (chain + word: visit-weighted
(0.80, 0.42) vs the oracle's (0.76, 0.44), correlation across contexts 0.95 for the chain's exponent, 0.09 for
word's; three experts (0.70, 0.42, 0.27) vs (0.69, 0.38, 0.14), |diff| (0.066, 0.068, 0.164)); coded on the
probability pool they TIE the per-context grid posterior for chain + word (−0.0021 ± 0.0022) and reach 86% of it for
three experts (+0.0145 ± 0.0020), from 36 solves of a 2 × 2 / 3 × 3 system on counts, no grid; the hindsight fit is
no better than the causal one (2.1532 vs 2.1495), so 75 000 training samples were enough and the statistic's
asymptote is what it is. (2) **The pool's SCALE is a separate object the Gaussian model gets wrong.** The same
exponents on the odds pool that eq. 4.1 literally defines give −0.031 / +0.002 / +0.152 (the top character's odds
raised to the w-th power over-sharpen); the LDA's own read-out, the per-event sigmoid renormalised, is +0.104 /
+0.050 / +0.064 (count-pooled LDA is worse everywhere: +0.195 chain alone, +0.965 chain + order4). So the LDA
supplies the exponents' level and per-context pattern (the between/within ratio on the stretch scale) and WHICH pool
to raise them on is decided by the code length, not by the model; why the stretch-fitted level matches the
probability pool's optimum is measured, not derived (not designed). The one exception is M = 1: the chain's
per-context temperature is found by the LDA on the ODDS pool (w = 0.89, −0.0414, tying the temperature posterior
−0.0392), not on the probability pool (−0.0244); the log-probability feature fails there (w = 0.56, +0.161: too
skewed for LDA). (3) **On NESTED experts the cross-term fails.** For chain + order4 (order4 is a coarsening of the
chain; Allard eq. 17's exact exponent ≈ 0; the grid finds 0.26 with the chain at 0.87) the LDA gives order4 0.85 with
sum 1.60 and codes +0.032 ABOVE the chain (hindsight +0.078; the grid −0.051): Ξ⁻¹ discounts by the LINEAR
correlation of the stretched forecasts (0.55–0.84 per context), far from 1 for a deterministic coarsening whose
forecast differs from the chain's exactly where the deep counts are small — a Gaussian second moment cannot express
"a function of another source". C3 is worse there (+0.736, w = (1.24, −0.82): the extremisation subtracts the shared
information with the wrong sign) and on the diverse sets recovers only 1/5–1/3 of the grid (−0.014 / −0.023;
its reliability assumption Cov(X_m, Z) = Var(X_m) fails for KT tables per context, as I.0 said; with empirical
instead of model-implied centering it is +0.135 / +0.128). Status: refuted for own-evidence functions; for the
cross-term closed forms on Latin — CONFIRMED on 2–3 diverse experts (a linear solve on counted second moments is
within 0.002–0.015 of the per-context grid posterior) and REFUTED where the 18-expert library actually lives (nested
and cue-sharing experts, 8 of E34's 18 being orders 1–8 inside the chain), where the exact exponent is a conditional
log-likelihood ratio that second moments cannot see. Design choices fixed before the run and not tuned: equal-pooled
Ξ, ridge 0.01, P_CLIP 10⁻⁷, output floor 10⁻⁶, C3's Var(Z | X) floor 0.01 (never hit in 15 810 solves), the 3-expert
grid's caps (b_match ≤ 0.5, b_word ≤ 0.7, so its oracle is a lower bound on a finer grid); one slice, one book.

**T7 — A large part of what the mixers were asked to fix is the CALIBRATION of the chain's KT escape, and a count
table (PAQ's SSE/APM, Ranjan & Gneiting's recalibration by table) fixes it: one table on the chain alone (2.179)
beats E34's linear and geometric mixtures at the same slice (2.211 / 2.191) but not the per-context grid posterior
(2.152) or A2-min (2.168) — see the ledger, §I.2. Round 2: the miscalibration is CLASS-CONDITIONAL (word-start vs
inside-word), a counted escape keyed by the previous character's class recovers 82% of the SSE's gain inside
`Chain.prob`, and the SSE and the per-context exponents are ~90% the SAME operation (interaction +0.043 ± 0.005).**
Mathematics: Ranjan & Gneiting Theorem 1; Allard Theorem 1 (a calibrated log-linear pool is asymptotically the
maximum-likelihood one); PAQ's SSE is a count table (`p += (y − p)/(n + δ)`, lpaq1 `APM`), gradient-free since PAQ2,
worth 2–3% in PAQ's Calgary history.
Numbers (`expC_sse_refinement.log`, prefix 300k; table = hits/n per (context, 24 buckets of stretch(p_raw) on
[−10, 6]), refined r = (hits + p_raw)/(n + 1), mixed p ∝ (1 − λ) p_raw + λ r, λ = 3/4): bucket only (`none`) 2.2097
(−0.019); `deep` −0.022; `char` −0.030; `conf` (deep, log2 N, one-symbol flag) −0.035; `cls` −0.037; `prev` −0.040;
`conf_cls` 2.1806 (−0.048, 2.2%); two-stage `cls_deep>char` 2.1793 (−0.0495, the best). That "best" is the best of
90 variants (18 tables + 9 averages + 3 stacks, each at λ = 1/2, 3/4, 1) selected on the one 60k test slice, and its
0.0013 lead over `conf_cls` is below the 0.005 noise floor — the two are tied. Same data, unmodified
`e34.py --mixtures 1` (`expC_e34_mixtures_300k.json`): linear 2.2110 (−0.018), geometric 2.1907 (−0.038), product
8.558. The calibration table of the `none` context: characters at raw 1e−5 arrive at 1.1e−4 (10×); raw 0.57 arrives
at 0.65; by deepest order the SIGN flips (raw 0.41 → 0.22 at order 1, → 0.47 at order 8; raw 0.57 → 0.32 at order 1,
→ 0.69 at order 8) — the escape mass (V/2)/(N + V/2) = 35 pseudo-counts in `Chain.prob` is too pessimistic for deep
near-deterministic contexts and too optimistic at shallow ones, and a single global temperature cannot fix a
sign-flipping error, which is why "chain tempered alone" is best at a = 1.0 in `expA` while the `deep`/`conf` keys
separate the two regimes. The gains do not add (averaging two tables gives nothing over the best single; stacking
0.001–0.009; interpolation 0.006) and the count cap hurts. How the relative gain moves with training data has not
been measured (no artefact under `research/` or `runs/` has a 20k-character training run; an earlier "2.6% at 20k
train" here had no source and is withdrawn) [pending expC re-run: `--train_chars 20000`].
Round 2 (`expH_additivity_and_escape.json` / `.log`, prefix 300k / 60k, every arm paired against the KT chain
2.2288 with the SE over six 10k blocks; in-pass reproductions exact: linear 2.2129, `grid_fresh` 2.1516, chain + SSE
`conf_cls` 2.1806 and `cls_deep>char` 2.1793, A2-min fresh 2.1678). **(a) SSE and the per-context exponents do not
add.** SSE on top of the chain + word grid: `conf_cls` 2.1462, `cls_deep>char` 2.1462 (−0.0826, SE 0.0062, vs the
grid's own 2.1513); SSE's residual on the grid −0.0051 (SE 0.0010) / −0.0052 (SE 0.0016), against −0.0482 / −0.0495
on the chain; the interaction (measured minus the sum of the two solo gains) +0.0431 (SE 0.0052) / +0.0444 (SE
0.0058). On A2-min (SSE warmed on the last 240k training characters): a2 + SSE 2.1593 / 2.1533 (−0.0695 / −0.0755),
residual −0.0073 (SE 0.0019) / −0.0132 (SE 0.0023), interaction +0.0398 / +0.0375. So ~90% of what `conf_cls` fixes
on the chain the per-context exponent a on the chain (0.5–0.9, one per class × deepest-order context, renormalised —
a per-context temperature, T4 round 2 (a)) already fixes; what the SSE adds beyond it is the non-power-law bucket
shape, worth 0.005. (The per-character stack table is non-stationary: with a 60k warm-up instead of 240k,
a2 + `cls_deep>char` is 2.1489 (−0.0800) and chain + `cls_deep>char` 2.1716 (−0.0572) — better with LESS warm-up;
a forgetting rate for that table is not designed.) **(b) The escape by counting, inside `Chain.prob`** (§I.3 item 4,
now built as `EscChain`): the KT term (V/2)/(N + V/2) replaced by a counted escape e = (escapes + ½)/(events + 1) per
bucket (order r, ⌊log2 N⌋, distinct symbols d) in five structures, chain alone vs KT: `blend` (the estimator swapped
in place in the chain's own top-down blend) 2.9010, +0.672 (SE 0.018) — catastrophic, the escaped mass compounds
through 8 levels; `excl` (bottom-up PPM exclusion) 2.3880, +0.159; `ppm` (top-down PPM with exclusions and
conditional counts) 2.4127, +0.184; `cal` (the blend's interpolation KEPT, p_r = w · p_{r−1} + (1 − w) · cnt/N with
w = min(1, e / p_{r−1}(unseen set)), so the seen symbols get exactly 1 − e) 2.2214, −0.0074 (SE 0.0047, 4/6 blocks);
`cal_cls` (`cal` with `Stream.cls` of the previous character in the key — the one post-hoc variant, added after
`cal`'s result) 2.1893, **−0.0395 (SE 0.0056) = 82% of SSE's −0.0482**, and vs KT chain + SSE `conf_cls` only +0.0087
(SE 0.0041). Structural finding: (V/2)/(N + V/2) in `Chain.prob` is a blending WEIGHT, not an escape probability;
the counted escape helps only when the interpolation of the seen symbols by the lower levels is kept. The counted
rates (`cal`, d marginalised) vs KT's: order 8, N = 1: 0.474 vs 0.972; order 8, N ∈ [16, 32): 0.116 vs 0.686; order
1, N ≥ 1024: 0.003 vs 0.033 — KT is 2–6× too pessimistic at every populated bucket. SSE's residual on its own chain:
on `cal` it GROWS (−0.0602 / −0.0659) — the standard SEE key (order, log2 N, d) does not remove what the SSE fixes;
on `cal_cls` it shrinks to −0.0302 (SE 0.0024) / −0.0353 (SE 0.0033). So the SSE's gain on the chain is mostly escape
miscalibration, but the miscalibration is conditional on the previous character's CLASS, which the standard PPM SEE
key misses. E34's 2-expert linear `Mixer` on `cal_cls` is WORSE than that chain alone (+0.0100, SE 0.0004).
**(c) The stack's floor.** The chain + word per-context grid on `cal` 2.1221 (−0.1067, SE 0.0057 vs KT; −0.0293, SE
0.0013, vs the grid on KT), on `cal_cls` 2.1190 (−0.1098, SE 0.0063; −0.0323, SE 0.0017, vs the grid on KT); grid +
SSE `cls_deep>char` on `cal` 2.1168 (−0.1120, SE 0.0057), on `cal_cls` 2.1166 (−0.1122, SE 0.0064; the two stacks
differ by −0.0002, SE 0.0013). The escape's gain adds to the GRID (−0.032) and not to the SSE; A2-min + SSE (2.1533)
stays 0.007 (SE 0.002) above the KT grid + SSE. The three closed forms — per-context exponents, SSE, counted
class-keyed escape — are three routes to the same ≈ −0.11 (5.0%), not three additive gains; the number to build on is
**2.117 at prefix 300k**. Status: supported at 300k and re-read — the APM-sized 2–3% is the class-conditional
miscalibration of the KT escape, fixable inside the chain by counting (`cal_cls`) or outside it by a table (SSE) or by
a per-context exponent (T4), and only once.

**T8 (cross-problem) — Whether unit exponents (plain counting) are right is a property of the REDUNDANCY between
the evidence sources, not of their quality: the product with exponent 1 is right where the cues are near
conditionally independent given the target, and wrong for the 18 nested, cue-sharing text contexts.**
Numbers: frames, `r_window_neighbours.json`: naive Bayes over the 9 cells (P(y) Π_i P(w_i | y), 9 × 17 × 16 counts)
0.926 / 0.870 / 0.811 at 0.63 / 0.77 / 0.91 bits per cell — above the hard nearest rule at t ≥ 0.5 (paired frame
bootstrap +0.026 [0.020, 0.032] at t = 0.5, +0.081 [0.072, 0.090] at t = 0.75; −0.015 at t = 0.25), on both boards,
and defined for every window; the linear mixture of the same nine tables is last (0.889 / 0.822 / 0.792). It IS
over-sharpened, measured (`research/check_frames_apparatus.json`, the r_window apparatus, seed 0): bits − entropy
per cell = +0.330 / +0.185 / +0.044 at t = 0.25 / 0.5 / 0.75 (bits 0.625 / 0.754 / 0.905 vs entropy 0.295 / 0.569 /
0.860; mean max-probability 0.930 / 0.858 / 0.774 vs accuracy 0.926 / 0.874 / 0.820), i.e. over-confident, most at
low t; the soft kernel is under-confident (gap −0.118 / −0.175 / −0.184). And the conditional-independence mechanism
is measurably violated: naive Bayes over the centre + its 4 edge neighbours (5 cells) beats naive Bayes over all 9
cells at every t (0.934 / 0.881 / 0.828 vs 0.926 / 0.874 / 0.820) — adding the corner cells HURTS, T2's
double-counting in miniature. Text at ±3 (`expF_cluster_features_text.log`): the six-position naive-Bayes product
0.309 vs the exact-window table 0.317, equal on the chain (0.307), and as the last backoff behind the exact r3, r2
tables 0.320 (the best row, +0.003, inside the ±0.003 draw noise). Same product, 18 nested contexts: 7.665 / 8.558.
Round 2 (`expI_frames_chowliu_loo.json` / `.log`, five held-out boards, §0; pooled cell accuracy with the paired
frame bootstrap in brackets and the board sign count). **(a) The product's "over-confidence" was a palette floor, not
the double counting.** On in-palette cells naive Bayes over 9 cells is UNDER-confident at every t (bits − entropy
−0.003 / −0.052 / −0.137 at t = 0.25 / 0.5 / 0.75; mean max-probability 0.858 vs accuracy 0.863 at t = 0.5); the
all-cells gap (+0.495 / +0.327 / +0.162 on five boards, +0.33 / +0.19 / +0.04 on LP2 + CA2 in round 1) is the 2.1% of
cells whose clean colour never occurs as a training target, on which nb9 pays 25.5 / 19.7 / 15.9 bits each (α = 0.5
gives an unseen target colour P(y) ≈ 1.6 · 10⁻⁵; the hard rule 10.6 / 10.0 / 10.0, the kernel 11.8 / 10.6 / 9.8, the
centre table 14.2 / 15.2 / 15.8), i.e. 0.54 / 0.41 / 0.33 bits per cell — all of it. **(b) Where the redundancy is,
measured.** The plug-in conditional MI of the cells given y (17 × 17 × 16 tables from 30 640 pairs, biased upward for
sparse pairs, the ORDER robust): the centre vs ANY other cell 0.021 / 0.028 / 0.031 bits — the independence
assumption is right for the centre; adjacent RING pairs 0.3–0.6 (corner–adjacent-edge mean 0.375 / 0.319 / 0.287,
max pair (1, 2) 0.60 / 0.50 / 0.45; edge–edge 0.214 / 0.130 / 0.078); for corners 2 and 6 the shared 0.3–0.4 bits
exceed their own I(cell; y) of 0.06–0.11 (the centre carries 0.79 / 0.45 / 0.16, corners 0 and 8 0.29–0.30 / 0.26–0.27
/ 0.24–0.25, edges 0.11–0.22). The Chow–Liu tree (max-weight spanning tree on the CMI, rooted at the centre) is a single
PATH around the ring, 4→5→2→1→0→3→6→7→8 at t = 0.5, attached to the centre by one 0.03-bit edge (0.02 at t = 0.25).
The corners' measured harm to naive Bayes, nb5 − nb9: +0.005 [+0.004, +0.007] (4/5) / +0.000 [−0.002, +0.003] (3/5)
/ +0.008 [+0.006, +0.011] (5/5) — real, small, absent at t = 0.5. **(c) The tree does not fix it in closed form and is
not calibrated** (F3 round 2): tan9 (m = 5) − nb9 = +0.006 [+0.005, +0.008] (4/5) / +0.004 [+0.002, +0.007] (5/5) /
−0.014 [−0.016, −0.011] (2/5: −0.022 / −0.008 / −0.018 on the three boards of the trained games, +0.015 / +0.010 on the
two never-trained games); in-palette gap +0.063 / +0.097 / +0.148 (OVER-confident) against nb9's under-confidence;
the m ladder moves both monotonically toward nb9 (m = 1: +0.003 / +0.004 / −0.015, gap +0.120 / +0.194 / +0.289;
m = 100: +0.013 [+0.011, +0.014] (5/5) / +0.009 [+0.007, +0.011] (5/5) / −0.003 [−0.005, −0.000] (3/5), gap +0.015 /
−0.009 / −0.032). The ordering vs the hard rule holds on five boards: nb9 − hard −0.006 [−0.009, −0.004] / +0.025
[+0.022, +0.028] (5/5) / +0.074 [+0.070, +0.078] (5/5). Status: supported as an ORDERING in both domains (unit
exponents rank above the hard rule on 9 cells at t ≥ 0.5, on five boards, and at 7.665 on 18 nested contexts); the
mechanism is now measured rather than inferred — unit exponents are right for the centre given y (CMI 0.02–0.03 bits)
and double-count the ring (0.3–0.6 bits per adjacent pair), the product's over-confidence in round 1 was the palette
floor and on in-palette cells it is under-confident; the text half is argmax accuracy, not bits.

### I.2 The ledger at prefix 300k / 60k (one slice, one book)

"vs chain" is a paired difference on this slice (the KT chain 2.2288 in every pass); where a per-block SE (six 10k
blocks) was measured it is given (`check_text_slice_noise.json`, `check_per_context_oracle.json`, and every round-2
JSON keeps per-block bits for every row); elsewhere the 0.005 floor applies. Per-context posteriors carry a prior
cost n_ctx · log2|G| / n (given) and are to be judged against their per-context oracle (T4), not against a global
best point; the causal C2 rows carry no prior charge (plug-in prequential sums); hindsight rows are oracles and are
not in the table. Rows are grouped by what they combine; within a group the rows are ALTERNATIVES.

| predictor | file | bits/char | vs chain | note |
|---|---|---|---|---|
| chain alone (KT escape) | any | 2.229 | — | |
| raw product of seen experts | `expC_e34_mixtures_300k.json` | 8.558 | +6.3 | |
| **one chain, recalibrated** | | | | |
| linear Bayesian mixture, 18 experts (E34 `Mixer`) | `expC_e34_mixtures_300k.json` | 2.211 | −0.018 ± 0.006 | 2-expert version −0.016 ± 0.006 |
| geometric, Bayesian exponents over seen experts (E34) | same | 2.191 | −0.038 ± 0.006 | |
| chain alone, per-context TEMPERATURE posterior (10 points) | `expG_cross_term_exponents_chain.json` | 2.190 | −0.039 ± 0.005 | prior 0.002; per-context oracle 2.185; best global temperature 1.0 (= no gain) |
| chain alone, C2 stretch-LDA on the odds pool | same | 2.187 | −0.041 ± 0.007 | vs the temperature posterior −0.002 ± 0.002; w = 0.89 |
| `cal_cls` chain alone (counted escape keyed by order, log2 N, d, previous-character class, inside `Chain.prob`) | `expH_additivity_and_escape.json` | 2.189 | −0.040 ± 0.006 | `cal` (without the class key) 2.221, −0.007 ± 0.005 |
| A1: Bayes over 11 temperatures, decaying switch | `mix_temperature_bayes_decay.json` | 2.184 | −0.045 | prior 0.002 |
| SSE `conf_cls` on the chain alone | `expC_sse_refinement.json` | 2.181 | −0.048 ± 0.006 | |
| SSE two-stage `cls_deep>char` on the chain alone | same | 2.179 | −0.050 ± 0.007 | best of 90 variants on this slice; tied with `conf_cls`; 60k warm-up 2.172 |
| `cal_cls` chain + SSE `cls_deep>char` | `expH_additivity_and_escape.json` | 2.154 | −0.075 ± 0.007 | SSE's residual on `cal_cls` −0.035 ± 0.003 (vs −0.050 on KT) |
| **18 experts, count-based exponents** | | | | |
| naive Bayes, flat exponents of total 1.86, chain + word + match + line | `expB_naive_bayes_product.json` | 2.180 | −0.049 | |
| A2-min: Bayes over 42 (β_chain, β_rest) points, decaying switch | `mix_temperature_bayes_decay.json` | 2.168 | −0.061 ± 0.004 | prior 0.003; per-context oracle 2.166, posterior − oracle +0.002 ± 0.001 |
| `sym_key_cmi@m1`: geometric over seen experts, e_m ∝ gain_m · (1 − max_j r_key-CMI(m, j)), mass 1 | `expJ_effective_sources.json` | 2.167 | −0.062 ± 0.005 | best of 48 rows on this slice; = chain 0.69 + match 0.19 on the simplex; the PRIMARY rows (mass k_eff) are all above the chain |
| A2-min + SSE `cls_deep>char` (240k warm-up) | `expH_additivity_and_escape.json` | 2.153 | −0.076 ± 0.006 | SSE's residual on A2 −0.013 ± 0.002; 60k warm-up 2.149 (−0.080 ± 0.007) |
| **2 experts (chain + word) per mixing context** | | | | |
| per-context grid posterior (63 points, share 0.01) | `expA_exponent_grid_fine.json` | 2.152 | −0.077 ± 0.005 | prior 0.0032 (32 used contexts) – 0.0036 (all 36); per-context oracle 2.144, posterior − oracle +0.008 ± 0.001; second slice −0.072 ± 0.007 |
| C2 stretch-LDA exponents on the probability pool (36 solves of a 2 × 2 system) | `expG_cross_term_exponents.json` | 2.150 | −0.079 ± 0.007 | vs the grid posterior −0.002 ± 0.002 (a tie); the same exponents on the odds pool 2.198 |
| per-context grid + SSE `cls_deep>char` (or `conf_cls`, both 2.1462) | `expH_additivity_and_escape.json` | 2.146 | −0.083 ± 0.006 | SSE's residual on the grid −0.005 ± 0.001 / −0.005 ± 0.002; interaction +0.043 ± 0.005 |
| per-context grid on the `cal_cls` chain | same | 2.119 | −0.110 ± 0.006 | −0.032 ± 0.002 vs the grid on KT; the grid's own gain on `cal_cls` −0.070; per-context oracle 2.111 |
| per-context grid + SSE `cls_deep>char` on the `cal_cls` chain | same | **2.117** | **−0.112 ± 0.006** | on `cal` 2.1168 (−0.0002 ± 0.0013 apart); SSE's residual on the grid −0.0025 ± 0.0008 (`conf_cls` on top of it HURTS, +0.0046 ± 0.0011) |
| **3 experts (chain + word + match) per mixing context** | | | | |
| per-context grid posterior (336 points; b_word ≤ 0.7, b_match ≤ 0.5) | `expG_cross_term_exponents.json` | 2.128 | −0.101 ± 0.005 | prior 0.005; per-context oracle 2.117; vs the 2-expert posterior −0.0235 |
| C2 stretch-LDA exponents on the probability pool (3 × 3 solves) | same | 2.143 | −0.086 ± 0.007 | vs the grid posterior +0.015 ± 0.002 (86% of its gain) |

At prefix 1.2M (`expA_exponent_grid_1200k.json`): chain 1.817; linear 1.814 / 1.823 / 1.807; per-context grid
posterior 1.776 / 1.790 / 1.791 (word / match / order4) — its per-context oracle at 1.2M is not measured (the check
ran at 300k only) [pending check_per_context_oracle --train_chars 1200000]. At E34 full: chain 1.823, linear 1.820,
geometric 1.801. The closed-form ladder's best paired gain against the chain is now **5.0% at 300k** (0.112 / 2.229:
the `cal_cls` chain + the chain + word per-context grid + the two-stage SSE, `expH`) and 4.5% for the 3-expert
per-context grid on the KT chain (0.101 / 2.229, `expG`); 2.3% at 1.2M (0.041 / 1.817, 2 experts, not combined). The
rows HAVE now been combined (round 2), and the finding is that they largely do not add: SSE on top of the per-context
grid is worth −0.005, on top of A2-min −0.007 / −0.013 (interaction +0.04 in each case, T7 round 2 (a)); the counted
escape adds to the GRID (−0.032) and not to the SSE; the three recalibrations of the chain — per-context exponent,
SSE table, counted class-keyed escape — are three routes to the same ≈ −0.11 floor, reached from `cal` or `cal_cls`
alike (2.1168 / 2.1166). What DOES add is a second independent partner: match on top of chain + word −0.0235 on the
per-context grid (T4 round 2 (b)); chain + word + match on the `cal_cls` chain has not been run [pending: the
3-expert grid on `cal_cls`]. The 18-expert closed forms (A2-min 2.168, `sym_key_cmi@m1` 2.167) both reduce to
"chain + one partner on the simplex" and sit 0.015 above the 2-expert per-context posterior and 0.040 above the
3-expert one: nothing count-based has yet shown the other 15 experts to be worth anything beyond chain + word +
match.

### I.3 What remains unexplained (Part I)

1. **The stacking weight from counts — run (T6 round 2), and the open part is the NESTED case.** C2 (LDA on the
   stretched one-vs-rest predictions per mixing context, the outcome counted) recovers the per-context exponents of
   2–3 DIVERSE experts to within 0.06–0.08 and codes within −0.002 ± 0.002 (chain + word) / +0.015 ± 0.002 (three
   experts) of the per-context grid posterior; C3 (Σ⁻¹ diag(Σ) from the forecasts alone, Satopää, Pemantle & Ungar
   2016) recovers 1/5–1/3 of it (−0.014 / −0.023), its reliability assumption Cov(X_m, Z) = Var(X_m) failing per
   context as predicted. On the nested pair chain + order4 BOTH fail (+0.032 and +0.736 vs the grid's −0.051): the
   exact exponent of a function of another source is a conditional log-likelihood ratio (Allard eq. 17) that a
   Gaussian second moment cannot see (the stretched forecasts correlate 0.55–0.84, not 1). The binarisation question
   is answered by construction (ONE-VS-REST over the 70 characters, 1 positive + 69 negatives per position; no 7-bit
   decomposition needed). What is NOT designed: a count-based exponent that is 0 for a function of another source —
   Allard eq. 17 by prequential counting (the code length of c under KT(K_m) vs KT(K_m, bucket of K_j)) or LDA on the
   difference feature stretch_order4 − stretch_chain — and the test is chain + order4 against −0.051.
2. **The information structure of the 18 experts — measured (T5 round 2), and it does not supply an allocation.**
   `expJ`: gain over the KT unigram (4.294 bits) where seen: chain +1.81, order3 +1.30, order2 +1.23, word +1.05,
   pair1_3 +1.01, order4 +0.87, pair1_4 +0.83, match +0.80, order1 +0.79, line +0.74, skip2 +0.35, order5 +0.29,
   skip3 +0.14, skip4 +0.07, and NEGATIVE order6 −0.15, prevword −0.31, order7 −0.43, order8 −0.60 (seen on 67 / 20 /
   46 / 30% of positions). Symmetrised KL: chain–order8 10.0, chain–order4 7.7, chain–word 6.7, chain–match 4.6,
   word–match 3.3, order1–line 0.21, skip3–skip4 0.42. Three pairwise matrices (shuffle-corrected key-CMI, argmax-CMI,
   code-length correlation partialled on the next character) give k_eff = 1.62 / 4.30 / 2.64, none the calibrated
   mass, and their allocations lose to flat except for the nesting bit (r = 1.00 for key-function pairs). The
   unresolvable pairs (chain, order7, order8, prevword: 70k–87k distinct keys in 100k positions; the plug-in and its
   null both read ≈ 1) need a PREQUENTIAL conditional information (the code-length gain of adding K_m to a table keyed
   by K_j), resolvable at any key cardinality — not designed. Heskes' quadratic ambiguity overstates the exact one 2×
   (1.498 vs 0.740 bits), so C5's QP is off the list unless it uses −log2 Z.
3. **Additivity — measured (T7 round 2 (a), T4 round 2 (b)).** SSE does not add to the per-context exponents (−0.005
   on the chain + word grid, −0.007 / −0.013 on A2-min; interaction +0.04); word and match partly add (−0.0235 on the
   per-context grid, against −0.073 for match alone). Open: the 3-expert grid on the `cal_cls` chain (does the 2.117
   floor move with an independent partner once the chain is calibrated?), and the per-character stack table's
   forgetting rate (60k warm-up beats 240k by 0.004–0.006 on both the chain and A2-min; no cap, no decay — not
   designed).
4. **The escape inside the chain — built (T7 round 2 (b)), with a structural finding.** (V/2)/(N + V/2) in
   `Chain.prob` is a blending WEIGHT, not an escape probability: swapping in a counted escape in place (`blend`) costs
   +0.672, PPM structures without interpolation lose (+0.159 / +0.184), and the counted escape helps only when the
   lower levels' interpolation of the seen symbols is kept (`cal`, w = min(1, e / p_{r−1}(unseen)) — my construction,
   no literature source; how often the cap at 1 discards the deep level is unmeasured). The standard SEE key (order,
   log2 N, d) gains −0.007; the previous character's CLASS in the key gains −0.040 (the one post-hoc variant). The
   counted rates are 2–6× below KT's at every populated bucket. Not run: `expA`'s order4 check on `cal_cls` (the
   prediction "b ≈ 0 for order4 once the escape is counted" is still untested; the refuted b = 0.3 was on KT);
   PPMII-style information inheritance inside `cal` (the within-seen-set cnt/N smoothed by the lower level with a
   counted weight), the last place the chain differs from the two-stage table; the second-slice replication of the
   `cal_cls`, grid-on-`cal_cls` and grid + SSE rows.
5. **Symbol-level vs bit-level — half answered.** The one-vs-rest construction of `expG` fits ONE exponent per
   expert per context from 70 binary events and ties the symbol-level grid, so the per-outcome factor is not where
   the 2–3-expert gain was. Not tried: the per-context posterior conditioned also on the partner's count bucket (0 /
   1–3 / 4–15 / 16+), and a 2-D SSE keyed by two experts' buckets. **The pool's SCALE is the new open item** (T4
   round 2 (c), T6 round 2 (2)): the probability pool Π p_m^{w_m}/Z beats the odds pool exp(Σ w_m stretch(p_m))/Z
   that eq. 4.1 and PAQ define, by +0.011–0.013 on the grid oracles and by 0.05–0.15 for the same LDA exponents; the
   stretch-fitted LDA level matches the probability pool's optimum for M ≥ 2 and the odds pool's for M = 1 — measured,
   not derived. The JSON holds both grid oracles per context beside the LDA's scale factor; a derivation (or a
   refutation by construction) is the cheapest open theory item.
6. **The Bayesian GLN (A3).** A tree of 17 two-input grid mixers, each a Bayesian mixture over a 15 × 15 grid of
   exponent pairs, at ~0.02 bits/char of prior cost. With FIXED per-node exponents the tree's leaf exponents are the
   products of the node exponents along each path (GLN eq. 5, "the logit and sigmoid functions cancel", holds for
   fixed weights), and the 15-level grid on [0, 3] represents only products of grid values. With per-node BAYESIAN
   posteriors each node emits a LINEAR mixture of geometric mixers; feeding that into the next geometric node is no
   longer a geometric mixture of the leaves, so neither the representability claim nor the dominance bound is
   inherited by the tree — which is why A3 has no bound. Whether greedy per-node posteriors reach what a jointly
   fitted 18-vector reaches has not been run.
7. **Switching rate.** CTS's lesson (the global rate n⁻¹ beat the per-context n_c⁻¹) has no analogue yet for a
   posterior over exponents; `expA` found the exponents stationary within the book (no-share best), `mix_temperature`
   found the decaying rate worth 0.015 — the two are consistent (both say: forget slowly) but the right rate is
   not designed.
8. **How much is at stake, exactly.** The learned-exponent reference number is not to be built on, has not been
   run on this library, and cannot be under the no-gradient rule. Mattern 2012's same-experts comparison measures
   learned SIMPLEX exponents on binary decisions against Bayesian linear weights — 3.4% (average) / 4.4% (book1)
   relative to BETA — the class E34's `geometric` already belongs to (1.2% recovered, 1.820 → 1.801); what free
   exponents (S_w ≠ 1) are worth has no published same-experts comparison. "A learned mixer should land near 1.75"
   is an unmeasured extrapolation of those Calgary percentages to E34's 1.820, not a target. The closed-form ladder
   has reached **5.0% at 300k** (`cal_cls` chain + chain + word per-context grid + two-stage SSE, 0.112 ± 0.006;
   `expH`), 4.5% with three experts per context on the KT chain (0.101 ± 0.005, `expG`), 3.5% with two (0.077 ±
   0.005; 3.2% on the second slice) and 2.3% at 1.2M (two experts, not combined) — the combination is done and is a
   floor, not a sum (§I.2). The count-based references in hand: the per-context hindsight oracles (2.144 chain + word,
   2.117 chain + word + match on KT, 2.111 chain + word on `cal_cls`, 2.166 A2-min), which the posteriors sit
   0.002–0.011 above, and the closed-form C2 solve, which sits 0.006 (chain + word) / 0.025 (three experts) above
   its oracle with no grid at all. The 1.2M per-context oracle is still unmeasured [pending
   `check_per_context_oracle.py --train_chars 1200000`].
9. **A count-based mass rule.** T5 round 2 refuted k_eff from every pairwise matrix; what survived as a candidate
   is the calibration gap itself — bits − entropy is measurable prequentially per mixing context, and the mass that
   calibrates is 2 at flat allocation and ≈ 1 at chain + match. A gap-driven per-context mass (raise the total while
   the pool is under-confident, lower it while over-confident) has not been run.

### I.4 What a gradient would be doing that the closed forms so far do not

The exponent vector's maximum-likelihood problem per mixing context is convex in 18 dimensions, and strictly
convex provided no non-zero v makes vᵀ log P(x) constant in x — Mattern 2012 §3.2's proof needs Jensen's inequality
to be strict, i.e. no expert's log-probability vector may be an affine function of the others'. With
r_cmi(chain, order8) = 1.00 measured in `expB` the Hessian is near-singular in the nested directions, which is why
any 18-dimensional fit there (grid, Newton or gradient) is ill-conditioned. PAQ's online gradient solves it
continuously; its solution is the joint redundancy structure of the expert set per context (T6), i.e. the stacking
weights. The closed forms measured here approximate that joint object by (a) a posterior over a LOW-dimensional grid
of exponent vectors (2–3 dimensions in `expA`, `expG` and A2-min; the full 18-dimensional posterior integral is
closed form but exponential in m — `bayesian_alternatives_to_learned_mixing.md` §2B, prior cost ≈ 450 bits, the
integral is the cost), (b) a count-based TOTAL with a flat or matrix-shaped allocation (T5, refuted for the mass and
for every graded allocation), (c) a per-context recalibration of the chain — a temperature, an SSE table or a counted
escape, three routes to one −0.11 (T7), and (d) since round 2, a cross-term computed from data: the LDA solve on
counted second moments (C2).

What the round-2 numbers say a learned mixer would STILL be doing, stated per regime:

- **At 2–3 diverse experts per context: nothing that a gradient is needed for.** The linear solve lands within
  −0.002 ± 0.002 (chain + word) and +0.015 ± 0.002 (chain + word + match) of the per-context grid posterior, which
  itself sits 0.008 / 0.011 above its per-context hindsight oracle; the hindsight-fitted LDA is no better than the
  causal one (2.1532 vs 2.1495), so the exponents are stationary within the book and per-symbol continuous adaptation
  is not what a step-size method would be buying. The ceiling for ANY per-context fixed exponent vector at this scale
  is the oracle gap: 0.006 bits/char over C2 for chain + word, 0.025 for three experts (the 336-point grid's oracle is
  a lower bound on a finer grid, so this figure is soft).
- **The scale the mixer is written in is a separate choice the data made against PAQ's.** A PAQ-style mixer fits
  exponents on the stretch (odds) scale; on this 70-ary alphabet the odds pool is worse than the probability pool at
  the SAME exponents (grid oracles +0.011 / +0.013; the LDA's own exponents 0.05–0.15 worse on the odds pool; the
  literal per-event sigmoid +0.10). A learned mixer on the stretch scale would start from the worse pool; on the
  probability scale the closed forms already hold the exponents.
- **What remains is the NESTED / cue-sharing part of the library, and it is a non-Gaussian cross-term.** For chain +
  order4 the exact exponent is ≈ 0 (Allard eq. 17: order4 is a function of the chain's inputs), the grid posterior
  finds 0.26 with the chain tempered to 0.87 and gains −0.051, and both second-moment solves fail (C2 +0.032, 0/6
  blocks; C3 +0.736) because the stretched forecasts of a deterministic coarsening correlate only 0.55–0.84 with the
  chain's — the redundancy lives in WHERE the forecasts differ (small deep counts), not in their covariance. The
  pairwise statistics of `expJ` supply only the nesting bit (r = 1.00 for key-function pairs) and reduce the library
  to chain + match (2.167). So what a learned mixer would be doing on the 18 experts, precisely: minimising the
  per-context code length jointly in 18 dimensions — a convex problem whose Hessian is near-singular in the nested
  directions — to find the conditional log-likelihood-ratio exponents of the 8 order-k experts, the skips, the pairs,
  line and prevword GIVEN the chain, word and match. Whether that is worth anything is UNMEASURED: no 18-dimensional
  per-context fit exists (grid: exponential; LDA: fails on nested pairs), and the two 18-expert closed forms (A2-min
  2.168, `sym_key_cmi@m1` 2.167) are 0.040 ABOVE the 3-expert per-context grid (2.128). The honest statement is not
  "nothing is left for a gradient" but "what is left is confined to the nested part of the library, its value is
  bounded above by nothing measured, and the count-based route to it is a conditional (prequential) exponent, not a
  covariance" — §I.3 item 1's not-designed rule, or C2 run on all 18 experts per context (36 solves of an 18 × 18
  system on the `logD` matrix) as the direct test of whether the nested mis-allocation sinks the rule at library
  scale (§3 item 1).
- **Calibration is not the gradient's either.** The APM-sized 2–3% is the class-conditional miscalibration of the KT
  escape (T7 round 2), fixable inside the chain by counting (`cal_cls`) and then worth −0.032 more to the per-context
  grid; SSE on top of all that is −0.0025 ± 0.0008 (and `conf_cls` on top of the `cal_cls` grid hurts, +0.0046 ±
  0.0011).

---

## Part II — Features inside a block without gradients

### II.0 What the survey fixed (from `gradient_free_features_and_denoisers.md`)

In every gradient-free feature method the features are fitted by a closed form or by counting — eigenvectors
(PCANet), fixed wavelets (ScatNet), k-means means or random exemplars (Coates), random spectral draws (Rahimi–Recht),
alternating SVD steps (K-SVD), a local rule whose fixed point is soft k-means (SoftHebb) — and the target enters only
at a SOLVED readout (least squares, an SVM, or a count table keyed on a discrete code: PCANet's block histogram). The
residual gap to backprop is 4–10 points on CIFAR-10/ImageNet-class tasks and about 1 dB in Gaussian denoising:
DnCNN 2017 is +0.6 dB over BM3D on BSD68, DRUNet 2021 (Zhang et al., TPAMI, arXiv:2008.13751 Table 1) +0.9 dB
(29.48 vs 28.57 at σ = 25; 26.59 vs 25.60 at σ = 50; 31.91 vs 31.08 at σ = 15). In classification the gap is
"features chosen FOR the target"; that part of the denoising gap is the receptive field is this note's conjecture, not
a result of any cited paper. RandNet matches PCANet only at 60k training examples (MNIST: 0.63 vs 0.66 error); at
10k (MNIST basic) it is 1.25 vs 1.06 — at E35's 30k-window scale the fitted projections are expected to matter.
Non-local means with kernel width h → 0 is the single nearest patch; Levin & Nadler 2011's optimum for a k-window
is the Gaussian-KERNEL average over the database (their eq. 12, non-local means with bandwidth σ), not the nearest
neighbour; their neighbour-density statement (3 × 3: 99% of patches have > 2 000 neighbours; 9 × 9: 13% have none)
is about when their lower bound is tight, i.e. where a window's density gives out (§II.2 item 3).

### II.1 The theories, each with its mathematics and its number

**F1 — E35's memory block is not a lookup table; at t ≥ 0.5 it is a Hamming nearest-neighbour rule (92–99.5% of
held-out windows unseen), and its accuracy is NOT a density limit of the 9-cell window.**
Mathematics: `e35.NearestRule.predict_nearest` = Nadaraya–Watson on Hamming distance with h → 0. Levin & Nadler's
optimum for a k-window is the KERNEL average over the database (their eq. 12), not the nearest neighbour; their
neighbour-density statement is about when their bound is tight. The hard rule sits 0.05–0.10 BELOW the kernel
estimator on the SAME stored windows (F2: +0.051 / +0.099 at t = 0.5 / 0.75), so its accuracy is the hard rule's,
not the window's.
Numbers (`r_window_neighbours.json`, one shot, 30 640 training pairs): distinct training windows 21 029 / 28 944 /
30 509 at t = 0.25 / 0.5 / 0.75; held-out windows never stored 0.626 / 0.923 / 0.995; hard-nearest accuracy by
Hamming distance at t = 0.5: d = 0 → 1.000, 1 → 0.980, 2 → 0.888, 3 → 0.748, 4 → 0.641, 5 → 0.505.
`expE_nlm_block.log`, distance profile: at t = 0.5 only 7% of query windows are stored exactly and there are on
average 2.6 / 31 / 165 stored windows at d = 1 / 2 / 3; at t = 0.75, 0.3% stored, 0.2 / 3.5 / 31.
Round 2 (`expI_frames_chowliu_loo.json`, five boards): the hard rule (nearest stored window's counts) is 0.926 /
0.837 / 0.729 pooled (per board at t = 0.5: 0.841 / 0.822 / 0.856 / 0.836 / 0.825), its in-palette calibration gap
+0.456 / +1.308 / +2.405 bits per cell (mean max-probability 0.986 vs accuracy 0.837 at t = 0.5 — the one predictor
that is grossly over-confident), and the leave-one-out-selected kernel over the SAME windows beats it by +0.014
[+0.012, +0.016] (boards +0.009 ± 0.004, 4/5, Toggle −0.006) / +0.053 [+0.051, +0.056] (+0.048 ± 0.005, 5/5) /
+0.088 [+0.085, +0.092] (+0.077 ± 0.009, 5/5). Status: measured on five boards, two of them games never trained on;
the hard rule's accuracy is the rule's, not the window's, at t ≥ 0.5 on every board.

**F2 — Softening the kernel is a gradient-free generalisation by shared structure (literal cell overlap): a vote over
~15 windows sharing 6–8 of 9 cells averages the target noise of one stored window; the neighbour SET carries the gain,
not its weights, and the neighbourhood must stay local.**
Mathematics: non-local means proper, P(y | w) ∝ Σ_s exp(−d_H(w, s)/h) n_s(y); the full kernel at large h drifts to
the frame's colour prior.
Numbers, within one apparatus at a time. `r_window_neighbours.json` (135 frames, 1 draw per frame, its own rng
stream; hard rule 0.941 / 0.849 / 0.731): soft kernel over all stored windows with h by leave-one-out likelihood on
a 2 000-window training subsample (h = 0.5 / 0.75 / 0.75): 0.954 / 0.900 / 0.829, +0.014 [0.010, 0.018] / +0.058
[0.052, 0.064] / +0.093 [0.086, 0.101] over hard (paired frame bootstrap, `check_frames_apparatus.json`; the
`r_window` file's own draw gives +0.013 / +0.051 / +0.099), at 0.35 / 0.69 / 0.87 bits per cell.
`expE_nlm_block.json` (73 kernels on the SAME 135 frames; its k = 1 reproduces E35's strict 0.880 / 0.800 (chain) and
0.854 / 0.732 (one shot) exactly on all 405 corrupted frames, and on the 135-frame grid k = 1 gives chain 0.876 /
0.801, one shot 0.852 / 0.734 at t = 0.5 / 0.75 — the pairs below are against THOSE): k = 15, h = 2, count-weighted,
chain 0.924 / 0.855 (+0.048 / +0.054 over k = 1); k = all, h = 0.5, one shot 0.903 / 0.816 (+0.051 / +0.082); ONE
soft one-shot block (k = all, h = 0.5: 0.903 at t = 0.5) beats the whole 8-block hard chain of the same run (k = 1
chain 0.876) by +0.027; majority vs count vs full-distribution weighting differ by ≤ 0.005; 66 of 72 soft kernels beat
k = 1 on both chain columns, the 6 that do not are all k = all with h = 2 (0.859 at t = 0.5, below k = 1), where 99.5%
of the kernel's mass lies beyond the 15 nearest windows (77% at h = 0.5, 97% at h = 1). Voting only where E35 fell
back to the nearest window captures most of the gain (0.916 / 0.842 vs 0.921 / 0.850). The chain's lead over one shot
shrinks under the soft kernel (t = 0.75: +0.067 at k = 1 → +0.036 at k = all, h = 0.5) but stays positive at every
kernel. Status after round 1: supported on two boards, with k and h selected among 73 kernels on the held-out frames.
Round 2 (`expI_frames_chowliu_loo.json`, five boards; k ∈ {1, 3, 7, 15, 50, 100, 200, all} × h ∈ {0.25 … 3}, picked
PER BLOCK by leave-one-out log-likelihood on a 3 000-window subsample of the block's own stored windows; the two LOO
definitions — whole window out, one observation out — agree everywhere except chain block 1). **The honest selection
picks a LOCAL kernel, not `expE`'s full one:** one shot k = 50, h = 0.75 / k = 100, h = 1.0 / k = 200, h = 1.0 at
t = 0.25 / 0.5 / 0.75 (LOO bits 0.207 / 0.402 / 0.740 vs 0.682 / 1.354 / 2.605 at k = 1 and 0.269 / 0.552 / 0.834 at
the best k = all; the t = 0.75 pick lies on the grid's finite-k boundary — k = 100 is 0.010 bits worse, a plateau);
chain blocks 1..6: k = 50 at h = 0.5, 0.5, 0.75, 1.0, then k = 15 at h = 1.5, 1.5. Held out at the pick: one shot
0.939 / 0.891 / 0.817 (per board 0.951 / 0.919 / 0.964 / 0.914 / 0.891 at t = 0.25; 0.900 / 0.871 / 0.912 / 0.880 /
0.856; 0.825 / 0.797 / 0.837 / 0.815 / 0.794), chain 0.944 / 0.910 / 0.840; soft − hard as in F1; chain soft − chain
hard +0.015 [+0.013, +0.017] (4/5) / +0.043 [+0.040, +0.046] (boards +0.043 ± 0.001, 5/5) / +0.050 [+0.046, +0.055]
(+0.047 ± 0.004, 5/5). **The selection bias of round 1 is ≈ 0.001 in accuracy and 0.03–0.41 bits per cell in code
length:** `expE`'s held-out-selected configuration (k = all, h = 0.5) minus the LOO pick is −0.001 [−0.002, −0.001] /
+0.001 [+0.000, +0.003] / +0.001 [−0.002, +0.003] pooled (boards −0.002 ± 0.002 / −0.001 ± 0.002 / −0.009 ± 0.008; at
t = 0.75 it loses 0.025 / 0.034 on MultiKey / Toggle, the two never-trained games), but its bits per cell (all
cells) are 0.449 vs 0.423 / 0.736 vs 0.620 / 1.268 vs 0.858 (in-palette 0.233 vs 0.182 / 0.608 vs 0.408 / 1.190 vs
0.668; its entropy 0.49 / 1.26 / 2.19 bits, gap −0.041 / −0.524 / −0.921: the full kernel drifts to the colour prior,
as the mathematics line says). **The chain's lead over one shot belongs to
MEMORY blocks:** under the LOO kernel chain − one shot is +0.004 [+0.003, +0.005] / +0.019 [+0.017, +0.021] (5/5) /
+0.023 [+0.020, +0.026] (5/5), about half the hard chain's +0.003 / +0.030 / +0.061; for the factored blocks it is
absent or negative — nb5 −0.003 / −0.016 [−0.018, −0.014] / −0.053 [−0.055, −0.050] (0/5 positive), nb9 −0.001 /
−0.004 / −0.015 (point), tan9 +0.004 / +0.006 [+0.003, +0.008] / +0.010 [+0.007, +0.013] (5/5). Status: supported on
five boards with the selection done on the training windows; THE numbers are 0.891 / 0.817 one shot and 0.910 / 0.840
chain at t = 0.5 / 0.75. The kernel keeps all ~30k entries and treats the 9 cells as equally relevant — the two
places where a feature would enter; the LOO likelihood is now measured as the criterion that would price an entry
(F5), and it picks 50–200 neighbours, not 15 and not all.

**F3 — A closed-form factored block (naive Bayes over the cells) is defined for every window and ranks above the
hard rule at t ≥ 0.5 from 9 × 17 × 16 counts; it is over-confident, and the cells are NOT conditionally independent
given the centre — T8's frame half.**
Numbers: `r_window_neighbours.json` 0.926 / 0.870 / 0.811 (0.63 / 0.77 / 0.91 bits per cell), second to the soft
kernel, above the hard rule at t ≥ 0.5 (+0.026 [0.020, 0.032], +0.081 [0.072, 0.090]; below it at t = 0.25,
−0.015), on both boards, from 9 × 17 × 16 counts instead of 30k stored windows; the linear mixture of the same
tables is last. Calibration, measured (`check_frames_apparatus.json`): naive Bayes over 9 cells is over-confident by
+0.33 / +0.19 / +0.04 bits per cell (bits − entropy) at t = 0.25 / 0.5 / 0.75, the soft kernel under-confident by
−0.12 / −0.18 / −0.18. Naive Bayes over the centre + 4 edge neighbours beats all 9 cells at every t (0.934 / 0.881 /
0.828 vs 0.926 / 0.874 / 0.820): the corner cells' evidence is double-counted. The mechanism that would explain the
ranking is therefore not "near conditional independence"; what is consistent with the numbers but not yet measured
is a weaker dependence structure that a tree captures — the pairwise conditional MI of the 9 cells given the centre
(the Chow–Liu matrix, §II.2 item 6) and a per-t calibration read-out would test it.
Round 2 (`expI_frames_chowliu_loo.json`, five boards; the numbers are in T8 round 2, this is the reading). (a) The
"over-confident" of the headline was wrong: on in-palette cells nb9 is UNDER-confident at every t (−0.003 / −0.052 /
−0.137); the +0.33 / +0.19 / +0.04 was the 2.1% out-of-palette cells at 25.5 / 19.7 / 15.9 bits each. (b) The
dependence structure is measured and is not "given the centre": the centre is conditionally independent of every
other cell given y (0.02–0.03 bits); the RING is a chain of 0.3–0.6-bit adjacent dependencies, and the Chow–Liu tree
is exactly that path hanging off the centre by one 0.03-bit edge. (c) The tree (Friedman–Geiger–Goldszmidt TAN,
P(y | w) ∝ P(y) P(w_c | y) Π P(w_i | w_pa(i), y), pairwise conditionals m-estimated toward P(w_i | y)) does NOT fix the
double counting in closed form on new boards: +0.006 / +0.004 over nb9 and −0.014 at t = 0.75 (2/5 boards; it hurts
on the three boards of the trained games and helps on the two new games), OVER-confident in-palette (+0.063 / +0.097
/ +0.148 at m = 5; +0.120 / +0.194 / +0.289 at m = 1); the most smoothed tree is the best and the best calibrated
(m = 100: +0.013 / +0.009 / −0.003, gap +0.015 / −0.009 / −0.032), so what decides the tree's calibration is the
prior weight m, not the tree — the 17 × 17 × 16 pairwise conditionals fitted on four boards' wall geometry are what a
new board violates. tan5 (tree over the centre + 4 edges) − nb5 = +0.004 / +0.004 / −0.010 (0/5 at t = 0.75). The
tree does help the CHAIN: chain_tan9 − chain_nb9 +0.011 / +0.014 / +0.011 (5/5 each), chain_tan9 − tan9 +0.004 /
+0.006 / +0.010 (5/5). (d) Every factorisation stays 0.02–0.03 below the LOO kernel at t ≥ 0.5 (soft_loo − tan9
+0.014 (3/5) / +0.024 [+0.021, +0.026] (4/5) / +0.028 [+0.025, +0.030] (5/5)); on the two never-trained games at
t = 0.25 the tree equals or beats the kernel (−0.001, −0.011). Status: ordering measured on five boards (nb9 > hard at
t ≥ 0.5, 5/5); the mechanism named — the ring's double counting, the centre clean; the Chow–Liu tree removes it in
principle and gains ≤ 0.013 in practice, with m unselected (Friedman's default 5; selecting m by the same LOO as k, h
is the same closed form and is not run). The count-based move the measured structure suggests — P(y) P(w_c | y) × a
kernel over the 8 RING cells only, or the MI-weighted Hamming distance with I(cell; y) from the block's own counts
(0.79 for the centre against 0.06–0.30 per ring cell at t = 0.25) — is unrun (§II.2 item 2).

**F4 — On text at ±3 the "shared structure" IS the relevance of a position, which falls by half per step outward;
E35's symmetric shrink already encodes it, target-blind codes (Hamming k-means, sign-quantised PCA) throw it away,
and MI-weighting from counts recovers most of it.**
Mathematics: PCANet's recipe (closed-form projection → discrete code → count table); RandNet matches PCANet only at
60k training examples (MNIST 0.63 vs 0.66 error) — at 10k (MNIST basic) it is 1.25 vs 1.06, so at E35's 30k-window
scale the fitted projections are expected to matter, and "the code and the counts do the work" is a 60k-example
statement; Coates: the encoder matters more than the dictionary.
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
own under the existing two-part code, is not designed. Round 2 gives the price a measured form: the leave-window-out
likelihood over the block's own stored windows (`expI`, 0.207 / 0.402 / 0.740 bits at the pick) is the number a
sleep pass would have to keep from rising, and the neighbourhood it says matters is the 50–200 nearest windows at
h = 0.75–1.0 — so "drop an entry when the vote of the others already predicts it" now has a concrete vote. Not run.
A second price question appeared: 2.1% of held-out cells (colours never seen as a training target) cost 10–27 bits
each under every table and dominate every all-cells bits number; a colour-relabelled (canonical-order) window key or a
copy-input fallback for unseen target colours is not designed.

### II.2 What remains unexplained (Part II)

1. **Sleep under the kernel** — drop a stored window when the LOO-selected vote of the others (k = 50, h = 0.75 at
   t = 0.25; k = 100–200, h = 1.0 at t ≥ 0.5, `expI`) already predicts its majority; report entries kept vs the
   five-board accuracy, against E35's collapse (28 811 → 16 entries). Unrun.
2. **The metric** — a Hamming distance that weights cell c by its mutual information with the target colour, from
   the block's own counts: the smallest "feature" a block can have, and the frame analogue of `expF`'s MI weighting
   that recovered 0.254 → 0.339 on text. The weights are now measured (`expI` `trees[label].mi_cell_target_bits`:
   centre 0.79 / 0.45 / 0.16, corners 0 and 8 ≈ 0.30 / 0.27 / 0.25, corners 2 and 6 ≈ 0.10 / 0.07 / 0.05, edges
   0.11–0.22) and the structure says the centre should enter as its own table factor (conditionally independent of
   the ring, 0.02–0.03 bits) with the kernel over the 8 ring cells only. Unrun.
3. **The receptive field** — at radius 2 (25 cells) the raw key is all-unseen, the Hamming kernel's distance is
   noise-dominated, and factored/coded features are where the survey says generalisation continues. This is where
   Levin & Nadler's neighbour-density statement applies: their bound is tight for small windows (3 × 3: 99% of
   patches have > 2 000 neighbours) and the density gives out at large ones (9 × 9: 13% have none), so a kernel
   estimator over stored windows runs out of neighbours with the window, not with the rule; the learned denoisers'
   ~1 dB over BM3D (DnCNN 0.6, DRUNet 0.9) is the size of what a model with shared structure adds there, whatever
   its cause. The crossover is unmeasured; the five-board set and the LOO selection now exist to measure it on.
4. **Chain vs one shot — measured (F2 round 2).** The chain's gain is a property of MEMORY blocks: +0.019 / +0.023
   (5/5) under the LOO kernel, +0.030 / +0.061 under the hard rule, and absent or negative for the factored blocks
   (nb5 −0.016 / −0.053 on 0/5 boards positive; nb9 −0.004 / −0.015; tan9 +0.006 / +0.010). So "the intermediate
   targets add context" is refuted for naive Bayes and weak for the tree; what the chain averages is the memory
   block's variance, and a soft one-shot block already takes half of that. Why the factored chain LOSES (the argmax
   steps feed a factored block a cleaner-looking but wrong window?) is not measured.
5. **Bits, not accuracy** — every text feature number is argmax accuracy; the sleep price judges by prequential
   bits, under which naive Bayes (~31k counts) vs the exact table (197 026 windows + 3 shrink tables) may rank
   differently. Unmeasured on text. On frames the bits are now measured (`expI`, all cells / in-palette): at t = 0.5
   the LOO kernel 0.620 / 0.408 bits per cell, nb9 0.895 / 0.494, tan9 0.912 / 0.481, the hard rule 1.641; the ranking
   by bits agrees with the ranking by accuracy except for the held-out-selected full kernel (equal accuracy, 0.736 /
   0.608 bits — worse by 0.12 / 0.20).
6. **Target-aware closed forms — one run.** Chow–Liu with the target as root (`expI`) names the dependence exactly
   (the ring path) and gains ≤ 0.013 over naive Bayes at the best smoothing, less than the target-blind kernel; its
   m is unselected (selecting it by the same leave-window-out likelihood as k, h is the same closed form, ~10 s,
   unrun), and a Miller–Madow-corrected CMI would check the ring path is not a sparse-table artefact (the ORDER
   ring ≫ centre is robust, the values are not). MI-weighting (item 2), information-gain splits from counts, and
   batch EM for a mixture of categoricals (SoftHebb's fixed point without a learning rate) for better keys than
   k-modes remain unrun. LDANet's closed-form LDA filters gained nothing over PCA on MNIST — the survey's warning
   that a closed-form target-aware feature need not help stands, and the tree is now a second instance of it.
7. **Text as an expert** — similarity-weighted contexts (a kernel over SEEN contexts of the same order on the
   Jensen–Shannon distance of their next-character counts) and Brown-style class contexts; the cost per character
   (a kernel over all seen contexts of an order) and the distance to use are not designed.
8. **New board vs new game.** MultiKey L1 and Toggle L0 are new GAMES (colours 9, 10 and 8 never seen as training
   targets); several differences flip sign on them (tan9 − nb9 at t = 0.75: −0.022 / −0.008 / −0.018 on the trained
   games' boards, +0.015 / +0.010 on the new games; the full kernel loses 0.025 / 0.034 there at t = 0.75). The
   training set has no MultiKey L0; adding it (or holding out LockPath L1) would separate the two effects. Unrun.
9. **The palette floor** — 2.1% of held-out cells at 10–27 bits each (F5); no method reaches them; a colour-relabelled
   key or a copy-input fallback is not designed, and every calibration number on ALL cells must be re-read once it
   exists.

### II.3 What a gradient would be doing that the closed forms so far do not

Fit the features to the target inside the block. The soft kernel (F2) and the factored block (F3) generalise by a
FIXED similarity (cell overlap; cell-wise independence); the codes of F4 by a target-blind projection. In the survey
that is exactly the residual gap (4–10 points in classification, about 1 dB in denoising). Of the count-based,
target-aware moves — MI-weighted metric, Chow–Liu, information-gain splits, MI-weighted k-means already measured on
text — one has now been run on frames: the Chow–Liu tree with the target as root (`expI`, five boards). It finds the
dependence structure exactly (the ring path, the centre clean) and turns it into ≤ 0.013 of accuracy at the best
smoothing, over-confident on new boards at Friedman's default, and stays 0.02–0.03 BELOW the target-blind LOO kernel
at t ≥ 0.5. So "features need a gradient" is still neither supported nor refuted: the one target-aware closed form
tried is worse than a fixed similarity, and the two count-based moves the measured structure points at — the
centre as its own factor times a kernel over the ring, and the MI-weighted Hamming distance — are unrun. What the
numbers DO say, now on five held-out boards with the selection done on the training windows: the block's first
failure was not features — a soft one-shot block beats the whole hard chain (0.891 vs 0.867 at t = 0.5, 0.817 vs
0.789 at t = 0.75); the honest selection costs ≈ 0.001 in accuracy and saves 0.03–0.41 bits per cell over the
held-out one; the chain's gain belongs to memory blocks and vanishes for factored ones; and the price (the entry
price that collapsed E35, and the palette floor that costs every table 10–27 bits on 2.1% of cells) is where the
next loss is, not the block.

---

## 3. Next cheap experiments, in order

Each ≤ 5 minutes on CPU, no gradient, on the existing apparatus (`e34.py`, `e35.py`, the `research/` scripts).
Round 1's items 1–5 and 8 were run in round 2 (`expG`, `expH`, `expJ`, `expI`); what follows is the order the round-2
numbers set. The two round-1 pendings without a run (`expC --train_chars 20000`; `pool_math_check` with
`free_grid(M, step=0.5, top=6.0)`) and the 1.2M per-context oracle stay tagged where they occur and are not
re-listed.

1. **(I) C2 stretch-LDA on ALL 18 experts per mixing context** (36 solves of an 18 × 18 system on counts, the `logD`
   matrix `run_mixtures` already builds, on the probability pool; ~3 min): the direct test of whether the nested
   mis-allocation (+0.032 on chain + order4) sinks the rule at library scale, against A2-min 2.168, `sym_key_cmi@m1`
   2.167, E34 geometric 2.191 and the 3-expert grid 2.128. This is the one measurement §I.4's open regime lacks.
2. **(I) A count-based exponent that is 0 for a function of another source.** Allard eq. 17 by prequential counting
   — the code length of c under KT(K_m) vs KT(K_m, bucket of K_j) — or LDA on the difference feature
   stretch_order4 − stretch_chain; test on chain + order4 first (target: the grid's −0.051, C2's +0.032). The same
   prequential conditional information replaces the plug-in key-CMI of `expJ` (resolvable at any key cardinality, so
   the chain/order8 redundancy becomes measurable and the unresolvable-pair artefact disappears).
3. **(I) `cal_cls` in the whole library, and the 3-expert grid on it.** (a) The A2 pass with `experts[0] =
   EscChain('cal_cls')` (one line in `run_a2_pass`): the 2-expert linear mixer already loses on `cal_cls` (+0.010), so
   how much of E34's 18-expert −0.062 was the chain's escape. (b) chain + word + match per-context grid on `cal_cls`
   (`expG`'s 336-point grid): does the 2.117 floor move once the chain is calibrated AND a second independent partner
   is present? (c) `expA`'s order4 check on `cal_cls` (`--others order4`): is b ≈ 0 once the escape is counted?
4. **(I) The nesting bit used structurally.** Detect r = 1.00 key-function pairs, hand the survivors (chain, match,
   word; order3 if the chain is discounted) to the per-context grid posterior with the partner's count bucket
   (0 / 1–3 / 4–15 / 16+) in the context; against 2.128.
5. **(I) The pool's scale — derive or refute.** Compare, per context, the LDA scale factor with the code-optimal
   temperature from the two grid oracles (`expG`'s JSON holds both): why do stretch-fitted exponents calibrate the
   probability pool at M ≥ 2 and the odds pool at M = 1? A derivation, or a constructed counterexample, before any
   further pool is built on the stretch scale.
6. **(I) A gap-driven mass.** Set the total exponent mass per mixing context online from bits − entropy (flat wants
   2, chain + match wants ≈ 1); the one count-based mass rule T5 round 2 left standing.
7. **(I) Replication and the residual.** The three headline rows (`cal_cls` chain, grid on `cal_cls`, grid + SSE) on
   `check_text_slice_noise`'s second slice (60k–120k; paired differences should move ≤ 0.005); expC's `char` table
   alone on `cal_cls` and PPMII-style inheritance inside `cal` (the −0.030 / −0.035 SSE residual); a forgetting rate
   for the per-character stack table (60k warm-up beats 240k by 0.004–0.006).
8. **(II) The ring kernel and the MI-weighted metric.** P(y | w) ∝ P(y) P(w_c | y) × [the LOO kernel over the 8 ring
   cells], and the Hamming distance weighted by I(cell; y) from the block's own counts, on the five boards against
   soft_loo 0.891 / 0.817 and tan9_m100 0.872 / 0.800.
9. **(II) The tree's m by the same leave-window-out likelihood** (~10 s) reported beside nb9, and a Miller–Madow-
   corrected CMI to check the ring path.
10. **(II) Sleep under the LOO kernel**: drop a stored window when the k = 50–200 vote of the others predicts its
    majority; entries kept vs the five-board accuracy, against E35's 28 811 → 16.
11. **(II) Remove the palette floor** (colour-relabelled key or copy-input fallback for unseen target colours), then
    re-read every all-cells calibration number; and **add MultiKey L0 to training** (or hold out LockPath L1) to
    separate new-board from new-game generalisation.
12. **(II) Radius 2** (25 cells) with the three families (LOO kernel, naive Bayes, tree) on the five boards — the
    crossover of §II.2 item 3, now with an honest selection and enough boards.
13. **(II) Text in bits**: the exact table, the MI-weighted cluster keys and naive Bayes under prequential KT code
    length; then push MI-weighted k-means to 16k–64k with a (near, far) product-quantised key.
14. **(II) Text as an expert**: similarity-weighted order-k contexts (Jensen–Shannon on next-character counts) as one
    more expert in the E34 library, judged by the per-context grid mixture of item 3(b).
15. **(I) A3** (the tree of two-input Bayesian grid mixers over all 18 experts) only after item 1: if the 18 × 18
    LDA solve already lands near the 3-expert grid there is nothing for A3 to add; Heskes' QP (C5) is dropped unless
    it uses the exact ambiguity −log2 Z (measured 0.740 vs the quadratic form's 1.498).
