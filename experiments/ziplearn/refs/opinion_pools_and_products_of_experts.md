# Opinion pools and products of experts — the mathematics of combining probability distributions — reference notes

Written 2026-09-21 for E34's open question (RESULTS.md, "E34"; DESIGN §20 "Where a gradient is needed", point (1)):
eighteen context experts on the Latin stream mixed by Bayesian (linear) weights per mixing context give 1.820
bits/char against the chain's 1.823; the same weights used as exponents of a geometric mixture over the experts whose
context was seen give 1.801; the plain product of the seen experts 7.665. The question this note answers from the
literature: what a linear pool and a logarithmic (geometric) pool each ARE mathematically, when a product of
experts is exactly right, what the exponents of a log pool MEAN, how redundant sources (nested contexts) must be
corrected for, and whether the exponents can be obtained from statistics of the sources in closed form instead of by
a gradient. Question (2) of §20 (features inside a block, E35) is NOT covered here.

Reading discipline: every source below was opened and read (the scanned Genest & Zidek PDF page by page; the others
as extracted text). Quotations are kept to short attributed fragments; the mathematics is restated in one notation.
"From memory" marks the few facts not re-verified today. The feel-experiment that checks the claims is
`experiments/ziplearn/research/pool_math_check.py` → `pool_math_check.json` (17 s, no gradient).

Notation used throughout: M sources (experts) give distributions p_1..p_M over the next symbol x ∈ X (|X| = V = 70 on
the Latin stream); p_0 is the prior (for us: the unigram, `Expert.uni`); w_i ≥ 0 are weights; S_w = Σ_{i≥1} w_i.

## 1. The two pools and the axioms that force each (Genest & Zidek 1986)

**Paper:** Christian Genest and James V. Zidek, *Combining Probability Distributions: A Critique and an Annotated
Bibliography*, Statistical Science 1(1), 114–135 (paper; with discussion to 148), February 1986,
DOI 10.1214/ss/1177013825. Read from the Project Euclid scan (22 pages, image-only PDF, rendered and read page by
page; scratchpad `gz_p*.png`). Verified 2026-09-21.

The paper's structure: §2 what a subjective probability is; §3 the AXIOMATIC pooling formulas (each derived from a
qualitative property a pooling operator T(P_1..P_n) should have); §4 the SUPRA-BAYESIAN approach (a decision maker
treats the experts' stated distributions as DATA and applies Bayes' rule — the only route on which the experts'
DEPENDENCE enters); §5 pools of odds and other encodings; then the annotated bibliography.

**Linear opinion pool (their eq. 3.1):** T = Σ_i w_i P_i, w_i ≥ 0, Σ w_i = 1 (Stone 1961; attributed to Laplace).
McConway (1981) and Wagner (1982) proved independently that T is NECESSARILY of this form if T(P_1..P_n)(A) depends
only on the numbers P_1(A)..P_n(A) — the "strong setwise function property" (eq. 3.2). Consequences they list:
zero preservation (all experts give A probability 0 ⇒ so does the pool); it cannot preserve independence of events
unless it is a DICTATORSHIP (one w_i = 1; Lehrer & Wagner 1983; Dalkey 1972/1975 — the "impossibility theorem").
Relaxing 3.2 to let the function depend on the event A (McConway's weak setwise function property) is equivalent to
the MARGINALIZATION PROPERTY (the pool commutes with coarsening the event space, eq. 3.6), and Aczél–Ng–Wagner 1984 /
Genest 1984c show this still forces a GENERALIZED linear pool T = Σ_{i=0}^n w_i P_i with an extra imposed measure
P_0 and weights in [−1, 1] summing to one (eq. 3.7). Genest & Zidek's verdict on the weights (p. 118): no definite
indications can be given about their choice or interpretation. Empirically (p. 119, citing Staël von Holstein 1970,
Winkler 1971) linear pools are "relatively insensitive to the choice of expert weights."

**Logarithmic opinion pool (their eq. 3.11):** T ∝ Π_i p_i^{w_i} (densities), normalised; in §3 they restrict to
Σ w_i = 1 and say explicitly that formulas of the same form "with unrestricted weights can be derived within the
Bayesian framework" (§4). Properties they state: unlike the linear pool it is typically UNIMODAL and LESS DISPERSED;
it is invariant under rescaling of the individual degrees of belief; with a 0–1 utility it is the Nash product; a
zero from any expert is a VETO ("zeros will constitute vetos", p. 120).

**External Bayesianity (Madansky 1964/1978; eq. 3.12):** pooling COMMUTES with Bayesian updating by a likelihood
common to all experts — update each expert then pool = pool then update. In the authors' opinion "the most compelling
reason for using a logarithmic opinion pool". Genest (1984b, Ann. Statist. 12, 1100–1105) proved that among pools of
the pointwise form T(θ) ∝ F[p_1(θ)..p_n(θ)] (eq. 3.13) the log pool is the ONLY externally Bayesian one, and the
linear pool is not externally Bayesian unless it is a dictatorship (Genest 1984a, JRSS-B 46, 403–405: "a conflict
between two axioms"). Genest, McConway & Schervish (1986, Ann. Statist. 14, 487–501) drop the pointwise regularity and
get the GENERALIZED log pool T ∝ g · Π_i p_i^{w_i} (eq. 3.14) with an arbitrary bounded g that they say should be
read as a LIKELIHOOD, not as a prior. Their closing judgment on weights (p. 120): the log pool "suffers from the same
problem as its linear counterpart" — no normative basis for the weights EXCEPT when the same formula is also derived
by the Bayesian route of §4.

So the two axioms — marginalization (∴ linear) and external Bayesianity (∴ logarithmic) — are INCOMPATIBLE except
for dictatorships, and neither axiom says anything about the weights. What the weights mean has to come from a
MODEL of how the experts came by their opinions. That is §4.

## 2. The supra-Bayesian route: the weights are statistics of the experts (Genest & Zidek §4–5)

The pooling operator is Bayes' rule with the experts' opinions as data: T(θ) = p(θ | p_1..p_n) ∝ p(θ) ℓ(p_1..p_n | θ)
(Winkler 1968; Morris 1974/1977). Everything then hinges on the likelihood ℓ, which must encode the experts'
reliability AND their mutual dependence. Results they collect:

- **Lindley (1985), finite Θ, eq. (4.1) — the key closed form.** Let q_ij = log p_i(θ_j) and, for two states,
  Q = (q_11 − q_12, …, q_n1 − q_n2) be the vector of the experts' LOG ODDS. Assume Q | θ_j is multivariate normal with
  mean μ_j and common covariance Ξ. Then the decision maker's posterior log odds are
      log[T(θ_1)/T(θ_2)] = log[p(θ_1)/p(θ_2)] + (μ_1 − μ_2)ᵀ Ξ⁻¹ [Q − (μ_1 + μ_2)/2].
  Taking antilogs gives (their words) essentially the log pool (3.11) "equipped with weights proportional to the
  independent 'information content' of each assessment" (Freeling 1981). The weight vector is w = Ξ⁻¹(μ_1 − μ_2):
  an expert's exponent is its between-state signal divided by its within-state variance, with Ξ⁻¹ DISCOUNTING experts
  whose log-odds co-vary — i.e. redundant experts. This is linear discriminant analysis on the experts' log-odds. The
  weights are NOT constrained to the simplex and their sum is whatever the covariance says. (4.1) has neither the
  marginalization property nor external Bayesianity except in special cases, which is why Lindley called both
  "essentially adhockeries"; French (1985) notes it obeys a variant of external Bayesianity in which the common data's
  log-likelihood is filtered out of the experts' posteriors.
- **Winkler (1981) / Lindley (1983), continuous Θ, eq. (4.2):** normal likelihood with expert means μ_i, stated
  standard deviations σ_i and a covariance matrix Ξ between the experts' ERRORS; with a diffuse prior the posterior
  mean is μ* = Σ w_i μ_i with w_i = Σ_j ξ^{ij} / Σ_k Σ_j ξ^{kj} (ξ^{ij} the entries of Ξ⁻¹). Winkler observes the
  weights "can sometimes be negative" (the combination is not convex). The off-diagonal of Ξ must be supplied or
  estimated (inverted-Wishart prior; Gokhale & Press 1982 on eliciting correlations).
- **Clemen & Winkler (1985), Operations Research 33, 427–442:** dependent sources are equivalent "to a much smaller
  number of independent sources" (G&Z p. 121–122). Derived here for the record (not quoted): for k sources with
  unit-variance errors and common correlation ρ, the optimal combination has precision k/(1 + (k−1)ρ) times one
  source's — bounded by 1/ρ however many sources are added. This is the "effective number of independent sources".
- **Genest & Schervish (1985), Ann. Statist. 13, 1198–1212, eqs. (4.3)–(4.4):** with only the first moments of the
  experts' probabilities specified, the only coherent pools are T = p + Σ w_i (p_i − μ_i) with w the multiple-regression
  coefficients of the event indicator on the p_i's (weights possibly negative) — the linear pool IS Bayes' rule in
  that case; assuming instead that the experts are CONDITIONALLY INDEPENDENT GIVEN THE EVENT (and given its
  complement) yields "a kind of logarithmic pool with possibly negative weights" (4.4), which exhibits the "group
  polarization" of Bordley (1982, 1983a): when every p_i > μ_i and weights are positive the pool is MORE extreme than
  any expert.
- **Bordley (1982), Management Science 28(10), 1137–1148, their eq. (5.6):** from additive conjoint measurement, the
  law of odds and a "weak likelihood ratio axiom" (= external Bayesianity in odds), the consensual ODDS must be
  T ∝ Π_{i=0}^n p_i^{w_i} with the ONLY constraint Σ_{i=0}^n w_i = 1, p_0 the prior odds. That is: the expert
  exponents w_1..w_n are FREE, and the prior gets exponent w_0 = 1 − S_w, which is NEGATIVE when S_w > 1 — the prior
  is divided out (S_w − 1) times. French (1985) notes p_0 is here a by-product determined after the experts report.
- **Dalkey (1975)** (annotated on p. 130): two log pools, one with every weight 1/n "by analogy to the theory of
  errors", one with every weight 1 by Bayes' theorem applied to "the degree of dependence among the experts". Those are exactly the two ends of the exponent-sum axis (S_w = 1 and S_w = n).

## 3. The modern statement of the same facts (Allard, Comunian & Renard 2012)

**Paper:** D. Allard, A. Comunian, P. Renard, *Probability Aggregation Methods in Geoscience*, Mathematical
Geosciences 44, 545–581 (2012), DOI 10.1007/s11004-012-9396-3. Read in full from the author's PDF
(scratchpad `allard2012.txt`). Verified 2026-09-21. The clearest single reference for this note: a formal review
with the exact decomposition, the property table, and a simulation comparison.

- **Log-linear pool with a prior (their eq. 15):** P_G(A) ∝ P_0(A)^{1 − S_w} Π_{i=1}^n P_i(A)^{w_i}, no restriction on
  w. With Σ_{i=0}^n w_i = 1 always holding, external Bayesianity is satisfied for ANY w. Their reading of S_w
  (p. 552): if S_w = 1 the prior is filtered out and unanimity is preserved; if S_w > 1 the prior has negative weight
  and the pool is always FURTHER from the prior than the unanimous opinion; if S_w < 1 always closer — S_w is the knob that sets the prior's influence.
- **The exact decomposition (their eqs. 16–18) — the theorem behind "redundancy-adjusted exponents".** For any joint
  model, writing D_{<i} = D_1 ∩ … ∩ D_{i−1},
      P(A | D_1..D_n) ∝ P(A) Π_i P(D_i | A, D_{<i})  =  P(A)^{1 − S_w} Π_i P(A | D_i)^{w_{A,D_1..D_n}},
      with  w_{A,D_1..D_n} = ln P(D_i | A, D_{<i}) / ln P(D_i | A).
  This is EXACT with one weight per combination (A, D_1..D_n); log-linear pooling is the simplifying assumption that
  this ratio is one number w_i per source (eq. 18). READ OFF: the exact exponent of a source is the ratio of its
  CONDITIONAL log-likelihood (given the sources already counted) to its MARGINAL log-likelihood. A source that is a
  deterministic function of the sources already counted has P(D_i | A, D_{<i}) = 1, so its exponent is 0. Krishnan
  (2008, Math. Geosci. 40, 705–727) gives the same exponents in odds form for the Tau model; Allard et al. remark that
  computing them needs the full joint, "unfortunately only of academic interest" in geoscience — for us the joint over
  QUANTISED predictions is countable (see §8).
- **Conditional independence ⇔ maximum entropy ⇔ all exponents 1 (their Props. 1–2, eq. 21):** under
  P(D_1..D_n | A) = Π_i P(D_i | A) the pool is P_0^{1−n} Π_i P_i, and this is also the maximum-entropy distribution
  given the univariate and bivariate marginals. Their inclusion chain (eq. 23):
  {max-ent ≡ cond.-ind.} ⊂ {externally Bayesian (Σ_{i=0}^n w_i = 1)} ⊂ {log-linear, w free}.
- **Bordley ≡ Tau model ≡ log-linear, in the binary case only (Props. 3–5).** For K > 2 outcomes the odds-product
  and probability-product pools DIFFER, and the "generalized log-linear" pool (a per-outcome factor ν(A) times the
  weighted product, i.e. G&Z's g) was best in their trinary test (Table 7: log-lik 18 554 vs 18 744 for plain
  log-linear vs 24 124 linear; Brier 0.187 vs 0.189 vs 0.222). Linear pooling was worst in every experiment;
  log-linear best in every binary one; max-entropy (exponents 1) "surprisingly" good on the Markovian Boolean model.
- **Their Theorem 1 (§5.5):** if a CALIBRATED log-linear pooling exists, then asymptotically it is the log-linear
  pooling with parameters estimated by maximum likelihood. Calibration and the ML (gradient) fit coincide.

## 4. Why a linear pool cannot sharpen (Ranjan & Gneiting 2010; Gneiting & Ranjan 2013; Satopää & Ungar 2015)

**Paper:** R. Ranjan and T. Gneiting, *Combining probability forecasts*, JRSS-B 72(1), 71–91 (2010). Abstract read on
the OUP page (verified 2026-09-21): linear pooling is the most popular method, but any non-trivial weighted average of two or more distinct calibrated
forecasts is "necessarily uncalibrated and lacks sharpness" — their Theorem 1 (calibrated, distinct components, positive weights summing to 1 ⇒ the pool is uncalibrated, sits closer to
the climatological base rate than its recalibrated version, and is beaten by recalibration under every strictly proper
score). Their fix, the beta-transformed linear pool (BLP): p̂ = H_{α,β}(Σ w_i p_i) with H the beta CDF — a two-parameter
RECALIBRATION applied after averaging.

**Paper:** T. Gneiting and R. Ranjan, *Combining predictive distributions*, Electronic J. Statist. 7, 1747–1782
(2013), DOI 10.1214/13-EJS823 (text extracted; verified). Theorem 3.1 generalises: a linear pool is at least as
dispersed as its least dispersed component, MORE dispersed if the components are "regular", and OVERDISPERSED if the
components are calibrated ("neutrally dispersed"); the tendency to increase dispersion, they note, explains why linear pooling succeeds in practice: the components
are "frequently underdispersed" (p. 1762).
CORRECTION (2026-09-22, round-1 check against the paper, arXiv:1106.1638): their "geometric pool" with link
h(x) = log x is a generalized linear pool of CUMULATIVE DISTRIBUTION FUNCTIONS, G(y) = h⁻¹(Σ w_i h(F_i(y))) (their
eq. 5, §3.2), not the log-linear pool of densities; and their eq. (7) is the variance of the probability-integral
transform of the SPREAD-ADJUSTED LINEAR POOL (Proposition 3.6). Neither says anything about a log-linear density
pool's dispersion; the sentence that stood here ("eq. (7) bounds the pooled variance on the link scale by (Σ w_i)²
times the largest component variance") had no basis in the paper and is withdrawn. For the Gaussian case the fact
wanted is elementary and is derived here, not cited: for components N(μ_i, 1/τ_i) the log-linear pool ∝ Π N_i^{w_i}
is Gaussian with precision Σ w_i τ_i, so with exponents on the simplex the pool's precision is a convex combination
of the components' (never above the sharpest) and only S_w = Σ w_i > 1 can exceed it. For a DISCRETE alphabet with
V > 2 a simplex geometric pool CAN put more probability on the true symbol than any component, through its
normaliser (Mattern 2013 Example 4.8; `research/check_math_refutations.json` key `ex48_V70`: 0.994 vs 0.6); on a
binary alphabet it cannot (the pooled logit is a convex combination of the logits).

**Paper:** V. A. Satopää and L. H. Ungar, *Combining and Extremizing Real-Valued Forecasts*, arXiv:1506.06405 (2015);
their earlier logit aggregator is Satopää, Baron, Foster, Mellers, Tetlock, Ungar, Int. J. Forecasting 30(2), 344–356
(2014) (abstract only). Under the PARTIAL INFORMATION framework — each "reliable" forecaster reports the conditional
expectation of the outcome given a sub-σ-field of the full information, X_j = E(Y | F_j) — their Theorem 2.1 says the
optimal aggregate E(Y | X_1..X_N) is marginally consistent, reliable and VARIANCE-EXPANDING (at least as variable as
the most variable forecaster), while Theorem 2.2 says every non-trivial weighted average is marginally consistent but
UNRELIABLE and must be "extremized" — moved away from the marginal mean. In the Gaussian specification (their eq. 3):
(Y, X_1..X_N) jointly normal with Cov(X_i, Y) = Var(X_i) = δ_i (the fraction of the information forecaster i holds)
and Cov(X_i, X_j) = ρ_ij (their information OVERLAP), the optimal aggregate is
      E(Y | X) = diag(Σ)ᵀ Σ⁻¹ X          (their "revealed aggregator"),
and — the sentence that matters for us (end of their §4) — fitting the extremized average needs outcomes, whereas "Σ can be learned from the forecasts alone." Their two worked structures: five forecasters with NO overlap ⇒ weights (1,1,1,1,1),
the plain SUM (weights sum to 5); with high overlap (all ρ = 0.12) ⇒ weights (−3, 1, 1, 1, 1): the shared information
is SUBTRACTED. Extremizing matters more "as the forecasters' information sets become more diverse".

## 5. Products of experts, the sum rule, and Bayesian model averaging (Hinton 2002; Kittler et al. 1998; Minka 2002)

**Paper:** G. E. Hinton, *Training Products of Experts by Minimizing Contrastive Divergence*, Neural Computation 14,
1771–1800 (2002). Read (scratchpad `hinton2002.txt`). A PoE multiplies expert distributions and renormalises; the
latent variables of different experts become conditionally independent given the data, so inference is easy and
generation hard. Hinton's argument for the product over the mixture (p. 1772): a mixture's posterior "cannot be
sharper than the individual models in the mixture", whereas each expert of a product "can constrain a different
subset of the dimensions" and the product constrains all of them. All exponents are 1; the sharpness is TRAINED into
the experts (by contrastive divergence, a gradient) rather than put into exponents.

**Paper:** J. Kittler, M. Hatef, R. P. W. Duin, J. Matas, *On Combining Classifiers*, IEEE PAMI 20(3), 226–239
(1998). Read. The PRODUCT rule (their eq. 7) follows from assuming the classifiers' representations are conditionally
statistically independent given the class; the SUM rule is derived from it under the further (stronger) assumption
that the posteriors barely deviate from the priors, and yet the sum rule wins their experiments (e.g. 98.05 % vs
84.69 % on one task). Their §6 explains why: in the product rule each estimation error ε_ki is "amplified by
1/P(ω_k | x_i)" and the amplified errors ADD; in the sum rule errors are not amplified and are damped by the sum of
the posteriors. The product's sharpness and its error amplification are the same mechanism.

**Note:** T. P. Minka, *Bayesian model averaging is not model combination*, 2 pp., 2002 (tminka.github.io). Read.
BMA is "soft model selection": it answers the question "which ONE hypothesis generated all the data", so as data
grows the weight concentrates on the single most probable hypothesis even when a uniform vote of the hypotheses would
be perfect (his three-circles example; error rates 20 % vs 19.99 % still end in a dictatorship). To combine, ask
instead "which linear (or other) combination generated the data" — BMA over the space of combined models.

## 6. Choosing log-pool weights from statistics (Heskes 1998; Kahn 2004; Carvalho et al. 2023)

**Paper:** T. Heskes, *Selecting weighting factors in logarithmic opinion pools*, NIPS 10 (1997), 266–272. Read.
The log pool is the distribution closest to the experts in weighted KL, p̄ = argmin_p Σ_a w_a KL(p ‖ p_a) (his eq. 1;
Σ w_a = 1). His eq. (3) decomposes the pool's error: KL(q, p̄) = Σ_a w_a KL(q, p_a) − A, with the AMBIGUITY
A ≥ 0 so the pool is never worse than the weighted average of its experts and "the larger the ambiguity, the larger
the benefit"; eq. (4)–(5) approximate A by ½ Σ_{a,b} w_a w_b [KL(p_a,p_b) + KL(p_b,p_a)] (exact for Gaussians of
equal variance), which is QUADRATIC in w, so the weights are found by a quadratic program from (i) each expert's own
error and (ii) the pairwise symmetrised KL between experts — the latter measurable WITHOUT the truth. Constraint:
simplex weights only, so this optimises the shape, not the sharpening.

**Paper:** J. Kahn, *A Generative Bayesian Model for Aggregating Experts' Probabilities*, UAI 2004, arXiv:1207.4144
(abstract). A generative model with an event prior and per-expert bias, calibration, accuracy and a dependence measure
"results in a weighted logarithmic opinion pool (LogOps)" that is externally Bayesian, with analytic solutions for
independent and for exchangeable experts — the same construction as Lindley (4.1) with the parameters estimated.

**Paper:** L. M. Carvalho, D. A. M. Villela, F. C. Coelho, L. S. Bastos, *Bayesian Inference for the Weights in
Logarithmic Pooling*, Bayesian Analysis 18(1), 223–251 (2023). Read (scratchpad). They restate the standard reading —
the weights "represent the (relative) reliability of each opinion" — restrict them to the simplex, put a hierarchical
(Dirichlet-type) prior on them and learn them by MCMC from data; they report identifiability problems for some
prior/data configurations. Useful only as confirmation that the simplex-weight literature has no closed form beyond
Lindley's model.

## 7. Context mixing in compression (Mahoney 2005/2013; Mattern 2012/2013; Veness et al. 2017)

**Text:** M. Mahoney, *Data Compression Explained* (mattmahoney.net/dc/dce.html), §4.3. Read. PAQ6's "linear
evidence mixing" (2005) sums weighted bit COUNTS, S_1 = ε + Σ w_i n_1i, weights moved along the coding-cost gradient
and clipped at 0. PAQ7's LOGISTIC mixing: p = squash(Σ_i w_i stretch(p_i)), stretch(p) = ln(p/(1−p)),
w_i ← w_i + λ (y − p) stretch(p_i), λ ≈ 0.01, "the optimal weight update … by taking the partial derivative of the
coding cost"; "Unlike linear mixing, weights can be negative"; one weight set is selected by a small context; ZPAQ's
MIX "does not constrain the weights to add to 1" and "A 2 input MIX often gives better compression than a MIX2"
(which does). SSE/APM (§4.3.3): a COUNT TABLE from (quantised stretched probability, small context) to a refined
probability, "adjusted to reduce the prediction error, typically by about 1 %" — a running mean, no gradient through
anything; it is PAQ's recalibration stage, present since PAQ2 (2003).

**Paper:** C. Mattern, *Mixing Strategies in Data Compression*, DCC 2012, arXiv:1302.2839. Read. Derives geometric
weighting as the minimiser of Σ_i w_i KL(Q ‖ P_i) (his eq. 3, same as Heskes eq. 1), obtaining
P(x) ∝ Π_i P_i(x)^{w_i / Σ_j w_j} (eq. 11 — weights NORMALISED to sum one), shows PAQ7's logistic mixer is exactly
this for a binary alphabet (eq. 20–22), and that weight optimisation is strictly convex. His experiment (Table 1,
Calgary corpus, seven order-0..6 context models + a match model, weight sets keyed by an order-1 context and the match
length, weights renormalised to sum 1 after each step, binary decomposition of bytes, step size 1/16): geometric
beats linear by about 2 % and beta-weighting by 3.6 % on average. The measured gap between geometric and linear
mixing OF THE SAME MODELS is small.

**Paper:** C. Mattern, *Linear and Geometric Mixtures — Analysis*, DCC 2013, arXiv:1302.2820 (abstract read): code
length bounds for online gradient descent with fixed step on "nice" mixtures, holding for arbitrary sequences; the
first theoretical analysis of PAQ7.

**Paper:** J. Veness, T. Lattimore, A. Bhoopchand, A. Grabska-Barwinska, C. Mattern, P. Toth, *Online Learning with
Gated Linear Networks*, arXiv:1712.01897 (2017). Read §2. Geometric mixing geo_w(x=1; p) = Π p_i^{w_i} / (Π p_i^{w_i}
+ Π (1−p_i)^{w_i}) = σ(w · logit(p)); w_i = 1/m is the geometric mean; larger |w_i| means more belief in expert i, a
negative w_i reverses it, w = 0 gives 1/2; every expert "has 'the right of veto'". They name the ancestry — Genest &
Zidek's log pool, Hinton's PoE — and state that the strongest case for it is simply its empirical performance when w is adapted by online
gradient descent; the loss is convex in w (their Prop. 1).

**From memory (not re-verified):** Ng & Jordan, NIPS 2001, on the generative/discriminative pair of the SAME
linear-in-log-odds model (naive Bayes/LDA vs logistic regression): the discriminative fit has the lower asymptotic
error, the generative closed form reaches its (higher) asymptote much faster (O(log n) vs O(n) samples); Efron 1975:
when the Gaussian model holds, logistic regression is only ½–⅔ as efficient as LDA. Knoll & de Freitas 2012
("A machine learning perspective on predictive coding with PAQ8") describe PAQ's mixer as online logistic regression.

## 8. The mathematics, in our words

1. **What each pool IS.** Linear pool = the mixture "exactly one expert is right", weights = the posterior over which;
   with counts it is Bayesian model averaging, which is soft SELECTION (Minka) and converges to a dictatorship (E34:
   mean weight 0.60 on the chain, the rest 0.01–0.05). Log pool = the conjunction "every expert holds evidence";
   exponent 1 each and the prior divided out (n−1) times is EXACT iff the experts' contexts are conditionally
   independent given the symbol (Allard Prop. 2 = Genest & Schervish 4.4 = Kittler's product rule = Dalkey's
   weights-1 pool). The linear pool minimises Σ w_i KL(p_i ‖ p̄); the log pool minimises Σ w_i KL(p̄ ‖ p_i) (Heskes,
   Mattern; Abbas 2009 from memory). Linear preserves marginalisation; log preserves external Bayesianity; no
   non-dictatorial pool does both (Genest 1984a).
2. **What the exponents MEAN (the exact statement).** Allard eq. 17: exponent_i = ln P(c_i | x, c_<i) / ln P(c_i | x),
   the conditional-to-marginal log-likelihood ratio of context i given the contexts already counted. It is 1 for a
   source conditionally independent of the earlier ones, 0 for a source that is a FUNCTION of them, between for
   partial overlap, and can exceed 1 when the sources are conditionally DEPENDENT in the opposite direction (the
   partial-information case: `pool_math_check` "disjoint" wants exponents ≈ 3 each on its grid — the grid's cap).
   S_w = Σ exponents is NOT a count of sources (corrected 2026-09-22): it depends on the scale the pool is written
   in. On the same "disjoint" structure the exact posterior is a geometric pool of the calibrated sources on the
   PROBIT scale with exponents (2.55, 2.65, 2.74), S_w = 7.9, while Satopää's aggregator combines the same sources
   with weights (1.003, 0.998, 1.003), S_w = 3.0, on the conditional-expectation scale
   (`research/check_math_refutations.json` key `disjoint_Sw_two_scales`); the pooled distribution is what is
   invariant. Clemen & Winkler's k/(1 + (k−1)ρ) is the precision of a linear combination of k equicorrelated point
   estimates, a different object, and is not a statement about S_w. S_w = 1 is the conservative,
   correlation-agnostic rule (Covariance Intersection: the geometric mean with simplex weights is consistent for ANY
   correlation — Julier & Uhlmann 1997, Hurley 2002, both from memory beyond Hurley's abstract), S_w = n the
   independence rule on the scale where the sources are conditionally independent.
3. **Why E34's three numbers came out as they did.** `run_mixtures` (e34.py) computes `g` with
   `wg = w[sm] / w[sm].sum()` — exponents forced to sum to 1 (Mattern's eq. 11, CI). With exponents on the simplex
   the geometric pool CAN give the true symbol more probability than every component through its normaliser whenever
   V > 2 (Mattern 2013 Example 4.8; `research/check_math_refutations.json` key `ex48_V70`: 0.994 vs 0.6 at V = 70);
   on a BINARY alphabet with simplex exponents it cannot (the pooled logit is a convex combination of the logits).
   S_w > 1 is a second, separate sharpening. (The sentence that stood here — "by Gneiting & Ranjan eq. 7 such a pool
   cannot be sharper than its sharpest expert; it can only re-shape" — was false for V > 2 and mis-cited, §4;
   corrected 2026-09-22.) So 1.801 below 1.820 is a simplex geometric pool over 70 symbols doing what Example 4.8
   allows, and Mattern's ~2 % (his LIN vs GEO, both learned) is the same move on binary decisions where only
   re-shaping is available. `pr_` uses exponent 1 on every SEEN expert with NO prior division:
   the eighteen experts are mostly NESTED (chain ⊃ order-8 ⊃ … ⊃ order-1; skips and pairs are sub-contexts of the
   orders), so the exact exponents of all but one of them are ≈ 0 and exponent 1 counts the same evidence up to
   eighteen times — 7.665 bits/char is the arithmetic of that, not a property of products (the synthetic nested case
   reproduces it: exponents-1 product 0.474 vs 0.332 for the single longest context). The linear mixture 1.820 is
   Ranjan & Gneiting's theorem: averaging calibrated experts under-sharpens; a recalibration stage (BLP, or PAQ's SSE
   count table) is what a linear pool NEEDS to gain.
4. **Where a multiplicative gain can live at all.** Only in experts that are not functions of one another: word,
   prevword, line, match, and the ORDER experts relative to the chain only through the chain's own backoff
   uncertainty (when the long context's count N is small, the exact conditional P(c_short | x, c_long) is 1 for the
   TRUE distribution but not for the COUNT estimate — the blending/backoff regime, which the linear Bayesian weight
   already handles). Prediction: an exponent fit would put ~0 on orders 1–8 given the chain, and positive exponents
   with S_w > 1 on {word, match, line, skip} — the "diverse sources must be heavily extremized" case of Satopää.

## 9. What it means for us — gradient-free candidates, each with the formula and what it needs

All of these change ONLY the mixing stage of `e34.py` (`Mixer`, `run_mixtures`); the experts, the KT counts, the
mixing contexts (`s.cls() * 9 + deep`) stay. Ranked by how much of the literature's gain they can reach.

- **C1 — the one-scalar sharpening S_w (Bordley's w_0, Allard's S_w, Satopää's a).** In `run_mixtures`, replace
  `wg = w[sm]/w[sm].sum()` by `wg = S * w[sm]/w[sm].sum()` and multiply `logD` by the prior term
  `(1 − S) · log p_0(x)` (the unigram) before normalising. S is one number per mixing context (or one overall),
  chosen by the prequential code on the 30k selection prefix — a grid over {1, 1.5, 2, 3, 4}, no gradient. Cheapest
  test of whether sharpening beyond the simplex is worth anything on text; the nested structure predicts a small
  optimal S (most of the mass is on one expert already). Expected: between 1.80 and PAQ; probably closer to 1.80.
- **C2 — Lindley/LDA closed-form exponents (Genest & Zidek eq. 4.1).** Per bit (binarise the 70 symbols as PAQ does,
  seven binary decisions, each expert's binary probability summed from its `dist`), per mixing context: accumulate
  the class-conditional means μ_1, μ_0 and pooled covariance Ξ of the stretched expert predictions s_i = logit(p_i)
  by COUNTING (running sums of s and s sᵀ per outcome); mix with logit(p) = logit(π) + (μ_1−μ_0)ᵀ Ξ⁻¹ (s − (μ_1+μ_0)/2).
  The weights are exactly PAQ's mixer weights under a Gaussian model of the stretched predictions (the
  generative twin of its logistic regression); Ξ⁻¹ performs the redundancy correction (nested experts co-vary, so
  they share one weight); S_w is whatever the statistics say. Needs the outcome (it is supervised, like PAQ) but no
  gradient. Cost: M = 18 → an 18×18 solve per (bit, context) — fine. How close to the learned mixer: Ng & Jordan's
  regime argument says it reaches a slightly worse asymptote faster; `pool_math_check` puts it within 0.005–0.025
  bits of the optimum on Gaussian data (0.3373 vs 0.3320 nested; 0.3517 vs 0.3270 disjoint; 0.3496 vs 0.3262
  overlap). On text the Gaussian assumption is false, so this is a number to measure, not to predict.
- **C3 — partial-information weights from the forecasts ALONE (Satopää's revealed aggregator).** The probit-scale
  construction for PROBABILITY forecasts is Satopää, Pemantle & Ungar 2016, *Modeling probability forecasts via
  information diversity*, JASA 111(516), 1623–1633 (arXiv:1406.2148) — not Satopää & Ungar 2015 (arXiv:1506.06405),
  which treats real-valued forecasts (§4; corrected 2026-09-22). Per bit and mixing context:
  δ_i = Var(s_i)/(1 + Var(s_i)) on the probit scale (each expert's information fraction, recovered from the spread of
  its own stretched forecast), X_i = s_i √(1−δ_i), Σ = Cov(X), w = diag(Σ)ᵀ Σ⁻¹, output
  Φ(wᵀX / √(1 − diag(Σ)ᵀ Σ⁻¹ diag(Σ))). Unsupervised (no outcome needed) and closed form; exact on its own model
  in all three synthetic structures (0.3333/0.3270/0.3262 against exact 0.3320/0.3270/0.3262; weights (−0.01, 0.09,
  0.94) nested, (1,1,1) disjoint, (0.5,0.5,0.5) overlap — it found the information structure). Its assumption,
  stated: Cov(X_i, Y) = Var(X_i) for every i (each forecast is a calibrated conditional expectation of the outcome
  under a Gaussian latent) — this is what lets Σ be estimated without outcomes; the KT experts violate it per
  mixing context (the synthesis note's T7: the chain's escape is miscalibrated with a depth-dependent sign) and are
  calibrated only in the aggregate, so on text C3 runs under a false premise and its number is to be measured, not
  predicted. The 1/√Var(z|X) factor IS the extremization, derived rather than fitted.
- **C4 — recalibrate the linear mixture by counting (BLP by table = PAQ's SSE/APM).** Keep `Mixer` as is; add a
  count table keyed by (mixing context, quantised stretch(p_lin) in 32 buckets) → refined probability, updated as a
  running mean toward the outcome (this is E24's counting with forgetting, not a gradient). Ranjan & Gneiting say the
  linear pool's whole deficit is calibration/sharpness; Allard's Theorem 1 says a calibrated log-linear pool equals
  the ML one. Per-symbol version: an SSE per bit after binarisation, or a ν(A)-style per-symbol multiplicative
  correction table (Allard's generalized log-linear, the best pool in their K = 3 test). Cheapest of all to build.
- **C5 — Heskes' quadratic program on measured quantities.** Weights on the simplex from each expert's online bits
  and the pairwise symmetrised KL between experts (accumulated from `dist` vectors); solves the shape only. Combine
  with C1 for the sharpening. Convex QP, no gradient on the coding loss, but a fit nonetheless.
- **C6 — the exact tau exponents by counting (Allard eq. 17 on quantised predictions).** exponent_i = the ratio of
  the conditional to the marginal log-likelihood of expert i's prediction bucket given the buckets of the experts
  before it, estimated from counts of (x, bucket_i, bucket_<i) with bucket_<i reduced to the bucket of the single
  best earlier expert (otherwise the joint explodes). NOT DESIGNED beyond this sentence; it is the discrete
  counterpart of C2's Ξ⁻¹.

What none of these is: PAQ's full stack (hundreds of contexts, bit-level nonstationary counters, two mixer layers,
SSE chains). Mattern's 2 % is the honest expectation for geometric-over-linear WITH THE SAME experts; the rest of the
enwik8 gap (1.9 → 1.2–1.3) is the model library and the calibration stages, which are count tables.

## 10. The feel-experiment (`research/pool_math_check.py`, N = 50 000, seed 0, 17 s)

Bits/symbol on a synthetic binary outcome y = 1[z > 0], z = u_1 + … + u_4 with variances (0.35, 0.30, 0.25, 0.10),
three calibrated sources each seeing a subset of the u's; weights by grid search (no gradient):

| structure | best single | exact | linear (simplex) | geometric (sum 1) | geometric (free exponents) | product exp. 1 | C2 LDA | C3 partial-info |
|---|---|---|---|---|---|---|---|---|
| nested {1}⊂{1,2}⊂{1,2,3} | 0.3320 | 0.3320 | 0.3320 | 0.3320 | 0.3320 at (0, 0, 1) | 0.4741 | 0.3373 | 0.3333 |
| disjoint {1},{2},{3} | 0.8172 | 0.3270 | 0.7511 | 0.7180 | 0.3309 at (3, 3, 3) | 0.4596 | 0.3517 | 0.3270 |
| overlap {1,2},{2,3},{3,1} | 0.6069 | 0.3262 | 0.5370 | 0.4733 | 0.3295 at (1, 1, 1) | 0.3295 | 0.3496 | 0.3262 |

Readings: nested ⇒ the exact exponents are (0, 0, 1) and the exponent-1 product is the E34 catastrophe in miniature;
sum-1 geometric ≈ linear ≈ best single (no sharpening possible); disjoint ⇒ the exponent-1 product is NOT exact
(marginal independence ≠ conditional independence given a thresholded latent) and the grid's optimum (3, 3, 3) is
its CORNER (the free grid is capped at 3.0 per exponent, `check_math_refutations.json` key
`pool_math_check_free_grid_cap_per_exponent`) — the analytic optimum on the probit scale is (2.55, 2.65, 2.74),
S_w = 7.9, which equals the exact posterior (a re-run with `free_grid(M, step=0.5, top=6.0)` so that the optimum is
interior is pending); overlap ⇒
the sum-1 pool recovers less than half of the available gain (0.607 → 0.473 vs 0.326), the free exponents nearly all
of it. The closed forms C2 and C3 land within 0.03 bits of the optimum everywhere, C3 exactly (its own model). The
apparatus caveat: these are Gaussian-latent sources; the Latin experts are KT count tables over nested strings.

## 11. Open questions

1. The information structure of E34's eighteen experts is a measurement, not an assumption: per mixing context, the
   matrix of pairwise conditional information between the experts' predictions (C3's Σ, or a count-based mutual
   information) would say how many effective independent sources there are and where any multiplicative gain can
   live. Predicted: ≈ 1 for the chain-plus-orders block, plus fractions for word/match/line.
2. Whether C2 (supervised closed form) or C3 (unsupervised closed form) on the Latin stream approaches a logistic
   mixer's number is unknown; the literature gives the relation only under the Gaussian model (Ng & Jordan, Efron).
   The gradient arm is not to be run under the standing rule; the comparison is C2/C3 against 1.820/1.801 and against
   PAQ-class numbers from the literature.
3. Bit-level versus symbol-level: PAQ mixes binary decisions; our experts emit 70-way distributions. Allard et al.
   show the odds and probability pools separate for K > 2 and that a per-symbol factor (their ν(A), G&Z's g) matters
   there — whether one shared exponent per expert across 70 symbols is too coarse is untested.
4. Whether recalibration by counting (C4) after the Bayesian mixture recovers most of what exponents would give —
   Ranjan & Gneiting's theorem makes the linear pool's deficit a calibration deficit; Allard's Theorem 1 makes a
   calibrated pool the ML pool. If C4 alone closes the gap, the exponent question dissolves.
5. The nested contexts' exact exponent is 0 only for the TRUE conditionals; for COUNT estimates the shorter context
   carries information exactly when the longer context's count is small (N in `Expert.predict`). Whether the right
   structure is "linear across the backoff chain, multiplicative across the diverse experts" (a two-level pool) is
   not designed.
6. Non-Gaussian closed forms for the exponents (C6) are not designed; the Tau-model literature computes them from
   the full joint only.
