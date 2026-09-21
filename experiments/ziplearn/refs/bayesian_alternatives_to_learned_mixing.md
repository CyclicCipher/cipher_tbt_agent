# Bayesian / closed-form alternatives to learned mixing weights — reference notes

Written 2026-09-22 after E34 (`RESULTS.md`, "2026-09-22 — E34"): eighteen context experts on the Latin stream, mixed by
the Bayesian (linear) weights of `e34.py: Mixer` per mixing context, give 1.820 bits/char on the held-out book against
1.823 for the blended order-8 table alone (`Chain`); the same experts combined geometrically, with the Bayesian weights
renormalised over the experts whose context was seen used as exponents (`run_mixtures`, key `geometric`), give 1.801; the
plain product of the seen experts (exponent 1 each) gives 7.665 (`runs/e34/e34_mixtures.json`). PAQ's mixer learns the
exponents by an online gradient step on the coding loss, and E34's verdict called that "the first irreducible use of a
gradient". The standing rule (2026-09-22): no gradient patch; research the mathematics first. This note answers the
question the rule poses: can the exponents of a multiplicative mixture be replaced by a Bayesian average, or be inferred
in closed form from counts — and at what cost.

Every quote below was read in the source on 2026-09-21/22 (PDFs fetched and text-extracted; abstracts checked against
the arXiv API where an arXiv version exists). Where a statement is my own derivation it says so. No PDFs are copied
beside this file: the sources are IEEE, Springer, Wiley and Kluwer papers (not Creative Commons) plus arXiv preprints.

## 0. The vocabulary, fixed by two theorems (Mattern 2012)

Let m experts give distributions P_1..P_m over the alphabet at one step; w ≥ 0 are weights.

- **Linear mixture** P(x) = Σ_i w'_i P_i(x), w' = w/Σw. Mattern, *Mixing Strategies in Data Compression*, DCC 2012
  (arXiv:1302.2839), §4.2: it is the solution of `argmin_Q Σ_i w_i D(P_i ‖ Q)` — "the normalized weights w'_i correspond
  to the switching probabilities W(i|x^{k−1}). Thus, the cost function in (25) would be proportional to the expected
  redundancy of a switching source": the model class is *one expert at a time*. With w'_i = the Bayesian posterior
  W(i)P_i(x^{k−1})/P(x^{k−1}) this is **β-weighting** (Kufleitner, Binder, Fries 2009, a CTW spin-off), which is exactly
  `Mixer.update` without its fixed-share line.
- **Geometric mixture** P(x) ∝ Π_i P_i(x)^{w_i}. Same paper, eq. (3): `P := argmin_Q Σ_i w_i D(Q ‖ P_i)` — the divergences
  point the other way, "P_i plays the role of a model distribution and we seek an approximate source distribution".
  For a binary alphabet it is PAQ7's logistic mixer: σ(w·logit(p)). Both weight-estimation problems are strictly convex
  (Mattern 2012 §3.2, §4.2). The abstract, verbatim: "We propose geometric weighting as a novel method to combine
  multiple models in data compression. Our results reveal the rationale behind PAQ-weighting and generalize it to a
  non-binary alphabet. Based on a similar technique we present a new, generic linear mixture technique. All novel mixture
  techniques rely on given weight vectors. We consider the problem of finding optimal weights and show that the weight
  optimization leads to a strictly convex (and thus, good-natured) optimization problem. Finally, an experimental
  evaluation compares the two presented mixture techniques for a binary alphabet. The results indicate that geometric
  weighting is superior to linear weighting."

**The one measurement in the literature of "Bayesian weights vs learned exponents, same experts"** is Mattern 2012
Table 1 (Calgary corpus, bits per character; eight models: order-0..6 context models + a match model; weight sets
selected by an order-1 context and the match length; GEO and LIN weights by online gradient descent with step 1/16 and
1/32; BETA = the Bayesian posterior weights, floored at 2^−8):

| file | GEO (learned exponents) | LIN (learned linear) | BETA (Bayesian linear) |
|---|---|---|---|
| book1 | 2.212 | 2.304 | 2.313 |
| book2 | 1.864 | 1.943 | 1.965 |
| paper1 | 2.274 | 2.327 | 2.343 |
| average (14 files) | 2.187 | 2.231 | 2.265 |

"On average LIN compresses about 2% and BETA compresses about 3.6% worse than GEO" — on text (book1) 4.4%. So with the
SAME eight experts, learning the exponents is worth ≈4% over Bayesian linear weights; the rest of PAQ's distance to
PPM-class compressors comes from its models (hundreds of contexts, bit-level, SSE), not from the mixer. For E34 the
analogue is 1.820 → ≈1.75, not 1.820 → 1.3.

Mattern, *Linear and Geometric Mixtures — Analysis*, DCC 2013 (arXiv:1302.2820), Remark 4.7, for weights on the simplex
S: "max_{w∈S} geo(x; w, P) ≥ max_i geo(x; e_i, P) = p_max(x; P) = max_{w∈S} lin(x; w, P)": a linear mixture can never
give the true symbol more probability than the best expert gives it; a geometric mixture can, even with exponents
summing to one, through its normaliser (mass an expert rules out is removed — GLN's "right of veto"). That is why
E34's `geometric` (1.801) beats `linear` (1.820) with the same weights, and why exponents summing to MORE than one
(PAQ's are unconstrained) sharpen further — up to the product's 7.665 when eighteen near-duplicate contexts are each
given exponent 1.

## 1. Sources

### Cover 1991 — Universal Portfolios; Cover & Ordentlich 1996; Blum & Kalai 1999

T. M. Cover, *Universal Portfolios*, Mathematical Finance 1(1):1–29, January 1991 (Wiley; abstract not fetchable, the
paper's text is in a custom font). T. M. Cover and E. Ordentlich, *Universal portfolios with side information*, IEEE
Trans. Inform. Theory 42(2):348–363, 1996. A. Blum and A. Kalai, *Universal Portfolios With and Without Transaction
Costs*, Machine Learning 35(3):193–205, 1999 (read: cs.cmu.edu/~avrim/Papers/portfolio.pdf).

The construction (Blum & Kalai, Definition 1): "The universal portfolio algorithm, UNIVERSAL, at time i is specified by
b̂_i = ∫_β b S_{i−1}(b, x^{i−1}) dμ(b) / ∫_β S_{i−1}(b, x^{i−1}) dμ(b), i = 1, 2, … with μ equal to the uniform
distribution over portfolios b." and "wealth of UNIVERSAL = E_{b∈β}[wealth of CRP b]" — i.e. it is the BAYESIAN MIXTURE
over the whole simplex of constant rebalanced portfolios (fixed convex combinations), with the uniform (Cover 1991) or
Dirichlet(½,…,½) (Cover & Ordentlich 1996) prior. Bound (Blum & Kalai Theorem 1, "as in (Cover and Ordentlich, 1996a)"):
"wealth of UNIVERSAL / wealth of best CRP ≥ (n+m−1 choose m−1)^{−1} ≥ 1/(n+1)^{m−1}, for all markets with m stocks
and n periods" (the binomial is typeset as a coefficient in the source); the Dirichlet(½) version "has the better
guaranteed ratio of" — the text extraction reads "2√(1/(n+1)^{m−1})", which cannot be a lower bound on a ratio ≤ 1
for small n, so a fraction bar was lost; Cover & Ordentlich 1996 Theorem 2 as I remember it (not fetched, Wiley/IEEE
paywalled): S_n^*/Ŝ_n ≤ 2(n+1)^{(m−1)/2}, i.e. the universal wealth is within a factor ½(n+1)^{−(m−1)/2} of the best CRP.
In code-length terms: (m−1) log n bits of regret with the uniform prior (Blum & Kalai Theorem 3 below), ((m−1)/2) log n
+ 1 with Dirichlet(½). The proof for the uniform prior is one paragraph: portfolios within α of b* earn at least
(1−α)^n of b*'s wealth and occupy an α^{m−1} fraction of the simplex; take α = 1/(n+1).

**The language-model reading is in the same paper** (Blum & Kalai §5): "A price relative x_ij corresponds to a
conditional probability P_j(w_i|h_{i−1}) and a portfolio b corresponds to a daily mixture of language models … the
probability that a combined language model algorithm assigns to a sequence h^n … is simply the value of the holdings
of the corresponding portfolio algorithm." Theorem 3: "the log probability assigned to h^n by the universal algorithm P
is at least the log probability assigned to h^n by the best mixture, minus (m−1) log n." — a closed-form (no learned
parameter) predictor that competes with the best FIXED LINEAR mixture, not merely the best expert (which is all
β-weighting / `Mixer` competes with; CTS eq. 3 below).

Cost: "The implementation presented in (Cover and Ordentlich, 1996a) has space and time requirements which grow like
n^{m−1}" (Li & Hoi survey, arXiv:1212.2129 §3.2.1: "time complexity O(n^m)"; Kalai & Vempala 2002 give a randomised
polynomial-time version, "O(m^7 n^8)"). Blum & Kalai's practical remedy is the RANDOM GRID: "First choose N portfolios
uniformly at random. Then invest a 1/N fraction of the money in each of the N CRPs and let it sit … using
N = (R−1)/ε²δ random portfolios, with probability at least 1−δ, the approximation achieves a wealth at least 1−ε times
as large as the universal algorithm."

What it gives us: the exact template for "a mixture over the simplex replacing weight learning" — but for LINEAR
mixtures, whose ceiling is the best-expert probability (§0). Its transfer to exponents is §2.

### Hazan 2016/2019 — EWOO; the continuous Bayesian mixture over any exp-concave parametrisation

E. Hazan, *Introduction to Online Convex Optimization*, arXiv:1909.05207 (book), §4.3 "Exponentially Weighted Online
Convex Optimization", Algorithm 11: weights w_t(x) = e^{−α Σ_{τ<t} f_τ(x)} over the convex set K, play the mean
x_t = ∫_K x w_t(x) dx / ∫_K w_t(x) dx. Theorem 4.4: "Regret_T(EWOO) ≤ (d/α) log T + 2/α." The bibliographic note: "The
EWOO algorithm was essentially given in Cover's paper for the application of portfolio selection and logarithmic loss
functions, and extended to exp-concave loss functions in [Hazan et al., 2006]. The randomized extension of Cover's
algorithm that runs in polynomial running time is due to Kalai and Vempala [2003], and it naturally extends to EWOO."
And: "The downside of EWOO is its running time. A naive implementation would run in exponential time of the dimension."

This is the general statement: for ANY parametrised predictor whose per-step loss is α-exp-concave in the parameter,
the Bayesian average over the parameter space (posterior ∝ prior × likelihood, prediction = the posterior mean's
prediction or the posterior-averaged prediction) has logarithmic regret against the best fixed parameter, with no
learned parameter and no step size. Geometric mixing qualifies (next source), which makes "Bayesian mixture over the
exponent vectors" a well-defined closed-form object; §2 computes its cost and its useless constant.

### Veness et al. 2017 / 2019 — Gated Linear Networks (what they prove; no Bayesian GLN exists)

J. Veness, T. Lattimore, A. Bhoopchand, A. Grabska-Barwinska, C. Mattern, P. Toth, *Online Learning with Gated Linear
Networks*, arXiv:1712.01897 (Dec 2017, 40 pp.). Abstract: "This paper describes a family of probabilistic architectures
designed for online learning under the logarithmic loss. Rather than relying on non-linear transfer functions, our
method gains representational power by the use of data conditioning. We state under general conditions a learnable
capacity theorem that shows this approach can in principle learn any bounded Borel-measurable function on a compact
subset of euclidean space; the result is stronger than many universality results for connectionist architectures
because we provide both the model and the learning procedure for which convergence is guaranteed."

J. Veness, T. Lattimore, D. Budden, A. Bhoopchand, C. Mattern, A. Grabska-Barwinska, E. Sezener, J. Wang, P. Toth,
S. Schmitt, M. Hutter, *Gated Linear Networks*, arXiv:1910.01526 (v2 June 2020; AAAI 2021). Abstract: "This paper
presents a new family of backpropagation-free neural architectures, Gated Linear Networks (GLNs). What distinguishes
GLNs from contemporary neural networks is the distributed and local nature of their credit assignment mechanism; each
neuron directly predicts the target, forgoing the ability to learn feature representations in favor of rapid online
learning. Individual neurons can model nonlinear functions via the use of data-dependent gating in conjunction with
online convex optimization. We show that this architecture gives rise to universal learning capabilities in the limit,
with effective model capacity increasing as a function of network size in a manner comparable with deep ReLU networks.
Furthermore, we demonstrate that the GLN learning mechanism possesses extraordinary resilience to catastrophic
forgetting, performing comparably to a MLP with dropout and Elastic Weight Consolidation on standard benchmarks. These
desirable theoretical and empirical properties position GLNs as a complementary technique to contemporary offline deep
learning methods."

The neuron (2019 §2): "geometric mixing predicts σ(w^⊤σ^{−1}(p)) … = Π p_i^{w_i} / (Π p_i^{w_i} + Π (1−p_i)^{w_i}),
which makes it clear that geometric mixing implements a type of product of experts [Hin02] … setting w_i = 1/d is
equivalent to taking the geometric mean … every forecaster has 'the right of veto'". Gating (2019 §3): "associated with
each neuron is a context function c : Z → C … the context function c is responsible for mapping a given piece of side
information z_t ∈ Z to a particular row w_{c(z_t)} of W, which we then use with standard geometric mixing" — halfspace
gates on the input, random directions; E34's 36 mixing contexts (`Stream.cls()` × `Chain.deepest`) are a context
function in this sense. Stacked, "the network behaves like a linear network, but with weight matrices that are
data-dependent" (eq. 5).

What is proved (2017 Proposition 1, 2019 §2 and §5): "∇ℓ_t^geo(w) = (geo_w(1; p_t) − x_t) logit(p_t)"; "ℓ_t^geo(w) is a
convex function of w"; "If p_t ∈ [ε, 1−ε]^m … ℓ_t^geo : W → R is α-exp-concave with α = σ(log(ε/(1−ε)) max_{w∈W}
‖w‖_1)"; "‖∇ℓ_t^geo(w)‖_2 ≤ √m log(1/ε)". Hence: "Online Gradient Descent (OGD) [Zin03] with a learning rate
proportional to 1/√t has total regret of O(√T) with respect to the best w* ∈ W chosen in hindsight. The loss function
ℓ_t is exp-concave, so Online Newton Step [HAK07] can improve the regret to O(log T), but is computationally more
expensive." With a fixed learning rate, "O(s√T) regret guarantees with respect to data sequences composed of s pieces
(Mattern, 2013)". Capacity (2017 Theorem 1) is asymptotic, for i.i.d. data, given a no-regret learner per neuron.

**Is there a Bayesian or closed-form GLN?** Not in either paper, and a web search for "Bayesian gated linear network",
"universal geometric mixing", "Bayesian mixture of geometric mixers" (2026-09-22) found none; the only Bayesian
ingredient GLNs use is at a different level — 2017 §5 "Adaptive Regularization via Sub-network Switching": a Bayesian
mixture over SEQUENCES OF NEURONS with the run-length prior w_τ(ν_{1:n}) = w_τ(ν_{<n}) × ((n−1)/n · 1[ν_n = ν_{n−1}] +
1/(n(|M|−1)) · 1[ν_n ≠ ν_{n−1}]), bound "−log w_τ(ν_{1:n}) ≤ (s(ν_{1:n}) + 1)(log|M| + log n)" (eq. 18), computed in
O(|M|) per step by u_ik^{(t+1)} = 1/((t+1)(|M|−1)) + ((t|M| − t − 1)/((t+1)(|M|−1))) · u_ik^{(t)} ρ_ik(x_t|x_{<t}) /
τ(x_t|x_{<t}). That is `Mixer.update` with a DECAYING switch rate α_t = 1/(t+1) instead of the fixed 0.02. The
"Gaussian GLN" (Budden et al., NeurIPS 2020) is the regression variant, also gradient-trained. The Bayesian
counterpart of the neuron itself — a mixture over its exponent vectors — is a construction, not a published result;
§2 states it.

### Herbster & Warmuth 1998 — fixed share; Volf & Willems 1998 — the switching method; Koolen & de Rooij 2013

M. Herbster and M. K. Warmuth, *Tracking the Best Expert*, Machine Learning 32:151–178, 1998. Abstract (excerpt): "The
generalization allows the sequence to be partitioned into segments, and the goal is to bound the additional loss of
the algorithm over the sum of the losses of the best experts for each segment … When the number of segments is k+1 and
the sequence is of length ℓ, we can bound the additional loss of our algorithm over the best partition by
O(k log n + k log(ℓ/k)) … As in the original algorithms, we keep one weight per expert, and spend O(1) time per weight
in each trial." The log-loss form (Koolen & de Rooij Theorem 4, "Herbster and Warmuth [6]"): with k experts, a
comparator sequence with m blocks, α* = (m−1)/(t−1) and H the cross-entropy, "ln P_{ξ^t}(x^t)/FS_{w,α}(x^t) ≤ m ln k +
(t−1) H(α*, α)" — the (t−1)H term is the price of a wrong fixed α: E34's α = 0.02 charges −log2(0.98) = 0.029 bits
per character of slack in the bound, ten times the 0.003 bits/char the mixture gains over the chain (the bound is not
tight, but it says the fixed rate is not free).

P. A. J. Volf and F. M. J. Willems, *Switching between two universal source coding algorithms*, DCC 1998, pp. 491–500
(Snowbird). Not fetched; characterised by Koolen & de Rooij §IV-D: "Volf and Willems describe an algorithm called the
switching method, which is very similar to Herbster and Warmuth's Fixed Share, except that it is able to learn the
optimal switching rate α on-line … the switching method 'integrates out' the parameter using Jeffreys' prior (which is
Beta[½, ½])." Theorem 12: "For any switching rate α and data x^t, ln FS_{w,α}(x^t)/SM_w(x^t) ≤ ln 2 + ½ ln t" — the
switch rate is learned in closed form (a KT estimator over the switch/no-switch bits) at a total price of ½ log t; but
"the number of states quadratic. The quadratic running time O(kt²) restricts its use to moderately sized data sets."
The Veness 2012 decaying rate α_t = 1/t (CTS Lemma 1: "−log2 w(i_{1:n}) ≤ (m(i_{1:n}) + 1)(log2|M| + log2 n)") keeps
O(|M|) per symbol and the same order of bound. W. M. Koolen and S. de Rooij, *Universal Codes from Switching
Strategies*, IEEE Trans. IT 59(11):7168–7185, 2013 (arXiv:1311.6536) is the unified account (fixed share, the switching
method, run-length, "elementwise mixtures" = linear mixtures as HMMs).

What it gives us: the switching family is the closed-form answer to WHICH expert, over time. It never combines two
experts' evidence at one step (Koolen & de Rooij's "Bayesian mixture" is exactly β-weighting). Relevant to E34 only as
a free tweak to `Mixer.update` (§3, candidate F).

### Willems, Shtarkov & Tjalkens 1995 — CTW; Veness et al. 2012 — CTS; Veness et al. 2013 — PTW

F. M. J. Willems, Y. M. Shtarkov, T. J. Tjalkens, *The context-tree weighting method: basic properties*, IEEE Trans.
IT 41(3):653–664, 1995. Abstract (TU/e research portal): "Describes a sequential universal data compression procedure
for binary tree sources that performs the 'double mixture.' Using a context tree, this method weights in an efficient
recursive way the coding distributions corresponding to all bounded memory tree sources, and achieves a desirable
coding distribution for tree sources with an unknown model and unknown parameters. Computational and storage complexity
of the proposed procedure are both linear in the source sequence length. The authors derive a natural upper bound on
the cumulative redundancy of the method for individual sequences. The three terms in this bound can be identified as
coding, parameter, and model redundancy. The bound holds for all source sequence lengths, not only for asymptotically
large lengths … the proposed context-tree weighting procedure is optimal in the sense that it achieves the Rissanen
(1984) lower bound". The recursion (as restated in CTS eq. 14): CTW^c_D = ½ ξ_KT(x^c) + ½ CTW^{0c}_{D−1} CTW^{1c}_{D−1};
the class C_D of suffix trees has |C_5| = 458 330 members and the mixture over ALL of them costs O(D) per symbol; the
bound (CTS Theorem 2): redundancy ≤ Γ_D(S) + |S| γ(n/|S|) + 2 with Γ_D(S) ≤ 2|S|−1 the code length of the tree and
γ(k) = ½ log2 k + 1 the KT parameter cost per leaf.

J. Veness, K. S. Ng, M. Hutter, M. Bowling, *Context Tree Switching*, DCC 2012 (arXiv:1111.3182). Abstract: "This paper
describes the Context Tree Switching technique, a modification of Context Tree Weighting for the prediction of binary,
stationary, n-Markov sources. By modifying Context Tree Weighting's recursive weighting scheme, it is possible to mix
over a strictly larger class of models without increasing the asymptotic time or space complexity of the original
algorithm. We prove that this generalization preserves the desirable theoretical properties of Context Tree Weighting
on stationary n-Markov sources, and show empirically that this new technique leads to consistent improvements over
Context Tree Weighting as measured on the Calgary Corpus." Theorem 3: "−log2 CTS_D(x_{1:n}) < Γ_D(S) + [d(S)+1] log2 n +
|S| γ(n/|S|) − log2 Pr(x_{1:n}|S, Θ_S)". Their Table 1 (bits per byte, depth 48, KT leaves, no enhancements): book1
CTW 2.31 / CTS 2.32; enhanced CTS at depth 160: book1 2.18, bib 1.77; Table 2, size-weighted Calgary average: PPM* 2.09,
CTW 1.99, PPMZ 1.93, CTS* 1.93, DEPLUMP 1.89. Note the switch rate: "our first thought was to use α_n^c = n_c^{−1}.
However this choice gave poor empirical performance … a much better alternative was to set α_n^c = n^{−1} for any
sub-context."

J. Veness, M. White, M. Bowling, A. György, *Partition Tree Weighting*, DCC 2013 (arXiv:1211.0587). Abstract: "This
paper introduces the Partition Tree Weighting technique, an efficient meta-algorithm for piecewise stationary sources.
The technique works by performing Bayesian model averaging over a large class of possible partitions of the data into
locally stationary segments. It uses a prior, closely related to the Context Tree Weighting technique of Willems, that
is well suited to data compression applications. Our technique can be applied to any coding distribution at an
additional time and space cost only logarithmic in the sequence length. We provide a competitive analysis of the
redundancy of our method, and explore its application in a variety of settings. The order of the redundancy and the
complexity of our algorithm matches those of the best competitors available in the literature, and the new algorithm
exhibits a superior complexity-performance trade-off in our experiments." Corollary 2: against any switching model with
|S(i_{1:n})| segments, "−log τ(x_{1:n}) + log ξ_{i_{1:n}}(x_{1:n}) ≤ (2 + κ)|S(i_{1:n})|(⌈log n⌉ + 1)".

What these give us: the paradigm of "a closed-form Bayesian mixture over an exponentially large class computed in
logarithmic time" — over TREES (which context to use), over TIME (when to switch). In every case the recursion works
because the class factorises (a tree is its two subtrees; a partition is two half-partitions) and the mixture is
LINEAR at each node. CTW's own answer to "combine orders" is the ½/½ node weighting — the `Chain`'s blending is its
PPM cousin — and it stops at PPM-class numbers (Table 2 above): none of these mixes evidence from two contexts
multiplicatively. That is not an oversight; the class of a geometric mixture with continuous exponents does not
factorise, so no CTW-style recursion exists for it (my statement, not a theorem in the sources).

### Mattern 2013 — bounds for OGD-learned exponents; Soft-Bayes 2017 — tempered Bayes; Heskes 1997 — weights from KL statistics

Mattern 2013 (above), abstract: "Both mixture schemes rely on weight vectors, which heavily determine their performance.
Typically weight vectors are identified via Online Gradient Descent. In this work we show that one can obtain strong
code length bounds for such a weight estimation scheme. These bounds hold for arbitrary input sequences … The results
hold for PAQ7 mixtures as well, thus we provide the first theoretical analysis of PAQ7." Table 1 gives, for a step size
∝ 1/√n, code length ≤ Σ over s pieces of the best fixed-weight code length + O(m s B² √n) (B = the bits any expert can
cost per symbol). §5 notes: "Veness [9] gave a bound for linear mixtures using a non-OGD weight estimation scheme which
is identical to Table 1 row 2 except the penalty term, which is O(s log n) in place of O(s√n)." — the Bayesian
(switching) weights have the BETTER bound, for the linear mixture; the gap to geometric is in the class, not the rate.

L. Orseau, T. Lattimore, S. Legg, *Soft-Bayes: Prod for Mixtures of Experts with Log-Loss*, ALT 2017 (arXiv:1901.02230).
Abstract: "We consider prediction with expert advice under the log-loss with the goal of deriving efficient and robust
algorithms. We argue that existing algorithms such as exponentiated gradient, online gradient descent and online Newton
step do not adequately satisfy both requirements. Our main contribution is an analysis of the Prod algorithm that is
robust to any data sequence and runs in linear time relative to the number of experts in each round. Despite the
unbounded nature of the log-loss, we derive a bound that is independent of the largest loss and of the largest
gradient, and depends only on the number of experts and the time horizon. Furthermore we give a Bayesian
interpretation of Prod and adapt the algorithm to derive a tracking regret." The update: w_{t+1}^i = w_t^i (1 − η +
η p_t^i / M_t) — "if η = 1 … w_t^i is the posterior of the Bayesian mixture … the algorithm does not compete with convex
combinations of experts. From a Bayesian perspective, there is no reason to believe that it should because the convex
combinations lies outside the class of the learner." Theorem 3: R_T(a) ≤ 2√(T N ln N) + ln N against the best fixed
convex combination a, in O(N) per step. So `Mixer.update` with η < 1 ("slowed-down Bayes") would compete with the best
fixed LINEAR mixture at linear cost — the cheap cousin of Cover's algorithm — still inside the linear ceiling.

T. Heskes, *Selecting Weighting Factors in Logarithmic Opinion Pools*, NIPS 10 (1997), pp. 266–272. Abstract (excerpt):
"If we interpret the output of a network as a probability statement, the sum-squared error corresponds to minus the
loglikelihood or the Kullback-Leibler divergence, and linear averaging of the outputs to logarithmic averaging of the
probability statements: the logarithmic opinion pool. The crux of this paper is that this whole story about model
averaging, bias/variance decompositions, and quadratic programming to find the optimal weighting factors, is not
specific for the sum-squared error, but applies to the combination of probability statements of any kind in a
logarithmic opinion pool, as long as the Kullback-Leibler divergence plays the role of the error measure." The
decomposition, eq. (3)–(5): K(q, p̄) = Σ_a w_a K(q, p_a) − A, with the "ambiguity" A ≈ ½ Σ_{a,b} w_a w_b [K(p_a,p_b) +
K(p_b,p_a)] (exact for Gaussians, third-order error otherwise): "The righthand side of this expression is quadratic in
the weighting factors w_a"; with Σ w = 1, w ≥ 0 "the approximation (5) leads to a quadratic programming problem"; "The
solution of the quadratic programming problem usually ends up at the edge of the unit cube with many weighting factors
equal to zero." The inputs are COUNT STATISTICS: each expert's mean code length K(q, p_a) and the pairwise mean
symmetric divergences K(p_a, p_b) + K(p_b, p_a), both running averages over the stream. Also cited there: the pool is
"externally Bayesian" (Genest & Zidek 1986, Bordley 1982) — the log-linear pool is the only pooling rule that commutes
with Bayesian updating.

### Mahoney — PAQ's own history of count-based mixing and SSE (*Data Compression Explained*, §4.3)

"Early versions expressed predictions as pairs of counts (observed zeros and ones) and added them together. This
implicitly gave greater weight to predictions near 0 or 1 because such predictions can only occur when one of the
counts is large." — PAQ1–PAQ3's mixer was closed form: p = Σ_i n1_i / Σ_i (n0_i + n1_i) with fixed per-model weights
("Early versions used fixed weights"). §4.3.1, PAQ6 (2005): "S0 = ε + Σ_i w_i n0_i … p1 = S1/S … Weights are adjusted
in the direction that minimizes coding cost in weight space" — the first gradient. §4.3.2, PAQ7: "p = squash(Σ_i w_i
stretch(p_i)) … w_i := w_i + λ (y − p) stretch(p_i) … λ … typically around 0.01 … Unlike linear mixing, weights can be
negative. Compression can often be improved by using a set of weights selected by a small context, such as a bytewise
order 0 context." And the reason given: "Mattern (2012) proved that logistic mixing is optimal in the sense of
minimizing Kullback-Leibler divergence, or wasted coding space, of the input predictions from the output mix."

§4.3.3 SSE/APM, the OTHER gradient-free gain: "SSE (secondary symbol estimation) is implemented in all PAQ versions
beginning with PAQ2. Like in ppmonstr, it inputs a prediction and a context and outputs a refined prediction. The
prediction is quantized typically to 32 or 64 values on a nonlinear scale with finer resolution near 0 and 1 and
sometimes interpolated between the two closest values. On update, one or both values are adjusted to reduce the
prediction error, typically by about 1%. A typical place for SSE is to adjust the output of a mixer using a low order
(0 or 1) context. SSE components may be chained in series with contexts typically in increasing order." A table of
(context, quantised probability) → probability, moved toward each outcome by a fraction: a discounted count ratio, i.e.
the same object as E24's tables with forgetting, not a learned exponent.

Calibration from the same document's `calgary.tar` table (3 152 896 bytes; sizes → bits/char): ctw 997 381 (2.53);
ppmd J -o16 754 243 (1.91); lpaq1 (seven contexts + one logistic mixer + two APMs — from memory of its source) 682 512 (1.73); ppmonstr J (PPMII:
counts, information inheritance, SSE — no logistic mixer) 669 497 (1.70); paq8l 595 533 (1.51). The gradient-free
PPMII+SSE line reaches the same place as one logistic mixer; the last 11% is hundreds of models plus mixer networks.

## 2. What is closed form, and what it costs (derived here from the sources above)

Take E34's numbers: m = 18 experts, alphabet V = 70, test book T = 240 588 characters, mixing contexts C = 36
(`n_ctx = 4 * 9`), each expert's distribution floored by the add-V/2 smoothing in `Chain.dist`/`Expert.dist`.

**A. A Bayesian mixture over a FINITE set G of exponent vectors** (the exact analogue of Cover's construction, for the
geometric mixture). Predict P(x) = Σ_{g∈G} π_t(g) geo_g(x); after x_t, π_{t+1}(g) ∝ π_t(g) geo_g(x_t) — `Mixer.update`
applied to "experts" that are themselves geometric mixers. By the dominance inequality (CTS eq. 3, GLN 2017 eq. 16):
total bits ≤ min_{g∈G} bits(geo_g) + log2|G|, exactly, for every sequence, with no step size, no ε and no
exp-concavity constant; per mixing context, log2|G| bits each. Fixed share on top (Herbster–Warmuth) tracks a changing
best g at (s+1)(log2|G| + log2 T) bits. Cost per symbol: |G| × m × V multiply-adds for the geometric mixers (one
(|G|×m)·(m×V) matrix product on the log-probability matrix that `run_mixtures` already builds as `logD`) plus |G| × V
exponentials; the posterior update is |G| multiplies. The regret against the best CONTINUOUS w* adds the grid error:
|ℓ_t(w) − ℓ_t(w')| ≤ √m log(1/ε) ‖w − w'‖ (GLN Prop. 1(4b)), so a grid of spacing δ loses at most T √m log(1/ε) δ √m
nats — with a full m-dimensional grid the count |G| = (L/δ)^m is the whole problem: at 3 levels per exponent, 3^18 =
3.9·10^8 mixers per symbol. The grid must be low-dimensional:
  - A1, **one temperature**: g = β · w_Bayes with β ∈ {½, ¾, 1, 1½, 2, 3, 4, 6} (8 mixers; 3 bits per context; ~8×70
    exponentials per symbol — negligible). Tests exactly the hypothesis E34 left open: that the exponents' SHAPE (the
    Bayesian weights) is right and only their total mass (1 for `geometric`, 18 for `product`) is wrong. Precedent in
    forecasting: Satopää et al., Int. J. Forecasting 30(2):344–356, 2014, aggregate logit = a × mean logit, one
    "extremizing" parameter (the abstract, via search, says "a single, easily-interpretable parameter"; the
    a × mean-logit form is from memory; paper not fetched).
  - A2, **one exponent per GROUP of experts** (chain | orders 1–8 + skips + pairs | word + prevword + line | match):
    4 dimensions × 6 levels = 1296 mixers; per symbol a (1296×18)(18×70) product = 1.6 M flops → ~1 min over the
    book in numpy; 10.3 bits per context. Grouping by hand is a prior, not a rule; the honest version is A3.
  - A3, **a tree of two-input mixers, each a Bayesian mixture over a 2-D grid** — the Bayesian counterpart of a GLN
    neuron (a construction; not in the literature): mixer k combines two inputs with exponents (u, v) on a 15×15 grid
    of [0, 3]² (225 points; 7.8 bits per context per mixer), 17 mixers for 18 experts, each input a distribution over
    V, cost 17 × 225 × 70 ≈ 2.7·10^5 flops per symbol; total prior cost 17 × 36 × 7.8 ≈ 4 800 bits ≈ 0.02 bits/char on
    the book. Stacked geometric mixers stay geometric (GLN eq. 5: "the logit and sigmoid functions cancel"), so the tree
    represents any product-form exponent vector; whether the tree's greedy-per-node Bayes finds what a jointly learned
    vector finds is the open question (§4).

**B. The continuous Bayesian mixture (EWOO / Cover) over the exponent hypercube.** Closed form as an integral,
regret (d/α) log T + 2/α (Hazan Thm 4.4) with α = σ(log(ε/(1−ε)) max‖w‖₁) (GLN Prop. 1(4a)): for ε = 10^−4 and
‖w‖₁ ≤ 1, α ≈ 10^−4 and the bound is ≈ 2·10^6 nats — vacuous on a 240k-character book (my arithmetic). The bound that
matters for log-loss is the prior-mass one: a uniform prior on [0, L]^m puts mass (δ/L)^m within δ (per axis) of w*,
and by Prop. 1(4b) every step's loss differs by at most √m log(1/ε) · δ√m = m log(1/ε) δ nats there, so bits ≤
bits(w*) + m log2(L/δ) + T·m·log(1/ε)·δ/ln 2; choosing δ = 1/(T m log(1/ε)) gives ≈ m log2(T m L log(1/ε)) + 1.4 ≈
18 × 25 + 1.4 ≈ 450 bits (L = 1; +28 bits for L = 3) ≈ 0.002 bits/char (my derivation from GLN Prop. 1(4b), which is
stated for the binary mixer, and the dominance inequality; the V-ary mixer's gradient has the same log(1/ε) form). So
the PRICE of
inferring 18 exponents Bayesian-style is negligible; only the INTEGRAL is expensive (Kalai–Vempala sampling, or the
posterior is log-concave — convex loss — so its mode is the convex programme Mattern 2012 §3.2 describes, which is the
thing we are not allowed to run by gradient). Blum & Kalai's random grid does not rescue the 18-dimensional case: N
random points give N^{1/18} effective levels per axis (10^4 points → 1.7 levels).

**C. Weights in closed form from count statistics (Heskes).** Per mixing context keep, as running means over the
stream, each expert's code length k_a = ⟨−log2 p_a(x)⟩ (18 numbers, already the `solo` array in `run_stream`) and the
pairwise symmetric divergences S_ab = ⟨D(p_a‖p_b) + D(p_b‖p_a)⟩ (153 numbers, from the same `logD` matrix). Solve
min_w Σ_a w_a k_a − ½ Σ_ab w_a w_b S_ab subject to Σw = 1, w ≥ 0 — an 18-variable QP whose active-set steps are linear
solves; refresh every few thousand symbols. No gradient on the coding loss; the price is the third-order approximation
(Heskes eq. 5) and Σw = 1 (no sharpening beyond the normaliser), which A1's temperature supplies on top. This is the
one candidate that infers a full 18-vector without a grid; it is an approximation whose error is not bounded in the
source.

**D. Switching / tempering the linear mixture** (Veness 2012 α_t = 1/(t+1); Volf–Willems; Soft-Bayes η < 1): closed
form, O(m) per symbol, and provably competitive with the best fixed convex combination (Soft-Bayes Thm 3, Blum–Kalai
Thm 3) — but inside the linear ceiling of §0. Expected effect on E34: within the 0.029 bits/char fixed-share slack;
cannot produce the multiplicative gain.

**E. SSE/APM on the output** (PAQ2+, ppmonstr): a counted table (context × 33 quantised stretch(p) buckets, interpolated)
that recalibrates the chain's or the mixture's probability of the predicted symbol; gradient-free by construction
(counts with forgetting). For a 70-symbol alphabet the binary SSE becomes a recalibration of the top-1 probability
with the rest rescaled, or a binary decomposition of the alphabet (Mattern 2012 §5 and CTS* do the latter).

## 3. Ranked candidates for E34 (gradient-free, each a short script under `research/`)

| # | mechanism | learned parameters | closed-form price (bits, total) | cost per symbol (m=18, V=70) | tests |
|---|---|---|---|---|---|
| A1 | Bayes over 8–11 temperatures on the Bayesian-weight geometric mixture | none | 3.5 × 36 contexts ≈ 125 | 11 geo mixers ≈ 10^4 flops | is total exponent mass the missing DOF? (§3b: no; per-context temperature, yes a little) |
| A3 | tree of 17 two-input Bayesian grid mixers (225 points each) | none | ≈ 4 800 | ≈ 2.7·10^5 flops | can greedy per-node Bayes over exponents match a learned 18-vector? |
| A2 | 4 group exponents × 6 levels (§3b ran the 2-group, 42-point version) | none (grouping is a hand prior) | 10.3 × 36 ≈ 370 | ≈ 1.6·10^6 flops | as A3, with a cheaper, hand-shaped basis (§3b: −2.7% vs the chain) |
| C | Heskes QP from running KL statistics (+ A1 temperature) | none (a linear solve) | not bounded in the source | ≈ 18² running means + a 18-var QP per refresh | do count statistics alone locate the exponents? |
| F | `Mixer.update` with α_t = 1/(t+1) (Veness 2012) or Soft-Bayes η | none | (s+1)(log2 18 + log2 T) | unchanged | measured 0.014–0.015 bits/char on the grid posteriors (§3b); for `Mixer` itself still inside the linear ceiling |
| E | SSE/APM on the chain's output, keyed on (`cls`, `deepest`, quantised p_top) | none (counts) | none (a table) | 1 table lookup | the PPMII/ppmonstr route, orthogonal to mixing |

The measure for all of them is the one E34 used: online bits/char on De Bello Civili after the 2.7M-character training
pass (`run_mixtures` reports `linear`, `geometric`, `product`; a new key per candidate). What would refute the
closed-form thesis: A3 and C both landing within 0.005 bits/char of `geometric` (1.801) — then the exponents' value is
not in their mass or their count statistics but in the per-symbol adaptation a gradient provides, and the honest record
is "the exponent learner is a gradient"; a learned-exponent arm run ONCE as the reference number would settle how much
is at stake (Mattern's Table 1 predicts ≈ 4%: ≈ 1.75).

## 3b. Measured (2026-09-22): A1 and the smallest A2, in one script

`experiments/ziplearn/research/mix_temperature_bayes.py` (reuses `e34.py` unchanged; 300 000 training characters and
the first 60 000 of the held-out book, so every number here sits ≈0.4 bits/char above E34's full-data numbers; each
run ≈55 s CPU). Outputs `research/mix_temperature_bayes.json` (fixed share α = 0.02 on the grid posteriors, as in
`Mixer`) and `research/mix_temperature_bayes_decay.json` (`--switch decay`: α_t = 1/(t+1) per mixing context, the
Veness 2012 rate). Online bits/char on the 60k slice:

| predictor | fixed share 0.02 | decaying rate |
|---|---|---|
| chain alone (grid point β_chain = 1, β_rest = 0) | 2.229 | 2.229 |
| linear Bayesian mixture (E34 `linear`) | 2.211 | — (the `Mixer` itself was left at 0.02) |
| geometric, Bayes weights, β = 1 (E34 `geometric`) | 2.191 | 2.191 |
| geometric at the best single β in hindsight | 2.191 (β = 1.0; β = 1.25 gives 2.193, β = 2 gives 2.467) | same |
| **A1**: Bayes over 11 temperatures β ∈ {0.5 … 8} | 2.198 | **2.184** |
| **A2-min**: Bayes over the 42-point grid (β_chain ∈ {0.5 … 2}) × (β_rest ∈ {0 … 3}) | 2.183 | **2.168** |
| best single grid point in hindsight | 2.203 (β_chain 0.75, β_rest 0.5) | same |
| product of the seen experts (E34 `product`) | 8.558 | — |

Readings. (i) The total exponent MASS is not the missing degree of freedom: the best fixed β is 1.0 and the landscape
is steep on both sides (0.75 → 2.285, 1.5 → 2.252). (ii) The SHAPE is: moving mass from the chain to the other
experts' geometric mean (0.75, 0.5) beats β = 1 by 0.012 bits/char alone, and the per-context posterior over the grid
beats its own best fixed point by 0.035 — different mixing contexts want different exponents (the A1 posterior-mean
temperatures range from 1.1 in shallow-context cells to 2.0 in the deepest letter cell). (iii) The switching prior is
not innocuous at this scale of gain: fixed share 0.02 costs A1 0.014 and A2-min 0.015 bits/char against the decaying
rate, the same order as the gains being measured — which is the Herbster–Warmuth slack term of §1 made visible; E34's
`Mixer` still runs at 0.02. (iv) Against the chain the closed-form ladder is linear −0.8%, geometric −1.7%, A1 −2.0%,
A2-min −2.7%; Mattern 2012's learned exponents gained 3.6% (average) and 4.4% (book1) over Bayesian weights with eight
experts on a different corpus — so a 2-parameter closed-form grid with the right switching prior already recovers a
large part of what learning the exponents is worth in the one published comparison. The next rungs are A3 (a tree of
two-input grid mixers, the full shape without a hand grouping) and running the decaying rate inside `Mixer` itself;
the learned-exponent arm run once would give the reference number the rule forbids building on.

## 4. Open questions (not designed)

- A3's grid is per node; the joint optimum over 18 exponents need not be reachable by greedy per-node posteriors — a
  bound for a tree of Bayesian geometric mixers against the best joint exponent vector does not exist in the sources.
- The switch-rate lesson of CTS ("α_n = n^{−1} for any sub-context" beat the per-context n_c^{−1}) has no analogue yet
  for a grid over exponents: how fast should a mixing context's posterior over G forget?
- Whether E34's eighteen experts contain evidence that a product CAN combine (the gain is bounded by what the experts
  know jointly and the chain does not): the product's 7.665 says their errors are highly dependent; the Heskes matrix
  S_ab measures that dependence directly and would say, before any mixer is built, how much is there.
- Beyond mixing: PAQ's remaining gap to a table is models + SSE (§1, calibration), which are all counting; the note
  does not touch problem (2), features inside a block.
