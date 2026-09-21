# Context mixing (PAQ, lpaq, ZPAQ, cmix) and Mattern's mixing theory — which parts are counted, which are gradient — reference notes

**Why this note exists.** E34 (`experiments/ziplearn/e34.py`, `runs/e34/*.json`) found that a library of eighteen context
experts on the Latin stream gains nothing over E33's blended table when mixed by Bayesian (linear) weights (1.820 vs
1.823 bits/char), gains 0.02 as a geometric mixture with the Bayesian weights as exponents (1.801), and over-sharpens as a
raw product (7.665). DESIGN §20 then named "combining evidence multiplicatively with LEARNED exponents" as the first place
ZipLearn seems to need a gradient. This note reads the compressor literature that owns that gain and separates, component
by component, what is learned by counting from what is learned by a gradient step, how big the gradient part is, and what
geometric mixing buys over linear in theory and in measured bits. Verified 2026-09-21 against the sources below.

**Sources (all read in full unless noted).**
- Matt Mahoney, *Adaptive Weighing of Context Models for Lossless Data Compression*, Florida Tech TR CS-2005-16 (2005),
  6 pages. http://mattmahoney.net/dc/cs200516.pdf — local copy `mahoney_2005_adaptive_weighing_cs200516.pdf`.
- Matt Mahoney, *Data Compression Explained* (2010–2012, Dell Inc.; "cited quotations totaling less than one page" are
  fair use under its licence; the rest is paraphrased). http://mattmahoney.net/dc/dce.html — §4.1.2 (bitwise counters),
  §4.1.3 (indirect models), §4.3 (context mixing), §4.3.1 (linear), §4.3.2 (logistic), §4.3.3 (SSE), §4.3.4 (ISSE),
  §4.3.5 (match), §4.3.6 (PAQ history), §4.3.7 (ZPAQ).
- Matt Mahoney, `lpaq1.cpp` (July 24 2007, GPL v2), the smallest complete PAQ-class program: 7 models, one mixer, two
  SSE stages, 846 lines. http://mattmahoney.net/dc/lpaq1.zip — local copy `lpaq1.cpp`. Its header comments are the
  clearest written account of each update rule; the code is the ground truth for the fixed-point constants.
- Christopher Mattern, *Mixing Strategies in Data Compression*, DCC 2012 (arXiv:1302.2839, IEEE preprint, "personal use
  permitted"). Local copy `mattern_2012_mixing_strategies_1302.2839.pdf`.
- Christopher Mattern, *Linear and Geometric Mixtures – Analysis*, DCC 2013 (arXiv:1302.2820). Local copy
  `mattern_2013_linear_geometric_mixtures_1302.2820.pdf`.
- Christopher Mattern, *On Statistical Data Compression*, PhD thesis, TU Ilmenau, defended 2016-01-18
  (urn:nbn:de:gbv:ilm1-2016000010). NOT read: the repository (db-thueringen.de) serves the PDF behind a proof-of-work bot
  challenge and CORE has no copy; the thesis is the compilation of the two DCC papers above plus arXiv:1311.1723
  (relative frequencies with discount) and arXiv:1501.01202 (exponential smoothing), whose abstracts were read. Its
  abstract as indexed by the search engine: the models and mixers considered "perform nearly as well as idealized
  competitors that may adapt to the input", and PAQ's mixing "is a special form of the Geometric Mixture Distribution
  coupled with Online Gradient Descent".
- Byron Knoll, cmix v21 (2024-09-10): http://www.byronknoll.com/cmix.html and the source at
  https://github.com/byronknoll/cmix (`src/mixer/mixer.cpp`, `mixer/sse.cpp`, `mixer/mixer-input.cpp`,
  `predictor.cpp`, `models/direct.cpp`, `models/indirect.cpp`, `models/match.cpp`, `states/nonstationary.cpp`,
  `states/run-map.cpp` read in full).
- Veness, Lattimore, Bhoopchand, Grabska-Barwinska, Mattern, Toth, *Online Learning with Gated Linear Networks*,
  arXiv:1712.01897 (abstract only) — the modern name for the PAQ mixer architecture.
- Matt Mahoney, Large Text Compression Benchmark, http://mattmahoney.net/dc/text.html (the enwik8/enwik9 table, read
  2026-09-21) — the numbers in §4.

Notation used below: a probability p, its stretch st(p) = ln(p/(1−p)), squash(x) = 1/(1+e^−x) = st^−1; y ∈ {0,1} the
coded bit; bits/char on enwik8 = compressed bytes × 8 / 10^8.

## 1. The PAQ pipeline, component by component

A PAQ-class compressor codes ONE BIT at a time (the byte's 8 bits, most significant first, the already-coded bits of the
current byte being part of every context). Per bit the pipeline is: contexts → per-context statistics → a probability
per model → the mixer → one or two SSE stages → the arithmetic coder. Only the mixer (and ZPAQ's ISSE) hold weights
moved by a gradient; everything else is a table filled by counting or a fixed designed table.

### 1.1 Direct counters (gradient-free)

DCE §4.1.2: a context holds a prediction and a count; the update is `m.prediction += (bit - m.prediction) / (m.count +
DELTA)` with the count capped at `LIMIT`, so the estimate is the running mean while the count grows ("stationary") and an
exponential moving average with rate 1/(LIMIT+DELTA) after ("adaptive"). ZPAQ fixes DELTA = 1/2, LIMIT ∈ {4, …, 1020};
the DCE table shows LIMIT 32 best on the Calgary corpus order-0 (1,683,890 bytes) and LIMIT 1020 best on the digits of π
(415,566), i.e. the discount is a per-source choice. cmix `models/direct.cpp` is this rule verbatim in floats
(`predictions_ += (bit - predictions_) * 1/(count + delta)`, `limit` 30, `delta` 0 for the direct models).

`lpaq1.cpp` class `StateMap` (the map from a bit-history state or a context to a probability): 22-bit probability + 10-bit
count in one 32-bit word, `dt[i] = 16384/(i+i+3)`, update `t[cxt] += (((y<<22)-p)>>3)*dt[n]`. Checked numerically
(2026-09-21): the effective rate is 1/(n+1.5) with n the count BEFORE increment, i.e. the header's "p := p + (y −
p)/(n + 0.5)" with n after increment; the closed form is the KT-type estimator with pseudo-count 1/4,
p = (n1 + 1/4)/(n + 1/2), until n reaches `limit` (1023 for the models, 255 for the APMs), then an EMA at rate
1/(limit + 1.5).

### 1.2 Bit-history state machines and indirect models (gradient-free, one fixed designed table)

DCE §4.1.3: "An indirect model learns the answer by observing what happened after similar sequences appeared. The model
uses two tables. The first table maps a context to a bit history, a state representing a past sequence of bits. The
second table maps the history to a prediction, just like a direct context model." The state is one byte; the transition
table is FIXED (lpaq1/paq8l: 253 states; ZPAQ: 219 states, "determined experimentally"); `lpaq1.cpp` `State_table[256][2]`
is the whole thing. States 1–30 record every history of up to 4 bits exactly; the rest record a count pair (n0, n1) plus
the last bit when n0+n1 < 16, with bounds "(41,0), (40,1), (12,2), (5,3), (4,4)" and the rule that "when a bit is
observed and the count of the opposite bit is large, then part of this count is discarded to favor newer data over old."
The second table is the `StateMap` above (counting). In ZPAQ the second table has "no count and the learning rate is
fixed at 1/4096 of the error. The initial prediction for each bit history is (n1 + 0.5)/(n0 + n1 + 1)." Why it wins over a
direct counter (DCE table: Calgary order-3 964,942 indirect vs 1,017,354 direct): the model LEARNS how much to trust a
history like 0000000001 instead of assuming stationarity, and the state costs one byte instead of four. cmix's
`states/nonstationary.cpp` is a 256-state table of this kind; `states/run-map.cpp` a 256-state run-length machine.

The 2005 TR's earlier form, before state machines (§3): the "semi-stationary update rule" — "If bit x occurs then
increment n_x; if n_{1−x} > 2 then set n_{1−x} = floor(n_{1−x}/2 + 1)" — so 0000000000 then 11111 gives (2,5), p1 = 5/7,
not the stationary 1/3.

### 1.3 The match model (gradient-free)

Find the last occurrence of the current high-order context (a hash to a pointer into the history buffer); predict the bit
that followed; the CONFIDENCE is a counter indexed by the match length (lpaq1: a `StateMap` over (predicted bit, match
length 1..15 or quantised by 4 up to 62, last byte); cmix `models/match.cpp`: `predictions_[match_length]` updated by the
counter rule with `limit` 200). ZPAQ's fixed rule: "the next bit will be the same with probability 1 − 1/8L" for a match
of L bytes. E34's match expert (2.88 bits/char solo) is this component.

### 1.4 Linear evidence mixing, PAQ1–PAQ6 (2002–2004): fixed weights, then a gradient on counts

TR 2005 eq. (1): each model emits counts (n0_i, n1_i); S0 = ε + Σ_i w_i n0_i, S1 = ε + Σ_i w_i n1_i, p1 = S1/(S0+S1).
PAQ1 used FIXED weights: "The eight general purpose contexts of length n are weighted w = (n + 1)^2." PAQ4 (Oct 2003)
introduced the gradient: eq. (2) "w_i ← max[0, w_i + (x − p1)(S n1_i − S1 n_i) / S0 S1]", "adjusted along the cost
gradient in weight space", with "8 sets of weights, selected by a 3 bit context consisting of the 3 most significant bits
of the previous whole byte" for 18 contexts. "Experimentally, equation (2) was found to be remarkably robust. The only
tuning parameter is ε, and even this has very little effect on compression ratio over a wide range." The count pair
itself carries the confidence: DCE §4.3 — "This implicitly gave greater weight to predictions near 0 or 1 because such
predictions can only occur when one of the counts is large."

### 1.5 Logistic mixing, PAQ7 onward (Dec 2005): the geometric mixture with gradient-learned exponents

DCE §4.3.2, verbatim: "p = squash(Σ_i w_i stretch(p_i))", "stretch(p) = ln(p/(1 − p))", "squash(x) = stretch^−1(x) =
1/(1 + e^−x)". "The probability computation is essentially a neural network evaluation taking stretched probabilities as
input. Again we find the optimal weight update by taking the partial derivative of the coding cost with respect to the
weights. The result is that the update for bit y (0 or 1) is simpler than back propagation (which would minimizes RMS
error instead). w_i := w_i + λ (y − p) stretch(p_i) where λ is the learning rate, typically around 0.01, and (y − p) is
the prediction error. Unlike linear mixing, weights can be negative." "Compression can often be improved by using a set
of weights selected by a small context, such as a bytewise order 0 context."

`lpaq1.cpp` header: "w_i := w_i + L * t_i * (y − p) where (y − p) is the prediction error and L ~ 0.002 is the learning
rate. This is a standard single layer backpropagation network modified to minimize coding cost rather than RMS
prediction error (thus dropping the factors p * (1 − p) from learning)." Code: `err = ((y<<12)-pr)*7;
w[i] += t[i]*err+0x8000 >> 16` with t_i the 8-bit-scaled stretch (range ±2047 = ±8) and p 12-bit. "One of 80 neural
networks are selected by a context that depends on the 3 high order bits of the last whole byte plus the context order
(quantized to 0, 1, 2, 3, 4, 6, 8, 12, 16, 32)." So lpaq1's entire gradient-learned state is 80 × 7 = 560 integer
weights. Weights start at 0 (`calloc`), i.e. p = 1/2 until trained. ZPAQ (DCE §4.3.7) widened weights to 20-bit signed
with range ±8 and has three mixer kinds: AVG (fixed user weights), MIX2 (two inputs, weights constrained to sum to 1,
selected by a context), MIX (any number of inputs, unconstrained).

The rule IS a gradient step — of the coding loss −log p(y) with respect to the weights of ONE logistic unit, on a
CONVEX loss (§3 below). It is not backpropagation: there is no chain rule through anything, and each mixer's inputs
are the models' outputs, which the update does not touch. What makes it a gradient rather than a count is that the
target of a weight is not observable — no (input, target) pair exists for w_i, only the loss.

### 1.6 SSE / APM: refinement by counting (gradient-free)

DCE §4.3.3: "Like in ppmonstr, it inputs a prediction and a context and outputs a refined prediction. The prediction is
quantized typically to 32 or 64 values on a nonlinear scale with finer resolution near 0 and 1 and sometimes interpolated
between the two closest values. On update, one or both values are adjusted to reduce the prediction error, typically by
about 1%. A typical place for SSE is to adjust the output of a mixer using a low order (0 or 1) context." "The table is
initialized so that the output prediction is equal to the input prediction for all contexts." Introduced by Serge Osnach
in PAQ2 (May 2003, 64 levels, no interpolation), interpolation in PAQ3 (Sept 2003); TR 2005 §4.2: a "2-D table which
inputs the probability from the mixer … and a small 10-bit context (the partially coded byte and match prediction)".

`lpaq1.cpp` class `APM` (a `StateMap` subclass): the input is stretched and mapped to 24 bins over [−8, 8], the output is
the linear interpolation of the two neighbouring cells, "and then only the nearest entry is updated" with the StateMap
counting rule, cells "initialized to p = pr and n = 6 (to slow down initial adaptation) with a limit n <= 255"; two
stages in series, contexts = the partial byte `c0` (256 contexts) and a hashed order-1 context (2^14 contexts); each
stage's output is averaged with its input, "weighted by 1/4": `pr = pr + 3*a1.pp(y, pr, c0) >> 2`. Sizes: 256×24 = 6,144
and 16,384×24 = 393,216 cells, all count-learned. cmix's final SSE (`mixer/sse.cpp`, "written by Eugene Shelwien") is
the same idea in two interpolated 7-bin tables (`SSEi<7>`) whose update is an EMA toward the bit, `X.P = X.P*(SCALE-wr0)
>> SCALElog; if (c==0) X.P += wr0;`, each followed by a ONE-weight two-input mixer (`struct Mixer { int w; … }`) that
blends the SSE output with its input; the mixer's `w` moves by `e*(p0−p1)*wq` — a gradient step on a single weight.

What SSE buys, from the PAQ Calgary history (DCE §4.3.6 table): PAQ1 716,704 → PAQ2 (+SSE) 702,382 (−2.0%) → PAQ3
(interpolated SSE) 696,616 (−0.8%). ppmonstr (Shkarin) gets its lead over ppmd from the same device on the PPM side:
"a direct context model taking as input the quantized prediction and a (very complex) context" (DCE §4.2.2).

ISSE (DCE §4.3.4, paq9a/ZPAQ): a bit-history state selects the pair of weights of a 2-input mixer whose inputs are the
incoming prediction and the constant 4.0, weights initialised to (1, 0); so per state 2 gradient-learned weights, and the
chain of ISSEs in increasing order IS ZPAQ's default model (mid.cfg). Its motivating sentence is worth keeping for E35:
"a large model with lots of free parameters (each table entry is a free parameter) will overfit the training data and
have no predictive power for future input. As a general rule, a model should not be larger than the input it is trained
on."

### 1.7 cmix (Knoll, 2014–2024): the same pipeline at scale, plus one true backprop component

From the cmix page: "cmix v21 uses a total of 2,077 independent models." "cmix uses a similar neural network architecture
to paq8. This architecture is also known as a gated linear network. cmix uses three layers of weights. Loss function:
cross entropy + L2 regularizer. Activation function: logistic. Optimization procedure: stochastic gradient descent.
Every neuron in the network directly tries to minimize cross entropy, so there is no backpropagation of gradients between
layers." "Only a small subset of neurons are activated for each prediction. The activations are based on manually
defined contexts (i.e. functions of the recent input history). One neuron is activated for each context." "Instead of
using a global learning rate, each context set has its own learning rate parameter. There is also learning rate decay."

`src/mixer/mixer.cpp`: inputs are the stretched model probabilities (`mixer-input.cpp` clamps p to [1e−4, 1−1e−4] and
takes the logit through a 100,001-entry table); `Mix()` is a dot product with the weight vector of the context's
`ContextData` (at most 10,000 weight sets per mixer, then a shared fallback set); `Perceive(bit)`:
`decay = 0.9 / pow(1e-7*steps + 0.8, 0.8); decay *= 1.5 − steps_of_this_set/max_steps; update = decay * learning_rate *
(Logistic(p) − bit); weights -= update * inputs;` and every 1,024 updates `weights *= 1 − 3e−6` (the L2 term).
`predictor.cpp` builds layer 0 from 26 mixers each keyed by its own context (orders 0–3 hashes, recent bytes, line
break, longest match, the WRT dictionary context, five hand-drawn 256-entry byte-class "interval" maps, …) with per-mixer
learning rates from 0.00005 to 0.005; layer 1 has 20 mixers over layer 0's outputs plus the PAQ8/fxcm "auxiliary"
predictions; layer 2 is one mixer (rate 0.0003), then `sse_.Predict(p)`. The byte-level LSTM (`ByteMixer`; the constructor is `Lstm(vocab, vocab, 200, 2, 100, 0.03, 10)`, read as 200 cells,
2 layers; trained by BPTT with Adam — "The byte-level mixer uses long short-term memory (LSTM) trained using
backpropagation through time") is the one component in this lineage that is real backpropagation; it entered at v12
(Nov 2016): LTCB enwik8 v11 15,566,358 → v12 15,440,186 (−0.8%).

Size of cmix's gradient-learned part: per mixer up to 10,000 sets × (~2,077 + extra) floats, allocated lazily; 47
mixers (26 + 20 + 1); so up to ~10^9 weights in principle, in practice bounded by the contexts that occur. The models' tables (hashed
bit histories, counters) take most of the 21 GB.

## 2. Which components are learned by which mechanism — the table

| component | what it stores | learned by | free parameters |
|---|---|---|---|
| context hashing / selection | which bytes/bits form each context | designed (fixed) | 0 |
| bit-history state table (`State_table`, `Nonstationary`) | 253/219/256-state transition graph | designed once, "experimentally" | 0 at run time |
| direct counter / `StateMap` | p per context or per state, with count | COUNTING: p += (y−p)/(n+δ), n capped | one p (+count) per cell |
| indirect model | context → state → p | counting (both tables) | one byte per context + 256 p's |
| match model | p per match length | counting | ~64 p's |
| PAQ1 mixer | (n+1)² per order | fixed | 0 |
| PAQ4–6 linear mixer | w_i ≥ 0 over count pairs | GRADIENT of coding cost, one linear unit | 18 × 8 (PAQ4) |
| PAQ7+/lpaq/ZPAQ/cmix logistic mixer | w_i ∈ ℝ per weight set | GRADIENT of coding cost, one logistic unit per weight set, no chain rule | lpaq1: 560; cmix: up to ~10^9, sparse |
| SSE / APM | calibrated p per (context, stretch bin) | COUNTING (StateMap rule / EMA), interpolated | lpaq1: 399,360 cells |
| ISSE (ZPAQ) | 2 weights per bit-history state | gradient | 2 × 256 per component |
| cmix LSTM byte mixer | 2 × 200 cells | BACKPROP through time (Adam) | ~10^5–10^6 |
| learning rates, `limit`, `delta`, bin counts, 1/4 blends | scalars | hand-tuned per program | tens |

The dividing line is exactly the one DESIGN §20 drew: a cell whose target is OBSERVABLE (a probability of the bit that
then arrives; a calibrated probability for an input bin) is learned by counting; a weight whose only feedback is the
loss is learned by a gradient of that loss with respect to that weight alone.

## 3. Mattern: what the logistic mixer is, why its weights are learnable, and what the bounds assume

### 3.1 Geometric vs linear mixture as two divergence minimisations (DCC 2012)

Abstract (verbatim): "We propose geometric weighting as a novel method to combine multiple models in data compression.
Our results reveal the rationale behind PAQ-weighting and generalize it to a non-binary alphabet. Based on a similar
technique we present a new, generic linear mixture technique. All novel mixture techniques rely on given weight vectors.
We consider the problem of finding optimal weights and show that the weight optimization leads to a strictly convex
(and thus, good-natured) optimization problem. Finally, an experimental evaluation compares the two presented mixture
techniques for a binary alphabet. The results indicate that geometric weighting is superior to linear weighting."

The two constructions, in the paper's equations. Given model distributions P_1..P_m over an alphabet X and weights
w_i ≥ 0:
- eq. (3): P := argmin_Q Σ_i w_i D(Q ‖ P_i) — the distribution CLOSEST to all models, each model treated as a MODEL of an
  unknown source — has the solution eq. (11), the geometric mixture P(x) ∝ Π_i P_i(x)^{w_i/Σw}. §3.3 shows PAQ7's
  squash(Σ w_i st(p_i)) IS this formula for a binary alphabet (with the weights not normalised).
- eq. (25): P := argmin_Q Σ_i w_i D(P_i ‖ Q) — each model treated as a SOURCE and the mixture as the model of a
  "switching source" that picks source i with probability w_i — gives the linear mixture P(x) = Σ_i w'_i P_i(x). The
  Bayesian posterior weights (β-weighting, eq. (24): β_i(k) = β_i(k−1)·P_i(x_k)/P(x_k)) are a special case of the OGD
  update with a weight-dependent step (the remark after eq. (27)). E34's `Mixer` is this β-weighting.

The weight problem: §3.2 proves the code length Σ_k −log geo(x_k; w) is STRICTLY CONVEX in w (Hessian = a
covariance of the log-probability vectors under the mixture, positive definite by Jensen), so it has one global
minimiser; §4.2 proves the same for the linear mixture. Both are then solved online by "iterative gradient descent"
with a fixed step; PAQ7's update (eq. 21) "is an instance of iterative gradient descent, where α_k = α is constant".
Weight vectors are selected by an order-1 context and the match length.

Measured (Table 1, Calgary corpus, 8 models: orders 0–6 plus a match model; binary decomposition of bytes; GEO step
1/16, LIN 1/32; the paper: "We did not notice significant changes in compression, when the step size was sufficiently
small (in the scale of 10^−2)"):

| file | GEO | LIN (OGD) | BETA (Bayesian) |
|---|---|---|---|
| book1 | 2.212 | 2.304 | 2.313 |
| book2 | 1.864 | 1.943 | 1.965 |
| paper1 | 2.274 | 2.327 | 2.343 |
| pic | 0.813 | 0.871 | 0.922 |
| obj1 | 3.672 | 3.603 | 3.610 |
| average (14 files) | 2.187 | 2.231 | 2.265 |

"On average LIN compresses about 2% and BETA compresses about 3.6% worse than GEO, respectively. When we compare LIN
and BETA we see that BETA produces worse compression in every case, 1.5% on average." On the text files the GEO–BETA
gap is 4.6% (book1), 5.4% (book2), 3.0% (paper1), all relative to GEO. This is the cleanest published measurement of "what geometric buys
over linear" with the SAME experts: a few percent — and note the experts here are RAW orders (the mixer does PPM's
blending job), whereas E34 mixed a chain that already blends the orders.

### 3.2 Regret bounds for OGD on "nice" mixtures (DCC 2013)

Abstract (verbatim): "Linear and geometric mixtures are two methods to combine arbitrary models in data compression.
Geometric mixtures generalize the empirically well-performing PAQ7 mixture. Both mixture schemes rely on weight vectors,
which heavily determine their performance. Typically weight vectors are identified via Online Gradient Descent. In this
work we show that one can obtain strong code length bounds for such a weight estimation scheme. These bounds hold for
arbitrary input sequences. For this purpose we introduce the class of nice mixtures and analyze how Online Gradient
Descent with a fixed step size combined with a nice mixture performs. These results translate to linear and geometric
mixtures, which are nice, as we show. The results hold for PAQ7 mixtures as well, thus we provide the first theoretical
analysis of PAQ7."

What is assumed (Definition 3.1, "nice"): the weight domain W is compact and convex; the per-step loss ℓ(x, mix(w,P))
is convex and differentiable in w; and a self-bounding gradient, |∇_w ℓ|² ≤ a·ℓ for a constant a. Nothing is assumed
about the data or the experts beyond p_i(x) ≥ ε = 2^−B. The competitor is the best FIXED weight vector (Prop. 3.3) or the
best PIECEWISE-CONSTANT sequence with s pieces (Thm 3.7). The bounds: with step α = 2(1 − 1/b)/a, code length ≤ b·ℓ* +
(a/4)(b²/(b−1))|w_1 − w*|² for any b > 1; choosing b = 1 + n^−1/2 gives ℓ* + O(s√n) (Obs. 3.5, eq. (4)), so the excess per
symbol vanishes. The step size must be set from a (and from n, or use the doubling trick); "It turns out that the choice
of the step size is of great importance."

Where geometric and linear differ in the bounds (Lemmas 4.1, 4.2, Table 1): the constant a — hence the penalty and the
step size — is a ≥ m·log²(p_max/p_min)/log e for GEO (polynomial in B) but a ≥ m·log²(e)·p²_max /(p²_min·log(1/p_min))
for LIN (exponential in B: the row-1 penalty is 17ms·4^B/(4B) for linear against 7msB²/5 for geometric). In words: in the
stretch domain the loss is well conditioned; in the probability domain the gradient blows up whenever an expert is
confident and wrong. Veness et al.'s Context Tree Switching gives the linear mixture an O(s log n) penalty with a
non-OGD (Bayesian switching) weight scheme, i.e. the linear case has a gradient-free learner with a BETTER bound — but it
is still linear.

Remark 4.7 + Example 4.8 — the theorem behind E34's "0.4 and 0.4 give 0.4": for w on the simplex, max_w lin(x; w, P) =
p_max(x; P) (a linear mixture can never assign a symbol more probability than its most confident expert), while
geo(x; w, P) ≥ p_min(x; P) always (never worse than linear in the worst case per symbol) AND "There exist situations
where max_w geo(x; w, P) > max_w lin(x; w, P)": two experts that each put q on symbol 1 and spread the rest over
DIFFERENT symbols give geo(1) > q. Sharpening beyond the best expert is possible only multiplicatively; the price is
that the exponents must be chosen, and choosing them wrong over-sharpens (E34's product, 7.665).

### 3.3 The counters have the same kind of guarantee (thesis chapters, from the arXiv abstracts)

arXiv:1311.1723: relative frequencies with periodic discount ("Algorithm RFD") have small code length "above an
arbitrary piecewise stationary model"; arXiv:1501.01202: exponential smoothing (the EMA counter of §1.1) has redundancy
"O(s√n)" against a piecewise-stationary model with s segments, "an improvement over redundancy O(s√(n log n)) of previous
approaches". So both halves of PAQ — counting cells and OGD mixer — are covered by the same style of adversarial bound
against a piecewise-constant competitor; the counters need no gradient because their target is the bit.

### 3.4 Gated linear networks (2017): the mixer stack named

arXiv:1712.01897 abstract (verbatim): "This paper describes a family of probabilistic architectures designed for online
learning under the logarithmic loss. Rather than relying on non-linear transfer functions, our method gains
representational power by the use of data conditioning. We state under general conditions a learnable capacity theorem
that shows this approach can in principle learn any bounded Borel-measurable function on a compact subset of euclidean
space; the result is stronger than many universality results for connectionist architectures because we provide both
the model and the learning procedure for which convergence is guaranteed." "Data conditioning" = the weight set selected
by a context (PAQ4's 3-bit context, lpaq1's 80 sets, cmix's 47 mixers × up to 10,000 sets); each neuron is a geometric mixture
trained on its own convex loss, which is why cmix can say there is no backpropagation between layers.

## 4. The measured ladder on enwik8 (LTCB, bytes → bits/char), and a correction to E34's framing

| program | enwik8 bytes | bits/char | what it is |
|---|---|---|---|
| xz 5.2.1 (LZMA2) | 24,703,772 | 1.976 | LZ77 |
| ppmd J1 -o10 | 21,388,296 | 1.711 | PPM, counts + escape (no mixer, no SSE) |
| zpaq 1.03 mid.cfg | 20,941,558 | 1.675 | order-0 indirect model, chain of 5 ISSEs (orders 1–5), order-7 match, one MIX keyed by the last byte |
| lpaq1 9 | 19,755,948 | 1.580 | 7 models, ONE mixer (560 weights), 2 APMs |
| ppmonstr J -o16 | 19,055,092 | 1.524 | PPMII + SEE + SSE — count-based throughout |
| paq8l -6 (2007) | 18,518,485 | 1.481 | hundreds of models, mixers, SSE |
| zpaq 6.42 max6 | 17,855,729 | 1.428 | CM |
| paq8hp12any -8 | 16,230,028 | 1.298 | paq8 tuned for text + dictionary preprocessing |
| durilca'kingsize | 16,209,219 | 1.297 | PPM, 13 GB, benchmark-specific dictionary |
| paq8px_v206fix1 | 15,849,084 | 1.268 | CM |
| phda9 1.8 | 15,010,414 | 1.201 | CM (Hutter Prize lineage) |
| nncp v3.2 | 14,915,298 | 1.193 | Transformer |
| cmix v21 | 14,623,723 | 1.170 | 2,077 models, 3 mixer layers, LSTM, SSE, dictionary |

Correction. E34's log and DESIGN §20 put "PPM-class tables" at ~1.9 and "PAQ/cmix" at ~1.2–1.3 and attributed the step
to the learned multiplicative mixer. The table says otherwise: plain PPM is 1.71, PPM with count-based SEE/SSE
(ppmonstr) is 1.52 — BELOW lpaq1's learned logistic mixer over 7 models (1.58) — and the step from 1.5 to 1.2 is
bought by hundreds of models (word, sparse, indirect, record, bracket …), bit-level modelling, dictionary
preprocessing, more SSE, and in cmix an LSTM, not by the mixer alone. The mixer's own contribution, with the experts held
fixed, is Mattern's 2–4% (§3.1) and the PAQ6→PAQ7 step (Calgary 648,892 → 611,684, −5.7%, confounded with new image
models and a rewrite). Applied to E34: a gradient-learned geometric mixer over the SAME eighteen experts should be
expected to land near 1.75, not near 1.3; 1.3 on Latin would need the library, not the mixer.

## 5. What this settles for ZipLearn, and the gradient-free candidates

**No closed form exists for the combination itself.** DCE §4.3: "Probability theory does not answer the question. It is
possible to create sequences where p can be anything at all for any pa and pb." A combination rule is a hypothesis about
the source — product = the experts are conditionally independent evidence; average = a switching source (Mattern §4);
geometric with exponents = a tempered product. Any fixed rule is wrong on some data; what can be inferred in closed form
is a posterior over a SET of rules. That reframes E34's finding: the linear Bayesian mixer did not fail to learn — it
learned the best member of a family that cannot sharpen (Remark 4.7).

**Candidates that need no gradient (concrete, in E34's apparatus).**

1. *Bayesian mixture over a finite family of geometric mixtures.* Hypotheses h ∈ H, each a fixed exponent vector
   (e.g. τ·w_Bayes for τ on a grid, or {0, ½, 1} on the three strongest experts, or "chain alone"); per mixing context
   keep the prequential weight of each h (a count of bits, exactly E34's `Mixer` machinery with the hypotheses in place
   of the experts); predict with the posterior-weighted average of the geometric mixtures. Regret vs the best fixed h is
   ≤ log2 |H| bits over the whole stream (|H| = 64 → 6 bits on 240k characters); with fixed share it tracks a changing
   best h at O(s·log(|H|·n)). Cost: |H| geometric mixtures per character, |H|×70×18 multiplications; |H| ≤ 64 fits the
   5-minute budget on a 60k-character test slice. This is linear over hypotheses and geometric inside each: the
   sharpening is available, the exponents are selected by counting. The step-size and n-dependence of Mattern's bounds
   disappear (there is no step).
2. *SSE/APM on the output, by counting.* Quantise st(p_mix(x)) of the geometric mixture (or of the raw product) into ~24
   bins per mixing context, learn a calibrated probability per bin by the StateMap rule, interpolate. For the 70-symbol
   alphabet this needs the bitwise decomposition PAQ uses (⌈log2 70⌉ = 7 binary decisions per character, each an SSE
   input) or a per-symbol calibration with renormalisation; the latter is the cheaper first try. PAQ's history prices
   SSE at 2–3% of the code length; ppmonstr's 1.524 shows counting-based refinement can reach where a small mixer does
   not. It corrects a global over/under-sharpening per context — exactly the product's failure mode — but not per-expert
   weights.
3. *2-input SSE trees (Mattern's M1, Martelock's CCM).* A table indexed by (context, quantised st(p_a), quantised
   st(p_b)) learned by counting is a nonparametric combination function for a PAIR — it contains the product, the
   average, and everything else; a tree of such tables combines more experts. Same bitwise-decomposition requirement;
   cells fill slowly (32×32×contexts), so interpolation and a prior of "p_a·p_b tempered" are needed.
4. *Indirect calibration of each expert before mixing.* Map each expert's count STATE (n0, n1, last outcome — the
   bit-history idea) to a probability by counting instead of using the raw frequency; PAQ's indirect models beat direct
   ones by 5% at order 3 (DCE table) with no gradient. This shrinks what the mixer must fix (the experts' confidence
   becomes honest) and is exactly what PPMII's information inheritance and SEE do on the PPM side.
5. *Bayesian switching over the experts themselves* (Volf's switching, Veness's CTS): gradient-free, O(s log n) — but
   linear, so it inherits Remark 4.7; listed to close the door on it, not to open it.
6. Not gradient-free, listed for the record: the exponent vector's maximum-likelihood problem is strictly convex
   (§3.1), so a sleep pass could solve it by Newton's method on stored (stretch vector, symbol) pairs per mixing context
   in a handful of iterations — derivatives, but no online SGD and no chain rule. Whether that counts as "a gradient
   patch" under the 2026-09-22 rule is the user's call; it is the same computation PAQ does online, done once offline.

**What the literature does NOT offer:** any count-based rule that learns per-expert exponents directly. Every program
that multiplies evidence with learned exponents learns them by a gradient of the coding loss (Mahoney 2005 for the
linear case, PAQ7/lpaq/ZPAQ/cmix for the logistic case, Mattern's proofs for both); the gradient-free components are the
cells with observable targets and the selection of which weight set to use.

## 6. Open questions

- Does candidate 1 recover Mattern's 2–4% on E34's stream? The test that isolates the mixer: mix the RAW orders 0–8 plus
  the match expert (no chain) geometrically with counted exponents and compare against the chain (1.823); if the
  gradient-free geometric mixer ≈ chain, the mixer buys PPM's blending and nothing more, and the library is the lever.
- Is the "sharper than the best expert" case (Example 4.8) actually frequent on Latin? Count, per character, how often
  two experts agree on the argmax with different supports elsewhere; that is the only situation where a multiplicative
  combination can beat the best expert.
- The bitwise decomposition: PAQ's SSE and 2-D SSE presuppose binary decisions. Is a 7-decision binarisation of the
  70-symbol alphabet (a fixed tree by frequency) acceptable in E34's apparatus, or is a per-symbol calibration with
  renormalisation enough?
- Mattern's bounds are worst-case against piecewise-constant weights; his experiments used 8 raw-order experts. Whether
  the GEO–LIN gap persists when one expert is already a good blend (E34's chain) is not measured anywhere I found.
- The thesis itself (2016) may contain the non-binary geometric experiments the DCC 2012 paper deferred ("We defer an
  exhaustive experimental study … to future research"); it is unread.
