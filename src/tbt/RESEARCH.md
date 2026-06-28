# Research notes — thalamo-cortical architecture

Living research log backing `THALAMO_CORTICAL_ARCHITECTURE.md`. Each entry: a load-bearing question,
what the literature says, and the implication for the build. Honest about what's solved vs still open.

---

## R1 — Online / single-shot learning of the task graph (the §9 unknown)

**Question.** TEM learns its structural code offline over many environments; on the ARC replica a task
dependency ("key unlocks door") must be inferred from a **single** playthrough, mid-level. How does
single-shot relational-structure learning work?

**Answer: a fast/slow weight decomposition.** This is the recurring theme across every relevant model:
- **SLOW weights** (SGD/EM, learned across many environments): the structural **prior** — the *space* of
  possible structures / causal schemas, and *how* to bind a new instance. Meta-learned, offline.
- **FAST weights** (Hebbian, one-shot, within-episode): bind **this** environment's specific rules in a
  single shot.

You do **not** learn structure from scratch online. You meta-learn the structure *space* offline and bind
the *instance* one-shot online. (Munkhdalai & Trischler 2018, *Metalearning with Hebbian Fast Weights*:
fast Hebbian weights do one-shot binding; slow SGD weights learn the representations *and the binding
rule*. TEM is the same shape — slow structural `g`, fast Hebbian memory `M`.)

**The slow substrate — schema spaces:**
- **CSCG** (Clone-Structured Cognitive Graph, George et al. 2021, Nat. Commun.): a cloned HMM — multiple
  latent "clones" per observation give context-dependent states; learns a latent graph from action-
  observation sequences (EM/Baum-Welch). Yields **transferable schemas, transitive inference, hierarchical
  abstraction, and vicarious (simulated) planning**. This is the task-space-map substrate the hippocampal–
  PFC model is built on.
- **Schema Networks** (Kansky/George 2017, ICML): an object-relational **generative causal** model of
  dynamics; learns causal rules (entity attrs → effects), reasons backward from goal through causes, and
  **transfers zero-shot**. This is the causal-rule (key→door) substrate.

**The fast trigger — exafference → a causal edge:** the **prediction-error / exafference** (the world
changing in a way self-motion did not predict — door vanished after stepping on the key) is exactly the
signal to bind a **new causal edge one-shot** (cause = recent action/state, effect = the change) via fast
Hebbian weights, into the task column's graph. The task graph then grows **incrementally, one edge per
surprising effect**. This reuses the thalamus's deviance signal and the exafference our agent already
separates.

**So: single-shot structure learning =** fast Hebbian binding of causal edges (triggered by exafference)
**into** a meta-learned schema space (slow). Slow = the schema/causal prior (or, cold, a minimal generic
"actions cause local effects / contact causes change" prior); fast = one-shot binding of this level's
specific rules.

**Honest residual hard parts (NOT solved by these papers):**
1. **Cold start.** A genuinely novel structure with no meta-learned prior leaves only the generic causal
   prior. ARC deliberately supplies novel structures — so how rich the generic prior must be is open.
2. **One-shot causal attribution.** *Which* recent cause produced the effect? (the blicket-detector
   ambiguity). One co-occurrence is weak evidence; needs a causal prior and/or a few confirmations.
3. **Full dependency structure.** Binding one edge is one-shot; inferring *ordering* and *conjunction*
   ("block AND pad AND goal") from few observations is harder than a single edge.

**Implication for the build.** The **task column** = a CSCG/schema-style latent graph + **fast Hebbian
edge insertion on exafference**, over a (slow, meta-learned or generic) schema prior. Slow schema-space
learning across levels/games + fast online instance binding. Ties to: thalamus deviance (the trigger),
exafference (already separated in `agent/column`), reward model (values which rule-sequence yields score).

Sources: [CSCG, George 2021](https://www.nature.com/articles/s41467-021-22559-5) ·
[Schema Networks, Kansky 2017](https://arxiv.org/abs/1706.04317) ·
[Metalearning w/ Hebbian Fast Weights, Munkhdalai & Trischler 2018](https://arxiv.org/abs/1807.05076) ·
TEM (Whittington 2020). See memory `reference_hierarchy_substrate`, `reference_exploration_replay`.

### R1 correction (the number-line reframing) — fast/slow was the WRONG framing for our substrate

The fast/slow, offline-meta requirement above is a property of **gradient/EM-trained models** (TEM, CSCG,
Schema Networks need many passes to move weights), **not of structure-learning itself.** Our column does
**not** learn structure by SGD — it **binds** it (grid + operator + Hebbian conjunction), which is
one-shot. Proof: the column learned the **number line from scratch** (successor transitions) and
generalised to all arithmetic. So:
- **Local cause→effect is 1-shot online** — a key→door edge is the same as a successor edge; observe the
  transition, bind the operator. **No meta-prior; cold-start is NOT the blocker.**
- **The genuinely hard part is LONG-RANGE credit assignment** — tracing a present failure to a distant past
  action. Mechanism = **reverse replay** (Mattar–Daw, `reference_exploration_replay`): replay the trajectory
  backward from the failure to find the action that mattered. It is a **learned** skill (replay is
  prioritised/learned), matching the intuition that one "had to learn how to go back and re-think."

**Revised open problem:** *long-range credit assignment via learned replay*, **not** cold-start
meta-learning. This is more optimistic and more correct.

---

## R2 — The location code: one SR-eigenvector frame for ANY topology (and why the hex grid is a prior)

**Question.** The column placed entities two different ways: a hard-coded universal **hex grid** (`learn_domain`)
and, online, a **metric embedding** discovered from inverse-action pairs (approach B). B placed line / ring /
2-D grid from transitions but **broke on a tree** (non-metric: no consistent axis displacement embeds it —
18/28, 27/60). A fallback (detect the metric collision, switch to distinct codes) was rejected as too
special-cased. So: one general mechanism, or accept a switch?

**Answer (biology decides): the grid IS the SR eigenbasis — compute it, don't hard-code it.** Stachenfeld
2017: place cells = the successor representation; grid cells = its **eigenvectors**. Hexagonal *only because*
open 2-D space has that transition structure — on a 1-D track, a hairpin, or a tree the eigenbasis is not
hexagonal, it is whatever the topology dictates. The grid also encodes **traversability**, not Euclidean
distance, so it re-tunes to the task (rescales — Barry 2007; fragments — Derdikman 2009; warps around
barriers — Carpenter 2015; warps to reward — Boccara/Butler 2019). So the hard-coded hex grid was only ever
the SR of the *default* (open-2-D) environment. Compute the SR frame of the **actual** transition graph and
one mechanism covers line / ring / 2-D / **tree** with no switch. Caveat (honest): the SR-eigenvector reading
(predictive map) competes with the continuous-attractor reading (Burak & Fiete 2009: hex as rigid hardwired
path-integration) — truth ≈ a scaffold + a learned SR; that hybrid is our prior-vs-learned tension. Full
biology log: memory `reference_grid_sr_eigenbasis`.

**Built (this is the implemented resolution).**
- `consolidate` and `learn_domain` now BOTH place via the **SR-eigenvector frame** of the transition graph
  (`_sr_frame`: symmetric-normalized adjacency → `eigh` → drop the trivial mode → near-orthonormal node
  codes). ONE code source — no parallel systems. Per-relation operators (L5) read off any relation.
- Verified: line 11/11, arithmetic 1806/1806, 2-D 164/164, **tree 28/28 & 60/60** (was B's hard limit),
  carry 60/60, factored 200/200, `unified_demo` all 1.000, 80 smoke tests.
- The hard-coded **hex grid is demoted to an innate metric PRIOR** (vector-nav to unvisited goals — Bush
  2015; CRT error-correcting code — Sreenivasan & Fiete 2011), switched off, to add back when a task needs
  those metric superpowers (they are *intrinsically* metric — you cannot vector-navigate a tree).
- **Emergent finding:** structure-specific frames AUTO-SEPARATE different domains (no-remap interference
  vanished, 0.18→1.0); only *identical* structures still need orthogonal slots (two rings: 0.500 → 1.000).

**A's real limitations (worked around / logged, not papered over).**
1. **Codes span only the explored structure** — no free address space for the unexplored. This is what
   forced `factored.py` to give POSITIONS their own discovered frame (a second column) instead of plucking
   grid cells — i.e., place value genuinely *needs* multi-column. (For non-metric structure this is correct:
   there is nothing to extrapolate to.)
2. **Batch `eigh()`** every consolidate. NEXT internal step: an incremental, persistent **TD-learned SR**
   (biological — modules persist and adapt gradually; engineering — dissolves the eigendecomposition cost).
3. **Cross-environment transfer** of a learned frame (TEM's generalization) — the thing that earns the word
   "universal" without a hard-coded lattice — is **not built**. Open.

**Implication.** Multi-column factorization is no longer assumed; it is forced by A's limitation #1. Emergent
column allocation (stage 6, `multicolumn.py`) and, after it, the goal-state control loop (§5/§6) sit on top
of this one frame.

Sources: [Stachenfeld 2017](https://www.nature.com/articles/nn.4650) · Barry 2007 · Derdikman 2009 ·
Carpenter 2015 · Boccara 2019 · Butler 2019 · Bush 2015 · Sreenivasan & Fiete 2011 · Burak & Fiete 2009 ·
TEM (Whittington 2020). Memory: `reference_grid_sr_eigenbasis`.

---

## R3 — Disentanglement: discovering THAT a space factors, from action (the §12.3c open problem)

**Question.** `multicolumn.py` (stage 6) was handed two PRE-SEPARATED streams and only had to allocate them to
columns. Upstream of allocation is the harder thing: given ONE entangled stream, discover that it factors into
independent generative factors at all, and which variation belongs to which — with no labels.

**Why it's not just unimplemented — it's provably impossible from statistics alone.** Locatello et al. 2019
(ICML): unsupervised disentanglement from i.i.d. observations is impossible without an inductive bias — for any
factored representation there's an entangled one with the identical data distribution. So nothing in the static
statistics can pick out the factors. The escape: you must use how states *transform*.

**The definition that says what to look for.** Higgins et al. 2018 ("Towards a Definition of Disentangled
Representations") defines it group-theoretically: a representation is disentangled w.r.t. `G = G1 × G2 × …` if
the space splits `V = V1 ⊕ V2 ⊕ …` with each `Gi` acting on its own `Vi` and trivially on the others. Crucial
corollary (the bridge to Locatello): the group is identifiable only through INTERACTION (action), not static
snapshots. We are a sensorimotor column — we have the action. Connects to [[project_pure_math_tbt_objective]]
(the generalizing circuit = the group's irreps).

**Built (the implemented operationalization).** `tbt/factorize.py` reads the group decomposition off action
ORBITS: each action generates a cyclic subgroup whose orbits partition the states; actions with the same
orbit-partition (an action and its inverse, or two actions of one factor) are the same factor; two factors are
a DIRECT PRODUCT iff their partitions are TRANSVERSE (every pair of classes meets in one state, sizes multiply
to |S|). A factor's coordinate of a state = its class under all the OTHER factors' partitions (moving along the
factor changes that coordinate, leaves the rest fixed — Higgins' condition, made operational). Each factor →
one column over its coordinate-classes; the thalamus binds them into the joint state.

**Result (`precursor/disentangle.py`).** Input = an N×N torus shown as ONE shuffled symbol per cell, 4 actions
unlabelled as to axis (the factorization is invisible in the observation). Discovered: 2 independent factors,
actions correctly split [0,1]|[2,3], direct-product confirmed, joint dynamics predicted COMPOSITIONALLY
(144/144 at N=6, exact at N=45). Capacity payoff (cf. stage-2 wall, updated by R4): with sparse coding (R4) a
single column is now high-capacity, so both models fit at N=45 — but the factored cost is LINEAR (2 columns
of N) vs the holistic's QUADRATIC (an N²-state column → N²×N² consolidation), so the holistic blows up in
compute and degrades by ~N=90 while the factored stays cheap+exact. So factoring is for SCALE + compositional
transfer, not a small-N wall.

**Honest residual hardness (logged).**
1. **Coupled (non-direct-product) factors.** Place-value carry couples digits: increment-units and
   increment-tens commute everywhere EXCEPT at a carry boundary → a *semidirect* product. The transverse test
   will flag "not a clean product"; the sparse non-commuting set IS the carry, to be learned as a cross-factor
   edge (the bottom-up §5 dependency channel). This is the next disentanglement step.
2. **Abelian subtlety.** Translations all commute, so non-commutativity won't separate them — the orbit/
   transversality test does (it uses orbits, not commutators), but the general non-abelian case needs more.
3. **Granularity = model selection** (how many factors, at what level) and **non-factored action spaces**
   (one ARC button touching several factors) remain open — there the statistical/MDL route stacks on top.

Sources: Locatello 2019 (impossibility) · Higgins 2018 (group definition) · Schölkopf 2021 (independent
mechanisms / sparse mechanism shift) · Caselles-Dupré 2019 (symmetry-based, needs interaction) · TEM 2020.

---

## R4 — Column capacity: the orthonormal cap, and the cortical fix (sparse expansion)

**Question (surfaced by disentangle.py).** A single column's content codebook was `qr(randn(feat_dim, n))` —
DENSE ORTHONORMAL codes — which caps HARD at `feat_dim` (you cannot fit > feat_dim orthonormal vectors in
feat_dim dims), and overflowing didn't degrade, it CRASHED (QR silently returns only feat_dim columns). Same
logic caps LOCATIONS at `d_mem` (place codes near-orthonormal). Both LINEAR in the dimension — because we
demanded zero crosstalk.

**How cortex gets capacity: sparse expansion.** The brain does NOT use dense orthonormal codes — it expands
into a higher-D space and makes codes SPARSE: dentate gyrus (several× more granule cells than EC input, ~2-5%
active), cerebellum (Marr-Albus granule cells), fly mushroom body (~50 inputs → 2000 Kenyon cells, ~5% active
= literally LSH, Dasgupta-Stevens-Navlakha 2017). Random k-sparse codes in high-D are near-orthogonal in
EXPONENTIALLY many numbers (~ C(D,k)), not just D — capacity >> dimension, tolerating ~k/D crosstalk cleaned
up by the argmax/attractor (Kanerva's Sparse Distributed Memory; hyperdimensional computing).

**Built.** (1) L4 content codebook → SPARSE high-D codes (k active units, no cap). Result: the number line
now reaches **513** (was 96) — i.e. 513 distinct numbers in a feat_dim=256 codebook, impossible with
orthonormal codes, so content capacity now genuinely EXCEEDS the dimension; arithmetic 5166/5166 at n=256.
(2) The LOCATION codes (places) get the SAME trick: ≤ d_mem stays exact near-orthonormal (dense), and past
d_mem consolidate uses **sparse place codes** — LSH (random projection of the SR frame + winner-take-all),
capacity C(d_mem,k) >> d_mem — plus an attractor **cleanup** (snap to the nearest place code) in `add`/`predict`
so operator composition on sparse codes doesn't accumulate error. Result: the number line now runs to **2000+
at 300/300** (was a hard crash at 514). The cleanup is GATED on the sparse path, so the dense ≤d_mem case is
byte-identical → all 8 precursor demos + unified_demo + 80 smoke tests unaffected.

**Honest residual.** Sparse place codes give exponential DISTINCTNESS (many distinct locations) but trade away
METRIC generalization at scale — past d_mem they are distinct-but-not-finely-metric, so per-edge operators are
exact for OBSERVED transitions (the number line, fully observed) but the codes no longer interpolate. For
metric RANGE beyond a module (predict unvisited metric positions) the cortical answer is MULTI-SCALE / CRT grid
modules (factor position across scales = the demoted hex prior) — itself a factorization, so it folds into the
multi-column line, not a separate capacity hack. Note: the multi-column capacity win is orthogonal and stronger
still — factoring needs N codes+places per factor vs N² (disentangle.py, R3).

Sources: Dasgupta, Stevens & Navlakha 2017 (fly LSH) · Kanerva (Sparse Distributed Memory; hyperdimensional
computing) · Marr-Albus (cerebellum) · Johnson-Lindenstrauss (random projection).

---

## R5 — Both active AND passive learning (the efference copy is the switch)

Active learning (own actions, efference-tagged) is qualitatively better — it intervenes, breaks spurious
correlations (Pearl/Schölkopf), and yields the controllable factors (orbits, R3). But passive learning gives
a capability active cannot: a model of what the world does ON ITS OWN, learned by watching, so the agent can
ANTICIPATE an uncontrolled process. Both fit in ONE column via the reafference/exafference split (von Holst &
Mittelstaedt): a SELF-CAUSED transition trains a controllable ACTION operator; an OBSERVED autonomous one
trains a WORLD operator. These are the SAME machinery — passive learning needs NO new mechanism: `observe(s,
'world', s2)`, and anticipation = composing the world operator forward (the same `add`). Built
(`precursor/passive.py`): watch an autonomous ring, 1-step 20/20, 5-step roll-out 20/20; active+passive in one
column 16/16 each. Passive disentanglement (factor the autonomous dynamics without action labels) = Cartesian-
product graph / spectral tensor-structure decomposition — harder than the active orbit route, logged for later.
Biology: passive statistical learning is real (Saffran) but self-generated action is necessary for development
(Held & Hein 1963 kitten carousel). Sources: von Holst & Mittelstaedt 1950 · Held & Hein 1963 · Hyvärinen
(nonlinear ICA identifiability from temporal structure).

## R6 — Coupled-carry disentanglement (the semidirect case): detection done, extraction next

disentangle.py (R3) handled the clean DIRECT product. Place value is COUPLED: Z_{b²} is NOT Z_b × Z_b
(gcd(b,b)≠1), a non-split group extension whose cocycle IS the carry. Built (`precursor/coupled.py`, DETECTION):
on 2-digit base-b with actions +1 (units, carries) and +b (tens), discover_factors returns a TRIVIAL second
factor (n=1) and is_product=FALSE (vs the torus's TRUE) — the transverse test FLAGS the coupling and localizes
it to +1 (whose orbit is the whole b²-cycle because the carry spirals through tens). +b still gives a clean
units factor (size b); the carry is sparse (b of b²). STILL OPEN (the EXTRACTION): recover the carry from data
as the HOLONOMY of the +1-connection around the units-cycle — align the +b-orbits by the majority of (carry-
free) +1 edges; the alignment fails to close by a tens-shift of 1, and that holonomy = the carry. Then the
factored model = units × tens columns + the carry as a learned sparse cross-factor edge (the §5 bottom-up
dependency), predicting multi-digit transitions compositionally. This is the bridge from disentanglement back
to the learned place value of factored.py/carry.py — discovered, not given.
BUT (R7): do NOT build the holonomy extractor — it is BESPOKE (works for carry, ~nothing else). See R7.

---

## R7 — Minimize the work: the scope of what's not learnable yet, and the ONE general mechanism

A bespoke detector per structure type (orbits for products, holonomy for carry, …) is the maximal-work,
anti-bitter-lesson path. Scope of what the column can't yet learn from experience:
- **A** structured deviations from a clean product — carry (cocycle), context-dependence F_τ(C), exceptions,
  conditionals; **B** hierarchy / recursion / chunking (currently flat); **C** the encoding rule (observation
  → factors, e.g. number ↔ digits — GIVEN in factored.py/carry.py); **D** factorization when actions don't
  isolate factors (granularity, non-factored action spaces).
- **E** perception (raw input → symbols); **F** goal/value; **G** long-range credit (reverse replay).

**Key insight: A–D are ONE problem — structure in the RESIDUAL** (regularity the current clean/first-order
model fails to predict). Carry = residual of the units-successor model; context = residual that varies with
context; hierarchy = residual compressed into chunks; encoding = the factorization that minimises the residual.

**The one general mechanism = RECURSIVE RESIDUAL MODELLING (Merge / MDL):** (1) model transitions with the
column; (2) take the prediction errors (residual); (3) apply the SAME column machinery to the residual; (4)
compose (base + correction) and recurse until the residual is unstructured (MDL stop). Carry falls out with no
holonomy: the base mispredicts at wraps, the residual events share a feature (units=b-1), the correction is
systematic (tens+1) → a feature-conditioned residual correction. Same loop → context, exceptions, hierarchy
(each recursion level = a chunk), and the encoding rule (C). ONE mechanism, not N detectors — and it is the
repo's own lines converging: recursive consolidation (Broca's Merge), discovery-as-residual (self-revising-
systems paper), MDL/epiplexity (Alemi). Capacity feeds it: column saturates → residual grows → overflow is the
trigger to factor/chunk into a new column (the multi-column interface; reference_cortical_capacity).

**Outside the loop (separate general mechanisms, don't force them in):** E perception = the same compression
at the INPUT boundary (symbols that compress the sensory stream); F value = the reward model; G credit =
reverse replay (uses the model). **Honest boundary:** recursive-residual expresses any recursive composition
of the column's PRIMITIVES {SR frame, operators, factorization, binding}; deeply recursive / counting /
unbounded-arity structure (real language) may need a new primitive (the symbolic-AI / Merge-primitive line) —
the one place "more than the column" might be required.

**Build plan:** do NOT build the holonomy extractor. Build the recursive-residual loop ONCE; its first test =
recover the carry + place value from data (kills the bespoke holonomy AND learns the GIVEN encoding rule C).
If it passes, A–D are one mechanism and the frontier is just E/F/G.

**BUILT + VALIDATED (`tbt/residual.py`, `precursor/residual.py`).** A decision-list-by-residual-peeling over a
state's factored coordinates: base = the dominant coordinate-delta; the states it mispredicts = the residual;
the dominant residual delta + the simplest predicate (coordinate ==value, or a conjunction up to the
coordinate count) that selects them and breaks no already-correct state = the next rule, prepended; recurse;
STOP when a residual group shares no coordinate value (no compressing predicate = the MDL stop, not a
per-state lookup). ONE loop, FIVE structurally different problems, zero structure-specific code:
- **2-digit carry 100/100** — discovered `c0==9 & c1==9 → (-9,-9)` (top wrap), `c0==9 → (-9,+1)` (carry), base
  `(+1,0)` (units+1) = place-value carry, NO holonomy, NO place value given;
- **3-digit carry 1000/1000** — nested recursion (carry, double-carry, triple-wrap);
- **context-dependence 29/29** (rule magnitude depends on a context coordinate, F_τ(C));
- **feature exceptions 43/43** (a feature value triggers a different move);
- **random exceptions REFUSED 41/49** (base rule only — incompressible noise is NOT memorised; the MDL stop).
So A–D are confirmed ONE mechanism. **Both edges now CLOSED:** (1) predicates are conjunctions of per-coordinate
literals ==/>=/<= — the RANGES make wraps expressible, so context (32/32) and exceptions (48/48) now work on
RINGS, while carry still uses clean == and random noise is still refused; (2) END-TO-END (`cyclic_coords`):
from RAW shuffled symbols + unlabelled {+1, +b} actions with NO coordinates given, the base (=10) is discovered
from the +b orbit count and the +1 cycle order gives the coordinates, then the residual loop predicts carry
**100/100 (2-digit), 1000/1000 (3-digit)** — place value + the encoding + the carry rule, all from data, one
shot. Frontier now genuinely E (perception) / F (value) / G (credit) → the ARC replica.

---

## R8 — Sequence memory: L6 as a path-integrated state (and the projection-vs-table lesson)

**Question.** The column — and the language probe built on it — predicts the next token from the CURRENT token
only (Markov-1). It has no memory of the past sequence, so "B after A" and "B after X" are indistinguishable.
What does the architecture say about sequence memory, and does adding it help?

**Answer (TBT/HTM): a compressed STRUCTURED recurrent state, not lossless attention.** TBT = path integration:
the column's LOCATION is the integrated history (grid cells path-integrate movement). HTM Temporal Memory
(Hawkins & Ahmad 2016) = context-specific cells via lateral prediction (WHICH cell fires encodes the past).
Both are a compressed recurrent state updated incrementally — and the engineering form is a SELECTIVE linear
state-space recurrence, i.e. **Mamba**, which the repo already has. Attention (lossless, O(n²)) is the
alternative and it bridges to L23 associative memory (modern Hopfield = attention = the episodic/hippocampal
side) — but the cortical answer is the compressed state. So: make **L6 a DYNAMIC path-integrated state** (it
was modelled on grid cells, which path-integrate; we had been using it as a STATIC lookup), updated by a
selective per-channel gate, learned online with a **1-step-truncated** local rule (no BPTT):
  α(x)=σ(G[x])∈(0,1)^d ;  h_t = α⊙h_{t-1} + (1−α)⊙E[x_t] ;  p_t = Op·h_t (navigate, L5) ;  P(next) ∝ softmax E·p_t.
The convex form keeps h in the convex hull of the codes → bounded even when CARRIED across ~1500 steps.

**Built + validated (`precursor/language*.py`, pooled Latin + Middle/Old High German ~480K tokens, next-token PPL).**
- `language.py` (PASSIVE): the SR-frame IS a word embedding (Levy-Goldberg: SGNS factorizes PMI; Stachenfeld:
  grid = SR eigenbasis — same object). Coherent geometry, generalizes next-token on RARE contexts (the
  E3-on-unseen-pairs win). Static factorization, PPL 166.
- `language_active.py` (ACTIVE, Markov-1): predict = a MOTOR ACT (apply the displacement operator). PPL 181 —
  WORSE than passive. Honest finding: on a FIXED corpus the agent can't choose data, so "active" collapses to
  online error-correction, which closed-form SVD does more efficiently; active's real edge is CONTROL over the
  data distribution (embodied: ARC/Minecraft), not corpus reading.
- `language_recurrent.py` (RECURRENT, selective SSM memory): PPL 181 → **152 (best)**, rare-context 155 →
  **136** (beats passive 141). The controlled ±memory comparison (same model family) is unambiguous:
  **sequence memory is the win.**

**THE PROJECTION-vs-TABLE LESSON (record this).** The per-channel gate must be a DIRECTLY-LEARNED per-token
TABLE `G (V×d)`, `α=σ(G[x])`. The Mamba-canonical SHARED PROJECTION `α=σ(Wₐ·E[x])` **under-trains at this
scale**: `Wₐ` init tiny → `Wₐ·E[x]≈0` → α stuck at its init (~0.61) for every token → collapses to a FIXED
(non-selective) decay → REGRESSES (163.6, vs the table's 152.4, vs even a scalar per-token table's 158). Why:
the projection's gradient is diffuse across one shared d×d matrix and needs many passes to develop per-token
differentiation; the per-token table gives each token's gate a strong, directly-accumulated signal (the scalar
table learned α∈[0.01,1.0] cleanly). At a few hundred K tokens the **direct table wins**; the projection is the
right form only at scale (more data + passes). **Logged in case we scale up and the projection begins to pay —
do not assume the Mamba projection is strictly better; here it wasn't.**

**Interpretability bonus.** The learned gate is linguistically meaningful, BILINGUALLY: high-α
"carry/transparent" = particles/copula (et, est, enim); low-α "reset/overwrite" = PREPOSITIONS (ad, ab, ex,
inter, de) + German function words (der, von). It rediscovered "a preposition introduces a fresh phrase" from
raw next-token prediction.

**Honest residuals / scope.** (1) 1-step-truncated learning: the forward state carries history but credit is
1-step local — works here, may limit very-long-range credit. (2) The convex gate biases toward recent tokens.
(3) This is NOT a full LM — exactly the R7 boundary: deeply recursive / unbounded-arity language structure may
need a **Merge PRIMITIVE beyond the column.** What IS settled: recurrence/sequence memory is a real, working
COLUMN capability — the temporal dimension the multi-column neocortex needs (the §5/§6 control loop is itself a
recurrence over subgoals). NEXT at scale: revisit the per-dim PROJECTION gate; carry state across whole books;
B/C-selectivity (input + readout gates); the Merge-primitive boundary for real language.

Sources: Mamba/SSD (Dao & Gu 2024) · HTM Temporal Memory (Hawkins & Ahmad 2016) · TBT path integration
(Hawkins 2019) · Stachenfeld 2017 (grid = SR) · Levy & Goldberg 2014 (SGNS = PMI) · Ramsauer 2020 (modern
Hopfield = attention). Memory: `project_grid_in_mamba_hybrid`.

---

## R9 — How columns COMMUNICATE: the Cortical Messaging Protocol + voting (the gap Cipher flagged)

**Question.** We built the thalamus (bind/read), basal ganglia, emergent allocation, egocentric vs absolute
columns — but never specified HOW columns actually talk to each other. What does the literature (NOT our docs)
say the inter-column protocol is? (Cipher, 2026-06-27: "the one thing I don't think we ever addressed is how
columns communicate; do external research; don't forget LATERAL.")

**Answer: ONE message format (the Cortical Messaging Protocol) + three routing directions; lateral = VOTING.**
From the Thousand Brains Project / Monty (Numenta 2024, arXiv 2412.18354) and Hawkins-Ahmad-Cui 2017:
- **The CMP — one currency for every message** (lateral, feedforward, feedback): `(content, pose, confidence)`
  + sender-id + validity. *Content* = what object/feature; *pose* = its location + 3×3 orientation in a
  reference frame; *confidence* ∈ [0,1]. TWO load-bearing facts: (a) columns exchange **beliefs (object+pose
  hypotheses), NEVER raw input features** — a compressed structured message, not sensory data; (b) the message
  carries the **spatial relationship**, so consensus is by **triangulation, not a bag-of-features** ("it's a
  mug, AND given my pose you should be sensing it over there").
- **Feedforward (bottom-up):** a lower column's recognized object ID becomes a **FEATURE in the higher
  column's model** → compositionality. (= our space→task channel; how key→door enters the task column.)
- **Feedback (top-down):** the higher column **biases** the lower (sets the goal-state). (= our task→space.)
- **Lateral (VOTING):** peers exchange the **UNION of all their (object, pose) hypotheses** + confidence;
  consensus by **evidence accumulation + spatial consistency**, terminal when one hypothesis survives. Resolves
  ambiguity fast (multi-view, multimodal, aliasing). (= the piece we downplayed; the egocentric×absolute bind.)

**Implication for the build (the reconnect spine).** Our thalamus binds `content ⊗ place` but the messages are
BARE — no pose/confidence, no hypothesis-union, no voting. Upgrade: every inter-column message is **CMP-shaped
`(content, pose, confidence)`**, carried through the thalamus; add **lateral voting** (exchange hypothesis sets,
consensus by spatial consistency) as a first-class channel beside top-down/bottom-up; the BG gates which routes
open; `reward.py` values; the recurrence carries context. This is the missing PROTOCOL on top of the thalamus
SUBSTRATE.

**The reconnect plan (Option A, the multi-column control loop; chosen by Cipher 2026-06-27).** The current flat
`value.py` BFS is the architecture's explicitly-rejected exhaustive VI (§6 "not blind one-step replanning";
reward.py "prioritized sweeping"). Stages, each gated on the levels that pass now (SK0-2, LP0-1):
1. **The communication spine + the control loop** — CMP messages over the thalamus (top-down goal-state +
   bottom-up achieved/exafference + lateral voting), task ⊕ space columns, navigate via the SR (reward.py
   prioritized replay), retire the flat planner. **← STARTED HERE.**
2. **Sub-goals EMERGE** — eigenoptions (off `_sr_frame`) + affordances (reach a discovered effect's
   precondition, from `residual.py`); doors become sub-goals, so LP3's bits FACTOR (no 2^bits). Validate LP2/LP3.
3. **Relational push via the egocentric column** — bind the egocentric G (`forward.py`) to the absolute map
   column by lateral voting; the push is the egocentric dynamics rolled forward + valued, not BFS-over-joint.
4. **Emergent allocation + incremental SR** — BG allocates columns (`multicolumn.py`); TD-SR for scale.

Sources: [Thousand Brains Project / Monty (Numenta 2024)](https://arxiv.org/html/2412.18354v1) ·
Hawkins, Ahmad & Cui 2017 (voting) · [TBT voting overview (Numenta)](https://www.numenta.com/blog/2019/01/16/the-thousand-brains-theory-of-intelligence/) ·
cortico-cortical laminar connectivity (feedforward L2/3→L4, feedback L5/6→L1, lateral uniform). Memory: see
`reference_hierarchy_substrate`, `reference_exploration_replay`; architecture doc §3/§5/§6.


## R10 — The TBT-faithful refactor: the column IS the forward model, value is SIGNED, location is continuous

**The flag (Cipher, 2026-06-27, after A③+S2 shipped).** The agent works (Sokoban 4/4, LockPath 4/4, MultiKey 2/2,
all autonomous — the world model learned from pixels+score), but it drifted back to the SYMBOLIC HACKS the bitter
lesson forbids: a `WorldModel` role SCHEMA (pushable / door / consumable / harmful), a SEPARATE `DynamicsModel`
the planner reads, TYPED sub-goals (cover / reach / affordance), a discrete CELL graph, and a POSITIVE-ONLY value
field. Three questions: (1) why does the planner know "cell / role / consumable / harmful" — shouldn't a faithful
column LEARN these? (2) why a separate `DynamicsModel` — shouldn't the COLUMN learn AND predict dynamics? (3)
shouldn't reward go NEGATIVE? Constraints: location must be GENERAL + continuous-safe (not tied to "cells", must
not break on continuous coords); there must be FEWER scripts at the end than the start.

**Answer: yes to all three — they are one drift. Faithful TBT = a column that PREDICTS + a SIGNED value that
EVALUATES; the roles dissolve into prediction + value.**

- **1. The unit (Monty / Thousand-Brains Systems, arXiv 2507.04494, 2025).** ONE repeating unit — the **Learning
  Module** — that MODELS an object, PREDICTS, and ACTS via a **Goal-State Generator**. The unit that learns IS the
  unit that plans: no separate planner, no separate dynamics module. (Sensor Modules turn raw input into the common
  language; the LM models it.) → kills the standalone `DynamicsModel` / `forward.py`; the column must learn + predict
  dynamics itself (Q2).
- **2. Location (same paper).** CONTINUOUS + pose-based: a feature at a pose `x ∈ ℝ³`, path-integrated by
  displacement `x_t = x_{t-1} + v_t` ("dead reckoning"), movement transformed into the object's frame by the
  hypothesised rotation (`v_M = R⁻¹ v_B`). NO grid of discrete cells. → our discrete SR-symbol / cell graph is the
  un-general part; the faithful location is the displacement/recurrence machinery on a CONTINUOUS pose code
  (Q1-location; continuous-safe by construction).
- **3. Composition (same).** The **CMP** carries `(features, pose, confidence)` for feedforward / lateral voting /
  **goal-states** — goal-directed action = emit a desired pose, move to close it (matches R9; the thalamus is the
  substrate). NB Monty has **NO value and NO object-behaviour dynamics yet** — those are OUR additions, not a copy.
- **4. Value — SIGNED; planning = active inference (Bogacz, eLife 53262, 2020).** Dopamine = a SIGNED reward-
  prediction error `δ = actual − expected`, NEGATIVE for worse-than-expected / aversive (negative δ flips the weight
  update = reversal learning). **Planning = within-trial action that MINIMISES the gap between the DESIRED state and
  the EXPECTED state** (`da/dt ∝ ∂F/∂a`, `F = −δ²`; terminates when δ→0). Preferences = DESIRED STATES, not absolute
  reward; cortex supplies the forward model, basal ganglia + dopamine supply value. → our `reward.py` is positive-
  only (R_ext=1, novelty+, no negatives), which is exactly WHY we LABEL death/harmful and do graph-surgery instead
  of letting value avoid them. Sign the value and those roles DISSOLVE: "harmful" = a predicted state with negative
  value; "collect-all" = the desired state is all-items-gone; "door" = a predicted state-change the value wants.
  The planner just closes the gap (Q1-roles + Q3).

**The build (Phase 2 — supersedes the role-threading "finish the suite" patches; full plan in
[[project_reorient_and_reconnect]]).** TARGET: `column.py` = the LM (continuous-pose frame via the recurrence +
an IN-COLUMN forward model + a goal-state generator); `value.py` = SIGNED value; `neocortex.py` = a THIN active-
inference loop (roll the forward models, score predicted states by signed value, emit goal-states — NO roles, NO
typed sub-goals); a THIN sensor module (no role schema). STEPS, each a general method + validated on the suite +
NET-deleting files: **A** the column becomes the forward model (fold `dynamics.py` + `forward.py` in; location →
pose/continuous) [−2]; **B** signed value + active-inference planning (roll the column model + signed value + goal-
states; replaces the cell-graph nav + typed sub-goals + openers + role-avoidance — CollectAll + Toggle FALL OUT)
DELETE `planner.py`, `control.py`, neocortex → thin [−2]; **C** dissolve the role schema (perception → thin sensor
module; the LM learns dynamics + the value field learns the goal) DELETE `learn.py`, shrink perceive/scene [−1].
NET ~24 → ~19 agent files. Step B is the crux = MuZero / EZ-V2 search over the LEARNED model (see `EZV2_NOTES`).

Sources: [Thousand-Brains Systems (arXiv 2507.04494, 2025)](https://arxiv.org/abs/2507.04494) ·
[The Thousand Brains Project (arXiv 2412.18354)](https://arxiv.org/abs/2412.18354) ·
[Dopamine role in learning and action inference — Bogacz, eLife 2020](https://elifesciences.org/articles/53262) ·
[Locations in the Neocortex: cortical grid cells (Frontiers 2019)](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2019.00022/full) ·
[Thousand Brains principles](https://thousandbrains.org/learn/thousand-brains-principles/).
Memory: [[reference_efficientzero_v2]], [[feedback_bitter_lesson]], [[project_general_world_model]].


## R11 — Objects as column-models recognized by voting (Step C: dissolve the role schema)

**The flag (Cipher, 2026-06-27, after Step B shipped).** Two questions converged: (1) what classes of problem does Step B's signed-value rollout cover, and where does it break? (2) how does a column represent an OBJECT, and does the integer-label codebook scale? Both land on the same gap, and Step C ("dissolve the role schema; objects become column-models; the LM learns dynamics + value learns the goal") is the answer. This entry is the deep HTM/TBT research that grounds it.

**What Step B's flip taught us (the rollout-state ceiling, NOT a column ceiling).** The Toggle flip (XOR a door-bit) is an *involution*; it also covers levers, mode-switches, and **parity locks** (open iff an even number of presses). But the limit it exposes is the *forward-model glue*, not the column: `control.py` threads only a boolean door-bitset + a few positions through the rollout, so it cannot express **mod-N counters (N>2)**, **conjunctive/sequential locks** (the effect lookup is keyed on the stepped-on cell and *ignores the rolled state*, even though `DynamicsPerceiver` already emits a 16-bit presence-context and the residual faculty can LEARN context-dependent rules — R7), **continuous world-state**, **object spawn/teleport**, **autonomous/timed dynamics** (the rollout freezes everything but agent+focus+doors), and **joint multi-object constraints** (we factor one focus mover). The column has already shown counters (the number line), context, and carry (R7) — they are simply not plumbed into the rollout. So Step C = plumb the LEARNED model in.

**How a column represents an OBJECT (we verified our code + the TBT sources).**
- *Ours now (`l4_feature_location.py`):* the L4 codebook `E` is `(n_entities, feat_dim=256)` of **k=12-sparse unit vectors**; the integer label is only a row index, the representation is the sparse code, bound to a LOCATION (SR-frame place code) via an outer product into the shared L23 memory. Capacity is **C(256,12) ≈ 10¹⁸**, not 256 (R4) — so raw label count is NOT the bottleneck. The gaps: `n_entities` is fixed at construction (no online recruitment); objects are **atomic** entries; and in OUR usage the column's "entities" are *cells of the spatial map*, not game objects — game objects live in perception's role schema + `segment`'s connected components, NOT as column object-models.
- *Monty (TBT, arXiv 2412.18354 / 2507.04494):* an object model is a **graph** — nodes = (3-D object-centric position, orientation, pose-independent features), edges = **displacement vectors** (the action to move between them) — built incrementally **from the displacement trajectory of sensations, not static snapshots**. This is *exactly our column's shape*: L4 content + L6 SR-frame location + L5 displacement operators + L23 memory IS "features-at-poses with displacement edges." The difference is only what we point it at (the map vs. each object's morphology). In ARC an object's morphology is cheap: `Obj.shape` is already the translation-invariant cell-set.

**Recognition = evidence accumulation (Monty's Evidence-Based LM).** Hypotheses = (object, location, rotation); each step: rotate the body-relative displacement into the object frame to PREDICT the next location, then compare sensed morphology/features to the stored node there — evidence in [−1,1] for morphology (angle between normals/curvatures), [0,1] for features (features only add). Recognized when one hypothesis dominates (max − threshold·max). Ours has the pieces (the recurrence's loc_move/loc_sense path-integration + `anchor`'s Bayes location correction) but NOT the multi-object multi-hypothesis accumulation.

**Voting (the R9 gap, now concrete).** Lateral votes exchange **CMP hypothesis-sets** `{object_id, pose_location, pose_orientation, evidence, sender_id}` — beliefs, never raw features. Consensus = **intersect the hypothesis spaces + check geometric consistency** (LM-A "handle at P1" and LM-B "body at P2" must be compatible given the sensors' separation) + accumulate cross-module evidence. It is **constraint satisfaction across reference frames**. KEY HONEST POINT: voting's *purpose is disambiguation* (multi-view, partial, aliasing). Our **full-observation** replica has no sensory ambiguity, so full hypothesis-voting buys little there — its one live use is the **egocentric⊗absolute relational vote** (the focus-mover's push, voted against the absolute map), which Step B's `_forward_model` already does implicitly. Full voting is forward-looking infrastructure for **partial observation / real ARC** (multi-cell objects, occlusion, the click action) — not a replica win.

**Object behaviors/dynamics (docs.thousandbrains.org/docs/object-behaviors).** Behaviors use the SAME mechanism as morphology but in a SEPARATE, **state-conditioned** reference frame (state = position in a temporal sequence; a global interval timer broadcasts elapsed time on L1); "behavior models are sequences of changes at locations," recognized independently of the object. This validates our **separate dynamics faculty** (`predict_effect` = a state-conditioned change) distinct from morphology, and says the faithful upgrade is to make it **state-conditioned over the rolled state** (closing the conjunctive/context gap above) + temporal.

**Goal-states.** A goal-state is CMP `{desired_pose, feature_id, confidence}`, generated by (A) **uncertainty reduction** (query the forward model for the displacement that most reduces hypothesis entropy) or (B) **goal-conditioned** (inverse model: the displacement to reach location L). Hierarchical: a higher LM's abstract goal-state is decomposed by a lower LM into motor commands. We already emit goal-states (the thalamus routes them) and CLOSE the gap by signed-value rollout (active inference) — Monty has no value; that is our addition (R10).

**The Step C build (full — objects as column-models + dissolve the schema), prototype-first + test-gated (the Step B discipline), keep the agent working:**
1. **Learned dynamics into the rollout (C1).** Replace `control.py`'s decoded `opener/closer/flip` with the column's **state-conditioned `predict_effect`**, so the toggle EMERGES from the learned context-dependent rule (the presence-context already exists) and the hand-coded flip dissolves; closes the conjunctive/context-lock gap. Validated by the 5 games (esp. Toggle).
2. **Objects as per-object column-models.** Each segmented `Obj` → a column: morphology = (colour, shape) in its own frame, pose = absolute position; its ROLE (mover/blocker/goal/trigger) EMERGES from the learned behavior (does it translate when pushed? block? change another object on contact?), replacing the perception role schema. Recognition by evidence accumulation; the codebook holds object IDs (sparse, R4).
3. **Egocentric⊗absolute voting**, formalized as a first-class channel on the thalamus (R9), used for the relational push now and for partial-obs/real-ARC disambiguation later.
4. **Dissolve the role schema.** The forward model reads object-models + learned dynamics + signed value directly; DELETE `learn.py`, shrink `perceive.py` (drop the role evidence) + `scene.py` (drop the `WorldModel` role schema → a thin sensor: frame → (feature, pose) observations). `WorldModel.harmful` is already unused (Step B).
Honest scope: steps 1, 2, 4 are replica-validatable; step 3's full hypothesis-voting is forward-looking (no ambiguity in full-obs) — build the relational-vote now, the disambiguation use when partial-obs/real-ARC needs it.

**BUILT (2026-06-28) — `tbt/recognize.py`, the pose-invariant recogniser (Tetris bench).** The crux Cipher named — *recognise a known object rotated into an UNFAMILIAR pose, in continuous space* — is precisely the limitation the 2019 grid-cell model states outright ("recognizes only at the learned orientation", PMC6491744). The Tetris `shape→shape` rotation table was brute-forcing around it (19 shapes, not 7 objects+pose) AND had a learning bug (spawn/respawn transitions polluted it, 5/18 entries wrong, e.g. I→T — found by a fidelity diagnostic vs the real game). The fix is Monty's Evidence-Based LM, built and validated:
- A hypothesis is `(object, pose=(theta, t))` with continuous evidence. **Pose is SOLVED, not recalled**: align the sensed local-patch displacements onto a model node's (continuous `theta` from `arctan2` differences; symmetric patches yield several candidates — Monty's "two rotation hypotheses"). Movement accumulates evidence: rotate the body displacement by `R(-theta)`, predict the next object-frame node, compare morphology (+1 match / −1 mismatch).
- The rotation is **one universal OPERATOR** `cells_at = R(theta)·model + t` (continuous; the grid's 90° is the discrete special case via the CW orbit), **correct by construction** — the table's wrong-entry bug class cannot exist. Continuous-space grounding: 2-D rotation is a steerable/irrep phase-rotation, and for a grid the L6 SR-frame already IS the translation irreps ([[reference_tbt_pose_invariant_recognition]], [[reference_grid_sr_eigenbasis]]).
- **Learning is online + label-free** (`add_if_novel`): a shape recognised as a rotation of a known object IS that object; else it is new — the object set is *discovered by watching*, never injected ([[feedback_bitter_lesson]]). Watching play learns exactly the **7 one-sided tetrominoes** (chiral S/Z, J/L kept distinct — recognition uses rotations only, not reflections).
- **Multi-column lateral VOTING** (`vote`): pool columns' `(object, pose)` hypotheses by their *shared world pose* and sum evidence — an object's world `(theta, t)` is the same for every column sensing it, so agreement IS consensus. Resolves single-glance ambiguity: **1 column 1 glance 64% → 2 columns voting 100%**.
- Results: continuous-pose recognition **100%** id+pose over unseen random angles; partial-obs **70%→100%** by 2 fixations (evidence accumulation = the sensorimotor claim); irregular continuous (non-grid) objects 100% (not grid-tied). **Integrated**: the SAME `tbt.agent.Agent` plays Tetris L0/L1 via recognition + the operator; `TetrisLearner`/`shape_of`/the table are DELETED; suite 66/66. This is steps 2+3 done the Monty way — a dedicated continuous Evidence-Based recogniser (matching Monty's actual implementation, which uses continuous poses + displacement edges, not SR-frame phase codes), rather than per-object columns on the spatial map. Open: wire recognition into real-ARC perception (the click action, occlusion — where voting earns its keep); the value/L2 multi-piece-clear thread now runs on a *faithful* model (no irreducible model error to fight — EZ-V2 robustness reserved for real-ARC's genuinely-imperfect online models).

Sources: [The Thousand Brains Project (arXiv 2412.18354)](https://arxiv.org/html/2412.18354v1) ·
[Thousand-Brains Systems (arXiv 2507.04494)](https://arxiv.org/pdf/2507.04494) ·
[Monty object behaviors](https://docs.thousandbrains.org/docs/object-behaviors) ·
[TBT voting overview (Numenta)](https://www.numenta.com/blog/2019/01/16/the-thousand-brains-theory-of-intelligence/).
Memory: [[project_reorient_and_reconnect]], [[reference_cortical_capacity]], [[reference_hierarchy_substrate]], [[feedback_bitter_lesson]], [[feedback_reuse_canonical_components]].
