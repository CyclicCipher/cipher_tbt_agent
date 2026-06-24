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
