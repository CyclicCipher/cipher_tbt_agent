# MATH_PHASE — is hypothesis learning GUIDED BY DOMAIN STRUCTURE? (an experiment, not a plan)

*2026-07-01. This is a RESEARCH phase; a concrete set of implementation steps is deliberately NOT the goal. We use EXACT
calculation (strictly base-10) as a controlled microworld to probe the hardest open question surfaced by the GSG work —
hypothesis GENERATION, not testing (it is easy to test a hypothesis, hard to generate one). Math is chosen because we
know BOTH the target hypotheses AND the domain structure, with no perception/mechanic confound. NB the ANS (analog
magnitude) is NOT the object of study here — EXACT calculation is.*

## The central question
**If the meaning of an operation (e.g. `+`) is unknown to the model, can it FIND the answer LATENT IN THE STRUCTURE of
the domain — when the domain is engineered so — by a mechanism that GENERALISES to learning Sokoban?** I.e. is hypothesis
generation actually *structure-reading*: given the right representation, the hypothesis is a short, discoverable step
rather than a free invention. Sokoban is the target (the push-to-goal rule is latent in the learned dynamics); math is
the clean rehearsal where we can see exactly what is discovered vs. supplied.

## The thesis under test (why structure could do the generative work)
Kemp/Perfors/Tenenbaum: inductive learning is impossible without **OVERHYPOTHESES** — constraints on the space of
first-order hypotheses. Their result (the *blessing of abstraction*): acquiring the right hypothesis SPACE accelerates
learning **more than** improved selection within a fixed one, the abstraction is learned from the *same* data, and it
**TRANSFERS** to new domains. Restated for us: the domain's STRUCTURE supplies the overhypothesis that makes "the meaning
of `+`" a geodesic in the right representation. If true, **"generate a hypothesis" reduces to "represent the domain well,
then read off the short (MDL) description"** — and the transfer property is exactly the Sokoban generalisation we want.
This is the concrete form of [[reference_hypothesis_generation]]'s SELECTION-vs-CONSTRUCTION split: structure may turn
CONSTRUCTION back into (cheap) selection.

## What "find `+` in the structure" means concretely
- Succession is a TRANSLATION operator (a chain). Addition is its COMPOSITION: `a+b` = apply-successor `b` times. So the
  meaning of `+` is LATENT in the succession structure — `+b` *is* the b-fold successor. If the model has learned
  succession as an operator, discovering `+` = discovering that the answer to `a+b` is REACHABLE from `a` by `b`
  succession steps. The hypothesis is not invented; it is a PATH in an already-learned structure.
- The ladder is composition-on-composition: multiplication = repeated addition; distributivity (`13×7 = 10×7 + 3×7`) —
  which learners DISCOVER for themselves when allowed their own methods (enactments of structure). Each rung's
  overhypothesis is the operator learned at the previous rung (the blessing of abstraction, made a curriculum).
- **CARRY / place value is the honest hard case.** It is the recursive/compositional rung — see THE MASTER BOUNDARY below,
  which turns "where does read-off stop" from an empirical guess into an algebraic prediction.

## THE MASTER BOUNDARY (Fable, 2026-07-01) — free/abelian = READ-OFF vs quotient/relational = CONSTRUCT/SEARCH
The A/B/C split is not empirical luck; it is PREDICTED by the algebra of the domain's transition monoid.
- **A hypothesis is READABLE OFF structure iff it is a homomorphism out of a FREELY-presented object — determined by its
  action on GENERATORS** (the universal property of the free monoid the doc already names: `+` = the free monoid on
  succession). Know where the generators go (the learned operator) and the extension to all `+b` is UNIQUE and FORCED — no
  search; that IS "read off the geodesic." Free/abelian → outcome A (succession; addition-as-composition; geodesics = coord
  differences).
- **The moment the target needs a QUOTIENT — a relation NOT entailed by the generators — read-off fails:** the relation is
  extra data to search for or discover. Base-10 place value is exactly a quotient ("ten units ≡ one ten" is a CONGRUENCE
  imposed on the free count, not entailed by succession) → outcome B (construct) or C (procedural). So "A for succession/plus,
  B/C for carry" is an ALGEBRAIC prediction the experiment CONFIRMS, not luck.
- **The regress bottoms out at the FREE OBJECT (answers "turtles all the way down" for learning the frame):** a free
  structure needs no overhypothesis above it — it is determined by its generators alone, and the generators are the primitive
  actions the agent is GIVEN by embodiment. The world hands you the moves; the free monoid on them needs nothing further. The
  action set is the one non-circular anchor (given, not learned).
- **The substrate has an ABELIAN CEILING (pushback to "operation-space costs an extra grid").** Free/abelian part:
  operation-space ≅ state-space (a translation ≅ the point it reaches) → the EXISTING state grid IS the operation grid, ONE
  grid. Non-abelian part: `l6_grid` (plane waves = the translation group's 1-D characters) CANNOT represent it — that needs
  non-abelian harmonic analysis (matrix-valued irreps), which we lack. So the hard part is not under-budgeted, it is
  UNREPRESENTABLE in the current frame; the representational burden and the difficulty are the SAME boundary.

## CARRY IS DEEP — a correction to "carry is free as phase-rollover"
That claim (in THE REFERENCE FRAME below) holds ONLY in the SERIAL regime: path-integrating `b` unit steps rolls the tens for
free but pays `O(b)` sequential rotations (= unary addition). As a PARALLEL operation on digit-strings (what generalizes, what
humans do), carry PROPAGATES — `999+1` ripples across every digit. Carry-propagation has NON-CONSTANT circuit depth (binary
addition ∉ AC⁰; hardware needs carry-lookahead), so NO depth-bounded read-off produces it over unbounded digits. Carry lives
in ITERATION-TO-A-FIXED-POINT depth (a DEQ), not in the map read at equilibrium.
- **What our own (shelved) DEQ line already found** ([[project_recurrent_world_model]]): BOUNDED depth is learnable —
  fixed-unroll BPTT of a shallow local operator BEAT the matched baseline on OOD ("recurrent depth earns its place").
  ADAPTIVE/equilibrium depth is NOT — a single weight-tied operator can't separate CONTRACTION from EXPRESSIVITY (the band
  contractive enough for a fixed point is too contractive to compute; needs a two-module split). ⇒ exact carry generalises
  across the TRAINED chain-length band and FAILS at unbounded ripple = our recurring OOD length-extrapolation PLATEAU (a
  REPRESENTABILITY wall, not an optimizer one). Carry is bounded-depth-learnable, unbounded-depth-walled = outcome C with
  teeth. NB that line already had the composition-fidelity probe `M(a∘b)=M(a)·M(b)` (= the free-monoid test) and the
  scratchpad-buys-depth / TC⁰ thread — Fable's critique CONVERGES with results we hold.

## The engineering knob (the crux, and the bitter-lesson line)
"Engineered so" = we control HOW MUCH STRUCTURE the representation exposes, then watch whether the model can FIND the
operation. **The line we must not cross:** provide the STRUCTURE (the succession geometry, the domain's transitions — the
world provides structure too), NEVER the ANSWER (the meaning of `+`, the carry rule). The experiment is therefore a
LADDER OF DECREASING SCAFFOLDING, and the RESULT is the shape of that curve:
- most scaffolded — succession given as a translation OPERATOR → is `+` findable as its composition?
- less — succession given only as examples → must the operator be discovered first?
- least — raw tokens → must the whole structure be discovered?
The question behind the curve: **how much structure must be given before the hypothesis becomes findable, and is that
amount domain-GENERAL (the same knob helps Sokoban) or bespoke (we cheated)?**

## What we already know — baseline probes (2026-07-01; see [[project_math_hypothesis_probe]])
- **Naive atomic succession** (digit tuples treated as atoms): the SR learns a FLAT LINE (`cos(code(n),code(n+10))=0.058`),
  perfect IN-range memorisation (carries included), and ZERO extrapolation (an unseen `n` → returned UNCHANGED). The
  representation exposes NO reusable structure — nothing to read a hypothesis off.
- **Positive control — the units `Z/10` RING:** the SR CLOSES the ring (wrap similarity = adjacent). The SR *can* hold
  periodic structure; the number line didn't, because it was a finite OPEN chain of ATOMS learned per-edge — no
  translation-invariant operator, no path integration.
- ⇒ the missing prerequisite is a **translation-invariant successor OPERATOR + PATH INTEGRATION** over multi-scale
  periodic modules (a grid / residue code — Fiete; incommensurate periods = huge unambiguous range + extrapolation),
  reusing our SHELVED `l6_grid` (`scales=(11,13,17)`) — the SAME machinery P1 built for space. Only then does a structure
  exist to read a hypothesis off. This is TBT-native (grid/path-integration), NOT the pruned symbolic `factorize.py`/
  `residual.py` (orbit-partitions + predicate decision-lists — the abandoned CTKG flavour; a warning, not a tool).

## THE REFERENCE FRAME — a translation-invariant operator × an (adaptive) multi-scale grid
The frame we are after: a number is a POSITION reached by path-integrating ONE operator over a multi-scale periodic code.
- **Translation-invariance (`L5.move_delta`, ALREADY LIVE):** `+1` = one displacement applied everywhere → it GENERALISES
  (predict any successor) and, crucially, COMPOSES: `a+b` = apply `+1` `b` times = **path-integrate `a` by `b`**. Addition
  is not invented; it is the operator's composition, read off the frame. (Baseline P0 missed this — it used per-state edges
  over opaque atoms, never the `move_delta`/`track` path.)
- **Multi-scale (`l6_grid`, COMPLETE but SHELVED):** magnitude that EXTRAPOLATES (residue/CRT over incommensurate scales);
  and if the scales are NESTED powers of ten, each module's phase IS a digit (place value) and — **only in the SERIAL regime,
  see CARRY IS DEEP above** — advancing position by 1 rotates the λ=100 phase by one tenth of its cycle, so ten unit-steps
  roll the tens over. **CAVEAT (do not overclaim):** that rollover is `O(b)` serial (unary); PARALLEL multi-digit carry
  PROPAGATES and is depth-unbounded, so it is NOT free — succession/addition are path integration (free/abelian), carry is
  the DEEP/quotient part.
- **Continuous / fractional displacement + the "+1" GAUGE:** the substrate is continuous — `move_delta` is a float, and
  `l6_grid.path_integrate` rotates phases by ANY real displacement — so a fraction is a position BETWEEN integer landmarks
  (a finer codebook). The DISCRETE unit is a gauge fixed by LOOP CLOSURE: the multi-scale error-correcting `decode`/`place`
  snaps to the nearest codeword, and the incommensurate scales CORRECT path-integration drift — so multi-scale is what keeps
  `+1` CONSISTENT over an unbounded count (stability, not just range). Live gap: the grid is shelved (raw `_fovea` is used),
  and the successor needs canonical-unit re-anchoring at landmarks (a pure EWMA drifts if step sizes vary).

## ADAPTIVE SCALES — how the frame is LEARNED (the real lever, currently fixed `(11,13,17)`)
Two principled routes, both reusing existing machinery:
1. **Scales = the dominant eigen-FREQUENCIES of the learned SR / transition operator** (grid = SR eigenvectors, Stachenfeld;
   we ALREADY compute this eigenframe as the eigenpurpose). A cyclic domain of period `p` peaks the spectrum at `1/p` →
   adopt scale `p`; an OPEN counting line is BROADBAND → no discrete scale emerges (self-consistent with the P0 line).
2. **An over-complete scale BANK + learned sparse SELECTION** (the biology: universal scales + attention) — don't relearn
   frequencies, learn which are load-bearing per domain.
Criterion for both = **MDL / prediction-error reduction** (a scale earns its place by compressing) — the same principle as
the construction engine (outcome B). **The catch, precise:** adaptive scales discover INTRINSIC periodicity (cycles, wraps,
modular dynamics). Base-10 place value is NOTATIONAL, not an intrinsic period of counting → it will NOT fall out of a raw
count; it must be present in the input (numerals) or the task made modular (clock arithmetic). That IS the scaffolding knob.

## FACTORED LOOP CLOSURE — how the cyclic frame EMERGES from naive input (and why it is a GENERAL map upgrade)
The mechanism *behind* adaptive scales — the TRIGGER that discovers the periods — and the answer to "why did naive base-10
give a LINE when the model can clearly hold a cycle?"
- **Diagnosis:** what folds a dead-reckoned LINE into a CYCLE is **loop closure** — recognising a state you have RETURNED
  to. The `Z/10` control closed because the state RECURRED; counting gave a line because every number is a UNIQUE state, so
  nothing recurs at the full-state grain, so nothing closes. The missing ingredient was a loop-closure TRIGGER, not a
  cyclic representation.
- **The fix: loop-close on FACTORS, not whole states.** Base-10 cyclicity lives in a PROJECTION — the units digit recurs
  every 10 while the number never does. Recognise the recurring units-content → close a loop in the UNITS SUBSPACE → a
  10-cycle; tens recurs every 100 → a 100-cycle; nested → place value. The PERIODS are DISCOVERED as the recurrence
  intervals (not imposed) and become the grid scales; carry = the operator coupling (units-wrap → tens+1); extrapolation is
  free (cycles are defined at any magnitude).
- **Why it MUST be factored:** naive WHOLE-state loop closure would wrongly MERGE 0 and 10 (same units). Closing only in the
  units subspace while the tens keeps them distinct = a PRODUCT of cycles, not a collapse. Disentanglement and per-factor
  loop closure are two sides of one coin: factor the state SO THAT loop closure can fire; FIND the factors as the
  projections that recur periodically. **The criterion (Fable, stronger than "SR-eigenspectrum"):** close the COARSEST
  partition that stays a PREDICTIVE SUFFICIENT STATISTIC for the future — the wrong-merge (0≡10) is detectable because it
  DESTROYS predictive sufficiency (you then mispredict the tens-dependent future). That is the causal-state / ε-machine
  (Crutchfield) + bisimulation criterion: **MDL SUBJECT TO predictive sufficiency** gives the product structure (MDL alone
  gives the wrong-merge). "Which projection to loop-close" = the general structured-world-model problem, not a base-10 case.
- **RISK — the similarity-kernel smuggle (top "fooling ourselves" candidate, Fable):** factored closure needs a kernel that
  calls two states "same units, different tens" — but that kernel IS the units projection, the factorisation we claim to
  discover. Honest scope: numeral notation HANDS us the coordinates (positions); a generic recognition primitive gives
  positional token-equality; only the PERIODS per position are discovered. So "periods discovered, not imposed" must read
  "periods discovered GIVEN the coordinates." Discovering the coordinate itself from a RAW count is the real hard part —
  and it does not happen (a raw count has no recurring projection); it needs the notation or a modular task. Do not overclaim.
- **Boundary (self-consistent):** from ATOMIC counting (no digit projection) nothing recurs → still a line (base-10 is
  notational). MODULAR tasks (mod 10) make the FULL state recur → ordinary loop closure → a ring (the P0 control). Positional
  place value is exactly what needs the FACTORED version.
- **Why it is a general win, not a math hack:** loop closure is THE mechanism that turns path integration into a correct
  TOPOLOGICAL map (SLAM; hippocampus; `l6_grid` defers topology to "loop closure in the layers above"). At the FACTOR grain
  it discovers PRODUCT/cyclic structure that whole-state closure misses → smaller, correctly-topological, transferable maps
  of ANY environment with recurring substructure: a re-entered room's corner (map closes, drift corrected); a game's
  counter/toggle/patrol (a cycle, not an ever-growing tree of "new" states); periodic dynamics (a short cycle). It is the
  shelved GRID MODULES earning their keep — the multi-scale periodic codes ARE the per-factor loop-closure codes,
  complementary to the conjunctive full-state SR — i.e. the TEM move (factorise structure/content → generalise).

## THE UNDERLYING MATHEMATICS — a compass, not an engine
- **Group representation theory is the rigorous WHY** (the load-bearing half). Translation-invariance is a GROUP; grid cells
  are its unitary (Fourier) REPRESENTATION — path integration REQUIRES the group-representation condition (Gao et al. 2021),
  and the grid eigenvectors ≈ Fourier plane waves = the irreducible reps of the translation group (Sorscher/Ganguli 2019).
  So "composition" and the periodic "scales" are not a coincidence — they are the group structure and its representation;
  **addition = the free monoid on the successor.**
- **Category theory is the language ONE LEVEL UP, for cross-domain generalisation** (the Sokoban/ARC hope). A domain = a
  CATEGORY (states = objects, operators = morphisms); a hypothesis/solution = a COMPOSITE MORPHISM to a goal object;
  cross-domain transfer = a FUNCTOR (shared structure, content varies) = exactly TEM's structure/content split; transfer =
  a universal construction (pullback). It unifies "hypothesis = a path to a goal" across math / Sokoban / ARC, and its
  universal constructions are the "forced, no ad-hoc design" appeal that could SAVE WORK.
- **Caution (bitter lesson):** both are DESCRIPTIVE — they say WHAT structure to capture, not HOW to learn it from data (the
  geometric mechanism stays ours). The project already ran a CT-SYMBOLIC line (CTKG / Kan extensions) and PAUSED it for the
  geometric TBT line. ⇒ use them as a COMPASS (is our operator set a group, closed under composition? do transfer + the
  universal constructions fall out?), a focused STUDY to see if category theory names the ONE unifying abstraction — NOT a
  resurrected symbolic engine to build in place of learning. See [[reference_discovery_regime_transition]] (Kan-extension
  transport was the CTKG framing).

## The probes (open-ended — each is a QUESTION: what is GIVEN, what is DISCOVERED, what we MEASURE, the Sokoban analogue)
- **P0 (done)** — naive atomic succession → baseline: no structure, no generalisation.
- **P-succession** — learn `+1` as a translation-invariant operator + path-integrate. *Does a grid/periodic code emerge,
  and does succession EXTRAPOLATE past the trained range?* (Does structure form at all?) Sokoban analogue: learn the
  agent-move operator as translation-invariant.
- **P-plus** — with succession learned, present `a+b` examples with `+` OPAQUE. *Can the model DISCOVER the answer is `b`
  succession-steps from `a` — read `+` off the structure?* **The FALSIFIABLE prediction (the one place CT earns its keep):
  if `+` is the free monoid on succession, a system that has correctly learned succession as an OPERATOR must produce
  arbitrary `a+b` for UNSEEN `b` with NO addition-specific training — the extension is unique and forced. If it needs
  addition-specific data to generalise over `b`, read-off is FALSIFIED (`+` is being learned as its own object).** This is
  the composition-fidelity probe `M(a∘b)=M(a)·M(b)` from [[project_recurrent_world_model]]. Sokoban analogue: the rewarding
  configuration is REACHABLE by composing learned moves, with no reward-specific data.
- **P-carry** — multi-digit: the recursive rung. *Is carry FREE in a nested multi-scale frame (the rollover of nested
  periodic phases), or does structure-reading break?* If it breaks, does it need an explicit CONSTRUCTION step (discover the
  nested scales + compress the recurring carry into a reusable sub-operator during CONSOLIDATION — the DreamCoder *principle*:
  MDL compression into a growing library; TBT-native = TEM/grid + replay, not a symbolic DSL) — or is carry irreducibly
  procedural? NB whether the nested (base-matched) scales are DISCOVERED vs imposed is the scaffolding knob here.
- **(later) P-mult / distributivity** — does a discovered operator become an OVERHYPOTHESIS the next rung composes (the
  ladder / blessing of abstraction, and its TRANSFER)?

## Competing outcomes (what we might learn — not mutually exclusive across rungs)
- **A — STRUCTURE-GUIDED discovery WORKS:** with the right geometry `+` is found as composition, cheaply, by a
  domain-general knob → strong support for building the GSG as STRUCTURE-READING; directly informs Sokoban.
- **B — a CONSTRUCTION step is needed:** discovery requires DISCOVERING the frame's scales (the SR-eigenspectrum / MDL scale
  selection) and compressing recurring structure into a new reusable operator during consolidation (library-learning,
  TBT-native via TEM/grid + replay — NOT symbolic factorize/residual).
- **C — exact carry is irreducibly PROCEDURAL/symbolic:** it does not fall out of geometry — a real boundary (the
  ANS-vs-symbolic fork) telling us where the geometric agent ends and a learned tool / the heterarchy begins.
Likely A for succession/plus, B or C for carry — but this is now an ALGEBRAIC PREDICTION (free → A, quotient → B/C), so the
experiment CONFIRMS the master boundary rather than fishing.

## DISCRIMINATING EXPERIMENTS + better microworlds (sharpened, Fable)
- **The decisive measurement = the CARRY-POSITION ERROR PROFILE.** Train succession, train `+` on a BAND of `b`, then test
  `a+b` with `b` OUTSIDE the band. Read error as a function of carry position: **A** (free read-off) — generalises to unseen
  `b` immediately, error roughly UNIFORM across positions; **B** (construction) — generalises only AFTER a replay/compression
  phase, uniform post-consolidation; **C** (deep/procedural) — SPIKES exactly at rollover boundaries (`…9→…0`), spike height
  SCALES with carry-chain length (`999+1` worst). A and C make OPPOSITE predictions about the SHAPE (not just level) of the
  error — one run discriminates all three.
- **Base-10 is NOT the cleanest microworld** — it conflates the free part, the quotient/carry part, AND a notational
  confound (numerals hand you the units coordinate — the similarity-kernel smuggle). Prefer a TWO-MICROWORLD design, base-10
  as the later composition test:
  1. **Free monoid on 2 generators, NO relations** — isolates read-off in its purest form; a clean GO/NO-GO on the whole
     thesis before spending anything on carry. Any failure here is the MECHANISM's, not the domain's.
  2. **A small NON-ABELIAN group learned from transitions (e.g. S₃)** — dials COMMUTATIVITY as the independent variable
     (abelian = geodesic = coordinate difference, read off; non-abelian = geodesic needs SEARCH). Directly measures where
     read-off breaks as a function of relational structure = the master boundary made experimental.
- **The domain-independent shared mechanism (answers "state it without either domain"):** find the SHORTEST composition of
  learned GENERATORS that carries the current position to where the recurrence/reward signal fires — i.e. GEODESIC-FINDING in
  the learned Cayley graph, "shortest" = MDL over the generator alphabet. It is CHEAP (read-off) when the group is
  free/abelian and EXPENSIVE (search) when non-abelian — the same boundary. `+` = shortest succession-composition to the
  answer; Sokoban = shortest move-composition to the goal config; ARC's read-off-able "recurring factors" = the ABELIAN
  sub-monoids (a counter/toggle/patrol); order-dependent, non-commuting interactions are the search-forced part.

## What CLOSES this phase
A clear read on outcome **A/B/C per rung**, and thus an answer to: *does domain structure guide hypothesis generation,
how much structure is needed, and is that the SAME mechanism Sokoban needs?* That decision feeds straight back into the
ARC/GSG line — whether the GSG is built as structure-reading (find the hypothesis latent in the learned geometry) plus,
if B, a consolidation-compression step.

## Sources / links
Overhypotheses + blessing of abstraction — Kemp, Perfors & Tenenbaum 2007 (Dev Sci); Tenenbaum et al. 2011 (Science).
Grid/residue codes — Fiete/Sreenivasan; Wei, Prentice & Balasubramanian. Structure/content factorisation — Whittington
et al. 2020 (TEM, Cell). Library learning (the PRINCIPLE) — Ellis et al. 2020 (DreamCoder). Number in the brain (context
only; ANS is NOT our target) — Dehaene; Nieder. **Master boundary (Fable 2026-07-01):** free monoid / universal property
(read-off ⇔ homomorphism on generators); carry depth = circuit complexity (binary addition ∉ AC⁰, carry-lookahead / TC⁰);
predictive-sufficiency factorisation = causal states / ε-machines (Crutchfield) + bisimulation; contraction⊥expressivity +
bounded-BPTT vs equilibrium-depth from [[project_recurrent_world_model]] (DEQ line; composition-fidelity `M(a∘b)=M(a)·M(b)`).
Internal: [[reference_hypothesis_generation]], [[project_math_hypothesis_probe]], [[reference_grid_sr_eigenbasis]],
[[reference_vector_navigation]], `GROUNDING_PLAN.md`.
