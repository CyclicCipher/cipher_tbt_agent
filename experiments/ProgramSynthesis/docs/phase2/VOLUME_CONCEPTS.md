# Volume Concepts — planning & knowledge store

> **Status: Phase 2 charter** (opened 2026-06-13) — the start of the redesign that follows Phase 1
> (the built-and-proven symbolic world-model agent; see `FINDINGS.md`). **Goal:** investigate storing
> knowledge as *volumes / regions* in a space (concepts as regions, not points) as an **inductive
> bias** — first in our symbolic agent, possibly transferred to neural nets. We are borrowing a
> **knowledge-representation idea**, *not* adopting anyone's architecture wholesale.
>
> **Provenance caveat (read first).** The seed came from two YouTube videos by the channel
> **Inductica**, presenting **"Conceptron"** by **Ron Pisaturo**. We do **not** have a verbatim
> transcript (transcript services are bot-blocked; YouTube exposes only boilerplate). What we have:
> the video *descriptions*, a *paraphrased* summary, and **direct quotes supplied by the project
> owner** (the most reliable source — marked `[owner-quote]` below). Pisaturo's "architecture" is a
> **concept with no implementation**; treat his specifics as motivation, not method. Going from
> concept → implementation is our job, and his initial framing may not be the best one.

---

## 0 · Resolved design decisions (2026-06-13)

Two design questions are settled; everything below is read in their light.

**Resolution 1 — Not one embedding space: a *graph of local spaces*.** A single global embedding
space is a *neural-net* constraint (backprop needs a fixed differentiable substrate); a symbolic
model has no such obligation. A concept that cares about 3 dimensions should be a region in *its own
3-dimensional subspace*, not a slab pinned on 3 axes and "don't care" across the rest of a global
space (wasteful, noisy, and it invents false comparisons between incommensurable concepts). So:
- **Nodes** = concepts, each a *region in its own small subspace* (only its relevant dimensions).
- **Edges** = structure-preserving maps between local spaces — *sideways* relations (e.g.
  `4 →opens→ 5`) and *upward* `is-a-dimension-of` edges (Resolution 2).
- **Shared dimensions glue** the local spaces, and the gluing must be consistent — a **sheaf**
  condition (the formal scaffold from the paused CTKG line).

This is the "modification of a knowledge graph" we were reaching for: a KG whose nodes carry *local
geometry*. Classic KG = symbols + edges (no geometry); neural net = one global geometry (no graph);
**this = a graph of small geometries.** Our current schema (`blocker_colors`, `contact_effect`,
`move_model`) is already a discrete instance — relations over 1–3 quality dimensions each — so this
is an **upgrade path, not a rewrite**.

**Resolution 2 — Dimensions are a discovered, MDL-promoted, *grounded tower*.** Most dimensions are
not primitive. A concept becomes a *dimension* for higher concepts: "democracy-ness" is not a given
axis — you must learn the concept of democracy before you can rate a country along it, and democracy
is itself a region over lower concepts, down to raw sensory dimensions. So:
- **Depth = a tower of concepts-as-dimensions** — the explicit, grounded version of why neural nets
  need depth (a CNN's edges→parts→objects is the same recursion, entangled in weights). A high node's
  subspace is *spanned by the lower nodes it connects to*; every chain bottoms out at sensory nodes ⇒
  **every concept is grounded by construction** (a structural answer to symbol-grounding).
- **MDL/compression decides the geometry** — both *how many* dimensions a concept needs and *which*
  concepts get **promoted** to dimensions (a concept earns axis-hood when treating it as one
  compresses higher-level structure). This is the precise junction with the discovery notes
  (`DISCOVERY_PROGRAM.md`): the discovery loop (regime-change, MDL) operates *on* this representation,
  and MDL is the single objective that carves subspaces, fits regions, proposes edges, and promotes
  dimensions. **Compactness (H2) stops being a perk and becomes the organizing principle.**
- **Some abstract dimensions form only when instrumentally useful** — pulled into existence top-down
  by a task / residual / value, not by unsupervised clustering alone (active-inference / MuZero
  value-equivalence). "Democracy-ness" crystallizes because it *does work* (compresses consequences).

**The prior floor (what we do NOT eliminate).** Three domain-general priors remain; going below them
isn't bitter-lesson virtue, it's non-function:
1. a **sensory interface** (the modality's raw substrate — I/O, not knowledge);
2. a **metric / similarity** (so you can cluster at all);
3. a **compression + prediction drive** (MDL + residual — what builds and promotes the tower).

Everything conceptual — objectness, persistence, contact, agency, … "democracy-ness" — is
**discovered**, not seeded. This is the operational form of "minimize priors, bitter-lesson-max."
Ref: [[feedback_bitter_lesson]].

**Honest difficulties (open).**
- **Cycles / mutually-defining concepts** (democracy ↔ legitimacy): a strict bottom-up DAG can't
  express co-definition; the tower must *settle* (co-determine regions jointly) — exactly a job for
  the RecurrentWorldModel settling operator. The clean layered picture is an approximation.
- **Falsifiable dimensions:** a promoted dimension must keep paying its MDL/predictive rent under
  distribution shift or be **demoted** — the guard against spurious axes that only compress the
  training distribution (the memorization trap).

**Open framing question (owner, 2026-06-13) — is a learned rule a *concept* or an *operation on
concepts*?** "Concepts as volumes" may conflate two different kinds. A box-region is a *kind* (a
"what" — a category/node); a **law/rule is a relation or transformation** (a "how" — it *acts on*
quantities). In the graph these are different sides: **nodes = regions (concepts)**, **edges = maps
(operations/laws)**. So a law likely belongs on the **edge** side, not as a node-region — and a box
is doubly the wrong shape for it (see §11A). The genuinely open part: whether an operation can be
*reified* into a concept (a concept *of* the operation — à la `fn`-as-value, or Merge making a
morphism into an object). Do **not** assume rules are just another kind of region; carry this
distinction explicitly as the representation grows. Tracked in §8.

---

## 1 · The core idea (and why it's attractive)

A **point** embedding gives each concept one location. A **region/volume** gives it an *extent*,
which unlocks an **algebra done as geometry**:

- **Intersection = AND / context-narrowing.** "bank" ∩ finance-context vs "bank" ∩ river-context →
  context-dependent meaning falls out as set intersection.
- **Containment = IS-A / instantiation.** A *concrete* (instance) is a point inside its *concept's*
  region. Asymmetric — cosine similarity (symmetric) cannot represent this; it is exactly the
  concept↔concrete relation, and a **geometric handle on the binding problem** (bind instance→type =
  containment test, no attention required). `[owner: "concretes and how concepts relate to them
  helps with the binding problem"]`
- **Volume = measure** (typicality / generality); **union = OR**, **complement = NOT**.

> Why we care: *spaces let you do algebra and logic through geometry and transformations.* For a
> symbolic system this is a way to get graded, context-dependent meaning that discrete graphs lack;
> for a neural net it is an **inductive bias** toward the structure LLMs may only reach emergently
> with enormous data.

## 2 · What Pisaturo actually proposed (paraphrase + owner-quotes)

- **Concepts as volumes in a hyperspace**, citing **Gärdenfors' Conceptual Spaces** and **Ayn
  Rand's** theory of concept-formation (measurement-omission). *(paraphrase)*
- **Diagnosis ("Modern AI's Fundamental Weakness"):** "LLMs do not encode the semantic meaning of
  the tokens they output" — pure statistical next-token prediction; hallucination is the symptom;
  knowledge graphs are too brittle/discrete to hold graded meaning. *(video description)*
- **Vector → matrix.** Give the embedding a second row/column so it becomes a **matrix encoding the
  *range* of the concept across the space**, i.e. more than a point. `[owner-quote, paraphrased]`
  → *This is literally the literature's mean+covariance (Gaussian embedding) or center+offset (box
  embedding); see §5. Good instinct, not novel — which is reassuring, it means the math exists.*
- **Concepts and concretes share one hyperspace.** `[owner-quote]` *"Concepts as volumes has this
  great advantage because the embeddings for both concepts and concretes are stored in the same
  hyperspace, which gives the relationship between concepts and concretes. But there are still at
  least 2 challenges to overcome: finding a method of arriving at the volumes by gradient descent,
  and how to generate text responses."*
- **Unsupervised concept formation via clustering** was mentioned as the route to volumes. *(paraphrase)*

### Pisaturo's two named challenges — and what the literature already says
1. **Reaching volumes by gradient descent.** Hard region boundaries have zero/ill-behaved gradient.
   *Already addressed:* probabilistic / smoothed boxes, **Gumbel boxes** (Dasgupta et al. 2020),
   Gaussian embeddings (differentiable by construction). → tractable.
2. **Generating text (decoding from a region, not a point).** Open-ish: sample within the volume, or
   map region → token distribution. Our symbolic setting sidesteps this first (we act, not emit text).

## 3 · The owner's hypotheses (what we want to test)

Marked as hypotheses, not facts:

- **H1 — Inductive bias beats emergence.** LLMs *might* form conceptual spaces emergently with enough
  data/training, but an explicit volume bias may get there cheaper and more reliably. Our **symbolic
  model cannot do it emergently at all**, so the bias is the only route there.
- **H2 — Compactness.** Volume storage may be **vastly more compact** than a comparable knowledge
  graph (one region generalizes over many enumerated triples — intensional vs extensional) or than
  distributed weights. *Testable:* bits/params to cover a domain as regions vs as a KG vs as weights.
- **H3 — Interpretability.** A model that stores knowledge as regions **from day 1** should make the
  computational spaces and their algebras **easy to find** — vs reverse-engineering them out of
  weights (the mech-interp problem). This could in turn teach us **how to make NNs smart**: how to
  give a net *innate / core knowledge* like our symbolic agent has, and how to *study the dynamics of
  thinking* directly.
- **H4 — Bidirectional transfer.** Prove it in the symbolic model → port the lesson to NNs; or the
  reverse. We start wherever it's cheapest to test (symbolic).
- **H5 — Attention may be (partly) unnecessary.** *(owner's sharpest speculation)* If context-
  dependent meaning lives in the representation (a volume carrying its senses), you may not need
  **attention to transform the embedding** to disambiguate (e.g. different uses of the same word) —
  you'd **intersect** the volume with the current context instead. See §4 for the honest assessment.

## 4 · Honest assessment of H5 (does volume rep remove attention?)

Attention does at least **two** jobs:
1. **Context-dependent representation** (disambiguation / binding) — *plausibly replaceable* by
   "intersect the concept's volume with the context region." This is the part H5 targets, and it's
   credible.
2. **Routing / mixing across positions** (move & compose information between tokens; long-range
   dependency). Intersection is a **binary op on two given volumes** — it does **not** tell you
   *which* volumes to intersect. Choosing what binds to what *is* the routing problem attention
   solves by content. So volumes shrink attention's job (#1) but don't obviously eliminate #2.

**Where this points (research-worthy):** pair **volume representations** (handle #1 by geometry) with
a **non-attention router** for #2 — e.g. an SSM/recurrent state (Mamba already drops attention for
routing), or a *structural/positional* binding (PoPE-style; ties into our binding / temporal-PoPE
thread). "Volume rep + cheap router" is a concrete, falsifiable path to *less or no* attention. Do
not assume it for free.

## 5 · Academic grounding (the rigorous version of "volumes")

The legitimate lineage Pisaturo gestures at — these are what we'd actually build on:

- **Gärdenfors 2000 — *Conceptual Spaces: The Geometry of Thought*** (MIT Press). Concepts as convex
  regions over quality dimensions; similarity & betweenness as geometry. The foundation he cites.
  *(Note: Rand's "measurement-omission" ≈ a convex region spanning a range along a retained quality
  dimension — an elegant correspondence, but motivation, not method.)*
- **Vilnis & McCallum 2014 — *Word Representations via Gaussian Embedding*** —
  [arXiv:1412.6623](https://arxiv.org/abs/1412.6623). Word = Gaussian (mean μ + covariance Σ). The
  covariance **is** Pisaturo's "range" / second matrix dimension. Captures entailment & specificity.
- **Vilnis, Li, Zaheer, McCallum 2018 — *Probabilistic Embedding of KGs with Box Lattice Measures***
  — [arXiv:1805.06627](https://arxiv.org/abs/1805.06627). Concepts as **boxes**; containment =
  entailment; volume = probability.
- **Dasgupta et al. 2020 — *Improving Local Identifiability… Gumbel Boxes*** —
  [arXiv:2010.04831](https://arxiv.org/abs/2010.04831). Makes boxes **trainable by gradient descent**
  (answers Pisaturo's challenge #1).
- **Ren, Hu, Leskovec 2020 — *Query2Box*** — [arXiv:2002.05969](https://arxiv.org/abs/2002.05969).
  Multi-hop **logical** KG queries (∧, ∃) as **box operations** — intersection-as-reasoning, working.
- **Lattice Representation Hypothesis of LLMs 2026** —
  [arXiv:2603.01227](https://arxiv.org/abs/2603.01227). Argues concepts are **intersections of
  half-spaces** *already emergent inside LLMs* — both a caution to Pisaturo's "LLMs encode no
  meaning" and evidence the region structure is real (relevant to H1/H3).
- **Analogical Reasoning Within a Conceptual Hyperspace 2024** —
  [arXiv:2411.08684](https://arxiv.org/abs/2411.08684). Reasoning/analogy inside a conceptual
  hyperspace — close academic cousin.
- *(Also relevant: cone embeddings & hyperbolic entailment cones for hierarchy; density/Gaussian
  process variants.)*

## 6 · Connection to our existing work

- **F_τ(C) / QKG context-dependent goal** (`world_model.py: required_absent`, `goal_sufficient`): our
  `required_absent` is a **discrete set**; the volume idea is its **continuous generalization** — the
  goal condition becomes a *win-enabling region* and `goal_sufficient` becomes a **point-in-region**
  test. See [[reference_quantum_knowledge_graph]] (we *discover* context; they annotate it).
- **Representation vs discovery (the key complementarity).** Conceptual-spaces give a better
  *representation* of context-dependent meaning; our agent already *discovers* the context from the
  sparse residual. Keep our discovery loop; borrow their representation.
- **Binding problem / temporal-PoPE thread** — concretes-in-concept-volume = geometric binding; ties
  to "how information needs to be bound" and to H5's routing question.
- **RecurrentWorldModel** — a settling operator's **fixed-point set under partial clamping is itself
  a region** of consistent completions, so "concept as volume" might *emerge from the dynamics*
  rather than be a designed primitive (speculative; relevant to the "Represent" mode).

## 7 · The bitter-lesson guardrail (non-negotiable)

Gärdenfors' quality dimensions are usually **hand-specified**; convex-region priors are **designed
structure**. That is the same "seeded language" trap as `LIMITATIONS.md §4`. **The region's shape and
its relevant dimensions must be *discovered / learned*, not designed** — otherwise we have only moved
the hand-coding into geometry. This is why **unsupervised concept formation (clustering / self-
supervision)** is load-bearing, not optional. Ref: [[feedback_bitter_lesson]].

## 8 · Open questions

1. Which region family? **Resolved in principle (2026-06-13) — don't pick one; let MDL grow shape from
   one general primitive (the half-space).** Axis-aligned faces are cheap to encode, oriented ones
   expensive, so MDL defaults to a box and buys orientation only when it pays → box / rotated polytope
   / cone emerge per concept. Built + benchmarked (`volume/halfspace.py`, `volume/benchmark.py`); see
   the §9 P0 note. Open remainder: curvature is offloaded to the dimension-tower and multi-modality to
   `union` (the convex limit the benchmark exposes), not to exotic single-region shapes.
2. How are the **dimensions of the space discovered**, not given? (the §7 guardrail)
3. Gradient path to volumes — Gumbel/soft boxes vs Gaussian; does it survive our setting?
4. Does intersection truly offload attention's job #1, and what is the cheapest router for #2 (H5)?
5. Compactness (H2): how much smaller than our KG / a weight matrix, measured?
6. Interpretability (H3): can we *read* the algebra back out, and does that teach us how to seed core
   knowledge into a net?
7. Decoding/acting from a region (Pisaturo challenge #2) — for us: pick an action from a region.
8. Unsupervised concept discovery — which clustering, and does it recover concepts we never labelled?
9. **Is a learned rule a *concept* or an *operation on concepts*?** (owner, §0) — node-region vs
   edge-map; can an operation be *reified* into a concept (`fn`-as-value / Merge)? Don't assume rules
   are just another region.
10. **How are relations/laws represented and discovered** — as functional constraints/manifolds on
    *edges*, via MDL-scored equation discovery (§10A), not as box-regions.

## 9 · Proposed path (symbolic-first; cheapest test of the idea)

- **P0 — Formalize.** Pick an initial region family (start with **boxes**: trainable via Gumbel,
  cleanest algebra) and define the ops we need: intersection, containment, volume, union.
  > **Built (2026-06-13):** `volume/box.py` — a `BoxConcept` (region over a *chosen subspace*,
  > unconstrained elsewhere) + `fit_box_concept`, which **discovers the relevant subspace** by greedy
  > forward selection under a two-part MDL code (model = the box; data = *conditional label entropy*).
  > Tests (`tests/test_volume_concept.py`, 3/3): from noisy 6-D data it recovers the planted 2-D
  > subspace + region, generalizes to held-out (>0.9), and — the knife-edge — **invents no structure
  > from noise labels**. First validation of Resolution 1 + "MDL decides the geometry" in miniature.
  > Caveat: the data term is conditional entropy (info-gain), not yet a fully bits-exact code; boxes
  > are hard-edged (Gumbel/soft boxes are the differentiable upgrade for when we need gradients).
  > **Algebra built (2026-06-13):** `volume/algebra.py` — `meet` (AND / context-narrowing over the
  > union of subspaces, empty when disjoint), `entails` (IS-A / specificity — the *asymmetric*
  > relation points can't express), `volume` (generality). Tests (`tests/test_volume_algebra.py`,
  > 7/7) confirm conjunction-across-subspaces, shared-dim tightening, the greatest-lower-bound
  > property (meet ⊑ both), entailment asymmetry, and context-narrowing ⇒ a strictly more specific
  > concept. *Deliberate structure:* concepts in `box.py`, operations in `algebra.py` — mirroring
  > §0's concept-vs-operation distinction.
  > **Shape discovery built (2026-06-13):** `volume/halfspace.py` — `HalfspaceConcept` (intersection of
  > K half-spaces) + `fit_halfspace_concept`, greedy MDL over one primitive (the half-space) with an
  > axis-cheap / oriented-expensive encoding, so **shape emerges, never chosen**. `volume/benchmark.py`
  > pits the fixed axis-box against MDL-shape on known truths; `tests/test_volume_shapes.py` (5/5) lock
  > the bitter-lesson result: **axis box recovered with 0 oriented faces** (parsimony); **diamond &
  > triangle discover orientation and beat the box** (triangle = 2 axis + 1 oriented hypotenuse); **disk
  > ties** (MDL stays parsimonious at this sample size); **two_blobs exposes the convex limit** → the
  > honest case for `union`. This *retires the box-shape question by removing the choice*, per §8.1.
  > **Union built (2026-06-13):** `volume/union.py` — `UnionConcept` (membership = OR of convex
  > regions) + `fit_union_concept`, which **discovers the number of modes** by MDL (cluster positives →
  > fit a half-space region per mode → keep the M with lowest union code). Tests
  > (`tests/test_volume_union.py`, 4/4): convex concepts stay **M=1** (parsimony), **two_blobs splits
  > into M=2** (0.89→**0.97** held-out, beating the single convex region). Closes the convex limit;
  > curvature still routes to the dimension-tower, not to exotic single-region shapes.
- **P1 — Retrofit F_τ(C) as a region.** Replace the discrete `required_absent` with a **win-enabling
  region** in a small feature space; `goal_sufficient` = point-in-region. Same agent, richer
  condition — a contained first experiment in a domain we fully control.
- **P2 — Unsupervised concepts via clustering.** Discover object/feature concepts as **regions** from
  observation (not labels), honoring §7. Test: does it recover the LockPath color-roles we currently
  get for free, *without* being told them?
- **P3 — Test H2/H3.** Measure compactness vs the current schema/KG; check whether the learned
  algebra is human-readable.
- **P4 — NN transfer (only if P1–P3 promise).** Give a small net a **box/Gaussian representation
  layer** as an inductive bias; test data-efficiency of context-dependence, interpretability, and the
  **attention-reduction** hypothesis (volume rep + SSM router vs attention).

## 10 · Future directions (recorded 2026-06-13)

**A · Laws & relations are EDGES, not box-nodes (Q1).** Physical laws (gas laws `PV=nRT`, Newton
`F=ma`) are *not* regions/kinds — they are **functional constraints**: curved, measure-zero manifolds
in the joint quantity-space that no axis-aligned box (or any convex region) can represent. In the
graph of local spaces a law is an **edge** — a structure-preserving map / constraint between
quantity-dimensions — i.e. exactly the inter-space map (= Merge) flagged as the hard part. To learn
them, add **relation/equation discovery** alongside region-fitting, scored by the *same MDL objective*:
- **Symbolic regression / equation discovery** is the template — **AI Feynman** (Udrescu & Tegmark,
  [arXiv:1905.11481](https://arxiv.org/abs/1905.11481)) recovered 100 physics equations from data;
  **SINDy** (Brunton, Proctor & Kutz, PNAS 2016) discovers the *differential* laws (Newton needs
  derivatives → dynamics/ODE discovery) by sparse regression. MDL/sparsity = shortest law that fits.
- **Dimensional analysis (Buckingham-π)** is the domain-general prior worth keeping: laws relate
  *dimensionless groups*; units massively prune the search. Few, general, not per-task — bitter-lesson-safe.
- **The variables come from the tower** (§0 Res. 2): AI Feynman is *given* P,V,T; discovering that
  "pressure"/"temperature" are the right quality-dimensions from raw sensorimotor data is the upstream
  dimension-tower problem.
- **Use in modelling/planning:** a learned law is a transportable piece of the forward model
  `T(s,a)→s'` — the agent *simulates* an action's consequence through the law and plans to a goal
  (MuZero learn-model-and-plan / active-inference rollout). Payoff over a black-box net (`FINDINGS §5.1`):
  an explicit law **extrapolates** to unseen regimes and **composes** — the invariant that makes shift
  non-catastrophic.
- **Caveat (owner, §0 framing question):** a law is an *operation/relation*, not a concept — do not
  fold it into the region representation. This is the strongest argument that boxes (and regions in
  general) are not the whole story.

**B · SIGReg / LeJEPA — anti-collapse & sketching (Q3).** SIGReg (Balestriero & LeCun, *LeJEPA*, 2025)
replaces the ad-hoc anti-collapse stack (VICReg terms, stop-grad, EMA teachers) with one principled
target: make the embedding distribution an **isotropic Gaussian**. The theorems, and what transfers
even though we are not building JEPA:
- **Isotropic Gaussian = max-entropy** for a fixed covariance (no collapsed or privileged directions);
  shown to minimize a downstream worst-case risk → the least-assumption healthy representation.
- **Cramér–Wold + sketching:** a distribution is Gaussian iff all its 1-D projections are, so push
  *random 1-D projections* toward Gaussian with a closed-form statistic — O(d), unbiased, one
  hyperparameter, no d×d covariance (vs VICReg).
- **Lessons:** (1) any self-supervised concept/dimension learner (ours) needs an explicit
  **non-degeneracy pressure** — borrow a principled one, don't re-invent penalties; (2) **sketching is
  a general high-dim tool** — check isotropy / dimension-independence / region well-formedness via
  random projections instead of full densities; (3) **max-entropy ⟺ MDL** — SIGReg's target and our
  "MDL decides the geometry" are duals (least-committed ⟺ shortest code); (4) **layering** — isotropy
  keeps the *substrate* full-rank and metric-meaningful (our prior-floor "metric"), while the discovery
  loop carves the meaningful *anisotropic* concept-tower on top; no real conflict (cf. the Lattice
  Representation Hypothesis: concepts as half-space intersections inside an otherwise unstructured space).

**C · Relation/edge design basis — rules as conditionable manifolds (2026-06-13, settled w/ owner).**
A rule/law/relation is **not** primarily a symbolic equation — the dog learns how things move with no
symbols, so a rule is a **(soft-capable) manifold in the joint (input × output) / (state × next-state)
space**: *stored* like a concept-region (the same machinery, lifted to a product space) and *applied*
by **conditioning** (slice the manifold at the input → read off the output region). This reconciles
the §0 concept-vs-operation question — a relation is a *region in product space* (storage) that is an
*operation* when conditioned (use). Symbols are an **optional downstream read-out** (compress a clean
manifold into a formula when one exists), never the substrate.

**MDL-arbitrated cascade for rule-learning** (cheapest description wins):
1. **simple region/manifold** over the joint subspace (clean functional / low-face relation) — reuse
   box/halfspace. A *thin* functional relation (`y≈x+c` = a diagonal slab) is captured by **oriented**
   half-spaces; with no negatives, the constraint = the **low-variance directions** of the joint data
   (codimension chosen by MDL — ties to SIGReg's variance view).
2. **multivalued / branchy** → **union** (already built).
3. **irregular but structured (spiral)** → discover the **coordinate change** that makes it simple
   (spiral → polar → line) — MDL-gated. *This is the hard case (below).*
4. **genuinely structureless** → store a **soft energy landscape / density** — the honest "memorize the
   residual" floor (the soft, T>0 generalization of our hard regions; a cheap geometric energy, no NN
   training required).

**Owner's concern — case 3 needs two things, both open:**
- **(i) Vocabulary completeness.** To find "spiral→polar" the transform search must be able to
  *express* polar. Need a basis expressive enough to convey all the math in a space yet parsimonious
  enough for MDL to score. Candidate: a **Sheffer-like minimal-complete basis** — one operator that
  composes into all elementary functions: the **eml operator** ([[reference_eml_operator]],
  `eml(x,y)=exp(x)−ln(y)`, all 36 calculator functions as depth-≤8 trees), so description length =
  composition-tree size. The cheap/expensive asymmetry lifts to transforms: **linear/affine cheap**
  (rotations, scalings), **nonlinear (eml-composed) expensive** — MDL buys nonlinearity only when it
  compresses. Honest limit: this is the `LIMITATIONS §4` vocabulary boundary recurring at the transform
  level — a complete-but-finite basis covers elementary coordinate changes, not arbitrary functions.
- **(ii) Escaping local minima.** MDL search over transforms is non-convex and *deceptive* (a partial
  polar transform doesn't simplify the spiral, so greedy — our box/halfspace fitter — stalls). Portfolio:
  (a) **soft-energy + annealing/Langevin** (case 4 doubles as a search-smoother — softness gives
  gradients, temperature escapes minima); (b) **staged compositional Merge / the tower** — discover `r`,
  make it a primitive dimension, then `θ`, then the spiral is linear; staging changes the search *space*
  rather than descending a fixed deceptive one (the regime-change move); (c) **population / beam / MCTS**
  guided search (DISCOVERY_PROGRAM's MuZero-guided-search fallback) + randomized restarts; (d)
  **residual-directed proposals** — the un-compressed residual points at what the coordinates miss,
  seeding the next transform. This *is* the across-regime directed search the discovery program names as
  the unsolved core; MDL stays the guard against overfitting (a transform must **compress**, not just fit).

**Scope:** case 3 is **deferred** — the ARC-AGI-3 replica's dynamics are discrete/linear (translations,
color-swaps), so the minimal relation/edge it needs is **case 1** (a conditionable transition manifold
in a discovered joint subspace; constraints = low-variance directions). Build that; bring in cases 3–4
only when a domain demands curved coordinates or genuinely irregular relations.

> **Case-1 built (2026-06-13):** `volume/relation.py` — `RelationConcept` + `fit_relation`. A rule is
> stored as the **low-variance directions** of the standardized joint (input, output) data (its
> near-constant linear combinations = its constraints) and *applied by conditioning* (`predict`:
> least-squares-solve the constraints for the output given the input). MDL-flavoured selection picks
> the **codimension** by a spectral gap on standardized eigenvalues (scale-free ⇒ "constraint" =
> linear dependence). Tests (`tests/test_volume_relation.py`, 5/5): recovers a linear function, a
> planar relation, and a **codimension-2** relation (predicts both outputs) exactly; is noise-robust;
> and **declines** (codim 0 → `predict` None) when the output is independent of the input — it invents
> no relation. This is the concrete form of "rule = region in product space (storage), operation when
> conditioned (use)" — the §0 concept-vs-operation reconciliation, working in code.

## 11 · Sources

- Inductica, *Modern AI's Fundamental Weakness* — [youtu.be/qQ21SC9TZIY](https://www.youtube.com/watch?v=qQ21SC9TZIY).
- Inductica, *New AI Architecture Uses Volumes Instead of Embeddings* (Conceptron, Ron Pisaturo) —
  [youtu.be/9k5XdbStgNE](https://www.youtube.com/watch?v=9k5XdbStgNE).
- Academic anchors: §5 (Gärdenfors; Gaussian, Box, Gumbel-Box, Query2Box embeddings; Lattice
  Representation Hypothesis; Conceptual-Hyperspace analogy).
- Owner quotes recorded inline (`[owner-quote]`) — primary source pending a real transcript.
- Related project docs: `LIMITATIONS.md §4`, `FINDINGS.md §5.1`, `world_model.py` (`required_absent`),
  memories [[reference_quantum_knowledge_graph]], [[feedback_bitter_lesson]].
