# Domain-General Structure Discovery

**Goal:** Replace all domain-specific consolidation code with mathematically
principled algorithms that discover categorical structure (products, coproducts,
NNO, functors, natural transformations, adjunctions) from raw observational
data in ANY domain.

**Constraint:** No function may reference digits, separators, carry rules, NNO
tokens, or any other domain vocabulary. Structure emerges from graph dynamics
and observation patterns only.

---

## 1. Formal Concept Analysis (FCA) — Type Hierarchy Discovery

**What it does:** Given a binary incidence relation (objects x attributes), FCA
discovers the complete concept lattice. Each formal concept is a maximal
rectangle in the incidence matrix: a set of objects sharing exactly a set of
attributes, and vice versa.

**Connection to category theory:** FCA is a special case of **Isbell duality**
(Avery & Leinster 2021). The derivation operators form a **Galois connection**
= adjunction between poset categories. The concept lattice viewed as a thin
category has:
- **Meet = product** (tokens sharing all the same contexts)
- **Join = coproduct** (contexts shared by all the same tokens)
- The Galois connection itself IS an adjunction

**Application to CTKG:** Build the incidence matrix from co-occurrence: rows =
token nodes, columns = observation contexts. FCA directly discovers:
- The type hierarchy (subconcept-superconcept = subtype-supertype)
- Products and coproducts as lattice operations
- Adjunctions as the derivation operators

**Library:** Custom bottom-up implementation in `ctkg/logic/fca.py`. No external
dependency — the `concepts` library was tested and confirmed identical results,
then removed. Bottom-up construction is faster and avoids exponential blowup
via max_concepts cap.

**Key references:**
- Avery & Leinster, "Isbell conjugacy and the reflexive completion" (2021)
  https://arxiv.org/pdf/2102.08290
- nLab: https://ncatlab.org/nlab/show/formal+concept+analysis
- n-Category Cafe discussion (Baez, Leinster):
  https://golem.ph.utexas.edu/category/2013/09/formal_concept_analysis.html
- Hitzler, Krotzsch & Zhang, "A categorical view on algebraic lattices in FCA"
  Fundamenta Informaticae 74(2-3), 2006

**Status:** Library installed. Needs integration with KnowledgeGraph's
co-occurrence data.

---

## 2. Equational Theory Discovery (Ruler / QuickSpec)

**What it does:** Given a grammar of operations and an evaluator, automatically
discovers equational rewrite rules (e.g., `x + 0 = x`, `x + y = y + x`).

**Ruler algorithm (OOPSLA 2021, Distinguished Paper):**
1. Enumerate all terms up to size i
2. Add to an e-graph, run equality saturation with known rules R
3. Find candidate equations: terms in different e-classes but with matching
   evaluation vectors (same output on all test inputs)
4. Validate candidates (SMT, fuzzing, model checking)
5. Select minimal subset, add to R
6. Repeat with larger terms

**QuickSpec (Haskell):** Simpler variant — enumerate terms, evaluate on random
inputs, group terms that always agree. No e-graphs needed.

**babble (POPL 2023):** Anti-unification over e-graphs discovers common
abstractions from a corpus. Finds patterns equivalent modulo the theory.

**Application to CTKG:** The BFM operations ARE the evaluator. Enumerate
compositions of observed morphisms, group by output, extract equations. This
discovers algebraic laws from data without any domain knowledge.

**Limitations to examine:**
- Term enumeration is exponential in term size
- Enumo (OOPSLA 2023) addresses this with programmable exploration DSL
- Requires a well-defined evaluator (we have BFM, but what about partial data?)
- QuickSpec needs random input generation (we can sample from observations)
- Neither discovers STRUCTURE (products, NNO) — only EQUATIONS between
  existing operations. FCA discovers structure; Ruler discovers laws.

**Key references:**
- Ruler: https://arxiv.org/abs/2108.10436 / https://github.com/uwplse/ruler
- Enumo: https://ajpal.github.io/assets/files/enumo-paper.pdf
- babble: https://cseweb.ucsd.edu/~npolikarpova/publications/popl23.pdf
- QuickSpec: https://github.com/nick8325/quickspec
- Tao et al., Equational Theories Project (2024):
  https://github.com/teorth/equational_theories
- CCLemma (ICFP 2024):
  https://cseweb.ucsd.edu/~npolikarpova/publications/icfp24.pdf

**Status:** No Python e-graph library installed yet. The core idea (enumerate
+ evaluate + group) can be implemented directly on the CTKG without external
dependencies. Ruler's Rust implementation is reference, not a dependency.

---

## 3. Krohn-Rhodes Decomposition — Algebraic Skeleton

**What it does:** Given any finite transformation semigroup (e.g., the
transition monoid of token sequences), factors it into a cascade (iterated
wreath product) of **simple groups** (reversible symmetries) and **aperiodic
semigroups** (irreversible collapses).

**Why it matters:** The token transition graph IS a transformation semigroup.
Krohn-Rhodes reveals its algebraic skeleton:
- Each simple group component = a reversible symmetry in the data
- Each aperiodic component = an irreversible state collapse
- The NNO appears as the free cyclic monoid (cascade of flip-flops)
- Addition appears as a group component (Z/nZ for modular arithmetic)

**Holonomy decomposition algorithm (SgpDec):**
1. Enumerate image sets of all elements
2. Compute holonomy groups at each level of the image set lattice
3. Build the cascade product

**Cayley graph recognition (Cece & Marin, DMTCS 2020):** A graph is a Cayley
graph of a monoid iff it is deterministic, source-complete, with a propagating
out-simple root. Testable from graph structure alone.

**Key references:**
- SgpDec GAP package: https://github.com/gap-packages/sgpdec
- Egri-Nagy & Nehaniv, "Computational Holonomy Decomposition" (2015):
  https://arxiv.org/pdf/1508.06345
- Egri-Nagy & Nehaniv, "Covering Lemma" (CIAA 2024):
  https://arxiv.org/html/2404.11923
- Egri-Nagy & Nehaniv, "Representation Independent Decompositions" (2025):
  https://arxiv.org/html/2504.04660
- Cece & Marin, Cayley graphs of basic algebraic structures (DMTCS 2020):
  https://dmtcs.episciences.org/6287/pdf

**Status:** SgpDec is GAP (not Python). The holonomy decomposition algorithm
can be implemented in Python. The Cayley graph recognition test is simple
enough to implement directly.

---

## 4. Kan Extensions — The Master Construction

**What it does:** All limits, colimits, adjunctions, and data migrations are
special cases of Kan extensions. Given F: C -> D and G: C -> E:
- Left Kan extension Lan_G(F)(e) = colim over the comma category (G/e) of F
- Right Kan extension Ran_G(F)(e) = lim over the comma category (e/G) of F

Perrone & Tholen (2022) proved: pointwise left Kan extensions = partial colimit
evaluation. This gives a direct computational recipe.

**Spivak's functorial data migration:** A schema morphism F: C -> D induces
three adjoint data migration functors:
- Delta_F (pullback) — restrict data along F
- Sigma_F (left Kan extension) — freely extend data along F
- Pi_F (right Kan extension) — cofreely extend data along F

**Application to CTKG:** When transferring knowledge between domains (e.g.,
from arithmetic to a new domain), the Kan extension computes the "best
approximation." The predictor's back-off chain is essentially computing
successive Kan extension approximations.

**Key references:**
- Rydeheard & Burstall, *Computational Category Theory* (free PDF):
  https://www.cs.man.ac.uk/~david/categories/book/book.pdf
- Catlab.jl: https://github.com/AlgebraicJulia/Catlab.jl
- Patterson, computational category theory wiki:
  https://www.epatters.org/wiki/algebra/computational-category-theory.html
- GATlab (2024): https://arxiv.org/html/2404.04837v2
- Perrone & Tholen (2022): https://arxiv.org/pdf/2101.04531

**Status:** Needs implementation. Rydeheard-Burstall provides ML pseudocode.
The core algorithm is: for each target object e, compute the comma category,
then take the (co)limit of the restricted diagram.

---

## 5. Sheaf Laplacian Learning — Quantitative Consistency

**What it does:** Given signals on graph vertices, learns the sheaf structure
(restriction maps between adjacent stalks) by optimization. The sheaf
Laplacian generalizes the graph Laplacian; its spectrum encodes consistency.

**Key results:**
- Hansen & Ghrist (2019): spectral theory of cellular sheaves
- Bodnar et al. (2022): Neural Sheaf Diffusion — learns sheaf structure via
  gradient descent during GNN training
- Robinson, *Topological Signal Processing* — sheaf cohomology detects where
  local data doesn't glue to a consistent global section

**Application to CTKG:** Learn restriction maps between adjacent nodes from
observed activation patterns. Extends sheaf_check() from binary pass/fail to
quantitative consistency measure. Cohomology classes identify WHERE the data is
locally consistent but globally inconsistent — these are the interesting
structural boundaries.

**Key references:**
- Hansen & Ghrist, spectral theory of cellular sheaves:
  https://link.springer.com/article/10.1007/s41468-019-00038-7
- Bodnar et al., Neural Sheaf Diffusion (2022)
- Sheaf theory survey (2025): https://arxiv.org/html/2502.15476v1
- Copresheaf TNNs (NeurIPS 2025): https://arxiv.org/abs/2505.21251
- Hypergraph Neural Sheaf Diffusion (2025): https://arxiv.org/abs/2505.05702

**Status:** No external dependency needed — the sheaf Laplacian is a matrix
computation over the existing graph structure. Needs implementation.

---

## 6. NNO as Initial Algebra — Universal Property Test

**What it does:** The Natural Number Object is the initial algebra of the
endofunctor `X -> 1 + X`. "Initial" means: for ANY other algebra `(A, a: 1->A,
f: A->A)`, there exists a UNIQUE morphism `h: N -> A` such that:
- h(zero) = a
- h(succ(n)) = f(h(n))

**Discovery algorithm:** Search for `(z, s)` pairs in the graph where:
1. `z` is a node (candidate zero)
2. `s` is an endomorphism: a set of edges forming a function on a subset S
3. For every other `(a, f)` pair observed in the data, the unique extension
   property holds: there's a consistent mapping from S to the target that
   commutes with the diagram

This is testable from the graph without knowing what "digits" are. The NNO is
the `(z, s)` pair that satisfies the universal property for the most targets.

**Connection to Gavranovic et al. (ICML 2024):** Neural architecture = initial
algebra of an endofunctor. Discovering the NNO is the same problem as
discovering the right endofunctor — the one whose initial algebra generates the
observed structure.

**Key references:**
- Gavranovic et al., "Categorical Deep Learning" (ICML 2024):
  https://arxiv.org/abs/2402.15332
- Rydeheard & Burstall, Chapter 5 (initial algebras)
- nLab: https://ncatlab.org/nlab/show/natural+numbers+object

**Status:** Needs implementation. The universal property test is the key — it's
a search over candidate (z, s) pairs with a commutation check.

---

## Additional References

### Surveys and Frameworks
- Category-Theoretical Frameworks in ML survey (2025):
  https://www.mdpi.com/2075-1680/14/3/204
- Category Theory for AGI (AGI 2024):
  https://dl.acm.org/doi/10.1007/978-3-031-65572-2_13
- bgavran/Category_Theory_Machine_Learning (curated paper list):
  https://github.com/bgavran/Category_Theory_Machine_Learning

### Compositional AI
- Spivak & Shapiro, "Dynamic Operads, Dynamic Categories" (2023)
- Lambert & Patterson, "Cartesian double theories," Advances in Mathematics
  444 (2024)
- Cruttwell & Gavranovic, "Deep Learning with Parametric Lenses" (2024):
  https://arxiv.org/abs/2404.00408

### Knowledge Representation
- Categorical knowledge reasoning via pushouts (IEEE)
- Categorical framework for knowledge graphs (topos-theoretic, 2025)
- Algebraic Dynamical Systems in ML (Applied Categorical Structures, 2023)

---

## 7. Enriched Categories — The Unifying Framework

### The Problem with the Discrete Pipeline

The FCA → Krohn-Rhodes → NNO → Ruler pipeline works for discrete symbolic
domains (math, logic, language, game states). It fails for continuous physical
domains (robotics, control, sensor data) because:
- Krohn-Rhodes requires a well-defined transition monoid; physical transitions
  are stochastic and continuous
- NNO assumes discrete successor structure; joint angles are continuous
- Ruler requires exact equality; physical actions produce noisy results
- FCA requires binary incidence; sensor readings are continuous

But the project needs general intelligence — the same system must be both a
program analyzer and a robot controller.

### Enriched Category Theory

An ordinary category has hom-**sets** (morphisms form a set). Replace "set"
with any monoidal category V and you get a V-enriched category:

| Enrichment V | Hom-object | What it models |
|---|---|---|
| **Set** | {morphisms} | Discrete structure (standard CT) |
| **[0,∞]** (with +) | distance d(A,B) | Metric spaces (Lawvere 1973) |
| **[0,1]** (with ×) | probability/weight | Fuzzy/probabilistic relations |
| **Pos** | partial order | Order-enriched categories |
| **Vect** | vector space | Linear algebra |

**Key insight (Lawvere 1973):** A metric space IS a category enriched over
([0,∞], +, 0). Objects are points. The hom-object d(A,B) is the distance. The
triangle inequality d(A,C) ≤ d(A,B) + d(B,C) IS the composition law. Products,
limits, Kan extensions — the entire categorical toolkit applies without
discretization.

**The CTKG's edge weights already define an enrichment.** They are continuous
values in approximately [0,1]. This makes the CTKG a category enriched over
[0,1]. All categorical constructions work, and they continuously interpolate
between the discrete case (weights near 0/1) and the continuous case
(weights spread across the interval).

### Connection to Grid Cells and the Hippocampal Map

The brain already implements enriched category theory:

**Grid cells = the enrichment.** The entorhinal cortex provides a metric over
whatever space the hippocampus is indexing. Moser & Moser (Nobel 2014)
discovered grid cells for physical space. Constantinescu, O'Reilly & Behrens
(Science 2016) showed the same grid-like codes activate for **abstract
conceptual spaces** (bird morphology parameterized by neck/leg dimensions).
The entorhinal cortex doesn't know the domain — it provides hom-objects
(distances at multiple scales) over whatever the hippocampus indexes.

**Place cells = objects.** Each fires at a specific point in the metric space.
In abstract domains, "concept cells" (e.g., the Jennifer Aniston neuron) are
place cells for social-conceptual space.

**Hippocampal replay = enriched Kan extension.** Sleep replay computes the
best approximation of experience at novel points, weighted by distance in the
metric. This IS a colimit in the enriched category.

**Predictive coding = error in the enriched hom.** Prediction error is the
distance between predicted and actual activation — measured in the metric
defined by edge weights. The Hebbian update adjusts enriched hom-objects to
minimize this distance.

| Brain | CTKG | Enriched CT |
|---|---|---|
| Grid cells (multi-scale metric) | Edge weights | Enrichment over [0,1] |
| Place cells | Nodes | Objects |
| Hippocampal replay | Consolidation replay | Enriched Kan extension |
| Grid scale modules | Multi-threshold FCA | Filtration of the metric |
| Predictive coding error | Surprise | Distance in enriched hom |
| Spatial navigation | Spread activation | Composition |

### Multi-Scale Structure: Persistent Homology via FCA

The brain has grid cells at **multiple discrete scales** (~1.4x ratio between
adjacent modules — Stensola et al. 2012). Structure that persists across many
scales is real; structure at one scale only is noise.

Implementation: instead of binary FCA (token-in-context: yes/no), threshold
the continuous co-occurrence weights at multiple levels and run FCA at each:

```
threshold 0.1: coarse lattice (everything co-occurs)
threshold 0.3: moderate lattice (frequent co-occurrence)
threshold 0.5: fine lattice (strong co-occurrence)
threshold 0.7: very fine (very strong co-occurrence only)
threshold 0.9: ultra-fine (near-certain co-occurrence)
```

Concepts that appear in the lattice at ALL thresholds are **persistent** —
these are the robust, domain-general structures. This IS persistent homology
applied to the co-occurrence metric. The Betti numbers at each threshold tell
you how many independent "holes" (= independent compositional structures)
exist at each scale.

For math tokens: weights are near-binary → one dominant scale → standard FCA.
For robot sensors: weights are continuously distributed → multi-scale structure
→ persistent homology reveals geometry at multiple precisions.

Same code. Same algorithm. The enrichment handles both.

### How Each Algorithm Generalizes

**FCA → Quantitative Concept Analysis:** Binary incidence (token ∈ context)
becomes weighted incidence (co-occurrence strength). Threshold at multiple
levels for persistent structure. The Galois connection becomes a
**quantitative Galois connection** (Bělohlávek 2002, fuzzy concept lattices).

**Krohn-Rhodes → Enriched Semigroup Decomposition:** The transition monoid
becomes a **weighted** transition system. Instead of "applying generator g
maps state s to state t," we have "applying g from s reaches t with weight w."
The decomposition still applies — the holonomy groups become weighted/graded.
For continuous domains, contractive maps replace bijections.

**NNO → Initial Algebra of Enriched Endofunctor:** The successor endofunctor
X → 1+X becomes a **contractive endofunctor** in the enriched setting. The
initial algebra exists by Banach's fixed point theorem (= enriched initiality).
For the robot, this means: find endomorphisms whose repeated application
converges — these are the "successor-like" structures in continuous space.

**Kan Extensions → Enriched Kan Extensions:** The formula is the same:
Lan_G(F)(e) = colim over (G/e) of F. In the enriched setting, the colimit
is a **weighted colimit** — nearby points contribute more than distant ones.
This is exactly what hippocampal replay does: reconstruct experience at a
novel point by weighting known experiences by proximity.

### Key References

- Lawvere, "Metric spaces, generalized logic, and closed categories" (1973).
  The foundational paper: metric spaces as enriched categories.
  Reprinted: TAC Reprints No. 1.
- Constantinescu, O'Reilly & Behrens, "Organizing conceptual knowledge in
  humans with a gridlike code" Science 352:1464-1468 (2016).
- Stensola et al., "The entorhinal grid map is discretized" Nature 492 (2012).
- Bělohlávek, "Fuzzy Galois connections" Mathematical Logic Quarterly 45(4)
  (1999). FCA generalized to [0,1]-valued incidence.
- Kelly, "Basic Concepts of Enriched Category Theory" (1982). The standard
  reference. Reprinted: TAC Reprints No. 10.
  http://www.tac.mta.ca/tac/reprints/articles/10/tr10abs.html
- Leinster, "The magnitude of metric spaces" Documenta Mathematica 18 (2013).
  Size of enriched categories; connects to biodiversity indices.

---

## Implementation Plan

### Phase 0: Multi-Scale Metric (foundation)
- Edge weights already define the enrichment — no new data structure needed
- Implement multi-threshold FCA: run FCA at thresholds [0.1, 0.3, 0.5, 0.7, 0.9]
- Track which concepts persist across thresholds (persistent structure)
- This replaces the missing "grid scale modules" in the brain-CTKG mapping

### Phase 1: Quantitative FCA (immediate)
- Build weighted incidence matrix from KnowledgeGraph co-occurrence data
- Run FCA at multiple thresholds [0.1, 0.3, 0.5, 0.7, 0.9]
- Track persistent concepts (appear at all thresholds)
- Extract products (meets), coproducts (joins), type hierarchy
- Store as categorical structure in the graph

### Phase 2: Algebraic Skeleton (Krohn-Rhodes lite)
- Extract transition monoid from TRANSITION edges
- For continuous domains: identify contractive endomorphisms
- Identify cyclic subgroups (potential NNOs) and group components
- The enriched version: weight transitions, decompose weighted semigroup

### Phase 3: Universal Property Search
- Implement enriched initial algebra test for endofunctors
- For discrete: find (z, s) pairs satisfying NNO universal property
- For continuous: find contractive endomorphisms (Banach fixed point)
- Discover NNO, free monoids, and other universal constructions

### Phase 4: Enriched Kan Extensions
- Implement weighted colimit computation (left Kan extension)
- Nearby points contribute more than distant ones (metric weighting)
- Use as the principled replacement for the predictor's back-off chain
- This IS hippocampal replay in categorical form

### Phase 5: Sheaf Laplacian
- Compute sheaf Laplacian from activation patterns
- Learn restriction maps from observed co-variation
- Identify cohomological inconsistencies (persistent boundary structures)
- Use as quantitative consistency measure

### Phase 6: Equational Theory Discovery (post-structure)
- Only after Phases 1-3 have discovered types and operations
- Implement QuickSpec-style enumerate-evaluate-group
- Discover algebraic laws from observed compositions
- Requires the evaluator that earlier phases provide

---

## Installed Libraries

No external dependencies required. FCA is implemented natively in
`ctkg/logic/fca.py` via bottom-up hierarchical lattice construction.
