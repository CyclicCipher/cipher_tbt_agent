# CipherNet — Variational Manifold Learning on Graphs

## Core thesis

True learning is discovering a lower-dimensional manifold on which
the data lies within a higher-dimensional space.

Every rule is a manifold. Every fact is a point on that manifold.
Prediction is projection onto the manifold. Uncertainty is distance
from the manifold. Generalization is the manifold extending beyond
the observed points.

## Architecture

The fundamental data structure is a **graph** that serves as a
learned latent space, analogous to grid cells in cortical columns.

```
Stimulus/Data ←→ Latent Space Graph ←→ Manifolds
  (sensory)       (grid cell analog)     (learned rules)
```

### The graph

Nodes represent positions in the latent space. Two types of edges:

- **Spatial edges** (A ↔ B): undirected. Encode metric structure.
  "A and B are adjacent in the latent space." These form the
  coordinate system. Analogous to grid cell connectivity.

- **Temporal edges** (A → B): directed. Encode transitions and
  causality. "A precedes B" or "A causes B." Analogous to place
  cell sequences during theta cycles.

Because graphs can implicitly represent any metric, any dimensionality,
and any topology, this single structure can build ANY latent space:
a 1D number line, a 2D plane, a high-dimensional protein conformation
space, a discrete logical space, or a space with holes and curvature.

### Connecting data to the graph

When the system observes a stimulus (e.g., the symbol "3" or the
sentence "3 + 4 = 7"), it connects the stimulus to positions in the
latent space graph. For arithmetic:

- "3" connects to position 3 on the number line subgraph
- "+" connects to the addition manifold
- "7" connects to position 7
- The triple (3, 4, 7) is a point in the 3D space formed by three
  number-line axes

These connections are the raw examples. The system accumulates them.

### Manifold discovery

As examples accumulate, the system dissolves them into a manifold —
a continuous surface that generalizes from the specific points to
ALL valid points.

For addition: after 3 non-collinear examples, the system fits a plane
z = ax + by + c through the points. The plane IS the rule of addition.
New predictions: project (x, y) onto the plane, read z.

For probabilistic domains: the manifold is a probability distribution
(a cloud), not a sharp surface. Valid points have high probability;
invalid points have low probability. The manifold is the
high-probability region.

Formally: the manifold is a probability density p(x, y, z) that
concentrates on the rule surface. For deterministic rules like
addition, the density is a delta function on the plane. For noisy
or uncertain rules, the density is a distribution around the surface.
This is the "variational" part — the manifold is learned as a
variational distribution that best explains the data with minimum
complexity.

### Inference

Given partial information (e.g., x=3, y=4, z=?), constrain to
the manifold and read off the answer. No separate forward/inverse
operations. The manifold is queried from any direction.

For deterministic manifolds: project the known values onto the surface,
read the unknown. For probabilistic manifolds: condition the
distribution on the known values, sample or take the mode of the
conditional for the unknown.

Uncertainties in inference:
- Multiple manifolds may be compatible with the known values
  (ambiguity → return a distribution, not a point)
- The manifold may not be fully determined yet (insufficient examples
  → wider distribution)
- The query point may be far from any training example
  (extrapolation → lower confidence)

## Design principles

### 1. Local subgraph training

When learning a specific problem (e.g., addition), create a new
tiny subgraph in the latent space. Train it locally — only the
subgraph's edges and manifold parameters update. This is
computationally cheap.

Later, integrate the subgraph into the main graph by connecting
it to related subgraphs. Global training passes (propagating
signals through the entire graph) happen less frequently, for
cross-domain integration.

This mirrors the brain: hippocampus learns fast (local), cortex
consolidates slow (global).

### 2. Structured priors

For problems with known latent space structure (like arithmetic),
we can directly construct the subgraph:
- Number line: 0 — 1 — 2 — 3 — ... (spatial edges)
- Successor: 0 → 1 → 2 → 3 → ... (temporal edges)
- 3D addition space: three copies of the number line as axes

This is not cheating — it's providing the coordinate system.
The MANIFOLD (z = x + y) still must be learned from examples.
The prior just says "the space has this shape." The rule still
has to be discovered.

For domains without known structure, the graph starts empty
and grows from observations (like BDH's emergent topology).

### 3. Categorical structure within and across domains

Manifold learning discovers the geometric rules within a domain.
Categorical structure exists at BOTH levels:

**Within a domain:**
- Composition: the manifold for "add then multiply" is the composition
  of the addition and multiplication manifolds
- Adjunctions: the sum manifold and the difference manifold are
  adjoint — they're the same surface queried from opposite sides
- Initial algebras: the number line itself is an initial algebra
  (zero + successor generates all positions)
- Pullbacks: a sudoku cell satisfying row AND column AND box
  constraints is the pullback of three manifolds

**Across domains:**
- Functors: "addition in base-10 and addition in base-2 are the same
  operation" → functor between two number-line graphs
- Natural transformations: "every commutative group has the same
  abstract structure" → same manifold shape in different spaces

The relationship: manifold learning discovers the GEOMETRY (the shape
of rules). Category theory discovers the ALGEBRA (how rules compose,
invert, and transfer). Both are needed within a single domain.
Category theory also operates across domains at the meta-level.

### 4. Probabilistic by default

Every manifold is a probability distribution. Deterministic rules
are the special case where the distribution is concentrated
(variance → 0). This means:

- Noisy data is handled naturally (wider distribution)
- Uncertainty is quantified (distance from manifold center)
- Conflicting evidence is represented (multimodal distribution)
- Confidence increases with more examples (distribution sharpens)
- Unknown regions have high entropy (honest about ignorance)

### 5. Thousand Brains Theory (Jeff Hawkins)

The architecture is inspired by Hawkins' Thousand Brains Theory:

- Each cortical column learns a model of a complete object
  → Each subgraph learns a complete manifold for one rule/domain
- Columns vote on the identity of the object
  → Subgraphs vote on the interpretation of the stimulus
- Each column has the same algorithm
  → Each subgraph uses the same manifold-learning algorithm
- Reference frames (grid cells) provide the coordinate system
  → The graph's spatial edges provide the reference frame
- Objects are learned as compositions of features at locations
  → Rules are learned as manifolds in the latent space

## Relationship to prior work

### What CipherNet replaces

- **CTKG (symbolic_ai_v2)**: The CTKG tried to discover all structure
  categorically (FCA, Krohn-Rhodes, initial algebra). This works for
  discrete symbolic structure but fails for continuous/geometric rules.
  CipherNet uses manifold learning for geometric structure and categorical
  learning for algebraic structure — both within and across domains.
  The CTKG's tools (FCA, algebra, sheaf) operate on discovered manifolds
  to find how they compose, invert, and relate.

- **BDH benchmarks**: The BDH experiments showed that a neural architecture
  can learn structured tasks but doesn't tell us HOW the rules are
  represented. CipherNet makes the representation explicit: rules are
  manifolds in a graph-structured latent space. BDH's code is a source
  of ideas for sparse computation and parallelization.

- **CatPlan**: The planning language and search algorithms carry forward.
  CatPlan's planner operates on the categorical structure that CipherNet's
  meta-level discovers. The planner doesn't change — the representation
  of what it plans over does.

### What carries forward

| Component | From | Status |
|-----------|------|--------|
| CatPlan language + parser | Baby Dragon Hatchling | Keep as-is |
| CatPlan planner (A*, FF, CG heuristics) | Baby Dragon Hatchling | Keep as-is |
| Domain classifier | Baby Dragon Hatchling | Adapt for manifold signatures |
| Heuristic selector | Baby Dragon Hatchling | Keep as-is |
| World simulators (chemistry, physics, protein) | Baby Dragon Hatchling | Keep as-is |
| FCA, Krohn-Rhodes, sheaf, NT discovery | symbolic_ai_v2 | Elevate to meta-level |
| Invariant extraction | Baby Dragon Hatchling | Replace with manifold fitting |
| Spatial rule discovery | Baby Dragon Hatchling | Replace with manifold fitting |
| Grid cell multi-scale representation | Baby Dragon Hatchling | Integrate into graph construction |

## Implementation plan

### Phase 1: Core graph + manifold fitting

1. Graph data structure: nodes, spatial edges, temporal edges
2. Subgraph creation and integration
3. Manifold fitting: given points in the graph, fit the simplest
   manifold (plane, then curved surface, then general)
4. Probabilistic manifold: represent as a distribution, not a surface
5. Inference: condition on known values, query unknowns

Validate on: addition (3 examples → plane → generalize to all pairs).

### Phase 2: Multi-domain learning

1. Structured priors for arithmetic (number line subgraph)
2. Learn addition, multiplication, exponentiation as manifolds
3. Unstructured learning for chemistry, physics (build graph from data)
4. Domain classifier using manifold signatures

### Phase 3: Categorical meta-level

1. Functor discovery between domain subgraphs
2. Adjunction discovery (inverse operations)
3. Natural transformation discovery (shared structure)
4. Cross-domain planning via CatPlan

### Phase 4: Scaling and integration

1. Local subgraph training (fast, cheap)
2. Global consolidation passes (slow, expensive)
3. BDH-inspired sparse computation for large graphs
4. CatPlan planner operating on learned domains

## Theoretical foundations

- **Manifold hypothesis** (Bengio et al.): real-world data lies on
  low-dimensional manifolds in high-dimensional spaces.
- **Variational inference** (Blei et al.): approximate complex
  posteriors with simpler distributions.
- **Thousand Brains Theory** (Hawkins): each cortical column learns
  a complete model using grid cell reference frames.
- **Enriched category theory** (Lawvere): manifolds with metrics
  are enriched categories. The graph IS an enriched category.
- **Free energy principle** (Friston): learning minimizes the
  divergence between the internal model (manifold) and observations.

## Open questions

1. What is the right manifold fitting algorithm for the graph setting?
   Standard manifold learning (UMAP, Isomap) assumes Euclidean ambient
   space. We need fitting on graphs.

2. How does the probabilistic manifold interact with the categorical
   meta-level? A functor between probabilistic manifolds is a
   Markov kernel — this connects to the Markov category work in
   the CTKG.

3. How many examples are needed for different types of rules?
   Addition: 3 (plane). Multiplication: ? (curved surface). General
   polynomial: degree + 1 points per dimension?

4. Can the system discover when it needs a new dimension in the
   latent space? (e.g., multiplication requires the 2D product space,
   not just the 1D number line)

5. How does local subgraph training interact with global consistency?
   The sheaf Laplacian measures local-global consistency — this
   is the right tool, but applied to manifolds instead of boolean
   predicates.
