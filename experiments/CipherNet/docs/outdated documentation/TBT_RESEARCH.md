# Thousand Brains Theory — Research for CipherNet

## Question 1: What coordinate systems can the neocortex use?

### What biology shows

Grid cells form **toroidal manifolds** — confirmed by persistent
homology applied to entorhinal cortex recordings (Gardner et al.,
[Nature 2022](https://www.nature.com/articles/s41586-021-04268-7)).
The toroidal topology appears BEFORE the animal navigates, suggesting
it's a structural prior (Trettel et al.,
[bioRxiv 2026](https://www.biorxiv.org/content/10.64898/2026.03.10.710908v1.full)).

Grid cell firing fields form a **hexagonal lattice** — the most
efficient packing of periodic points on a 2D surface. This implies
the neural space is a **conformally isometric embedding** of physical
space: local displacements in neural space are proportional to local
displacements in physical space.

The brain uses at least three types of reference frames simultaneously:
- **Egocentric**: centered on the body (where is it relative to me?)
- **Allocentric**: centered on the environment (where is it in the room?)
- **Object-centric**: centered on the object (where on the cup is my finger?)

[Allocentric and egocentric representations COEXIST](https://www.nature.com/articles/s41467-024-54699-9)
in medial entorhinal cortex, with deeper layers encoding egocentric
bearing and distance to landmarks. Conversion between frames is
handled by retrosplenial cortex.

### What TBT implements

The current implementation uses **3D Cartesian space** for reference
frames. Object models are **explicit 3D graphs** where:
- Nodes store: position (3D), orientation (3x3 orthonormal basis from
  surface differential geometry), and features (color, texture, curvature)
- Edges store: displacement vectors between connected nodes

All models have an inductive bias toward 3D space + time.

However, the TBT paper acknowledges: "the exact structure of space can
potentially be learned, such that the lower-dimensional space of a melody,
or the abstract space of a family tree, can be represented."

Grid cells are mentioned as the future representation, replacing
explicit Cartesian coordinates.

### What the neocortex CANNOT represent (stated limitations)

- Not suited for: "multiplying arbitrary large numbers, or predicting
  the structure of a protein given its genetic sequence" — these need
  computation, not spatial models
- The Cartesian 3D graph has fixed dimensionality — can't natively
  represent higher-dimensional spaces
- The hexagonal grid structure implies the native coordinate system
  is 2D (torus) at each module level — 3D is built by combining
  multiple 2D modules
- Grid cells have NO ORIGIN — positions are relative, not absolute.
  This means the coordinate system is inherently relational.

### Constraints for CipherNet

1. The coordinate system is **periodic** (toroidal, not Euclidean)
2. The native dimensionality is **2D per module**, higher dimensions
   built by combining modules
3. There is **no origin** — everything is relative displacement
4. The metric is **conformal** — local distances are preserved but
   global structure can warp
5. Both **egocentric** (body-relative) and **allocentric** (world-relative)
   frames coexist and can be converted between
6. **Abstract spaces** use the same machinery as physical space

## Question 2: How does a column go from observations to a manifold?

### The TBT learning algorithm, step by step

**What a column stores:**

An object model is a GRAPH of (feature, location) bindings.
For a coffee cup: node 1 = (brown color, smooth texture, low curvature)
at position (3, 5, 2) in the cup's reference frame. Node 2 = (brown,
smooth, high curvature) at position (3, 5, 4). Edge between them
stores the displacement (0, 0, 2).

The model is NOT a surface equation. It's a discrete set of observed
points with features, connected by displacements. The "manifold" is
the graph itself — the set of all (feature, location) pairs that have
been observed or can be interpolated.

**Step 1: First observation**

The column receives: "I sense smooth brown surface with curvature 0.5."
It doesn't know what object this is. Every stored object is a candidate.
For each candidate, for each location on that candidate that has a
similar feature, the column creates a HYPOTHESIS: "I'm touching object X
at location Y in orientation Z."

Evidence is assigned: morphological match (point normal, curvature)
in the range [-1, +1]. Feature match (color, texture) in [0, +1].

**Step 2: Movement and prediction**

The column knows its sensor moved (from motor efference copy).
The displacement is in BODY coordinates (egocentric). For each
hypothesis, the column rotates the displacement into the OBJECT's
reference frame: "if I'm at location Y on object X in orientation Z,
and I moved by vector D in body frame, then I should now be at
location Y' = Y + R(Z) * D on the object."

**Step 3: Prediction testing**

At the new location Y', the column predicts what features it should
sense. It checks: does the model have a feature stored at or near Y'?
If yes, does it match what I actually sense?

- Match: evidence for this hypothesis INCREASES
- Mismatch (morphology): evidence DECREASES (could be the wrong object)
- Mismatch (features only): evidence stays neutral (could be same object,
  different color)

**Step 4: Evidence accumulation**

Evidence accumulates across many sense-move-sense cycles. Hypotheses
that consistently predict correctly accumulate high evidence. Those
that fail get low evidence. Eventually, one hypothesis dominates:
"I'm touching the coffee cup, my finger is on the handle."

**Step 5: Learning (adding to the model)**

If the system recognizes the object but encounters a location it hasn't
stored before, it ADDS the new (feature, location) pair to the graph.
The model grows through exploration — each new observation that doesn't
match an existing model point creates a new node and connects it to
the previous location with a displacement edge.

**Step 6: Novel objects**

If NO hypothesis accumulates sufficient evidence (nothing matches),
the system creates a NEW object model and starts populating it with
the current observations. The first observation becomes the first
node; subsequent observations become new nodes connected by
displacement edges.

### How this relates to learning addition

The coffee cup and addition are structurally identical:

| Cup | Addition |
|-----|----------|
| Feature: brown, smooth, curvature=0.5 | Feature: digit pair (3, 4) |
| Location: (3, 5, 2) on the cup | Location: position on the number line |
| Displacement: move finger 2cm right | Displacement: the operation (+) takes you from inputs to output |
| Model: graph of (feature, location) pairs | Model: graph of (input pair, result) triples |
| Prediction: "at location X, I should feel Y" | Prediction: "given inputs 3 and 4, the result should be 7" |
| Error: felt something unexpected | Error: predicted the wrong sum |
| Learning: add new node to graph | Learning: add new example, update displacement pattern |

The column doesn't know it's learning a cup or learning addition.
It does the same thing: sense features at locations, predict what
features to expect after displacement, update evidence based on errors,
and grow the model by adding new (feature, location) bindings.

### What this means for "fitting a manifold"

The TBT column does NOT fit a manifold. It builds a GRAPH of specific
observations connected by displacements. The "manifold" is the graph
itself — a discrete approximation that gets denser with more observations.

Prediction at novel locations happens via DISPLACEMENT: "I know the
feature at location A. If I displace by D, I predict the feature at
location A + D." This doesn't require a surface equation. It requires
that displacements are CONSISTENT — the same displacement always leads
to the same feature change.

For addition: "starting at 3, displacing by 4, I arrive at 7" is one
observation. "Starting at 8, displacing by 7, I arrive at 15" is
another. If the column discovers that the displacement rule is
consistent (displacement = b, from any starting point a, arrives at
a + b), it can predict from ANY starting point. The rule IS the
consistency of displacements.

This is what our invariant extraction found: distance(a, c) co-terminates
with distance(0, b). That IS the displacement consistency rule.

## Question 3: Navigation, displacement, and knowing when to stop

### How TBT navigation works

The TBT system does NOT walk and count steps. It tracks its current
location relative to a goal and checks whether it has arrived.

Ref: [Thousand Brains Project 2024, arXiv](https://arxiv.org/html/2412.18354v1)

**The process:**

1. **Goal state**: a target pose/location stored in the system.
   "I want to be at position 7." This is a feature-at-location
   specification in the model's reference frame.

2. **Current state**: the believed current location, continuously
   updated by integrating displacements. "I am at position 3."

3. **Each movement**: apply a displacement, update believed location.
   The system "calculates a pose displacement" between consecutive
   observations — "the difference between the current location and
   the previous location relative to the body."

4. **Update hypotheses**: for each hypothesis about where the sensor
   is, rotate the displacement into the object's reference frame
   and predict the new location.

5. **Arrival check**: compare current believed location with goal.
   Match → stop. No match → continue moving.

6. **Terminal condition**: "recognizing the object or exceeding the
   maximum number of permitted steps." Recognition occurs when
   evidence thresholds are satisfied.

### How this applies to arithmetic on the number line

For addition (3 + 4 = ?):
1. Start at position 3 (current location on the number line)
2. Goal: apply displacement 4 — "where do I end up?"
3. Move along successor edges, tracking displacement
4. After each step, check: has my displacement reached 4?
5. When displacement = 4, read current position → node 7 → answer

**How the system knows displacement has reached 4:**
Co-termination — two simultaneous processes:
- REFERENCE walk: starts at node 0, follows successor toward node 4
- COMPUTATION walk: starts at node 3, follows successor
Both advance in lockstep (one step per timestep). When the reference
arrives at node 4, the computation's current position IS the answer.

This is graph-native: both walks are activation waves propagating
along edges. The graph's step() function advances both simultaneously.
No counting. No Python arithmetic. Just wave propagation and
coincidence detection.

### Dual-wave mechanism for graph-native arithmetic

The reference walk and computation walk are two simultaneous
activation waves on the number line. They must NOT interfere.

Biological solution: they use different neurons. In the cortex,
different information streams use different LAYERS within the
same column. Layer 2/3 might carry one signal while Layer 5
carries another.

For CipherNet: each number line node could have multiple
activation channels, OR we use two parallel number line subgraphs
(reference line and computation line) connected so they advance
in lockstep but carry independent signals.

### Implications for CipherNet

1. **Numbers as positions on a graph**: 0-9 (or 0-20) as explicit
   nodes connected by successor edges. Within ANS range.

2. **Addition as co-terminating walks**: two activation waves,
   one reference (0 → b), one computation (a → c). Graph-native.

3. **Large numbers as sequential digit processing**: 307 is '3','0','7'
   processed one digit at a time by the PFC. Same mechanism that
   chains 4+5+1+9. The brain doesn't represent 307 as a magnitude —
   it represents it as a symbol sequence plus a learned place value rule.

4. **Arrival detection**: the reference wave hitting its target node
   triggers a "done" signal. This is coincidence detection — when
   the reference node b's activation exceeds a threshold, the
   computation is complete.

5. **Goal representation**: the target node (b for reference,
   the unknown c for computation) is held in PFC WM. The PFC sets
   the goal, the number line executes the walk, the thalamus gates
   the result back to PFC.

## Question 4: Column communication and PFC coordination

### Column-to-column communication

Each cortical column has its own model. Columns that sense the same
object from different perspectives VOTE on the object's identity.
The voting happens via long-range lateral connections between columns
in Layer 2/3. Columns whose hypotheses agree reinforce each other;
those that disagree compete.

This is consensus formation: thousands of semi-independent models
converge on a shared interpretation. The mechanism is similar to
belief propagation in graphical models.

### PFC coordination

The PFC controls which columns are active, what they attend to,
and how results are sequenced. See PFC_PLAN.md and PFC_RESEARCH.md.

## Implications for CipherNet

### What types of hypotheses do we need?

The TBT answer: we don't need hypotheses about manifold SHAPE (plane,
saddle, etc.). We need hypotheses about DISPLACEMENT CONSISTENCY.

The column asks: "is the displacement pattern consistent?" Not:
"is the data on a plane?"

For addition: "displacement b from position a arrives at a + b" —
consistent for all a. ONE displacement rule.

For multiplication: "displacement b scaled by a arrives at a * b" —
consistent for all (a, b). A DIFFERENT displacement rule.

The question becomes: what is the vocabulary of displacement rules?

### Possible displacement rules

1. **Translation**: displace by b → arrive at a + b. (Addition.)
2. **Scaling**: scale a by factor b → arrive at a * b. (Multiplication.)
3. **Rotation**: rotate by angle b → arrive at rotated position. (Geometric.)
4. **Composition**: apply rule R1, then rule R2. (Chaining.)
5. **Projection**: project onto a subspace. (Dimensional reduction.)

These are the ISOMETRIES and SIMILARITIES of the reference frame
geometry. In Euclidean 3D: translations, rotations, reflections,
scaling. In toroidal geometry: modular shifts.

### Is there a finite set of hypothesis types?

Possibly. The isometries of a given geometry form a GROUP, which is
finite-dimensional. For Euclidean 3D: the isometry group is SE(3) —
translations (3D) + rotations (3D) = 6 parameters. Adding scaling
gives the similarity group Sim(3) = 7 parameters.

If the neocortex's native geometry is toroidal (as grid cells suggest),
the isometry group of a torus is SMALLER than Euclidean: translations
along the two cycle directions + discrete rotational symmetries.

**Hypothesis**: the set of displacement rules the neocortex can learn
is exactly the isometry group of its grid cell geometry. This is a
finite-dimensional Lie group. Fitting a "manifold" reduces to
identifying which group element (which displacement rule) is consistent
with the observations.

If true, this means:
- There IS a small, fixed number of hypothesis types
- They correspond to the symmetries of the toroidal grid
- Learning = identifying which symmetry the data exhibits
- This is a GROUP THEORY problem, not a regression problem

### Open questions

1. Is the isometry group of the toroidal grid sufficient to capture
   all arithmetic operations? Addition = translation, multiplication
   = scaling. What about exponentiation, logarithms, roots?

2. Can the column discover NEW displacement types that aren't in the
   pre-existing isometry group? Or is the group fixed by the geometry?

3. How does the column distinguish "this is addition" from "this is
   multiplication" from the same set of (a, b, c) observations?
   By checking which displacement rule (translation vs scaling) is
   consistent.

4. For non-arithmetic domains (chemistry, physics, language), what
   are the "displacements"? Are chemical reactions displacements in
   molecular space? Are grammatical transformations displacements in
   syntactic space?

## Summary: How TBT Solves Our Problems

| Problem | TBT Solution | CipherNet Implementation |
|---------|-------------|--------------------------|
| How to represent numbers | Positions on a spatial graph (like positions on an object) | Number line: 0-9 as nodes with successor edges |
| How to do addition | Displacement (walk along the number line, like moving a finger along a cup) | Co-terminating activation waves |
| How to know when to stop | Goal-state matching (compare current position to target) | Reference wave arriving at target triggers "done" |
| How to handle large numbers | Sequential symbol processing (like exploring an object one touch at a time) | PFC sequences through digits, each within ANS range |
| How to chain operations | PFC holds intermediate results, sequences operations (like planning a series of movements) | WM stripes + BG gating + thalamic routing |
| How operations compose | Displacement composition (movement A then movement B) | Sequential walks on the same number line |
| How to learn when to gate | Dopamine reward prediction error trains BG Go/NoGo | Edge weight learning on BG D1/D2 edges |

The central principle: **intelligence is navigation in reference frames.**
Recognizing a cup is navigating the cup's shape. Doing arithmetic is
navigating the number line. Planning is navigating a task space. The
same cortical column mechanism — sense features at locations, predict
what comes next, track displacement — handles all of it.

## Key references

- [Gardner et al. 2022 — Toroidal topology of grid cell activity](https://www.nature.com/articles/s41586-021-04268-7)
- [Trettel et al. 2026 — Toroidal topology precedes navigation during development](https://www.biorxiv.org/content/10.64898/2026.03.10.710908v1.full)
- [Allocentric + egocentric coexist in entorhinal cortex, Nature Comms 2024](https://www.nature.com/articles/s41467-024-54699-9)
- [Grid-like entorhinal representation of abstract value space, Nature Comms 2024](https://www.nature.com/articles/s41467-024-45127-z)
- [Hawkins et al. 2019 — Grid cells in the neocortex framework, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6336927/)
- [Lewis et al. 2019 — Locations in neocortex via cortical grid cells, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6491744/)
- [Thousand Brains Project 2024 paper, arXiv](https://arxiv.org/html/2412.18354v1)
- [Thousand Brains Project documentation](https://thousandbrainsproject.readme.io/)
- [Predictive coding approximates backprop via Hebbian plasticity, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5467749/)
