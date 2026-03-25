# CTKG Architecture

## The Goal

A single symbolic AGI that learns, reasons, and acts in any environment —
physics problems, mathematics, natural language, video games — using one
unified architecture with no environment-specific code and no hardcoded domain
knowledge.

## The Core Insight

The human neocortex is a spiking neural network that, when zoomed out, acts
like an unlabeled knowledge graph. Neurons are nodes. Synapses are edges.
Activation is working memory. Hebbian learning is edge creation. Decay is
forgetting. Sensory input fires neurons directly — there is no separate
"observation format" that gets "converted to" the graph.

The CTKG is this graph. There is one graph. It is the knowledge, the workspace,
and the processor. When the environment produces a token, that token IS a node
that activates. When two tokens co-occur, the edge between them strengthens.
When a token follows an action, that is a directed transition edge.

## The Core Files

```
ctkg/logic/
├── graph.py           — the one graph: nodes, edges, activation, spread, learn
├── loop.py            — AgenticLoop: observe → spread → learn → act
├── hippocampus.py     — episodic memory: activation snapshots + observation records
└── consolidation.py   — the slow path: replay, prune, structure discovery

ctkg/connectors/
└── tokenizer.py       — character-level tokenizer at the environment boundary
```

## Nodes and Edges

Every node carries:
- `activation: float` — current firing level, [0, 1]
- `resting: float` — baseline importance (grows with exposure)
- `preferred: float` — homeostatic target (active inference prior)

Every edge carries:
- `alpha, beta: float` — Beta posterior counts (evidence for/against)
- `weight: float` — computed: (alpha - beta) / (alpha + beta), range [-1, 1]
- `confidence: float` — computed: alpha + beta (total evidence)
- `role: int` — COOCCURRENCE (0) or TRANSITION (1)
- `sigma: float` — transient context-dependent state (reset each timestep)
- `dist_sum, dist_sq_sum, dist_count` — PoPE positional statistics

Working memory = {nodes where activation > threshold}. There is no separate
WorkingMemory object. Activation levels on KG nodes ARE working memory.

Two edge roles:
- **Co-occurrence** — A and B appeared in the same observation. **Forward-only**
  (causal masking): earlier tokens predict later tokens, not the reverse. This
  prevents backward associations that corrupt attention (e.g., training
  `[5, succ, 6]` should NOT create `6→5`).
- **Transition** — A preceded B across a timestep boundary (an action happened
  between them). Directed. Used for temporal prediction (spread), NOT for action
  selection.

The system discovers which tokens are intero vs extero vs action from the
edge structure, not from labels.

### Excitation and Inhibition

Edge weights can be positive (excitatory) or negative (inhibitory). Spread
handles them differently, following cortical circuit architecture:

**Excitatory edges are normalised per source.** From each active source node,
outgoing positive-weight edges form a probability distribution (stochastic
kernel): weights normalised to sum to 1. This is winner-take-all competition
— excitatory neurons compete with each other for which target to activate,
just as pyramidal cell populations compete via shared inhibition in cortical
WTA circuits.

**Inhibitory edges set a threshold.** Negative-weight edges contribute their
raw weight as suppression — they are NOT normalised. This mirrors the role of
parvalbumin basket cells in cortex: they target the soma of pyramidal cells
and set a firing threshold that excitation must exceed. The inhibition is
subtractive (raising the bar), not competitive (not winner-take-all among
inhibitors). A target node activates only if total excitation from all
sources exceeds total inhibition from all sources.

This asymmetry — normalised excitation, raw inhibition — prevents noise
amplification. A tiny inhibitory edge (-0.03) contributes only -0.03 of
suppression, not an amplified -0.5 that would result from normalisation.
Strong inhibitory edges (-0.9) provide strong suppression. The strength is
proportional to actual evidence, not an artifact of being the only negative
edge from a source.

### PoPE: Positional Encoding on Edges

Each edge tracks the distribution of relative positions at which its endpoints
co-occur (Gopalakrishnan et al. 2025). This decouples "what" (content
association strength) from "where" (typical positional offset). Two tokens with
strong content affinity maintain high association regardless of position. Two
tokens that happen to be adjacent but have no meaningful relationship have low
content weight despite high positional proximity.

Currently: positional statistics are recorded on each co-occurrence edge
(observe_distance). Not yet used in action selection because the answer appears
in a separate observation, making cross-observation position matching invalid.
Will be used for within-observation attention when needed.

## The Single Process

Induction, deduction, and abduction are not three modules. They are three
descriptions of one process: **spread, compare, update.**

Each timestep:

```
1. DECAY      — all node activations *= decay_factor
2. INPUT      — observed tokens activate their nodes (via tokenizer)
3. EDGES      — co-occurrence edges between same-observation tokens strengthen
                 (causal masking: forward-only, equal strength)
               — transition edges from previous timestep's active nodes strengthen
4. LEARN      — compare the PREVIOUS timestep's spread prediction against
                THIS timestep's actual tokens:
                  confirmed → strengthen edge (induction)
                  wrong     → weaken edge (revision)
                  surprise  → create edge (abduction)
5. SPREAD     — active nodes propagate activation along weighted TRANSITION
                edges → produces prediction for the NEXT timestep
6. STORE      — Hippocampus records activation snapshot + observation tokens
```

One function. Three names for its three update cases:
- **Deduction**: spread predicts from known edges.
- **Induction**: confirmed predictions strengthen edges.
- **Abduction**: surprising observations create new edges.

## Action Selection: Dot-Product Attention

The co-occurrence edge weight matrix IS the attention matrix. For each
candidate action c given context tokens {k₁, k₂, ...}:

```
logit(c) = Σᵢ activation(kᵢ) × normalised_cooccur(kᵢ, c)
```

where normalised_cooccur(k, c) = cooccur(k, c) / Σⱼ cooccur(k, cⱼ).

Each context token's co-occurrence weights to candidates are normalised to
sum to 1. This is the "projection" that makes attention discriminating:

- succ → {0,1,...,9} all equal → each gets 0.1 → non-discriminating
- 6 → {7:0.9, others:0.01} → 7 gets ~0.9 → highly discriminating

The dot product Q·K_norm automatically amplifies discriminating context
tokens and attenuates non-discriminating ones.

**No transition edges in action selection.** Transition edges carry noise
from wrong answers (each wrong digit→FEEDBACK_wrong is a transition). The
co-occurrence attention from the counting warmup and training observations
is the sole action selection signal.

**No pragmatic bias, no epistemic bonus, no backward spread.** Earlier
versions used heuristic terms (selectivity weighting, voting, harmonic
extension, IDF). All were removed. The attention mechanism alone produces
98% on succession — adding heuristics degrades performance.

**Shared sensory-motor nodes** (APC, Rao 2024): digit tokens are the same
nodes whether perceived in the observation or produced as actions. The
efference copy (transition edge from action to next observation) records
the sensory consequence of the action. No separate digit_ prefix needed.

## The Slow Path (Consolidation)

The fast path (the single process above) runs every timestep. It handles
edge weights. Consolidation runs periodically and handles *structure*:

- **Replay** — re-run spread→learn on stored snapshot pairs from Hippocampus.
  Transition edges that were discovered but oscillated to near-zero weight
  get re-confirmed across multiple replayed episodes. This is how rare-but-
  important edges survive.

- **Prune** — remove edges with deeply negative weight (dead connections).
  Remove nodes with no edges and zero resting potential (forgotten tokens).

- **Structure discovery** — colimits, functors, natural transformations,
  adjunctions. See "Categorical Structure" below.

The fast path is a neuron firing. The slow path is sleep.

## AgenticLoop — The Only Door

Every environment, test, benchmark, and application interfaces with the system
exclusively through AgenticLoop. No test calls graph methods directly for
learning. No adapter reaches into the KnowledgeGraph to create edges manually.

```python
loop = AgenticLoop(kg)

while not env.done:
    obs = env.observe()
    loop.observe([t[0] for t in obs], [t[1] for t in obs])

    actions = env.available_actions()
    chosen = loop.act(actions)
    if chosen is None:
        chosen = random.choice(actions)

    loop.observe([chosen], [2])  # edge_type 2 = action
    env.act(chosen)
```

## Hippocampus

Stores two things per timestep:

1. **Activation snapshot** — {NodeId: activation_level} for ALL active nodes,
   including decay residuals. Used by replay.

2. **Observation record** — the specific NodeIds from a single observe() call.
   No decay residuals. Used by structure discovery to find which tokens
   genuinely co-occur in the same observation.

Replay = reactivate a stored pattern so the fast path can re-strengthen edges.

## Tokenization

The tokenizer lives at the connector boundary between the environment and the
graph. It converts raw strings to opaque integer IDs. Downstream of the
tokenizer, everything is nodes and edges. No code inspects token content.
No code calls float(), int(), or str() on token values.

The tokenizer is invertible for display purposes only.

---

## Categorical Structure

The activation-based graph with attention is the substrate. It handles one-hop
associations (succession) at 98%. Category theory provides the machinery for
multi-hop composition (addition, multiplication) and cross-domain transfer.

### Why Categorical Structure Matters: The Poincare Principle

Henri Poincare made most of his mathematical discoveries not by working
harder on the problem at hand, but by recognising that the structure of one
problem was the same as the structure of another. His insight about Fuchsian
functions came on a bus — he suddenly saw that the transformations he'd been
studying had the same structure as non-Euclidean geometry. The insight wasn't
about either domain. It was about the **map between them**. A functor.

And it happened on a bus, not at his desk. That's consolidation — the slow
path replayed structures from two different contexts and found they were
isomorphic.

The general principle: **learning structure-preserving maps between subgraphs.**
For addition, the subgraphs are sub-chains of the NNO. For language, the
subgraphs are syntactic constituents. For analogy, the subgraphs are pairs with
parallel morphisms. The categorical machinery is the same — functors, natural
transformations, colimits, adjunctions — but the subgraphs being mapped differ
by domain.

Without categorical structure, every domain is learned from scratch. With
it, solving one problem teaches you about all problems with the same shape.

### omega-Category Levels

The KnowledgeGraph is an omega-category: a structure with cells at every level,
where each level n contains morphisms between (n-1)-cells.

| Level | Name | Contents |
|-------|------|----------|
| 0 | Instance | raw token observations, individual values |
| 1 | Morphism | laws, relations between instances (edges) |
| 2 | Natural transformation | structural relations between parallel morphisms |
| 3 | Functor | structure-preserving maps between sub-categories |
| 4 | Modification | relations between functors |
| n | n-cell | n-th level abstraction, unbounded |

Abstraction IS the colimit at every level. "Room" is the colimit of all
room-instance tokens. "Navigation law" is the colimit of all room-transition
edges. "Spatial structure" is the colimit of all navigation laws.

### Functors (next to implement)

A functor is a structure-preserving map between two subgraphs. Concretely:
a function F that maps nodes of subgraph A to nodes of subgraph B such that
for every edge a₁ → a₂ in A, the edge F(a₁) → F(a₂) exists in B.

**Addition as a functor**: `3 + 4 = 7` is a functor from the sub-chain
`[0, 1, 2, 3, 4]` (second operand's NNO position) to the sub-chain
`[3, 4, 5, 6, 7]` (starting from first operand). The functor sends
`0→3, 1→4, 2→5, 3→6, 4→7` and preserves successor edges.

The answer is the image of the second operand under this functor. No integer
extraction. No counting. Just a structure-preserving map between two sub-chains
of the same number line. The system discovers this by observing that training
examples always exhibit this correspondence.

**Discovery algorithm**: given training examples of an operation, find pairs
of sub-chains (one from the operand's NNO position, one from the first
operand to the answer) where a structure-preserving map exists. Verify by
simultaneously walking both chains and checking that successor edges commute.

**How this generalises**: finding that `[0,1,2,3,4]` maps to `[3,4,5,6,7]`
preserving SUCC edges is the same algorithmic operation as finding that
`{king, queen}` maps to `{man, woman}` preserving a role-flip edge. The
discovery algorithm is domain-agnostic: it finds structure-preserving maps
between any subgraphs, regardless of what the tokens mean.

### Natural Transformations

A natural transformation is a structural relation between two parallel
functors (same source structure, same target structure). The "+1" functor
and the "+2" functor are parallel: both map NNO sub-chains to sub-chains
starting from the first operand. A natural transformation says "they have
the same structure, just different chain lengths."

Why it matters: NTs enable generalisation across operations. Knowing "+1",
"+2", "+3" as separate functors, the NT discovers the PATTERN: "for any N,
+N maps the first N successor edges to the answer chain." This lets the
system predict "+7" without having seen 7-step iteration explicitly.

### Adjunctions

An adjunction is a pair of functors (F, G) where F compose G and G compose F
are both (approximately) identity. F left-adjoint to G means "F and G undo
each other."

**Addition/subtraction**: `add(_, 4)` and `sub(_, 4)` are adjoint. The system
discovers this by observing that every `add(n, 4) = r` has a corresponding
`sub(r, 4) = n`. The adjunction creates a bidirectional link between the
add-4 and sub-4 functor nodes, so knowing one gives the other for free.

**Discovery**: given two functors F and G over the same domain, check whether
F(G(x)) = x for all observed x. If so, record the adjunction. This enables
predicting G from F without independent training on G.

### Colimits

A colimit glues a diagram (subgraph with internal morphisms) into a single
node with universal properties. See Architecture.md for the full formal
definition.

For addition: the colimit of all "+4" training examples is the "+4" concept
node. Its cocone maps connect to each specific example, and its factored edges
connect to the answer digits via the universal property.

### Probability (Markov Category)

Edges are stochastic kernels. Spread computes weighted sums, which is a form
of stochastic prediction. The categorical structure of probability composition
(d-separation, conditional independence) is not yet explicit but is partially
present via the normalised attention mechanism.

### Sheaf Consistency

Knowledge learned in one context must agree with knowledge learned in an
overlapping context (gluing axiom). Context-dependent co-occurrence edges
(from causal masking and normalised attention) are the current approximation.
Full sheaf structure (validity domains, restriction maps, consistency checking)
is future work.

### MDL as Global Objective

The Minimum Description Length principle determines whether structural changes
improve the model. Creating a functor that compresses 10 training examples into
one reusable structure = shorter DL. Creating a spurious functor that adds
complexity without compressing = longer DL = rejected.

### Revision Log

Append-only history of structural changes. Non-amnesiac: the system remembers
what it tried and what failed. Prevents oscillation in consolidation.

### Credit Assignment: Prospective Configuration

Before edge weights change, activations settle to a prospective state (Song,
Bogacz et al. 2024). Two phases: (1) inference: relax activations given target
observation, (2) weight update: consolidate the settled pattern. Avoids
catastrophic interference.

---

## Architectural Laws

**One Graph**: There is one graph. It is the KnowledgeGraph. There is no
observation graph, no separate WorkingMemory object, no shadow data structures.
When something is not in the graph, it does not exist.

**One Process**: There is one process: spread → compare → update. Deduction,
induction, and abduction are three cases of this process, not three modules.

**Attention is the mechanism**: Action selection uses dot-product attention over
normalised co-occurrence edges. No heuristic biases (pragmatic, epistemic,
familiarity). The co-occurrence edges ARE the learned Q·K matrix.

**Opacity**: All values are opaque tokens. No code downstream of the tokenizer
may call float(), int(), or str() on a token value, fit a curve to token
values, or perform arithmetic on token values. Values have no meaning except
through their edges in the graph.

**Causal masking**: Co-occurrence edges are forward-only. Earlier tokens in an
observation predict later tokens. No reverse edges.

**No Pairs**: Observations are token activations. Edges connect nodes. There is
nothing to extract.

**Iron Law**: No dispatch on string names. All identity is by NodeId (opaque
integer). A system that works with {mul, add, sub} must work identically with
{X, Y, Z}. If it does not, it has hardcoded knowledge.

**AgenticLoop is the only door**: Every test, benchmark, and environment
adapter interfaces with the model exclusively through AgenticLoop. No test
calls graph learning methods directly. No adapter creates edges manually.

**Universality**: The same graph.py, loop.py, hippocampus.py, and
consolidation.py run on every test and every environment. Environment-specific
logic in these files means the abstraction is wrong.

## What Does Not Belong Here

- Environment simulators — live in `environments/`
- Neural networks — not part of this architecture
- Any code that calls float(), int(), or str() on token values
- Any code that extracts, decomposes, or preprocesses observations into tuples
- Any separate module for deduction, induction, or abduction
- Any test that bypasses AgenticLoop
- Any data structure parallel to the KnowledgeGraph
- Any heuristic bias in action selection (pragmatic, epistemic, familiarity)
