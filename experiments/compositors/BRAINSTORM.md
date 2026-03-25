# Compositors — Brainstorming

## What is a Compositor?

A neural network architecture where the embedding layer is a pointer into a
categorical knowledge graph, and a novel **composition layer** queries that
graph to enrich token representations with relational structure.

The thesis: the embedding layer (nn.Embedding — a flat lookup table) is the
weakest component of current architectures. It can only encode semantic
similarity through vector proximity. It cannot represent directed
relationships, typed relationships, composition, or hierarchy. And because
of weight tying, it is BOTH the first and last thing data touches — the
model's entire interface with the symbolic world.

Replace it with a pointer into a persistent categorical graph, add a
composition layer that queries that graph, and the existing transformer
machinery (attention for routing, SwiGLU for graph navigation) should
produce compositional generalisation.

## What We Know

1. **Opaque tokens.** Inputs and outputs are tokens with no inherent
   meaning. Everything the model knows about a token is learned purely
   from data.

2. **Atomic tokenization.** Tokenize at the smallest indivisible level:
   individual ASCII characters, individual pixel color values, etc.
   Whole-word or subword tokenization destroys internal structure that
   is relevant. The model must discover word boundaries and word meaning
   from character co-occurrence, just as a child does.

3. **World model, not similarity.** The model must NOT merely learn how
   semantically similar two tokens are — at the single-character level,
   similarity is nearly meaningless anyway. It must learn a world model
   structured as an omega-category: objects (instances), morphisms
   (relations), natural transformations (relations between relations),
   and so on, with composition at every level.

4. **Backpropagation is necessary but not sufficient.** The neural network
   trains via backpropagation. But the graph can ALSO be updated by
   non-gradient mechanisms — rule discovery, composition closure,
   structural operations that pure gradient descent cannot perform
   efficiently. The symbolic CTKG experiment proved that explicit
   relational structure achieves 100% OOD on tasks where neural
   networks score 0%. The question is how to couple symbolic graph
   operations with neural network training.

5. **Standard transformer block (revised roles).** Attention (with Q/K/V)
   handles dynamic routing between positions. SwiGLU FFN learns to
   NAVIGATE the graph — given relational context, which graph traversal
   minimises prediction loss? PoPE handles positional encoding.
   RMSNorm and residual connections handle normalisation and gradient
   flow.

---

## Concrete Design Target

Given ONLY this training data (character-level tokens):

```
"1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11..."          # number line
"1 < 2, 2 < 3, 3 < 4, 8 < 11"                    # less-than
"4 > 3, 3 > 2, 2 > 1, 9 > 5"                     # greater-than
"1 + 1 = 2, 1 + 2 = 3, 2 + 1 = 3, 2 + 2 = 4,    # addition
 2 + 3 = 5, 3 + 2 = 5"
```

Design a model that generalises PERFECTLY on:
- Succession (what comes after 47?)
- Greater-than (is 13 > 7?)
- Less-than (is 3 < 19?)
- Addition (what is 8 + 5?)

Key constraints:
- Addition = "start at n, apply succession m times." The model must LEARN
  this and then CARRY IT OUT.
- "1 + 3 = 4" requires each token to attend to each other token — the `+`
  tells you this is addition, the `1` and `3` are operands, the `=` signals
  the result position.
- The model must learn AND execute computations, not just pattern-match.

### What the Model Must Discover

From the number line alone:
- Each multi-character number is a unit (word boundary discovery)
- Numbers have an ORDER (the sequence is monotonically increasing)
- There is a SUCCESSOR relation: succ(n) = n+1

From the comparison data:
- `<` and `>` are INVERSE relations (adjoint pair)
- Both are DERIVED from the ordering discovered in the number line
- `a < b` iff `b` comes after `a` in the successor chain

From the addition data:
- `+` is a BINARY operation
- `a + b = c` means "apply succ to a exactly b times to get c"
- This is a FOLD over the successor morphism — the model must discover
  this compositional structure, not memorise the 6 addition facts

---

## Architecture

### Two structures with distinct roles

A Compositor has two distinct structures:

1. **The graph** — a persistent categorical knowledge graph. Stores BOTH
   knowledge (what objects exist, what morphisms connect them) AND
   relationships (composition, typing, hierarchy). This is the model's
   world model. It is NOT just a lookup table — it has categorical
   structure: identity, composition, associativity.

2. **The neural network** — attention + SwiGLU + norms. Its role is to
   NAVIGATE the graph: given an input sequence, determine which graph
   nodes and morphisms are relevant, compose them, and produce a
   prediction. The neural network does NOT store knowledge — it stores
   strategy for using knowledge.

The embedding vector is an **address**, not a description. It tells you
where the token lives in the graph. When you need to know the
relationships, you query the graph using the address.

### Division of labour

| What                         | Where it lives              |
|------------------------------|-----------------------------|
| Token identity               | Graph (node embeddings)     |
| Relationships between tokens | Graph (morphisms)           |
| Composition of relationships | Graph (categorical closure) |
| Higher-level concepts        | Graph (concept nodes)       |
| "Which positions matter?"    | Attention                   |
| "How to traverse the graph?" | SwiGLU FFN                  |
| Position encoding            | PoPE                        |
| Normalisation & gradient flow| RMSNorm + residuals         |

### What stays from a transformer

| Component          | Role                                      | Status    |
|--------------------|-------------------------------------------|-----------|
| Attention (Q/K/V)  | Dynamic routing between positions         | **Keep**  |
| PoPE               | Content/position separation in Q and K    | **Keep**  |
| SwiGLU FFN         | Graph navigation strategy                 | **Keep**  |
| RMSNorm            | Activation normalisation                  | **Keep**  |
| Residual connections | Gradient flow                           | **Keep**  |
| out_proj -> logits | Hidden state -> vocabulary scores         | **Keep**  |

### What changes

| Component          | Standard transformer         | Compositor                    |
|--------------------|------------------------------|-------------------------------|
| Embedding layer    | `nn.Embedding(V, d_model)`   | Pointer into categorical graph|
|                    | Flat lookup: token -> vector | Token -> graph address        |
|                    | Encodes similarity only      | Graph encodes typed morphisms |
| FFN role           | Stores knowledge             | Navigates the graph           |
| Compositor block   | Attention -> FFN             | Attention -> Composition -> FFN|
| Weight tying       | Same flat matrix for in/out  | Same graph for in/out         |

### The Compositor Block

```
Input: sequence of per-position states (d_model vectors)
  |
  +---> RMSNorm
  +---> Attention (Q/K/V with PoPE)
  |       "Which positions are relevant to each other?"
  |       Unchanged from a standard transformer.
  |
  +---> Residual connection
  |
  +---> RMSNorm
  +---> Composition Layer (THE NOVEL COMPONENT)
  |       "What morphisms in the graph connect the positions that
  |        attention identified as relevant? What do they compose into?"
  |       Queries the graph using position states as addresses.
  |       Enriches each position's representation with categorical
  |       structure -- not just "these positions are related" but
  |       "they are related BY THIS MORPHISM."
  |
  +---> Residual connection
  |
  +---> RMSNorm
  +---> SwiGLU FFN
  |       "Given the relational context from the graph, how should
  |        I navigate the graph to minimise prediction loss?"
  |       Learns traversal strategy, not facts.
  |
  +---> Residual connection

Output: updated sequence of per-position states
```

The three sub-layers have distinct roles:
- **Attention**: routing (who talks to whom)
- **Composition**: structure (what relationships connect them)
- **SwiGLU**: navigation (how to use those relationships to predict)

### The categorical embedding layer

Replaces `nn.Embedding`. Each token ID maps to a pointer (a `d_model`
vector) into the graph. The graph is a separate persistent structure
holding objects and morphisms.

The Yoneda lemma justifies this: an object IS its pattern of morphisms
to and from every other object. The pointer doesn't describe what a token
"is" — it locates the token within a web of relationships. The
relationships are in the graph, not in the vector.

Weight tying: the same graph is used for output. Given the final hidden
state, query the graph to find morphism strength to each vocabulary token.
The token with the strongest "next-token" morphism becomes the prediction.

### The composition layer (the novel component)

This is the ONLY new sub-layer. Its job:

1. Take per-position states (which are graph pointers enriched by attention)
2. Query the graph: "what morphisms connect the objects these positions
   point to?"
3. Compose the retrieved morphisms where applicable
4. Return enriched per-position states that carry relational context

The critical question: what does this look like as differentiable
computation? The graph must be queryable via soft lookups (not discrete
addressing), so that gradients flow through both the query and the graph.

### The graph's categorical structure

The graph MUST have actual category theory machinery, not just learned
tensors. The v0.1 experiment proved that bilinear relation heads without
categorical structure are just redundant attention — the gradient doesn't
use them, all morphism scores stay at zero, and the neural network routes
everything through attention + SwiGLU instead.

**What the graph needs:**

1. **Identity morphisms.** Every object has an identity morphism to itself.
   This is structurally enforced, not learned.

2. **Composition.** Given morphism f: A -> B and morphism g: B -> C,
   the graph automatically contains g . f: A -> C. This is the
   fundamental operation that makes a category a category. Without it,
   the graph is just a set of disconnected edges.

3. **Associativity.** (h . g) . f = h . (g . f). This should be
   structurally guaranteed by how composition is implemented.

4. **Typed morphisms.** Morphisms have source and target types. A
   morphism from "digit" to "digit" is different from a morphism from
   "operator" to "digit". The type system constrains which compositions
   are valid.

5. **Functors (eventual).** Structure-preserving maps between subgraphs.
   "The successor relation on single-digit numbers has the same structure
   as the successor relation on two-digit numbers." This is how the model
   generalises from small to large.

### Graph learning: beyond pure backprop

The v1 symbolic CTKG experiment achieved 100% OOD accuracy on succession,
addition, subtraction, multiplication, and more. It did this by:

- **BFM completion:** discovering binary function maps and filling ALL
  gaps, so that OOD pairs work by interpolation
- **Relation discovery:** finding that certain operations compose
  (e.g., iterated successor = addition) without being told
- **Positional role schema:** learning that inputs have fixed-length
  positional structure without hardcoded arity assumptions

These are structural operations that gradient descent cannot perform
efficiently — they require discrete decisions (create a new node, close
under composition, recognise a pattern across examples). The Compositor
graph needs a HYBRID learning system:

- **Gradient descent** updates the neural network (attention, SwiGLU)
  and the continuous parameters of the graph (node embeddings, morphism
  strengths)
- **Structural operations** update the graph's discrete structure
  (create new nodes, compose morphisms, discover rules). These can be
  triggered periodically during training (e.g., every N epochs) or
  when the gradient signal indicates the neural network is stuck.

The key constraint: structural operations must be DERIVED from data the
model has seen, not from human-engineered rules. The symbolic CTKG's
failure mode was that the programmer kept encoding the discovery instead
of writing code that discovers. The structural operations must be
general-purpose categorical constructions (composition closure, product
formation, colimit detection) — not task-specific algorithms.

---

## Key Insights

### 1. The graph stores knowledge; the FFN navigates it

In a transformer, the FFN stores facts as key-value memories in its weight
matrices. In a Compositor, the graph stores facts as morphisms. The FFN's
role changes from "knowledge store" to "navigation strategy" — given the
relational context the composition layer retrieved, how should I traverse
the graph to minimise prediction loss?

This is analogous to the difference between a database and a query engine.
The graph is the database. The FFN is the query engine. You don't store
the data in the query engine.

### 2. The v0.1 graph learned nothing — and why

The v0.1 experiment's graph had bilinear relation heads but no categorical
structure. All morphism scores stayed at zero. The gradient routed
everything through attention + SwiGLU because those were more efficient
paths — the composition layer's bilinear scoring was just redundant
attention with worse initialisation.

The fix is not to make the bilinear scoring better. The fix is to give
the graph capabilities that attention CANNOT replicate:
- **Composition** (transitive closure — attention can't do this in one layer)
- **Persistent structure** (attention is dynamic per-sequence; the graph
  persists across sequences and accumulates knowledge)
- **Discrete operations** (node creation, rule discovery — attention is
  purely continuous)

### 3. Higher-level structure lives in the layers, not the embedding

A transformer's embedding table only has vectors for vocabulary tokens. But
by middle layers, the residual stream represents concepts that don't exist
in the vocabulary. The higher structure is in the LAYER WEIGHTS, not the
embedding table.

In a Compositor, higher-level structure should live in the GRAPH (as
concept nodes and composed morphisms), not in the layer weights. The layer
weights should learn navigation strategy, not facts.

### 4. Attention already handles routing and variable binding

Attention is NOT limited to learning semantic similarity. It learns whatever
Q.K^T patterns reduce the training loss. PoPE separates content (magnitude)
from position (angle) in Q and K. Together, they can derive structural
roles from context. There is no need to replace attention.

### 5. Human pattern recognition is associative, not template-matching

Seeing "x^2 + 2x + 1" and recognising "(x + 1)^2" is associative recall:
the whole expression, as a gestalt, activates a memory of an equivalent
expression. The equivalence is a known morphism. If no association fires,
you fall back to procedures.

### 6. Efficiency demands direct higher-level operations

Iterated succession is correct for LEARNING addition. But once addition is
learned, it must become a DIRECT morphism. Each level of the omega-category
operates at its own level.

### 7. The embedding is a pointer, not a description

The embedding vector is an address into the graph. When relationships are
needed, the composition layer queries the graph. This avoids fitting a
relational tensor into a d_model-width pipeline.

---

## Open Problems — Notes

### 1. Object creation

Two mechanisms needed:

**Chunking** (product/coproduct): multiple tokens -> one object.
Characters `4`, `7` become entity "47".

**Abstraction** (colimit/limit): pattern across objects -> higher-level
object. "Each number maps to the next number" -> entity "successor".

**Open question:** Can products and colimits emerge from gradient descent,
or do they need structural operations (non-gradient graph updates)?
The symbolic CTKG created new nodes explicitly when patterns were
detected. A hybrid approach may be needed.

### 2. Variable binding

Attention + PoPE handles this for the initial design. The graph may
improve binding by providing typed relational constraints.

### 3. Exact tensor shapes

The composition layer must support:
- Soft lookup by pointer (differentiable)
- Morphism retrieval between pointers
- Composition of retrieved morphisms (matrix multiply over adjacency)
- Gradient flow through all of the above

### 4. Recurrence vs depth for iterative computation

Relevant prior work: "Scaling Latent Reasoning via Looped Language Models"
(arXiv, proof of concept: github.com Ouro). A single Compositor block
looped N times gives N composition steps. The model could learn to iterate
until convergence (a fixed point) with a learned halting criterion.

### 5. Hybrid learning: coupling gradient descent with structural operations

The graph needs two learning mechanisms working in tandem:

**Continuous:** gradient descent updates node embeddings, morphism
strengths, and all neural network weights.

**Discrete:** periodic structural operations update the graph's topology:
- **Composition closure:** if f: A->B and g: B->C exist, create g.f: A->C
- **Product formation:** if tokens frequently co-occur in fixed patterns,
  create a product node
- **Colimit detection:** if multiple morphisms share the same structure,
  create an abstract morphism node
- **BFM completion:** if a binary function map has gaps, fill them by
  interpolation (as the symbolic CTKG did)

The trigger for structural operations: the neural network's loss plateaus.
If the gradient can't reduce loss further, the graph's structure may need
discrete changes that gradient descent can't make.

The constraint: structural operations must be general-purpose categorical
constructions, not task-specific algorithms. The programmer does not
encode the discovery.
