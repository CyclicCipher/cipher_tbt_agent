# CTKG Design: Category Theory Knowledge Graph

Design document for the Category Theory Knowledge Graph — the external long-term memory and reasoning engine complementing the neural net's short-term memory. The CTKG stores principled knowledge (not just facts), provides interpretable priors, imposes constraints, and can perform structured reasoning independently.

## Role in the System

| Memory Tier | Implemented By | Brain Analog |
|---|---|---|
| Sensory buffer | SSM fast-decay channels | Early sensory cortex |
| Working memory | SSM surprise-gated state | Prefrontal + parietal cortex |
| Long-term knowledge | **CTKG** | Hippocampal-entorhinal cognitive map + anterior temporal semantic hub |
| Constraints | **CTKG** | Prefrontal constraint system (vmPFC, dlPFC, OFC, ACC) |
| Innate priors | **CTKG** (pre-populated before training) | Genome / developmental program |

The neural net handles perception and working memory (~4 chunk-references). The CTKG handles everything else.

## Foundational Framework

### Category Theory Basics (For Reference)

| Concept | Meaning | CTKG Role |
|---|---|---|
| **Object** | A type or concept | Entity, concept, mathematical structure |
| **Morphism** | A relationship or transformation between objects | Relation, function, derivation step |
| **Functor** | A structure-preserving map between categories | Analogy, abstraction, translation between domains |
| **Natural transformation** | A systematic map between functors | Meta-relationships (e.g., "every group homomorphism induces a ring homomorphism") |
| **Commutative diagram** | Path equivalence (two routes give the same result) | Constraint, invariant, law |
| **Limit / Colimit** | Universal construction (product, pullback / coproduct, pushout) | Query result, aggregation, synthesis |
| **Adjunction** | Optimal relationship between two functors (free ↔ forgetful) | Search/retrieval, optimal approximation |
| **Kan extension** | Best approximation of a functor along another | Generalization, prediction |

### The Functorial Data Model

The CTKG uses Spivak's functorial data model:

- **Schema**: A finitely-presented category S (objects = types, morphisms = relationships, commutative diagrams = constraints)
- **Instance**: A functor I: S → Set (assigns a set of elements to each type, a function to each relationship)
- **Query**: A functor from a query schema to the data schema; results are limits/colimits

Every relational database is isomorphic to a typed graph (and vice versa). The CTKG is both simultaneously.

Reference: Spivak & Kent (2012), "Ologs: A Categorical Framework for Knowledge Representation", arXiv:1102.1889.

## Structure: What the CTKG Contains

### Layer 1: Mathematical Foundations (Pre-populated)

Loaded before training begins. These are the "innate priors":

```
Category: Arithmetic
  Objects: Nat, Int, Rat, Real, Complex
  Morphisms: +, ×, -, ÷, ^, √, ...
  Diagrams: commutativity (a+b = b+a), associativity, distributivity, ...

Category: Logic
  Objects: Prop, Bool, Proof
  Morphisms: ∧, ∨, ¬, →, ∀, ∃
  Diagrams: modus ponens, de Morgan's laws, ...

Category: LinearAlgebra
  Objects: Vector, Matrix, Scalar, VectorSpace
  Morphisms: matmul, transpose, inverse, det, eigendecompose, ...
  Diagrams: (AB)C = A(BC), det(AB) = det(A)det(B), ...

Functors:
  embed: Nat → Int → Rat → Real → Complex    (embeddings)
  forget: VectorSpace → Set                    (forgetful)
  free: Set → VectorSpace                      (free construction)
```

### Layer 2: Domain Knowledge (Accumulated)

Grows during training and deployment:

```
Category: DanganronpaWorld
  Objects: Character, Location, Item, Event, Relationship
  Morphisms: located_at, possesses, witnessed, trusts, suspects, ...
  Diagrams: consistency constraints (a character can't be in two places at once)

Category: GameMechanics
  Objects: Trial, Evidence, Testimony, Contradiction
  Morphisms: supports, contradicts, implies, ...
  Diagrams: logical rules of evidence
```

### Layer 3: Constraints (Alignment)

```
Category: Constraints
  Objects: Action, LegalStatus, MoralStatus, SafetyStatus
  Morphisms:
    legal_check: Action → LegalStatus
    moral_check: Action → MoralStatus
    safety_check: Action → SafetyStatus
  Diagrams:
    legal_check(action) = illegal  ⟹  do_not_execute(action)    (hard constraint)
    moral_check(action) = harmful  ⟹  warn_user(action)         (soft constraint)
```

These constraints impose energy terms on the neural net's predictions. An action that violates a commutative diagram in the Constraints category produces high energy, driving the system away from that output.

## Traversal: How To Find Information

### Pattern Matching via Functors

A query is a small category Q (the "pattern") and we seek all functors Q → CTKG. Each functor is a match — it maps the query's objects and morphisms into the CTKG while preserving composition and identities.

Example: "Find all characters who suspect someone they trust"

```
Query category Q:
  Objects: A, B
  Morphisms: trusts: A → B, suspects: A → B

Results: all functors Q → DanganronpaWorld
  = all (character_a, character_b) such that both trusts and suspects edges exist
```

This is equivalent to a graph pattern match / SPARQL query, but with the additional constraint that commutative diagrams must be preserved.

### Pullbacks as Joins

A pullback finds all pairs of objects that share a common relationship. Given:

```
A --f--> C <--g-- B
```

The pullback A ×_C B = {(a, b) | f(a) = g(b)} is exactly a SQL JOIN on the shared key C.

Example: "Find all events witnessed by characters in the library"

```
Character --located_at--> Location <--location_of-- Event
                              |
                          = "Library"

Pullback = {(character, event) | character.location = event.location = Library}
```

### Limits for Multi-Constraint Queries

For queries with multiple constraints simultaneously, take the limit of the query diagram. The limit is the universal object satisfying ALL constraints — it's the "best" answer.

### Kan Extensions for Generalization

Given knowledge in domain A and a mapping A → B, the left Kan extension provides the "best generalization" of that knowledge to domain B. This is how the CTKG can reason about new domains by analogy.

Example: "We know how arithmetic works for natural numbers. What should arithmetic look like for matrices?"

```
Kan extension along embed: Nat → Matrix gives the "free" extension of arithmetic to matrices
```

### Yoneda Lemma for Associative Recall

An object X is **completely determined** by all morphisms into it: Hom(-, X). Given partial relational information about an unknown object, the Yoneda embedding narrows the candidates.

This IS content-addressable associative memory. "I'm looking for something that is located in the library, was seen by Makoto, and is related to the murder weapon" — each constraint is a morphism into the unknown object, and the intersection of representable functors identifies it.

## Writing: How To Add Information

### Adding Objects and Morphisms

New knowledge = new generators for the finitely-presented category. Adding an object or morphism is a **pushout** in the category of schemas — it glues new structure onto existing structure while preserving all existing constraints.

```
Existing schema S
        |
        | inclusion
        ▼
Extended schema S' = S + new object/morphism
```

The pushout ensures: all existing data remains valid, all existing constraints still hold, the new element is integrated coherently.

### Functorial Data Migration

When the schema evolves (new concept added, relationship refined), existing data is migrated via three adjoint functors:

- **Δ_F (pullback)**: Restrict data to the new schema's view of the old schema. Safe, always works.
- **Σ_F (left Kan extension)**: Push data forward, merging where the new schema identifies previously-distinct concepts. Union/merge semantics.
- **Π_F (right Kan extension)**: Push data forward conservatively, taking products where ambiguous. Join/product semantics.

These provide **automatic, provably correct** schema migration. No manual ETL scripts.

### Incremental Updates via Rewriting

For fine-grained modifications (updating a value, adding an edge), use **double-pushout (DPO) rewriting** on C-sets. An update rule is a span L ← K → R:

- L = pattern to match (the "before")
- K = context preserved (the "interface")
- R = replacement (the "after")

Applying the rule: find L in the data, remove L - K, add R - K. The pushout guarantees the result is a valid instance of the schema.

Reference: AlgebraicRewriting.jl (AlgebraicJulia project).

## Retrieval: How To Get Information Back

### For the Neural Net (Working Memory Interface)

The neural net queries the CTKG by producing a query embedding from its current state. The CTKG interface translates this into a categorical query, executes it, and returns a compact result:

```
Neural net state → query vector q
    ↓
CTKG interface layer (learned):
    q → categorical query Q (pattern matching / pullback / limit)
    Q applied to CTKG → result set R
    R → compressed embedding e (back to neural net)
    ↓
Neural net receives e as additional input (concatenated or cross-attended)
```

The result is a **chunk-reference**: a compact embedding representing the CTKG's answer, which the neural net holds in working memory as one of its ~4 active chunks.

### Associative Memory (Pattern Completion)

Given partial information (some morphisms known, some objects known), find the completion:

1. Construct a partial diagram D from known information
2. Compute the limit of D in the CTKG — this is the "most constrained" completion
3. If the limit is unique, the completion is determined
4. If multiple completions exist, return the set (or the "most likely" under a probability measure on the CTKG)

This is the categorical version of Hopfield network pattern completion: partial cue → attractor → complete pattern.

Formally, this uses **sheafification**: partial locally-consistent data (a presheaf) is completed to globally-consistent data (a sheaf) via the sheafification functor (left adjoint to inclusion). This is the *universal* pattern completion — the "best possible" given the constraints.

Reference: Sheaving as semantic compositionality (PMC6939351).

### Episodic Memory (Temporal Sequences)

Episodic memories are sections of a **presheaf over a temporal category**:

```
Category: Timeline
  Objects: time intervals [t1, t2]
  Morphisms: inclusions [t1, t2] ⊆ [t1', t2'] when t1' ≤ t1 and t2 ≤ t2'

Episodic presheaf: F: Timeline^op → Set
  F([t1, t2]) = set of events/states during [t1, t2]
  restriction maps: F([t1', t2']) → F([t1, t2]) = "what happened in this sub-interval"
```

Temporal queries ("what happened between t1 and t2?") are just restriction maps. The sheaf condition ensures consistency across overlapping time intervals.

Reference: Schultz & Spivak (2019), "Temporal Type Theory", arXiv:1710.10258.

## Computation: How the CTKG Reasons

### Morphism Composition as Multi-Step Reasoning

Multi-step derivation = morphism composition. If `f: A → B` and `g: B → C`, then `g ∘ f: A → C`.

Example — solving `2(x + 3) = 10`:

```
Objects: equations    Morphisms: valid algebraic steps

2(x+3) = 10 --[divide by 2]--> x+3 = 5 --[subtract 3]--> x = 2

Composite morphism: "divide by 2 then subtract 3" : {2(x+3) = 10} → {x = 2}
```

Commutativity of diagrams ensures: any two valid derivation paths from the same premise to the same conclusion are equivalent.

### Curry-Howard-Lambek: Logic = Types = Categories

| Logic | Type Theory | CTKG |
|---|---|---|
| Proposition P | Type P | Object P |
| Proof of P | Term of type P | Morphism 1 → P |
| P implies Q | Function P → Q | Morphism P → Q |
| P and Q | Product (P, Q) | Product P × Q |
| P or Q | Sum (Either P Q) | Coproduct P + Q |
| For all x, P(x) | Dependent product Π | Right adjoint |
| Exists x, P(x) | Dependent sum Σ | Left adjoint |
| Modus ponens | Function application | Evaluation: Q^P × P → Q |

A CTKG structured as a **cartesian closed category** IS a programming language and a logic simultaneously. Proofs are morphisms. Computation is composition. The CTKG doesn't need an external reasoner — reasoning is navigating morphisms.

### Topos for Internal Logic

If the CTKG is structured as a **topos** (a category with subobject classifier, finite limits, and exponentials), it has **full higher-order intuitionistic logic** as its internal language. You can:

- Quantify over objects (∀, ∃)
- Form implications and conjunctions
- Prove theorems by constructing morphisms
- The proofs ARE programs (Curry-Howard)

The logic is **intuitionistic** (constructive) — no proofs by contradiction. This is actually a feature: constructive proofs correspond to algorithms, so every proof in the CTKG is an executable computation.

### String Diagrams for Visual Reasoning

Morphisms in monoidal categories can be represented as **string diagrams**: boxes (operations) connected by wires (types). Equational reasoning = diagram manipulation. This is the natural graphical syntax for categorical computation.

String diagrams are sound and complete for symmetric monoidal categories — any equation that holds in the category can be proved by diagram manipulation, and vice versa.

### Multi-Step Integration Example

Problem: Compute ∫₀¹ 2x dx

```
Category: Calculus
  Objects: Expression, Antiderivative, BoundedValue
  Morphisms:
    integrate: Expression × Variable → Antiderivative
    evaluate_bounds: Antiderivative × Bounds → BoundedValue
    simplify: Expression → Expression

Computation (morphism chain):
  2x --[integrate w.r.t. x]--> x² + C --[evaluate at 0,1]--> (1² + C) - (0² + C) --[simplify]--> 1
```

The CTKG can perform this entirely symbolically by composing morphisms. The neural net's role would be: (1) parsing the problem into a CTKG query, (2) interpreting the result.

## Existing Tools

| Tool | Language | Maturity | Best For |
|---|---|---|---|
| **CQL** (Categorical Query Language) | Java | Production-ready (single-node) | Data integration, schema migration, theorem proving |
| **Catlab.jl** | Julia | Research-grade, active development | Applied category theory, C-sets, rewriting |
| **discopy** | Python | Research-grade | String diagrams, quantum, NLP |
| **catgrad** | Python | Experimental | Categorical gradient-based learning |

CQL has an embedded theorem prover for data integrity. Catlab.jl has the most active community and the best C-set implementation.

For our system, the most likely path is:
1. Start with **Catlab.jl** for prototyping (Julia has good interop with Python via PyCall/PythonCall)
2. Or build a custom lightweight implementation in **Python/PyTorch** for tight integration with the neural net
3. Use **CQL** for formal verification of schema correctness

## Addressing OOLONG and BrowseComp-Plus

### OOLONG (Aggregation Over Entire Context)

OOLONG requires counting, temporal reasoning, and distributional queries across all tokens. The CTKG addresses this via:

1. **Neural net processes the stream**: surprise-gated, only novel information enters working memory
2. **Novel entities/relations written to CTKG**: as new objects/morphisms via pushout
3. **CTKG performs aggregation**: colimits for counting, temporal presheaf queries for temporal reasoning
4. **CTKG returns answer**: as a categorical construction, translated to natural language

The key insight: OOLONG's aggregation operations (counting, filtering, temporal ordering) are all **categorical constructions** (colimits, pullbacks, presheaf operations). The CTKG performs them exactly, not approximately.

### BrowseComp-Plus (Multi-Hop Reasoning Over 6-11M Tokens)

The neural net reads documents one at a time (within context window). For each:
1. Extract entities and relations → write to CTKG as objects and morphisms
2. The CTKG accumulates a growing knowledge graph across all documents
3. Multi-hop reasoning = morphism composition in the CTKG
4. The answer follows from navigating the graph

This is directly analogous to how the RLM (Recursive Language Model) works, but with categorical structure instead of Python REPL decomposition. The CTKG provides the decomposition structure naturally.

## Open Questions

1. **CTKG implementation language**: Julia (Catlab.jl) vs custom Python vs hybrid?
2. **Neural net ↔ CTKG interface**: How to translate between continuous embeddings and categorical structures? Learned encoder/decoder? Hard-coded templates?
3. **CTKG scale**: How many objects/morphisms before queries become slow? CQL's theorem prover is the bottleneck.
4. **Uncertainty in the CTKG**: Category theory is deterministic. How to represent uncertain knowledge? Probability monads? Fuzzy categories? Enriched categories over [0,1]?
5. **Learning new categories**: Can the neural net discover new categorical structure (new objects, morphisms, commutative diagrams) from data? This is category learning, not just category querying.
6. **Decidability**: The word problem for finitely-presented categories is undecidable in general. What restrictions ensure tractability? (Sketch theory provides sufficient conditions.)

## Key References

### Category Theory Foundations
- Spivak & Kent (2012), "Ologs: A Categorical Framework for Knowledge Representation", arXiv:1102.1889
- Spivak (2012), "Functorial Data Migration", arXiv:1009.1166
- Fong & Spivak (2019), "Seven Sketches in Compositionality", arXiv:1803.05316
- Patterson (2017), "Knowledge Representation in Bicategories of Relations", arXiv:1706.00526
- Schultz et al. (2016), "Algebraic Databases", arXiv:1602.03501

### Computation and Logic
- Lambek & Scott (1986), "Introduction to Higher Order Categorical Logic"
- Mac Lane & Moerdijk (1992), "Sheaves in Geometry and Logic"
- Shiebler (2022), "Kan Extensions in Data Science and Machine Learning", arXiv:2203.09018
- Aguinaldo et al. (2023), "A Categorical Representation Language for Knowledge-Based Planning", arXiv:2305.17208

### Memory and Cognition
- Schultz & Spivak (2019), "Temporal Type Theory", arXiv:1710.10258
- PMC6939351, "Sheaving — a universal construction for semantic compositionality"

### Tools
- CQL: https://categoricaldata.net/ | https://github.com/CategoricalData/CQL
- Catlab.jl: https://github.com/AlgebraicJulia/Catlab.jl
- AlgebraicJulia: https://www.algebraicjulia.org/
- discopy: https://github.com/discopy/discopy
