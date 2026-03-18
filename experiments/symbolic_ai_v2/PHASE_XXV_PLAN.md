# Phase XXV: The Bayesian CTKG — Architecture Redesign

## Preamble: Why This Document Exists

The math benchmark exposed a fundamental architectural flaw: the system cheats. Not
deliberately — but structurally. `RelationStore`, `SkeletonStore`, `HankelCount`,
`ProcessRule`, `ChainRule` all live outside the CTKG in Python dicts and dataclasses.
The inference engine dispatches on operator name strings (`if op == 'linsolve'`).
Knowledge is stored in RAM, not in the graph. The bitter lesson was violated by
construction, not by choice.

This document specifies the complete replacement.

---

## Part I: The Two Laws (Anti-Amnesia Cages)

These laws must be **architecturally enforced**, not merely remembered. If you are
reading this document in a future session and have forgotten why they exist, the
benchmark described below will break if you violate them. That is the cage.

### The Iron Law
**No `if op == 'some_op'` dispatch. No operator-specific code paths.**

All reasoning must emerge from data flowing through a uniform graph traversal. The
system must not know what "addition" means. It must only know how to follow edges.

*Architectural enforcement*: The benchmark uses randomly chosen novel symbols for
all operators (see Part II). A system that hardcodes `add` semantics scores 0% on
a benchmark that uses `⊕` for addition.

### The Bitter Lesson (Sutton 2019)
**General methods that leverage computation always win over methods that encode human knowledge.**

Every time AI researchers built in domain expertise — hand-crafted features, structured
representations, expert rules — they were eventually beaten by methods that simply searched
or learned harder with more compute. The knowledge encoding gave a short-term advantage
but a long-term ceiling. General methods have no ceiling.

Applied here: every piece of mathematical knowledge you are tempted to hardcode (what
addition means, what a derivative is, that `linsolve` uses an equalizer) is a ceiling.
The system must instead learn these things from data through general graph operations.
The benchmark enforces this by making the domain unrecognizable (anonymous symbols).
No amount of domain knowledge survives anonymization; only general computation does.

*Architectural enforcement*: Anonymous symbol tables. A system whose performance
degrades when operator names change encoded domain knowledge, not learned structure.

### The OOD Generalization Law
**No memorization. All knowledge must generalize.**

A system that stores seen (input, output) pairs and retrieves them on test scores
0% on out-of-distribution (OOD) test sets. The benchmark's train/test split is
designed so that memorization of training pairs is structurally insufficient.

*Architectural enforcement*: The benchmark's test partition specifically contains
inputs NOT present in training. A lookup table — a flat store of (input → output)
pairs — is illegal and will fail all OOD tests. Every inference must be a
computation, not a retrieval.

---

## Part II: The Benchmark — Anti-Cheat Design

### Core Principle: Structural Anonymization

Every benchmark session instantiates a fresh **symbol table** mapping abstract
role names to randomly drawn Unicode symbols:

```
INDUCTION session: {succ → ⊞, pred → ⊟, eq → ↦, step → ∘, ans → ⊡}
DEDUCTION session: {compose → ⊗, identity → ε, object → ◦, morphism → →}
```

The system sees only the symbols. It never sees the words. A system that
hardcodes "if I see ⊞ and a number, output the next number" must regenerate
that knowledge fresh every session from examples. It cannot rely on the string
`'succ'` appearing in its code.

This is not optional. The benchmark runner generates the symbol table at test
time and it changes between runs.

### Symbol Table Generation

```python
# Pseudocode — actual implementation in benchmark runner
import random, string

PRINTABLE_UNICODE = [chr(i) for i in range(0x2200, 0x22FF)]  # mathematical operators

def fresh_symbol_table(role_names: list[str]) -> dict[str, str]:
    symbols = random.sample(PRINTABLE_UNICODE, len(role_names))
    return dict(zip(role_names, symbols))
```

The system is given the symbol table at the start of each session as raw
character-level observations, not as a parsed dict. It must infer which symbol
plays which role by observing examples — which is exactly induction.

### Three Tracks

#### Track I: Induction
Given a stream of examples `(input_seq → output_seq)`, infer the rule.

*Difficulty levels*:
1. **I-1**: Single symbol function on naturals (3 examples → predict next)
2. **I-2**: Binary operation on naturals (5 examples → predict on unseen inputs)
3. **I-3**: Composed operation (the output of one rule is input to another)
4. **I-4**: Operation with multiple cases (base case + recursive case)
5. **I-5**: Operation family (same structure, different parameters)
6. **I-6**: Algebraic law discovery (commutativity, associativity from examples)
7. **I-7**: Structural law discovery (functor laws, natural transformation laws)
8. **I-8**: Theory discovery (infer an entire consistent rule system)

*Train/test split rule*: Training gives examples on inputs `{0..N}`. Test queries
on `{N+1..2N}`. Memorization of training pairs scores 0%.

#### Track D: Deduction
Given a set of rules (expressed as CTKG edges), derive conclusions.

*Difficulty levels*:
1. **D-1**: Single-step rule application
2. **D-2**: Two-step chain (compose two rules)
3. **D-3**: Three-way chain
4. **D-4**: Case split (deduction requires choosing a branch)
5. **D-5**: Modus ponens in propositional logic
6. **D-6**: Universal quantifier instantiation
7. **D-7**: Proof search (find a derivation of length ≤ k)
8. **D-8**: Type-theoretic derivation (dependent types as propositions)

*Cheating prevention*: Rules are expressed with novel symbols. The system cannot
hardcode "modus ponens"; it must discover that `A → B, A ⊢ B` is valid from
examples of valid and invalid derivation steps.

#### Track A: Abduction
Given observations and a partial theory, hypothesize the missing rule.

*Difficulty levels*:
1. **A-1**: Single anomalous example — what rule explains it?
2. **A-2**: Two competing hypotheses — which fits the data better?
3. **A-3**: Hypothesis that requires extending the theory
4. **A-4**: Anomaly that falsifies a current rule — retract and replace
5. **A-5**: Latent variable hypothesis (unobserved intermediate)
6. **A-6**: Multiple anomalies requiring a single unified explanation
7. **A-7**: Theory revision that preserves explained phenomena
8. **A-8**: Paradigm shift — the anomaly requires a new ontology

*Cheating prevention*: The anomaly is structurally valid under the new rule
but structurally impossible under any previous rule. A lookup table cannot
generate the hypothesis; graph structure must change.

### The Einstein Test (Final Level)

Level E: Given only:
- Observations that match Newtonian mechanics for low velocities
- Observations that match Maxwell's equations
- The Michelson-Morley experiment result (null result for ether)
- The perihelion precession of Mercury

Derive a consistent theory that predicts all four.

This is not a near-term deliverable. It is a north star. The system must reach
Track I-8, D-8, A-8 before it can attempt E. Reaching E means the system can:
1. Induct over structured observation streams (I-8)
2. Derive consequences of a formal theory (D-8)
3. Perform principled theory revision under anomaly (A-8)

*Current status*: The math benchmark covers approximately I-1 through I-3 for
a fixed symbol table. We are at the beginning.

### Cheating-Resistance Summary

| Attack vector | Enforcement mechanism |
|---|---|
| Hardcode operator semantics | Novel symbols each session |
| Memorize training pairs | OOD test partition by construction |
| Store (input→output) lookup table | OOD inputs not in any training pair |
| Hardcode rule templates | Rules expressed structurally, no string names |
| Hardcode data formats | Format varies by session |
| Hardcode thresholds | Thresholds are learned from data |

---

## Part III: The Bayesian Outer Loop

### Current Problem

The current inference pipeline is a decision tree:
```
Level 1f: SkeletonStore
Level 1c: RelationRule
Level 0.5: FC/adj lookup
Level 0.6: NNO chain
Level 0.7: NNO fold
→ {}  (give up)
```

This is not Bayesian. It does not track confidence. It does not revise beliefs
on new evidence. It does not detect anomalies. It cannot ask "why did this
fail?" The system has no model of its own uncertainty.

### Replacement: Belief-Surprise-Revision

The outer loop becomes:

```
observe(token_sequence)
  → Surprise?
    YES → trigger Revision
    NO  → update Belief
  → Belief selects active Theory
  → Theory generates Prediction
  → Prediction compared to observation
  → loop
```

#### Belief

A **Belief** is a probability distribution over theories. It is stored as a
weighted set of CTKG subgraphs (each subgraph = one theory). The weights are
maintained as node-level float annotations on a special "belief" node in the CTKG.

Belief is NOT stored in a Python dict. It is a subgraph of the CTKG where:
- Each theory is represented as a cluster of morphism nodes
- Each cluster has an edge to a shared "belief-state" node
- The edge carries a `weight` annotation (the probability)
- Marginalizing beliefs = traversing these edges and summing weights

#### Surprise

A **Surprise** signal is generated when the current theory's prediction diverges
from observation beyond a threshold. Formally:

```
surprise(obs, prediction) = KL(obs_distribution || predicted_distribution)
```

This threshold is itself a learned value stored as an edge annotation in the
CTKG (not a Python float constant). It adapts as the system encounters more
examples.

High surprise triggers Revision. Low surprise triggers Belief update (reweight
the theory that predicted correctly; discount theories that predicted wrongly).

#### Revision

A **Revision** is a graph edit on the CTKG that:
1. Identifies the subgraph responsible for the failed prediction
2. Generates candidate modifications (new edges, new nodes, retracted edges)
3. Evaluates each candidate against all stored observations
4. Adopts the candidate that maximizes posterior probability

This is abductive inference over graph edits. The candidate generator is a
CTKG traversal that follows "abduction templates" — morphism patterns that,
when instantiated, produce new rule candidates.

Revision is the hardest part and the last to implement. Initial versions will
use a restricted hypothesis space (only rules of the forms already seen).

---

## Part IV: The Fully Unlabeled CTKG

### Current Violation

Every node in the current system is identified by a Python string:
`'add'`, `'succ'`, `'p0'`, `'eq'`. String identity IS node identity.
This means the system "knows" that `'succ'` means successor because its
Python code hardcodes that meaning. The graph structure is irrelevant.

### The Fix: NodeId = int

Node identity must come only from edges. The only nodes with intrinsic
meaning are character nodes (NodeId 0–127 = `ord(char)`). All other nodes
are assigned opaque integer IDs at creation time.

Character nodes are special for the same reason sensory primitives are
special in any grounded system: they are the boundary where the external
world enters. A character node does not derive its identity from edges
because there is nothing more primitive to derive it from — it IS the
primitive. This is the CTKG's equivalent of a sensory receptor.

This generalises to other modalities. When vision is added, pixel intensity
values (or patch embeddings discretised to a finite alphabet) will be the
sensory primitives of that modality — their own set of intrinsic-identity
nodes, not derived from edges. When audio is added, discretised frequency
bins or raw PCM sample values play the same role. When mouse input is added,
(x, y, button) atoms are the primitives. Each modality has its own sensory
floor: a finite set of nodes whose identity is given by the world, not by
the graph. Everything above that floor — tokens, concepts, rules, theories
— is unlabeled and derives identity purely from structure.

This is specified in detail in the existing NodeId Refactor plan (see the
plan file rippling-hopping-feather.md). That plan must be executed as a
prerequisite to Phase XXV. After it, no module in the pipeline can contain
a string like `'succ'` — only integer IDs.

### Zero Pre-made Nodes

Under the old design, `TOKEN_GRAPH` pre-registers well-known nodes:
`EQ_NODE`, `STEP_NODE`, `ANS_NODE`, etc. This is still a form of hardcoding.

Under Phase XXV, the system starts with only character nodes (0–127).
ALL other nodes — including structural tokens like `eq` and `step` —
are created by the system when it first observes them. The benchmark
runner presents the symbol table as observations. The system creates nodes
for each symbol by observing the character sequence.

*Implementation*: The benchmark runner writes a "bootstrap" sequence of
observations in the anonymous symbol language. The system's first act is
to parse these characters and create nodes for each distinct token. Those
nodes have no semantic labels — they are identified only by their character
subgraph (the edges connecting them to the 0–127 character nodes that spell
their name).

Wait — this is a subtle point. Even character-sequence identity is a form
of labeling. The true unlabeled ideal would have even character nodes be
anonymous. But we need a grounding truth: the real world anchors meaning in
perception, and character codes are our minimal perceptual primitives. So:

**Axiom**: Nodes 0–127 are the perceptual atoms. Their identity IS their
value. All other identity emerges from structure.

### Knowledge Storage in the CTKG

Currently, the system stores knowledge in:
- `RelationStore._relations: dict[str, list[Relation]]`
- `SkeletonStore._store: dict[...] `
- `HankelCount._data: dict[...]`
- `ProcessRule._patterns: dict[...]`

After Phase XXV, ALL of this is stored as CTKG edges. Specifically:

#### Rules as Morphisms
A rule `f(a, b) = c` is a morphism in the CTKG:
- Source object: the type of `(a, b)` — a product type node
- Target object: the type of `c`
- The morphism node carries edge annotations (weight, support count, etc.)

A relational rule `bfm_op(role_a, role_b) → output_role` is a span in the
CTKG:
```
input_type_a ──edge_a──> rule_node <──edge_b── input_type_b
                            |
                          edge_c
                            ↓
                        output_type
```

#### Belief as Subgraph Weights
As described above — edge weights on "theory" subgraphs.

#### Process Programs as Path Sequences
A process like `succ(succ(x))` is stored as a path in the CTKG:
```
x_node ──succ_morphism──> y_node ──succ_morphism──> z_node
```
The "program" is the path. Executing it is following the edges.

#### No Lookup Tables
A lookup table is a flat store of `(input → output)` pairs with no structure.
It is illegal because:
1. It cannot generalize (OOD inputs have no entry)
2. It violates the unlabeled principle (inputs are identified by string equality)
3. It stores knowledge outside the CTKG

Every former lookup table is replaced by a structured CTKG subgraph that
supports graph-traversal inference.

---

## Part V: Categorical Representation of I/D/A

### Induction as Functor Discovery

Inductive learning = discovering a functor F: Observations → Rules.

Given observations O₁, O₂, ..., Oₙ as morphisms in an observation category,
induction finds a functor F that:
- Maps each Oᵢ to a morphism in the rule category
- Preserves composition: F(O₂ ∘ O₁) = F(O₂) ∘ F(O₁)

The functor is stored in the CTKG as a mapping between subgraphs. Learning
= extending the functor to cover more of the observation category.

### Deduction as Morphism Composition

Deductive inference = morphism composition in the rule category.

Given rules A → B and B → C, conclude A → C by composing. This is literally
the composition operation in the CTKG. Deduction IS graph traversal.

A deduction proof is a path in the CTKG from hypothesis to conclusion. Proof
search = path-finding. Proof validation = path existence check.

### Abduction as Hypothesis Generation via Left Kan Extension

Abductive inference = given a partial functor, extend it minimally to cover
the anomaly.

Formally: given functor F: C → D on a subcategory C ⊂ C', and an anomalous
observation O in C' \ C that F cannot explain, find the **left Kan extension**
Lan_ᵢF: C' → D that extends F to cover O.

The left Kan extension is the "most conservative" extension — it adds only
what is necessary to explain O, nothing more. This is exactly the principle
of abductive reasoning: hypothesize the minimum revision that explains the
anomaly.

*Implementation note*: Full Kan extension computation is expensive. The initial
implementation uses a restricted version: only consider extensions that add
a single new morphism (analogous to adding one new rule). More general
extensions are added in later phases.

### Markov Category for Belief

The belief distribution over theories is a morphism in a Markov category
(following Fritz 2020, already referenced in CTKG architecture).

A belief state = a Markov kernel `κ: 1 → Theories` (a probability distribution
over theories, viewed as a morphism from the terminal object).

Bayesian update = composition with a likelihood kernel:
```
posterior = likelihood ∘ prior  (normalized)
```

This is stored in the CTKG as a morphism in the "belief" layer — a subgraph
that contains probability-annotated edges between theory nodes.

---

## Part VI: Migration Plan

### What to Keep

- **CTKG hypergraph structure** (`ctkg/core/morphism_graph.py`) — keep, extend
- **Categorical vocabulary** (functors, morphisms, natural transformations) — keep
- **Character-level NodeId foundation** (from NodeId Refactor plan) — keep
- **Test infrastructure** (`ctkg/tests/`) — keep, adapt
- **Math benchmark OOD results** — use as regression baseline

### What to Throw Out

| Module | Reason |
|---|---|
| `ctkg/learning/relation_store.py` | External Python dict storage |
| `ctkg/learning/skeleton_lambda.py` | External Python dict storage |
| `ctkg/learning/hankel_count.py` | External lookup table |
| `ctkg/learning/process_discover.py` | String-comparison dispatch |
| `ctkg/inference/predict.py` (outer loop) | Decision-tree, not Bayesian |
| `ctkg/inference/reason.py` | Hardcoded rule templates |
| All `if op == '...'` dispatch | Iron Law violation |
| All `str → dict` lookups | Bitter Lesson violation |

### What Gets Replaced By

| Old | New |
|---|---|
| `RelationStore._relations` | CTKG morphism subgraph |
| `SkeletonStore._store` | CTKG path-pattern subgraph |
| `HankelCount._data` | CTKG Hankel matrix as bipartite subgraph |
| `predict.py` decision tree | Belief-Surprise-Revision outer loop |
| `ProcessRule._patterns` | CTKG composition paths |
| `if op == 'X'` | CTKG edge traversal from op node |

---

## Part VII: Implementation Stages

### Stage 0 (Prerequisite): NodeId Refactor
Complete the plan in `rippling-hopping-feather.md`. After this, no string
token names exist inside the pipeline.

### Stage 1: Belief Layer in CTKG
Add a "belief" subgraph to the CTKG:
- Theory nodes (each theory = a cluster of morphism nodes)
- Belief edges (theory → belief-state node, weight annotation)
- Methods: `update_belief(theory_id, evidence)`, `get_belief(theory_id)`
- No Python dicts — all state in graph edges

### Stage 2: Observation Parser
Replace `RelationStore.update_batch(seqs)` with an observation parser that:
- Takes character sequences (not string tokens)
- Creates NodeIds for each new token via character-subgraph construction
- Writes observations as typed edges in the CTKG
- No special-casing of any token type

### Stage 3: Induction via Functor Discovery
Implement functor discovery as CTKG subgraph matching:
- Given observation edges, find morphism patterns that cover them
- Write discovered morphisms back to CTKG
- No external storage — the CTKG IS the rule store

### Stage 4: Deduction via Path Traversal
Replace the prediction engine with a path-finding traversal:
- Given a query (source node, target type), find a path
- Path = composition of morphisms
- No decision tree, no level hierarchy — one algorithm

### Stage 5: Surprise Detection
Implement KL-based surprise scoring:
- Compare predicted path distribution to observed output
- Surprise threshold stored as CTKG edge annotation
- High surprise → flag for Revision

### Stage 6: Revision via Minimal Graph Edit
Implement restricted abduction:
- Identify failing subgraph
- Generate single-morphism extension candidates
- Score by posterior probability
- Adopt best candidate

### Stage 7: Benchmark Integration
Run I/D/A benchmark with anonymous symbol tables:
- All three tracks
- Score per level
- Regression against math benchmark OOD results

### Stage 8: Einstein Test Scaffolding
Define the observation streams for the Einstein test prerequisites:
- Newtonian mechanics stream (I-1 through I-5)
- Maxwell's equations stream (I-5 through I-7)
- Lorentz invariance anomaly (A-4 through A-7)
- Mercury precession anomaly (A-6 through A-8)

---

## Part VIII: Verification Criteria

### After Stage 0
- All 495 existing tests pass
- No string tokens inside pipeline (confirmed by grep for `== 'succ'` etc.)

### After Stage 3
- I-1 through I-3 benchmark: ≥90% on each level
- Anonymous symbol table confirmed by running with 10 different symbol tables
  and variance < 5%

### After Stage 4
- D-1 through D-4 benchmark: ≥90% on each level
- Math benchmark OOD results unchanged (regression)

### After Stage 5
- Surprise correctly flags 95%+ of anomalous inputs
- False positive rate < 5% on normal inputs

### After Stage 6
- A-1 through A-3 benchmark: ≥80% on each level
- Belief correctly reweights theories after revision

### After Stage 7
- Full I/D/A benchmark scores documented
- Anonymous symbol table variance < 5% across 20 sessions

### Einstein Test: Not a completion criterion for Phase XXV.
It is the long-term north star. Phase XXV ends when Stage 7 verification
is satisfied.

---

## Appendix: Why Lookup Tables Are Illegal

A lookup table stores pairs `{key → value}`. To query it:
1. Hash the key
2. Return the stored value

This is NOT reasoning. It is retrieval. It generalizes to exactly zero OOD
inputs. It is the opposite of what this system is trying to do.

The correct structure is a **morphism in a category**: a rule that maps
objects of type A to objects of type B, with defined behavior on ALL objects
of type A (not just the ones you happened to store).

A morphism stored in the CTKG can be:
- Composed with other morphisms (deduction)
- Extended by functor discovery (induction)
- Revised under anomaly (abduction)

A lookup table cannot do any of these things.

Every time you are tempted to write `dict[str, ...]` to store a rule,
ask: what is the CTKG morphism this rule corresponds to? Then store that.

---

## Appendix: Why External Python Modules Are Illegal for Knowledge Storage

`RelationStore` is a Python class. Its knowledge dies when the process ends.
It cannot be traversed by the inference engine. It cannot be revised by the
abduction engine. It cannot be combined with other knowledge via categorical
composition. It cannot be queried using the same graph algorithms as everything
else. It requires special-case code in the inference loop.

The CTKG is the universal substrate. Everything that matters must be in it.

External Python modules are permitted only for:
- Benchmark infrastructure (not knowledge)
- Parsing/serialization at system boundary
- Numerical computation (numpy operations on subgraphs)

They are illegal for:
- Storing rules, patterns, or frequencies
- Making predictions
- Storing belief states
- Anything that should generalize to new inputs

---

*This document is the specification. Implementation begins with Stage 0.*
*Current date: 2026-03-16.*
*All future development is gated on this plan.*
