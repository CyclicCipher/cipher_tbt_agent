# Category Theory Knowledge Graph (CTKG) — Design Document

**Date:** 2026-02-16 (updated 2026-03-05)
**Status:** Universal type system + DSL parser + arithmetic domain + sheaf consistency + logic domain + epistemic reasoning + standalone symbolic AI runtime implemented.
**Context:** Mistake #44 showed that missing prerequisites between counting and arithmetic caused the model to memorize instead of compose. A knowledge graph with structural constraints would have caught this automatically — like a compiler catching a missing import. This motivates building the CTKG as a general-purpose infrastructure component, not just a curriculum tool.

---

## What the CTKG Is

A directed acyclic graph where:
- **Nodes** = concepts/skills (objects in the category)
- **Edges** = "is prerequisite for" relationships (morphisms)
- **Each node** carries metadata: what it teaches, how to verify it's learned, what scratchpad format it uses, what tokens it produces/consumes
- **Each edge** carries metadata: how the source skill is used by the target, codomain/domain type annotations

The graph enforces **structural constraints** that prevent ill-formed curricula. The key insight: if the graph is correct, the curriculum is correct. Mistakes happen when the graph is incomplete (missing nodes/edges), not when the training procedure is wrong.

---

## Universal Type System

The CTKG uses a small set of universal primitives that compose into any domain-specific type. This ensures that "digit," "price," or "chemical bond" are not opaque strings but structurally defined objects the system can reason about.

### The Problem

Without universal primitives, every new domain requires reprogramming. `DOT_TEN_sequence` means nothing to the computer — it's just a label on a wire. If a customer in finance creates `price_sequence`, the system would happily connect it to `column_scratchpad` if the strings accidentally matched. We need types with structure, not just names.

### The Insight

At the level of a language model processing tokens, everything reduces to: **pick one symbol from a finite set, then maybe pick another.** The structure is in how those sets relate to each other.

### Type Primitives (IMPLEMENTED)

**Type constructors** — how to build types:

| Constructor | Syntax | Meaning | Example |
|-------------|--------|---------|---------|
| `symbol` | `symbol(a, b, c, ...)` | Element from a named finite set | `digit = symbol(0..9)` |
| `nat` | `nat` | Natural number | Built-in |
| `bool` | `bool` | True / false | Built-in |
| `seq` | `seq(T)` | Variable-length sequence of T | `count_seq = seq(digit)` |
| `tuple` | `tuple(T1, T2, ...)` | Fixed-length product | `digit_pair = tuple(digit, digit)` |
| `tagged` | `tagged(l1: T1, l2: T2)` | Sum type / variant | `result = tagged(ok: nat, err: bool)` |
| `expr` | `expr` | Quoted expression (code-as-data) | Built-in |
| `proposition` | `proposition` | Logical statement | Built-in |

**Structure annotations** — properties on types:

| Annotation | Meaning | Unlocks |
|------------|---------|---------|
| `ordered` | Elements have a total order | `succ`, `pred`, `compare` |
| `invertible` | Operations can be reversed | Adjunction support |
| `commutative` | `a op b = b op a` | Symmetry optimization |
| `associative` | `(a op b) op c = a op (b op c)` | Regrouping |
| `periodic(k)` | Wraps around every k elements | Modular arithmetic |
| `metric` | Elements have a distance function | Similarity, approximation |

**Builtin types** — always available without declaration: `nat`, `bool`, `expr`, `proposition`.

### Three Levels of Process Primitives

Every concept's `process` field uses operations from three levels. Higher levels subsume lower ones.

**Level 1 — Computation** (applying known operations):

| Primitive | Signature | Meaning |
|-----------|-----------|---------|
| `succ(x)` | ordered T → T | Next element (non-wrapping integer) |
| `pred(x)` | ordered T → T | Previous element (non-wrapping integer) |
| `compare(a, b)` | ordered T × T → {GT, LT, EQ} | Comparison |
| `lookup(concept, *args)` | concept × args → tuple | Call a named concept (returns tuple) |
| `fold(n, init, step)` | nat × S × (S→S) → S | Apply step n times from init |
| `fold_until(max, init, step, stop)` | nat × S × (S→S) × (S→bool) → S | fold with early exit; bounded by max |
| `fn(param, body)` | — | Lexical-scope closure (special form) |
| `pair(a, b)` | T × U → tuple(T,U) | Two-element tuple constructor |
| `triple(a, b, c)` | T × U × V → tuple(T,U,V) | Three-element tuple constructor |
| `first(s)` | tuple(T,...) → T | First element |
| `second(s)` | tuple(T,...) → T | Second element |
| `third(s)` | tuple(T,...) → T | Third element |
| `emit(x, ...)` | T... → output | Produce output tuple |
| `if(c, then, else)` | bool × T × T → T | Conditional |

**Level 2 — Logic** (reasoning about values):

| Primitive | Signature | Meaning |
|-----------|-----------|---------|
| `equal(a, b)` | T × T → bool | Equality test |
| `and(a, b)` | bool × bool → bool | Conjunction |
| `or(a, b)` | bool × bool → bool | Disjunction |
| `not(a)` | bool → bool | Negation |
| `implies(a, b)` | bool × bool → bool | Implication |
| `forall(x in T, P(x))` | T × (T→bool) → bool | Universal (finite) |
| `exists(x in T, P(x))` | T × (T→bool) → bool | Existential (finite) |

**Level 3 — Transform** (operating on expressions as data):

Implemented as **Level C — Symbolic AST** (tagged Python tuples; interpreter.py):

| Primitive | Signature | Meaning |
|-----------|-----------|---------|
| `sym_num(n)` | int → expr | Numeric constant node |
| `sym_var(name)` | str → expr | Variable node (use literal X, Y, Z, T) |
| `sym_add(e1, e2)` | expr × expr → expr | Addition node (constant-folds) |
| `sym_sub(e1, e2)` | expr × expr → expr | Subtraction node (= sym_add + sym_neg) |
| `sym_mul(e1, e2)` | expr × expr → expr | Multiplication node (constant-folds) |
| `sym_pow(e, n)` | expr × nat → expr | Power node (constant-folds) |
| `sym_neg(e)` | expr → expr | Negation node |
| `sym_eval(expr, var, val)` | expr × str × int → int | Numerically evaluate at a point |
| `sym_diff(expr, var)` | expr × str → expr | Differentiate (power + product rule) |
| `sym_subst(expr, var, replacement)` | expr × str × expr → expr | Substitute sub-expression |
| `sym_str(expr)` | expr → str | Human-readable string |

Expressions are tagged Python tuples: `('NUM', n)`, `('VAR', 'X')`, `('ADD', e1, e2)`, `('MUL', e1, e2)`, `('POW', e, n)`, `('NEG', e)`. Constructors constant-fold (e.g., `sym_add(('NUM',2), ('NUM',3))` → `('NUM',5)`).

**Still unimplemented from original Level 3 plan:** `quote`, `match`, `rewrite`, `decompose`, `compose` — needed for grammar induction and algorithm synthesis. `sym_match` is the next required primitive for linguistics/pattern tasks.

Level 1 lets you DO arithmetic. Level 2 lets you PROVE things about arithmetic. Level 3 lets you IMPROVE algorithms (meta-reasoning, code-as-data).

### Primitive Minimality

The implemented primitives form a practical set. The theoretical minimum (fewest primitives for equivalent expressiveness):

**Definitively redundant — can be removed with zero expressiveness loss:**
- `triple(a,b,c)` = `pair(a, pair(b, c))`
- `third(s)` = `second(second(s))`
- `sym_sub(e1, e2)` = `sym_add(e1, sym_neg(e2))`
- `equal(a, b)` = the `==` syntactic operator already in the language

**Derivable but kept for readability:**
- `pred(n)` — derivable as `fold_until` with stop on succ reaching n; kept because it's a genuine primitive (biological number sense)
- `compare(a, b)` — derivable from repeated succ/pred; kept for performance and clarity
- `fold(n, init, step)` — special case of `fold_until` with no stop condition; kept because simpler templates synthesize addition, multiplication
- `sym_neg(e)` — = `sym_sub(sym_num(0), e)`; kept for convenience

**Genuine theoretical minimum (17 forms):**
`succ`, `fold_until`, `fn`, `if`, `pair`, `first`, `second`, `lookup`, `emit`,
`sym_num`, `sym_var`, `sym_add`, `sym_mul`, `sym_pow`, `sym_diff`, `sym_eval`, `sym_subst`

**Still missing for full completeness:**
- `sym_match` — pattern matching on expression trees (needed for grammar/linguistics)
- Float arithmetic — needed for real-valued physics/control domains
- Sequence primitives beyond pairs — `seq_cons`, `seq_head`, `seq_tail`

### Type Validation (IMPLEMENTED)

The validator checks that every type name used in a concept's `input`/`output` resolves against the type registry. This catches errors at parse time:

```
CTKG validation error:
  Concept 'foo' input type 'nonexistent_type' not defined in type registry
```

### Cross-Domain Universality

The same primitives work for any domain:

```
-- Finance
type price = symbol(0..999) ordered metric
type signal = symbol(BUY, SELL, HOLD)

-- Chemistry
type element = symbol(H, He, Li, ...) ordered
type bond = tagged(single: tuple(element, element), double: tuple(element, element))

-- Logic
type prop = proposition
type proof_step = tuple(prop, rule_name, seq(prop))
```

---

## Four Use Cases (Ordered by Implementation Priority)

### 1. Curriculum Compiler (IMMEDIATE)

The graph compiles into a valid training curriculum. This is analogous to a build system:

- **Topological sort** of the graph = valid stage ordering
- **Type checking** on edges = codomain/domain matching (the output format of skill A must match the input format of skill B)
- **Completeness check** = every internal node has all prerequisite edges satisfied
- **Cycle detection** = impossible dependencies are flagged
- **Missing import** = a node references a concept not in the graph

**What Mistake #44 would have looked like:**
```
CTKG validation error:
  Node 'single_digit_addition' has no incoming edge for:
    - ordinality (digits must have order for counting-up)
    - comparison (digits must be comparable quantities)
  Node 'single_digit_addition' is marked as fact_stage but has >20 entries (155).
    Hint: decompose into sub-graph with counting-based derivation.
```

The "compiler" catches the bug before training runs. No wasted GPU hours.

**Concrete deliverable:** `ctkg.validate()` that checks all structural constraints and reports errors in the format above.

### 2. Structured Training Data Generator (NEAR-TERM)

Each node in the graph is associated with a `ProblemGenerator` (from the scratchpad framework). The graph structure determines:

- **What to train on:** The current frontier node + replay from ancestors
- **In what order:** Topological sort, with advancement gates (>=95% test)
- **With what replay:** When training node B, replay problems from all ancestors of B (weighted by recency and graph distance)
- **Format consistency:** The scratchpad format for node B must literally contain the formats of its prerequisite nodes as sub-sequences

This replaces the manual stage numbering in `train_arithmetic.py` with automatic curriculum generation from the graph.

**Concrete deliverable:** `ctkg.generate_curriculum()` that returns an ordered list of `(stage_number, generator, replay_generators)` tuples.

### 3. External Knowledge Store / Long-Term Memory (MEDIUM-TERM)

The CTKG stores **knowledge** (structured relationships between concepts), not just text. At inference time, the relevant subgraph is prefetched and injected into the model's context. This differs from RAG in that retrieval is structure-aware:

- For a problem of type X, the CTKG knows exactly which prerequisite concepts are relevant
- The retrieval path follows the graph edges, not embedding similarity
- The injected context includes not just facts but the **derivation structure** (how to reduce the problem to simpler sub-problems)

This compensates for our 4GB VRAM constraint: the model's weights encode general reasoning patterns, while the CTKG in system RAM holds the specific knowledge. The model doesn't need to memorize arithmetic facts if the CTKG can provide the relevant subgraph at inference time.

**Analogy to DeepSeek's Engram:** Both systems separate fast-access knowledge (GPU) from large-capacity knowledge (RAM/disk). The difference: Engram uses learned routing to select which knowledge to load; CTKG uses graph structure. Graph structure is deterministic and auditable — you can prove which knowledge is relevant without relying on a learned router that might hallucinate.

**Concrete deliverable:** `ctkg.retrieve(problem_type)` that returns the relevant subgraph as a token sequence suitable for prepending to the model's input.

### 4. Computational Aid / Deterministic Solver (LONG-TERM)

For problems where the CTKG has a complete derivation path from axioms to answer, the graph can **execute the computation deterministically** instead of asking the neural network to predict it. This is computationally cheaper, more reliable, and produces provably correct results.

Example: For a 50-step calculus problem, the CTKG:
1. Identifies which rules apply (chain rule, product rule, etc.)
2. Applies them in the correct order (following the graph structure)
3. Returns the answer with a proof trace (the sequence of morphisms applied)

The neural network's role shifts from "compute the answer" to "decide which subgraph to invoke" — a routing/planning task that's much easier than the computation itself.

**Concrete deliverable:** `ctkg.solve(problem, max_steps)` that returns a `(answer, proof_trace)` if the problem is within the graph's coverage.

---

## Categorical Structure

### Objects, Morphisms, Composition

```
Category ARITH:
  Objects:
    counting         -- count physical objects (DOTs, TENs) → digit
    ordinality       -- digits have a natural order (successor/predecessor)
    comparison       -- digits represent comparable quantities (GT/LT/EQ)
    addition         -- a + b = count up b steps from a
    subtraction      -- a - b = count down b steps from a
    place_value      -- positional notation (ones, tens, hundreds)
    column_addition  -- multi-digit addition via column-wise single-digit ops
    column_subtraction -- multi-digit subtraction with borrowing

  Morphisms:
    counting → ordinality       -- counting defines digit ordering
    ordinality → comparison     -- ordering enables comparison
    ordinality → addition       -- successor function enables counting-up
    ordinality → subtraction    -- predecessor function enables counting-down
    addition → column_addition  -- single-digit addition is the column operation
    subtraction → column_subtraction
    place_value → column_addition   -- positional notation enables column decomposition
    place_value → column_subtraction

  Composition:
    counting → ordinality → addition → column_addition
    (transitive: counting is a prerequisite for column addition)
```

### Type System

**See "Universal Type System" section above for the complete reference.**

Types are now defined using universal constructors (`symbol`, `seq`, `tuple`, `tagged`) with structure annotations (`ordered`, `metric`, etc.). The validator checks that all type names in concept inputs/outputs resolve against the type registry.

Example from `arithmetic.ctkg`:
```
type digit = symbol(0, 1, 2, 3, 4, 5, 6, 7, 8, 9) ordered
type carry = symbol(0, 1)
type arith_result = tuple(carry, digit)

concept single_digit_addition
  input digit op digit
  output arith_result        -- type validated against registry
  ...
```

The old edge `counting → single_digit_addition_fact` was INVALID because:
- counting output: `digit` (as quantity label)
- fact addition input: `digit` (as arbitrary symbol in lookup table)
- Types DON'T match: counting's digit-as-quantity ≠ fact table's digit-as-symbol

### Functors (Cross-Domain Transfer)

A **functor** maps an entire subgraph from one domain to another while preserving structure. Example:

```
F: ARITH → LOGIC
  F(counting) = proposition_counting  -- count propositions satisfying a predicate
  F(ordinality) = logical_ordering    -- partial order on propositions
  F(addition) = disjunction           -- combining propositions
  F(column_addition) = proof_composition  -- combining sub-proofs
```

The functor preserves composition: if `counting → addition` in ARITH, then `proposition_counting → disjunction` in LOGIC. This means a model trained on the arithmetic curriculum could transfer its compositional reasoning to logic.

This is speculative and long-term, but the point is: the CTKG structure is domain-independent. The same graph operations (validate, compile, retrieve) work for any domain.

---

## Data Model

### Node

```python
@dataclass
class Concept:
    name: str                          # unique identifier
    description: str                   # what this concept teaches
    domain: str                        # which category (e.g., 'arithmetic')

    # Type annotations
    input_type: List[str]              # token types consumed
    output_type: List[str]             # token types produced

    # Scratchpad integration
    scratchpad_format: str             # template showing work area format
    generator_class: Optional[str]     # ProblemGenerator subclass name
    n_result: Union[int, str]          # fixed int or 'variable'

    # Verification
    pass_threshold: float = 0.95       # test accuracy to consider learned
    max_epochs: int = 100              # budget before declaring failure

    # Classification
    is_atomic: bool = False            # True only for genuinely irreducible facts
    # (no is_fact_stage — that's a design smell flag, not a classification)
```

### Edge

```python
@dataclass
class Prerequisite:
    source: str                        # prerequisite concept name
    target: str                        # dependent concept name
    role: str                          # how source is used ("successor for counting-up")
    codomain_type: List[str]           # what source produces in this context
    domain_type: List[str]             # what target expects in this context
```

### Graph

```python
class KnowledgeGraph:
    concepts: Dict[str, Concept]
    prerequisites: List[Prerequisite]

    def add_concept(self, concept: Concept) -> None
    def add_prerequisite(self, prereq: Prerequisite) -> None

    # Curriculum compiler (Use Case 1)
    def validate(self) -> List[ValidationError]
    def topological_sort(self) -> List[str]
    def generate_curriculum(self) -> List[CurriculumStage]

    # Queries
    def ancestors(self, name: str) -> Set[str]       # all transitive prereqs
    def descendants(self, name: str) -> Set[str]      # all that depend on this
    def frontier(self, learned: Set[str]) -> Set[str]  # next teachable concepts
    def missing_for(self, name: str, learned: Set[str]) -> Set[str]

    # Knowledge store (Use Case 3)
    def retrieve(self, problem_type: str) -> SubGraph
    def inject_context(self, problem_type: str, vocab: Vocab) -> List[int]
```

### Validation Rules

```python
class ValidationError:
    """Base class for graph validation errors."""

class MissingPrerequisite(ValidationError):
    """Node X depends on concept Y, but Y has no node in the graph."""

class TypeMismatch(ValidationError):
    """Edge source.output_type doesn't match edge target.input_type."""

class LargeFactTable(ValidationError):
    """Node marked is_atomic=True but has >20 entries. Likely decomposable."""

class OrphanNode(ValidationError):
    """Internal node with no incoming edges (should be atomic or has missing prereqs)."""

class CycleDetected(ValidationError):
    """Circular dependency — impossible to teach in any order."""

class FormatInconsistency(ValidationError):
    """Target scratchpad doesn't contain source scratchpad as sub-sequence."""
```

---

## The Arithmetic Domain Graph

The first concrete instantiation. This replaces the manual stage list in CONTINUATION.md:

```
                    counting
                   /        \
          counting_dots    counting_tens
                   \        /
                combined_counting
                       |
                  ordinality (successor/predecessor)
                  /         \
          comparison       addition_as_counting
                             |           \
                    subtraction_as_counting  \
                             |               |
                     column_subtraction   column_addition
                              \              /
                          two_digit_arithmetic
```

Each node maps to a ProblemGenerator:

| Node | Generator | Stage | Problem Count |
|------|-----------|-------|---------------|
| counting_dots | QueryCountingGenerator(query='DOT') | 1a | 100 |
| counting_tens | QueryCountingGenerator(query='TEN') | 1b | 100 |
| combined_counting | CombinedCountingGenerator | 2 | 100 |
| ordinality | SuccessorGenerator | 3 | 20 |
| comparison | ComparisonGenerator | 4 | 100 |
| addition_as_counting | CountingAdditionGenerator | 5 | 100 |
| subtraction_as_counting | CountingSubtractionGenerator | 6 | 55 |
| column_addition | TwoDigitSingleGenerator(op='+') | 7a | 900 |
| column_subtraction | TwoDigitSingleGenerator(op='-') | 7b | 900 |
| two_digit_arithmetic | TwoDigitGenerator | 8 | ~12,195 |

The topological sort produces a valid curriculum. Multiple valid orderings exist (e.g., comparison could come before or after addition), but the graph ensures no ordering violates prerequisites.

---

## Integration with Existing Code

### With scratchpad framework

The CTKG doesn't replace the scratchpad framework — it orchestrates it. Each `Concept` node references a `ProblemGenerator`. The CTKG's `generate_curriculum()` produces a list of generators in valid topological order, which `train_arithmetic.py` consumes.

### With train_arithmetic.py

Currently, `train_arithmetic.py` has a hardcoded stage list. The CTKG replaces this with:

```python
graph = build_arithmetic_graph()
errors = graph.validate()
if errors:
    for e in errors:
        print(f"CTKG ERROR: {e}")
    sys.exit(1)

curriculum = graph.generate_curriculum()
for stage in curriculum:
    train_stage(model, stage.generator, stage.replay_generators, ...)
```

### With future models

The CTKG is model-agnostic. It produces curricula (ordered generators with replay policies), not model-specific training code. Any model that consumes token sequences can use it.

---

## Implementation Plan

### Phase 1: Graph data structures + validation (Curriculum Compiler)

- `graph.py` — `Concept`, `Prerequisite`, `KnowledgeGraph`, `ValidationError` subclasses
- `validate()` — all six validation rules
- `topological_sort()` — Kahn's algorithm
- `arithmetic.py` — `build_arithmetic_graph()` instantiating the arithmetic domain
- Tests: validate catches missing prerequisites, type mismatches, large fact tables

### Phase 2: Curriculum generation

- `generate_curriculum()` — topological sort + replay policy
- `frontier()` — given learned concepts, which are next teachable?
- Integration with `train_arithmetic.py` — replace hardcoded stage list

### Phase 3: External knowledge store

- `retrieve()` — given a problem type, return relevant subgraph
- `inject_context()` — convert subgraph to token sequence for model input
- Benchmark: does injecting the relevant subgraph improve generalization?

### Phase 4: Deterministic solver

- `solve()` — follow derivation path from axioms to answer
- Proof trace generation — the sequence of morphisms applied
- Hybrid mode: model plans, CTKG executes

---

## Epiplexity: Measuring What a Stage Teaches

**Reference:** Alemi (2025), "Epiplexity and the Solomonoff Prior."

**Epiplexity** (S_preq) = area under the loss curve above the final loss. Formally, for a training run of T epochs with losses l_1, ..., l_T and final loss l_T:

```
S_preq = sum_{i=1}^{T} (l_i - l_T)
```

This measures the **structural information** the model extracted from the data during training. High S_preq means the model slowly discovered rich compositional structure. Low S_preq means the stage was trivially memorizable or contained little learnable structure.

### Why This Matters for the CTKG

Each node in the graph should carry an empirical epiplexity score from training runs. This gives a diagnostic beyond pass/fail (test accuracy):

| S_preq | Interpretation | Action |
|--------|---------------|--------|
| High (slowly declining loss) | Stage teaches rich structure | Good — this is where generalization comes from |
| Near-zero (rapid convergence) | Stage is trivial or memorizable | Warning — model may pass without learning reusable circuits |
| High final loss (never converges) | Stage is too hard or missing prereqs | Error — check prerequisites in graph |

**The "too easy" trap:** A stage with low S_preq can pass at 95% test accuracy while contributing nothing to downstream composition. Stage 3 (successor, 20 problems) is a likely candidate — the model may memorize all 20 facts without internalizing ordinality. Epiplexity catches this: if S_preq ≈ 0, the stage should be redesigned to require actual pattern learning (more combinatorial complexity, or compositional structure with earlier stages).

**Relation to curriculum compiler:** The `validate()` method can flag stages where empirical S_preq is below a threshold, suggesting the stage is too simple to teach what it claims. This is a data-driven complement to the structural (graph-based) validation.

### Updated Data Model

```python
@dataclass
class Concept:
    # ... existing fields ...

    # Epiplexity diagnostics (populated after training runs)
    empirical_epiplexity: Optional[float] = None   # S_preq from training
    epiplexity_threshold: float = 1.0              # minimum expected S_preq
```

---

## Factorization Order as a Design Variable

**Reference:** Alemi (2025) — Chess factorization experiment.

The **order in which tokens appear** in a sequence affects what representations the model learns. Same data, different ordering → different epiplexity → different OOD transfer.

The chess experiment showed:
- **Forward factorization** (moves → board): easy, mechanical application. Low epiplexity.
- **Reverse factorization** (board → moves): hard, requires induction. Higher epiplexity AND better OOD transfer to new positions.

### Implications for Scratchpad Design

Each scratchpad format is a **factorization choice**. The question is: does this ordering force the model to build rich representations, or does it allow mechanical application?

**Current forward factorization (Stage 3):**
```
3 + 4 WORK 0 7       — see operands, produce result
```
The model applies the operation mechanically. Each output token follows deterministically from the inputs and the operation rule.

**Reverse factorization (Stage 3):**
```
? + 4 = 0 7 WORK 0 3   — see result + one operand, find the missing one
3 + ? = 0 7 WORK 0 4   — same, other operand missing
```
The model must **induct** the missing operand from the result. This requires understanding the inverse relationship — what addition MEANS, not just how to apply it.

### Design Principle

For each concept node in the graph, consider both forward and reverse factorizations:
- **Forward:** given inputs, produce outputs (application)
- **Reverse:** given outputs (and partial inputs), produce missing inputs (induction)

Training on both forces bidirectional understanding. The graph should track which factorizations are available for each node.

### Updated Data Model

```python
@dataclass
class Concept:
    # ... existing fields ...

    # Factorization support
    supports_reverse: bool = False                 # can this concept be presented in reverse?
    reverse_generator_class: Optional[str] = None  # generator for reverse problems
```

```python
@dataclass
class Prerequisite:
    # ... existing fields ...
    invertible: bool = False    # is this morphism reversible?
```

---

## Categorical Structure — Honest Assessment

The current implementation uses category theory vocabulary (objects, morphisms, composition) but not its substance. What we have is a well-validated DAG. Here's what we need to add and why.

### What we have

| Feature | Status | Notes |
|---------|--------|-------|
| Objects (concepts) | Implemented | Concept dataclass |
| Morphisms (prerequisites) | Implemented | Prerequisite dataclass |
| Composition (transitive) | Implemented | Via ancestors() / descendants() |
| Validation (7 error types) | Implemented | validate() in graph.py |
| Curriculum generation | Implemented | topological_sort + generate_curriculum |
| Sheaf consistency | Implemented | sheaf_check(), sheaf_merge(), SheafViolation |
| Interfaces | Implemented | Interface dataclass, DSL `interface` blocks |

### What we need (prioritized by practical impact)

**1. Typed composition (HIGH priority)**

Current type system: list-of-string equality (`['digit'] == ['digit']`). No semantic distinction between "digit as quantity" vs "digit as ordinal" vs "digit as symbol."

Need: semantic type labels so the compiler catches that `counting.digit_as_quantity ≠ fact_table.digit_as_symbol`, which is exactly what Mistake #44 was about.

**2. Computation morphisms (HIGH priority)**

Morphisms currently carry only metadata (role, type annotations). They need to carry **computation rules** — how the target concept's process composes the source concept's process. This is what enables auto-generating problems at higher stages from lower stages' rules.

**3. Functors (MEDIUM priority)**

Structure-preserving maps between domains. F: Arithmetic → Logic maps concepts to concepts and prerequisites to prerequisites while preserving composition. Needed for cross-domain curriculum transfer.

**4. Adjunctions (MEDIUM priority)**

Forward/inverse pairs with round-trip verification. Currently approximated by `supports_reverse` flag but not formalized. Needed for teaching bidirectional understanding.

**5. Identity morphisms (LOW priority)**

id_X: X → X for every concept. Would model "review" as a formal operation. Not needed until spaced repetition is formalized.

**6. Natural transformations, limits, colimits, pullbacks (LOW priority)**

Theoretical elegance. Natural transformations compare competing functors. Limits/colimits generalize constraint satisfaction. Not needed until multiple domains are active and we need to prove structural equivalences.

---

## Sheaf Consistency (IMPLEMENTED)

When composing multiple domains into a single knowledge graph, we need a guarantee that overlapping definitions agree. This is where sheaf theory provides the right mathematical framework.

### The Problem

Suppose the arithmetic domain defines `digit = symbol(0..9) ordered` and a separate finance domain also defines `digit = symbol(0..9)` but without the `ordered` annotation. If we naively merge these, the `digit` type in the merged graph is ambiguous — does it have ordering or not? Concepts from arithmetic that rely on ordering would silently break.

More subtly: two domains might define a concept with the same name but different input/output types. Merging them produces a graph where the concept's type depends on which domain you ask — a contradiction that no amount of validation can catch after the merge.

### The Sheaf Framework

In sheaf theory:
- Each **domain** is an "open set" in the topology of knowledge
- A **section** over a domain is a consistent assignment of data (types, concepts, prerequisites) to that domain
- The **restriction** of a section to a smaller domain is just subgraph extraction
- The **gluing axiom** says: if two sections agree on their overlap, they can be composed into a global section

For the CTKG, this translates to:
- Each `.ctkg` file defines a section over its domain
- Overlapping type names must have identical structure (same constructor, params, annotations)
- Overlapping concept names must have compatible input/output types
- If these conditions hold, the merge is well-defined; if not, we get a `SheafViolation` error

### Implementation

**Interface declarations** — each domain declares what it exports:

```
interface arithmetic
  exports types digit carry op cmp_result arith_result column_result
  exports concepts successor predecessor comparison single_digit_addition
```

**Type compatibility** — structural equality on `TypeDef`:

```python
def types_compatible(a: TypeDef, b: TypeDef) -> bool:
    return (a.constructor == b.constructor
            and a.params == b.params
            and a.annotations == b.annotations)
```

**Sheaf check** — `KnowledgeGraph.sheaf_check(other)` returns a list of `SheafViolation` errors for any overlapping types or concepts that disagree.

**Sheaf merge** — `KnowledgeGraph.sheaf_merge(source)` first checks consistency, then merges if clean. If violations are found, the target graph is left unmodified and the violations are returned.

### Example: Compatible Merge

Arithmetic defines `digit = symbol(0..9) ordered`. Logic uses builtins (`bool`, `nat`) but defines its own types (`connective`, `truth_value`). No overlap on custom types → merge succeeds cleanly.

```python
arith = build_arithmetic_graph()
logic = build_logic_graph()
violations = arith.sheaf_merge(logic)
assert violations == []  # clean merge
```

### Example: Sheaf Violation

```python
graph_a = parse("type status = symbol(OK, ERR)")
graph_b = parse("type status = symbol(GOOD, BAD, UNKNOWN)")
violations = graph_a.sheaf_check(graph_b)
# → [SheafViolation: Type 'status' defined incompatibly across domains]
```

### DSL Grammar Addition

```
interface    = 'interface' NAME NL (interface_field NL)*
interface_field = 'exports' ('types' | 'concepts') NAME+
```

### Data Model

```python
@dataclass
class Interface:
    name: str
    types: List[str]      # exported type names
    concepts: List[str]   # exported concept names

class SheafViolation(ValidationError):
    """Overlapping definitions are incompatible across domains."""
```

---

## Three Curriculum Patterns

Every category theory concept maps to one of three curriculum patterns. The CTKG DSL must express all three.

### Pattern 1: Process (composition, functors)

The scratchpad shows step-by-step execution of the concept. The model learns by training on process tokens.

Example — composition (column addition composes single-digit addition):
```
2 3 + 4 8 WORK 3 + 8 + 0 = 1 1 SEP 2 + 4 + 1 = 0 7 SEP 0 7 1
                    ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
                    Stage 3 process     Stage 3 process (reused)
```

Example — functor (mapping arithmetic to a new domain):
```
DOMAIN arithmetic: 3 + 4 WORK 4 5 6 7 = 0 7
MAP arithmetic logic: + → OR, digit → proposition
DOMAIN logic: P OR Q WORK [disjunction process tokens]
```

### Pattern 2: Relationship (adjunctions, natural transformations)

The scratchpad shows two related computations and a VERIFY step proving their relationship. The model learns structural linkage.

Example — adjunction (forward + inverse with round-trip):
```
FORWARD: 3 + 4 WORK 4 5 6 7 = 0 7
INVERSE: 7 - 4 WORK 6 5 4 = 0 3
VERIFY: 3 + 4 - 4 = 3 ✓
```

Example — natural transformation (two functors agree):
```
PATH1: arithmetic → sets via F1: addition → union
PATH2: arithmetic → sets via F2: addition → multiset_sum
VERIFY: F1(3+4) ≡ F2(3+4)
```

### Pattern 3: Constraint (limits, pullbacks)

The scratchpad shows multiple constraints and a WORK step finding the object satisfying all of them. The model learns constraint satisfaction.

Example — limit (find the most specific common structure):
```
CONSTRAINT: X is a group
CONSTRAINT: X has total order
WORK: group ∩ total_order = ordered_group
VERIFY: ordered_group satisfies both
```

---

## CTKG DSL — Domain-Specific Language

### Motivation

The CTKG is intended as a commercial system. Customers must be able to define domains without writing Python — just data. A custom DSL is more concise than JSON/YAML, more readable for domain experts, and gives us control over syntax evolution.

### Design principles

1. **Indentation-based** — no braces, no XML tags
2. **Keywords over punctuation** — `requires`, `map`, `verify` not `->`, `=>`, `<=>`
3. **Extensible** — new keywords can be added without breaking existing files
4. **Comments** — `--` line comments
5. **Sections** — top-level keywords (`concept`, `functor`, `adjunction`) start blocks

### Example

```
-- Arithmetic domain: counting through column operations

concept query_counting
  domain arithmetic
  atomic
  description "Count DOTs or TENs given a query"
  input DOT_TEN_sequence query_token
  output digit
  process count(filter(input, query))
  threshold 0.95

concept combined_counting
  domain arithmetic
  description "Count both via successor count-up with STOP"
  input DOT_TEN_sequence
  output count_up_sequence count_up_sequence
  requires query_counting via "individual counting composes into dual"
  process
    dot_count = count_up(count(filter(input, DOT)))
    ten_count = count_up(count(filter(input, TEN)))

concept single_digit_addition
  domain arithmetic
  description "Single-digit +/- producing carry + ones"
  input digit op digit
  output carry digit
  requires combined_counting via "counting grounds digit semantics"
  reversible
  process
    forward: carry, ones = apply_op(a, op, b)
    inverse: missing = invert_op(result, known, op)

-- Cross-domain transfer

functor arith_to_logic
  from arithmetic to logic
  map query_counting -> proposition_counting
  map addition -> disjunction
  map column_addition -> proof_composition
  preserves composition

-- Forward/inverse pairing

adjunction addition_subtraction
  forward addition
  inverse subtraction
  unit a + b - b = a
  counit a - b + b = a
```

### Grammar (informal)

```
file         = (statement NL)*
statement    = typedef | concept | functor | adjunction | interface | comment
comment      = '--' TEXT

typedef      = 'type' NAME '=' CONSTRUCTOR
             | 'type' NAME '=' CONSTRUCTOR '(' param_list ')' (ANNOTATION)*
CONSTRUCTOR  = 'symbol' | 'nat' | 'bool' | 'seq' | 'tuple'
             | 'tagged' | 'expr' | 'proposition'
param_list   = PARAM (',' PARAM)*
ANNOTATION   = 'ordered' | 'invertible' | 'commutative'
             | 'associative' | 'metric' | 'periodic(' INT ')'

concept      = 'concept' NAME NL (concept_field NL)*
concept_field = 'domain' NAME
             | 'atomic'
             | 'description' STRING
             | 'input' type_list
             | 'output' type_list
             | 'requires' NAME 'via' STRING
             | 'reversible'
             | 'threshold' FLOAT
             | 'max_epochs' INT
             | 'process' (EXPR | NL (INDENT EXPR NL)*)

functor      = 'functor' NAME NL (functor_field NL)*
functor_field = 'from' NAME 'to' NAME
             | 'map' NAME '->' NAME
             | 'preserves' NAME

adjunction   = 'adjunction' NAME NL (adj_field NL)*
adj_field    = 'forward' NAME
             | 'inverse' NAME
             | 'unit' EXPR
             | 'counit' EXPR

interface    = 'interface' NAME NL (interface_field NL)*
interface_field = 'exports' ('types' | 'concepts') NAME+

type_list    = NAME (NAME)*
NAME         = [a-zA-Z_][a-zA-Z0-9_]*
STRING       = '"' [^"]* '"'
FLOAT        = [0-9]+ '.' [0-9]+
INT          = [0-9]+
EXPR         = (any text until end of line or dedent)
INDENT       = 2+ spaces deeper than parent
NL           = newline
```

### What the parser produces

The parser reads `.ctkg` files and emits Python graph objects:
- `type` lines → `TypeDef` instances (registered in graph's type registry)
- `concept` blocks → `Concept` instances (with `process` lines preserved)
- `requires` fields → `Prerequisite` instances
- `functor` blocks → `Functor` instances
- `adjunction` blocks → `Adjunction` instances
- `interface` blocks → `Interface` instances (exported types and concepts)

The graph is constructed by loading one or more `.ctkg` files, then calling `validate()`.
Validation checks type resolution: all type names in concept inputs/outputs must exist in the type registry. Multi-domain merging uses `sheaf_merge()` to enforce consistency on overlapping definitions.

---

## Commercial Architecture

Three-layer design where customers define WHAT to teach, the system handles HOW.

### Layer 1 — Graph (customer provides)

`.ctkg` files defining concepts, prerequisites, functors, adjunctions. Pure data, no code. The customer brings domain expertise.

### Layer 2 — Computation rules (customer provides)

The `process` field in each concept defines how outputs derive from inputs. Rules reference other concepts by name — the system resolves references using the graph and composes computations automatically.

Key property: a composite concept's process is derivable from its prerequisites' processes. The customer defines atomic rules; the system derives the full computation tree.

### Layer 3 — Engine (we provide)

- **Parser**: `.ctkg` → Python graph objects
- **Validator**: Graph structure, type matching, prerequisite completeness
- **Curriculum compiler**: Topological sort + replay policy
- **Problem generator**: Samples inputs, evaluates computation rules, produces process tokens + answers
- **Trainer**: Model-agnostic training loop
- **Diagnostics**: Per-stage epiplexity, per-token accuracy, train/test breakdown

The engine is domain-agnostic. Same engine for arithmetic, electronics, formal logic, or any domain the customer defines.

---

## Updated Implementation Plan

### Phase 1: Graph data structures + validation ✅ DONE

- `graph.py` — Concept, Prerequisite, KnowledgeGraph, 6 validation errors
- `domains/arithmetic.py` — 5 implemented concepts (now loads from .ctkg)

### Phase 2: DSL parser + universal types ✅ DONE

- Grammar specification with `type` keyword
- `parser.py` — tokenizer + parser for `.ctkg` files (type, concept, functor, adjunction)
- `TypeDef` dataclass with constructor, params, annotations
- Builtin types: nat, bool, expr, proposition
- `UndefinedType` validation error — catches unresolved type names
- `domains/arithmetic.ctkg` — 9 concepts, 13 types, 12 prerequisites, 1 adjunction
- `test_parser.py` — 6 tests, all passing
- `arithmetic.py` → thin wrapper that loads from .ctkg

### Phase 2.5: Sheaf consistency + multi-domain ✅ DONE

- `Interface` dataclass — declares exported types and concepts per domain
- `SheafViolation` error — caught when overlapping definitions disagree
- `types_compatible()` — structural equality on TypeDef
- `sheaf_check()` — compare two graphs for overlap consistency
- `sheaf_merge()` — merge with consistency enforcement (refuses on violation)
- `interface` block in DSL — `exports types` / `exports concepts`
- `domains/logic.ctkg` — 5 logic concepts, 8 types, interface declaration
- `test_parser.py` — 11 tests, all passing (5 new sheaf tests)

### Phase 2.7: Probabilistic structure (Markov category) ✅ DONE

The CTKG now supports probabilistic reasoning via three mechanisms grounded
in categorical probability theory.

**Markov kernel weights on prerequisites:**
- `Prerequisite.transfer_probability` (default 1.0) = P(can learn target | mastered source)
- DSL syntax: `requires NAME via "ROLE" [0.75]` — trailing `[probability]` is optional
- Categorically: morphisms in FinStoch (Fritz, Advances in Mathematics 2020)

**d-Separation (Bayes-ball algorithm):**
- `KnowledgeGraph.d_separated(x, y, given)` — conditional independence test
- Uses Shachter (1998) Bayes-ball algorithm
- Categorically: the conditional independence relation in the Markov category
  (Fritz & Klingler, JMLR 2023)
- Use case: "given that the student has mastered set Z, is their performance
  on concept X independent of concept Y?"

**Entropy (Baez-Fritz-Leinster characterisation):**
- `concept_entropy(name)` — H(C) = log2(|problem_space|), the maximum uncertainty
- `conditional_entropy(name, learned)` — H(C | learned), remaining uncertainty
  given prerequisites, weighted by transfer probability
- `mutual_information(name, learned)` — I(C; learned) = H(C) - H(C | learned)
- `information_flow()` — per-edge information transfer in bits
- The Baez-Fritz-Leinster theorem (2011) proves Shannon entropy is the *unique*
  functorial information measure: any function that is (1) functorial, (2)
  convex-linear, (3) continuous must be Shannon entropy. This gives a principled
  foundation for epiplexity tracking.

**Intervention / do-calculus (string diagram surgery):**
- `KnowledgeGraph.intervene(do_concepts)` — returns mutilated graph
- Removes all incoming edges to intervened concepts
- Models "what if we skip/force-teach these concepts?"
- Categorically: an endofunctor performing diagram surgery
  (Jacobs, Kissinger, Zanasi 2019)

**MasteryState:**
- `MasteryState(graph)` — per-concept mastery levels in [0, 1]
- `observe(concept, score)` — Bayesian update from assessment
- `expected_readiness(concept)` — min(mastery × transfer_probability) over prereqs
- `frontier(threshold)` — concepts ready to learn (readiness > threshold)
- `information_gain(concept)` — expected bits gained from learning this concept
- Categorically: a functor from the knowledge graph to [0, 1]
  (Fritz et al. 2024, Hidden Markov models and the Bayes filter)

**Deferred probabilistic extensions (designed, not yet implemented):**
- Imprecise probability / credal sets (Liell-Cock & Staton, POPL 2025) —
  soft prerequisites as convex sets of distributions rather than point estimates
- Categorical gradient learning (Cruttwell, Gavranovic et al. 2022) —
  lens/optic framework for forward (teaching) and backward (assessment) passes
- Possibility theory (Fritz & Teran 2024) — alternative to probabilistic
  weights using t-norm-based Markov categories

### Phase 2.9: Epistemic reasoning (IMPLEMENTED)

The CTKG now supports critical thinking via four mechanisms:

**Epistemic tiers on concepts:**
- `Concept.tier` = `'axiom'` | `'theorem'` | `'conjecture'` | `'heuristic'`
- `Concept.assumes` = list of assumption names the concept depends on
- `Concept.defaults` = dict of default properties (for heuristic-tier concepts)
- DSL: `tier conjecture`, `assumes NAME1 NAME2`, `default NAME = VALUE`

**Assumption-conditioned prerequisites:**
- `Prerequisite.assuming` = name of the assumption making this prerequisite hold
- `Prerequisite.assumption_status` = `'axiomatic'` | `'derived'` | `'empirical'` | `'heuristic'`
- DSL: `requires NAME via "ROLE" assuming ASSUMPTION [STATUS]`

**Challenge edges:**
- `Challenge(source, target, role, strength)` — evidence weakening a concept
- `KnowledgeGraph.challenges` — list of all challenge edges
- DSL: `challenges NAME via "REASON"` inside concept blocks
- Validation: `ChallengedConjecture` warning when a conjecture has active challenges

**Defaults and overrides (Fido problem):**
- `Override(instance, default_concept, property, value, reason)` — instance exception
- `KnowledgeGraph.overrides` — list of all override edges
- DSL: `overrides NAME with PROP = VALUE via "REASON"` inside concept blocks
- `KnowledgeGraph.resolve_default(concept, property, instance)` — returns override value if exists, else default

**Counterfactual exploration:**
- `KnowledgeGraph.what_if_not(concept)` — returns set of concepts that become unblocked if concept is removed
- `KnowledgeGraph.challenged_concepts()` — returns concepts with active challenges + their challengers
- `KnowledgeGraph.assumption_dependents(assumption)` — returns all concepts/prereqs that depend on an assumption

### Phase 3: Computation rule interpreter (NEXT)

- Parse `process` field into an AST (currently stored as raw strings)
- Define evaluation semantics for Level 1 primitives (succ, pred, fold, scan, etc.)
- Evaluate rules by composing prerequisite rules
- Auto-generate problems from rule evaluation + input sampling
- Auto-assemble scratchpads from rule execution trace

### Phase 4: Curriculum generation from `.ctkg` files

- End-to-end: `.ctkg` file → validated graph → curriculum → training data
- Replace hardcoded stage list in `train_arithmetic.py`
- Replay policy based on graph distance

### Phase 5: Functor validation

- Validate that functor preserves composition
- Curriculum transfer: train on domain A, apply functor to generate domain B curriculum

### Phase 6: Level 2-3 primitives

- Logic primitives (equal, forall, exists, implies) for proof-supervised training
- Transform primitives (quote, match, rewrite) for meta-reasoning / algorithm improvement
- Process AST evaluation for all three levels

---

## Epistemic Reasoning — Critical Thinking Infrastructure (IMPLEMENTED)

**Motivation (Light Yagami roleplay, 2026-02-19):** An AI playing Light Yagami accepted without question the claim that it was an EEG-derived mind copy — a claim that is technologically impossible given the character's 2003 timeframe. The model built elaborate strategic frameworks on top of an unexamined false premise. This revealed that the CTKG's fixed-graph model of knowledge (concept A requires concept B, full stop) cannot teach the model to question whether a prerequisite is fundamental or merely assumed.

The Fido problem crystallizes this: "Dogs have 4 legs" + "Fido is a dog" → "Fido has 4 legs." But Fido lost a leg. The system needs to represent default properties that admit exceptions, derived results that depend on assumptions that might be wrong, and active challenges from new evidence.

**Design principle:** The CTKG shouldn't just encode what's known — it should encode *how confidently* it's known and *what would change if it were wrong*. Knowledge is a presheaf over assumption contexts, not a fixed graph.

### Epistemic Tiers on Concepts

Every concept carries a `tier` field indicating its epistemic status:

| Tier | Meaning | Audit frequency | Example |
|------|---------|-----------------|---------|
| `axiom` | Mathematical/logical necessity within its domain. Don't question during normal work. | Never (within domain) | Conservation of energy, group axioms |
| `theorem` | Rigorously derived from stated premises. Valid iff premises hold. | When stuck, question premises | "Alcubierre requires negative energy" (given original metric) |
| `conjecture` | Widely believed, possibly evidence-supported, but unproven. | Maintain active skepticism | "FTL signaling is impossible" |
| `heuristic` | Useful approximation with known exceptions. | Expect exceptions | "Dogs have 4 legs", "heavy elements are stable" |

The model's reasoning strategy is tier-dependent. When stuck, climb the dependency chain and look for the highest-tier concept it can afford to question — conjectures before theorems, theorems before axioms.

```
concept no_ftl_signaling
  domain physics
  tier conjecture
  description "No information can travel faster than light"
  assumes special_relativity pointlike_observers
  ...
```

### Assumption-Conditioned Prerequisites

Prerequisites carry an explicit assumption context: a list of names identifying which assumptions make the prerequisite hold, and a `status` indicating whether it's axiomatic, derived, empirical, or heuristic.

```
requires negative_energy via "metric solution" assuming original_alcubierre_metric [derived]
```

This makes the dependency *transparent*. The model can see that the requirement flows from a specific assumption, not from physics itself. When exploring alternatives, it knows exactly which assumption to relax. Drop `original_alcubierre_metric`, and the `negative_energy` requirement detaches — opening the search space to Lentz-type solutions.

DSL syntax extends the existing requires line:
```
requires NAME via "ROLE" [probability] assuming ASSUMPTION_NAME [STATUS]
```

Both `[probability]` and `assuming` are optional. `[STATUS]` after the assumption defaults to `derived`.

### Challenge Edges

A new edge type alongside prerequisites. Challenges say "evidence E weakens the claim that concept C or prerequisite P holds."

```
concept lentz_soliton
  domain physics
  description "Positive-energy warp metric using shift vector"
  challenges negative_energy_requirement via "positive-energy reformulation"
```

When the model encounters a concept with active challenges, it is *forced to branch*: it cannot simply accept the challenged premise. This externalizes MIMO hypothesis tracking into the graph structure.

Challenge edges also create a natural "audit trigger." When new research is added to the CTKG and includes a challenge edge, every concept downstream of the challenged premise is flagged for re-evaluation.

### Defaults and Overrides (The Fido Problem)

Heuristic-tier concepts express default properties that admit exceptions. An **override** is an instance-level assertion that contradicts a default.

```
concept dogs_have_four_legs
  domain biology
  tier heuristic
  description "Dogs typically have four legs"
  default legs = 4

concept fido
  domain biology
  description "A specific dog"
  overrides dogs_have_four_legs with legs = 3 via "lost a leg in accident"
```

Semantics: when the model reasons about a specific instance, it first checks for overrides. If an override exists, it takes precedence over the default. If no override exists, the default applies.

This connects to the challenge edge mechanism: an override is a challenge scoped to a specific instance rather than to the general concept. The `overrides` DSL keyword creates an `Override` edge in the graph.

Categorically: defaults are natural transformations from the heuristic concept to instances; overrides are modifications (whiskering) of the natural transformation at specific components.

### Counterfactual Exploration: `what_if_not()`

Extension of the existing `intervene()` method. Instead of just removing incoming edges (do-calculus), `what_if_not()` removes a concept entirely and returns the set of concepts that become *unrequired* — the search space that opens up.

```python
opened = graph.what_if_not('negative_energy_density')
# Returns: concepts that were blocked only by negative_energy_density
```

This is the dual of `missing_for()`. Instead of "what do I need to reach X?", it's "what becomes reachable if I stop assuming Y?" If the opened set is large and the removed concept is merely a conjecture, that's a high-value research direction.

### Updated Data Model

```python
@dataclass
class Concept:
    # ... existing fields ...

    # Epistemic tier
    tier: str = 'theorem'  # 'axiom' | 'theorem' | 'conjecture' | 'heuristic'
    assumes: List[str] = field(default_factory=list)  # assumption names

    # Defaults (for heuristic-tier concepts)
    defaults: Dict[str, str] = field(default_factory=dict)  # property -> value

@dataclass
class Prerequisite:
    # ... existing fields ...

    # Assumption context
    assuming: Optional[str] = None     # which assumption makes this hold
    assumption_status: str = 'derived' # 'axiomatic' | 'derived' | 'empirical' | 'heuristic'

@dataclass
class Challenge:
    """A challenge edge — evidence weakening a concept or prerequisite."""
    source: str        # challenging concept
    target: str        # challenged concept
    role: str          # how the challenge works
    strength: float = 1.0  # 0.0 = weak hint, 1.0 = full refutation

@dataclass
class Override:
    """An instance-level exception to a heuristic default."""
    instance: str      # the instance concept
    default_concept: str  # the heuristic being overridden
    property: str      # which property
    value: str         # override value
    reason: str = ''   # why the override exists
```

### Updated DSL Grammar

```
concept_field += 'tier' TIER
             |  'assumes' NAME+
             |  'default' NAME '=' VALUE
             |  'challenges' NAME 'via' STRING
             |  'overrides' NAME 'with' NAME '=' VALUE 'via' STRING
requires_ext  = 'requires' NAME 'via' STRING ['[' FLOAT ']'] ['assuming' NAME ['[' STATUS ']']]
TIER          = 'axiom' | 'theorem' | 'conjecture' | 'heuristic'
STATUS        = 'axiomatic' | 'derived' | 'empirical' | 'heuristic'
```

### Validation Extensions

Two new validation rules:

1. **ChallengedConjecture** — a concept with tier `conjecture` that has active challenge edges should be flagged as "under active dispute — consider branching."

2. **UngroundedAssumption** — a prerequisite with `assuming X` where X is not defined as a concept in the graph. Catches dangling assumption references.

### Example: Warp Drive Domain

```
concept alcubierre_drive
  domain physics
  tier theorem
  description "FTL warp by contracting space ahead and expanding behind"
  assumes original_alcubierre_metric
  requires negative_energy via "metric solution" assuming original_alcubierre_metric [derived]
  requires spacetime_manipulation via "GR field equations"

concept negative_energy_requirement
  domain physics
  tier conjecture
  description "Warp drives require exotic matter with negative energy density"
  assumes original_alcubierre_metric

concept lentz_soliton
  domain physics
  tier theorem
  description "Positive-energy warp metric using shift vector"
  assumes lentz_metric
  challenges negative_energy_requirement via "positive-energy reformulation"
```

Running `what_if_not('negative_energy_requirement')` on this graph would show that `alcubierre_drive` becomes reachable without exotic matter — pointing the model toward the Lentz reformulation.

### Integration with Roleplay Findings

The epistemic reasoning system directly addresses the three failure modes identified in the Light Yagami session:

1. **Premise acceptance** → Epistemic tiers + challenge edges force the model to distinguish between axioms it shouldn't question and conjectures it should actively probe.

2. **Information boundary collapse** → Assumption contexts make explicit which facts depend on which premises, preventing the model from treating derived results as ground truth.

3. **Missing MIMO hypothesis tracking** → Challenge edges externalize parallel hypotheses into the graph structure. The model doesn't need to spontaneously generate alternatives — the graph tells it where alternatives exist.

---

## Symbolic AI Runtime (IMPLEMENTED)

The process language defined in `.ctkg` files is now fully executable. The `experiments/symbolic_ai/` system demonstrates that a structured symbolic reasoner can generalize from ≤10 examples with 100% accuracy — vs. Mamba3 (1.26M params, 50 epochs, 100 examples) achieving ~45% test accuracy on the same task (Mistake #42).

### Architecture

```
experiments/symbolic_ai/
├── interpreter.py   — ProcessInterpreter: executes process lines (List[str] → outputs)
├── memory.py        — ExampleStore: stores examples + KL divergence metric
├── synthesis.py     — Synthesizer: template-based program synthesis (consolidation)
├── engine.py        — SymbolicAI: ties interpreter + memory + synthesizer together
└── run_experiment.py — 11-phase arithmetic + symbolic differentiation experiment
```

### 11-Phase Experiment Results

| Phase | What | Result |
|-------|------|--------|
| 1 | Built-in ops (succ, pred, compare) | PASS |
| 2 | Learn addition from 20 examples | PASS (100%, discovers fold+succ+carry) |
| 3 | Learn subtraction from 15 examples | PASS (100%, discovers fold+pred+borrow) |
| 4 | Minimum examples sweep | PASS (succeeds at N=2) |
| 5 | Prerequisite enforcement | PASS (synthesis blocked without ancestors) |
| 6 | Learn multiplication from 10 examples | PASS (double-nested fold+fn) |
| 7 | Learn exponentiation from 10 examples | PASS (triple-nested fold+fn) |
| 8 | Learn division from 10 examples | PASS (fold_until template) |
| 9 | Verify remainder (lookup process) | PASS (90/90) |
| 10 | Verify GCD (Euclidean, fold_until+lookup) | PASS (81/81) |
| 11 | Symbolic differentiation (Level C) | PASS (23/23) |

### Synthesis Templates

The `Synthesizer` walks templates from simplest to most complex until one is consistent with all stored examples. Templates are gated by the concept's transitive ancestors in the CTKG — if `successor` is not in the ancestor set, no fold+succ template is generated, so addition synthesis fails correctly (Phase 5).

| Level | Lines | Template | Requires |
|-------|-------|----------|---------|
| 1 | 1 | `emit(succ(a))` | successor ancestor |
| 1 | 1 | `emit(pred(a))` | predecessor ancestor |
| 1 | 1 | `emit(compare(a,b))` | comparison ancestor |
| 2 | 2 | `result = fold(x, y, succ); emit(result)` | successor |
| 2 | 4 | fold + carry detection | successor |
| 2 | 4 | fold + borrow detection | predecessor |
| 3 | 2 | double-nested fold via fn (multiplication) | successor |
| 4 | 2 | triple-nested fold via fn (exponentiation) | successor |
| 5 | 3 | fold_until with pair state (division) | predecessor + comparison |

### Key Design Decisions

- **Integer succ, not digit succ.** `fold(b, a, succ)` for addition requires succ to be non-wrapping. The digit constraint applies only to I/O types, not intermediate values.
- **Constant folding in sym_* constructors.** `sym_add(('NUM',2), ('NUM',3))` → `('NUM',5)` at construction time.
- **X, Y, Z, T are literals.** Added to `_LITERALS` frozenset so they resolve as strings, not env lookups. Enables `sym_var(X)` → `('VAR', 'X')`.
- **lookup() always returns tuple.** Use `first(lookup(...))` to extract a scalar.
- **fold_until has hard safety cap.** max_steps is always set to an input variable, guaranteeing termination in at most max_steps steps. Cannot infinite-loop.
- **`ai._interp`** is the ProcessInterpreter attribute in SymbolicAI (private, not `ai.interpreter`).

---

## Open Questions

1. **How to represent "variable n_result"?** Stages 5-8 have variable-length scratchpads. The `problems_to_tensors()` function pads to seq_len, but the graph needs to know the maximum length for validation.

2. **Replay policy:** When training node B, how much to replay from ancestors? Options: uniform, distance-weighted, recency-weighted, loss-weighted. This is an empirical question.

3. **Type checking granularity:** The current type system is coarse (list of token type names). Should we use a richer type system (e.g., dependent types where the type of the output depends on the input)?

4. **Cross-domain functors:** The functor concept is elegant but speculative. Do we need it for the arithmetic domain, or is it only relevant when we add logic/language domains?

5. **Engram-style prefetching:** The retrieval mechanism needs to be fast enough for inference. Graph traversal is O(V+E), which is fine for small graphs. For large graphs (thousands of concepts), we may need indexing.

6. **Epiplexity threshold calibration:** What S_preq value indicates "too easy"? Likely domain-dependent. Need empirical data from initial curriculum runs to set thresholds.

7. **Reverse factorization coverage:** Which stages benefit from reverse problems? Arithmetic stages (3-5) have natural inverses. Counting stages (1-2) don't have clean reverses (reconstruction is ambiguous). Should reverse coverage be tracked per-node in the graph?

---

## Key Principles

1. **The graph is the source of truth.** If the graph says a prerequisite is missing, trust it. Don't work around it with more training epochs.

2. **Validation before training.** Always run `validate()` before starting a training run. A clean validation = a structurally sound curriculum.

3. **Nodes before edges.** Define all concepts before defining relationships. This forces explicit enumeration of what the model needs to know.

4. **Types are contracts.** The input/output types on nodes and edges are contracts that the scratchpad format must satisfy. If the types don't match, the scratchpad is wrong.

5. **Small atomic, large composite.** Atomic nodes should have <20 entries. Large collections are a signal to decompose into sub-graphs with compositional structure.

6. **Measure, don't assume.** Track epiplexity (S_preq) per stage. A stage that passes doesn't necessarily teach structure — it might just be memorizable. Epiplexity distinguishes "learned to apply" from "learned to look up."

7. **Both directions.** When a concept has a natural inverse, train on both forward and reverse factorizations. Reverse problems force induction, which builds richer representations than mechanical forward application alone.

8. **Sheaf before merge.** Never merge two domain graphs without `sheaf_check()`. Incompatible overlapping definitions produce silent bugs that are nearly impossible to diagnose after the merge. The sheaf condition is cheap to check and catches contradictions at composition time, not at training time.

9. **Know what you don't know.** Every concept has an epistemic tier. Axioms are trusted. Theorems are trusted given their premises. Conjectures are actively probed. Heuristics expect exceptions. When stuck, question the highest-tier thing you can afford to question.

10. **Challenges are first-class.** When new evidence weakens an existing claim, add a challenge edge. This forces the model to branch rather than ignore the contradiction. Unaddressed challenges are technical debt.

11. **Defaults are not facts.** Heuristic-tier concepts express defaults, not universal truths. Always check for overrides before applying a default. The Fido problem: "dogs have 4 legs" is a heuristic, not an axiom.

12. **Assumptions are explicit.** Every derived result should name the assumptions it depends on. When an assumption is weakened, all results that depend on it are automatically flagged.
