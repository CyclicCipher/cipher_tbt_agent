# Category Theory Knowledge Graph (CTKG) — Design Document

**Date:** 2026-02-16 (updated 2026-02-18)
**Status:** Universal type system + DSL parser + arithmetic domain implemented. 6/6 tests pass.
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
| `succ(x)` | ordered T → T | Next element |
| `pred(x)` | ordered T → T | Previous element |
| `compare(a, b)` | ordered T × T → {GT, LT, EQ} | Comparison |
| `lookup(table, key)` | Table × Key → Value | Finite function |
| `fold(seq, init, step)` | seq(T) × S × (S×T→S) → S | Reduce sequence |
| `scan(seq, init, step)` | seq(T) × S × (S×T→S) → seq(S) | Running fold |
| `count(seq, filter)` | seq(T) × Pred → nat | Count matches |
| `emit(x)` | T → output | Produce token |
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

| Primitive | Signature | Meaning |
|-----------|-----------|---------|
| `quote(expr)` | T → expr | Reify expression |
| `match(expr, pattern)` | expr × pattern → bindings | Pattern match |
| `substitute(expr, var, val)` | expr × name × T → expr | Replace variable |
| `rewrite(expr, rule)` | expr × rule → expr | Apply transformation |
| `decompose(expr)` | expr → seq(expr) | Break into parts |
| `compose(a, b)` | expr × expr → expr | Combine expressions |

Level 1 lets you DO arithmetic. Level 2 lets you PROVE things about arithmetic. Level 3 lets you IMPROVE algorithms (meta-reasoning, code-as-data).

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
| Validation (6 error types) | Implemented | validate() in graph.py |
| Curriculum generation | Implemented | topological_sort + generate_curriculum |

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
statement    = typedef | concept | functor | adjunction | comment
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

The graph is constructed by loading one or more `.ctkg` files, then calling `validate()`.
Validation now checks type resolution: all type names in concept inputs/outputs must exist in the type registry (either defined in the `.ctkg` file or built-in).

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
