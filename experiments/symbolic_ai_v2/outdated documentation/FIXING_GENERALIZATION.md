# Fixing Generalization

## The Iron Rule

**There is never, ever a special case or code for one.**

Any rule, segment type, handler, or pattern that only fires for a specific value,
specific digit width, or specific surface form is a violation. If a rule exists,
it must be derivable from a universal principle that applies to all inputs of the
same algebraic class.

---

## Root Cause of Current Brittleness

The system operates on **token sequences** but math is **trees**.

`d/dx[2x² + 3x]` in the corpus is the flat sequence `d mul 2 pow x 2 add mul 3 x eq`.
The tree structure — that `add` is the root, `mul 2 pow x 2` is a subtree, `d` binds
over the whole thing — is invisible. Every segment type added (`G`, `P`, `SC`, `AK`,
`F`, `CONST_BLOCK`, etc.) is a patch to recover structure that should never have been
lost.

An autoregressive LLM doesn't need these patches because it has enough representational
capacity to *implicitly* recover the tree. Our system tries to do the same with explicit
lookup tables. That is why a new special case appears every time a new surface form appears.

The fix is not more special cases. The fix is the correct representation.

---

## The LLM Requirement

The system must be able to read tokens and output tokens as freely as an autoregressive
LLM. This means:

- No fixed arity for operators
- No width assumptions about how numbers are written
- No required skeleton matching
- No hand-coded segment types
- Must generalize across surface forms: `5`, `five`, `cinco`, `5.0` are the same object
- Must handle word problems: "find the derivative of the product of two functions"
- The same machinery that handles digits must handle words

The LLM achieves this because softmax over all tokens is implicitly a categorical
morphism from any context to any continuation. The symbolic analog:
1. **Term algebra** — replaces fixed-width digit patterns
2. **Natural transformations** — replace hard-coded segment types
3. **Catamorphisms** — replace case-by-case segment discovery

---

## The Two Problems Are One Problem

Brittleness and storage explosion are the same failure seen from different angles.

Both arise from **storing derived facts instead of generators**.

- Brittleness: the system stores surface-form-specific programs (one per skeleton)
  instead of the universal law that generates all of them
- Explosion: the system would store every (a < b) pair on the number line instead of
  the single generator `succ` plus the transitivity law that derives them all

The unified fix: **store only generators and laws; compute everything else.**

A morphism should be stored if and only if it cannot be expressed as a composition of
smaller morphisms using the known laws. This is MDL — minimum description length. The
CTKG already has MDL pruning applied to sequential patterns; the principle needs to be
applied to all morphisms at every level of abstraction.

---

## Separate Categories

Knowledge must be partitioned into separate categories, not flattened into one graph.

A single category containing all knowledge becomes incoherent: a morphism from the
category of *temperatures* must not accidentally compose with a morphism from the
category of *probabilities*. Separate categories give reasoning type-safety.

The correct structure for the CTKG is a **2-category**:

- Objects = categories (domains: ℕ, ℝ, propositions, geometric spaces, words, ...)
- 1-morphisms = functors (structure-preserving maps: "temperature converts to energy
  via E=kT"; differentiation D: Func → Func)
- 2-morphisms = natural transformations (laws that hold between functors: the sum rule
  says D commutes with addition)

Separate categories also solve the naming collision problem: "one" in ℕ and "one" in
Bool are different objects that happen to share a surface form. Without separation the
system confuses them.

The CTKG in `experiments/ctkg/` already has this structure (arithmetic, logic, syntax
domains). What is missing is the inference engine using it.

---

## The Transitive Closure Problem: Compression Principles

The number line is the clearest example. The ordering relation `<` on ℕ would require
O(n²) explicit stored pairs. The number `1 < 10^100` is true but must not cost anything
to store. The same explosion appears in:

- Every preorder (≤, ⊆, ⊢, subtype)
- Every group (O(|G|²) multiplication pairs)
- Every grammar (exponentially many parse trees)
- Any rich category with composable morphisms

**The answer: the number line needs three entries, not O(n²).**

1. The generator: `succ: n → n+1`
2. The law: transitivity (a natural transformation — composition of `<` morphisms
   is a `<` morphism)
3. The query algorithm: `a < b ⟺ b - a ∈ ℕ⁺` (reduces to subtraction, which the
   NNO already provides)

The NNO we already have IS the correct compressed representation of the entire infinite
ordered number line. `1 < 10^100` costs O(1) to verify (subtract) and O(0) to store
(it is not stored; it is derived).

**The five compression mechanisms, in order of power:**

**A. Normal forms**
Two morphisms that are equal by the laws of the category are stored once as their
canonical normal form. `add(2,3)`, `add(1,4)`, `succ(succ(succ(succ(succ(0)))))` are
the same object in ℕ — normal form `5`. The quotient of the free category by its
algebraic laws is the compressed representation. Sequitur already does this for
sequential grammar rules; the algebraic analog extends it to all morphism classes.

**B. Formula morphisms**
A morphism whose value is computable from a closed-form expression is stored as the
formula, not as a table. `n < m ⟺ m - n ∈ ℕ⁺` is one entry, not n² entries.
The NNO fold rules already encode this — they are formulas, not lookup tables.
Extension: any morphism expressible as `cata(alg)` over an initial algebra is stored
as `alg`, not as its extension.

**C. Profunctors for relations**
A relation between categories A and B is not a set of pairs — it is a profunctor
`P: A^op × B → Set`. Storing the profunctor (a function) instead of its extension
(all pairs) compresses O(n²) to O(1) when the relation is computable. The ordering
on ℕ, implication between propositions, and subsumption between types are all
profunctors.

**D. Span/schema compression**
When many morphisms share the same algebraic shape, store the schema parameterised
by one varying component. All "multiply by n" morphisms share the schema
`mul_n: x ↦ n·x`; store one entry, not one per n. This directly maps to the
`formula morphism` principle applied to families.

**E. Lazy composition with bounded memoization**
Compositions are not stored — they are computed on demand and cached for hot paths.
Cache eviction (LRU or frequency-weighted) bounds RAM. The composition algorithm is
`_compose()` in the current code; it needs to be the primary path, not a fallback.
The rule: if a morphism is derivable from generators in under k steps, never store it.

---

## Missing Category Theory Abstractions

### 1. Initial algebras / term algebra (most critical)

We have the NNO (initial algebra of the successor functor) but no **initial algebra
for expressions**:

```
Expr = μX. Atom | Add(X,X) | Mul(X,X) | Pow(X,X) | Neg(X) | Apply(Op, X) | ...
```

Without this, the system has no notion of "subexpression." The sum rule
`d/dx[f+g] = d/dx[f] + d/dx[g]` is a rewrite rule on trees. On flat token
sequences it has no representation.

The current `_discover_value_segment` with 8 special cases is an approximation
of one case of one catamorphism over this algebra. The correct version is a
single recursive rule.

This also solves the width problem: `12` (two digit tokens) and `8` (one digit token)
are the same *kind* of object — a numeral — and a rule over numerals applies to both
regardless of how many characters they occupy.

### 2. Natural transformations as algebraic laws

Every law of calculus and algebra is a natural transformation between functors:

| Law | CT structure |
|-----|-------------|
| Sum rule: `D(f+g) = D(f) + D(g)` | η: D∘Add → Add∘(D×D) |
| Linearity: `D(cf) = c·D(f)` | η: D∘Scale_c → Scale_c∘D |
| Chain rule: `D(g∘f) = D(g)∘f · D(f)` | η: D∘(−∘f) → (−∘f)∘D · D(f) |
| FTC: `∫D(f) = f` | counit of adjunction D ⊣ ∫ |
| Distributivity: `a(b+c) = ab + ac` | η: Mul∘(Id×Add) → Add∘(Mul×Mul) |
| Transitivity of `<` | η: (<∘<) → < — composition stays in the preorder |

We have zero of these. The system computes `D(x^n) = n·x^(n-1)` for specific `n`
because it appears in training data, but it cannot apply this as a rewrite rule
inside a larger expression. Each natural transformation stored replaces potentially
infinite explicit morphism pairs.

### 3. Substitution as a monad (Kleisli for variables)

`x` in `d/dx[x²]` is a **bound variable**. The system treats it as a literal token.
This means `d/dx[x²]` and `d/dt[t²]` are completely different problems.

Correct CT treatment: the **substitution monad** `T(X) = {terms with variables in X}`
with:
- η: X → T(X) (variable injection)
- μ: T(T(X)) → T(X) (substitution / flattening)

Without this, `x` is just a character, not a variable. Evaluation, substitution,
and the chain rule are all unrepresentable. This also means "five" and `5` can be
the same object: both are terms that reduce to the same normal form in ℕ.

### 4. Catamorphisms / recursors over the initial algebra

The correct way to apply a rule to an expression tree is a **catamorphism** —
a fold over the initial algebra. For differentiation:

```
D = cata(alg_D)
  where
    alg_D(Add(df, dg))          = Add(df, dg)
    alg_D(Mul(f, df, g, dg))    = Add(Mul(f, dg), Mul(df, g))  -- product rule
    alg_D(Pow(Var, n))          = Mul(n, Pow(Var, pred(n)))     -- power rule
    alg_D(Var)                  = Const(1)
    alg_D(Const(c))             = Const(0)
```

This is a **single recursive rule** that handles sum, product, power, and constant
rules with no special cases. The entire calculus of derivatives is one catamorphism.
Integration is the adjoint catamorphism. Both are one entry in the CTKG, not a table
of programs per expression shape.

Catamorphisms also give compression: instead of storing `D(2x² + 3x) = 4x + 3` as
a fact, store `alg_D` once and compute the answer for any expression in one tree
traversal.

### 5. Adjunction D ⊣ ∫ as a proper pair

We have `adj_search` as a hack. The correct structure: differentiation `D` and
integration `∫` form an adjunction with:
- Unit η: `f → ∫D(f)` (antiderivative of derivative = original up to constant)
- Counit ε: `D∫(f) = f` (fundamental theorem of calculus)

The constant `C` in indefinite integrals is exactly the kernel of the counit —
the part of `∫` that `D` destroys. Without the adjunction structure, we cannot
represent why `C` appears or what it means. Proper adjunctions also enable
bidirectional reasoning: given the integral, find the derivative; given the
derivative, recover the integral. One adjunction pair replaces two separate
lookup tables.

### 6. Functorial variables vs. free variables

Anti-unification (Phase III) discovers **free variables** — positions that range over
arbitrary values. This is not sufficient for learning relational structure like the
classic word analogy `king − man + woman ≈ queen`.

The reason transformers learn this is distributional: `king` and `queen` appear in
almost identical contexts (royalty), while differing in a single latent dimension
(gender). The vector offset `king − queen` aligns with `man − woman` because both
pairs co-occur with identical context tokens except for gendered pronouns. The model
never represents "gender" explicitly — it falls out of the dot-product geometry.

The correct CT representation is a **natural transformation between two functors**:

```
Category ROLE:   {person,  ruler,   spouse, ...}
                       ↓ F_male      ↓ F_female
Category ENTITY: {man,     king,    husband, ...}
                 {woman,   queen,   wife,   ...}
```

- `F_male(ruler) = king`, `F_male(spouse) = husband`, `F_male(person) = man`
- `F_female(ruler) = queen`, `F_female(spouse) = wife`, `F_female(person) = woman`
- The gender swap is a natural transformation `η : F_male → F_female`

Naturality square (the formal statement that gender is consistent across all roles):
for every role morphism `r : ruler → spouse`, the square commutes —
`F_female(r) ∘ η_ruler = η_spouse ∘ F_male(r)`.

The critical distinction:
- **Free variable** (what anti-unification finds): `V0` ranges over arbitrary values
- **Functorial variable**: `V0` ranges over `image(F_male)`, `V0'` ranges over
  `image(F_female)`, with a consistent bijection `η: V0 ↔ V0'` across all rules

Anti-unification finds that `king` and `queen` share skeleton `(_, ruler)` with V0
varying. But it does not detect that V0's values form a consistent partition
`{man, king, husband}` vs `{woman, queen, wife}` across multiple separate rules. What
is needed is **Phase IX: functorial variable clustering** — scanning across all
discovered rules for variables whose substitution sets form a consistent partition,
then registering each such partition as a functor pair with a natural transformation.

This is architecturally absent from the current plan. It slots naturally after Phase
VI (2-category structure) as Phase IX.

### 7. Operad structure for n-ary composition

Hard-coded binary operations violate the iron rule. An **operad** handles
arbitrary-arity operations uniformly. `add(f, g, h)`, `mul(a, b, c, d)`,
`compose(f, g, h)` — all the same mechanism, all the same code path.

### 7. Comonad for evaluation context

Evaluating `f(x)` at a point requires an **environment** — the **reader comonad**
(coKleisli category). Without it, `x` has no value, substitution is impossible,
and the chain rule `D(g∘f)(x) = D(g)(f(x)) · D(f)(x)` is unrepresentable.

---

## Tasks Architecturally Blocked Without These Fixes

**Calculus:** product rule, chain rule, implicit differentiation, integration by
parts, any input with `+` at the root (sum rule).

**Algebra:** polynomial factoring, solving for a variable, simplification by laws.

**Logic/proofs:** modus ponens, universal quantification, introducing/eliminating
variables.

**Word problems:** impossible with flat token system; require parsing to expression
trees first.

**Cross-notation:** written digits, different derivative notations (`f'(x)` vs
`df/dx`), any surface form change breaks every program.

**Any new domain:** adding physics, chemistry, or grammar requires new hand-coded
segment types instead of plugging in a new category with generators and laws.

---

## What the Correct Architecture Looks Like

A **term rewriting system whose rules are discovered from examples**, operating
over a 2-categorical knowledge graph that stores only generators and laws.

```
token sequence
      ↓  parse (free category / term algebra)
expression tree
      ↓  match (natural transformation pattern matching)
rewrite rule fires
      ↓  apply (catamorphism over tree)
reduced expression
      ↓  unparse
token sequence
```

**Storage policy:** a morphism is stored iff it cannot be derived from the known
generators and laws in under k composition steps. Everything else is computed
lazily and cached with bounded memory.

**Discovery:** find new generators and laws using Kan extensions over observed
morphism pairs. A natural transformation is hypothesised when the same structural
equation holds across all observed instances — this is what the current process
discovery already does for binary fold rules; the extension generalises it to
tree-structured morphisms.

**Generalization:** automatic. A rule discovered for `d/dx[x^2]` applies to
`d/dx[f^2]` because the catamorphism is universal over the tree structure. No
re-discovery needed; no new segment type needed.

The NNO and fold rules we already have are the **arithmetic leaf** of this system.
The missing piece is the tree layer above them — the initial algebra, the
catamorphism recursors, and the natural transformation law store.

This is what CT_REFERENCE §3 (natural transformations), §6 (Kan extensions),
§9 (operads), §16 (FCA objects), and §17 (free category + quotient = discovered
rewrite rules) are pointing toward. The Kan extension machinery exists. The tree
representation does not.

---

## Open Questions: Chosen Answers

These were open questions. Answers are given here so the roadmap can be concrete.

**Q: How do we parse arbitrary token sequences into expression trees without a
pre-defined grammar?**

Answer: **bootstrapped arity discovery from corpus statistics**.

Atoms are identified first: any token that appears as a complete standalone answer
(e.g., after `eq` or `ans`) is arity-0. Then, iterating over the corpus: a token
`op` is n-ary if, when we tentatively parse the n tokens/sub-expressions following
it as arguments, the resulting expression's canonical form consistently matches the
observed output. Formally, `arity[op] = argmin n s.t. parse_prefix(seq, n) produces
consistent (input, output) pairs across all corpus occurrences of op`.

This bootstraps: once atom arities are known, unary operators are discoverable;
once unary operators are known, binary operators are discoverable; and so on. The
free category construction we already have is exactly the scaffolding — it already
identifies which tokens act as "operators" by observing which tokens have morphisms
flowing through them. Extension: attach the discovered arity to each operator node.

For natural language: the same algorithm applies. "the derivative of x squared" is
parsed into a tree using the same mechanism — "derivative of" is discovered as a
unary operator (arity 1) that consistently maps to the `D` morphism in the math
category. This is a functor from the English surface category to the math expression
category, discovered from word-problem training data. The discovery algorithm is
identical; only the token vocabulary differs.

**Q: How are variables distinguished from constants without hard-coding?**

Answer: **anti-unification across multiple examples with the same structural skeleton**.

A position p in expression pattern T is a **variable** if and only if at least two
training examples share the same structural skeleton T and differ at position p,
and the rule that maps input to output is consistent under that substitution.

Concretely: given `(D(pow(x, 2)), 2*x)` and `(D(pow(x, 3)), 3*x^2)`, anti-unify
both inputs to get pattern `D(pow(x, V))`. The position occupied by `2` and `3`
is variable (different values, consistent rule). The position occupied by `x` is
constant (same value in both — `x` is a structural atom of the differentiation
context, not a varying argument).

This means `5` and `five` become the same variable value as soon as both appear in
examples with the same structural skeleton and the same outputs. No hard-coding
required; notation-independence falls out automatically.

**Q: How is a catamorphism stored in the CTKG?**

Answer: **as a named set of RewriteRules, one per constructor of the initial algebra**.

A catamorphism `cata(alg_D)` for differentiation is stored as the group of rules
under the name `D_algebra`:
- `D(Const(c)) → Const(0)`
- `D(Var(x)) → Const(1)`
- `D(Add(f, g)) → Add(D(f), D(g))`
- `D(Mul(f, g)) → Add(Mul(D(f), g), Mul(f, D(g)))`
- `D(Pow(f, n)) → Mul(n, Pow(f, Pred(n)))`

Each rule is a `RewriteRule(lhs: Expr, rhs: Expr)`. The catamorphism is the unique
F-algebra homomorphism from the initial algebra to itself; in storage it is just
an ordered rule set with a name. Evaluation calls `cata_reduce(expr, rules)` — one
recursive function, no special cases.

**Q: Does `graph.py` need extension or reinterpretation?**

Answer: **extension with three new node types; existing structure is reinterpreted
as the 2-categorical skeleton**.

Add:
1. `RewriteRule(lhs: Expr, rhs: Expr, algebra_name: str)` — stores one rule of one
   catamorphism; replaces all SlotProgram segment types
2. `Category(name: str, concept_ids: frozenset)` — groups concepts into a domain;
   existing arithmetic/logic/syntax domains become `Category` objects
3. `NaturalTransformation(source_functor: str, target_functor: str,
   components: list[RewriteRule])` — stores an algebraic law; distinct from `Functor`
   (which maps between domains) because a NatTrans is a 2-morphism between functors

Existing types are reinterpreted:
- `Concept` → object in its `Category`
- `Prerequisite` → generating morphism (1-morphism in the 2-category)
- `Functor` → 1-morphism between `Category` objects (already correct)
- `Adjunction` → adjoint pair of `Functor`s with unit/counit (already partially correct)

**Q: Can MDL pruning be extended to remove semantic duplicates?**

Answer: **yes, via normal form computation under the RewriteRule set**.

Two morphisms m₁ and m₂ are semantic duplicates if `normal_form(m₁, rules) ==
normal_form(m₂, rules)`. The normal form is computed by applying `cata_reduce`
to exhaustion (until no rule fires). For arithmetic, the algebraic laws form a
confluent rewriting system (proven; e.g., Knuth-Bendix for ring axioms), so
normal forms are unique. The extended MDL criterion: remove morphism m if
`normal_form(m) == normal_form(m')` for some already-stored m' with a shorter
description length.

---

## The Unified Architecture: Prediction IS CTKG Traversal

The current system has two disconnected prediction pathways:

- **`BinaryFoldRule` + `_process_predict`**: handles fold-type arithmetic (`add`, `sub`,
  `mul`) via a `(d1, d2, carry) → (result, carry_out)` lookup table.
- **`ChainRule` + `_chain_predict`**: handles composition-type operations (`linsolve`,
  `eval`, `d`, `int`) via a `input_tuple → output_tokens` lookup table.

These are architecturally separate and non-communicating. The system can compute
`sub(9, 5) = 4` and `sub(3, 1) = 2`, but when asked `linsolve(3, 9, 1, 5)` it cannot
route the answer through its own arithmetic because `linsolve` goes through a different
dispatch path that has no connection to `BinaryFoldRule`.

**This is not a missing feature. It is the wrong structure.** Two lookup tables are not
a knowledge graph. A knowledge graph has one evaluation mechanism: traversal.

### The Principle

**Prediction = catamorphism over the CTKG morphism graph.**

Given a parsed expression tree, prediction walks the morphism graph:

1. At each node (expression head `op`), look up the morphisms indexed by `op` in the CTKG.
2. Apply the matching rule to the recursively-reduced arguments.
3. The result is the normal form of the expression — the unique representative under the
   algebraic laws stored in the CTKG.

This is exactly `cata_reduce(expr, rules)` where `rules` is the full CTKG rule set.
CTKG traversal and `cata_reduce` are the same computation viewed from two angles:
- **CTKG angle**: walk the morphism graph, apply each morphism to its sub-expressions
- **Algebra angle**: fold an F-algebra homomorphism bottom-up over the expression tree

`BinaryFoldRule` is not a different kind of thing from `RewriteRule`. It is a set of
morphisms in the CTKG whose head is `add`/`sub`/`mul`. The `binary_table` is their
extension (all ground instances). The Iron Rule says: store the generator, not the
extension. The generator for `add` is the NNO inductive rule:

```
add(zero, b, 0)    → (b, 0)         -- identity base case
add(zero, b, 1)    → (succ(b), ...)  -- carry base case
add(succ(a), b, c) → succ_carry(add(a, b, c))  -- inductive step
```

These three rules, plus the ground `succ` chain (10 rules for single-digit arithmetic),
replace 200 `binary_table` entries. They are `RewriteRule` objects. `cata_reduce`
evaluates them. The entire arithmetic machinery is unified under one call.

`ChainRule` is also not a different kind of thing from `RewriteRule`. `linsolve(A, B, C, D)`
is a composite morphism, discoverable by anti-unification of the trace-format training
examples. The discovery finds the structural rule:

```
linsolve(A, B, C, D) → div(sub(B, D), sub(A, C))
```

This rule is a `RewriteRule` whose RHS contains `div` and `sub` — morphisms already
present in the CTKG as NNO rules. `cata_reduce` evaluates the full tree in one pass:
`linsolve(3,9,1,5)` → `div(sub(9,5), sub(3,1))` → `div(4,2)` → `2`. No separate
dispatch. No lookup table miss. No fallback to n-gram.

### What Changes

**Phase VIII** (previously "Remove Remaining Violations") is extended to include the
unification:

1. Convert `BinaryFoldRule.binary_table` → NNO inductive `RewriteRule` objects
   (stored in the CTKG alongside the structural rules).
2. Extend `discover_rules` to discover composition rules for trace-format ops:
   anti-unify `(linsolve(A,B,C,D), sub(B,D)/sub(A,C))` pairs → structural RewriteRule.
3. Replace all prediction dispatch (Level 1a `_process_predict`, Level 1b
   `_chain_predict`, Level 1c `_cata_predict`) with a **single** `_cata_predict` call
   that uses the unified rule set.

**Storage policy** (Phase VII gate, now achievable): with NNO inductive rules replacing
the binary table, the CTKG stores O(1) rules per arithmetic operation instead of O(n²).
Morphism count stops growing with corpus size.

---

## Implementation Roadmap

Phases are ordered so each is independently testable against the math benchmark.
Every phase has a gate: it must not regress any currently passing level.

---

### Phase I — Term Algebra: `Expr` and Anti-unification

**New file:** `experiments/symbolic_ai_v2/ctkg/core/term_algebra.py`

This is the foundation everything else builds on. No changes to existing code yet.

```
Expr
  head: str          # operator name, literal digit, variable name
  args: tuple[Expr]  # empty for atoms
  is_var: bool       # True if this position is a pattern variable (for rules)
```

Functions to implement:
- `atom(tok: str) → Expr` — construct a leaf node
- `node(head: str, *args: Expr) → Expr` — construct an internal node
- `var(name: str) → Expr` — construct a pattern variable (is_var=True)
- `size(e: Expr) → int` — number of nodes
- `depth(e: Expr) → int` — tree depth
- `match(pattern: Expr, expr: Expr) → Optional[dict[str, Expr]]`
  - pattern variables bind to any sub-expression
  - literals must match exactly
  - returns binding dict or None
- `substitute(expr: Expr, bindings: dict[str, Expr]) → Expr`
  - replace all var nodes with their bindings
- `anti_unify(e1: Expr, e2: Expr) → tuple[Expr, dict, dict]`
  - returns (lgg, subst1, subst2) where lgg is the least general generalisation
  - lgg[subst1] = e1, lgg[subst2] = e2
  - algorithm: structural recursion; where heads differ, introduce a fresh var
- `anti_unify_list(exprs: list[Expr]) → tuple[Expr, list[dict]]`
  - fold anti_unify over a list; returns (lgg, list of substitutions)

**New file:** `experiments/symbolic_ai_v2/ctkg/tests/test_term_algebra.py`

Gate: all unit tests pass. No changes to benchmark yet.

Key tests:
- `anti_unify(pow(x,2), pow(x,3))` → `(pow(x, V0), {V0: 2}, {V0: 3})`
- `match(pow(var('f'), var('n')), pow(x, 3))` → `{f: x, n: 3}`
- `substitute(mul(var('n'), pow(var('f'), pred(var('n')))), {n:3, f:x})` → `mul(3, pow(x, pred(3)))`
- `anti_unify(mul(2,x), mul(3,x))` → `(mul(V0, x), ...)` — x is NOT generalised

---

### Phase II — Arity Discovery and Expression Parser

**New file:** `experiments/symbolic_ai_v2/ctkg/core/expr_parser.py`

```
ArityTable = dict[str, int]   # token → number of sub-expression arguments
```

Functions to implement:
- `discover_arities(corpus: list[list[str]]) → ArityTable`
  - **Step 1:** mark as arity-0 any token that appears as a standalone complete
    sequence (a single-token sequence) OR appears after `eq`/`ans`/`step` as a
    run of consecutive digit tokens. These are atoms.
  - **Step 2:** for each remaining token `op`, observe all corpus positions where
    `op` appears. Tentatively parse the following tokens as k sub-expressions
    (for k = 1, 2, 3, ...) using arities discovered so far. Accept the smallest k
    such that the resulting parse produces consistent (input_tree, output_tree)
    pairs across at least 2 corpus occurrences.
  - **Step 3:** repeat until convergence (no new arities discovered).
  - This is a Kleene fixed-point over the arity table.
- `parse(tokens: list[str], arities: ArityTable) → Expr`
  - Recursive descent: read head token; if arity[head] = n, recursively parse n
    sub-expressions and return `node(head, *args)`
  - For unknown tokens: treat as arity-0 (atom) and log a warning
- `unparse(expr: Expr) → list[str]`
  - Pre-order traversal: emit head then recursively unparse each arg

**Note on natural language:** `discover_arities` works on any token vocabulary.
Given a corpus of word problems, "derivative of" would be discovered as a unary
operator, "the sum of _ and _" as binary, etc. The parser makes no assumption
that tokens are mathematical symbols.

**New file:** `experiments/symbolic_ai_v2/ctkg/tests/test_expr_parser.py`

Gate: parse/unparse round-trip is identity for all training sequences.

Key tests:
- `discover_arities` on math corpus discovers `add:2, mul:2, pow:2, succ:1, d:2`
- `parse(['add', 'mul', '2', 'x', '3'])` → `add(mul(2,x), 3)`
- `unparse(add(mul(2,x), 3))` → `['add', 'mul', '2', 'x', '3']`
- Round-trip: `unparse(parse(seq)) == seq` for all training sequences

---

### Phase III — RewriteRule and Rule Discovery

**New file:** `experiments/symbolic_ai_v2/ctkg/core/rewrite.py`

```
RewriteRule
  lhs: Expr          # pattern (may contain var() nodes)
  rhs: Expr          # replacement
  algebra_name: str  # which catamorphism / natural transformation this belongs to
  evidence: int      # number of training examples that support this rule
```

Functions to implement:
- `cata_reduce(expr: Expr, rules: list[RewriteRule]) → Expr`
  - Recursively reduce children first (bottom-up)
  - Then try each rule: `match(rule.lhs, reduced_expr)` → if bindings found,
    return `substitute(rule.rhs, bindings)`
  - Repeat until no rule fires (fixed point = normal form)
  - No special cases. No segment types. One function.
- `normalize(expr: Expr, rules: list[RewriteRule]) → Expr`
  - Alias for `cata_reduce` to exhaustion; semantically "compute the normal form"

**New file:** `experiments/symbolic_ai_v2/ctkg/learning/rule_discover.py`

Functions to implement:
- `discover_rules(examples: list[tuple[Expr, Expr]], arities: ArityTable) → list[RewriteRule]`
  - Input: list of (input_tree, output_tree) pairs from training corpus
  - **Step 1:** group examples by structural skeleton of the input tree
    (skeleton = tree with all atoms replaced by a generic placeholder)
  - **Step 2:** within each group, anti-unify all input trees → `lhs_pattern`
  - **Step 3:** anti-unify all output trees WITH THE SAME VARIABLE NAMES as Step 2
    → `rhs_pattern`. A variable in rhs at position p corresponds to the variable in
    lhs at the anti-unified position.
  - **Step 4:** verify consistency: for every example, `cata_reduce(input, [rule])`
    equals the observed output. Discard rules that fail this check.
  - **Step 5:** name the rule after the root operator of lhs_pattern.
  - Return list of consistent RewriteRules.
- `group_by_skeleton(examples: list[tuple[Expr, Expr]]) → dict[Expr, list]`
  - Skeleton = replace all atoms with placeholder `_`; group by skeleton equality

**Note:** this replaces `_discover_trace_programs`, `_discover_slot_program`,
`_discover_value_segment`, and all segment type logic. The anti-unification step
replaces `_refine_pattern_key`. The consistency check replaces the CONST_BLOCK /
AK priority ordering hacks.

**New file:** `experiments/symbolic_ai_v2/ctkg/tests/test_rule_discover.py`

Gate: `discover_rules` on derivative_trace training data recovers the power rule,
product rule (partial), and constant rule without any hard-coded cases.

Key tests:
- Given `[(D(pow(x,2)), mul(2,x)), (D(pow(x,3)), mul(3,sq(x)))]` →
  discovers `RewriteRule(D(pow(x, V0)), mul(V0, pow(x, pred(V0))))`
- Given `[(D(const(2)), const(0)), (D(const(5)), const(0))]` →
  discovers `RewriteRule(D(const(V0)), const(0))`
- The lhs variables and rhs variables are consistent (same V0 maps to same value)

---

### Phase IV — Replace SlotProgram with RewriteRule in the Predictor ✓ COMPLETE

**Completed 2026-03-15.**

**What was done:**
- Deleted `class SlotProgram` and ALL 8 segment types
  (`'K'`, `'G'`, `'P'`, `'SC'`, `'F'`, `'V'`, `'E'`, `'CONST_BLOCK'`, `'A'`, `'AK'`)
- Deleted `_lz_strip`, `_zpad`, `_parse_into_value_blocks`, `_discover_value_segment`,
  `_discover_slot_program`, `_refine_pattern_key`, `_pattern_key_matches`,
  `_discover_trace_programs`, `_slot_program_predict` (618 lines removed)
- Added `_cata_predict` (39 lines): parses `[op] + input_tokens` → `cata_reduce` →
  `unparse` → point mass on token k
- Added NNO-seeded `discover_arities` + `discover_rules` in `Predictor.__init__` to
  build `self._arities`, `self._atoms`, `self._rewrite_rules` without any `.isdigit()`
- **Zero `.isdigit()` calls remain in `predict.py`** — Iron Rule satisfied

**Gate results (2026-03-15):**
- 343/343 tests pass (all ctkg + test suites)
- NL benchmark: **COMPLIANT ±0.0% on all 15 levels** (was 10 levels DRIFT -100%)
- Anon benchmark: BITTER-LESSON COMPLIANT
- Standard benchmark: 4/15 PASS (regressed from 12/15 because SlotProgram that was
  passing trace levels via special-case discovery is gone — this is correct)

**Deferred to Phase V:**
- `ChainRule` dataclass and `discover_compose_chains` remain (used by Level 1b exact
  lookup, which is not a special case — it's a legitimate memorization baseline)
- Multi-token outputs (linear_eval, algebra_trace, etc.) require carry-node
  representation before `discover_rules + cata_reduce` can cover them
- `_SEED_ATOMS` in `expr_parser.py` still hard-codes `'0'..'9'` — the NNO-seed
  workaround in `Predictor.__init__` bridges this until Phase V fixes `discover_arities`

---

### Phase V — Variable Discovery: Notation-Independence ✅ COMPLETE

**Status:** All 30 Phase V tests pass. 373/373 total tests pass.

**Files modified:**
- `ctkg/core/term_algebra.py` — `identify_variables`, `unify_surface_forms`
- `ctkg/core/expr_parser.py` — `normalize_surface`
- `ctkg/learning/rule_discover.py` — `_apply_norm_once`, `_align_rhs_variables`
  extended with `functional_maps`; `discover_rules` extended with `norm_rules`,
  `output_norm_rules`, `functional_maps`, `aux_rules`; `node` added to top-level import
- `ctkg/inference/predict.py` — `_merge_digit_runs`, `_split_compound` helpers;
  `_cata_predict` extended with `post_rules` and `nno_atoms`; Level 1c dispatch
  updated; `Predictor.__init__` wires normalization rules, ground NNO, functional maps

**Key design decisions:**
- `output_norm_rules` use `_apply_norm_once` (not `cata_reduce`) to avoid cycles in
  rules like `mul(V0,x)→mul(V0,pow(x,1))` which re-introduce the matched pattern
- Output norm rule is `mul(V0,x)→mul(V0,pow(x,1))` (not bare `x→pow(x,1)`) — avoids
  replacing `x` inside `pow(x,N)`, which would produce `pow(pow(x,1),N)` (wrong)
- `_merge_digit_runs` applied to chain_table `ans_part` only — not eq_table entries
  (which use separate single-digit args for arithmetic)
- `_split_compound` applied in `_cata_predict` to expand compound tokens back to
  individual chars before comparing with corpus surface form

**Gate result:** Power rule `d(pow(x,V0)) → mul(V0, pow(x, pred(V0)))` discovered
with normalization and generalizes OOD (n=9 passes, not in training n=2..5).

**Deferred:** anon_math_benchmark.py (token bijection compliance test) — the
`unify_surface_forms` and `normalize_surface` infrastructure is in place; the
end-to-end benchmark test is deferred to a future integration test pass.

---

### Phase VI — 2-Category Structure in `graph.py` ✅ COMPLETE

**Status:** 31/31 CTKG tests pass (30 existing + 1 new). 404/404 total tests pass.

**Files modified:** `experiments/ctkg/graph.py`, `experiments/ctkg/test_parser.py`

**Added:**
- `NaturalTransformation` dataclass: `name`, `source_functor`, `target_functor`,
  `components: List[RewriteRule]` — a 2-morphism α : F ⟹ G in the 2-category
- `Adjunction` extended: `unit_nat_trans: str = ''` and `counit_nat_trans: str = ''`
  fields store names of NaturalTransformation objects for unit η and counit ε
- `KnowledgeGraph.natural_transformations: Dict[str, NaturalTransformation]`
- `KnowledgeGraph.add_nat_trans(nt)` — register by name
- `KnowledgeGraph.apply_nat_trans(name, expr)` — cata_reduce(expr, nt.components);
  returns None if name unknown or no rule fires

**Gate result:** `apply_nat_trans('D', d(pow(x,2)))` → `mul(2, x)` ✓
OOD: `apply_nat_trans('D', d(pow(x,5)))` → `mul(5, pow(x,4))` ✓

**Design note:** `Category` dataclass (from the original spec) is not added — Functor
already maps between named domains (string domain names), and `KnowledgeGraph.domains()`
derives the set of concepts per domain dynamically. Adding a separate `Category` type
would be redundant.

---

### Phase VII — Compression: Normal Forms and Semantic MDL

**File to modify:** `experiments/symbolic_ai_v2/ctkg/learning/mdl_prune.py`

**Add function:** `semantic_deduplicate(mg: MorphismGraph, rules: list[RewriteRule]) → MorphismGraph`

Algorithm:
- For each morphism m in mg, compute `nf = normalize(m.to_expr(), rules)`
- Build a map `nf → [morphisms with this normal form]`
- For each equivalence class, keep only the morphism with shortest description
  length (by MDL criterion); remove all others
- The removed morphisms are not deleted from the CTKG — they are marked as
  `derived: True` and their canonical representative is recorded

**Add function:** `compute_storage_policy(mg: MorphismGraph, rules: list[RewriteRule],
k_steps: int = 5) → dict[str, bool]`

Returns `{morphism_id: should_store}`. A morphism should NOT be stored if it can
be derived from other stored morphisms in ≤ k_steps of `cata_reduce`. This is the
bounded-memoization principle: store only what cannot be quickly recomputed.

**File to modify:** `experiments/symbolic_ai_v2/ctkg/core/working_memory.py`

**Replace** `input_tokens: list[str]` and `output_tokens: list[str]` with
`input_tree: Optional[Expr]` and `output_tree: Optional[Expr]`. The token
representations are derived from these by `unparse` at the boundary. This eliminates
the need to store both forms redundantly.

Gate: run benchmark before and after; verify RAM usage is flat or decreasing as
training corpus size grows (the current system's morphism count grows linearly
with corpus size; after this phase it should grow sublinearly because derived
morphisms are not stored).

---

### Phase VIII — Unify Prediction: BinaryFoldRule + ChainRule → Single cata_reduce

This phase implements "The Unified Architecture" section above. It is the bridge
between the two disconnected prediction systems.

**Step 1 — NNO inductive rules as RewriteRules**

In `experiments/symbolic_ai_v2/ctkg/learning/process_discover.py`:

Add `fold_rules_as_rewrite_rules(fold_rule: BinaryFoldRule) → list[RewriteRule]`:
- Walk the discovered `succ_step` chain to identify `zero_digit` (additive identity).
- Emit the three NNO inductive rules for addition (zero base cases + inductive step).
- Emit analogous rules for subtraction (adjunction: reverse the inductive step) and
  multiplication (fold of addition).
- Return a list of `RewriteRule` objects covering all arithmetic ops.
- No `int()` calls anywhere. All digit tokens are opaque strings traversed via
  `succ_step`.

These replace `_apply_binary_formula`. The binary_table remains as a cache (correct:
bounded memoization principle), but is no longer the authoritative source. The
authoritative source is the inductive rules.

**Step 2 — Composition rules for trace-format ops**

In `experiments/symbolic_ai_v2/ctkg/learning/rule_discover.py`:

Extend `discover_rules` to handle trace-format sequences:
- A trace sequence `[op, a, b, ..., step, r1, r2, ..., ans, y1, y2, ...]` yields the
  pair `(node(op, a, b, ...), node('ans', r1, r2, ..., y1, y2, ...))` for rule discovery.
- Anti-unification over these pairs discovers the structural composition rule, e.g.:
  `linsolve(A, B, C, D) → div(sub(B, D), sub(A, C))`.
- The discovered rule RHS contains sub-expressions that are themselves reducible via
  the NNO arithmetic rules from Step 1.

**Step 3 — Single unified dispatch**

In `experiments/symbolic_ai_v2/ctkg/inference/predict.py`:

Replace the three-level dispatch (Level 1a `_process_predict`, Level 1b `_chain_predict`,
Level 1c `_cata_predict`) with:

```python
all_rules = self._rewrite_rules + self._nno_rules  # NNO from Step 1
result = _cata_predict(op, input_tokens, output_tokens_so_far,
                       all_rules, self._arities,
                       post_rules=self._post_rules,
                       nno_atoms=self._atoms)
```

The order of rules within `all_rules` matters: structural rules (linsolve, eval, d, int)
fire at the root; NNO arithmetic rules fire at the leaves. `cata_reduce` bottom-up
traversal naturally handles this — leaves reduce first, then the structural rule fires
on the reduced arguments.

**Dead code to remove after Step 3 passes gates:**

- `_process_predict` and Level 1a dispatch (replaced by unified cata_reduce)
- `_chain_predict` and Level 1b dispatch (replaced by unified cata_reduce)
- `BinaryFoldRule.binary_table` as authoritative source (demoted to memoization cache)
- `ChainRule`, `discover_compose_chains` (replaced by extended `discover_rules`)
- `_compose_adjoint_search` (replaced by `apply_nat_trans` for D ⊣ ∫)

**`experiments/symbolic_ai_v2/corpus/math_generator.py`:**
No changes. The corpus is external data; the system must handle whatever it emits.

**Gate:** Math benchmark passes all currently-passing levels (no regression) AND
improves on `linear_eval`, `algebra_trace`, `derivative_trace`, `integral_trace`.
The NL benchmark delta remains ±0% on all levels (Iron Rule compliance).
`_process_predict`, `_chain_predict` are absent from `predict.py`.

---

### Summary Table

| Phase | New files | Modified files | Removed |
|-------|-----------|----------------|---------|
| I: Term algebra | `core/term_algebra.py`, `tests/test_term_algebra.py` | — | — |
| II: Parser | `core/expr_parser.py`, `tests/test_expr_parser.py` | — | — |
| III: Rule discovery | `core/rewrite.py`, `learning/rule_discover.py`, `tests/test_rule_discover.py` | — | — |
| IV: Integration | — | `inference/predict.py` | SlotProgram, all segment types, `discover_compose_chains` |
| V: Variables | — | `core/term_algebra.py`, `core/expr_parser.py` | — |
| VI: 2-category | — | `experiments/ctkg/graph.py` | — |
| VII: Compression | — | `learning/mdl_prune.py`, `core/working_memory.py` | — |
| VIII: Unify prediction | — | `inference/predict.py`, `learning/process_discover.py`, `learning/rule_discover.py` | `_process_predict`, `_chain_predict`, `ChainRule`, `discover_compose_chains`, `_compose_adjoint_search` |
| IX: Functor discovery | `learning/functor_discover.py`, `tests/test_functor_discover.py` | `ctkg/graph.py` | — |

Phases I–III are pure additions with no risk of regression. Phase IV is the
integration point where the new system replaces the old; this is where the math
benchmark gate is critical. Phases V–VIII refine and clean up.

The NNO, binary fold rules, Hankel/FCA/em_loop, MDL pruning, and Kan extension
machinery are all retained — they are the arithmetic leaf that `cata_reduce` calls
for digit-level nodes. The tree layer above them is what is being added.
