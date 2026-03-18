# Fixing Generalization — Part 2

> **Status: Brainstorming phase.**

---

## The Iron Rule

**There is never, ever a special case or code for one.**

Any rule, segment type, handler, or pattern that only fires for a specific value,
specific digit width, or specific surface form is a violation. If a rule exists,
it must be derivable from a universal principle that applies to all inputs of the
same algebraic class. No `int()` calls. No hardcoded token names. No arity tables.

---

## What the Relational Triple Store Got Right (and Where It Falls Short)

The relational triple store approach is directionally correct. The core insight —
some tokens are structural separators, some are values; named roles > positional
arities — is a genuine improvement. `linear_trace` at 100% OOD proves it works
for the single-step case.

What's wrong is the **scope**: the relational store handles single-step ops but
doesn't compose. A Kleisli morphism is **two** relational lookups composed via
bind. The store needs to discover that `linear_trace` decomposes as `mul` then
`add`, not that it's a single undecomposable relation.

---

## Trace Ops Are Kleisli Morphisms

The traces (`linear_trace`, `power_trace`, etc.) are not being solved because the
system doesn't recognise them as what they are: **Kleisli morphisms**.

`linear_trace(a, b, x, at, c)` is `linear_eval(a, b, x, at, c)` lifted into the
**Writer monad**:

```
linear_trace(a, b, at, c) =
  let step = mul(a, c) in       -- η: inject into Writer
  let ans  = add(step, b) in    -- μ: bind + collapse
  [step, ans]
```

The system already knows how to compute `linear_eval`. It does not know that
`linear_trace` is just `linear_eval` with a Writer monad wrapper. If it did,
`linear_trace` would be solved trivially. Similarly: `power_trace` is `pow`
lifted into Writer. `derivative_trace` is `sym_diff` lifted into Writer.

### The `step` and `ans` delimiters are monadic operations

- `step` = Kleisli unit η (inject intermediate result into the monad)
- `ans` = Kleisli multiplication μ (collapse to final result)

Multi-step traces are Kleisli morphisms in a monad. The system needs to discover
that `linear_trace(a, b, x)` is `mul(a, x) >>= \s -> add(s, b)` where `>>=` is
Kleisli composition.

---

## What the Architecture Gets Right

The architecture isn't wrong — the NNO + FC + relational store is the right
substrate. What's missing is **monad structure discovery**: learn that trace ops
are Writer monad liftings of scalar ops, and reconstruct the trace by running the
scalar op step-by-step.

Concretely: after RelationStore learns `linear_trace`, it should discover that:

1. The `step` output role = the result of applying some known BFM op to the inputs
2. The `ans` output role = applying another known BFM op to `step` + remaining inputs

This is already what `discover_relation_rules` does for the single-step case. For
multi-step, it needs to discover the **composition chain** — which is exactly what
`discover_compose_chains` was originally designed for, before it got buried under
the arity machinery.

### Path forward

Extend `discover_relation_rules` to handle multi-output-role sequences by building
a **dependency graph** over output roles. `step → ans` is a topological sort order.
The algorithm already handles this for `linear_trace` (step before ans). It needs
to handle `power_trace` (multiple intermediate steps).

---

## Theoretical Foundations (Brainstorm)

The following three areas of mathematics are directly relevant to the next
architectural steps. These are notes from a brainstorming session.

---

### 1. Topos / Topoi

A **topos** is a category that behaves like the category of sets — but in a
universe where logic is constructive (no excluded middle by default).

**Two defining properties:**

1. **Cartesian closed** — exponential objects `B^A` exist. The "space of all
   functions A→B" is itself an object in the category. This is what makes lambda
   calculus internal to it.

2. **Subobject classifier Ω** — a special object generalising `{true, false}`. A
   "subset" `S ⊆ A` corresponds exactly to a morphism `A → Ω`. In Set, Ω =
   {⊤, ⊥}. In a non-Boolean topos, Ω has more "truth values."

The **internal logic** of any topos is higher-order intuitionistic type theory.
You can do mathematics *inside* the topos — quantify over objects, form
propositions, prove theorems — and the proofs are morphisms.

A **Grothendieck topos** is a category equivalent to sheaves on a site (a category
with a Grothendieck topology). These arise naturally in algebraic geometry and
forcing (Cohen proved independence of CH by constructing a topos where CH fails).

**Why it matters for this project:** If the system's knowledge is a sheaf (locally
consistent, globally potentially inconsistent), then a topos is the right home for
it. The subobject classifier gives a categorical handle on "this sequence belongs
to relation X" as a morphism, not a set-membership predicate. The internal logic
of a topos is the right setting for expressing what the system knows and doesn't
know, without committing to classical logic.

---

### 2. Lambda Calculus

A formal system for computation as function application and abstraction. Three
syntactic forms:

```
M ::= x          -- variable
    | λx. M      -- abstraction (anonymous function)
    | M N        -- application
```

One reduction rule (β-reduction):
```
(λx. M) N  →β  M[x := N]    -- substitute N for x in M
```

**Everything is a function.** There are no built-in data types, no primitives,
no special cases. Church encodings demonstrate this: natural numbers are
functions that apply another function n times.

```
0 = λf. λx. x
1 = λf. λx. f x
2 = λf. λx. f (f x)
n = λf. λx. f^n x

succ = λn. λf. λx. f (n f x)
add  = λm. λn. λf. λx. m f (n f x)
mul  = λm. λn. λf. m (n f)
```

Apply `succ` to `2`: β-reduces to `λf. λx. f (f (f x))` = `3`. No integers,
no arithmetic primitives — just substitution. Booleans, pairs, lists, trees:
all encodable as lambda terms by the same principle.

**Recursion** requires a fixed-point combinator because λ has no self-reference:
```
Y = λf. (λx. f (x x)) (λx. f (x x))
Y f = f (Y f)
```
`Y f` is the value `v` such that `f(v) = v` — the fixed point. General
recursion is derivable from pure function abstraction.

**Untyped lambda calculus** is Turing-complete but some terms diverge
(`(λx. x x)(λx. x x)` loops). Fixed-point combinators exist.

**Simply typed (STLC)**: Types `A ::= base | A → B`. Every well-typed term
normalises. At the cost of expressiveness — can't write `Y` with a simple type.

**System F** (polymorphic lambda calculus) adds universal quantification:
`∀α. T`. A term of type `∀α. α → α` is the identity function for *any* type.
This is the formal mechanism for cross-domain transfer: the same lambda term
applies to multiple domains because its type is universally quantified over them.

**The Curry-Howard correspondence** — the deepest unification in logic/CS:

```
Types         ↔  Propositions
Terms         ↔  Proofs
A → B         ↔  "A implies B"
A × B         ↔  "A and B"
A + B         ↔  "A or B"
∀α. T         ↔  "For all types α, T holds"
```

A term of type `A → B` IS a proof of `A ⊢ B`. **Running a program and
normalizing a proof are the same computation** (β-reduction).

**Connection to CT:** STLC is the internal language of **Cartesian Closed
Categories**. Every CCC has an internal STLC (objects = types, morphisms =
terms), and every STLC model is a CCC. Topos = CCC + subobject classifier, so:
topos = higher-order lambda calculus. The internal language of a topos is the
**Calculus of Inductive Constructions** (CiC) — CoC plus inductive types. This
is the type theory underlying Coq and Lean.

**Why it matters for this project:** Every discovered rule is a lambda term (a
morphism in some CCC). `add(succ(a), b) = succ(add(a, b))` is a proof that
`add ∘ (succ × id) = succ ∘ add` — a naturality square. The Kleisli morphism
structure of trace ops (`linear_trace = mul >>= add`) is exactly the monadic
lambda calculus: `>>=` is the lambda calculus `let`-binding.

The trace format is literally do-notation:

```haskell
linear_trace a b c = do
  step <- mul a c      -- 'step' delimiter = Writer tell
  ans  <- add step b   -- 'ans' delimiter  = Writer tell
  return ans
```

Recognising this means: the system should learn that `step` and `ans` are
**monadic bind points**, not arbitrary token names.

---

### 3. Enriched Category Theory

Ordinary categories: `Hom(A, B)` is a **set**. Enriched categories: replace that
set with an **object of some monoidal category V**.

A **V-enriched category** C has:
- Objects
- For each pair `(A, B)`: a hom-**object** `C(A, B) ∈ V` (not just a set)
- Composition: a morphism `C(B,C) ⊗ C(A,B) → C(A,C)` in V
- Identity: `I → C(A,A)` where I is V's monoidal unit

**Key examples:**

| V | Enriched category is | Example |
|---|---|---|
| Set | ordinary category | any category |
| Ab (abelian groups) | preadditive category | R-Mod |
| Cat | strict 2-category | morphisms between morphisms |
| Vect | linear category | quantum groups, TQFT |
| `[0,∞]` (reals, addition) | **Lawvere metric space** | any metric space |
| `[0,1]` (probabilities) | probabilistic relation | Markov kernel |

**The Lawvere metric space insight:** a metric space is *exactly* a
`[0,∞]`-enriched category. Objects = points. `C(A,B) = d(A,B)`. The triangle
inequality IS the composition law: `d(A,C) ≤ d(A,B) + d(B,C)`. This is not
analogy — it IS the same structure.

**Change of base:** there are functors between enriching categories. The forgetful
functor `[0,∞] → Set` (a metric space has an underlying set of points) corresponds
to "forget the distances."

**Why it matters for this project:**

- The BFM (Binary Function Map) is a Set-enriched structure (morphisms are elements
  of digit sets). If you enrich over a probability monad, morphisms become
  distributions — soft prediction.
- If you enrich over `[0,∞]`, you get costs/distances for rule application. This is
  the natural setting for MDL pruning.
- The Cruttwell-Gavranovic lens framework for backprop uses enrichment to make
  gradient descent a categorical operation — relevant to the lens_update.py module.
- **Most directly relevant:** enriching the RelationStore over `[0,1]` (probability)
  instead of Set (lookup tables) would give soft rule matching — the system could
  say "this input *probably* matches rule R with confidence 0.8" rather than hard
  lookup / fail. This is the path to handling OOD inputs gracefully instead of
  falling through to n-gram.

---

## How These Three Connect

```
Topos         → the logical foundation: what the system "knows" is a subobject,
                classified by Ω. The internal logic is constructive.

Lambda calc   → the computation model: every rule is a lambda term, every trace
                is do-notation, every inference step is a β-reduction.

Enriched cats → the quantitative layer: replace hom-sets with probabilities,
                costs, or distances. Gives soft prediction, MDL, and lens updates
                as a single categorical structure.
```

Together they suggest a **unified representation**:

- Rules are typed lambda terms in the internal language of a topos
- The RelationStore is a Vect-enriched (or Stoch-enriched) free multicategory
- Trace ops are Kleisli morphisms in the Writer monad (lambda do-notation)
- Generalisation = finding the simplest lambda term (shortest proof) consistent
  with the observed training morphisms

---

## Lambda Calculus as the Reasoning Substrate

### Creative Problem-Solving as β-Abstraction + β-Reduction

Consider the following anecdote. A professor explains that rocket fuel must be
vaporized before combustion, and is atomized into the smallest possible droplets
to maximize surface area, causing faster evaporation. A student, with no
deliberate goal other than general interest in water infrastructure, immediately
thinks of applying the same technique to water purification — maximizing the
surface area of water droplets to lower the energy cost of evaporation.

The student's hypothesis about their own reasoning: some kind of graph traversal
on their own knowledge graph.

In lambda calculus terms, the reasoning step decomposes as follows.

The professor's statement is received as a **specific application** of a general
lambda term to a specific argument:

```
phase_transition_opt rocket_fuel atomization
```

The creative step is **β-abstraction** — recovering the general lambda term from
the specific application:

```
λsubstance. λtechnique.
  maximize_surface_area(substance, technique) → faster_phase_transition(substance)
```

Then **β-reduction** on a new argument:

```
phase_transition_opt water ?
```

The `?` is an open variable — the system is now searching for a `technique` that
maximizes water's surface area. The traversal the student experienced is
**type-directed search**: find lambda terms whose type unifies with the problem's
type, then apply them. It succeeds because `rocket_fuel` and `water` share a
common supertype — both are `substance_undergoing_phase_transition`. That shared
type is what makes the abstraction valid and the transfer non-arbitrary.

This is not metaphor. β-abstraction followed by β-reduction on a new typed
argument IS creative analogical reasoning. The traversal is not across random
associations; it is across the concept lattice, constrained by type unification.

### The Bitter-Lesson Requirement Met

Lambda calculus has exactly one computation rule (β-reduction) and three
syntactic forms. Every computation — arithmetic, symbolic differentiation,
language parsing, cross-domain analogy — is β-reduction. There is no special
case for any domain.

The Church numeral encoding shows why: `succ` applied to the token `'g'` works
identically to `succ` applied to `'1'` if both tokens satisfy the NNO universal
property (the same abstract type). This is precisely what the anonymization
test verifies. The anonymization test IS a type-checking test: it checks whether
the system uses the type of a token (its role in the algebra) or its surface form
(its string label).

### What Is Needed Beyond STLC

Three extensions are required:

**1. System F (polymorphism).** Cross-domain transfer requires `∀α. T`. The
rocket/water abstraction has type `∀α: phase_transition_substance. ...`. Without
universal quantification the system cannot form the abstraction — it can only
memorize specific applications.

**2. Dependent types (CiC).** The NNO's type depends on the carrier object. The
sheaf condition requires dependent products. The internal language of a topos is
CiC, not STLC or System F. In practice: the type of `add(a, b)` depends on the
types of `a` and `b`, which depend on what NNO the system has discovered. This
is a dependent type.

**3. The probability monad.** Lambda calculus is deterministic. The CTKG needs to
reason under uncertainty. The fix: morphisms become Kleisli morphisms in the
probability monad, `f: A → Dist(B)`. β-reduction with the probability monad is
probabilistic inference. A stochastic dependent type theory — exactly what a
sheaf of probability distributions is (enriching over `[0,1]` instead of `{T,F}`
for the subobject classifier Ω).

The two extensions — dependent types and the probability monad — compose cleanly
into stochastic dependent type theory.

### The Practical Implication: Partial Evaluation

The current prediction pipeline applies rules as **fully evaluated** beta-normal
forms — lookup tables (BFM, RelationStore, ChainTable). The result of running
`mul(a, c)` was pre-computed during training and stored.

What is missing is **partial evaluation**: the ability to hold an **open lambda
term** with unresolved variables and reduce it incrementally as new tokens arrive.
An open term is a lambda expression with free variables — it cannot be fully
reduced until those variables are bound. As each input token arrives, one more
variable gets bound and the term reduces one step further.

This is what would give the system the creative transfer in the anecdote: an
abstract lambda term waiting for a new argument, found by type unification with
the current problem. The "graph traversal" is: search the space of learned lambda
terms for one whose open type unifies with the current partial prefix, then
partially evaluate it with the tokens seen so far.

Concretely for trace ops: `linear_trace` is an open term
`λa. λb. λc. let s = mul(a,c) in add(s,b)`. As input tokens `a`, `b`, `c` arrive,
the term is partially reduced step by step. `step` is emitted when the first bind
is reduced; `ans` when the second is. The system does not look up `linear_trace`
in a table — it evaluates the lambda term.

---

## Categorical Completeness — Why Partial Implementations Fail

A full AGI system capable of passing the benchmark requires every one of the
following category-theoretic tools without exception. Partial or implicit
implementations do not suffice. The evidence for this is already in the
benchmark numbers: the one place where the system computes a genuine categorical
construction (the NNO initial algebra / fold colimit) achieves 100% OOD
generalization. Every other failure corresponds to a categorical construction
that is approximated, hacked around, or missing entirely.

---

### Sheaf Theory

**Current state:** 0% implemented in v2. The `symbolic_ai_v2/` system has no
sheaf structures at any layer — not in the morphism graph, not in the operad,
not in the prediction pipeline.

**What is missing:** CT_REFERENCE §10 gives the correct design. The context
category has discourse states as objects and context refinements as morphisms
(`c' → c` means `c'` is more specific). A presheaf `F: C^op → Set` assigns to
each context `c` the set of admissible representations `F(c)`, and to each
refinement `c' → c` a **restriction map** `F(c) → F(c')` that specialises the
representation as context becomes more specific. The sheaf condition: locally
consistent sections glue uniquely to global sections.

Every place in `predict.py` where prediction branches on context is a missing
restriction map. The guard `"eq" not in prefix` is the most explicit case: it is
a hardcoded discrimination between two objects in the context category (eq-format
context vs step/ans-format context). The correct design encodes this as a
restriction morphism from the general output-phase context to the specific
format context. The restriction map replaces the string check and generalises
correctly to novel format contexts.

The fixpoint iteration in `_fixpoint_iteration` uses n-gram keys as a proxy
for restriction maps, not genuine presheaf restriction. It is labelled
"presheaf restrict" in comments but does not satisfy the presheaf functoriality
condition — it does not compose restriction maps, it reuses n-gram lookups as
a heuristic.

**Why indispensable for AGI:** The context-dependency problem is the sheaf
problem. "Bank" at coarse context has two representations; at refined context
(financial conversation) it has one. This is the restriction map. Without sheaves
there is no principled mechanism to specialise general knowledge to specific
contexts. The system will accumulate knowledge from multiple domains with no
coherence guarantee and will require ever-more-elaborate ad hoc disambiguation
logic — exactly what the fallback cascade in `predict_next` is.

---

### Limits

**Current state:** ~10% implemented. The NNO is the initial algebra for
`F(X) = 1 + X` — an initial object in `Alg(F)`, which is a limit. The fold
engine uses this universal property correctly and is responsible for the system's
only genuine OOD generalization results.

**What is missing:** Pullbacks (constraint intersection), equalizers (equation
solving), and the codensity monad (principled limit-based completion from
training data) are all absent.

A **pullback** of `f: A → C` and `g: B → C` is the universal object `P` with
maps to `A` and `B` that agree on `C`. This is the categorical model of
constraint satisfaction: "find all entities that satisfy constraint A AND
constraint B simultaneously." Multi-step traces fail precisely because each
step imposes a constraint on the intermediate value and the final answer must
satisfy the conjunction. The system has no mechanism to compute this
intersection — it tries rules independently and falls to n-gram when they
conflict.

An **equalizer** of `f, g: A → B` is the universal object `E` where `f` and
`g` agree. This is equation solving: "find all `x` such that `f(x) = g(x)`."
The missing `linear_eval` path — find `x` such that `a*x + b = c` — is an
equalizer computation over the BFM.

The **codensity monad** `Ran_K id` (CT_REFERENCE §20) is the right Kan
extension of the identity along the observation functor
`K: examples → CTKG objects`. It is the most conservative generalization
forced by the training data — Occam's razor in categorical form. Every n-gram
fallback is an unprincipled approximation to this limit that discards all
discovered categorical structure.

**Why indispensable for AGI:** Limits are required for anything involving
multiple simultaneous constraints. Type-checking is a limit. World-model
coherence is a limit. Conservation laws are limits. Any system that cannot
compute categorical limits cannot enforce consistency between its beliefs.

---

### Colimits

**Current state:** ~20% implemented in v2. The NNO fold engine computes a
genuine colimit (the universal property of the initial algebra). The free
category construction `FC(G)` is a colimit (free object on the token graph).
`ctkg/core/operad.py` produces `MultiMorphism` objects via GraphZip / SEQUITUR
grammar induction — each grammar rule `R → (child_1, ..., child_k)` is a k-ary
operad morphism and the repeated digram replacement is an implicit coequalizer
(quotienting the free category by the discovered digram equations). The free
category quotient `FC(G)/E` is an implicit coequalizer more broadly, discovered
ad hoc rather than computed categorically.

**What is missing:** The left Kan extension formula
`(Lan_K F)(e) = colim_{K(c) → e} F(c)` is the categorical formulation of
generalization. It is not computed for any op except those handled by the
NNO fold engine.

The **JSD Kan extension** in Level 3 of the prediction pipeline is
misnamed. It computes `exp(-JSD(query, centroid_c))`-weighted nearest-neighbor
prediction — a heuristic, not a colimit. The genuine left Kan extension would
compute the colimit over the comma category `(K ↓ e)` of all training examples
`c` such that `K(c)` maps to `e`. The JSD weighting is an unprincipled
approximation that neither satisfies the universal property nor respects the
categorical structure discovered upstream.

**Coproducts** (`A ⊔ B`) are completely absent. The RelationStore has no
mechanism to express "this op can resolve to either interpretation A or
interpretation B." At the language level, natural language ambiguity requires
coproducts at every merge step. Without coproducts the system must commit to
one interpretation or fail.

**Pushouts** for rule merging: when the system has a rule for context A and
a rule for context B, and encounters context A∩B, the pushout of the two rules
is the correct rule for A∩B. There is no such mechanism.

The NNO result (100% OOD) is the proof of concept. The question for every
other op is: what is the diagram whose colimit gives the correct generalization
rule? For trace ops it is the Kleisli composition chain. For eq-format ops it
is the adjunction diagram. Computing these colimits is the core engineering
task of the next phase.

---

### Summary

| Construction | Current state | AGI necessity |
|---|---|---|
| Sheaves / restriction maps | 0% in v2 | Indispensable — context-dependency is the sheaf problem |
| Limits (pullback, equalizer, codensity) | ~10%, NNO initial algebra only | Critical — all multi-constraint satisfaction |
| Colimits (Kan ext, pushout, coproduct) | ~20%, NNO fold + FC + operad grammar quotient | Critical — all generalization to OOD inputs |

---

## Phase X — Remove All Fallbacks (Cleanroom Testing Gate)

**This phase must precede any serious evaluation of the categorical system.**

The prediction pipeline currently contains four fallback levels that fire when
the categorical levels fail:

- **Level 0 — n-gram**: direct left-context lookup in the Hankel index.
  An n-gram is not a categorical construction. It memorizes surface-form
  co-occurrence and has no generalization beyond seen contexts. For the
  arithmetic domain it will score near-zero on OOD inputs (it never saw
  anonymized digit tokens). For natural language it memorizes collocations
  without understanding structure. **Remove entirely** for evaluation.

- **Level 2 — Fixpoint iteration + morphism marginalization**: a heuristic
  iteration using n-gram keys as a proxy for presheaf restriction maps. Does
  not satisfy the sheaf condition. The "Sheaf obstruction detection" comment
  is misleading — it detects an iteration cycle, not a sheaf cohomological
  obstruction. **Remove entirely** for evaluation; replace with genuine
  sheaf restriction when sheaves are implemented.

- **Level 3 — JSD "Kan extension"**: a JSD-weighted nearest-neighbor
  prediction. Not a Kan extension. Does not satisfy the universal property
  of a left Kan extension. **Remove entirely** for evaluation; replace with
  the genuine left Kan extension colimit computation.

- **Level 4 — Uniform marginal**: uniform distribution over vocabulary.
  This is always wrong (it implies the system has no information about
  what token comes next). **Remove entirely** for evaluation.

**Exception:** Level 0.5 (FC direct lookup) and Level 0.6 (NNO chain
prediction) are not fallbacks — they are exact categorical computations
(initial algebra universal property, adjunction-mediated lookup). These stay.

**The gate condition:** after removing all fallbacks, every prediction must
be justified by a categorical construction that was explicitly implemented
and tested. A prediction failure becomes an honest failure (the system does
not know) rather than a lucky n-gram hit that masks a missing categorical
implementation. This is the only way to know what the system actually does.

In a production-ready AGI, a graceful fallback for genuinely missing
information is appropriate — but only after all the categorical constructions
above are implemented. The n-gram fallback in a production system should be
replaced by the codensity monad `Ran_K id` (the principled limit-based
completion), which is the categorical formalisation of "I don't know the
answer, but here is the most conservative prediction that is consistent with
everything I have observed."

---

## Phase XI — Unify the Hypergraph Layers

### The Current Situation

Two hypergraph representations exist in v2 and are not connected to each other:

- **`ctkg/core/operad.py` — `MultiMorphism`**: k-ary morphism produced by Phase 4
  (GraphZip / SEQUITUR grammar induction). Operates on typed concept distributions
  (`TypeDist = dict[ConceptId, float]`). Inputs are anonymous and positional:
  `child_type_dists[0]`, `child_type_dists[1]`. Always binary in practice because
  SEQUITUR produces digram rules. Single output. No named roles.

- **`ctkg/learning/relation_store.py` — `Relation`**: named-role tuple produced
  by Phase 5+ (named-role extraction from ChainRule sequences). Operates on raw
  token strings. Inputs and outputs are both named: `input_roles`, `output_roles`.
  Multiple output roles. Strictly more expressive.

They converged on the same mathematical structure — a hyperedge / multi-morphism
— from opposite directions, and the convergence was not noticed.

### Why This Happened

`operad.py` was part of the original v2 pipeline design, built to sit at the
end of the grammar-induction chain (HankelCount → FCA → MorphismGraph →
GraphZip → MultiMorphism). `relation_store.py` was built later as a direct
response to prediction failure — specifically to replace the broken arity-based
approach that was breaking the math benchmark. The person building
`relation_store.py` was focused on making `linear_trace` work and found named-role
tuples to be the right solution. This was correct. But it was built in isolation
from the operad, which already had a concept for multi-input operations.

The root cause is **accretion without architecture**: features were added to
solve immediate problems, each in isolation, without asking whether the existing
codebase already contained a structure representing the same mathematical concept.

### The Structural Irony

`operad.py` feeds into Level 2 (heuristic type-assignment iteration) and Level 3
(JSD nearest-neighbor approximation) — both of which are designated heuristic
fallbacks to be removed in Phase X.

`relation_store.py` feeds into Level 1c-relational — the structural prediction
level that is correct and should be extended.

The older, more theoretically motivated layer is feeding the wrong path. The
newer, more pragmatically motivated layer is feeding the right path. The
multi-morphism structure we actually need exists in the layer that was built to
fix a benchmark failure, not in the layer that was planned from the start.

### Secondary Violation: `KNOWN_INPUT_SEPS`

```python
KNOWN_INPUT_SEPS: frozenset[str] = frozenset({'x', 'at', 'dx'})
```

This pre-seeded set of domain-specific structural keywords is a violation of the
Iron Rule. The 80% threshold mechanism for learning separators from data is the
right idea. The seed set is a special case — it assumes that only `x`, `at`, and
`dx` can be structural separators, which is false for any domain outside symbolic
math. The correct design discovers structural separators purely from distributional
statistics with no pre-seeded set: any token that appears at a consistent position
in ≥80% of training sequences for a given op is a separator, regardless of its
surface form.

### The Fix

`Relation` is the canonical form. It is strictly more expressive than
`MultiMorphism`: named roles on both input and output sides, multiple output
roles, operates on concrete tokens. `MultiMorphism` is a degenerate special case:
anonymous, positional, binary-only, type-distribution level.

The unification:

1. Extend `Relation` to optionally carry type information alongside its token
   strings — `input_type_dists` and `output_type_dists` fields, derived from
   the ConceptLattice when available.
2. Every multi-input operation discovery mechanism — whether grammar induction
   (SEQUITUR) or named-role extraction — produces a `Relation`. SEQUITUR's
   grammar rule output is translated to a `Relation` at the point where types
   are assigned, rather than producing a separate `MultiMorphism`.
3. Since `MultiMorphism` only fed the heuristic levels being removed in Phase X,
   `operad.py`'s `MultiMorphism` class becomes orphaned after Phase X.
   The SEQUITUR grammar induction itself remains valuable for discovering
   hierarchical compression structure; only its output type changes.
4. Remove the `KNOWN_INPUT_SEPS` pre-seeded set. The separator discovery
   algorithm retains the 80% threshold but drops the explicit keyword list.
   Any token can be a separator if the data supports it.

### The Architectural Principle to Prevent Recurrence

This is the architectural equivalent of the Iron Rule, and must be stated as
explicitly:

**There is exactly one canonical representation for each mathematical concept
in the system. Before adding a new data structure, ask: does a structure already
exist in this codebase that represents this mathematical concept? If yes, use it.
If no, define it once, document it as canonical, and require all future code to
use it.**

In categorical terms: an object is determined by its relationships (Yoneda).
Two objects with the same relationships are isomorphic and must be identified —
taking the quotient that collapses them is not optional. `Relation` and
`MultiMorphism` existing as separate classes is a failure to take that quotient.

Practically, CTKG_ARCHITECTURE.md must maintain a **canonical type table**:

| Mathematical concept | Canonical Python type | File |
|---|---|---|
| Multi-input operation / hyperedge | `Relation` | `learning/relation_store.py` |
| Unary chain / NNO step | `ProcessRule` | `learning/process_discover.py` |
| Binary scalar function | `BinaryFoldRule` | `learning/process_discover.py` |
| Typed morphism (binary) | `CTKGMorphism` | `core/morphism_graph.py` |
| Token sequence with delimiters | `ChainRule` | `learning/process_discover.py` |

Any new code that introduces a type belonging to an existing row is a violation.
Any new code that introduces a new row must document the canonical type before
any implementation begins.

---

## Phase IX — Functorial Variable Discovery

*(Moved from Fixing Generalization Part 1)*

**New file:** `experiments/symbolic_ai_v2/ctkg/learning/functor_discover.py`

This phase discovers **functorial variables** — variables whose substitution sets
form consistent partitions across multiple rules, indicating a latent functor pair
with a natural transformation between them.

Example: after Phase III, the rule store might contain:
- `king → [V0, ruler]` with `V0 ∈ {man, woman}` across two examples
- `husband → [V0, spouse]` with `V0 ∈ {man, woman}` across two examples
- `son → [V0, child]` with `V0 ∈ {man, woman}` across two examples

The algorithm detects that `V0` is not truly free — it always takes values in the
same two-element partition `{man, woman}` across all three rules, with `man` always
corresponding to one functor and `woman` to the other.

**Functions to implement:**

- `collect_variable_values(rules: list[RewriteRule], corpus_examples) → dict`

  For each discovered rule and each variable in its lhs, collect the set of
  observed substitution values. Returns `{(rule_id, var_name): [values_observed]}`.

- `cluster_consistent_partitions(variable_values: dict) → list[FunctorCandidate]`

  Groups variables from different rules that share the same partition structure.
  Two variables `V` (from rule R1) and `W` (from rule R2) belong to the same
  functor pair if:
  1. Their observed value sets have the same cardinality
  2. There is a consistent bijection between the values (the same value always
     appears in the same slot across all rules containing either variable)

  Each group produces a `FunctorCandidate(partition_a: set, partition_b: set,
  supporting_rules: list[str])`.

- `register_as_nat_trans(candidate: FunctorCandidate, kg: KnowledgeGraph)`

  Converts a `FunctorCandidate` into a proper `NaturalTransformation` in the
  2-categorical CTKG:
  - Creates `Category ROLE` with abstract objects for each distinct role
  - Creates `Functor F_a` mapping ROLE → partition_a image
  - Creates `Functor F_b` mapping ROLE → partition_b image
  - Creates `NaturalTransformation η: F_a → F_b` with components from the
    consistent bijection

```
FunctorCandidate
  partition_a: frozenset[str]   # e.g. {man, king, husband}
  partition_b: frozenset[str]   # e.g. {woman, queen, wife}
  bijection: dict[str, str]     # man↔woman, king↔queen, husband↔wife
  supporting_rules: list[str]   # rule ids that provide evidence
  evidence: int                 # number of supporting rules
```

**New file:** `experiments/symbolic_ai_v2/ctkg/tests/test_functor_discover.py`

Gate: given a corpus with at least 3 role-entity pairs (king/queen, husband/wife,
man/woman), `cluster_consistent_partitions` recovers the gender partition as a
single `FunctorCandidate` with bijection `{man↔woman, king↔queen, husband↔wife}`
and evidence ≥ 3. The recovered `FunctorCandidate` is registered as a
`NaturalTransformation` in the knowledge graph.

---

## Master Roadmap

This roadmap is ordered by dependency. Each phase has a gate condition: a
measurable pass criterion that must be satisfied before the next phase begins.
The anonymization test (`anon_math_benchmark.py`) is a standing gate for every
phase — any phase that passes on standard tokens but fails on anonymized tokens
is a Iron Rule violation and must be fixed before proceeding.

Phases are grouped into five tiers. Tiers 1 and 2 are near-term. Tiers 3–5
are medium-to-long-term and correspond to implementing the categorical
constructions identified in the Categorical Completeness section.

---

### Tier 1 — Cleanroom (prerequisite for all measurement)

These phases must be completed before any benchmark numbers are meaningful.
Without them, passing scores may reflect fallback memorization rather than
structural understanding.

**Phase X — Remove All Fallbacks**
Remove Levels 0 (n-gram), 2 (heuristic fixpoint), 3 (JSD approximation), and
4 (uniform marginal) from `predict.py`. Every prediction must be justified by
an explicit categorical construction. A prediction failure becomes an honest
failure rather than a masked one.
- Gate: `predict_next` returns `{}` (empty dict) when no categorical level
  fires, rather than returning a heuristic distribution. All existing tests
  still pass. Benchmark numbers reflect structural coverage only.

**Phase XI — Unify Hypergraph Layers**
Establish `Relation` as the single canonical type for multi-input operations.
Translate SEQUITUR grammar rule output into `Relation` objects rather than
`MultiMorphism`. Remove `KNOWN_INPUT_SEPS` pre-seeded set; separator discovery
uses only the 80% distributional threshold. Publish the canonical type table
in CTKG_ARCHITECTURE.md and enforce it going forward.
- Gate: `MultiMorphism` is no longer produced by any active code path.
  `KNOWN_INPUT_SEPS` is gone. Separator discovery correctly identifies `x`,
  `at`, `dx` from the math corpus data alone without being told their names.
  All existing tests pass.

---

### Tier 2 — Relational System Extension (short-term)

These phases extend the existing relational prediction system to cover the
cases that currently fall to zero with the fallbacks removed.

**Phase XII — Multi-Step Kleisli Chain Discovery**
Extend `discover_relation_rules` to build a dependency graph over output roles
rather than assuming a flat single-step structure. `step → ans` is a two-node
chain; `power_trace` has multiple intermediate steps. The algorithm performs a
topological sort over output roles and discovers the BFM op for each step
conditioned on the results of all prior steps.
- Gate: `power_trace` and `derivative_trace` reach >50% OOD accuracy using
  the relational path only (no fallbacks). `linear_trace` remains at 100%.

**Phase XIII — Eq-Format Relational Prediction**
Remove the `"eq" not in prefix` guard from the Level 1c-relational block.
Extend `RelationStore` to handle eq-format sequences (single output role after
`eq`) as a degenerate case of the general multi-role schema. The eq-format is
a Kleisli morphism with one output role — it is not a different kind of thing.
- Gate: `linear_eval` returns to non-zero OOD accuracy via the relational
  path. The format guard is replaced by data-driven schema detection.

**Phase XIV — Positional Schema for Separator-Free Ops**
Ops without named input separators (e.g. `pow`, `sq`, `bern_p1`) cannot use
the current RelationStore which requires separator tokens. Extend the schema
learning to assign positional role names (`p0`, `p1`, ...) when no separators
are found. This is the degenerate case of named roles where names are ordinals.
- Gate: `pow` and `sq` are handled by the relational path with positional
  roles. No special-case handling for separator-free ops.

**Phase IX — Functorial Variable Discovery** *(see section above)*
Discover natural transformations between functor pairs from consistent variable
partition patterns across rules. Produces `FunctorCandidate` objects registered
as `NaturalTransformation` in the knowledge graph.
- Gate: gender partition recovered from word-analogy corpus with evidence ≥ 3.

---

### Tier 3 — Colimit Constructions (medium-term)

These phases implement the categorical constructions identified as missing in
the Colimits section. Each is a genuine universal construction, not a heuristic.

**Phase XV — Coproducts (Multiple Competing Rules)**
The coproduct `A ⊔ B` in a category is the "either A or B" type: the disjoint
union with injection morphisms `i_A: A → A ⊔ B` and `i_B: B → A ⊔ B`. A
function `[f, g]: A ⊔ B → C` handles both cases. In the RelationStore,
multiple qualifying rules for the same output role form exactly this structure:
each rule is an injection (one path through the coproduct); prediction is the
copairing that selects the right injection based on context evidence.

This matters at two scales:

- **Arithmetic (degenerate case):** For fully-determined ops like `add`, only
  one rule fires per output role — the coproduct degenerates to a point. The
  implementation is backward-compatible: a single-element alternatives list.
  Evidence weighting does nothing. No behavior change for arithmetic.

- **NLP (non-trivial case):** "bank" in an underspecified context has output
  type `financial_institution ⊔ river_margin`. Both interpretations are valid
  until context (the Grothendieck topology's restriction maps, Phase XIX)
  resolves the ambiguity. Without coproduct support, the system must commit
  to one interpretation at rule-discovery time — wrong for any genuinely
  ambiguous token.

Concretely: `discover_relation_rules` previously kept only the single
highest-evidence rule per output role. It now keeps **all** rules that meet
the evidence and tolerance thresholds. `predict_alternatives_from_rules`
evaluates all alternatives and returns a weighted distribution. Level 1c in
`predict_next` uses this distribution directly, making prediction a Kleisli
morphism `input → Dist(output)` in the probability monad rather than a
deterministic function `input → output`.

- Gate: `discover_relation_rules` returns multiple rules for the same output
  role when the training data supports them. `predict_alternatives_from_rules`
  returns a multi-element list for genuinely ambiguous ops, a single-element
  list for deterministic ops. All arithmetic benchmarks remain at their
  current OOD accuracy (no regression). Prediction for an op with two
  competing rules of equal evidence produces a 50/50 distribution.

**Phase XVI — Left Kan Extension (Genuine Colimit Generalization)**
Replace the JSD nearest-neighbor approximation (`KanExtension` in
`core/kan_extension.py`) with the genuine left Kan extension computation:
`(Lan_K F)(e) = colim_{K(c) → e} F(c)`. The comma category `(K ↓ e)` indexes
all training examples `c` such that `K(c)` has a morphism to `e`; the colimit
over this category gives the canonical extension. This is the principled
replacement for the n-gram fallback for novel inputs.
- Gate: for a novel op not seen in training, the Kan extension correctly
  predicts the output by generalizing from the most structurally similar
  seen ops. Anonymization test passes.

**Phase XVII — Equalizers (Equation Solving)**
Implement equalizer computation over the BFM: given `f, g: A → B`, find all
inputs `x` where `f(x) = g(x)`. The immediate application: `linear_eval`
inverse path — find `x` such that `a*x + b = c` — is an equalizer of
`λx. a*x + b` and the constant function `λx. c` over the digit domain.
- Gate: `linear_eval` solves the inverse correctly OOD (given `a`, `b`, `c`,
  find `x`). The same code path solves any equation of the same algebraic
  form without special-casing `linear_eval`.

**Phase XVIII — Pullbacks (Constraint Intersection)**
Implement pullback computation: given `f: A → C` and `g: B → C`, compute
the universal object `P` of all pairs `(a, b)` with `f(a) = g(b)`. This is
the categorical model of multi-constraint satisfaction. The immediate
application: multi-step trace verification, where the intermediate `step`
value must simultaneously satisfy the constraint from the first BFM op and
serve as valid input to the second.
- Gate: `algebra_trace` and `bernoulli_trace` reach non-zero OOD accuracy
  by constraint intersection. The pullback code is domain-agnostic.

---

### Tier 4 — Context and Sheaves (long-term)

These phases implement the sheaf-theoretic foundation identified as 0%
implemented and indispensable for AGI.

**Phase XIX — Context Category and Restriction Maps**
Build the context category: discourse states as objects, context refinements
as morphisms (`c' → c` when `c'` is more specific). Every prediction rule
becomes a section of a presheaf `F: C^op → Set`. Implement restriction maps
that specialise a general rule to a specific context. Replace all string-branch
context detection in `predict.py` (including the format guards, the phase
detection, and the op-type detection) with restriction map applications.
The sheaf condition — locally consistent sections glue to global sections —
is enforced as a consistency check on the learned rules.
- Gate: the `"eq" not in prefix` guard and all analogous format-detection
  guards are gone. Context specialisation is handled by restriction maps.
  Multi-domain knowledge (arithmetic + NLP) integrates without contradiction.

**Phase XX — Partial Evaluation and Open Lambda Terms**
Store discovered rules as partially-evaluated lambda terms with free variables,
not as fully-evaluated lookup tables. As input tokens arrive, reduce the term
one step at a time — bind one variable, reduce one β-redex. `linear_trace`
prediction is produced by partial evaluation of
`λa. λb. λc. let s = mul(a,c) in add(s,b)`, not by ChainTable lookup.
This is the mechanism for the creative reasoning in the anecdote: an abstract
lambda term waiting for a new argument, found by type-directed search and
then partially evaluated with the current prefix.
- Gate: trace op prediction is provably produced by lambda term evaluation,
  not lookup. A novel trace op not in training data can be handled if its
  lambda term can be assembled from known primitive terms. Creative transfer
  test: given a new domain with the same algebraic structure, the system
  correctly applies the known lambda term to the new domain.

---

### Tier 5 — Full Type Theory (far-term)

These phases implement the complete internal language of the topos — CiC with
the probability monad — which is the theoretical target of the architecture.

**Phase XXI — Dependent Type Inference**
Implement a lightweight dependent type system where the type of a term can
depend on the types of its arguments. The NNO's type depends on the carrier
object; `add(a, b)`'s output type depends on the NNO types of `a` and `b`.
Type inference runs at discovery time and assigns every learned `Relation` a
dependent type. The anonymization test becomes a theorem: if two tokens have
the same type (satisfy the same NNO universal property), they are
computationally indistinguishable.
- Gate: the type system correctly assigns dependent types to all arithmetic
  ops from data alone. Tokens that satisfy the NNO property are automatically
  grouped into the same type regardless of surface form. Anonymization test
  passes with 0% accuracy gap.

**Phase XXII — Probability Monad and Enrichment over [0,1]**
Lift morphisms from `A → B` to Kleisli morphisms `A → Dist(B)` in the
probability monad. Enrich the RelationStore over `[0,1]` instead of `{hit,
miss}`: every rule match produces a confidence score, not a binary decision.
The subobject classifier Ω becomes a sheaf of probability distributions —
graded truth replacing classical Boolean truth. This is the categorical
foundation for soft prediction, uncertainty quantification, and MDL pruning
in a single unified structure.
- Gate: `RelationRule.evaluate` returns a `dict[str, float]` (distribution)
  instead of `Optional[str]`. Confidence propagates through Kleisli
  composition. The system gracefully degrades on uncertain inputs instead of
  failing hard.

---

### Gate Summary

| Phase | Tier | Key gate condition |
|---|---|---|
| X — Remove fallbacks | 1 | `predict_next` returns `{}` on miss; no n-gram/fixpoint/JSD |
| XI — Unify hypergraph | 1 | `MultiMorphism` gone; `KNOWN_INPUT_SEPS` gone |
| XII — Multi-step Kleisli | 2 | `power_trace`, `derivative_trace` > 50% OOD |
| XIII — Eq-format relational | 2 | `linear_eval` > 0% OOD via relational path |
| XIV — Positional schema | 2 | `pow`, `sq` via relational path with positional roles |
| IX — Functor discovery | 2 | Gender partition recovered with evidence ≥ 3 |
| XV — Coproducts | 3 | All rules meeting evidence threshold stored; `predict_alternatives_from_rules` returns weighted distribution; no arithmetic regression |
| XVI — Left Kan extension | 3 | Novel op predicted from structural similarity |
| XVII — Equalizers | 3 | `linear_eval` inverse OOD via equalizer computation |
| XVIII — Pullbacks | 3 | `algebra_trace`, `bernoulli_trace` > 0% OOD |
| XIX — Context / sheaves | 4 | All format guards replaced by restriction maps |
| XX — Partial evaluation | 4 | Trace ops produced by lambda term evaluation |
| XXI — Dependent types | 5 | Anonymization gap = 0% by theorem |
| XXII — Probability monad | 5 | Rules return distributions; confidence propagates |

**Standing gate (every phase):** `anon_math_benchmark.py` accuracy within ±2%
of `math_benchmark.py` on all tasks that were passing before the phase began.
Any regression on the anonymization test is a Iron Rule violation.
