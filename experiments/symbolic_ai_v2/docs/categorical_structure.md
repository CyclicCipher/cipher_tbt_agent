# Categorical Structure for the CTKG

This document specifies the category theory machinery the CTKG needs to
generalise beyond training data. Each concept is defined precisely, illustrated
with a concrete example in our system, and assessed for implementation priority.

The running example throughout: **multi-digit succession**. The system trains
on counting 0-50 and must answer succ(5000) = 5001. This requires discovering
that succession is a compositional operation on digit sequences with carry
propagation — not memorising individual successor pairs.

---

## Part 1: Definitions and Examples

### 1. Products

**Definition.** Given objects A, B in a category C, their product A x B is an
object with projections pi_1: A x B -> A and pi_2: A x B -> B such that for
any X with f: X -> A and g: X -> B, there exists a unique h: X -> A x B with
pi_1 . h = f and pi_2 . h = g.

**In our system.** The number 25 is a two-token sequence [2, 5]. As a product,
it is an element of Digit x Digit with pi_1(25) = 2 (tens) and pi_2(25) = 5
(units). The universal property says: anything that can produce both a tens
digit and a units digit factors uniquely through this product.

**Why it matters.** Products enable compositional representation. Instead of
learning succ for every two-digit number (100 facts), the system learns
successor on single digits (10 facts) plus carry logic. The product structure
decomposes the problem. Without products, multi-digit numbers are opaque;
with them, the system reasons about each digit position independently.

**How to discover.** During consolidation, when two tokens consistently appear
at fixed relative positions (e.g., tokens at position 0 and position 1 in
two-digit counting observations), the product structure is inferred. The
projections are the "extract position 0" and "extract position 1" operations.

### 2. Coproducts

**Definition.** Given A, B, their coproduct A + B has injections iota_1: A -> A + B
and iota_2: B -> A + B such that for any X with f: A -> X and g: B -> X, there
exists a unique h: A + B -> X with h . iota_1 = f and h . iota_2 = g.

**In our system.** The coproduct Digit + Operator represents "either a digit or
an operator." When the system processes a token, it is injected into the
coproduct via its type. The universal property says: any uniform treatment of
both digits and operators factors through this coproduct.

**Why it matters.** Coproducts handle alternatives. The two cases of succession
— carry (ones=9) and no-carry (ones!=9) — form a coproduct. The successor
function case-splits via the coproduct: apply one rule for the carry case,
another for no-carry, and the coproduct composes them.

**How to discover.** When consolidation finds that a morphism (e.g., successor)
has two distinct behaviours depending on some condition (ones digit = 9 vs not),
the condition defines a coproduct decomposition.

### 3. Limits

**Definition.** The limit of a diagram D: J -> C is an object L with cone
morphisms lambda_j: L -> D(j) for each j in J, commuting with the diagram's
internal morphisms, and universal: any other cone factors uniquely through L.

Limits are the dual of colimits. Where colimits merge (quotient), limits
refine (intersect). A limit extracts what all objects in the diagram have
in common.

**Special case: Equalizer.** Given f, g: A -> B, the equalizer is the subobject
of A on which f and g agree. Comes with e: E -> A such that f . e = g . e.

**In our system.** The limit of all observed two-digit successions extracts
the common structure: "increment ones digit, carry if 9." The equalizer of
"increment-then-project-ones" vs "project-ones-then-increment" isolates the
non-carry cases (ones != 9). The carry boundary is discovered automatically
as the equalizer's complement.

**Why it matters.** Limits extract invariant structure. The limit of training
examples is the abstract pattern that transfers to unseen inputs. For
succ(5000), the limit of all two-digit succession examples gives the rule
that applies at EVERY digit position, regardless of how many digits.

### 4. Initial and Terminal Objects

**Definition.** An initial object 0 has exactly one morphism 0 -> X for every X.
A terminal object 1 has exactly one morphism X -> 1 for every X.

**In our system.** The terminal object is the "trivial observation" — "something
happened." Every observation maps to it by forgetting all structure. The initial
object is the empty observation — no tokens, no edges. It maps uniquely into
any observation (vacuously).

More important: in the NNO (Natural Number Object), the initial object provides
the zero morphism z: 1 -> N. The number 0 IS the morphism from the terminal
object into the number line. This is the foundation for recursive definitions.

**Why it matters.** The initial object is the starting point for all recursive
constructions. The NNO is defined relative to it: "N is the object with a
morphism from 1 (zero) and an endomorphism (successor) that is universal for
recursion." Without the initial object, there is no NNO, and without the NNO,
there is no universal recursion.

### 5. Isomorphism Detection

**Definition.** A and B are isomorphic if there exist f: A -> B and g: B -> A
with g . f = id_A and f . g = id_B.

**In our system.** Two subgraphs are isomorphic when a bijection between their
nodes preserves all edge structure in both directions. The subgraph around
"25 -> 26" (no carry) and "35 -> 36" (no carry) are isomorphic: both involve
incrementing the ones digit from 5 to 6 with tens digit unchanged. The
isomorphism maps 2->3 on the tens position and is identity on everything else.

Similarly, "19 -> 20" and "29 -> 30" are isomorphic: both carry.

**Why it matters.** Isomorphism is the core mechanism for transfer. Once the
system detects that carry-free succession is isomorphic across all tens digits,
it factors this into: (a) an abstract carry-free pattern, and (b) a
parametrisation by tens digit. This factorisation is what enables answering
succ(75) from having only seen succ(25) and succ(35).

**How to detect.** During consolidation, compare morphism profiles of subgraphs.
Two nodes with identical in/out edge patterns (up to relabeling) are isomorphism
candidates. Verify by checking that the mapping preserves all composition.

### 6. Duality

**Definition.** Every category C has an opposite category C^op with the same
objects but all morphisms reversed. Every theorem in C has a dual in C^op.

**Dual pairs in our system:**

| Construction | Dual | Computes | Dual computes |
|---|---|---|---|
| Colimit | Limit | Merge/identify | Refine/intersect |
| Coproduct | Product | Either/or | Both/and |
| Initial | Terminal | Canonical source | Canonical sink |
| Coequalizer | Equalizer | Force agreement | Find agreement |

**Why it matters.** Every consolidation operation (merging similar things)
automatically gives an analysis operation (finding shared structure) by
reversing arrows. The system implements one and gets the other for free.

If consolidation discovers colimits (merging co-occurring tokens into concepts),
it also discovers limits (the common structure of all concept instances) without
additional code — just reverse the morphisms and run the same algorithm.

### 7. Monads

**Definition.** A monad on C is (T, eta, mu) where T: C -> C is an endofunctor,
eta: Id -> T is the unit (embedding), mu: T^2 -> T is the multiplication
(flattening), satisfying associativity and unit laws.

**The Kleisli category.** Morphisms A -> B in the Kleisli category C_T are
morphisms A -> T(B) in C. Composition chains through T using mu to flatten.

**In our system.** The "one-step reachability" endofunctor T maps a token to
the set of tokens reachable from it. T(25) = {26, 24, ...}. T^2(25) = tokens
reachable in two steps. mu flattens: reachable-in-two-steps is still reachable.
eta: every token is trivially reachable from itself.

For carry propagation: the monad packages the "possibly multiple carries"
effect. succ(999) requires carries through three digit positions. The Kleisli
composition of [increment-ones, carry-to-tens, carry-to-hundreds,
carry-to-thousands] handles variable-length chains without special-case code.

**The free monoid monad and repetition.** T(A) = List(A). eta(a) = [a].
mu flattens lists of lists. The natural numbers are T(1) = List({*}) — lists
of the single element, i.e., lengths. This is the NNO: N = T(1). Addition is
list concatenation. Succession is appending one element.

**Why it matters.** Monads provide a uniform interface for composing operations
with effects (carry, overflow, variable-length computation). The system doesn't
need separate machinery for "simple successor" vs "successor with carry" vs
"multi-digit successor with cascading carries." The monad packages the effect
and Kleisli composition handles the plumbing.

### 8. Adjunctions

**Definition.** F -| G means Hom(F(A), B) is naturally isomorphic to
Hom(A, G(B)). Equivalently: unit eta: Id -> GF and counit epsilon: FG -> Id
satisfying the triangle identities.

**In our system.** add_n -| sub_n means: "a + n = b" if and only if
"a = b - n." The adjunction encodes that subtraction is the unique solution
to the addition equation. The unit eta: a -> (a + n) - n = a says "add then
subtract recovers the original." The counit epsilon: (b - n) + n = b says
"subtract then add recovers the original."

For succession: succ -| pred. The system discovers this when it observes that
succ and pred consistently undo each other across all observations. The
adjunction then licenses using pred wherever succ needs to be "undone," even
in unseen contexts.

**Why it matters.** Adjunctions give inverse operations for free. Learning
addition automatically provides subtraction. Learning succession provides
predecessor. More broadly, adjunctions capture the "best approximation"
relationship between domains.

### 9. The Natural Number Object (NNO)

**Definition.** In a category with terminal object 1, the NNO is an object N
with z: 1 -> N (zero) and s: N -> N (successor) such that for any X with
q: 1 -> X and f: X -> X, there exists a unique h: N -> X with h(z) = q and
h . s = f . h.

```
1 --z--> N --s--> N
|        |        |
q        h        h
v        v        v
X -----> X --f--> X
```

**Why it's central.** The NNO universal property says: if the system discovers
zero and successor, it gets ALL recursive definitions for free. Addition is the
unique h for q = m, f = s (count up from m, s times). Multiplication is the
unique h for q = 0, f = add_m (add m, n times). Every arithmetic operation is
an instance of the same universal diagram.

**In our system.** The counting warmup establishes z (token "0") and s (the
successor co-occurrence edges). The NNO universal property is not yet
exploited — the system memorises individual successor pairs instead of using
the universal recursion. To generalise to succ(5000), the system needs to
APPLY the NNO universal property: the unique h for the two-digit-successor
is defined by q = "00" and f = "increment ones, carry if 9." The same h
works for any number of digits.

### 10. Sub-Categories and Slice Categories

**Sub-categories.** A full sub-category of C is a subset of objects with all
morphisms between them. In our graph: Digit = full sub-category on {0,...,9},
Operator = full sub-category on {+, -, *, succ, pred, =}.

**How to discover.** Cluster nodes by morphism profile similarity. Nodes with
isomorphic neighborhoods belong to the same sub-category.

**Slice categories.** For an object C, the slice C/C has as objects all
morphisms f: X -> C, and as morphisms the commuting triangles. The slice over
"carry" contains all numbers whose succession involves carry: {9, 19, 29, ...}.
This is a natural sub-category discovered when the system notices these numbers
share the morphism pattern "ones 9->0, tens increments."

**Why it matters.** Sub-categories enable modular learning. Learn single-digit
successor in Digit, then lift to multi-digit via the product functor. Slice
categories enable context-sensitive reasoning: "what leads to carry?" is a
categorical query with a precise answer.

---

## Part 2: Implementation Roadmap

### Priority Order

The priority is determined by what unlocks multi-digit generalisation (the
current blocking problem) and what provides the most leverage across domains.

#### Phase A: Products + Equalizers (unlock multi-digit)

**Products** decompose multi-digit numbers into (tens, ones) pairs. The system
learns successor on each component independently, then composes. This directly
solves the 0% OOD problem: succ(5000) decomposes into per-digit operations
that the system has already learned from single-digit training.

**Equalizers** discover the carry boundary. The equalizer of "increment ones
then project" vs "project ones then increment" isolates the non-carry cases.
The complement is the carry cases. No hardcoded carry logic.

**Implementation:**
1. Product discovery in consolidation: detect token pairs at fixed relative
   positions across multiple observations. Create product nodes with projections.
2. Equalizer discovery: for each pair of parallel morphisms (same source/target
   type), find the subobject where they agree. The complement is where they
   disagree (the case split).
3. Lift single-digit successor to products: compose projection, single-digit
   succ, and product assembly. The carry case uses the coproduct decomposition.

**Test:** succ(N) for N in [51, 99, 100, 255, 999, 5000]. Target: >90%.

#### Phase B: NNO Universal Property (unlock recursion)

The NNO guarantees that zero + successor gives all arithmetic via universal
recursion. Currently the system has zero and successor but doesn't exploit
the universal property.

**Implementation:**
1. Recognise the NNO structure: identify the zero morphism (token "0") and
   the successor endomorphism (the succ co-occurrence edges).
2. For any pair (q, f) — base case and step function — compute the unique
   recursive morphism h via the universal diagram.
3. Apply to addition: h(m, 0) = m, h(m, s(n)) = s(h(m, n)). The system
   computes h by walking the NNO structure, not by looking up stored pairs.

**Test:** add(a, b) for a, b in [0..9] including sums > 9. Target: >90%.

#### Phase C: Adjunctions (unlock inverse operations)

Discover that succ -| pred and add -| sub by checking that compositions are
identity on observed pairs. Create bidirectional links.

**Implementation:**
1. For each pair of functors (F, G) over the same domain, check F . G = id
   and G . F = id on all observed pairs.
2. Record the adjunction with unit and counit morphisms.
3. Use the adjunction to answer inverse queries: sub(7, 3) = ? transposes to
   add(?, 3) = 7, which the addition functor can answer.

**Test:** sub(a, b) for a >= b, inferred from addition. Target: >80%.

#### Phase D: Isomorphism Detection (unlock transfer)

Detect that succession at different tens digits is structurally identical.
Factor into abstract pattern + parametrisation.

**Implementation:**
1. Compare morphism profiles of subgraph neighborhoods.
2. Cluster isomorphic subgraphs.
3. Factor: abstract pattern (the colimit) + parameter (which instance).
4. Apply abstract pattern to new parameters.

**Test:** succ(N) for N in [5000, 12345, 99999]. Target: >95%.

#### Phase E: Monads (unlock variable-length composition)

Package the carry effect into a monad. Kleisli composition handles cascading
carries (e.g., succ(999) = 1000 requires three carries).

**Implementation:**
1. Identify the carry effect as a monad: T(digit) = digit x {carry, no_carry}.
2. Kleisli composition of per-digit successor: each step produces a digit and
   a carry flag, which feeds into the next step.
3. The monad handles variable-length chains without special-case code.

**Test:** succ(999), succ(9999), succ(99999). Target: 100%.

#### Phase F: Duality (double the constructions)

Implement limit computation by running colimit on the opposite category.
Every colimit discovery gives a limit for free.

**Implementation:**
1. Add a "reverse morphisms" operation to the graph.
2. Run existing colimit code on the reversed graph.
3. The result is the limit of the original diagram.

#### Phase G: Sub-Categories and Slice Categories

Discover natural groupings of tokens. Use slice categories for context-
sensitive queries ("what tokens lead to carry?").

**Implementation:**
1. Cluster nodes by morphism profile.
2. Create full sub-categories for each cluster.
3. Compute slice categories for key objects (carry, equals, etc.).

---

## Dependencies

```
Phase A (Products + Equalizers)
  |
  v
Phase B (NNO Universal Property)  ← depends on products for multi-digit
  |
  v
Phase C (Adjunctions)  ← depends on NNO for inverse computation
  |
  v
Phase D (Isomorphism)  ← depends on having multiple functor instances to compare
  |
  v
Phase E (Monads)  ← depends on products (digit x carry_flag) and NNO
  |
  v
Phase F (Duality) + Phase G (Sub-categories)  ← can be done in parallel
```

Phase A is the critical path. Without products, multi-digit numbers are opaque
tokens. With products, every subsequent construction can decompose numbers into
components and reason about them independently.
