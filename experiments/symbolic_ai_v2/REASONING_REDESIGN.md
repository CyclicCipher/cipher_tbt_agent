# Domain-Agnostic Rule Learning: Categorical Foundations and Redesign

**Compiled:** 2026-03-11
**Purpose:** Technical report guiding a complete redesign of the reasoning layer
(`rule_store.py` + `variable_binding.py`) to be truly domain-agnostic.
**Status of the current system:** The existing reasoning layer works only for
mathematical domains because it hardcodes searches for `'eq'`-terminated
sequences and fits integer polynomials. This report provides the theoretical
foundations needed to replace both mechanisms with domain-agnostic analogues.

---

## Preamble: What the Redesign Must Achieve

The current system's pathology is localised and precise:

1. **`_find_endofunctor_patterns`** searches for sequences ending in the atom
   `'eq'` and then tries to fit a univariate integer polynomial to the
   preceding atoms. Both the separator token and the fitting procedure are
   hardcoded for arithmetic.

2. **`predict_via_frame_match`** looks for four-atom windows matching
   `[op, arg1, arg2, result]`, which presupposes the arity and positional
   structure of arithmetic expressions.

3. **`backward_chainer`** chains adjunctions in a strictly numerical direction.

The root cause is not laziness but a conceptual gap: arithmetic happened to be
the first domain tested, and the two mechanisms above are the domain-specific
*instantiations* of something that is actually universal. This report identifies
what that universal thing is, across eight theoretical perspectives, then
synthesises concrete replacement algorithms at the end.

---

## Topic 1: Natural Numbers Object and Primitive Recursion

### 1.1 The NNO and its Universal Property

In a category **E** with a terminal object **1**, a **Natural Numbers Object**
(NNO) is an object **N** equipped with two morphisms:

    z : 1 → N          (zero)
    s : N → N          (successor)

satisfying the following **universal property**: for any object **A**, any
global element `q : 1 → A`, and any endomorphism `f : A → A`, there exists a
**unique** morphism `u : N → A` such that

    u ∘ z = q          (base case)
    u ∘ s = f ∘ u      (recursive step)

This single diagram captures all of primitive recursion. Every primitive
recursive function on natural numbers is, categorically, a unique morphism out
of **N** — i.e., a unique **fold** (catamorphism) into some algebra `(A, q, f)`.

**Lambek's Lemma** clarifies the fixed-point nature: the NNO is the initial
algebra for the endofunctor `F(X) = 1 + X` (the "maybe" functor). Lambek's
theorem states that the structure map `ι : F(µF) → µF` of any initial
F-algebra is an isomorphism, so `N ≅ 1 + N`. This makes **N** a fixed point of
`F` — the *least* such fixed point. (Reference: Lambek, J. (1968). "A fixed
point theorem for complete categories." *Mathematische Zeitschrift* 103,
151–161.)

### 1.2 Catamorphisms as Folds over Initial Algebras

A **catamorphism** (Meijer et al. 1991, "Functional Programming with Bananas,
Lenses, Envelopes and Barbed Wire") is the unique homomorphism from an initial
algebra to any other F-algebra. For `F(X) = 1 + X`:

    cata (q, f) : N → A
    cata (q, f) zero      = q
    cata (q, f) (succ n)  = f (cata (q, f) n)

This is `fold` in its purest form. The key insight: **any computation that
processes a sequence of tokens by accumulating state via a step function is a
catamorphism**. This includes:

- Summation: `(0, (+n))` maps a sequence of numbers to their sum.
- String reversal: `([], cons)` applied to a list.
- Parsing: `([], push/reduce)` applied to a token sequence.
- Arithmetic evaluation: `(empty_stack, eval_step)` over an RPN expression.

### 1.3 Recognising Folds in Observed Sequences

The critical observation for the redesign: **if a computation is a
catamorphism, its input-output pairs satisfy the fold equation universally**.
That is:

    output([]) = base
    output(x :: xs) = step(x, output(xs))    [right fold]
    output([x]) = step(x, base)
    output([x, y]) = step(x, step(y, base))

Given a collection of input-output pairs `{([a₁,...,aₙ], b)}`, one can test
the fold hypothesis by checking:

1. Does a pair `([], b₀)` exist? If yes, `base = b₀`.
2. For the pair `([a₁], b₁)`, does `step(a₁, b₀) = b₁` for some `step`?
3. For the pair `([a₁, a₂], b₂)`, does `step(a₁, step(a₂, b₀)) = b₂`?

If these constraints are consistent, the computation is (right) fold. The
"Origami" program synthesis framework (Fuketa et al., arXiv:2402.13828;
Martins et al., arXiv:2406.01500) formalises exactly this: given I/O examples,
it deduces the recursion scheme (catamorphism, anamorphism, hylomorphism,
paramorphism) and synthesises the inner step function.

**Connection to SEQUITUR:** SEQUITUR discovers context-free grammars — i.e.,
free monoidal category presentations — in O(n). A grammar rule `A → BC` is
exactly a fold over the list `[B, C]` with base `ε` and step `concat`. Every
grammar rule is a catamorphism over the rule's right-hand side. Therefore,
**SEQUITUR implicitly induces a family of catamorphisms**: one per grammar
rule. The MorphismGraph composition hierarchy IS the family of folds discovered
by Graph-SEQUITUR.

### 1.4 List Objects as NNOs over Arbitrary Types

Wraith (1985) showed that an NNO in a topos implies the existence of
**list objects**: for any type `A`, there is a list object `List(A)` with
constructors `nil : 1 → List(A)` and `cons : A × List(A) → List(A)`, satisfying
an analogous universal property.

This is the key generalisation: **the NNO is `List(1)`**. For arbitrary domain
symbol types, the relevant initial algebra is `List(A)`. Fold over
`List(A)` has type:

    fold : B → (A → B → B) → List(A) → B

This is domain-agnostic. The domain appears only in the type of `A` (the token
type) and `B` (the accumulator type). **Replacing the hardcoded integer
polynomial fitter with a fold detector over `List(A)` for arbitrary `A`
is the correct generalisation.**

---

## Topic 2: Adjunctions as the Categorical Foundation of Inverse Operations

### 2.1 Adjunctions in Posets: Galois Connections

The simplest adjunctions are **Galois connections** between posets. Given
partially ordered sets `(P, ≤)` and `(Q, ≤)`, a pair of order-preserving maps
`f : P → Q` and `g : Q → P` form a Galois connection if:

    f(p) ≤ q  ⟺  p ≤ g(q)   for all p ∈ P, q ∈ Q

This is the unit/counit characterisation restricted to posets, where:

    η : id_P → g ∘ f    (unit: p ≤ g(f(p)))
    ε : f ∘ g → id_Q    (counit: f(g(q)) ≤ q)

**Arithmetic examples as Galois connections on (ℤ, ≤):**

- Addition `+n` and subtraction `-n` are inverse functors on the discrete
  category ℤ (every element has only identity morphisms). Because ℤ is discrete,
  the adjunction degenerates to a bijection: `+n` and `-n` are strict inverses.

- For the divisibility order `(ℕ, |)`: the Galois connection `(×n, ÷n)` is
  given by `n×p ≤_| q ⟺ p ≤_| q÷n`. Here `÷n` is right adjoint to `×n`.

- For the exponential: `n^p ≤ q ⟺ p ≤ log_n(q)`. The logarithm is right
  adjoint to exponentiation.

### 2.2 Adjunctions in Categories: the General Case

An **adjunction** `F ⊣ G` between categories **C** and **D** consists of:

- Functors `F : C → D` (left adjoint) and `G : D → C` (right adjoint)
- A natural isomorphism `Hom_D(F(c), d) ≅ Hom_C(c, G(d))`

Equivalently, there are natural transformations `η : id_C → G∘F` (unit) and
`ε : F∘G → id_D` (counit) satisfying the triangle identities.

**The key insight for rule discovery**: an adjunction pair `(F, G)` is exactly
a pair of operations that are "inverse up to a canonical transformation." The
unit `η` says "applying F then G is at least as good as doing nothing" (from
C's perspective), and the counit `ε` says "applying G then F is at most as
good as doing nothing" (from D's perspective). Exact inverses (equivalences)
are the degenerate case where both `η` and `ε` are isomorphisms.

### 2.3 Detecting Adjunctions from Co-occurrence Data

**Residuation theory** (Krull 1924, Ward & Dilworth 1939) provides the key:
a pair `(f, g)` of order-preserving maps between posets is an adjoint pair
(Galois connection) if and only if `g` is the **residual** of `f`, i.e.,

    g(q) = max{p : f(p) ≤ q}

This max need not exist in general, but when it does, it is the right adjoint.

**Empirical detection algorithm** (sketch):

1. Build a co-occurrence table `T[a, b] = count(a followed by b)` from the
   corpus.
2. For each pair of relation types `(r₁, r₂)`, compute the composition
   `r₁ ; r₂`: for each source `x`, collect all `z` reachable by first
   following `r₁` then `r₂`.
3. An adjunction candidate `r₁ ⊣ r₂` is detected when:
   - `r₁ ; r₂ ⊇ id` (unit condition: `z ≥ x` in the composition order)
   - `r₂ ; r₁ ⊆ id` (counit condition: `z ≤ x`)
   - The asymmetry is consistent across the corpus.
4. Formalise via FCA (see Topic 4): run FCA on the object-attribute matrix
   where `[x, r₁, y]` and `[y, r₂, z]` edges are the binary relation. Closed
   biclusters that contain both `r₁`-triples and `r₂`-triples are adjunction
   candidates.

**Linguistic examples:**

- `plural ⊣ singular` (or its inverse): "dog → dogs → dog" is the round-trip.
- `encode ⊣ decode` in any coding domain.
- `stem → inflected_form` ⊣ `inflected_form → stem` in morphology.
- `question → answer` ⊣ `answer → question` in Q&A.

These are all detectable from corpus co-occurrences without domain knowledge.

### 2.4 Kan Extensions as Universal Adjunctions

**Kan extensions** are the most general form of adjunction. Given functors
`K : M → C` and `T : M → A`, the left Kan extension `Lan_K T : C → A` is the
"best approximation" to extending `T` along `K`. Mac Lane's slogan "all
concepts are Kan extensions" (from *Categories for the Working Mathematician*)
is precise: limits, colimits, adjoints, and even free constructions are all
special cases.

For the redesign: when the system discovers a partial map between domains
(a functor defined on some objects but not others), the left Kan extension
gives the **canonical completion** of that partial map to the entire source
domain. This is the categorical foundation for the "transfer" operation in
`tests/transfer_test.py`.

**Key property**: `Lan_K ⊣ (- ∘ K)` (precomposition). Discovering that the
left Kan extension exists and is "well-behaved" (pointwise) provides evidence
that the two domains share an algebraic structure — the functor `K` is a
morphism of Lawvere theories (see Topic 5).

---

## Topic 3: Cartesian Differential Categories

### 3.1 Axioms of CDC (Blute, Cockett, Seely 2009)

A **Cartesian differential category** (Blute, R.F., Cockett, J.R.B., and
Seely, R.A.G., "Cartesian Differential Categories," *Theory and Applications
of Categories* 22 (2009), no. 23, 622–672; available at
http://www.tac.mta.ca/tac/volumes/22/23/22-23abs.html) is a Cartesian left
additive category equipped with a **differential combinator**:

    D : Hom(A, B) → Hom(A × A, B)

satisfying seven axioms **[CD.1–CD.7]**:

    [CD.1]  D[f + g] = D[f] + D[g]           (linearity in f)
    [CD.2]  D[0] = 0                           (zero)
    [CD.3]  D[f ∘ ⟨g, h⟩] = D[f] ∘ ⟨g ∘ π₁, D[g] ∘ ⟨π₁, π₂⟩ + h⟩  (actually the chain rule form)
    [CD.4]  D[⟨f, g⟩] = ⟨D[f], D[g]⟩         (product)
    [CD.5]  D[f ∘ g] = (D[f] ∘ ⟨g ∘ π₁, D[g]⟩)  (chain rule)
    [CD.6]  D[D[f]] ∘ ⟨⟨a,0⟩, ⟨0,b⟩⟩ = D[f] ∘ ⟨a, b⟩  (linearity of D in first arg)
    [CD.7]  D[D[f]] ∘ ⟨⟨a,b⟩, ⟨c,d⟩⟩ = D[D[f]] ∘ ⟨⟨a,c⟩, ⟨b,d⟩⟩  (symmetry of mixed partials)

The canonical example: `Euc` (Euclidean spaces with smooth maps), where
`D[f](a, v) = Jac_f(a) · v` (directional derivative).

### 3.2 The Chain Rule as Functoriality

The most important axiom is **[CD.5]**: `D[f ∘ g] = D[f] ∘ ⟨g ∘ π₁, D[g]⟩`.

This is the chain rule `(f∘g)'(x) = f'(g(x)) · g'(x)` expressed without
coordinates. The functoriality statement is: the assignment `f ↦ (f, D[f])`
is a functor `α : C → T(C)` where `T(C)` is the "tangent bundle category."
The product `A × A` in `Hom(A × A, B)` is `TA` (tangent bundle): the first
copy carries the base point, the second copy carries the tangent vector.

**Tangent categories** (Cockett & Cruttwell 2014, "Differential Structure,
Tangent Structure, and SDG," arXiv:1406.0479) generalise CDCs further. A
tangent category has a tangent bundle functor `T : C → C` with natural
transformations:

    p : T → id         (projection to base)
    z : id → T         (zero section)
    l : T → T²         (vertical lift)
    + : T ×_base T → T (fibrewise addition)

satisfying coherence conditions. The `sym_diff` interpreter primitive is the
discrete-symbolic instantiation of `D[−]` in this framework.

### 3.3 Joyal Species and Combinatorial Differentiation

André Joyal (1981, "Une théorie combinatoire des séries formelles,"
*Advances in Mathematics* 42, 1–82) introduced **combinatorial species** as
functors `F : B → Set` where `B` is the category of finite sets and bijections.
The **derivative** of a species is:

    F'[n] = F[n+1]

Concretely: an element of `F'[n]` is an `F`-structure on an `(n+1)`-element
set where one distinguished element is the "hole." Removing the hole gives the
derivative structure.

**Classical example:** `C` = cyclic permutations. `C'[n] = C[n+1]` means:
take a cycle on `n+1` elements, remove one element; the result is a linear
order on the remaining `n` elements. Therefore `C' = L` (linear orders).
This recovers the differential equation `C' = L`.

**Relevance for the redesign**: Joyal differentiation gives a domain-agnostic
notion of "derivative of a rule." Given a grammar rule (species) `F`, its
derivative `F'` is the rule with one "slot" opened up. This is exactly what
happens when you generalise a specific instance to a rule with a free variable:
replacing a concrete token with a variable IS differentiation in the species
sense.

The `sym_diff` primitive in the interpreter is the symbolic-expression
specialisation of this. The redesign should recognise that **any slot in any
rule is a species derivative**, and generalise accordingly.

### 3.4 Connection to the Current sym_diff

In the interpreter, `sym_diff(expr, var)` computes the symbolic derivative of
`expr` with respect to `var`. Categorically:

- `expr` is a morphism in the free CDC on the arithmetic Lawvere theory.
- `sym_diff(expr, var)` applies the differential combinator `D[expr]`.
- The chain rule is enforced by [CD.5].

The redesign implication: `D[−]` should be defined for **any** Lawvere theory
(see Topic 5), not just arithmetic. In a language domain, `D[word]` with
respect to a slot is the set of words that can fill that slot — a contextual
distribution. This connects CDCs to distributional semantics.

---

## Topic 4: Formal Concept Analysis and Rule Discovery

### 4.1 The Galois Connection in FCA

A **formal context** is a triple `K = (G, M, I)`: objects `G`, attributes `M`,
incidence relation `I ⊆ G × M`. Define derivation operators:

    A' = {m ∈ M : ∀g ∈ A. (g,m) ∈ I}   (attributes shared by A)
    B' = {g ∈ G : ∀m ∈ B. (g,m) ∈ I}   (objects sharing all B)

The pair `(¹, ²)` forms an **antitone Galois connection** (order-reversing
adjunction) between `(2^G, ⊆)` and `(2^M, ⊆)`. A **formal concept** is a
closed pair `(A, B)` where `A = B'` and `B = A'`. The concept lattice
`B(K)` is the poset of all formal concepts ordered by `(A₁,B₁) ≤ (A₂,B₂)
⟺ A₁ ⊆ A₂` (equivalently `B₁ ⊇ B₂`).

**Theorem (Wille 1982):** The concept lattice is a complete lattice, and every
complete lattice is isomorphic to a concept lattice. The concept lattice
captures **all closed implications** (all rules) entailed by the binary
relation `I`.

### 4.2 Implications and the Duquenne-Guigues Basis

An **attribute implication** is a statement `A → B` (for `A, B ⊆ M`): every
object possessing all attributes in `A` also possesses all attributes in `B`.
The **Duquenne-Guigues basis** (canonical basis, stem base) is a minimal,
non-redundant set of implications that is a complete axiom system for all
implications holding in the context. It has size at most `|B(K)|`.

**Relevance for domain-agnostic rule discovery**: Given any domain's data as a
binary relation (objects × attributes), running FCA produces the complete set
of rules (implications) with no free parameters. The rules are derived
mathematically, not by polynomial fitting.

For a **mathematics corpus**:
- Objects `G` = observed computations `(input, output)`.
- Attributes `M` = intermediate steps, algebraic properties.
- `I`: computation `g` has property `m`.

For a **language corpus**:
- Objects `G` = observed word-context pairs.
- Attributes `M` = distributional features (left context, right context, POS).
- Rules derived: "words with attribute set A always have attribute set B" =
  morphological/syntactic implications.

For a **logic corpus**:
- Objects `G` = axiom sets used in proofs.
- Attributes `M` = theorems derived.
- Rules: implicational structure of the logical theory.

### 4.3 Formal Concepts as Closed Adjunction Pairs

The fundamental categorical insight: **each formal concept `(A, B)` is a fixed
point of the adjunction**, i.e., a stable "resonance" where the object set and
attribute set mutually determine each other. This is the categorical
characterisation of a **natural kind**: a cluster that is stable under the
Galois connection.

In the MorphismGraph context:
- Objects `G` = symbol IDs appearing as sources of some edge type `r`.
- Attributes `M` = symbol IDs appearing as targets.
- `I = {(g, m) : ∃ edge (g, r, m)}`.
- Formal concepts = maximally closed bicliques in the edge relation.

Each formal concept is an adjunction in the enriched-category sense: a pair
of "maximum-covering" roles that are dual to each other.

### 4.4 FCA and Chu Spaces

Vaughan Pratt's **Chu spaces** (`Chu(Set, K)`) unify FCA and linear logic.
A Chu space `(A, r, X)` over `K = {0,1}` is exactly a formal context: `A` are
objects, `X` are attributes, `r : A × X → {0,1}` is the incidence matrix.
The Chu construction is a `*`-autonomous category (Barr 1979), meaning it has
a dualizing object `⊥ = K`. Self-duality of Chu spaces corresponds to the
symmetry of the Galois connection: objects and attributes are structurally
interchangeable.

**Practical implication**: The same FCA algorithm applies to any binary relation.
For the MorphismGraph's edge-type adjacency data, FCA discovers both
"which source symbols co-occur with which edge types" and "which edge types
co-occur with which target symbols," simultaneously recovering the type system
and the adjunctions in a single pass.

---

## Topic 5: Lawvere Theories and Algebraic Theories

### 5.1 Definition

A **Lawvere theory** (Lawvere 1963, *Functorial Semantics of Algebraic
Theories*) is a small category `L` with a distinguished object `*` such that
every object of `L` is a finite power of `*` (i.e., `Lₙ = *ⁿ`), and `L`
has all finite products (the product structure makes `*ⁿ = * × ... × *`).

A **model** of `L` in a category **C** is a finite-product-preserving functor
`M : L → C`. In `Set`:
- `M(*)` is the underlying set (the "carrier").
- `M(f : *ⁿ → *)` is an n-ary operation on `M(*)`.
- The equations encoded in `L`'s morphism identities become equational axioms
  on `M(*)`.

**Examples:**
- `L_Grp`: groups. Models are exactly groups.
- `L_Rng`: rings. Models are rings.
- `L_Vect_k`: vector spaces over `k`. Models are `k`-modules.
- `L_Bool`: Boolean algebras.
- `L_Mon`: monoids (relevant: a context-free language is a monoid).

Every algebraic theory that can be presented by operations and equations has
a Lawvere theory. (Fields do NOT have a Lawvere theory because multiplicative
inverse is not defined on 0.)

### 5.2 Syntactic Categories: Extracting a Lawvere Theory from Data

Given a set of observed input-output pairs `{(inputs, output)}`, the
**syntactic category** `Syn(data)` is the free Lawvere theory generated by:
- Generators: one generating morphism per observed operation.
- Relations: the equational axioms provably satisfied by the observations.

The syntactic category is related to the initial model in the same way that
a free group is related to the integers: it captures exactly the structural
constraints entailed by the data, with no additional assumptions.

**Algorithm sketch for extracting a Lawvere theory from observations:**

1. **Signature discovery**: For each distinct operation schema `(arity, type)`,
   introduce a generator `σ_i : *ⁿ → *`.
2. **Equation mining**: For pairs of computation paths that produce the same
   output, introduce an equation between the corresponding composed morphisms.
3. **Quotient**: Form the free category on the generators, quotient by the
   equations. The result is the syntactic Lawvere theory.

This is precisely what **Graph-SEQUITUR does**, one level at a time: it
discovers generators (composition rules) and enforces the rule utility
invariant (discarding rules that appear only once = equations that are not
provable). The composition hierarchy in `MorphismGraph` is the syntactic
category of the observed data.

### 5.3 Models Generalise Across Domains

The power of Lawvere theories is that a single abstract theory `L` can have
models in many different categories:
- `L_Mon` has models in `Set` (monoids), `Vect` (vector spaces = "linear
  monoids"), `Cat` (monoidal categories), `Rel` (relational monoids).

**For the redesign**: A "rule" discovered in domain A can be lifted to a rule
in domain B if there is a **morphism of Lawvere theories** `φ : L_A → L_B`.
This is the categorical foundation of transfer learning. The current
`transfer_test.py` detects this empirically; the redesign should make it
explicit.

**Concrete example**: The Lawvere theory `L_Mon` has a morphism into both
`L_Arith` (arithmetic under addition) and `L_RegExp` (regular expressions
under concatenation). A fold rule discovered in arithmetic (e.g., "sum of
a list") transfers automatically to string concatenation by applying the
Lawvere theory morphism.

### 5.4 Connection to Term Rewriting

A **term rewriting system** (TRS) is a Lawvere theory with an orientation on
some equations: `l → r` means "rewrite left-hand side to right-hand side."
The **Knuth-Bendix completion algorithm** (Knuth & Bendix 1970) takes a set
of equations `E` and an ordering `>` and attempts to produce a confluent,
terminating TRS `R` with the same equational theory.

- **Confluence**: every term has a unique normal form.
- **Termination**: rewriting always halts.
- A confluent terminating TRS solves the **word problem** for the algebra.

For domain-agnostic rule discovery: Knuth-Bendix completion is the algorithm
that converts raw "things that are equal" (from FCA implications) into a
canonical rewrite system (from which reduction to normal form is deterministic).
**The `eq` token in the current system is the rewrite arrow `→` of the TRS.
Replacing `eq` with a generalised separator that marks any rewrite relation
is the correct fix.**

Gröbner bases (polynomial rings) and Knuth-Bendix completion (term algebras)
are both instances of "normalised completion" (Marché 1998), showing the
algorithm is domain-agnostic.

---

## Topic 6: String Diagrams and Graphical Calculus

### 6.1 String Diagrams for Monoidal Categories

In a **symmetric monoidal category**, morphisms `f : A ⊗ B → C ⊗ D` can be
drawn as boxes with wires:

    A ---[f]--- C
    B ---    --- D

Composition is vertical concatenation; tensor product is horizontal
juxtaposition. String diagrams are a **sound and complete** notation for
(symmetric) monoidal categories (Joyal & Street 1991, "The Geometry of Tensor
Calculus I").

### 6.2 Frobenius Algebras

A **Frobenius algebra** in a monoidal category is an object `A` with both
monoid structure `(μ : A⊗A → A, η : I → A)` and comonoid structure
`(δ : A → A⊗A, ε : A → I)` satisfying the Frobenius law:

    (id ⊗ μ) ∘ (δ ⊗ id) = δ ∘ μ = (μ ⊗ id) ∘ (id ⊗ δ)

In string diagram notation, this is the "spider" — wires can split and
merge freely, and the result depends only on the topology of the diagram,
not the order of operations. This gives a correspondence:

    String diagrams modulo Frobenius ↔ Hypergraphs

(Bonchi, Gadducci, Kissinger, Sobocinski, Zanasi 2022, "String Diagram Rewrite
Theory I: Rewriting with Frobenius Structure," *Journal of the ACM* 69(2);
arXiv:2012.01847.)

**Implication**: The MorphismGraph is a hypergraph. String diagram rewriting
modulo Frobenius gives a canonical framework for pattern matching and rewriting
on hypergraphs. Two diagrams are equal modulo Frobenius iff their underlying
hypergraphs are isomorphic. This is the theoretical foundation for
**topology-independent rule matching**.

### 6.3 String Diagram Rewriting as Grammar

String diagram rewriting systems are **double-pushout (DPO) rewriting systems**
on labelled hypergraphs. An rewriting rule is a pair `(L → I → R)` where `L` is
the left-hand side pattern, `R` the right-hand side, and `I` their interface.
Confluence and termination can be decided for finite DPO systems (Bonchi et al.
2022, Part III).

**Connection to SEQUITUR**: SEQUITUR on sequences is DPO rewriting on path
graphs. Graph-SEQUITUR is DPO rewriting on arbitrary topology graphs. Both are
instances of string diagram rewriting. The two SEQUITUR invariants (digram
uniqueness + rule utility) correspond to MDL-optimal DPO rule sets.

### 6.4 Compact Closed Categories and Pregroup Grammars

In a **compact closed category**, every object `A` has a dual `A*` with cups
`η : I → A* ⊗ A` and caps `ε : A ⊗ A* → I` satisfying the snake equations.
String diagrams for compact closed categories allow wires to "bend back."

**Pregroup grammars** (Lambek 1999) assign types from a pregroup (a compact
closed poset) to words. A sentence is grammatical iff its type reduces to the
sentence type `s`. Reductions are string diagram compositions.

**For the redesign**: Any domain with a notion of "type correctness" (not just
linguistic grammars, but also type-checked arithmetic, typed API calls, etc.)
can be modelled as a compact closed category. Rules are compact closed
morphisms; rule matching is string diagram pattern matching.

---

## Topic 7: Related AI/ML Work

### 7.1 Inductive Logic Programming (ILP)

ILP (Muggleton 1991, *Machine Intelligence* 13; survey: Cropper & Dumancic,
JAIR 2022 arXiv:2008.07912) induces logic programs (sets of Datalog/Prolog
rules) from positive and negative examples, using background knowledge.

**Categorical reformulation**: An ILP hypothesis is a functor `H : BK → Ex`
where `BK` is the background knowledge category and `Ex` is the example
category. Learning is the search for such a functor. Muggleton and de la
Raedt's FOIL/PROGOL algorithms are syntactic searches for this functor in a
language-biased space.

**Connection to MorphismGraph**: The MorphismGraph IS the background knowledge
category. The `rule_store` IS the ILP hypothesis space. The redesign should
recognise that rule discovery in any domain is an ILP problem, and that the
categorical version of ILP (functor finding) is exactly what needs to be made
domain-agnostic.

**ILASP** (Law, Russo, Broda 2015, arXiv:1906.10055) and **Metagol**
(Cropper & Muggleton 2016) extend ILP with meta-rules (second-order templates).
Meta-rules are exactly Lawvere theory generators: `P(x,y) :- Q(x,z), R(z,y)`
is the composition meta-rule. The current `compose` composition in
`MorphismGraph` is the Metagol "chain" meta-rule.

### 7.2 Curry-Howard-Lambek and Program Synthesis

The **Curry-Howard-Lambek correspondence** (Lambek 1970s):

    Logic           ↔  Type Theory      ↔  Category Theory
    Proposition     ↔  Type             ↔  Object
    Proof           ↔  Program/Term     ↔  Morphism
    Proof reduction ↔  Evaluation       ↔  Composition
    Implication A⊃B ↔  A → B           ↔  Exponential object B^A
    Conjunction A∧B ↔  Product A×B     ↔  Categorical product
    Disjunction A∨B ↔  Coproduct A+B   ↔  Categorical coproduct

**Implication for rule learning**: Every rule discovered is a proof (of the
proposition "from inputs, the output follows"). Rule composition is proof
concatenation. The **proof-theoretic completeness** of a rule set = the
completeness of the corresponding Lawvere theory (all theorems provable from
the axioms).

**Program synthesis via Curry-Howard**: Given a type signature (= a morphism
type in the Lawvere theory), synthesis = finding a proof of the corresponding
proposition. The Origami framework (Topic 1.3) instantiates this: given the
type `A → B`, it searches for the proof that has a catamorphism structure.

### 7.3 Analogical Reasoning as Functor Detection

**Gentner's Structure Mapping Theory** (Gentner 1983, "Structure-Mapping: A
Theoretical Framework for Analogy," *Cognitive Science* 7, 155–170) defines
analogy as the detection of a structure-preserving mapping between a source
domain and a target domain. The key principle: good analogies map relations
to relations (higher-order), not just attributes to attributes (first-order).

**Categorical reformulation**: An analogy is a functor `φ : S → T` where `S`
is the source category (graph of source domain relations) and `T` is the
target category. "Relations map to relations" is exactly functoriality:
`φ(f ∘ g) = φ(f) ∘ φ(g)`.

The DIANA system (Stojanov & Eliot 1994) implements a version of this in
Prolog. The categorical version: enumerate functor candidates between the
MorphismGraph of the source domain and the MorphismGraph of the target domain.
A functor candidate is a mapping of objects + edge types that preserves all
observed compositions. This is exactly what `transfer_test.py` tests.

**Gentner's systematicity principle** (prefer functors that map larger
connected subgraphs) corresponds categorically to preferring functors that
preserve more morphisms (fuller functors over sparse ones).

### 7.4 Neural Networks and Implicit Categorical Structure

**Equivariant networks** (Cohen & Welling 2016; Geometric Deep Learning,
Bronstein et al. 2021, arXiv:2104.13478): A neural network architecture is
equivariant iff it implements a functor from an input-symmetry category to an
output-symmetry category. CNNs are equivariant to the translation group;
GNNs to graph isomorphisms; Transformers to permutation symmetries.

**Gavranovic et al. (2024)** (arXiv:2402.15332): All architectures are monad
algebras in a 2-category of parametric maps. Discovering the symmetry
structure of data = discovering the monad.

**Implication**: Neural networks that generalise ARE finding the relevant monad
(= Lawvere theory). Symbolic rule discovery is doing this explicitly. The
redesign should output not just rules but the **monad/Lawvere theory** that
explains the rules — this is the form in which knowledge is transferable.

### 7.5 Omega-Categories and Rewriting

**ω-categories** (Street 1987; Steiner 2004, "Omega-categories and chain
complexes," arXiv:math/0403237) model rewriting systems at all levels
simultaneously:
- 0-cells: objects (terms)
- 1-cells: rewrite steps (term → term)
- 2-cells: "rewrites between rewrites" (proving two reduction sequences equal)
- n-cells: n-th order equivalences

The MorphismGraph is a 1-categorical structure (edges are relations between
tokens). The composition hierarchy adds 2-cells (compositions are 2-cells
in the grammar category). Extending to ω-categories would allow the system
to represent and reason about rewriting strategies, not just rewrites.

**Parity complexes** (Street 1991/1994) are combinatorial structures (analogous
to simplicial sets but for directed globular structures) that generate free
ω-categories. The Grammar-SEQUITUR output is a computad (= parity complex for
free ω-categories). Recognising this structure allows the system to use ω-
categorical coherence theorems to simplify the rule set.

---

## Topic 8: Practical Algorithms for Domain-Agnostic Rule Discovery

### 8.1 Fold Detection from Input-Output Pairs (Catamorphism Synthesis)

**The problem**: Given observations `{([a₁,...,aₙ], result)}` for varying
lengths `n`, determine if the computation is a fold, and if so, extract the
step function.

**Algorithm** (following Martins et al. arXiv:2402.13828 and the PLDI 2024
paramorphisms paper):

```
fold_detect(examples):
    # Step 1: Check base case
    base_examples = [([], b) for ([], b) in examples]
    if len(base_examples) == 0: return UNKNOWN
    base = base_examples[0].output

    # Step 2: For length-1 examples, infer step applied once
    step_examples = []
    for ([a], b) in examples:
        step_examples.append((a, base, b))
        # constraint: step(a, base) = b

    # Step 3: For length-2, check consistency
    for ([a1, a2], b) in examples:
        intermediate = lookup_in_length1(a2)  # step(a2, base)
        if intermediate is not None:
            step_examples.append((a1, intermediate, b))
            # constraint: step(a1, intermediate) = b

    # Step 4: Anti-unify the step constraints
    step_schema = anti_unify_all(step_examples)
    if step_schema is not None: return FOLD(base, step_schema)
    else: return NOT_A_FOLD
```

**Anti-unification** (Plotkin 1970; Cerna & Kutsia 2023, arXiv for survey)
computes the **least general generalisation** (lgg) of a set of term triples.
The lgg is the most specific pattern that subsumes all observed step
applications. This replaces the integer polynomial fitter: where the current
code fits `output = a * input + b`, the anti-unification approach finds the
most specific functor expression.

**Categorical interpretation**: anti-unification is the **equalizer** in the
category of substitutions (dual to Robinson unification = coequalizer). The
lgg is the "meet" of two terms in the anti-unification lattice.

### 8.2 Adjunction Detection from Raw Co-occurrence Data

**The problem**: Given the MorphismGraph edge table, detect pairs of relation
types `(r₁, r₂)` that form adjunction pairs without domain knowledge.

**Algorithm** (based on residuation theory, Topic 2.3):

```
detect_adjunctions(mg, edge_types):
    candidates = []
    for r1, r2 in product(edge_types, edge_types):
        if r1 == r2: continue

        # Test r1 ; r2 ⊇ id (unit condition)
        unit_holds = True
        for x in mg.all_symbols():
            r1_images = mg.follow(x, r1)      # {y : x -r1-> y}
            roundtrip = set()
            for y in r1_images:
                roundtrip |= mg.follow(y, r2)  # {z : y -r2-> z}
            if x not in roundtrip:
                unit_holds = False
                break

        # Test r2 ; r1 ⊆ id (counit condition)
        counit_holds = True
        for y in mg.all_symbols():
            r2_images = mg.follow(y, r2)
            for z in r2_images:
                r1_back = mg.follow(z, r1)
                if y not in r1_back:
                    counit_holds = False
                    break

        if unit_holds and counit_holds:
            candidates.append((r1, r2, 'adjunction'))
        elif unit_holds:
            candidates.append((r1, r2, 'unit_only'))
        elif counit_holds:
            candidates.append((r1, r2, 'counit_only'))

    return candidates
```

In practice, these conditions will only hold approximately. The detection
should use a **coverage score**: what fraction of observed `x` satisfy the
unit condition? A threshold-free approach: run FCA on the composition matrix
(objects = triples `(x,y,z)`, attributes = `{unit_r1r2, counit_r2r1}`) and
look for formal concepts that include both attributes.

### 8.3 Composition Pattern Mining

**The problem**: Given the MorphismGraph, find all patterns of the form
`f ∘ g = h` (composition rules) that hold universally (not just in a specific
instance).

This is the Graph-SEQUITUR task, already implemented. The categorical
interpretation: each discovered rule `A → BC` is a **generator** in the
syntactic Lawvere theory. Collecting all generators gives the theory.

For domain-agnostic detection: the key invariant is MDL (minimum description
length). A rule is worth creating iff the description length decreases. This
is exactly SEQUITUR's rule utility invariant, which is parameter-free. No
domain knowledge required.

### 8.4 Unification and Anti-Unification

**Unification** (Robinson 1965): Given terms `s` and `t`, find the most
general unifier (mgu) `θ` such that `θ(s) = θ(t)`. Categorically: the mgu
is the **coequalizer** in the Kleisli category of the free monad on the
term algebra (Goguen, "What is Unification? A Categorical View of Substitution,
Equation and Solution").

**Anti-unification** (Plotkin 1970, Reynolds 1970): Given terms `s` and `t`,
find the least general generalisation (lgg) `g` such that both `s` and `t` are
instances of `g`. Categorically: the lgg is the **equalizer** (dual
construction). The lgg is unique up to renaming of variables.

**Domain-agnostic rule extraction** uses anti-unification as follows:

1. Collect all observed instances of a relation type `r`: pairs `(s_i, t_i)`.
2. Anti-unify all source terms: `g_src = lgg(s_1, s_2, ...)`.
3. Anti-unify all target terms: `g_tgt = lgg(t_1, t_2, ...)`.
4. The rule is: `g_src -r-> g_tgt`.
5. Variables in `g_src` that also appear in `g_tgt` are **bound variables**
   (the rule transfers specific information).
6. Variables in `g_tgt` not in `g_src` are **fresh variables** (the rule
   introduces new information, e.g., from background knowledge).

This procedure works for ANY domain. For arithmetic: the lgg of `{(3,7), (2,6),
(5,9)}` under `+4` is `(x, x+4)` — the rule `add_four`. For morphology: the
lgg of `{(dog,dogs), (cat,cats), (hat,hats)}` under `plural` is `(x, x+s)` —
the default plural rule.

### 8.5 Knuth-Bendix as Canonical Rule Normalisation

Given a set of equations discovered by FCA/anti-unification and an ordering
on terms (e.g., by size or by the MorphismGraph's compression score), running
Knuth-Bendix completion produces a **confluent terminating TRS**: a canonical
rule set where every term has a unique normal form.

The normal form of a term is its canonical representation under the discovered
rules. For arithmetic: `3 + 4` normalises to `7`. For morphology: `run + -ing`
normalises to `running` (after applying morphophonological rules). For logic:
a proposition normalises to its canonical form (DNF/CNF) under propositional
identities.

**Key insight for the redesign**: The `eq` token in the current system marks
a "rewrite arrow" in a single-rule TRS over a specific arithmetic domain.
Replacing `eq` with a general rewrite separator that marks the boundary
between any left-hand side and any right-hand side, and running Knuth-Bendix
on the resulting equation set, makes the system domain-agnostic.

---

## Synthesis: Answering the Five Key Questions

### Q1: Minimal Categorical Structure for Domain-Agnostic Rule Discovery

The minimal structure needed is:

1. **A small category** (the "schema"): objects = symbol types, morphisms =
   observed relations between symbols. This is the MorphismGraph's topology.

2. **A monad on Set** (or equivalently, a Lawvere theory): the algebraic
   structure of the discovered operations. SEQUITUR's grammar is the presentation
   of this monad.

3. **An antitone Galois connection** (FCA): to detect closed rule patterns
   (implications) in the object-attribute incidence data.

4. **An initial algebra** (for fold detection): the NNO structure (or its
   generalisation to `List(A)`) that characterises which computations are folds.

Nothing else is required. Adjunctions, Kan extensions, string diagrams, and the
rest provide additional machinery for specific tasks (transfer, composition
optimisation, grammar simplification), but are not necessary for basic rule
discovery.

**The minimal algorithm**:
- SEQUITUR (O(n)) for composition discovery.
- FCA (O(K³)) for adjunction/implication discovery.
- Anti-unification (O(n²) per rule) for rule generalisation.
- Knuth-Bendix (semidecidable) for normalisation.

### Q2: Categorical Characterisation of "Rule" vs "Fact"

A **fact** is a closed morphism: `f : 1 → B` (a global element, a specific
value). It has no free variables; it is fully instantiated.

A **rule** is a morphism with free variables: `f : A → B` where `A ≠ 1`. The
free variables are the "inputs" to the rule. More precisely:

- A rule is a **generator of the syntactic Lawvere theory** of the domain.
- A fact is a **ground instance** of a rule: the result of substituting specific
  values for all free variables.
- A **meta-rule** (a rule about rules) is a **natural transformation** between
  functors representing different rules.

In the species framework (Topic 3.3): a fact is a species evaluated on a
specific set `F[n]`; a rule is the species `F` itself (a functor). The
derivative `F'` is a "partially applied" rule (one variable made explicit).

**Operational distinction**: A sequence in the corpus is a fact if its
anti-unification lgg with any other sequence produces only trivial
generalisations (all variables). It is a rule if its lgg with some other
sequence produces a non-trivial pattern (some structure preserved).

**Threshold-free criterion**: A pattern discovered by SEQUITUR is a rule if
it is used at least twice (rule utility invariant). A pattern used exactly once
is a fact until confirmed as a rule by a second occurrence.

### Q3: Fold Detection as Replacement for Integer Polynomial Fitting

**Current system**: Fits `output = a₀ + a₁·input + a₂·input² + ...` to a
sequence of observations where the input is an integer. Fails for any
non-polynomial domain.

**Replacement**: The fold detection algorithm (Topic 8.1):

1. Group observations by their input list length.
2. Test consistency of the fold equation across lengths.
3. Anti-unify the step function constraints to get the rule schema.
4. If the fold hypothesis is consistent, emit a FOLD rule; otherwise, try
   UNFOLD (anamorphism), HYLOMORPHISM (unfold then fold), or PARAMORPHISM
   (fold with access to original structure).

The recursion scheme taxonomy (catamorphism / anamorphism / hylomorphism /
paramorphism / zygomorphism / histomorphism) is fully domain-agnostic: it
depends only on the structure of the input TYPE (a list, a tree, a graph),
not on the VALUE DOMAIN of the elements.

**Concrete change to `rule_store.py`**:
- Remove the polynomial fitter entirely.
- Replace with `detect_recursion_scheme(observations)` that tests the four
  main schemes in order of generality.
- The step function in each scheme is extracted by anti-unification (not
  polynomial fitting).
- The base case is the empty-list observation.

### Q4: Adjunction Detection on Raw Co-occurrence Data

**Current system**: Adjunctions are detected only for `eq`-terminated
sequences in arithmetic. The detection hardcodes `(+, -)` and `(×, ÷)`.

**Replacement**:

1. For each pair of edge types `(r₁, r₂)` in the MorphismGraph, test the
   roundtrip condition (Algorithm in Topic 8.2).
2. Use a coverage score (fraction of observed symbols satisfying the unit and
   counit conditions) rather than exact equality, to handle noisy data.
3. Store adjunction candidates sorted by coverage score.
4. Use FCA on the edge-pair matrix to find formal concepts that correspond
   to closed adjunction pairs (Topic 4).
5. Emit an `ADJUNCTION(r₁, r₂, coverage)` record to the CTKG.

This works for ALL domains:
- In arithmetic: detects `(+n, -n)` and `(×n, ÷n)` pairs.
- In morphology: detects `(stem→plural, plural→stem)` pairs.
- In logic: detects `(A→B, ¬B→¬A)` (contrapositive) pairs.
- In code: detects `(serialize, deserialize)` pairs.
- In physics: detects `(encode_SI, decode_SI)` unit conversion pairs.

### Q5: The Correct Replacement for the Hardcoded `eq` Separator

**Root cause**: The `eq` token acts as a "rewrite arrow" in the current system.
It separates the left-hand side of a computation from its result. This is a
domain-specific convention: in mathematics, `3 + 4 = 7` uses `=`. In English,
there is no such separator; in code, `->` or `=>` might serve this role.

**General replacement**: Instead of looking for a specific token, the system
should detect **segment boundaries** — the same mechanism already used by
Graph-SEQUITUR for composition discovery.

A segment boundary occurs when a new symbol pair `(a, b)` is observed for
the FIRST time (pair count = 1, never seen before). A composition is triggered
when a pair is seen for the SECOND time. This threshold-free mechanism is
domain-agnostic by design.

**Concrete implementation**:

1. Remove all special-case logic for `eq`.
2. Instead, use the segment boundary signal from the MorphismGraph's pair
   count table: when pair count transitions from 1 to 2, a new composition
   rule candidate is created.
3. The "separator" is no longer a token but an EVENT (the pair count crossing
   2). The left-hand side context and right-hand side context of this event
   are the analogues of `lhs` and `rhs` in the arithmetic case.
4. For domains that DO have explicit separators (like `=` in arithmetic,
   or `->` in Haskell types), these will be discovered automatically as
   high-frequency pair boundaries, and their special role (connecting left-hand
   sides to right-hand sides) will emerge from the FCA adjunction detection.

**Why this works**: The SEQUITUR pair-count criterion is equivalent to MDL.
The "eq" token in arithmetic is just the domain-specific way that the MDL
boundary manifests: the sequence `3 4 = 7` has pair `(4, =)` appearing
frequently (before every result), making it a natural composition boundary.
In language, the same boundary detection will fire on high-frequency
syntactic junctions (NP-VP, modifier-head, etc.) without requiring any
domain-specific separator token.

---

## Concrete Redesign Specification

### Step 1: Replace `_find_endofunctor_patterns` in `rule_store.py`

**Current**: Search for `eq`-terminated sequences, fit polynomial.

**Replacement**:
```
def find_rules(mg: MorphismGraph, topo: Topology) -> List[Rule]:
    rules = []

    # 1. Collect all observed paths grouped by edge type
    for etype in topo.edge_types:
        obs = collect_observations(mg, etype)  # list of (input_path, output_symbol)

        # 2. Try fold detection
        fold = fold_detect(obs)
        if fold is not None:
            rules.append(FoldRule(etype=etype, base=fold.base, step=fold.step))
            continue

        # 3. Try anti-unification of I/O pairs
        schema = anti_unify_io_pairs(obs)
        if schema is not None and schema.num_variables < len(schema.tokens):
            rules.append(SchemaRule(etype=etype, schema=schema))

    # 4. Detect adjunction pairs
    adj_pairs = detect_adjunctions(mg, topo.edge_types)
    rules.extend(AdjunctionRule(r1, r2, cov) for r1, r2, cov in adj_pairs)

    # 5. Run Knuth-Bendix (optional, for canonicalisation)
    trs = knuth_bendix([(r.lhs, r.rhs) for r in rules if isinstance(r, SchemaRule)],
                        ordering=mg_compression_order(mg))
    rules.extend(RewriteRule(lhs, rhs) for lhs, rhs in trs.rules)

    return rules
```

### Step 2: Replace `predict_via_frame_match` in `variable_binding.py`

**Current**: Hard-wired `[op, arg1, arg2, result]` window.

**Replacement**:
```
def predict_via_rule_match(mg: MorphismGraph, atom_buf: List[int],
                            rules: List[Rule]) -> Optional[int]:
    for rule in sorted(rules, key=lambda r: r.specificity, reverse=True):
        match = rule.try_match(atom_buf)
        if match is not None:
            return rule.apply(match)
    return None
```

Where `try_match` uses anti-unification: check if the atom buffer is an
instance of the rule's left-hand side pattern. This is unification (Robinson's
algorithm = coequalizer), which is domain-agnostic.

### Step 3: Replace the `eq` Atom Check

**Current**:
```python
if atoms[-1] == eq_atom_id:
    ...
```

**Replacement**:
```python
# No special atom check. Instead, consult the MorphismGraph's
# pair-count table for segment boundaries.
for pos in range(len(atoms) - 1):
    pair = (atoms[pos], atoms[pos+1])
    if mg.pair_counts.get(pair, 0) >= COMPOSITION_THRESHOLD:
        # This is a composition boundary, not a separator token.
        lhs = atoms[:pos+1]
        rhs = atoms[pos+1:]
        yield (lhs, rhs)
```

This makes the rule learning system completely token-agnostic.

---

## Key Citations for This Report

- Lambek, J. (1968). "A fixed point theorem for complete categories."
  *Mathematische Zeitschrift* 103, 151–161.
- Meijer, E., Fokkinga, M., & Paterson, R. (1991). "Functional Programming
  with Bananas, Lenses, Envelopes and Barbed Wire." *FPCA*.
- Wille, R. (1982). "Restructuring Lattice Theory." *Ordered Sets*, Springer.
- Ganter, B. & Wille, R. (1999). *Formal Concept Analysis: Mathematical
  Foundations*. Springer.
- Blute, R.F., Cockett, J.R.B., & Seely, R.A.G. (2009). "Cartesian
  Differential Categories." *Theory and Applications of Categories* 22(23).
  http://www.tac.mta.ca/tac/volumes/22/23/22-23abs.html
- Joyal, A. (1981). "Une théorie combinatoire des séries formelles."
  *Advances in Mathematics* 42, 1–82.
- Lawvere, F.W. (1963). *Functorial Semantics of Algebraic Theories*. PhD
  thesis, Columbia University. Reprinted TAC 2004.
- Knuth, D.E. & Bendix, P.B. (1970). "Simple Word Problems in Universal
  Algebras." *Computational Problems in Abstract Algebra*, Pergamon.
- Plotkin, G.D. (1970). "A Note on Inductive Generalization." *Machine
  Intelligence* 5, 153–163.
- Robinson, J.A. (1965). "A Machine-Oriented Logic Based on the Resolution
  Principle." *JACM* 12(1), 23–41.
- Bonchi, F., Gadducci, F., Kissinger, A., Sobocinski, P., & Zanasi, F.
  (2022). "String Diagram Rewrite Theory I: Rewriting with Frobenius
  Structure." *JACM* 69(2). arXiv:2012.01847.
- Marcolli, M., Chomsky, N., & Berwick, R.C. (2025). *Mathematical Structure
  of Syntactic Merge*. MIT Press. arXiv:2305.18278.
- Goguen, J. (1989). "What is Unification? A Categorical View of Substitution,
  Equation and Solution." *Resolution of Equations in Algebraic Structures*,
  Academic Press.
- Pratt, V. (1999). "Chu Spaces." *School on Category Theory and Applications*,
  University of Coimbra.
- Barr, M. (1979). *\*-Autonomous Categories*. Springer.
- Steiner, R. (2004). "Omega-categories and chain complexes." *Homology,
  Homotopy and Applications* 6(1). arXiv:math/0403237.
- Gentner, D. (1983). "Structure-Mapping: A Theoretical Framework for
  Analogy." *Cognitive Science* 7, 155–170.
- Martins, R. et al. (2024). "Origami: (un)folding the abstraction of
  recursion schemes for program synthesis." arXiv:2402.13828.
- Cerna, D.M. & Kutsia, T. (2023). "Anti-unification and Generalization:
  A Survey." *IJCAI 2023*. arXiv survey paper.
- Fong, B. & Spivak, D.I. (2019). *Seven Sketches in Compositionality*.
  Cambridge University Press. arXiv:1803.05316.
- Gavranović, B. et al. (2024). "Position: Categorical Deep Learning is an
  Algebraic Theory of All Architectures." arXiv:2402.15332.
