# Category Theory Reference
## Companion document to CTKG_ARCHITECTURE.md

This is the standing reference for every category-theoretic concept used in the
CTKG architecture. Definitions are precise. Each concept ends with a note on its
role in the system. Do not look these up elsewhere.

Cross-references from CTKG_ARCHITECTURE.md cite sections by number, e.g. §16 = FCA.

---

## 1. Category

A **category** C consists of:
- A collection of **objects** ob(C)
- For each pair of objects A, B, a set **hom(A, B)** of **morphisms** (arrows) f: A → B
- For each object A, an **identity morphism** id_A: A → A
- A **composition operation**: if f: A → B and g: B → C then g ∘ f: A → C

Satisfying:
- **Associativity**: h ∘ (g ∘ f) = (h ∘ g) ∘ f
- **Unit laws**: id_B ∘ f = f = f ∘ id_A

**Key examples relevant to this project:**

| Category | Objects | Morphisms |
|---|---|---|
| **Set** | Sets | Functions |
| **Graph** | Graphs | Graph homomorphisms |
| **Mat(ℝ)** | Natural numbers | Real matrices (n × m for hom(m, n)) |
| **Stoch** | Finite sets | Stochastic matrices (Markov kernels) |
| **a free category on a graph** | Graph nodes | Finite paths (including empty paths as identities) |
| **the CTKG** | Concepts / entities | Learned relations and operations |
| **Pos** | Posets | Monotone functions |
| **a monoid** | One object (•) | Elements of the monoid (composition = multiplication) |

**In the CTKG:** Objects are concepts. Morphisms are everything the CTKG knows about
how concepts relate to each other. Every fact, every process, every relation is a
morphism. This uniformity is the architecture's central design principle.

---

## 2. Functor

A **functor** F: C → D assigns:
- To each object A ∈ ob(C), an object F(A) ∈ ob(D)
- To each morphism f: A → B in C, a morphism F(f): F(A) → F(B) in D

Satisfying:
- **Preservation of identity**: F(id_A) = id_{F(A)}
- **Preservation of composition**: F(g ∘ f) = F(g) ∘ F(f)

A **contravariant functor** F: C^op → D reverses arrows: from f: A → B it produces
F(f): F(B) → F(A). Equivalently, it is a functor from the **opposite category** C^op,
which has the same objects as C but all morphisms reversed.

**Key examples:**

- **Forgetful functor** U: Grp → Set forgets the group structure, keeps the set.
- **Free functor** F: Set → Grp sends a set S to the free group on S.
- **Hom functor** hom(A, -): C → Set sends B to the set hom(A, B) and f: B → C
  to the post-composition function (- ∘ f).
- **Power set functor** P: Set → Set sends a set to its power set.

A functor between two CTKGs (or two domains of the CTKG) is a **structure-preserving
domain translation**. If the arithmetic domain and the sequence domain both have a
"successor" concept, a functor maps one to the other preserving all compositional
relationships. This is how the CTKG transfers learned structure across domains.

---

## 3. Natural Transformation

Given functors F, G: C → D, a **natural transformation** η: F ⟹ G assigns to each
object A ∈ ob(C) a morphism η_A: F(A) → G(A) in D, called the **component at A**,
such that for every morphism f: A → B in C the following **naturality square** commutes:

```
F(A) ──η_A──→ G(A)
 │                │
F(f)            G(f)
 │                │
 ↓                ↓
F(B) ──η_B──→ G(B)
```

That is: η_B ∘ F(f) = G(f) ∘ η_A.

Natural transformations are **morphisms between functors**. The category **Fun(C, D)**
has functors as objects and natural transformations as morphisms.

A natural transformation η: F ⟹ G is a **natural isomorphism** if every component
η_A is an isomorphism.

**In the CTKG:** A natural transformation between two functors (domain translations)
is a systematic, structure-preserving mapping between two ways of representing the
same domain. For example, "the English encoding of arithmetic" and "the symbolic
encoding of arithmetic" are two functors from the universal domain to representations.
A natural transformation between them is a translation that commutes with all operations.

---

## 4. Adjunction

An **adjunction** F ⊣ G consists of functors F: C → D (left adjoint) and G: D → C
(right adjoint) together with a natural isomorphism:

```
hom_D(F(A), B) ≅ hom_C(A, G(B))
```

natural in both A ∈ ob(C) and B ∈ ob(D). Equivalently: there exist natural
transformations (the **unit** η: id_C ⟹ G ∘ F and **counit** ε: F ∘ G ⟹ id_D)
satisfying the **triangle identities**:

```
(ε_F) ∘ (F η) = id_F      (G ε) ∘ (η_G) = id_G
```

**Key properties:**
- Left adjoints preserve colimits; right adjoints preserve limits.
- Adjunctions characterize "free constructions": F(A) is the free D-object on A.
- Every adjunction gives rise to a monad (Section 12) and a comonad (Section 13).
- Galois connections (Section 16) are adjunctions in **Pos**.

**Canonical examples:**

| Left adjoint F | Right adjoint G | Universal property |
|---|---|---|
| Free group on S | Underlying set of G | Maps from S to U(G) ↔ group homomorphisms F(S) → G |
| Free category on graph | Underlying graph of C | Graph maps to U(C) ↔ functors F(graph) → C |
| Tensor product A ⊗ - | Internal hom [A, -] | Bilinear maps ↔ linear maps into [A, B] |
| Suspension Σ | Loop space Ω | (topology) |
| Existential quantifier ∃ | Substitution Δ | (logic) |

**In the CTKG:** Adjunctions are **inverse operation pairs**. Addition ⊣ Subtraction.
Encoding ⊣ Decoding. Merge ⊣ Parse. For every operation in the CTKG, the learning
algorithm should discover whether it has an adjoint — a systematic inverse. Discovered
adjoint pairs are first-class knowledge: knowing one operation and its adjoint doubles
the inferential reach of each.

---

## 5. Limits and Colimits

Given a **diagram** D: J → C (a functor from a small category J used as a "shape"),
a **limit** of D is an object lim D together with morphisms (the **cone**) πⱼ: lim D → D(j)
for each j ∈ J, universal in the sense that every other cone factors uniquely through it.

Dually, a **colimit** colim D is an object with morphisms ιⱼ: D(j) → colim D
satisfying the dual universal property.

**Named limits and colimits:**

| J shape | Limit | Colimit |
|---|---|---|
| Empty | Terminal object 1 | Initial object 0 |
| Discrete 2-object | Product A × B | Coproduct A ⊔ B |
| Parallel pair (f, g: A ⇒ B) | Equalizer | Coequalizer |
| Span (A ← C → B) | Pullback | Pushout |
| Cospan (A → C ← B) | — | — |

**Pullbacks** are the categorical model of "intersection" or "constraint satisfaction":
the pullback of f: A → C and g: B → C is the object P with maps to A and B that
agree on C, universal among all such. In type theory: dependent pair types.

**In the CTKG:**
- The **terminal object** is the "trivially true" concept (everything maps to it).
- **Products** A × B represent "A and B simultaneously" — joint constraints.
- **Coproducts** A ⊔ B represent "A or B" — alternatives, polymorphism.
- **Pullbacks** represent constraint intersection: "all entities that are both A and B."
- **Pushouts** represent merging: "unify A and B, identifying their shared boundary."
  Pushouts are how the CTKG merges representations from different sources when they
  share a common sub-structure.

---

## 6. Kan Extension

Given functors F: C → D and K: C → E, the **left Kan extension** Lan_K F: E → D
is the "best approximation to extending F along K" — formally, the left adjoint to
the restriction functor K*: Fun(E, D) → Fun(C, D).

```
      F
C ──────→ D
│          ↑
K       Lan_K F
│         ↗
↓
E
```

Pointwise formula: (Lan_K F)(e) = colim_{K(c) → e} F(c)

The **right Kan extension** Ran_K F is the right adjoint: (Ran_K F)(e) = lim_{e → K(c)} F(c).

Mac Lane: *"All concepts are Kan extensions."*

**Key instances:**

| Kan extension | What it does |
|---|---|
| Lan along inclusion C ↪ D | Extends F defined on C to all of D |
| Lan_K F for K: discrete set → ℕ | Weighted colimit / coend |
| Yoneda lemma | Every representable is a Kan extension of identity |
| Left Kan along K: {·} → C | Colimit of F over all morphisms out of K(·) |

**In the CTKG:** Every generalization from known examples to novel inputs is a Kan
extension. Given a morphism F defined on seen objects (training data), extending F
to unseen objects is Lan_K F for the inclusion K of seen objects into all objects.
The SP's tree composition back-off was an approximation to this; in the CTKG the Kan
extension is computed directly in the categorical structure. Crucially, the Kan
extension is the **unique universal** generalization — not a heuristic.

---

## 7. Yoneda Lemma

For any locally small category C, any object A ∈ ob(C), and any functor F: C → Set:

```
Nat(hom(A, -), F) ≅ F(A)
```

naturally in both A and F. This says: **natural transformations from the representable
functor hom(A, -) to F are in bijection with elements of F(A).**

Corollary — **Yoneda embedding**: the functor y: C → Fun(C^op, Set) defined by
y(A) = hom(-, A) is **fully faithful**. C embeds into its presheaf category.

This means: **an object is completely determined by the totality of morphisms into it
from all other objects.** There are no "intrinsic properties" — only relational ones.

**In the CTKG:** A concept is what it is by virtue of all its relationships. "Paris" as
a CTKG object is completely characterized by every morphism that maps to or from it.
Learning "Paris" = learning its full hom-set. This justifies the relational learning
approach: there is no separate "representation" to learn beyond the morphisms themselves.

The Yoneda lemma also underlies **in-context learning**: a foundation model's behavior
F(A) on input A is determined by natural transformations from the representable
hom(A, -) — i.e., by how similar inputs behave (Yuan et al. 2023).

---

## 8. Monoidal Category

A **monoidal category** (C, ⊗, I, α, λ, ρ) consists of:
- A category C
- A **tensor product** functor ⊗: C × C → C
- A **unit object** I
- Natural isomorphisms: **associator** α_{A,B,C}: (A ⊗ B) ⊗ C ≅ A ⊗ (B ⊗ C),
  **left unitor** λ_A: I ⊗ A ≅ A, **right unitor** ρ_A: A ⊗ I ≅ A

Satisfying coherence conditions (pentagon and triangle equations).

A **symmetric monoidal category** additionally has a natural isomorphism
σ_{A,B}: A ⊗ B ≅ B ⊗ A (swap) with σ_{B,A} ∘ σ_{A,B} = id.

A **closed monoidal category** has an **internal hom** [A, B] with
hom(C ⊗ A, B) ≅ hom(C, [A, B]) (currying/uncurrying adjunction).

A **cartesian monoidal category** has ⊗ = × (product) and I = 1 (terminal object).
Every cartesian monoidal category is symmetric and closed (if it has exponentials).

**String diagrams** are the graphical calculus for monoidal categories:
- Objects = wires
- Morphisms = boxes with wires in at the top, wires out at the bottom
- ⊗ = wires side by side (parallel composition)
- ∘ = boxes stacked (sequential composition)
- The calculus is **sound and complete**: two string diagrams are equal iff they
  represent the same morphism. Proofs become topological manipulations.

**In the CTKG:** The CTKG is monoidal. Parallel composition (⊗) models simultaneously
holding multiple concepts in working memory. Sequential composition (∘) models
deriving one concept from another. Closed monoidal structure gives the CTKG
**internal functions**: [A, B] is the object representing "all processes from A to B"
— enabling higher-order reasoning.

---

## 9. Operad and Multicategory

A **multicategory** (or **colored operad**) M consists of:
- A collection of **objects** (colors) ob(M)
- For each tuple of objects (A₁, ..., Aₙ) and object B, a set
  **M(A₁, ..., Aₙ; B)** of **multi-morphisms**
- **Composition**: given f ∈ M(A₁,...,Aₙ; B) and gᵢ ∈ M(B_{i,1},...,B_{i,kᵢ}; Aᵢ),
  their composite is in M(B_{1,1},...,B_{n,kₙ}; B)
- Satisfying associativity and unit laws

A **symmetric multicategory** additionally has an action of the symmetric group Σₙ
on each M(A₁,...,Aₙ; B) compatible with composition.

An **operad** is a multicategory with a single color (all operations have the same
input and output type). The **free operad** on a signature Σ is the collection of
all finite labeled rooted trees with leaves labeled by arguments.

**In the CTKG:** Merge is a multi-morphism. "The verb 'eats' takes a subject NP and
an object NP to produce a sentence S" is a multi-morphism eats: (NP, NP) → S.
"Addition takes two digits to produce a digit" is +: (digit, digit) → digit.

The **free operad** on the CTKG's discovered operations is the space of all possible
Merge trees — all possible hierarchical compositions. The CTKG's learning problem
is: given observed sequences (flat), discover the operad structure (tree) that
generated them.

Colored operads naturally give **context-dependent types**: the color of a constituent
determines its role in the composition. "Bank" as NP-subject has a different color
than "bank" as N-in-PP. Colors are not assigned by a dictionary — they are inferred
from the operad structure of the observed compositions.

---

## 10. Presheaf and Sheaf

A **presheaf** on a category C is a functor F: C^op → Set.

The **presheaf category** Set^{C^op} = [C^op, Set] = P(C) is the category of all
presheaves on C, with natural transformations as morphisms.

A **site** is a category C equipped with a **Grothendieck topology** — a specification,
for each object U ∈ ob(C), of which families of morphisms {Uᵢ → U} "cover" U.

A **sheaf** on a site (C, J) is a presheaf F satisfying the **sheaf condition**:
for every covering family {Uᵢ → U}, F(U) is the **limit** of the diagram built from
the Fᵢ = F(Uᵢ) and their pairwise intersections F(Uᵢ ×_U Uⱼ). Informally: **local
data that agree on overlaps glue uniquely to global data**.

The full subcategory of sheaves Sh(C, J) ⊆ P(C) is a **Grothendieck topos**.

**The key idea for context-dependent representations:**

Let C be the **context category**: objects are contexts (situations, discourse
states, dialogue histories), morphisms are context refinements (c' → c means c'
is a more specific context than c). A presheaf F: C^op → Set assigns to each
context c a set F(c) of admissible representations, and to each refinement c' → c
a **restriction map** F(c) → F(c') that specializes a representation as context
becomes more specific.

"Bank" at coarse context c might have F(c) = {financial_institution, river_margin}.
At refined context c' (financial conversation), F(c') = {financial_institution}.
The sheaf condition says: representations that agree locally (in each specific context)
patch together consistently to a global representation.

**In the CTKG:** The CTKG is a sheaf on the context category. Learning context-dependent
representations = learning the restriction maps. Every token's type is a section of
this sheaf. The sheaf condition is the well-formedness constraint on the learned
representations: local consistency ↔ global existence.

---

## 11. Topos

A **Grothendieck topos** is a category equivalent to Sh(C, J) for some site (C, J).

An **elementary topos** is a category E that:
- Has all finite limits
- Has a **subobject classifier** Ω (an object such that sub-objects of X correspond
  to morphisms X → Ω)
- Has exponentials (is cartesian closed)

Every Grothendieck topos is an elementary topos. Every elementary topos has an
**internal language** — a typed higher-order intuitionistic logic — in which one can
reason about the category's objects and morphisms.

**Subobject classifier Ω:** In Set, Ω = {true, false} and sub-objects of X are
exactly characteristic functions X → {true, false}. In a general topos, Ω may be
more complex — it is the "object of truth values" and may have more than two elements
(corresponding to partial truth, probabilistic truth, etc.).

**Internal logic:** Every statement in the internal language of a topos corresponds
to a morphism in the topos. Logical connectives (∧, ∨, ⟹, ¬, ∀, ∃) are all
morphisms. Proving a theorem = constructing a morphism. This is the Curry-Howard
correspondence in its most general form.

**In the CTKG:** The CTKG generates a topos. Reasoning within the CTKG is reasoning
in the internal logic of this topos. Truth is not binary (Ω ≠ {T, F}) — it is a
presheaf of evidence. "Paris is in France" is a morphism in the CTKG; "Is Paris in
France?" is asking for its existence. The probabilistic structure (Section 15) makes
Ω the presheaf of probability distributions — graded truth.

---

## 12. Monad

A **monad** on a category C is a triple (T, η, μ) where:
- T: C → C is a functor (the "computation type")
- η: id_C ⟹ T is the **unit** (natural transformation — "return" / "pure")
- μ: T² ⟹ T is the **multiplication** (natural transformation — "join" / "bind")

Satisfying:
- μ ∘ T(η) = id_T = μ ∘ η_T  (unit laws)
- μ ∘ T(μ) = μ ∘ μ_T  (associativity)

Every adjunction F ⊣ G gives a monad T = G ∘ F with η = unit and μ = G(ε_F).

**The Kleisli category** of a monad (T, η, μ) has the same objects as C and
Kleisli morphisms A → B defined as morphisms A → T(B) in C. Composition of Kleisli
morphisms is via the monad's bind: (A → T(B)) then (B → T(C)) composes to A → T(C).

**Key monads:**

| Monad T(A) | Effect modeled | Kleisli morphisms |
|---|---|---|
| A × S (state) | Mutable state | Stateful functions |
| A + E (exception) | Error handling | Partial functions |
| List(A) | Nondeterminism | Relations |
| Dist(A) | Probability | Stochastic maps |
| Maybe(A) | Partiality | Partial functions |
| Reader(E, A) = E → A | Read environment | Context-dependent functions |
| Writer(W, A) = W × A | Append-only log | Functions that emit |

**Monad algebras (T-algebras):** An algebra for monad T is an object A with a
morphism a: T(A) → A satisfying a ∘ η_A = id and a ∘ μ_A = a ∘ T(a). T-algebras
are exactly the objects that "know how to evaluate T-computations." The category of
T-algebras is the **Eilenberg-Moore category** C^T.

**In the CTKG:** The CTKG's working memory is a state monad T(A) = (Memory → A × Memory).
Every CTKG operation that reads from or writes to working memory is a Kleisli morphism.
The probability monad T(A) = Dist(A) models uncertain inference: CTKG morphisms are
Kleisli morphisms A → Dist(B) (stochastic maps). The **free monad** on the CTKG's
operation signature is the computation tree of all possible CTKG programs — the space
of all inference chains.

---

## 13. Comonad

A **comonad** on C is a triple (W, ε, δ) where:
- W: C → C is a functor
- ε: W ⟹ id_C is the **counit** ("extract" — get the current value)
- δ: W ⟹ W² is the **comultiplication** ("duplicate" — extend the context)

Satisfying the dual laws to a monad.

**The co-Kleisli category** has morphisms A → B defined as W(A) → B.

**Key comonads:**

| Comonad W(A) | Models | Co-Kleisli morphisms |
|---|---|---|
| (S → A) × S (Store) | Focus in a context | Context-dependent computations |
| Stream(A) = Aⁿ | Infinite sequence | Sliding window functions |
| Env(E, A) = E × A | Read environment | Functions using fixed context E |
| Traced(A) | History/trace | Path-dependent functions |

**Store comonad** in detail: W(A) = (S → A) × S where S is the "position" type.
- **extract**: given (f, s), return f(s) — get the current value
- **duplicate**: given (f, s), return (λs'. (f, s'), s) — the same store viewed from each position

A **co-Kleisli morphism** W(A) → B is a function that looks at the full context
(f: S → A, current position s) and produces a result. This is exactly the structure
of a **cellular automaton rule** or a **sliding window computation**.

**In the CTKG:** Working memory is a comonad. W(A) = (Context → A) × FocusToken:
the full context as a function, plus the current focus. Every CTKG operation that
uses context (not just the current token) is a co-Kleisli morphism. The comonad
structure ensures: (1) you can always extract the current value (ε), (2) you can
always re-examine the context from a different focus (δ). Attention is a co-Kleisli
morphism: given the full context, produce the attended representation.

---

## 14. Enriched Category

A **V-enriched category** C (where V is a monoidal category) has:
- Objects ob(C)
- For each pair A, B: a **hom-object** C(A, B) ∈ ob(V) instead of a set
- Composition: C(B, C) ⊗ C(A, B) → C(A, C) in V
- Identities: I → C(A, A) in V

When V = Set: ordinary categories. When V = Ab (abelian groups): preadditive.
When V = [0, ∞]: Lawvere metric spaces. When V = {0 ≤ 1}: preordered sets.

**Lawvere metric spaces** (V = ([0,∞], ≥, +, 0)): a V-enriched category where
hom(A, B) = d(A, B) is the "distance" from A to B (not necessarily symmetric).
Composition: d(A, C) ≤ d(A, B) + d(B, C) (triangle inequality). This is the
categorical formulation of a generalized metric space.

**In the CTKG:** The CTKG is enriched over probability distributions: hom(A, B) is
not just whether a morphism exists but a **confidence** — a probability distribution
over possible morphisms. V-enrichment formalizes this: hom(A, B) ∈ Dist gives
the distribution over relations from A to B. Composition is probabilistic
(marginalization). The Lawvere metric structure measures concept similarity.

---

## 15. Markov Category

A **Markov category** (Fritz 2020) is a symmetric monoidal category (C, ⊗, I) where
every object A has a **copy morphism** copy_A: A → A ⊗ A and **delete morphism**
del_A: A → I, satisfying coherence axioms making every object a **cocommutative
comonoid**.

**Intuition:** In a Markov category, morphisms are stochastic processes. copy lets
you duplicate a random variable (to use in two places); del lets you discard it.

**Key instance:** Stoch (finite sets, stochastic matrices) is a Markov category.
FinStoch is the full sub-Markov-category of finite sets.

**Concepts definable internally in any Markov category:**
- **Conditional independence**: X ⊥ Y | Z — defined without reference to probabilities
- **d-separation**: the graphical criterion (Bayes-ball) for conditional independence
- **Conditional distribution**: P(X | Y = y) as a morphism in the Kleisli category
- **Sufficient statistic**: a morphism that captures all relevant information
- **Intervention (do-calculus)**: surgery on the string diagram (Jacobs et al. 2019)

**Entropy** in a Markov category is characterized uniquely by three axioms
(Baez-Fritz-Leinster 2011): (1) functoriality, (2) convex-linearity,
(3) continuity ⟹ Shannon entropy. No other formula satisfies all three.

**In the CTKG:** All uncertain inference in the CTKG is Markov-categorical. Morphisms
are stochastic (hom(A, B) = probability distribution). d-separation answers "does
knowing C make A and B independent?" Intervention answers "if I force C to take value
c, what happens to B?" These are first-class CTKG operations, not special-case
add-ons.

---

## 16. Formal Concept Analysis (FCA) and Galois Connections

A **formal context** is a triple K = (G, M, I) where G is a set of **objects**,
M is a set of **attributes**, and I ⊆ G × M is the incidence relation ("object g
has attribute m").

Two **derivation operators**:
- A' = {m ∈ M | gIm for all g ∈ A} for A ⊆ G (attributes shared by all objects in A)
- B' = {g ∈ G | gIm for all m ∈ B} for B ⊆ M (objects possessing all attributes in B)

A **formal concept** is a pair (A, B) with A' = B and B' = A. The set of formal
concepts ordered by A₁ ⊆ A₂ (equivalently B₂ ⊆ B₁) forms the **concept lattice** B(K).

The operators (')^op ⊣ (') form a **Galois connection** — an adjunction in the
poset category. This adjunction is the mathematical core of FCA: object-extent and
attribute-intent are adjoints of each other.

**MDL justification:** The concept lattice is the maximum-compression representation
of the binary relation I. It has zero free parameters: both G and M are observed
data; the lattice structure follows necessarily.

**In the CTKG:** FCA discovers adjunction pairs from data. Given a corpus of
(context, token) observations, FCA finds clusters of contexts that share exactly
the same token distributions — these are CTKG objects. The Galois connection gives
the adjunction: context-extent ⊣ token-intent. This is the categorical version of
"token clusters with their defining contexts." FCA is the primary tool for
discovering the CTKG's objects from raw data, with no domain knowledge.

---

## 17. Free Category and Quotient Category

The **free category** FC(G) on a directed graph G has:
- Objects = nodes of G
- Morphisms from A to B = all finite paths in G from A to B (including empty paths)
- Composition = path concatenation
- Identity = empty path at each node

FC(G) is the "most general" category with the given graph as its skeleton — no
equations hold except the minimal ones required by the category axioms.

A **presentation** of a category C is a graph G plus a set of **equations**
(path equalities) E such that C ≅ FC(G) / E (the quotient obtained by imposing E).

The quotient FC(G) / E is computed by taking the **congruence closure** of E under
composition: two paths are equal in the quotient iff they can be connected by a
finite chain of equation applications.

**In the CTKG:** Learning from data = discovering which paths are equal.
- Start with FC(G) where G = all observed (antecedent, consequent) token pairs.
- Each new observation of "context C produces the same outcome as context C'" is an
  equation C = C' in E.
- The CTKG converges to FC(G) / E as more equations are discovered.
- MDL principle selects the minimal E that explains the data.

This formulation subsumes SP's k-gram statistics (equation: every context of depth k
that leads to the same output distribution is identified), Tucker role conditioning
(equation: same token at different roles in identical structural positions are
identified), and Kan extension (the universal property of the quotient gives the
unique generalization).

---

## 18. Hopf Algebra

A **bialgebra** over a field k is a vector space H that is simultaneously an algebra
(product m: H ⊗ H → H, unit η: k → H) and a coalgebra (coproduct Δ: H → H ⊗ H,
counit ε: H → k), with the two structures compatible:
```
Δ ∘ m = (m ⊗ m) ∘ (id ⊗ σ ⊗ id) ∘ (Δ ⊗ Δ)
```
(where σ is the swap map).

A **Hopf algebra** is a bialgebra with an **antipode** S: H → H satisfying:
```
m ∘ (S ⊗ id) ∘ Δ = η ∘ ε = m ∘ (id ⊗ S) ∘ Δ
```

**Linguistic Merge** (Marcolli, Chomsky, Berwick 2023): Syntactic objects form a
Hopf algebra where:
- **m** = Merge: combines two syntactic objects into one
- **Δ** = parse/segment: decomposes a syntactic object into its constituents
- **S** = the antipode corresponds to movement/displacement operations

The **cross-level functor** — the claim that the same algorithm operates at every
scale of hierarchical structure — follows from the Hopf algebra axioms: Δ(m(x, y))
is always defined and expresses the multi-scale decomposition.

**In the CTKG:** Merge is the primary composition operation. The Hopf algebra structure
gives it both a forward direction (build larger structures from smaller ones) and a
backward direction (parse larger structures into their constituents). The antipode
gives the algebra its group-like properties, enabling inverses and movement. This is
not a metaphor — the learned CTKG morphisms must satisfy the bialgebra compatibility
condition, which is a testable constraint on the learned structure.

---

## 19. Natural Number Object, F-Algebras, and Initial Algebras

This section exists to prevent a specific class of error: **algorithms that identify
structure by reading the string label attached to a node** (e.g., checking whether a
token is a "digit" by matching the string "digit", or recognising an addition table
by pattern-matching on tokens named "0".."9" and "+"). Such algorithms are domain-
specific, brittle to renaming, and learn nothing structural. The right replacement is
to check whether an object satisfies the appropriate **universal property**.

### F-Algebra

Given a functor F: C → C, an **F-algebra** is a pair (A, α) where A ∈ ob(C) and
α: F(A) → A is a morphism (the **structure map** or **evaluation map**). A
**morphism of F-algebras** from (A, α) to (B, β) is a morphism f: A → B in C such
that the following square commutes:

```
F(A) ──α──→ A
  │              │
F(f)           f
  │              │
  ↓              ↓
F(B) ──β──→ B
```

F-algebras and their morphisms form a category **Alg(F)**. The initiality condition
picks out the canonical, domain-free definition of any recursive structure.

### Initial F-Algebra (Lambek's Theorem)

An **initial F-algebra** (I, ι) is the initial object in **Alg(F)**: for every
F-algebra (A, α) there exists a unique algebra morphism (I, ι) → (A, α).

**Lambek's theorem**: if (I, ι) is initial then ι: F(I) → I is an isomorphism.
That is, the initial algebra is a fixed point of F: F(I) ≅ I.

This means: the initial F-algebra is simultaneously the *smallest* fixed point and
the *canonical* recursive type defined by F.

### Natural Number Object (NNO)

Let C be a category with a terminal object 1. The **natural number object (NNO)**
is an object N with morphisms

```
z: 1 → N     (zero)
s: N → N     (successor)
```

such that for any object X and morphisms q: 1 → X, f: X → X, there exists a
**unique** morphism u: N → X making both triangles commute:

```
1 ──z──→ N ──s──→ N
│          │           │
q          u           u
│          │           │
↓          ↓           ↓
X ══════ X ──f──→ X
```

i.e. u ∘ z = q and u ∘ s = f ∘ u.

The NNO is precisely the **initial algebra for the functor F(X) = 1 + X** (the
functor that "adds one point"). Its initial algebra structure map is
ι: 1 + N → N sending the left summand to z(*) and the right summand to s.

The unique morphism u is **primitive recursion**: given a base case q and a
step f, u computes the result at every natural number by recursion. The
universal property *guarantees* that u is the only coherent such function.

**Key consequence**: the NNO is defined entirely by its role in the category.
It does not matter whether the elements are called {0, 1, 2, ...}, {"zero",
"one", "two", ...}, {"〇", "一", "二", ...}, or {☆, ★, ★★, ...}.
Any object satisfying this universal property *is* the natural numbers, up to
unique isomorphism.

### Other Recursive Types as Initial F-Algebras

| Type | Functor F(X) | Initial algebra |
|---|---|---|
| Natural numbers ℕ | 1 + X | (ℕ, [zero, succ]) |
| Binary trees T(A) | 1 + A × X × X | (T(A), [leaf, node]) |
| Lists List(A) | 1 + A × X | (List(A), [nil, cons]) |
| Sequences over Σ | 1 + Σ × X | (Σ*, [ε, prepend]) |
| The Merge tree | 1 + X × X (or colored version) | Free operad on Merge |

Each of these is recognised by checking the **universal property** of the relevant
functor F, not by looking at labels.

### Why This Matters for the CTKG

Every time the CTKG's learning algorithm needs to identify "what kind of structure
is this?", it must use universal properties — not string matching.

**Example — detecting digit-like objects:**
An object D "behaves like digits" not because it is labelled "digit" but because
there exists a morphism s: D → D (successor) and a terminal-like injection z: 1 → D
that together satisfy the NNO universal property (or a finite truncation of it —
the integers mod 10 satisfy a cyclic version). The algorithm checks: does the
category of observed (D, s, z) satisfy the necessary commutation conditions?
If yes, D is a natural-number-like object regardless of what its elements are
named.

**Example — detecting operator-like objects:**
An object OP "behaves like a binary operator" because there exist morphisms
op: D × D → D (for some object D) satisfying the F-algebra structure for the
appropriate functor. The CTKG detects this by testing the F-algebra morphism
condition, not by checking whether a token is named "+", "×", or "⊕".

**The anti-pattern to avoid:**
```python
# WRONG — reads the label, not the structure:
if token_label in {"0", "1", "2", ..., "9"}:
    classify_as_digit()

# RIGHT — tests the universal property:
if has_successor_morphism(obj) and satisfies_nno_universal_property(obj):
    classify_as_nno_like()
```

**General rule**: any place in the CTKG code where a string comparison is used to
identify what an object IS is a latent domain-knowledge bug. Replace every such
check with a test of the appropriate universal property or F-algebra morphism
condition. The CTKG's type system — derived from FCA on the concept lattice — is
the correct mechanism for structural identification.

---

## 20. Profunctor

A **profunctor** (or **distributor**, or **bimodule**) P: C ↛ D is a functor
P: D^op × C → Set.

Equivalently: a profunctor P: C ↛ D is a functor C → P(D) into the presheaf category
of D. Profunctors compose via the **coend formula**:
```
(Q ∘ P)(d, c) = ∫^e P(e, c) × Q(d, e)
```
The category **Prof** has small categories as objects and profunctors as morphisms.

Every functor F: C → D gives a profunctor hom_D(-, F(-)): D^op × C → Set.
A profunctor not arising from a functor is a "generalized functor" or "partial map."

**In the CTKG:** Profunctors model **bidirectional relations with evidence**. A
profunctor P: C ↛ D assigns to each pair (target d, source c) the set of ways c
can be related to d. This is the right type for learned relations in the CTKG:
not a function (deterministic) nor just a set (unstructured), but a Set-valued
bimodule with both variance directions. When P is representable (comes from a
functor), the relation is functional. When it's not, it's genuinely relational
(one-to-many or partial).

---

## 20. Density Comonad and Codensity Monad

Given a functor K: C → D, the **left Kan extension** Lan_K (id_C) along K is called
the **density comonad** of K (when K is fully faithful, this is a genuine comonad
on D). Dually, the **codensity monad** of K is Ran_K (id_C), the right Kan extension.

The codensity monad T of K: C → D can be computed pointwise:
```
T(d) = ∫_c D(d, K(c))^{C(c, ?)}
```
(an end formula). For K: C → Set (a "classifier"), T(S) = the set of "ultrafilters"
on S relative to the type structure.

**Learning connection:** Given a dataset (a functor K: examples → predictions), the
codensity monad of K is the most general monad that is learned from the data — the
"monad you get if you take the examples seriously." This is the analogue of computing
the empirical distribution: Ran_K(id) captures the maximal statistical structure
extractable from the training examples without additional assumptions.

**In the CTKG:** The codensity monad of the "observation functor" (training corpus →
CTKG objects) is the principled completion of the CTKG from training data. It gives
the most conservative generalization: Ran_K(id) captures exactly the structure forced
by the observations, and nothing more. This is Occam's razor in categorical form.

---

