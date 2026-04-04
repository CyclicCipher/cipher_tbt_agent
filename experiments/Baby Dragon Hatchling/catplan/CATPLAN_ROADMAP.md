# CatPlan Completion Roadmap

## Current state (2026-03-30)

Working: parser, type system, action grounding, precondition checking
(simple predicates), effect application (assert/retract/wildcard),
frozenset state representation, A* search with goal-count heuristic.

Validated on blocks-world tier 1 (3/3 solved) and tier 2 Hanoi (solved
in 10 steps).

---

## Phase A: Language completeness

Make CatPlan handle everything PDDL can handle, plus typed.

### A.1: Derived predicates (axioms)
Define predicates computed from other predicates, not maintained manually.
```
  derived clear(b : Block) = not(exists b2 : Block . on(b2, b))
```
The planner recomputes derived predicates after each action, eliminating
the need to manually add/remove `clear` in every action's effects.

Implementation:
- New `DerivedPredicate` type: name, params, body (a logical formula)
- After applying effects, recompute all derived predicates from scratch
- Parser support for `derived` keyword
- **Test:** rewrite blocks domain with `clear` as derived. Same plans,
  cleaner domain file.

### A.2: Conditional effects
Effects that only apply when a condition holds.
```
  action pick(h : Hand, b : Block)
    require empty(h), clear(b)
    effect holding(h, b), not(empty(h))
    when on(b, x) : not(on(b, x)), clear(x)
    when on_table(b) : not(on_table(b))
```
Implementation:
- New `ConditionalEffect` type: condition + effect list
- `apply_action` checks each `when` condition against current state
- Parser support for `when` keyword within actions
- **Test:** reunify pick_from_table and pick_from_block into single
  `pick` action with conditional effects. Same plans.

### A.3: Quantified preconditions
`exists` and `forall` in preconditions.
```
  require exists x : Block . on(b, x)
  require forall x : Block . not(on(x, target))
```
Implementation:
- `ExistsPrecondition(var, type, body)` and `ForallPrecondition`
- Evaluate by iterating over all objects of the given type
- Parser support for `exists` and `forall` in require lines
- **Test:** blocks domain with `exists` preconditions

### A.4: Disjunctive preconditions
`or` in preconditions.
```
  require on_table(b) or exists x : Block . on(b, x)
```
Implementation:
- `OrCondition(conditions)` wrapping multiple conditions
- Evaluate: true if any sub-condition is true
- Parser support for `or` in require lines
- **Test:** single `pick` action with `or` precondition

### A.5: Equality and inequality constraints
```
  require b /= target
  require pos(a) = pos(b)
```
Implementation:
- `EqualityCondition(arg1, arg2, negated)`
- Evaluate by comparing bound object names
- Parser support for `=` and `/=` in require lines
- **Test:** stack action with explicit `/=` constraint

### A.6: Negative goals
Goals that require something to NOT be true.
```
  goal
    not(on(a, b))
    on_table(a)
```
Implementation:
- Extend goal representation to include negated atoms
- Update goal satisfaction check and heuristic
- **Test:** "clear the table" type goals

---

## Phase B: Invariants and verification

Make CatPlan verify its own work.

### B.1: Invariant evaluation
Actually evaluate the invariant expressions parsed in Phase A.3/A.4.
```
  invariant "hand holds at most one block"
    forall h : Hand . count(b : Block . holding(h, b)) <= 1
```
Implementation:
- Evaluate invariants as boolean formulas over a state
- Add `count` aggregation support
- Check invariants after each action application
- If an action would violate an invariant, it's inapplicable
  (stronger than preconditions — invariants are global constraints)
- **Test:** blocks domain invariant "a block is on exactly one surface"
  catches bugs in action definitions

### B.2: Plan validation
Given a plan (list of ground actions), independently verify:
1. Each action's preconditions hold in the state where it's applied
2. No invariant is violated at any intermediate state
3. The final state satisfies the goal
4. Report which step fails and why if invalid

Implementation:
- `validate_plan(domain, problem, plan) -> ValidationResult`
- **Test:** validate all plans from tier 1-2, also test with a
  deliberately broken plan

### B.3: Sheaf energy computation
Port sheaf.py concepts into CatPlan. For each state:
- Energy = sum of invariant violations (weighted)
- A valid state has energy 0
- Energy measures "how far from valid" a state is

Implementation:
- `sheaf_energy(state, domain) -> float`
- Weight each invariant (hard vs soft constraints)
- **Test:** compute energy for valid and invalid blocks states

### B.4: Sheaf energy as search heuristic
Replace or augment goal-count heuristic with sheaf energy.
- h(state) = goal_count(state) + sheaf_energy_of_goal_projection(state)
- The goal projection: "if we pretend the goal atoms are added to the
  state, how inconsistent is it?" High inconsistency = we're missing
  prerequisites.

Implementation:
- `sheaf_heuristic(state, goal, domain) -> float`
- Combine with goal-count in multi-heuristic search
- **Test:** compare A* node expansions with goal-count vs sheaf
  on Hanoi — sheaf should expand fewer nodes

---

## Phase C: Search improvements (borrowed from PDDL ecosystem)

Make CatPlan fast enough for real problems. Each technique is proven
in the PDDL competition ecosystem and is algorithm-level (not
PDDL-specific).

### C.1: Causal graph extraction
Build the causal graph: which state variables affect which other
variables through actions. A directed graph where nodes are predicates
and edges are "action A reads predicate P and writes predicate Q."

Fast Downward computes this at search time from grounded actions.
We can do it from the CatPlan domain structure directly — the typed
morphism signatures already encode which predicates are read/written
by which actions.

Additionally, apply Krohn-Rhodes (from algebra.py) to decompose the
causal graph into SCCs. Tightly coupled variables form groups; the
DAG of groups gives a solving order.

Implementation:
- `build_causal_graph(domain) -> dict[str, set[str]]`
- `decompose_causal_graph(cg) -> list[set[str]]` (SCCs via Tarjan)
- **Test:** blocks domain causal graph: on→clear→holding→empty chain

### C.2: Causal graph heuristic
Estimate goal distance by solving subproblems in each SCC independently.
Cost = sum of subproblem costs. This is Fast Downward's CG heuristic.

Implementation:
- For each goal variable: estimate cost by relaxed planning in its
  SCC (ignoring delete effects, standard relaxation)
- Sum costs across goal variables
- **Test:** compare node expansions CG heuristic vs goal-count on
  Hanoi and blocks tier 3

### C.3: Landmark analysis
A landmark is a predicate that MUST become true at some point in
every valid plan. The count of unachieved landmarks is an admissible
heuristic.

Categorical interpretation: landmarks are the terminal objects in the
subcategory of states reachable from init that also reach the goal.
Every path through the category must pass through them.

Implementation:
- `find_landmarks(domain, problem) -> set[GroundAtom]`
- Use relaxed planning graph to identify landmarks (standard method)
- Landmark count heuristic: h = |unachieved landmarks|
- **Test:** blocks tier 3: identify that "hand must hold each block
  at least once" as a landmark

### C.4: Preferred operators
When expanding a state, give priority to actions that:
1. Achieve an unachieved landmark
2. Reduce sheaf energy
3. Achieve a goal atom
These actions go into a preferred queue with higher priority.

Implementation:
- Dual open list: preferred queue + regular queue
- Alternate between them (as Fast Downward does)
- **Test:** measure wall-clock time improvement on Hanoi and tier 3

### C.5: Deferred heuristic evaluation
Don't compute expensive heuristics (CG, landmark) when a state is
first generated. Only compute when it reaches the front of the open
list and is about to be expanded. Many states are generated but never
expanded — deferred evaluation skips wasted heuristic computation.

Implementation:
- Lazy evaluation: store states with f=g (no h) initially
- When popped from open list, compute h, push back if f increases
- **Test:** measure heuristic computation count reduction

### C.6: Multi-heuristic search
Combine multiple heuristics in a single search, as LAMA does:
- Open list 1: goal-count heuristic
- Open list 2: sheaf energy heuristic
- Open list 3: landmark count heuristic
- Preferred operator list for each
- Round-robin expansion across lists

Implementation:
- `MultiHeuristicSearch` class with N open lists
- Round-robin or boost-based expansion selection
- **Test:** compare against single-heuristic A* on blocks tier 3
  and logistics tier 3

### C.7: Delete relaxation (FF heuristic)
The FF heuristic: estimate goal distance by solving a relaxed problem
where delete effects are ignored. The relaxed problem is solvable in
polynomial time (just greedily add atoms). The length of the relaxed
plan = heuristic estimate.

This is the most widely used PDDL heuristic. Well understood, fast to
compute, informative.

Implementation:
- Build relaxed planning graph (ignore delete effects)
- Extract relaxed plan via backward chaining from goal
- h = |relaxed plan|
- **Test:** compare FF heuristic vs goal-count on all blocks tiers

---

## Phase D: Categorical features (CatPlan's unique advantages)

These go beyond what PDDL can do. Each construct is a universal
planning primitive — domain-general machinery that applies to any
domain, not a special case for one problem type.

The guiding principle: every construct is a universal property.
Universal properties are inherently domain-general because they're
defined by what they DO (their interface), not what they ARE
(their implementation). A pullback is "the most general thing
satisfying two constraints simultaneously" — whether the constraints
are about blocks, molecules, equations, or grammar rules.

### D.1: Morphism composition (macro-operators)
**Universal property:** composition is associative and typed.

Define composite actions from existing ones:
```
  compose pick_and_stack(h, b, target) = stack(h, b, target) . pick(h, b)
```
The planner uses composites as macro-operators, reducing search depth.
Composition is type-checked at definition time: the postcondition of
the first action must satisfy the precondition of the second.

Domain-general: applies to any sequence of actions in any domain.
Blocks: pick_and_stack. Chemistry: dissolve_then_filter. Logistics:
load_drive_unload. The planner discovers useful compositions by
finding frequently co-occurring action pairs in successful plans.

Implementation:
- `CompositeAction`: ordered list of action references
- Type-check compatibility at composition time
- Planner can expand or use as single-step macros
- Auto-discovery: after solving N problems, identify frequent
  action subsequences and offer them as composites
- **Test:** define pick_and_stack, verify planner uses it and
  reduces search depth

### D.2: Adjunctions (inverse operations)
**Universal property:** F left adjoint to G means solving F(x) = y
is equivalent to x = G(y). The most general notion of "inverse."

```
  adjunction add_sub : add -| sub
    -- from add(a, b) = c, derive sub(c, b) = a and sub(c, a) = b
  adjunction encrypt_decrypt : encrypt -| decrypt
  adjunction push_pop : push -| pop
```

When the planner encounters a goal that matches the OUTPUT of an
adjoint pair, it can compute the input directly instead of searching.
"Find x such that add(x, 3) = 7" → apply right adjoint → x = sub(7, 3) = 4.
No search, no heuristic, direct computation.

Domain-general: any pair of operations where one undoes the other.
Math: +/-, ×/÷, d/dx / ∫dx. Chemistry: bond/break, oxidize/reduce.
Circuits: charge/discharge. Language: encode/decode.
Physics: accelerate/decelerate.

Implementation:
- `Adjunction` type: left functor, right functor, unit, counit
- When goal matches left adjoint's output pattern, apply right adjoint
- Planner inserts adjunction resolution as a zero-cost "computation
  step" before resorting to search
- Port from experiments/ctkg/graph.py (Adjunction already defined)
- **Test:** arithmetic domain: solve x + 3 = 7 via adjunction, not search

### D.3: Equalizers (single-equation solving)
**Universal property:** the equalizer of f and g is the largest
subobject where f = g. "Find all x where f(x) = g(x)."

```
  equalizer solution(x) where f(x) = g(x)
    -- e.g., find x where temperature(x) = target_temp
```

The planner recognizes when a goal is an equality between two
computed predicates and constructs the equalizer directly.

Domain-general: any "find the state where two things agree" problem.
Physics: equilibrium (forces balance). Chemistry: reaction equilibrium.
Economics: supply = demand. Geometry: intersection of curves.

Implementation:
- `Equalizer` type: two morphisms with common source and target
- Solve by evaluating both morphisms across all objects of the source type
- For numeric predicates (Phase D.8): use bisection or Newton's method
- **Test:** find the block where weight(b) = capacity(shelf)

### D.4: Pullbacks (simultaneous constraints)
**Universal property:** the pullback of f: A→C and g: B→C is the
largest thing mapping to both A and B such that f and g agree.
"Find x satisfying constraint1 AND constraint2 simultaneously."

```
  pullback valid_cell(row, col, box) where
    row_constraint(row) = digit
    col_constraint(col) = digit
    box_constraint(box) = digit
```

Sudoku IS a pullback computation. So is scheduling (resource
constraints AND time constraints AND dependency constraints).
The planner recognizes pullback patterns and solves them by
intersecting the solution sets of individual constraints.

Domain-general: any problem with multiple independent constraints
that must all be satisfied simultaneously. Sudoku, scheduling,
configuration, resource allocation, type inference.

Implementation:
- `Pullback` type: two morphisms with common codomain
- Solve by computing solution set of each constraint, then intersect
- For large solution sets: propagate constraints iteratively (arc
  consistency, same as constraint propagation in CSP solvers)
- **Test:** 4x4 sudoku expressed as a pullback of row/col/box constraints

### D.5: Initial algebras (recursion and induction)
**Universal property:** the initial algebra of F is the smallest
fixed point of F. It gives recursion and induction for free.

```
  initial_algebra Nat where
    zero : Unit -> Nat
    succ : Nat -> Nat
    -- induction: to define h: Nat -> X, give h(zero) and h(succ(n)) = f(h(n))

  initial_algebra List(T) where
    nil  : Unit -> List(T)
    cons : T × List(T) -> List(T)
```

When the planner knows a type is an initial algebra, it can:
1. Do induction: prove a property for all elements by proving
   base case + inductive step
2. Define recursive functions by giving base case + step case
3. Recognize that Tier 5 problems (discover 2^N - 1) are initial
   algebra computations

Domain-general: any recursive structure. Numbers, lists, trees,
expressions, derivations, proofs. Also: time steps (each state is
succ of previous), hierarchical decomposition (a plan is a tree
of sub-plans).

Implementation:
- `InitialAlgebra` type: carrier type, constructors (zero, succ, etc.)
- `catamorphism(algebra, target_algebra)`: the unique map from initial
  to any other algebra. This IS recursion.
- Port from experiments/symbolic_ai_v2/ctkg/logic/initial_algebra.py
- **Test:** define Nat, use catamorphism to compute factorial

### D.6: Kan extensions (generalization / extrapolation)
**Universal property:** the left Kan extension of F along K is the
best approximation of F on a larger domain, using only data from
the smaller domain. "Extend what you know to where you haven't been."

```
  extend learned_rule along inclusion(TrainingDomain, FullDomain)
    -- extrapolate multiplication from single digits to all numbers
    -- extrapolate grammar rules from seen sentences to unseen ones
```

When the planner has learned how an operation works on a subset
(e.g., single-digit multiplication), the Kan extension tells it
how to extend that operation to the full domain (multi-digit
multiplication) in the most general way consistent with what it's
seen.

Left Kan = best approximation from below (interpolation, free extension).
Right Kan = best approximation from above (extrapolation, cofreeness).

Domain-general: any situation where you've learned rules on a
subset and need to apply them more broadly. This is the categorical
formulation of generalization.

Implementation:
- `KanExtension` type: base functor F, inclusion functor K, extension
- Compute via colimit formula: Lan_K(F)(c) = colim_{K(d)→c} F(d)
- For finite categories: compute by enumeration
- **Test:** learn successor on 0-9, extend to 0-99 via Kan extension

### D.7: Galois connections (abstraction hierarchies)
**Universal property:** a pair of monotone maps (α: Concrete → Abstract,
γ: Abstract → Concrete) such that α(c) ≤ a iff c ≤ γ(a). The most
general abstraction-concretization pair.

```
  galois abstraction(concrete_domain, abstract_domain) where
    abstract : ConcreteState -> AbstractState
    concretize : AbstractState -> ConcreteState
    -- abstract(concretize(a)) ≤ a  (abstracting the concretization
    --   gives you something at most as specific as what you started with)
```

The planner solves the problem in the abstract domain first (smaller
state space, faster), then refines to the concrete domain. If the
abstract plan is valid, the concrete plan is guaranteed to exist.

This is hierarchical planning done right. FCA (already implemented)
computes the concept lattice, which IS a Galois connection between
objects and their properties.

Domain-general: any problem where you can define a meaningful
abstraction. Blocks: ignore colors, plan by shape. Logistics:
ignore exact locations, plan by region. Chemistry: ignore exact
molecules, plan by reaction type.

Implementation:
- `GaloisConnection` type: concrete category, abstract category, α, γ
- Abstract planning: solve in abstract domain, then refine
- Port from FCA lattice (experiments/symbolic_ai_v2/ctkg/logic/fca.py)
- **Test:** blocks domain: abstract "all blocks are identical",
  plan in abstract, refine to concrete assignment

### D.8: Enriched predicates (continuous values)
**Universal property:** enrichment replaces the boolean hom-set
{true, false} with a richer structure V (real numbers, probabilities,
costs, distances). Everything else (composition, identities, functors)
still works — just over V instead of {true, false}.

```
  enriched pred distance : Point -> Point -> Float
  enriched pred probability : Event -> [0,1]
  enriched pred cost : Action -> Float
```

State includes both boolean atoms and continuous assignments.
Actions have numeric effects: `effect temperature(room) += 5.0`.
The heuristic works over enriched values (e.g., minimize total cost).

Domain-general: any domain with quantities. Physics (forces, energy),
geometry (distances, angles), ecology (populations), economics
(prices, quantities), chemistry (concentrations).

Implementation:
- `EnrichedAtom`: predicate + args + value (float)
- Extended state: frozenset[GroundAtom] + dict[EnrichedAtom, float]
- Numeric preconditions: `require temperature(room) > 100.0`
- Numeric effects: `effect energy(system) -= work(action)`
- **Test:** geometry domain: plan to minimize total distance

### D.9: Operads (multi-input composition)
**Universal property:** an operad generalizes a category by allowing
morphisms with multiple inputs. A morphism f: (A, B, C) → D takes
three inputs and produces one output. Composition substitutes
outputs into inputs.

```
  operad reaction where
    combine : (Reagent, Reagent) -> Product
    catalyze : (Reaction, Catalyst) -> Reaction
```

Standard morphisms are single-input (f: A → B). But chemical
reactions, circuit junctions, recipe steps, and logical inferences
often combine multiple inputs. Operads handle this without encoding
multi-input as "pair then apply" (which loses structure).

Domain-general: any domain where operations take multiple inputs.
Chemistry (reactions), circuits (junctions), cooking (combining
ingredients), manufacturing (assembly), logic (modus ponens takes
a rule + a fact).

Implementation:
- `OperadicAction`: multiple typed input slots + output
- Composition: output of one operation plugs into input slot of another
- Action grounding: bind each input slot independently
- **Test:** chemistry: combine(H2, O) produces H2O

### D.10: Probabilistic morphisms (Markov category)
**Universal property:** morphisms are stochastic kernels (probability
distributions over outputs given inputs). Composition = Bayesian
chaining. The category of stochastic matrices.

```
  stochastic action move(r : Robot, from : Room, to : Room)
    require at(r, from), connected(from, to)
    effect 0.9 : at(r, to), not(at(r, from))
    effect 0.1 : at(r, from)  -- move fails
```

d-separation for conditional independence. Intervention (do-calculus)
for causal reasoning. Entropy for information-theoretic planning
("which action gives me the most information?").

Domain-general: any domain with uncertainty. Robotics, ecology,
medicine, finance, game playing.

Implementation:
- `StochasticEffect`: list of (probability, effect_list) pairs
- Planner: MDP solver (value iteration or MCTS)
- d_separated(), conditional_entropy(), intervene() from ctkg/graph.py
- **Test:** logistics with unreliable trucks, ecology with stochastic
  population dynamics

---

## Phase E: Performance and infrastructure

### E.1: State representation optimization
Replace frozenset[GroundAtom] with bit-packed representation.
Assign each ground atom an integer ID. State = bitset. Faster hashing,
membership testing, and set operations.

### E.2: Successor generator
Instead of iterating all ground actions and checking preconditions,
build an index: for each predicate, which actions have it as a
precondition? Given a state, only consider actions whose preconditions
reference predicates that are currently true.

This is Fast Downward's successor generator. Reduces branching factor
dramatically for large domains.

### E.3: Symmetry breaking
Detect symmetric objects (blocks with identical properties) and prune
symmetric branches from the search. "pick(hand, a)" and "pick(hand, b)"
are equivalent if a and b are interchangeable.

### E.4: Iterative deepening / anytime search
Instead of plain A*, use iterative deepening A* (IDA*) for memory
efficiency, or weighted A* with decreasing weights for anytime behavior
(find a plan fast, then improve it).

---

## Milestone checklist

| Milestone | Phases | What it proves |
|-----------|--------|----------------|
| M1: PDDL parity | A.1-A.6 | CatPlan can express anything PDDL can |
| M2: Self-verifying | B.1-B.4 | CatPlan catches its own mistakes |
| M3: Competitive search | C.1-C.7 | CatPlan scales to non-toy problems |
| M4: Computation | D.1-D.5 | CatPlan computes answers, not just searches |
| M5: Generalization | D.6-D.7 | CatPlan extrapolates beyond training |
| M6: Rich domains | D.8-D.10 | CatPlan handles continuous, multi-input, stochastic problems |
| M7: Production-ready | E.1-E.4 | CatPlan is fast enough for real use |

## Implementation order

**Phase A (language completeness):**
A.1 → A.2 → A.5 → A.3 → A.4 → A.6

**Phase B (verification) — starts after A.3:**
B.1 → B.2 → B.3 → B.4

**Phase C (search) — starts after A.1, overlaps with B:**
C.7 → C.1 → C.2 → C.3 → C.4 → C.5 → C.6

**Phase D (categorical features) — staged by dependency:**

D.1 (composition) — after A complete. Foundation for everything else.
  ↓
D.2 (adjunctions) — after D.1. Needs composition to express F∘G = id.
D.3 (equalizers) — after A.3 (needs quantified preconditions).
D.4 (pullbacks) — after D.3. Pullback = pair of equalizers + product.
  ↓
D.5 (initial algebras) — after D.1. Needs composition for catamorphism.
D.6 (Kan extensions) — after D.5. Needs initial algebras for colimit formula.
D.7 (Galois connections) — after B.1 (needs invariant evaluation for abstraction validity).
  ↓
D.8 (enriched predicates) — independent of D.1-D.7. Needs A.6 (numeric goals).
D.9 (operads) — after D.1. Extends composition to multi-input.
D.10 (probabilistic) — after D.8. Enrichment over [0,1].

Recommended order within D:
D.1 → D.2 → D.5 → D.3 → D.4 → D.6 → D.7 → D.9 → D.8 → D.10

Rationale: D.1 (composition) and D.2 (adjunctions) are the highest
value — adjunctions turn search into computation. D.5 (initial algebras)
enables recursion. D.3→D.4 (equalizers→pullbacks) enable constraint
solving. D.6 (Kan extensions) enables generalization. D.7-D.10 extend
to richer domain types.

**Phase E (performance) — starts after C:**
E.1 → E.2 → E.3 → E.4

## Dependency graph

```
A.1 ──→ A.2 ──→ A.5 ──→ A.3 ──→ A.4 ──→ A.6
 │                        │
 │                        ↓
 │                       B.1 → B.2 → B.3 → B.4
 │                                     │
 ↓                                     ↓
C.7 → C.1 → C.2 → C.3 → C.4 → C.5 → C.6
 │
 ↓
D.1 (composition)
 ├──→ D.2 (adjunctions) ──→ solve-without-search
 ├──→ D.5 (initial algebras) ──→ D.6 (Kan extensions)
 ├──→ D.3 (equalizers) ──→ D.4 (pullbacks)
 ├──→ D.9 (operads)
 │
 │   B.1 ──→ D.7 (Galois connections)
 │
 │   A.6 ──→ D.8 (enriched predicates) ──→ D.10 (probabilistic)
 │
 ↓
E.1 → E.2 → E.3 → E.4
```

## Critical path to value

The shortest path to "CatPlan does something PDDL can't" is:

A.1 → A.2 → D.1 → D.2 (adjunctions)

With adjunctions, CatPlan solves equations by computation instead
of search. This is demonstrably beyond PDDL's capabilities.

The shortest path to "CatPlan generalizes" is:

A.1 → D.1 → D.5 → D.6 (Kan extensions)

With Kan extensions, CatPlan extrapolates learned rules to unseen
inputs. This is demonstrably beyond any existing planner.

The shortest path to "CatPlan handles hard constraints" is:

A.1 → A.3 → D.3 → D.4 (pullbacks)

With pullbacks, CatPlan solves sudoku-class constraint satisfaction
as a categorical computation, not as search.
