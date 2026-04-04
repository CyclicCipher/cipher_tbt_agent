# Neurosymbolic System Roadmap

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Layer 3: BDH Action Policies                    │
│  plan step → concrete action                     │
│  small per-operator neural policies              │
├──────────────────────────────────────────────────┤
│  Layer 2: CatPlan (Categorical Planner)          │
│  typed symbolic state → verified plan            │
│  morphism composition in a learned category      │
├──────────────────────────────────────────────────┤
│  Layer 1: BDH Perception                         │
│  raw input → typed symbolic state                │
│  grounding predicates from sensory data          │
└──────────────────────────────────────────────────┘
```

Each layer has its own roadmap. They can be developed in parallel but
integrate at defined interface points.

---

## Layer 1: BDH Perception (raw input → symbolic state)

**Goal:** Given raw input (text, pixels, sensor data), produce a typed
symbolic state: a set of grounded predicates with values.

### Phase L1.1: Baseline BDH on structured tasks
- Run full benchmark suite (copy, reverse, sort, arithmetic, successor,
  sudoku 4x4, calendar, sudoku 9x9) on all four model configs
- Establish what BDH-GPU can learn at <=1.4M params
- Identify failure modes and capability boundaries
- **Deliverable:** benchmark results table, documented in results.md

### Phase L1.2: BDH as predicate grounder
- Train BDH to output predicate truth values instead of next token
- Input: raw text description of a state (e.g., "block A is on block B")
- Output: predicate vector (on_A_B=1, clear_A=1, clear_B=0, ...)
- Start with blocks-world (known ground truth from PDDL)
- **Deliverable:** predicate grounding accuracy on blocks-world

### Phase L1.3: BDH as state change detector
- Given two consecutive observations, output which predicates changed
- This is the perception half of operator extraction — the system needs
  to see what changed when an action happened
- **Deliverable:** state-diff accuracy on demonstration trajectories

### Phase L1.4: Multi-modal grounding (stretch)
- Extend from text to images (object detection + predicate grounding)
- BDH processes image features → predicate values
- **Deliverable:** visual predicate grounding on a simple manipulation domain

---

## Layer 2: CatPlan (symbolic state → plan)

**Goal:** A categorical planning language that is typed, compositional,
transferable, and learnable from data. Successor to PDDL.

### Design philosophy

Follow Fast Downward's proven three-phase architecture (translate →
compile → search) but with categorical structures instead of propositional
logic. Borrow every algorithm-level idea that isn't PDDL-specific.

### Phase L2.1: CatPlan language specification
- Define the formal syntax and semantics
- Core constructs:
  - **category** — a named planning domain
  - **type** — an object of the category (Block, Surface, etc.)
  - **morphism** — a predicate or relation (on: Block × Surface → Prop)
  - **action** — a typed state transition with pre/postconditions
  - **functor** — a structure-preserving map between domains
  - **natural transformation** — an operator schema that generalizes
    across types
- Actions have typed pre/post conditions expressed as pullbacks
- Composition is the primitive operation
- Key advantage over PDDL: predicates are typed and multi-valued from
  the start. PDDL uses booleans, then Fast Downward translates them
  into multi-valued variables at search time. CatPlan skips that step
  because a morphism `position: Block → Surface` IS a multi-valued
  variable by definition.
- Write a parser (reuse the .ctkg DSL parser pattern from experiments/ctkg/)
- **Deliverable:** CatPlan spec document + parser that reads .catplan files

### Phase L2.2: Internal representation + forward planner
- **Translation** (borrowing from Fast Downward): CatPlan domain → internal
  representation. Since CatPlan types are already multi-valued, this is
  simpler than PDDL translation. Extract:
  - Variable set (one per typed morphism to Prop or typed value morphism)
  - Action schemas with typed parameter bindings
  - Initial state assignment
  - Goal condition
- **Knowledge compilation** (borrowing from Fast Downward): extract a
  **causal graph** from the category structure. This is cheaper than
  PDDL's causal graph extraction because the category already encodes
  dependencies — morphism structure IS the causal graph.
  - SCC decomposition of the causal graph = tightly coupled variable groups
  - DAG of SCCs = causal ordering
  - These come from Krohn-Rhodes (already implemented in algebra.py)
- **Search:** A* with a basic heuristic. Type checking prunes invalid
  actions before expanding them — smaller branching factor than PDDL.
- Validate on blocks-world and Towers of Hanoi (known PDDL benchmarks)
- **Deliverable:** planner that solves small problems correctly

### Phase L2.3: Sheaf-based plan verification and heuristic
- Given a candidate state, compute sheaf consistency energy
- Energy = 0 means valid state (all constraints satisfied).
  Energy > 0 means violation — localize which predicate/action fails.
- Port sheaf.py from symbolic_ai_v2 into the CatPlan framework
- **Use sheaf energy as a search heuristic:** actions that reduce
  energy are preferred. This gives a universal, domain-independent
  heuristic (unlike PDDL heuristics which must be re-derived per domain).
- **Deliverable:** plan verifier + sheaf-energy heuristic for A* search

### Phase L2.4: Advanced heuristics (borrow from PDDL ecosystem)
- Implement additional heuristics borrowing from Fast Downward / LAMA:
  - **Causal graph heuristic:** estimate goal distance by solving
    subproblems in the causal graph independently. Our algebraic
    skeleton (Krohn-Rhodes SCC decomposition) gives us this graph
    for free.
  - **Landmark analysis:** certain predicates MUST become true in any
    valid plan. Landmarks = terminal objects in the relevant subcategory.
    Count of unachieved landmarks = admissible heuristic.
  - **Preferred operators:** actions the heuristic considers promising
    get search priority. Use morphisms that reduce sheaf energy.
  - **Deferred evaluation:** don't compute expensive heuristics until
    a state is about to be expanded. Copy directly from Fast Downward.
  - **Multi-heuristic search:** combine sheaf energy + causal graph +
    landmark count in a single best-first search (as LAMA combines
    FF heuristic + landmark heuristic).
- Benchmark against Pyperplan (Python PDDL planner) on standard
  planning domains. Fair comparison: both Python, similar code maturity.
  Not against Fast Downward (C++, 25 years of optimization) yet.
- **Deliverable:** planner competitive with Pyperplan on standard domains

### Phase L2.5: Domain learning from demonstrations
- Given trajectories (sequences of states), learn the CatPlan domain:
  - **FCA** discovers types (concept lattice from predicate co-occurrence).
    Already implemented in fca.py.
  - **Krohn-Rhodes** discovers primitive operators (irreducible components
    of the state transition monoid). Already implemented in algebra.py.
  - **Natural transformation discovery** generalizes operators into schemas.
    Already implemented in Consolidation.py.
  - **Initial algebra / NNO discovery** finds recursive structure.
    Already implemented in initial_algebra.py.
- This is the key differentiator: automatic domain authoring.
- **Deliverable:** system that watches demonstrations and outputs a
  .catplan domain file

### Phase L2.6: Functor-based transfer
- Given a learned domain A and a new domain B with partial demonstrations,
  discover a functor F: A → B that maps types and actions
- The functor lets you reuse all of A's planning knowledge in B,
  including heuristics — a causal graph heuristic learned in domain A
  transfers to domain B via the functor
- Test: learn blocks-world in one setting, transfer to a new setting
  with different object names/types
- **Deliverable:** transfer learning via functor discovery

### Phase L2.7: Probabilistic planning (stretch)
- Extend CatPlan with Markov category morphisms
- Actions have probabilistic effects (stochastic kernels)
- Planner computes expected-value-optimal plans
- Reuse d_separated(), conditional_entropy() from experiments/ctkg/
- **Deliverable:** probabilistic CatPlan planning on stochastic domains

---

## Layer 3: BDH Action Policies (plan step → concrete action)

**Goal:** Given a symbolic plan step (e.g., "pick(blockA)"), execute it
in the environment. Small per-operator neural policies.

### Phase L3.1: Text-output action policies
- For text domains (sudoku, calendar), the "action" is generating the
  output string
- Train one small BDH policy per operator type
- Input: symbolic action + current state predicates
- Output: text tokens for the result
- **Deliverable:** per-operator BDH policies for sudoku and calendar

### Phase L3.2: Operator-conditioned policy
- Instead of one model per operator, train a single BDH model conditioned
  on the operator type
- The operator type is prepended as a special token
- Test whether a shared model matches per-operator accuracy
- **Deliverable:** single BDH model that executes any operator

### Phase L3.3: Termination conditions
- Each policy must know when it's done
- Learn a termination classifier: given current state, has the operator's
  postcondition been achieved?
- Sheaf energy as termination signal: energy drops to 0 when the
  postcondition is satisfied
- **Deliverable:** reliable operator termination detection

### Phase L3.4: Physical action policies (stretch)
- For robotics domains, output motor commands instead of text
- Diffusion-based policies (as in the neuro-symbolic paper)
- BDH as the backbone instead of a diffusion model
- **Deliverable:** proof-of-concept on a simulated manipulation task

---

## Integration Milestones

### Milestone 1: BDH benchmarks complete (Layer 1 only)
- Phase L1.1
- We know what BDH can do at small scale
- Timeline: immediate

### Milestone 2: CatPlan solves toy problems (Layer 2 only)
- Phases L2.1, L2.2, L2.3
- We have a working planner with a formal language
- Hand-authored domains, verified by sheaf energy
- CatPlan's typed multi-valued variables skip PDDL's translation step
- This is the first demonstrable product component

### Milestone 3: End-to-end on text domains (all layers)
- Phases L1.2, L2.4, L3.1
- BDH reads a problem → CatPlan plans → BDH executes
- Test on sudoku: BDH parses the grid → CatPlan plans the fill order
  → BDH outputs each digit
- Test on calendar: BDH parses the date → CatPlan plans the arithmetic
  → BDH outputs the result

### Milestone 4: Automatic domain learning (the moat)
- Phase L2.5
- The system watches demonstrations and writes its own CatPlan domain
- This is what nobody else has
- All the structure discovery code already exists (FCA, Krohn-Rhodes,
  initial algebra, sheaf, NT discovery). Needs reframing from "CTKG
  cognitive architecture" to "CatPlan domain learner."
- Demo: show it 50 blocks-world demonstrations, it discovers the
  domain, then solves novel instances

### Milestone 5: Transfer (the multiplier)
- Phase L2.6
- Learn one domain, transfer to another via functor
- Heuristics transfer too — causal graph structure maps through functor
- Demo: learn stacking in context A, transfer to context B with zero
  additional demonstrations

### Milestone 6: First paying customer
- Package Milestones 3-5 into a product for one vertical
- The customer demonstrates their workflow
- The system extracts the formal structure
- The system plans and executes novel instances
- The customer pays because it works, it's verifiable, and it runs
  on commodity hardware

---

## What we build vs what exists

| Component | Status | Source |
|-----------|--------|--------|
| BDH model | Use upstream | pathwaycom/bdh (MIT) |
| CatPlan language | Build | Novel — no categorical planning language exists |
| CatPlan parser | Build | Reuse .ctkg DSL parser pattern |
| Translation layer | Build (simplified) | CatPlan types are already multi-valued; skip PDDL→SAS+ step |
| Causal graph extraction | Adapt from existing | algebra.py (Krohn-Rhodes) gives SCC decomposition |
| A* search | Build | Standard algorithm |
| Sheaf energy heuristic | Adapt from existing | sheaf.py → universal domain-independent heuristic |
| Causal graph heuristic | Build, borrow ideas | Fast Downward's CG heuristic adapted to categorical structure |
| Landmark heuristic | Build, borrow ideas | Fast Downward/LAMA landmark analysis → terminal objects |
| Preferred operators | Build, copy idea | Fast Downward's preferred operators → morphisms reducing energy |
| Deferred evaluation | Copy directly | Fast Downward technique, algorithm-level, not PDDL-specific |
| Multi-heuristic search | Build, copy idea | LAMA's multi-heuristic best-first search |
| FCA (type discovery) | Already built | experiments/symbolic_ai_v2/ctkg/logic/fca.py |
| Krohn-Rhodes (operator discovery) | Already built | experiments/symbolic_ai_v2/ctkg/logic/algebra.py |
| Initial algebra (recursion) | Already built | experiments/symbolic_ai_v2/ctkg/logic/initial_algebra.py |
| Sheaf consistency | Already built | experiments/symbolic_ai_v2/ctkg/logic/sheaf.py |
| NT discovery (schemas) | Already built | experiments/symbolic_ai_v2/ctkg/logic/Consolidation.py |
| Object detection | Use existing | YOLOv8 or similar |

**Key reuse:** over half the stack exists in symbolic_ai_v2. The planner
search algorithms borrow ideas from 25 years of PDDL research (Fast
Downward, LAMA, Pyperplan) but adapted to categorical structure.

---

## Structural advantages of CatPlan over PDDL

| Aspect | PDDL | CatPlan |
|--------|------|---------|
| Variables | Boolean predicates (then translated to multi-valued by Fast Downward) | Multi-valued from the start (typed morphisms). No translation step. |
| Type safety | Weak (PDDL 2.1+ has types but no type checking on predicates) | Strong (morphism composition is type-checked; invalid actions pruned before search) |
| Causal graph | Computed at search time from grounded actions | Falls out of category structure + Krohn-Rhodes decomposition |
| Heuristic | Domain-dependent (different heuristics work on different domains) | Sheaf energy is universal and domain-independent |
| Composition | Not primitive (HTN bolts it on as separate formalism) | Primitive operation — actions compose via morphism composition |
| Transfer | Impossible (each domain is independent) | Functors map types + actions between domains; heuristics transfer too |
| Learning | Static (human writes domain file) | FCA + Krohn-Rhodes + NTs discover domain from demonstrations |
| Uncertainty | Awkward extensions (PPDDL) | Markov category morphisms (built into the formalism) |
| Consistency | Silent failure on contradictory domains | Sheaf Laplacian gives quantitative consistency score |

---

## Key references

### PDDL ecosystem (borrow from)
- Helmert 2006. "The Fast Downward Planning System." JAIR 26. Three-phase
  architecture, causal graph heuristic, preferred operators, deferred
  evaluation. (arXiv:1109.6051)
- Richter & Westphal 2010. "The LAMA Planner." Multi-heuristic search,
  landmark heuristic + FF heuristic combination. (arXiv:1401.3839)
- Pyperplan: simple Python PDDL planner. Fair comparison target.

### Neurosymbolic (our approach extends)
- "The Price Is Not Right" (arXiv:2602.19260). PDDL + small neural
  policies beats VLAs: 77x less energy, 3x better accuracy.

### Category theory (our tools)
- FCA: bottom-up lattice discovery (fca.py)
- Krohn-Rhodes: algebraic skeleton, SCC decomposition (algebra.py)
- Initial algebra: NNO/recursion discovery (initial_algebra.py)
- Sheaf Laplacian: consistency measure (sheaf.py)
- Natural transformations: operator schema discovery (Consolidation.py)
- Markov categories: probabilistic reasoning (graph.py in experiments/ctkg/)

### BDH (neural component)
- Kosowski et al. "The Dragon Hatchling." (arXiv:2509.26507)
  BDH-GPU: sparse monosemantic neurons, linear attention, Hebbian gating.

---

## Open questions

1. Can a categorical planner match Pyperplan's speed on standard
   benchmarks? Type checking prunes the search space, but category
   theory adds overhead per step. Need Phase L2.4 to find out.

2. Does BDH's sparse monosemantic structure actually help with predicate
   grounding, or is it just a different flavor of neural network for
   this purpose? Need Phase L1.2 to find out.

3. How many demonstrations does FCA + Krohn-Rhodes need to discover a
   correct domain? The neuro-symbolic paper used 50. Can we do it in 10?
   Need Phase L2.5 to find out.

4. Is functor-based transfer actually better than just re-learning?
   If domain learning only needs 10 demonstrations, transfer might not
   be worth the complexity. Need Phase L2.6 to find out.

5. Is sheaf energy a good enough heuristic by itself, or will we always
   need domain-specific heuristics too? The claim is it's universal.
   Need Phase L2.4 benchmarks to verify.

6. What vertical market should we target first? Needs customer discovery,
   not more engineering.
