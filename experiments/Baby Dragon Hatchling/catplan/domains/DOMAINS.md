# CatPlan Domain Suite

## Design Principles

Each domain has 5 difficulty tiers:

- **Tier 1 (Trivial):** single-step, direct application of one rule
- **Tier 2 (Easy):** multi-step, chain of 2-5 rules
- **Tier 3 (Medium):** requires composition, discovering that A∘B solves C
- **Tier 4 (Hard):** requires transfer across sub-domains, or discovering
  a new operator from existing ones
- **Tier 5 (Research-grade):** structurally analogous to a real scientific
  discovery. Requires inventing a new concept, finding a hidden invariant,
  or unifying two apparently unrelated domains.

## Given vs Discovered domains

Some domains have hand-authored .catplan files — these test the PLANNER
(given known rules, can it find a plan?). Other domains have NO .catplan
file — these test DOMAIN LEARNING (given only a world simulator and
demonstrations, can it discover the rules?).

| Domain | .catplan file? | What we test |
|--------|---------------|--------------|
| Blocks | YES (given) | Planner: can it plan with known rules? |
| Logistics | YES (given) | Planner: can it scale and transfer? |
| Chemistry | NO (discover) | Domain learning: discover types, conservation laws |
| Circuits | NO (discover) | Domain learning: discover composition, duality |
| Ecology | NO (discover) | Domain learning: discover dynamics, feedback |
| Geometry | NO (discover) | Domain learning: discover continuous invariants |
| Language | NO (discover) | Domain learning: discover grammar from examples |
| Physics | NO (discover) | Domain learning: discover laws from observations |

For "discovered" domains, we provide:
1. A **world simulator** (Python class that executes actions and returns states)
2. **Demonstration trajectories** (sequences of (state, action, next_state))
3. **Test problems** (initial state + goal, no domain knowledge given)

The system must discover the domain (types, predicates, actions,
preconditions, effects, invariants) from the demonstrations alone,
then solve the test problems.

## Domains

### Domain 1: Blocks World (GIVEN)

The classic. Tests planner correctness and scaling.

**Types:** Block, Surface, Hand
**Predicates:** on, clear, holding, empty, on_table
**Actions:** pick, place_on_table, stack

| Tier | Problem | Tests |
|------|---------|-------|
| 1 | Move one block | Basic plan correctness |
| 2 | Towers of Hanoi (3 blocks) | Multi-step with constraints |
| 3 | Build specific structure from scramble | Operator composition |
| 4 | Hanoi N blocks (unseen N) | Generalization via NNO |
| 5 | Discover minimum-moves formula | Invariant from data |

Files: domain.catplan (given), tier1-5 problem files.

### Domain 2: Logistics (GIVEN)

Packages, trucks, cities, planes. Tests scaling and functor transfer.

**Types:** Package, Vehicle(Truck|Plane), Location(City|Airport)
**Predicates:** at, in_vehicle, connected
**Actions:** load, unload, drive, fly

| Tier | Problem | Tests |
|------|---------|-------|
| 1 | Direct delivery | Basic chain |
| 2 | Multi-hop delivery | Longer chains |
| 3 | Parallel deliveries, minimize steps | Shared resources |
| 4 | Transfer to unseen map | Functor |
| 5 | Discover that flying is faster | Novel action incorporation |

Files: domain.catplan (given), tier1-5 problem files.

### Domain 3: Chemistry (DISCOVER)

Atoms, bonds, molecules, reactions. Tests conservation law discovery.

**World simulator provides:** atoms with valences, bond/break actions,
observation of resulting molecules, charge conservation.

| Tier | Problem | Tests |
|------|---------|-------|
| 1 | Form water from atoms | Discover valence constraints |
| 2 | Balance a reaction | Discover conservation of atoms |
| 3 | Multi-step synthesis | Compose discovered operators |
| 4 | Predict products of novel reaction | NT: apply schema to new type |
| 5 | Discover periodic table structure | Hidden 2D organization from behavior |

Files: world simulator + demonstrations only. No .catplan file.

### Domain 4: Circuits (DISCOVER)

Components, connections, signals. Tests composition and duality discovery.

**World simulator provides:** circuit builder, voltage/current measurements
after connecting components.

| Tier | Problem | Tests |
|------|---------|-------|
| 1 | Light an LED | Discover connectivity requirements |
| 2 | Series vs parallel resistance | Discover composition rules |
| 3 | Design circuit to spec | Inverse problem from discovered rules |
| 4 | Find dual circuit | Discover series/parallel adjunction |
| 5 | Discover Kirchhoff's laws | Conservation laws from measurements |

Files: world simulator + demonstrations only. No .catplan file.

### Domain 5: Ecology (DISCOVER)

Species, habitats, populations. Tests probabilistic reasoning discovery.

**World simulator provides:** population counts per season, species
interactions (predator-prey), habitat carrying capacities.

| Tier | Problem | Tests |
|------|---------|-------|
| 1 | Predict next season population | Discover update rules |
| 2 | Restore ecosystem (reintroduction order) | Plan under feedback |
| 3 | Predict cascade from species removal | Indirect effects |
| 4 | Transfer to new ecosystem | Functor: same web, new species |
| 5 | Discover trophic cascade principle | Emergent principle from data |

Files: world simulator + demonstrations only. No .catplan file.

### Domain 6: Geometry (DISCOVER)

Points, lines, shapes, transformations. Tests continuous predicate discovery.

**World simulator provides:** coordinate geometry engine, measurement
functions (distance, angle, area), transformation application.

| Tier | Problem | Tests |
|------|---------|-------|
| 1 | Compute distance | Discover distance formula |
| 2 | Triangle congruence | Discover SSS/SAS/ASA rules |
| 3 | Construct shape to spec | Constraint satisfaction, continuous |
| 4 | Classify symmetry groups | Discover algebraic structure |
| 5 | Discover Pythagorean theorem | Universal law from measurements |

Files: world simulator + demonstrations only. No .catplan file.

### Domain 7: Language (DISCOVER)

Words, syntax, translation. Tests NT and functor discovery.

**World simulator provides:** sentences with meanings (semantic
representations), parallel corpora (same meaning in multiple languages).

| Tier | Problem | Tests |
|------|---------|-------|
| 1 | POS tagging | Discover word categories |
| 2 | Parsing | Discover recursive phrase structure |
| 3 | Translation | Discover functor between languages |
| 4 | Discover grammar rule in unknown language | NT from parallel data |
| 5 | Discover universal grammar | NT between language functors |

Files: world simulator + demonstrations only. No .catplan file.

### Domain 8: Physics (DISCOVER)

Particles, forces, motion, fields. The ultimate benchmark.

**World simulator provides:** particle positions/velocities over time,
force measurements, collision data. Newtonian mechanics under the hood.

| Tier | Problem | Tests |
|------|---------|-------|
| 1 | Predict position (constant velocity) | Discover v = dx/dt |
| 2 | Projectile trajectory | Discover F = ma under gravity |
| 3 | Discover momentum conservation | Invariant from collision data |
| 4 | Unify falling + orbits | Functor: same force law, different scales |
| 5 | Discover general relativity | Replace framework to explain anomaly |

Files: world simulator + demonstrations only. No .catplan file.

---

## Implementation order

1. **Blocks** (given domain) — validate planner works at all
2. **Logistics** (given domain) — validate scaling and transfer
3. **Chemistry** (discover) — first domain learning test
4. **Physics** (discover) — the ultimate benchmark, start Tier 1 early
5. **Circuits** (discover) — composition and duality
6. **Geometry** (discover) — continuous predicates
7. **Ecology** (discover) — probabilistic reasoning
8. **Language** (discover) — NT discovery, bridges to NLP
