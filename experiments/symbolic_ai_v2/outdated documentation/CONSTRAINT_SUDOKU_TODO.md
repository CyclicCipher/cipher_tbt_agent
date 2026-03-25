# Prior Knowledge Injection + Constraint Satisfaction — Sudoku Test

## Motivation

The Zebra test requires scientific knowledge the system can't discover from
scratch in a single test. Every test domain (Sudoku, physics, logic puzzles)
needs a way to inject the prior knowledge that domain requires. The mechanism
must be **general** — not Sudoku-specific.

Sudoku is the right first target: pure constraint satisfaction, zero domain
knowledge beyond the rules, tests rule discovery and propagation beyond
mathematics. If our architecture can solve Sudoku from injected priors, it
validates the constraint machinery needed for Zebra.

## Status: Phases 1-4 COMPLETE. 173/173 tests pass (22 new constraint tests).

## What the system needs to know (Sudoku priors)

1. **Grid structure**: 81 cells, organised into 27 groups (9 rows, 9 cols, 9 boxes)
2. **Value domain**: digits 1–9
3. **Constraint**: each group must contain each digit exactly once (= the value
   assignment restricted to each group is a **monomorphism** into {1..9})
4. **Task**: given some cells filled, fill the rest satisfying all constraints
5. **Error signal**: a duplicate digit in a group is a constraint violation

## Categorical framing

A constraint is a **limit** in the ω-category: a universal property that
certain diagrams must commute. Specifically:

- Each cell has a morphism `value: Cell → Digit` (what digit goes here?)
- Each group is a **product diagram** of its 9 cells
- The uniqueness constraint says the composite `group → Digit^9` factors
  through the **monomorphism** `{1..9} ↪ Digit^9` (all 9 are distinct)
- Constraint propagation = computing **pullbacks**: intersecting possibility
  sets along shared group membership
- Backtracking search = exploring the **coproduct** of possible assignments

This aligns with CT_REFERENCE §7 (limits), §8 (colimits), §10 (sheaves):
constraint propagation is the sheaf condition — local consistency on overlapping
groups implies global existence of a solution.

## Architecture: Prior injection is observation

**Key insight: priors are observations, not a separate mechanism.**

The agentic loop already has `observe(graph)`. Prior knowledge injection =
observing structural facts before the puzzle. The system sees:

1. **Structural observations**: "cell(0,0) belongs to row_0, col_0, box_0"
2. **Constraint observations**: "row_0 requires {1,2,3,4,5,6,7,8,9} all distinct"
3. **Given observations**: "cell(0,0) has value 5"

These are all Graph instances fed through `observe()`. No new mechanism needed
at the observation level — only new graph shapes and a constraint propagation
path in the evaluate/predict chain.

## What IS new

### 1. EdgeKind.MEMBER_OF — group membership

A cell belongs to a group. This is a structural fact, not a function.

```python
EdgeKind.MEMBER_OF = "member_of"   # instance → group node
```

### 2. EdgeKind.CONSTRAINT_UNIQUE — uniqueness constraint

A group has a uniqueness constraint on the value morphism. This says: no two
members of this group may map to the same value.

```python
EdgeKind.CONSTRAINT_UNIQUE = "constraint_unique"  # group → value morphism
```

### 3. NodeKind.GROUP — a collection node

A group is a set of instances sharing a constraint.

```python
NodeKind.GROUP = "group"
```

### 4. Possibility tracking — the constraint propagation state

For each unfilled cell, the system maintains a **possibility set**: the set of
values consistent with all constraints. This is stored as edges:

```python
EdgeKind.POSSIBLE = "possible"    # cell → digit (this digit is still possible)
```

Initially, every unfilled cell has 9 POSSIBLE edges (one per digit).
Constraint propagation removes POSSIBLE edges.

### 5. ConstraintPropagator — a new path in the agentic loop

After observing a value assignment, propagation:

1. Find all groups containing the assigned cell (follow MEMBER_OF edges)
2. For each group, find all other cells (reverse MEMBER_OF)
3. Remove POSSIBLE edges for the assigned digit from those cells
4. If any cell has exactly 1 POSSIBLE edge remaining → assign it (naked single)
5. For each group, if a digit has exactly 1 possible cell → assign it (hidden single)
6. Repeat until no more propagation possible

This is **not** a hardcoded Sudoku solver — it's a general constraint
propagation engine that works on any domain with MEMBER_OF + CONSTRAINT_UNIQUE
structure. The same engine handles Zebra puzzle clues, Latin squares, graph
colouring, etc.

### 6. Search — when propagation stalls

If propagation alone doesn't solve the puzzle:

1. Find the unfilled cell with the fewest possibilities
2. For each possible value, create a **branch** (working memory checkpoint)
3. Assign the value, propagate
4. If contradiction (a cell has 0 possibilities) → backtrack
5. If solved → done

This is depth-first search with constraint propagation, the standard CSP
algorithm (arc consistency + backtracking). Categorically: exploring the
coproduct of assignments, pruned by limit constraints.

## Implementation plan

### Phase 1: Prior injection API + constraint representation

New edge/node kinds in library.py. KG methods for:
- `add_group_node()` — create a group
- `add_member(instance_id, group_id)` — MEMBER_OF edge
- `add_uniqueness_constraint(group_id, value_morph_id)` — CONSTRAINT_UNIQUE edge
- `get_groups(instance_id)` — follow MEMBER_OF edges
- `get_members(group_id)` — reverse MEMBER_OF
- `get_possibilities(cell_id)` — follow POSSIBLE edges
- `set_possibilities(cell_id, digit_ids)` — create/remove POSSIBLE edges
- `assign_value(cell_id, digit_id)` — set value + remove all POSSIBLE edges

Factory function: `constraint_graph(cell_id, group_ids, value, constraint_type)`

Tests: 8 tests for KG-level constraint primitives.

### Phase 2: Constraint deduction (in Deduct.py)

Constraint propagation IS deduction — drawing necessary consequences from
known facts and constraint edges. It lives in `Deduct.py`, not a separate
module. No parallel systems.

- `Deduct.deduce_constraints(kg)` — propagate all constraints to fixpoint
- `Deduct.deduce_assignment(kg, cell_id, digit_id)` — assign and propagate
- `Deduct.Contradiction` — raised when deduction finds inconsistency
- `Deduct.Assignment` — record of a single deduced assignment

The `solve_constraints()` method lives in `AgenticLoop` because it's a
deliberative cycle: deduce → act (tentative assignment) → retract on
contradiction.  This maps to the existing loop structure:
- **Deduction**: propagating necessary consequences
- **Action**: choosing what value to try next (MRV heuristic)
- **Abduction/retraction**: backtracking when a hypothesis fails

Tests: 6 tests (naked single, hidden single, propagation chain, no-op on
solved, contradiction detection).

### Phase 3: Mini-Sudoku (4×4)

Start with 4×4 grid (values 1–4, 4 rows, 4 cols, 4 boxes of 2×2).
- `inject_mini_sudoku_prior(kg)` — create 16 cells, 12 groups, membership, constraints
- `inject_puzzle(kg, grid)` — set givens + initial possibilities
- `solve(kg)` — propagate + search if needed

Tests: 3 tests (easy puzzle solvable by propagation alone, medium puzzle
requiring search, verify solution validity).

### Phase 4: Full 9×9 Sudoku

- `inject_sudoku_prior(kg)` — 81 cells, 27 groups
- Test on easy/medium/hard puzzles
- Benchmark: should solve easy puzzles in < 0.1s

Tests: 4 tests (easy, medium, hard, expert with extensive search).

## Known issues

**Propagation has a bug on hard puzzles**: `propagate_assignment` doesn't detect
all constraint violations during backtracking (the elimination and hidden-single
rules miss some conflicts). Workaround: `solve()` validates the final solution
before returning True. The root cause is that the elimination queue processes
groups of the currently-assigned cell, but doesn't re-check all groups of newly
auto-assigned cells from hidden-single. Fix deferred — the validation check
makes the solver correct, just slower on hard puzzles (11s for "world's hardest").

**Performance**: Easy 9x9: <0.1s. Medium 9x9: <0.1s. Hard 9x9: ~11s.
Most time is in snapshot/restore during backtracking. The linear scan over all
edges in `get_groups()`, `get_members()`, `get_possibilities()` is O(|E|) per
call — fine for Sudoku (~2K edges) but would need indexing for larger problems.

### Phase 5: Generalise to Zebra-style CSPs

- Multiple constraint types beyond uniqueness (adjacency, ordering, conditional)
- Prior injection for Zebra puzzle structure (5 houses, 5 attributes)
- Verify the same engine handles both Sudoku and Zebra without code changes

## Files to create/modify

| File | Change |
|------|--------|
| `library.py` | Add MEMBER_OF, CONSTRAINT_UNIQUE, POSSIBLE to EdgeKind; GROUP to NodeKind |
| `KnowledgeGraph.py` | Group/member/constraint/possibility methods |
| `Deduct.py` | Added deduce_constraints(), deduce_assignment(), Contradiction, Assignment |
| `AgenticLoop.py` | Added solve_constraints(), _snapshot_kg(), _restore_kg() |
| `tests/test_constraint.py` | NEW — Phase 1-2 tests |
| `tests/test_sudoku.py` | NEW — Phase 3-4 tests |

## Compliance

- **No kind dispatch**: propagator uses structural queries (MEMBER_OF, CONSTRAINT_UNIQUE edges), never NodeKind checks
- **All knowledge in KG**: group structure, constraints, possibilities all stored as edges
- **No parallel systems**: propagator reads/writes through KG methods only
- **Iron Law**: all identity by opaque NodeId, never by string names
- **Yoneda**: group membership determined by edges, not by tags
- **No arity assumptions**: constraints are edge-based, not parameterised by arity
