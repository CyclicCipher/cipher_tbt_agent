# Sheaf Restriction: Unifying Constraints, Prediction, and Context

## The Insight

Constraint propagation, morphism-based prediction, and context-dependent
disambiguation (bank vs. bank) are **the same operation**: narrowing a
possibility set given context. They're all restriction maps in the sheaf.

| Domain | Context | Possibilities | Restriction |
|--------|---------|---------------|-------------|
| Sudoku | row_0 has a 5 | cell (0,3) values | eliminate 5 from {1..9} |
| Language | financial discourse | "bank" meanings | eliminate river_margin |
| Physics | low-velocity regime | applicable laws | eliminate relativistic morphisms |
| Zebra | "Brit in red house" | house-colour pairs | eliminate non-red for Brit |

The KG already has two mechanisms doing this:
1. **validity_domain** on morphisms — filters which morphisms fire based on
   WorkingMemory context (morphism-level restriction)
2. **POSSIBLE edges** on cells — filters which values are admissible for
   a constraint cell (instance-level restriction)

These are the same sheaf restriction map at different levels of the ω-category.

## Status: ALL 6 PHASES COMPLETE. 198/198 total tests pass.

## What's done

### Phase 1: Constraint deduction as deduction (DONE)

- `Deduct.deduce_constraints()` — propagate all necessary consequences
- `Deduct.deduce_assignment()` — assign a value and deduce consequences
- `AgenticLoop.solve_constraints()` — deliberative deduce→act→retract cycle
- `predict()` Path B — POSSIBLE edges become the prediction distribution
  when a query references a constraint cell (node_type="cell")
- Path A × Path B intersection — morphism predictions filtered by remaining
  possibilities (sheaf gluing condition)
- 25 tests: 8 primitives, 6 propagation, 5 mini-Sudoku, 3 full 9×9, 3 prediction

### What the tests prove

- `test_prediction_returns_possibilities`: Before deduction, 4 outcomes.
  After assigning a peer and deducing, fewer outcomes. Deduction narrows
  prediction.
- `test_prediction_point_mass_after_assignment`: Assigned cell → prediction
  collapses to a single outcome.
- `test_prediction_uniform_over_possibilities`: N remaining possibilities →
  uniform distribution (log(1/N) each).

## What's next

### Phase 2: Graded context affinity for morphism disambiguation — DONE

`CONTEXT_AFFINITY` edges: morphism → context node with graded weight.
`context_affinity_score(morph_id, active_context)` returns [0, 1]:
  1.0 = all preferred contexts active (or no preferences = context-free)
  0.0 = none of the preferred contexts active

`predict()` combines posterior_weight + log(affinity) for effective weight.
Morphisms with affinity 0 are skipped. Context-free morphisms (no affinity
edges) always fire with score 1.0. Existing validity_domain (hard filter)
is preserved — affinity is a soft layer on top.

This solves bank vs. bank: two morphisms with different context affinities.
Financial context active → financial morphism dominates. River context →
river morphism dominates. Both active → both contribute proportionally.

Tests (test_context_affinity.py, 10 tests):
- 4 KG-level: no affinities, full match, no match, partial match
- 6 prediction: financial context, river context, ambiguous, no context,
  context-free morph, validity_domain backward compat

### Phase 3: Constraint types beyond CONSTRAINT_UNIQUE — DONE

Three new constraint types implemented:

- **CONSTRAINT_EQUALITY**: "if cell_A = val_V then cell_B = val_W" (bidirectional).
  Stored as edges with arg_position=trigger_val, via_law=forced_val.
  Propagates in both `deduce_assignment()` and `deduce_constraints()`.
  Used for same-position clues ("Brit = Red") and directed adjacency
  ("Green immediately left of White").

- **CONSTRAINT_NEIGHBOR**: disjunctive "if cell_A = val_V then val_W must appear
  in at least one of [cell_B1, cell_B2, ...]".  Stored as multiple edges.
  Propagates eagerly when only one viable candidate remains; validated
  at solution check in `solve_constraints()`.
  Used for neighbor clues ("Blend smoker has a neighbor who keeps cats").

- **SAME_POSITION**: bidirectional link between cells sharing a position/slot.
  Used for permutation puzzles (Zebra).

KG methods: `add_equality_constraint()`, `get_equality_constraints()`,
`add_neighbor_constraint()`, `get_neighbor_constraints()`,
`link_same_position()`, `get_same_position_peers()`.

Tests (test_zebra.py, 4 tests):
- `test_direct_assignment_propagates`: uniqueness elimination in permutation grid
- `test_equality_constraint_fires`: Brit + Red co-assignment forced
- `test_3x3_puzzle_solve`: 3-house, 3-attribute puzzle (3 categories, 5 clues)
- `test_zebra_puzzle`: **full 5-house Zebra puzzle** — 5 categories,
  15 clues (8 same-position, 4 neighbor, 1 directed adjacency, 2 direct position).
  Solves in <1 second. Unique solution verified for all 25 cells.

### Phase 4: Prior injection mechanism — DONE

`PriorInjector` (`ctkg/logic/PriorInjector.py`): declarative builder that
translates high-level domain descriptions into KG edges.

API:
- `add_sort(name, values)` — define a value domain
- `add_instance(label, sort_name)` — create a cell linked to a sort's possibilities
- `add_group(label, members, constraint)` — create a uniqueness group
- `assign(instance, sort, value)` — given value (propagates constraints)
- `add_clue_equality(...)` — if A=V then B=W
- `add_clue_neighbor(...)` — disjunctive neighbor constraint
- `build_permutation_puzzle(n, categories)` — full Zebra-style setup
- `add_clue_same_position(...)` / `add_clue_neighbor_puzzle(...)` — convenience

Tests (in test_zebra.py):
- `test_mini_sudoku_via_injector`: 4×4 Sudoku built entirely through PriorInjector
- `test_zebra_via_injector`: full 5-house Zebra puzzle through PriorInjector

The mechanism is general: any domain that needs sorts, instances, groups,
and constraints can use PriorInjector. The `observe()` integration (routing
observations through the agentic loop) is deferred — direct KG manipulation
via PriorInjector is cleaner for structural priors that are known before
the reasoning loop starts.

### Phase 5: Working memory integration — DONE

`deduce_assignment(kg, cell, digit, wm=wm)` and `deduce_constraints(kg, wm=wm)`
accept an optional WorkingMemory parameter. Newly assigned digit IDs are
added to `wm.active_context_set` after deduction completes.

This connects constraint solving to morphism-level context filtering:
assigning digit_0 in a Sudoku cell puts digit_0 in WM, which enables
morphisms with `validity_domain = {digit_0}` to fire.

Tests (test_context_affinity.py, 3 tests):
- `test_deduce_assignment_updates_wm`: assigned digits enter WM
- `test_deduce_assignment_without_wm_unchanged`: backward compat (no crash)
- `test_constraint_deduction_enables_context_morph`: assignment enables
  a validity_domain-gated morphism via WM

### Phase 6: Contradiction as structural surprise — DONE

`AgenticLoop.observe_constraint_assignment(cell, digit)` attempts an
assignment and measures structural surprise:
- Success → `(True, 0.0)`
- Contradiction → `(False, inf)` + records `_last_contradiction`

Infinite surprise always passes the WM gate. The contradiction object is
available for downstream abduction (retract the bad hypothesis).

Tests (test_context_affinity.py, 3 tests):
- `test_contradiction_returns_inf_surprise`: conflicting assignment → inf
- `test_valid_assignment_returns_zero_surprise`: clean assignment → 0
- `test_contradiction_records_last_contradiction`: exception stored

## Compliance

- No parallel systems: constraint deduction lives in Deduct.py
- All knowledge in KG: groups, membership, constraints, possibilities are edges
- No kind dispatch: propagator uses structural queries
- Iron Law: all identity by opaque NodeId
- Yoneda: group membership determined by edges, not tags
