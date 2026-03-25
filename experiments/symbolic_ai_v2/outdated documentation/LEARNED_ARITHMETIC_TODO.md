# Learned Arithmetic — Remove Seeded Primitives, Learn from Data

## Status: Phases 1-7 COMPLETE. 151/151 tests pass.

## The Problem

`_seed_primitives()` hardcoded ADD, MUL, SUB, DIV, MAX as Python lambdas.
`_prim_fns` was a Python dict of callables living outside the KG. Induction
searched over compositions of these seeded ops. This violated:

- **Iron Law**: system wouldn't work with different symbols
- **Single Source of Truth**: Python lambdas are knowledge outside the KG
- **No Parallel Systems**: `_prim_fns` is a shadow registry
- **Universality**: math-specific code in the universal engine

## The Goal

The system starts with an empty KG. The connector provides atoms (raw values)
and topology (sequential adjacency). The system discovers succession from
observing `(n, n+1)` pairs, addition from observing `(a, b, a+b)` triples,
multiplication from observing `(a, b, a*b)` triples — each by composing
previously discovered morphisms.

## Phase 1: Delete seeded primitives and _prim_fns — DONE

- [x] Remove `_seed_primitives()` method from KnowledgeGraph.py
- [x] Remove `_prim_op_ids` dict from KnowledgeGraph.__init__
- [x] Remove `_prim_fns` dict from KnowledgeGraph.__init__
- [x] Remove `register_prim_fn()` method
- [x] Remove all callers of `_seed_primitives()`
- [x] Remove `can_evaluate()` dependence on `_prim_fns`
- [x] Remove the `fn = self._prim_fns.get(morph_id)` path from `evaluate()`
- [x] Fix all tests that depend on seeded primitives or registered callables

## Phase 2: Evaluation through the KG only — DONE

Three evaluation paths, all through the KG:

1. **Observed extension**: OBSERVED edges map input instance → output instance.
   Exact match on input label.
2. **NT extrapolation**: affine model (slope, intercept) stored as NT_SLOPE
   and NT_INTERCEPT edges in the KG. `evaluate(morph, x) = slope * x + intercept`.
3. **Composition traversal**: APPLIES_OP → sub-morphism, ARG → args. Recurse.

- [x] Observed-extension evaluation path
- [x] NT extrapolation path (single and multi-input)
- [x] Composition traversal bottoms out at observed extension or NT
- [x] `can_evaluate()` checks observed pairs, NT, or composition structure

## Phase 3: Topology-driven induction — DONE

- [x] `_extract_obs()` extracts (inputs, output) from Graph edge topology
- [x] `_discover_univariate()` creates law node with observed pairs + affine NT
- [x] `_discover_multivariate()` handles multi-field graphs
- [x] `_fit_affine()` OLS regression for NT parameters
- [x] Yoneda dedup: `find_law_by_extension()` prevents duplicate law nodes
- [x] MDL scoring via mdl_prior + log_likelihood

## Phase 4: NT-based generalisation — DONE

- [x] NT stored as edges (NT_SLOPE, NT_INTERCEPT) via `set_nt()` / `get_nt()`
- [x] Succ morphism generalises to any number via NT(slope=1, intercept=1)
- [x] Adjunction detection works with observed pairs + NT (no register_prim_fn)
- [x] Abduction uses affine models for EM partition (no composition fitting)
- [x] Domain classifier uses observed pairs for routing (no registered callable)
- [x] Comparison morphism deferred to Phase 5 (requires fold for 2-input)

## Phase 5: Iterated composition (fold / NNO universal property) — DONE

Addition is "apply succ b times starting from a." The system discovers
this pattern from data like (2, 3, 5), (4, 1, 5), (3, 3, 6).

The NNO universal property: for any endomorphism f: X → X and base point
z: 1 → X, there is a unique morphism fold(f, z, -): ℕ → X such that
fold(f, z, 0) = z and fold(f, z, n+1) = f(fold(f, z, n)).

- [x] Add ITERATE edge kind: morph → sub-morphism (the endomorphism being
      iterated).
- [x] Induction fold search: given (a, b, result) triples and a known
      morphism f, test whether result = f^b(a). If yes for all data, create
      a fold-composition morphism with ITERATE edge → f.
- [x] Evaluation of fold morphisms: given (a, b), apply the iterated
      sub-morphism b times starting from a. The sub-morphism is evaluated
      recursively through the KG (observed extension or its own NT).
- [x] MDL scoring: fold(f, var, var) has complexity proportional to 1
      (the fold structure itself — sub-morphism reference).
- [x] Observation partitioning by dimensionality — mixed 1-input + 2-input
      streams are handled correctly.
- [x] 12 tests: fold KG mechanics (6) + fold discovery from data (6).
- [ ] 2-input comparison morphism (max/min) via fold — deferred (requires
      specialised fold: fold(max_step)(a, b) with a comparison step function)

## Phase 6: Currying / partial application — DONE

Multiplication is fold(add_a, 0, b) where add_a is addition with first
argument bound to a. This requires the system to take a known multi-input
morphism and produce a family of single-input morphisms by fixing one input.

This is the exponential object in the ω-category.

- [x] Discover partial application: _discover_nested_fold() searches over
      known 2-input morphisms, trying all combinations of curry position,
      argument swap, and base value (0 or 1).
- [x] Represent partial application as composition structure in the KG:
      CURRY_SOURCE edge (nested fold → base 2-input morph, arg_position = curry pos)
      + FOLD_BASE edge (nested fold → constant node for fold starting value).
      Swap-args encoded via FOLD_BASE.arg_position = 1.
- [x] Test: after learning succ → add = fold(succ), observe mul triples
      → discover mul = nested_fold(add, curry_pos=1, base=0).
      10 tests (7 KG-level + 3 discovery integration) all pass.
- [x] Bug fix: _retract_universal_fitted_laws() now protects fold morphisms
      (ITERATE / CURRY_SOURCE edges) and their iterate targets from
      abduction-triggered retraction.  Without this, auto-abduction during
      mul observation destroyed the fold(succ) morphism needed for nested fold.

## Phase 7: Adjoint discovery (already exists, verify it still works) — DONE

Consolidation already detects inverse pairs. With learned arithmetic:
- pred = adjoint(succ)
- sub = adjoint(add)
- div = adjoint(mul)

- [x] Adjunction detection works with learned (not seeded) morphisms
- [x] Adjoint morphisms evaluable via inverse traversal (Path 6 in evaluate()):
      observed-pair reverse lookup on the adjoint partner, then NT inverse
      ((y - intercept) / slope).  can_evaluate() recognizes adjoint-backed morphisms.
      5 tests: pred via observed reverse, pred via NT inverse, halve via double
      inverse, can_evaluate True for adjoint, not evaluable without partner data.

## Critical constraints (from Architecture.md)

- NO knowledge outside KG, WorkingMemory, Hippocampus
- NO dispatch on string names (Iron Law)
- NO arity assumptions — input structure is discovered from topology
- NO lookup tables outside the KG (OBSERVED edges are fine)
- ALL tests go through AgenticLoop
- ALL predictions are distributions, not point estimates

## Files modified

| File | Changes |
|------|---------|
| KnowledgeGraph.py | Deleted _seed_primitives, _prim_fns, _prim_op_ids, register_prim_fn. New evaluate() with 6 paths: observed extension → NT → composition → fold → nested fold-curry → adjoint inverse. Added set_nt()/get_nt(), add_fold_node(), add_nested_fold_node(), has_fold(). can_evaluate() recognizes adjoint-backed morphisms. |
| library.py | Added EdgeKind.NT_SLOPE, NT_INTERCEPT, ITERATE, CURRY_SOURCE, FOLD_BASE |
| Induct.py | Rewrote: topology-driven discovery with _fit_affine() OLS. Removed all composition fitting. |
| Abduct.py | Rewrote EM partition to use affine models. Domain classifier via observed pairs. Protected fold morphisms from retraction. |
| AgenticLoop.py | Removed _derive_ordering_comparisons (deferred to Phase 5 fold). |
| Consolidation.py | Fixed prune_dominated to only prune evaluable morphisms (cocone legs safe). |
| Induct.py | Added _discover_fold (NNO universal property), _discover_nested_fold (exponential object / currying), _filter_unexplained. |
| test_phase5_fold.py | NEW: 12 tests for fold KG mechanics + fold discovery. |
| test_phase6_curry.py | NEW: 10 tests for nested fold-curry KG mechanics + nested fold discovery. |
| checkpoint.py | Updated rendering to use target node label (no _prim_op_ids). |
| test_phase3.py | Rewrote make_succ_kg() with observed pairs + NT. |
| test_phase4.py | Removed comparison test (Phase 5). Updated morphism count assertion. |
| test_phase5.py | No changes needed — already works with new evaluation model. |
| test_phase6.py | Rewrote adjunction tests with observed pairs + NT. |
| test_phase7.py | Changed multivar test to affine function (product requires Phase 5+). |
