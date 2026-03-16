# CTKG Architecture — Canonical Type Table

## Purpose

This document is the authoritative record of canonical Python types for each
mathematical concept in the CTKG system.  Every code change that introduces or
modifies a data structure must be checked against this table.

**Iron Rule:**
There is exactly one canonical representation for each mathematical concept.
Before adding a new data structure, ask: does a structure already exist that
represents this concept?  If yes, use it.  If no, define it here first.

Two objects with the same mathematical structure are isomorphic and must be
identified — taking the quotient that collapses them is mandatory, not optional.
(Yoneda: an object is determined by its relationships.)

---

## Canonical Type Table

| Mathematical concept | Canonical Python type | File | Notes |
|---|---|---|---|
| Multi-input operation / hyperedge | `Relation` | `ctkg/learning/relation_store.py` | Named input and output roles; token-level; strictly more expressive than MultiMorphism |
| Unary chain / NNO step | `ProcessRule` | `ctkg/learning/process_discover.py` | Successor chain; NNO universal property |
| Binary scalar function | `BinaryFoldRule` | `ctkg/learning/process_discover.py` | digit×digit→digit with carry; NNO-completed table |
| Typed morphism (binary) | `CTKGMorphism` | `ctkg/core/morphism_graph.py` | Typed, context-sensitive; BFM and FCA layer |
| Token sequence with delimiters | `ChainRule` | `ctkg/learning/process_discover.py` | step/ans delimiters; Kleisli chain |
| Expression tree | `Expr` | `ctkg/core/term_algebra.py` | Initial algebra; atoms, nodes, pattern vars |
| Rewrite rule (structural) | `RewriteRule` | `ctkg/core/rewrite.py` | lhs→rhs with pattern variables; cata_reduce |
| Relational discovery rule | `RelationRule` | `ctkg/learning/relation_store.py` | BFM-lookup rule for named output role |
| Grammar rule (compression) | `GrammarRule` | `ctkg/learning/graph_grammar.py` | SEQUITUR digram rule; feeds OperadStructure |

---

## Deprecated Types

| Deprecated Python type | File | Canonical replacement | Reason |
|---|---|---|---|
| `MultiMorphism` | `ctkg/core/operad.py` | `Relation` | Anonymous positional; no named roles; fed heuristic levels removed in Phase X |

---

## Invariants

1. **`Relation` is the only representation for multi-input ops.** Any new code
   that introduces a second class representing "a function with multiple named
   inputs" is a violation.

2. **`KNOWN_INPUT_SEPS` must not exist.** Separator identity is discovered
   purely from distributional statistics (≥80% positional consistency).  No
   pre-seeded keyword sets.  (Phase XI Iron Rule fix.)

3. **No `int()` on token strings in the learning pipeline.** Digit identity is
   discovered from the NNO chain; `int()` casts are bitter-lesson violations.
   (`_apply_binary_formula` was the violation; it is removed.)

4. **No arity hard-coding.** Arities are either discovered from data or derived
   from the Relation schema.

---

## Phase History

| Phase | Change |
|---|---|
| Phase X | Removed heuristic prediction levels 0, 2, 3, 4 from `predict.py`; `MultiMorphism` orphaned |
| Phase XI | Removed `KNOWN_INPUT_SEPS` pre-seeded set; extended `Relation` with optional type-dist fields; deprecated `MultiMorphism` |
| Phase XII | `SlotProgram` generalization: power/derivative traces reach 100% OOD; `TraceProgram` synthesis for eval/linsolve |
| Phase XIII | Arity-free `RelationStore`; `discover_relation_rules` finds BFM-op rules without hardcoded arities; relational triple store |
| Phase XIV | Positional role schema for fixed-length ops; `mismatch_tolerance` param; `concat`/`div` BFM ops; coverage guard |
| Phase XV | Coproduct rule storage: `discover_relation_rules` keeps ALL qualifying rules per output role (not just best-evidence); `predict_alternatives_from_rules` returns weighted distribution over alternatives; Level 1c produces `dict[str,float]` |
| Phase XVI | Bitter-lesson compliance verified: `anon_math_benchmark.py` fixed to use full predictor pipeline (FC + NNO + BFM + RelationStore); anonymization test passes with 0.0% delta on all levels |
| Phase XVII | Equalizer solve: `_equalizer_predict` enumerates NNO digit chain and evaluates `add(mul(A,v),B)==C` via BFM; `linear_solve_seqs()` added; `linear_solve` reaches 100% OOD; no int() calls |
| Phase XVIII | Pullback predict: `_pullback_predict` handles `linsolve` (equalizer+forward BFM) and `bern_p1/bern_p2` (mul+add/sub via `_compose` NNO engine); `algebra_trace` and `bernoulli_trace` both reach 100% OOD |
| Phase XIX | Context category and restriction maps: `ctkg/core/context_category.py` defines `ContextId` (ANY/EQ/TRACE/INPUT) and `ContextCategory`; all `'eq' in prefix` format-detection guards in `predict_next` replaced by `ctx_cat.is_refinement(ctx, ContextId.EQ)`; prediction levels become presheaf sections registered at named context objects |
