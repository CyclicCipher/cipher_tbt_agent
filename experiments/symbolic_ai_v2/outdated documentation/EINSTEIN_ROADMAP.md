# Roadmap to the Einstein Test

## The Goal

Given four streams of anonymous observations — low-velocity mechanics, electromagnetic
fields, Michelson-Morley null result, Mercury perihelion precession — derive special
and general relativity from first principles. No physics hardcoded. All laws discovered
from data. All dispatch by graph node ID, never by string name.

## Status

| Phase | Capability | Status |
|-------|-----------|--------|
| 1 | Expression trees as CTKG nodes | ✓ |
| 2 | Parametric morphisms + OLS fitting | ✓ |
| 3 | Continuous quantities, unit conversion | ✓ |
| 4 | Theory compartments, consistency checking, symmetry groups | ✓ |
| 5 | Closed-loop surprise + revision (5 defects fixed) | ✓ |
| 6 | Retraction + replacement | ✓ |
| 7 | Latent variables, ontology extension, paradigm shift, orchestrator | ✓ |
| 8 | Deep IDA benchmark (I/D/A 4–8, 15 tracks, 10 seeds, 0.00 pp gap) | ✓ |
| 9 | Einstein test formal requirements | blocked by 10 |
| **10** | **CTKG-grounded law induction (`induce_law`)** | **← NEXT** |
| 11 | Stream-to-theory pipeline (`learn_from_stream`) | blocked by 10 |
| 12 | Symmetry-conflict latent hypothesis generation | ✓ |
| 13 | Evidence-triggered revision pipeline | ✓ |
| 14 | Concept-grounded paradigm shift + run_discovery | ✓ |
| 15 | Full integration — real Einstein test | blocked by 10, 11 |
| 16 | Cross-domain structural isomorphism detection | after 15 |
| 17 | Algorithm discovery from input/output pairs | after 15 |

---

## Compliance Rules (apply to every phase)

**The cage:** Run under 10 independent anonymous symbol tables (random Unicode U+2200–U+22FF
replacing all domain tokens). Variance < 5 pp. Gap (named vs anonymous) < 5 pp. Failure
means domain knowledge is encoded as string recognition.

**The defect probe:** One targeted test for the specific known-bad pattern most likely
in that phase.

**Iron Law:** No `if label == "PRIM_MUL"` or any string dispatch on operator names.
All dispatch by MorphId (opaque integer).

**Bitter Lesson:** No hardcoded hypothesis classes. Structure is discovered from data
via graph traversal, not assumed.

---

## Completed Phases

### Phase 1 — Expression Trees
`ctkg/core/term_algebra.py`: `Expr`, `var`, `atom`, `expr_match`, `expr_subst`.
Tests: `test_term_algebra.py`.

### Phase 2 — Parametric Morphisms
`ctkg/core/schematic_law.py`: `SchematicLaw`, `discover_parametric_law`.
`ctkg/core/parameter_fitter.py`: `FittedLaw`, `fit_parameters`, `add_fitted_law`, `predict_continuous`.
Tests: `test_schematic_law.py`, `test_parameter_fitter.py`.

OLS fits free parameters *within* a given expression structure. Does not discover
the structure — that is Phase 10.

### Phase 3 — Continuous Quantities
`ctkg/core/quantity.py`: `QuantityNode`, `EvalContext`, `eval_expr`, `eval_expr_batch`,
`continuous_surprise`, unit conversion via UNIT_CONV morphisms.
Tests: `test_quantity.py`, `test_phase3_pipeline.py`.

### Phase 4 — Theory Compartments
`ctkg/inference/theory.py`: `TheoryManager`, `register_theory`, `assign_morphism`,
`predict_under_theory`, `blame_theory`, `consistency_check`, `compare_symmetry_groups`.
Tests: `test_theory_manager.py`, `test_phase4_pipeline.py`, `test_theory_extensions.py`.

### Phase 5 — Closed-Loop Revision
`ctkg/inference/revision.py`: `ClosedLoopReviser`.
Tests: `test_revision.py`, `test_phase5_pipeline.py`.

Fixed five defects in the original revise.py: (1) single anomaly scored 0,
(2) OBS_SEQ edges ignored by predictor, (3) no cross-sequence evidence accumulation,
(4) open-loop acceptance, (5) bigram candidates couldn't reach theory-level cause.

### Phase 6 — Retraction
`ctkg/inference/retract.py`: `RetractEngine`, `retract_morphism`.
Tests: `test_retract.py`, `test_phase6_pipeline.py`.

### Phase 7 — Latent Variables and Ontology Extension
`ctkg/inference/abduce.py`: `hypothesise_latent`.
`ctkg/inference/coverage.py`: multi-anomaly coverage abduction.
`ctkg/inference/preservation.py`: `PredictionLedger`, `check_preservation`.
`ctkg/inference/paradigm.py`: paradigm shift / Left Kan Extension.
`ctkg/inference/orchestrator.py`: `AbductionOrchestrator` (levels 1→4).
`ctkg/einstein/streams.py`: synthetic linear analogs (scaffolding, not the Einstein test).
Tests: `test_coverage.py`, `test_preservation.py`, `test_paradigm.py`,
`test_orchestrator.py`, `test_phase7_pipeline.py` through `test_phase12_pipeline.py`.

### Phase 8 — Deep IDA Benchmark
`ctkg/tests/test_deep_ida_benchmark.py` (42 tests).
`tests/deep_ida_benchmark.py` (standalone CLI, 15 tracks × 10 seeds).
All tracks: 100% mean, 0.00 pp std, 0.00 pp gap.

D-8 (dependent type derivation): `TypedDeductionEngine` in `ctkg/inference/deduct.py`,
tested in `test_deduct_typed.py` (15 tests).

### Phase 12 — Symmetry-Conflict Latent Generation
`ctkg/inference/latent_conflict.py`: `hypothesise_from_symmetry_conflict`, `ConflictLatent`.
Tests: `test_phase15_latent_conflict.py` (8 tests).

Two theories with different symmetry groups → `ConflictLatent` stored as a
LATENT_CONCEPT morphism in the graph.

### Phase 13 — Evidence-Triggered Revision Pipeline
`ctkg/inference/revision_pipeline.py`: `auto_revise_on_anomaly`, `RevisionPipelineResult`.
Tests: `test_phase16_revision_pipeline.py` (9 tests).

Caller provides `replacement_law: FittedLaw`. Pipeline handles: anomaly check →
residual check → retract worst morphism → add replacement → preservation check.
Wire `induce_law` (Phase 10) as the law provider.

### Phase 14 — Paradigm Shift Wiring
`ctkg/inference/paradigm.py`: `wire_paradigm_shift`, `propose_paradigm_shift`.
`ctkg/einstein/run_discovery.py`: `run_discovery`, `seed_physics_priors`, `DiscoveryResult`.
Tests: `test_phase17_paradigm_wiring.py` (9 tests), `test_paradigm.py`.

Wires a new theory to constituent morphisms via PROJECTION/INCLUSION edges. Stores
pre-fitted laws (caller-provided) in theories, adds PARADIGM_SHIFT morphism.

---

## Phase 9 — Einstein Test Formal Requirements

**Not complete.** The six formal requirements are the pass criteria:

1. **Discovery:** γ(v), Lorentz transform, spacetime curvature all derived from streams,
   not pre-installed.
2. **Inspectability:** `inspect_gr_discovery(mg, tm) → GRDiscoveryAudit` fully populated
   (newtonian_laws, maxwell_laws, ether_morphism, retraction_reason, lorentz_factor_morphism,
   lorentz_factor_expr, spacetime_concept, curvature_morphism, mercury_prediction_error,
   revision_history).
3. **Proof:** `verify_gr_discovery(audit)` returns True with all 8 structural properties:
   - `len(newtonian_laws) >= 3`, `len(maxwell_laws) >= 2`
   - ether generated as latent then retracted
   - `lorentz_factor_expr` structurally encodes `1/√(1−v²/c²)`
   - `mercury_prediction_error < 0.05`
   - `spacetime_concept` created during the run, not before
   - revision history order: [ether added, ether retracted, Lorentz added, curvature added]
4. **Naming:** `physics_streams.py` carries `"THIS MODULE IS NOT THE EINSTEIN TEST"` header.
5. **Prior knowledge:** `seed_physics_priors` installs only: PRIM_OP morphisms,
   FRAME_CONCEPT node, MEASUREMENT_SCHEMA, GAUSSIAN_NOISE_MODEL. Nothing learnable
   from the streams.
6. **Symbol invariance:** gap < 5 pp across 10 anonymous symbol seeds.

**Blocker:** No mechanism to discover the functional form of a law from observations.
`parameter_fitter.py` fits parameters for a given form; it does not discover the form.
The deleted `discover_law` discovered forms but was a standalone Python system disconnected
from the CTKG. Phase 10 replaces it correctly.

---

## Phase 10 — CTKG-Grounded Law Induction ← NEXT

### The Architectural Requirement

Law induction = morphism composition enumeration. The search traverses PRIM_OP edges in
the MorphismGraph; the search state is a path of MorphId values; the result is a new
FITTED_LAW morphism added to the graph. No standalone beam search. No external operator
list in Python. Operators come from `mg.source_morphisms(FLOAT_OBJ, morph_type="PRIM_OP")`.

### What the Research Shows

The closest existing system is **Φ-SO** (Physical Symbolic Optimization,
arxiv.org/abs/2303.03192). It dispatches purely by integer token ID, uses a typed
constraint system to mask invalid operators at each step, and stores the operator library
externally. The algorithm pattern: **typed-edge beam traversal with MDL scoring**.

The theoretical anchor is **grammar-guided symbolic regression** (GGSR). A CFG production
rule `expr → OP(expr, expr)` is structurally identical to a typed morphism `OP: T×T → T`
in the graph. Non-terminals are types; production rules are typed edges; a valid expression
tree is a composable path in the free category generated by the operator graph. GGSR
proves this is a complete, sound search over expression space.

Relevant papers:
- Φ-SO: arxiv.org/abs/2303.03192 (typed dispatch, external library, dimensional masking)
- DSR: arxiv.org/abs/1912.04871 (integer token IDs, pre-order traversal, RL policy)
- Grammar-guided SR: arxiv.org/abs/2202.04367 (grammar rules as action space)
- Exhaustive SR with CFG: arxiv.org/abs/2109.13895 (complete enumeration up to depth bound)
- MDLformer: arxiv.org/html/2411.03753 (MDL as scoring objective, monotone convergence)
- E-graph deduplication: arxiv.org/html/2501.17848v1 (semantic dedup, 60% redundancy removed)

### MDL Scoring

`score(path) = tree_cost + λ · log(MSE + ε)`

where `tree_cost = Σ -log₂(p(edge))` along the traversal path. Under a uniform prior
over operators: `tree_cost = n_nodes · log₂(|PRIM_OP set|)`. Under a learned prior
(PCFG weights on morphisms): `tree_cost = Σ -log₂(morph.weight)`. Minimum score wins.

### γ(v) Recovery Strategy

Searching for γ(v) = 1/√(1−v²/c²) directly requires depth 6–7. Two strategies reduce this:

**Output transform (primary):** Transform observations from γ to 1/γ² = 1−v²/c² before
searching. Now the target is a depth-3 polynomial (SUB(1.0, MUL(p0, SQ(v))) where p0=1/c²).
Find it, then wrap with SQRT and INV. This is the key insight: search in the co-domain
of a simple invertible transform, not in the original co-domain.

**Typed dimensionless reparametrisation:** If v and c carry dimensional types, the type
constraint system automatically forces the combination v/c before search begins, reducing
one free variable.

For γ specifically: beam width 100–500 + output transforms + depth ≤ 5 is sufficient
(confirmed in Φ-SO and PySR benchmarks on the Feynman dataset which includes this equation).

### Algorithm Design

```
seed_prim_ops(mg):
    FLOAT_OBJ = mg.get_or_create_object("__float__")
    for (name, arity, fn) in PRIMITIVES:   # PRIMITIVES is a module-level constant list
        mg.add_morphism(FLOAT_OBJ, FLOAT_OBJ,
                        morph_type="PRIM_OP",
                        payload={"arity": arity, "fn": fn})
    # NodeIds are opaque; name is metadata only, never dispatched on

induce_law(mg, observations, max_depth, beam_width, n_param_slots) → MorphId:
    ops = mg.source_morphisms(FLOAT_OBJ, morph_type="PRIM_OP")
    # ops is a list of Morphism objects; identity is morph.morph_id (int)

    beam = [(score=0, expr=var("x"), open_slots=0)]  # Expr objects, head is MorphId
    for depth in range(max_depth):
        candidates = expand_beam(beam, ops, n_param_slots)
        scored = [(mdl_score(e) + log(mse(e, observations) + ε), e)
                  for e in complete(candidates)]
        beam = keep_diverse(scored, beam_width)

    # Also search with output transforms (INV, SQ applied to observation targets)
    for transform in output_transforms(ops):
        transformed_obs = apply_transform(transform, observations)
        inner = induce_law(mg, transformed_obs, max_depth-1, beam_width, n_param_slots)
        wrapped = compose_with_inverse(inner, transform)
        if score(wrapped) < score(best):
            best = wrapped

    law = FittedLaw(schema=best_expr, params=fit_params(best_expr, observations))
    return add_fitted_law(mg, "__induced__", law)
```

Key invariant: `expr.head` at every internal node is a `morph.morph_id` from `ops`.
Never a `TOKEN_GRAPH.encode("PRIM_MUL")`. The beam contains `Expr` objects whose
operator identity is a MorphId, not a string.

### New Files

- `ctkg/inference/law_induction.py` — `seed_prim_ops(mg)`, `induce_law(mg, obs, ...)`
- `ctkg/tests/test_law_induction.py` — cage + defect probes (written first)

### Cage and Probes

**Cage:** Seed PRIM_OP morphisms with random anonymous NodeIds (not TOKEN_GRAPH names).
Present observations from `f(x) = k*x²`. `induce_law` must recover a structurally
equivalent depth-2 composition, fit k within 5%, prefer it over the linear candidate
by MDL+residual. Variance < 5 pp across 10 symbol seeds.

**Probe — Iron Law:** Any `if nid == TOKEN_GRAPH.encode(...)` in the search path fails.

**Probe — γ(v) recovery:** Observations of γ(v) with c=0.3. `induce_law` at max_depth=5
must recover the expression to residual < 0.01 and fit c within 5%.

---

## Phase 11 — Stream-to-Theory Pipeline

```python
learn_from_stream(
    mg: MorphismGraph,
    tm: TheoryManager,
    stream: PhysicsObservationStream,
    theory_name: str,
    max_depth: int = 4,
) -> TheoryId
```

Calls `induce_law` for each observation set in the stream; stores resulting FITTED_LAW
morphisms in a new theory compartment. Replaces the deleted `stream_learner.py`,
correctly grounded via `induce_law`.

New file: `ctkg/inference/stream_induction.py`.

**Cage:** Learn a theory from an anonymous observation stream. Morphisms predict held-out
observations correctly regardless of symbol table. Variance < 5 pp.

---

## Phase 15 — Full Integration (Real Einstein Test)

Blocked by Phases 10 and 11. The complete pipeline:

```
seed_prim_ops(mg)
learn_from_stream(mg, tm, stream_1) → Newton theory
learn_from_stream(mg, tm, stream_2) → Maxwell theory
consistency_check + compare_symmetry_groups → symmetry conflict detected
hypothesise_from_symmetry_conflict() → ether morphism (latent)
[stream_3: MM null result]
blame_theory() → ether morphism
induce_law(mg, γ_obs) → γ(v) law morphism
auto_revise_on_anomaly(replacement_law=γ_law) → ether retracted
[stream_4: Mercury precession]
blame_theory() → Newtonian gravity
propose_paradigm_shift() + wire_paradigm_shift() → spacetime concept node
induce_law(mg, curvature_obs) → curvature morphism
inspect_gr_discovery(mg, tm) → GRDiscoveryAudit
verify_gr_discovery(audit) → True
```

New files needed:
- `ctkg/inference/law_induction.py` (Phase 10)
- `ctkg/inference/stream_induction.py` (Phase 11)
- `ctkg/einstein/audit.py` — `GRDiscoveryAudit`, `inspect_gr_discovery`, `verify_gr_discovery`
- `ctkg/tests/test_phase9_einstein.py` — invokes all six formal requirements

---

## Post-Einstein (Phases 16–17)

The same pipeline generalises to any domain with numeric observations.

**Phase 16 — Structural Isomorphism Detection:** After `induce_law` registers a law,
scan other theories for Expr trees with the same structural signature (depth, arity
sequence, compatible MorphId bijection verified by numerical evaluation). Store matches
as ISOMORPHISM morphisms. New function: `find_structural_isomorphisms(mg, tm, new_law)`.

**Phase 17 — Algorithm Discovery:** Given a black-box `probe_fn`, generate (input, output)
pairs and call `induce_law` (numeric outputs) or `RelationStore` (symbolic outputs).
New function: `discover_algorithm(probe_fn, mg, n_samples, max_depth)`.
