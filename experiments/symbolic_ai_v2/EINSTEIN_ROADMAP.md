# Roadmap to the Einstein Test

## What Is the Einstein Test

Given four streams of observations:
1. Mechanical experiments at low velocity (consistent with Newton)
2. Electromagnetic experiments (consistent with Maxwell)
3. The Michelson-Morley null result (inconsistent with Newtonian ether)
4. Mercury's perihelion precession (inconsistent with Newtonian gravity)

Derive a consistent theory that predicts all four.

The answer is special relativity (for 1–3) and general relativity (for 4). The system
must arrive at these theories through induction, deduction, and abduction over observations
— not through hardcoded physics knowledge.

---

## Part I: What the Current Architecture Cannot Do

### 1. No symbolic expression trees at the node level

Every node in the current system is a token atom: a discrete integer NodeId whose
identity is stored as a character tuple. Morphisms carry payloads that are token
sequences or rule structs (`BinaryFoldRule`, `RelationRule`), but there are no
expression tree nodes — no `Mul(Param('F'), Param('a'))`, no `Eq(Expr, Expr)`.

This means the system cannot:
- Represent `F = ma` as a CTKG morphism (it would have to encode it as the token
  string `"F=ma"`, which is opaque)
- Pattern-match on subexpressions of a law
- Compose laws symbolically (substituting one law into another)
- Detect structural similarity between `F = ma` and `E = mc²` (both are scaling laws)

Everything above token level is invisible to the inference engine.

### 2. No parametric morphisms

Every learned morphism is ground: it maps concrete token tuples to concrete token tuples.
`functor_discover.py` extracts bijections like `{man → woman, king → queen}` — it finds
which specific tokens co-vary, but not the abstract parameter governing them.

This means the system cannot:
- Infer `scale(X, k) = k * X` from examples `scale(1,10)=10, scale(2,10)=20`
- Represent a law with a free parameter (e.g., Hooke's law `F = -kx` for unknown k)
- Fit a parameter value from data (infer k=9.81 from falling-body observations)
- Detect that two laws share the same structural schema with different parameter values

Functor discovery discovers natural transformations between finite concrete sets. It does
not discover mathematical laws.

### 3. No continuous/real-valued quantity representation

All nodes are discrete token atoms. The number π is encoded as the string `"3.14159"` —
an atomic token, not a mathematical object with algebraic properties. There is no way to
express that 3.14159... is a limit of rational approximations, or that it satisfies `e^(iπ)+1=0`.

This means the system cannot:
- Represent velocity, force, field strength, or spacetime coordinates
- Do numerical fitting (minimise residuals over a continuous parameter space)
- Represent limits, derivatives, or integrals as operations on real-valued functions
- Detect that two measurements agree to within experimental error

Every physical quantity in Newtonian mechanics and electromagnetism is continuous.
The current architecture cannot even represent them.

### 4. No multi-domain theory consistency checking

The MorphismGraph is a single undifferentiated graph. There is no notion of "this cluster
of morphisms is the Newtonian theory" versus "this cluster is Maxwell's theory." There is
no consistency checker that asks: "Do these two theories make contradictory predictions
for the same observable?"

This means the system cannot:
- Represent Newtonian mechanics and Maxwell's equations simultaneously as distinct theories
- Detect that Newtonian mechanics + ether + Maxwell together predict a non-null
  Michelson-Morley result (contradicting the observation)
- Identify which theory is responsible for the failed prediction and target it for revision

Without theory compartments, the Michelson-Morley result is just another anomalous
token — there is no mechanism to trace the anomaly to a specific set of morphisms and
propose their retraction.

### 5. No morphism retraction

`revise.py` only adds morphisms. The `RevisionEngine._generate_candidates()` creates
new `OBS_SEQ` or `IMPLIES` edges; there is no `retract_morphism()` path. Once a
morphism is added to the graph, it persists unless MDL pruning removes it during a
training loop.

The Michelson-Morley result requires retracting the ether hypothesis (a concrete
morphism that predicts a non-null fringe shift). A system that can only add morphisms
cannot perform this retraction. It will instead try to add new morphisms to explain
away the anomaly, producing an increasingly inconsistent theory.

### 6. No latent variable hypothesis generation

`abduce.py` assumes all relevant morphisms are already in the graph. When carry
inversion fails and the type-level fallback finds no existing morphism to invert, it
returns empty. It cannot hypothesise that an unobserved intermediate quantity (a latent
variable) explains the observed output.

Special relativity requires hypothesising the Lorentz factor `γ = 1/√(1-v²/c²)` — a
latent structure that is not directly observed but explains multiple anomalies
simultaneously. The current abduction engine cannot generate it.

### 7. No paradigm shift capability (new ontology nodes)

`revise.py` creates `CTKGObject` nodes only for observed token labels. It cannot
propose entirely new concept nodes — abstract objects that have no surface form in the
training data.

General relativity requires inventing spacetime: a new 4-dimensional object that merges
space and time into a single manifold. This is not a new morphism between existing
concepts — it is a new concept that subsumes two previously separate concepts and redefines
the relationships between all physics morphisms. The current graph has no mechanism for
this.

### 8. The surprise + revision loop is broken at a design level

The current mechanism has five concrete defects that are independent of capability gaps
and would cause Einstein test failure even with Phases 1–6 implemented.

**Defect 1: Single anomaly is always ignored.**
`_generate_candidates()` scores each candidate as `len(explains) - complexity_penalty`.
With the default `complexity_penalty=1.0`, a single unique anomaly scores `1 - 1 = 0.0`.
`revise()` returns `None` if `best.score <= 0.0`. The Michelson-Morley experiment is a
single class of anomaly — the system will observe maximal surprise and do nothing.

**Defect 2: Revision writes OBS_SEQ edges; the predictor never reads them.**
`_apply()` always writes `morph_type="OBS_SEQ"`. The prediction engine reads only
`FOLD_RULE`, `CHAIN_STEP`, `FC_EDGE`, `ADJ_EDGE`, and `RELATION_RULE` morphisms.
Revision operates in a disconnected shadow stratum. After revision, the predictor
runs the same unchanged theory. Revision never actually changes what is predicted.

**Defect 3: No evidence accumulation across separate observations.**
Each call to `revise(tokens)` processes one sequence in isolation. There is no running
posterior `P(theory | all_observations_so_far)`. Ten Michelson-Morley repetitions as
ten separate sequences each score `1 - 1 = 0.0`. The cumulative overwhelming evidence
is invisible.

**Defect 4: Open-loop — no re-evaluation after applying a candidate.**
`revise()` applies the best candidate and returns without checking whether the revision
reduced surprise on the anomalous data. Bad surface patches are accepted unconditionally.

**Defect 5: Bigram candidates cannot reach a theory-level cause.**
Candidates are generated from `(prev_tok, curr_tok)` pairs. When a Michelson-Morley
sequence surprises the system, the proposed candidate is a bigram edge between adjacent
observation tokens. The ether morphism sits several hops back in the theory graph.
There is no causal tracing from a surface token anomaly to the responsible theory
morphism.

### 9. Only I-1/I-3, D-1/D-3, A-1/A-2 benchmark coverage

The IDA benchmark currently reaches I-3 (composed unary ops), D-3 (3-hop deduction
chains), and A-2 (two competing hypotheses). The Einstein test requires roughly:
- I-8: Full theory induction (Newtonian mechanics from kinematics experiments)
- D-8: Type-theoretic derivation (Lorentz transformation from symmetry principles)
- A-8: Paradigm shift abduction (spacetime from multiple incompatible anomalies)

Each level from the current ceiling to those targets is a substantial research problem.

---

## Part II: What It Needs

The architecture needs seven new capabilities, each building on the previous:

1. **Expression trees as first-class CTKG nodes** — so laws can be represented,
   composed, and pattern-matched symbolically.

2. **Parametric morphisms** — so the system can represent and discover laws with free
   parameters, and fit parameter values from data.

3. **Continuous/real-valued quantities** — so physical measurements can enter the graph
   and laws can make numerical predictions.

4. **Theory compartments with consistency checking** — so multiple theories can coexist,
   their predictions can be compared for the same observable, and contradictions can be
   located and attributed.

5. **A redesigned surprise + revision loop** — closed-loop, evidence-accumulating,
   theory-morphism-targeting, with causal attribution. The current mechanism is
   disconnected from the predictor and cannot act on a single anomaly.

6. **Full revision cycle (add + retract + replace)** — so anomalies that falsify
   existing morphisms trigger targeted retraction and principled replacement, not just
   accretion. Requires capability 5 to have a working loop to retract within.

7. **Latent variable and ontology-extension abduction** — so the system can hypothesise
   unobserved structures (latent variables, new concept nodes) when no existing structure
   explains the anomaly.

---

## Part III: Roadmap

### Phase 1: Expression Trees as CTKG Nodes — ✓ COMPLETE

**Implemented in:** `ctkg/core/term_algebra.py` (`Expr`, `var`, `atom`, `expr_match`, `expr_subst`).
Tests: `test_term_algebra.py`.

**Goal:** Algebraic expressions are first-class graph objects. Laws can be stored,
composed, and pattern-matched as CTKG morphisms.

**What changes:**
- Add `ExprNode` to the node type system alongside `TokenAtom`. An ExprNode is a tree
  of operator nodes and variable/constant leaves — not a token string.
- `CTKGMorphism.payload` can now carry an `Expr` object as its rule body.
- Add `expr_match(pattern: Expr, target: Expr) → dict[str, Expr]` — unification/
  pattern matching that returns variable bindings.
- Add `expr_subst(expr: Expr, bindings: dict) → Expr` — substitution.
- Laws like `F = m*a` become morphisms whose payload is `Eq(Var('F'), Mul(Var('m'), Var('a')))`.

**What it enables:**
- Structural similarity detection between laws (scaling law schema)
- Composition of laws by substitution (substitute `a = F/m` into kinematic equations)
- Rewrite-rule discovery from observed expression transformations

**Verification:** A new test suite `test_expr_morphism.py` — given a stream of
`(F_value, m_value, a_value)` triples, the system represents and queries the morphism
`F = m*a` symbolically, including solving for any one variable given the other two.

**Does not require:** continuous quantities (Phase 3), parametric fitting (Phase 2).
All expressions can still have concrete token values at their leaves.

---

### Phase 2: Parametric Morphisms and Law Discovery — ✓ COMPLETE

**Implemented in:** `ctkg/core/schematic_law.py` (`SchematicLaw`), `ctkg/core/parameter_fitter.py`
(`FittedLaw`, `fit_parameters`, `add_fitted_law`).
Tests: `test_schematic_law.py`, `test_parameter_fitter.py`.

**Goal:** The functor discovery pipeline produces schematic laws with free parameters,
not just concrete bijections.

**What changes:**
- Extend `FunctorCandidate` to `SchematicLaw(params: list[str], body: Expr, evidence: int)`.
  A `SchematicLaw` has named free parameters that are inferred from training examples.
- Extend `functor_discover` with a **parameter extraction pass**: after collecting
  concrete variable bindings, factor out the common functional relationship. If
  `x → 10, 2x → 20, 3x → 30`, the extractor finds the linear schema `f(n) = 10*n`
  and registers `SchematicLaw(params=['k'], body=Mul(Var('k'), Var('n')))`.
- `RelationStore.discover_relation_rules()` generalises from concrete (input → output)
  pairs to `(input_schema → output_schema)` parametric rules.
- `predict_next` can instantiate a SchematicLaw with concrete values to make predictions.

**What it enables:**
- Hooke's law (`F = -kx`) discovered from spring force measurements with unknown k
- Ohm's law (`V = IR`) discovered from voltage/current pairs
- The gravitational law schema (`F = G*m1*m2/r²`) from force/distance pairs

**Verification:** IDA benchmark I-5: given 5 examples of `scale(X, k)` for a fixed k
and varying X, predict `scale(X', k)` for novel X'. Then vary k across a family of
laws and detect that the same schema applies.

**Does not require:** continuous fitting (can still use symbolic parameter discovery on
discrete examples). Phase 3 adds numerical fitting on top.

---

### Phase 3: Continuous Quantities and Numerical Fitting — ✓ COMPLETE

**Implemented in:** `ctkg/core/quantity.py` (`EvalContext`, `predict_continuous`).
OLS fitting via numpy in `parameter_fitter.py`.
Tests: `test_quantity.py`, `test_phase3_pipeline.py`.

**Goal:** The CTKG can represent and reason about real-valued physical quantities.
Laws can be fitted to numerical observations.

**What changes:**
- Add `QuantityNode(value: float, unit: str)` to the node type system. Quantity nodes
  live in the range NodeId 10000+ (above token atoms).
- Add dimensional analysis morphisms: `velocity = displacement / time` is a typed
  morphism between quantity types, not token types.
- Add a **parameter fitter** module: given a `SchematicLaw` and a stream of
  `(quantity_observations)`, minimise residuals over free parameters. Output:
  `FittedLaw(schema: SchematicLaw, params: dict[str, float], residual: float)`.
- Integrate `FittedLaw` into `CTKGMorphism.payload` — a morphism can now carry a
  fitted law with a residual error annotation.
- Surprise scoring gains a **continuous mode**: surprise for a `QuantityNode`
  observation is `(predicted - observed)² / σ²`, not KL on a discrete distribution.

**What it enables:**
- Newtonian mechanics: given (force, mass) pairs, fit `a = F/m` and predict
  acceleration for novel (force, mass) combinations
- Gravitational constant G fitted from planetary orbit data
- The speed of light c fitted from electromagnetic measurements

**Verification:** IDA benchmark I-5 extended to continuous domains. Given 10
(force, mass, acceleration) triples with Gaussian noise, recover F=ma to within 1%
parameter error.

---

### Phase 4: Theory Compartments and Cross-Domain Consistency — ✓ COMPLETE

**Implemented in:** `ctkg/inference/theory.py` (`TheoryManager`, `register_theory`,
`assign_morphism`, `blame_theory`, `consistency_check`).
Tests: `test_theory.py`, `test_phase4_pipeline.py`.

**Goal:** Multiple named theories coexist in the CTKG as isolated subgraphs. Their
predictions for shared observables can be compared. Contradictions are detected and
attributed.

**What changes:**
- Add `Theory(name: str, morphisms: set[MorphismId], domain: ObjectId)` as a CTKG
  node type. A theory is a subgraph with a named root.
- Add `TheoryManager` to `MorphismGraph`:
  - `register_theory(name: str) → TheoryId`
  - `assign_morphism_to_theory(morph_id, theory_id)`
  - `predict_under_theory(query, theory_id) → dict[str, float]`
  - `consistency_check(theory_a, theory_b, observable) → ConsistencyResult`
  - `blame_theory(anomaly, candidate_theories) → TheoryId` — identify which theory
    is responsible for a failed prediction.
- `SurpriseDetector` gains `scan_with_attribution(sequence, theories) → list[(position, surprise, blamed_theory)]`.

**What it enables:**
- Newtonian mechanics and Maxwell's equations represented as separate `Theory` nodes
- `consistency_check(Newtonian, Maxwell, fringe_shift_observable)` returns INCONSISTENT
  when Newtonian-ether predicts non-null and Maxwell-alone predicts null
- `blame_theory` identifies the ether morphism as the responsible assumption

**Verification:** IDA benchmark A-3: two theories make the same prediction on 90% of
observables but disagree on a specific one. The system attributes the anomaly to the
correct theory and does not revise the innocent one.

---

### Phase 5: Closed-Loop Surprise and Revision — ✓ COMPLETE

**Implemented in:** `ctkg/inference/revision.py` (`ClosedLoopReviser`). All five defects
from the roadmap fixed: Bayesian MDL scoring, theory-stratum routing, evidence accumulation
via `observe()`/`flush()`, closed-loop re-evaluation, causal attribution via `TheoryManager`.
Tests: `test_revision.py`, `test_phase5_pipeline.py`.

**Goal:** Redesign the surprise + revision mechanism so that it (a) can act on a single
anomaly, (b) targets theory morphisms instead of surface bigrams, (c) accumulates
evidence across separate observations, and (d) verifies that each revision actually
reduces surprise before accepting it.

This phase fixes five design defects in the current `surprise.py` / `revise.py` that
would cause Einstein test failure regardless of what Phases 1–4 deliver.

**What changes:**

**Fix 1 — Score single anomalies.**
Remove the `complexity_penalty` floor as the sole gate on adoption. Replace it with a
Bayesian posterior: `score = log P(data | candidate) - MDL(candidate)`. A single
strongly-surprising anomaly (KL = 20 nats) can produce a positive score even with a
complexity penalty, because `20 - MDL(one morphism)` is positive. The gate becomes
`score > 0` in log-probability space, not `count - 1 > 0`.

**Fix 2 — Route revision candidates into the prediction strata.**
`_apply()` must write candidates as the morph_type that `predict_next` actually reads.
When the anomaly is a failed fold-rule prediction, the candidate is a `FOLD_RULE`
modification. When the anomaly is a failed relation-rule prediction, the candidate is a
`RELATION_RULE` modification. `OBS_SEQ` edges remain only for raw observation logging,
not as a substitute for theory morphisms.

This requires `_generate_candidates()` to know which prediction stratum was active when
the anomaly fired — i.e., which `morph_type` in `_ctkg_path_find` returned the wrong
answer. `SurpriseAnnotation` gains an `active_stratum: str` field set by
`scan_sequence()` from the predictor's internal trace.

**Fix 3 — Accumulate evidence across observations.**
Add `RevisionEngine.observe(tokens)` — a method that scans a sequence, records
surprise annotations into a persistent `evidence_buffer: list[SurpriseAnnotation]`,
and does not immediately revise. Add `RevisionEngine.flush()` — called when the
buffer exceeds a threshold or when the caller requests it — that runs
`_generate_candidates()` over the entire buffer. Candidates that explain anomalies
across multiple sequences score higher than candidates that explain only one.

**Fix 4 — Close the loop.**
After `_apply(best)`, call `scan_sequence()` on the anomalous sequences again. If
surprise has decreased below threshold, accept. If not, roll back the edit, remove
the candidate from consideration, and try the next best. Maximum three retry steps
before giving up and logging the unresolved anomaly.

**Fix 5 — Causal attribution via theory compartments.**
Require Phase 4 (`TheoryManager`) as a prerequisite. `_generate_candidates()` takes
the blamed theory (from `TheoryManager.blame_theory()`) as input and only considers
morphisms belonging to that theory as retraction/modification candidates. This
prevents surface-symptom patching: the candidate set is restricted to morphisms in the
compartment that made the wrong prediction.

**What it enables:**
- A single Michelson-Morley null result triggers a revision candidate targeting the
  ether morphism (the blamed theory's prediction), not a surface bigram
- After three more null results in the evidence buffer, the ether morphism's score
  falls below zero and it is flagged for retraction (Phase 6 executes the retraction)
- The closed loop rejects candidates that do not reduce surprise, preventing the
  accumulation of useless OBS_SEQ patches

**Verification:** Extend IDA benchmark A-1 and A-2:
- A-1 (revised): a single anomaly with KL = 15 nats triggers adoption of the correct
  theory-stratum candidate, and surprise on the anomalous sequence drops below
  threshold after revision.
- A-2 (revised): two competing theory-stratum candidates; the closed loop selects the
  one that actually reduces surprise, not the one that covers more bigrams.

---

### Phase 6: Full Revision Cycle — Add, Retract, Replace — ✓ COMPLETE

**Implemented in:** `ctkg/inference/retract.py` (`RetractEngine`, `retract_morphism`,
`propose_replacement`). `RevisionHistory` stored as CTKG subgraph annotations.
Tests: `test_retract.py`, `test_phase6_pipeline.py`.

**Goal:** `RevisionEngine` can retract existing morphisms and propose replacements, not
just add new morphisms.

**What changes:**
- Add `RevisionEngine.retract_morphism(morph_id: MorphismId, reason: str)` — removes a
  morphism and logs the reason as a graph annotation.
- Extend `_generate_candidates` to produce `RetractionCandidate(morph_id, justification)`
  alongside `AdditionCandidate`. A `RetractionCandidate` scores by: how many anomalies
  does retracting this morphism resolve, minus the predictions it would break.
- Add `ReplacementCandidate(retract: MorphismId, add: SchematicLaw)` — the atomic unit
  of scientific theory change: remove one morphism, add a more general one.
- Posterior scoring: `P(replacement | data) ∝ P(data | replacement) * P(replacement)`.
  Complexity prior: shorter law bodies score higher (MDL principle, already in use).
- Add `RevisionHistory` — a log of all retractions and replacements, stored as a
  subgraph of the CTKG, so the system can reason about its own revision history.

**What it enables:**
- Retract ether hypothesis when Michelson-Morley anomaly is strong enough
- Replace `x' = x - vt` (Galilean transform) with `x' = γ(x - vt)` (Lorentz transform)
- Preserve the Newtonian approximation as a limiting case (γ → 1 as v/c → 0)

**Verification:** IDA benchmark A-4 and A-7. A-4: given an established morphism that
causes 100% false positives on a new observation class, retract it and score 0 false
positives after. A-7: revise without breaking 95%+ of previously-explained phenomena.

---

### Phase 7: Latent Variable and Ontology-Extension Abduction — ✓ COMPLETE

**Implemented in:** `ctkg/inference/abduce.py` (`hypothesise_latent`, `LatentHypothesis`,
`propose_new_concept`). MDL scoring via `mdl_score`.
Tests: `test_abduce.py`, `test_phase7_pipeline.py`.

**Supplementary work (beyond this phase, implemented as phases 8–12 in-session):**
- `ctkg/inference/coverage.py` — multi-anomaly A-6 coverage abduction
- `ctkg/inference/preservation.py` — A-7 `PredictionLedger` + `apply_with_preservation`
- `ctkg/inference/paradigm.py` — A-8 paradigm shift / Left Kan Extension
- `ctkg/inference/orchestrator.py` — `AbductionOrchestrator` cascading levels 1→4
- `ctkg/einstein/streams.py` — synthetic linear analogs of four Einstein scenarios
Tests: `test_coverage.py`, `test_preservation.py`, `test_paradigm.py`, `test_orchestrator.py`,
`test_phase8_pipeline.py` through `test_phase12_pipeline.py` (1085 total passing).

**Goal:** The abduction engine can hypothesise unobserved structures — both latent
variables within an existing theory and entirely new concept nodes.

**What changes:**

**Latent variable abduction:**
- Extend `abduce.py` with `hypothesise_latent(anomalies: list, theory: Theory) →
  list[LatentHypothesis]`. A `LatentHypothesis` posits an unobserved quantity X such
  that the anomalies are explained by a law involving X.
- Candidate latents are generated by: (a) inverting known laws with unknown inputs,
  (b) enumerating SchematicLaw templates with one free variable slot left unfilled.
- Scoring: `MDL(latent hypothesis) + residual(latent predicts observations)`.

**Ontology extension:**
- Extend `RevisionEngine` with `propose_new_concept(justification: str) → ObjectId`.
  A new concept is a fresh ObjectId with no character representation — purely structural.
  Its identity is defined entirely by the morphisms that connect it to existing objects.
- An ontology extension is a `ReplacementCandidate` that includes a `new_concept` field.
- Scoring: concepts that unify multiple existing morphisms under a common abstraction
  score higher (Kolmogorov complexity reward).

**What it enables:**
- Hypothesise the Lorentz factor γ as a latent quantity explaining both the
  Michelson-Morley result and the observed velocity-dependent mass increase
- Hypothesise spacetime as a new concept node that unifies the spatial and temporal
  transformation morphisms under a single 4D structure
- Propose curved spacetime as the ontology extension required to explain Mercury's
  perihelion precession

**Verification:** IDA benchmark A-5 and A-8. A-5: given 10 observations of `f(x)` where
f is `h ∘ g` and g is unobserved, recover g as a latent. A-8: given observations
inconsistent with any existing concept structure, propose a new abstract concept that
resolves the inconsistency.

---

### Phase 8: Deep IDA Benchmark (I-4 through I-8, D-4 through D-8, A-3 through A-8) — ✓ COMPLETE

**Implementation:** `ctkg/tests/test_deep_ida_benchmark.py` (42 tests, all passing).
**Standalone CLI runner:** `tests/deep_ida_benchmark.py` (15 tracks × 10 seeds, gap check).
**D-8 deferred:** Dependent-type derivation requires architectural extension to
DeductionEngine; all other tracks pass at 100% mean, 0.00 pp std, 0.00 pp gap.

**Goal:** The IDA benchmark is extended to test every level described in the Phase XXV
plan. Phases 1–7 are verified to compose correctly.

**New benchmark tracks added:**
- I-4: Multi-case induction (base case + recursive case)
- I-5: Parametric family discovery (SchematicLaw recovery)
- I-6: Algebraic law discovery (commutativity, associativity from examples)
- I-7: Functor law discovery (F(f∘g) = F(f)∘F(g))
- I-8: Full theory induction from structured observation streams
- D-4: Case split in deduction
- D-5: Modus ponens (propositional)
- D-6: Universal quantifier instantiation
- D-7: Bounded proof search
- D-8: Dependent type derivation
- A-3: Hypothesis requiring new morphism type
- A-4: Anomaly falsifies existing morphism → retract and replace
- A-5: Latent variable hypothesis
- A-6: Multiple anomalies → single unified explanation
- A-7: Theory revision preserving prior explanations
- A-8: Paradigm shift → new concept node

**Pass criteria:** Per Phase XXV Part VIII (≥90% for I/D, ≥80% for A, variance <5% across
10 anonymous symbol seeds).

This phase produces no new architectural components. It is purely verification and
benchmark extension.

---

### Phase 9: Einstein Test — ✗ NOT DONE (scaffolding only)

---

#### NOTICE: The abduction routing scaffold is NOT this test

`ctkg/tests/test_phase12_pipeline.py` and `tests/einstein_benchmark.py` test
*abduction routing machinery* on four synthetic linear scenarios (`f(x) = k*x`).
They verify that the correct orchestrator level (1/2/3/4) fires and that preservation
blocks inappropriate revisions. This is structural scaffolding for the real test.
The scaffolding verifies routing; it does not verify discovery.

**The scaffolding does NOT constitute Phase 9 completion. The scaffolding cannot
constitute Phase 9 completion regardless of pass rates, seed counts, or extensions.**

The four synthetic linear analogs (`newtonian_scenario`, `michelson_morley_scenario`,
`mercury_precession_scenario`, `maxwell_em_scenario`) in `ctkg/einstein/streams.py`
are scalar-magnitude proxies. They test whether the orchestrator escalates from
level 1 to level 2+ under preservation constraints. They say nothing about whether
the system can discover general relativity.

---

#### Formal requirements (all six are PASS/FAIL — not preferences)

**Requirement 1 — Discovery criterion (PASS/FAIL)**

The test is only complete if the system derives a consistent predictive theory from the
four observation streams using only the information Einstein had. Specifically:
- The Lorentz transform `x′ = γ(x − vt)` must be derived from observations, not
  installed as a graph node.
- The Lorentz factor `γ(v) = 1/√(1−v²/c²)` must be recovered by `discover_law` as a
  composed expression over PRIM_OP morphisms, not seeded.
- The spacetime curvature concept node must be generated by `propose_paradigm_shift` or
  `propose_new_concept` as a response to the Mercury anomaly, not pre-installed.
- A prediction of Mercury's perihelion residual ≥ 90% accurate on held-out data must
  follow from the derived (not installed) curvature morphism.

FAIL if: any of Lorentz transform, γ(v), or spacetime curvature are in the graph
before the observation streams are presented.

**Requirement 2 — Inspectability criterion (PASS/FAIL)**

The system's reasoning must be auditable via graph inspection. There must exist a
function `inspect_gr_discovery(mg: MorphismGraph, tm: TheoryManager) -> GRDiscoveryAudit`
that returns a structured audit trail containing:

```python
@dataclass
class GRDiscoveryAudit:
    newtonian_laws: list[MorphismId]   # morphisms learned from stream 1
    maxwell_laws: list[MorphismId]     # morphisms learned from stream 2
    ether_morphism: Optional[MorphismId]  # generated then retracted
    retraction_reason: Optional[str]   # why ether was retracted
    lorentz_factor_morphism: MorphismId  # γ(v) discovered by compose_search
    lorentz_factor_expr: Expr           # the expression tree (inspectable)
    spacetime_concept: ObjectId         # new concept node from paradigm shift
    curvature_morphism: MorphismId      # GR curvature law
    mercury_prediction_error: float     # |predicted - observed| / observed
    revision_history: list[RevisionRecord]  # full ordered revision log
```

FAIL if: any of these fields cannot be populated from the graph after the test runs.
A passing test that only asserts `success == True` without this audit is INSUFFICIENT.

**Requirement 3 — Proof criterion (PASS/FAIL)**

It must be provable from graph inspection — not from test assertions — that the system
discovered general relativity. A function `verify_gr_discovery(audit: GRDiscoveryAudit)
-> bool` must check all of the following structural properties:

1. `len(audit.newtonian_laws) >= 3` — at least F=ma, x(t), momentum conservation
   learned from stream 1.
2. `len(audit.maxwell_laws) >= 2` — at least field equation + wave propagation from
   stream 2.
3. `audit.ether_morphism is not None` — ether was generated as a latent hypothesis
   (derived, not installed; if it was never generated the MM anomaly wasn't explored).
4. `audit.retraction_reason is not None` — ether was retracted with justification.
5. `audit.lorentz_factor_expr` encodes the composition
   `DIV(1, SQRT(SUB(1, MUL(POW(v,2), POW(c,-2)))))` or structurally equivalent.
6. `audit.mercury_prediction_error < 0.05` — curvature morphism predicts to 5%.
7. `audit.spacetime_concept` is a NodeId generated during the test run, not before.
8. The revision history shows: [ether added, ether retracted, Lorentz added, curvature
   added] — in that causal order.

FAIL if `verify_gr_discovery(audit)` returns False.

**Requirement 4 — Naming criterion (PASS/FAIL)**

The scaffolding in `ctkg/einstein/streams.py`, `ctkg/tests/test_phase12_pipeline.py`,
and `tests/einstein_benchmark.py` is named and documented to distinguish it from this
test. DONE — already enforced by the "THIS MODULE IS NOT THE EINSTEIN TEST" headers
and the `TestAbductionRouting*` class names.

Any new file that tests synthetic linear scenarios must include the header:

```
THIS IS NOT THE EINSTEIN TEST.
```

in its module docstring and must not be named `test_einstein_*.py` or placed in a
path that suggests it is the Einstein test. FAIL if a file omits this header.

**Requirement 5 — Prior knowledge criterion (PASS/FAIL)**

All prior knowledge the system receives before seeing the observation streams must
exist as graph nodes installed by `seed_physics_priors(mg: MorphismGraph)`. There
must be NO physics knowledge encoded in Python as:
- hardcoded constants (`c = 299792458.0` in module scope)
- hardcoded law templates (`SchematicLaw` constructed before `discover_law` runs)
- hardcoded concept names (`TOKEN_GRAPH.encode("ether")` called at init)
- special-case dispatch on physical quantity names

The permitted prior knowledge is exactly:

| Node type | What it installs | Why permitted |
|-----------|-----------------|---------------|
| PRIM_OP morphisms | PRIM_MUL, PRIM_ADD, PRIM_SUB, PRIM_DIV, PRIM_SQRT, PRIM_POW, PRIM_NEG, PRIM_SQ, PRIM_INV | Domain-independent mathematical substrate; nothing derives them |
| FRAME_CONCEPT | Abstract reference frame ObjectId (no semantics, just the node) | Abstract structural concept needed to define transforms; not derivable from observations alone |
| MEASUREMENT_SCHEMA | That observations are (value: float, frame_id: ObjectId) pairs | Experimental protocol, not a physics law |
| GAUSSIAN_NOISE_MODEL | That residuals ≤ σ are consistent with a theory | Statistical prior, not physics |

The following are FORBIDDEN as installed nodes (they must emerge from observations):

| Forbidden prior | Why forbidden |
|----------------|---------------|
| Newtonian laws (F=ma, x(t), p=mv) | Learnable from stream 1 |
| Galilean transform (x′=x−vt) | Learnable from cross-frame kinematics pairs |
| Maxwell's equations | Learnable from stream 2 |
| Ether hypothesis | Derivable from Galilean invariance + Maxwell wave equation; must emerge as a latent candidate, then be falsified by MM |
| Speed of light c | Learnable from EM wave propagation data in stream 2 |
| Lorentz transform | Must be discovered — this is the test |
| γ(v) functional form | Must be recovered by discover_law — this is the test |
| Spacetime concept | Must be proposed by paradigm shift — this is the test |
| Spacetime curvature | Must be proposed by ontology extension — this is the test |

FAIL if `seed_physics_priors` installs any node in the forbidden list, or if any
module-level constant encodes a physical quantity that should be discovered.

Note on the ether hypothesis: the system must GENERATE the ether morphism as a latent
candidate when it first reconciles Galilean invariance (learned from stream 1) with
Maxwell's constant-c equation (learned from stream 2). The ether is not a separate
prior — it is what the system hypothesises to explain how Maxwell's equations can have
a constant wave speed in a Galilean universe. The MM null result then falsifies it.
If the ether morphism is installed, the MM scenario does not test discovery; it tests
routing (which is what the scaffolding already does).

**Requirement 6 — Written into the phase (PASS/FAIL)**

These six requirements are the Phase 9 pass criteria. They are not preferences or
aspirations. Phase 9 is not complete until `verify_gr_discovery(inspect_gr_discovery(mg, tm))`
returns True on held-out test data under 10 independently randomised anonymous symbol
tables, with gap (named vs anonymous) < 5 pp.

A test suite that does not invoke `verify_gr_discovery` and `inspect_gr_discovery`
cannot satisfy Phase 9, regardless of what other assertions pass.

---

#### What the current architecture cannot do (blockers)

1. **γ(v) = 1/√(1−v²/c²) requires compose_search depth ≥ 5.**
   `compose_search.py` implements the beam search, but Phase 9 needs to confirm
   it recovers this specific expression at depth=5 from velocity/time-dilation data.
   This has not been tested on physics streams — only on synthetic linear scenarios.

2. **Lorentz covariance detection is not implemented.**
   The system needs to detect that Maxwell's equations have Lorentz symmetry (not
   Galilean symmetry) by checking which transformation group leaves the field equation
   morphisms invariant. This requires `consistency_check` to compare symmetry groups
   of two theories — not currently in `theory.py`.

3. **Cross-theory quantitative inference is not implemented.**
   To predict Mercury's 43″/century from the Lorentz theory derived in stream 3,
   `TheoryManager` needs an inter-theory inference chain: take Lorentz transform from
   SR theory, compose with gravitational morphism from Newtonian theory, evaluate on
   Mercury orbit parameters. This is not in `orchestrator.py`.

4. **Spacetime concept node requires structural wiring.**
   `propose_paradigm_shift` creates an opaque new ObjectId. For the audit to satisfy
   Requirement 3 property 7, the new node must be wired to the spatial morphisms from
   stream 1 and the temporal transform morphisms from stream 3. `paradigm.py` does not
   currently do this wiring.

5. **Physics observation streams are not implemented.**
   `ctkg/einstein/physics_streams.py` does not exist. The real Einstein test requires
   streams encoding kinematic triples, EM field measurements, interferometer fringe
   data, and planetary orbit data — with realistic Gaussian noise, anonymous symbols
   for all physical quantities, and frame_id labels. These are separate from the
   synthetic `f(x) = k*x` scenarios in `ctkg/einstein/streams.py`.

---

#### Observation streams

These streams constitute the only information the system receives (beyond the permitted
installed priors). All physical quantity names are anonymous (symbol-permuted).

**Stream 1 — Newtonian kinematics and dynamics** (target: I-5/I-6)

Observations:
- Uniform motion triples: `(position_0, velocity, time) → position_1`
  where `position_1 = position_0 + velocity * time`
- Accelerated motion: `(position_0, velocity_0, accel, time) → position_1`
- Force-mass-acceleration: `(force, mass) → acceleration` where `accel = force / mass`
- Momentum: `(mass, velocity) → momentum` where `momentum = mass * velocity`
- Momentum conservation: `(momentum_before_A, momentum_before_B) → (momentum_after_A,
  momentum_after_B)` with constraint `sum_before = sum_after`
- Cross-frame pairs: same event observed from two frames moving at relative velocity v,
  with `x′ = x − v*t` (Galilean) holding to within experimental noise at v ≪ c

Expected induction: SchematicLaws for x(t), F=ma, p=mv; Galilean transform as a
two-parameter SchematicLaw with frame_id labelling.

**Stream 2 — Electromagnetic fields** (target: I-7)

Observations:
- Static field: `(charge, distance) → field_strength`
- Induced EMF: `(flux_rate) → emf`
- Wave propagation: `(wavelength, period) → wave_speed` with wave_speed constant
  across all wavelength/period combinations (this encodes c as a constant)
- Cross-frame field transform: E and B field values in two frames moving at relative
  velocity v, showing that the E/B mix is frame-dependent

Expected induction: field law SchematicLaws; c discovered as a constant parameter
(all wave propagation observations share the same wave_speed value); field transform
morphisms; detection that the symmetry group preserving the field equations is NOT
the Galilean group (because c is constant in all frames — contradicts Galilean).

**Stream 3 — Michelson-Morley** (target: A-4/A-6)

Observations:
- Interferometer fringe shift measurements at multiple apparatus orientations
- All fringe shifts: 0 (within noise) regardless of orientation or Earth's orbital
  velocity direction

Expected sequence of events:
1. `consistency_check(Newtonian_theory, Maxwell_theory, fringe_shift_observable)`:
   Newtonian + Galilean + ether predicts non-null; Maxwell predicts null. INCONSISTENT.
2. `blame_theory`: ether morphism (or Galilean transform) is at fault.
3. `hypothesise_latent`: generate latent quantity γ(v) as a candidate correction to the
   Galilean transform.
4. `discover_law` on the γ constraint: recovers `1/√(1−v²/c²)` using c from stream 2.
5. `retract_morphism(ether_morphism)` + `propose_replacement(galilean_transform,
   lorentz_transform)`.
6. Coverage check: γ(v) also unifies the frame-dependent E/B mixing from stream 2.

Expected induction: Lorentz transform replacing Galilean, γ(v) as a latent law,
ether retracted with the MM null result as justification.

**Stream 4 — Mercury perihelion** (target: A-8)

Observations:
- Angular position of Mercury perihelion across many orbits
- Newtonian prediction (using laws from stream 1): closed ellipses, 0 residual
- Observed residual: 43 arcseconds/century (encoded as a float with units arcsec/century)
- Other planetary perturbation corrections: insufficient to account for 43″/century

Expected sequence of events:
1. Newtonian theory predicts 0 residual; observation is 43″/century. ANOMALY.
2. SR theory (from stream 3): compute prediction using Lorentz theory. Prediction ≈ 43″.
3. But SR alone (flat spacetime) does not fully explain the GR correction. RESIDUAL ANOMALY.
4. `propose_new_concept` + `propose_paradigm_shift`: generate a new concept node
   (spacetime curvature) whose curvature morphism on a 4D manifold explains the residual.
5. Wire the new node: spacetime merges spatial morphisms (stream 1) with temporal
   transform morphisms (stream 3) under the Minkowski metric structure.
6. Curvature morphism fitted by `discover_law` on the GR correction term: predicts 43″.

Expected induction: spacetime concept node, curvature morphism, GR theory compartment.

---

#### Blockers and pre-requisites for implementation

Before `ctkg/einstein/physics_streams.py` is written:

1. `ctkg/inference/theory.py`: add `symmetry_group_check(theory, transform_schema)` —
   tests whether a transformation leaves the theory's morphisms invariant.
2. `ctkg/inference/theory.py`: add cross-theory inference: given morphisms from theory A
   and theory B, compose them to produce a prediction for a shared observable.
3. `ctkg/inference/paradigm.py`: extend `propose_paradigm_shift` to wire the new concept
   node into existing morphisms (not just return an opaque new ObjectId).
4. `ctkg/core/prim_ops.py`: verify that `discover_law` at depth=5 recovers
   `1/√(1−v²/c²)` from simulated time-dilation observations. This is a unit test that
   must pass BEFORE the physics streams are written.
5. `ctkg/einstein/physics_streams.py`: implement the four observation streams above.
   All physical quantities use anonymous symbols. All stream generators take `nid: int`
   and `ctx: EvalContext` (same interface as `ctkg/einstein/streams.py`).
6. `ctkg/einstein/audit.py`: implement `GRDiscoveryAudit`, `inspect_gr_discovery`,
   `verify_gr_discovery`.
7. `ctkg/tests/test_phase9_einstein.py`: the Phase 9 test suite invoking all six
   requirements.

---

#### Pass criteria (all must hold)

1. `verify_gr_discovery(inspect_gr_discovery(mg, tm))` returns True.
2. `audit.mercury_prediction_error < 0.05` (within 5% on held-out Mercury data).
3. `audit.lorentz_factor_expr` structurally encodes `1/√(1−v²/c²)` (up to tree
   isomorphism, symbol-invariant).
4. `audit.ether_morphism is not None` and `audit.retraction_reason is not None`.
5. `audit.spacetime_concept` was created during the test run (not before it).
6. Gap (named vs. anonymous symbol table): < 5 pp across 10 seeds.
7. `seed_physics_priors(mg)` installs only permitted nodes (enforced by a static
   analysis test that reads the function body and checks for forbidden patterns).

### Phase 10: Graph-Backed Compositional Fitting (replaces SchematicLaw hardcoding) — ✓ COMPLETE

**This is the blocking step for Phase 9.**

#### The architectural defect in Phases 2–3

`parameter_fitter.py` takes a `SchematicLaw` as *input*: the caller constructs
`SchematicLaw(pattern=Expr(nid, (var('k'), var('x'))), params={'k'})` and passes it
in. This encodes the functional form (linearity, one multiplicative parameter) before
fitting runs. The fitter does OLS within that form — it never asks whether the form is
correct.

This violates both laws simultaneously:

- **Iron Law violation**: The form `k*x` is special-case handling. Fitting `γ(v)`
  would require a separate `if nonlinear: ...` branch — the same violation repeated.
- **Bitter Lesson violation**: The system does not discover that the relationship is
  multiplicative. It assumes it. If the true law is `γ(v) = 1/√(1−v²/c²)`, no amount
  of OLS over `k*x` recovers it.

The same defect propagates into the IDA benchmark: I-4 through I-8 pass a pre-specified
`SchematicLaw` to `fit_parameters`. They test parameter recovery given the form, not
form discovery from data.

#### The correct design

**Fitting is induction over the graph.** The system's graph already contains (or should
contain) primitive operation morphisms: `PRIM_MUL`, `PRIM_ADD`, `PRIM_DIV`, `PRIM_SQRT`,
`PRIM_POW`, `PRIM_NEG`, `PRIM_COMPOSE`. These are typed edges identified by `morph_type`
string labels and `ObjectId` ints — no token-name dispatch.

Law discovery becomes:

```
discover_law(observations, ctx, mg, max_depth=5) -> SchematicLaw:
    1. Enumerate compositions of primitive morphisms up to max_depth
       (beam search over the expression DAG, pruned by MDL prior)
    2. For each candidate expression C(params, x):
       a. Solve for params via OLS / gradient descent
       b. Compute residual R = MSE(C(params, x_i), y_i)
       c. Score = MDL(C) + log(R + ε)
    3. Return the minimum-score candidate as a SchematicLaw
```

OLS within step 2b is not a Bitter Lesson violation: it is a general mathematical tool
applied to fit free parameters *after* the structure is discovered. The structure
search (step 1) is what must be graph-backed and symbol-invariant.

For `f(x) = k*x`: beam search at depth 1 finds `PRIM_MUL(CONST(k), VAR(x))`, OLS
recovers k. For `γ(v) = 1/√(1−v²/c²)`: beam search at depth 5 finds
`PRIM_DIV(CONST(1), PRIM_SQRT(PRIM_SUB(CONST(1), PRIM_MUL(PRIM_POW(VAR(v), 2),
PRIM_POW(CONST(c), -2)))))`, OLS recovers c. No hardcoded hypothesis class in either
case.

#### What changes

| Component | Current (wrong) | Replacement |
|-----------|----------------|-------------|
| `parameter_fitter.py` `fit_parameters(schema, obs, ctx)` | Takes schema as input; OLS | `discover_law(obs, ctx, mg)` — beam search + OLS |
| `SchematicLaw` construction in callers | Hand-built before calling fitter | Output of `discover_law` |
| I-4 through I-8 benchmark tests | Pass pre-specified schema | Pass only `(observations, ctx, mg)` |
| `ctkg/inference/latent.py` | Calls `fit_parameters` with given schema_g, schema_h | Calls `discover_law` for both g and h |
| `parameter_fitter.py` | Kept for step 2b (OLS within a discovered structure) | Kept but no longer called with hand-built schemas |

New file: `ctkg/inference/compose_search.py` — beam search over primitive morphism
compositions, MDL scoring, returns `SchematicLaw`.

New edge type in MorphismGraph: `PRIM_OP` morphisms for each arithmetic primitive.
These are seeded into the graph at initialisation (they are domain-independent, like
the NNO successor chain).

#### Cage test

Run `discover_law` on observations from `f(x) = k*x` under 10 anonymous symbol tables.
The system must recover a structure equivalent to `PRIM_MUL(CONST(k), VAR(x))` — not
the named form, just the structural pattern — and fit k within 5% across all 10 runs.
Variance < 5 pp. Gap between named and anonymous mode < 5 pp.

Defect probe: present observations from `f(x) = k*x²` alongside `f(x) = k*x`. The
system must prefer the quadratic form for the quadratic observations (MDL+residual
lower), not default to linear for both. A system with a hardcoded linear prior fails.

---

## Summary

| Phase | Core Capability Added | I/D/A Levels Unlocked | Status |
|---|---|---|---|
| 1 | Expression trees as CTKG nodes | Prerequisite for everything | ✓ COMPLETE |
| 2 | Parametric morphisms + law discovery | I-5, I-6 | ✓ COMPLETE |
| 3 | Continuous quantities + numerical fitting | I-5 (continuous), I-6 (physics) | ✓ COMPLETE |
| 4 | Theory compartments + consistency checking | I-7, I-8, D-4 | ✓ COMPLETE |
| 5 | Closed-loop surprise + revision (redesign) | A-1/A-2 (fixed), prerequisite for A-4+ | ✓ COMPLETE |
| 6 | Full revision cycle (retract + replace) | A-4, A-7 | ✓ COMPLETE |
| 7 | Latent variable + ontology extension | A-5, A-6, A-8 | ✓ COMPLETE |
| Supp. | Multi-anomaly coverage, preservation, paradigm shift, orchestrator, synthetic streams | A-6, A-7, A-8 (standalone) | ✓ COMPLETE (supplementary) |
| 8 | Full IDA benchmark (I/D/A 4–8) | Verification phase | ✓ COMPLETE |
| 9 | Einstein Test | End-to-end integration | ✗ NOT DONE — 6 formal requirements; 5 blockers; see Phase 9 section |
| 10 | Graph-backed compositional fitting (replaces SchematicLaw hardcoding) | I-5 (real), γ(v) recovery | ✓ COMPLETE |

**Phase 10 blocks Phase 9.** Without graph-backed compositional fitting, the Einstein
test's γ(v) latent is unreachable regardless of how many other phases are complete.
Phases 1–8 are necessary prerequisites; Phase 10 is the missing architectural piece
that was concealed by the linear scaffolding.

The phase ordering is: 10 → revised I-5/I-8 IDA benchmark → Phase 9 (real Einstein test).

---

## Part IV: Compliance Tests — Bitter Lesson and Iron Law

### Design principle

Every phase test has exactly two components:

1. **The cage** — a symbol-permutation invariance test. Replace every domain token
   with a randomly drawn Unicode symbol from U+2200–U+22FF. Run the phase's core
   functionality under 10 independently drawn symbol tables. Compute the variance of
   the primary metric across the 10 runs.

   - **Pass**: variance < 5 percentage points, mean ≥ phase target.
   - **Fail**: variance ≥ 5 pp, OR mean drops more than 10 pp relative to the named-symbol
     baseline. Either failure means the implementation encoded domain knowledge that
     depends on recognising specific token strings.

   The cage catches both laws simultaneously. If the code contains `if op == 'add'`
   (Iron Law violation), it breaks when `op` is `⊕`. If the code encodes what addition
   *means* (Bitter Lesson violation), it breaks when addition is presented as an
   unrecognised symbol.

2. **The defect probe** — a targeted test that would pass even for a correct
   implementation that cheats by memorising structure, but fails for the specific
   known-bad pattern the phase is most likely to introduce. Each probe is different
   per phase; they are described below.

---

### Phase 1 — Expression Trees — ✓ COMPLETE (cage: `test_term_algebra.py` TestCage class)

**Cage:** Build a CTKG with morphisms carrying `Expr` payloads using only anonymous
operator symbols. Invoke `expr_match` and `expr_subst`. The substitution result must
be structurally identical to the named-symbol case (up to symbol renaming) across all
10 tables.

**Defect probe — tree vs. string identity:**
Present two expressions that serialise to the same string but have different tree
structures: `∀(∃, ∂)` (left-associative) vs `∀(∃, ∂)` (right-associative, same
tokens different bracketing). The implementation must treat them as distinct. A string-
based implementation that ignores tree structure matches them as equal. The probe
passes only if `expr_match` returns distinct bindings for the two trees.

**What a violation looks like:** Any code path of the form
`if str(expr) == "..."` or storing expressions as serialised token strings rather than
tree objects.

---

### Phase 2 — Parametric Morphisms — ✓ COMPLETE (cage: `test_phase2_pipeline.py` TestCage class)

**Cage:** Present a parametric family (e.g., `f(X) = k * X` for k ∈ {2,3,5}) using
anonymous symbols for `f`, `*`, and the role labels. Run `discover_parametric_law`
under 10 symbol tables. The recovered `SchematicLaw` must have the same structural
form (one free multiplicative parameter) across all runs, with fitted k values
agreeing within 5%.

**Defect probe — parameter vs. constant confusion:**
Present two families: one where k varies across examples (parametric), one where k is
the same in every example (could be a constant, not a parameter). The implementation
must classify the first as `SchematicLaw(param=k)` and the second as a ground law
`GroundLaw(constant=k)`. A system that always produces a SchematicLaw (or always a
GroundLaw) fails. The probe is run with anonymous symbols so string matching on "k"
cannot be the distinguishing signal.

**What a violation looks like:** Hardcoded parameter names (`'k'`, `'a'`, `'b'`) in
the discovery code, or special-casing known law templates (Hooke, Ohm, etc.).

---

### Phase 3 — Continuous Quantities — ✓ COMPLETE (cage: `test_phase3_pipeline.py` TestCage class)

**Cage:** Present a physical law (F = m·a) as anonymous role triples
`(⊞_value, ⊟_value, ⊕_value)` with floating-point values. Run `parameter_fitter`
under 10 symbol tables. Fitted parameter values must agree within 1% across all 10
runs.

**Defect probe — unit confusion:**
Present two quantity streams that are structurally identical (same role graph) but
differ in units by a factor of 1000 (e.g., one in metres, one in millimetres). The
system must represent them as distinct `QuantityNode` types with a conversion morphism
between them — not conflate them because their numeric values overlap. A system that
stores values as raw floats without unit tracking conflates them and fits the wrong
parameter. The probe is run with anonymous unit labels so string matching on "metres"
cannot be the signal.

**What a violation looks like:** Any `float` stored without a unit annotation, or
dimensionless arithmetic between quantities of different types.

---

### Phase 4 — Theory Compartments — ✓ COMPLETE (cage: `test_phase4_pipeline.py` TestCage class)

**Cage:** Build two anonymous theories T1 and T2 that make contradictory predictions
for observable O. All theory tokens and observable tokens are anonymous. Run
`consistency_check(T1, T2, O)` and `blame_theory(anomaly, [T1, T2])` under 10 symbol
tables. Both must correctly identify the inconsistency and attribute it to the correct
theory in all 10 runs.

**Defect probe — blame locality:**
Build a theory with 10 morphisms, exactly one of which makes a wrong prediction for O.
The other 9 morphisms all make correct predictions. Run `blame_theory`. It must
identify the single bad morphism, not return the entire theory or a random morphism.
This fails if blame is implemented as "return the whole theory" rather than as causal
tracing through the theory graph. Run with anonymous symbols so morphism labels cannot
be the attribution signal.

**What a violation looks like:** `blame_theory` returning a theory ID rather than a
morphism ID, or attribution based on string matching on morphism labels.

---

### Phase 5 — Closed-Loop Surprise and Revision — ✓ COMPLETE (cage + 5 probes: `test_phase5_pipeline.py`)

This phase has five known prior defects. Each gets its own targeted probe in addition
to the cage.

**Cage:** Present an anomalous sequence where one token has KL = 15 nats (the
predicted token is assigned probability ≈ 3×10⁻⁷). All tokens are anonymous. Run
`revise()` under 10 symbol tables. In all 10 runs, `revise()` must return non-None,
the returned candidate must target a theory-stratum morphism (not `OBS_SEQ`), and
re-scanning the anomalous sequence after revision must show surprise below threshold.

**Probe 1 — single anomaly adoption (catches defect 1):**
Present exactly one anomalous observation with KL = 15 nats. Verify `revise()` returns
non-None. Under the old code (complexity_penalty=1.0, `score <= 0` gate) this
returns None. Any implementation that scores single anomalies as 0 fails.

**Probe 2 — stratum routing (catches defect 2):**
After `revise()` returns a candidate, call `predict_next` on the anomalous prefix and
verify the prediction has changed. Under the old code the predictor ignores OBS_SEQ
edges and the prediction is identical before and after revision. Any implementation
that writes OBS_SEQ edges and calls it done fails.

**Probe 3 — cross-sequence accumulation (catches defect 3):**
Present the same anomaly pattern as 5 separate one-observation sequences (not one
5-observation sequence). Call `observe()` 5 times, then `flush()`. Verify that
`flush()` returns a candidate with evidence_count = 5. Under the old code each call
to `revise()` sees 1 observation and scores 0; accumulation does not occur. Any
implementation that processes sequences in isolation fails.

**Probe 4 — closed loop rejection (catches defect 4):**
Inject a deliberately wrong candidate (one that adds a morphism making a different
wrong prediction). Verify the loop rejects it: after `_apply(bad_candidate)`,
re-scanning shows surprise unchanged or increased, and the revision is rolled back.
Any open-loop implementation accepts it unconditionally.

**Probe 5 — causal attribution (catches defect 5):**
Build a theory with two morphisms: a correct one C and an incorrect one I. I makes a
wrong prediction. Run `revise()`. Verify the candidate targets morphism I, not
morphism C and not a surface bigram. Any implementation that generates bigram
candidates regardless of theory structure fails.

**What a violation looks like:** Any of: `return None` on single strong anomaly,
`morph_type="OBS_SEQ"` in `_apply`, no `observe()`/`flush()` API, no re-scan after
apply, candidates generated from `(prev_tok, curr_tok)` pairs without theory
attribution.

---

### Phase 6 — Full Revision Cycle — ✓ COMPLETE (cage + probe: `test_phase6_pipeline.py`)

**Cage:** Establish an anonymous theory with one morphism M that causes systematic
prediction failures (100% false positive rate on a held-out class). Run the revision
cycle under 10 symbol tables. In all 10 runs, M must be retracted, a replacement
morphism R must be adopted, and the false positive rate on the held-out class must
drop to 0% while the true positive rate on other classes stays ≥ 90%.

**Defect probe — preservation under revision (A-7 analog):**
A theory has 20 morphisms. One is wrong for class C but correct for classes A and B.
After revision: M is retracted, replacement R is adopted, classes A and B still
predict correctly (≥ 90%), class C now predicts correctly (≥ 90%). A system that
retracts carelessly (deletes M without checking what it explains correctly) will break
A and B. A system that adds without retracting (old behaviour) will keep M and fail
on C. Run with anonymous symbols.

**What a violation looks like:** `revise.py` adding edges without any `retract_morphism`
call path, or `revise.py` retracting M without first computing which correct predictions
it covers.

---

### Phase 7 — Latent Variable and Ontology Extension — ✓ COMPLETE (cage + 2 probes: `test_phase7_pipeline.py`)

**Cage:** Present 10 observations of `f(x)` where f = h ∘ g, g is unobserved, and all
symbols are anonymous. Run `hypothesise_latent` under 10 symbol tables. In all 10
runs, the recovered latent g must have the correct functional form (verified by
checking that composing the recovered g with h predicts held-out observations at
≥ 90%). Variance in the functional form across symbol tables must be < 5 pp.

**Defect probe — minimal latent (Occam):**
Present observations consistent with two latent hypotheses: g₁ (1 free parameter) and
g₂ (3 free parameters, also fits the data). The system must prefer g₁ by MDL. A
system without an MDL prior selects g₂ (it fits the training data better) and
overfits held-out observations. Run with anonymous symbols so parameter labels cannot
bias selection.

**Defect probe — new concept node identity:**
Trigger ontology extension on an anomaly that requires a new abstract concept C.
Verify that C is identified by its structural role (the morphisms connecting it to
existing objects), not by any label. Specifically: run the same anomaly with two
different symbol tables, producing concept nodes C₁ and C₂. Verify that the
subgraphs rooted at C₁ and C₂ are isomorphic (up to NodeId renaming). Any
implementation that stores a string label on the new concept node and uses it for
identity fails.

**What a violation looks like:** Latent variable hypotheses that enumerate known
physical quantities (`velocity`, `force`) rather than structural roles, or new concept
nodes identified by string labels rather than graph structure.

---

### Phase 8 — Deep IDA Benchmark — ✓ COMPLETE

The benchmark is implemented as `ctkg/tests/test_deep_ida_benchmark.py` (42 tests)
and `tests/deep_ida_benchmark.py` (standalone CLI runner with gap check).

All 15 tracks (I-4 through I-8, D-4 through D-7, A-3 through A-8) pass at 100% mean,
0.00 pp std, 0.00 pp named-vs-anonymous gap across 10 seeds.

**D-8 (dependent type derivation):** `TypedDeductionEngine` implemented in
`ctkg/inference/deduct.py`. `predict_with_types(prefix, value_context, type_constraints)`
gates BFS edge traversal on per-node value predicates — the reachable conclusion depends
on the numeric VALUE of each node, not just its name (Curry-Howard: types as propositions,
values as proofs). Tested in `ctkg/tests/test_deduct_typed.py` (15 tests, all passing),
including the SR velocity-bound analog cage across 10 anonymous symbol seeds.

---

### Phase 9 — Einstein Test — ✗ NOT DONE (scaffolding only)

The six formal requirements and seven pass criteria for Phase 9 are specified in full
in the Phase 9 section of Part III. This compliance section restates them as the
concrete test assertions that `ctkg/tests/test_phase9_einstein.py` must make.

**Cage (Requirement 6 — symbol invariance):**

The four observation streams are presented entirely in anonymous symbols (random Unicode
from U+2200–U+22FF). This is the ultimate cage: Newtonian mechanics, Maxwell's equations,
and the Michelson-Morley result are unrecognisable from their token surface. The system
must derive the structural content of special relativity (Lorentz covariance, γ factor,
spacetime unification) from the data alone.

Run the full test under 10 independent symbol tables. Variance across symbol tables
< 5 pp on each criterion. Gap (named vs. anonymous) < 5 pp.

**Discovery probe (Requirement 1):**

Before presenting any observation stream, dump `mg.all_morphisms()`. After all streams
are presented, dump again. The Lorentz transform and γ(v) morphisms must be ABSENT from
the first dump and PRESENT in the second. Any morphism whose `morph_type` encodes a
known physics law (`"LORENTZ_TRANSFORM"`, `"GAMMA_FACTOR"`, `"GR_CURVATURE"`) in the
first dump is a FAIL — these are forbidden installed priors (Requirement 5).

**Inspectability probe (Requirement 2):**

`inspect_gr_discovery(mg, tm)` must return a `GRDiscoveryAudit` with no None fields
except the optional ether fields (which may be None if the ether was generated and
immediately retracted in the same step). Any `AttributeError` or `None` on a required
field is a FAIL.

**Proof probe (Requirement 3):**

`verify_gr_discovery(inspect_gr_discovery(mg, tm))` must return True. Each of the 8
structural properties listed in Requirement 3 is a separate assert. Report which ones
fail if any do.

**Naming probe (Requirement 4):**

`ctkg/einstein/physics_streams.py` (once created) must contain the string
`"THIS MODULE IS NOT THE EINSTEIN TEST"` in its module docstring — because it defines
the observation stream generators, not the test. The test file is
`ctkg/tests/test_phase9_einstein.py`. The scaffolding files (`streams.py`,
`test_phase12_pipeline.py`, `einstein_benchmark.py`) already carry the required headers
and their compliance is tested by `grep` in the CI configuration.

**Prior knowledge probe (Requirement 5):**

A static analysis test reads the body of `seed_physics_priors(mg)` and asserts that
it contains no calls to `TOKEN_GRAPH.encode(name)` where `name` matches any known
physical concept string (`"newton"`, `"lorentz"`, `"maxwell"`, `"ether"`, `"galilean"`,
`"spacetime"`, `"gamma"`, `"curvature"`, case-insensitive). A dynamic test seeds a
fresh `MorphismGraph`, calls `seed_physics_priors(mg)`, and asserts that the graph
contains only PRIM_OP morphisms, FRAME_CONCEPT node, MEASUREMENT_SCHEMA, and
GAUSSIAN_NOISE_MODEL — no physics laws.

**Five remaining blockers before this compliance test can be run for real:**
1. `compose_search.discover_law` depth-5 recovery of `1/√(1−v²/c²)` — unit test first
2. `theory.symmetry_group_check(theory, transform_schema)` — not yet implemented
3. `theory.cross_theory_inference(theory_a, theory_b, observable)` — not yet implemented
4. `paradigm.propose_paradigm_shift` structured concept node wiring — currently opaque
5. `ctkg/einstein/physics_streams.py` — the four observation stream generators

---

### Phase 10 — Graph-Backed Compositional Fitting — ✓ COMPLETE

**This is the primary blocker for Phase 9.**

**Cage:** Present observations from an anonymous function `f(x) = k*x²` (quadratic,
not linear) using only anonymous primitive morphisms and anonymous variable names.
Run `discover_law(observations, ctx, mg)` under 10 symbol tables. In all 10 runs,
the system must recover a depth-2 composition `PRIM_MUL(CONST(k), PRIM_POW(VAR(x), 2))`
(or structurally equivalent), fit k within 5%, and prefer it over the depth-1 linear
candidate `PRIM_MUL(CONST(k), VAR(x))` by MDL+residual score. Variance < 5 pp.

**Defect probe — no linear default:**
Present observations from `f(x) = k*x` (linear) and from `f(x) = k*x²` (quadratic)
in the same run. The system must independently select depth-1 for the linear case and
depth-2 for the quadratic case. A system with a hardcoded linear prior (or that always
returns depth-1) fails for the quadratic observations.

**Defect probe — Iron Law in the search:**
Run `discover_law` where the primitive morphisms have been renamed to anonymous
Unicode symbols. The beam search must produce an isomorphic expression DAG — same
depth, same arity pattern, same fitted parameter value — regardless of what the
primitive symbols are named. Any code path of the form
`if morph_label == 'PRIM_MUL': ...` fails.

**What a violation looks like:** `compose_search.py` iterating over a hardcoded list
of named primitives (`'mul'`, `'add'`, `'sqrt'`), or `discover_law` taking a
`SchematicLaw` as an argument rather than returning one.
