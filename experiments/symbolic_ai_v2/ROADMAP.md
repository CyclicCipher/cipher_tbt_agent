# symbolic_ai_v2 — Implementation Roadmap

Status key: ✅ done · 🔧 in progress · ⬜ not started · ❌ blocked

Last updated: 2026-03-11

---

## Phase 1 — Core data structure (Graph-SEQUITUR)

- ✅ `MorphismGraph` skeleton (`symbols`, `atoms`, `edges`, `pairs`, `rules`, `_out`, `_buf`)
- ✅ `observe(value, edge_type)` — O(1) amortised per token
- ✅ Segment-boundary detection (`pairs` count = 1 → novel triple)
- ✅ Composition creation (`_create_composition`) with `rules` / `rules_inv`
- ✅ `_compress_buf_tail` — greedy tail compression enabling depth > 1
- ✅ Digram-based composition trigger (`self.digrams`) — more aggressive than triple-based
- ✅ `_rewrite_buf` — retroactive buffer rewriting on composition creation
- ✅ `predict_dist(src_id, etype)` — O(degree) via `_out` index
- ✅ `perplexity` + `perplexity_multilevel` — cross-entropy evaluation
- ✅ `generate()` — top-down Hopf coproduct expansion (morphism.py)
- ✅ Segment-boundary callbacks (`on_segment`)
- ✅ Pre-observation hooks (`on_observe`) — fires BEFORE model update
- ✅ Pruning Mechanism 1: `_prune_rdigram` — composition-triggered, provably correct
- ✅ Pruning Mechanism 2: `prune(max_singleton_age)` — stability-window singletons
- ✅ Checkpoint: `save(path)` / `load(path)` — numpy `.npz` + JSON header (7/7 tests)
- ✅ `observe_edge(src, etype, tgt)` — explicit directed edge for 2D+ topologies

## Phase 2 — Topology layer

- ✅ `EdgeTypeRegistry` — string ↔ uint8 code mapping
- ✅ `Topology` — stream_tokens() + on_boundary() callbacks
- ✅ `sequence_1d()` — path graph (one edge type: "next")
- ✅ `grid_2d(rows, cols)` — grid graph (4 edge types: N/S/E/W)
- ⬜ `relational_graph(edges)` — arbitrary labelled graph topology

## Phase 3 — Prediction layer

- ✅ `predict(mg, context_id, etype, n_top)` — ranked list with back-off
- ✅ `predict_by_value(mg, value, etype_name, topology)` — string-level wrapper
- ✅ `perplexity(mg, sequences, topology)` — atom-level bigram baseline
- ✅ `perplexity_multilevel(mg, sequences, topology)` — composition-context prediction
- ✅ **Hopf coproduct smoothing** — Phase 10a: mix prediction from composition C
  with prediction from C's right constituent; lower count → more smoothing.
  Implemented in `predict_dist()` in `morphism.py`.
- ✅ **FCA adjunction back-off** — Phase 10b: when context has no direct edges,
  query LiveCTKG type-group and use the type-level marginal.
  `LiveCTKG.atom_type_map(mg)` + `type_map` argument in `predict()`.
- ✅ CTKG type back-off integrated into `perplexity_multilevel`

## Phase 4 — Reasoning: FCA + CTKG

- ✅ `fca.py` written — `chunk_concept_matrix`, `formal_concepts`, `concepts_from_chunk`
- ✅ `ctkg_live.py` written — `LiveCTKG` with `on_segment` callback
- ✅ `ctkg_live.py` fixed — Concept/TypeDef API mismatch corrected; sheaf_merge()
  return-value pattern (not try/except) now used correctly
- ✅ FCA end-to-end test — `test_fca_on_repeated_chunk` verifies concepts_from_chunk
  finds a concept when symbols share an edge type
- ✅ `ctkg_live` end-to-end test — `ctkg_live_test.py` (7 tests): no-crash, accumulates
  ≥1 concept, TypeDefs created, Latin corpus ≥10 concepts, sheaf_merge, summary format
- ✅ `sheaf_merge` pipeline — cross-boundary merging verified; violations → sense-disambiguate
- ✅ `active_inference.py` written — `prediction_error`, `free_energy`, `expected_info_gain`
- ✅ `active_inference` connected — `ActiveInferenceTracker` wraps MorphismGraph via
  `on_observe` hook; fires BEFORE every model update; 7/7 tests pass
- ✅ `conftest.py` autouse fixture — every MorphismGraph in every test gets a tracker
  automatically, enforcing the active-inference loop structurally
- ✅ **Sense disambiguation — full algorithm** (Phase 12): `split_atom()` creates a new
  atom ID for the minority sense, redistributes historical edges by etype partition,
  updates composition rules.  `_sense_disambiguate()` now calls `split_atom()`.
  Verified: 4/4 tests pass in `sense_disambiguation_test.py`.

## Phase 5 — Memory + checkpointing

- ✅ `memory.py` written — `ChunkMemory` + CTKG reference
- ✅ Checkpoint round-trip test — edge counts, atoms, compositions, stats identical
- ✅ Incremental learning test — first-half edges preserved; n_obs accumulates;
  (note: pairs table dropped at checkpoint → exact single-pass match not guaranteed)

## Phase 6 — Tests (all must pass)

### Already passing
- ✅ `long_context_test.py` — needle in haystack @ 100 / 1K tokens (5/5)
- ✅ `arithmetic_test.py` — count → succ → add → mul compositional chain (5/5)
- ✅ `topology_test.py` — same core logic for 1D / 2D (6/6)
- ✅ `corpus_benchmark.py` — multi-language perplexity table (4/4)

### Fixed / written this sprint
- ✅ `pruning_test.py` — 8 tests: rdigram correctness, singleton prune age/count guards,
  |pairs| saturation (< 5x growth 1K→10K), >50% reduction after prune() (8/8)
- ✅ `transfer_test.py` — 6 tests: isomorphic-domain score > 0.8, random permutation
  scores lower, translated perplexity, outgoing-etype preservation, asymmetry,
  held-out score > 0.8 (6/6)
- ✅ `causal_test.py` — 7 tests: d-separation before/after do(A), graph mutilation,
  idempotence, original-graph immutability, LiveCTKG+intervene (7/7)
- ✅ `ctkg_live_test.py` — LiveCTKG: 7 tests; accumulates concepts on Latin corpus (7/7)
- ✅ `checkpoint_test.py` — 7 tests: round-trip edges/atoms/compositions/stats,
  incremental learning, JSON metadata, str+Path variants (7/7)
- ✅ `active_inference_test.py` — 7 tests: PE decreases, pe_last ≥ 0, free energy finite,
  n_obs tracking, log_line format, no divergence from direct mg, flush delegates (7/7)

### Still needed
- ⬜ `latin_test.py` — perplexity < transformer baseline on EarlyModernLatin (needs corpus)
- ⬜ Corpus benchmark: add `mg.prune()` call per language, report pair counts
- ✅ `functor_discovery_test.py` — discovers correct functor from data (Phase 9) — 5/5 pass
- ✅ `planning_test.py` — `best_action()` selects highest-EIG edge type (Phase 11) — 3/3 pass
- ⬜ `predict_smoothing_test.py` — Hopf + adjunction back-off improves perplexity (Phase 3)
- ✅ `causal_direction_test.py` — infers causal direction from observational counts (Phase 13) — 4/4 pass
- ✅ `mastery_test.py` — MasteryState tracks consolidation across training (Phase 14) — 4/4 pass
- ✅ `environment_test.py` — Environment ABC + GridWorldEnv + AgentLoop (Phase 15) — 6/6 pass
- ✅ `vision_test.py` — VisionEncoder + video_2d topology (Phase 16) — 6/6 pass

## Phase 7 — GOALS.md compliance audit

| # | Requirement | Status | Gap |
|---|-------------|--------|-----|
| 1 | Active inference from day one | ✅ | `ActiveInferenceTracker` + `on_observe` hook; `AgentLoop` closes the action half |
| 2 | Long context + STM tested | ✅ | long_context_test passes |
| 3 | Full CTKG toolkit from day one | 🔧 | FCA/sheaf/mastery/causal done; adjunctions used in prediction via back-off; functors discovered via KL matching |
| 4 | No superlinear algorithms | ✅ | core is O(n); pairs growth controlled by pruning |
| 5 | Zero hardcoded thresholds | ✅ | all decisions grounded in MDL / count comparisons |
| 6 | Drop-in transformer replacement | 🔧 | multilevel 2.79 bits/char (Latin); target < 2.0; Hopf smoothing implemented |
| 7 | Topology-agnostic | ✅ | 1D + 2D + video_2d tested |
| 8 | Bounded memory (pruning) | 🔧 | mechanisms implemented; not yet called in benchmark |

## Phase 8 — Compression target

- ⬜ `memory_growth_test.py` — train on 1K / 10K / 100K chars; plot `|pairs|` vs n;
  verify sub-linear growth after `prune()` calls
- ⬜ Wikipedia sample benchmark — run on 10MB Wikipedia sample; measure memory before
  and after `prune(500)`; target < 50MB for `|pairs|` after convergence
- ⬜ FCA abstraction — implement periodic FCA consolidation: merge concepts with
  identical edge-type distributions into a single abstract concept
- ⬜ Composition dissolution — dissolve compositions with usage_count < MDL threshold;
  free their edges and symbols

---

## Phase 9 — Functor Discovery  ← CLOSES GAP: "verify but not discover"

GOALS.md requires a cross-domain functor benchmark (Latin ↔ arithmetic).

**Algorithm**: for each atom a, its "profile" is the normalised outgoing edge
distribution `predict_dist(a_id, etype)` — the hom-object in the enriched category.
Two atoms a ∈ A and b ∈ B correspond iff their profiles are closest under KL.

- ✅ `reasoning/functor.py`:
  - `atom_profile(mg, atom_id, etype)` — normalised outgoing distribution
  - `discover_functor(mg_A, mg_B, etype_A, etype_B)` — KL-matching across domains
  - `apply_functor(mg_A, functor_map, mg_B)` — score the discovered map
  - `build_functor_levels(mg)` — coproduct-based cross-level functor map
- ✅ `functor_discovery_test.py` — 5 tests (all pass):
  1. `test_discovers_correct_mapping` — on isomorphic abcd↔0123, ≥ 3/4 atoms correct
  2. `test_discovered_functor_scores_above_random` — KL > random permutation
  3. `test_kl_decreases_with_training` — more data → tighter profiles
  4. `test_level_functor_is_coproduct` — maps comp C → (left, right) rule pair
  5. `test_functor_roundtrip` — held-out perplexity lower than random mapping

## Phase 10 — Prediction Improvements (Hopf + Adjunction back-off)
                                        ← CLOSES GAP: "discover but not use compositions/adjunctions in prediction"

### Phase 10a — Hopf coproduct smoothing  ✅

When `predict_dist(C_id, etype)` is called and C is a Composition with rule
`C = (left, e, right)`, mixes the direct prediction with `predict_dist(right_id, etype)`
weighted by `w = 1/max(direct_count, 1)` (lower count → more smoothing).

- ✅ Implemented in `predict_dist()` in `morphism.py`
- ✅ `predict_smoothing_test.py` tests 1-2 pass

### Phase 10b — FCA adjunction back-off  ✅

When context has no edges (completely unseen), falls back to the CTKG type-group
marginal over all atoms sharing the context's type-group.

- ✅ `type_map: dict[int, str] | None` argument added to `predict()` and `_marginal_dist()`
- ✅ `LiveCTKG.atom_type_map(mg)` implemented — returns `{atom_id: type_name}`
- ✅ `predict_smoothing_test.py` tests 3-4 pass

## Phase 11 — Active Inference as Planning  ← CLOSES GAP: "action half of AIF"

- ✅ `reasoning/planner.py`:
  - `epistemic_policy(mg, context_id, candidate_etypes, n_steps)` — greedy EIG lookahead
  - `simulate_observation(mg, context_id, etype, rng)` — read-only sampling
  - `active_scan(mg, start_id, topology, budget, rng)` — epistemic foraging loop
- ✅ `planning_test.py` — 3/3 pass

## Phase 12 — Sense Disambiguation — Full Algorithm  ← CLOSES GAP: "stub"

- ✅ `split_atom(atom_id, sense_a_etypes, sense_b_etypes)` in `morphism.py`:
  creates new Atom B, moves sense_b_etype edges to new atom, updates rules
- ✅ `_sense_disambiguate()` in `ctkg_live.py`: calls `split_atom()` with etype
  partition derived from the SheafViolation
- ✅ `sense_disambiguation_test.py` — 4/4 pass

## Phase 13 — Causal Direction Inference  ← CLOSES GAP: "can reason given DAG, can't discover direction"

- ✅ `reasoning/causal_discovery.py`:
  - `cond_independence(mg, x_id, z_id, y_id, etype)` — KL-based independence test
  - `skeleton(mg, etype, threshold)` — undirected adjacency from independence tests
  - `orient_edges(skeleton, mg, etype)` — v-structure orientation (PC step 2)
  - `causal_graph(mg, etype)` — full PC pipeline; returns directed edge dict
- ✅ `causal_direction_test.py` — 4/4 pass

## Phase 14 — MasteryState + Information Flow  ← CLOSES GAP: GOALS.md §3 remaining items

### Phase 14a — MasteryState per composition  ✅

- ✅ `LiveCTKG.mastery_state(mg)` — atoms: mastery=1.0; comps: `1 - 1/(1+count)`
- ✅ `LiveCTKG.frontier(mg)` — prerequisites mastered but concept not yet mastered
- ✅ `mastery_test.py` — 4/4 pass

### Phase 14b — Information flow per edge  ✅

- ✅ `LiveCTKG.information_flow(mg)` — reads `mg.rules` directly (not prerequisites,
  which are always empty since compositions don't appear in FCA observations).
  Formula: `log2(1+count) / log2(1+max_count)` per composition.
- ✅ `mastery_test.py` test 4: `test_information_flow_nonzero` passes

---

## Phase 15 — Environment ABC + AgentLoop  ← CLOSES GAP: "passive observer; no action loop"

GOALS.md §1: "perception, learning, and action are unified under one objective."
The active inference loop was implemented as a tracker (observation side) but had
no action side.  Phase 15 closes this with an application-agnostic Environment ABC
and an AgentLoop that wires observe → update → EIG-action → act.

- ✅ `environment.py`:
  - `Environment` ABC: `topology`, `observe()`, `act()`, `available_actions()`,
    `action_etypes()` (default: match action names to edge type names), `reset()`
  - `GridWorldEnv`: 3×3 neighbourhood, N/S/E/W actions, wall collision detection
  - `SequenceEnv`: wraps a token list; 'next' action maps to 'next' edge type,
    enabling full EIG-based action scoring
- ✅ `agent_loop.py`:
  - `AgentLoop(mg, env, ctkg=None)`: wires AIT + optional LiveCTKG segment callback
  - `step()`: observe → mg.observe() → EIG-choose-action → act
  - `run(n_steps)`: returns list of chosen actions
  - Action selection priority: (1) EIG over action_etypes() map, (2) EIG over
    all known etypes → index, (3) round-robin cold-start fallback
- ✅ `environment_test.py` — 6/6 pass

## Phase 16 — FovealEncoder  ← CLOSES GAP: "no visual input pipeline"

The agent needs to process image frames.  A biologically-motivated foveal
two-level pyramid replaces the original uniform-patch design, which was
insufficient for reading (16px patches can't resolve 10px-wide characters).

**Foveal parameters at 2ft from laptop:**
- Screen: 1920×1080 at 141 PPI (15.6" 1080p)
- Viewing distance: 2ft = 61cm
- 1° visual angle = 61cm × 141px/2.54cm × tan(1°) ≈ 59 pixels
- Human fovea ≈ 5° diameter → `foveal_radius = 148px` (≈150px)
- `foveal_patch = 2px`: character (16px) spans 5×8 patches → distinguishable

**Two-level pyramid:**
- Level 0 (fovea): 2*foveal_radius×2*foveal_radius crop around gaze, 2px patches → 150×150 = 22,500 tokens.  Resolves 12pt text.
- Level 1 (context): full 1920×1080 at 16px patches → 120×67 = 8,040 tokens.  Guides saccade planning.

**Topology — 8 edge types (fit in uint8):**
- right(0), left(1), down(2), up(3) — spatial, same level
- zoom_in(4), zoom_out(5) — cross-level at same screen position
- prev_frame(6), next_frame(7) — temporal across frames

**Saccade learning:**
`zoom_in` edges teach the MorphismGraph: "context patch X at screen location
predicts foveal detail Y".  `best_saccade_target(mg)` returns the screen
position of the context patch with the highest EIG for zoom_in, which the
AgentLoop uses to direct gaze via `set_gaze(x, y)`.

- ✅ `vision/__init__.py`
- ✅ `vision/encoder.py`:
  - `foveal_2d()` topology: 8 edge types (right/left/down/up/zoom_in/zoom_out/prev_frame/next_frame)
  - `FovealEncoder(foveal_radius=150, foveal_patch=2, context_scale=8, n_color_levels=4, n_frames=3)`:
    - `encode_fovea(frame)` → foveal crop patch grid
    - `encode_context(frame)` → full-frame coarse grid
    - `stream_edges(frame)` → spatial + zoom + temporal edges for `mg.observe_edge()`
    - `set_gaze(x, y)` → direct foveal attention
    - `best_saccade_target(mg)` → screen (x, y) with highest EIG for zoom_in
    - `reset()` → clear frame buffers at scene/episode boundaries
    - `for_screen(width, height, dpi, dist_cm, foveal_degrees)` — classmethod that computes foveal_radius from physical parameters
  - GPU path: `torch.nn.functional.avg_pool2d` (~10× faster on 1080p frames)
  - Numpy path: reshape + mean (fast enough for 60fps on CPU)
  - Pure-Python fallback
  - `VisionEncoder` (legacy uniform patches) retained for non-reading tasks
- ✅ `vision_test.py` — 10/10 pass

---

## Summary: Gap → Phase mapping

| Gap (from gap analysis) | Phase | Status |
|-------------------------|-------|--------|
| Verify but not discover: cross-domain functors | Phase 9 | ✅ |
| Discover but not use: FCA adjunctions in prediction | Phase 10b | ✅ |
| Discover but not use: Hopf coproduct smoothing | Phase 10a | ✅ |
| Verify but not discover: causal direction | Phase 13 | ✅ |
| Discover but not use: EIG / best_action planning | Phase 11 | ✅ |
| Stub: sense disambiguation | Phase 12 | ✅ |
| Not implemented: MasteryState | Phase 14a | ✅ |
| Not implemented: information flow per edge | Phase 14b | ✅ |
| Passive observer; no action loop | Phase 15 | ✅ |
| No visual input pipeline | Phase 16 | ✅ |

---

## Phase 17 — Improving the Model Through Algebraic Generalization

**Motivation:** The math benchmark (2026-03-10) revealed a fundamental architectural
limitation: Graph-SEQUITUR achieves 0% test accuracy on successor, subtraction, powers,
and conservation law tasks.  Root cause: the architecture is a lossless compressor with
no variable binding.  It memorises ground instances (`succ 3 eq 4`) but cannot apply
the rule (`succ(N) = N+1`) to a novel input N.  Gary Marcus's critique applies directly.

The benchmark showed the split clearly:

| Task | Train Acc | Test Acc | Root cause of gap |
|------|-----------|----------|-------------------|
| counting | 94.4% | 100% | — (type back-off covers it) |
| successor | 100% | 0% | variable binding required |
| addition | 97.4% | 55.3% | commutativity covers ~half; rest needs rules |
| subtraction | 81.2% | 0% | no commutativity overlap; pure memorisation |
| multiplication | 98.0% | 57.9% | commutativity covers ~half |
| powers | 83.8% | 0% | no structural overlap |
| integrals | 92.9% | 80.0% | FTC constant + dx generalise via type back-off |
| conservation | 92.4% | 0% | requires algebraic constraint reasoning |
| bernoulli | 51.1% | 13.3% | pattern memorisation only |

This phase adds three capabilities in order, with a benchmark run after each:

---

### Phase 17a — Extended Category-Theoretic Approaches  ⬜

**Goal:** Exhaust what pure category theory can give us before adding variable binding.
Scientifically important: if CT alone closes the gap, we avoid adding machinery.

**1. Endofunctor discovery (within one domain)**

Currently `discover_functor` only runs between two *separate* MorphismGraphs.
Add `discover_endofunctor(mg, etype)` that maps MG → MG itself.  If an operator
acts as an endofunctor on the number type-group (e.g. succ shifts every number to its
successor), the algorithm recovers this from training data by:
- Restricting attention to atoms belonging to the same CTKG type-group
- Profile-matching those atoms to themselves under the operator's induced permutation
- Returning the discovered permutation as a partial endofunctor map

**2. Natural transformation discovery**

A natural transformation α: F → G between two discovered functors is a coherent
family of morphisms that converts one way of mapping into another.  Practically: find
pairs of operators whose induced atom maps are related by a systematic offset.
`succ` and `pred` are natural transformations of the identity endofunctor on numbers —
one shifts +1, the other -1.  `add(·, k)` and `sub(·, k)` are a natural transformation
pair parameterised by k.

Algorithm: for each pair (op_A, op_B), check if `f_B = f_A ∘ constant_shift` fits
the observed (input, output) pairs for both operators.  If yes, register an
`Adjunction` node in the CTKG connecting the two operators.

**3. Automatic adjunction detection**

`add` and `sub` form an adjunction (add ⊣ sub in the appropriate sense).
Currently adjunctions are hand-specified in the CTKG DSL.  This step discovers them
from data: for operators F and G, check if `G(F(A, B), B) ≈ A` holds on all
observed training instances.  If yes, register `F ⊣ G` as a CTKG Adjunction.

This closes the gap in GOALS.md §3: adjunctions should be *discovered*, not only
verified after manual specification.

**4. Wire discovered structures into the prediction back-off chain**

Currently the back-off chain is:
  1. Hopf-smoothed composition context
  2. FCA type back-off on composition
  3. Raw atom bigram
  4. FCA type back-off on atom
  5. Corpus-wide marginal

After this phase, add a new level 0 (before Hopf smoothing):
  0. Endofunctor application: if context matches a known endofunctor pattern,
     apply the functor map to predict the output atom directly.

**5. Higher-order relational reasoning (relations between relations)**

Penn, Holyoak & Povinelli (2008) argue this is a uniquely human cognitive capacity
and may be what makes variable binding feel natural.  In our framework, a relation
between relations is a natural transformation between two functors.  Concretely:
"addition and multiplication are related by distributivity" is a natural transformation
`α: add ∘ (id × mul) → mul ∘ (add × id)`.  Discovering such second-order relations
requires:
- First discovering the first-order functors (add, mul as endofunctors on numbers)
- Then checking structural coherence conditions between pairs of functors
- Registering the coherence as a CTKG `Interface` or `Adjunction` node

This may emerge automatically from natural transformation discovery if both functors
are found first.  If not, add explicit second-order search after Phase 17a step 2.

**Deliverable:** Run math benchmark after implementing 17a.  Record test accuracy
for all 11 levels.  Document whether the CT approaches alone close the gap on any
of the 0%-test-accuracy tasks.

---

### Phase 17b — Variable Binding (Gary Marcus Approach)  ✅ (revised)

**Goal:** Implement the minimum machinery for algebraic generalization that the
developmental evidence supports: the *mechanism* for variable binding is innate;
the specific rules are learned from data.

**Core design principle (representation independence):**  The algorithm must work
for ANY domain — Danganronpa (atoms named `'Kyoko'`, `'gymnasium'`), mathematics
(atoms named `'0'`, `'1'`, `'2'`), or music (atoms named `'C4'`, `'D4'`).
Calling `int(atom.value)` is a naming accident: it works only because the math
corpus happened to name atoms after the integers they represent.  The correct
approach is **graph-structural rank derivation** — read the ordinal structure
from the graph topology, not from atom names.

**1. Template extraction**

For each operator atom `op` observed in training:
- Collect all composition chains matching the structural pattern `[op, *, eq, *]`
  (or `[op, *, *, eq, *]` for binary operators, etc.)
- Identify fixed positions (same atom in every chain) vs. variable positions (differ)
- Output: `Frame(op, fixed_positions, variable_slots)`, e.g.
  `Frame('succ', {0: 'succ', 2: 'eq'}, {1: '?N', 3: '?M'})`

**2. Graph-structural rank derivation**

Before fitting any hypothesis, the system must derive an ordinal structure from
the graph.  The `succ` endofunctor map `{atom_0→atom_1, atom_1→atom_2, ...}` IS
the total order on atoms.  **`_is_chain(ef_map)`** checks that the map is a
simple acyclic chain (no branching, no cycles).  **`_build_rank_map(ef_map)`**
walks the chain to assign rank 0, 1, 2, ... to each atom, producing:
- `rank_map: {atom_id → rank}` — position in the chain
- `inv_rank_map: {rank → atom_id}` — inverse lookup

These maps are stored as `mg._rank_map` and `mg._inv_rank_map` after
`build_variable_binding` runs.  An atom "has ordinal type" if and only if it
appears in `rank_map` — this is the type membership test (see Phase 19 Level 1).

**3. Functional relationship discovery (symbolic regression on rank pairs)**

For each operator whose endofunctor map is chain-structured, extract `(rank_N,
rank_M)` pairs and fit a hypothesis.  The hypothesis space operates on **ranks**,
not on atom names:

| Form | Example (rank space) |
|------|---------------------|
| `M = N + k` | successor: k=1, predecessor: k=-1 |
| `M = k * N` | doubling |
| `M = N ^ k` | powers |
| `M = isqrt(N)` | integer square root |
| `M = N // k` | floor division |

Binary operators: extract `(rank_N1, rank_N2, rank_M)` triples from the
endofunctor map (using `mg._rank_map` to convert all three operands).
Fit `M = f(N1, N2)`.

Search strategy: try forms in order, accept first form with zero residual on all
training rank-pairs.  Reject if no form fits (too complex for this phase).

**4. RuleStore — rule crystallization and registration**

Store discovered rules as `AlgebraicRule(op, arity, formula, fn, evidence)`.
Persist alongside the `MorphismGraph` (include in checkpoint).  All arithmetic
`fn` lambdas operate in rank space; back-conversion uses `mg._inv_rank_map`.

**5. Unification at prediction time**

At prediction time, look up argument atom_ids → ranks via `mg._rank_map`, apply
the rule lambda, and convert the result rank back to an atom via
`mg._inv_rank_map`.  If the result rank is not in `inv_rank_map` (novel output),
call `mg.get_or_create_atom(str(rank), coarse_type)` to create a predicted atom.

```
query: succ 5 eq ?
→ arg_id = mg.atoms['5']
→ arg_rank = mg._rank_map[arg_id]   # = 5 (graph-structural, not int('5'))
→ result_rank = rule.fn(arg_rank)   # = 6
→ result_id = mg._inv_rank_map.get(6) or mg.get_or_create_atom('6', 'num')
→ return {result_id: rule.confidence}
```

**6. Higher-order relational reasoning (Phase 17a)**

Covered by adjunction discovery in `rule_store.py`.  `succ ⊣ pred` and
`add ⊣ sub` are discovered from the endofunctor maps without requiring
integer interpretation.

**Deliverable:** Run math benchmark after implementing 17b.  Record test accuracy
for all 11 levels.  The key test: does successor reach >0% test accuracy?  Does
conservation reach >0%?  Document which tasks are fixed by variable binding alone
and which still require additional machinery.

---

### Phase 17c — Cross-Domain Transfer Benchmark  ⬜

**Goal:** Measure whether mathematical knowledge helps linguistic compression and
vice versa.  This tests the central architectural claim — one algorithm for all
token domains — against the empirical question of whether the knowledge representations
actually transfer.

**Experiment design:**

Two models, identical hyperparameters, trained in opposite orders:

| Model | Training order |
|-------|---------------|
| Model A | Corpus benchmark (multilingual text) FIRST, then math benchmark |
| Model B | Math benchmark FIRST, then corpus benchmark |

For each model, collect:
- Final perplexity on corpus benchmark held-out test sets (all languages)
- Final perplexity on math benchmark test sets (all 11 levels)
- Test accuracy on math benchmark (all 11 levels)
- Number of compositions, max composition level, CTKG concept count

**Hypotheses:**

1. If the MorphismGraph is truly domain-agnostic, the order should not matter:
   both models should reach similar perplexity on both benchmarks.

2. If mathematical structure helps linguistic compression (or vice versa),
   the model trained on that domain first should show lower perplexity on the
   second domain (positive transfer).

3. If the domains interfere (e.g. mathematical tokens corrupt linguistic
   composition hierarchies), the model trained on that domain first should show
   *higher* perplexity on the second domain (catastrophic interference).

**Procedure:**

```python
# Model A: corpus first
mg_A = MorphismGraph(topology=topo)
for lang_seqs in corpus_benchmark_data:
    for seq in lang_seqs:
        mg_A.observe_sequence(seq, topo)
mg_A.prune()
# Evaluate corpus benchmark
# Then train on math:
for seq in math_benchmark_train:
    mg_A.observe_sequence(seq, topo)
mg_A.prune()
# Evaluate both benchmarks

# Model B: math first (symmetric)
```

Note: the two benchmarks use different topologies (corpus = sequence_1d or
math_topology; math = math_topology).  Need to decide: use a shared topology
for the transfer experiment, or use separate MorphismGraphs that share only
the CTKG.  The former tests full integration; the latter tests knowledge-level
transfer only.

**Deliverable:** Table comparing Model A vs Model B on both benchmarks.
Document evidence for positive transfer, interference, or independence.

---

## Phase 17 — Current results (as of 2026-03-10)

Phase 17a and 17b are **complete**.  Summary of what was achieved and what remains:

| Task | Train Acc | Test Acc | Status |
|------|-----------|----------|--------|
| counting | 94.4% | 100% | ✅ type back-off |
| successor | 100% | 100% | ✅ variable binding (Phase 17b) |
| addition | 97.4% | 100% | ✅ variable binding (Phase 17b) |
| subtraction | 81.2% | 100% | ✅ variable binding (Phase 17b) |
| multiplication | 98.0% | 100% | ✅ variable binding (Phase 17b) |
| powers | 83.8% | 92.3% | ✅ variable binding (Phase 17b) |
| integrals | 92.9% | 80.0% | 🔧 partial; multi-token results needed |
| linear_eval | n/a | 6.9% | ❌ rule chaining needed |
| derivatives | n/a | 33.3% | ❌ multi-token results needed |
| conservation | 92.4% | 0% | ❌ backward chaining needed |
| bernoulli | 51.1% | 13.3% | ❌ all three capabilities needed |

**8 algebraic rules discovered:** succ (M=N+1), pred (M=N-1), sq (M=N^2),
sqrt (M=isqrt(N)), add (M=N1+N2), sub (M=N1-N2), mul (M=N1*N2), pow (M=N1^N2).
**1 adjunction discovered:** sub ⊣ add (90% confidence).
**137/137 tests passing.**

The three remaining failures are architecturally distinct and map to three missing
capabilities — each requiring a new component in the generative model hierarchy.
Phase 17c (cross-domain transfer benchmark) is **deferred** until after Phase 18 and
Phase 19 are complete; integer interpretation in rule learning must be addressed first.

---

## Phase 18 — Multi-Step Reasoning (Three Missing Capabilities)

**Motivation:** After Phase 17, the architecture can apply a single algebraic rule
to a single novel input.  The ceiling has been hit.  The remaining tasks —
`linear_eval`, `derivatives`, `conservation`, `bernoulli` — require three capabilities
that go beyond single-rule application.  Each maps to a distinct biological system and
a distinct active-inference construct.

**Active inference framing (key insight):**  In active inference, a goal is a *prior
expectation* over a future observation.  When that expectation is violated the system
either (a) updates its generative model (learning) or (b) takes action to bring
reality into conformity with the expectation.  Some expectations are not revisable —
e.g. logical consistency, conservation laws — and a violation there should propagate
a strong error signal rather than a model update.  The goal stack below is exactly
this hierarchy of prior expectations, stratified by revisability.

The three capabilities map to three levels of the generative model hierarchy:

| Level | Capability | Biological analog | Active inference analog |
|-------|-----------|-------------------|------------------------|
| 2 | Rule chaining | PFC working memory + BG gating | Prior expectations over intermediate states |
| 3 | Sequence goals | Hippocampal theta-gamma binding | Prior over a *sequence* of future observations |
| 4 | Backward chaining | ACC + cerebellum inverse model | Goal as a hard prior; action = rule inversion |

All three share a single data structure — the **GoalStack** — and extend the existing
`ActiveInferenceTracker` rather than replacing it.

---

### Phase 18a — Rule Chaining (GoalStack + RuleChainer)

**Root cause of failure:** `linear_eval` presents `[linear, x=5, eq, ?]`.  The answer
requires two sequential rule applications: `mul(2, 5) = 10`, then `add(10, 3) = 13`.
The current frame matcher fires once and returns nothing (frame is 7 tokens wide,
not 3 or 4).

**Biological analog:** Prefrontal cortex maintains intermediate results in working
memory.  Basal ganglia gate each result into the next slot.  Neither structure is
needed for single-rule inference; both are needed for multi-step chains.

**Active inference analog:** Each intermediate result is a prior expectation at level 2
of the generative model.  The system predicts "if I apply rule R₁ to current input,
I expect intermediate state S₁; if I then apply R₂ to S₁, I expect final state S₂
= the observed answer."  Prediction error at level 2 is the mismatch between
predicted S₂ and the observed answer token.

**Key design decision — revisable vs. fixed priors:**  Algebraic rules are revisable
(evidence-weighted).  But the *goal* at the top of the stack — "the system must produce
the correct answer" — is not revisable.  A mismatch at the goal level triggers rule
revision, not goal revision.  This asymmetry is why learning happens at the rule level
and not the goal level.

**Implementation plan:**

```
reasoning/goal_stack.py
  GoalState(target_id: int, confidence: float)
  GoalStack:
    push(goal: GoalState)
    pop() -> GoalState | None
    peek() -> GoalState | None

reasoning/rule_chainer.py
  RuleChainer(mg: MorphismGraph):
    chain(atom_values: list[str], max_depth: int = 5) -> dict[int, float]
      # Attempt to reach a result by chaining known algebraic rules.
      # 1. Parse atom_values into operator + arguments (supports nested frames).
      # 2. Evaluate each argument recursively (DFS, depth-limited).
      # 3. Apply outer rule to evaluated arguments.
      # 4. Return {result_atom_id: confidence} or {} if no chain found.
    _eval(expr: list[str], depth: int) -> int | None
      # Recursive evaluator: try variable binding first, then recurse on sub-frames.
```

**Back-off chain extension:**  Add level 0d before frame match:
```
0a. Endofunctor table (seen inputs, certainty 1.0)
0b. Variable binding via ctx_id decomposition
0c. Frame match on raw atom buffer (3 or 4 tokens)
0d. Rule chainer (arbitrary-depth, depth-limited to 5)    ← NEW
1.  Hopf-smoothed composition context
...
```

**Deliverable:** `reasoning/rule_chainer.py` + `tests/rule_chaining_test.py` (≥5 tests).
Key tests: `linear_eval` reaches > 50% test accuracy; chaining correctly decomposes
two-step expressions; depth limit prevents infinite recursion; single-rule case
produces same result as frame matcher.

---

### Phase 18b — Sequence Goals (Multi-token Results)

**Root cause of failure:** `derivatives` and `integrals` produce multi-token answers:
`d/dx(x²) = [mul, 2, x]`.  The current prediction interface returns `dict[int, float]`
— a distribution over *single* atoms.  A three-token answer cannot be expressed.

**Biological analog:** Hippocampal theta-gamma coupling binds a sequence of cortical
firing events into a single episodic chunk that can be retrieved and replayed as a
unit.  The chunk has identity as a whole; its internal structure is accessible on
demand.

**Active inference analog:** A sequence goal is a prior expectation over a *sequence*
of future observations.  The system predicts "I expect to observe tokens [mul, 2, x]
in order."  Each token in the sequence generates a prediction error at level 0; the
sequence as a whole generates a prediction error at level 3.

**Implementation plan:**

```
core/predict.py — extend prediction interface:
  SequenceGoal(atoms: list[int], confidence: float)
  _predict_sequence(mg, atom_values) -> SequenceGoal | None
    # Wraps rule_chainer; if result is itself a Composition (not a leaf Atom),
    # decompose it into its constituent atom sequence via _decompose().
    # Returns SequenceGoal with atoms = [mul_id, 2_id, x_id] for d/dx(x²).

  # Extend perplexity_multilevel / _accuracy to handle SequenceGoal:
  # - Compare predicted token sequence against actual next-N tokens
  # - Credit = fraction of sequence matched correctly
```

**Composition as sequence:** The MorphismGraph already stores composition hierarchies.
When a result is a composition node C with rule `C = (left, e, right)`, `_decompose(C)`
yields its leaf atoms in order.  Phase 18b reuses this: if `rule_chainer` produces a
composition ID as its result, `_decompose` unpacks it into the answer sequence.

**Format decision:** Multi-token answers appear in the training data as composition
nodes (the model has already learned `[mul, 2, x]` as a composition during training).
The question is whether the *training* examples include enough composed-expression
answers to drive learning.  If the training data uses only atom-level answers, Phase
18b also needs a mechanism to form composition nodes for answers on-the-fly.

**Deliverable:** `tests/sequence_goal_test.py` (≥5 tests).  Key tests: `derivatives`
reaches > 60% test accuracy; multi-token answer correctly matches next-N ground-truth
tokens; sequence goal confidence propagates correctly; `_decompose` round-trip.

---

### Phase 18c — Backward Chaining (Adjunction-based Constraint Solving)

**Root cause of failure:** `conservation` presents `A + B = C + ?`.  This is a
constraint satisfaction problem: given `add(A, B) = add(C, ?)`, solve for `?`.
Algebraically: `? = (A + B) - C`.  The adjunction `sub ⊣ add` was already discovered
in Phase 17a — we know that `sub` is the right adjoint of `add`.  The system just
has no engine to *use* that adjunction for backward inference.

**Biological analog:** Anterior cingulate cortex detects the mismatch between expected
and observed outcome.  The cerebellum maintains an inverse model: given the desired
motor output, compute the muscle command needed to achieve it.  In our context: given
the desired symbolic output (the conserved quantity), compute the input that the
operator must have received.

**Active inference analog:** The goal at the top of the stack is a *hard prior* — a
conservation law is not revisable.  The system must act to satisfy the constraint by
finding the missing value.  This is pure action, zero learning: the generative model's
inverse is used to compute what the world must be like to satisfy the prior.

**Implementation plan:**

```
reasoning/backward_chainer.py
  BackwardChainer(mg: MorphismGraph):
    solve(frame: list[str], unknown_pos: int) -> dict[int, float]
      # frame = ['add', '3', '?', 'eq', '7']  (? at position 2)
      # 1. Identify operator from frame[0].
      # 2. Look up adjunction: if op = 'add', adjoint = 'sub'.
      # 3. Rearrange: compute known_result and known_args.
      # 4. Apply adjoint rule to solve for unknown.
      # 5. Return {result_atom_id: confidence} or {} if no adjunction known.
    _find_adjunction(op: str) -> str | None
      # Query mg._adjunctions for known (F, G) pairs.
      # Returns the name of the right adjoint of op, or None.
```

**Back-off chain extension:**  Add level 0e after rule chaining:
```
0c. Frame match (3 or 4 tokens, forward)
0d. Rule chaining (arbitrary-depth)
0e. Backward chaining (constraint solving via adjunction)    ← NEW
```

**Detection of constraint frames:**  A constraint frame is identified by the presence
of more than one `eq` marker or by a wildcard/unknown token in a non-final position.
The frame parser in `backward_chainer.py` scans `atom_buf` for this pattern.

**Deliverable:** `reasoning/backward_chainer.py` + `tests/backward_chaining_test.py`
(≥5 tests).  Key tests: `conservation` reaches > 50% test accuracy; simple one-step
constraint (`3 + ? = 7`) solved correctly; adjunction lookup from `mg._adjunctions`;
unknown at arg1 position vs arg2 position both handled; no adjunction found → returns {}.

---

## Phase 19 — Three Missing Levels of Algebraic Generalization

**Background:** The MorphismGraph composition level already implements Marcus's algebraic
mechanism: edge types are typed variables, Graph-SEQUITUR fires on edge-type pairs
(not specific atoms), and compositions are discovered universally.  But *prediction*
falls back to statistical edge counts.  Three structural levels were omitted that are
required for Marcus-style algebraic generalization throughout the prediction hierarchy.

---

### Level 1 — Type Membership (atoms must know their formal concept)  ✅

**The gap:** Atoms carry only a string value.  They do not know which formal concept
(FCA extent) they belong to.  The FCA already computes concepts from the graph
structure, but the results are not stored back on the atoms.  This means rules cannot
be universally quantified over types: you cannot say "for any atom of type VERB_ROOT,
apply rule R" without re-running FCA at query time.

Marcus's type-token distinction requires each token (individual atom) to be registered
as a member of its type (formal concept) at creation time.  Only then can a rule
defined over a type variable generalise immediately to any new token of that type.

**Implemented (2026-03-11):**

- `Atom.concept_ids: frozenset[int]` — added to `Atom` dataclass.  Defaults to `frozenset()`.
  Populated by `LiveCTKG._write_concept_ids()` after each FCA pass.
- `mg.fca_type_id(attr_set)` — assigns stable integer IDs to FCA attribute sets
  (frozenset of edge types).  Same attr_set → same ID across all chunks and time.
- `mg.atoms_of_type(type_id)` — returns all atoms currently registered as members.
- `LiveCTKG._write_concept_ids(concepts, mg)` — called at the start of `on_segment()`,
  before building the local CTKG.  Writes type IDs onto every atom in each concept's
  extent.  Concept IDs accumulate: an atom observed in multiple chunks with different
  edge-type sets receives all corresponding type IDs.
- `mg._fca_type_ids: dict[frozenset, int]` and `mg._fca_types: list[frozenset]` —
  FCA type registry on MorphismGraph.

**Two type layers:**

- Ordinal type (`atom_id in mg._rank_map`): set by `build_variable_binding()`.
  Derived from graph topology: whether the atom belongs to a discovered chain.
  Valid for any domain — successor chain in math, chapter sequence in Danganronpa,
  beat sequence in music, etc.
- Fine type (`atom.concept_ids`): ✅ NOW IMPLEMENTED.  Set by FCA after each segment
  boundary.  Multiple concepts are valid (a concept lattice is not a partition).
  An atom can simultaneously belong to 'word' (etype={0}) and 'content-word'
  (etype={0,1}) — both type IDs are stored.

**Rule constraint:** `fit_rule` checks whether the endofunctor's argument atoms
are in `mg._rank_map`.  Arithmetic hypotheses (H1–H5) are only attempted when
at least `min_pairs` atoms in the ef_map have assigned ranks.  For all other atoms
no hypothesis is attempted — the MorphismGraph statistical associations are the
correct and complete representation (Marcus's "associative memory" subsystem for
non-algebraic domains).  This is not an enumeration of more hypotheses; it is a
*guard* that prevents spurious rules.

```python
def fit_rule(op, ef_map, mg, rank_map: dict | None = None) -> AlgebraicRule | None:
    if rank_map is None:
        rank_map = getattr(mg, '_rank_map', {})
    # Gate: only proceed if we have rank-mapped atom pairs (ordinal type check).
    # If no atoms in ef_map have ranks, this operator is non-ordinal; return None.
    # The MorphismGraph edge counts (associative memory) are the correct fallback.
    is_binary = isinstance(next(iter(ef_map)), tuple)
    if is_binary:
        ranked_triples = [(rank_map[a1], rank_map[a2], rank_map[r])
                         for (a1, a2), r in ef_map.items()
                         if a1 in rank_map and a2 in rank_map and r in rank_map]
        result = _fit_binary(ranked_triples)
    else:
        ranked_pairs = [(rank_map[a], rank_map[r])
                        for a, r in ef_map.items()
                        if a in rank_map and r in rank_map]
        result = _fit_unary(ranked_pairs)
    if result:
        formula, fn = result
        return AlgebraicRule(op=op, arity=2 if is_binary else 1,
                             formula=formula, fn=fn,
                             evidence=len(ranked_triples if is_binary else ranked_pairs))
    return None
```

**Revisable vs. fixed (active inference):**
- Numeric rules: revisable — evidence count updated; rule replaced if falsified.
- Homeostatic priors (conservation laws, logical axioms): **not revisable** — mark
  with `fixed=True` on `AlgebraicRule`.  A falsifying observation generates a
  `SheafViolation` (sense disambiguation) rather than a model update.

---

### Level 2 — Relational Rules (structural equality between slot positions)

**The gap:** Phase 17b's `AlgebraicRule` encodes `f(content_of_slot) → result_value`.
This requires looking *inside* the atom (reading its integer value).  But Marcus's
foundational example — A-B-A — is a rule that requires *zero* content knowledge:
`slot[0] == slot[2]`.  It is a constraint on **positions in a composition**, not on
values.  Similarly:

| Rule | Formula | Content knowledge needed? |
|------|---------|--------------------------|
| Identity | `f(X) = X` | None — purely structural |
| A-B-A | `slot[0] == slot[2]` | None — equality between positions |
| Commutativity | `f(a,b) = f(b,a)` | None — permutation of argument positions |
| Successor | `f(X) = succ(X)` | Requires integer value of X |
| Past tense | `f(X) = X + "ed"` | Requires string content of X |

Rules in the top half of this table are the *most primitive* algebraic rules — they
sit *below* Phase 17b in the hierarchy (they need less information) and are the level
closest to Marcus's infant experiments.

**Fix:** Add a `RelationalRule` dataclass alongside `AlgebraicRule`.  A `RelationalRule`
is a predicate over the *positions* of atoms within a composition frame, with no
reference to atom contents.

```python
@dataclass
class RelationalRule:
    """A structural rule over slot positions in a composition frame.

    op         : the operator atom value
    arity      : number of argument slots (1 or 2)
    relation   : one of 'identity', 'aba', 'commutative', 'constant'
    evidence   : number of training frames that confirm this rule
    """
    op:       str
    arity:    int
    relation: str          # 'identity' | 'aba' | 'commutative' | 'constant'
    constant_id: int | None = None   # for 'constant' relation: the fixed result atom
```

**Discovery (`fit_relational_rule`):**

```python
def fit_relational_rule(op, ef_map, mg) -> RelationalRule | None:
    if not ef_map or len(ef_map) < 2:
        return None
    is_binary = isinstance(next(iter(ef_map)), tuple)

    if not is_binary:
        # Unary: check identity (result == arg for all pairs)
        if all(res_id == arg_id for arg_id, res_id in ef_map.items()):
            return RelationalRule(op=op, arity=1, relation='identity', evidence=len(ef_map))
        # Unary: check constant (same result for all args)
        results = set(ef_map.values())
        if len(results) == 1:
            return RelationalRule(op=op, arity=1, relation='constant',
                                  constant_id=next(iter(results)), evidence=len(ef_map))
    else:
        # Binary: check commutativity (f(a,b) = f(b,a))
        if all(ef_map.get((a2, a1)) == res
               for (a1, a2), res in ef_map.items()
               if (a2, a1) in ef_map):
            return RelationalRule(op=op, arity=2, relation='commutative', evidence=len(ef_map))

    return None
```

**Prediction (`predict_via_relational_rule`):** Consult `mg._relational_rules` (a
`dict[str, RelationalRule]`) during prediction.  For `identity`: result = arg.  For
`constant`: result = `constant_id`.  For `commutative`: if `(arg2, arg1)` is in the
endofunctor map, use that result.

**Storage:** `build_variable_binding` calls `fit_relational_rule` for each operator and
stores results in `mg._relational_rules`.  These are checked *before* numeric rules in
the back-off chain (they need less information and are more reliable).

---

### Level 3 — Novel Atom Generation (predict beyond the training vocabulary)

**The gap:** When `AlgebraicRule` or `RelationalRule` computes a result value not yet
in `mg.atoms`, the system returns `{}`.  Marcus's critical capability — applying a rule
to a truly novel input and producing a novel output — is blocked.

This is the level that distinguishes genuine algebraic generalization from
sophisticated pattern matching.  `past(rick) = ricked` is only meaningful if "ricked"
can be created when the rule fires on an unseen verb.  For numeric rules: `succ(999)`
should produce `1000` even if 1000 was never seen in training.

**Fix:** Add `mg.get_or_create_atom(value: str, coarse_type: str) -> int`.  When a
rule fires and the result string is not in `mg.atoms`, call `get_or_create_atom` to
register it as a new Atom node.  The new atom has:
- `value = str(result)`
- `level = 0` (leaf)
- `coarse_type` inherited from the result's expected edge type (from topology)
- Zero edges initially — it is a *predicted* atom, not an observed one

Predicted atoms are tagged with `predicted=True` so they can be distinguished from
observed atoms.  They participate in prediction but not in endofunctor map building
(which requires observed input-output pairs).

```python
# In AlgebraicRule evaluation:
result_id = mg.atoms.get(str(result_val))
if result_id is None:
    result_id = mg.get_or_create_atom(str(result_val), coarse_type='num')
return {result_id: rule.confidence}
```

**Guard:** Only create atoms when `abs(result_val) <= 10_000` (same cap as
`RuleChainer`) and when the rule's evidence is above a minimum threshold (≥ 3 pairs),
to prevent runaway atom creation from spurious rules.

---

### Complete updated back-off chain (Phases 17–19)

```
0a. Endofunctor table       — exact composition match (certainty 1.0)
0b. Relational rule         — slot-position rule (identity, constant, commutative) ← NEW Level 2
0c. Variable binding        — AlgebraicRule on ctx_id decomposition (numeric)
0d. Frame match             — AlgebraicRule on raw atom_buf strings (numeric)
0e. Rule chaining           — recursive prefix evaluation (Phase 18a)
0f. Backward chaining       — adjunction constraint solving (Phase 18c)
1.  Hopf-smoothed edge counts
2.  FCA type back-off
3.  Bigram
4.  Type back-off on atom
5.  Marginal
```

Novel atom creation (Level 3) applies at steps 0c–0e whenever the computed result is
not in `mg.atoms`.

---

### Deliverables

**Modified files:**
- `core/morphism.py` — add `coarse_type` field to `Atom`; add `get_or_create_atom()`;
  add `predicted: bool = False` flag
- `reasoning/variable_binding.py` — add `RelationalRule`; add `fit_relational_rule()`;
  add `predict_via_relational_rule()`; gate `_fit_numeric` on `coarse_type == 'num'`;
  add `fixed: bool = False` to `AlgebraicRule`; add `SheafViolation` check for fixed rules;
  extend numeric rule evaluation to call `get_or_create_atom` when result not in `mg.atoms`
- `core/predict.py` — add 0b back-off step (relational rules) before 0c

**New test file:** `tests/variable_binding_domain_test.py` (≥ 9 tests):
- Numeric fit regression test (succ, add still discovered and correct)
- Relational identity rule discovered and applied
- Relational constant rule discovered and applied
- Relational commutativity rule discovered (if symmetric pairs present)
- No arithmetic rule fitted for non-`num` coarse-type atoms (spurious rule guard)
- Novel atom created when numeric rule fires beyond training range
- Novel atom is tagged `predicted=True`
- `fixed=True` rule survives a falsifying observation unchanged
- `SheafViolation` raised when a fixed rule is violated

---

## Phase 17c — Cross-Domain Transfer Benchmark  (DEFERRED)

**Deferred until Phase 18 and Phase 19 are complete.**

Reason: the transfer benchmark measures knowledge-level transfer across domains.
Until the rule learning is domain-agnostic (Phase 19) and multi-step reasoning
is in place (Phase 18), the transfer benchmark will measure only the baseline
MorphismGraph perplexity improvement — which has already been measured in Phase 17.
Running it earlier would produce a misleading baseline.

When unblocked, the experiment design from the original Phase 17c entry above stands
unchanged.

---

## Current test count: 237/237 passing (+ 35 math_benchmark = 272 total)

**Overall: 237/237 tests passing across all symbolic_ai_v2 test files.**
**Phase 21 adds 12 new tests (phase21_test.py).**

| Phase | Tests | Status |
|-------|-------|--------|
| Phase 21 — NL generalisation + variadic fold | 12 new tests | ✅ |
| Phase 20 — Q&A / word problems / variadic equations | 19 tests | ✅ |
| Phase 19 L1 — FCA concept_ids + type registry | 8 tests | ✅ |
| Phase 19 L2 — RelationalRule | 3 tests | ✅ |
| Phase 19 L3 — Novel atom creation | 3 tests | ✅ |
| Phase 17b — domain independence | 11 tests | ✅ |
| All prior phases | 181 tests | ✅ |

---

## Immediate next steps (priority order)

1. ✅ **Fix `ctkg_live.py`** — Concept/TypeDef/sheaf_merge API corrected
2. ✅ **FCA + ctkg_live end-to-end test** — 7/7 tests pass; reasoning pipeline working
3. ✅ **`pruning_test.py`** — 8/8 pass; saturation + both pruning mechanisms verified
4. ✅ **`transfer_test.py`** — 6/6 pass; functor_alignment_score > 0.8 on held-out domain
5. ✅ **Checkpoint save/load** — 7/7 tests; round-trip + incremental verified
6. ✅ **Wire `active_inference`** — `ActiveInferenceTracker` via `on_observe` hook; 7/7 tests
7. ✅ **`causal_test.py`** — 7/7 pass; do(A) d-separation, graph surgery, idempotence
8. ✅ **Phase 9**: `reasoning/functor.py` + `functor_discovery_test.py`
9. ✅ **Phase 10a**: Hopf coproduct smoothing in `predict_dist()`
10. ✅ **Phase 11**: `reasoning/planner.py` + `planning_test.py`
11. ✅ **Phase 10b**: FCA adjunction back-off + `LiveCTKG.atom_type_map()`
12. ✅ **Phase 12**: `split_atom()` + full sense disambiguation
13. ✅ **Phase 13**: `reasoning/causal_discovery.py` + `causal_direction_test.py`
14. ✅ **Phase 14**: MasteryState + information flow wiring
15. ✅ **Phase 15**: `environment.py` + `agent_loop.py` + `environment_test.py`
16. ✅ **Phase 16**: `vision/encoder.py` + `video_2d()` + `vision_test.py`
17. ✅ **Phase 17a**: Endofunctor discovery, adjunction detection, natural transformations
18. ✅ **Phase 17b**: Variable binding — symbolic regression, AlgebraicRule, frame matcher
19. ⬜ **Phase 18a**: `reasoning/goal_stack.py` + `reasoning/rule_chainer.py` + tests
    → `linear_eval` target: > 50% test accuracy
20. ⬜ **Phase 18b**: Sequence goals in `core/predict.py` + tests
    → `derivatives` target: > 60% test accuracy
21. ⬜ **Phase 18c**: `reasoning/backward_chainer.py` + tests
    → `conservation` target: > 50% test accuracy
22. 🔧 **Phase 19**: Three missing levels of algebraic generalization —
    Level 1: coarse-type membership on atoms + type-gated `fit_rule`;
    Level 2: `RelationalRule` (identity, constant, commutativity) + `fit_relational_rule()`;
    Level 3: `get_or_create_atom()` + `predicted=True` flag for novel outputs;
    + `fixed=True` on `AlgebraicRule`, `SheafViolation` on homeostatic violations
23. ✅ **Phase 20**: Multi-token Q&A, word problems, variadic equations (19/19 tests)
24. ✅ **Phase 21**: NL generalisation + variadic fold (12/12 tests)
    - Level D: NL word problems generalise to unseen (N1,N2) pairs via numeral scan
      (ATOM_BUF_SIZE raised 8→16 to fit 11-token NL prompts)
    - Level E: variadic fold — `vadd N1 ... Nk eq` for k≥3 via iterated binary rule
25. ⬜ **Phase 17c**: Cross-domain transfer benchmark (unblocked after 18+19)
25. ⬜ **Phase 8**: `memory_growth_test.py` — sub-linear growth under pruning
26. ⬜ **GOALS.md §6**: perplexity < 2.0 bits/char on EarlyModernLatin
