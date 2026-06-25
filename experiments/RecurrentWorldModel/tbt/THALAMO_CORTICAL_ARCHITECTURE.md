# Thalamo-Cortical Architecture

A reusable system for hierarchical, multi-column world-modeling — the way the transformer is reusable:
**fixed machinery, different hyperparameters → different models.** Where a transformer composes N
attention+FFN blocks, this composes N **columns** (the maps/experts) with a **thalamus** (the router /
controller) and **basal ganglia** (the gate selector), trained by **reward** (dopamine).

> One mechanism — *learn a model, predict from it* — instantiated as columns, composed by the thalamus,
> gated by the basal ganglia, valued by reward.

Status: design doc. Implemented today = the Column (`experiments/RecurrentWorldModel/tbt/`) and the flat
reward model (`reward.py`) + a single-column agent (`experiments/ProgramSynthesis/agent/column/`). The
thalamus, basal-ganglia gate, and the second (task) column are the build this doc plans. Research backing:
see memories `reference_hierarchy_substrate`, `reference_exploration_replay`.

---

## 1. The Column — the reusable unit (a TEM module)

A column learns **one structured map** and predicts from it. It is a Tolman-Eichenbaum-Machine module
(Whittington 2020); our layers map onto TEM exactly:

| Layer | Role | TEM |
|---|---|---|
| **L6** grid/structure code `g` | the SR-eigenvector frame of the transition graph (Stachenfeld: grid cells ARE the SR eigenvectors) — grid-like on metric graphs, correct on trees; hard-coded hex kept as an innate metric PRIOR; **can also be driven as a DYNAMIC path-integrated state for sequence memory** (selective recurrence — §14 stage 11 / R8) | `g` (MEC) |
| **L5** displacement operators | per-action/relation operators that path-integrate L6 — the **efference copy** drives them | action operators |
| **L4** content `x` | feature codebook; binds feature ⊗ location | `x` (LEC) |
| **L23** conjunction `p` / object memory `S` | bound feature-at-location memory; pooled, delta-rule revised, votes laterally | `p` (hippocampus) |

Properties (validated): factorizes **structure** (L6, reused/generalized) from **content** (L4, per-domain);
path-integration on arbitrary graphs (spatial *and* relational — transitive inference); subgoal **scale
hierarchy = the SR eigenspectrum of its own transition graph** (eigenoptions, §6).

Hyperparameters: grid scales (range), dims, structure class (metric/relational), place sparsity `k`.

---

## 2. Factorization — why one column for storage, separate columns for composition

Two distinct uses of factorization; conflating them is the trap.

- **STORAGE (capacity):** many **independent** domain-models live in **orthogonal slots** of one column,
  recalled one at a time, no interference (`unified_demo.py`: arithmetic/family/social/spatial → 1.000).
  Orthogonality *is* the non-interference — and it also forbids cross-talk between slots.
- **COMPOSITION (interaction):** two structures **active and interacting** (task gates space) must be
  **separate columns**, kept **factored** (space ⊕ task), joined by a **thin switchable interface** (one
  goal-state channel via the thalamus).

**Why not one joint column for composition?** Encoding the joint *space × task* structure in one column is
the **product space = the 2^K conjunctive explosion** (the measured scaling wall, `scaling_probe.py`).
Separate columns keep it additive. Rule of thumb:

- Same structure / independent recall → **one column** (orthogonal slots).
- Different structures that must interact concurrently → **separate columns + thalamic routing.**

---

## 3. The Thalamus — router + controller (not a relay to skip)

Every long-range cortico-cortical (column→column) message is **paralleled by a cortico-thalamo-cortical
route** (Sherman & Guillery; transthalamic pathways). The thalamus is the brain's dynamic-routing /
attention fabric — the analogue of transformer attention.

- **Transthalamic routing:** L5 output (the efference copy) branches — to motor *and* through higher-order
  thalamus (pulvinar/MD) to other columns. Inter-column messages route here.
- **Gating (TRN):** the inhibitory reticular shell selects which routes are open — the attention spotlight.
- **Context maintenance + switching (MD ↔ PFC):** sustains the active subgoal/context in the task column,
  suppresses the irrelevant, switches under uncertainty/completion. This is the controller's *state*.
- **Deviance + gain (L6 feedback):** detects prediction-error vs the column's model; sets gain; the
  surprise / course-correction signal.

Hyperparameters: columns routed, gating topology, context-switch threshold.

---

## 4. The Basal Ganglia — the gate SELECTOR (when / where / why)

The thalamus is the gate; the BG decide when it opens (O'Reilly CCN; PBWM, O'Reilly & Frank 2006).

- **Default closed:** GPi/SNr tonically inhibit the thalamus.
- **WHERE:** direct (Go/D1) pathway disinhibits the *selected* thalamic loop; indirect (NoGo/D2) suppresses
  competitors.
- **WHY:** dopamine (reward-prediction-error) trains Go/NoGo weights → open the highest-value gate for the
  current context.
- **WHEN:** striatal threshold crossing (context + candidate options). PBWM: the same gate decides **update
  vs maintain** the task context = **subgoal switching**.

Decision loop: **cortex proposes** (candidate subgoals/routes, L5→thalamus) → **BG selects** (disinhibits
the winner) → **thalamus gates** → **dopamine learns**. Actor-critic: BG = actor, reward/value = critic.
Our `reward.py` (value + eigenoption candidate subgoals) is that **critic + candidate generator**.

---

## 5. Connections — the explicit spec

For columns **A** (task/relational) and **B** (spatial/metric):

### Short-range (WITHIN a column) — implemented
- **L5 → L6**: path integration — the efference copy (issued action/relation) advances the structure code.
- **L4 → L23**: feature ⊗ location binding into the object memory.
- **L23 lateral**: local pooling / delta-rule revision.
Builds one structural map. No thalamus involved.

### Long-range (BETWEEN columns) — via the thalamus, gated by the BG
- **TOP-DOWN (A → B, task → space):** A's active subgoal → a **goal state** in B's structure (a target
  location/feature). B then **vector-navigates** to it (displacement cells, Bush 2015). Path: A's L5/L6 →
  higher-order thalamus → B's input. **Gated open by the BG** when A selects the subgoal; **latched** by MD
  thalamus as the active context.
- **BOTTOM-UP (B → A, space → task):** B's **achieved state** (reached the goal) or **prediction-error /
  exafference** (block-on-pad, door-opened) → updates A's state: **advance the task graph and *learn* the
  dependency.** Path: B's L5 (deviance) → thalamus (deviance detect) → A. **This is where dynamics are
  learned.**
- **LATERAL (A ↔ B, same level):** **consensus voting at L23** (the binding layer) for agreement —
  multi-column object recognition; TEM's parallel streams "combined at retrieval." Used for ambiguous
  perception / multi-view, not for the task↔space control loop.

Reciprocity: top-down (goal) + bottom-up (achieved/deviance) form the **closed control loop**; lateral is
consensus. Every long-range edge is a **transthalamic, BG-gated** route — not a hard wire.

---

## 6. The hierarchical loop — plan → execute → monitor → switch

1. **Discover** (reward-free): eigenoptions over each column's SR/Laplacian → candidate **subgoals**
   (bottleneck = low-frequency eigenvector extrema; low-freq = global, high-freq = local).
2. **Plan:** task column A sequences subgoals; **BG selects** the active one; thalamus **latches** it as
   context and **routes** the goal-state to spatial column B.
3. **Execute:** B vector-navigates to the goal-state (displacement cells); fast model-free policy for
   low-level control.
4. **Monitor:** thalamus watches **L6 deviance** (efference-copy prediction vs actual). Match → continue.
5. **Switch / revise:** on **arrival** → BG gates A to **advance** (next subgoal). On **surprise/exafference**
   → re-anchor (relocalize), **revise** the model (learn the dynamics), re-plan.

This is "think before doing" (plan via the model) + prediction-error course-correction — the System-2 loop,
not blind one-step replanning.

---

## 7. Goal & subgoal discovery

- **Subgoals — reward-free**, from each column's transition graph: SR / graph-Laplacian eigenvectors →
  eigenpurposes → eigenoptions (Machado 2017). Bottlenecks = Fiedler-cut extrema. Scale hierarchy = the
  eigenspectrum (no extra columns for scale).
- **Goal — valued from sparse reward**: the reward model values which subgoal *sequence* yields score;
  dopamine-style RPE trains the BG gate. Intrinsic (novelty/empowerment) value kept **separable** from
  extrinsic (score) — as in OFC.

---

## 8. Hyperparameters & the LockPath instantiation (smallest non-trivial model)

- **Columns:** 1 spatial-metric (hex grid, scales 11/13/17) + 1 task/relational (dependency/ordinal graph).
- **Thalamus:** routes the two columns; latches the active subgoal.
- **Basal ganglia:** selects the subgoal, gates task-column updates; trained by the level-completion score.
- **Connections:** top-down (subgoal → spatial goal-cell), bottom-up (reached / exafferent effect → task
  advance), lateral (n/a, single view).
- Bigger games: add object columns / more structure-columns — **same machinery, different hyperparameters.**

---

## 9. Reward model & agentic wrapper — the plug-and-play layer

These make the architecture **reusable across experiments**, not just LockPath.

**Reward model (`reward.py`) — the critic + planner.** Domain-agnostic: it operates on *any* column's
transition graph + a *sparse reward signal*, learns what is rewarding, values it (prioritized sweeping /
the Mattar–Daw replay), and supplies the candidate subgoals (eigenoptions) and value that **train the BG
gate** (§4). It knows nothing about LockPath — give it transitions and a scalar reward and it plans. This
is the actor-critic's critic; it ports unchanged to arithmetic, a new game, or a robotics loop.

**Agentic wrapper (`agent.py`) — the loop + a generic Environment interface.** One reusable loop drives
columns + thalamus + BG + reward:
`perceive → efference-copy localize → update model → plan (§6) → act → learn`.
It plugs into any experiment through a thin interface:

```
Environment:  reset() -> observation
              step(action) -> (observation, reward, done)
              actions      -> available actions
```

An experiment implements `Environment`; the wrapper just drives it (choose action → step → feed the
transition to the column). **The COLUMN owns the model**: `column.observe`/`consolidate` discover a
structure from transitions (online; the SR-eigenvector frame is computed here), beside `column.learn_domain`
(the same frame, structure given); `column.predict`/`add`/`infer` read from it. So there is one place that owns
"learn a model", and `agent.py` is a ~30-line driver that delegates to the column. The LockPath
`ColumnAgent` is the first instantiation; swapping the Environment swaps the task with **no change to the
machinery** — exactly the transformer-style reuse goal.

---

## 10. Package layout — `tbt/` is the home of every component

```
experiments/RecurrentWorldModel/tbt/
  l6_grid.py            # Column: L6 grid / structure code            [done]
  l5_displacement.py    # Column: L5 displacement operators           [done]
  l4_feature_location.py# Column: L4 content codebook + binding       [done]
  l23_object.py         # Column: L23 conjunction / object memory     [done]
  column.py             # Column = TEM module + OWNS learning: learn_domain (batch) AND observe/consolidate (online structure discovery) + predict/add/infer/revise/anchor  [done]
  reward.py             # Reward model / critic (domain-agnostic; pure stdlib)            [done]
  env.py                # Environment contract (reset/step/actions; pure stdlib)          [done]
  agent.py              # Thin env-driver: feeds transitions to the column, delegates everything else       [done]
  eigenoptions.py       # Reward-free subgoal discovery (SR/Laplacian)                    [planned]
  thalamus.py           # Inter-column routing / cross-column conjunction (bind/read)     [done — conjunction; goal-state control loop ahead]
  basal_ganglia.py      # Gate selector / emergent allocator (Go/NoGo + load-balance + dopamine-RPE; pure stdlib) [done]
  factorize.py          # Disentanglement: discover factors from action-orbits (Higgins 2018; pure stdlib)  [done — direct-product case]
  residual.py           # Recursive residual modelling — the ONE general structured-deviation learner (pure stdlib) [done]
  dynamics.py           # Dynamics column: learn the world's conditional effects (precondition→effect) from exafference [done]
  __init__.py           # lazy exports (torch loads only when a torch component is touched)
  RESEARCH.md           # research log
  THALAMO_CORTICAL_ARCHITECTURE.md   # this document

experiments/RecurrentWorldModel/precursor/
  numberline.py         # stage-1 environment + runner (doc §14)
  factored.py / carry.py / grid2d.py / tree.py   # stages 3–5 (place value, learned carry, 2-D, non-metric tree)
  multicolumn.py        # stage-6 emergent allocation: pool of columns + BG gate + thalamus
  disentangle.py        # stage-7 Higgins disentanglement: discover an entangled torus factors, from action
  passive.py            # stage-8 passive learning: a world model learned by watching (no actions) + anticipation
  coupled.py            # stage-9 coupled-carry: the transverse test FLAGS place value's semidirect coupling
  residual.py           # stage-10 recursive residual: ONE loop learns carry/context/exceptions, refuses noise
  dynamics.py           # ARC step-1: a dynamics column learns conditional effects (key/switch/pad → door) from experience
  language.py           # stage-11: the column's SR-frame IS a word embedding (passive geometry probe)
  language_active.py    # stage-11: active sensorimotor language — predict = a motor act (Markov-1)
  language_recurrent.py # stage-11: sequence memory — L6 as a selective-SSM path-integrated state (the win); RESEARCH.md R8
```

An experiment = `import tbt`; assemble `columns + thalamus + basal_ganglia + reward + agent` with chosen
hyperparameters; plug in the experiment's `Environment`. The reward model now lives in `tbt/`; the LockPath
`ColumnAgent` stays in `ProgramSynthesis/` as an experiment instantiation (it depends on `arc_agi_3`, which
`tbt/` must not).

---

## 11. Dependencies

- **Python 3.11+.**
- **PyTorch (`torch`)** — the only heavy dependency. Used for the column/grid math (L6 hex grid, place
  codes, path integration) and eigendecomposition for eigenoptions (`torch.linalg.eigh`). **CPU is
  sufficient** at current scale (small grids, ≤ a few hundred locations); GPU optional. The repo pins
  `torch 2.10+cu126`.
- **Python stdlib** (`random`, `collections`, `dataclasses`, `math`) — the reward model (prioritized
  sweeping), the control loop, and the BG gate are **pure stdlib**, no torch needed.
- **`numpy`** — optional, only if any component prefers numpy arrays over torch tensors.
- **Not** required: Triton (that's the paused Mamba line), `transformers`, or any RL framework — the
  architecture is self-contained `torch` + stdlib.
- **Per-experiment:** the environment package (e.g. `arc_agi_3` for LockPath) is the experiment's own
  dependency, hidden behind the `Environment` interface — not a dependency of `tbt/` itself.

---

## 12. Open problems (load-bearing unknowns)

1. **Online task-graph learning** — RESEARCHED + CORRECTED (`RESEARCH.md`, R1): cold-start is **NOT**
   the blocker. Local cause→effect is **1-shot online** — our column already learned the number line from
   scratch and generalised to all arithmetic; a key→door edge is just a successor edge, bound (not
   SGD-trained). The genuinely hard part is **long-range credit assignment** — tracing a present failure to
   a distant past action — solved by **reverse replay** (Mattar–Daw), a *learned* backtracking skill.
2. **Online eigenoption computation** per layout (eigen-decomposition cost / approximation).
3. **Column allocation / emergent specialization** — columns are NOT pre-assigned roles. Mountcastle:
   one universal cortical algorithm; specialization follows INPUTS/connectivity, not design (ferret
   rewiring — auditory cortex given visual input learns to see). So provide a POOL of identical columns +
   the thalamus/BG as a **learned gating router** (Mixture-of-Experts gating = the BG-gate), and let which
   column models what (number line, place value, reward, …) **emerge** via routing + competition + reward;
   a task competitively recruits the closest "neuronal niche" (Dehaene/Anderson neural reuse). Sub-problems:
   (a) gating collapse (MoE rich-get-richer → load-balancing); (b) symmetry breaking (random-init niches);
   (c) **factorization discovery** — how the system discovers the task even factorizes (digit × position) so
   columns have separate structures to specialize to (disentanglement / TEM). The thalamus/BG is therefore
   the learned ALLOCATOR, not just a carry-router — more fundamental than first framed (do NOT hand-assign
   column roles).
   STATUS: **allocation EMERGENCE now demonstrated** (`precursor/multicolumn.py`, stage 6) — a pool of
   identical columns + the BG gate allocate digit vs position by random-init symmetry break + load-balancing
   (a→no collapse, b→broken) + dopamine-RPE; which column takes which role varies by seed and the gate
   routes a known structure back to its specialist; the thalamus composes the gate-chosen columns → 200/200.
   (c) **disentanglement** — discovering THAT the space factors — is now demonstrated for the clean
   DIRECT-PRODUCT case (`precursor/disentangle.py`, stage 7): from an entangled torus (shuffled symbols,
   unlabelled actions) the factorization is read off action-orbit partitions (Higgins 2018; Locatello 2019:
   you need action, not statistics), the two independent factors recovered + modelled compositionally, and an
   N²=100 joint space that overflows one column's codebook is handled by two columns of 10. STILL OPEN: the
   COUPLED case (carry = a semidirect product, not a clean direct product) + content-keyed gating (recognize
   a structure from its transitions, not a given stream id).
4. **Stability** of the BG-gated switching loop (avoid thrashing between subgoals).

---

## 13. Mapping to the transformer (the reuse claim)

| Transformer | Thalamo-cortical |
|---|---|
| FFN block / expert | **Column** (TEM module: L6/L5/L4/L23) |
| Self-attention (dynamic routing) | **Thalamus** (transthalamic routing + TRN gating) |
| Gating / routing decision | **Basal ganglia** (Go/NoGo disinhibition, dopamine-trained) |
| Residual + norm (stability) | **L6→thalamus deviance + gain control** |
| Stacking N blocks | Composing N columns + the thalamic router |
| Loss / gradient | **Reward (dopamine RPE)** training the gate + value |

Fixed machinery, varying hyperparameters — a column is a column whether it maps space, a task graph, or an
object; the thalamus routes whatever columns exist; the BG gates by learned value.

---

## 14. Build & validation plan — the number-line → arithmetic precursor

Build and validate the architecture on a **known-learnable** domain first (no perception or novel-structure
noise from ARC), staged so each stage lights up one more component and tests a specific claim.

1. **Learn the number line.** ✅ DONE (`precursor/numberline.py`, `tbt/agent.py`). One column learns the
   structure from successor transitions **through the agentic wrapper** (cold-start, 1-shot edges): it
   discovers the order of *shuffled* symbols, path-integrates positions via the efference copy, binds
   content, learns the successor operator, and predicts successors column-natively — **11/11 correct across
   5 seeds, all 12 symbols placed.** Settles the R1 correction empirically: cold-start structure learning is
   1-shot, no meta-prior. *Built:* the `Environment` contract + the reusable wrapper.
2. **Arithmetic on ONE column.** ✅ DONE (`precursor/arithmetic.py`). Addition = navigation: a + b = apply
   the learned successor operator b times. **Perfect to ~512** (the line reaches d_mem, then degrades
   gracefully). NB the original feat_dim=96 wall is GONE — L4 now uses SPARSE content codes (capacity >>
   feat_dim, the cortical trick), so the limit moved down to the LOCATION capacity (d_mem place codes; see
   the L4 / capacity work). KEY EMPIRICAL FINDING (settles "why >1 column?"): a single number line is still
   **linear** in capacity (≈1 symbol per number) — the second structure is **not** assumed; it is forced by
   **efficiency at scale**: place value (10 digit-symbols reused across positions) is the **logarithmic**
   representation that handles any magnitude. So multi-column is motivated by the linear-vs-log scaling, not
   theory. *Built:* the `add`
   composition path.
3. **Factored (place-value) representation.** ✅ BUILT (`precursor/factored.py`). A number is stored in the
   column as `digit ⊗ place` bindings — its native **What(L4) × Where(L6)** binding, = the brain's place
   value (Grossberg categorical-What × spatial-Where; Dehaene recycling). **It GENERALISES perfectly to
   UNSEEN numbers at any magnitude: 200/200 on 1-, 2-, 3-, 5-, and 8-digit additions**, from a column that
   learned ONLY the single-digit number line (0..2b−1) and never saw a multi-digit number. So the factored
   representation composes 10 digit-symbols instead of memorising numbers — the answer to "learn the rule,
   not every combination." (Gotcha: an unbound place must read 0 — threshold the readout, else leading
   garbage.) HONEST split: the column does the representation + single-digit arithmetic (learned, by
   navigation); the place-value decomposition + carry rule was PROVIDED (the symbolic/cultural layer).
4. **Learned carry.** ✅ BUILT (`precursor/carry.py`). The carry rule IS learnable — and it is **modular**:
   the carry is the **WRAP of a learned cyclic (mod-base) digit line**, which is exactly what grid cells are.
   Learn a cyclic digit line; the successor operator includes the wrap edge (base−1→0); single-digit add =
   navigation, **digit = where you land, carry = how many wraps — no `%` or `//` anywhere**. **60/60 on
   unseen 1/2/3/5-digit additions.** The base (= cycle length) is itself discovered by traversing the cycle.
   The cyclic line is discovered the SAME universal way as everything else (stage 5): the SR-eigenvector
   frame of a RING has a cyclic spectrum, so the successor operator includes the wrap edge (base−1 → 0) with
   no special case (carry.py: `discovered_wrap` = True, not told). STILL hand-coded in carry.py: the outer
   number↔digits decomposition (factored.py showed that part is genuine in-column). So both halves are proven
   separately — representation in the column (stage 3) + carry learnable from modular structure (stage 3b).

5. **One structure-discovery for ANY topology — the SR frame.** ✅ DONE (`tbt/column.py` `consolidate`;
   `precursor/grid2d.py`, `tree.py`). The column discovers structure as the **SR-eigenvector frame** of its
   observed transition graph (Stachenfeld 2017: grid cells ARE the SR eigenvectors; hexagonal is just the
   eigenbasis of open 2-D). ONE mechanism, no metric-vs-non-metric switch: line / ring / 2-D grid AND a
   non-metric **tree** (28/28, 60/60) — the tree was approach B's (metric-embedding) hard limit. The
   hard-coded hex grid is demoted to an innate metric PRIOR (vector-nav / CRT error-correction) to switch on
   when a task needs it. `learn_domain` (structure GIVEN) and `consolidate` (structure DISCOVERED) now share
   this ONE code source (no parallel systems); structure-specific frames auto-separate DIFFERENT domains,
   while identical structures still need orthogonal slots (`unified_demo.py`: 0.500 → 1.000). Biology log:
   memory `reference_grid_sr_eigenbasis`. NEXT internal step: an incremental (TD-learned) persistent SR
   instead of a batch `eigh()` — more biological and it dissolves the eigendecomposition cost.

6. **Multi-column composition + EMERGENT allocation.** ✅ DONE (`tbt/thalamus.py`, `tbt/basal_ganglia.py`,
   `precursor/multicolumn.py`). The cross-column binding is extracted into the **thalamus** — a register
   R = Σ content ⊗ place (Smolensky / VSA, across two columns); `factored.py` now composes its two columns
   through it (200/200). Then a POOL of identical columns + a **basal-ganglia gate** ALLOCATE the digit line
   vs the position line to columns by competition (random-init symmetry break) + load-balancing + dopamine-
   RPE — which column takes which role **emerges** (varies by seed), is not hand-assigned (§12.3,
   Mountcastle); the gate routes a known structure back to its specialist; the thalamus composes the
   gate-chosen columns → place value (200/200), allocated not designed. STILL given: disentanglement (that
   the task factors into digit × position; §12.3c). NEXT: the goal-state CONTROL loop (task column sets a
   goal-state in the spatial column, §5/§6) — its necessity shows on LockPath, where the subgoal sequence is
   data-dependent, not a fixed position loop.

7. **Disentanglement — discovering THAT the space factors (Higgins-style).** ✅ DONE for the clean case
   (`tbt/factorize.py`, `precursor/disentangle.py`). Stage 6 was handed pre-separated streams; here the input
   is ENTANGLED — an N×N torus seen as ONE shuffled symbol per cell, 4 actions unlabelled as to axis. The
   factorization is read off **how states transform under action** (Higgins 2018; Locatello 2019: static
   statistics provably cannot): each action's orbit-partition is computed, actions with the same orbits are
   one factor (an action and its inverse coincide), two factors are a direct product iff their partitions are
   transverse. Result: from the entangled torus the 2 independent factors are recovered (actions split
   [0,1]|[2,3]), modelled one-column-each, joint dynamics predicted compositionally (144/144 at N=6, exact at
   N=45). Capacity (post the sparse-coding work, R4): a single column is now high-capacity, so both models fit
   at N=45 — but the factored cost is LINEAR (2 columns of N) vs the holistic's QUADRATIC (an N²-state column
   → N²×N² consolidation), so the holistic blows up in compute and degrades by ~N=90 while factored stays
   cheap+exact. STILL OPEN: the COUPLED case (place-value carry = a semidirect product).

**Capacity (infrastructure, not a stage).** L4 content codes AND the place codes are now SPARSE high-D codes
(dentate-gyrus / cerebellum / fly-mushroom-body expansion) instead of dense orthonormal, so capacity is
EXPONENTIAL not linear: the number line runs to 2000+ (was a hard wall at 96, then 513). Sparse place codes
need an attractor CLEANUP (snap to the nearest place code) in composition, gated on the sparse path so the
dense ≤d_mem case is byte-identical. See RESEARCH.md R4. The cortical lesson: generous sparse capacity, then
FACTORIZE on overflow (which is the whole multi-column line).

8. **Passive learning.** ✅ DONE (`precursor/passive.py`). A column learns a WORLD model by WATCHING (no
   actions) and ANTICIPATES (rolls the world operator forward): autonomous ring, 1-step 20/20, 5-step roll-out
   20/20. NO new machinery — a 'world' operator is learned exactly like an action operator; the efference copy
   (reafference vs exafference) is the only thing distinguishing what the agent CONTROLS (active, orbit-
   disentanglable) from what it merely WATCHES (passive). Both coexist in one column (16/16 each). Active alone
   could never learn the autonomous process.

9. **Coupled-carry disentanglement — detection.** ✅ DONE (`precursor/coupled.py`). Place value (units × tens
   with carry) is a SEMIDIRECT product — Z_{b²}, NOT Z_b × Z_b. The transverse test FLAGS it: discover_factors
   returns a TRIVIAL second factor (n=1) and is_product=FALSE (vs the torus's TRUE), localizing the coupling to
   the carry action (+1, whose orbit is the whole cycle); the carry is sparse (b of b²). STILL OPEN: EXTRACTING
   the carry from data = the HOLONOMY of +1 around the units-cycle (the extension's cocycle) — the next build.

10. **Recursive residual modelling — the ONE general mechanism (RESEARCH.md R7).** ✅ DONE (`tbt/residual.py`,
   `precursor/residual.py`). Instead of a bespoke detector per structure type (the holonomy would have been
   one), ONE loop: model the coordinate-deltas, take the residual, re-model it with the same machinery,
   recurse, stop at the MDL boundary. ONE loop, FIVE structurally different problems: 2-digit carry 100/100
   (DISCOVERED place-value carry as rules — `c0==9 → tens+1` — no holonomy, no place value given), 3-digit
   carry 1000/1000 (nested), context-dependence 29/29, feature exceptions 43/43, and incompressible random
   noise REFUSED (41/49, base only = the MDL stop, not memorisation). So scope items A–D (couplings, hierarchy,
   encoding, granularity = all "residual structure") are ONE mechanism. Open: range/disjunction predicates;
   wiring disentangle → residual end-to-end (raw transitions, not given coordinates).

11. **Sequence memory (recurrence) + language.** ✅ DONE (`precursor/language.py`, `language_active.py`,
   `language_recurrent.py`; RESEARCH.md R8). The Markov-1 ceiling (predict next from the CURRENT token only) is
   broken by making **L6 a DYNAMIC path-integrated state** (TBT path integration / HTM context cells) via a
   **selective linear recurrence** (Mamba — the compressed structured state, NOT lossless attention), learned
   online with a 1-step-truncated local rule, the state carried across the whole ~1500-token chunk. On pooled
   Latin + Middle/Old High German (next-token perplexity): memory takes Markov-1 **181 → 152 (best of all
   approaches)**, rare-context **155 → 136** (beats the passive SR-frame factorization, 166/141). The controlled
   ±memory comparison (same model family) is unambiguous — sequence memory is the win. **Key build lesson (R8):**
   the per-channel selective gate must be a DIRECTLY-LEARNED per-token TABLE `G (V×d)`; the Mamba-canonical
   SHARED PROJECTION `σ(Wₐ·E[x])` UNDER-trains at this scale (init-tiny `Wₐ` → α frozen at init → fixed,
   non-selective decay → regresses 163.6) — the projection is the right form only at scale, the direct table
   wins at a few hundred K tokens. The gate is interpretable: prepositions (ad/ab/ex/de, der/von) learn to RESET
   context, particles/copula (et/est/enim) to CARRY it — "a preposition introduces a fresh phrase", discovered
   from raw prediction. NOT a full LM (the R7 Merge-primitive boundary), but recurrence is now a working COLUMN
   capability — the temporal dimension the multi-column neocortex needs (the §5/§6 control loop is itself a
   recurrence over subgoals).

Stage 1 settled cold-start empirically; stage 2 "needs >1 column" (no, until the capacity wall — now lifted by
sparse coding); stage 5 one frame for every topology; stage 6 the factorization's ALLOCATION emerges from the
gate; stage 7 its DISCOVERY (clean direct-product); stage 8 passive watch-and-anticipate learning; stage 9
flagged the coupled (carry) case; stage 10 replaced the bespoke per-structure detectors with ONE recursive-
residual mechanism (carry DISCOVERED, not hand-coded); stage 11 gave the column SEQUENCE MEMORY (L6 as a
selective-SSM recurrent state — memory beats no-memory and the static factorization). The frontier before ARC
is now genuinely just E perception / F value / G credit (plus wiring disentangle → residual end-to-end), with
the recurrence carrying into the multi-column control loop.
