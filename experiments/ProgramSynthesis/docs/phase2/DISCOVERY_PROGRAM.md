# Discovery-Systems Program — the symbolic relaxation-and-search agent

> Grounded in *Self-Revising Discovery Systems for Science* ([arXiv:2606.01444](https://arxiv.org/abs/2606.01444),
> [[reference_discovery_regime_transition]]). **Written 2026-06-12, revised same day** after:
> (1) symbolic AI became the focus — 4 GB rules out meaningful transformer training;
> (2) the old three-way bake-off collapsed into one bet — a self-revising symbolic world-model
> agent; (3) a correction — discovery is **relaxation for the within-regime work *plus directed
> search* for the across-regime work**, not relaxation alone; (4) lessons from **MuZero**. Phase 1
> (the binding sweep) is closed — see RESULTS.md. Agent component/IO: `AGENT_DESIGN.md`.
>
> **Status legend:** `directional` empirical · `theoretical` · `open` · `commitment`.
>
> **Phase-2 fusion (2026-06-13):** the discovery mechanism here (MDL / representation-regime change)
> is the *learning loop* for the **Volume Concepts** representation — see `VOLUME_CONCEPTS.md §0`.
> MDL is the shared objective: it carves a concept's local subspace, fits its region, proposes the
> graph edges, and **promotes** lower concepts to dimensions of higher ones. Discovery decides the
> geometry; the geometry is what discovery revises.

---

## 0 · The frame (and what changed)

- **Symbolic pivot.** Transformers are parked for resources → the Merge-layer experiment leaves the
  active program (see *Parked*). One symbolic line, not a bake-off.

- **Relaxation + directed search** (corrected from the earlier "relaxation, not search"). Two
  levels, not rivals:
  - **Relaxation** = the *within-regime* work, given the structure you already have — plan over
    known rules, fit given primitives, settle to equilibrium. Cheap; the inner loop; the bulk.
  - **Directed search** = the *across-regime* work — going beyond the current boundary. Two kinds,
    **both impossible by relaxation** (you cannot relax into a representation you don't have):
    *data-gathering* (act to reach states the model can't predict — epistemic value) and
    *structure-proposing* (propose a new rule/primitive to explain a residual).
  - The enemy is **blind / exhaustive** search, not search. The whole game is keeping the search
    *directed* — by residual, epistemic value, MDL, and a learned value (MuZero) — so it stays
    tractable. (Physarum does both: it spreads exploratory foraging fronts *and* relaxes the
    network. MuZero does both: MCTS expands to new nodes *and* a learned policy/value guide it.)
  - **Symbolic makes the search unavoidable.** A neural net can hide discovery as relaxation —
    structure is latent in an overparameterized space and grokking just surfaces it. A *minimal
    symbolic schema has no such reservoir*: a new primitive must be **explicitly proposed** (search).
    We chose the regime where structure-search is explicit, so "relaxation only" is especially wrong
    for this path.

- **The binding lesson is a hard constraint.** Represent in the invariants (object-relative,
  relational) so structure transports; absolute binding (`content`) doesn't.

## Shared primitives (paper concepts)

| primitive | paper | here |
|---|---|---|
| **regime** | schema category | current schema — objects + relations + rules |
| **transport** | left Kan extension | derive / explain new data from the current schema |
| **residual** | what won't transport | the unexplained — where directed search is aimed |
| **revision** | verified regime transition | **directed structure-search**: propose structure where the residual is high, accept iff it shortens total description *and* preserves prior derivations; relaxation then optimizes within it |

**Yardsticks:** composition gap (residual after transport, `agent/gap_report.py`) + skill-
acquisition efficiency (the native ARC-AGI-3 metric).

---

## Phase 0 · The gate — can we keep the search directed? `open` · **do first**

Corrected gate. The question is **not** "is discovery purely downhill (relaxation)?" — it isn't;
discovery is irreducibly part search (crossing regime boundaries). The real gate: **can we keep the
across-regime search *directed* enough to stay tractable, while relaxation handles the within-regime
bulk?**

- **Why it gates everything.** If the only way to find new structure is blind/exhaustive proposal,
  the symbolic line inherits the combinatorial explosion and the resource-light bet dies.
- **The directedness is the actual content.** Guidance candidates: residual/surprise (where to
  propose), epistemic value (where to explore), MDL (which proposal to accept), a learned value (how
  deep the search must go — MuZero). Active inference offers a unifying potential: free-energy
  descent for the relaxation, expected-free-energy for the directed exploration.
- **What to do.** Theory (what bounds the structure-search — MDL guidance, value truncation,
  submodularity) + one toy where we watch whether a residual-guided *propose-and-relax* loop
  discovers the right structure without exploding.
- **Gate.** If even a toy needs blind enumeration, stop and reconsider.

## Phase 1 · Representation — the invariant substrate `commitment`

The relaxation-and-search runs *over a representation*; the binding lesson dictates its form:
object-relative and relational (objects, relations, causal links), never absolute coordinates
(`content`'s Phase-1 failure). Seeded with **Core Knowledge priors** only (objectness, agentness,
number, geometry/topology — which double as the rule-language vocabulary; see `AGENT_DESIGN.md` §2).
No domain vocabulary ([[feedback_no_seeded_primitives]]).

## Phase 2 · The mechanism — relaxation + directed structure-search `theoretical`

- **Substrate:** an explicit schema — typed objects + relations/rules.
- **Within-regime (relaxation):** reinforce structure that carries explanatory flux (shortens the
  description of observations), starve structure that carries none.
- **Across-regime (directed search):** propose new structure where the residual concentrates; accept
  a proposal iff it shortens total description **and** transports prior derivations.
- **Replaces** both the discrete Builder/Breaker loop and blind search-over-mutations — it keeps the
  *directed* proposal and lets relaxation do the rest.
- **Measure:** does description length fall with **discrete drops at growth events** (regime
  transitions / grokking-epiplexity)? Does OOD jump? Knife-edge: the propose-and-accept rule must be
  domain-blind (this *is* Phase 0's deliverable).

## Phase 3 · The agent — self-revising symbolic world-model on the replica `theoretical`

Loop: perceive → object graph → induce (relax + propose) → infer goal from score → **plan by
relaxation toward the goal, guided by a learned distance-to-goal value (MuZero)** → explore by
*directed* search (epistemic value) → transport the schema across levels (induce only for the new
mechanic). Component / input-output design: **`AGENT_DESIGN.md`.** Measured by composition gap +
skill-acquisition efficiency.

## MuZero — precedent and fallback `reference`

- **Learn-the-model-and-plan-in-it without given rules** — the existence proof for this architecture.
- **A learned value** makes long-horizon, sparse-reward planning tractable (bootstraps, truncates
  depth). Added to the Planner; it *is* the pragmatic-value / EFE term.
- **Value-equivalence vs explanatory.** MuZero models only what's relevant to value — efficient but
  goal-specific, so it may *not transport* when the goal changes. We choose the **explanatory** model
  (it transports across levels — what the benchmark demands), with active-inference **precision-
  weighting** as the way to keep it efficient (model everything, allocate effort by relevance).
- **MuZero is guided search** (policy prunes, value truncates) — the proven **fallback** if the
  relaxation Planner fails. The value helps either way, so add it regardless.
- **Sample-inefficient** (millions of games); ARC-AGI-3 demands few-shot — a reason for the
  symbolic/compositional route over a neural latent.

## Parked · transformer Merge layer `directional`

Composition as an architectural primitive (a learned `M(z_a, z_b)`). Parked for resources; design
was unsettled. Revisit only if the symbolic line needs a sub-symbolic perception component that
justifies the VRAM (see `AGENT_DESIGN.md` §5).

---

## Open questions / risks

- **Phase 0 is the bet:** keep the across-regime structure-search *directed*, or inherit the explosion.
- **Value-equivalence vs transport** (MuZero): the explanatory model transports but costs efficiency;
  precision-weighting is the hoped resolution.
- **Guided planning is itself a (directed) search** — watch that the Planner's value/EFE guidance
  doesn't quietly become exhaustive.
- **Bitter-lesson knife-edge** on every operator (segmenter, inducer, propose-and-accept, goal-inferrer).
- **Objectness from raw grids** is the perceptor precondition, hard in ambiguous cases.
- **Relation to RWM:** this is now its own (symbolic) line; the paper is the shared spine.
