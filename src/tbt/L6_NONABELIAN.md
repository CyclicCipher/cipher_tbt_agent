# L6_NONABELIAN — from an abelian grid to a learned GROUP REPRESENTATION (a refactor)

*2026-07-01. A PLANNING doc for a REFACTOR (not a from-scratch rewrite — the abelian grid survives as the special case).
Motivation → the one core change → why it (hopefully) unlocks Sokoban → a staged plan with gates+fallbacks → risks →
what is preserved. Grounds: `MATH_PHASE.md` (THE MASTER BOUNDARY + the ABELIAN CEILING), the Fable critique, TEM
(Whittington 2020 — learned transition matrices + structure/content factorisation), Gao et al. 2021 (path integration ⇔
a GROUP REPRESENTATION + isometry). North star: L6 should represent the domain's transformation GROUP, of which the
current translation/plane-wave grid is the abelian special case.*

## Why — the abelian ceiling (the thing this refactor removes)
L5's operator today is an ADDITIVE displacement (`move_delta[a]=(dx,dy)`, you ADD it); L6's grid is PLANE WAVES and
`path_integrate` ROTATES phases by that displacement. Both are the abelian move: addition and phase-rotation COMMUTE, and
plane waves are exactly the *characters (1-D irreps) of the abelian translation group*. **Commutativity is not learned — it
is hard-wired into the substrate.** A code of commuting phases physically cannot store a PATH-DEPENDENT (non-abelian) result,
because there is no place to record "the answer depended on the order." Non-abelian structure — rotations, orderings,
constrained dynamics (Sokoban-with-walls: move-right-then-up ≠ up-then-right) — is therefore unrepresentable in L6 as built.
NB the *active* substrate (the online SR) already handles a non-abelian Cayley graph, but only as a memorised LOOKUP (a node
per state, NO generalisation/extrapolation). The GRID is what generalises, and it is abelian. **So the gap is precisely:
non-abelian structure WITH generalisation.**

## The one core change (stated three ways)
- **L5 (primary):** operators go from additive DISPLACEMENTS (vectors, commute) to composable OPERATORS — learned LINEAR
  MAPS (matrices) applied to the L6 code, composing by MATRIX PRODUCT, which is *non-commutative by construction* (`A·B ≠
  B·A` records order-dependence for free). NB the abelian grid is already "block-diagonal 2×2 rotation matrices in disguise"
  (a phase rotation IS a 2×2 orthogonal matrix per module) — the generalisation is to let those matrices be LEARNED,
  LARGER, and NON-commuting.
- **L6 (consequence):** the location code goes from a phase/torus point to a LATENT VECTOR the L5 operators act on — a group
  representation. The abelian grid falls out as the special case (commuting orthogonal rotations = the plane waves we have).
- **Constraint (Gao — do not skip):** the learned operators must satisfy the GROUP-REPRESENTATION condition (orthogonal/
  unitary, composition-consistent) so path integration + extrapolation SURVIVE; a *freely* learned matrix keeps neither. =
  TEM's learned per-action transition matrices + Gao's isometry constraint.
- **Relations discovered, not assumed:** the FREE composition of operators (the tree of operator-words) path-integrates;
  RELATIONS — including COMMUTATIVITY itself, since the commutator `aba⁻¹b⁻¹` is a loop that returns to start — are
  DISCOVERED by LOOP CLOSURE (a word that returns to a seen state = a relation). The factored-loop-closure machinery
  (`MATH_PHASE.md`) thus generalises from cyclic factors (abelian) to RELATION discovery (non-abelian). Read-off = the
  free/tree part; SEARCH = the relations (the quotient). The abelian grid = "assume every commutator loop closes."

## Why this (hopefully) unlocks Sokoban — honestly
Sokoban's dynamics are non-abelian and config-dependent: the effect of a move depends on what is ahead (push vs. not),
and move-orders do not commute once walls/blocks intervene. The current abelian grid can only map FREE translation (the
agent on open ground), never the push dynamics — so the agent has no faithful MAP to plan over, only the memorising SR.
A representation-constrained operator L6 gives a **faithful COMPOSITIONAL map of order/config-dependent dynamics** — the
substrate a planner needs. With it, planning = **geodesic-finding in the learned Cayley graph** (the master-boundary shared
mechanism), and the GSG proposes goal-configurations. **Necessary, not proven sufficient:** the GSG / multi-step planning /
commitment (the vector-nav + GSG line) still apply on top; Stage 3 is where the "unlocks Sokoban" hope is CONFIRMED or
FALSIFIED, and the staged gates are designed so that a remaining failure is diagnosable as REPRESENTATION (this refactor)
vs PLANNING/GSG (the other line). Do not sell the refactor as a Sokoban solve; sell it as the missing SUBSTRATE.

## The staged plan (each stage gated + a fallback; abelian behaviour preserved throughout)
- **Stage 0 — STOP hard-coding commutativity (the negative first step). DONE (2026-07-01).** `tbt/operator.py` = the
  `Operator` primitive (a matrix; `apply` = act, `then` = compose by matrix product = NON-commutative in general;
  `translation`/`rotation` factories; `commutes_with`). `L5.operator(a)` = the abelian TRANSLATION view of `move` (Stage 1
  makes it a learned matrix); `l6_grid.operator(disp)` = `path_integrate` re-expressed as a block-diagonal phase rotation.
  *Gate met (`test_operator.py`, suite 122):* translation is a faithful abelian representation (composition fidelity +
  commutes); the interface HOLDS a non-commuting operator (rotation ∘ translation ≠ translation ∘ rotation) — commutativity
  is no longer baked in; `operator(disp).apply(z) == path_integrate(z, disp)`; and L5's operator reproduces the additive
  `move` on a NavGame path + a counting succession (NO regression — the live `track` still uses `move`, unchanged). The door
  is open; nothing is exploited yet.
- **Stage 1 — LEARN operators as a representation (TEM + Gao).** Learn a per-action operator matrix from observed
  transitions, CONSTRAINED to a proper representation (orthogonal, composition-consistent). L6 = the acted-on latent.
  *Gate (abelian):* the learned rep matches the hand-coded grid's EXTRAPOLATION — composition fidelity `M(a∘b)=M(a)·M(b)`
  (the [[project_recurrent_world_model]] probe). *Gate (non-abelian):* a small non-abelian task (planar rotations / S₃) —
  the learned operators represent the order-dependence the abelian grid cannot. *Fallback:* keep the hand-coded grid for
  abelian domains; use learned operators only where non-abelian structure is present.
- **Stage 2 — DISCOVER relations by loop closure (the quotient).** Free composition path-integrates; relations (incl.
  commutativity) are found by loop closure under the **predictive-sufficiency** criterion (causal states / bisimulation, per
  `MATH_PHASE.md`) — close the coarsest partition that stays a sufficient statistic. *Gate:* on a task with a KNOWN
  presentation, the discovered relations match; the free part reads off, the relations are found. Fully dissolves the
  abelian assumption. *Fallback:* if online relation-discovery is unstable, restrict to relations that recur within a bounded
  horizon (bounded loop length) — the analogue of bounded-depth carry.
- **Stage 3 — SOKOBAN / non-abelian planning.** Faithful non-abelian map → planning = geodesic-in-Cayley-graph; the GSG
  proposes goal-configs; commitment holds the multi-step maneuver. *Gate:* Sokoban — the map now represents push dynamics
  and the planner searches the non-abelian graph. Confirms or falsifies the unlock. *Diagnosis on failure:* the Stage-0..2
  gates isolate whether the residual gap is representation (here) or planning/GSG (elsewhere).

## Risks (the spine — where this breaks)
1. **LEARNABILITY of the constrained representation.** Can operator matrices constrained to a proper rep be learned ONLINE
   from few samples? Our own DEQ line is the warning: **constraint ⊥ expressivity** (contraction ⊥ expressivity killed the
   single-operator equilibrium — [[project_recurrent_world_model]]). The representation/isometry constraint may be as hard
   to satisfy online as the contraction was. This is the linchpin risk.
2. **COST / dimensionality.** Non-abelian irreps are higher-dimensional (matrices, not phases); the code + operators are
   heavier. Needs truncation / band-limiting (low-order irreps only), the group analogue of the grid's finite scales.
3. **EXTRAPOLATION without the engineered guarantee.** A learned rep may not extrapolate like the hand-coded grid; Gao's
   constraint is what buys it, and it may hold only approximately → drift on long compositions (the carry-depth wall, again).
4. **RELATION DISCOVERY at scale.** Discovering the quotient online is the hard structured-world-model problem; loop closure
   + predictive-sufficiency is the plan but is unproven beyond toys; the wrong-merge failure mode is ever-present.
5. **SUFFICIENCY for Sokoban.** Necessary (a faithful map) but not obviously sufficient — the GSG/planning/commitment still
   apply. The refactor may unlock the SUBSTRATE without solving the task. Keep the claim honest.

## Preserved / reused (why it's a refactor, not a rewrite)
- **The abelian grid stays as the special case** (commuting rotations) → no regression on navigation / counting / magnitude.
- **L5 already has an operator notion** — the per-location forward model + the pose group-operators recognition reads — so
  generalise its interface, don't replace it. (One-step config-dependent effects already live in the forward model; this
  refactor is the COMPOSED, path-integrated FRAME you plan over.)
- **The online SR stays** as the conjunctive/lookup complement (any graph, no generalisation) — the grid/representation is
  the generalising half.
- **Loop closure / recognition (L2/3)** is reused for relation discovery — the same machinery, generalised from cyclic
  factors to group relations.

## Connections
`MATH_PHASE.md` (the master boundary: free/read-off vs quotient/search; the abelian ceiling; geodesic-in-Cayley-graph;
predictive-sufficiency factorisation). TEM — Whittington et al. 2020 (learned transition matrices + structure/content
factorisation). Gao et al. 2021 (path integration ⇔ group representation + isotropic/conformal isometry). [[reference_grid_sr_eigenbasis]]
(grid = SR eigenvectors — the abelian eigenbasis; non-abelian → matrix coefficients / Peter–Weyl). [[reference_tbt_reference_frame]]
(L6 = the movement-bootstrapped position frame). [[project_recurrent_world_model]] (composition-fidelity probe; the
constraint⊥expressivity learnability warning). [[project_math_hypothesis_probe]]. `HETERARCHY_PLAN.md` (non-abelian
composition is also where multi-frame binding via the thalamus starts to matter).
