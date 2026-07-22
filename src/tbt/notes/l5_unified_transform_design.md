# notes/l5_unified_transform_design.md — ONE L5 transform, THREE projection roles (design of record)

Status: DESIGN (2026-07-22), before code. Supersedes the *shape* of `behavior.ContactDynamics` and the scene column's
`_cue_weights`. Motivated by a blunt and correct critique: **both of those are hardcoded machinery for one hyper-specific
niche, which is not what L5 does.**

## 1. The indictment (why this note exists)

`ContactDynamics` is a hand-written decision tree over a hand-chosen three-way enum:
`if object moved → YIELD (fit T) / elif body blocked → RESIST / elif body advanced through → PASS`. Every part is mine, not
learned: the three KINDS enumerate exactly the outcomes Sokoban has; the discrimination is an `if/elif` I wrote; "sticky YIELD"
is a patch added when a seed broke; "contact at the leading face" is hardcoded as *the* condition. That violates
`feedback_subgoal_types_from_dynamics` outright (no env-specific enumerations — discover the KINDS from the dynamics). The
column's `_cue_weights` is the same disease in a different niche (relational-pose cues hand-picked for support/gravity).

Worse, they are **two mechanisms for one job** — object dynamics already had an owner (the scene column's L5 dynamics), and
`behavior.py` was built alongside it rather than extending it (`feedback_one_model`, RULES #5). The column path is now
loop-orphaned but still exposed on the agent and exercised by tests.

**L5 does one generic thing:** learn a DISPLACEMENT — a delta in whatever dimension the action changes — and apply it
invariantly (`reference_l5_operator_kinds`). It carries no taxonomy of physics outcomes.

## 2. The realization that dissolves the enum

**Yield / resist / pass are not kinds. They are the DELTAS of two bodies.**

| outcome | agent's delta | object's delta |
|---|---|---|
| "yield" | efference | `T · efference` |
| "resist" | 0 | 0 |
| "pass" | efference | 0 |

There is no taxonomy — only *what delta did each body take*. The enum is an artifact of categorizing outcomes instead of
learning deltas; drop it and the decision tree evaporates. **Solidity, pushability and passability stop being categories and
become learned conditionings of the two bodies' deltas.**

## 3. The unified transform

> **For each body, L5 learns `delta = W · context`**, where `context` = the action's efference ⊕ the relational / contact
> features the column has, `W` is learned by PREDICTION ERROR (the delta rule), and **cue competition (Rescorla-Wagner)
> discovers WHICH context features actually condition the delta** (`reference_cue_competition_key_discovery`).

Because the efference is IN the context vector, one linear form covers both effect shapes we needed: a **fixed** delta
(gravity: context has no efference term) and an **interaction-parameterised** delta (`T·efference`, the push). This subsumes all
four bespoke things at once:

- the **nav operator** — the self's delta, conditioned on the action alone;
- **ContactDynamics** — an object's delta, conditioned on action ⊕ contact, parameterised by the efference;
- the **RW override** — an object's delta, conditioned on action ⊕ support;
- **solidity** — the *self's* delta, conditioned on contact with an unmoving body (no hand-coded rollout coupling).

**The self stops being special: it is simply the body whose delta is unconditioned.** That is a better story than "nav operator
here, object dynamics there, coupling hardcoded in the rollout."

## 4. THREE projection roles (the sublayers — one transform, several output paths)

L5 is not one thing; each cortical layer is two projection classes (`reference_cortical_layers_research`), and the
long-range paper (arXiv:2507.05888, [[reference_long_range_connections]]) gives L5 **three** distinct long-range roles. These
are OUTPUT PATHS of the single transform above, not separate transforms:

| projection | role | ours |
|---|---|---|
| **L5PT / L5a → higher-order thalamus** | ENACT the self's delta (motor command), emit the EFFERENCE COPY, drive the TRANSTHALAMIC HIERARCHICAL route to the next higher region (which is how compositional child-of structure is learned) | `apply`, `efference`, `apply_efference`, `broadcast_efference`; `Agent.place_object` (correctly transthalamic) |
| **L5IT → striatum** | the action PROPOSAL: the transform read BACKWARD into per-action drive, projected to the BG, which selects (priority = salience ⊕ value) | `Column.striatum` (context half only — see the defect below) |
| **L5b → long-range LATERAL** | participate in cross-column VOTING / consensus | **missing** — our voting is mis-located in the thalamus (debt recorded in STATUS) |

**Forward and inverse are the two directions of the ONE transform, exiting through DIFFERENT projections:** forward
(action → displacement, enacted/broadcast) leaves via **L5PT**; inverse (goal-vector → per-action drive, proposed for
selection) leaves via **L5IT**. That is why the anatomy has two classes, and it is the structure the planner should follow.

### Defect this exposes in the shipped step 1
`Agent._nav_inverse` ends with `self.bg.select((), len(movement), salience=salience)` — an **empty cortical context**, and the
per-action drive computed **in the agent**, bypassing L5IT entirely. Both belong in the column's L5IT projection (context AND
per-channel drive). It also violates `feedback_thin_shell_agent`. Fix as part of this work, before building step 2 on it.

## 5. The priors are load-bearing — state them, do not smuggle them

The bespoke versions were sample-efficient (1–2 observations) because they encoded priors. A general `W·context` must express
these as explicit INITIALISATION / STRUCTURE, or it will regress while looking more principled:
- **identity prior on the interaction term** — why ONE push generalised to all directions;
- **free kernel = stay** — the default delta when nothing conditions it;
- **monotone evidence** ("sticky yield") — a body observed to move is movable; a later non-move is OBSTRUCTION, not a new kind.
If the general form cannot hold these, that is the finding: the priors are real and belong in the mechanism as priors.

## 6. Falsification (what pins the design)
Port is only valid if ALL survive unchanged:
- `test_key_discovery` — cue competition rejects the spurious neighbour, support is kept as the real condition;
- `test_object_dynamics` — ONE demo generalises to any position / orientation / object;
- `test_push` — Push L1 at **oracle 6, 12/12 seeds**, block learned movable, wall learned immovable;
- `test_game_loop` / `test_inverse_model` — nav transfer + the read-out.

## 7. Open (do NOT pretend these are solved)
- **Which features are in the context, and how are they DISCOVERED rather than hand-picked?** Hand-picking cues is the same sin
  one level up. This is the `feedback_subgoal_types_from_dynamics` frontier and the honest weak point of this design.
- **Effect KINDS beyond displacement** (recolor, appear/vanish) — `reference_l5_operator_kinds` says the operator is "a delta in
  whatever dim an action changes"; a non-positional dim needs the same treatment, not a new module.
- Sample-efficiency vs generality (§5) — measure, do not assume.

## 8. Build order
1. Define the context representation + the transform with §5's priors as explicit structure.
2. Port the CONTACT case onto it; pin with `test_push`.
3. Port the STATE-CONDITIONED case onto it; pin with `test_key_discovery` + `test_object_dynamics`.
4. RETIRE `behavior.ContactDynamics`, the column's `_cue_weights`/`learn_object_move`/`predict_object_move`, and the four agent
   passthroughs — one owner, no parallel mechanism.
5. Wire the projections properly: L5IT for the proposal (fixing §4's defect), L5PT for enact/efference; L5b lateral voting when
   voting is moved out of the thalamus.
