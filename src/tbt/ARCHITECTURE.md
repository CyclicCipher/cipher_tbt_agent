# ARCHITECTURE — the one model, the rules, the plan

*The single source of truth for the TBT agent. If the code and this document disagree, one of them is a bug. This
document must remain explainable, in full, to anyone fluent in the domain jargon — if it cannot, the architecture is
wrong and the architecture is what changes. It supersedes the older plan web (`L6_NONABELIAN.md`, `MATH_PHASE.md`,
`VECTOR_NAV_PLAN.md`, `FORWARD_MODEL_PLAN.md`, `COLUMN_AUDIT.md`, `GROUNDING_PLAN.md`, …), which are demoted to background
references. There is one plan, here.*

---

## 1. The model, in one paragraph (the paper abstract)

A single reusable **cortical column** learns the structure of any domain as a navigable **reference frame** and predicts
within it. **L6** is the *location*: an online successor-representation frame whose eigenvectors are grid cells
(Stachenfeld 2017). **L5** is the *operator*: one learned per-action transition on the location code — a group
representation, of which additive translation is the abelian special case and rotations/orderings are the non-abelian
general case. **Path integration is applying the operator.** **L4** is *content*: the feature bound at a location
(feature-at-location). The **forward model** is L5's operator predicting the next L4 content at the L6 location it
path-integrates to, given an action — *predict the next observation given position and action* (TEM, Whittington 2020).
**L2/3** recognises **objects** — a stored map of content-at-displacements — by incremental evidence voting, inferring
pose so a known object is recognised at a pose never seen. A domain-agnostic **value critic** learns expected future
reward (walls/hazards are negative value, not special objects); the **basal ganglia** select which goal-state to bring
about; the column's **motor** acts to fulfil the prediction (active inference). Composed through the **thalamus**
(content⊗location binding), columns vote — the Thousand-Brains consensus. No hand-coded rules and no domain priors:
structure, content, value, and goals are all learned online from the sparse signal.

## 2. The model, one line per part (the simple explanation)

| part | question it answers | the one mechanism |
|---|---|---|
| **L6** | *where am I?* | a learned SR location frame (grid cells = its eigenvectors) |
| **L5** | *how does it change?* | one learned operator per action; path integration = applying it |
| **L4** | *what is here?* | content bound at a location |
| **forward model** | *what will I see next?* | L5's operator → L4 content at the next L6 location |
| **L2/3** | *which object is this?* | recognition by evidence voting; pose inferred |
| **value** | *how good?* | expected reward + cost, one currency |
| **basal ganglia** | *what should I do?* | select the goal-state to bring about |
| **thalamus** | *do the columns agree?* | bind content⊗location and vote |

## 3. Vocabulary — one definition each (the anti-spaghetti glossary)

Every term has EXACTLY this meaning everywhere in the code and the docs. A second meaning is a bug, not a nuance.

- **location** `g` — a point in a learned reference frame = an SR code (L6). The ONLY location representation. There is
  no separate fovea / pose-matrix / binned-node belief; those were parallel estimators (deleted, see §6 P0).
- **operator** — the learned per-action transition on the location code (L5), a group-representation matrix. The ONLY
  transition and the ONLY path integrator. Translation is its abelian special case, not a second mechanism.
- **content** `x` — the feature at a location = L4's code. Location-invariant (the *what*, separate from the *where*).
- **feature-at-location** — the binding of content to a location, `g ⊗ x` (L4). The ONLY map.
- **forward model** — L5's operator predicting the next content at the path-integrated location, given an action. The
  ONLY forward model. Local/propagation dynamics live *inside* it (content over neighbouring locations), never as a
  second, location-blind predictor.
- **object** — a learned reference frame holding content-at-locations, recognised by matching sensed
  features-at-displacements to the stored map (L2/3 over L4⊗L6). NOT a segmented colour-blob, NOT a change-log, NOT a
  tracked mover. Boundaries emerge from prediction MISMATCH, not a segmentation heuristic.
- **recognition** — settling an object's identity + pose by incremental evidence voting (L2/3).
- **prediction error / surprise** — the mismatch between predicted and sensed content. The ONLY "something changed"
  signal; there is no stored change-log (`object_state`/`_changed` were bookkeeping — deleted).
- **value** — expected future reward, learned by the critic (`reward.py`). Cost (walls/hazards/slow/risky) is negative
  value in the same currency. The ONLY value; obstacles are not objects.
- **goal** — a target-state (a location/content to bring about). The agent acts to fulfil it (active inference).
- **selection** — the basal ganglia choosing among goals/actions.

## 4. The five rules (development law)

These are hard constraints. A change that breaks one is reverted, not documented around.

1. **No parallel systems — ever, including for experiments.** Exactly one way to forward-model, one way to
   path-integrate, one feature-at-location, one way a grid module learns its structure, one recogniser, one value, one
   selector. Manage comparison and risk with **git branches**, never by keeping two mechanisms in the tree. A flag that
   switches between two implementations (e.g. an abelian-vs-non-abelian `heading_dependent` fork) is a parallel system in
   disguise — forbidden; the general mechanism must contain the special case.
2. **One definition per concept.** See §3. A new meaning for an existing word is a bug to fix, not a footnote to add.
3. **The column and the agent are thin coordinators.** They hold references + routing — never math or state. Every
   belief, map, operator, and value lives in a layer/module. If `column.py` or `agent.py` grows a subsystem with its own
   state, that state belongs in a layer.
4. **No load-bearing harness, no domain-specific code, no special-casing, no ungrounded arbitration.** Nothing branches
   on which game/domain it is. Every arbitration (explore/exploit, tabular/forward, goal selection) must name the brain
   mechanism it implements (basal-ganglia selection, tonic-dopamine gain, STN commitment, …) or it is removed.
5. **No symbolic estimators, object heuristics, or change logs.** No hand-coded "what is an object / how to split it,"
   no Kalman-style tracker banks (fovea centroids, pose matrices, binned nodes kept in parallel), no dicts of "what
   changed." Structure is learned; change is carried by prediction error; the object is a recognition construct.

## 5. The architecture (who owns what; one mechanism per function)

| module | owns (the ONE mechanism) |
|---|---|
| `l6_sr` (**L6**) | the location frame — the online TD successor representation; grid cells = its eigenvectors; path integration = applying the L5 operator to the code |
| `l5_displacement` (**L5**) | the per-action **operator** (transition), the **forward model** (operator over L4-content-at-L6-location), the **motor** (act to fulfil the prediction), the thalamus **driver** |
| `l4_feature_location` (**L4**) | **feature-at-location** (bind + readout), the content codebook, the feature descriptor |
| `l23_object` (**L2/3**) | **object recognition** (evidence voting, pose inferred) and the object map; boundaries from prediction mismatch |
| `operator` | the group-representation primitive the operator IS (compose, learn, powers, relation/factor discovery) |
| `thalamus` | content⊗location binding + cross-column voting |
| `basal_ganglia` | goal/action **selection** |
| `reward` | the **value** critic (reward + cost, one currency) and planning (SR value / prioritised sweeping) |
| `column` | **routing only** — wire the layers; hold no state |
| `agent` | the **loop only** — perceive → predict → select → act; hold no solver |

## 6. The plan (one dependency-ordered spine)

The old Phase/Stage/slice/step web is retired. What actually depends on what:

- **P0 — Converge the code to this document (the deletion pass).** Make the code equal §1. Concretely: collapse the two
  forward models into one (delete the location-blind `field_rule` CA; the operator-over-content-at-location is the one);
  collapse the location estimators into the SR code path-integrated by the operator (delete the fovea / pose-matrix /
  `state_node` / `_obs` / `heading_dependent` fork); make the object a recognition construct (delete the
  colour-segmentation heuristic and `object_state`/`_changed`); move the cost field into the value critic; make `column`
  and `agent` thin (subsystems → layers). Suite-green throughout; git branches for anything risky. **This is the bulk of
  the work and it is mostly DELETION.**
- **P1 — Factored perception.** L2/3 recognition + L4 content deliver `(location, content)` factored, from the live
  frame — not a raw entangled patch. This is the prerequisite the forward model always assumed and never had (why the
  FM could not be built before).
- **P2 — The forward model.** Trivial once P1 exists: L5's operator over L4-content-at-L6-location, predicting the next
  observation. One model, one place. (This was the tangled "c".)
- **P3 — Relations & planning.** Operators (learned) → relations by loop closure → geodesic planning over the learned
  frame (SR value / prioritised sweeping, never explicit search) → the order/config-dependent case (Sokoban).
- **P4 — The goal loop.** A goal-state generator proposes target-states → basal-ganglia selection → the motor achieves
  → value confirms. The heterarchy (multi-column voting via the thalamus) scales the same loop.

Honest status: the **operator** primitive and **relation/factor discovery** (part of P3) exist and are tested; the SR is
the one L6; everything else is entangled with the estimator stack listed in P0 and must go through P0 first. We are at P0.

## 7. Acceptance test for every change (the paper test)

Before any change lands, BOTH must hold:

1. **Explainability** — the change is stateable in one sentence that fits the §1 paragraph and the §3 vocabulary. If
   explaining it needs a new term, a second meaning, or a "well, in this mode…", stop.
2. **The five rules** — it introduces no parallel system, no second definition, no coordinator bloat, no
   harness/special-case/ungrounded arbitration, and no symbolic estimator/heuristic/change-log.

If a change cannot pass both, the design is wrong. Fix the design, not the change.
