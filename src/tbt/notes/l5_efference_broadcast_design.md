# L5IT/PT + the EFFERENCE BROADCAST — design pass (the moving-sensor fix), 2026-07-18

Mechanism check before code (`feedback_check_tbt_accuracy_per_step`). Sources: `reference_layer5_role`,
`reference_cortical_layers_research` (+ `notes/cortical_layers_and_sublaminae_research.md`), `reference_l5_operator_kinds`,
`reference_hippocampus` (the moving-sensor problem + its UPDATE 2026-07-18), ARCHITECTURE.md §9/§12. Verified against `column.py`.

## §1 THE MECHANISM (verified)
L5 is TWO projection classes, not one layer (Harris & Shepherd 2015):
- **L5IT** (slender, intratelencephalic) — cortico-cortical + striatum→BG; the ASSOCIATIVE INTEGRATOR; **NOT motor**.
- **L5PT** (thick-tufted, pyramidal-tract) — subcortical **MOTOR** + higher-order-thalamus **DRIVER**. Its **branching axon
  IS the efference copy up the hierarchy** (`reference_cortical_layers_research`). Hawkins: L5PT cells "alternately represent
  movements sent subcortically, and compositional objects sent up." Displacement cells = L5PT (TBT prediction).

`reference_layer5_role`, the unification: **L5 emits the chosen DISPLACEMENT, which is simultaneously the MOTOR COMMAND (enact
it), the EFFERENCE COPY (predict its effect), and the FEED-FORWARD message to OTHER columns via the higher-order thalamus. One
object, three roles.** The moving-sensor problem is precisely the THIRD role missing.

## §2 CURRENT STATE (verified against column.py)
- **BUILT / substituted:** the DISPLACEMENT function — `self.operator` (ego=True, the SELF's body path-integration = L6a),
  `self.dynamics` (ego=False, an object's motion = the L5 engine), `relate`/`observe_relation`/`relation_of` (L5PT displacement:
  location+location→relation). The `_Readout` in `agent.py` substitutes L5's MOTOR OUTPUT (but it lives OUTSIDE the column).
- **DECLARED but UNDRIVEN:** `layers["L5IT"]` (proximal=L23 → IT), `layers["L5PT"]` (proximal=L5IT, context=goal → PT) — the
  §12 wiring metadata is right, but nothing `.observe()`s them.
- **MISSING (the gap):** the HO-thalamus efference **BROADCAST to other columns**. One column (the nav/self column) has the
  efference (it path-integrates the body); the OTHERS do not, so if the body moves they cannot tell self-motion from
  world-motion (`reference_hippocampus`, "one column had the efference, the others did not").

## §3 THE PROBLEM, AND WHY THE GROUND IS NOW GOOD
Egocentric sensing: the body moves by Δ ⇒ a STATIC thing appears to shift by −Δ. To read that shift correctly you must
SUBTRACT the predicted self-flow (flow parsing, Rushton & Warren) using the efference; the residual is the true world-motion.
The destination frame for the corrected observations is the ALLOCENTRIC map — which we BUILT this session (`hippocampus/map.py`;
`reference_hippocampus` UPDATE). So piece 1 (the allocentric frame) is DONE; piece 2 (the efference broadcast) is this slice.
`reference_l5_operator_kinds`: there is NO self-vs-other operator split — the **self is the DISCOVERED controllable ROOT** (the
object whose motion correlates with the action); the efference is just the action applied to the root, read by everything else.

## §4 DESIGN DECISION — build ALL of L5 honestly; NO inert placeholders, NO deferral
CORRECTION (user, 2026-07-18). The first pass deferred L5IT and kept the declared-but-undriven `layers["L5IT"]`/`layers["L5PT"]`
HTMLayers. Wrong on two counts: (1) an inert HTMLayer carrying a docstring about "displacement / motor / efference copy" is DEAD
WEIGHT that LIES about what it does — a reader will see it; (2) "not needed for ARC" is not a valid reason to half-build a brain
region — this is a TBT brain, not an ARC-shaped hack, and we do not even know real ARC-AGI-3 games are all full-observation.
**Every declared part of L5 does its advertised job, or it does not exist.**

The technical point that STANDS: the CONTINUOUS displacement is the `MotionOperator`'s job, not an HTMLayer's (SDR sequence
memory is the wrong tool for continuous geometry). So the fix is NOT to force the displacement through the L5PT HTMLayer — it is
to make L5's REAL mechanism explicit + complete and DELETE the inert placeholder (or give it its genuine DISCRETE job).

**The honest L5 — both classes BUILT:**
- **L5PT** (thick-tufted, PT — the OUTPUT layer). Its geometric ENGINE = the operator/dynamics/relate (already built; now
  explicitly OWNED by L5PT, not sitting beside a dead HTMLayer). Build its MISSING outputs:
  - **EFFERENCE** — `Column.efference(action)` = the WORLD-frame self-displacement (`R·d`, exactly what `path_integrate`
    applies); broadcast via the Phase-5 thalamus register to peer columns; `Column.apply_efference(Δ)` = a peer path-integrates
    its OWN L6a by the self-motion (flow parsing). An object's world-motion = egocentric shift + efference: a STATIC object nets
    to 0 (no false dynamics), a MOVING one nets to its real motion. [the moving-sensor fix]
  - **MOTOR** — fold the agent's `_Readout` decode INTO the column: L5 IS the cortex's motor output, so the displacement→action
    decode belongs in L5PT, not outside it.
  - the `layers["L5PT"]` HTMLayer earns its keep as the discrete goal-gated motor/affordance SELECTION (object+goal → the action
    to enact), the operator supplying that action's geometry — OR it is removed. Nothing inert.
- **L5IT** (slender, IT — the INTEGRATOR). BUILD it (not deferred): observe L2/3's object identity → project it to the BASAL
  GANGLIA as the selection context (which the ad-hoc `_decision_col` in `agent.py` currently FAKES) + gate L5PT. This retires
  the decision-column hack: the sensory/task columns feed the BG through their OWN L5IT, as the cortex does.

Result: no undriven placeholders, no L5 function scattered OUTSIDE the column (`_Readout`, `_decision_col`), no deferral. The
self = the discovered controllable ROOT (`reference_l5_operator_kinds`); the efference is the action on the root, read by all.

## §5 THE MINIMAL EXERCISABLE SLICE + FALSIFIER
Two spatial columns (a SELF/root column + a PEER column), one body move:
1. the body takes an action; the self column path-integrates; `efference(action)` = the world-frame Δ.
2. broadcast Δ; the peer applies it (path-integrates its L6a by the self-motion).
3. the peer, observing a STATIC object, computes its world-motion = egocentric shift + Δ = **0** ⇒ learns NO false dynamics;
   a MOVING object nets to its true motion ⇒ learned correctly.
**Falsifier (both sides):** WITHOUT the broadcast, the peer attributes the −Δ egocentric shift to the object ⇒ spuriously
learns "the static object moved by −Δ"; WITH it, the static object is correctly static. (This reuses `dynamics`/`learn_object_move`
— the world-motion is what the dynamics should see.) The self-vs-world separation is the property tested.

## §6 SCOPE — only GENUINE limits (no ARC-only excuses, nothing half-built)
- **Vestibular / independent self-motion estimate** — NOT needed and NOT a dodge: our efference IS the motor-command copy (the
  operator's learned Δ), and the action is known, so there is nothing to estimate independently.
- **The ego→allo gain-field transform for a sensor whose OWN ORIENTATION is uncertain** (retrosplenial) — a genuine FRONTIER,
  named not deferred: the world-frame Δ we broadcast assumes the self's orientation is tracked; a sensor that must ALSO infer
  its heading from flow is harder, and that is the moving-camera game where it gets built + tested. Every part below IS built now.

## §7 BUILD ORDER (each: wired from agent.py + a falsifier; suite GREEN throughout)
1. **L5PT EFFERENCE ✅ DONE (2026-07-18, `test_efference`)** — `Column.efference(action)` (world-frame Δ) + `apply_efference(Δ)`
   (peer L6a correction) + `to_world(ego)` (egocentric→world flow-parsing output); `Agent.broadcast_efference` (the agent as the
   thalamic relay — the efference is a motor message, not a content⊗location binding, so it is routed by the plumbing, not
   forced through `bind`/`bundle`). Falsifier passes both sides: a static object stays static WITH the efference (20,10) and is
   misread as moved WITHOUT (19,10); a moving object nets to its true world-motion. [the moving-sensor fix substrate]
2. **L5PT MOTOR** — fold the `_Readout` decode INTO the column as L5PT's motor output (the displacement/action enacted); the
   agent stops owning the readout. Falsifier: the column emits the action; the decision test still passes.
3. **L5IT → BG** — the column's L5IT projects its object-state to the basal ganglia (retire `_decision_col`); the decision loop
   runs cortex(L5IT)→BG→thalamus→L5PT(motor). Falsifier: the BG selection test passes driven by a real column's L5IT.
4. **CLEANUP** — no inert `layers[...]` entry remains (each driven or deleted); reconcile ARCHITECTURE §9/§12/§15 + STATUS so
   the docs = the code. L5 is whole: every declared part does its advertised job.
