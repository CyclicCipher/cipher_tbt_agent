"""column.py — THE CORTICAL COLUMN: a composition of `HTMLayer`s wired as the real microcircuit (not a feedforward stack).

This module is written PLAN-FIRST per the user's directive ("make the whole column at once; lay out the research and the
plan in the comments"). PART A is an exhaustive, cited synthesis of the layers + sublaminae the typical TBT explanation
smooths over (from a six-thread primary-source review, 2026-07-09). PART B is the build plan for the whole column. The code
below PART B is a structural scaffold that encodes the plan; the per-timestep dynamics await the §15 decisions.

STATUS: STANDALONE / under construction (allow-listed in test_reachability). NOT wired into agent.py, NOT exercised
end-to-end yet — so by RULES.md #3 it is explicitly NOT "done." It exists so the whole-column structure + research are
captured in code before we implement the dynamics.

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
PART A — RESEARCH: the layers & sublaminae, and what the popular TBT account smooths over
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

§0. THE HEADLINE (the "smoothed over" summary)
The popular TBT column — L4 (feature input) ⇄ L2/3 (object output), location from L6a, L5 = motor, grid cells in L6 — is a
DELIBERATE, HEAVY simplification. Numenta say so: their 2017 simulation implements only TWO pyramidal layers, folds all
inhibition into activation rules (no interneurons as cells), omits L1 / sublaminae / the L2-vs-L3 distinction, admits
"cells we describe as residing in separate layers may actually intermingle," and flags grid-cell + displacement-cell
placement as PREDICTIONS, not facts (Hawkins, Ahmad & Cui 2017; Hawkins et al. 2019 "Framework"; TBP 2024). Monty (the
software) drops layers, spikes, grid cells and the thalamus entirely. The six corrections the real anatomy forces:
  (1) A "layer" is really defined by PROJECTION CLASS (IT / PT / CT), not by laminar number (Harris & Shepherd 2015).
  (2) L2/3, L5, and L6 are each really TWO classes doing different jobs.
  (3) The circuit is RECURRENT, not a relay: thalamus is only ~5% of L4's excitatory synapses; local recurrence (~34%)
      exceeds the feedforward interlaminar loop (~21%). Cortex AMPLIFIES a weak thalamic drive (Binzegger/Douglas/Martin
      2004; Douglas & Martin 2004).
  (4) It is NOT a serial L4→L2/3→L5→L6 stack: thalamus drives L5/L6 DIRECTLY, in parallel with L4 — L4 is not an
      obligatory first relay (Constantinople & Bruno 2013).
  (5) There are TWO COUNTERSTREAMS: feedforward from SUPRAGRANULAR (L2/3, → higher L4, driving) carrying prediction
      ERRORS up; feedback from INFRAGRANULAR (L5/L6, → lower L1, modulatory) carrying PREDICTIONS down (Markov 2014;
      Bastos 2012; Shipp 2007).
  (6) L1 is the top-down/feedback LANDING PAD (not "irrelevant"), and inhibition (PV/SST/VIP/NGF) does structured control
      our single kWTA does not capture.

§1. L4 — the INPUT layer (granular)
  • SUBLAMINAE: in primate V1, L4 is 4A / 4B / 4Cα / 4Cβ carrying THREE segregated thalamic streams — magno→4Cα,
    parvo→4Cβ, konio→4A — with divergent upper-layer targets (4Cα→4B→MT/motion; 4Cβ→3B/4A→form/colour) (Callaway 1998;
    Yabuta & Callaway 1998; Hendry & Reid 2000). Rodent barrel cortex sublaminates TANGENTIALLY instead: one barrel per
    whisker (Petersen 2007).
  • CELL TYPES: spiny stellate (no apical dendrite, star-shaped, LOCAL) vs star pyramid vs pyramid; ~80% stellate in
    barrel L4; all REGULAR-SPIKING; L4 excitatory cells are LOCAL/non-projecting (an intracortical amplifier stage), not
    an output class (Feldmeyer 2012).
  • THE THALAMIC MINORITY: thalamus is only ~5–20% of L4 excitatory synapses; the LARGEST source is L6 feedback (~45%),
    then recurrent L4→L4 (~28%) (Ahmed 1994; Binzegger 2004). Thalamus drives anyway because its synapses are weak but
    NUMEROUS + SYNCHRONOUS (Bruno & Sakmann 2006). L4→L2/3 is the strong, ~unidirectional feedforward step.
  • AGRANULAR CORTEX: motor + much of prefrontal cortex have NO granular L4. M1 has a cryptic L4-equivalent at the L3/5A
    border (all pyramidal, weaker/diffuse thalamic input) (Yamawaki et al. 2014). So a "uniform L4-input" column is closest
    to primary SENSORY cortex; the honest defense is serial homology (the input STAGE is conserved even where the granular
    layer is not — Harris & Shepherd 2015).
  • TBT SMOOTHS: the M/P/K sublaminar streams; spiny-stellate vs pyramid; agranular cortex; that thalamus is a minority
    input and L6 the majority; that deep layers are driven by thalamus in parallel (not strictly via L4).

§2. L2/3 — NOT one layer
  • WHY LUMPED: a rodent architectonic convenience (in rodent L2/3 is a smooth depth-CONTINUUM — Weiler 2023). In
    PRIMATE/HUMAN it is distinct: L2 vs L3 differ physiologically (L2 sparser, smaller spikes; L3 sharper RFs, direction-
    selective — Economides/Gur/Snodderly 2008); L3 is sublaminated 3A/3B/3C; human deep-L3 has distinctive long-range
    (SMI-32) cells (Berg 2021).
  • THE FUNCTIONAL SPLIT (Barone 2000): DEEP L3 (bordering L4) = FEEDFORWARD output → higher-area L4. L2 + UPPER L3 =
    FEEDBACK-biased (matrix-thalamic POm + top-down apical input) → lower-area L1/L2. L3 is more core-thalamic-coupled; L2
    more matrix/feedback.
  • CLASS + OUTPUTS: both are IT (cortico-cortical + striatum). Outputs: → L5 (the strong, conserved L2/3→L5 within-column
    descending path); → higher cortex (deep-L3 feedforward); → lower cortex (L2/upper-L3 feedback); dense horizontal
    L2/3↔L2/3 = the substrate TBT calls VOTING.
  • COMPETING STORY: a large literature makes superficial pyramids PREDICTION-ERROR units (Bastos 2012), not (only) object
    pools — a different computation for the same layer, and one the apical/feedback wiring of L2 seems built for.
  • TBT SMOOTHS: L2 vs L3; that "the output layer" is really TWO different outputs (deep-L3 FF-export vs L2/upper-L3 FB) at
    different depths; the sublaminae; primate double-bouquet minicolumn inhibition; the error-coding role.

§3. L5 — TWO classes (IT vs PT)
  • IT (intratelencephalic; slender-tufted; upper L5/L5A; REGULAR-SPIKING) → cortico-cortical + BILATERAL striatum;
    telencephalic-only; NOT a direct motor output. PT (pyramidal-tract / extratelencephalic; THICK-TUFTED; lower L5/L5B;
    INTRINSIC-BURSTING) → subcortical (brainstem, spinal cord, tectum, pons) AND higher-order thalamus as a DRIVER
    (Harris & Shepherd 2015; Frontiers 2022; Shepherd 2013).
  • IT→PT is UNIDIRECTIONAL: IT integrates local computation and gates it into PT, which broadcasts. Effective chain:
    L4 → L2/3 → L5-IT → L5-PT → subcortical/thalamic.
  • PT IS THE TRUE "MOTOR OUTPUT" — and more: a SINGLE PT axon BRANCHES to a subcortical motor target AND to higher-order
    thalamus, which relays to the next cortical area. So PT emits a motor command AND a COPY of it up the hierarchy (a
    built-in EFFERENCE COPY / transthalamic feedforward) (Frontiers 2021; Sherman & Guillery 2024).
  • LARKUM (the apical mechanism): the thick-tufted L5 cell is a TWO-COMPARTMENT coincidence detector — basal (feedforward)
    × apical-tuft-in-L1 (feedback), within a ~5–30 ms NMDA window, triggers a Ca²⁺ BAC spike → a BURST. This is the
    BIOLOGICAL ARCHETYPE of HTM's apical tiebreak — but it is a MULTIPLICATIVE gain/burst, whereas HTM approximates it as a
    subthreshold depolarization + earlier-firing + WTA (same computation, different currency) (Larkum 2013).
  • DISPLACEMENT CELLS (object composition; behavior = a sequence of displacement vectors) are HYPOTHESIZED in L5
    thick-tufted — a TBT PREDICTION, unconfirmed, and in tension with grid cells being placed in L6a (Framework 2019).
  • TBT SMOOTHS: IT vs PT (calls all of L5 "motor"); the HO-thalamus DRIVER / transthalamic hierarchy role (L5's arguably
    most important job); the multiplicative Larkum gate (→ scalar depolarization); that displacement-cell placement is a
    bare hypothesis.

§4. L6 — TWO classes (CT vs CC) + L6b
  • CT (corticothalamic; upright; 30–50% of L6) → thalamus + the reticular nucleus (nRT, the attentional gate);
    MODULATORY. CC (corticocortical; inverted/horizontal) → lateral cortex. Plus L6b (a SUBPLATE remnant; distinct
    markers; orexin/arousal-sensitive; → higher-order thalamus) (Thomson 2010; Marx & Feldmeyer 2017).
  • THE L6→L4 LOOP: numerically LARGE (~45% of L4 synapses) but individually WEAK & MODULATORY (facilitating, mGluR, onto
    DISTAL dendrites) = GAIN CONTROL, not drive; reciprocal and topographically narrow (Binzegger 2004; Briggs 2010). This
    is the anatomical hook TBT hangs "L6a location drives L4" on — and TBT is honest the connection is weak/distal.
  • L6-CT → thalamus is a MODULATOR (vs L5-PT = driver); core vs matrix; L6 uniquely reaches nRT.
  • GRID CELLS IN L6 = a TBT PREDICTION. Grid cells are established in ENTORHINAL cortex, NOT observed in sensory-cortex L6
    ("our prediction is they will be in L6"; the evidence "is mute on what layers contain grid cells" — Framework 2019).
  • TBT SMOOTHS: CT vs CC (it uses the CC/L6a cell for the L4 loop and sets aside CT→thalamus gain control — L6's defining
    job); L6a vs L6b; that the L6→L4 signal is physiologically GAIN CONTROL (relabeled "location"); the unobserved
    grid-cell claim.

§5. L1 — the FEEDBACK LANDING PAD (not irrelevant; our old notes were wrong to call it so)
  • CONTAINS: the apical TUFTS of L2/3 + L5 pyramids; long-range cortico-cortical FEEDBACK axons; thalamic MATRIX axons;
    L1 interneurons (neurogliaform → slow GABA_B, canopy, VIP, α7); dense cholinergic/neuromodulatory input (brain-state).
    Cajal-Retzius cells are developmental/transient (Schuman et al. 2021).
  • ROLE: the substrate for top-down × bottom-up COINCIDENCE (Larkum apical amplification) and for DISINHIBITORY gating of
    when that amplification is allowed. HTM's apical dendrite ("predict, don't fire") IS this — what HTM omits is the L1
    INTERNEURON control of the apical channel.

§6. INHIBITION — three groups ≈ 100% of interneurons
  • PV (~40%; perisomatic basket + axo-axonic chandelier) = gain / spike-timing / gamma + output veto at the AIS ≈ what
    our kWTA already does. SST (~30%; Martinotti, axon ascends to L1) = DENDRITIC inhibition = the knob on the apical /
    PREDICTION channel. VIP (disinhibition: VIP→SST→pyramid) = the CONTEXT / attention / plasticity GATE that transiently
    unlocks apical amplification. NGF (L1; slow GABA_B volume transmission) = diffuse blanket (Tremblay/Lee/Rudy 2016;
    Pfeffer 2013; Pi 2013).
  • OUR kWTA ≈ PV PERISOMATIC ONLY. Missing: SST (dynamic dendritic gate on the prediction channel) and VIP (context-gated
    unlock of the top-down channel + learning). Matters once the column must LEARN dynamic control of its feedback.

§7. THE REAL WIRING — recurrent + two counterstreams (NOT a stack)
  • RECURRENCE DOMINATES: thalamus ~5% of L4 synapses; intralaminar self-connection ~34% > feedforward interlaminar ~21%
    (Binzegger 2004). Deep layers driven directly by thalamus IN PARALLEL with L4 (Constantinople & Bruno 2013).
  • DIRECTED EXCITATORY CONNECTIONS (rough): thalamic-core → L4 (+ direct to L5B/L6); L4 → L2/3 (strong FF, ~one-way);
    L2/3 → L5 (strong, conserved); L5 → L2/3 (WEAK); L6 → L4 (numerous, weak, modulatory); L5-IT → L5-PT (one-way);
    L5 ↔ L6; local recurrence everywhere.
  • TWO COUNTERSTREAMS: FEEDFORWARD from SUPRAGRANULAR (L2/3) → higher-area L4 — driving, gamma, = prediction ERRORS up.
    FEEDBACK from INFRAGRANULAR (L5/L6) → lower-area L1 (AVOIDS L4) — modulatory, alpha/beta, = PREDICTIONS down (Markov
    2014; Bastos 2012). Superficial = error units; deep = expectation/prediction units.
  • THREE OUTPUT "MOUTHS": L2/3-IT → cortico-cortical; L5B-PT → subcortical motor (+ HO thalamus); L6-CT → thalamus.

§8. THALAMUS — core vs matrix; driver vs modulator
  • CORE (parvalbumin⁺; → L4 focal, topographic; DRIVER; feedforward; specific content) vs MATRIX (calbindin⁺; → L1 + L5a,
    diffuse; MODULATOR; feedback/synchronizing) (Jones 2001). Matrix→L1 converges with cortical feedback on the SAME apical
    tufts. Core ≈ feedforward-drive, matrix ≈ feedback-context.
  • DRIVER vs MODULATOR (Sherman & Guillery) is the more robust axis: L5-PT → HO thalamus = DRIVER (the transthalamic route
    that builds the next cortical level); L6-CT → thalamus = MODULATOR. Core/matrix as a hard partition is contested (2024
    reappraisal) — hold it loosely.

§9. TBT's OWN mapping + what Numenta explicitly admit
  • Input = L4; Output = L2/3; Location = L6a; grid cells PREDICTED in L6. "Possibly TWICE": the same input/output motif in
    L4↔L2/3 AND in L6a↔L5. L5 = displacement cells (thick-tufted), TIME-MULTIPLEXING movement (sent subcortically) and the
    compositional object (sent via HO thalamus to higher regions). Hierarchy = columnar + transthalamic + lateral VOTING.
  • ADMITTED SIMPLIFICATIONS: 2 pyramidal layers simulated; inhibition = activation rules, not cells; L1 / sublaminae /
    L2-vs-L3 omitted; grid + displacement placement = predictions; Monty drops layers, spikes, grid cells, and the thalamus.
  • BIGGEST DIVERGENCE from the mainstream canonical microcircuit: TBT turns the L6a→L4 MODULATORY/gain-control feedback
    into an active GRID-CELL LOCATION code, and claims grid cells exist in EVERY column.

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
PART B — THE PLAN: how we build the whole column
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

§10. DESIGN PRINCIPLES
  P1. PROJECT BY CLASS, NOT LAMINAR NUMBER (Harris & Shepherd). A layer = ONE `HTMLayer` + a declared
      (proximal-in, context-in, apical-in, target-out); `target-out` IS the projection class (IT / PT / CT / local /
      FF-export / FB). This is exactly the htm.py wiring framing.
  P2. TWO COUNTERSTREAMS = PREDICTION vs ERROR. Deep (L5/L6) output = the PREDICTION / feedback-down (+ subcortical /
      thalamic). Superficial (L2/3) feedforward-up = the prediction ERROR (our HTM burst / anomaly). This ALIGNS HTM's own
      predictive-cells (prediction) + burst (error) with the anatomy and with predictive coding (Bastos 2012).
  P3. STRUCTURE IS SUPPLIED, NOT DISCOVERED (ARCHITECTURE §7/§8; the credit-assignment finding). Each layer's role is its
      WIRING, asserted by hand; the location code (L6a) and any task-state channel are GIVEN, not learned.
  P4. UNIFORM MECHANISM. Every layer is the ONE `HTMLayer`. Proximal is column-level (an upstream `SpatialPooler` only at a
      raw front-end; otherwise a lower layer's output SDR taken directly). We do NOT add proximal dendrites to `HTMLayer`.
  P5. HONEST ABSTRACTION (RULES: no silent omissions). We model the FUNCTIONAL skeleton and NOTE every sublamina /
      inhibitory / thalamic detail we abstract, so it can be added when a task needs it. Nothing is silently dropped.

§11. THE LAYERS WE MODEL (and the sublaminae we abstract, noted)
  L4    — granular INPUT (spiny-stellate feature-at-location).        model: YES. abstract: M/P/K streams (one channel),
          star-pyramid, agranular L4-equivalent.
  L23   — supragranular IT OUTPUT/object + lateral VOTING.            model: YES, as one layer with TWO output wires
          (deep-L3 → FF-export; L2/upper-L3 → FB). abstract: L2-vs-L3 as separate cell pops, 3A/3B/3C, double-bouquet.
  L5IT  — slender-tufted associative INTEGRATOR.                      model: YES (recommended) or fold into one L5 first.
  L5PT  — thick-tufted DRIVER/EFFECTOR = motor + efference-copy-to-  model: YES. abstract: intrinsic bursting; the Larkum
          hierarchy + displacement cells.                             multiplicative gate (approximated by apical tiebreak).
  L6a   — corticocortical LOCATION (grid cells) + the L4 gain loop.   model: YES. abstract: L6b; L6-CT thalamic gain control.
  L1    — the apical/FEEDBACK landing (NOT a cell layer).             model: as the apical-IN channel. abstract: SST/VIP.
  Inhibition: kWTA (≈ PV) via the SpatialPooler / HTMLayer WTA. SST/VIP dynamics DEFERRED (§14).
  Thalamus: partly EXTERNAL (the input source) + the L6-CT / L5-PT outputs. core/matrix + full transthalamic relay
  DEFERRED (§14) — the L5-PT→HO-thalamus→higher-region hop is approximated as a direct cross-region wire for now.

§12. THE WIRING TABLE  (per layer:  proximal-in │ context-in (basal) │ apical-in (feedback) │ target-out)
  L4   │ thalamic/sensory core via SP, OR a lower region's output │ L6a LOCATION │ L2/3 object feedback │ → L2/3 (local FF)
  L23  │ L4                                                       │ own recurrence (object pooling) + lateral VOTING (peer
         columns) │ L1 top-down feedback │ → {deep-L3: higher-region L4 (FF/error-up); L2: lower-region L1 (FB); → L5}
  L5IT │ L2/3                                                     │ recurrence + direct thalamus │ L1 │ → cortico-cortical
         + striatum (→ basal ganglia)
  L5PT │ L5IT (IT→PT)                                             │ sensorimotor state + top-down GOAL │ L1 (the Larkum
         coincidence = the apical tiebreak/gate) │ → {subcortical MOTOR; HO thalamus → higher region = EFFERENCE COPY};
         these ARE the displacement cells
  L6a  │ L5 movement/efference (path integration) + thalamus      │ own recurrence (the grid path-integration operator) │
         (apical: t.b.d.) │ → L4 (LOCATION, modulatory) + thalamus (CT gain)
  Note the two load-bearing loops: L6a↔L4 (location predicts the next feature) and L5PT→L6a (the efference copy
  path-integrates the location for the next step). The path-integration ITSELF is the TRANSFORM primitive
  (`operator.MotionOperator`), NOT the L6a HTMLayer's sequence memory — ARCHITECTURE §8 (a memorised per-position
  transition fails place-invariance); the L6a HTMLayer remains for L6a's ASSOCIATE roles.

§13. STEP ORDER + COUNTERSTREAM DATAFLOW (per timestep = one "movement")
  1. FEEDBACK/prediction settles first (deep→superficial): the efference copy from L5PT path-integrates L6a → L6a location
     becomes L4's basal context; L2/3 object → L4 apical; higher-region feedback → L1.
  2. FEEDFORWARD/drive: new sensory input → L4 proximal; L4's predicted-vs-actual mismatch = the burst/ERROR; L4 → L2/3
     (pool + vote); L2/3 → L5IT → L5PT.
  3. OUTPUT: L5PT emits the motor command (subcortical) + the efference copy (→ L6a next step, + HO thalamus → higher
     region); deep-L3 → higher-region L4 (FF/error-up); L6-CT → thalamus (gain).
  Recurrence: each layer's basal context defaults to its own previous active cells (temporal memory), EXCEPT where an
  external context is wired (L4←L6a; L5PT←goal). NOT a serial stack: L6a/L5 (deep) may update in parallel with L4.

§14. WHAT WE DEFER (with reasons — noted, never silently dropped)
  • SST/VIP inhibitory DYNAMICS (learned dendritic gating of the prediction channel + plasticity control) — kWTA(PV)
    suffices for a first column; add when a task needs learned top-down control.
  • FULL THALAMUS (core/matrix as stages; nRT attention; transthalamic relay as its own node) — approximated by direct
    cross-region wires + the L6-CT/L5-PT outputs.
  • L4 M/P/K streams; agranular L4-equivalent; L6b; primate double-bouquet minicolumn inhibition — single-channel
    abstractions.
  • The Larkum MULTIPLICATIVE burst — approximated by HTM's apical tiebreak (depolarize + earlier-fire + WTA); it is a gain
    approximation, flagged.
  • L2-vs-L3 as separate cell populations — modeled as ONE L2/3 with two OUTPUT wires (deep-L3 FF, L2 FB) first.

§15. DECISIONS (LOCKED 2026-07-10 — the user took the recommendations; first slice = the sensorimotor increment)
  D1 → SPLIT L5 into IT + PT (kept in the composition below).
  D2 → L2/3 = ONE layer with two OUTPUT wires (FF_EXPORT + FB), not two cell populations.
  D3 → FULL structure, SUBSET-driven first: the first runnable slice drives L4 (feature) conditioned on a factored STATE
       carried by a second (TASK) column (project_place_invariance_needs_factored_state); L2/3, L5IT, L5PT, L6a are present
       and wired but not yet driven.
  D4 → adopt "superficial = prediction ERROR / feedforward-up; deep = PREDICTION / feedback-down" as the organizing principle.
  FIRST SLICE (wired 2026-07-10): `Column.observe` (below) drives L4; `agent.Agent` composes a SENSORY + a TASK column and
  runs the sensorimotor increment; `test_column_arithmetic` reproduces the place-invariance win (a place never trained ≈100%)
  through the real column composition — turning option2.py's validated mechanism into wired architecture. SIMPLIFICATIONS
  (honest, per RULES): the factored state enters L4 via the PROXIMAL path (concatenated, then SP), not the basal `context=`
  channel — a single HTMLayer's active cells are content-determined, so proximal is what reaches a cell-readout (htm.py
  §PROXIMAL); order=1 (the successor is a first-order map). Basal-context + high-order are refinements for a spatial task.

FULL RESEARCH (unabridged, every citation + URL + uncertainty flag): `src/tbt/notes/cortical_layers_and_sublaminae_research.md`.
Sources (compact): Harris & Shepherd 2015 Nat Neurosci; Douglas & Martin 2004
Annu Rev Neurosci; Binzegger/Douglas/Martin 2004 J Neurosci; Thomson & Lamy 2007 Front Neurosci; Bastos et al. 2012 Neuron;
Markov et al. 2014 J Comp Neurol; Constantinople & Bruno 2013 Science; Bruno & Sakmann 2006 Science; Callaway 1998 & Yabuta
& Callaway 1998 (V1 sublaminae); Feldmeyer 2012 Front Neuroanat (barrel L4); Yamawaki et al. 2014 eLife (M1 "L4");
Economides/Gur/Snodderly 2008 & Berg et al. 2021 Nature (L2/3); Larkum 2013 Trends Neurosci; Frontiers 2022 (L5 IT/PT);
Sherman & Guillery 2024 J Neurosci (transthalamic); Thomson 2010 & Briggs 2010 & Marx & Feldmeyer 2017 (L6/L6b); Schuman et
al. 2021 Annu Rev Neurosci (L1); Tremblay/Lee/Rudy 2016 Neuron (interneurons); Jones 2001 Trends Neurosci (core/matrix);
Hawkins/Ahmad/Cui 2017, Hawkins et al. 2019 "Framework", Lewis et al. 2019, TBP 2024 (the TBT mapping + admissions).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Optional

from tbt.encoders import SDR, GridEncoder, SpatialPooler
from tbt.htm import HTMLayer
from tbt.operator import MotionOperator, compose, dist, eye, gram_schmidt, invert, rotate, solve_rotation, sub
from tbt.pooler import ColumnPooler

_TOL = 1e-6          # geometric slack: distances are exact floats here, so this only absorbs round-off


@dataclass(frozen=True)
class Hypothesis:
    """ONE (object, pose) hypothesis with its accumulated EVIDENCE — Monty's unit of recognition
    (`reference_tbt_pose_invariant_recognition`): the object's IDENTITY plus its pose, `pose = orientation + location`, both
    CONTINUOUS (never discretised). `rotation` = the object's orientation as an n×n ROTATION MATRIX (Monty's "three
    orthonormal vectors"; `operator.to_angle` reads it as degrees in 2-D, and SO(3) has no such scalar). `origin` = where the
    object's frame origin sits in the sensor's frame, so a model point ℓ is sensed at `rotate(rotation, ℓ) + origin`. The
    identity SDR is L2/3's; `label` is its library index (the read-out). `evidence` RANKS rival hypotheses; `refuted_at` is the
    INDEX of the first fixation this hypothesis fails to explain (the model has something else there, or nothing), or None if
    it explains the whole sweep. That index answers two questions at once — *is* it refuted (which is ART's VIGILANCE at
    ρ=1: nothing may go unexplained) and *where* (the object BOUNDARY, `Column.commit`).

    **`choice` is what RANKS rivals** — ART's `T_j = |I ∧ w_j| / (α + |w_j|)`, normalised by the CATEGORY, so the smallest
    model that explains the sweep wins (`ColumnPooler.choice`). `evidence` is the raw accumulated support, kept because it
    breaks exact ties and reads well in a trace — but it must NOT rank, since a sum lets a big model seen partly tie a small
    model seen whole."""
    identity: frozenset
    label: int
    rotation: tuple
    origin: tuple
    evidence: float
    refuted_at: Optional[int] = None
    choice: float = 0.0


# ── projection classes (the target-out half of a layer's wiring; §10 P1) ───────────────────────────────────────────
IT = "IT"            # intratelencephalic  — cortico-cortical + striatum (L2/3, L5a)
PT = "PT"            # pyramidal-tract     — subcortical MOTOR + higher-order thalamus DRIVER (L5b) = efference copy
CT = "CT"            # corticothalamic     — thalamus MODULATOR + nRT (L6)
LOCAL = "local"      # non-projecting local amplifier (L4)
FF_EXPORT = "ff"     # feedforward corticocortical → higher region's L4 (deep L3) — carries prediction ERROR (§10 P2)
FB = "fb"            # feedback corticocortical → lower region's L1 (L2/upper-L3, L5/L6) — carries PREDICTION


@dataclass
class Layer:
    """ONE cortical layer = one HTMLayer + its declared wiring (§10 P1, htm.py docstring). A layer is NOT a subclass; the
    mechanism is uniform and the differentiation is entirely in these fields. `sp` is present ONLY at a raw front-end
    (proximal = column-level SpatialPooler); elsewhere proximal-in is another layer's output SDR taken directly (§10 P4)."""
    name: str
    htm: HTMLayer
    target_out: tuple                       # projection class(es): which of IT/PT/CT/LOCAL/FF_EXPORT/FB this layer drives
    sp: Optional[SpatialPooler] = None      # proximal front-end (raw input → columns); None = takes an SDR directly
    proximal_from: Optional[str] = None     # source of the feedforward drive (a layer name, "sensory", or "efference")
    context_from: Optional[str] = None      # basal context source ("recurrence" | "location" | "goal" | a layer name)
    apical_from: Optional[str] = None       # apical feedback source ("L1" | a higher-region signal)


class Column:
    """A cortical column: L4, L2/3, L5(IT,PT), L6a as ONE HTMLayer each, wired per the §12 table. Composition + wiring are
    real here; the full counterstream `step` (§13) is the target — the FIRST SLICE (§15 D3) drives L4 only, via `observe`.

    ALWAYS-multi-column note (ARCHITECTURE §5.1): a Column is instantiated at least twice (a SENSORY column on physical
    space and a PFC/TASK column on an abstract space) — same class, different reference frame. The task column's output
    conditions the sensory column's L4 (the factored-state mechanism from project_place_invariance_needs_factored_state).
    `order` sets every layer's HTM order — 2+ (high-order, real sensorimotor sequences) by default; 1 for the first-order
    arithmetic first slice. Pass `location=GridEncoder(...)` for a SPATIAL column: L6a then gains the TRANSFORM primitive
    (`operator.MotionOperator`) and the path-integration API (`locate`/`learn_move`/`path_integrate`/`where`, ARCHITECTURE §8).
    The location frame's `dims` sets the space: n=2 for an ARC frame, n=3 for a 3-D environment — ONE mechanism either way,
    because the pose's orientation is a rotation MATRIX rather than a scalar angle."""

    def __init__(self, sensory_n: int, n_cols: int = 1024, order: int = 2, seed: int = 0,
                 location: Optional[GridEncoder] = None) -> None:
        # Proximal front-end: only L4 turns a RAW input space into columns (§10 P4, §12). All other layers take SDRs.
        self.n_cols = int(n_cols)
        l4_sp = SpatialPooler(n_inputs=sensory_n, n_cols=n_cols, seed=seed)
        # L4's basal threshold: for a SPATIAL column the L6a location drives L4's context, and a SHARP place field is needed
        # (adjacent grid codes overlap heavily) — require most of the location's active bits to match (ARCHITECTURE §8). A
        # plain (non-spatial) column keeps the HTMLayer default.
        if location is not None:
            loc_w = location.mw * len(location.modules())            # the location code's active-bit count
            thr = max(2, round(0.75 * loc_w))                        # a sharp place field: predict AND pick a winner cell only
            l4_htm = HTMLayer(order=order, activation_threshold=thr, min_threshold=thr)   # at a near-exact location match, so
        else:                                                        # a feature at (0,0) vs (1,0) uses DIFFERENT cells
            l4_htm = HTMLayer(order=order)
        self.layers: dict[str, Layer] = {
            # feature-at-location: proximal sensory (via SP), basal context = L6a location, apical = L2/3 object feedback.
            "L4":   Layer("L4", l4_htm, target_out=(LOCAL,), sp=l4_sp,
                          proximal_from="sensory", context_from="location", apical_from="L23"),
            # object / output + voting: proximal L4, basal = own recurrence + peer-column votes, apical = L1 top-down. The
            # temporal POOLING that forms the stable object IDENTITY is `self.pooler` (a spatial column), ARCHITECTURE §8 —
            # a decoupled stable output this HTMLayer can't express; this HTMLayer entry is reserved for L2/3's other roles.
            "L23":  Layer("L23", HTMLayer(order=order), target_out=(FF_EXPORT, FB, IT),
                          proximal_from="L4", context_from="recurrence", apical_from="L1"),
            # associative integrator: proximal L2/3, → cortico-cortical + striatum(→BG).  (D1: split kept.)
            "L5IT": Layer("L5IT", HTMLayer(order=order), target_out=(IT,),
                          proximal_from="L23", context_from="recurrence", apical_from="L1"),
            # driver/effector = displacement cells: proximal L5IT (IT→PT), basal = state+goal, apical = Larkum gate;
            # → subcortical MOTOR + HO-thalamus efference copy.
            "L5PT": Layer("L5PT", HTMLayer(order=order), target_out=(PT,),
                          proximal_from="L5IT", context_from="goal", apical_from="L1"),
            # location / grid: proximal = L5 efference (path integration) + thalamus, basal = grid recurrence,
            # → L4 (location, modulatory) + thalamus (CT gain).
            "L6a":  Layer("L6a", HTMLayer(order=order), target_out=(CT, LOCAL),
                          proximal_from="efference", context_from="recurrence", apical_from=None),
        }
        # L6a — the TRANSFORM engine (ARCHITECTURE §8). The location STATE is CONTINUOUS (`_pose` = (position, R)) and
        # the grid code is a per-fixation READ-OUT of it (`_code`). That split is the 2026-07-14 cut-over: a discrete SDR
        # state made each action a PERMUTATION, which only represents group elements that map the code's lattice onto itself
        # — exact translation only axis-aligned, exact rotation only at 360/N (measured: quantised phases drift linearly;
        # continuous modular phases need N > 2π·radius modules). Continuous state ⇒ translation AND rotation exact at ANY
        # vector/angle, and the read-out's quantisation is bounded and NEVER accumulates (a fresh encode each fixation).
        # The ORIENTATION is likewise a rotation MATRIX, not a scalar angle (the 2026-07-15 cut-over — the same lesson: a
        # scalar is an SO(2)-only encoding, while Monty's pose is "three orthonormal vectors" and Gao 2021's operator is a
        # group-representation matrix). So n=2 and n=3 are ONE code path, and degrees are a 2-D read-out (`operator.to_angle`).
        # `MotionOperator` LEARNS what each action does (body-frame Δ + ΔR); rotation-as-hypothesis is geometry the solve applies.
        self.location = location                                # the READ-OUT encoder (what L4 binds to), not the state
        self.operator = MotionOperator() if location is not None else None
        self._pose = None                                       # (position, R) — continuous; None until located
        # L5 — OBJECT DYNAMICS (ARCHITECTURE §9). The SAME TRANSFORM primitive, applied to an OBJECT's pose instead of the
        # sensor's: L6a's operator path-integrates where the SENSOR is; this one learns how an ACTION moves a THING out
        # there. It is L5's because L5 owns displacement/behaviour ("grid: location + movement → location; displacement:
        # location + location → the relation" — `reference_tbt_layers_4_23`), and it is EXTRINSIC because an object's motion
        # does not care which way the object faces (a shoved block goes where it was shoved; a rock falls down).
        self.dynamics = MotionOperator(ego=False) if location is not None else None
        # L5PT — DISPLACEMENT cells (ARCHITECTURE §9): the RELATION between two object frames (`location + location → the
        # relation`, the INVERSE of L6a's `location + movement → location`; `reference_tbt_layers_4_23`). A relation is a
        # relative pose that stays STABLE as the pair moves — "resting on" / "part of" / "attached". NB L5 = IT + PT (§15 D1):
        # this is the PT thick-tufted displacement role; the L5IT associative integrator that gates into it is deferred.
        self._relations: dict = {}                              # (id_a, id_b) -> [relative pose, count, still-consistent]
        # COMPOSITIONAL column (ARCHITECTURE §9): the SCENE — recognised objects (object-id → pose) treated as this column's
        # own features-at-locations ("object id as a FEATURE → compositional objects", `reference_tbt_layers_4_23`), the same
        # Column one level up. Fed from the sensory column via the thalamus. Only the scene column uses it.
        self._scene_objects: dict = {}                          # object_id -> pose
        # L2/3's POOLING engine (ARCHITECTURE §8): the stable object-IDENTITY that pools the L4 feature-at-location stream
        # (`pooler.ColumnPooler`) — a decoupled stable output + persistence, which the L2/3 HTMLayer (associate) cannot do.
        self.pooler = ColumnPooler(seed=seed + 4) if location is not None else None
        # The OBJECT-CENTRIC frame anchor: a canonical origin the frame is RE-ORIGINED to at each object onset, so a location
        # is measured RELATIVE to the object (grid frames have no origin — Lewis 2019; an arbitrary origin gives translation
        # invariance). One shared origin suffices because the L2/3 pooler individuates objects, not the phase.
        self._anchor = tuple(0.0 for _ in range(location.dims)) if location is not None else None
        # `_sweep` = the episode BUFFER of sensed fixations (Monty's Buffer) that `recognize` REPLAYS per hypothesis;
        # `_prev_sweep` = the one before it, kept ONLY so `commit` can see what MOVED (common fate — the sole cue that can
        # segment a scene no model explains yet); `_pop` = the LIVE (object, pose) population `perceive` narrows per fixation.
        self._sweep: list = []
        self._prev_sweep: list = []
        self._pop: list = []
        # The **L4 → L6a associative link** (Lewis 2019): a sensed feature activates the UNION OF LOCATIONS where it occurs,
        # which is what seeds recognition when we do NOT know where on the object we are. Learned in `perceive`. The object
        # model is DISTRIBUTED, never one layer's data structure (`reference_tbt_layers_4_23`): the FEATURE is the key (L4),
        # the LOCATION is in L6a's object-centric frame, the IDENTITY is L2/3's — this dict is only the L4→L6a *link* between
        # them. It is NOT the retired rotation table: that memorised the ANSWER per orientation; this stores the object ONCE
        # in its own frame and the pose is SOLVED from it (`notes/rotation_invariance_plan.md` R4).
        self._link: dict = {}                                   # feature key -> [(identity, location in the object frame)]

    # ── L6a path integration (the TRANSFORM primitive; ARCHITECTURE §8) — a SPATIAL column only ─────────────────────
    def _require_location(self) -> None:
        if self.operator is None:
            raise ValueError("this Column has no location frame — pass location=GridEncoder(...) to enable L6a path integration.")

    def _code(self) -> SDR:
        """The grid-code READ-OUT of the continuous state — what L4 binds to. A FRESH encode each fixation, so its
        quantisation is bounded and never accumulates (unlike repeatedly applying a rounded shift)."""
        return self.location.encode(self._pose[0])

    def locate(self, coord) -> None:
        """Fix the location by sensory anchor (orientation = identity). The state is the CONTINUOUS coordinate; the code is
        derived."""
        self._require_location()
        self._pose = (tuple(float(c) for c in coord), eye(self.location.dims))

    def set_pose(self, coord, rotation) -> None:
        """Fix the full pose: location + ORIENTATION as an n×n rotation matrix (`operator.from_angle(deg)` builds one in 2-D).
        Continuous — no ring, no discretisation, and no scalar-angle special case."""
        self._require_location()
        self._pose = (tuple(float(c) for c in coord), tuple(tuple(float(x) for x in row) for row in rotation))

    def learn_move(self, action, before_coord, after_coord) -> None:
        """Learn `action`'s effect from one observed coordinate move (the orientation-free case: pure translation)."""
        self._require_location()
        n = self.location.dims
        self.operator.learn(action, (tuple(map(float, before_coord)), eye(n)), (tuple(map(float, after_coord)), eye(n)))

    def learn_pose_move(self, action, before_pose, after_pose) -> None:
        """Learn `action`'s effect from one observed pose move `(position, R)`. The operator stores the BODY-frame displacement
        + body-frame rotation, so one observation generalises to every position AND every orientation."""
        self._require_location()
        self.operator.learn(action, before_pose, after_pose)

    def path_integrate(self, action):
        """Dead-reckon: apply the learned action to the continuous pose — exact, nothing rounds, so nothing drifts. The
        body-frame displacement is mapped through the CURRENT orientation, so orientation-dependent motion is non-commutative
        (FORWARD;TURN ≠ TURN;FORWARD — and in 3-D the rotations themselves stop commuting) with no keying and no ring. An
        unlearned action is the identity. Returns the new pose."""
        self._require_location()
        if self._pose is None:
            raise ValueError("path_integrate before locate()/set_pose(): L6a has no state yet.")
        self._pose = self.operator.apply(self._pose, action)
        return self._pose

    def where(self):
        """The current location — the state itself; no decode needed. None before locate()."""
        self._require_location()
        return None if self._pose is None else self._pose[0]

    def pose(self):
        """The current pose `(position, R)`; None before locate()/set_pose()."""
        self._require_location()
        return self._pose

    # ── the L4↔L6a loop (ARCHITECTURE §8): predict the FEATURE at the path-integrated LOCATION (order-invariant) ──────
    def sense_at(self, feature: SDR, learn: bool = True) -> None:
        """L4 FEATURE-AT-LOCATION binding — the real §12 wiring: drive L4 with the FEATURE (its SDR bits ARE the active
        columns; a pre-transduced feature needs no SP), basal context = L6a's CURRENT location code. Binds the feature to
        WHERE it is sensed, so prediction later comes from the LOCATION, not the previous feature (order-invariant)."""
        self._require_location()
        if self._pose is None:
            raise ValueError("sense_at before locate()/path_integrate(): L6a has no location yet.")
        code = self._code().active           # the grid READ-OUT of the continuous state
        l4 = self.layers["L4"].htm
        l4.depolarize(code)                  # the LOCATION predicts the feature-at-location FIRST (so firing is
        l4.observe(feature.active, context=code, learn=learn)               # location-specific, not recurrent-sequence)

    def predict_feature(self) -> set:
        """Predict the FEATURE columns at L6a's current location BEFORE sensing — the feature-at-location read
        (`HTMLayer.predict_at`). Empty at an unbound location. Decode 'which feature' by overlap against the known feature
        codes (the peripheral's job — the encoder's inverse)."""
        self._require_location()
        if self._pose is None:
            raise ValueError("predict_feature before locate()/path_integrate(): L6a has no location yet.")
        return self.layers["L4"].htm.predict_at(self._code().active)

    # ── L2/3 (ARCHITECTURE §8): the STABLE object IDENTITY ───────────────────────────────────────────────────────────
    def label_of(self, identity) -> int:
        """The stable integer label of an identity SDR (−1 if unknown/empty) — L2/3's identity decoded for a caller. Takes
        the identity rather than reading L2/3's settled state, because `perceive`/`commit` already RETURN what they concluded
        — asking the layer again afterwards was a second route to the same answer."""
        return self.pooler.objects.index(identity) if identity in self.pooler.objects else -1

    # ── OBJECT-CENTRIC frame + the emergent boundary (ARCHITECTURE §8): one event anchors the frame AND the identity ───
    def look_again(self) -> None:
        """Declare that the NEXT sweep is THE SAME SCENE, LATER — so `commit` may group it by COMMON FATE (what moved
        together is one thing). This is explicit, and that is the point: "the previous EPISODE" is NOT "the same scene, a
        moment later" — objects are routinely studied one after another, and treating the last object's sweep as this
        scene's past invents motion between two unrelated things. Measured: rolling the buffer automatically at every onset
        split a chiral pair into fragments, because the two forms were learned back-to-back and differ in one cell. Only the
        caller knows the scene is the same one; the column will not assume it."""
        self._require_location()
        self._prev_sweep = self._sweep

    def start_object(self) -> None:
        """An object ONSET — the ONE coupled event (Lewis 2019: a fresh grid phase is BOTH the frame origin AND the object's
        unique location space). Re-ANCHOR the L6a frame to the canonical origin (so subsequent path integration is
        object-relative → translation-invariant) AND reset L2/3 to start a fresh identity. Called explicitly at a
        learning-time boundary (the honest minimal episode cue), and fired EMERGENTLY by `perceive` on recognition failure."""
        self._require_location()
        self.locate(self._anchor)                       # a place to START — NOT the object's origin: that is SOLVED
        self.pooler.reset()                             # (`perceive`) or minted from the sweep's first fixation (`commit`)
        self._sweep = []
        self._pop = []

    def perceive(self, feature: SDR):
        """INFER, online: sense a feature wherever the sensor actually is, and SOLVE which object this is — and where on it
        we are. Returns the L2/3 identity SDR (empty = nothing recognised, or genuinely ambiguous).

        THE POPULATION IS THE STATE (Monty's evidence-based LM). Each fixation NARROWS a live set of (object, pose)
        hypotheses; nothing is recomputed from scratch. Seeding needs `dims` fixations (n−1 displacements determine a
        rotation), so before that the only evidence is the L4→L6a UNION — "the sensory input activates the union of
        locations" (Lewis 2019): sensing a feature that only ONE object carries identifies it outright, and a feature two
        objects share leaves it honestly ambiguous. That is inference, not an assumption.

        THIS REPLACED AN ASSUMPTION. `perceive` used to bind and recall at whatever coordinate the caller supplied, so it
        only ever recognised an object entered AT ITS LEARNED ORIGIN — measured: the same object shifted to (7,3) read
        `[-1,-1,-1]`, entered mid-object `[-1,-1]`, while the buffered path solved the identical presentation. Two things
        followed: the caller's coordinate frame was a SILENT contract (nothing checked it), and there was no online
        `(object, pose)` stream for dynamics to track. Solving both is one change — this one (ARCHITECTURE §8/§9).

        THE BOUNDARY IS STILL EMERGENT, and now needs no re-anchoring: when every hypothesis is refuted we have LEFT the
        object, so the episode restarts HERE and the next object's pose is SOLVED rather than assumed."""
        self._require_location()
        if self._pose is None:
            raise ValueError("perceive before locate()/set_pose(): L6a has no location yet.")
        pos = self._pose[0]
        self._sweep.append((pos, feature))
        if self._pop:
            self._pop = self._narrow(pos, feature)       # incremental: every live hypothesis meets the new fixation
        if not self._pop:
            self._pop = [h for h in self.recognize() if h.refuted_at is None]      # (re)seed by SOLVING from the buffer
        if not self._pop and len(self._sweep) > 1:
            self._sweep = [(pos, feature)]               # every hypothesis died ⇒ a BOUNDARY: the episode starts HERE
        candidates = {h.identity for h in self._pop} or self._union_identities(feature)
        return self.pooler.settle(next(iter(candidates)) if len(candidates) == 1 else frozenset())

    def _narrow(self, pos, feature: SDR) -> list:
        """One fixation, every live hypothesis: does it still explain me? Refuted hypotheses are DROPPED (a contradiction is
        decisive — `commit`), the rest accumulate evidence. Several survivors is not a failure: agreeing on the OBJECT while
        differing on the pose is a symmetry orbit, which `perceive` reports as the object; disagreeing on the object is
        genuine ambiguity, which it reports as nothing rather than forcing a choice."""
        out = []
        for h in self._pop:
            delta, refuted, _predicted = self._fit(h.identity, h.rotation, h.origin, pos, feature)
            if not refuted:
                out.append(replace(h, evidence=h.evidence + delta))
        return out

    def _union_identities(self, feature: SDR) -> set:
        """The objects a sensed feature could belong to — the L4→L6a union read as identities. Before `dims` fixations no
        rotation is determined, so this is ALL the evidence there is; it is still real inference (a feature unique to one
        object names it), and it is what Monty gets from ONE sensation only because its features carry a local frame."""
        return {ident for ident, _loc in self._union_for(feature)}

    def _link_feature(self, feature: SDR, identity: frozenset) -> None:
        """Learn the L4→L6a link for one fixation: remember that this FEATURE occurs at this OBJECT-FRAME LOCATION on this
        OBJECT. Re-sensing the same feature-at-location on the same object is the SAME fact, not a new one (idempotent, so
        repeated learning passes do not inflate the union)."""
        entry = (identity, self._pose[0])
        union = self._link.setdefault(self._key(feature), [])
        if entry not in union:
            union.append(entry)

    @staticmethod
    def _key(feature: SDR):
        """The associative key for a feature. EXACT-match on the feature's active bits — honest scope (RULES): a NOISY or
        partial feature would need overlap recall (the union of everything sufficiently similar). Our features come from a
        clean categorical transducer, so exact match is the whole story here; overlap recall is the refinement."""
        return frozenset(feature.active)

    def _union_for(self, feature: SDR) -> list:
        """The L4→L6a recall: the UNION of (identity, object-frame location) where this feature has been sensed — across
        every object (Lewis 2019: "the sensory input activates the union of locations"). This is the hypothesis SEED; it is
        the model talking, not a lookup of the answer."""
        return self._link.get(self._key(feature), [])

    # ── RECOGNITION by EVIDENCE over a hypothesis POPULATION; the pose is SOLVED (plan R4; Monty) ───────────────────
    def sense_sweep(self, feature: SDR) -> None:
        """Sense a feature at the current location and BUFFER the fixation (Monty's Buffer) for `recognize`. Used when the
        object's pose is UNKNOWN: online pooling cannot recognise it until the pose is undone, so we record the sweep — the
        CONTINUOUS sensed position and the feature — and let `recognize` solve the pose. (`perceive` is the online path, for
        an object already in its canonical pose.)"""
        self._require_location()
        if self._pose is None:
            raise ValueError("sense_sweep before locate(): L6a has no location yet.")
        self._sweep.append((self._pose[0], feature))

    def recognize(self) -> list:
        """Recognise the buffered sweep's object AND its pose, at ANY rotation, ENTERED ANYWHERE. Returns the surviving
        hypothesis POPULATION (the best-evidence `Hypothesis` objects, tied); the pooler is left settled on the winner.

        THE MECHANISM (`reference_tbt_pose_invariant_recognition`; `notes/rotation_invariance_plan.md` R4/R6) — the pose is
        **SOLVED, never scanned**: "you recognize an unseen orientation because you SOLVE for the rotation; you don't recall
        it." Monty solves it by ALIGNING FRAMES, because its features carry a local frame (surface normal + curvature). Ours
        are colour-at-location — NON-morphological, no intrinsic orientation — so we build the frame from the inter-fixation
        DISPLACEMENT geometry instead:
          • an object at pose (R, t) puts every model point ℓ at  p = rotate(R, ℓ) + t;
          • differences CANCEL t:  pᵢ − p₀ = rotate(R, ℓᵢ − ℓ₀);
          • **n−1 independent displacements determine R** (`operator.solve_rotation`, the TRIAD method: orthonormalize each
            side into a right-handed frame, R = Uᵀ·V), then  t = p₀ − rotate(R, ℓ₀).  Closed-form, exact, ANY orientation.
        So 2-D needs 2 fixations and 3-D needs 3 — `dims` of them, picked by `_pin_rotation` — and there is NO angular
        resolution to sample, which is why SO(3) costs no more than SO(2) here. What is HYPOTHESISED is the correspondence
        (which model point each fixation touched), seeded by the L4→L6a union (`_union_for`); the pose is DERIVED from it and
        then VERIFIED by the model's own prediction (`_replay`).

        The prunes are the GROUP structure we already committed to (ARCHITECTURE §8), not domain priors: a rotation is an
        ISOMETRY, so EVERY pairwise distance must be preserved; degenerate fixations leave R undetermined; and R is always a
        PROPER rotation, so a MIRRORED object is refuted by the evidence rather than special-cased (reflections ∉ SO(n)).

        THE POPULATION IS THE ANSWER (`reference_population_code_belief`): several tied hypotheses mean the evidence does not
        separate them — a 4-fold object returns its whole symmetry ORBIT, which is correct because it genuinely has no single
        pose. Returning one there would be a lie. An empty list = nothing recognised."""
        self._require_location()
        picks = self._pin_rotation()
        if picks is None:
            return []                                    # too few / degenerate fixations — they fix no rotation
        sensed = [self._sweep[i][0] for i in picks]
        unions = [self._union_for(self._sweep[i][1]) for i in picks]
        seen, scored = set(), []
        for combo in itertools.product(*unions):
            identity = combo[0][0]
            if any(entry[0] != identity for entry in combo):
                continue                                 # a correspondence must lie within ONE object
            solved = self._solve(sensed, [entry[1] for entry in combo])
            if solved is None:
                continue                                 # not a rigid motion (or R undetermined) — pruned by geometry
            R, origin = solved
            key = (identity, tuple(round(x, 6) for row in R for x in row), tuple(round(c, 6) for c in origin))
            if key in seen:
                continue                                 # different correspondences, same pose = the same hypothesis
            seen.add(key)
            evidence, _, refuted_at, matched = self._replay(identity, R, origin)
            scored.append(Hypothesis(identity, self.label_of(identity), R, origin, evidence, refuted_at,
                                     self._choice(matched, identity)))
        if not scored:
            return []
        # RANK BY ART's CHOICE (the size principle), evidence only to break exact ties. Ranking by evidence alone was the
        # bug: a SUM lets a big model explaining part of itself tie a small model explaining all of itself.
        top = max((h.choice, h.evidence) for h in scored)
        best = [h for h in scored if (h.choice, h.evidence) >= top]
        self.pooler.settle(best[0].identity)             # L2/3 holds what the population concluded
        return best

    def _choice(self, matched: int, identity: frozenset) -> float:
        """ART's CHOICE function `T_j = |I ∧ w_j| / (α + |w_j|)`, at the FEATURE-AT-LOCATION granularity: `matched` fixations
        over model size (`α + |extent|`). Normalised by the CATEGORY, so in the conservative limit (small α) the SMALLEST
        model that explains the sweep wins — Grossberg's mechanism and Tenenbaum's SIZE PRINCIPLE in one expression (seeing
        exactly a small model's cells and none of a bigger one's others is a *suspicious coincidence*). This is what lets a
        torn-off PIECE beat its parent BLOB (matched 2 / extent 2 = 0.99 vs 2 / 4 = 0.50) and is why the blob need never be
        un-bound — it simply stops being chosen (ARCHITECTURE §9)."""
        return matched / (self.pooler.alpha + max(1, len(self._extent(identity))))

    def _pin_rotation(self):
        """Pick the `dims` buffered fixations that PIN A ROTATION DOWN: the first, plus each next one whose displacement from
        it adds an INDEPENDENT direction (n−1 independent displacements determine R in n-space). Returns their indices, or
        None if the sweep never spans enough directions. None is the honest answer, not a failure: one fixation fixes no
        rotation at all, and a wholly collinear sweep in 3-D leaves a genuine 1-parameter family of poses that a discrete
        population cannot express — inventing one would be a lie. (Choosing WHERE to look next to break that is active
        sensing — L5's job, deferred.)"""
        n = self.location.dims
        if len(self._sweep) < n:
            return None
        picks, dirs, p0 = [0], [], self._sweep[0][0]
        for i in range(1, len(self._sweep)):
            candidate = dirs + [sub(self._sweep[i][0], p0)]
            if gram_schmidt(candidate) is None:
                continue                                 # dependent on what we already have — adds no constraint
            picks.append(i)
            dirs = candidate
            if len(picks) == n:
                return picks
        return None

    def _solve(self, sensed, model):
        """SOLVE the pose `(R, origin)` from one hypothesised correspondence: fixations `sensed` touched model points `model`.
        None if the correspondence is not a rigid motion or leaves R undetermined."""
        for i in range(len(sensed)):
            for j in range(i + 1, len(sensed)):
                if abs(dist(sensed[i], sensed[j]) - dist(model[i], model[j])) > _TOL:
                    return None                          # a rotation is an ISOMETRY — this correspondence cannot be rigid
        R = solve_rotation([sub(p, sensed[0]) for p in sensed[1:]], [sub(m, model[0]) for m in model[1:]])
        if R is None:
            return None                                  # degenerate on the model side — no orienting cue
        return R, sub(sensed[0], rotate(R, model[0]))

    def _replay(self, identity, rotation: tuple, origin: tuple, learn: bool = False):
        """Replay the buffered sweep under ONE (object, pose) hypothesis: map each sensed position back into the object's own
        frame and sense it there. Returns `(evidence, predicted, refuted_at, matched)` — the evidence for `identity`, how many
        fixations L4 PREDICTED (did not burst), the INDEX of the FIRST fixation that refuted the hypothesis (None if none did),
        and how many fixations MATCHED it (predicted AND supported by this identity).

        SCORING (`learn=False`) asks THE MODEL whether it expected each fixation. Monty's update — a match ADDS [0,+1], a
        mismatch SUBTRACTS [−1] — with two distinct mismatches, both object-SPECIFIC:
          • L4 BURSTS ⇒ nothing known is here at all, so this object is not here either;
          • L4 predicts but L2/3's support for THIS identity is ~0 ⇒ something is here and it is SOMEONE ELSE's.
            (Support is the graded weight, so a hypothesis is never rewarded for a location another object explains.)
        `refuted_at` is ART's VIGILANCE at ρ=1 (any refutation ⇒ not this object) and, when the prefix exhausted the model,
        the object BOUNDARY. `matched` is what ranks rivals via ART's CHOICE (`recognize`), NOT raw `evidence`: choice counts
        at the FEATURE-AT-LOCATION granularity (matched fixations over model size), which is burst-independent — the reason
        an earlier L4-CELL choice tied a piece with its parent blob (minting binds burst cells, ~M per column; recognition
        fires ~1, so a piece overlapped only 1/M of its own inflated receptive field).

        BINDING (`learn=True`) walks the SAME traversal and commits it: L4 learns the feature-at-location, and each PREDICTED
        code is bound to `identity` in L2/3 + the L4→L6a link. Learning and scoring being one traversal is what makes an
        object get learned in exactly the frame it was recognised in — so re-sweeping a known object at a NOVEL pose
        reinforces it rather than corrupting it with rotated coordinates. `identity=None` trains L4 only (the probe below)."""
        evidence, predicted, refuted_at = 0.0, 0, None
        matched = 0
        for i, (pos, feature) in enumerate(self._sweep):
            delta, refuted, was_predicted = self._fit(identity, rotation, origin, pos, feature, learn=learn)
            evidence += delta
            predicted += 1 if was_predicted else 0
            matched += 1 if (was_predicted and not refuted and identity is not None) else 0
            if refuted and refuted_at is None:
                refuted_at = i
        return evidence, predicted, refuted_at, matched

    def _fit(self, identity, rotation: tuple, origin: tuple, pos, feature: SDR, learn: bool = False):
        """THE scoring atom — ONE fixation against ONE (object, pose) hypothesis. Returns `(evidence, refuted, predicted)`.
        Both callers use it, so there is ONE rule and not two: `_replay` loops it over a whole buffer (batch), `_narrow`
        calls it once per live hypothesis as each new fixation arrives (online, incremental — "recognition is INCREMENTAL
        evidence accumulation, NEVER recomputed", `reference_tbt_layers_4_23`).

        IMAGINATION, NOT MOVEMENT: this senses the fixation where the HYPOTHESIS says it falls on the object, so it must not
        disturb where the sensor actually is — hence the save/restore. That the sensor's pose was ever scratch space for a
        replay is exactly why the online path could not solve while the buffered one could."""
        real = self._pose
        try:
            self._pose = (rotate(invert(rotation), sub(pos, origin)), eye(self.location.dims))   # in the object's OWN frame
            self.sense_at(feature, learn=learn)
            l4 = self.layers["L4"].htm
            if l4.bursting():
                return -1.0, True, False                 # nothing known is here ⇒ this object is not here either
            if identity is None:
                return 0.0, False, True                  # the L4-training probe (`commit`): no identity to score against
            support = self.pooler.support(l4._active, identity)
            if learn:                                    # BIND before scoring: a freshly minted identity supports nothing
                self.pooler.bind(l4._active, identity)   # yet, so scoring first would refuse to ever bind it. Only
                self._link_feature(feature, identity)    # PREDICTED codes reach here (a burst is location-agnostic).
            if support < self.pooler.recognize_frac:
                return -1.0, True, True                  # ~0 support ⇒ this is someone ELSE's feature here
            return support, False, True
        finally:
            self._pose = real

    # ── L5 OBJECT DYNAMICS (ARCHITECTURE §9): what an ACTION does to a THING, STATE-CONDITIONED ─────────────────────
    def learn_object_move(self, action, before_pose, after_pose, state=frozenset()) -> None:
        """Learn what `action` does to an OBJECT, from ONE observed `(pose → pose)` transition, CONDITIONED on the object's
        STATE. The poses are what `recognize`/`perceive` SOLVE — the model's own output, not a hand-fed coordinate.

        WHY ONE DEMONSTRATION IS ENOUGH FOR EVERY OBJECT, EVERYWHERE — the §7 lesson, not statistics. The delta is stored in
        the frame that holds it INVARIANT, so it applies at every position and orientation BY CONSTRUCTION; the FRAME
        generalizes, not the data. Nothing is keyed on WHICH object.

        STATE-CONDITIONING (`reference_tbt_object_behaviors`): the effect is keyed on `(action, STATE)`, so the SAME action has
        DIFFERENT effects in different states — STEP drops a FREE object but not a SUPPORTED one. `state=∅` is the free kernel,
        the null-state default (the aggressive "everything falls" prior, `feedback_prefer_generalize_then_correct`). The state
        is GEOMETRY-keyed (`state_of`), so the effect learned in a state transfers to any object in that state — TBP's
        object-independent behaviour frame. "Which state feature is the TRUE condition vs a spurious correlate" is the KEY
        problem, still open."""
        self._require_location()
        self.dynamics.learn((action, state), before_pose, after_pose)

    def predict_object_move(self, pose, action, state=frozenset()):
        """Predict where `action` puts an object at `pose` in `state` — the FORWARD MODEL over objects. Uses the state-specific
        effect where one is learned, else FALLS BACK to the free kernel (`state=∅`): "a specific context overrides the
        default", the HTM high-order-over-first-order structure — NOT a hand-coded `if supported`."""
        self._require_location()
        key = (action, state)
        return self.dynamics.apply(pose, key if self.dynamics.known(key) else (action, frozenset()))

    # ── COMPOSITIONAL column (ARCHITECTURE §9): the SCENE, object STATES, and STATE-CONDITIONED behaviour ──────────────
    def place_object(self, object_id, pose) -> None:
        """Put a recognised object into this column's SCENE at a pose — the compositional column's features-at-locations are
        whole objects (fed from the sensory column via the thalamus)."""
        self._require_location()
        self._scene_objects[object_id] = pose

    def clear_scene(self) -> None:
        """Start a fresh scene configuration."""
        self._scene_objects = {}

    def state_of(self, object_id) -> frozenset:
        """The object's relational STATE: the set of (quantised) relative poses to the OTHER objects in the scene
        (`reference_tbt_object_behaviors`: behaviour is STATE-CONDITIONED). GEOMETRY-keyed, not identity-keyed, so the SAME
        state arises whichever objects realise it — which is why a behaviour learned in one state transfers to any object in
        that state. The EMPTY state (an object alone, or with only never-learned relations → the free-kernel fallback) is the
        null default."""
        self._require_location()
        me = self._scene_objects[object_id]
        return frozenset(self._quantise(self.relate(me, other))
                         for oid, other in self._scene_objects.items() if oid != object_id)

    @staticmethod
    def _quantise(rel) -> tuple:
        """Bucket a relative pose so the SAME geometric relation is ONE key (position to the nearest unit, rotation to 3 dp).
        Honest scope: exact-match on the bucket — two support configs generalise only if they share the bucket, so it
        transfers across objects of the SAME relative geometry, not yet across geometry variants (a bigger table at a
        different offset). Overlap recall over relation SDRs is the refinement — the same note as `_key`, and part of the KEY
        problem."""
        dp, dr = rel
        return (tuple(round(c) for c in dp), tuple(round(x, 3) for row in dr for x in row))

    def learn_behavior(self, action, object_id, after_pose) -> None:
        """Observe `object_id` (at its current SCENE pose) move to `after_pose` under `action`, and learn the effect
        CONDITIONED on its current relational STATE; then update the scene. This is where "supported objects don't fall" is
        LEARNED, not coded: whatever state happens to hold (support) gets its own keyed effect (stay), the null state keeps
        the free kernel (fall). No `if supported` anywhere — the state-keying does it."""
        self._require_location()
        before = self._scene_objects[object_id]
        self.learn_object_move(action, before, after_pose, self.state_of(object_id))
        self._scene_objects[object_id] = after_pose

    def predict_behavior(self, action, object_id):
        """Predict `object_id`'s next pose under `action`, gated by its current relational state — the free kernel by default,
        the state-specific effect where the scene puts it in a learned state."""
        self._require_location()
        return self.predict_object_move(self._scene_objects[object_id], action, self.state_of(object_id))

    # ── L5PT DISPLACEMENT cells (ARCHITECTURE §9): the RELATION between two objects ─────────────────────────────────
    def relate(self, pose_a, pose_b):
        """The DISPLACEMENT cell's output: the pose of object B expressed in object A's frame — `location + location → the
        relation` (the inverse of L6a's operator; `reference_tbt_layers_4_23`). Position- AND orientation-invariant BY
        CONSTRUCTION: if the whole pair translates or turns RIGIDLY, this is unchanged, because it is B *relative to* A. That
        invariance is the whole point — a relation does not depend on WHERE the pair sits, only on how the two are arranged."""
        (pa, ra), (pb, rb) = pose_a, pose_b
        return rotate(invert(ra), sub(pb, pa)), compose(invert(ra), rb)

    def observe_relation(self, id_a, pose_a, id_b, pose_b):
        """Observe two recognised objects together and update their RELATION — a relation is a displacement that PERSISTS as
        the pair moves ([[feedback_prefer_generalize_then_correct]]: ASSUMED fixed from the first view, then DISSOLVED the
        moment the two move independently). 'Resting on', 'part of', 'attached' are exactly this; a compositional object is
        sub-objects at fixed relative displacements (Hawkins 2019). Returns the current relative pose."""
        self._require_location()
        rel = self.relate(pose_a, pose_b)
        rec = self._relations.get((id_a, id_b))
        if rec is None:
            self._relations[(id_a, id_b)] = [rel, 1, True]      # first view — assume it is a fixed relation
        elif rec[2] and self._pose_close(rec[0], rel):
            rec[1] += 1                                          # still consistent — the assumption holds, reinforce it
        else:
            rec[2] = False                                      # they moved apart ⇒ NOT a fixed relation (corrected)
        return rel

    def relation_of(self, id_a, id_b, min_count: int = 2):
        """The STABLE relative pose of the pair, or None if they have no fixed relation (never confirmed, or broken by
        independent motion). `min_count` = how many consistent views make it a relation rather than a coincidence."""
        rec = self._relations.get((id_a, id_b))
        return rec[0] if rec and rec[2] and rec[1] >= min_count else None

    @staticmethod
    def _pose_close(a, b, tol: float = 1e-6) -> bool:
        """Two relative poses equal within tolerance (position + rotation)."""
        (pa, ra), (pb, rb) = a, b
        return (all(abs(x - y) < tol for x, y in zip(pa, pb))
                and all(abs(x - y) < tol for rowa, rowb in zip(ra, rb) for x, y in zip(rowa, rowb)))

    # ── LEARNING: the end-of-episode commitment (plan R5) ───────────────────────────────────────────────────────────
    def commit(self) -> frozenset:
        """END OF A LEARNING EPISODE: commit the buffered sweep to L2/3 — RECOGNISE first (does a known object, at ANY pose,
        explain this sweep?), then BIND: reinforce that object, or mint a new identity and bind the sweep to it. This is
        Monty's end-of-episode learning step (Buffer → update an existing graph, or build a new one). Returns the committed
        identity, or empty if the commitment was deferred (below).

        WHY THE EPISODE, NOT THE FIXATION (the R5 fix). L2/3 used to commit at the FIRST fixation and persist unconditionally,
        which MERGED two objects sharing their first feature-at-location into one chimeric identity (measured). At fixation 1
        those objects are genuinely indistinguishable, so recognising the first is correct INFERENCE — the bug was the lack of
        REVISION, and a per-fixation loop *cannot* revise: by the time a later fixation contradicts the choice, the earlier
        ones are already bound to the wrong identity. Refutation needs the object's EXTENT, which only the whole sweep gives.

        Because recognition here is POSE-INVARIANT, re-sweeping a known object at a NOVEL pose reinforces it instead of
        minting a duplicate — identity and pose stay factored (`recognize`), which is the point of R4.

        THE BAR IS "NOTHING REFUTES IT", NOT A SCORE. A known object explains this sweep iff NO fixation contradicts it: a
        PARTIAL view of it is all match and no contradiction (so it reinforces, never fragments), while one genuine
        contradiction means it is a different object, however much else agrees. That is threshold-free, and it is also the
        SAFE rule: `_replay(learn=True)` binds every predicted fixation, so tolerating a contradiction would BIND it —
        i.e. tolerance silently corrupts the model, which is exactly how the merge used to happen. (An earlier
        `evidence ≥ ½·fixations` bar merged a CHIRAL pair by a single hair: 3 matches − 1 refutation = 2 = the bar. Its
        tolerance was arbitrary — it depended on how many OTHER fixations agreed — and it was corrupting.) Tolerating k
        contradictions is a question for a sensor-NOISE model, deferred with noise itself; with an exact sensor a
        contradiction is decisive.

        THE SWEEP SPLITS ITSELF AT AN OBJECT BOUNDARY (R7). A contradiction has two readings — the sweep LEFT this object
        (a boundary), or the sweep is one DIFFERENT object that merely shares a prefix — and both fit the same evidence.
        `_exhausts` is the model answering: you leave an object when you reach its EDGE, so if the prefix visited EVERY
        location in the object's model it ended there and a new object begins; if the prefix covered only part of it, the
        sweep never reached an edge and "one object sharing a prefix" is the better parse. Then the remainder is simply a
        fresh episode — the same `commit`, recursively — so a continuous sweep over several objects needs NO boundary cue
        from the caller.

        COMMON FATE OVERRIDES RECOGNITION (R7's cold-start blob, closed). Checked FIRST, above recognition, because motion
        can refute a single-object hypothesis that recognition would otherwise accept: if the buffered scene split into >1
        MOTION group, its parts moved apart, so it is not one object — even if a known blob explains every fixation as a
        partial view (vigilance passes). Motion carries evidence a static match cannot (`reference_tbt_segmentation_and_grouping`;
        Xu & Carey: spatiotemporal individuation precedes featural). `_commit_split` is then the ART orienting RESET — recruit
        fresh categories for the parts rather than reinforce the over-spanning whole (Grossberg)."""
        self._require_location()
        if not self._sweep:
            return frozenset()
        groups = self._common_fate_groups()
        if len(groups) > 1:                              # motion split the scene ⇒ it was never one object (checked FIRST)
            return self._commit_split(groups)
        best = self.recognize()
        if best and best[0].refuted_at is None:
            h = best[0]                                  # a KNOWN object (possibly at a novel pose) — reinforce it
            self._replay(h.identity, h.rotation, h.origin, learn=True)
            return self.pooler.settle(h.identity)
        if best and self._exhausts(best[0]):
            h, cut = best[0], best[0].refuted_at         # a BOUNDARY: this object ENDED here, and the rest is another
            whole = self._sweep
            self._sweep = whole[:cut]
            self._replay(h.identity, h.rotation, h.origin, learn=True)    # reinforce the object we just left
            self._sweep = whole[cut:]
            self.pooler.reset()
            return self.commit()                         # the remainder is its own episode — recognise it, or mint it
        return self._mint_sweep()

    def _commit_split(self, groups: list) -> frozenset:
        """The buffered scene split into >1 MOTION group (common fate) — so it is >1 object, and the ART orienting RESET
        fires: NO single object may claim more than one group, because one object cannot be in two places at once. Commit
        each group as its own episode, admitting a known identity ONLY IF its whole model fits WITHIN the group's features —
        an object that reaches OUTSIDE the group would have to be partly elsewhere, which motion has just shown it is. When
        nothing fits, RECRUIT a fresh identity (mint) — never erode the over-spanning whole; it is simply not chosen, and
        dies of disuse (Grossberg; the un-binding answer, ARCHITECTURE §9).

        This is what CREATES the rival the size principle then needs. Before it, a piece of a learned blob was only ever a
        partial view (vigilance 1.0), so recognition reinforced the blob and the piece was never individuated (measured: six
        passes studying a part alone left the library at 1). The containment veto is the motion evidence recognition lacks;
        once a piece exists, ART CHOICE keeps it winning over the blob (a small model seen whole beats a big one seen half),
        so no duplicate is minted on later looks."""
        whole, last = self._sweep, frozenset()
        for g in groups:
            self._sweep = [whole[i] for i in g]
            here = {self._key(f) for _p, f in self._sweep}
            best = self.recognize()
            self.pooler.reset()
            if best and best[0].refuted_at is None and self._model_keys(best[0].identity) <= here:
                h = best[0]                              # a known object that fits WITHIN this group — reinforce it
                self._replay(h.identity, h.rotation, h.origin, learn=True)
                last = self.pooler.settle(h.identity)
            else:
                last = self._mint_sweep()                # over-spanning or unknown ⇒ recruit (ART reset)
        self._sweep = whole
        return last

    def _mint_sweep(self) -> frozenset:
        """Recruit a NEW identity for the buffered sweep. A NEW object DEFINES its own frame at the onset — the first
        fixation is its origin (Lewis 2019: a fresh grid phase IS the object's origin; `reference_tbt_object_frame_bootstrap`).
        "NEW" and "not yet LEARNED" differ, and L4 tells them apart: until L4 predicts a fixation its code is a
        location-agnostic BURST supporting every object carrying that feature ANYWHERE (the feature-only trap), so there is
        nothing to ground an identity on — sense the sweep (training L4) and mint only once L4 predicts some of it."""
        canonical, origin = eye(self.location.dims), self._sweep[0][0]
        _, predicted, _, _ = self._replay(None, canonical, origin, learn=True)
        if not predicted:
            self.pooler.reset()                          # nothing committed ⇒ L2/3 holds no object
            return frozenset()                           # deferred: another look will mint, once L4 has learned the features
        identity = self.pooler.mint()
        self._replay(identity, canonical, origin, learn=True)
        return self.pooler.settle(identity)

    def _model_keys(self, identity: frozenset) -> set:
        """The feature-KEYS this identity's model uses — enough to ask "does the whole model fit within a group's features?".
        The stored key suffices here because RECRUITING (not un-binding) never re-senses a model; the SDR would only be
        needed to replay a model's own features, which this path does not do."""
        return {k for k, entries in self._link.items() if any(i == identity for i, _loc in entries)}

    def _common_fate_groups(self) -> list:
        """Group the buffered fixations by HOW THEY MOVED since the previous look — the Gestalt cue, and the only one that can
        segment a scene NO MODEL EXPLAINS YET. Returns groups of sweep indices (one group = no split).

        WHY THIS IS THE MISSING CUE. Everywhere else the boundary is a PREDICTION MISMATCH against a model
        (`reference_tbt_segmentation_and_grouping`: "it relies on feature and morphology mismatch to implicitly detect
        boundaries"), which is why a wholly novel scene could only ever mint ONE blob — with no model there is no object.
        Motion needs no model: a feature seen at `p` last look and at `p'` now moved by `d = p' − p`, and fixations sharing
        `d` moved TOGETHER, so they are one thing. Same principle as everywhere else — the mismatch is just against the
        scene's own motion instead of against a stored model. It is also why "what IS an object" and "what does an object DO"
        are one question (ARCHITECTURE §9).

        WHAT IT REFUSES TO DO, and why the refusals are the load-bearing part. Motion tells you nothing unless you can say
        WHICH thing was where, and correspondence here is by EXACT feature match — so the cue is only trustworthy when every
        feature occurs ONCE in each look. A REPEATED feature (a 4-fold symmetric object senses the same feature at four
        places) would silently pair the wrong ones and invent four different displacements, shattering the object; a feature
        with no counterpart leaves nothing to compare. In both cases this reports ONE group rather than guess. Measured: without
        that guard it splits a symmetric object into its four cells and a chiral pair into fragments. The general fix is the
        same one `_key` needs — overlap recall over a POPULATION of correspondences, i.e. motion should narrow hypotheses
        rather than be read off a dict.

        SCOPE: this GROUPS a look; making the grouping PERSIST as objects is the other half and is not built — the moved part
        lands where L4 has never sensed it (so its mint defers), and the next static look groups the scene as one again. A
        blob already learned would need UN-BINDING to split. Noted, not hidden (ARCHITECTURE §9)."""
        n = len(self._sweep)
        if not self._prev_sweep or not n:
            return [list(range(n))]
        keys = [self._key(f) for _p, f in self._sweep]
        prev_keys = [self._key(f) for _p, f in self._prev_sweep]
        if len(set(keys)) != len(keys) or len(set(prev_keys)) != len(prev_keys):
            return [list(range(n))]                      # a REPEATED feature ⇒ correspondence is a guess ⇒ refuse
        was = dict(zip(prev_keys, (p for p, _f in self._prev_sweep)))
        groups: dict = {}
        for i, (pos, _feature) in enumerate(self._sweep):
            before = was.get(keys[i])
            if before is None:
                return [list(range(n))]                  # no counterpart ⇒ nothing to compare ⇒ do not guess
            groups.setdefault(tuple(round(c, 6) for c in sub(pos, before)), []).append(i)
        return list(groups.values())

    def _extent(self, identity: frozenset) -> set:
        """Every OBJECT-FRAME location this identity's model holds — the object's EXTENT, read off the L4→L6a link. Rounded,
        because a location recovered through a solved rotation carries float round-off the stored one does not."""
        return {tuple(round(c, 6) for c in loc)
                for union in self._link.values() for ident, loc in union if ident == identity}

    def _exhausts(self, h: Hypothesis) -> bool:
        """Did the sweep's PREFIX (up to `h.refuted_at`) cover the WHOLE of `h`'s model? That is what tells a boundary from a
        merely-shares-a-prefix object: you leave an object when you reach its EDGE. False if the object has no extent yet."""
        extent = self._extent(h.identity)
        if not extent:
            return False
        undo = invert(h.rotation)
        seen = {tuple(round(c, 6) for c in rotate(undo, sub(pos, h.origin)))
                for pos, _f in self._sweep[:h.refuted_at]}
        return extent <= seen

    # ── the FIRST-SLICE drive (§15 D3): sense a feature at L4, conditioned on a factored state ──────────────────────
    def observe(self, feature: SDR, context: Optional[SDR] = None, learn: bool = True) -> list:
        """Drive L4: sense a FEATURE (proximal), optionally conditioned on a factored CONTEXT/state (the task column's
        output — project_place_invariance_needs_factored_state). Return L4's ACTIVE CELLS (flat indices) for a downstream
        readout. SIMPLIFICATION (noted, §15): the context enters via the PROXIMAL path (concatenated → SP), not L4's basal
        `context=` — a single HTMLayer's active cells are content-determined, so proximal is what a cell-readout sees. The
        other layers (L2/3, L5, L6a) exist and are wired but are not driven by this slice (D3)."""
        l4 = self.layers["L4"]
        inp = feature if context is None else SDR.concat([feature, context])
        cols = l4.sp.encode(inp.dense(), learn=learn)
        l4.htm.observe(cols.active, learn=learn)
        return [c * l4.htm.M + cell for (c, cell) in l4.htm._active]

    def step(self, sensory: SDR, efference: Optional[SDR] = None, feedback: Optional[SDR] = None):
        """The FULL two-counterstream dataflow of §13 (all layers). TARGET, not yet built — the first slice drives L4 via
        `observe`; the deep layers (L5/L6) + the feedback stream come as tasks exercise them (D3). Raises so a half-built
        loop fails loudly (RULES #3)."""
        raise NotImplementedError(
            "column.step: the full §13 counterstream is the target; the wired first slice uses `Column.observe` (L4 only, "
            "§15 D3). Build the deep-layer + feedback dynamics against a task that exercises them.")
