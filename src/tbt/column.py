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
  (`operator.ModularOperator`), NOT the L6a HTMLayer's sequence memory — ARCHITECTURE §8 (a memorised per-position
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

from dataclasses import dataclass, field
from typing import Optional

from tbt.encoders import SDR, GridEncoder, SpatialPooler
from tbt.htm import HTMLayer
from tbt.operator import ModularOperator
from tbt.pooler import ColumnPooler

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
    (`operator.ModularOperator`) and the path-integration API (`locate`/`learn_move`/`path_integrate`/`where`, ARCHITECTURE §8)."""

    def __init__(self, sensory_n: int, n_cols: int = 1024, order: int = 2, seed: int = 0,
                 location: Optional[GridEncoder] = None, heading: Optional[GridEncoder] = None) -> None:
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
        # L6a's TRANSFORM engine (ARCHITECTURE §8): when a location frame is given (a SPATIAL column), path integration is
        # the learned OPERATOR over the grid code — the "grid path-integration operator" §12 names — NOT the L6a HTMLayer's
        # sequence memory (which would fail place-invariance). `self._loc` = L6a's dynamic location state (the bump we move).
        self.location = location
        self.operator = ModularOperator(location) if location is not None else None
        self._loc: Optional[SDR] = None
        # SE(2) NON-ABELIAN path integration (ARCHITECTURE §8): an optional HEADING ring + its own operator. The LOCATION
        # operator above is reused, keyed by (action, heading), so its shift DEPENDS on heading (the semidirect product
        # R²⋊SO(2)); `head_op` shifts the heading ring (abelian). No tensor code — the conditioning is the key.
        self.heading = heading
        self.head_op = ModularOperator(heading) if heading is not None else None
        self._head: Optional[SDR] = None
        # L2/3's POOLING engine (ARCHITECTURE §8): the stable object-IDENTITY that pools the L4 feature-at-location stream
        # (`pooler.ColumnPooler`) — a decoupled stable output + persistence, which the L2/3 HTMLayer (associate) cannot do.
        self.pooler = ColumnPooler(seed=seed + 4) if location is not None else None
        # The OBJECT-CENTRIC frame anchor: a canonical origin the L6a frame is RE-ORIGINED to at each object onset, so a
        # location is measured RELATIVE to the object (grid frames have no origin — Lewis 2019; the arbitrary origin gives
        # translation invariance). One shared origin suffices because the L2/3 pooler individuates objects, not the phase.
        self._anchor = tuple(0 for _ in range(location.dims)) if location is not None else None

    # ── L6a path integration (the TRANSFORM primitive; ARCHITECTURE §8) — a SPATIAL column only ─────────────────────
    def _require_location(self) -> None:
        if self.operator is None:
            raise ValueError("this Column has no location frame — pass location=GridEncoder(...) to enable L6a path integration.")

    def locate(self, coord) -> SDR:
        """Fix L6a's location state to a coordinate (a sensory anchor / reset of the path integrator)."""
        self._require_location()
        self._loc = self.location.encode(coord)
        return self._loc

    def learn_move(self, action, before_coord, after_coord) -> None:
        """L6a/L5 TRANSFORM learning (ARCHITECTURE §8): learn `action`'s effect on the location code from one observed
        (before → after) coordinate move (per-module phase-delta voting). Position-invariant by construction — the same
        shift is read at every position, so it generalises to positions never visited."""
        self._require_location()
        self.operator.learn(self.location.encode(before_coord), action, self.location.encode(after_coord))

    def path_integrate(self, action) -> SDR:
        """Dead-reckon: apply the learned operator to the current location state, no sensory input (an unlearned action is
        the identity). Returns the new location code. Composes over a sequence (abelian group)."""
        self._require_location()
        if self._loc is None:
            raise ValueError("path_integrate before locate(): L6a has no location state yet.")
        self._loc = self.operator.apply(self._loc, action)
        return self._loc

    def where(self):
        """Decode L6a's current location state to a coordinate (the peripheral read of the bump); None before locate()."""
        self._require_location()
        return None if self._loc is None else self.location.decode(self._loc)

    # ── SE(2) NON-ABELIAN path integration (ARCHITECTURE §8): heading-conditioned location shift + heading shift ──────
    def _require_heading(self) -> None:
        if self.head_op is None:
            raise ValueError("this Column has no heading frame — pass heading=GridEncoder(...) for SE(2) path integration.")

    def set_pose(self, coord, heading) -> None:
        """Fix the full SE(2) pose: location `coord` + `heading` (a sensory anchor / reset of the pose integrator)."""
        self._require_location()
        self._require_heading()
        self._loc = self.location.encode(coord)
        self._head = self.heading.encode(heading)

    def learn_pose_move(self, action, before_pose, after_pose) -> None:
        """SE(2) TRANSFORM learning (ARCHITECTURE §8): from an observed pose move `*_pose = ((x, y), heading)`, learn (i) the
        LOCATION shift keyed by (action, heading) — the heading-CONDITIONED, non-abelian part — and (ii) the HEADING shift
        keyed by action. Both are position-invariant per (action, heading), so they generalise across the whole space."""
        self._require_location()
        self._require_heading()
        (bloc, bh), (aloc, ah) = before_pose, after_pose
        self.operator.learn(self.location.encode(bloc), (action, bh), self.location.encode(aloc))
        self.head_op.learn(self.heading.encode(bh), action, self.heading.encode(ah))

    def path_integrate_pose(self, action):
        """Dead-reckon the SE(2) pose: decode the CURRENT heading, apply the heading-conditioned LOCATION operator, then the
        HEADING operator. Non-commutative (FORWARD;TURN ≠ TURN;FORWARD) — FORWARD's location shift is keyed on heading."""
        self._require_location()
        self._require_heading()
        if self._loc is None or self._head is None:
            raise ValueError("path_integrate_pose before set_pose(): the pose has no state yet.")
        h = self.heading.decode(self._head)[0]      # the heading ring is 1-D → the scalar heading (matches the learn key)
        self._loc = self.operator.apply(self._loc, (action, h))
        self._head = self.head_op.apply(self._head, action)
        return self._loc, self._head

    def pose(self):
        """Decode the current SE(2) pose to ((x, y), heading); None before set_pose()."""
        self._require_location()
        self._require_heading()
        if self._loc is None or self._head is None:
            return None
        return self.location.decode(self._loc), self.heading.decode(self._head)[0]

    # ── the L4↔L6a loop (ARCHITECTURE §8): predict the FEATURE at the path-integrated LOCATION (order-invariant) ──────
    def sense_at(self, feature: SDR, learn: bool = True) -> None:
        """L4 FEATURE-AT-LOCATION binding — the real §12 wiring: drive L4 with the FEATURE (its SDR bits ARE the active
        columns; a pre-transduced feature needs no SP), basal context = L6a's CURRENT location code. Binds the feature to
        WHERE it is sensed, so prediction later comes from the LOCATION, not the previous feature (order-invariant)."""
        self._require_location()
        if self._loc is None:
            raise ValueError("sense_at before locate()/path_integrate(): L6a has no location yet.")
        l4 = self.layers["L4"].htm
        l4.depolarize(self._loc.active)      # the LOCATION predicts the feature-at-location FIRST (so firing is
        l4.observe(feature.active, context=self._loc.active, learn=learn)   # location-specific, not recurrent-sequence)

    def predict_feature(self) -> set:
        """Predict the FEATURE columns at L6a's current location BEFORE sensing — the feature-at-location read
        (`HTMLayer.predict_at`). Empty at an unbound location. Decode 'which feature' by overlap against the known feature
        codes (the peripheral's job — the encoder's inverse)."""
        self._require_location()
        if self._loc is None:
            raise ValueError("predict_feature before locate()/path_integrate(): L6a has no location yet.")
        return self.layers["L4"].htm.predict_at(self._loc.active)

    # ── L2/3 temporal pooling (ARCHITECTURE §8): pool the L4 stream into a STABLE object IDENTITY ────────────────────
    def reset_object(self) -> None:
        """An object BOUNDARY for L2/3 — start recognising/learning a fresh object (drops the current identity)."""
        self._require_location()
        self.pooler.reset()

    def pool(self, learn: bool = True) -> frozenset:
        """Pool L4's CURRENT feature-at-location code (its active cells, set by the last `sense_at`) into the L2/3 identity —
        STABLE across fixations, recognised incrementally. Passes L4's BURST (was the last sensation unpredicted?) so the
        pooler treats a surprising fixation as NOVELTY, not as recognition of a different object (feature-only trap). Returns
        the object-identity SDR (empty if unrecognised in infer)."""
        self._require_location()
        l4 = self.layers["L4"].htm
        return self.pooler.pool(l4._active, learn=learn, bursting=l4.bursting())

    def object_id(self) -> int:
        """A stable integer label for the currently-recognised object (−1 if none) — the L2/3 identity decoded."""
        self._require_location()
        return self.pooler.which()

    # ── OBJECT-CENTRIC frame + the emergent boundary (ARCHITECTURE §8): one event anchors the frame AND the identity ───
    def start_object(self) -> None:
        """An object ONSET — the ONE coupled event (Lewis 2019: a fresh grid phase is BOTH the frame origin AND the object's
        unique location space). Re-ANCHOR the L6a frame to the canonical origin (so subsequent path integration is
        object-relative → translation-invariant) AND reset L2/3 to start a fresh identity. Called explicitly at a
        learning-time boundary (the honest minimal episode cue), and fired EMERGENTLY by `perceive` on recognition failure."""
        self._require_location()
        self.locate(self._anchor)
        self.pooler.reset()

    def perceive(self, feature: SDR, learn: bool = True):
        """Sense a feature at the current OBJECT-RELATIVE location and pool it into the L2/3 identity — and, if this is a
        RECOGNITION FAILURE (an active object hypothesis that no longer holds, with nothing else recognised), fire the coupled
        object-onset here: re-anchor the frame + start a fresh identity, then bind this feature as the new object's first
        (origin) feature. That single non-recognition event does both — no symbolic segmenter. During LEARNING the pooler
        persists (never 'fails' mid-object), so the boundary only fires at INFERENCE, exactly as the theory supports.
        Returns the object-identity SDR."""
        self._require_location()
        had_object = bool(self.pooler.active)
        self.sense_at(feature, learn)
        identity = self.pool(learn)
        if had_object and not identity:                 # recognition FAILURE ⇒ a new object begins HERE
            self.start_object()                         # ONE event: re-anchor frame + reset identity
            self.sense_at(feature, learn)               # bind this feature at the new object's origin
            identity = self.pool(learn)
        return identity

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
