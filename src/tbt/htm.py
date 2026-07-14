"""The ONE cortical-layer mechanism — HTM sequence memory (Hawkins & Ahmad 2016, "Why Neurons Have Thousands of Synapses",
arXiv 1511.00083). Salvaged + MERGED 2026-07-09 (the old `sequence.py`/`SequenceMemory` is folded in here — they were never
two things). See `ARCHITECTURE.md` / `STATUS.md`.

WHY THERE IS ONE MECHANISM, NOT A "DYNAMICS LAYER". Per Hawkins & Ahmad, sequence memory is "the most fundamental operation
of ALL neocortical tissue." It is not a module and not a special layer — EVERY cortical layer runs THIS mechanism; a layer
learns the forward model (the dynamics) of its OWN content by learning TRANSITIONS through its dendrites. So a column does
NOT get a separate dynamics layer — every layer is already a dynamics/sequence memory.

THE MECHANISM (one pyramidal cell has three dendrite zones):
  * PROXIMAL  — the feedforward drive: WHICH minicolumns are active now (a COLUMN-level question). SDR-native: the input
    SDR's active bits ARE the active columns (`observe(active)`). This layer does NOT learn proximal synapses — column
    selection from a raw input space is `encoders.SpatialPooler`, kept DELIBERATELY UPSTREAM: proximal is COLUMN-level,
    basal/apical are CELL-level (which cell within a firing column) — different operations at different grains. `proximal-in`
    is `SpatialPooler(raw)` ONLY at the sensory front-end; between layers it is a lower layer's output SDR taken directly
    (no pooling). So `observe(active, …)` always receives an already-resolved column set — do NOT add proximal dendrites here.
  * BASAL / DISTAL — the CONTEXT that DEPOLARISES a cell into the PREDICTIVE state. Learning a transition = growing basal
    synapses onto the context (`seg`, `_predicted`, `_learn_segment`). **What drives the basal context is the ONLY thing
    that distinguishes the layers.**
  * APICAL — top-down feedback: a PURE tiebreak that NARROWS a basal prediction; an active apical segment alone never fires
    a cell or bursts (Numenta apical_tiebreak_temporal_memory).
A predicted cell fires and inhibits its siblings → the code is context-specific; an unpredicted column BURSTS (all cells) =
the novelty signal. Presynaptic "cells" are any hashable, so an SDR's bits, a location code, etc. can all drive the basal.

THE LAYERS = this ONE class, differentiated by TWO wirings declared when a column is COMPOSED (never by subclassing the
mechanism), because biology fixes a layer's role from two different sources:
  * the basal CONTEXT-IN that drives its dendrites (location / own recurrence / top-down goal / another column) — biology
    sets this by CONNECTIVITY within the UNIFORM canonical microcircuit (thalamic input is only ~10% of synapses; a region's
    function comes from WHAT it is wired to — Mountcastle; Douglas & Martin 2004). This is the `context=` knob.
  * the projection TARGET-OUT its output drives (motor+subcortical / thalamus / cross-column / feedforward) — biology sets
    this by CELL-TYPE identity fixed in development (birth-order inside-out + a transcription-factor code: Fezf2/Ctip2 → L5b
    subcortical MOTOR; Tbr1 → L6 cortico-THALAMIC; Satb2 → cortico-cortical). So "L5 is the motor output" / "L6 drives the
    thalamus" are NOT derivable from a layer's input — they are asserted by hand when we wire the column, exactly as
    development asserts them genetically.
So a layer = ONE HTMLayer + a declared WIRING: the three dendrite-zone INPUTS (proximal-in = SP(raw) at the front-end else a
lower layer's output SDR; context-in; apical-in — all already exposed as `observe` arguments) plus TARGET-OUT (where its
output projects). context-in + target-out are the two DIFFERENTIATING knobs above. The mechanism stays uniform; the
differentiation is entirely the wiring. BUILT/NOT markers refer to that LAYER WIRING; the mechanism itself is built here.
  * L4 — sensory INPUT (granular). Basal context = the L6a LOCATION → predict the FEATURE-AT-LOCATION. This is the TBT
    advance over pure temporal memory: predict the next feature from the next MOVEMENT (location), not from the previous
    feature — so the model is order-INVARIANT (an object = features-at-locations, sensed in ANY movement order). [This class
    SUPPORTS it via `observe(x, context=location)` / `predict_at(location)`; the L4↔L6 loop itself is NOT BUILT.]
  * L2/3 — OUTPUT / object (supragranular). NB **two anatomical layers (L2 + L3)**. Basal context = its own recurrence;
    POOLS the L4 feature-at-location stream into the STABLE OBJECT (larger than L4's receptive field) and L3 adds LATERAL
    VOTING across columns. [The temporal POOLING is BUILT as `pooler.ColumnPooler` — the pooler's decoupled stable output is
    ASSOCIATE-in-a-pooling-regime, not this HTMLayer (ARCHITECTURE §8); L3 lateral VOTING (cross-column) is NOT BUILT — the
    thalamus's job.]
  * L5 — OUTPUT / MOTOR (infragranular). NB **sublayers with DIFFERENT jobs**: L5a slender-tufted / IT (intratelencephalic)
    = cortico-cortical association; L5b thick-tufted / PT (pyramidal tract) = the SUBCORTICAL MOTOR OUTPUT — projects to
    brainstem/spinal cord/tectum (this is where BEHAVIOUR is enacted) and drives the thalamus. Content = sequences of MOTOR
    commands (skills/behaviours) + DISPLACEMENT cells (object-to-object relations; Lewis et al. 2019). Basal context = the
    current sensorimotor state + top-down goal. The IT→PT flow (L4 → IT → PT) makes the motor output specifically L5b/PT.
    [NOT BUILT.]
  * L6 — LOCATION (infragranular). NB **sublayers**: L6a = the grid-cell LOCATION code (drives L4's location signal — ~45%
    of L4 input is from L6), PATH-INTEGRATED by movement; L6b = the sensor ORIENTATION; L6-CT (corticothalamic) = the
    thalamus driver. [The location CODE exists (`encoders.GridEncoder`); the L6 path-integration LAYER + the L4↔L6 loop are
    NOT BUILT.]

WHAT THE "4 layers" SHORTHAND HIDES (do not smooth over): the neocortex is 6 anatomical layers + sublaminae + distinct cell
types. "L2/3" is two layers; "L5" collapses cortico-cortical (L5a/IT) AND motor-output (L5b/PT) cells; "L6" collapses
location (L6a), orientation (L6b), and the thalamus driver (CT). We start from the simplified functional view and refine.

BUILT HERE: the ONE mechanism (this class) — proximal SDR-in, pluggable basal context, apical, learn, predict, burst.
NOT BUILT: any actual layer's wiring (the L4↔L6 sensorimotor loop, L2/3 pooling+voting, L5 motor/displacement, L6 path
integration) or a column composing them — those come with the multi-column loop (ARCHITECTURE.md). Pure stdlib.
Sources: Hawkins & Ahmad 2016 (1511.00083); Thousand Brains Project (2412.18354, 2507.04494); Numenta columns (Hawkins 2019).
"""

from __future__ import annotations


class HTMLayer:
    """ONE cortical layer = an HTM sequence memory (proximal input + basal context + apical feedback). It becomes L4 / L2·3 /
    L5 / L6 by CHOOSING what drives its basal context — see the module docstring; every layer is a dynamics/forward-model
    memory. `observe(active)` steps it (context defaults to the recurrent PHASE → a temporal sequence); pass `context=` to
    drive the basal from a location / task state (sensorimotor / PFC). `predict()` / `predict_at(context)` read the prediction.
    `M` cells per minicolumn give HIGH-ORDER context (the SAME bit fires a DIFFERENT cell in a different context); `order<=1`
    → 1 cell/column = first-order."""

    def __init__(self, order: int = 2, cells_per_column: int = 8, activation_threshold: int = 4, min_threshold: int = 2,
                 connected: float = 0.5, init_perm: float = 0.40, perm_inc: float = 0.15, perm_dec: float = 0.05,
                 predicted_dec: float = 0.01, max_syn: int = 24):
        self.M = 1 if order <= 1 else int(cells_per_column)     # cells per minicolumn (context capacity)
        self.activation_threshold = int(activation_threshold)   # connected synapses onto the context to DEPOLARISE a cell
        self.min_threshold = int(min_threshold)                 # potential synapses to count a segment as MATCHING (winner pick)
        self.connected, self.init_perm = float(connected), float(init_perm)
        self.perm_inc, self.perm_dec, self.max_syn = float(perm_inc), float(perm_dec), int(max_syn)
        self.predicted_dec = float(predicted_dec)               # punishment on a basal segment that predicted a column that did not activate
        self.seg: dict = {}                                    # cell (col, cell) → list of BASAL segments; a segment = {presyn: permanence}
        self.apical_seg: dict = {}                             # cell → list of APICAL segments {apical_bit: permanence}
        self._active: set = set()                              # active cells this step (the PHASE / recurrent basal context)
        self._winners: set = set()                            # winner cells (presynaptic candidates for growth)
        self._predictive: set = set()                         # cells PREDICTIVE for next step
        self._active_segs: dict = {}                          # cell → segment(s) that made it predictive (reinforce/punish)
        self._burst = False

    # ---- segment matches + the one Hebbian learn rule ---------------------------------------------------------
    def _connected_match(self, seg, cells) -> int:
        return sum(1 for c, p in seg.items() if p >= self.connected and c in cells)

    def _potential_match(self, seg, cells) -> int:
        return sum(1 for c in seg if c in cells)

    def _learn_segment(self, seg, active, grow) -> None:
        """HEBBIAN: reinforce synapses onto cells that WERE active (+inc), weaken the rest (−dec, pruned at 0), and GROW
        new synapses onto the `grow` cells the segment does not yet cover (up to `max_syn`)."""
        for c in list(seg):
            if c in active:
                seg[c] = min(1.0, seg[c] + self.perm_inc)
            else:
                seg[c] -= self.perm_dec
                if seg[c] <= 0.0:
                    del seg[c]
        for c in grow:
            if c not in seg and len(seg) < self.max_syn:
                seg[c] = self.init_perm

    # ---- prediction (basal) + the basal-determined burst winner ----------------------------------------------
    def _predicted(self, context):
        """The cells DEPOLARISED by the context = those with a connected BASAL segment matching it (≥
        `activation_threshold`), plus the matching segments (for reinforce/punish). Returns `(cells, {cell: [segments]})`."""
        pred, segs = set(), {}
        for cell, ss in self.seg.items():
            for s in ss:
                if self._connected_match(s, context) >= self.activation_threshold:
                    pred.add(cell)
                    segs.setdefault(cell, []).append(s)
        return pred, segs

    def _winner(self, col, context):
        """The winner cell of a bursting column, BASAL-determined: the cell whose best segment MATCHES the context most, if
        ≥ `min_threshold`; else the DETERMINISTIC least-used cell (fewest segments, lowest index)."""
        best_cell, best_seg, best_score = None, None, self.min_threshold - 1
        for c in range(self.M):
            for s in self.seg.get((col, c), []):
                m = self._potential_match(s, context)
                if m > best_score:
                    best_cell, best_seg, best_score = (col, c), s, m
        if best_cell is not None:
            return best_cell, best_seg
        least = min(range(self.M), key=lambda c: (len(self.seg.get((col, c), [])), c))
        return (col, least), None

    # ---- apical (top-down feedback) — a PURE TIEBREAK, never fires a cell or bursts ---------------------------
    def _apical_predicted(self, apical) -> set:
        """The cells CONNECTED-supported by the top-down feedback — the tiebreak set that narrows a basal prediction."""
        return {cell for cell, ss in self.apical_seg.items()
                if any(self._connected_match(s, apical) >= self.activation_threshold for s in ss)}

    def _learn_apical(self, cell, apical) -> None:
        """Reinforce/grow the cell's best-matching APICAL segment onto the feedback bits (DOWNSTREAM of the basal choice)."""
        ss = self.apical_seg.setdefault(cell, [])
        best, best_m = None, self.min_threshold - 1
        for s in ss:
            m = self._potential_match(s, apical)
            if m > best_m:
                best, best_m = s, m
        if best is None:
            best = {}
            ss.append(best)
        self._learn_segment(best, apical, apical)

    def _apical_narrow(self, basal_cells, apic_pred):
        """The APICAL TIEBREAK: narrow a column's basally-predicted cells to the apically-supported subset IF ANY exist,
        else keep ALL of them (refine, never suppress to empty or burst)."""
        supported = [cell for cell in basal_cells if cell in apic_pred]
        return supported or basal_cells

    # ---- the layer STEP: observe / predict (folded from the old sequence.py) ----------------------------------
    def observe(self, active, context=None, apical=None, learn: bool = True) -> None:
        """One input arrived: `active` = the input SDR's active bits (the PROXIMAL drive = the active columns). A column with
        a basally-predicted cell fires it (APICAL narrows to the top-down-supported subset if any); an unpredicted column
        BURSTS (all cells, a basal-determined winner). Firing cells reinforce their basal segment onto the DRIVING CONTEXT
        and, downstream, learn their apical segment. `context` overrides the recurrent phase (the sensorimotor/PFC driver —
        e.g. a location for L4); `apical` = top-down feedback bits (None = the plain temporal step). `learn=False` =
        INFERENCE ONLY (step the phase + re-predict, but no reinforce/grow/punish) — for a FROZEN held-out generalization test."""
        cols = set(active)
        drive = self._active if context is None else set(context)          # the basal driver: recurrent phase, or an external context
        grow = self._winners if context is None else set(context)          # presynaptic growth candidates
        apical = set(apical) if apical else None
        apic_pred = self._apical_predicted(apical) if apical else set()
        new_active, new_winners = set(), set()
        self._burst = False
        for col in cols:
            basal = [(col, c) for c in range(self.M) if (col, c) in self._predictive]
            if basal:                                                       # PREDICTED: the depolarised cells fire
                for cell in self._apical_narrow(basal, apic_pred):
                    new_active.add(cell)
                    new_winners.add(cell)
                    if learn:
                        for s in self._active_segs.get(cell, []):
                            self._learn_segment(s, drive, grow)
                        if apical is not None:
                            self._learn_apical(cell, apical)
            else:                                                           # BURST: no prediction (surprise)
                self._burst = True
                for c in range(self.M):
                    new_active.add((col, c))
                wcell, best = self._winner(col, drive)
                new_winners.add(wcell)
                if learn and grow:                                          # a sequence start has no context to learn from
                    if best is None:
                        best = {}
                        self.seg.setdefault(wcell, []).append(best)
                    self._learn_segment(best, drive, grow)
                if learn and apical is not None:
                    self._learn_apical(wcell, apical)
        if learn:
            for cell in self._predictive:                                   # PUNISH: predicted a column that did NOT activate
                if cell[0] not in cols:
                    for s in self._active_segs.get(cell, []):
                        for c in list(s):
                            if c in drive:
                                s[c] -= self.predicted_dec
                                if s[c] <= 0.0:
                                    del s[c]
        self._active, self._winners = new_active, new_winners
        self._predictive, self._active_segs = self._predicted(self._active)

    def predict(self) -> set:
        """The predicted next SDR = the columns of the cells predictive for next step (empty on a burst). Decode 'which
        content' by OVERLAP against known SDRs externally (the encoder's job) — there is no symbol table here."""
        return {col for (col, _c) in self._predictive}

    def predict_at(self, context) -> set:
        """SENSORIMOTOR query: the columns depolarised by an EXTERNAL basal context (a location / task context), without
        stepping the phase — the feature-at-location read (L4). Pure query; does not mutate state."""
        cells, _segs = self._predicted(set(context))
        return {col for (col, _c) in cells}

    def predict_best(self, candidates):
        """Decode helper: the candidate SDR (a bit-set) with the greatest OVERLAP with the prediction, or None."""
        pred = self.predict()
        best, best_ov = None, 0
        for sdr in candidates:
            ov = len(pred & set(sdr))
            if ov > best_ov:
                best, best_ov = sdr, ov
        return best

    def bursting(self) -> bool:
        """Did the last `observe` contain an unpredicted (bursting) column — i.e. was the input surprising?"""
        return self._burst

    def reset(self) -> None:
        """A sequence boundary: clear the phase (active/winner/predictive); the learned segments PERSIST."""
        self._active, self._winners, self._predictive, self._active_segs = set(), set(), set(), {}
        self._burst = False
