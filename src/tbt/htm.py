"""The shared HTM cell/dendrite mechanism (Hawkins & Ahmad 2016; MICROCIRCUIT.md) — the ONE home of the minicolumn × cell
structure, distal SEGMENTS (synapses with permanences), the Hebbian learn rule, the connected/potential match, the
BASAL-determined burst winner, and the APICAL tiebreak.

Both temporal and sensorimotor memory build on it, differing ONLY in what drives the dendrites (rule 1 — no parallel
systems): `sequence.SequenceMemory` (basal = the previous active cells; predicts the NEXT element) and `l4.L4Layer`
(basal = the L6 LOCATION; predicts the feature AT a location). A cell is depolarised (predictive) by a connected BASAL
segment; an active APICAL segment ALONE never predicts a cell — it only NARROWS the basal prediction (Numenta
apical_tiebreak_temporal_memory). Pure stdlib.
"""

from __future__ import annotations


class HTMLayer:
    """A sheet of minicolumns × `M` cells with distal basal + apical segments. Subclasses drive `observe` with the layer's
    context (temporal / sensorimotor); this base owns the segment machinery so there is exactly one of it."""

    def __init__(self, cells_per_column: int = 8, activation_threshold: int = 4, min_threshold: int = 2,
                 connected: float = 0.5, init_perm: float = 0.40, perm_inc: float = 0.15, perm_dec: float = 0.05,
                 max_syn: int = 24):
        self.M = int(cells_per_column)                          # cells per minicolumn (the context capacity)
        self.activation_threshold = int(activation_threshold)   # connected synapses onto the context to DEPOLARISE a cell
        self.min_threshold = int(min_threshold)                 # potential synapses onto the context to count a segment as MATCHING (winner pick)
        self.connected, self.init_perm = float(connected), float(init_perm)
        self.perm_inc, self.perm_dec, self.max_syn = float(perm_inc), float(perm_dec), int(max_syn)
        self.seg: dict = {}                                    # cell (col, cell) → list of BASAL segments; a segment = {presyn: permanence}
        self.apical_seg: dict = {}                             # cell → list of APICAL segments {apical_bit: permanence}

    # ---- segment matches + the one Hebbian learn rule ---------------------------------------------------------
    def _connected_match(self, seg, cells) -> int:
        return sum(1 for c, p in seg.items() if p >= self.connected and c in cells)

    def _potential_match(self, seg, cells) -> int:
        return sum(1 for c in seg if c in cells)

    def _learn_segment(self, seg, active, grow, punish: bool = True) -> None:
        """HEBBIAN: reinforce synapses onto cells that WERE active (+inc), weaken the rest (−dec, pruned at 0), and GROW
        new synapses onto the `grow` cells the segment does not yet cover (up to `max_syn`). `punish=False` is the
        GROW-ONLY variant (no −dec): the L2/3 proximal pooler accumulates the UNION of an object's feature-locations, all
        of which are positive evidence, so punishing the currently-inactive ones would erode the object's own union."""
        for c in list(seg):
            if c in active:
                seg[c] = min(1.0, seg[c] + self.perm_inc)
            elif punish:
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
        """The winner cell of a bursting column, BASAL-determined (apical learning is downstream, never biases this): the
        cell whose best segment MATCHES the context most, if ≥ `min_threshold`; else the DETERMINISTIC least-used cell
        (fewest segments, lowest index) so a fresh context takes a fresh cell and its downstream accumulates stably."""
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
        """The cells CONNECTED-supported by the top-down feedback (an active apical segment) — the tiebreak set that
        narrows a basal prediction. An active apical segment ALONE never predicts a cell (Hawkins & Ahmad 2016)."""
        return {cell for cell, ss in self.apical_seg.items()
                if any(self._connected_match(s, apical) >= self.activation_threshold for s in ss)}

    def _learn_apical(self, cell, apical) -> None:
        """Reinforce/grow the cell's best-matching APICAL segment onto the current feedback bits (DOWNSTREAM of the basal
        choice); grow a NEW apical segment if none matches yet. Reuses the one Hebbian rule."""
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
        """The APICAL TIEBREAK on a column's basally-predicted cells: narrow to the apically-supported subset IF ANY
        exist, else keep ALL of them (an active apical segment refines, never suppresses to empty or bursts)."""
        supported = [cell for cell in basal_cells if cell in apic_pred]
        return supported or basal_cells
