"""Layer 4 — the sensorimotor INPUT layer (MICROCIRCUIT.md Stage 2): feature-at-location, HTM-native.

Minicolumns = the feed-forward FEATURE (a content SDR's active bits); cells-per-column = the LOCATION context (basal
segments onto the L6 grid SDR); apical = the OBJECT (a pure TIEBREAK — object SELECTION is L2/3's job, Stage 3). L4 IS the
object's feature-at-location MAP: `predict_feature(location)` recalls the feature THERE without sensing it (visited or not
— the map recalls, and generalises to nearby locations via grid-SDR overlap); `observe(feature, location)` learns the map
and BURSTS when a sensed feature was NOT predicted at that location (surprise / new / object-mismatch — the recognition
signal). Basal = the LOCATION (not the previous cells) is the ONLY difference from the temporal `SequenceMemory`: the SAME
`HTMLayer` cell/segment mechanism (rule 1 — no parallel systems). Pure stdlib.
"""

from __future__ import annotations

from .htm import HTMLayer


class L4Layer(HTMLayer):
    """A sheet of feature minicolumns whose cells are location-contexts. `observe`/`predict_feature` drive the shared
    `HTMLayer` dendrites with the L6 LOCATION as the basal context and the object as the apical tiebreak."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._active: set = set()                              # the cells active after the last observe
        self._burst = False                                   # did the last observe burst (a feature unpredicted at its location)?

    def observe(self, feature_cols, location, obj=None, learn: bool = True) -> None:
        """Sense `feature_cols` (the active feature minicolumns) at `location` (the L6 grid-SDR bits). For each feature
        column: if the location DEPOLARISED a cell there (this feature was expected at this location), fire it (apical
        tiebreak by `obj`) and reinforce its basal segment onto the location; else BURST — grow a segment binding this
        feature TO this location (its winner basal-determined). Firing/winner cells learn their apical segment onto `obj`.
        `learn=False` = pure INFERENCE (the recognition / anchoring search): predicted cells FIRE but nothing is reinforced
        or grown, and a burst adds NO cell (an unrecognised feature-at-location contributes no evidence)."""
        location = set(location)
        obj = set(obj) if obj else None
        basal_pred, basal_segs = self._predicted(location)      # cells this LOCATION depolarises
        apic_pred = self._apical_predicted(obj) if obj else set()
        self._burst = False
        active = set()
        for col in feature_cols:
            basal = [(col, c) for c in range(self.M) if (col, c) in basal_pred]
            if basal:                                           # PREDICTED: the feature was expected at this location
                for cell in self._apical_narrow(basal, apic_pred):
                    active.add(cell)
                    if learn:
                        for s in basal_segs.get(cell, []):
                            self._learn_segment(s, location, location)   # reinforce this cell's basal onto the location
                        if obj is not None:
                            self._learn_apical(cell, obj)
            elif learn:                                         # BURST: this feature was NOT predicted here → bind it to the location
                self._burst = True
                wcell, best = self._winner(col, location)
                if best is None:
                    best = {}
                    self.seg.setdefault(wcell, []).append(best)
                self._learn_segment(best, location, location)
                active.add(wcell)
                if obj is not None:
                    self._learn_apical(wcell, obj)
            else:                                               # INFERENCE: an unpredicted feature is just a burst signal (no cell)
                self._burst = True
        self._active = active

    def predict_feature(self, location, obj=None) -> set:
        """The FEATURE predicted AT `location` = the minicolumns the location depolarises (narrowed within each column by
        the object). A PURE query — works for a location never sensed this episode (imagination) as long as the object's
        map covers it (or a nearby location does, via grid overlap). Returns the predicted feature COLUMNS (an SDR)."""
        location = set(location)
        obj = set(obj) if obj else None
        basal_pred, _ = self._predicted(location)
        apic_pred = self._apical_predicted(obj) if obj else set()
        by_col: dict = {}
        for (col, cell) in basal_pred:
            by_col.setdefault(col, []).append((col, cell))
        return {col for col, cells in by_col.items() if self._apical_narrow(cells, apic_pred)}

    def burst(self) -> bool:
        """Did the last `observe` burst? — a sensed feature was not predicted at its location (surprise / object-mismatch)."""
        return self._burst
