"""
temporal_memory.py — Temporal Memory algorithm with optional location conditioning.

Operates on the spatial pooler's output (active minicolumns) and adds
cell-level context that enables higher-order sequence learning.

This module implements the core mechanism of Layer 3 (lower L2/3) in the
TBT cortical column. Layer 3 receives the bound feature-location
representation from Layer 4, and learns sequences of those bindings using
temporal memory. Cells within minicolumns encode which sequence context
is currently active, so the same minicolumn can mean different things in
different sequential contexts.

TBT Extension — Location Conditioning (L6a → L3):
  In standard HTM temporal memory, cells become predictive based solely on
  which other cells were active in the prior timestep (sequence context).
  In TBT's Layer 3, cells must also condition on the current location from
  Layer 6a — binding "what feature" to "at what position in the object's
  reference frame."

  This is the mechanism that distinguishes TBT from plain HTM. Without
  location conditioning, the column learns sequences of features but cannot
  associate features with positions. With it, the cell that fires for a
  given sensory input depends on both the sequential context AND the current
  location in the reference frame.

  Implementation:
  Each dendritic segment has two synapse populations:
    1. Cell synapses (standard): presynaptic = prior active cells (sequence)
    2. Location synapses (new): presynaptic = current L6a location SDR bits

  Segment activation counts both. A segment becomes active when:
    (connected cell synapses to prior active cells) +
    (connected location synapses to current L6a location bits)
    >= segment_activation_threshold

  Set location_sdr_length=0 (default) for standard HTM temporal memory
  with no location conditioning.

Timing:
  Phase 1 (activation): checks whether cells were correctly predicted using
    prev_active cells + prev_location (L6a state from prior timestep).
  Phase 2 (prediction): computes new predictive state using current active
    cells + current L6a location.
  Learning: grows cell synapses toward prev_winner cells, grows location
    synapses toward prev_location active bits.

Layer notes:
  - This implements Layer 3 (lower L2/3). Layer 2 (upper L2/3), which
    integrates L3 output with L5a motor context and projects long-range,
    is not yet separately modeled.
  - location_sdr should be the output of a GridCellLayer (grid_cells.py),
    representing the L6a location in the object's reference frame.
  - The L6a → L3 connection is implemented here as basal context synapses.
    The L6a → L4 modulatory connection is separate and not yet implemented.

Usage:
    # Standard HTM temporal memory (no location)
    tm = TemporalMemory(num_minicolumns=512, cells_per_col=4)
    active_cells = tm.compute(active_cols_sdr, learn=True)

    # TBT location-conditioned temporal memory (L3 with L6a context)
    tm = TemporalMemory(
        num_minicolumns=512,
        cells_per_col=4,
        location_sdr_length=192,   # total length of concatenated L6a SDR
    )
    active_cells = tm.compute(
        active_cols_sdr,
        location_sdr=grid_layer.get_location_sdr(),
        learn=True,
    )
"""

import numpy as np
from typing import Optional, Tuple


class TemporalMemory:
    """Temporal Memory with basal dendritic segments, Hebbian learning,
    and optional location conditioning from Layer 6.

    Args:
        num_minicolumns: Number of minicolumns (matches spatial pooler output).
        cells_per_col: Number of cells per minicolumn.
        max_segs_per_cell: Maximum dendritic segments per cell.
        max_synapses_per_seg: Maximum cell-to-cell synapses per segment.
        location_sdr_length: Length of the L6 location SDR. 0 = disabled.
        max_loc_synapses_per_seg: Maximum location synapses per segment.
            Only used when location_sdr_length > 0.
        segment_activation_threshold: Minimum connected active inputs
            (cell + location combined) to activate a segment.
        min_threshold: Minimum potential active inputs for "matching" segment
            selection during bursting.
        permanence_threshold: Synapse connection threshold.
        permanence_inc: Permanence increase for active synapses (both types).
        permanence_dec: Permanence decrease for inactive synapses (both types).
        permanence_punish: Permanence decrease for incorrect predictions.
        initial_permanence: Starting permanence for newly grown synapses.
        max_new_synapses_per_seg: Target new synapses per learning step.
        max_new_loc_synapses_per_seg: Target new location synapses per step.
        min_loc_contribution: When location conditioning is enabled, the
            minimum number of connected location synapses that must be active
            for a segment to fire. Enforces conjunctive cell+location gating:
            the location is REQUIRED context, not just additive. Default 0
            (location is additive, same as before). Set to 1 or higher to
            require location context.
        location_activation_threshold: When > 0, a segment becomes predictive
            if its connected location synapses alone reach this count, regardless
            of cell synapse activity. This makes the reference frame
            position-addressable: placing the grid at position p primes the
            correct cells immediately, even after a reset with no prior context.
            Biologically: location signal on basal/distal dendrites produces a
            dendritic spike that primes the cell without requiring sequential
            cell context. Default 0 (disabled, old behaviour).
        seed: Random seed.
    """

    def __init__(
        self,
        num_minicolumns: int,
        cells_per_col: int = 4,
        max_segs_per_cell: int = 32,
        max_synapses_per_seg: int = 32,
        location_sdr_length: int = 0,
        max_loc_synapses_per_seg: int = 16,
        segment_activation_threshold: int = 10,
        min_threshold: int = 8,
        permanence_threshold: float = 0.5,
        permanence_inc: float = 0.1,
        permanence_dec: float = 0.1,
        permanence_punish: float = 0.01,
        initial_permanence: float = 0.21,
        max_new_synapses_per_seg: int = 20,
        max_new_loc_synapses_per_seg: int = 10,
        min_loc_contribution: int = 0,
        location_activation_threshold: int = 0,
        seed: Optional[int] = None,
    ):
        self.num_minicolumns = num_minicolumns
        self.cells_per_col = cells_per_col
        self.total_cells = num_minicolumns * cells_per_col
        self.max_segs_per_cell = max_segs_per_cell
        self.max_synapses_per_seg = max_synapses_per_seg
        self.location_sdr_length = location_sdr_length
        self.max_loc_synapses_per_seg = max_loc_synapses_per_seg
        self.segment_activation_threshold = segment_activation_threshold
        self.min_threshold = min_threshold
        self.permanence_threshold = permanence_threshold
        self.permanence_inc = permanence_inc
        self.permanence_dec = permanence_dec
        self.permanence_punish = permanence_punish
        self.initial_permanence = initial_permanence
        self.max_new_synapses_per_seg = max_new_synapses_per_seg
        self.max_new_loc_synapses_per_seg = max_new_loc_synapses_per_seg
        self.min_loc_contribution = min_loc_contribution
        self.location_activation_threshold = location_activation_threshold
        self.use_location = location_sdr_length > 0

        self.rng = np.random.default_rng(seed)
        self.total_segs = self.total_cells * max_segs_per_cell

        # ── Cell state ────────────────────────────────────────────────────
        self.cell_active = np.zeros(self.total_cells, dtype=bool)
        self.cell_predictive = np.zeros(self.total_cells, dtype=bool)
        self.cell_winner = np.zeros(self.total_cells, dtype=bool)

        self.prev_active = np.zeros(self.total_cells, dtype=bool)
        self.prev_winner = np.zeros(self.total_cells, dtype=bool)
        self.prev_predictive = np.zeros(self.total_cells, dtype=bool)

        # ── Location state ────────────────────────────────────────────────
        # prev_location: location SDR from the previous timestep.
        # This is what was active when the current predictions were formed,
        # and so is the target for growing new location synapses.
        self.prev_location = np.zeros(location_sdr_length, dtype=bool)

        # ── Cell-to-cell segment synapses ─────────────────────────────────
        self.seg_syn_cells = np.full(
            (self.total_segs, max_synapses_per_seg), -1, dtype=np.int32
        )
        self.seg_syn_perm = np.zeros(
            (self.total_segs, max_synapses_per_seg), dtype=np.float32
        )

        # ── Location synapses (TBT extension) ─────────────────────────────
        # seg_loc_bits[s, k] = index into the location SDR (-1 = empty)
        # seg_loc_perm[s, k] = permanence for that location synapse
        if self.use_location:
            self.seg_loc_bits = np.full(
                (self.total_segs, max_loc_synapses_per_seg), -1, dtype=np.int32
            )
            self.seg_loc_perm = np.zeros(
                (self.total_segs, max_loc_synapses_per_seg), dtype=np.float32
            )
        else:
            self.seg_loc_bits = None
            self.seg_loc_perm = None

        self.cell_num_segs = np.zeros(self.total_cells, dtype=np.int32)
        self._created_segs = np.array([], dtype=np.int32)  # cache, updated on segment creation
        self.iteration = 0

    # ── Indexing helpers ──────────────────────────────────────────────────

    def _col_cells(self, col: int) -> np.ndarray:
        start = col * self.cells_per_col
        return np.arange(start, start + self.cells_per_col)

    def _cell_segs(self, cell: int) -> np.ndarray:
        n = self.cell_num_segs[cell]
        if n == 0:
            return np.array([], dtype=np.int32)
        base = cell * self.max_segs_per_cell
        return np.arange(base, base + n, dtype=np.int32)

    # ── Segment activity ──────────────────────────────────────────────────

    def _compute_segment_activity(
        self,
        active_cells: np.ndarray,
        location_sdr: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute connected and potential active counts per segment.

        Returns:
            seg_connected_active: int32 (total_segs,) — cell+loc combined
            seg_potential_active: int32 (total_segs,)
            seg_loc_connected_active: int32 (total_segs,) — location only
        """
        seg_conn = np.zeros(self.total_segs, dtype=np.int32)
        seg_pot  = np.zeros(self.total_segs, dtype=np.int32)
        seg_loc  = np.zeros(self.total_segs, dtype=np.int32)

        created = self._created_segs
        if len(created) == 0:
            return seg_conn, seg_pot, seg_loc

        sc = self.seg_syn_cells[created]
        sp = self.seg_syn_perm[created]
        valid      = sc >= 0
        syn_active = valid & active_cells[np.where(valid, sc, 0)]
        connected  = valid & (sp >= self.permanence_threshold)

        seg_conn[created] = (connected & syn_active).sum(axis=1)
        seg_pot[created]  = syn_active.sum(axis=1)

        if self.use_location and location_sdr is not None:
            lc = self.seg_loc_bits[created]
            lp = self.seg_loc_perm[created]
            loc_valid  = lc >= 0
            loc_active = loc_valid & location_sdr[np.where(loc_valid, lc, 0)]
            loc_conn   = loc_valid & (lp >= self.permanence_threshold)
            loc_counts = (loc_conn & loc_active).sum(axis=1)
            seg_conn[created] += loc_counts
            seg_pot[created]  += loc_active.sum(axis=1)
            seg_loc[created]   = loc_counts

        return seg_conn, seg_pot, seg_loc

    # ── Phase 1: Activation ───────────────────────────────────────────────

    def _activate_cells(
        self,
        active_minicolumns: np.ndarray,
        seg_connected_active: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Determine which cells fire and which columns burst."""
        active = np.zeros(self.total_cells, dtype=bool)
        winner = np.zeros(self.total_cells, dtype=bool)
        bursting_cols = []

        for col in np.where(active_minicolumns)[0]:
            cells = self._col_cells(col)
            predictive_cells = cells[self.prev_predictive[cells]]

            if len(predictive_cells) > 0:
                active[predictive_cells] = True
                winner[predictive_cells] = True
            else:
                active[cells] = True
                bursting_cols.append(col)

        return active, winner, np.array(bursting_cols, dtype=np.int32)

    # ── Phase 2: Prediction ───────────────────────────────────────────────

    def _predict_cells(self, seg_connected_active: np.ndarray,
                       seg_loc_active: Optional[np.ndarray] = None) -> np.ndarray:
        """Cells whose segments are active become predictive.

        A segment makes its owning cell predictive if EITHER:
          (a) Combined (cell + location) connected active synapses
              >= segment_activation_threshold  [standard HTM path], OR
          (b) Location connected active synapses alone
              >= location_activation_threshold  [new — position-addressable]

        Condition (b) is what eliminates burst-on-first-step after reset:
        the location signal from L6a can prime the correct cells without any
        prior sequential cell context. This maps onto the biological model
        where the location signal on basal/distal dendrites produces a
        dendritic spike independent of prior activity.

        If min_loc_contribution > 0, condition (a) additionally requires
        at least that many location synapses (conjunctive gate).
        """
        predictive = np.zeros(self.total_cells, dtype=bool)

        # Condition (a): combined threshold
        eligible = seg_connected_active >= self.segment_activation_threshold
        if self.use_location and self.min_loc_contribution > 0 and seg_loc_active is not None:
            eligible &= seg_loc_active >= self.min_loc_contribution

        # Condition (b): location-alone threshold (OR with condition a)
        if (self.use_location
                and self.location_activation_threshold > 0
                and seg_loc_active is not None):
            loc_only = seg_loc_active >= self.location_activation_threshold
            eligible = eligible | loc_only

        active_segs = np.where(eligible)[0]
        owning_cells = active_segs // self.max_segs_per_cell
        predictive[owning_cells] = True
        return predictive

    # ── Learning helpers ──────────────────────────────────────────────────

    def _best_matching_segment(
        self, cell: int, seg_potential_active: np.ndarray
    ) -> int:
        n = self.cell_num_segs[cell]
        if n == 0:
            return -1
        base = cell * self.max_segs_per_cell
        seg_slice = seg_potential_active[base:base + n]
        best_local = int(seg_slice.argmax())
        if seg_potential_active[base + best_local] >= self.min_threshold:
            return base + best_local
        return -1

    def _least_used_cell(self, col: int) -> int:
        cells = self._col_cells(col)
        seg_counts = self.cell_num_segs[cells]
        min_count = seg_counts.min()
        candidates = cells[seg_counts == min_count]
        return int(self.rng.choice(candidates))

    def _grow_cell_synapses(
        self, seg: int, prev_winner: np.ndarray, n_desired: int
    ) -> None:
        """Grow new cell-to-cell synapses targeting previous winner cells."""
        if n_desired <= 0:
            return
        existing = set(int(c) for c in self.seg_syn_cells[seg] if c >= 0)
        candidates = np.where(prev_winner)[0]
        candidates = candidates[
            np.array([c not in existing for c in candidates])
        ]
        if len(candidates) == 0:
            return
        n_grow = min(n_desired, len(candidates))
        empty_slots = np.where(self.seg_syn_cells[seg] == -1)[0]
        n_grow = min(n_grow, len(empty_slots))
        if n_grow <= 0:
            return
        chosen = self.rng.choice(candidates, size=n_grow, replace=False)
        for i in range(n_grow):
            self.seg_syn_cells[seg, empty_slots[i]] = chosen[i]
            self.seg_syn_perm[seg, empty_slots[i]] = self.initial_permanence

    def _grow_location_synapses(
        self, seg: int, location_sdr: np.ndarray, n_desired: int
    ) -> None:
        """Grow new location synapses targeting active bits in location_sdr."""
        if not self.use_location or n_desired <= 0 or not location_sdr.any():
            return
        existing = set(int(b) for b in self.seg_loc_bits[seg] if b >= 0)
        candidates = np.where(location_sdr)[0]
        candidates = candidates[
            np.array([b not in existing for b in candidates])
        ]
        if len(candidates) == 0:
            return
        n_grow = min(n_desired, len(candidates))
        empty_slots = np.where(self.seg_loc_bits[seg] == -1)[0]
        n_grow = min(n_grow, len(empty_slots))
        if n_grow <= 0:
            return
        chosen = self.rng.choice(candidates, size=n_grow, replace=False)
        for i in range(n_grow):
            self.seg_loc_bits[seg, empty_slots[i]] = chosen[i]
            self.seg_loc_perm[seg, empty_slots[i]] = self.initial_permanence

    def _reinforce_segment(
        self,
        seg: int,
        prev_active: np.ndarray,
        prev_location: Optional[np.ndarray] = None,
    ) -> None:
        """Hebbian update for both cell and location synapses on a segment."""
        # Cell synapses
        valid = self.seg_syn_cells[seg] >= 0
        cell_indices = np.where(valid, self.seg_syn_cells[seg], 0)
        was_active = prev_active[cell_indices] & valid
        self.seg_syn_perm[seg, was_active] += self.permanence_inc
        self.seg_syn_perm[seg, valid & ~was_active] -= self.permanence_dec
        np.clip(self.seg_syn_perm[seg], 0.0, 1.0, out=self.seg_syn_perm[seg])

        # Location synapses
        if self.use_location and prev_location is not None:
            loc_valid = self.seg_loc_bits[seg] >= 0
            loc_indices = np.where(loc_valid, self.seg_loc_bits[seg], 0)
            loc_was_active = prev_location[loc_indices] & loc_valid
            self.seg_loc_perm[seg, loc_was_active] += self.permanence_inc
            self.seg_loc_perm[seg, loc_valid & ~loc_was_active] -= self.permanence_dec
            np.clip(
                self.seg_loc_perm[seg], 0.0, 1.0,
                out=self.seg_loc_perm[seg]
            )

    def _create_segment(self, cell: int) -> int:
        """Create or recycle a segment on a cell. Returns global segment index.

        When a new (non-recycled) segment is created, appends its index to
        self._created_segs so _compute_segment_activity can find it without
        rebuilding the index list from scratch.
        """
        n = self.cell_num_segs[cell]
        base = cell * self.max_segs_per_cell
        if n < self.max_segs_per_cell:
            seg = base + n
            self.cell_num_segs[cell] += 1
            # Append to cache — this index is newly created
            self._created_segs = np.append(self._created_segs, np.int32(seg))
        else:
            # Recycle least-used segment — index already in cache
            seg_range = np.arange(base, base + self.max_segs_per_cell)
            syn_counts = (self.seg_syn_cells[seg_range] >= 0).sum(axis=1)
            seg = int(seg_range[syn_counts.argmin()])
            self.seg_syn_cells[seg] = -1
            self.seg_syn_perm[seg] = 0.0
            if self.use_location:
                self.seg_loc_bits[seg] = -1
                self.seg_loc_perm[seg] = 0.0
        return seg

    # ── Learning ──────────────────────────────────────────────────────────

    def _learn(
        self,
        active_minicolumns: np.ndarray,
        seg_connected_active: np.ndarray,
        seg_potential_active: np.ndarray,
        bursting_cols: np.ndarray,
    ) -> None:
        """Apply all three learning rules.

        Synapses grow toward prev_winner (cell context) and prev_location
        (location context) — the state that existed when the prediction
        being reinforced was formed.
        """
        prev_a = self.prev_active
        prev_w = self.prev_winner
        prev_loc = self.prev_location if self.use_location else None

        bursting_set = set(bursting_cols.tolist())

        # ── Rule 1: Correctly predicted cells ────────────────────────────
        for col in np.where(active_minicolumns)[0]:
            if col in bursting_set:
                continue
            cells = self._col_cells(col)
            for cell in cells[self.cell_active[cells]]:
                for seg in self._cell_segs(cell):
                    if seg_connected_active[seg] >= self.segment_activation_threshold:
                        self._reinforce_segment(seg, prev_a, prev_loc)
                        n_cell_syn = int((self.seg_syn_cells[seg] >= 0).sum())
                        self._grow_cell_synapses(
                            seg, prev_w,
                            self.max_new_synapses_per_seg - n_cell_syn
                        )
                        if self.use_location and prev_loc is not None:
                            n_loc_syn = int((self.seg_loc_bits[seg] >= 0).sum())
                            self._grow_location_synapses(
                                seg, prev_loc,
                                self.max_new_loc_synapses_per_seg - n_loc_syn
                            )

        # ── Rule 2: Bursting columns ──────────────────────────────────────
        for col in bursting_cols:
            cells = self._col_cells(col)
            best_seg = -1
            best_seg_cell = -1
            best_potential = self.min_threshold - 1

            for cell in cells:
                seg = self._best_matching_segment(cell, seg_potential_active)
                if seg >= 0 and seg_potential_active[seg] > best_potential:
                    best_potential = seg_potential_active[seg]
                    best_seg = seg
                    best_seg_cell = cell

            if best_seg >= 0:
                winner_cell = best_seg_cell
                self.cell_winner[winner_cell] = True
                self._reinforce_segment(best_seg, prev_a, prev_loc)
                n_cell_syn = int((self.seg_syn_cells[best_seg] >= 0).sum())
                self._grow_cell_synapses(
                    best_seg, prev_w,
                    self.max_new_synapses_per_seg - n_cell_syn
                )
                if self.use_location and prev_loc is not None:
                    n_loc_syn = int((self.seg_loc_bits[best_seg] >= 0).sum())
                    self._grow_location_synapses(
                        best_seg, prev_loc,
                        self.max_new_loc_synapses_per_seg - n_loc_syn
                    )
            else:
                winner_cell = self._least_used_cell(col)
                self.cell_winner[winner_cell] = True
                if prev_w.any():
                    new_seg = self._create_segment(winner_cell)
                    self._grow_cell_synapses(
                        new_seg, prev_w, self.max_new_synapses_per_seg
                    )
                    if self.use_location and prev_loc is not None:
                        self._grow_location_synapses(
                            new_seg, prev_loc,
                            self.max_new_loc_synapses_per_seg
                        )

        # ── Rule 3: Punish incorrect predictions ──────────────────────────
        for col in np.where(~active_minicolumns)[0]:
            cells = self._col_cells(col)
            for cell in cells:
                if not self.prev_predictive[cell]:
                    continue
                for seg in self._cell_segs(cell):
                    if seg_connected_active[seg] >= self.segment_activation_threshold:
                        valid = self.seg_syn_cells[seg] >= 0
                        self.seg_syn_perm[seg, valid] -= self.permanence_punish
                        np.clip(
                            self.seg_syn_perm[seg], 0.0, 1.0,
                            out=self.seg_syn_perm[seg]
                        )
                        if self.use_location:
                            loc_valid = self.seg_loc_bits[seg] >= 0
                            self.seg_loc_perm[seg, loc_valid] -= self.permanence_punish
                            np.clip(
                                self.seg_loc_perm[seg], 0.0, 1.0,
                                out=self.seg_loc_perm[seg]
                            )

    # ── Main compute ──────────────────────────────────────────────────────

    def compute(
        self,
        active_minicolumns: np.ndarray,
        location_sdr: Optional[np.ndarray] = None,
        learn: bool = True,
    ) -> np.ndarray:
        """Run one timestep of temporal memory (Layer 3).

        Args:
            active_minicolumns: Bool (num_minicolumns,) — spatial pooler
                output from Layer 4.
            location_sdr: Bool (location_sdr_length,) — current L6a location
                SDR from the grid cell layer, or None if not using location
                conditioning.
            learn: Whether to update permanences and grow synapses.

        Returns:
            Bool (total_cells,) — active cells this step.
        """
        assert len(active_minicolumns) == self.num_minicolumns

        if location_sdr is None:
            location_sdr = np.zeros(self.location_sdr_length, dtype=bool)

        # Save previous state
        self.prev_active = self.cell_active.copy()
        self.prev_winner = self.cell_winner.copy()
        self.prev_predictive = self.cell_predictive.copy()
        prev_location = self.prev_location.copy()

        # Update stored previous location for next step's learning
        if self.use_location:
            self.prev_location = location_sdr.copy()

        # Phase 1: segment activity vs prev_active + prev_location
        seg_conn_prev, seg_pot_prev, _ = self._compute_segment_activity(
            self.prev_active,
            prev_location if self.use_location else None,
        )

        active, winner, bursting_cols = self._activate_cells(
            active_minicolumns, seg_conn_prev
        )
        self.cell_active = active
        self.cell_winner = winner

        # Phase 2: segment activity vs current_active + current_location
        seg_conn_cur, _, seg_loc_cur = self._compute_segment_activity(
            self.cell_active,
            location_sdr if self.use_location else None,
        )
        self.cell_predictive = self._predict_cells(seg_conn_cur, seg_loc_cur)

        if learn:
            self._learn(
                active_minicolumns,
                seg_conn_prev,
                seg_pot_prev,
                bursting_cols,
            )

        self.iteration += 1
        return self.cell_active.copy()

    def reset(self) -> None:
        """Reset all cell state. Call between unrelated sequences."""
        self.cell_active[:] = False
        self.cell_predictive[:] = False
        self.cell_winner[:] = False
        self.prev_active[:] = False
        self.prev_winner[:] = False
        self.prev_predictive[:] = False
        self.prev_location[:] = False

    # ── Diagnostics ───────────────────────────────────────────────────────

    def get_anomaly_score(self) -> float:
        """Fraction of active columns that burst (0 = fully predicted)."""
        col_matrix = self.cell_active.reshape(
            self.num_minicolumns, self.cells_per_col
        )
        active_cols = col_matrix.any(axis=1)
        n_active = active_cols.sum()
        if n_active == 0:
            return 0.0
        n_bursting = (col_matrix.all(axis=1) & active_cols).sum()
        return float(n_bursting / n_active)

    def get_prediction_density(self) -> float:
        """Fraction of cells in predictive state."""
        return float(self.cell_predictive.sum() / self.total_cells)

    def get_segment_stats(self) -> dict:
        """Summary statistics on segment and synapse usage."""
        total_created = int(self.cell_num_segs.sum())
        cells_with_segs = int((self.cell_num_segs > 0).sum())
        valid_cell_syn = int((self.seg_syn_cells >= 0).sum())
        valid_loc_syn = int(
            (self.seg_loc_bits >= 0).sum() if self.use_location else 0
        )
        return {
            "total_segments_created": total_created,
            "cells_with_segments": cells_with_segs,
            "total_cell_synapses": valid_cell_syn,
            "total_location_synapses": valid_loc_syn,
            "mean_segs_per_cell": float(self.cell_num_segs.mean()),
            "max_segs_per_cell_used": int(self.cell_num_segs.max()),
        }