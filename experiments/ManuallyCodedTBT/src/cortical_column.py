"""
cortical_column.py — Cortical Column assembly (Phase 8, updated Phase 10).

Wires together all implemented layer components:
  - Encoder          (optional: accepts pre-built or None)
  - SpatialPooler    (L4: feature representation)
  - TemporalMemory   (L3: sequence + location binding)
  - GridCellLayer    (L6a: reference frame)
  - DisplacementLayer (L5b: displacement cell modules)
  - L5aReadout       (L5a: conjunctive readout, optional — Phase 10)

The column is a wiring harness. Pre-built components should be passed in.
Sensible defaults are constructed if no component is provided.

Per-timestep computation (one 5ms cycle):
  1. Encode raw input → input_sdr           (if encoder provided)
  2. Spatial pooler → active_minicolumns    (L4: feature representation)
  3. Get location_sdr from grid layer       (L6a → L3 basal context)
  4. Temporal memory → active_cells         (L3: sequence + location binding)
  5a. If L5aReadout provided: compute displacement from (L3, L4) state
  5b. Apply displacement to grid layer      (L5b → L6a: path integration)
      Source: L5aReadout output (learned) or external float (injected)

Design notes:
  - Encoder is optional. If None, compute() requires a pre-encoded SDR.
  - DisplacementLayer is auto-created paired to grid_layer if not provided.
  - All state is owned by the component objects. The column holds references.
  - compute() returns a dict with all relevant state for downstream use.
  - l5a_readout: if provided, compute() returns 'l5a_displacement' in the
    result dict. External 'displacement' arg overrides L5a when both provided
    (useful for injecting ground-truth displacement during training).

Usage — Phase 10 with learned L5a:
    from l5a_readout import L5aReadout

    l5a = L5aReadout(
        num_l3_cells=col.tm.total_cells,
        num_minicolumns=col._num_minicolumns,
        learning_rate=0.005,
        use_supervised=True,
    )
    col = CorticalColumn(..., l5a_readout=l5a)

    # Operator step — L5a computes and applies displacement automatically
    result = col.compute(encode_symbol('+'), learn=True)

    # Result step — teach L5a from anomaly
    result = col.compute(encode_number(result_val), learn=True)
    l5a.learn(anomaly_score=result['anomaly_score'],
               true_displacement=b)   # supervised mode
"""

import numpy as np
from typing import Optional, Union

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from spatial_pooler import SpatialPooler
from temporal_memory import TemporalMemory
from grid_cells import GridCellLayer, make_number_line_layer
from displacement_layer import DisplacementLayer, make_displacement_layer_from_grid

# L5a is optional — import lazily so the file works without it
try:
    from l5a_readout import L5aReadout as _L5aReadout
except ImportError:
    _L5aReadout = None


class CorticalColumn:
    """A single cortical column wiring L4, L3, L6a, L5b, and optionally L5a.

    Args:
        encoder: Optional encoder object with an encode(value) method.
            If None, compute() must receive a pre-encoded boolean SDR.
        spatial_pooler: Optional pre-built SpatialPooler (L4).
        temporal_memory: Optional pre-built TemporalMemory (L3).
        grid_layer: Optional pre-built GridCellLayer (L6a).
        displacement_layer: Optional pre-built DisplacementLayer (L5b).
        l5a_readout: Optional pre-built L5aReadout (L5a). If provided,
            compute() will call l5a.compute(l3_cells, l4_cols) and apply
            the resulting displacement to L6a automatically. External
            'displacement' argument to compute() overrides L5a when provided.
        input_size: Length of the input SDR. Used when building default SP.
        num_minicolumns: Number of SP minicolumns.
        cells_per_col: Cells per minicolumn in TM.
        active_per_step: SP active minicolumns per step.
        sp_kwargs: Extra keyword arguments passed to SpatialPooler.
        tm_kwargs: Extra keyword arguments passed to TemporalMemory.
        seed: Random seed.
    """

    def __init__(
        self,
        encoder=None,
        spatial_pooler: Optional[SpatialPooler] = None,
        temporal_memory: Optional[TemporalMemory] = None,
        grid_layer: Optional[GridCellLayer] = None,
        displacement_layer: Optional[DisplacementLayer] = None,
        l5a_readout=None,
        input_size: int = 256,
        num_minicolumns: int = 512,
        cells_per_col: int = 4,
        active_per_step: int = 20,
        sp_kwargs: Optional[dict] = None,
        tm_kwargs: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        self.encoder = encoder
        self.l5a = l5a_readout
        self.iteration = 0

        # ── L6a: Grid Cell Layer ──────────────────────────────────────────
        if grid_layer is not None:
            self.grid_layer = grid_layer
        else:
            self.grid_layer = make_number_line_layer(
                max_value=100, num_modules=3,
                sdr_length_per_module=64, sdr_width_per_module=9,
            )

        # ── L5b: Displacement Layer ───────────────────────────────────────
        if displacement_layer is not None:
            self.displacement_layer = displacement_layer
        else:
            self.displacement_layer = make_displacement_layer_from_grid(
                self.grid_layer
            )

        # ── L4: Spatial Pooler ────────────────────────────────────────────
        if spatial_pooler is not None:
            self.sp = spatial_pooler
            self._num_minicolumns = spatial_pooler.num_minicolumns
        else:
            sp_kw = sp_kwargs or {}
            self.sp = SpatialPooler(
                input_size=input_size,
                num_minicolumns=num_minicolumns,
                active_per_step=active_per_step,
                seed=seed,
                **sp_kw,
            )
            self._num_minicolumns = num_minicolumns

        # ── L3: Temporal Memory with L6a location conditioning ────────────
        if temporal_memory is not None:
            self.tm = temporal_memory
        else:
            loc_len = self.grid_layer.total_sdr_length
            tm_kw = dict(tm_kwargs or {})
            # Only inject location_sdr_length if not already specified
            if 'location_sdr_length' not in tm_kw:
                tm_kw['location_sdr_length'] = loc_len
            self.tm = TemporalMemory(
                num_minicolumns=self._num_minicolumns,
                cells_per_col=cells_per_col,
                seed=seed,
                **tm_kw,
            )

        # ── Output state ──────────────────────────────────────────────────
        self._last_result: dict = {}

    # ── Core compute ──────────────────────────────────────────────────────

    def compute(
        self,
        sensory_input,
        displacement: Optional[float] = None,
        learn: bool = True,
    ) -> dict:
        """Run one timestep of the cortical column.

        Steps:
          1. Encode sensory_input → input_sdr  (if encoder provided)
          2. SP.compute(input_sdr) → active_minicolumns  [L4]
          3. grid_layer.get_location_sdr() → location_sdr  [L6a]
          4. TM.compute(active_minicolumns, location_sdr) → active_cells [L3]
          5a. L5a.compute(active_cells, active_minicolumns) → l5a_displacement
          5b. Apply displacement to L6a (external > L5a, both > None)

        Args:
            sensory_input: Either a raw value (encoded by self.encoder) or
                a pre-encoded boolean numpy array of length input_size.
            displacement: Optional scalar displacement to apply to the grid
                layer. Overrides L5a output when both are provided. None
                with no L5a = no movement.
            learn: Whether to update permanences and grow synapses.

        Returns:
            dict with keys:
              'input_sdr':           bool (input_size,)
              'active_minicolumns':  bool (num_minicolumns,)
              'location_sdr':        bool (total_location_sdr_length,)
              'active_cells':        bool (num_minicolumns * cells_per_col,)
              'predictive_cells':    bool (num_minicolumns * cells_per_col,)
              'anomaly_score':       float in [0, 1]
              'displacement_applied': float or None
              'l5a_displacement':    float or None (L5a output, if L5a present)
        """
        # Step 1: Encode
        if isinstance(sensory_input, np.ndarray):
            input_sdr = sensory_input.astype(bool)
        elif self.encoder is not None:
            input_sdr = self.encoder.encode(sensory_input)
        else:
            raise ValueError(
                "No encoder provided and sensory_input is not an ndarray. "
                "Either pass a pre-encoded SDR or construct the column with "
                "an encoder."
            )

        # Step 2: L4 — Spatial Pooler
        active_minicolumns = self.sp.compute(input_sdr, learn=learn)

        # Step 3: L6a — Location context for this timestep
        location_sdr = self.grid_layer.get_location_sdr()

        # Step 4: L3 — Temporal Memory with location conditioning
        active_cells = self.tm.compute(
            active_minicolumns,
            location_sdr=location_sdr,
            learn=learn,
        )

        # Step 5a: L5a — Conjunctive readout (if present)
        l5a_displacement = None
        if self.l5a is not None:
            l5a_displacement = self.l5a.compute(
                self.tm.cell_active, active_minicolumns
            )

        # Step 5b: L5b → L6a — Path integration
        # External displacement takes priority over L5a (allows training with
        # ground-truth displacement labels alongside learned L5a)
        effective_displacement = displacement if displacement is not None else l5a_displacement
        if effective_displacement is not None:
            self.displacement_layer.apply_displacement_to(
                effective_displacement, self.grid_layer
            )

        self.iteration += 1

        result = {
            'input_sdr': input_sdr,
            'active_minicolumns': active_minicolumns,
            'location_sdr': location_sdr,
            'active_cells': active_cells,
            'predictive_cells': self.tm.cell_predictive.copy(),
            'anomaly_score': self.tm.get_anomaly_score(),
            'displacement_applied': effective_displacement,
            'l5a_displacement': l5a_displacement,
        }
        self._last_result = result
        return result

    def reset(self) -> None:
        """Reset temporal state. Call between unrelated sequences.

        Resets TM cell state and L5a credit-assignment state.
        Does NOT reset SP permanences, grid phases, or L5a weights.
        """
        self.tm.reset()
        if self.l5a is not None:
            self.l5a.reset()

    def reset_position(self, position: float = 0.0) -> None:
        """Set the grid layer to a known position.

        Args:
            position: The position to move to in the reference frame.
        """
        self.grid_layer.set_position(position)

    def prime_from_location(self) -> None:
        """Initialise predictive state from the current grid position alone.

        Call this after reset() + reset_position(p) to prime the correct cells
        for position p WITHOUT requiring a prior compute() step. This allows
        the column to predict feature(p) on the very first observation after
        jumping to a known position.

        Mechanism: computes location-only segment activity using the current
        L6a location SDR (no active cells), then applies location_activation_threshold
        to set prev_predictive. The next compute() call will see these predictions
        and fire the correct cells without bursting — IF enough location synapses
        have been trained on segments at this position.

        Biological interpretation: L6a sends the location signal to basal/distal
        dendrites, producing dendritic spikes that depolarise specific cells
        without requiring any prior firing. The cell is 'ready to fire' when
        proximal sensory input arrives.

        No-op if the TM has no location_activation_threshold set (== 0).
        """
        if not self.tm.use_location or self.tm.location_activation_threshold == 0:
            return

        location_sdr = self.grid_layer.get_location_sdr()
        zeros = np.zeros(self.tm.total_cells, dtype=bool)

        # Compute location-only segment activity (cell part contributes nothing)
        _, _, seg_loc = self.tm._compute_segment_activity(zeros, location_sdr)

        # _predict_cells OR-gate: seg_loc >= location_activation_threshold
        combined_zeros = np.zeros(self.tm.total_segs, dtype=np.int32)
        self.tm.cell_predictive = self.tm._predict_cells(combined_zeros, seg_loc)

        # Expose as prev_predictive so the NEXT compute()'s phase 1 sees it
        self.tm.prev_predictive = self.tm.cell_predictive.copy()

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def anomaly_score(self) -> float:
        """Current anomaly score (fraction of active columns that burst)."""
        return self.tm.get_anomaly_score()

    @property
    def prediction_density(self) -> float:
        """Fraction of cells currently in a predictive state."""
        return self.tm.get_prediction_density()

    @property
    def active_cells(self) -> np.ndarray:
        """Boolean array of currently active cells."""
        return self.tm.cell_active.copy()

    @property
    def predictive_cells(self) -> np.ndarray:
        """Boolean array of currently predictive cells."""
        return self.tm.cell_predictive.copy()

    @property
    def location(self) -> np.ndarray:
        """Current grid cell phases (position in reference frame)."""
        return self.grid_layer.get_phases()

    @property
    def location_sdr(self) -> np.ndarray:
        """Current location SDR from the grid layer."""
        return self.grid_layer.get_location_sdr()

    def get_stats(self) -> dict:
        """Summary statistics for diagnostics."""
        sp_counts = self.sp.get_connected_counts()
        seg_stats = self.tm.get_segment_stats()
        odc_stats = self.sp.get_overlap_duty_cycle_stats()
        return {
            'iteration': self.iteration,
            'anomaly_score': self.anomaly_score,
            'prediction_density': self.prediction_density,
            'sp_entropy': self.sp.get_entropy(),
            'sp_max_entropy': self.sp.get_max_entropy(),
            'sp_min_connections': int(sp_counts.min()),
            'sp_mean_connections': float(sp_counts.mean()),
            'sp_overlap_dc_underperforming':
                odc_stats['pct_underperforming'],
            **{f'tm_{k}': v for k, v in seg_stats.items()},
            'grid_phases': self.location.tolist(),
        }

    def __repr__(self) -> str:
        n_cols = self._num_minicolumns
        cells = self.tm.total_cells
        loc_len = self.grid_layer.total_sdr_length
        has_enc = self.encoder is not None
        return (
            f"CorticalColumn("
            f"minicolumns={n_cols}, cells={cells}, "
            f"location_sdr={loc_len}, encoder={has_enc})"
        )