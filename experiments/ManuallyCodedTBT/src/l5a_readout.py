"""
l5a_readout.py — Layer 5a Conjunctive Displacement Readout (SDR-native).

Implements L5a as a bank of Spatial Poolers, one per L6a/L5b displacement
cell module. Every signal is an SDR — no scalar floats cross layer
boundaries.

Architecture (Hawkins 2019 + revised design):

  Input:  L3 active cells (bool, total_l3_cells)
            — encodes the FULL prior sequence context via TM higher-order
              memory. Sufficient for arbitrary operator semantics:
                addition:            context at 'b' encodes [a, +, b]
                multiplication:      context at 'b' encodes [a, ×, b]
                chained multiply:    context at 'c' encodes [a, ×, b, ×, c]
              The L3 cell state is the variable-length accumulator.
          L4 is NOT needed: L3's higher-order memory already distinguishes
            feature 'b' in different operator contexts.

  Output: displacement SDR (bool, n_modules × sdr_length_per_module)
            — concatenation of each module's phase SDR.
            — same format as DisplacementLayer.get_displacement_sdr().
            — applied to L6a via DisplacementLayer.apply_from_sdr().

  Structure: one SpatialPooler per displacement cell module.
    SP i: input_size=total_l3_cells,
           num_minicolumns=sdr_length_per_module,
           active_per_step=sdr_width_per_module.
    Output of SP i = displacement module i's phase SDR.

  The SP substrate is identical to L4 — the whole system now shares one
  computational substrate. No dense weight matrices, no gradient descent.

Variable-length context:
  L3 active cells at the operand step encode the entire prior context:
    'b' in [3, ×, 4, ×, 5]  ≠  'b' in [3, +, 5]  ≠  'b' in [3, ×, 5]
  L5a SPs learn different associations for each. No architectural change
  is needed to support longer sequences — the TM handles it.

Learning:

  Supervised (use_supervised=True, default):
    For each step, call learn_supervised(l3_active, true_displacement):
      Encodes true_displacement % λᵢ as the target SDR for module i,
      then calls SP.learn_with_target(l3_active, target_sdr).
      The SP Hebbian permanence update grows synapses from active L3 cells
      to the correct output bits.

    Call with true_displacement=0.0 for non-operator steps (observing 'a',
    '+', '=', result). This trains the SPs to output a zero-phase (identity)
    displacement for those contexts, preventing spurious grid movement.

  Anomaly-driven Hebbian (use_supervised=False):
    compute() stores the L3 pattern. At the result step, if anomaly is low,
    reinforce by calling learn_from_anomaly(anomaly_score).

Usage:
    l5a = L5aReadout.from_displacement_layer(
              displ, total_l3_cells=col.tm.total_cells, seed=42)
    col.l5a = l5a

    # Training — for each step in sequence [a, +, b, =, result]:
    col.compute(encode_number(a), displacement=None, learn=True)
    l5a.learn_supervised(col.tm.cell_active, 0.0)          # not operator step

    col.compute(encode_symbol('+'), displacement=None, learn=True)
    l5a.learn_supervised(col.tm.cell_active, 0.0)          # not operator step

    col.compute(encode_number(b), displacement=float(result), learn=True)
    l5a.learn_supervised(col.tm.cell_active, float(result)) # operator step

    col.compute(encode_symbol('='), displacement=None, learn=True)
    l5a.learn_supervised(col.tm.cell_active, 0.0)

    col.compute(encode_number(result), displacement=None, learn=True)
    l5a.learn_supervised(col.tm.cell_active, 0.0)

    # Inference — L5a runs inside col.compute() automatically.
    # No external displacement needed; L5a's SDR output is applied via
    # DisplacementLayer.apply_from_sdr().
"""

import numpy as np
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sdr import encode_periodic, concatenate
from spatial_pooler import SpatialPooler


class L5aReadout:
    """L5a as a bank of Spatial Poolers mapping L3 cells → displacement SDR.

    Args:
        total_l3_cells:        TM total_cells — input size for each SP.
        n_modules:             Number of displacement / grid cell modules.
        sdr_length_per_module: Phase SDR length per module.
        sdr_width_per_module:  Phase SDR active bits per module.
        periods:               Grid/displacement module periods (λᵢ).
        sp_permanence_threshold: Synapse connection threshold.
        sp_permanence_inc:     Permanence increment for active synapses.
        sp_permanence_dec:     Permanence decrement for inactive synapses.
        sp_initial_synapses:   Initial potential synapses per output column.
            L3 inputs are sparser (~1%) than sensory inputs (~2%), so more
            initial connections are needed for reliable overlap scores.
        use_supervised:        True = forced-winner Hebbian (learn_supervised).
                               False = anomaly-gated Hebbian (learn_from_anomaly).
        seed:                  Random seed.
    """

    def __init__(
        self,
        total_l3_cells: int,
        n_modules: int,
        sdr_length_per_module: int,
        sdr_width_per_module: int,
        periods: List[float],
        sp_permanence_threshold: float = 0.3,
        sp_permanence_inc: float = 0.08,
        sp_permanence_dec: float = 0.01,
        use_supervised: bool = True,
        seed: Optional[int] = None,
    ):
        assert len(periods) == n_modules, (
            f"len(periods)={len(periods)} != n_modules={n_modules}"
        )
        self.total_l3_cells = total_l3_cells
        self.n_modules = n_modules
        self.sdr_length_per_module = sdr_length_per_module
        self.sdr_width_per_module = sdr_width_per_module
        self.periods = list(periods)
        self.use_supervised = use_supervised
        self.total_sdr_length = n_modules * sdr_length_per_module

        rng = np.random.default_rng(seed)

        # potential_pct=1.0: each output minicolumn can form synapses to ANY
        # L3 cell. Full connectivity is correct here because L3 inputs are
        # very sparse (~1%), so the SP needs access to all inputs to reliably
        # grow connections to the small set of active cells.
        # boost_strength=0: boosting distorts duty cycles when forced-winner
        # training sees each context at different frequencies.
        self.sps: List[SpatialPooler] = [
            SpatialPooler(
                input_size=total_l3_cells,
                num_minicolumns=sdr_length_per_module,
                active_per_step=sdr_width_per_module,
                potential_pct=1.0,
                permanence_threshold=sp_permanence_threshold,
                permanence_inc=sp_permanence_inc,
                permanence_dec=sp_permanence_dec,
                boost_strength=0.0,
                seed=int(rng.integers(0, 2**31)),
            )
            for _ in range(n_modules)
        ]

        # Stored L3 state for anomaly-driven Hebbian mode
        self._stored_l3: Optional[np.ndarray] = None
        self._has_pending: bool = False

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_displacement_layer(
        cls,
        displacement_layer,
        total_l3_cells: int,
        use_supervised: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "L5aReadout":
        """Create L5aReadout paired to an existing DisplacementLayer."""
        periods = [m.period for m in displacement_layer.modules]
        return cls(
            total_l3_cells=total_l3_cells,
            n_modules=displacement_layer.num_modules,
            sdr_length_per_module=displacement_layer.sdr_length_per_module,
            sdr_width_per_module=displacement_layer.sdr_width_per_module,
            periods=periods,
            use_supervised=use_supervised,
            seed=seed,
            **kwargs,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def compute(
        self,
        l3_active_cells: np.ndarray,
        learn: bool = False,
    ) -> np.ndarray:
        """Map L3 active cells → displacement SDR.

        Each SP maps l3_active_cells → one module's phase SDR. Results
        are concatenated into a single displacement SDR in DisplacementLayer
        format, ready for DisplacementLayer.apply_from_sdr().

        Args:
            l3_active_cells: Bool (total_l3_cells,) — TM cell_active.
            learn:           Whether SPs run their normal Hebbian update
                             (usually False; use learn_supervised instead).

        Returns:
            Bool (total_sdr_length,) — full displacement SDR.
        """
        module_sdrs = [
            sp.compute(l3_active_cells, learn=learn)
            for sp in self.sps
        ]

        if not self.use_supervised:
            self._stored_l3 = l3_active_cells.copy()
            self._has_pending = True

        return concatenate(module_sdrs)

    # ── Learning ──────────────────────────────────────────────────────────────

    def learn_supervised(
        self,
        l3_active_cells: np.ndarray,
        true_displacement: float,
    ) -> None:
        """Forced-winner Hebbian update toward the correct displacement SDR.

        For each module i, encodes true_displacement % λᵢ as the target
        phase SDR, then calls SP.learn_with_target(l3_active, target_sdr).
        The SP permanently associates the L3 pattern with the correct bits.

        Call at EVERY timestep:
          - Operator steps (e.g. 'b' in [a, +, b]):  true_displacement = a+b
          - All other steps:                          true_displacement = 0.0
            (zero phase = identity displacement — no grid movement)

        Args:
            l3_active_cells:  Bool (total_l3_cells,).
            true_displacement: Correct displacement for this step.
        """
        for sp, period in zip(self.sps, self.periods):
            phase = true_displacement % period
            target_sdr = encode_periodic(
                phase,
                self.sdr_length_per_module,
                self.sdr_width_per_module,
                0.0,
                period,
            )
            sp.learn_with_target(l3_active_cells, target_sdr)

    def learn_from_anomaly(
        self,
        anomaly_score: float,
        reinforce_threshold: float = 0.3,
    ) -> None:
        """Anomaly-gated Hebbian reinforcement (unsupervised mode).

        Uses the L3 state stored during the most recent compute() call.
        Reinforces only when anomaly < threshold (displacement was correct).

        Args:
            anomaly_score:       TM anomaly at the result step.
            reinforce_threshold: Anomaly below which the output is reinforced.
        """
        if not self._has_pending or self._stored_l3 is None:
            return
        if anomaly_score < reinforce_threshold:
            for sp in self.sps:
                sp.compute(self._stored_l3, learn=True)
        self._has_pending = False
        self._stored_l3 = None

    def reset(self) -> None:
        """Clear pending Hebbian state between sequences."""
        self._has_pending = False
        self._stored_l3 = None

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def decode_displacement(self, displacement_sdr: np.ndarray) -> List[float]:
        """Decode a displacement SDR to per-module phase values.

        Returns the decoded phase for each module (not a single scalar,
        since the CRT reconstruction is only valid over the full module set).

        Args:
            displacement_sdr: Bool (total_sdr_length,).

        Returns:
            List of floats, one per module: [phase_mod_0, phase_mod_1, ...]
        """
        from sdr import decode_periodic as _dp
        L = self.sdr_length_per_module
        return [
            _dp(displacement_sdr[i * L: (i + 1) * L], 0.0, period)
            for i, period in enumerate(self.periods)
        ]

    def get_stats(self) -> dict:
        return {
            f"module_{i}_entropy": float(sp.get_entropy())
            for i, sp in enumerate(self.sps)
        }

    def __repr__(self) -> str:
        return (
            f"L5aReadout("
            f"l3_cells={self.total_l3_cells}, "
            f"n_modules={self.n_modules}, "
            f"sdr={self.sdr_length_per_module}×{self.sdr_width_per_module}, "
            f"supervised={self.use_supervised})"
        )