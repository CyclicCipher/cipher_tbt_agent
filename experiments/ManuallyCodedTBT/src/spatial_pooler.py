"""
spatial_pooler.py — Spatial Pooling algorithm.

Takes an input SDR of arbitrary size and produces a fixed-size output SDR
of active minicolumns. This implements the core feedforward mechanism of
Layer 4 in the TBT cortical column.

Layer 4 role in TBT:
  Layer 4 is the primary recipient of thalamic/encoder sensory input. It
  is also where feature-location binding begins: L6a sends a location
  prediction to L4, and L4's output to Layer 3 carries the combined
  feature-location representation. This spatial pooler implements L4's
  feedforward sensory processing (encoder → sparse minicolumn SDR).

  NOTE: The L6a → L4 modulatory connection (location prediction modulating
  spatial pooling) is NOT yet implemented. The spatial pooler currently
  receives only encoder output. Adding the location modulation pathway is
  a future step, once the basic column is assembled and tested.

The spatial pooler:
  - Maintains fixed sparsity in the output regardless of input density
  - Preserves semantic overlap from the input (similar inputs → similar outputs)
  - Learns stable representations through Hebbian permanence updates
  - Uses boosting to ensure all minicolumns participate in representations

Data Oriented Design: no Minicolumn class. The spatial pooler owns flat
arrays indexed by minicolumn index. Each minicolumn's state is a row
in a matrix or an element in a vector.

Corrections vs. original implementation (sourced from BAMI v0.56):
  1. Boost target is the mean active duty cycle of neighbors (global mean
     under global inhibition), not a fixed rate derived from active_per_step.
  2. A second boosting mechanism is now implemented: overlap duty cycle
     tracking with permanence scaling for columns that can't even reach
     stimulus threshold. This prevents column death.
  3. stimulus_threshold is now an explicit parameter (BAMI: 0-5, default 0).
  4. Default parameters updated to BAMI recommendations:
       permanence_threshold: 0.5 -> 0.2
       permanence_inc:       0.05 -> 0.03
       permanence_dec:       0.008 -> 0.015

Usage:
    from spatial_pooler import SpatialPooler
    import numpy as np

    sp = SpatialPooler(
        input_size=256,
        num_minicolumns=2048,
        active_per_step=40,
    )

    output = sp.compute(input_sdr, learn=True)
"""

import numpy as np
from typing import Optional


class SpatialPooler:
    """Spatial Pooler with Hebbian learning, dual boosting, and global inhibition.

    Args:
        input_size: Length of the input SDR.
        num_minicolumns: Number of minicolumns in the output.
            BAMI recommends a minimum of 2048.
        active_per_step: Number of minicolumns active per timestep (k).
            BAMI recommends ~2% of num_minicolumns (e.g. 40 for 2048).
        potential_pct: Fraction of input bits each minicolumn can potentially
            connect to. 1.0 = global. Set so that at least 15-20 inputs are
            connected on average after initialization.
        connected_pct: Initial fraction of potential synapses that start
            above permanence_threshold.
        permanence_threshold: Permanence value above which a synapse is
            considered connected. BAMI default: 0.2.
        permanence_inc: Permanence increase for active synapses on winning
            columns. BAMI default: 0.03.
        permanence_dec: Permanence decrease for inactive synapses on winning
            columns. BAMI default: 0.015.
        boost_strength: Controls strength of exponential boost function.
            0.0 = no boosting.
        duty_cycle_period: Window (in timesteps) for duty cycle moving averages.
        min_pct_overlap_duty_cycle: A column's overlap duty cycle must exceed
            this fraction of the maximum overlap duty cycle among its neighbors,
            or its permanences are scaled up. BAMI default: 0.001.
        stimulus_threshold: A column must have at least this many active
            connected synapses to be eligible for activation. Prevents noise
            from activating columns. BAMI: 0-5, not sensitive, default 0.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        input_size: int,
        num_minicolumns: int,
        active_per_step: int,
        potential_pct: float = 1.0,
        connected_pct: float = 0.5,
        permanence_threshold: float = 0.2,
        permanence_inc: float = 0.03,
        permanence_dec: float = 0.015,
        boost_strength: float = 3.0,
        duty_cycle_period: int = 1000,
        min_pct_overlap_duty_cycle: float = 0.001,
        stimulus_threshold: int = 0,
        seed: Optional[int] = None,
    ):
        self.input_size = input_size
        self.num_minicolumns = num_minicolumns
        self.active_per_step = active_per_step
        self.potential_pct = potential_pct
        self.permanence_threshold = permanence_threshold
        self.permanence_inc = permanence_inc
        self.permanence_dec = permanence_dec
        self.boost_strength = boost_strength
        self.duty_cycle_period = duty_cycle_period
        self.min_pct_overlap_duty_cycle = min_pct_overlap_duty_cycle
        self.stimulus_threshold = stimulus_threshold

        self.rng = np.random.default_rng(seed)
        self.iteration = 0

        # ── Potential pool ────────────────────────────────────────────────
        # potential_pool[i, j] = True if minicolumn i CAN form a synapse
        # to input bit j. Fixed at initialization.
        self.potential_pool = np.zeros(
            (num_minicolumns, input_size), dtype=bool
        )
        num_potential = max(1, int(input_size * potential_pct))
        for i in range(num_minicolumns):
            indices = self.rng.choice(input_size, size=num_potential, replace=False)
            self.potential_pool[i, indices] = True

        # ── Permanence matrix ─────────────────────────────────────────────
        # permanence[i, j] = synapse permanence from minicolumn i to input j.
        # Only meaningful where potential_pool[i, j] is True.
        # Initialized with a normal distribution centered on the threshold
        # so roughly connected_pct of potential synapses start connected.
        self.permanence = np.zeros((num_minicolumns, input_size), dtype=np.float32)
        from scipy.stats import norm as scipy_norm
        shift = scipy_norm.ppf(connected_pct) * 0.05
        for i in range(num_minicolumns):
            pot_indices = np.where(self.potential_pool[i])[0]
            values = self.rng.normal(
                loc=permanence_threshold,
                scale=0.05,
                size=len(pot_indices),
            )
            values += shift
            values = np.clip(values, 0.0, 1.0)
            self.permanence[i, pot_indices] = values.astype(np.float32)

        # ── Boosting state ────────────────────────────────────────────────
        # active_duty_cycle[i]: running average of how often column i wins
        # after inhibition.
        # overlap_duty_cycle[i]: running average of how often column i has
        # raw overlap >= stimulus_threshold (before boosting). Tracks whether
        # the column's synapses are connected to anything useful at all.
        target = active_per_step / num_minicolumns
        self.active_duty_cycle = np.full(num_minicolumns, target, dtype=np.float32)
        self.overlap_duty_cycle = np.full(num_minicolumns, target, dtype=np.float32)
        self.boost_factors = np.ones(num_minicolumns, dtype=np.float32)

        # ── Output state ──────────────────────────────────────────────────
        self.active_minicolumns = np.zeros(num_minicolumns, dtype=bool)

    # ── Core computation ──────────────────────────────────────────────────

    def _connected_synapses(self) -> np.ndarray:
        """Boolean matrix: True where permanence >= threshold AND in potential pool."""
        return (self.permanence >= self.permanence_threshold) & self.potential_pool

    def _compute_overlap(self, input_sdr: np.ndarray) -> np.ndarray:
        """Raw overlap: connected synapses active in input, per minicolumn.

        Returns float array of shape (num_minicolumns,).
        """
        connected = self._connected_synapses()
        return connected.astype(np.float32) @ input_sdr.astype(np.float32)

    def _apply_boosting(self, overlaps: np.ndarray) -> np.ndarray:
        """Multiply raw overlaps by boost factors (element-wise)."""
        return overlaps * self.boost_factors

    def _inhibit(self, raw_overlaps: np.ndarray,
                 boosted_overlaps: np.ndarray) -> np.ndarray:
        """Global inhibition: top-k by boosted overlap, subject to stimulus_threshold.

        BAMI phase 3: a column is only eligible if its raw overlap (before
        boosting) exceeds stimulus_threshold. This prevents noise columns
        from winning via boost alone.

        Ties broken by small random perturbation.

        Returns boolean array of shape (num_minicolumns,).
        """
        noisy = boosted_overlaps + self.rng.uniform(
            0, 1e-6, size=self.num_minicolumns
        )

        active = np.zeros(self.num_minicolumns, dtype=bool)
        if self.active_per_step >= self.num_minicolumns:
            active[:] = True
        else:
            top_k = np.argpartition(noisy, -self.active_per_step)[
                -self.active_per_step:
            ]
            active[top_k] = True

        # Enforce stimulus_threshold on raw overlaps (BAMI phase 3, line 8)
        active &= raw_overlaps > self.stimulus_threshold

        return active

    def _learn(self, input_sdr: np.ndarray, active: np.ndarray) -> None:
        """Hebbian permanence update for winning columns only.

        Vectorised: operates on all active columns at once using masked
        matrix arithmetic rather than a Python loop per column.
        """
        active_idx = np.where(active)[0]
        if len(active_idx) == 0:
            return

        # Subset permanences and potential pool for active columns only
        perm = self.permanence[active_idx]         # (n_active, input_size)
        pot  = self.potential_pool[active_idx]     # (n_active, input_size)

        inc_mask = pot &  input_sdr   # broadcast over rows
        dec_mask = pot & ~input_sdr

        perm[inc_mask] += self.permanence_inc
        perm[dec_mask] -= self.permanence_dec

        np.clip(perm, 0.0, 1.0, out=perm)
        self.permanence[active_idx] = perm

    def _update_duty_cycles(self, active: np.ndarray,
                            raw_overlaps: np.ndarray) -> None:
        """Update both duty cycle moving averages.

        active_duty_cycle: fraction of timesteps each column wins after
            inhibition.
        overlap_duty_cycle: fraction of timesteps each column has raw
            overlap > stimulus_threshold. Tracks whether columns have any
            useful connections at all, independent of whether they win.
        """
        period = min(self.iteration + 1, self.duty_cycle_period)
        alpha = 1.0 / period

        self.active_duty_cycle = (
            (1.0 - alpha) * self.active_duty_cycle
            + alpha * active.astype(np.float32)
        )

        had_overlap = (raw_overlaps > self.stimulus_threshold).astype(np.float32)
        self.overlap_duty_cycle = (
            (1.0 - alpha) * self.overlap_duty_cycle
            + alpha * had_overlap
        )

    def _update_boost_factors(self) -> None:
        """Recompute boost factors from active duty cycles.

        BAMI fix: compares each column to the mean active duty cycle of its
        neighbors. Under global inhibition, neighbors = all columns, so the
        reference is the global mean — a running adaptive target rather than
        a fixed rate tied to active_per_step / num_minicolumns.

        boost(c) = exp(boost_strength * (mean_duty_cycle - duty_cycle(c)))

        Columns below the mean are boosted above 1.
        Columns above the mean are suppressed below 1.
        """
        mean_duty = self.active_duty_cycle.mean()
        self.boost_factors = np.exp(
            self.boost_strength * (mean_duty - self.active_duty_cycle)
        ).astype(np.float32)

    def _apply_permanence_boost(self) -> None:
        """Second boosting mechanism: scale up permanences for dying columns.

        BAMI phase 4, lines 23-25: if a column's overlap duty cycle falls
        below min_pct_overlap_duty_cycle * max(overlap_duty_cycle of
        neighbors), increase all its potential permanences by
        0.1 * permanence_threshold.

        Under global inhibition the reference is the global max overlap duty
        cycle. This rescues columns whose synapses have decayed to the point
        where they cannot reach stimulus_threshold on any input — they would
        never recover via Hebbian learning alone because they never win.
        """
        min_duty = self.min_pct_overlap_duty_cycle * self.overlap_duty_cycle.max()
        underperforming = self.overlap_duty_cycle < min_duty

        if underperforming.any():
            increment = 0.1 * self.permanence_threshold
            self.permanence[underperforming] = np.where(
                self.potential_pool[underperforming],
                self.permanence[underperforming] + increment,
                self.permanence[underperforming],
            )
            np.clip(self.permanence, 0.0, 1.0, out=self.permanence)

    def learn_with_target(
        self, input_sdr: np.ndarray, target_output: np.ndarray
    ) -> None:
        """Hebbian update with an externally specified winner set.

        Treats target_output as the winning minicolumns and runs the
        normal permanence update. Used by L5a during supervised training:
        the correct displacement SDR bits are the forced winners, and the
        SP grows synapses from the L3 context pattern to those bits.

        Over many training steps, the SP learns to produce the correct
        output bits when presented with the corresponding L3 pattern,
        without needing any gradient or external error signal at the
        synapse level — the target is specified as the winner set.

        Args:
            input_sdr:     Bool (input_size,)  — the presynaptic pattern.
            target_output: Bool (num_minicolumns,) — forced winner columns.
        """
        active = target_output.astype(bool)
        self._learn(input_sdr, active)
        # Update duty cycles so boosting reflects actual usage.
        # Pass raw_overlaps=None → duty cycle uses zeros (no ODC update).
        # Active duty cycle is updated normally.
        self._update_duty_cycles(active, raw_overlaps=np.zeros(
            self.num_minicolumns, dtype=np.float32))

    def compute(self, input_sdr: np.ndarray, learn: bool = True) -> np.ndarray:
        """Run one timestep of the spatial pooler.

        Phase 2 (BAMI): Compute raw overlaps, apply boost.
        Phase 3 (BAMI): Inhibition — top-k by boosted overlap, filtered by
                         stimulus_threshold on raw overlap.
        Phase 4 (BAMI): Hebbian learning, duty cycle updates, boost factor
                         update, permanence boost for dying columns.

        Args:
            input_sdr: Boolean array of shape (input_size,).
            learn: Whether to update permanences and duty cycles.

        Returns:
            Boolean array of shape (num_minicolumns,), True for active.
        """
        assert len(input_sdr) == self.input_size, (
            f"Input size mismatch: expected {self.input_size}, got {len(input_sdr)}"
        )

        # Phase 2 — overlap
        raw_overlaps = self._compute_overlap(input_sdr)
        boosted_overlaps = self._apply_boosting(raw_overlaps)

        # Phase 3 — inhibition
        active = self._inhibit(raw_overlaps, boosted_overlaps)

        # Phase 4 — learning
        if learn:
            self._learn(input_sdr, active)
            self.iteration += 1
            self._update_duty_cycles(active, raw_overlaps)
            self._update_boost_factors()
            self._apply_permanence_boost()

        self.active_minicolumns = active
        return active.copy()

    # ── Diagnostics ───────────────────────────────────────────────────────

    def get_sparsity(self) -> float:
        """Fraction of minicolumns currently active."""
        return float(self.active_minicolumns.sum() / self.num_minicolumns)

    def get_connected_counts(self) -> np.ndarray:
        """Number of connected synapses per minicolumn."""
        return self._connected_synapses().sum(axis=1)

    def get_entropy(self) -> float:
        """Entropy of active duty cycle distribution.

        Higher = minicolumns used more uniformly.
        Max = log2(num_minicolumns).
        """
        dc = self.active_duty_cycle.copy()
        dc = dc[dc > 0]
        dc /= dc.sum()
        return float(-np.sum(dc * np.log2(dc)))

    def get_max_entropy(self) -> float:
        """Maximum possible entropy (uniform duty cycle distribution)."""
        return float(np.log2(self.num_minicolumns))

    def get_overlap_duty_cycle_stats(self) -> dict:
        """Summary statistics for the overlap duty cycle.

        Useful for diagnosing column death: if min is near zero and a large
        fraction of columns are below the minimum threshold, permanence
        boosting is working hard and the system may be under-connected
        relative to the input sparsity.
        """
        odc = self.overlap_duty_cycle
        min_duty = self.min_pct_overlap_duty_cycle * odc.max()
        return {
            "mean": float(odc.mean()),
            "min": float(odc.min()),
            "max": float(odc.max()),
            "pct_underperforming": float((odc < min_duty).mean()),
        }