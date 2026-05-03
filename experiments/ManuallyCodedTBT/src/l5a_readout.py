"""
l5a_readout.py — Layer 5a Conjunctive Displacement Readout.

Implements the L5a (intratelencephalic IT neuron) conjunctive readout that
maps (L3 active cells, L4 active minicolumns) → scalar displacement.

Architecture (Hawkins 2019 + pre-compaction design):
  L3 provides operator context: "we are in an after-+ state."
  L4 provides current feature magnitude: "the number is 5."
  Neither input alone is sufficient. L5a integrates both and outputs a
  scalar displacement d̂, which is passed to L5b (DisplacementLayer) to
  update the L6a grid cell reference frame.

  L3 → L5a (confirmed: IT neurons receive from lower L2/3)
  L4 → L5a (barrel cortex literature: L5 receives from multiple laminar sources)
  L5a → L5b (local: IT neurons drive ET neurons within the column)
  L5b → L6a (bilateral: displacement shifts each module's phase by d mod λᵢ)

  L3 does NOT directly drive L5b. This is confirmed by Hawkins 2019.

Weight matrix W:
  Shape: (total_L3_cells + num_minicolumns,)
  Represents one weight per presynaptic unit. The displacement output is
  the dot product of W with the concatenated (L3 active ∥ L4 active) vector,
  scaled by a gain factor.

  Because SDR inputs are binary and sparse, the dot product is equivalent to
  summing the weights of active units. Only active cells/columns contribute.

Learning rule (Hebbian, prediction-error-driven):
  At step b (the operand step — when L4 shows the number b):
    1. L5a computes d̂ = W · (L3 cells ∥ L4 cols) * gain
    2. d̂ is applied to L6a via DisplacementLayer
    3. The (L3, L4) state is stored for credit assignment

  At the result step (after "="):
    4. Anomaly score at this step is the teaching signal.
       anomaly = 0 → prediction correct → reinforce: W += lr * (target - d̂) * active
       anomaly > 0 → prediction wrong   → weaken:    W -= lr * anomaly * active * decay

  'target' at reinforcement time is the true displacement (a+b - a = b),
  which is available from the training data. For the biologically-plausible
  version, target is not needed: we use the sign and magnitude of anomaly
  as a surrogate error (positive anomaly → displacement was wrong → weaken).

Two modes:
  SUPERVISED (use_supervised=True):
    At each training step, the true displacement is provided. W is updated
    by a simple delta rule: ΔW = lr * (target - d̂) * active_input.
    Fast convergence. Used to validate the architecture before tackling the
    Hebbian version.

  HEBBIAN (use_supervised=False):
    The anomaly at the result step is the teaching signal. No true displacement
    label is provided at training time. ΔW = lr * (1 - anomaly) * d̂ * active_input
    (reinforce when anomaly is low, weaken when high). Biologically plausible.

Usage:
    from l5a_readout import L5aReadout

    l5a = L5aReadout(
        num_l3_cells=tm.total_cells,
        num_minicolumns=sp.num_minicolumns,
        learning_rate=0.01,
        use_supervised=True,
    )

    # At operand step: compute displacement and store state
    d_hat = l5a.compute(tm.cell_active, sp_active_cols)

    # At result step: learn from anomaly (Hebbian) or true target (supervised)
    l5a.learn(anomaly_score=col.anomaly_score)           # Hebbian
    l5a.learn(true_displacement=b, anomaly_score=anom)   # Supervised
"""

import numpy as np
from typing import Optional


class L5aReadout:
    """Layer 5a conjunctive readout: (L3, L4) → displacement scalar.

    Args:
        num_l3_cells: Total cells in the L3 temporal memory (total_cells).
        num_minicolumns: Number of L4 spatial pooler minicolumns.
        learning_rate: Hebbian/supervised weight update step size.
        weight_decay: L2 regularization coefficient (keeps weights near 0
            for neutral context inputs).
        gain: Scalar applied to the dot-product output. Useful for
            scaling the weight magnitudes to the expected displacement range.
        use_supervised: If True, learn() requires true_displacement and uses
            a delta rule. If False, uses Hebbian learning from anomaly signal.
        seed: Random seed for weight initialisation.
    """

    def __init__(
        self,
        num_l3_cells: int,
        num_minicolumns: int,
        learning_rate: float = 0.005,
        weight_decay: float = 0.0001,
        gain: float = 1.0,
        use_supervised: bool = True,
        seed: Optional[int] = None,
    ):
        self.num_l3_cells = num_l3_cells
        self.num_minicolumns = num_minicolumns
        self.input_size = num_l3_cells + num_minicolumns
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gain = gain
        self.use_supervised = use_supervised

        rng = np.random.default_rng(seed)
        # Small random init — near zero so neutral context outputs near zero
        self.W = rng.normal(0.0, 0.01, size=self.input_size).astype(np.float32)

        # Stored state for credit assignment (prev step's active input + output)
        self._prev_active: Optional[np.ndarray] = None
        self._prev_d_hat: float = 0.0
        self._has_pending: bool = False   # True if a displacement was applied last step

    # ── Forward ──────────────────────────────────────────────────────────────

    def compute(
        self,
        l3_active_cells: np.ndarray,
        l4_active_minicolumns: np.ndarray,
    ) -> float:
        """Compute displacement estimate from current L3 and L4 state.

        Stores the active input and output for the next learn() call.

        Args:
            l3_active_cells: Bool (num_l3_cells,) — TM cell_active.
            l4_active_minicolumns: Bool (num_minicolumns,) — SP output.

        Returns:
            Scalar displacement estimate d̂. Pass this to
            DisplacementLayer.apply_displacement_to(d_hat, grid_layer).
        """
        active = self._concat(l3_active_cells, l4_active_minicolumns)
        d_hat = float(np.dot(self.W, active) * self.gain)

        # Store for credit assignment
        self._prev_active = active
        self._prev_d_hat = d_hat
        self._has_pending = True

        return d_hat

    # ── Learning ─────────────────────────────────────────────────────────────

    def learn(
        self,
        anomaly_score: float,
        true_displacement: Optional[float] = None,
    ) -> None:
        """Update weights based on the prediction outcome.

        Call this at the result step, after observing the outcome of the
        displacement that was applied at the previous compute() call.

        Args:
            anomaly_score: Anomaly at the result step (0=correct, 1=burst).
                Used as the primary error signal in both modes.
            true_displacement: Optional. If use_supervised=True and this is
                provided, use the delta rule directly. If None in supervised
                mode, falls back to Hebbian.
        """
        if not self._has_pending or self._prev_active is None:
            return

        active = self._prev_active
        d_hat  = self._prev_d_hat

        if self.use_supervised and true_displacement is not None:
            # Delta rule: ΔW = lr * (target - d̂) * active_input
            error = true_displacement - d_hat / self.gain
            delta = self.learning_rate * error * active
        else:
            # Hebbian: reinforce when anomaly low, weaken when high
            # ΔW = lr * (1 - anomaly) * d̂/gain * active  (positive = reinforce)
            # ΔW = -lr * anomaly * d̂/gain * active        (combined)
            # Simplified: ΔW = lr * (1 - 2*anomaly) * d̂/gain * active
            signal = (1.0 - 2.0 * anomaly_score) * (d_hat / self.gain if self.gain != 0 else d_hat)
            delta = self.learning_rate * signal * active

        # Weight decay (L2 regularisation — pulls unused weights toward 0)
        self.W += delta
        self.W *= (1.0 - self.weight_decay)

        self._has_pending = False

    def skip(self) -> None:
        """Call when a step does NOT produce a displacement to learn from.

        Clears pending state without updating weights. Use when the current
        timestep is a non-operator step (observing a number or symbol that
        should not trigger a displacement).
        """
        self._has_pending = False
        self._prev_active = None
        self._prev_d_hat = 0.0

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def predict(
        self,
        l3_active_cells: np.ndarray,
        l4_active_minicolumns: np.ndarray,
    ) -> float:
        """Compute displacement estimate without updating stored state.

        Use for evaluation/inspection only — does not affect learning.
        """
        active = self._concat(l3_active_cells, l4_active_minicolumns)
        return float(np.dot(self.W, active) * self.gain)

    def get_stats(self) -> dict:
        """Summary statistics for monitoring."""
        return {
            "w_mean": float(self.W.mean()),
            "w_std": float(self.W.std()),
            "w_max": float(self.W.max()),
            "w_min": float(self.W.min()),
            "w_nonzero": int((np.abs(self.W) > 1e-6).sum()),
            "pending": self._has_pending,
        }

    def reset(self) -> None:
        """Clear pending credit-assignment state between sequences."""
        self._has_pending = False
        self._prev_active = None
        self._prev_d_hat = 0.0

    # ── Internal ─────────────────────────────────────────────────────────────

    def _concat(
        self,
        l3_active_cells: np.ndarray,
        l4_active_minicolumns: np.ndarray,
    ) -> np.ndarray:
        """Concatenate L3 and L4 active patterns into a single float32 vector."""
        return np.concatenate([
            l3_active_cells.astype(np.float32),
            l4_active_minicolumns.astype(np.float32),
        ])
