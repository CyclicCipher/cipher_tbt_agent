"""
displacement_layer.py — Displacement Cell Modules (Layer 5b).

Implements the L5b displacement cell module system that mirrors L6a's grid
cell module system. Each L6a grid cell module is paired with one L5b
displacement cell module using the same period.

Theory:
  A displacement d is encoded modularly across all modules as:
      (d mod λ₁, d mod λ₂, ..., d mod λₖ)

  This is the same residue number system as L6a position encoding, but
  representing differences between positions rather than positions themselves.
  A single module is ambiguous (cannot uniquely represent a displacement);
  multiple modules together uniquely represent a displacement — same logic
  as grid cells.

  Applying displacement d to a GridCellLayer updates each module's phase:
      φᵢ ← (φᵢ + d) mod λᵢ

  This is path integration — the same `grid_layer.integrate(d)` call that
  our grid_cells.py already implements. The displacement layer's job is to
  encode the displacement as an SDR and apply it.

Two roles (per Hawkins 2019):
  1. Movement: updating L6a reference frame when the sensor physically moves.
  2. Composition: encoding the spatial relationship between two separate
     object reference frames (e.g., where a logo sits on a cup).

Source of displacement signal:
  Does not have to come from this column. For the addition test, "+"
  provides a displacement that can be injected externally — equivalent to
  a separate "operator" column sending its L5b signal to this column's L6a.

Usage:
    from displacement_layer import DisplacementLayer
    from grid_cells import GridCellLayer

    # Must use the same periods as the paired GridCellLayer
    grid   = GridCellLayer(periods=[7.0, 11.0, 13.0], ...)
    displ  = DisplacementLayer(periods=[7.0, 11.0, 13.0], ...)

    # Encode displacement +5 and apply to grid layer
    displ.set_displacement(5.0)
    displ.apply_to(grid)

    # Subtraction: displacement -3 (handled natively via modular arithmetic)
    displ.set_displacement(-3.0)
    displ.apply_to(grid)
"""

import numpy as np
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sdr import encode_periodic, concatenate
from grid_cells import GridCellLayer


class DisplacementModule:
    """A single displacement cell module paired with one grid cell module.

    Encodes a scalar displacement as a phase in [0, period), using the same
    periodic SDR encoding as its paired GridCellModule.

    Args:
        period: The spatial period λ of this module — must match the paired
            GridCellModule's period.
        sdr_length: Length of the displacement phase SDR.
        sdr_width: Number of active bits in the displacement phase SDR.
    """

    def __init__(self, period: float, sdr_length: int, sdr_width: int):
        self.period = period
        self.sdr_length = sdr_length
        self.sdr_width = sdr_width
        self.phase = 0.0  # current displacement phase ∈ [0, period)

    def set_displacement(self, displacement: float) -> None:
        """Set the displacement phase: δ = displacement mod period."""
        self.phase = float(displacement) % self.period

    def get_sdr(self) -> np.ndarray:
        """Encode the current displacement phase as a periodic SDR."""
        return encode_periodic(
            self.phase,
            n=self.sdr_length,
            w=self.sdr_width,
            min_val=0.0,
            max_val=self.period,
        )

    def apply_to_grid_module_phase(self, grid_phase: float) -> float:
        """Compute the updated grid module phase after applying this displacement.

        Returns (grid_phase + self.phase) mod period.
        """
        return (grid_phase + self.phase) % self.period

    def __repr__(self) -> str:
        return (
            f"DisplacementModule(period={self.period}, "
            f"phase={self.phase:.4f})"
        )


class DisplacementLayer:
    """L5b displacement cell module system, mirroring a GridCellLayer.

    One displacement cell module per grid cell module, with identical
    period structure. Encodes displacements as SDRs and applies them
    to a paired GridCellLayer via path integration.

    Args:
        periods: List of module periods — must match the paired GridCellLayer.
        sdr_length_per_module: SDR length per module (match GridCellLayer).
        sdr_width_per_module: Active bits per module (match GridCellLayer).
    """

    def __init__(
        self,
        periods: List[float],
        sdr_length_per_module: int = 64,
        sdr_width_per_module: int = 9,
    ):
        if len(periods) == 0:
            raise ValueError("Need at least one displacement module")

        self.modules = [
            DisplacementModule(p, sdr_length_per_module, sdr_width_per_module)
            for p in periods
        ]
        self.sdr_length_per_module = sdr_length_per_module
        self.sdr_width_per_module = sdr_width_per_module
        self.total_sdr_length = len(periods) * sdr_length_per_module

    @property
    def num_modules(self) -> int:
        return len(self.modules)

    def set_displacement(self, displacement: float) -> None:
        """Encode a scalar displacement across all modules.

        Each module computes displacement mod its period independently.
        Negative displacements work natively — subtraction is modular
        addition of the additive inverse.

        Args:
            displacement: The displacement magnitude. Positive = forward,
                negative = backward. The system handles both identically
                via modular arithmetic.
        """
        for module in self.modules:
            module.set_displacement(displacement)

    def get_displacement_sdr(self) -> np.ndarray:
        """Return the full displacement SDR: concatenation of all module phases.

        Two displacements that are numerically close will have high SDR
        overlap within each module (up to the module's period).
        """
        return concatenate([m.get_sdr() for m in self.modules])

    def get_phases(self) -> np.ndarray:
        """Return the current displacement phase of each module."""
        return np.array([m.phase for m in self.modules])

    def apply_to(self, grid_layer: GridCellLayer) -> None:
        """Apply the current displacement to a paired GridCellLayer.

        Calls grid_layer.integrate(d) using the displacement value
        reconstructed from the current module phases. This performs path
        integration: φᵢ ← (φᵢ + δᵢ) mod λᵢ for each module i.

        The grid_layer must have the same number of modules and the same
        periods as this DisplacementLayer.

        Args:
            grid_layer: The L6a GridCellLayer to update.
        """
        if len(grid_layer.modules) != self.num_modules:
            raise ValueError(
                f"Module count mismatch: DisplacementLayer has "
                f"{self.num_modules} modules, GridCellLayer has "
                f"{len(grid_layer.modules)} modules."
            )

        # Apply each module's displacement phase to its paired grid module
        for i, (d_mod, g_mod) in enumerate(
            zip(self.modules, grid_layer.modules)
        ):
            if abs(d_mod.period - g_mod.period) > 1e-9:
                raise ValueError(
                    f"Period mismatch at module {i}: "
                    f"displacement={d_mod.period}, grid={g_mod.period}"
                )
            g_mod.phase = (g_mod.phase + d_mod.phase) % g_mod.period

    def apply_displacement_to(
        self, displacement: float, grid_layer: GridCellLayer
    ) -> None:
        """Convenience method: set displacement and apply to grid layer.

        Equivalent to:
            self.set_displacement(displacement)
            self.apply_to(grid_layer)

        Args:
            displacement: Scalar displacement to apply.
            grid_layer: The L6a GridCellLayer to update.
        """
        self.set_displacement(displacement)
        self.apply_to(grid_layer)

    def __repr__(self) -> str:
        periods = [m.period for m in self.modules]
        phases = [f"{m.phase:.3f}" for m in self.modules]
        return (
            f"DisplacementLayer(periods={periods}, phases={phases}, "
            f"sdr_total={self.total_sdr_length})"
        )


def make_displacement_layer_from_grid(
    grid_layer: GridCellLayer,
) -> DisplacementLayer:
    """Create a DisplacementLayer paired with an existing GridCellLayer.

    Copies the periods, SDR length, and SDR width exactly from the grid
    layer — guaranteeing compatibility.

    Args:
        grid_layer: The L6a GridCellLayer this layer is paired with.

    Returns:
        A DisplacementLayer with matching structure.
    """
    periods = [m.period for m in grid_layer.modules]
    return DisplacementLayer(
        periods=periods,
        sdr_length_per_module=grid_layer.sdr_length_per_module,
        sdr_width_per_module=grid_layer.sdr_width_per_module,
    )
