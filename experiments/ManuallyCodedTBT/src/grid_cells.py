"""
grid_cells.py — Grid Cell Modules for Layer 6 (Reference Frame Layer).

Implements the residue number system (RNS) representation used by Layer 6
to track where a sensor is in an object's reference frame.

Theory recap:
  Each grid cell module has a spatial period λ. The module represents
  position as a PHASE: φ = position mod λ. Multiple modules with different
  periods form a residue number system. By the Chinese Remainder Theorem,
  coprime periods give a unique position representation over the range
  [0, λ₁ × λ₂ × ... × λₖ).

  Path integration (displacement) is modular addition, performed
  independently and in parallel within each module:
      φᵢ ← (φᵢ + d) mod λᵢ

  This is carry-free: no module needs to communicate with any other during
  path integration. Addition on the number line IS path integration.

  The full location SDR is the concatenation of each module's phase SDR
  (using periodic encoding from sdr.py).

Dimensionality:
  Modules are agnostic to dimensionality. In 1D (number line) each phase
  is a scalar. In 2D (visual field) each phase is a 2D vector mod the
  module's lattice basis. Higher dimensions follow the same pattern.
  This module implements 1D for now, with the interface designed so nD
  can be substituted without changing the cortical column code.

Subtraction:
  Subtraction is addition of the additive inverse. In ℤ/λℤ, the inverse
  of d is λ - d. The system handles this natively.

Usage:
    from grid_cells import GridCellLayer
    import numpy as np

    layer = GridCellLayer(
        periods=[7, 11, 13],     # coprime → unique range 0..1000
        sdr_length_per_module=64,
        sdr_width_per_module=7,
    )

    # Start at position 3
    layer.set_position(3.0)
    loc_sdr = layer.get_location_sdr()

    # Apply displacement +5 (path integration)
    layer.integrate(5.0)
    loc_sdr = layer.get_location_sdr()
    # loc_sdr now represents position 8
"""

import numpy as np
from typing import Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sdr import encode_periodic, concatenate, overlap


class GridCellModule:
    """A single grid cell module representing position mod period.

    Encodes the current phase as a periodic SDR. Supports path integration
    via modular addition.

    Args:
        period: The spatial period λ of this module. Positions that differ
            by a multiple of λ map to the same phase.
        sdr_length: Length of the phase SDR produced by this module.
        sdr_width: Number of active bits in the phase SDR (bucket width).
            Must satisfy: sdr_width > sdr_length / period for adjacent
            positions to have overlapping SDRs.
    """

    def __init__(self, period: float, sdr_length: int, sdr_width: int):
        self.period = period
        self.sdr_length = sdr_length
        self.sdr_width = sdr_width
        self.phase = 0.0  # current phase ∈ [0, period)

    def set_phase(self, position: float) -> None:
        """Set the phase by computing position mod period."""
        self.phase = float(position) % self.period

    def integrate(self, displacement: float) -> None:
        """Apply a displacement via modular addition.

        This is path integration: (φ + d) mod λ.
        Works correctly for negative displacements (subtraction).
        """
        self.phase = (self.phase + displacement) % self.period

    def get_sdr(self) -> np.ndarray:
        """Encode the current phase as a periodic SDR.

        Returns a boolean array of length sdr_length.
        """
        return encode_periodic(
            self.phase,
            n=self.sdr_length,
            w=self.sdr_width,
            min_val=0.0,
            max_val=self.period,
        )

    def displacement_sdr(self, displacement: float) -> np.ndarray:
        """Encode a displacement as a periodic SDR in this module's space.

        A displacement d has the same periodic structure as a position:
        it wraps at the module's period. This lets you compute what the
        phase would be AFTER applying the displacement without changing
        the current phase.
        """
        return encode_periodic(
            displacement % self.period,
            n=self.sdr_length,
            w=self.sdr_width,
            min_val=0.0,
            max_val=self.period,
        )

    def __repr__(self) -> str:
        return (
            f"GridCellModule(period={self.period}, "
            f"phase={self.phase:.4f}, "
            f"sdr=({self.sdr_length},{self.sdr_width}))"
        )


class GridCellLayer:
    """Collection of grid cell modules forming a residue number system.

    The full location SDR is the concatenation of all modules' phase SDRs.
    Path integration is applied to all modules independently.

    Args:
        periods: List of module periods λ₁, λ₂, ..., λₖ.
            For maximum unique range: choose coprime integers.
            Example: [7, 11, 13] → unique range = 7×11×13 = 1001.
            For noise robustness: geometric progression (e.g. ratio ~1.4).
        sdr_length_per_module: SDR length for each module's phase encoding.
        sdr_width_per_module: Active bits per module's phase SDR.
            Rule of thumb: sdr_width > sdr_length / min(periods)
            to ensure adjacent positions have overlapping SDRs.
    """

    def __init__(
        self,
        periods: List[float],
        sdr_length_per_module: int = 64,
        sdr_width_per_module: int = 9,
    ):
        if len(periods) == 0:
            raise ValueError("Need at least one grid cell module")

        self.modules = [
            GridCellModule(p, sdr_length_per_module, sdr_width_per_module)
            for p in periods
        ]
        self.sdr_length_per_module = sdr_length_per_module
        self.sdr_width_per_module = sdr_width_per_module
        self.total_sdr_length = len(periods) * sdr_length_per_module
        self.total_sdr_width = len(periods) * sdr_width_per_module

    @property
    def num_modules(self) -> int:
        return len(self.modules)

    @property
    def unique_range(self) -> float:
        """Approximate unique representable range.

        For coprime integer periods this equals the product of all periods
        (Chinese Remainder Theorem). For non-coprime periods the actual
        range is lcm(periods).
        """
        result = 1.0
        for m in self.modules:
            result *= m.period
        return result

    def set_position(self, position: float) -> None:
        """Set all modules' phases corresponding to a position.

        Each module independently computes position mod its period.
        """
        for m in self.modules:
            m.set_phase(position)

    def integrate(self, displacement: float) -> None:
        """Apply a displacement to all modules via path integration.

        Each module independently computes (φ + d) mod λ.
        This is the core computation: carry-free modular addition.
        Negative displacements (subtraction) work natively.
        """
        for m in self.modules:
            m.integrate(displacement)

    def get_location_sdr(self) -> np.ndarray:
        """Return the full location SDR: concatenation of all module phase SDRs.

        Returns boolean array of length total_sdr_length.
        Two positions whose difference is small relative to any module period
        will have high SDR overlap in that module's slice.
        """
        return concatenate([m.get_sdr() for m in self.modules])

    def get_displacement_sdr(self, displacement: float) -> np.ndarray:
        """Encode a displacement as a concatenated SDR across all modules.

        Each module encodes (displacement mod period) independently.
        """
        return concatenate([m.displacement_sdr(displacement) for m in self.modules])

    def get_phases(self) -> np.ndarray:
        """Return the current phase of each module as a float array."""
        return np.array([m.phase for m in self.modules])

    def estimated_position(self) -> float:
        """Decode the current phase vector back to an approximate position.

        Uses a simple approach: find the position in [0, unique_range) whose
        phase vector best matches the current module phases.

        For the 1D case with coprime integer periods, this is the Chinese
        Remainder Theorem reconstruction. For general periods, we do a
        brute-force search (practical only for small ranges).

        For large ranges use this only for diagnostics — the SDR itself
        is the canonical representation.
        """
        # Try CRT-style reconstruction first (exact for coprime integer periods)
        try:
            pos = _crt_reconstruct(
                [m.phase for m in self.modules],
                [m.period for m in self.modules],
            )
            if pos is not None:
                return pos
        except Exception:
            pass

        # Fallback: brute force over a reasonable range
        best_pos = 0.0
        best_score = -1.0
        current_sdrs = [m.get_sdr() for m in self.modules]

        test_range = min(int(self.unique_range), 10000)
        for pos in range(test_range):
            score = sum(
                overlap(current_sdrs[i], self.modules[i].displacement_sdr(pos))
                for i in range(self.num_modules)
            )
            if score > best_score:
                best_score = score
                best_pos = float(pos)

        return best_pos

    def __repr__(self) -> str:
        periods = [m.period for m in self.modules]
        phases = [f"{m.phase:.3f}" for m in self.modules]
        return (
            f"GridCellLayer(periods={periods}, phases={phases}, "
            f"sdr_total={self.total_sdr_length})"
        )


def _crt_reconstruct(
    remainders: List[float], moduli: List[float]
) -> Optional[float]:
    """Chinese Remainder Theorem reconstruction for integer-valued inputs.

    Given remainders r₁, r₂, ..., rₖ and moduli m₁, m₂, ..., mₖ
    (assumed coprime integers), find x such that x ≡ rᵢ (mod mᵢ) for all i.

    Returns None if moduli are not coprime integers or reconstruction fails.
    """
    # Check that moduli are close to integers and phases are close to integers
    int_moduli = [round(m) for m in moduli]
    int_remainders = [round(r) for r in remainders]

    if any(abs(m - im) > 0.01 for m, im in zip(moduli, int_moduli)):
        return None  # non-integer moduli, fall back

    # Standard CRT
    M = 1
    for m in int_moduli:
        M *= m

    x = 0
    for r, m in zip(int_remainders, int_moduli):
        Mi = M // m
        # Find modular inverse of Mi mod m
        inv = _mod_inverse(Mi, m)
        if inv is None:
            return None
        x += r * Mi * inv

    return float(x % M)


def _mod_inverse(a: int, m: int) -> Optional[int]:
    """Extended Euclidean Algorithm to find modular inverse of a mod m."""
    if m == 1:
        return 0
    g, x, _ = _extended_gcd(a % m, m)
    if g != 1:
        return None  # No inverse (not coprime)
    return x % m


def _extended_gcd(a: int, b: int):
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def make_number_line_layer(
    max_value: int = 100,
    num_modules: int = 3,
    sdr_length_per_module: int = 64,
    sdr_width_per_module: int = 9,
) -> GridCellLayer:
    """Convenience constructor for a 1D number line reference frame.

    Selects coprime periods whose product exceeds max_value, ensuring
    every integer in [0, max_value) has a unique phase combination.

    Args:
        max_value: The largest value that must be uniquely representable.
        num_modules: Number of grid cell modules. More modules = larger range
            and better error correction at the cost of longer SDRs.
        sdr_length_per_module: SDR length per module.
        sdr_width_per_module: Active bits per module SDR.

    Returns:
        GridCellLayer configured for the number line.
    """
    # Choose small coprime primes whose product exceeds max_value
    # Starting from small primes gives the widest coverage per module
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    periods = []
    coverage = 1
    for p in primes:
        if len(periods) >= num_modules:
            break
        periods.append(float(p))
        coverage *= p
        if coverage > max_value:
            break

    # If we haven't used all requested modules and still have coverage, add more
    i = len(periods)
    while len(periods) < num_modules and i < len(primes):
        periods.append(float(primes[i]))
        i += 1

    return GridCellLayer(
        periods=periods,
        sdr_length_per_module=sdr_length_per_module,
        sdr_width_per_module=sdr_width_per_module,
    )
