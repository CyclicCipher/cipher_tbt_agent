"""
encoder.py — Sensory encoders for the TBT cortical column.

Every sense must be converted into an SDR before it reaches the brain.
This module provides a clean, consistent interface over the encoding
functions in sdr.py.

Design:
  - All encoders expose a single method: encode(value) -> np.ndarray
  - All encoders are stateless except RDSEncoder (which must track its
    bucket-to-bits mapping as it grows).
  - MultiEncoder combines semantically related quantities into one SDR
    by concatenating sub-encoder outputs.

The key constraint for any encoder:
  1. Semantically similar inputs must produce SDRs with high overlap.
  2. The same input must always produce the same output.
  3. Output length and sparsity must be fixed and consistent.

Usage:
    from encoder import ScalarEncoder, PeriodicEncoder, MultiEncoder
    from encoder import RDSEncoder  # stateful

    # Number line encoder for the addition test
    enc = ScalarEncoder(n=256, w=21, min_val=0, max_val=20)
    sdr = enc.encode(7.0)

    # Datetime MultiEncoder
    enc = MultiEncoder([
        PeriodicEncoder(n=128, w=11, min_val=0, max_val=7),   # day of week
        PeriodicEncoder(n=128, w=11, min_val=0, max_val=24),  # hour
    ])
    sdr = enc.encode([2.0, 14.5])  # Wednesday, 2:30pm
"""

import numpy as np
from typing import List, Optional, Union

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import sdr as sdr_lib


class ScalarEncoder:
    """Encodes a continuous scalar value into a consecutive-bucket SDR.

    Semantically similar values (nearby on the number line) produce SDRs
    with high bit overlap because their active-bit buckets overlap.

    Key constraint: for adjacent values to have any overlap, the bucket
    width w must exceed n / (max_val - min_val). Equivalently:
        w > n / num_distinct_values

    Args:
        n: SDR length (total bits).
        w: Number of active bits (bucket width).
        min_val: Minimum encodable value (inclusive).
        max_val: Maximum encodable value (inclusive).
    """

    def __init__(self, n: int, w: int, min_val: float, max_val: float):
        if max_val <= min_val:
            raise ValueError(f"max_val ({max_val}) must be > min_val ({min_val})")
        if w <= 0 or w >= n:
            raise ValueError(f"w ({w}) must be in (0, n={n})")

        self.n = n
        self.w = w
        self.min_val = min_val
        self.max_val = max_val
        self.output_length = n

        # Warn if adjacent integer values won't overlap
        step_size = (max_val - min_val) / (n - w)
        if step_size > w:
            import warnings
            warnings.warn(
                f"ScalarEncoder: bucket width ({w}) < step size ({step_size:.1f}). "
                f"Adjacent values will NOT have overlapping SDRs. "
                f"Consider increasing w or decreasing the value range.",
                stacklevel=2
            )

    def encode(self, value: float) -> np.ndarray:
        """Encode a scalar value. Values outside [min_val, max_val] are clamped."""
        return sdr_lib.encode_scalar(
            value, self.n, self.w, self.min_val, self.max_val
        )

    def decode(self, sdr: np.ndarray) -> float:
        """Approximate inverse. Returns the scalar value closest to the SDR."""
        return sdr_lib.decode_scalar(sdr, self.w, self.min_val, self.max_val)

    def __repr__(self) -> str:
        return (
            f"ScalarEncoder(n={self.n}, w={self.w}, "
            f"range=[{self.min_val}, {self.max_val}])"
        )


class PeriodicEncoder:
    """Encodes a scalar with wrap-around (periodic / cyclic encoding).

    The max value wraps back to min, making them semantically adjacent.
    Useful for: time of day, compass heading, angle, day of week.

    Args:
        n: SDR length.
        w: Number of active bits.
        min_val: Start of the periodic range.
        max_val: End of the periodic range (exclusive — wraps to min_val).
    """

    def __init__(self, n: int, w: int, min_val: float, max_val: float):
        if max_val <= min_val:
            raise ValueError(f"max_val ({max_val}) must be > min_val ({min_val})")
        self.n = n
        self.w = w
        self.min_val = min_val
        self.max_val = max_val
        self.output_length = n

    def encode(self, value: float) -> np.ndarray:
        """Encode a periodic value. Values outside range wrap automatically."""
        return sdr_lib.encode_periodic(
            value, self.n, self.w, self.min_val, self.max_val
        )

    def __repr__(self) -> str:
        return (
            f"PeriodicEncoder(n={self.n}, w={self.w}, "
            f"period=[{self.min_val}, {self.max_val}))"
        )


class RDSEncoder:
    """Random Distributed Scalar Encoder (stateful).

    Instead of consecutive buckets, active bits are randomly scattered
    throughout the SDR. Adjacent values share w-1 bits, values k steps
    apart share w-k bits (down to 0). This gives more uniform overlap
    across the value range than bucket encoding.

    The encoder is STATEFUL: it builds its bucket-to-bits mapping
    incrementally as new values are encountered.

    Args:
        n: SDR length.
        w: Number of active bits.
        resolution: How many scalar units correspond to one bit difference.
            resolution=1.0 means every integer gets a distinct SDR that
            differs from its neighbor by exactly 1 bit.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n: int,
        w: int,
        resolution: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.n = n
        self.w = w
        self.resolution = resolution
        self.output_length = n
        self._encoder = sdr_lib.RDSEncoder(n, w, resolution, seed)

    def encode(self, value: float) -> np.ndarray:
        return self._encoder.encode(value)

    def decode(self, sdr: np.ndarray) -> Optional[float]:
        return self._encoder.decode(sdr)

    def __repr__(self) -> str:
        return (
            f"RDSEncoder(n={self.n}, w={self.w}, "
            f"resolution={self.resolution})"
        )


class CategoryEncoder:
    """Encodes a categorical value (integer index) into a fixed SDR.

    Each category gets a unique random SDR with no overlap with others.
    Useful for: token identities (symbols like +, -, =), labels, classes.

    Unlike scalar encoders, categorical SDRs deliberately have NO semantic
    similarity structure — the SDR for category 3 is no more similar to
    category 4 than to category 100. Use this when the categories have
    no meaningful ordering or metric relationship.

    Args:
        num_categories: Total number of categories.
        n: SDR length.
        w: Number of active bits per category.
        seed: Random seed.
    """

    def __init__(
        self,
        num_categories: int,
        n: int,
        w: int,
        seed: Optional[int] = None,
    ):
        if w * num_categories > n * 0.5:
            import warnings
            warnings.warn(
                f"CategoryEncoder: high risk of collision. "
                f"{num_categories} categories × {w} bits in SDR of length {n}. "
                f"Consider increasing n.",
                stacklevel=2
            )
        self.num_categories = num_categories
        self.n = n
        self.w = w
        self.output_length = n

        rng = np.random.default_rng(seed)
        self._sdrs = np.zeros((num_categories, n), dtype=bool)
        for i in range(num_categories):
            indices = rng.choice(n, size=w, replace=False)
            self._sdrs[i, indices] = True

    def encode(self, category: int) -> np.ndarray:
        """Encode a category index. Must be in [0, num_categories)."""
        if not (0 <= category < self.num_categories):
            raise ValueError(
                f"Category {category} out of range [0, {self.num_categories})"
            )
        return self._sdrs[category].copy()

    def decode(self, sdr: np.ndarray) -> int:
        """Find the category whose SDR has the highest overlap with the input."""
        overlaps = (self._sdrs & sdr).sum(axis=1)
        return int(overlaps.argmax())

    def __repr__(self) -> str:
        return (
            f"CategoryEncoder(num_categories={self.num_categories}, "
            f"n={self.n}, w={self.w})"
        )


class MultiEncoder:
    """Combines multiple encoders into a single SDR by concatenation.

    Use when encoding semantically related quantities that should be
    represented together. Example: a datetime encoder combining
    day-of-week, time-of-day, and season into one SDR.

    The output SDR length = sum of all sub-encoder output lengths.
    The output population = sum of all sub-encoder populations.

    Args:
        encoders: List of encoder instances. Each must have an
            `output_length` attribute and an `encode(value)` method.
    """

    def __init__(self, encoders: list):
        if len(encoders) == 0:
            raise ValueError("MultiEncoder requires at least one sub-encoder")
        self.encoders = encoders
        self.output_length = sum(e.output_length for e in encoders)

    def encode(self, values: Union[list, np.ndarray]) -> np.ndarray:
        """Encode a list of values, one per sub-encoder.

        Args:
            values: Sequence of values, one for each sub-encoder.
                Length must match number of sub-encoders.

        Returns:
            Concatenated SDR of length self.output_length.
        """
        if len(values) != len(self.encoders):
            raise ValueError(
                f"Expected {len(self.encoders)} values, got {len(values)}"
            )
        parts = [enc.encode(val) for enc, val in zip(self.encoders, values)]
        return sdr_lib.concatenate(parts)

    def __repr__(self) -> str:
        enc_str = ", ".join(repr(e) for e in self.encoders)
        return f"MultiEncoder([{enc_str}])"


# ── Convenience constructors ──────────────────────────────────────────────────

def make_number_encoder(
    max_value: int = 20,
    n: int = 256,
    w: Optional[int] = None,
) -> ScalarEncoder:
    """Scalar encoder for integers on the number line.

    Automatically selects w such that adjacent integers have overlapping
    SDRs. The rule: w > n / (max_value - 0) = n / max_value.
    We use w = ceil(n / max_value) + 2 for a comfortable overlap margin.

    Args:
        max_value: Largest integer to encode.
        n: SDR length.
        w: Active bits. If None, auto-selected for adjacent overlap.
    """
    if w is None:
        import math
        w = math.ceil(n / max_value) + 4
        # Cap at n // 2 to avoid overly dense SDRs
        w = min(w, n // 2)

    return ScalarEncoder(n=n, w=w, min_val=0.0, max_val=float(max_value))


def make_symbol_encoder(
    symbols: List[str],
    n: int = 128,
    w: int = 11,
    seed: Optional[int] = None,
) -> "SymbolEncoder":
    """Create an encoder for a fixed vocabulary of symbols (e.g. +, -, =).

    Returns a SymbolEncoder that maps strings to SDRs.
    """
    return SymbolEncoder(symbols, n, w, seed)


class SymbolEncoder:
    """Encodes named symbols (strings) into SDRs.

    A thin wrapper around CategoryEncoder that maps string symbols to
    integer indices, then to SDRs. The SDRs have no semantic similarity
    structure — each symbol gets a distinct random SDR.

    Use for operators (+, -, *, =), tokens, labels, or any discrete
    vocabulary where ordering doesn't matter.

    Args:
        symbols: List of symbol strings. Order determines category index.
        n: SDR length.
        w: Active bits per symbol.
        seed: Random seed.
    """

    def __init__(
        self,
        symbols: List[str],
        n: int = 128,
        w: int = 11,
        seed: Optional[int] = None,
    ):
        self.symbols = symbols
        self.symbol_to_idx = {s: i for i, s in enumerate(symbols)}
        self.output_length = n
        self._cat = CategoryEncoder(len(symbols), n, w, seed)

    def encode(self, symbol: str) -> np.ndarray:
        """Encode a symbol string to an SDR."""
        if symbol not in self.symbol_to_idx:
            raise ValueError(
                f"Unknown symbol '{symbol}'. Known: {self.symbols}"
            )
        return self._cat.encode(self.symbol_to_idx[symbol])

    def decode(self, sdr: np.ndarray) -> str:
        """Find the closest matching symbol."""
        idx = self._cat.decode(sdr)
        return self.symbols[idx]

    def __repr__(self) -> str:
        return (
            f"SymbolEncoder(symbols={self.symbols}, "
            f"n={self._cat.n}, w={self._cat.w})"
        )
