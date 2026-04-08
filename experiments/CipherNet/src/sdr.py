"""Sparse Distributed Representation (SDR).

An SDR is a binary vector where ~2% of bits are active. The pattern
of active bits IS the representation. Two SDRs sharing active bits
share semantic properties (similarity = overlap).

This is how cortical columns represent features at every level.
Unlike strings (exact match only), SDRs support:
- Similarity: overlap(A, B) / size → 0.0 to 1.0
- Union: A | B (combine features)
- Intersection: A & B (shared features)
- Generalization: partial matches still recognized

Implemented as numpy arrays for vectorized operations.
"""
from __future__ import annotations

import numpy as np


class SDR:
    """A sparse distributed representation.

    Internally: a sorted array of active bit indices.
    Total bits (n) and number of active bits (w) define the space.
    """

    __slots__ = ('n', 'indices')

    def __init__(self, n: int, indices: np.ndarray | list[int] | None = None):
        self.n = n
        if indices is None:
            self.indices = np.array([], dtype=np.int32)
        elif isinstance(indices, np.ndarray):
            self.indices = np.sort(indices.astype(np.int32))
        else:
            self.indices = np.sort(np.array(indices, dtype=np.int32))

    @property
    def w(self) -> int:
        """Number of active bits (sparsity width)."""
        return len(self.indices)

    @property
    def sparsity(self) -> float:
        """Fraction of active bits."""
        return self.w / self.n if self.n > 0 else 0.0

    def overlap(self, other: 'SDR') -> int:
        """Count of shared active bits."""
        return len(np.intersect1d(self.indices, other.indices))

    def similarity(self, other: 'SDR') -> float:
        """Overlap similarity: shared / min(self.w, other.w).

        1.0 = identical active bits. 0.0 = no overlap.
        Uses min(w) denominator so a subset always matches fully.
        """
        if self.w == 0 or other.w == 0:
            return 0.0
        shared = self.overlap(other)
        return shared / min(self.w, other.w)

    def union(self, other: 'SDR') -> 'SDR':
        """Union: all bits active in either SDR."""
        combined = np.union1d(self.indices, other.indices)
        return SDR(max(self.n, other.n), combined)

    def intersection(self, other: 'SDR') -> 'SDR':
        """Intersection: bits active in BOTH SDRs."""
        shared = np.intersect1d(self.indices, other.indices)
        return SDR(max(self.n, other.n), shared)

    def to_dense(self) -> np.ndarray:
        """Convert to dense binary vector."""
        v = np.zeros(self.n, dtype=np.float32)
        if self.w > 0:
            v[self.indices] = 1.0
        return v

    @staticmethod
    def from_dense(v: np.ndarray) -> 'SDR':
        """Create SDR from a dense binary vector."""
        return SDR(len(v), np.where(v > 0.5)[0])

    @staticmethod
    def random(n: int, w: int, rng: np.random.RandomState | None = None) -> 'SDR':
        """Create a random SDR with w active bits out of n total."""
        if rng is None:
            rng = np.random.RandomState()
        indices = rng.choice(n, size=w, replace=False)
        return SDR(n, indices)

    def __eq__(self, other):
        if not isinstance(other, SDR):
            return False
        return self.n == other.n and np.array_equal(self.indices, other.indices)

    def __hash__(self):
        return hash((self.n, tuple(self.indices)))

    def __repr__(self):
        if self.w <= 10:
            return f"SDR(n={self.n}, w={self.w}, bits={list(self.indices)})"
        return f"SDR(n={self.n}, w={self.w})"


class SDREncoder:
    """Encode continuous values or discrete tokens into SDRs.

    For VQ codebook codes: each code gets a unique random SDR.
    For continuous values: scalar → bucket → SDR (thermometer encoding).
    """

    def __init__(self, n: int = 256, w: int = 10, seed: int = 42):
        """
        Args:
            n: total bits in the SDR.
            w: active bits per encoded value.
            seed: random seed for reproducible encodings.
        """
        self.n = n
        self.w = w
        self.rng = np.random.RandomState(seed)
        self._cache: dict[str, SDR] = {}

    def encode(self, token: str) -> SDR:
        """Encode a discrete token (e.g., VQ code "v42") to an SDR.

        Same token always produces the same SDR (cached).
        Different tokens produce SDRs with low overlap (random).
        """
        if token in self._cache:
            return self._cache[token]
        sdr = SDR.random(self.n, self.w, self.rng)
        self._cache[token] = sdr
        return sdr

    def encode_scalar(self, value: float, min_val: float = 0.0,
                      max_val: float = 1.0) -> SDR:
        """Encode a continuous scalar to an SDR (thermometer-style).

        Adjacent values produce overlapping SDRs (smooth encoding).
        """
        # Bucket the value into n_buckets positions.
        n_buckets = self.n - self.w + 1
        if n_buckets < 1:
            n_buckets = 1
        frac = (value - min_val) / (max_val - min_val + 1e-10)
        frac = max(0.0, min(1.0, frac))
        start = int(frac * (n_buckets - 1))
        indices = np.arange(start, start + self.w, dtype=np.int32)
        return SDR(self.n, indices)

    def n_cached(self) -> int:
        return len(self._cache)
