"""
sdr.py — Sparse Distributed Representation library.

The core data type for the TBT architecture. Every signal in the system
(sensory input, motor commands, inter-layer communication, inter-column voting)
is an SDR.

An SDR is represented as a 1D numpy boolean array of length n, where only w
bits are active (True). w/n is the sparsity ratio, typically ~0.02.

All functions operate on plain numpy arrays — no wrapper classes. This is
Data Oriented Design: the data is the interface.

Usage:
    from sdr import *

    a = sdr_random(n=2048, w=40)
    b = sdr_random(n=2048, w=40)
    print(overlap(a, b))
    print(match(a, b, theta=3))
"""

import numpy as np
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Creation
# ══════════════════════════════════════════════════════════════════════════════

def sdr_random(n: int, w: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Create a random SDR with exactly w active bits out of n total.

    Args:
        n: Total number of bits (SDR length).
        w: Number of active bits (population).
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Boolean array of shape (n,) with exactly w True values.
    """
    if rng is None:
        rng = np.random.default_rng()
    sdr = np.zeros(n, dtype=bool)
    indices = rng.choice(n, size=w, replace=False)
    sdr[indices] = True
    return sdr


def sdr_from_indices(n: int, indices: np.ndarray) -> np.ndarray:
    """Create an SDR from explicit active bit indices.

    Args:
        n: Total number of bits.
        indices: Array of integer indices to set active.

    Returns:
        Boolean array of shape (n,) with specified indices set True.
    """
    sdr = np.zeros(n, dtype=bool)
    sdr[indices] = True
    return sdr


def sdr_empty(n: int) -> np.ndarray:
    """Create an SDR with no active bits."""
    return np.zeros(n, dtype=bool)


def active_indices(sdr: np.ndarray) -> np.ndarray:
    """Return the indices of active bits in an SDR."""
    return np.where(sdr)[0]


def population(sdr: np.ndarray) -> int:
    """Return the number of active bits (Hamming weight)."""
    return int(sdr.sum())


# ══════════════════════════════════════════════════════════════════════════════
# Bitwise operations
# ══════════════════════════════════════════════════════════════════════════════

def sdr_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bitwise AND (intersection). Active bits present in both SDRs."""
    return a & b


def sdr_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bitwise OR (union). Active bits present in either SDR."""
    return a | b


def sdr_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bitwise XOR. Active bits present in exactly one SDR."""
    return a ^ b


def sdr_not(a: np.ndarray) -> np.ndarray:
    """Bitwise NOT. Flip all bits."""
    return ~a


# ══════════════════════════════════════════════════════════════════════════════
# Similarity and matching
# ══════════════════════════════════════════════════════════════════════════════

def overlap(a: np.ndarray, b: np.ndarray) -> int:
    """Overlap score: number of active bits shared by both SDRs.

    This is the fundamental similarity metric for SDRs.
    Equivalent to population(sdr_and(a, b)), but faster.
    """
    return int((a & b).sum())


def match(a: np.ndarray, b: np.ndarray, theta: int) -> bool:
    """Check whether two SDRs match (overlap >= theta).

    Args:
        a, b: SDRs to compare.
        theta: Minimum overlap score to declare a match.

    Returns:
        True if overlap(a, b) >= theta.
    """
    return overlap(a, b) >= theta


def overlap_score_normalized(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized overlap: overlap / max(population(a), population(b)).

    Returns a float in [0.0, 1.0]. Useful for comparing SDRs with
    different populations.
    """
    max_pop = max(population(a), population(b))
    if max_pop == 0:
        return 0.0
    return overlap(a, b) / max_pop


# ══════════════════════════════════════════════════════════════════════════════
# Union operations
# ══════════════════════════════════════════════════════════════════════════════

def union(sdrs: list[np.ndarray]) -> np.ndarray:
    """Create a union SDR from a list of SDRs.

    The union has an active bit wherever ANY input SDR has an active bit.
    Useful for fast approximate membership testing — check if an SDR
    matches the union instead of comparing against each SDR individually
    (avoids O(n^2)).

    Warning: unions saturate as more SDRs are added. Monitor the
    population of the union relative to its length.
    """
    if len(sdrs) == 0:
        raise ValueError("Cannot create union of empty list")
    result = sdrs[0].copy()
    for s in sdrs[1:]:
        result |= s
    return result


def union_saturation(union_sdr: np.ndarray) -> float:
    """How saturated is a union SDR? Returns population/length.

    Values above ~0.5 mean the union is losing discriminative power.
    """
    return population(union_sdr) / len(union_sdr)


def match_union(sdr: np.ndarray, union_sdr: np.ndarray, theta: int) -> bool:
    """Check if an SDR matches any member of a union.

    This is an APPROXIMATION. As the union saturates, false positive
    rate increases. But it's O(1) per check instead of O(n).
    """
    return match(sdr, union_sdr, theta)


# ══════════════════════════════════════════════════════════════════════════════
# Subsampling
# ══════════════════════════════════════════════════════════════════════════════

def subsample(sdr: np.ndarray, k: int,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Subsample an SDR by keeping only k of its active bits.

    Because SDRs have extremely low false positive rates, you can
    store/transmit only a fraction of the active bits and still get
    reliable matching. The resulting SDR has the same length but
    population k (or less if the input has fewer than k active bits).

    Args:
        sdr: Input SDR.
        k: Number of active bits to keep.
        rng: Optional random generator.

    Returns:
        A new SDR with at most k active bits, all of which were active
        in the input.
    """
    if rng is None:
        rng = np.random.default_rng()
    indices = active_indices(sdr)
    if len(indices) <= k:
        return sdr.copy()
    kept = rng.choice(indices, size=k, replace=False)
    return sdr_from_indices(len(sdr), kept)


# ══════════════════════════════════════════════════════════════════════════════
# Concatenation
# ══════════════════════════════════════════════════════════════════════════════

def concatenate(sdrs: list[np.ndarray]) -> np.ndarray:
    """Concatenate multiple SDRs into a single longer SDR.

    Used for combining outputs of multiple sub-encoders (e.g. grid cell
    modules with different scales) into one representation.
    The resulting SDR has length = sum of input lengths, and
    population = sum of input populations.
    """
    return np.concatenate(sdrs)


def split(sdr: np.ndarray, lengths: list[int]) -> list[np.ndarray]:
    """Split a concatenated SDR back into its component SDRs.

    Args:
        sdr: The concatenated SDR.
        lengths: The length of each component SDR. Must sum to len(sdr).

    Returns:
        List of SDRs, one per length.
    """
    if sum(lengths) != len(sdr):
        raise ValueError(
            f"Lengths sum to {sum(lengths)}, but SDR has length {len(sdr)}"
        )
    parts = []
    offset = 0
    for length in lengths:
        parts.append(sdr[offset:offset + length].copy())
        offset += length
    return parts


# ══════════════════════════════════════════════════════════════════════════════
# Capacity and false positive estimation
# ══════════════════════════════════════════════════════════════════════════════

def capacity(n: int, w: int) -> float:
    """Number of unique SDRs with length n and population w.

    This is C(n, w) = n! / (w! * (n-w)!).
    For large n, this number is astronomically large.
    Returns a float because the exact integer can overflow.
    """
    from math import comb
    return float(comb(n, w))


def false_positive_probability(n: int, w: int, theta: int,
                               num_sdrs_in_set: int = 1) -> float:
    """Estimate the probability of a false positive match.

    Given a random SDR of length n and population w, what is the
    probability it matches (overlap >= theta) with at least one SDR
    in a set of `num_sdrs_in_set` random SDRs?

    Uses the hypergeometric distribution for a single comparison,
    then union bound for the set.

    Args:
        n: SDR length.
        w: SDR population (same for both compared SDRs).
        theta: Match threshold.
        num_sdrs_in_set: Number of SDRs to compare against.

    Returns:
        Approximate probability of at least one false positive.
    """
    from scipy.stats import hypergeom

    # Probability of overlap >= theta between two random SDRs
    # This is a hypergeometric distribution:
    # Drawing w balls from an urn with w white and (n-w) black balls,
    # probability of drawing >= theta white balls.
    p_single = hypergeom.sf(theta - 1, n, w, w)

    # Union bound: P(at least one match in set) <= num * p_single
    # This is an upper bound, tight when p_single * num_sdrs_in_set << 1
    p_set = min(1.0, num_sdrs_in_set * p_single)

    return p_set


# ══════════════════════════════════════════════════════════════════════════════
# Noise injection (for testing robustness)
# ══════════════════════════════════════════════════════════════════════════════

def add_noise(sdr: np.ndarray, num_flips: int,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add noise to an SDR by flipping bits.

    Randomly turns off `num_flips` active bits and turns on `num_flips`
    inactive bits, keeping the population constant.

    Args:
        sdr: Input SDR.
        num_flips: Number of bits to flip in each direction.
        rng: Optional random generator.

    Returns:
        A new noisy SDR with the same population as the input.
    """
    if rng is None:
        rng = np.random.default_rng()

    active = active_indices(sdr)
    inactive = np.where(~sdr)[0]

    if num_flips > len(active) or num_flips > len(inactive):
        raise ValueError(
            f"Cannot flip {num_flips} bits: only {len(active)} active "
            f"and {len(inactive)} inactive bits available"
        )

    turn_off = rng.choice(active, size=num_flips, replace=False)
    turn_on = rng.choice(inactive, size=num_flips, replace=False)

    noisy = sdr.copy()
    noisy[turn_off] = False
    noisy[turn_on] = True
    return noisy


# ══════════════════════════════════════════════════════════════════════════════
# Encoding — Scalar
# ══════════════════════════════════════════════════════════════════════════════

def encode_scalar(value: float, n: int, w: int,
                  min_val: float, max_val: float) -> np.ndarray:
    """Encode a scalar value into an SDR using bucket encoding.

    Creates a consecutive block of w active bits whose position slides
    along the SDR proportionally to where `value` falls in [min_val, max_val].

    Semantically similar values (nearby on the number line) produce SDRs
    with high overlap because their active-bit buckets overlap.

    Args:
        value: The scalar to encode.
        n: SDR length.
        w: Number of active bits (bucket width).
        min_val: Minimum of the encoding range.
        max_val: Maximum of the encoding range.

    Returns:
        Boolean SDR of length n.
    """
    if max_val <= min_val:
        raise ValueError(f"max_val ({max_val}) must be > min_val ({min_val})")

    # Clamp to range
    value = max(min_val, min(max_val, value))

    # Calculate the start position of the active bucket
    # The bucket can start from index 0 to index (n - w)
    range_size = max_val - min_val
    fraction = (value - min_val) / range_size
    start = int(round(fraction * (n - w)))
    start = max(0, min(n - w, start))

    sdr = np.zeros(n, dtype=bool)
    sdr[start:start + w] = True
    return sdr


def decode_scalar(sdr: np.ndarray, w: int,
                  min_val: float, max_val: float) -> float:
    """Decode a scalar-encoded SDR back to an approximate scalar value.

    Finds the center of mass of the active bits and maps it back
    to the scalar range.

    Args:
        sdr: A scalar-encoded SDR.
        w: The bucket width used during encoding.
        min_val: Minimum of the encoding range.
        max_val: Maximum of the encoding range.

    Returns:
        The approximate scalar value.
    """
    n = len(sdr)
    indices = active_indices(sdr)
    if len(indices) == 0:
        return min_val

    # Center of mass of active bits
    center = indices.mean()

    # The center of a bucket that starts at `start` is at start + w/2 - 0.5
    # start ranges from 0 to (n - w)
    # So center ranges from (w/2 - 0.5) to (n - w + w/2 - 0.5) = (n - w/2 - 0.5)
    center_min = (w - 1) / 2.0
    center_max = (n - 1) - (w - 1) / 2.0

    if center_max <= center_min:
        return (min_val + max_val) / 2.0

    fraction = (center - center_min) / (center_max - center_min)
    fraction = max(0.0, min(1.0, fraction))
    return min_val + fraction * (max_val - min_val)


# ══════════════════════════════════════════════════════════════════════════════
# Encoding — Periodic
# ══════════════════════════════════════════════════════════════════════════════

def encode_periodic(value: float, n: int, w: int,
                    min_val: float, max_val: float) -> np.ndarray:
    """Encode a scalar with wrap-around (periodic encoding).

    Like scalar encoding, but the representation wraps: the max value
    is semantically close to the min value. Useful for cyclic data
    like time of day, compass heading, angle.

    The active bits can wrap around the end of the array.

    Args:
        value: The scalar to encode.
        n: SDR length. The full length is used for the period.
        w: Number of active bits.
        min_val: Start of the periodic range.
        max_val: End of the periodic range (wraps back to min_val).

    Returns:
        Boolean SDR of length n.
    """
    if max_val <= min_val:
        raise ValueError(f"max_val ({max_val}) must be > min_val ({min_val})")

    range_size = max_val - min_val
    # Normalize to [0, 1) within the period
    fraction = ((value - min_val) % range_size) / range_size
    start = int(round(fraction * n)) % n

    sdr = np.zeros(n, dtype=bool)
    for i in range(w):
        sdr[(start + i) % n] = True
    return sdr


# ══════════════════════════════════════════════════════════════════════════════
# Encoding — Random Distributed Scalar Encoding (RDSE)
# ══════════════════════════════════════════════════════════════════════════════

class RDSEncoder:
    """Random Distributed Scalar Encoder.

    Instead of consecutive buckets, each scalar value maps to a set of
    randomly scattered active bits. Adjacent values share all but one bit,
    controlled by the resolution parameter.

    This encoder is STATEFUL: it builds up a mapping as new values are
    encountered. This is the tradeoff — better noise properties than
    bucket encoding, but requires persistent state.

    Args:
        n: SDR length.
        w: Number of active bits.
        resolution: How many scalar units correspond to one bit change.
            E.g., resolution=1 means every integer gets a distinct SDR
            that differs from its neighbors by exactly 1 bit.
    """

    def __init__(self, n: int, w: int, resolution: float = 1.0,
                 seed: Optional[int] = None):
        self.n = n
        self.w = w
        self.resolution = resolution
        self.rng = np.random.default_rng(seed)

        # Map from bucket index to the set of active bit indices
        # A bucket is floor(value / resolution)
        self._bucket_to_bits: dict[int, np.ndarray] = {}
        # The first bucket we ever see gets a fully random assignment
        self._initialized = False

    def _get_bucket(self, value: float) -> int:
        return int(np.floor(value / self.resolution))

    def _ensure_bucket(self, bucket: int) -> np.ndarray:
        """Ensure a bucket has an assigned set of active bits.

        If the bucket doesn't exist yet, derive it from the nearest
        existing bucket by flipping one bit.
        """
        if bucket in self._bucket_to_bits:
            return self._bucket_to_bits[bucket]

        if not self._initialized:
            # First bucket: fully random
            bits = self.rng.choice(self.n, size=self.w, replace=False)
            bits.sort()
            self._bucket_to_bits[bucket] = bits
            self._initialized = True
            return bits

        # Find the nearest existing bucket and walk toward this one
        existing = sorted(self._bucket_to_bits.keys())
        # Binary search for nearest
        idx = np.searchsorted(existing, bucket)
        if idx == 0:
            nearest = existing[0]
        elif idx == len(existing):
            nearest = existing[-1]
        else:
            # Pick whichever is closer
            if abs(existing[idx] - bucket) <= abs(existing[idx - 1] - bucket):
                nearest = existing[idx]
            else:
                nearest = existing[idx - 1]

        # Walk from nearest to bucket, flipping one bit per step
        direction = 1 if bucket > nearest else -1
        current = nearest
        while current != bucket:
            next_bucket = current + direction
            if next_bucket not in self._bucket_to_bits:
                prev_bits = self._bucket_to_bits[current].copy()
                # Turn off one random active bit
                off_idx = self.rng.integers(0, self.w)
                old_bit = prev_bits[off_idx]
                # Turn on one random inactive bit
                all_inactive = np.setdiff1d(np.arange(self.n), prev_bits)
                new_bit = self.rng.choice(all_inactive)
                prev_bits[off_idx] = new_bit
                prev_bits.sort()
                self._bucket_to_bits[next_bucket] = prev_bits
            current = next_bucket

        return self._bucket_to_bits[bucket]

    def encode(self, value: float) -> np.ndarray:
        """Encode a scalar value into an SDR.

        Args:
            value: The scalar to encode.

        Returns:
            Boolean SDR of length n with w active bits.
        """
        bucket = self._get_bucket(value)
        bits = self._ensure_bucket(bucket)
        return sdr_from_indices(self.n, bits)

    def decode(self, sdr: np.ndarray) -> Optional[float]:
        """Decode an RDSE SDR back to an approximate scalar value.

        Finds the bucket with the highest overlap to the given SDR.

        Returns:
            The scalar value (bucket center), or None if no buckets exist.
        """
        if not self._bucket_to_bits:
            return None

        best_bucket = None
        best_overlap = -1
        query_indices = set(active_indices(sdr))

        for bucket, bits in self._bucket_to_bits.items():
            ov = len(query_indices & set(bits))
            if ov > best_overlap:
                best_overlap = ov
                best_bucket = bucket

        if best_bucket is None:
            return None
        return (best_bucket + 0.5) * self.resolution


# ══════════════════════════════════════════════════════════════════════════════
# Multi-encoding
# ══════════════════════════════════════════════════════════════════════════════

def encode_multi(values: list[float], encoders_config: list[dict]) -> np.ndarray:
    """Encode multiple semantically related values into a single SDR.

    Each value is encoded independently by a sub-encoder, then the
    results are concatenated. This is how a datetime encoder would work:
    separate sub-encoders for day-of-week, time-of-day, season, all
    concatenated into one SDR.

    Args:
        values: List of scalar values to encode.
        encoders_config: List of dicts, each with keys:
            'n': sub-encoder SDR length
            'w': sub-encoder population
            'min_val': encoding range minimum
            'max_val': encoding range maximum
            'periodic': bool, whether to use periodic encoding

    Returns:
        Concatenated SDR.
    """
    if len(values) != len(encoders_config):
        raise ValueError(
            f"Got {len(values)} values but {len(encoders_config)} encoder configs"
        )

    parts = []
    for val, cfg in zip(values, encoders_config):
        if cfg.get('periodic', False):
            part = encode_periodic(val, cfg['n'], cfg['w'],
                                   cfg['min_val'], cfg['max_val'])
        else:
            part = encode_scalar(val, cfg['n'], cfg['w'],
                                 cfg['min_val'], cfg['max_val'])
        parts.append(part)

    return concatenate(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Printing / debugging utilities
# ══════════════════════════════════════════════════════════════════════════════

def sdr_to_string(sdr: np.ndarray, max_display: int = 80) -> str:
    """Human-readable string representation of an SDR.

    Shows the active indices and population.
    """
    indices = active_indices(sdr)
    n = len(sdr)
    w = len(indices)
    idx_str = str(indices.tolist())
    if len(idx_str) > max_display:
        idx_str = idx_str[:max_display] + "..."
    return f"SDR(n={n}, w={w}, active={idx_str})"


def sdr_density_bar(sdr: np.ndarray, width: int = 60) -> str:
    """Visual density bar for an SDR, showing where active bits cluster.

    Divides the SDR into `width` bins and shows density per bin.
    """
    n = len(sdr)
    bin_size = max(1, n // width)
    bar = []
    for i in range(0, n, bin_size):
        chunk = sdr[i:i + bin_size]
        density = chunk.sum() / len(chunk)
        if density == 0:
            bar.append('·')
        elif density < 0.25:
            bar.append('░')
        elif density < 0.5:
            bar.append('▒')
        elif density < 0.75:
            bar.append('▓')
        else:
            bar.append('█')
    return ''.join(bar[:width])