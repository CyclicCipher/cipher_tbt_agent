"""Keyed noise families. Both hashes are implemented here; Python's `hash()` is salted per process and never used.

noisy:   h = splitmix64(key ^ polyhash(x)); y_i = splitmix64(h + i) % 10 for i < len(x)
partial: a Fisher-Yates permutation of x driven by a splitmix64 stream seeded with h
"""
from __future__ import annotations

M64 = (1 << 64) - 1


def splitmix64(z):
    z = (z + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def polyhash(x):
    h = 0
    for d in x:
        h = (h * 1000003 + d + 1) & M64
    return h


def noisy(key, x):
    h = splitmix64((key ^ polyhash(x)) & M64)
    return tuple(splitmix64((h + i) & M64) % 10 for i in range(len(x)))


def partial(key, x):
    h = splitmix64((key ^ polyhash(x)) & M64)
    out, s = list(x), h
    for i in range(len(out) - 1, 0, -1):
        s = splitmix64(s)
        j = s % (i + 1)
        out[i], out[j] = out[j], out[i]
    return tuple(out)


def derive_seed(*parts):
    """A 64-bit seed from any tuple of ints/strings, for numpy Generators. Deterministic across processes."""
    h = 0x1234567
    for p in parts:
        if isinstance(p, str):
            p = polyhash(p.encode())
        h = splitmix64((h ^ (int(p) & M64)) & M64)
    return h
