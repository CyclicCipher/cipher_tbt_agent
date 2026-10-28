"""The real adelic toolkit (docs/REAL_ADELIC.md): places, local coordinates, local distances and local densities.

The facts it implements:
- Z/N is (1/N)Z/Z inside Q/Z, and Q/Z splits into its p-adic parts Q_p/Z_p. The p-adic part of x/N is nonzero only
  for p dividing N. With N = p^k * m (p not dividing m) it is (x * m^-1 mod p^k) / p^k, written by its k base-p
  digits. The Archimedean image of x/N is x/N mod 1 on the circle R/Z.
- On classes mod N the p-adic distance is |a - b|_p = p^-min(v_p(a - b), k); on the circle it is the chord.
- Local-global counting (Hardy-Littlewood, Siegel): the number of solutions of a counting problem is approximately
  the product of its local densities over all places: the real density (the singular integral) times the p-adic
  densities (the singular series). A local density of zero is an exact certificate that no solution exists.
"""
import math

import numpy as np


def factor(n):
    """{p: k} with n = prod p^k."""
    out, d = {}, 2
    while d * d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def valuation(n, p):
    """v_p(n) for n != 0."""
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def places(N):
    """The finite places of Z/N: [(p, k, u)] for each p^k exactly dividing N, with u = (N / p^k)^-1 mod p^k."""
    return [(p, k, pow(N // p ** k, -1, p ** k)) for p, k in sorted(factor(N).items())]


def local_digits(x, p, k, u):
    """The p-adic part of x/N as its k base-p digits, least significant first: [..., k]. x: int array."""
    y = (np.asarray(x) * u) % p ** k
    return np.stack([(y // p ** j) % p for j in range(k)], -1)


def padic_distance(N, p, k):
    """[N, N]: |a - b|_p on classes mod N, rescaled to [0, 1] (0: equal at p; 1: different first digit)."""
    d = (np.arange(N)[:, None] - np.arange(N)[None, :]) % p ** k
    v = np.array([[k if x == 0 else min(valuation(int(x), p), k) for x in row] for row in d])
    return (p ** -v.astype(float) - p ** -k) / (1 - p ** -k)


def circle_distance(N):
    """[N, N]: (1 - cos(2 pi (a - b) / N)) / 2, the squared chord on R/Z rescaled to [0, 1]."""
    a = np.arange(N)
    return (1 - np.cos(2 * np.pi * (a[:, None] - a[None, :]) / N)) / 2


# ---- local densities of subset sums ------------------------------------------------------------------------------------
def moduli(limit=64, primes_up_to=31):
    """One modulus per prime p <= primes_up_to: the largest power of p not above `limit` (at least p itself)."""
    ps = [p for p in range(2, primes_up_to + 1) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    out = []
    for p in ps:
        q = p
        while q * p <= limit:
            q *= p
        out.append(q)
    return out


def densities_without_each(pool, m):
    """[n, m]: for each i, the distribution over Z/m of the sum of a uniformly random subset of pool minus element i.
    Exact (prefix and suffix dynamic programs, then a cyclic convolution); zeros are exact zeros."""
    n = len(pool)
    pre = np.zeros((n + 1, m))
    suf = np.zeros((n + 1, m))
    pre[0, 0] = suf[n, 0] = 1.0
    for j, c in enumerate(pool):
        pre[j + 1] = 0.5 * (pre[j] + np.roll(pre[j], c % m))
    for j in range(n - 1, -1, -1):
        suf[j] = 0.5 * (suf[j + 1] + np.roll(suf[j + 1], pool[j] % m))
    idx = (np.arange(m)[:, None] - np.arange(m)[None, :]) % m          # [t, s] -> (t - s) mod m
    return np.einsum("is,its->it", pre[:n], suf[1:][:, idx])


def log_real_density(pool_sum, pool_sq, r):
    """log of the normal approximation to the density of a random subset's sum at r (mean pool_sum / 2,
    variance pool_sq / 4): the singular integral, up to the common factor 2^|pool|."""
    var = pool_sq / 4.0
    if var <= 0:
        return 0.0 if r == 0 else -math.inf
    return -0.5 * (r - pool_sum / 2.0) ** 2 / var - 0.5 * math.log(2 * math.pi * var)
