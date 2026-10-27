"""Brute-force checks of adelic.py: exact local densities against enumeration, the CRT character identity
(x/N = sum over places of the local part, mod 1), and the strong triangle inequality of the p-adic distances.
    python check_adelic_toolkit.py
"""
import itertools

import numpy as np

import adelic as A

rng = np.random.default_rng(0)
for trial in range(20):
    pool = [int(x) for x in rng.integers(1, 200, size=rng.integers(1, 9))]
    for m in (2, 7, 9, 16, 25):
        D = A.densities_without_each(pool, m)
        for i in range(len(pool)):
            rest = pool[:i] + pool[i + 1:]
            ref = np.zeros(m)
            for bits in itertools.product((0, 1), repeat=len(rest)):
                ref[sum(b * c for b, c in zip(bits, rest)) % m] += 1
            ref /= 2 ** len(rest)
            assert np.allclose(D[i], ref), (pool, m, i)
            assert ((D[i] > 0) == (ref > 0)).all()
for N in (105, 81, 97, 360):
    pl = A.places(N)
    x = np.arange(N)
    codes = np.concatenate([A.local_digits(x, p, k, u) for p, k, u in pl], -1)
    assert len({tuple(c) for c in codes}) == N            # the local coordinates determine x (CRT)
    # the product formula / CRT character identity: x/N = sum_p (local part)/p^k mod 1
    frac = sum(((x * u) % p ** k) / p ** k for p, k, u in pl) % 1
    assert np.allclose(frac, (x / N) % 1)
    for p, k, u in pl:
        D = A.padic_distance(N, p, k)
        assert np.allclose(np.diag(D), 0) and D.max() <= 1
        # ultrametric (strong triangle inequality)
        i, j, l = rng.integers(0, N, (3, 500))
        assert (D[i, l] <= np.maximum(D[i, j], D[j, l]) + 1e-12).all()
print("moduli", A.moduli())
print("all adelic.py checks pass")
