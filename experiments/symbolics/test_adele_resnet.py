"""
test_adele_resnet.py (the real version; Gemini's original is gemini_original/test_adele_resnet.py)
Greedy subset sum guided by the local-global principle (REAL_ADELIC.md).

The residual greedy picks one element per layer. Which one? The real adelic answer: the one that leaves the most
ways to finish. The number of subsets of the remaining pool summing to the new residual r' is estimated as the
product of its local densities over all places (Hardy-Littlewood / Siegel):
    real place      the normal density of a random subset's sum at r', and the bound 0 <= r' <= sum(pool)
    p-adic places   m * Pr[random subset's sum = r' mod m], for m = the largest power of each prime p <= 31 up to 64
                    (exact, by a dynamic program over Z/m)
A local density of zero proves that no completion exists (a local obstruction), so those moves are never taken.

Arms: euclid (closest element, Gemini's baseline), smooth (Gemini's "adele distance", a smoothness discount),
arch (the real place only), padic (the p-adic places only), adelic (both: the local-global estimate).
    python test_adele_resnet.py
"""
import math
import random
import time

import numpy as np

import adelic as A
from controls_adelic_search import planted
from gemini_original.test_adele_resnet import AdeleSymbolicResNet

MODULI = A.moduli(limit=64, primes_up_to=31)


def local_global_greedy(pool, target, mode):
    """mode: 'arch', 'padic' or 'adelic'. Returns (success, candidates scored, chosen elements)."""
    pool, r, chosen, evals = list(pool), target, [], 0
    while r != 0 and pool:
        n = len(pool)
        dens = [A.densities_without_each(pool, m) for m in MODULI] if mode != "arch" else None
        total, total_sq = sum(pool), sum(c * c for c in pool)
        best, best_score = None, -math.inf
        for i, c in enumerate(pool):
            rr = r - c
            if rr < 0:
                continue
            evals += 1
            if rr == 0:
                best, best_score = i, math.inf
                break
            if n == 1:
                continue
            score = 0.0
            if mode in ("arch", "adelic"):
                rest, rest_sq = total - c, total_sq - c * c
                score += -math.inf if rr > rest else A.log_real_density(rest, rest_sq, rr)
            if mode in ("padic", "adelic"):
                for m, d in zip(MODULI, dens):
                    q = d[i, rr % m]
                    score += math.log(m * q) if q > 0 else -math.inf
            if score > best_score:
                best, best_score = i, score
        if best is None or best_score == -math.inf:
            break
        r -= pool[best]
        chosen.append(pool.pop(best))
    return r == 0, evals, chosen


def uniform_planted(rng, n, top, k_lo, k_hi):
    nums = [rng.randint(1, top) for _ in range(n)]
    return nums, sum(rng.sample(nums, rng.randint(k_lo, k_hi)))


FAMILIES = {   # name: (instances, generator); "small" and "structured" are the CONTROLS.md instances exactly
    "small": (2000, lambda rng: planted(rng, False)),
    "structured": (2000, lambda rng: planted(rng, True)),
    "dense": (500, lambda rng: uniform_planted(rng, 20, 1000, 5, 15)),
    "hard": (500, lambda rng: uniform_planted(rng, 24, 10 ** 6, 6, 18)),
}
ARMS = ("euclid", "smooth", "arch", "padic", "adelic")


def solve(arm, nums, target):
    if arm in ("euclid", "smooth"):
        ok, ev, _, _ = AdeleSymbolicResNet(use_adele=arm == "smooth", alpha=0.5).fit(nums, target)
        return ok, ev
    ok, ev, _ = local_global_greedy(nums, target, arm)
    return ok, ev


def main():
    print("The four hand-built cases:")
    for target, nums in ((42, [1, 2, 4, 8, 16, 32, 64, 128]), (100, [51, 50, 50, 20, 10]),
                         (100, [53, 51, 49, 50, 50, 20, 10]), (90, [46, 30, 30, 30, 15])):
        print(f"   target {target:3d}  set {nums}: " + "  ".join(
            f"{arm} {'ok' if solve(arm, nums, target)[0] else '--'}" for arm in ARMS))
    for family, (count, gen) in FAMILIES.items():
        rng = random.Random(0)
        wins, evals, secs = dict.fromkeys(ARMS, 0), dict.fromkeys(ARMS, 0), dict.fromkeys(ARMS, 0.0)
        for _ in range(count):
            nums, target = gen(rng)
            for arm in ARMS:
                t0 = time.time()
                ok, ev = solve(arm, nums, target)
                secs[arm] += time.time() - t0
                wins[arm] += ok
                evals[arm] += ev
        print(f"\n{family} ({count} instances): success rate (candidates scored per instance, ms per instance)")
        for arm in ARMS:
            print(f"   {arm:7s} {wins[arm] / count:6.1%}   ({evals[arm] / count:5.0f}, {secs[arm] / count * 1e3:6.1f} ms)")


if __name__ == "__main__":
    main()
