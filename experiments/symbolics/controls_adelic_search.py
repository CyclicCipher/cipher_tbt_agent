"""Control for test_adele_resnet.py (CONTROLS.md): does the adelic distance help greedy subset sum beyond the four
hand-built cases? Planted instances: n numbers, a random subset's sum is the target. Two families:
    random     - all numbers uniform in 1..100
    structured - the solution elements are multiples of a factor f of the target, the decoys are not (the shape of the
                 hand-built cases)
Success = the greedy residual search reaches the target exactly (any subset).
Exploratory arm (added after S1/S2 were scored): "shuffled" discounts each distance by the p-adic factor of an
unrelated random integer in 1..100 -- the same distribution of discounts, no relation to the residual.
    python controls_adelic_search.py
"""
import random

from gemini_original import test_adele_resnet as T
from gemini_original.test_adele_resnet import AdeleSymbolicResNet


class ShuffledDistance(T.AdeleDistance):
    rng = random.Random(1)

    def compute(self, residual, candidate):
        diff = abs(residual - candidate)
        if diff == 0:
            return 0.0
        r, factor = self.rng.randint(1, 100), 1.0
        for p in self.prime_base:
            v = T.p_adic_valuation(r, p)
            if v > 0:
                factor *= p ** (-self.alpha * v)
        return diff * factor


N_INSTANCES, N = 2000, 10


def planted(rng, structured):
    k = rng.randint(2, 5)
    if not structured:
        nums = [rng.randint(1, 100) for _ in range(N)]
        return nums, sum(rng.sample(nums, k))
    f = rng.choice([2, 3, 5, 7, 10])
    sol = [f * rng.randint(1, 100 // f) for _ in range(k)]
    decoys = []
    while len(decoys) < N - k:
        d = rng.randint(1, 100)
        if d % f:
            decoys.append(d)
    nums = sol + decoys
    rng.shuffle(nums)
    return nums, sum(sol)


def main():
    for family in ("random", "structured"):
        rng = random.Random(0)
        wins = {"euclidean": 0, "adelic": 0, "shuffled": 0}
        evals = {"euclidean": 0, "adelic": 0, "shuffled": 0}
        for _ in range(N_INSTANCES):
            nums, target = planted(rng, family == "structured")
            for name, adele in (("euclidean", False), ("adelic", True)):
                ok, ev, _, _ = AdeleSymbolicResNet(use_adele=adele, alpha=0.5).fit(nums, target)
                wins[name] += ok
                evals[name] += ev
            T.AdeleDistance, real = ShuffledDistance, T.AdeleDistance
            wins["shuffled"] += AdeleSymbolicResNet(use_adele=True, alpha=0.5).fit(nums, target)[0]
            T.AdeleDistance = real
        print(f"{family:10s}: success euclidean {wins['euclidean'] / N_INSTANCES:.3f}  adelic "
              f"{wins['adelic'] / N_INSTANCES:.3f}  shuffled {wins['shuffled'] / N_INSTANCES:.3f}  (evals per instance {evals['euclidean'] / N_INSTANCES:.0f} / "
              f"{evals['adelic'] / N_INSTANCES:.0f}; exhaustive would be {2 ** N})")


if __name__ == "__main__":
    main()
