"""
test_adele_resnet.py
Benchmarking the Adele Metric ResNet against Euclidean ResNet.

Combines the Archimedean place (Euclidean magnitude) with non-Archimedean 
places (p-adic divisibility) to expose combinatorial traps.
"""

from typing import List, Tuple, Set
import math


def get_prime_factors(n: int) -> Set[int]:
    """Extract prime factors of an integer n."""
    factors = set()
    d = 2
    temp = abs(n)
    while d * d <= temp:
        if temp % d == 0:
            factors.add(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.add(temp)
    return factors


def p_adic_valuation(n: int, p: int) -> int:
    """Returns v_p(n), the highest power of prime p dividing n."""
    if n == 0:
        return 999  # Convention: 0 is divisible by infinite power of p
    count = 0
    temp = abs(n)
    while temp % p == 0:
        count += 1
        temp //= p
    return count


class AdeleDistance:
    """
    Computes distance in the Ring of Adeles (A):
        d_A(R, e) = |R - e|_inf * Prod_{p in P} p^(-alpha * v_p(|R - e|))
    """
    def __init__(self, target: int, candidate_pool: List[int], alpha: float = 0.5):
        self.alpha = alpha
        # Prime base: primes dividing the target or candidates
        self.prime_base = set()
        self.prime_base.update(get_prime_factors(target))
        for x in candidate_pool:
            self.prime_base.update(get_prime_factors(x))

    def compute(self, residual: int, candidate: int) -> float:
        diff = residual - candidate
        if diff == 0:
            return 0.0

        # 1. Archimedean (Euclidean) magnitude
        euclidean_mag = abs(diff)

        # 2. Non-Archimedean (p-adic) valuation product
        p_adic_factor = 1.0
        for p in self.prime_base:
            v = p_adic_valuation(euclidean_mag, p)
            if v > 0:
                p_adic_factor *= (p ** (-self.alpha * v))

        # Adele Composite Distance
        return euclidean_mag * p_adic_factor


class AdeleSymbolicResNet:
    """
    Symbolic ResNet using the Adele Metric to route candidate selection.
    Complexity: O(depth * N * |P|) -> Strictly Linear/Polynomial!
    """
    def __init__(self, use_adele: bool = True, alpha: float = 0.5):
        self.use_adele = use_adele
        self.alpha = alpha
        self.eval_count = 0

    def fit(self, candidate_set: List[int], target: int) -> Tuple[bool, int, List[int], List[str]]:
        self.eval_count = 0
        pool = list(candidate_set)
        selected = []
        running_sum = 0
        logs = []

        adele_metric = AdeleDistance(target, candidate_set, self.alpha)

        for layer in range(len(candidate_set)):
            residual = target - running_sum
            if residual == 0:
                return True, self.eval_count, selected, logs

            best_elem = None
            best_dist = float("inf")
            best_idx = -1
            layer_evals = []

            for idx, elem in enumerate(pool):
                self.eval_count += 1
                if elem > residual:
                    continue  # Overshoot constraint for non-negative subset sum

                if self.use_adele:
                    dist = adele_metric.compute(residual, elem)
                else:
                    dist = abs(residual - elem)  # Standard Euclidean

                layer_evals.append((elem, dist))
                if dist < best_dist:
                    best_dist = dist
                    best_elem = elem
                    best_idx = idx

            if best_elem is None:
                break

            layer_evals.sort(key=lambda x: x[1])
            eval_summary = ", ".join([f"{e}(d={d:.2f})" for e, d in layer_evals[:4]])
            logs.append(f"  Layer {layer+1} [Residual={residual:3d}]: Picked {best_elem:2d} | Ranked: [{eval_summary}]")

            selected.append(best_elem)
            running_sum += best_elem
            pool.pop(best_idx)

        return (running_sum == target), self.eval_count, selected, logs


def run_benchmark():
    print("=" * 75)
    print("THE ADELE METRIC RESNET BENCHMARK")
    print("=" * 75)

    test_cases = [
        {
            "name": "TEST 1: Class P (Super-Increasing Powers of 2)",
            "target": 42,
            "set": [1, 2, 4, 8, 16, 32, 64, 128],
            "solution": [32, 8, 2]
        },
        {
            "name": "TEST 2: The Original Frustration Trap (Multiples of 10)",
            "target": 100,
            "set": [51, 50, 50, 20, 10],
            "solution": [50, 50]
        },
        {
            "name": "TEST 3: Multi-Intruder Decoy Trap",
            "target": 100,
            "set": [53, 51, 49, 50, 50, 20, 10],
            "solution": [50, 50]
        },
        {
            "name": "TEST 4: 3-adic / 5-adic Factor Trap",
            "target": 90,
            "set": [46, 30, 30, 30, 15],
            "solution": [30, 30, 30]
        }
    ]

    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        print(f"Target: {tc['target']} | Candidate Set: {tc['set']}")

        # 1. Standard Euclidean ResNet
        euc_model = AdeleSymbolicResNet(use_adele=False)
        euc_ok, euc_evals, euc_res, _ = euc_model.fit(tc['set'], tc['target'])

        # 2. Adele Metric ResNet
        adele_model = AdeleSymbolicResNet(use_adele=True, alpha=0.5)
        adele_ok, adele_evals, adele_res, adele_logs = adele_model.fit(tc['set'], tc['target'])

        print(f"[*] Euclidean ResNet : Success={str(euc_ok):<5} | Evals={euc_evals:<2} | Solution={euc_res}")
        print(f"[*] Adele ResNet     : Success={str(adele_ok):<5} | Evals={adele_evals:<2} | Solution={adele_res}")

        if adele_ok and not euc_ok:
            print("[+] RESULT: Adele Metric bypassed the local minima trap in linear time!")
            print("    Layer-by-Layer Adele Distance Trace:")
            for log in adele_logs:
                print("  ", log)


if __name__ == "__main__":
    run_benchmark()
