"""Is the "adelic" math in the three scripts the real thing? (CONTROLS.md, "The math")
1. test_transformer_grokking.py, the Tate term: does it vanish on CORRECT answers, as a character identity should?
   The real identity (CRT): x/105 = x*(2/3 + 1/5 - 6/7) mod 1, since 1 = 35*2 + 21*1 + 15*(-6).
2. test_transformer_discrete_logic.py, the "2-adic ultrametric": is D2 = 1 - 2^-(tree path length) an ultrametric?
   The real 2-adic distance between branch sequences read as 2-adic integers is 2^-(common prefix length).
3. test_adele_resnet.py, the "adele distance" |x| * prod_p |x|_p^alpha: the real adelic absolute value of a nonzero
   rational is 1 (the product formula). What does alpha = 1 give, and what is alpha = 0.5 in closed form?
"""
import itertools
import math
import random

import test_adele_resnet as A
import test_transformer_discrete_logic as L

# ---- 1. the Tate term -------------------------------------------------------------------------------------------------
N = 105
script = [1 - math.cos(2 * math.pi * x / N - 2 * math.pi * (x % 3 / 3 + x % 5 / 5 + x % 7 / 7)) for x in range(N)]
real = [1 - math.cos(2 * math.pi * x / N - 2 * math.pi * (2 * (x % 3) / 3 + (x % 5) / 5 - 6 * (x % 7) / 7))
        for x in range(N)]
print("1. Tate term on correct answers (should be 0 for every x):")
print(f"   as scripted: mean {sum(script) / N:.3f}, zero for {sum(v < 1e-9 for v in script)}/{N} values of x")
print(f"   with the CRT weights (2, 1, -6): max {max(real):.1e}")

# ---- 2. the tree metric -----------------------------------------------------------------------------------------------
random.seed(0)
violations, triples = 0, 0
for _ in range(300):
    paths, _ = L.parse_tree_to_2adic_paths(L.generate_deep_logic_tree(1, L.MAX_TREE_DEPTH - 1))
    D = L.compute_2adic_tree_distance_matrix(paths, len(paths))
    for i, j, k in itertools.combinations(range(len(paths)), 3):
        triples += 1
        violations += D[i, k] > max(D[i, j], D[j, k]) + 1e-6           # the strong triangle inequality
print(f"2. scripted D2: strong-triangle violations in {violations}/{triples} node triples "
      f"({violations / triples:.1%}); an ultrametric has none")


def prefix(p, q):
    n = 0
    for a, b in zip(p, q):
        if a != b:
            break
        n += 1
    return n


paths, _ = L.parse_tree_to_2adic_paths(["AND", "OR", "1", "0", "NOT", "1"])
print("   e.g. AND(OR(1,0), NOT(1)); real 2-adic distance 2^-(common prefix) from the root AND to every node:",
      [2.0 ** -prefix(paths[0], q) for q in paths[1:]], "(all equal: a node is equally far from all of its subtree)")

# ---- 3. the adele distance --------------------------------------------------------------------------------------------
print("3. the adele distance of a difference x, all of whose primes are in the base:")
for x in (12, 49, 50, 64, 90):
    pool = [x]
    d1 = A.AdeleDistance(x, pool, alpha=1.0).compute(x, 0)
    d5 = A.AdeleDistance(x, pool, alpha=0.5).compute(x, 0)
    print(f"   x = {x:3d}: alpha 1 -> {d1:.3f} (the product formula: always 1);  alpha 0.5 -> {d5:.3f} = sqrt(x) "
          f"{math.sqrt(x):.3f}")
x, pool = 47, [2, 3, 5]
print(f"   x = 47 (prime not in the base): alpha 0.5 -> {A.AdeleDistance(10, pool, 0.5).compute(47, 0):.1f} = x")
