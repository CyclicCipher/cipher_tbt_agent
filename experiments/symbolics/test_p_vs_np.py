"""
test_p_vs_np.py
Benchmarking the Symbolic ResNet against P vs NP boundary conditions:
1. Class P Instance: Linearly separable numbers (Super-increasing / Decoupled).
2. NP-Hard Instance: Frustrated, non-separable combinatorial trap.
3. Scalability Test: How candidate evaluations scale as N increases.
"""

from symbolic_p_vs_np_resnet import CombinatorialSymbolicResNet


def run_p_vs_np_experiment():
    print("=" * 70)
    print("SYMBOLIC RESNET vs. P vs NP BOUNDARY EXPERIMENT")
    print("=" * 70)

    solver = CombinatorialSymbolicResNet()

    # -------------------------------------------------------------
    # Test 1: Class P Problem (Decoupled / Super-Increasing Sequence)
    # The elements are additive basis vectors (powers of 2).
    # -------------------------------------------------------------
    print("\n--- TEST 1: CLASS P INSTANCE (DECOUPLED / SUPER-INCREASING) ---")
    separable_set = [1, 2, 4, 8, 16, 32, 64, 128]  # N = 8 -> 2^8 = 256 subsets
    target_1 = 42  # 32 + 8 + 2

    # A. Monolithic Exhaustive Search
    mono_ok, mono_evals, mono_res = solver.fit_monolithic_exhaustive(separable_set, target_1)
    # B. Symbolic ResNet Greedy Search
    res_ok, res_evals, res_res = solver.fit_greedy_residual(separable_set, target_1)

    print(f"Target: {target_1} | Candidate Set: {separable_set}")
    print(f"[*] Monolithic Exhaustive : Success={mono_ok} | Evals={mono_evals} | Solution={mono_res}")
    print(f"[*] Symbolic ResNet       : Success={res_ok} | Evals={res_evals} | Solution={res_res}")
    print(f"[+] Collapse Factor: ResNet evaluated {res_evals} vs {mono_evals} ({mono_evals/res_evals:.1f}x speedup).")
    assert res_ok and sum(res_res) == target_1

    # -------------------------------------------------------------
    # Test 2: The NP-Hard Trap (Frustrated / Strongly Coupled)
    # Target = 100.
    # Set = [51, 50, 50, 20, 10]
    # True solution is 50 + 50 = 100.
    # Greedy ResNet will pick 51 first (leaving residual 49), and get stuck!
    # -------------------------------------------------------------
    print("\n--- TEST 2: THE NP-HARD FRUSTRATION TRAP (LOCAL MINIMA) ---")
    frustrated_set = [51, 50, 50, 20, 10]
    target_2 = 100

    mono_ok, mono_evals, mono_res = solver.fit_monolithic_exhaustive(frustrated_set, target_2)
    res_ok, res_evals, res_res = solver.fit_greedy_residual(frustrated_set, target_2)

    print(f"Target: {target_2} | Candidate Set: {frustrated_set}")
    print(f"[*] Monolithic Exhaustive : Success={mono_ok} | Evals={mono_evals} | Solution={mono_res}")
    print(f"[*] Symbolic ResNet       : Success={res_ok} | Evals={res_evals} | Solution={res_res}")

    if not res_ok:
        print("\n[!] THE RESULT: The Symbolic ResNet FAILED on the frustrated NP-hard instance!")
        print("    Why? Layer 1 greedily picked 51 because abs(100 - 51) = 49 was the closest.")
        print("    Once 51 was locked in, the true solution (50 + 50) was mathematically blocked.")
        print("    To solve it, the system would need BACKTRACKING (exponential search).")

    # -------------------------------------------------------------
    # Test 3: Scaling Curve (N = 5 to N = 20)
    # Demonstrating the mathematical divergence of O(N*d) vs O(2^N)
    # -------------------------------------------------------------
    print("\n--- TEST 3: SCALING CURVES (POLYNOMIAL vs EXPONENTIAL) ---")
    print(f"{'N (Items)':<10} | {'Monolithic Exhaustive O(2^N)':<30} | {'Symbolic ResNet O(N*d)':<25}")
    print("-" * 70)
    for n in [5, 10, 15, 20, 25]:
        mono_complexity = 2 ** n
        resnet_complexity = n * (n // 2)  # Average depth ~ N/2
        print(f"{n:<10} | {mono_complexity:<30,d} | {resnet_complexity:<25,d}")


if __name__ == "__main__":
    run_p_vs_np_experiment()
