"""
test_p_vs_np_attnres.py
Benchmarking Attention Residuals (AttnRes) against the NP-Hard Frustration Trap.

Compares three architectures on Target = 100, Set = [51, 50, 50, 20, 10]:
  1. Monolithic Exhaustive Search  : Guaranteed exact O(2^N)
  2. Standard Greedy ResNet        : Additive residual stream O(N*d)
  3. AttnRes Dynamic Depth ResNet  : Softmax depth attention over layer history
  4. Continuous Attentive AttnRes  : Continuous gradient relaxation over attention logits
"""

from typing import List, Tuple
import numpy as np


class CombinatorialAttnResBenchmark:

    # -------------------------------------------------------------
    # 1. Monolithic Exhaustive (O(2^N))
    # -------------------------------------------------------------
    @staticmethod
    def solve_exhaustive(candidates: List[int], target: int) -> Tuple[bool, int, List[int]]:
        n = len(candidates)
        evals = 0
        for i in range(1 << n):
            evals += 1
            subset = [candidates[j] for j in range(n) if (i & (1 << j))]
            if sum(subset) == target:
                return True, evals, subset
        return False, evals, []

    # -------------------------------------------------------------
    # 2. Standard Greedy Additive ResNet (O(N*d))
    # -------------------------------------------------------------
    @staticmethod
    def solve_standard_resnet(candidates: List[int], target: int) -> Tuple[bool, int, List[int]]:
        pool = list(candidates)
        selected = []
        running_sum = 0
        evals = 0

        for _ in range(len(candidates)):
            residual = target - running_sum
            if residual == 0:
                return True, evals, selected

            best_elem, best_diff, best_idx = None, float("inf"), -1
            for idx, elem in enumerate(pool):
                evals += 1
                diff = abs(residual - elem)
                if diff < best_diff:
                    best_diff, best_elem, best_idx = diff, elem, idx

            if best_elem is None or best_diff >= abs(residual):
                break

            selected.append(best_elem)
            running_sum += best_elem
            pool.pop(best_idx)

        return (running_sum == target), evals, selected

    # -------------------------------------------------------------
    # 3. Sequential AttnRes: Depth-Wise Attention over Layer History
    # -------------------------------------------------------------
    @staticmethod
    def solve_sequential_attnres(candidates: List[int], target: int) -> Tuple[bool, int, List[int], str]:
        """
        At each layer l, AttnRes computes dynamic attention weights alpha over
        all selected past layers {v_1, ..., v_l} to decide how much to keep/suppress them.
        """
        pool = list(candidates)
        selected = []
        evals = 0
        log = []

        for layer in range(len(candidates)):
            # Residual computed against the current best attention-weighted sum
            current_sum = sum(selected)
            residual = target - current_sum
            if residual == 0:
                return True, evals, selected, "\n".join(log)

            # Query candidate pool
            best_elem, best_diff, best_idx = None, float("inf"), -1
            for idx, elem in enumerate(pool):
                evals += 1
                diff = abs(residual - elem)
                if diff < best_diff:
                    best_diff, best_elem, best_idx = diff, elem, idx

            if best_elem is None:
                break

            selected.append(best_elem)
            pool.pop(best_idx)

            # Depth Attention Mechanism:
            # Re-evaluate all chosen elements so far with attention routing
            # Can attention 'suppress' (weight -> 0) an earlier mistake?
            # We test candidate attention configurations (binary mask over past choices)
            best_mask = None
            min_layer_err = float("inf")
            n_sel = len(selected)
            for mask_int in range(1, 1 << n_sel):
                evals += 1
                active_sum = sum(selected[k] for k in range(n_sel) if (mask_int & (1 << k)))
                err = abs(target - active_sum)
                if err < min_layer_err:
                    min_layer_err = err
                    best_mask = mask_int

            active_elements = [selected[k] for k in range(n_sel) if (best_mask & (1 << k))]
            log.append(
                f"  Layer {layer+1}: Selected {best_elem} | Active Pool: {selected} "
                f"| Attn Attenuation -> Active: {active_elements} (Sum={sum(active_elements)}, Error={min_layer_err})"
            )

            if min_layer_err == 0:
                return True, evals, active_elements, "\n".join(log)

            # If error is stuck or diverging, break
            if len(selected) > 1 and min_layer_err >= abs(residual) and best_diff > 0:
                break

        final_sum = sum(selected)
        return (final_sum == target), evals, selected, "\n".join(log)

    # -------------------------------------------------------------
    # 4. Continuous Attentive Relaxation (Softmax AttnRes via Adam)
    # -------------------------------------------------------------
    @staticmethod
    def solve_continuous_attnres(candidates: List[int], target: int, steps: int = 500) -> Tuple[bool, int, List[float]]:
        """
        Attempts to find a soft attention vector over candidates using gradient descent:
            Loss = (target - sum(alpha_i * c_i))^2 + penalty * (alpha_i * (1 - alpha_i))
        """
        c = np.array(candidates, dtype=float)
        logits = np.zeros(len(candidates))  # Initial uniform attention
        lr = 0.1
        evals = 0

        m, v = np.zeros_like(logits), np.zeros_like(logits)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for step in range(1, steps + 1):
            evals += 1
            # Sigmoid independent attention gate per candidate
            alpha = 1.0 / (1.0 + np.exp(-logits))
            pred = np.sum(alpha * c)
            err = pred - target
            
            # Loss: MSE + binarization barrier penalty
            loss = err ** 2
            dL_dalpha = 2.0 * err * c + 0.1 * (1.0 - 2.0 * alpha)
            dL_dlogits = dL_dalpha * alpha * (1.0 - alpha)

            m = beta1 * m + (1 - beta1) * dL_dlogits
            v = beta2 * v + (1 - beta2) * (dL_dlogits ** 2)
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            logits -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # Check if discrete threshold hits exact target
            binary_selection = (alpha > 0.5).astype(int)
            if np.sum(binary_selection * c) == target:
                return True, evals, alpha.round(2).tolist()

        final_alpha = (1.0 / (1.0 + np.exp(-logits))).round(2)
        final_discrete = (final_alpha > 0.5).astype(int)
        return (np.sum(final_discrete * c) == target), evals, final_alpha.tolist()


def run_p_vs_np_attnres_experiment():
    print("=" * 70)
    print("ATTENTION RESIDUALS (AttnRes) vs. P vs NP BOUNDARY EXPERIMENT")
    print("=" * 70)

    benchmark = CombinatorialAttnResBenchmark()

    # -------------------------------------------------------------
    # TEST 1: Decoupled / Super-Increasing (Class P)
    # -------------------------------------------------------------
    print("\n--- TEST 1: CLASS P INSTANCE (DECOUPLED / SUPER-INCREASING) ---")
    easy_set = [1, 2, 4, 8, 16, 32, 64, 128]
    target_1 = 42  # 32 + 8 + 2

    ok_mono, ev_mono, res_mono = benchmark.solve_exhaustive(easy_set, target_1)
    ok_std, ev_std, res_std = benchmark.solve_standard_resnet(easy_set, target_1)
    ok_attn, ev_attn, res_attn, _ = benchmark.solve_sequential_attnres(easy_set, target_1)

    print(f"Target: {target_1} | Set: {easy_set}")
    print(f"[*] Monolithic Exhaustive : Success={ok_mono:<5} | Evals={ev_mono:<4} | Solution={res_mono}")
    print(f"[*] Standard ResNet       : Success={ok_std:<5} | Evals={ev_std:<4} | Solution={res_std}")
    print(f"[*] AttnRes Sequential    : Success={ok_attn:<5} | Evals={ev_attn:<4} | Solution={res_attn}")

    # -------------------------------------------------------------
    # TEST 2: The NP-Hard Trap (Strongly Coupled / Frustrated)
    # -------------------------------------------------------------
    print("\n--- TEST 2: THE NP-HARD FRUSTRATION TRAP ---")
    trap_set = [51, 50, 50, 20, 10]
    target_2 = 100  # True solution: 50 + 50 = 100

    print(f"Target: {target_2} | Candidate Set: {trap_set}")
    print(f"Ground Truth Solution: [50, 50]")

    ok_mono2, ev_mono2, res_mono2 = benchmark.solve_exhaustive(trap_set, target_2)
    ok_std2, ev_std2, res_std2 = benchmark.solve_standard_resnet(trap_set, target_2)
    ok_attn2, ev_attn2, res_attn2, log_attn = benchmark.solve_sequential_attnres(trap_set, target_2)
    ok_cont2, ev_cont2, alpha_cont2 = benchmark.solve_continuous_attnres(trap_set, target_2)

    print(f"\n[*] 1. Monolithic Exhaustive : Success={ok_mono2:<5} | Evals={ev_mono2:<4} | Solution={res_mono2}")
    print(f"[*] 2. Standard ResNet       : Success={ok_std2:<5} | Evals={ev_std2:<4} | Solution={res_std2}")
    print(f"[*] 3. AttnRes (Sequential)  : Success={ok_attn2:<5} | Evals={ev_attn2:<4} | Solution={res_attn2}")
    print(f"[*] 4. AttnRes (Continuous)  : Success={ok_cont2:<5} | Evals={ev_cont2:<4} | Attention={alpha_cont2}")

    print("\n--- [AttnRes Sequential Diagnostic Trace] ---")
    print(log_attn)


if __name__ == "__main__":
    run_p_vs_np_attnres_experiment()
