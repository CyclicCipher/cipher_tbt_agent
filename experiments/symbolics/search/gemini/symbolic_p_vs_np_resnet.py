"""
symbolic_p_vs_np_resnet.py
A Generalized Symbolic ResNet for Combinatorial & Additive Problems.
Attempts to solve Subset Sum by iteratively peeling off elements to cancel
the residual:
    Residual_l = Target - Sum(Selected_Elements_so_far)
"""

from typing import List, Tuple, Optional


class CombinatorialSymbolicResNet:
    def __init__(self, max_depth: Optional[int] = None):
        self.max_depth = max_depth
        self.selected_elements: List[int] = []
        self.eval_count: int = 0

    def fit_greedy_residual(self, candidate_set: List[int], target: int) -> Tuple[bool, int, List[int]]:
        """
        Greedy Symbolic ResNet:
        Layer l selects the candidate that minimizes abs(residual - candidate).
        Complexity: O(depth * N) -> Linear/Polynomial!
        """
        self.selected_elements = []
        self.eval_count = 0
        remaining_pool = list(candidate_set)
        running_sum = 0
        
        limit = self.max_depth or len(candidate_set)

        for layer_idx in range(limit):
            current_residual = target - running_sum
            if current_residual == 0:
                return True, self.eval_count, self.selected_elements

            best_elem = None
            best_res_diff = float("inf")
            best_idx = -1

            # Layer-wise search: find element that best explains/cancels the residual
            for idx, elem in enumerate(remaining_pool):
                self.eval_count += 1
                diff = abs(current_residual - elem)
                if diff < best_res_diff:
                    best_res_diff = diff
                    best_elem = elem
                    best_idx = idx

            # If no element can make progress without overshooting, check if exact match exists
            if best_elem is None or best_res_diff >= abs(current_residual):
                break

            # Additive Skip Connection: accumulate into residual stream
            self.selected_elements.append(best_elem)
            running_sum += best_elem
            remaining_pool.pop(best_idx)

        success = (running_sum == target)
        return success, self.eval_count, self.selected_elements

    def fit_monolithic_exhaustive(self, candidate_set: List[int], target: int) -> Tuple[bool, int, List[int]]:
        """
        Exhaustive Combinatorial Search:
        Explores all 2^N combinations. Guaranteed to find the solution if it exists.
        Complexity: O(2^N) -> Exponential!
        """
        self.eval_count = 0
        n = len(candidate_set)
        
        # Test all 2^N subsets
        for i in range(1 << n):
            self.eval_count += 1
            subset = [candidate_set[j] for j in range(n) if (i & (1 << j))]
            if sum(subset) == target:
                return True, self.eval_count, subset
                
        return False, self.eval_count, []
