"""
test_symbolic_thousand_brains.py (With Contextual Attention Gating)
A 100-Column Symbolic Thousand Brains Architecture for Continual Learning.

New Feature:
- Apical Dendrite Gating (Attention mechanism via Query-Key matching):
  Inhibits columns whose context keys do not match the current input query,
  eliminating the 3-way tie.
"""

import math
import random

random.seed(42)

PRIMITIVES = {
    "no_op": lambda x: 0.0,
    "identity": lambda x: x,
    "square": lambda x: x ** 2,
    "neg": lambda x: -x,
    "relu": lambda x: max(0.0, x),
    "double": lambda x: 2.0 * x,
    "const_1": lambda x: 1.0,
    "const_2": lambda x: 2.0,
    "const_3": lambda x: 3.0,
}

def mse(preds, targets):
    return sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(targets)

def normalize_vector(vec):
    norm = math.sqrt(sum(v ** 2 for v in vec))
    if norm < 1e-6:
        return [0.0 for _ in vec]
    return [v / norm for v in vec]


class CorticalColumn:
    def __init__(self, column_id, num_features=3, max_layers=2):
        self.id = column_id
        self.receptive_feature = random.randint(0, num_features - 1)
        self.max_layers = max_layers
        self.layers = []
        self.is_committed = False
        self.committed_task = None
        
        # Contextual Key (The environmental profile under which this column activates)
        self.context_key = None

    def forward(self, x_vector):
        x_val = x_vector[self.receptive_feature]
        running_val = 0.0
        for op_name in self.layers:
            running_val += PRIMITIVES[op_name](x_val)
        return running_val

    def fit_local(self, dataset, max_error_threshold=1e-4):
        """2-step lookahead search to avoid greedy local minima."""
        if self.is_committed:
            return float("inf"), []

        x_vals = [sample[0][self.receptive_feature] for sample in dataset]
        y_targets = [sample[1] for sample in dataset]

        best_loss = float("inf")
        best_layers = []

        # 1. Test 1-layer solutions (with no_op for layer 2)
        for op1 in PRIMITIVES:
            preds = [PRIMITIVES[op1](x) for x in x_vals]
            loss = mse(preds, y_targets)
            if loss < best_loss:
                best_loss = loss
                best_layers = [op1, "no_op"]
                if best_loss <= max_error_threshold:
                    return best_loss, best_layers

        # 2. Test 2-layer combinations (Lookahead)
        for op1 in PRIMITIVES:
            p1 = [PRIMITIVES[op1](x) for x in x_vals]
            for op2 in PRIMITIVES:
                p2 = [PRIMITIVES[op2](x) for x in x_vals]
                combined = [a + b for a, b in zip(p1, p2)]
                loss = mse(combined, y_targets)
                if loss < best_loss:
                    best_loss = loss
                    best_layers = [op1, op2]
                    if best_loss <= max_error_threshold:
                        return best_loss, best_layers

        return best_loss, best_layers


class ThousandBrainsSymbolicNet:
    def __init__(self, num_columns=100, num_features=3):
        self.num_features = num_features
        self.columns = [CorticalColumn(i, num_features=num_features) for i in range(num_columns)]

    def learn_task(self, task_name, dataset, cohort_size=6):
        print(f"\n[*] Learning {task_name} (Dataset size: {len(dataset)})...")
        candidates = []
        for col in self.columns:
            if not col.is_committed:
                loss, layers = col.fit_local(dataset)
                if loss < 1e-4:
                    candidates.append((col, loss, layers))

        if len(candidates) == 0:
            print(f"[-] ERROR: Zero columns were able to learn {task_name}!")
            return False

        recruited = candidates[:cohort_size]

        # Compute the Context Key for this task (Sensory Energy Profile)
        raw_key = [0.0 for _ in range(self.num_features)]
        for x_vec, _ in dataset:
            for i, val in enumerate(x_vec):
                raw_key[i] += abs(val)
        normalized_key = normalize_vector(raw_key)

        for col, loss, layers in recruited:
            col.layers = layers
            col.is_committed = True
            col.committed_task = task_name
            col.context_key = normalized_key  # Store apical key

        print(f"[+] Successfully recruited {len(recruited)} columns for {task_name}.")
        sample_col = recruited[0][0]
        print(f"    Sample Column #{sample_col.id} Receptive Field: Feature {sample_col.receptive_feature}")
        print(f"    Sample Column #{sample_col.id} Architecture   : {' + '.join(sample_col.layers)}")
        print(f"    Learned Context Key (K) : {[round(k, 3) for k in normalized_key]}")
        return True

    def predict(self, x_vector, consensus_tolerance=0.1, attention_threshold=0.5):
        """
        Inference with Apical Attention Gating + Lateral Consensus Voting.
        """
        # 1. Construct the Query Vector from incoming input
        query = normalize_vector([abs(v) for v in x_vector])

        active_votes = []

        # 2. Apical Gating: Query-Key matching
        for col in self.columns:
            if col.is_committed:
                # Dot-product attention score: Q . K
                attn_score = sum(q * k for q, k in zip(query, col.context_key))

                # Apical Inhibition: Silence column if context doesn't match
                if attn_score >= attention_threshold:
                    pred = col.forward(x_vector)
                    active_votes.append((col, pred, attn_score))

        if not active_votes:
            return 0.0

        # 3. Lateral Consensus Voting among the uninhibited columns
        best_pred = None
        max_weighted_agreement = -1.0

        for col_i, pred_i, attn_i in active_votes:
            # Agreement weighted by attention score
            weighted_agreement = sum(
                attn_j for _, pred_j, attn_j in active_votes 
                if abs(pred_i - pred_j) <= consensus_tolerance
            )
            if weighted_agreement > max_weighted_agreement:
                max_weighted_agreement = weighted_agreement
                best_pred = pred_i

        return best_pred


if __name__ == "__main__":
    net = ThousandBrainsSymbolicNet(num_columns=100, num_features=3)

    task1_train = [([x, 0.0, 0.0], x ** 2) for x in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]]
    task1_test  = [([4.0, 0.0, 0.0], 16.0), ([-5.0, 0.0, 0.0], 25.0)]

    task2_train = [([0.0, x, 0.0], max(0.0, x) + 2.0) for x in [-3.0, -1.0, 0.0, 2.0, 4.0]]
    task2_test  = [([0.0, 3.0, 0.0], 5.0), ([0.0, -4.0, 0.0], 2.0)]

    task3_train = [([0.0, 0.0, x], 2.0 * x + 3.0) for x in [-2.0, -1.0, 0.0, 1.0, 2.0]]
    task3_test  = [([0.0, 0.0, 5.0], 13.0), ([0.0, 0.0, -3.0], -3.0)]

    # Sequential Training (Zero Replay)
    net.learn_task("Task_1 (x0^2)", task1_train, cohort_size=6)
    net.learn_task("Task_2 (relu(x1) + 2)", task2_train, cohort_size=6)
    net.learn_task("Task_3 (2*x2 + 3)", task3_train, cohort_size=6)

    # Column Summary
    committed_counts = {}
    for col in net.columns:
        if col.is_committed:
            committed_counts[col.committed_task] = committed_counts.get(col.committed_task, 0) + 1
    uncommitted = sum(1 for c in net.columns if not c.is_committed)

    print("\n" + "=" * 60)
    print("COLUMN ALLOCATION SUMMARY (100 Columns Total):")
    for task, count in committed_counts.items():
        print(f"  {task:25s} : {count} columns locked")
    print(f"  {'Uncommitted (Plastic)':25s} : {uncommitted} columns free for future tasks")
    print("=" * 60)

    # Continual Learning Test
    print("\n--- RUNNING CONTINUAL LEARNING EVALUATION (ZERO REPLAY) ---")
    all_passed = True
    eval_tasks = [
        ("Task 1", task1_test),
        ("Task 2", task2_test),
        ("Task 3", task3_test),
    ]

    for task_name, test_set in eval_tasks:
        errors = []
        for x_vec, y_true in test_set:
            y_pred = net.predict(x_vec)
            err = (y_pred - y_true) ** 2
            errors.append(err)
            print(f"[{task_name}] Input: {x_vec} | Expected: {y_true:5.1f} | Consensus Pred: {y_pred:5.1f}")

        task_mse = sum(errors) / len(errors)
        if task_mse < 1e-4:
            print(f"  ==> {task_name} STATUS: [PASS] (No Forgetting, MSE: {task_mse:.6f})\n")
        else:
            print(f"  ==> {task_name} STATUS: [FAIL] (Forgetting Detected, MSE: {task_mse:.6f})\n")
            all_passed = False

    # Dynamic Depth Test
    task1_cols = [c for c in net.columns if c.committed_task == "Task_1 (x0^2)"]
    has_used_noop = any("no_op" in c.layers for c in task1_cols)

    print("=" * 60)
    if has_used_noop:
        print("[PASS] Dynamic Depth Test: Columns for Task 1 chose 'no_op' for layer 2.")
    else:
        print("[FAIL] Dynamic Depth Test: Columns failed to utilize 'no_op'.")

    if all_passed:
        print("[PASS] Continual Learning Test: All 3 tasks preserved with 0.0% forgetting!")
    else:
        print("[FAIL] Continual Learning Test: Catastrophic forgetting occurred.")
    print("=" * 60)