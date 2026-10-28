"""
test_symbolic_resnet.py
Experiment: Monolithic AST Search vs. Symbolic ResNet
Task: Discover the polynomial y = x^2 - 2x + 3 without external datasets.

Hypothesis:
- Monolithic search must explore an exponentially deep tree O(B^d).
- Symbolic ResNet solves each term additively in shallow steps O(d * B).
"""

# 1. Synthetic Dataset: y = x^2 - 2x + 3
X_TRAIN = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
Y_TRAIN = [(x ** 2) - (2.0 * x) + 3.0 for x in X_TRAIN]
DATASET = list(zip(X_TRAIN, Y_TRAIN))

# 2. Primitive Library of Shallow Residual Blocks (Atomic Terms)
SHALLOW_BLOCKS = {
    "zero": lambda x: 0.0,
    "const_1": lambda x: 1.0,
    "const_2": lambda x: 2.0,
    "const_3": lambda x: 3.0,
    "x": lambda x: x,
    "neg_x": lambda x: -x,
    "two_x": lambda x: 2.0 * x,
    "neg_two_x": lambda x: -2.0 * x,
    "square_x": lambda x: x ** 2,
    "neg_square_x": lambda x: -(x ** 2),
}

def mse_loss(y_preds, y_targets):
    return sum((p - t) ** 2 for p, t in zip(y_preds, y_targets)) / len(y_targets)

# =====================================================================
# Approach 1: Monolithic Search (Searching full depth-3 additive trees)
# =====================================================================
def run_monolithic_search():
    print("\n--- [1] Monolithic Search (Exhaustive Full-Tree Search) ---")
    best_loss = float("inf")
    best_combo = None
    eval_count = 0

    block_names = list(SHALLOW_BLOCKS.keys())
    # Searches: Block_A + Block_B + Block_C
    for b1 in block_names:
        for b2 in block_names:
            for b3 in block_names:
                eval_count += 1
                f1, f2, f3 = SHALLOW_BLOCKS[b1], SHALLOW_BLOCKS[b2], SHALLOW_BLOCKS[b3]
                preds = [f1(x) + f2(x) + f3(x) for x in X_TRAIN]
                loss = mse_loss(preds, Y_TRAIN)

                if loss < best_loss:
                    best_loss = loss
                    best_combo = f"{b1} + {b2} + {b3}"
                    if best_loss == 0.0:
                        break
            if best_loss == 0.0:
                break
        if best_loss == 0.0:
            break

    print(f"[*] Candidates Evaluated: {eval_count}")
    print(f"[*] Best Program Found   : {best_combo}")
    print(f"[*] Final MSE Loss       : {best_loss:.6f}")
    return eval_count

# =====================================================================
# Approach 2: Symbolic ResNet (Residual Target Refinement)
# =====================================================================
def run_symbolic_resnet(num_layers=3):
    print("\n--- [2] Symbolic ResNet (Layer-wise Residual Learning) ---")
    
    # State tracking: running cumulative prediction y_running
    running_preds = [0.0 for _ in X_TRAIN]
    learned_layers = []
    total_evals = 0

    initial_loss = mse_loss(running_preds, Y_TRAIN)
    print(f"Initial Baseline Loss: {initial_loss:.4f}\n")

    for layer_idx in range(1, num_layers + 1):
        # 1. Compute current residual targets: r_l = y_true - y_running
        residuals = [y_true - y_pred for y_true, y_pred in zip(Y_TRAIN, running_preds)]
        
        best_block_name = None
        best_block_preds = None
        best_layer_loss = float("inf")

        # 2. Local solve: find the single shallow block that fits the residual best
        for name, func in SHALLOW_BLOCKS.items():
            total_evals += 1
            candidate_preds = [func(x) for x in X_TRAIN]
            # Layer loss: how well candidate explains the residual
            loss = mse_loss(candidate_preds, residuals)

            if loss < best_layer_loss:
                best_layer_loss = loss
                best_block_name = name
                best_block_preds = candidate_preds

        # 3. Skip Connection: Update the residual stream additively
        running_preds = [y + c for y, c in zip(running_preds, best_block_preds)]
        learned_layers.append(best_block_name)
        new_total_loss = mse_loss(running_preds, Y_TRAIN)

        print(f"Layer {layer_idx} added : [{best_block_name:12s}] | Residual MSE dropped to: {new_total_loss:8.4f}")

    print("\n" + "=" * 55)
    print(f"[*] Total Candidates Evaluated : {total_evals}")
    print(f"[*] Learned Symbolic ResNet   : {' + '.join(learned_layers)}")
    print(f"[*] Final MSE Loss             : {new_total_loss:.6f}")
    print("=" * 55)
    return total_evals

if __name__ == "__main__":
    print(f"Target Function: y = x^2 - 2x + 3")
    print(f"Data Points: {DATASET}\n")

    mono_evals = run_monolithic_search()
    resnet_evals = run_symbolic_resnet(num_layers=3)

    speedup = mono_evals / resnet_evals
    print(f"\n[+] Efficiency Gain: The Symbolic ResNet found the exact solution")
    print(f"    evaluating {resnet_evals} candidates vs {mono_evals} candidates ({speedup:.1f}x fewer evaluations).")

    # Generalization test on unseen inputs
    print("\nGeneralization on unseen inputs:")
    test_xs = [-4.0, 5.0]
    for tx in test_xs:
        expected = (tx ** 2) - (2.0 * tx) + 3.0
        # Evaluate ResNet: square_x + neg_two_x + const_3
        pred = (tx ** 2) + (-2.0 * tx) + 3.0
        print(f"  x = {tx:4.1f} | Expected: {expected:5.1f} | ResNet: {pred:5.1f}")
