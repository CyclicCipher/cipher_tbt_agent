
"""
test_fuzzy_binary_tree.py
Symbolic AST Search with:
1. Binary operations (add, mul, sub)
2. Nonlinear unary operations (square, relu)
3. Fuzzy Execution (continuous MSE loss)
4. Equivalence-class pruning (canonical representation of commutative ops)
"""

import math

# 1. Discrete Primitive Library
UNARY_OPS = {
    "square": lambda x: x ** 2,
    "relu": lambda x: max(0.0, x),
    "neg": lambda x: -x,
}

BINARY_OPS = {
    "add": (lambda a, b: a + b, True),   # (func, is_commutative)
    "sub": (lambda a, b: a - b, False),
    "mul": (lambda a, b: a * b, True),
}

CONSTANTS = [1.0, 2.0, 3.0]


class ASTNode:
    def __init__(self, op_type, name, left=None, right=None, value=None):
        self.op_type = op_type  # 'var', 'const', 'unary', 'binary'
        self.name = name
        self.left = left
        self.right = right
        self.value = value

    def eval(self, env):
        if self.op_type == "var":
            return float(env[self.name])
        elif self.op_type == "const":
            return float(self.value)
        elif self.op_type == "unary":
            return UNARY_OPS[self.name](self.left.eval(env))
        elif self.op_type == "binary":
            func = BINARY_OPS[self.name][0]
            return func(self.left.eval(env), self.right.eval(env))

    def to_string(self):
        if self.op_type == "var":
            return self.name
        elif self.op_type == "const":
            return str(int(self.value) if self.value.is_integer() else self.value)
        elif self.op_type == "unary":
            return f"{self.name}({self.left.to_string()})"
        elif self.op_type == "binary":
            return f"({self.left.to_string()} {self.name} {self.right.to_string()})"

    # Equivalence-class hashing (Canonical Form)
    def canonical_hash(self):
        if self.op_type in ("var", "const"):
            return self.to_string()
        elif self.op_type == "unary":
            return f"{self.name}({self.left.canonical_hash()})"
        elif self.op_type == "binary":
            is_comm = BINARY_OPS[self.name][1]
            left_h = self.left.canonical_hash()
            right_h = self.right.canonical_hash()
            if is_comm and left_h > right_h:
                left_h, right_h = right_h, left_h  # Canonical sort: add(b, a) -> add(a, b)
            return f"({left_h} {self.name} {right_h})"


def compute_fuzzy_loss(tree, dataset):
    """Fuzzy evaluation: computes Mean Squared Error (MSE)."""
    total_error = 0.0
    for x_val, y_true in dataset:
        try:
            y_pred = tree.eval({"x": x_val})
            total_error += (y_pred - y_true) ** 2
        except (OverflowError, ZeroDivisionError, ValueError):
            return float("inf")
    return total_error / len(dataset)


def generate_candidate_trees():
    """Generates a bounded hypothesis space up to depth-2 ASTs."""
    var_x = ASTNode("var", "x")
    leaves = [var_x] + [ASTNode("const", "c", value=c) for c in CONSTANTS]

    trees = []
    seen_hashes = set()

    def register(tree):
        h = tree.canonical_hash()
        # Identity pruning: prune trivial ops like mul(x, 1) or add(x, 0)
        if "mul 1" in h or "1 mul" in h:
            return
        if h not in seen_hashes:
            seen_hashes.add(h)
            trees.append(tree)

    # 1. Leaves
    for leaf in leaves:
        register(leaf)

    # 2. Depth 1: Unary over leaves
    for leaf in leaves:
        for u_name in UNARY_OPS:
            register(ASTNode("unary", u_name, left=leaf))

    # 3. Depth 1: Binary over leaves
    for i, left in enumerate(leaves):
        for right in leaves:
            for b_name in BINARY_OPS:
                register(ASTNode("binary", b_name, left=left, right=right))

    # 4. Allow nested unary: Unary(Unary(x)) -> e.g. square(relu(x))
    nested_unaries = []
    for u1 in UNARY_OPS:
        for u2 in UNARY_OPS:
            nested_unaries.append(ASTNode("unary", u1, left=ASTNode("unary", u2, left=var_x)))

    # 5. Combine nested unary with leaves in binary ops
    for n_tree in nested_unaries:
        for leaf in leaves:
            for b_name in BINARY_OPS:
                register(ASTNode("binary", b_name, left=n_tree, right=leaf))

    return trees


if __name__ == "__main__":
    # Target function: Nonlinear with binary composition: y = relu(x)^2 + 2
    # For positive x, this behaves as x^2 + 2. For negative x, it floors to 2.
    dataset = [(-2.0, 2.0), (-1.0, 2.0), (0.0, 2.0), (1.0, 3.0), (2.0, 6.0), (3.0, 11.0)]

    print(f"[*] Target dataset: {dataset}")
    candidates = generate_candidate_trees()
    print(f"[*] Generated {len(candidates)} distinct canonical AST candidates after equivalence pruning.")

    best_tree = None
    best_loss = float("inf")

    # Evaluate using fuzzy execution
    for tree in candidates:
        loss = compute_fuzzy_loss(tree, dataset)
        if loss < best_loss:
            best_loss = loss
            best_tree = tree
            print(f"    [Candidate] Loss: {loss:.4f} -> {tree.to_string()}")
            if math.isclose(best_loss, 0.0, abs_tol=1e-5):
                break

    print("\n" + "=" * 50)
    print(f"[+] Best Discovered Program : {best_tree.to_string()}")
    print(f"[+] Final Fuzzy Loss (MSE) : {best_loss:.6f}")
    print("=" * 50)

    # Generalization test on unseen points
    test_inputs = [-3.0, 4.0]
    for tx in test_inputs:
        pred = best_tree.eval({"x": tx})
        expected = (max(0.0, tx) ** 2) + 2.0
        print(f"Eval on unseen x={tx:4.1f} | Expected: {expected:5.1f} | Predicted: {pred:5.1f}")