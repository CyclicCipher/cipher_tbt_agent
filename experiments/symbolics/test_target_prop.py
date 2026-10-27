"""
test_target_prop.py
A minimal 2-layer Symbolic Network trained via Target Propagation / Inversion.
Goal: Discover a composition of two functions (Layer 1 -> Layer 2) that satisfies 
target input-output pairs without using continuous gradients.
"""

# 1. Define a discrete primitive library with forward functions and their inverses
PRIMITIVES = {
    "add_3": {
        "forward": lambda x: x + 3,
        "inverse": lambda y: y - 3,
    },
    "mul_2": {
        "forward": lambda x: x * 2,
        "inverse": lambda y: y // 2 if y % 2 == 0 else None,
    },
    "square": {
        "forward": lambda x: x ** 2,
        "inverse": lambda y: int(y ** 0.5) if int(y ** 0.5) ** 2 == y else None,
    },
    "sub_1": {
        "forward": lambda x: x - 1,
        "inverse": lambda y: y + 1,
    },
}

class SymbolicLayer:
    def __init__(self, name):
        self.name = name
        self.op_name = None

    def forward(self, x):
        return PRIMITIVES[self.op_name]["forward"](x)

    def invert(self, target_y):
        return PRIMITIVES[self.op_name]["inverse"](target_y)


class TwoLayerSymbolicNet:
    def __init__(self):
        self.l1 = SymbolicLayer("Layer_1")
        self.l2 = SymbolicLayer("Layer_2")

    def forward(self, x):
        h = self.l1.forward(x)
        return self.l2.forward(h)

    def fit_target_propagation(self, dataset):
        print(f"[*] Training on {len(dataset)} examples via Target Propagation...")
        
        # Test candidate operators for Layer 2 and propagate targets back to Layer 1
        for op2_name in PRIMITIVES:
            self.l2.op_name = op2_name
            valid_op2 = True
            intermediate_targets = []

            # Target propagation step: invert Layer 2 to find required hidden targets (h*)
            for x, y_target in dataset:
                h_target = self.l2.invert(y_target)
                if h_target is None:
                    valid_op2 = False
                    break
                intermediate_targets.append((x, h_target))

            if not valid_op2:
                continue

            # Layer 1 local solve: find an op that satisfies (x -> h*)
            for op1_name in PRIMITIVES:
                self.l1.op_name = op1_name
                if all(self.l1.forward(x) == h_target for x, h_target in intermediate_targets):
                    print(f"\n[+] Success! Learned Composition:")
                    print(f"    Input (x) -> [{self.l1.op_name}] -> h -> [{self.l2.op_name}] -> Output (y)")
                    return True

        print("[-] Failed to find a valid composition.")
        return False


if __name__ == "__main__":
    # Target function to discover: y = (x + 3)^2
    # Primitives needed: add_3 followed by square
    data = [(1, 16), (2, 25), (3, 36), (4, 49)]

    print(f"Target dataset: {data}")
    net = TwoLayerSymbolicNet()
    success = net.fit_target_propagation(data)

    if success:
        # Test on unseen input
        test_x = 5
        pred = net.forward(test_x)
        print(f"\nVerification on unseen x={test_x}:")
        print(f"    Expected: {(test_x + 3)**2} | Predicted: {pred}")
        assert pred == 64
        print("[+] Test passed successfully!")
