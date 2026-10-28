"""Control for continual/gemini/test_symbolic_thousand_brains.py (docs/CONTROLS.md, T1): is "no forgetting" a property of the columns, or
of the three tasks reading three different input features (so the gating key is a task label)?
Same network and procedure as the original; the tasks are the original three, then the same three functions all
reading feature 0.
    python -m continual.claude.controls_continual      (from experiments/symbolics)
"""
import random

from continual.gemini import test_symbolic_thousand_brains as T

XS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
FUNCS = [("x^2", lambda x: x ** 2), ("relu(x)+2", lambda x: max(0.0, x) + 2.0), ("2x+3", lambda x: 2.0 * x + 3.0)]


def run(features):
    random.seed(42)
    net = T.ThousandBrainsSymbolicNet(num_columns=100, num_features=3)
    vec = lambda f, x: [x if i == f else 0.0 for i in range(3)]
    for (name, fn), f in zip(FUNCS, features):
        net.learn_task(f"{name} on x{f}", [(vec(f, x), fn(x)) for x in XS])
    out = []
    for (name, fn), f in zip(FUNCS, features):
        mse = sum((net.predict(vec(f, x)) - fn(x)) ** 2 for x in (4.0, -5.0, 2.5)) / 3
        out.append(f"{name} on x{f}: test MSE {mse:.3g} {'PASS' if mse < 1e-4 else 'FAIL'}")
    return out


if __name__ == "__main__":
    for label, feats in (("disjoint features (as in the original)", (0, 1, 2)), ("one shared feature", (0, 0, 0))):
        res = run(feats)
        print(f"\n=== {label} ===")
        print("\n".join(res))
