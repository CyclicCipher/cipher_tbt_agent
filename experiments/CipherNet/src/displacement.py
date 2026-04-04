"""Displacement-based learning via isometry group identification.

The set of learnable operations = the isometry group of the grid cell
geometry. Each operation is a TRANSFORMATION that maps input positions
to output positions. Learning = identifying which transformation is
consistent with all examples. Prediction = applying the transformation.

Transformation types (the group elements):
1. Translation:  c = a + b           (shift by b)
2. Scaling:      c = a * b           (scale by b)
3. Affine:       c = a * b + k       (scale then shift)
4. Power:        c = a ^ b           (repeated scaling)
5. Logarithmic:  c = log_a(b)        (inverse of power)
6. Division:     c = a / b           (inverse scaling)

Each type has:
- check(a, b, c) → bool: is this example consistent with this transformation?
- apply(a, b) → c: compute the output
- solve_b(a, c) → b: inverse query
- solve_a(b, c) → a: inverse query

The column maintains a set of CANDIDATE transformations. Each new
example eliminates candidates that are inconsistent. The surviving
candidate(s) are the learned operation.

This is a finite search over a known group — not regression, not
curve fitting, not hypothesis generation. The group is determined
by the grid cell geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    from .graph import Graph, TEMPORAL, BINDING
except ImportError:
    from graph import Graph, TEMPORAL, BINDING


# ---------------------------------------------------------------------------
# Transformation types (elements of the isometry group)
# ---------------------------------------------------------------------------

class Transformation:
    """Base class for a displacement rule."""
    name: str = "base"

    def fit(self, examples: list[tuple[float, float, float]]) -> bool:
        """Try to fit this transformation to the examples.
        Returns True if consistent with ALL examples."""
        raise NotImplementedError

    def apply(self, a: float, b: float) -> float:
        """Compute c from a and b."""
        raise NotImplementedError

    def solve_b(self, a: float, c: float) -> float | None:
        """Solve for b given a and c."""
        raise NotImplementedError

    def solve_a(self, b: float, c: float) -> float | None:
        """Solve for a given b and c."""
        raise NotImplementedError

    def residual(self, examples: list[tuple[float, float, float]]) -> float:
        """Total residual across all examples."""
        total = 0.0
        for a, b, c in examples:
            try:
                pred = self.apply(a, b)
                total += (pred - c) ** 2
            except (ValueError, ZeroDivisionError, OverflowError):
                total += 1e10
        return total


class Translation(Transformation):
    """c = a + b + k.  (k=0 for pure addition.)"""
    name = "translation"

    def __init__(self):
        self.k = 0.0

    def fit(self, examples):
        if not examples:
            return False
        # k = c - a - b for each example. Must be consistent.
        ks = [c - a - b for a, b, c in examples]
        self.k = ks[0]
        return all(abs(ki - self.k) < 0.01 for ki in ks)

    def apply(self, a, b):
        return a + b + self.k

    def solve_b(self, a, c):
        return c - a - self.k

    def solve_a(self, b, c):
        return c - b - self.k


class Scaling(Transformation):
    """c = k * a * b + offset.  (k=1, offset=0 for pure multiplication.)"""
    name = "scaling"

    def __init__(self):
        self.k = 1.0
        self.offset = 0.0

    def fit(self, examples):
        if len(examples) < 2:
            return False
        # Need to solve: c_i = k * a_i * b_i + offset for all i.
        # Two unknowns (k, offset) → need at least 2 examples.
        # With examples that include (a*b = 0): offset is directly observable.

        # Find an example with a*b != 0 to anchor k.
        nonzero = [(a, b, c) for a, b, c in examples if abs(a * b) > 0.01]
        zero = [(a, b, c) for a, b, c in examples if abs(a * b) < 0.01]

        if zero:
            # From zero examples: c = offset.
            self.offset = zero[0][2]
        else:
            self.offset = 0.0

        if nonzero:
            # k = (c - offset) / (a * b) for nonzero examples.
            ks = [(c - self.offset) / (a * b) for a, b, c in nonzero]
            self.k = sum(ks) / len(ks)
        else:
            self.k = 1.0

        # Check consistency.
        for a, b, c in examples:
            pred = self.k * a * b + self.offset
            if abs(pred - c) > 0.01:
                return False
        return True

    def apply(self, a, b):
        return self.k * a * b + self.offset

    def solve_b(self, a, c):
        if abs(self.k * a) < 1e-12:
            return None
        return (c - self.offset) / (self.k * a)

    def solve_a(self, b, c):
        if abs(self.k * b) < 1e-12:
            return None
        return (c - self.offset) / (self.k * b)


class Power(Transformation):
    """c = a ^ b.  (Exponentiation.)"""
    name = "power"

    def fit(self, examples):
        for a, b, c in examples:
            if a <= 0 or c <= 0:
                # Can't check log for non-positive values easily.
                # Skip — power doesn't apply to negatives simply.
                if a == 0 and c == 0:
                    continue
                return False
            try:
                pred = a ** b
                if abs(pred - c) > max(0.01, abs(c) * 0.001):
                    return False
            except (ValueError, OverflowError):
                return False
        return True

    def apply(self, a, b):
        return a ** b

    def solve_b(self, a, c):
        if a <= 0 or a == 1 or c <= 0:
            return None
        try:
            return math.log(c) / math.log(a)
        except (ValueError, ZeroDivisionError):
            return None

    def solve_a(self, b, c):
        if b == 0 or c <= 0:
            return None
        try:
            return c ** (1.0 / b)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None


class Difference(Transformation):
    """c = a - b + k.  (Subtraction — translation in the opposite direction.)"""
    name = "difference"

    def __init__(self):
        self.k = 0.0

    def fit(self, examples):
        if not examples:
            return False
        ks = [c - a + b for a, b, c in examples]
        self.k = ks[0]
        return all(abs(ki - self.k) < 0.01 for ki in ks)

    def apply(self, a, b):
        return a - b + self.k

    def solve_b(self, a, c):
        return a - c + self.k

    def solve_a(self, b, c):
        return c + b - self.k


class Division(Transformation):
    """c = a / b (+ offset).  (Inverse scaling.)"""
    name = "division"

    def __init__(self):
        self.offset = 0.0

    def fit(self, examples):
        for a, b, c in examples:
            if abs(b) < 1e-12:
                if abs(a) < 1e-12 and abs(c) < 1e-12:
                    continue
                return False
        if not examples:
            return False
        offsets = [c - a / b for a, b, c in examples if abs(b) > 1e-12]
        if not offsets:
            return False
        self.offset = offsets[0]
        return all(abs(o - self.offset) < 0.01 for o in offsets)

    def apply(self, a, b):
        if abs(b) < 1e-12:
            return None
        return a / b + self.offset

    def solve_b(self, a, c):
        denom = c - self.offset
        if abs(denom) < 1e-12:
            return None
        return a / denom

    def solve_a(self, b, c):
        return (c - self.offset) * b


# The full vocabulary of displacement rules.
ALL_TRANSFORMATIONS = [
    Translation,
    Scaling,
    Power,
    Difference,
    Division,
]


# ---------------------------------------------------------------------------
# Learning module: identify the consistent transformation
# ---------------------------------------------------------------------------

class DisplacementLearner:
    """Learns which transformation type is consistent with the data.

    Maintains a candidate set. Each new example eliminates inconsistent
    candidates. When one candidate survives, the operation is identified.
    """

    def __init__(self):
        self.examples: list[tuple[float, float, float]] = []
        self.identified: Transformation | None = None
        self._candidates: list[Transformation] | None = None

    def learn(self, a: float, b: float, c: float) -> dict[str, Any]:
        """Learn one example. Returns info about the learning step."""
        # Predict before learning (to measure error).
        prediction = self.predict({"a": a, "b": b}, "c") if self.examples else None
        error = abs(prediction - c) if prediction is not None else float('inf')

        self.examples.append((a, b, c))

        # Re-evaluate candidates.
        candidates = []
        for TransClass in ALL_TRANSFORMATIONS:
            t = TransClass()
            if t.fit(self.examples):
                candidates.append(t)

        self._candidates = candidates

        if len(candidates) == 1:
            self.identified = candidates[0]
        elif len(candidates) > 1:
            # Multiple candidates — pick the one with lowest residual.
            # But don't commit yet — more examples may disambiguate.
            self.identified = min(candidates, key=lambda t: t.residual(self.examples))
        else:
            # No candidate fits. Fall back to the best-fitting one.
            all_fits = []
            for TransClass in ALL_TRANSFORMATIONS:
                t = TransClass()
                t.fit(self.examples)  # fit even if not perfect
                all_fits.append((t.residual(self.examples), t))
            all_fits.sort()
            if all_fits:
                self.identified = all_fits[0][1]

        return {
            "error": error,
            "n_candidates": len(candidates),
            "candidates": [c.name for c in candidates],
            "identified": self.identified.name if self.identified else None,
        }

    def predict(self, known: dict[str, float], unknown: str) -> float | None:
        """Predict using the identified transformation."""
        if self.identified is None:
            return None

        if unknown == "c":
            a, b = known.get("a"), known.get("b")
            if a is None or b is None:
                return None
            try:
                return self.identified.apply(a, b)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None

        elif unknown == "b":
            a, c = known.get("a"), known.get("c")
            if a is None or c is None:
                return None
            return self.identified.solve_b(a, c)

        elif unknown == "a":
            b, c = known.get("b"), known.get("c")
            if b is None or c is None:
                return None
            return self.identified.solve_a(b, c)

        return None

    def predict_int(self, known: dict[str, float], unknown: str) -> int | None:
        result = self.predict(known, unknown)
        return round(result) if result is not None else None


# ---------------------------------------------------------------------------
# Cortical column (graph-integrated)
# ---------------------------------------------------------------------------

class ManifoldColumn:
    """A cortical column in the graph with displacement-based learning."""

    def __init__(self, graph: Graph, name: str):
        self.graph = graph
        self.name = name
        self.learner = DisplacementLearner()

        sg = f"column:{name}"
        graph.create_subgraph(sg)

        self.input_a = graph.add_node(label=f"{name}:L4:a", subgraph=sg, layer=4)
        self.input_b = graph.add_node(label=f"{name}:L4:b", subgraph=sg, layer=4)
        self.input_c = graph.add_node(label=f"{name}:L4:c", subgraph=sg, layer=4)
        self.process = graph.add_node(label=f"{name}:L23", subgraph=sg, layer=23)
        self.output = graph.add_node(label=f"{name}:L5", subgraph=sg, layer=5)
        self.feedback = graph.add_node(label=f"{name}:L6", subgraph=sg, layer=6)

        for inp in [self.input_a, self.input_b, self.input_c]:
            graph.add_edge(inp, self.process, edge_type=TEMPORAL)
        graph.add_edge(self.process, self.output, edge_type=TEMPORAL)
        graph.add_edge(self.output, self.feedback, edge_type=TEMPORAL)
        for inp in [self.input_a, self.input_b, self.input_c]:
            graph.add_edge(self.feedback, inp, edge_type=TEMPORAL)

    def learn(self, a: float, b: float, c: float) -> dict:
        info = self.learner.learn(a, b, c)
        self.graph.activate(self.feedback, min(1.0, info["error"] / 10.0))
        return info

    def predict(self, known: dict[str, float], unknown: str) -> float | None:
        result = self.learner.predict(known, unknown)
        if result is not None:
            self.graph.activate(self.output, 1.0)
        return result


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Isometry Group Displacement Learning")
    print("  Learning = elimination over a finite group")
    print("=" * 60)

    g = Graph()

    # --- Addition: 3 examples ---
    print("\n--- Addition (3 examples) ---")
    add = ManifoldColumn(g, "add")
    for a, b, c in [(1, 2, 3), (8, 7, 15), (20, 13, 33)]:
        info = add.learn(a, b, c)
        print(f"  ({a},{b},{c}): candidates={info['candidates']}, identified={info['identified']}, error={info['error']:.2f}")

    add_tests = [
        ({"a": 3, "b": 4}, "c", 7),
        ({"a": 307, "b": 456}, "c", 763),
        ({"a": 1000000, "b": 1}, "c", 1000001),
        ({"a": 3, "b": -5}, "c", -2),
        ({"a": -10, "b": -20}, "c", -30),
        ({"a": 3.5, "b": 2.5}, "c", 6.0),
        ({"a": 3, "c": 7}, "b", 4),
        ({"b": 456, "c": 763}, "a", 307),
    ]
    n = sum(1 for k, u, e in add_tests
            if (r := add.predict(k, u)) is not None and abs(r - e) < 0.01)
    for known, unk, exp in add_tests:
        r = add.predict(known, unk)
        ok = r is not None and abs(r - exp) < 0.01
        ks = ", ".join(f"{k}={v}" for k, v in sorted(known.items()))
        print(f"  {ks} -> {unk}={r:.4f}  {'OK' if ok else f'WRONG (exp {exp})'}" if r else
              f"  {ks} -> {unk}=None WRONG")
    print(f"  Score: {n}/{len(add_tests)}")

    # --- Multiplication: 8 examples ---
    print("\n--- Multiplication (8 examples) ---")
    mul = ManifoldColumn(g, "mul")
    for a, b, c in [(3,4,12),(2,5,10),(7,3,21),(3,-2,-6),(-4,3,-12),(-3,-4,12),(0,7,0),(5,0,0)]:
        info = mul.learn(a, b, c)
        print(f"  ({a},{b},{c}): candidates={info['candidates']}, identified={info['identified']}")

    mul_tests = [
        ({"a": 6, "b": 7}, "c", 42),
        ({"a": 100, "b": 100}, "c", 10000),
        ({"a": -6, "b": -7}, "c", 42),
        ({"a": 0.5, "b": 4}, "c", 2.0),
        ({"a": 6, "c": 42}, "b", 7),
        ({"a": -3, "c": 12}, "b", -4),
    ]
    n_mul = sum(1 for k, u, e in mul_tests
                if (r := mul.predict(k, u)) is not None and abs(r - e) < 0.01)
    for known, unk, exp in mul_tests:
        r = mul.predict(known, unk)
        ok = r is not None and abs(r - exp) < 0.01
        ks = ", ".join(f"{k}={v}" for k, v in sorted(known.items()))
        print(f"  {ks} -> {unk}={r:.4f}  {'OK' if ok else f'WRONG (exp {exp})'}" if r else
              f"  {ks} -> {unk}=None WRONG")
    print(f"  Score: {n_mul}/{len(mul_tests)}")

    # --- Power: 4 examples ---
    print("\n--- Power (4 examples) ---")
    pw = ManifoldColumn(g, "pow")
    for a, b, c in [(2, 3, 8), (3, 2, 9), (2, 10, 1024), (5, 3, 125)]:
        info = pw.learn(a, b, c)
        print(f"  ({a},{b},{c}): candidates={info['candidates']}, identified={info['identified']}")

    pow_tests = [
        ({"a": 2, "b": 8}, "c", 256),
        ({"a": 10, "b": 3}, "c", 1000),
        ({"a": 2, "c": 1024}, "b", 10),
        ({"b": 3, "c": 125}, "a", 5),
    ]
    n_pow = sum(1 for k, u, e in pow_tests
                if (r := pw.predict(k, u)) is not None and abs(r - e) < max(0.01, abs(e)*0.001))
    for known, unk, exp in pow_tests:
        r = pw.predict(known, unk)
        ok = r is not None and abs(r - exp) < max(0.01, abs(exp) * 0.001)
        ks = ", ".join(f"{k}={v}" for k, v in sorted(known.items()))
        print(f"  {ks} -> {unk}={r:.4f}  {'OK' if ok else f'WRONG (exp {exp})'}" if r else
              f"  {ks} -> {unk}=None WRONG")
    print(f"  Score: {n_pow}/{len(pow_tests)}")

    # --- Subtraction: 3 examples ---
    print("\n--- Subtraction (3 examples) ---")
    sub = ManifoldColumn(g, "sub")
    for a, b, c in [(10, 3, 7), (20, 8, 12), (5, 5, 0)]:
        info = sub.learn(a, b, c)
        print(f"  ({a},{b},{c}): candidates={info['candidates']}, identified={info['identified']}")

    sub_tests = [
        ({"a": 100, "b": 37}, "c", 63),
        ({"a": 5, "b": 10}, "c", -5),
        ({"a": 100, "c": 63}, "b", 37),
    ]
    n_sub = sum(1 for k, u, e in sub_tests
                if (r := sub.predict(k, u)) is not None and abs(r - e) < 0.01)
    for known, unk, exp in sub_tests:
        r = sub.predict(known, unk)
        ok = r is not None and abs(r - exp) < 0.01
        ks = ", ".join(f"{k}={v}" for k, v in sorted(known.items()))
        print(f"  {ks} -> {unk}={r:.4f}  {'OK' if ok else f'WRONG (exp {exp})'}" if r else
              f"  {ks} -> {unk}=None WRONG")
    print(f"  Score: {n_sub}/{len(sub_tests)}")

    # --- Graph ---
    print(f"\n--- Graph ---")
    print(f"  {g.summary()}")
