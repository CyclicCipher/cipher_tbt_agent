"""Number line world — arithmetic as spatial navigation.

Every number is a position on a line. Successor connects adjacent
positions. Arithmetic relations are PATHS on this graph.

    sum(a, b, c)     = there are exactly b successor steps from a to c
    product(a, b, c) = there are exactly b paths of length a from 0 to c
                       (equivalently: walk a steps, b times, from 0)

No inputs, no outputs. No functions. Just positions and paths.
The system discovers the rules by correlating observed relation triples
with path structure in the successor graph.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import sys, os
_CATPLAN_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _CATPLAN_DIR not in sys.path:
    sys.path.insert(0, _CATPLAN_DIR)

from worlds.base import Observation, Transition, Demonstration


class NumberLine:
    """A world of positions connected by successor."""

    def __init__(self, max_pos: int = 50):
        self.max_pos = max_pos
        # The successor graph IS the number line.
        # succ[i] = i + 1 for all positions.
        self._known_relations: set[tuple[str, tuple[int, int, int]]] = set()

    def reset(self):
        self._known_relations.clear()

    def walk(self, start: int, steps: int) -> int | None:
        """Walk `steps` successor steps from `start`. Returns destination.

        This is the ONLY primitive operation — spatial navigation.
        """
        pos = start
        for _ in range(steps):
            if pos + 1 > self.max_pos:
                return None
            pos = pos + 1
        return pos

    def walk_back(self, end: int, steps: int) -> int | None:
        """Walk backward: find start such that walk(start, steps) = end."""
        pos = end
        for _ in range(steps):
            pos = pos - 1
            if pos < 0:
                return None
        return pos

    def observe(self) -> Observation:
        facts = set()

        # Positions exist.
        for p in range(self.max_pos + 1):
            facts.add(("position", (str(p),), True))

        # Successor structure.
        for p in range(self.max_pos):
            facts.add(("succ", (str(p), str(p + 1)), True))

        # Known relations.
        for rel_name, (a, b, c) in self._known_relations:
            facts.add((rel_name, (str(a), str(b), str(c)), True))

        return Observation(facts=frozenset(facts))

    def establish_sum(self, a: int, b: int) -> int | None:
        """Establish sum(a, b, c) by walking b steps from a."""
        c = self.walk(a, b)
        if c is not None:
            self._known_relations.add(("sum", (a, b, c)))
            self._known_relations.add(("sum", (b, a, c)))  # commutativity
        return c

    def establish_product(self, a: int, b: int) -> int | None:
        """Establish product(a, b, c) by walking a steps, b times, from 0."""
        pos = 0
        for _ in range(b):
            pos = self.walk(pos, a)
            if pos is None:
                return None
        self._known_relations.add(("product", (a, b, pos)))
        self._known_relations.add(("product", (b, a, pos)))  # commutativity
        return pos

    def generate_demonstrations(self, n: int = 50, n_steps: int = 15,
                                 seed: int = 42) -> list[Demonstration]:
        import random
        rng = random.Random(seed)
        demos = []

        for d in range(n):
            self.reset()
            transitions = []

            for _ in range(n_steps):
                a = rng.randint(0, 12)
                b = rng.randint(0, 12)
                op = rng.choice(["sum", "product"])

                before = self.observe()

                if op == "sum":
                    c = self.establish_sum(a, b)
                    action = "establish_sum"
                else:
                    c = self.establish_product(a, b)
                    action = "establish_product"

                if c is None:
                    continue

                after = self.observe()
                transitions.append(Transition(
                    before=before,
                    action=action,
                    action_args=(str(a), str(b)),
                    after=after,
                ))

            demos.append(Demonstration(transitions=transitions))

        return demos


# ---------------------------------------------------------------------------
# Rule discovery: correlate relation triples with path structure
# ---------------------------------------------------------------------------

class RuleDiscoverer:
    """Discover arithmetic rules from observed relation triples
    by correlating with path structure on the number line.

    The key idea: for each observed relation (e.g., sum(3,5,8)),
    check if the triple is consistent with a PATH hypothesis:
    "c is b steps from a in the successor graph."

    If ALL observed triples of a relation satisfy the same path
    hypothesis, that IS the rule.
    """

    def __init__(self, max_pos: int = 50):
        self.max_pos = max_pos

    def _walk(self, start: int, steps: int) -> int | None:
        end = start + steps
        return end if 0 <= end <= self.max_pos else None

    def discover_rule(
        self,
        relation_name: str,
        triples: list[tuple[int, int, int]],
    ) -> dict[str, Any] | None:
        """Try to explain observed triples as a spatial rule.

        Hypotheses tested:
        1. "c = walk(a, b)" — b successor steps from a
        2. "c = walk(b, a)" — a successor steps from b
        3. "c = walk(0, a) repeated b times" — multiplication
        """
        if not triples:
            return None

        # Hypothesis 1: c = a + b (walk b steps from a).
        h1_matches = sum(1 for a, b, c in triples if self._walk(a, b) == c)

        # Hypothesis 2: c = walk(0, a) * b (walk a steps, b times, from 0).
        def _repeated_walk(a, b):
            pos = 0
            for _ in range(b):
                if pos + a > self.max_pos:
                    return None
                pos += a
            return pos

        h2_matches = sum(1 for a, b, c in triples if _repeated_walk(a, b) == c)

        n = len(triples)
        results = {}

        if h1_matches == n:
            results["walk"] = {
                "rule": f"{relation_name}(a, b, c) iff walk(a, b) = c",
                "meaning": "c is b successor steps from a",
                "confidence": 1.0,
                "matches": h1_matches,
                "total": n,
            }

        if h2_matches == n:
            results["repeated_walk"] = {
                "rule": f"{relation_name}(a, b, c) iff repeated_walk(a, b) = c",
                "meaning": "c is reached by walking a steps, b times, from 0",
                "confidence": 1.0,
                "matches": h2_matches,
                "total": n,
            }

        return results if results else None

    def discover_from_demonstrations(
        self,
        demos: list[Demonstration],
    ) -> dict[str, dict]:
        """Discover rules for all relations observed in demonstrations."""
        # Collect all relation triples from observations.
        relation_triples: dict[str, set[tuple[int, int, int]]] = defaultdict(set)

        for demo in demos:
            for trans in demo.transitions:
                for obs in [trans.before, trans.after]:
                    for pred, args, val in obs.facts:
                        if val is True and len(args) == 3:
                            try:
                                a, b, c = int(args[0]), int(args[1]), int(args[2])
                                relation_triples[pred].add((a, b, c))
                            except ValueError:
                                pass

        discovered = {}
        for rel_name, triples in sorted(relation_triples.items()):
            rules = self.discover_rule(rel_name, list(triples))
            if rules:
                discovered[rel_name] = rules

        return discovered


# ---------------------------------------------------------------------------
# Constraint solver using DISCOVERED rules (not hardcoded)
# ---------------------------------------------------------------------------

class SpatialSolver:
    """Solve arithmetic using discovered spatial rules.

    Once the RuleDiscoverer has found that sum(a,b,c) = walk(a,b)=c,
    this solver can answer ANY query by navigating the number line.
    """

    def __init__(self, discovered_rules: dict[str, dict], max_pos: int = 50):
        self.rules = discovered_rules
        self.max_pos = max_pos

    def solve(self, relation: str, known: dict[str, int]) -> int | None:
        """Solve for the unknown position using the discovered rule."""
        if relation not in self.rules:
            return None

        rule_info = self.rules[relation]
        a = known.get("a")
        b = known.get("b")
        c = known.get("c")

        if "walk" in rule_info:
            # Rule: c = walk(a, b) = a + b
            if a is not None and b is not None and c is None:
                result = a + b
                return result if result <= self.max_pos else None
            elif a is not None and c is not None and b is None:
                result = c - a
                return result if result >= 0 else None
            elif b is not None and c is not None and a is None:
                result = c - b
                return result if result >= 0 else None

        if "repeated_walk" in rule_info:
            # Rule: c = repeated_walk(a, b) = a * b
            if a is not None and b is not None and c is None:
                result = a * b
                return result if result <= self.max_pos else None
            elif a is not None and c is not None and b is None:
                if a == 0:
                    return None
                result = c // a
                return result if c % a == 0 else None
            elif b is not None and c is not None and a is None:
                if b == 0:
                    return None
                result = c // b
                return result if c % b == 0 else None

        return None

    def solve_compound(self, steps: list[tuple[str, dict[str, Any]]]) -> list[int | None]:
        """Solve a multi-step problem."""
        results = []
        env: dict[str, int] = {}
        for relation, known in steps:
            resolved = {}
            for k, v in known.items():
                if isinstance(v, str) and v.startswith("$"):
                    ref = v[1:]
                    resolved[k] = env.get(ref)
                else:
                    resolved[k] = v
            result = self.solve(relation, resolved)
            results.append(result)
            if result is not None:
                env[f"step{len(results)}"] = result
        return results


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  SPATIAL ARITHMETIC: Rules Learned from Number Line Structure")
    print("=" * 60)

    nl = NumberLine(max_pos=100)

    # Phase 1: Generate demonstrations.
    print("\n--- Phase 1: Demonstrations ---")
    demos = nl.generate_demonstrations(n=100, n_steps=20, seed=42)
    n_trans = sum(len(d.transitions) for d in demos)
    print(f"{len(demos)} demos, {n_trans} transitions")

    # Phase 2: Discover rules by correlating triples with path structure.
    print("\n--- Phase 2: Rule Discovery ---")
    discoverer = RuleDiscoverer(max_pos=100)
    rules = discoverer.discover_from_demonstrations(demos)

    for rel_name, rule_info in rules.items():
        print(f"\n  {rel_name}:")
        for hypothesis, details in rule_info.items():
            print(f"    {hypothesis}: {details['rule']}")
            print(f"      meaning: {details['meaning']}")
            print(f"      evidence: {details['matches']}/{details['total']} triples match")

    # Phase 3: Solve using discovered rules.
    print("\n--- Phase 3: Solving with Discovered Rules ---")
    solver = SpatialSolver(rules, max_pos=100)

    tests = [
        ("Forward addition", "sum", {"a": 3, "b": 5}, 8),
        ("Backward (subtraction)", "sum", {"a": 3, "c": 8}, 5),
        ("Backward (other direction)", "sum", {"b": 5, "c": 8}, 3),
        ("Forward multiplication", "product", {"a": 6, "b": 7}, 42),
        ("Backward (division)", "product", {"a": 6, "c": 42}, 7),
        ("Backward (other direction)", "product", {"b": 7, "c": 42}, 6),
        ("Large (never in training)", "sum", {"a": 50, "b": 50}, 100),
        ("Large multiply", "product", {"a": 12, "b": 8}, 96),
    ]

    all_correct = True
    for desc, rel, known, expected in tests:
        result = solver.solve(rel, known)
        ok = result == expected
        if not ok:
            all_correct = False
        print(f"  {desc:30s}: {rel}({known}) = {result} {'OK' if ok else f'WRONG (expected {expected})'}")

    # Compound problems.
    print(f"\n--- Compound Problems ---")
    results = solver.solve_compound([
        ("sum", {"a": 3, "b": 4}),
        ("product", {"a": "$step1", "b": 2}),
    ])
    print(f"  (3 + 4) * 2 = step1={results[0]}, step2={results[1]}  {'OK' if results[1] == 14 else 'WRONG'}")

    results2 = solver.solve_compound([
        ("product", {"b": 2, "c": 16}),
        ("sum", {"a": 3, "c": "$step1"}),
    ])
    print(f"  x*2=16 -> x={results2[0]}, 3+y=x -> y={results2[1]}  {'OK' if results2[1] == 5 else 'WRONG'}")

    # The key question: did it LEARN the rule?
    print(f"\n--- Did it learn the rule? ---")
    if "sum" in rules and "walk" in rules["sum"]:
        r = rules["sum"]["walk"]
        print(f"  sum rule: {r['rule']}")
        print(f"  This was discovered from {r['total']} observed triples.")
        print(f"  The system checked: does walk(a, b) = c for every observed sum(a,b,c)?")
        print(f"  Answer: YES for all {r['matches']} triples.")
        print(f"  Now it can solve sum(50, 50, ?) = 100 — NEVER seen in training.")
    else:
        print("  sum rule NOT discovered!")

    if "product" in rules and "repeated_walk" in rules["product"]:
        r = rules["product"]["repeated_walk"]
        print(f"\n  product rule: {r['rule']}")
        print(f"  Discovered from {r['total']} triples.")
        print(f"  walk a steps, b times, from 0 = a * b. Checked all {r['matches']} triples.")
    else:
        print("  product rule NOT discovered!")

    print(f"\n  ALL TESTS PASSED: {all_correct}")
