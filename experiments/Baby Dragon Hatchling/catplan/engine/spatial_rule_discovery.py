"""Spatial rule discovery — learn arithmetic from graph structure alone.

The ONLY primitives:
1. Follow a successor edge (walk one step on the number line)
2. Simultaneous co-termination (walk two paths in lockstep,
   check if they arrive at their destinations on the same step)

From these, the system discovers:
- sum(a, b, c): the walk a→c co-terminates with the walk 0→b
- product(a, b, c): walking in a-length segments b times from 0 reaches c

No addition, no counting, no arithmetic of any kind is hardcoded.
The system discovers these relationships purely from graph traversal.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


class SuccessorGraph:
    """A directed graph where each node has at most one successor.

    This is the number line: 0 → 1 → 2 → 3 → ...
    The system knows ONLY the edges. It cannot compute, add, or count.
    """

    def __init__(self, edges: dict[int, int]):
        """edges: {node: successor_node}"""
        self.succ = dict(edges)
        self.pred = {v: k for k, v in edges.items()}  # inverse for backward walking

    @staticmethod
    def number_line(max_n: int) -> 'SuccessorGraph':
        return SuccessorGraph({i: i + 1 for i in range(max_n)})

    def step(self, node: int) -> int | None:
        """Follow one successor edge. Returns None if no successor."""
        return self.succ.get(node)

    def step_back(self, node: int) -> int | None:
        """Follow one predecessor edge."""
        return self.pred.get(node)


# ---------------------------------------------------------------------------
# Primitive 2: Simultaneous co-termination
# ---------------------------------------------------------------------------

def co_terminate(graph: SuccessorGraph,
                 start_a: int, end_a: int,
                 start_b: int, end_b: int,
                 max_steps: int = 200) -> bool:
    """Walk two paths simultaneously, one step each.

    Path 1: start_a → end_a (following successor)
    Path 2: start_b → end_b (following successor)

    Returns True if both paths arrive at their destinations
    on the SAME step. This is the comparison primitive.

    No counting. No numbers computed. Just lockstep walking.
    """
    pos_a = start_a
    pos_b = start_b

    for _ in range(max_steps):
        arrived_a = (pos_a == end_a)
        arrived_b = (pos_b == end_b)

        if arrived_a and arrived_b:
            return True   # co-terminated!
        if arrived_a or arrived_b:
            return False  # one arrived but not the other

        next_a = graph.step(pos_a)
        next_b = graph.step(pos_b)

        if next_a is None or next_b is None:
            return False  # one path ended (no successor)

        pos_a = next_a
        pos_b = next_b

    return False  # exceeded max steps


# ---------------------------------------------------------------------------
# Primitive 3: Segment walking (for multiplication)
# ---------------------------------------------------------------------------

def segment_walk(graph: SuccessorGraph,
                 segment_end: int,    # walk this many steps per segment (= position of segment length)
                 n_segments_end: int, # repeat this many times (= position of count)
                 max_steps: int = 200) -> int | None:
    """Walk in segments from 0.

    Each segment: walk from current position, in lockstep with a walk
    from 0 to segment_end. When the reference walk arrives, the
    segment is complete. Then start a new segment.

    Repeat for as many segments as the walk from 0 to n_segments_end takes.

    Returns the final position, or None if invalid.

    This uses ONLY co-termination. No counting or arithmetic.
    """
    # Outer walk: 0 → n_segments_end, one step per segment.
    # Inner walk: each segment is a walk synchronized with 0 → segment_end.
    current_pos = 0
    outer_pos = 0

    for _ in range(max_steps):
        if outer_pos == n_segments_end:
            return current_pos  # done all segments

        # One segment: walk current_pos forward, synchronized with 0 → segment_end.
        ref_pos = 0
        for _ in range(max_steps):
            if ref_pos == segment_end:
                break  # segment complete

            next_ref = graph.step(ref_pos)
            next_cur = graph.step(current_pos)
            if next_ref is None or next_cur is None:
                return None

            ref_pos = next_ref
            current_pos = next_cur

        # Advance outer walk by one step.
        next_outer = graph.step(outer_pos)
        if next_outer is None:
            return None
        outer_pos = next_outer

    return None


# ---------------------------------------------------------------------------
# Structural templates: ways to relate three positions using walks
# ---------------------------------------------------------------------------

def template_walk_coterminate(graph: SuccessorGraph, a: int, b: int, c: int) -> bool:
    """Template: walk(a→c) co-terminates with walk(0→b).

    If true: sum(a, b, c) — "c is b steps from a."
    """
    return co_terminate(graph, a, c, 0, b)


def template_walk_coterminate_swap(graph: SuccessorGraph, a: int, b: int, c: int) -> bool:
    """Template: walk(b→c) co-terminates with walk(0→a).

    If true: sum(a, b, c) with a and b swapped — same thing for commutative.
    """
    return co_terminate(graph, b, c, 0, a)


def template_segment_walk(graph: SuccessorGraph, a: int, b: int, c: int) -> bool:
    """Template: walking in a-length segments b times from 0 reaches c.

    If true: product(a, b, c).
    """
    result = segment_walk(graph, a, b)
    return result is not None and result == c


def template_segment_walk_swap(graph: SuccessorGraph, a: int, b: int, c: int) -> bool:
    """Template: walking in b-length segments a times from 0 reaches c."""
    result = segment_walk(graph, b, a)
    return result is not None and result == c


# All templates.
ALL_TEMPLATES = [
    ("walk_a_to_c_length_b", template_walk_coterminate),
    ("walk_b_to_c_length_a", template_walk_coterminate_swap),
    ("segment_a_repeated_b", template_segment_walk),
    ("segment_b_repeated_a", template_segment_walk_swap),
]


# ---------------------------------------------------------------------------
# Rule discovery: try all templates on all triples
# ---------------------------------------------------------------------------

class SpatialRuleDiscoverer:
    """Discover rules by testing structural templates against observed triples.

    For each relation (e.g., "sum"), try each template on every observed
    triple. If a template matches ALL triples, it IS the rule.

    No arithmetic. No hardcoded rules. Just graph traversal templates.
    """

    def __init__(self, graph: SuccessorGraph):
        self.graph = graph

    def discover(
        self,
        relation_triples: dict[str, list[tuple[int, int, int]]],
    ) -> dict[str, list[dict]]:
        """Discover rules for each relation.

        Returns {relation_name: [matching_templates]}.
        """
        results: dict[str, list[dict]] = {}

        for rel_name, triples in relation_triples.items():
            if not triples:
                continue

            matches = []
            for template_name, template_fn in ALL_TEMPLATES:
                n_match = 0
                n_total = len(triples)

                for a, b, c in triples:
                    if template_fn(self.graph, a, b, c):
                        n_match += 1

                if n_match == n_total:
                    matches.append({
                        "template": template_name,
                        "matches": n_match,
                        "total": n_total,
                        "confidence": 1.0,
                    })

            results[rel_name] = matches

        return results


# ---------------------------------------------------------------------------
# Solver: use discovered templates to answer queries
# ---------------------------------------------------------------------------

class SpatialSolver:
    """Solve queries using discovered structural templates.

    Given a relation and two known positions, find the third by
    searching for a position where the template holds.
    """

    def __init__(self, graph: SuccessorGraph, discovered: dict[str, list[dict]]):
        self.graph = graph
        self.discovered = discovered

    def _get_template_fn(self, template_name: str):
        for name, fn in ALL_TEMPLATES:
            if name == template_name:
                return fn
        return None

    def solve(self, relation: str, a: int | None, b: int | None, c: int | None,
              max_search: int = 200) -> int | None:
        """Find the missing value by searching for a position where the template holds."""
        if relation not in self.discovered or not self.discovered[relation]:
            return None

        template_name = self.discovered[relation][0]["template"]
        template_fn = self._get_template_fn(template_name)
        if template_fn is None:
            return None

        # Search for the unknown value.
        for candidate in range(max_search):
            if a is None:
                if template_fn(self.graph, candidate, b, c):
                    return candidate
            elif b is None:
                if template_fn(self.graph, a, candidate, c):
                    return candidate
            elif c is None:
                if template_fn(self.graph, a, b, candidate):
                    return candidate

        return None


# ---------------------------------------------------------------------------
# Extract triples from demonstrations
# ---------------------------------------------------------------------------

def extract_triples(demos) -> dict[str, list[tuple[int, int, int]]]:
    """Extract relation triples from world demonstrations."""
    triples: dict[str, set[tuple[int, int, int]]] = defaultdict(set)
    for demo in demos:
        for trans in demo.transitions:
            for obs in [trans.before, trans.after]:
                for pred, args, val in obs.facts:
                    if val is True and len(args) == 3:
                        try:
                            a, b, c = int(args[0]), int(args[1]), int(args[2])
                            triples[pred].add((a, b, c))
                        except ValueError:
                            pass
    return {k: list(v) for k, v in triples.items()}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
    from worlds.number_line import NumberLine

    print("=" * 60)
    print("  SPATIAL RULE DISCOVERY")
    print("  (no arithmetic, only graph walking)")
    print("=" * 60)

    # Build the number line graph.
    graph = SuccessorGraph.number_line(150)

    # Generate demonstrations.
    nl = NumberLine(max_pos=100)
    demos = nl.generate_demonstrations(n=100, n_steps=20, seed=42)
    n_trans = sum(len(d.transitions) for d in demos)
    print(f"\n{len(demos)} demos, {n_trans} transitions")

    # Extract triples.
    triples = extract_triples(demos)
    for rel, trips in triples.items():
        print(f"  {rel}: {len(trips)} triples (e.g., {trips[:3]})")

    # Discover rules.
    print("\n--- Rule Discovery (graph walking only) ---")
    discoverer = SpatialRuleDiscoverer(graph)
    discovered = discoverer.discover(triples)

    for rel, matches in discovered.items():
        print(f"\n  {rel}:")
        if not matches:
            print("    (no matching template)")
        for m in matches:
            print(f"    {m['template']}: {m['matches']}/{m['total']} triples match")

    # Solve novel problems.
    print("\n--- Solving with Discovered Rules ---")
    solver = SpatialSolver(graph, discovered)

    tests = [
        ("sum(3, 5, ?)", "sum", 3, 5, None, 8),
        ("sum(3, ?, 8)", "sum", 3, None, 8, 5),
        ("sum(?, 5, 8)", "sum", None, 5, 8, 3),
        ("sum(50, 50, ?)", "sum", 50, 50, None, 100),
        ("sum(0, 0, ?)", "sum", 0, 0, None, 0),
        ("product(6, 7, ?)", "product", 6, 7, None, 42),
        ("product(?, 7, 42)", "product", None, 7, 42, 6),
        ("product(6, ?, 42)", "product", 6, None, 42, 7),
        ("product(12, 8, ?)", "product", 12, 8, None, 96),
    ]

    all_ok = True
    for desc, rel, a, b, c, expected in tests:
        result = solver.solve(rel, a, b, c)
        ok = result == expected
        if not ok:
            all_ok = False
        print(f"  {desc:25s} = {result}  {'OK' if ok else f'WRONG (expected {expected})'}")

    print(f"\n  ALL CORRECT: {all_ok}")

    # Show what was NOT hardcoded.
    print("\n--- What the system used ---")
    print("  Primitives: successor edges, co-termination, segment walking")
    print("  NOT used: addition, subtraction, multiplication, counting, numbers-as-values")
    print("  The system walked the graph and found which template matches all triples.")
