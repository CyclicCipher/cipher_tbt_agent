"""Invariant extraction — discover rules by finding what doesn't change.

Given examples of a relation (e.g., 3+4=7, 2+5=7, 9+10=19):
1. For each example, compute ALL spatial relationships between positions.
2. Intersect across all examples.
3. The surviving invariant IS the rule.

No hypotheses. No templates. No arithmetic. Just: what's the same
every time?

Spatial relationships are defined by co-termination of walks on the
successor graph. Two intervals "match" if walking both in lockstep,
they arrive at their destinations on the same step.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .spatial_rule_discovery import SuccessorGraph, co_terminate


# ---------------------------------------------------------------------------
# Interval computation: walk between two positions on the graph
# ---------------------------------------------------------------------------

def walk_length(graph: SuccessorGraph, start: int, end: int,
                max_steps: int = 200) -> int | None:
    """Walk from start toward end. Return number of steps, or None if unreachable."""
    if start == end:
        return 0
    pos = start
    steps = 0
    for _ in range(max_steps):
        next_pos = graph.step(pos)
        if next_pos is None:
            return None
        steps += 1
        pos = next_pos
        if pos == end:
            return steps
    return None


# ---------------------------------------------------------------------------
# Structural roles in an expression
# ---------------------------------------------------------------------------

# An expression like "3 + 4 = 7" has:
# - Positions: 3, 4, 7 (mapped to the number line)
# - Structural tokens: +, = (not on the number line)
# - The = splits the expression into left and right
# - The + separates positions on the left side
#
# We label the positions by their structural role:
# - left_1 = first position on the left (3)
# - left_2 = second position on the left (4)
# - right  = position on the right (7)
# - origin = position 0 (always available as reference point)


def parse_expression(expr: str, symbol_to_pos: dict[str, int]) -> dict[str, int] | None:
    """Parse "3 + 4 = 7" into structural roles.

    Returns {"left_1": 3, "left_2": 4, "right": 7, "origin": 0}
    or None if unparseable.
    """
    # Split on '='
    parts = expr.split("=")
    if len(parts) != 2:
        return None

    left_str = parts[0].strip()
    right_str = parts[1].strip()

    # The right side should be a single position.
    right_tokens = right_str.split()
    if len(right_tokens) != 1:
        return None
    if right_tokens[0] not in symbol_to_pos:
        return None

    # The left side has positions separated by an operator.
    left_tokens = left_str.split()
    # Find positions and operators.
    positions = []
    operators = []
    for tok in left_tokens:
        if tok in symbol_to_pos:
            positions.append(symbol_to_pos[tok])
        else:
            operators.append(tok)

    if len(positions) < 2:
        return None

    result = {
        "left_1": positions[0],
        "left_2": positions[1],
        "right": symbol_to_pos[right_tokens[0]],
        "origin": 0,
    }
    if operators:
        result["operator"] = operators[0]

    return result


# ---------------------------------------------------------------------------
# Compute ALL spatial relationships for one example
# ---------------------------------------------------------------------------

def compute_all_coterminations(
    graph: SuccessorGraph,
    roles: dict[str, int],
) -> set[tuple[str, str, str, str]]:
    """For a set of named positions, find all pairs of intervals that co-terminate.

    An interval is (start_role, end_role). Two intervals co-terminate
    if walking both in lockstep, they arrive on the same step.

    Returns a set of (start_A, end_A, start_B, end_B) tuples where
    interval(A_start → A_end) co-terminates with interval(B_start → B_end).
    """
    role_names = sorted(roles.keys())
    # Only consider roles that are positions (not "operator").
    pos_roles = [r for r in role_names if isinstance(roles.get(r), int)]

    coterminations: set[tuple[str, str, str, str]] = set()

    # All ordered pairs of intervals.
    intervals = []
    for r1 in pos_roles:
        for r2 in pos_roles:
            if r1 != r2 and roles[r1] <= roles[r2]:  # forward walks only
                intervals.append((r1, r2))

    # Check all pairs of intervals for co-termination.
    for i, (a_start, a_end) in enumerate(intervals):
        for j, (b_start, b_end) in enumerate(intervals):
            if i >= j:
                continue  # avoid duplicates
            if co_terminate(graph, roles[a_start], roles[a_end],
                           roles[b_start], roles[b_end]):
                coterminations.add((a_start, a_end, b_start, b_end))

    return coterminations


# ---------------------------------------------------------------------------
# Find invariants across all examples
# ---------------------------------------------------------------------------

def find_invariant_coterminations(
    graph: SuccessorGraph,
    examples: list[dict[str, int]],
) -> set[tuple[str, str, str, str]]:
    """Find co-termination relationships that hold for EVERY example.

    Computes all co-terminations for each example, then intersects.
    The surviving invariants ARE the rules.
    """
    if not examples:
        return set()

    # Compute for first example.
    result = compute_all_coterminations(graph, examples[0])

    # Intersect with each subsequent example.
    for ex in examples[1:]:
        ex_coterms = compute_all_coterminations(graph, ex)
        result = result & ex_coterms

    return result


def describe_invariant(inv: tuple[str, str, str, str]) -> str:
    """Human-readable description of a co-termination invariant."""
    a_start, a_end, b_start, b_end = inv
    return f"interval({a_start} -> {a_end}) = interval({b_start} -> {b_end})"


# ---------------------------------------------------------------------------
# Full pipeline: expressions → rule
# ---------------------------------------------------------------------------

def discover_rule(
    expressions: list[str],
    symbol_to_pos: dict[str, int],
    graph: SuccessorGraph,
) -> list[tuple[str, str]]:
    """Given expressions like ["3 + 4 = 7", "2 + 5 = 7", ...],
    discover the rule by finding invariant spatial relationships.

    Returns list of (invariant_description, interpretation).
    """
    # Parse expressions.
    examples = []
    operators = set()
    for expr in expressions:
        parsed = parse_expression(expr, symbol_to_pos)
        if parsed is not None:
            examples.append(parsed)
            if "operator" in parsed:
                operators.add(parsed["operator"])

    if not examples:
        return []

    # Find invariant co-terminations.
    invariants = find_invariant_coterminations(graph, examples)

    # Interpret.
    results = []
    for inv in sorted(invariants):
        desc = describe_invariant(inv)

        # Interpret what this means.
        a_start, a_end, b_start, b_end = inv
        if a_start == "left_1" and a_end == "right" and b_start == "origin" and b_end == "left_2":
            interp = "The result is reached by walking from left_1 by a distance equal to left_2's position from origin."
        elif a_start == "left_2" and a_end == "right" and b_start == "origin" and b_end == "left_1":
            interp = "The result is reached by walking from left_2 by a distance equal to left_1's position from origin (commutativity)."
        else:
            interp = "(structural relationship)"

        results.append((desc, interp))

    return results


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  INVARIANT EXTRACTION")
    print("  No hypotheses. Just: what's the same every time?")
    print("=" * 60)

    graph = SuccessorGraph.number_line(200)

    # Symbol-to-position mapping (the system knows this).
    sym = {str(i): i for i in range(200)}

    # --- Addition ---
    print("\n--- Addition examples ---")
    add_exprs = [
        "3 + 4 = 7",
        "2 + 5 = 7",
        "9 + 10 = 19",
        "5 + 6 = 11",
        "1 + 1 = 2",
        "0 + 8 = 8",
        "12 + 3 = 15",
    ]
    for e in add_exprs:
        print(f"  {e}")

    rules = discover_rule(add_exprs, sym, graph)
    print(f"\nInvariants found: {len(rules)}")
    for desc, interp in rules:
        print(f"  {desc}")
        print(f"    -> {interp}")

    # --- Multiplication ---
    print("\n--- Multiplication examples ---")
    mul_exprs = [
        "3 * 4 = 12",
        "2 * 5 = 10",
        "6 * 3 = 18",
        "7 * 2 = 14",
        "1 * 9 = 9",
        "4 * 4 = 16",
    ]
    for e in mul_exprs:
        print(f"  {e}")

    rules_mul = discover_rule(mul_exprs, sym, graph)
    print(f"\nInvariants found: {len(rules_mul)}")
    for desc, interp in rules_mul:
        print(f"  {desc}")
        print(f"    -> {interp}")

    # --- What about a NOVEL operator? ---
    print("\n--- Novel operator: @ (which is actually max) ---")
    max_exprs = [
        "3 @ 7 = 7",
        "5 @ 2 = 5",
        "4 @ 4 = 4",
        "1 @ 9 = 9",
        "6 @ 3 = 6",
    ]
    for e in max_exprs:
        print(f"  {e}")

    rules_max = discover_rule(max_exprs, sym, graph)
    print(f"\nInvariants found: {len(rules_max)}")
    for desc, interp in rules_max:
        print(f"  {desc}")
        print(f"    -> {interp}")

    if not rules_max:
        print("  (no co-termination invariants — this operation")
        print("   doesn't have a simple walk-based spatial rule)")
