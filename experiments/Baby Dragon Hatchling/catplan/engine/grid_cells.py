"""Multi-scale grid cell representation for rule discovery.

Biological basis: entorhinal grid cells fire at regular intervals
at multiple scales simultaneously. A position is represented not as
a single point but as a VECTOR of activations at different scales.

For the number line:
- Scale 1: every position is distinct (identity)
- Scale 2: positions 0,2,4,6,... are equivalent (even/odd)
- Scale 3: positions 0,3,6,9,... are equivalent (mod 3)
- Scale k: positions 0,k,2k,3k,... are equivalent (mod k)

A position p at scale k is represented by:
- phase: p mod k  (where in the cycle)
- count: p // k   (how many full cycles from origin)

Multiplication emerges as a 2D pattern:
product(a, b, c) means c is at grid point (count_a, count_b) in
the 2D lattice formed by scales a and b.

Segment decomposition: the interval from 0 to c can be decomposed
into segments of length k. The number of segments = c // k (if exact).
This decomposition is computed by walking, not by division.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .spatial_rule_discovery import SuccessorGraph, co_terminate


# ---------------------------------------------------------------------------
# Grid cell representation
# ---------------------------------------------------------------------------

class GridCell:
    """A single grid cell that fires periodically at a given scale.

    At scale k, this cell fires (activation = 1) at positions
    0, k, 2k, 3k, ... and is silent in between.
    """

    def __init__(self, scale: int):
        self.scale = scale

    def fires_at(self, position: int) -> bool:
        """Does this grid cell fire at the given position?"""
        return position % self.scale == 0

    def phase(self, position: int) -> int:
        """Phase of the position within this cell's cycle."""
        return position % self.scale

    def count(self, position: int) -> int:
        """How many full cycles from origin to this position."""
        return position // self.scale


class MultiScaleGrid:
    """A set of grid cells at multiple scales.

    Represents positions as vectors of (phase, count) at each scale.
    This is the neural code: a position is identified by its
    activations across all scales.
    """

    def __init__(self, scales: list[int] | None = None, max_pos: int = 100):
        if scales is None:
            # Default: scales 1 through 12 (covers factors of common numbers).
            scales = list(range(1, 13))
        self.scales = scales
        self.cells = [GridCell(s) for s in scales]
        self.max_pos = max_pos
        self.graph = SuccessorGraph.number_line(max_pos)

    def encode(self, position: int) -> dict[int, tuple[int, int]]:
        """Encode a position as {scale: (phase, count)}."""
        return {cell.scale: (cell.phase(position), cell.count(position))
                for cell in self.cells}

    def firing_scales(self, position: int) -> list[int]:
        """Which scales fire at this position? (phase = 0)"""
        return [cell.scale for cell in self.cells if cell.fires_at(position)]


# ---------------------------------------------------------------------------
# Segment decomposition via walking (no division!)
# ---------------------------------------------------------------------------

def decompose_interval(graph: SuccessorGraph, start: int, end: int,
                       segment_length: int, max_steps: int = 500) -> int | None:
    """Can the interval start→end be decomposed into exact segments
    of the given length?

    Walk from start in segments. Each segment: walk in lockstep with
    a reference walk 0→segment_length. Count how many segments fit.

    Returns the number of segments if exact decomposition exists,
    None otherwise.

    NO DIVISION. Only walking and co-termination.
    """
    if segment_length == 0:
        return 0 if start == end else None

    pos = start
    n_segments = 0

    for _ in range(max_steps):
        if pos == end:
            return n_segments

        # Walk one segment: lockstep with 0→segment_length.
        ref = 0
        for _ in range(max_steps):
            if ref == segment_length:
                break
            next_ref = graph.step(ref)
            next_pos = graph.step(pos)
            if next_ref is None or next_pos is None:
                return None
            ref = next_ref
            pos = next_pos

        n_segments += 1

        if pos > end:
            return None  # overshot — not an exact decomposition

    return None


# ---------------------------------------------------------------------------
# Multi-scale invariant extraction
# ---------------------------------------------------------------------------

def compute_multiscale_relationships(
    grid: MultiScaleGrid,
    roles: dict[str, int],
) -> set[tuple]:
    """Compute all multi-scale spatial relationships for one example.

    For each scale, compute:
    - Phase relationships between positions
    - Segment decomposition relationships
    - Grid point coincidences

    Returns a set of relationship tuples that describe this example.
    """
    relationships: set[tuple] = set()
    pos_roles = {k: v for k, v in roles.items() if isinstance(v, int)}

    for cell in grid.cells:
        s = cell.scale

        # Phase of each position at this scale.
        for role, pos in pos_roles.items():
            phase = cell.phase(pos)
            relationships.add(("phase", role, s, phase))

        # Segment decomposition: can interval(origin→right) be decomposed
        # into segments of length left_1 or left_2?
        if "origin" in pos_roles and "right" in pos_roles:
            c = pos_roles["right"]
            for input_role in ["left_1", "left_2"]:
                if input_role in pos_roles:
                    seg_len = pos_roles[input_role]
                    if seg_len > 0:
                        n_segs = decompose_interval(grid.graph, 0, c, seg_len)
                        if n_segs is not None:
                            relationships.add(("exact_segments", input_role, n_segs))

        # Grid firing: does the result position fall on a grid point
        # at the scale of one of the inputs?
        if "right" in pos_roles:
            c = pos_roles["right"]
            for input_role in ["left_1", "left_2"]:
                if input_role in pos_roles:
                    input_val = pos_roles[input_role]
                    if input_val > 0 and cell.scale == input_val:
                        if cell.fires_at(c):
                            count = cell.count(c)
                            relationships.add(("grid_fires", input_role, s, count))

    # Cross-scale: does the segment count at scale=left_1 equal left_2?
    # This is: interval(0→c) has exactly left_2 segments of length left_1.
    if "left_1" in pos_roles and "left_2" in pos_roles and "right" in pos_roles:
        a = pos_roles["left_1"]
        b = pos_roles["left_2"]
        c = pos_roles["right"]
        if a > 0:
            n_a = decompose_interval(grid.graph, 0, c, a)
            if n_a is not None:
                # Does n_a co-terminate with walk(0→b)?
                if co_terminate(grid.graph, 0, n_a, 0, b):
                    relationships.add(("segments_of_a_equals_b",))
        if b > 0:
            n_b = decompose_interval(grid.graph, 0, c, b)
            if n_b is not None:
                if co_terminate(grid.graph, 0, n_b, 0, a):
                    relationships.add(("segments_of_b_equals_a",))

    # Single-walk co-termination (from invariant_extraction).
    if "left_1" in pos_roles and "right" in pos_roles and "origin" in pos_roles and "left_2" in pos_roles:
        if co_terminate(grid.graph, pos_roles["left_1"], pos_roles["right"],
                       pos_roles["origin"], pos_roles["left_2"]):
            relationships.add(("walk_coterminate", "left_1", "right", "origin", "left_2"))
        if co_terminate(grid.graph, pos_roles["left_2"], pos_roles["right"],
                       pos_roles["origin"], pos_roles["left_1"]):
            relationships.add(("walk_coterminate", "left_2", "right", "origin", "left_1"))

    return relationships


def find_multiscale_invariants(
    grid: MultiScaleGrid,
    examples: list[dict[str, int]],
) -> set[tuple]:
    """Find relationships that hold for EVERY example.

    Computes all multi-scale relationships per example, intersects.
    """
    if not examples:
        return set()

    result = compute_multiscale_relationships(grid, examples[0])
    for ex in examples[1:]:
        ex_rels = compute_multiscale_relationships(grid, ex)
        result = result & ex_rels

    return result


def describe_multiscale_invariant(inv: tuple) -> str:
    """Human-readable description."""
    if inv[0] == "walk_coterminate":
        return f"interval({inv[1]} -> {inv[2]}) = interval({inv[3]} -> {inv[4]})"
    elif inv[0] == "segments_of_a_equals_b":
        return "interval(0 -> result) decomposes into exactly left_2 segments of length left_1"
    elif inv[0] == "segments_of_b_equals_a":
        return "interval(0 -> result) decomposes into exactly left_1 segments of length left_2"
    elif inv[0] == "exact_segments":
        return f"interval(0 -> result) = {inv[2]} segments of length {inv[1]}"
    elif inv[0] == "grid_fires":
        return f"result falls on grid of scale {inv[1]}(={inv[2]}), count={inv[3]}"
    elif inv[0] == "phase":
        return f"phase({inv[1]}, scale={inv[2]}) = {inv[3]}"
    else:
        return str(inv)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def discover_rules_multiscale(
    expressions: list[str],
    symbol_to_pos: dict[str, int],
    max_pos: int = 200,
) -> list[tuple[str, str]]:
    """Discover rules from expressions using multi-scale grid representation."""
    from .invariant_extraction import parse_expression

    grid = MultiScaleGrid(scales=list(range(1, 15)), max_pos=max_pos)

    examples = []
    for expr in expressions:
        parsed = parse_expression(expr, symbol_to_pos)
        if parsed is not None:
            examples.append(parsed)

    if not examples:
        return []

    invariants = find_multiscale_invariants(grid, examples)

    # Filter out trivial invariants (phase relationships that hold for
    # every possible triple, not just these specific ones).
    # Keep: walk co-termination, segment decomposition, grid firing.
    meaningful = []
    for inv in sorted(invariants, key=str):
        if inv[0] in ("walk_coterminate", "segments_of_a_equals_b",
                       "segments_of_b_equals_a"):
            desc = describe_multiscale_invariant(inv)
            meaningful.append((inv[0], desc))

    return meaningful


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .invariant_extraction import parse_expression

    print("=" * 65)
    print("  MULTI-SCALE GRID CELL RULE DISCOVERY")
    print("  Biological: grid cells + segment decomposition + intersection")
    print("=" * 65)

    sym = {str(i): i for i in range(300)}

    # --- Addition ---
    print("\n--- Addition ---")
    add_exprs = ["3 + 4 = 7", "2 + 5 = 7", "9 + 10 = 19",
                 "5 + 6 = 11", "1 + 1 = 2", "0 + 8 = 8", "12 + 3 = 15"]
    results = discover_rules_multiscale(add_exprs, sym)
    for kind, desc in results:
        print(f"  [{kind}] {desc}")

    # --- Multiplication ---
    print("\n--- Multiplication ---")
    mul_exprs = ["3 * 4 = 12", "2 * 5 = 10", "6 * 3 = 18",
                 "7 * 2 = 14", "1 * 9 = 9", "4 * 4 = 16", "5 * 5 = 25"]
    results_mul = discover_rules_multiscale(mul_exprs, sym)
    for kind, desc in results_mul:
        print(f"  [{kind}] {desc}")

    if not results_mul:
        print("  (no invariants found)")

    # --- Exponentiation ---
    print("\n--- Exponentiation (power) ---")
    pow_exprs = ["2 ^ 3 = 8", "3 ^ 2 = 9", "2 ^ 4 = 16",
                 "5 ^ 2 = 25", "4 ^ 2 = 16", "2 ^ 5 = 32"]
    results_pow = discover_rules_multiscale(pow_exprs, sym)
    for kind, desc in results_pow:
        print(f"  [{kind}] {desc}")

    if not results_pow:
        print("  (no invariants found)")

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"  Addition:       {len(discover_rules_multiscale(add_exprs, sym))} invariants found")
    print(f"  Multiplication: {len(discover_rules_multiscale(mul_exprs, sym))} invariants found")
    print(f"  Exponentiation: {len(discover_rules_multiscale(pow_exprs, sym))} invariants found")
