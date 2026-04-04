"""Graph-native number line and arithmetic via co-terminating activation waves.

Numbers are NODES on a graph connected by successor edges.
Addition is two simultaneous activation waves:
  - Reference wave: 0 → b (measures the displacement)
  - Computation wave: a → c (applies the displacement)
Both advance one edge per step(). When the reference arrives at b,
the computation's position IS the answer.

No Python arithmetic. The graph IS the calculator.

The number line has two LAYERS per position — one for the reference
wave and one for the computation wave — so they don't interfere.
This mirrors the brain's use of different cortical layers for
different information streams within the same column.
"""
from __future__ import annotations

try:
    from .graph import Graph, SPATIAL, TEMPORAL, BINDING
except ImportError:
    from graph import Graph, SPATIAL, TEMPORAL, BINDING


def build_number_line(graph: Graph, max_n: int = 20,
                      subgraph_name: str = "number_line") -> dict[str, any]:
    """Build a graph-native number line from 0 to max_n.

    Each position has TWO nodes:
      - ref:N  — carries the reference wave (for measuring displacement)
      - comp:N — carries the computation wave (for computing results)

    Both layers are connected by successor edges within their layer.
    A SYNC edge between ref:N and comp:N ensures they advance in lockstep.

    Also creates:
      - done:N — a detector node that fires when the reference wave
        arrives at position N (coincidence detection for stopping)

    Returns dict with node IDs and metadata.
    """
    graph.create_subgraph(subgraph_name)

    ref_nodes = {}    # position -> node_id for reference layer
    comp_nodes = {}   # position -> node_id for computation layer
    done_nodes = {}   # position -> node_id for arrival detector

    # Create nodes for each position.
    for n in range(max_n + 1):
        ref_id = graph.add_node(
            label=f"numline:ref:{n}", subgraph=subgraph_name,
            layer="ref", position=n)
        comp_id = graph.add_node(
            label=f"numline:comp:{n}", subgraph=subgraph_name,
            layer="comp", position=n)
        done_id = graph.add_node(
            label=f"numline:done:{n}", subgraph=subgraph_name,
            layer="done", position=n)
        ref_nodes[n] = ref_id
        comp_nodes[n] = comp_id
        done_nodes[n] = done_id

    # Successor edges WITHIN each layer.
    for n in range(max_n):
        # Reference layer: ref:n → ref:n+1
        graph.add_edge(ref_nodes[n], ref_nodes[n + 1],
                       edge_type=TEMPORAL, weight=1.0)
        # Computation layer: comp:n → comp:n+1
        graph.add_edge(comp_nodes[n], comp_nodes[n + 1],
                       edge_type=TEMPORAL, weight=1.0)

    # SYNC edges: ref:n activating means comp:n should also advance.
    # These are spatial (undirected) — they couple the two layers
    # so waves propagate at the same speed.
    for n in range(max_n + 1):
        graph.add_edge(ref_nodes[n], comp_nodes[n],
                       edge_type=SPATIAL, weight=0.0)
        # Weight 0 — sync edges don't transfer activation, they just
        # mark that these nodes correspond to the same position.

    # Done detector: ref:n arriving triggers done:n.
    # done:n fires when ref:n has high activation.
    for n in range(max_n + 1):
        graph.add_edge(ref_nodes[n], done_nodes[n],
                       edge_type=TEMPORAL, weight=1.0)

    return {
        "ref": ref_nodes,
        "comp": comp_nodes,
        "done": done_nodes,
        "max_n": max_n,
        "subgraph": subgraph_name,
    }


def setup_addition(graph: Graph, numline: dict,
                   a: int, b: int) -> dict:
    """Set up the graph for computing a + b.

    Activates:
      - ref:0 (reference wave starts at 0, heading toward b)
      - comp:a (computation wave starts at a)

    The target is done:b — when the reference wave reaches position b,
    the computation wave's active position is the answer.

    Returns setup info including the target done node.
    """
    ref_nodes = numline["ref"]
    comp_nodes = numline["comp"]
    done_nodes = numline["done"]

    # Clear all number line activations.
    for n in range(numline["max_n"] + 1):
        graph.get_node(ref_nodes[n]).activation = 0.0
        graph.get_node(comp_nodes[n]).activation = 0.0
        graph.get_node(done_nodes[n]).activation = 0.0

    # Activate starting positions.
    graph.activate(ref_nodes[0], 1.0)    # reference starts at 0
    graph.activate(comp_nodes[a], 1.0)   # computation starts at a

    return {
        "a": a,
        "b": b,
        "target_done": done_nodes[b],
        "ref_start": ref_nodes[0],
        "comp_start": comp_nodes[a],
    }


def propagate_waves(graph: Graph, numline: dict, n_steps: int = 1):
    """Advance both waves by n_steps along the number line.

    Uses a custom propagation that moves activation ONE step along
    successor edges in both ref and comp layers. This is NOT the
    full graph step() — it's targeted propagation on the number line
    only, to avoid interference with PFC and other subgraphs.

    Each call advances both waves by exactly one position.
    """
    ref_nodes = numline["ref"]
    comp_nodes = numline["comp"]
    done_nodes = numline["done"]
    max_n = numline["max_n"]

    for _ in range(n_steps):
        # Snapshot current activations.
        ref_acts = {n: graph.get_node(ref_nodes[n]).activation
                    for n in range(max_n + 1)}
        comp_acts = {n: graph.get_node(comp_nodes[n]).activation
                     for n in range(max_n + 1)}

        # Clear all.
        for n in range(max_n + 1):
            graph.get_node(ref_nodes[n]).activation = 0.0
            graph.get_node(comp_nodes[n]).activation = 0.0

        # Propagate: each active node passes its activation to successor.
        for n in range(max_n):
            if ref_acts[n] > 0.01:
                graph.activate(ref_nodes[n + 1], ref_acts[n])
            if comp_acts[n] > 0.01:
                graph.activate(comp_nodes[n + 1], comp_acts[n])

        # Update done detectors.
        for n in range(max_n + 1):
            ref_act = graph.get_node(ref_nodes[n]).activation
            if ref_act > 0.5:
                graph.activate(done_nodes[n], 1.0)
            else:
                graph.get_node(done_nodes[n]).activation = 0.0


def compute_addition(graph: Graph, numline: dict,
                     a: int, b: int) -> int | None:
    """Compute a + b using co-terminating waves on the number line.

    Returns the answer (position of comp wave when ref wave hits b),
    or None if the computation fails.
    """
    if a + b > numline["max_n"]:
        return None  # result exceeds number line

    setup = setup_addition(graph, numline, a, b)
    target_done = setup["target_done"]

    # Special case: b = 0 means no displacement. Answer is a.
    # The reference wave starts at 0 and target IS 0 — already there.
    if b == 0:
        return a

    # Propagate waves until the reference arrives at b.
    for step in range(numline["max_n"] + 1):
        propagate_waves(graph, numline, n_steps=1)

        # Check: has the reference wave arrived at b?
        if graph.get_node(target_done).activation > 0.5:
            # Find where the computation wave is.
            comp_nodes = numline["comp"]
            for n in range(numline["max_n"] + 1):
                if graph.get_node(comp_nodes[n]).activation > 0.5:
                    return n
            return None  # comp wave lost somehow

    return None  # reference never arrived (shouldn't happen)


def read_comp_position(graph: Graph, numline: dict) -> int | None:
    """Read the current position of the computation wave."""
    comp_nodes = numline["comp"]
    for n in range(numline["max_n"] + 1):
        if graph.get_node(comp_nodes[n]).activation > 0.5:
            return n
    return None


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Graph-Native Number Line Arithmetic")
    print("  No Python math. The graph IS the calculator.")
    print("=" * 60)

    g = Graph()
    numline = build_number_line(g, max_n=20)
    print(f"\nNumber line: {numline['max_n'] + 1} positions")
    print(f"Graph: {g.summary()}")

    # --- Test addition via co-terminating waves ---
    print("\n--- Addition via wave co-termination ---")
    tests = [
        (0, 0, 0),
        (1, 1, 2),
        (3, 4, 7),
        (5, 5, 10),
        (8, 7, 15),
        (0, 9, 9),
        (10, 10, 20),
        (1, 0, 1),
        (7, 6, 13),
    ]

    correct = 0
    for a, b, expected in tests:
        result = compute_addition(g, numline, a, b)
        ok = result == expected
        if ok:
            correct += 1
        print(f"  {a} + {b} = {result}  {'OK' if ok else f'WRONG (expected {expected})'}")

    print(f"\n  Score: {correct}/{len(tests)}")

    # --- Trace a single computation step by step ---
    print("\n--- Trace: 3 + 4 = ? ---")
    setup = setup_addition(g, numline, 3, 4)
    print(f"  Initial: ref at 0, comp at 3, target done:4")

    for step in range(5):
        ref_pos = None
        comp_pos = None
        for n in range(21):
            if g.get_node(numline["ref"][n]).activation > 0.5:
                ref_pos = n
            if g.get_node(numline["comp"][n]).activation > 0.5:
                comp_pos = n

        done_4 = g.get_node(numline["done"][4]).activation
        print(f"  Step {step}: ref={ref_pos}, comp={comp_pos}, done:4={done_4:.1f}")

        if done_4 > 0.5:
            print(f"  ARRIVED! comp position = {comp_pos} = ANSWER")
            break

        propagate_waves(g, numline, n_steps=1)

    # --- Inverse: given a and c, find b ---
    print("\n--- Inverse: 3 + ? = 7 ---")
    # To find b: we need to walk from a to c and count the steps.
    # Set up: ref starts at a (=3), comp starts at c (=7).
    # Walk BOTH backward... or: just compute c - a = 7 - 3 = 4.
    # For now, use the forward method: try each b until we get c.
    # (A proper inverse would use backward waves — future work.)
    for b_candidate in range(21):
        result = compute_addition(g, numline, 3, b_candidate)
        if result == 7:
            print(f"  Found: b = {b_candidate} (3 + {b_candidate} = 7)")
            break
