"""
Algebraic skeleton discovery — Krohn-Rhodes lite.

Extracts the transition monoid from the knowledge graph's TRANSITION edges
and decomposes it into:

1. **Strongly connected components (SCCs):** maximal subsets where every
   node can reach every other via transitions. These are the "group-like"
   components — reversible structure (cyclic groups, permutation groups).

2. **Condensation DAG:** collapsing each SCC to a single node gives a DAG
   showing the irreversible flow between components.

3. **Maximal chains:** longest directed paths through the condensation DAG.
   These are candidate NNOs — ordered sequences with a clear start and
   successor structure.

4. **Endomorphisms within SCCs:** consistent one-step-forward patterns.
   Within a cyclic SCC, the successor function is an endomorphism.

The full Krohn-Rhodes decomposition factors the transition monoid into
simple groups and aperiodic semigroups. We compute the structural
invariants (SCCs, chains, cycles) that reveal the same information
without the full algebraic machinery.

No domain-specific code. Operates on abstract graph structure only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId, COOCCURRENCE, TRANSITION,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AlgebraicComponent:
    """A strongly connected component of the transition graph."""
    members: frozenset[NodeId]
    is_trivial: bool           # single node with no self-loop
    cycle_length: int = 0      # length of the dominant cycle (0 = acyclic/trivial)
    cycle_order: list[NodeId] = field(default_factory=list)  # nodes in cycle order


@dataclass
class Chain:
    """A maximal directed path through the condensation DAG."""
    nodes: list[NodeId]        # ordered sequence of nodes
    length: int = 0            # number of edges (len(nodes) - 1)


@dataclass
class AlgebraResult:
    """Result of algebraic skeleton discovery."""
    components: list[AlgebraicComponent]
    chains: list[Chain]
    condensation_edges: list[tuple[int, int]]  # (src_comp_idx, tgt_comp_idx)


# ---------------------------------------------------------------------------
# Transition graph extraction
# ---------------------------------------------------------------------------

def _extract_transition_graph(
    kg: KnowledgeGraph,
    min_weight: float = 0.05,
) -> dict[NodeId, list[NodeId]]:
    """Extract the positive-weight TRANSITION subgraph as adjacency lists.

    Returns {source: [targets]} for all transition edges with weight >= min_weight.
    Only includes nodes that participate in at least one transition.
    """
    adj: dict[NodeId, list[NodeId]] = {}
    for (src, tgt), edge in kg._edges.items():
        if edge.role != TRANSITION:
            continue
        if edge.weight < min_weight:
            continue
        adj.setdefault(src, []).append(tgt)
        # Ensure target appears as a key even if it has no outgoing edges.
        adj.setdefault(tgt, [])
    return adj


# ---------------------------------------------------------------------------
# Tarjan's SCC algorithm
# ---------------------------------------------------------------------------

def _tarjan_scc(adj: dict[NodeId, list[NodeId]]) -> list[list[NodeId]]:
    """Tarjan's algorithm for strongly connected components.

    Returns a list of SCCs, each as a list of NodeIds. SCCs are in
    reverse topological order (sinks first).
    """
    index_counter = [0]
    stack: list[NodeId] = []
    on_stack: set[NodeId] = set()
    index: dict[NodeId, int] = {}
    lowlink: dict[NodeId, int] = {}
    result: list[list[NodeId]] = []

    def strongconnect(v: NodeId) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            component: list[NodeId] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            result.append(component)

    # Use iterative deepening to avoid stack overflow on large graphs.
    # For typical CTKG sizes (< 1000 nodes), recursion is fine.
    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, len(adj) + 100))
    try:
        for v in adj:
            if v not in index:
                strongconnect(v)
    finally:
        sys.setrecursionlimit(old_limit)

    return result


# ---------------------------------------------------------------------------
# Cycle detection within SCCs
# ---------------------------------------------------------------------------

def _find_cycle_in_scc(
    members: frozenset[NodeId],
    adj: dict[NodeId, list[NodeId]],
) -> list[NodeId]:
    """Find a Hamiltonian cycle or longest cycle within an SCC.

    For small SCCs (≤ 20 members), attempts to find a cycle visiting
    all members. For larger SCCs, finds the longest simple cycle from
    an arbitrary start node.

    Returns the cycle as an ordered list of nodes, or empty if no cycle.
    """
    if len(members) <= 1:
        # Check self-loop.
        node = next(iter(members))
        if node in adj.get(node, []):
            return [node]
        return []

    # For small SCCs, try to find a Hamiltonian path then check if it cycles.
    # Greedy approach: start from each node, always pick unvisited neighbors.
    best_cycle: list[NodeId] = []

    for start in sorted(members):
        path = [start]
        visited = {start}
        current = start

        while True:
            # Pick an unvisited neighbor within the SCC.
            next_node = None
            for n in adj.get(current, []):
                if n in members and n not in visited:
                    next_node = n
                    break
            if next_node is None:
                break
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        # Check if the path forms a cycle (last node → start).
        if len(path) > 1 and start in adj.get(path[-1], []):
            if len(path) > len(best_cycle):
                best_cycle = path

        # For small SCCs, keep trying all starts.
        if len(members) > 20:
            break  # just use the first start for large SCCs

    return best_cycle


# ---------------------------------------------------------------------------
# Chain extraction (longest paths in condensation DAG)
# ---------------------------------------------------------------------------

def _build_condensation(
    sccs: list[list[NodeId]],
    adj: dict[NodeId, list[NodeId]],
) -> tuple[dict[NodeId, int], dict[int, list[int]]]:
    """Build the condensation DAG from SCCs.

    Returns (node_to_comp, comp_adj) where:
    - node_to_comp: maps each NodeId to its SCC index
    - comp_adj: adjacency list between SCC indices
    """
    node_to_comp: dict[NodeId, int] = {}
    for comp_idx, scc in enumerate(sccs):
        for nid in scc:
            node_to_comp[nid] = comp_idx

    comp_adj: dict[int, set[int]] = {i: set() for i in range(len(sccs))}
    for src, targets in adj.items():
        src_comp = node_to_comp.get(src)
        if src_comp is None:
            continue
        for tgt in targets:
            tgt_comp = node_to_comp.get(tgt)
            if tgt_comp is not None and tgt_comp != src_comp:
                comp_adj[src_comp].add(tgt_comp)

    return node_to_comp, {k: list(v) for k, v in comp_adj.items()}


def _find_longest_paths(
    comp_adj: dict[int, list[int]],
    n_comps: int,
) -> list[list[int]]:
    """Find all maximal paths in the condensation DAG.

    A maximal path starts at a source (no incoming edges) and ends at
    a sink (no outgoing edges). Returns paths as lists of component indices,
    sorted by length (longest first).
    """
    # Find sources (no incoming edges).
    has_incoming: set[int] = set()
    for targets in comp_adj.values():
        for t in targets:
            has_incoming.add(t)
    sources = [i for i in range(n_comps) if i not in has_incoming]

    if not sources:
        # All nodes have incoming edges — pick the one with fewest.
        in_count = [0] * n_comps
        for targets in comp_adj.values():
            for t in targets:
                in_count[t] += 1
        sources = [min(range(n_comps), key=lambda i: in_count[i])]

    # DFS from each source to find all maximal paths.
    paths: list[list[int]] = []

    def dfs(node: int, path: list[int]) -> None:
        targets = comp_adj.get(node, [])
        if not targets:
            paths.append(list(path))
            return
        for t in targets:
            if t not in set(path):  # avoid cycles (shouldn't happen in DAG)
                path.append(t)
                dfs(t, path)
                path.pop()

    for src in sources:
        dfs(src, [src])

    paths.sort(key=len, reverse=True)
    return paths


def _expand_component_path(
    comp_path: list[int],
    sccs: list[list[NodeId]],
    adj: dict[NodeId, list[NodeId]],
) -> list[NodeId]:
    """Expand a component-level path to a node-level chain.

    For each SCC in the path, picks one representative node. For
    non-trivial SCCs with cycles, picks a node that connects to the
    next component. Chains the representatives together.
    """
    if not comp_path:
        return []

    chain: list[NodeId] = []

    for path_idx, comp_idx in enumerate(comp_path):
        scc = sccs[comp_idx]

        if len(scc) == 1:
            chain.append(scc[0])
        else:
            # Pick a node that has an edge to the next component.
            if path_idx < len(comp_path) - 1:
                next_comp = sccs[comp_path[path_idx + 1]]
                next_set = set(next_comp)
                for nid in scc:
                    if any(t in next_set for t in adj.get(nid, [])):
                        chain.append(nid)
                        break
                else:
                    chain.append(scc[0])
            else:
                chain.append(scc[0])

    return chain


# ---------------------------------------------------------------------------
# Main: discover algebraic structure
# ---------------------------------------------------------------------------

def discover_algebraic_structure(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    min_weight: float = 0.05,
    min_chain_length: int = 3,
    min_scc_size: int = 2,
) -> dict[str, Any]:
    """Discover the algebraic skeleton of the transition graph.

    Extracts SCCs (group components), the condensation DAG (irreversible
    flow), maximal chains (candidate NNOs), and cycle structure.

    Called by Consolidation.consolidate().

    Returns stats dict with discovered structure.
    """
    adj = _extract_transition_graph(kg, min_weight=min_weight)
    if not adj:
        return {
            "transition_nodes": 0, "components": 0, "nontrivial_components": 0,
            "chains_found": 0, "longest_chain": 0, "cycles_found": 0,
        }

    # --- SCC decomposition ---
    raw_sccs = _tarjan_scc(adj)

    components: list[AlgebraicComponent] = []
    for scc in raw_sccs:
        members = frozenset(scc)
        is_trivial = len(scc) == 1 and scc[0] not in adj.get(scc[0], [])

        cycle_order: list[NodeId] = []
        cycle_length = 0
        if not is_trivial and len(scc) >= 2:
            cycle_order = _find_cycle_in_scc(members, adj)
            cycle_length = len(cycle_order)

        components.append(AlgebraicComponent(
            members=members,
            is_trivial=is_trivial,
            cycle_length=cycle_length,
            cycle_order=cycle_order,
        ))

    # --- Condensation DAG ---
    node_to_comp, comp_adj = _build_condensation(raw_sccs, adj)

    condensation_edges: list[tuple[int, int]] = []
    for src, targets in comp_adj.items():
        for tgt in targets:
            condensation_edges.append((src, tgt))

    # --- Chain extraction ---
    comp_paths = _find_longest_paths(comp_adj, len(raw_sccs))

    chains: list[Chain] = []
    for cp in comp_paths:
        if len(cp) < min_chain_length:
            continue
        node_chain = _expand_component_path(cp, raw_sccs, adj)
        if len(node_chain) >= min_chain_length:
            chains.append(Chain(nodes=node_chain, length=len(node_chain) - 1))

    # --- Materialize significant findings ---
    nontrivial_comps = [c for c in components if not c.is_trivial]
    cycles_found = sum(1 for c in components if c.cycle_length >= min_scc_size)

    # Materialize non-trivial SCCs as concept nodes (group-like structure).
    scc_nodes_created = 0
    for comp in nontrivial_comps:
        if len(comp.members) < min_scc_size:
            continue
        scc_key = ("__scc__", comp.members)
        if scc_key in kg._value_to_node:
            continue
        scc_nid = kg.get_or_create(scc_key)
        node = kg.node(scc_nid)
        if node is not None:
            node.resting = 0.3
        # Cocone edges: member → SCC node.
        for member_nid in comp.members:
            if member_nid in kg._nodes:
                edge = kg.get_or_create_edge(member_nid, scc_nid, role=COOCCURRENCE)
                edge.strengthen(0.3)
        scc_nodes_created += 1

    # Materialize chains as ordered sequences.
    # Store the chain ordering via transition edges between consecutive nodes.
    # These are CANDIDATE successor structures — Phase 3 (universal property
    # test) will verify which ones are genuine NNOs.
    chains_materialized = 0
    for chain in chains:
        chain_key = ("__chain__", tuple(chain.nodes))
        if chain_key in kg._value_to_node:
            continue
        chain_nid = kg.get_or_create(chain_key)
        node = kg.node(chain_nid)
        if node is not None:
            node.resting = 0.4
        # Cocone edges: chain members → chain node.
        for member_nid in chain.nodes:
            if member_nid in kg._nodes:
                edge = kg.get_or_create_edge(member_nid, chain_nid, role=COOCCURRENCE)
                edge.strengthen(0.2)
        chains_materialized += 1

    return {
        "transition_nodes": len(adj),
        "components": len(components),
        "nontrivial_components": len(nontrivial_comps),
        "cycles_found": cycles_found,
        "chains_found": len(chains),
        "longest_chain": max((c.length for c in chains), default=0),
        "scc_nodes_created": scc_nodes_created,
        "chains_materialized": chains_materialized,
    }
