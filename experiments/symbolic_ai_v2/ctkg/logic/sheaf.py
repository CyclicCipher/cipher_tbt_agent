"""
Sheaf Laplacian — quantitative consistency measure.

Each node has a "stalk" (its activation pattern across observations).
Each edge has a "restriction map" (how activations relate between
adjacent nodes). The sheaf Laplacian measures inconsistency: where
local data doesn't glue to a consistent global section.

The sheaf Laplacian energy for edge (A→B) with weight w is:
    E = |w| * (act_A * w - act_B)²

This generalizes the graph Laplacian (which assumes the restriction
map is identity). The minimum energy configuration is the harmonic
extension — the most consistent assignment of activations given
the edge structure.

The spectral gap of the sheaf Laplacian tells you how consistent
the overall data is. A small gap means the data is nearly consistent
everywhere. A large gap means there are significant inconsistencies
that the current edge structure can't explain — these are the places
where new structure needs to be discovered.

Three outputs:
1. **Per-edge inconsistency**: how much each edge's endpoints disagree.
   High inconsistency = the edge weight is wrong, or there's missing
   structure that this edge is trying (and failing) to capture.
2. **Per-node inconsistency**: total inconsistency at each node.
   High = this node is being pulled in contradictory directions by
   its neighbors. The node is at a type boundary or structural seam.
3. **Global consistency score**: total Laplacian energy normalized by
   edge count. Lower = more consistent. Track over time to measure
   whether consolidation is actually improving the graph's coherence.

No domain-specific code. Operates on abstract graph structure only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId, Edge, COOCCURRENCE, TRANSITION,
    ACTIVATION_THRESHOLD,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SheafDiagnostics:
    """Result of sheaf Laplacian computation."""
    edge_inconsistency: dict[tuple[NodeId, NodeId], float]
    node_inconsistency: dict[NodeId, float]
    global_energy: float           # total Laplacian energy
    global_consistency: float      # 1 - normalized energy (0=total disagreement, 1=perfect)
    worst_edges: list[tuple[NodeId, NodeId, float]]  # top inconsistent edges
    worst_nodes: list[tuple[NodeId, float]]           # top inconsistent nodes


# ---------------------------------------------------------------------------
# Sheaf Laplacian computation
# ---------------------------------------------------------------------------

def compute_sheaf_energy(
    kg: KnowledgeGraph,
    activations: dict[NodeId, float] | None = None,
    role_filter: int | None = None,
) -> SheafDiagnostics:
    """Compute the sheaf Laplacian energy of the current activation pattern.

    For each edge (A→B) with weight w:
    - If w > 0 (excitatory): E = w * (act_A - act_B)²
      Agreement: similar activations → low energy.
    - If w < 0 (inhibitory): E = |w| * (act_A * act_B)²
      Disagreement: both active → high energy (should be mutually exclusive).

    Parameters
    ----------
    kg : KnowledgeGraph
    activations : override activations. If None, uses kg.active_nodes().
    role_filter : only consider edges of this role (None = all).

    Returns SheafDiagnostics with per-edge, per-node, and global scores.
    """
    if activations is None:
        activations = kg.active_nodes()

    edge_incon: dict[tuple[NodeId, NodeId], float] = {}
    node_incon: dict[NodeId, float] = {}
    total_energy = 0.0
    edge_count = 0

    for (src, tgt), edge in kg._edges.items():
        if role_filter is not None and edge.role != role_filter:
            continue

        w = edge.weight
        if abs(w) < 0.01:
            continue

        act_s = activations.get(src, 0.0)
        act_t = activations.get(tgt, 0.0)

        if w > 0:
            # Excitatory: endpoints should agree.
            energy = w * (act_s - act_t) ** 2
        else:
            # Inhibitory: endpoints should not both be active.
            energy = abs(w) * (act_s * act_t) ** 2

        if energy > 0.001:
            edge_incon[(src, tgt)] = energy
            node_incon[src] = node_incon.get(src, 0.0) + energy
            node_incon[tgt] = node_incon.get(tgt, 0.0) + energy
            total_energy += energy

        edge_count += 1

    # Normalize: global consistency score in [0, 1].
    max_possible = max(edge_count, 1)  # rough normalization
    consistency = max(0.0, 1.0 - total_energy / max_possible)

    # Top inconsistent edges and nodes.
    worst_edges = sorted(
        ((s, t, e) for (s, t), e in edge_incon.items()),
        key=lambda x: -x[2],
    )[:20]

    worst_nodes = sorted(
        ((n, e) for n, e in node_incon.items()),
        key=lambda x: -x[1],
    )[:20]

    return SheafDiagnostics(
        edge_inconsistency=edge_incon,
        node_inconsistency=node_incon,
        global_energy=total_energy,
        global_consistency=consistency,
        worst_edges=worst_edges,
        worst_nodes=worst_nodes,
    )


# ---------------------------------------------------------------------------
# Sheaf consistency from observation replay
# ---------------------------------------------------------------------------

def compute_sheaf_consistency(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    max_snapshots: int = 200,
) -> dict[str, Any]:
    """Compute average sheaf consistency across recent activation snapshots.

    Replays stored snapshots through the sheaf Laplacian to measure
    how consistently the graph explains the observed activation patterns.

    Returns stats including per-snapshot consistency, trend, and
    the worst persistent inconsistencies (edges/nodes that are
    inconsistent across many snapshots).
    """
    snapshots = hippo.all_snapshots()
    recent = snapshots[max(0, since_index):]
    if len(recent) > max_snapshots:
        recent = recent[-max_snapshots:]

    if not recent:
        return {
            "snapshots_analyzed": 0,
            "mean_consistency": 1.0,
            "mean_energy": 0.0,
            "worst_edge_count": 0,
            "worst_node_count": 0,
        }

    total_consistency = 0.0
    total_energy = 0.0

    # Track persistent inconsistencies across snapshots.
    edge_incon_count: dict[tuple[NodeId, NodeId], int] = {}
    node_incon_count: dict[NodeId, int] = {}

    for snap in recent:
        # Use the snapshot's activation pattern.
        activations = {
            nid: lvl for nid, lvl in snap.activations.items()
            if nid in kg._nodes and lvl >= ACTIVATION_THRESHOLD
        }
        if not activations:
            total_consistency += 1.0
            continue

        diag = compute_sheaf_energy(kg, activations)
        total_consistency += diag.global_consistency
        total_energy += diag.global_energy

        # Count how often each edge/node appears as inconsistent.
        for (src, tgt), e in diag.edge_inconsistency.items():
            if e > 0.01:
                key = (src, tgt)
                edge_incon_count[key] = edge_incon_count.get(key, 0) + 1

        for nid, e in diag.node_inconsistency.items():
            if e > 0.01:
                node_incon_count[nid] = node_incon_count.get(nid, 0) + 1

    n = len(recent)
    mean_consistency = total_consistency / n
    mean_energy = total_energy / n

    # Persistent offenders: inconsistent in > 50% of snapshots.
    threshold = n * 0.5
    persistent_edges = [
        (src, tgt, count)
        for (src, tgt), count in edge_incon_count.items()
        if count > threshold
    ]
    persistent_nodes = [
        (nid, count)
        for nid, count in node_incon_count.items()
        if count > threshold
    ]

    return {
        "snapshots_analyzed": n,
        "mean_consistency": round(mean_consistency, 4),
        "mean_energy": round(mean_energy, 4),
        "worst_edge_count": len(persistent_edges),
        "worst_node_count": len(persistent_nodes),
    }


# ---------------------------------------------------------------------------
# Main entry point for consolidation
# ---------------------------------------------------------------------------

def discover_sheaf_structure(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
) -> dict[str, Any]:
    """Run sheaf Laplacian analysis during consolidation.

    Computes consistency metrics and identifies structural seams.
    Does NOT modify the graph — purely diagnostic. The information
    can guide future consolidation phases (e.g., pruning persistently
    inconsistent edges, or flagging boundary nodes for FCA attention).

    Called by Consolidation.consolidate().
    """
    return compute_sheaf_consistency(kg, hippo, since_index=since_index)
