"""
Consolidation — the slow path.

Runs periodically (not every timestep). Replays stored activation patterns
from Hippocampus to re-strengthen important edges, prunes dead structure,
and discovers higher-level abstractions (colimits, morphisms).

This is the symbolic analog of sleep: offline processing that stabilises
what was learned during waking (the fast path).

Three operations:

1. **Replay** — re-run spread→learn on consecutive snapshot pairs.
   Uses _outgoing adjacency index for O(active * degree) per pair.

2. **Prune** — remove edges with deeply negative weight and isolated nodes.
   Maintains _outgoing adjacency index consistency.

3. **Colimit/morphism formation** — delegated to colimit.py and morphism.py.
   These discover directed diagrams and consistent input→output patterns
   from observation records, with no domain-specific knowledge.

All structure discovery is domain-agnostic. No function in this file
references digits, separators, carry rules, NNO tokens, or any other
domain vocabulary. Structure emerges from graph dynamics and observation
patterns only.
"""
from __future__ import annotations

from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId, Edge, TRANSITION, COOCCURRENCE,
    ACTIVATION_THRESHOLD,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


# ---------------------------------------------------------------------------
# Replay (selective: skip low-surprise pairs)
# ---------------------------------------------------------------------------

def replay(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    n_passes: int = 1,
    replay_strength: float = 0.1,
    since_index: int = 0,
) -> dict[str, Any]:
    """Replay consecutive snapshot pairs with Hebbian edge updates.

    Only replays snapshots from since_index onward (incremental).
    Uses _outgoing adjacency index: O(active * degree) per pair.
    """
    snapshots = hippo.all_snapshots()
    # Only replay recent snapshots.
    snapshots = snapshots[max(0, since_index):]
    if len(snapshots) < 2:
        return {"replayed": 0, "edges_strengthened": 0, "edges_weakened": 0}

    total_strengthened = 0
    total_weakened = 0
    total_replayed = 0
    total_skipped = 0

    for _ in range(n_passes):
        for i in range(len(snapshots) - 1):
            snap_t = snapshots[i]
            snap_t1 = snapshots[i + 1]

            active_t = {nid for nid, lvl in snap_t.activations.items()
                        if nid in kg._nodes and lvl >= ACTIVATION_THRESHOLD}
            active_t1 = {nid for nid, lvl in snap_t1.activations.items()
                         if nid in kg._nodes and lvl >= ACTIVATION_THRESHOLD}

            if not active_t or not active_t1:
                total_skipped += 1
                continue

            if active_t == active_t1:
                total_skipped += 1
                continue

            for src_nid in active_t:
                for edge in kg._outgoing.get(src_nid, ()):
                    if edge.role != TRANSITION:
                        continue
                    if edge.target in active_t1:
                        edge.strengthen(replay_strength)
                        total_strengthened += 1
                    else:
                        edge.weaken(replay_strength)
                        total_weakened += 1

            total_replayed += 1

    return {
        "replayed": total_replayed,
        "edges_strengthened": total_strengthened,
        "edges_weakened": total_weakened,
    }


# ---------------------------------------------------------------------------
# Prune (maintains _outgoing index)
# ---------------------------------------------------------------------------

def prune(
    kg: KnowledgeGraph,
    edge_threshold: float = -0.5,
    node_min_resting: float = 0.001,
) -> dict[str, int]:
    """Remove dead edges and isolated nodes.

    Maintains _outgoing adjacency index consistency.
    """
    dead_edges = [
        key for key, edge in kg._edges.items()
        if edge.weight < edge_threshold
    ]
    for src, tgt in dead_edges:
        kg.remove_edge(src, tgt)

    connected: set[NodeId] = set()
    for (src, tgt) in kg._edges:
        connected.add(src)
        connected.add(tgt)

    dead_nodes = [
        nid for nid, node in kg._nodes.items()
        if nid not in connected and node.resting < node_min_resting
    ]
    for nid in dead_nodes:
        del kg._nodes[nid]
        kg._outgoing.pop(nid, None)
        to_remove_val = [v for v, n in kg._value_to_node.items() if n == nid]
        for v in to_remove_val:
            del kg._value_to_node[v]

    return {
        "edges_pruned": len(dead_edges),
        "nodes_pruned": len(dead_nodes),
    }


# ---------------------------------------------------------------------------
# Natural transformation discovery (domain-agnostic)
# ---------------------------------------------------------------------------

class MorphismTable:
    """A morphism represented as a mapping from source nodes to target nodes.

    Two tables are parallel if they have the same domain (set of sources).
    A natural transformation from F to G exists if the component map
    eta_a : F(a) -> G(a) preserves domain structure.
    """
    __slots__ = ('table', 'weight', 'tag')

    def __init__(self, table: dict[NodeId, NodeId], weight: float = 1.0,
                 tag: Any = None):
        self.table = table
        self.weight = weight
        self.tag = tag

    @property
    def domain(self) -> frozenset[NodeId]:
        return frozenset(self.table.keys())

    @property
    def codomain(self) -> frozenset[NodeId]:
        return frozenset(self.table.values())

    def is_identity(self) -> bool:
        return all(s == t for s, t in self.table.items())

    def agrees_with(self, other: "MorphismTable", threshold: float = 0.8) -> bool:
        shared = self.domain & other.domain
        if not shared:
            return False
        agree = sum(1 for s in shared if self.table[s] == other.table[s])
        return agree / len(shared) >= threshold

    def isomorphic_to(self, other: "MorphismTable", threshold: float = 0.8) -> bool:
        return self.agrees_with(other, threshold)


def find_natural_transformations(
    kg: KnowledgeGraph,
    morphism_tables: list[MorphismTable],
    min_cluster_size: int = 2,
    agreement_threshold: float = 0.8,
) -> dict[str, Any]:
    """Universal natural transformation discovery.

    Given a list of morphism tables (each representing a parallel morphism
    with a tag), finds clusters of tables that are naturally isomorphic —
    they implement the same mapping pattern.

    For each cluster, the dominant table (most complete) becomes the template.
    All tables in the cluster that are missing entries get completed from
    the template.

    This is domain-agnostic: it works on ANY set of parallel morphisms.
    """
    if len(morphism_tables) < min_cluster_size:
        return {"clusters": [], "extended": 0}

    non_empty = [i for i in range(len(morphism_tables)) if morphism_tables[i].table]
    empty = [i for i in range(len(morphism_tables)) if not morphism_tables[i].table]

    remaining = list(non_empty)
    clusters: list[tuple[MorphismTable, list[int]]] = []

    while remaining:
        seed_idx = max(remaining, key=lambda i: len(morphism_tables[i].table))
        seed = morphism_tables[seed_idx]

        cluster_indices = []
        for idx in remaining:
            if morphism_tables[idx].isomorphic_to(seed, agreement_threshold):
                cluster_indices.append(idx)

        if len(cluster_indices) >= min_cluster_size:
            clusters.append((seed, cluster_indices))

        for idx in cluster_indices:
            remaining.remove(idx)
        if len(cluster_indices) < min_cluster_size and seed_idx in remaining:
            remaining.remove(seed_idx)

    if clusters and empty:
        best_cluster_idx = 0
        best_identity = 0
        for ci, (template, members) in enumerate(clusters):
            id_count = sum(1 for s, t in template.table.items() if s == t)
            if id_count > best_identity:
                best_identity = id_count
                best_cluster_idx = ci
        clusters[best_cluster_idx][1].extend(empty)

    extended = 0
    completed_domain: set[NodeId] = set()

    for template, member_indices in clusters:
        all_sources: set[NodeId] = set()
        for mt in morphism_tables:
            all_sources |= mt.domain
            all_sources |= mt.codomain

        from collections import Counter
        completed: dict[NodeId, NodeId] = {}
        for src in all_sources:
            targets = Counter()
            for idx in member_indices:
                t = morphism_tables[idx].table.get(src)
                if t is not None:
                    targets[t] += 1

            if targets:
                completed[src] = targets.most_common(1)[0][0]
            elif template.is_identity():
                completed[src] = src
                completed_domain.add(src)

        for idx in member_indices:
            mt = morphism_tables[idx]
            for src, tgt in completed.items():
                if src not in mt.table:
                    mt.table[src] = tgt
                    extended += 1

    return {
        "clusters": [(t.tag, [morphism_tables[i].tag for i in idxs])
                      for t, idxs in clusters],
        "extended": extended,
        "completed_domain": completed_domain,
        "n_clusters": len(clusters),
    }


# ---------------------------------------------------------------------------
# Main consolidation entry point
# ---------------------------------------------------------------------------

def consolidate(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    replay_passes: int = 1,
    replay_strength: float = 0.1,
    since_index: int = 0,
) -> dict[str, Any]:
    """Consolidation: replay → prune → FCA → algebra → initial algebra → sheaf → colimit → morphism.

    since_index: only process snapshots/observations from this index onward
    (incremental consolidation — don't re-process old data).
    """
    stats: dict[str, Any] = {}

    replay_stats = replay(kg, hippo, n_passes=replay_passes,
                          replay_strength=replay_strength,
                          since_index=since_index)
    stats.update({f"replay_{k}": v for k, v in replay_stats.items()})

    prune_stats = prune(kg)
    stats.update({f"prune_{k}": v for k, v in prune_stats.items()})

    # PMI + IDF cache: compute from observation records for action selection.
    from experiments.symbolic_ai_v2.ctkg.logic.initial_algebra import _compute_pmi
    kg._pmi = _compute_pmi(hippo, kg, since_index=since_index)
    stats["pmi_pairs"] = len(kg._pmi)

    # IDF: log(N / count(token)) for each token.
    import math as _math
    all_obs = hippo.all_observations()
    obs_recent = all_obs[max(0, since_index):]
    n_obs = max(len(obs_recent), 1)
    _token_doc_count: dict[int, int] = {}
    for _obs in obs_recent:
        for _nid in set(_obs.token_nids):
            _token_doc_count[_nid] = _token_doc_count.get(_nid, 0) + 1
    kg._idf = {
        nid: _math.log(n_obs / count) if count > 0 else 0.0
        for nid, count in _token_doc_count.items()
    }

    # FCA: discover type hierarchy from observation co-occurrence.
    from experiments.symbolic_ai_v2.ctkg.logic.fca import discover_fca_structure
    fca_stats = discover_fca_structure(kg, hippo, since_index=since_index)
    stats.update({f"fca_{k}": v for k, v in fca_stats.items()})

    # Algebraic skeleton: SCCs, chains, cycles from transition structure.
    from experiments.symbolic_ai_v2.ctkg.logic.algebra import discover_algebraic_structure
    alg_stats = discover_algebraic_structure(kg, hippo, since_index=since_index)
    stats.update({f"alg_{k}": v for k, v in alg_stats.items()})

    # Initial algebra: universal property test on candidate chains.
    from experiments.symbolic_ai_v2.ctkg.logic.initial_algebra import discover_initial_algebras
    ia_stats = discover_initial_algebras(kg, hippo, since_index=since_index)
    stats.update({f"ia_{k}": v for k, v in ia_stats.items()})

    # Sheaf Laplacian: consistency diagnostics.
    from experiments.symbolic_ai_v2.ctkg.logic.sheaf import discover_sheaf_structure
    sheaf_stats = discover_sheaf_structure(kg, hippo, since_index=since_index)
    stats.update({f"sheaf_{k}": v for k, v in sheaf_stats.items()})

    from experiments.symbolic_ai_v2.ctkg.logic.colimit import find_colimits
    colimit_stats = find_colimits(kg, hippo, since_index=since_index)
    stats.update({f"colimit_{k}": v for k, v in colimit_stats.items()})

    from experiments.symbolic_ai_v2.ctkg.logic.morphism import discover_morphisms
    morph_stats = discover_morphisms(kg, hippo, since_index=since_index)
    stats.update({f"morphism_{k}": v for k, v in morph_stats.items()})

    return stats
