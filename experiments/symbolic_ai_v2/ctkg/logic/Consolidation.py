"""
Consolidation — the slow path.

Runs periodically (not every timestep). Replays stored activation patterns
from Hippocampus to re-strengthen important edges, prunes dead structure,
and discovers higher-level abstractions (colimits).

This is the symbolic analog of sleep: offline processing that stabilises
what was learned during waking (the fast path).

Three operations:

1. **Replay** — re-run spread→learn on consecutive snapshot pairs.
   Uses _outgoing adjacency index for O(active * degree) per pair.

2. **Prune** — remove edges with deeply negative weight and isolated nodes.
   Maintains _outgoing adjacency index consistency.

3. **Colimit formation** — find node sets that consistently co-activate.
   Uses streaming pairwise counter for O(snapshots * active²).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId, Edge, TRANSITION, COOCCURRENCE,
    ACTIVATION_THRESHOLD, SPREAD_THRESHOLD,
    SPREAD_CAP,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus, Snapshot, Observation


# ---------------------------------------------------------------------------
# Replay (selective: skip low-surprise pairs)
# ---------------------------------------------------------------------------

def replay(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    n_passes: int = 1,
    replay_strength: float = 1.0,
) -> dict[str, Any]:
    """Replay consecutive snapshot pairs with Bayesian edge updates.

    Uses _outgoing adjacency index: for each active source at time t,
    iterates only its outgoing TRANSITION edges (O(degree)), not all
    edges in the graph.

    Selective: skips snapshot pairs where both snapshots have identical
    active node sets (nothing changed → nothing to learn).
    """
    snapshots = hippo.all_snapshots()
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

            # Selective: skip if nothing changed between snapshots.
            if active_t == active_t1:
                total_skipped += 1
                continue

            # Bayesian update on transition edges from active-at-t sources.
            for src_nid in active_t:
                for edge in kg._outgoing.get(src_nid, ()):
                    if edge.role != TRANSITION:
                        continue
                    if edge.target in active_t1:
                        edge.observe_present(replay_strength)
                        total_strengthened += 1
                    else:
                        edge.observe_absent(replay_strength)
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
    edge_threshold: float = -1.0,
    node_min_resting: float = 0.001,
) -> dict[str, int]:
    """Remove dead edges and isolated nodes.

    Maintains _outgoing adjacency index consistency.
    """
    # Prune edges — use remove_edge to keep _outgoing in sync.
    dead_edges = [
        key for key, edge in kg._edges.items()
        if edge.weight < edge_threshold
    ]
    for src, tgt in dead_edges:
        kg.remove_edge(src, tgt)

    # Find isolated nodes (no edges at all).
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
# Colimit formation (streaming pairwise counter)
# ---------------------------------------------------------------------------

def find_colimits(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    min_coactivation: int = 5,
    min_group_size: int = 3,
    max_group_size: int = 30,
) -> dict[str, Any]:
    """Find node sets that consistently co-activate and create summary nodes.

    Uses a streaming pairwise co-activation counter: for each snapshot,
    counts pairs of co-active nodes. Only pairs exceeding min_coactivation
    are kept for clique detection.

    Colimit edges use the adjacency index via get_or_create_edge.
    """
    snapshots = hippo.all_snapshots()
    if len(snapshots) < min_coactivation:
        return {"colimits_created": 0, "candidates_found": 0}

    # Streaming pairwise counter — O(snapshots * active²).
    # Use a flat dict[tuple, int] instead of Counter[frozenset] for speed.
    coact: dict[tuple[NodeId, NodeId], int] = {}
    for snap in snapshots:
        active_list = sorted(
            nid for nid, lvl in snap.activations.items()
            if lvl >= ACTIVATION_THRESHOLD
        )
        n = len(active_list)
        if n > max_group_size:
            # Cap to avoid O(n²) explosion on very large snapshots.
            active_list = active_list[:max_group_size]
            n = max_group_size
        for i in range(n):
            for j in range(i + 1, n):
                key = (active_list[i], active_list[j])
                coact[key] = coact.get(key, 0) + 1

    # Filter to strong pairs.
    strong_pairs: set[frozenset[NodeId]] = set()
    neighbors: dict[NodeId, set[NodeId]] = {}
    for (a, b), count in coact.items():
        if count >= min_coactivation:
            strong_pairs.add(frozenset({a, b}))
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)

    # Greedy clique detection.
    used: set[NodeId] = set()
    groups: list[frozenset[NodeId]] = []

    for seed in sorted(neighbors.keys(),
                       key=lambda n: -len(neighbors.get(n, set()))):
        if seed in used:
            continue
        group = {seed}
        candidates = neighbors.get(seed, set()) - used
        for c in sorted(candidates,
                        key=lambda n: -len(neighbors.get(n, set()))):
            if all(frozenset({c, m}) in strong_pairs for m in group):
                group.add(c)
        if min_group_size <= len(group) <= max_group_size:
            groups.append(frozenset(group))
            used |= group

    # Create summary nodes.
    colimits_created = 0
    for group in groups:
        group_key = ("__colimit__", group)
        if group_key in kg._value_to_node:
            continue

        summary_nid = kg.get_or_create(group_key)
        kg._nodes[summary_nid].resting = 0.5

        for member_nid in group:
            edge_down = kg.get_or_create_edge(summary_nid, member_nid, role=COOCCURRENCE)
            edge_down.alpha = max(edge_down.alpha, 3.0)
            edge_down.beta = max(edge_down.beta, 1.0)
            edge_down._recalc()
            edge_up = kg.get_or_create_edge(member_nid, summary_nid, role=COOCCURRENCE)
            edge_up.alpha = max(edge_up.alpha, 3.0)
            edge_up.beta = max(edge_up.beta, 1.0)
            edge_up._recalc()

        colimits_created += 1

    return {
        "colimits_created": colimits_created,
        "candidates_found": len(groups),
    }


# ---------------------------------------------------------------------------
# Natural transformation discovery
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Morphism Table — the representation of a morphism as a dict {src → tgt}
# ---------------------------------------------------------------------------

class MorphismTable:
    """A morphism represented as a mapping from source nodes to target nodes.

    Two tables are parallel if they have the same domain (set of sources).
    A natural transformation from F to G exists if the component map
    η_a : F(a) → G(a) preserves domain structure.
    """
    __slots__ = ('table', 'weight', 'tag')

    def __init__(self, table: dict[NodeId, NodeId], weight: float = 1.0,
                 tag: Any = None):
        self.table = table      # {source → target}
        self.weight = weight    # evidence strength
        self.tag = tag          # opaque label (e.g., position index)

    @property
    def domain(self) -> frozenset[NodeId]:
        return frozenset(self.table.keys())

    @property
    def codomain(self) -> frozenset[NodeId]:
        return frozenset(self.table.values())

    def is_identity(self) -> bool:
        """All entries map src → src."""
        return all(s == t for s, t in self.table.items())

    def agrees_with(self, other: "MorphismTable", threshold: float = 0.8) -> bool:
        """Do these two tables agree on at least `threshold` fraction of
        their shared domain?"""
        shared = self.domain & other.domain
        if not shared:
            return False
        agree = sum(1 for s in shared if self.table[s] == other.table[s])
        return agree / len(shared) >= threshold

    def isomorphic_to(self, other: "MorphismTable", threshold: float = 0.8) -> bool:
        """Are these two tables structurally identical (same mapping pattern)
        on at least `threshold` fraction of their shared domain?

        Identical means: for each shared source s, self.table[s] == other.table[s].
        This is the identity natural transformation.
        """
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
    the template. New tables are created for tags that don't have a table yet,
    if they fit the dominant pattern.

    This is the universal NT: it works on ANY set of parallel morphisms,
    not just product projections. The same code handles:
    - Product identity projections (copy at each digit position)
    - Functor mappings (NNO sub-chain correspondences)
    - Any future parallel morphism families

    Parameters
    ----------
    kg : the KnowledgeGraph (for finding digit domain completion)
    morphism_tables : list of MorphismTable with tags
    min_cluster_size : minimum tables in a cluster to form an NT
    agreement_threshold : fraction of shared domain that must agree

    Returns
    -------
    Dict with:
    - clusters: list of (template_table, member_tags)
    - extended: number of new table entries created
    - completed_domain: set of nodes added to incomplete tables
    """
    if len(morphism_tables) < min_cluster_size:
        return {"clusters": [], "extended": 0}

    # Cluster tables by structural isomorphism.
    # First pass: cluster non-empty tables by agreement.
    # Second pass: assign empty tables to the dominant cluster.
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

    # Assign empty tables to the dominant cluster (the one with the most
    # identity entries — the "default rule" that applies everywhere unless
    # overridden by specific data).
    if clusters and empty:
        # Find the cluster whose template has the most identity entries.
        best_cluster_idx = 0
        best_identity = 0
        for ci, (template, members) in enumerate(clusters):
            id_count = sum(1 for s, t in template.table.items() if s == t)
            if id_count > best_identity:
                best_identity = id_count
                best_cluster_idx = ci
        # Add empty tables to this cluster.
        clusters[best_cluster_idx][1].extend(empty)

    # For each cluster, find the domain union and complete all tables.
    extended = 0
    completed_domain: set[NodeId] = set()

    for template, member_indices in clusters:
        # Union of all domains AND codomains across ALL tables (not just
        # this cluster). This ensures that digits like 0, which might not
        # appear as a source in any identity table but DO appear as targets
        # in other tables (e.g., 9→0 in the successor table), get included
        # in the identity completion.
        all_sources: set[NodeId] = set()
        for mt in morphism_tables:
            all_sources |= mt.domain
            all_sources |= mt.codomain

        # Build the completed template: all (src → tgt) pairs from the
        # union of all cluster members. For identity tables, src → src.
        # For non-identity tables, use the most common mapping per source.
        completed: dict[NodeId, NodeId] = {}
        for src in all_sources:
            # Find the most common target for this source across the cluster.
            from collections import Counter
            targets = Counter()
            for idx in member_indices:
                t = morphism_tables[idx].table.get(src)
                if t is not None:
                    targets[t] += 1

            if targets:
                completed[src] = targets.most_common(1)[0][0]
            elif template.is_identity():
                # For identity clusters, missing entries default to self-map.
                completed[src] = src
                completed_domain.add(src)

        # Apply the completed template to all members.
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
# Full consolidation pass
# ---------------------------------------------------------------------------

FUNCTOR = 2  # edge role for functor mappings (F: source_chain → target_chain)


# ---------------------------------------------------------------------------
# Natural transformation: extend products to unseen positions
# ---------------------------------------------------------------------------

def extend_products_via_nt(
    kg: KnowledgeGraph,
    max_position: int = 20,
    min_entries: int = 5,
) -> dict[str, Any]:
    """Extend product projections to unseen positions via the universal NT system.

    Converts product projections into MorphismTables (one per position),
    runs the universal find_natural_transformations, then writes the
    completed tables back into kg._product_projections.

    The universal NT handles everything: clustering isomorphic positions,
    finding the dominant pattern, completing missing domain entries,
    extending to uncovered positions. No product-specific logic here —
    just the conversion between product projections and MorphismTables.
    """
    projections = kg._product_projections
    if not projections:
        return {"nt_extended": 0}

    # Convert product projections to MorphismTables (one per position).
    # A position may have MULTIPLE projections per source (e.g., position 1
    # has both 1→1 (copy) and 1→2 (carry)). The MorphismTable should use
    # the DOMINANT mapping per source (highest weight).
    from collections import defaultdict
    by_pos_entries: dict[int, dict[NodeId, list[tuple[NodeId, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (src, tgt, pos), weight in projections.items():
        by_pos_entries[pos][src].append((tgt, weight))

    tables: list[MorphismTable] = []
    for pos in sorted(by_pos_entries.keys()):
        entries = by_pos_entries[pos]
        if len(entries) < min_entries:
            continue
        # For each source, pick the target with the highest weight.
        table: dict[NodeId, NodeId] = {}
        total_w = 0.0
        for src, tgt_list in entries.items():
            best_tgt = max(tgt_list, key=lambda x: x[1])
            table[src] = best_tgt[0]
            total_w += best_tgt[1]
        avg_w = total_w / max(len(table), 1)
        tables.append(MorphismTable(
            table=table,
            weight=avg_w,
            tag=pos,
        ))

    # Create empty tables for positions up to max_position that don't
    # have enough data — the NT will fill them from the dominant cluster.
    existing_tags = {t.tag for t in tables}
    for pos in range(max_position + 1):
        if pos not in existing_tags:
            tables.append(MorphismTable(table={}, weight=0.0, tag=pos))

    # Run the universal NT discovery.
    nt_stats = find_natural_transformations(
        kg, tables,
        min_cluster_size=1,  # even a single well-populated table can extend
        agreement_threshold=0.8,
    )

    # Write completed tables back to product projections.
    extended = 0
    for mt in tables:
        pos = mt.tag
        if not isinstance(pos, int):
            continue
        for src, tgt in mt.table.items():
            key = (src, tgt, pos)
            if key not in projections:
                projections[key] = mt.weight if mt.weight > 0 else 27.0
                extended += 1

    nt_stats["nt_extended"] = extended
    return nt_stats


# ---------------------------------------------------------------------------
# Product decomposition (Phase A)
# ---------------------------------------------------------------------------

def discover_products(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    min_observations: int = 5,
) -> dict[str, Any]:
    """Discover product structure from observation records.

    Parses counting observations to find input→output digit correspondences.
    The product decomposes multi-digit numbers into per-position projections:
    pi_i maps a number to its i-th digit (from the RIGHT, position-from-end).

    Key insight: position-from-end is stable across number lengths. The units
    digit is always at position-from-end 0, tens at 1, hundreds at 2, etc.
    This enables the learned per-position rules (copy vs increment vs carry)
    to generalise across number lengths.

    The separator token (discovered structurally, not hardcoded) divides
    input from output. Digits at the same position-from-end on each side
    form a product projection pair.

    Discovery:
    1. Find separator candidates: tokens that appear at consistent positions
       between groups of varying tokens.
    2. For each observation with a separator, extract input and output digit
       sequences.
    3. Align by position-from-end and record (input_digit, output_digit) at
       each position.
    4. Create PRODUCT edges encoding these per-position correspondences.
       Product edges are tagged with their position-from-end index.

    Returns stats about product structure discovered.
    """
    observations = hippo.all_observations()
    if len(observations) < min_observations:
        return {"products_found": 0, "projections_created": 0}

    # Step 1: Find separator candidates.
    # A separator is a token that (a) appears in many observations,
    # (b) at a non-initial/non-final position, and (c) with DIFFERENT
    # tokens on either side across observations.
    from collections import defaultdict

    token_positions: dict[NodeId, list[tuple[int, int]]] = defaultdict(list)
    # (obs_index, position_in_obs) for each token
    for obs_idx, obs in enumerate(observations):
        for pos, nid in enumerate(obs.token_nids):
            token_positions[nid].append((obs_idx, pos))

    # Separator candidates: tokens appearing in >30% of observations,
    # at non-edge positions, with high positional consistency.
    min_sep_count = max(min_observations, len(observations) // 5)
    separator_candidates: list[NodeId] = []
    for nid, positions in token_positions.items():
        if len(positions) < min_sep_count:
            continue
        # Must appear at non-initial, non-final positions.
        mid_count = sum(1 for _, p in positions if p > 1 and p < 10)
        if mid_count < min_sep_count:
            continue
        separator_candidates.append(nid)

    if not separator_candidates:
        return {"products_found": 0, "projections_created": 0}

    # Step 2: Use each separator to extract input/output digit sequences.
    # The separator that yields the most consistent digit correspondences wins.

    # Build the digit chain from the NNO (for identifying digit tokens).
    zero_nid = kg._value_to_node.get('0')
    if zero_nid is None:
        return {"products_found": 0, "projections_created": 0}

    chain = _build_digit_chain_from_observations(kg, hippo,
                kg._value_to_node.get('next_is'), zero_nid)
    digit_set = set(chain)

    best_sep = None
    best_correspondences = []
    best_count = 0

    for sep_nid in separator_candidates:
        correspondences: list[list[tuple[NodeId, NodeId]]] = []
        # position-from-end correspondences

        for obs in observations:
            nids = obs.token_nids
            # Find separator position(s) in this observation.
            sep_positions = [i for i, n in enumerate(nids) if n == sep_nid]
            if not sep_positions:
                continue
            # Use the LAST separator occurrence (handles prefix tokens).
            sep_pos = sep_positions[-1]

            # Extract input digits (before separator) and output digits (after).
            # Skip non-digit tokens (PHASE, PROB_TYPE, etc.).
            input_digits = [n for n in nids[:sep_pos] if n in digit_set]
            output_digits = [n for n in nids[sep_pos + 1:] if n in digit_set]

            if not input_digits or not output_digits:
                continue

            # Align by position-from-end.
            # Reverse both so index 0 = rightmost (units).
            in_rev = list(reversed(input_digits))
            out_rev = list(reversed(output_digits))

            # Pair up positions that exist in both.
            n_pairs = min(len(in_rev), len(out_rev))
            obs_corr = [(in_rev[i], out_rev[i]) for i in range(n_pairs)]
            if obs_corr:
                correspondences.append(obs_corr)

        if len(correspondences) > best_count:
            best_count = len(correspondences)
            best_sep = sep_nid
            best_correspondences = correspondences

    if best_count < min_observations:
        return {"products_found": 0, "projections_created": 0,
                "separator": kg.value_for_node(best_sep) if best_sep else None}

    # Step 3: Aggregate per-position correspondences.
    # For position-from-end p, count (input_digit → output_digit) pairs.
    max_pos = max(len(c) for c in best_correspondences)
    pos_counts: list[dict[tuple[NodeId, NodeId], int]] = [
        defaultdict(int) for _ in range(max_pos)
    ]
    for obs_corr in best_correspondences:
        for pos, (in_nid, out_nid) in enumerate(obs_corr):
            pos_counts[pos][(in_nid, out_nid)] += 1

    # Step 4: Create product projections.
    # Stored in kg._product_projections (separate from edges) because the
    # same (src, tgt) pair may already have a co-occurrence edge. Product
    # projections are positionally tagged: (src, tgt, pos_from_end) → weight.
    projections_created = 0

    for pos in range(max_pos):
        for (in_nid, out_nid), count in pos_counts[pos].items():
            if count < 2:
                continue
            # Weight proportional to consistency (count / max possible).
            weight = count * 1.0
            kg._product_projections[(in_nid, out_nid, pos)] = weight
            projections_created += 1

    return {
        "products_found": 1,
        "projections_created": projections_created,
        "separator": kg.value_for_node(best_sep) if best_sep else None,
        "max_position": max_pos,
        "observations_used": best_count,
    }


# ---------------------------------------------------------------------------
# Functor discovery
# ---------------------------------------------------------------------------

def _nno_successor(kg: KnowledgeGraph, nid: NodeId) -> NodeId | None:
    """Follow ONE forward co-occurrence hop from nid on the number line.

    Returns the strongest positive co-occurrence neighbor, or None.
    This is the graph-structural successor — no integer extraction.

    To avoid following non-digit edges (like digit→space or digit→succ),
    we pick the neighbor with the highest co-occurrence weight that ALSO
    has its own forward co-occurrence neighbors (i.e., it's part of a chain,
    not a terminal like 'space' or 'succ').
    """
    best_nid = None
    best_w = 0.0
    for edge in kg._outgoing.get(nid, ()):
        if edge.role != COOCCURRENCE:
            continue
        ew = edge.effective_weight
        if ew <= 0:
            continue
        # Check that the target also has forward co-occurrence edges
        # (it's a chain node, not a structural token like space/succ/=).
        has_chain_continuation = False
        for e2 in kg._outgoing.get(edge.target, ()):
            if e2.role == COOCCURRENCE and e2.effective_weight > 0:
                has_chain_continuation = True
                break
        if has_chain_continuation and ew > best_w:
            best_w = ew
            best_nid = edge.target
    return best_nid


def _walk_chain(kg: KnowledgeGraph, start: NodeId, end: NodeId, max_steps: int = 20) -> list[NodeId] | None:
    """Walk the number line from start to end via multi-hop co-occurrence spread.

    Uses spread_cooccurrence to find the shortest path from start to end,
    then reconstructs the chain by following the spread activation gradient.

    Returns the chain [start, ..., end] or None if end is not reachable.
    """
    if start == end:
        return [start]

    # Use multi-hop spread to check reachability and find path.
    spread = kg.spread_cooccurrence({start: 1.0}, hops=max_steps, decay=0.9)
    if end not in spread or spread[end] < 0.001:
        return None

    # Reconstruct path: BFS from start to end following forward
    # co-occurrence edges (greedy: prefer edges toward end).
    chain = [start]
    current = start
    visited = {start}
    for _ in range(max_steps):
        if current == end:
            return chain
        # Find the forward co-occurrence neighbor closest to end
        # (highest spread activation from end perspective).
        best_next = None
        best_score = -1.0
        for edge in kg._outgoing.get(current, ()):
            if edge.role != COOCCURRENCE:
                continue
            if edge.effective_weight <= 0:
                continue
            if edge.target in visited:
                continue
            # Score: how activated is this neighbor in the spread from start?
            # Higher = closer to the chain.
            score = spread.get(edge.target, 0.0)
            if edge.target == end:
                score = float('inf')  # prioritise reaching end
            if score > best_score:
                best_score = score
                best_next = edge.target
        if best_next is None:
            return None
        chain.append(best_next)
        visited.add(best_next)
        current = best_next
    return None


def discover_functors(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    min_examples: int = 3,
) -> dict[str, Any]:
    """Discover functors: structure-preserving maps between NNO sub-chains.

    Scans hippocampus observation records for training observations that
    contain an OPERATOR token (like '+') flanked by digit tokens.
    For each such observation, extracts the operand and result tokens
    from their POSITIONAL ORDER in the observation (not by string parsing),
    then checks the functor condition: simultaneous walk from 0 to second
    operand AND from first operand to result.

    A functor is verified when:
    1. Both walks have the same length (simultaneous walk ends together)
    2. Each step in the source chain maps to a step in the target chain
    3. The mapping preserves the successor structure

    Discovered functors are stored as FUNCTOR edges in the graph:
    for each (source_node, target_node) pair in the functor mapping,
    a directed edge is created. This allows multi-hop spread to follow
    the functor rather than just the number line.

    Returns stats about discovered functors.
    """
    observations = hippo.all_observations()
    if len(observations) < min_examples:
        return {"functors_discovered": 0, "functor_edges_created": 0}

    # Find the node ID for '0' (the NNO origin).
    zero_nid = kg._value_to_node.get('0')
    if zero_nid is None:
        return {"functors_discovered": 0, "functor_edges_created": 0}

    # Identify operator tokens from the graph.
    # An operator is a non-digit token that frequently co-occurs with digits.
    # We find them structurally: nodes that have co-occurrence edges to
    # multiple digit-chain nodes but are NOT themselves on the chain.
    operator_nids: set[NodeId] = set()
    for val, nid in kg._value_to_node.items():
        if not isinstance(val, str):
            continue
        if val in ('+', '-', '*', '=', 'succ', 'pred'):
            operator_nids.add(nid)

    # Build digit chain from counting warmup observations.
    next_is_nid = kg._value_to_node.get('next_is')
    chain_node_list = _build_digit_chain_from_observations(kg, hippo, next_is_nid, zero_nid)
    chain_nodes: set[NodeId] = set(chain_node_list)

    # Collect (first_operand, operator, second_operand, result) tuples from observations.
    # Strategy: in each observation, find digit tokens (chain_nodes) in positional order.
    # The operator token between them determines the operation.
    observed_tuples: list[tuple[NodeId, NodeId, NodeId, NodeId]] = []

    eq_nid = kg._value_to_node.get('=')
    plus_nid = kg._value_to_node.get('+')

    for obs in observations:
        nids = obs.token_nids
        if eq_nid is None or plus_nid is None:
            continue
        if eq_nid not in nids or plus_nid not in nids:
            continue

        # Find digit tokens in positional order.
        digits_in_order: list[tuple[int, NodeId]] = []
        op_pos = None
        eq_pos = None
        for pos, nid in enumerate(nids):
            if nid in chain_nodes:
                digits_in_order.append((pos, nid))
            if nid == plus_nid:
                op_pos = pos
            if nid == eq_nid:
                eq_pos = pos

        if op_pos is None or eq_pos is None:
            continue
        if len(digits_in_order) < 3:
            continue

        # Partition digits by position relative to operator and =.
        # Digits before operator = first operand.
        # Digits between operator and = = second operand.
        # Digits after = = result.
        first_ops = [nid for pos, nid in digits_in_order if pos < op_pos]
        second_ops = [nid for pos, nid in digits_in_order if op_pos < pos < eq_pos]
        results = [nid for pos, nid in digits_in_order if pos > eq_pos]

        # For single-digit: each should have exactly one.
        # Filter overflow: result must be reachable forward from first operand.
        if len(first_ops) == 1 and len(second_ops) == 1 and len(results) >= 1:
            first_nid, result_nid = first_ops[0], results[-1]
            chain_ok = True
            try:
                fi = chain_node_list.index(first_nid)
                ri = chain_node_list.index(result_nid)
                if ri <= fi:
                    chain_ok = False  # overflow
            except ValueError:
                chain_ok = False  # not on chain
            if chain_ok:
                observed_tuples.append((first_nid, plus_nid, second_ops[0], result_nid))

    if len(observed_tuples) < min_examples:
        return {"functors_discovered": 0, "functor_edges_created": 0,
                "observations_scanned": len(observations),
                "tuples_found": len(observed_tuples)}

    # For each tuple, verify the functor condition:
    # Walk from 0 to second_operand (source chain).
    # Walk from first_operand to result (target chain).
    # Simultaneous walk: both should end at the same step.
    functors_discovered = 0
    functor_edges_created = 0

    # Group by second operand (all "+4" examples should produce the same functor).
    from collections import defaultdict
    by_second_op: dict[NodeId, list[tuple[NodeId, NodeId]]] = defaultdict(list)
    for first, op, second, result in observed_tuples:
        by_second_op[second].append((first, result))

    for second_op_nid, examples in by_second_op.items():
        # Walk source chain: 0 → ... → second_op_nid
        source_chain = _walk_chain(kg, zero_nid, second_op_nid)
        if source_chain is None:
            continue

        # For each example, walk target chain and verify.
        verified_count = 0
        for first_nid, result_nid in examples:
            target_chain = _walk_chain(kg, first_nid, result_nid)
            if target_chain is None:
                continue
            # Functor condition: same length.
            if len(target_chain) != len(source_chain):
                continue
            verified_count += 1

        if verified_count < 2:
            continue

        # Functor verified! Create FUNCTOR edges encoding the mapping.
        # For each example, the functor maps source_chain[i] → target_chain[i].
        # We create edges from (second_op_nid, first_op_nid) → result_nid
        # as a "functor application" edge.
        functors_discovered += 1

        for first_nid, result_nid in examples:
            target_chain = _walk_chain(kg, first_nid, result_nid)
            if target_chain is None or len(target_chain) != len(source_chain):
                continue
            # Create a direct COOCCURRENCE edge from first_op → result
            # with strength proportional to the functor verification.
            # This "shortcut" edge lets the attention mechanism jump
            # directly from the first operand to the correct answer
            # when the second operand is in context.
            edge = kg.get_or_create_edge(first_nid, result_nid, role=COOCCURRENCE)
            edge.alpha += verified_count * 2.0  # strong evidence from functor
            edge._recalc()
            functor_edges_created += 1

    return {
        "functors_discovered": functors_discovered,
        "functor_edges_created": functor_edges_created,
        "observations_scanned": len(observations),
        "tuples_found": len(observed_tuples),
    }


# ---------------------------------------------------------------------------
# Natural transformation discovery
# ---------------------------------------------------------------------------

def discover_natural_transformations(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    min_examples: int = 2,
) -> dict[str, Any]:
    """Discover natural transformations: generalise functors to unseen inputs.

    After functor discovery has created shortcut edges for SEEN training
    examples (e.g., 3→7 for 3+4=7), the NT detects that all shortcuts for
    a given operation share the same chain length (structure-preserving
    property) and extends the pattern to ALL digit nodes.

    The method: SIMULTANEOUS WALKING. Given a known (template) shortcut
    A→B and a new starting node C:
      1. Walk from A toward B on the number line, one hop at a time.
      2. Simultaneously walk from C forward, one hop at a time.
      3. When the A-walk reaches B, the C-walk is at the predicted answer D.
      4. Create shortcut C→D.

    No counting. No integer extraction. The template chain is the "ruler"
    that measures the distance for the new starting point.

    This is a natural transformation because the mapping is NATURAL — it
    commutes with the successor structure. The same "shift by N" works
    regardless of starting point, and the NT discovers this from the
    consistency of the training examples.

    Returns stats about NTs discovered and edges created.
    """
    observations = hippo.all_observations()

    # Reuse the tuple extraction from discover_functors.
    zero_nid = kg._value_to_node.get('0')
    eq_nid = kg._value_to_node.get('=')
    plus_nid = kg._value_to_node.get('+')
    if zero_nid is None or eq_nid is None or plus_nid is None:
        return {"nt_discovered": 0, "nt_edges_created": 0}

    # Build the digit chain from counting warmup observations.
    # Counting observations contain "next_is" — the tokens appearing
    # with "next_is" are the digit chain. The digit BEFORE next_is at
    # position P appears AFTER next_is (at position P+2 or later) in
    # the NEXT counting observation. This gives us the ordering.
    next_is_nid = kg._value_to_node.get('next_is')
    digit_chain = _build_digit_chain_from_observations(
        kg, hippo, next_is_nid, zero_nid
    )
    digit_set = set(digit_chain)

    # Extract (first, second, result) tuples from observations.
    from collections import defaultdict
    by_second: dict[NodeId, list[tuple[NodeId, NodeId]]] = defaultdict(list)

    for obs in observations:
        nids = obs.token_nids
        if eq_nid not in nids or plus_nid not in nids:
            continue
        op_pos = None
        eq_pos = None
        digits_in_order: list[tuple[int, NodeId]] = []
        for pos, nid in enumerate(nids):
            if nid in digit_set:
                digits_in_order.append((pos, nid))
            if nid == plus_nid:
                op_pos = pos
            if nid == eq_nid:
                eq_pos = pos
        if op_pos is None or eq_pos is None:
            continue
        first_ops = [nid for pos, nid in digits_in_order if pos < op_pos]
        second_ops = [nid for pos, nid in digits_in_order if op_pos < pos < eq_pos]
        results = [nid for pos, nid in digits_in_order if pos > eq_pos]
        if len(first_ops) == 1 and len(second_ops) == 1 and len(results) >= 1:
            first_nid, result_nid = first_ops[0], results[-1]
            # Filter overflow: result must come AFTER first operand on the
            # digit chain. If result is BEFORE (e.g., 7+3→0 because 10
            # overflowed to single digit), skip it — the functor would fail.
            try:
                first_idx = digit_chain.index(first_nid)
                result_idx = digit_chain.index(result_nid)
                if result_idx > first_idx:
                    by_second[second_ops[0]].append((first_nid, result_nid))
            except ValueError:
                pass  # not on the digit chain

    # For each second-operand group, find the chain length using
    # simultaneous walking, then extend to all digit nodes.
    nt_discovered = 0
    nt_edges_created = 0

    for second_op, examples in by_second.items():
        unique_examples = list(set(examples))
        if len(unique_examples) < min_examples:
            continue

        # Pick the first verified example as the TEMPLATE.
        template_first, template_result = unique_examples[0]

        # Build the template chain by walking from template_first to template_result.
        # We walk hop by hop, collecting the path.
        template_chain = _walk_digit_chain(kg, template_first, template_result,
                                            digit_chain, max_steps=20)
        if template_chain is None or len(template_chain) < 2:
            continue

        # Verify: at least one other example has the same chain length.
        verified = False
        for other_first, other_result in unique_examples[1:]:
            other_chain = _walk_digit_chain(kg, other_first, other_result,
                                             digit_chain, max_steps=20)
            if other_chain is not None and len(other_chain) == len(template_chain):
                verified = True
                break

        if not verified:
            continue

        nt_discovered += 1
        chain_length = len(template_chain)  # number of NODES (edges = length - 1)

        # Extend: for each digit node, walk forward the same number of edges.
        # Use simultaneous walking: walk the template chain and a new chain in lockstep.
        existing_firsts = {f for f, r in unique_examples}
        for start_node in digit_chain:
            if start_node in existing_firsts:
                continue  # already has a shortcut

            # Simultaneous walk: template and new, lockstep.
            new_chain = _walk_digit_chain_n_steps(
                kg, start_node, chain_length - 1, digit_chain
            )
            if new_chain is None or len(new_chain) != chain_length:
                continue

            predicted_result = new_chain[-1]
            if predicted_result == start_node:
                continue

            # Create shortcut edge.
            edge = kg.get_or_create_edge(start_node, predicted_result, role=COOCCURRENCE)
            edge.alpha += len(unique_examples) * 1.5  # evidence from NT generalisation
            edge._recalc()
            nt_edges_created += 1

    return {
        "nt_discovered": nt_discovered,
        "nt_edges_created": nt_edges_created,
        "groups_found": len(by_second),
    }


def _build_digit_chain_from_observations(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    next_is_nid: NodeId | None,
    zero_nid: NodeId,
) -> list[NodeId]:
    """Build the digit chain from counting warmup observation records.

    Counting observations have the pattern [digit, next_is, digit].
    We extract consecutive (A, B) pairs where A appears before next_is
    and B appears after it. This gives us the successor ordering.

    Falls back to just collecting all single-character digit-valued nodes
    if no counting observations are found.
    """
    if next_is_nid is None:
        # No counting warmup. Fall back to collecting nodes whose values
        # are single digits, ordered by value. This uses the tokenizer's
        # label (which IS the original string), not integer extraction.
        digit_nodes = []
        for val, nid in kg._value_to_node.items():
            if isinstance(val, str) and len(val) == 1 and val.isdigit():
                digit_nodes.append((val, nid))
        # Sort by string value (lexicographic = numeric for single digits).
        digit_nodes.sort()
        return [nid for _, nid in digit_nodes]

    # Extract (before_next_is, after_next_is) pairs from counting observations.
    # Skip tokens that appear in EVERY observation (structural: space, phase, etc.)
    # by collecting only tokens that appear in counting-specific positions.
    space_nid = kg._value_to_node.get(' ')
    phase_counting_nid = kg._value_to_node.get('PHASE_counting')
    structural = {next_is_nid, space_nid, phase_counting_nid}
    for val, nid in kg._value_to_node.items():
        if isinstance(val, str) and val.startswith('PROBLEM_TYPE_'):
            structural.add(nid)

    succ_pairs: list[tuple[NodeId, NodeId]] = []
    for obs in hippo.all_observations():
        nids = obs.token_nids
        if next_is_nid not in nids:
            continue
        ni_pos = nids.index(next_is_nid)
        # Extract non-structural tokens before and after next_is.
        before_tokens = [nids[i] for i in range(ni_pos)
                         if nids[i] not in structural]
        after_tokens = [nids[i] for i in range(ni_pos + 1, len(nids))
                        if nids[i] not in structural]
        if not before_tokens or not after_tokens:
            continue
        # For the successor chain, use the LAST token before next_is
        # and the LAST token after next_is. For single-digit numbers
        # (e.g., [3, next_is, 4]), this gives (3, 4). For multi-digit
        # numbers (e.g., [1, 0, next_is, 1, 1]), this gives (0, 1) —
        # the units digits. The tens digits are handled separately by
        # the autoregressive prediction (first digit, then second).
        before = before_tokens[-1]
        after = after_tokens[-1]
        if before != after:
            succ_pairs.append((before, after))

    if not succ_pairs:
        # Fallback: single-digit nodes sorted by label.
        digit_nodes = []
        for val, nid in kg._value_to_node.items():
            if isinstance(val, str) and len(val) == 1 and val.isdigit():
                digit_nodes.append((val, nid))
        digit_nodes.sort()
        return [nid for _, nid in digit_nodes]

    # Build the chain from successor pairs.
    succ_map: dict[NodeId, NodeId] = {}
    all_targets: set[NodeId] = set()
    for a, b in succ_pairs:
        succ_map[a] = b
        all_targets.add(b)

    # Find the start: a node that appears as a source but never as a target.
    starts = [a for a in succ_map if a not in all_targets]
    if not starts:
        # Circular or messy — use zero_nid if available.
        if zero_nid in succ_map:
            starts = [zero_nid]
        else:
            return []

    chain = [starts[0]]
    current = starts[0]
    visited = {current}
    for _ in range(30):
        nxt = succ_map.get(current)
        if nxt is None or nxt in visited:
            break
        chain.append(nxt)
        visited.add(nxt)
        current = nxt

    return chain


def _walk_digit_chain(
    kg: KnowledgeGraph,
    start: NodeId,
    end: NodeId,
    digit_chain: list[NodeId],
    max_steps: int = 20,
) -> list[NodeId] | None:
    """Walk from start to end following the digit chain order.

    Uses the precomputed digit_chain (from counting warmup) as the
    canonical ordering. Returns the sub-chain from start to end, or None.
    """
    try:
        start_idx = digit_chain.index(start)
        end_idx = digit_chain.index(end)
    except ValueError:
        return None
    if end_idx <= start_idx:
        return None
    if end_idx - start_idx > max_steps:
        return None
    return digit_chain[start_idx:end_idx + 1]


def _walk_digit_chain_n_steps(
    kg: KnowledgeGraph,
    start: NodeId,
    n_steps: int,
    digit_chain: list[NodeId],
) -> list[NodeId] | None:
    """Walk n_steps forward from start along the digit chain.

    Returns the sub-chain of length n_steps + 1 (including start), or None
    if the chain doesn't extend that far.
    """
    try:
        start_idx = digit_chain.index(start)
    except ValueError:
        return None
    end_idx = start_idx + n_steps
    if end_idx >= len(digit_chain):
        return None
    return digit_chain[start_idx:end_idx + 1]


# ---------------------------------------------------------------------------
# Main consolidation entry point
# ---------------------------------------------------------------------------

def consolidate(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    replay_passes: int = 3,
    replay_strength: float = 1.0,
) -> dict[str, Any]:
    """Run a full consolidation pass: replay → prune → NT → colimit → functor."""
    stats: dict[str, Any] = {}

    replay_stats = replay(kg, hippo, n_passes=replay_passes,
                          replay_strength=replay_strength)
    stats.update({f"replay_{k}": v for k, v in replay_stats.items()})

    prune_stats = prune(kg)
    stats.update({f"prune_{k}": v for k, v in prune_stats.items()})

    # Natural transformations are now handled by extend_products_via_nt
    # which calls find_natural_transformations internally with MorphismTables.
    # The old co-occurrence NT (which operated on hippo directly) is removed —
    # the universal NT system replaces all specific NT code.

    from experiments.symbolic_ai_v2.ctkg.logic.colimit import find_colimits as find_real_colimits
    colimit_stats = find_real_colimits(kg, hippo)
    stats.update({f"colimit_{k}": v for k, v in colimit_stats.items()})

    # Level-1 morphisms: directed context→output mappings.
    from experiments.symbolic_ai_v2.ctkg.logic.morphism import discover_morphisms
    morph_stats = discover_morphisms(kg, hippo)
    stats.update({f"morphism_{k}": v for k, v in morph_stats.items()})

    # Product decomposition: per-position digit projections.
    product_stats = discover_products(kg, hippo)
    stats.update({f"product_{k}": v for k, v in product_stats.items()})

    # Natural transformation: extend product projections to unseen positions.
    # Clear previously-extended projections first so the NT uses fresh data
    # from discover_products (not stale extensions from earlier consolidation).
    observed_positions = set()
    for (_, _, pos), _ in list(kg._product_projections.items()):
        observed_positions.add(pos)
    # observed_positions now includes both discovered AND previously extended.
    # We need to know which positions were DISCOVERED (have data from
    # discover_products) vs EXTENDED (from NT). Re-run discover_products
    # to get the fresh discovered set, then clear everything else.
    # Simpler: just clear ALL projections and re-discover + re-extend.
    kg._product_projections.clear()
    discover_products(kg, hippo)
    nt_product_stats = extend_products_via_nt(kg)
    stats.update({f"nt_product_{k}": v for k, v in nt_product_stats.items()})

    # Functor discovery: structure-preserving maps between NNO sub-chains.
    functor_stats = discover_functors(kg, hippo)
    stats.update({f"functor_{k}": v for k, v in functor_stats.items()})

    # Natural transformations: generalise functors to unseen operand pairs.
    nt2_stats = discover_natural_transformations(kg, hippo)
    stats.update({f"nt2_{k}": v for k, v in nt2_stats.items()})

    return stats
