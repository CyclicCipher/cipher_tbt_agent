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
# Chunking: replace token sequences with concept nodes in observations
# ---------------------------------------------------------------------------

def _chunk_observations(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    min_occurrences: int = 3,
    min_length: int = 2,
    max_length: int = 20,
) -> int:
    """Replace invariant contiguous subsequences with chunk nodes.

    Finds contiguous token subsequences that appear IDENTICALLY in
    multiple observations (at the same sequential position). These
    are invariant frame fragments — "The current date is", "the date
    will be", etc. Replaces each occurrence with a single chunk node.

    This preserves position: only CONTIGUOUS identical subsequences
    are chunked. A character that appears in both frame and content
    positions won't be chunked in the content position (because the
    surrounding context differs).

    Returns the number of subsequences chunked.
    """
    observations = hippo.all_observations()
    if len(observations) < min_occurrences:
        return 0

    # Collect all contiguous subsequences and count how many observations
    # they appear in at the same position.
    from collections import Counter
    subseq_count: Counter = Counter()

    for obs in observations:
        nids = obs.token_nids
        n = len(nids)
        for length in range(min_length, min(max_length, n) + 1):
            for start in range(n - length + 1):
                subseq = tuple(nids[start:start + length])
                subseq_count[(subseq, start)] += 1

    # Find subsequences that appear in many observations at the same position.
    # These are invariant frame fragments.
    invariants: list[tuple[tuple[NodeId, ...], int, int]] = []  # (subseq, start, count)
    for (subseq, start), count in subseq_count.items():
        if count >= min_occurrences and len(subseq) >= min_length:
            invariants.append((subseq, start, count))

    # Sort by length * count (prefer long, frequent fragments).
    invariants.sort(key=lambda x: -len(x[0]) * x[2])

    if not invariants:
        return 0

    # Greedily chunk: for each invariant, create a chunk node and
    # replace the subsequence in all matching observations.
    chunked_count = 0
    # Track which positions in each observation have been chunked.
    obs_chunked: dict[int, set[int]] = {}  # obs_index → set of chunked positions

    for subseq, start, count in invariants:
        length = len(subseq)

        # Create or reuse chunk node.
        chunk_key = ("__chunk__", subseq, start)
        chunk_nid = kg.get_or_create(chunk_key)
        node = kg.node(chunk_nid)
        if node is not None:
            node.resting = max(node.resting, 0.3)

        # Replace in all matching observations.
        for obs_idx, obs in enumerate(observations):
            nids = obs.token_nids
            # Check if this subsequence matches at this position.
            if start + length > len(nids):
                continue
            if tuple(nids[start:start + length]) != subseq:
                continue
            # Check no overlap with already-chunked positions.
            chunked_positions = obs_chunked.get(obs_idx, set())
            overlap = any(p in chunked_positions for p in range(start, start + length))
            if overlap:
                continue

            # Replace: put chunk_nid at start, remove the rest.
            new_nids = nids[:start] + [chunk_nid] + nids[start + length:]
            obs.token_nids = new_nids

            # Mark positions as chunked. Adjust for the shortened list.
            if obs_idx not in obs_chunked:
                obs_chunked[obs_idx] = set()
            obs_chunked[obs_idx].add(start)
            # Shift existing chunked positions after this point.
            shift = length - 1
            if shift > 0:
                obs_chunked[obs_idx] = {
                    p - shift if p > start else p
                    for p in obs_chunked[obs_idx]
                }

            chunked_count += 1

        # Only chunk a limited number of invariants to avoid
        # over-chunking in one pass.
        if chunked_count > len(observations) * 2:
            break

    return chunked_count


# ---------------------------------------------------------------------------
# Main consolidation entry point
# ---------------------------------------------------------------------------

def _single_pass(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    replay_passes: int = 1,
    replay_strength: float = 0.1,
    since_index: int = 0,
) -> dict[str, Any]:
    """One consolidation pass: replay → prune → PMI/IDF → FCA → algebra →
    initial algebra → sheaf → colimit → morphism.

    Returns stats dict including structure_created count (new nodes/edges
    materialized by FCA, algebra, initial algebra, colimit, morphism).
    """
    stats: dict[str, Any] = {}
    nodes_before = kg.node_count()
    edges_before = kg.edge_count()

    replay_stats = replay(kg, hippo, n_passes=replay_passes,
                          replay_strength=replay_strength,
                          since_index=since_index)
    stats.update({f"replay_{k}": v for k, v in replay_stats.items()})

    prune_stats = prune(kg)
    stats.update({f"prune_{k}": v for k, v in prune_stats.items()})

    # PMI + IDF cache.
    from experiments.symbolic_ai_v2.ctkg.logic.initial_algebra import _compute_pmi
    kg._pmi = _compute_pmi(hippo, kg, since_index=since_index)
    stats["pmi_pairs"] = len(kg._pmi)

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

    # FCA.
    from experiments.symbolic_ai_v2.ctkg.logic.fca import discover_fca_structure
    fca_stats = discover_fca_structure(kg, hippo, since_index=since_index)
    stats.update({f"fca_{k}": v for k, v in fca_stats.items()})

    # Cortical column: context nodes (layer 1) and displacement nodes (layer 2).
    from experiments.symbolic_ai_v2.ctkg.logic.cortical_column import discover_cortical_structure
    cc_stats = discover_cortical_structure(kg, hippo, since_index=since_index)
    stats.update({f"cc_{k}": v for k, v in cc_stats.items()})

    # Algebraic skeleton.
    from experiments.symbolic_ai_v2.ctkg.logic.algebra import discover_algebraic_structure
    alg_stats = discover_algebraic_structure(kg, hippo, since_index=since_index)
    stats.update({f"alg_{k}": v for k, v in alg_stats.items()})

    # Initial algebra.
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

    # Track how much new structure was created.
    nodes_after = kg.node_count()
    edges_after = kg.edge_count()
    stats["structure_new_nodes"] = nodes_after - nodes_before
    stats["structure_new_edges"] = edges_after - edges_before

    return stats


def consolidate(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    replay_passes: int = 1,
    replay_strength: float = 0.1,
    since_index: int = 0,
    max_depth: int = 5,
) -> dict[str, Any]:
    """Recursive consolidation — discovers patterns in patterns.

    Runs _single_pass. If new structure was created (new nodes/edges
    materialized by FCA, algebra, colimits, etc.), runs again — because
    the new structure is now data for the next pass to find patterns in.

    Repeats until no new structure is found (fixpoint) or max_depth
    is reached.

    This is the Broca's area merge operation: each pass discovers
    patterns at one level; the next pass discovers patterns in THOSE
    patterns. Level 0 = raw token patterns. Level 1 = patterns in
    FCA concepts. Level 2 = meta-patterns (context-dependent periods,
    conditional rules). Etc.

    since_index: only process observations from this index onward.
    max_depth: maximum number of recursive passes.
    """
    all_stats: dict[str, Any] = {}
    depth = 0

    while depth < max_depth:
        pass_stats = _single_pass(
            kg, hippo,
            replay_passes=replay_passes,
            replay_strength=replay_strength,
            since_index=since_index,
        )

        new_nodes = pass_stats.get("structure_new_nodes", 0)
        new_edges = pass_stats.get("structure_new_edges", 0)

        # Store prefixed stats for tracing depth.
        for k, v in pass_stats.items():
            all_stats[f"d{depth}_{k}"] = v
        # Unprefixed: depth 0 sets the base, deeper passes ACCUMULATE.
        if depth == 0:
            for k, v in pass_stats.items():
                all_stats[k] = v
        else:
            for k, v in pass_stats.items():
                if isinstance(v, (int, float)) and k in all_stats:
                    all_stats[k] = all_stats[k] + v
                else:
                    all_stats[k] = v

        depth += 1

        # Stop if no new structure was created (fixpoint).
        if new_nodes == 0 and new_edges <= 0:
            break

        # Inject discovered structure into observations for the next pass.
        # For each observation, replace sequences of tokens that match an
        # FCA concept's extent with the concept node. This creates
        # higher-level observations that the next pass can find patterns in.
        _chunk_observations(kg, hippo)

        # After first pass, don't re-replay or re-prune — just run
        # the structure discovery phases on the enriched graph.
        replay_passes = 0

    all_stats["consolidation_depth"] = depth
    return all_stats
