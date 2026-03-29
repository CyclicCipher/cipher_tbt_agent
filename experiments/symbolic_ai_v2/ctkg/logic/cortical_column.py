"""
Cortical column — context and displacement node discovery.

The cortical column has three layers:
- Identity (layer 0): one node per token. Already exists in the graph.
- Context (layer 1): one node per recurring activation pattern.
  Fires when a specific set of identity nodes are co-active.
  Like a place cell: encodes WHERE you are in the sequence.
- Displacement (layer 2): one node per recurring context transition.
  Fires when the active context changes in a specific way.
  Like a grid cell: encodes HOW you moved through the space.

Discovery during consolidation:
1. Scan hippocampal snapshots for recurring activation patterns.
2. Each pattern that recurs becomes a context node with edges
   from its constituent identity nodes.
3. Scan consecutive snapshot pairs for recurring context transitions.
4. Each transition that recurs becomes a displacement node with
   edges from the source context to the displacement and from
   the displacement to the target context.

No domain knowledge. Discovers positional structure from activation
pattern recurrence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId, Node, Edge,
    COOCCURRENCE, TRANSITION,
    IDENTITY, CONTEXT, DISPLACEMENT,
    ACTIVATION_THRESHOLD,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


# ---------------------------------------------------------------------------
# Context node discovery
# ---------------------------------------------------------------------------

def discover_context_nodes(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    min_occurrences: int = 3,
    min_pattern_size: int = 2,
    max_pattern_size: int = 8,
    max_contexts: int = 200,
) -> dict[str, Any]:
    """Discover context nodes from recurring activation patterns.

    Scans activation snapshots. For each snapshot, extracts the set
    of active IDENTITY nodes (above threshold). Patterns that recur
    across multiple snapshots become context nodes.

    A context node has:
    - Incoming edges from each identity node in the pattern (COOCCURRENCE)
    - The context node activates when ALL its constituent identity nodes
      are active (conjunction detection)

    Returns stats.
    """
    snapshots = hippo.all_snapshots()
    recent = snapshots[max(0, since_index):]
    if len(recent) < min_occurrences:
        return {"context_nodes_created": 0, "patterns_found": 0}

    # Extract identity-node activation patterns from each snapshot.
    # Use only the TOP-K most active identity nodes (sparse coding).
    # This filters out decay residuals and makes patterns more repeatable.
    TOP_K = max_pattern_size  # only the strongest activations

    patterns: list[frozenset[NodeId]] = []
    for snap in recent:
        # Get identity nodes sorted by activation.
        id_activations = [
            (nid, lvl) for nid, lvl in snap.activations.items()
            if lvl >= ACTIVATION_THRESHOLD
            and nid in kg._nodes
            and kg._nodes[nid].layer == IDENTITY
        ]
        id_activations.sort(key=lambda x: -x[1])

        # Take only the top-K.
        top = id_activations[:TOP_K]
        if len(top) < min_pattern_size:
            continue

        # Use a high threshold: only nodes with activation >= 0.5
        # (recently observed, not just decay residual).
        strong = frozenset(nid for nid, lvl in top if lvl >= 0.5)
        if min_pattern_size <= len(strong) <= max_pattern_size:
            patterns.append(strong)

    if not patterns:
        return {"context_nodes_created": 0, "patterns_found": 0}

    # Count pattern frequencies.
    pattern_counts = Counter(patterns)

    # Create context nodes for recurring patterns.
    contexts_created = 0
    for pattern, count in pattern_counts.most_common(max_contexts):
        if count < min_occurrences:
            break

        # Create context node (idempotent).
        ctx_key = ("__context__", pattern)
        if ctx_key in kg._value_to_node:
            continue

        ctx_nid = kg.get_or_create(ctx_key, layer=CONTEXT)
        node = kg.node(ctx_nid)
        if node is not None:
            node.resting = min(0.5, 0.1 * count)

        # Edges from identity nodes to context node.
        for id_nid in pattern:
            edge = kg.get_or_create_edge(id_nid, ctx_nid, role=COOCCURRENCE)
            edge.strengthen(min(0.1 * count, 0.8))

        contexts_created += 1

    return {
        "context_nodes_created": contexts_created,
        "patterns_found": len(pattern_counts),
    }


# ---------------------------------------------------------------------------
# Displacement node discovery
# ---------------------------------------------------------------------------

def discover_displacement_nodes(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    min_occurrences: int = 3,
    max_displacements: int = 200,
) -> dict[str, Any]:
    """Discover displacement nodes from recurring context transitions.

    Scans consecutive activation snapshots. For each pair (t, t+1),
    extracts the active CONTEXT nodes at each step. Transitions that
    recur become displacement nodes.

    A displacement node has:
    - Incoming edge from the source context (TRANSITION)
    - Outgoing edge to the target context (TRANSITION)
    - The displacement encodes: "from this context, this is what changed"

    Returns stats.
    """
    snapshots = hippo.all_snapshots()
    recent = snapshots[max(0, since_index):]
    if len(recent) < min_occurrences + 1:
        return {"displacement_nodes_created": 0, "transitions_found": 0}

    # Get context nodes in the graph.
    context_nids = set(kg.nodes_by_layer(CONTEXT))
    if not context_nids:
        return {"displacement_nodes_created": 0, "transitions_found": 0}

    # Extract active context nodes per snapshot.
    def _active_contexts(snap) -> frozenset[NodeId]:
        """Compute which context nodes WOULD be active given this snapshot.

        A context node fires when ALL its identity constituents are active.
        """
        active_ids = set(
            nid for nid, lvl in snap.activations.items()
            if lvl >= ACTIVATION_THRESHOLD
            and nid in kg._nodes
            and kg._nodes[nid].layer == IDENTITY
        )
        active_ctx: set[NodeId] = set()
        for ctx_nid in context_nids:
            val = kg.value_for_node(ctx_nid)
            if not isinstance(val, tuple) or val[0] != "__context__":
                continue
            pattern = val[1]  # frozenset of identity NodeIds
            if pattern.issubset(active_ids):
                active_ctx.add(ctx_nid)
        return frozenset(active_ctx)

    # Compute context transitions.
    transition_counts: Counter = Counter()
    for i in range(len(recent) - 1):
        ctx_t = _active_contexts(recent[i])
        ctx_t1 = _active_contexts(recent[i + 1])
        if ctx_t and ctx_t1 and ctx_t != ctx_t1:
            transition_counts[(ctx_t, ctx_t1)] += 1

    # Create displacement nodes for recurring transitions.
    displacements_created = 0
    for (src_ctxs, tgt_ctxs), count in transition_counts.most_common(max_displacements):
        if count < min_occurrences:
            break

        disp_key = ("__displacement__", src_ctxs, tgt_ctxs)
        if disp_key in kg._value_to_node:
            continue

        disp_nid = kg.get_or_create(disp_key, layer=DISPLACEMENT)
        node = kg.node(disp_nid)
        if node is not None:
            node.resting = min(0.4, 0.1 * count)

        # Edges: source contexts → displacement (TRANSITION).
        for src_nid in src_ctxs:
            edge = kg.get_or_create_edge(src_nid, disp_nid, role=TRANSITION)
            edge.strengthen(min(0.1 * count, 0.8))

        # Edges: displacement → target contexts (TRANSITION).
        for tgt_nid in tgt_ctxs:
            edge = kg.get_or_create_edge(disp_nid, tgt_nid, role=TRANSITION)
            edge.strengthen(min(0.1 * count, 0.8))

        displacements_created += 1

    return {
        "displacement_nodes_created": displacements_created,
        "transitions_found": len(transition_counts),
    }


# ---------------------------------------------------------------------------
# Context-tagged sequence tracking
# ---------------------------------------------------------------------------

def _active_contexts_for_snapshot(
    kg: KnowledgeGraph,
    snap,
    context_nids: set[NodeId],
) -> set[NodeId]:
    """Compute which context nodes fire given a snapshot's activations."""
    active_ids = set(
        nid for nid, lvl in snap.activations.items()
        if lvl >= ACTIVATION_THRESHOLD
        and nid in kg._nodes
        and kg._nodes[nid].layer == IDENTITY
    )
    active_ctx: set[NodeId] = set()
    for ctx_nid in context_nids:
        val = kg.value_for_node(ctx_nid)
        if not isinstance(val, tuple) or val[0] != "__context__":
            continue
        pattern = val[1]
        if pattern.issubset(active_ids):
            active_ctx.add(ctx_nid)
    return active_ctx


def _strongest_identity(
    kg: KnowledgeGraph,
    snap,
    exclude: frozenset[NodeId] | None = None,
) -> NodeId | None:
    """Return the most strongly activated identity node in a snapshot.

    Optionally exclude a set of nodes (e.g., the context's own members).
    """
    best_nid = None
    best_act = 0.0
    for nid, lvl in snap.activations.items():
        if nid not in kg._nodes:
            continue
        if kg._nodes[nid].layer != IDENTITY:
            continue
        if exclude and nid in exclude:
            continue
        if lvl > best_act:
            best_act = lvl
            best_nid = nid
    return best_nid


def discover_positional_periods(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    min_observations: int = 10,
    observation_subset: list | None = None,
    group_label: str = "",
) -> dict[str, Any]:
    """Discover token periodicity at each position across observations.

    For each position p in the observation token sequences, extract the
    token at position p across all observations. If periodic, record it.

    observation_subset: if provided, use this list instead of hippo.
    group_label: label for this group (for materializing period nodes).

    Returns stats including discovered periods per position.
    """
    if observation_subset is not None:
        recent = observation_subset
    else:
        observations = hippo.all_observations()
        recent = observations[max(0, since_index):]
    if len(recent) < min_observations:
        return {"positions_checked": 0, "periods_found": 0, "period_details": {}}

    # Find max observation length.
    max_len = max(len(obs.token_nids) for obs in recent)

    # For each position, extract the token sequence across observations.
    periods_found = 0
    positions_checked = 0
    period_details: dict[int, int] = {}  # position → period

    for pos in range(max_len):
        # Collect token at this position from each observation.
        seq: list[NodeId] = []
        for obs in recent:
            if pos < len(obs.token_nids):
                seq.append(obs.token_nids[pos])

        if len(seq) < min_observations:
            continue
        positions_checked += 1

        # Check if this position has a constant token (frame).
        unique = set(seq)
        if len(unique) == 1:
            continue  # invariant position, no period to discover

        # Detect period.
        for p in range(2, min(100, len(seq) // 2) + 1):
            max_check = min(len(seq) - p, len(seq) // 2)
            if max_check < 5:
                continue
            matches = sum(1 for i in range(max_check) if seq[i] == seq[i + p])
            if matches / max_check >= 0.85:
                period_details[pos] = p
                periods_found += 1

                # Materialize as a period node.
                # Extract the cycle tokens.
                cycle_tokens: list[NodeId] = []
                for phase in range(p):
                    phase_counts: dict[NodeId, int] = {}
                    for i in range(phase, len(seq), p):
                        nid = seq[i]
                        phase_counts[nid] = phase_counts.get(nid, 0) + 1
                    if phase_counts:
                        best = max(phase_counts, key=phase_counts.get)
                        cycle_tokens.append(best)

                period_key = ("__period__", group_label, pos, p, tuple(cycle_tokens))
                if period_key not in kg._value_to_node:
                    period_nid = kg.get_or_create(period_key, layer=CONTEXT)
                    node = kg.node(period_nid)
                    if node is not None:
                        node.resting = 0.5

                    # Edges from period node to cycle tokens.
                    for tok_nid in cycle_tokens:
                        edge = kg.get_or_create_edge(
                            period_nid, tok_nid, role=COOCCURRENCE,
                        )
                        edge.strengthen(0.4)

                break  # smallest period

    return {
        "positions_checked": positions_checked,
        "periods_found": periods_found,
        "period_details": period_details,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def discover_grouped_periods(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    min_group_size: int = 20,
    min_observations: int = 10,
) -> dict[str, Any]:
    """Partition observations by categorical positions, then detect periods.

    For each position in the observations, check if the token at that
    position is CATEGORICAL (small number of unique values that partition
    the observations) vs NUMERICAL (many values, potentially periodic).

    Categorical positions partition the observations into groups.
    Within each group, run period detection on the numerical positions.

    This discovers: "among observations where position 46 is 's' (seconds),
    position 21 cycles with period 10."

    Returns stats.
    """
    observations = hippo.all_observations()
    recent = observations[max(0, since_index):]
    if len(recent) < min_group_size:
        return {"groups_found": 0, "total_periods": 0, "group_details": {}}

    max_len = max(len(obs.token_nids) for obs in recent)

    # Find categorical positions: positions where only a few unique tokens
    # appear across all observations (< 20 unique values AND < 50% of obs count).
    categorical_positions: list[tuple[int, dict[NodeId, list]]] = []

    for pos in range(max_len):
        token_groups: dict[NodeId, list] = {}
        for obs in recent:
            if pos < len(obs.token_nids):
                tok = obs.token_nids[pos]
                token_groups.setdefault(tok, []).append(obs)

        n_unique = len(token_groups)
        # Categorical: few unique values, each with enough observations.
        if 2 <= n_unique <= 15:
            big_groups = sum(1 for g in token_groups.values() if len(g) >= min_group_size)
            if big_groups >= 2:
                categorical_positions.append((pos, token_groups))

    if not categorical_positions:
        # No categorical partition found — run on all observations.
        period_stats = discover_positional_periods(
            kg, hippo, since_index=since_index,
            min_observations=min_observations,
            group_label="all",
        )
        return {
            "groups_found": 0,
            "total_periods": period_stats["periods_found"],
            "group_details": {"all": period_stats["period_details"]},
        }

    # Use the categorical position with the fewest unique values
    # (most discriminating partition).
    best_pos, best_groups = min(categorical_positions, key=lambda x: len(x[1]))

    total_periods = 0
    group_details: dict[str, dict] = {}

    for tok_nid, group_obs in best_groups.items():
        if len(group_obs) < min_group_size:
            continue
        label = kg.label_for_node(tok_nid)
        group_stats = discover_positional_periods(
            kg, hippo,
            min_observations=min_observations,
            observation_subset=group_obs,
            group_label=f"pos{best_pos}={label}",
        )
        total_periods += group_stats["periods_found"]
        if group_stats["period_details"]:
            group_details[f"pos{best_pos}={label}"] = group_stats["period_details"]

    return {
        "groups_found": len(best_groups),
        "partition_position": best_pos,
        "total_periods": total_periods,
        "group_details": group_details,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def discover_cortical_structure(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
) -> dict[str, Any]:
    """Discover context nodes, positional periods, and displacement nodes.

    Called during consolidation. Order:
    1. Context nodes (layer 1) from activation patterns
    2. Grouped positional period detection
    3. Displacement nodes (layer 2) from context transitions
    """
    ctx_stats = discover_context_nodes(kg, hippo, since_index=since_index)
    period_stats = discover_grouped_periods(kg, hippo, since_index=since_index)
    disp_stats = discover_displacement_nodes(kg, hippo, since_index=since_index)

    stats: dict[str, Any] = {}
    stats.update({f"ctx_{k}": v for k, v in ctx_stats.items()})
    stats.update({f"period_{k}": v for k, v in period_stats.items()})
    stats.update({f"disp_{k}": v for k, v in disp_stats.items()})
    return stats
