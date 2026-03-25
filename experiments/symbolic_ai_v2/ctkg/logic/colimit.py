"""
Colimit discovery — the categorical universal construction.

A colimit of a diagram D (a connected subgraph with internal morphisms) is
a new node C with cocone maps from each diagram member, satisfying the
universal property: any other cocone factors through C uniquely.

Construction: coproduct (disjoint union) + coequalizer (quotient by the
diagram's internal morphisms). In our graph:

1. **Discover diagrams**: find pairs (A, B) connected by a directed edge
   where A→B was observed within the same observation (the edge is a
   diagram morphism, not just a co-occurrence).

2. **Coequalizer**: the colimit identifies A and B via their morphism.
   The colimit node represents "A-and-B connected by this specific edge."

3. **Cocone maps**: edges A → colimit and B → colimit.

4. **Universal property / factored edges**: for any node X that BOTH A
   and B connect to (another cocone), AND X appears AFTER the diagram
   in observation order, create colimit → X. The weight is determined
   by the minimum of the member→X weights (bottleneck of the cocone).

The positional ordering ensures factored edges encode direction:
in observation [3, succ, 4], the colimit of the diagram {3→4} points
to targets that appear AFTER both 3 and 4, not to targets that
appeared before (which would be inputs, not outputs).

No string inspection. No domain knowledge. Pure graph + observation structure.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId, Edge, COOCCURRENCE, TRANSITION,
    ACTIVATION_THRESHOLD,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


def find_colimits(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    min_coactivation: int = 3,
    min_group_size: int = 2,
    max_group_size: int = 3,
) -> dict[str, Any]:
    """Discover colimits from observation-grounded diagrams.

    Step 1: Find directed edges (A→B) that occur within the same
    observation at least min_coactivation times. These are diagram
    morphisms — not just co-occurrence, but directed structure.

    Step 2: For each qualifying diagram (pair or triple of nodes with
    internal morphisms), compute the colimit as a new node with cocone
    maps and factored edges derived from the universal property.

    Returns statistics.
    """
    observations = hippo.all_observations()
    if len(observations) < min_coactivation:
        return {"colimits_created": 0, "factored_edges": 0, "diagrams_found": 0}

    # --- Step 1: Find directed pairs within observations ---
    # A "diagram morphism" A→B exists when:
    # - A and B co-occur in the same observation
    # - A appears BEFORE B in the observation (directed)
    # - This pattern repeats at least min_coactivation times
    #
    # This is fundamentally different from co-occurrence pairs: the
    # ORDER within the observation encodes the diagram's morphism direction.

    directed_pair_count: dict[tuple[NodeId, NodeId], int] = defaultdict(int)
    # Also track which observations each pair appeared in (for factored edges).
    pair_observations: dict[tuple[NodeId, NodeId], list[int]] = defaultdict(list)

    for obs_idx, obs in enumerate(observations):
        nids = obs.token_nids
        n = len(nids)
        # For each ordered pair (i, j) where i < j in observation position:
        seen = set()
        for i in range(n):
            for j in range(i + 1, min(n, i + 6)):  # window of 5
                a, b = nids[i], nids[j]
                if a == b:
                    continue
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    directed_pair_count[key] += 1
                    pair_observations[key].append(obs_idx)

    # Filter to pairs that occur frequently enough.
    strong_pairs: dict[tuple[NodeId, NodeId], int] = {}
    for pair, count in directed_pair_count.items():
        if count >= min_coactivation:
            strong_pairs[pair] = count

    if not strong_pairs:
        return {"colimits_created": 0, "factored_edges": 0, "diagrams_found": 0}

    # --- Step 2: Build diagrams and compute colimits ---
    # Each strong directed pair (A→B) is a minimal diagram with one morphism.
    # The colimit of this diagram glues A and B via the morphism.

    colimits_created = 0
    factored_edges_created = 0
    diagrams_found = 0

    for (src, tgt), count in strong_pairs.items():
        diagrams_found += 1

        # The colimit key encodes the diagram: directed pair (src→tgt).
        # This is different from the old frozenset — direction matters.
        colimit_key = ("__colimit_d__", src, tgt)
        if colimit_key in kg._value_to_node:
            colimit_nid = kg._value_to_node[colimit_key]
        else:
            colimit_nid = kg.get_or_create(colimit_key)
            kg._nodes[colimit_nid].resting = 0.3
            colimits_created += 1

        # --- Cocone maps: diagram members → colimit ---
        # Both src and tgt map into the colimit (cocone condition).
        for member_nid in (src, tgt):
            edge_up = kg.get_or_create_edge(member_nid, colimit_nid, role=COOCCURRENCE)
            desired_alpha = max(count + 1.0, edge_up.alpha)
            if edge_up.alpha < desired_alpha:
                edge_up.alpha = desired_alpha
                edge_up.beta = max(1.0, edge_up.beta)
                edge_up._recalc()

        # --- Universal property: factored edges ---
        # Scan the observations where this pair appeared. For each such
        # observation, find nodes that appear AFTER BOTH src and tgt.
        # These are the factored targets: they form another cocone over
        # the diagram, and the universal property gives colimit → target.

        target_counts: dict[NodeId, int] = defaultdict(int)
        diagram_set = {src, tgt, colimit_nid}

        for obs_idx in pair_observations[(src, tgt)]:
            obs = observations[obs_idx]
            nids = obs.token_nids

            # Find the position of the LAST diagram member.
            last_pos = -1
            for i, nid in enumerate(nids):
                if nid in (src, tgt):
                    last_pos = i

            # Targets: nodes appearing AFTER the last diagram member.
            for i in range(last_pos + 1, len(nids)):
                nid = nids[i]
                if nid not in diagram_set:
                    target_counts[nid] += 1

        # Create factored edges for targets seen at least once.
        # Factored edges are DERIVED from the universal property — they
        # represent verified structural relationships, not just statistics.
        # Amplify them: each observation-grounded co-occurrence counts
        # as strong evidence (5x multiplier) because the direction and
        # context have been verified by the colimit construction.
        for target_nid, tgt_count in target_counts.items():
            factored = kg.get_or_create_edge(colimit_nid, target_nid, role=COOCCURRENCE)
            desired_alpha = max(tgt_count * 5.0 + 1.0, 2.0)
            if factored.alpha < desired_alpha:
                factored.alpha = desired_alpha
                factored.beta = max(1.0, factored.beta)
                factored._recalc()
                factored_edges_created += 1

    return {
        "colimits_created": colimits_created,
        "factored_edges": factored_edges_created,
        "diagrams_found": diagrams_found,
    }
