"""
Morphism discovery — level-1 directed structures.

A morphism is a directed mapping: input_pattern → output. Unlike colimits
(which are symmetric co-occurrence groups), morphisms have direction: some
nodes are inputs, one node is the output.

Discovery: scan consecutive observation pairs in the hippocampus.
- Observation t: the context (what was presented)
- Observation t+1: the action (what the agent did)

If a token appears in BOTH obs t and obs t+1, it's the output (the agent
reproduced what it saw). The remaining tokens in obs t are the input pattern.
When the same input pattern consistently maps to the same output across
multiple episodes, create a morphism node.

The morphism node M has:
- Input edges: context_node → M (for each context token)
- Output edge: M → output_node (the prediction)

During 2-hop context spread: context nodes activate M (hop 1), M activates
the output (hop 2). Direction is built into the edge structure.

No string inspection. No domain knowledge. Pure observation structure.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId, COOCCURRENCE, TRANSITION,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


def discover_morphisms(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    min_occurrences: int = 2,
    max_context_size: int = 6,
    since_index: int = 0,
) -> dict[str, Any]:
    """Discover level-1 morphisms from observation→action pairs.

    Scans consecutive observation records. When observation t contains
    tokens that overlap with observation t+1 (a single-token action),
    the overlapping token is the output and the non-overlapping tokens
    in obs t form the input context.

    When the same context→output mapping is seen min_occurrences times,
    creates a morphism node with input edges and an output edge.

    Returns statistics.
    """
    all_observations = hippo.all_observations()
    observations = all_observations[max(0, since_index):]
    if len(observations) < 3:
        return {"morphisms_created": 0, "patterns_found": 0}

    # --- Step 1: Extract (context_set, output_nid) from consecutive pairs ---
    # An action observation has exactly 1 token (the chosen action).
    # The preceding observation is the context.
    # Generate all SUBSETS of size 2..max_context_size from the context.
    # The minimal consistent subset is the morphism we want.
    raw_mappings: list[tuple[set[NodeId], NodeId]] = []

    for i in range(len(observations) - 1):
        obs_context = observations[i]
        obs_action = observations[i + 1]

        # Action observations have exactly 1 token.
        if len(obs_action.token_nids) != 1:
            continue

        action_nid = obs_action.token_nids[0]
        context_nids = set(obs_context.token_nids)

        # The output is the action token, but only if it also appeared
        # in the context observation (the agent is reproducing what it saw).
        if action_nid not in context_nids:
            continue

        # Context = everything in obs_context EXCEPT the output token.
        input_context = context_nids - {action_nid}

        if len(input_context) < 2:
            continue

        raw_mappings.append((input_context, action_nid))

    if not raw_mappings:
        return {"morphisms_created": 0, "patterns_found": 0}

    # --- Step 2: Generate size-2 subsets and count (subset, output) ---
    # Size 2 is the sweet spot: small enough to generalise, large enough
    # to distinguish. {succ, 3} → 4 captures the pattern without noise
    # from PHASE_train, SCORE_0, etc.
    pattern_counts: dict[tuple[frozenset[NodeId], NodeId], int] = defaultdict(int)
    # Also track: for each subset, does it ALWAYS map to the same output?
    pattern_outputs: dict[frozenset[NodeId], set[NodeId]] = defaultdict(set)

    for ctx, out in raw_mappings:
        ctx_list = sorted(ctx)
        n = len(ctx_list)
        # Generate pairs.
        for i in range(n):
            for j in range(i + 1, n):
                subset = frozenset({ctx_list[i], ctx_list[j]})
                pattern_counts[(subset, out)] += 1
                pattern_outputs[subset].add(out)

    # Filter to CONSISTENT patterns: subsets that always map to ONE output.
    # {succ, 3} → {4} is consistent. {succ, PHASE_train} → {0,1,2,...} is not.
    consistent_patterns: dict[tuple[frozenset[NodeId], NodeId], int] = {}
    for (subset, out), count in pattern_counts.items():
        if count < min_occurrences:
            continue
        if len(pattern_outputs[subset]) == 1:
            # This subset ALWAYS maps to this one output.
            consistent_patterns[(subset, out)] = count

    # --- Step 3: Create morphism nodes for consistent patterns ---
    morphisms_created = 0
    patterns_found = 0

    for (ctx, output_nid), count in consistent_patterns.items():
        patterns_found += 1

        # Check if this morphism already exists.
        morph_key = ("__morphism__", ctx, output_nid)
        if morph_key in kg._value_to_node:
            continue

        # Create morphism node.
        morph_nid = kg.get_or_create(morph_key)
        kg._nodes[morph_nid].resting = 0.4

        # Input edges: context_node → morphism (co-occurrence role).
        for ctx_nid in ctx:
            edge = kg.get_or_create_edge(ctx_nid, morph_nid, role=COOCCURRENCE)
            edge.strengthen(min(0.15 * count, 0.9))

        # Output edge: morphism → output (co-occurrence role).
        out_edge = kg.get_or_create_edge(morph_nid, output_nid, role=COOCCURRENCE)
        out_edge.strengthen(min(0.15 * count, 0.9))

        morphisms_created += 1

    return {
        "morphisms_created": morphisms_created,
        "patterns_found": patterns_found,
    }
