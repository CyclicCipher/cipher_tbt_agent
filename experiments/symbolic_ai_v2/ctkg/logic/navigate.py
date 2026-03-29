"""
Causal navigation — directed movement through the embedding space.

Learns operator displacement vectors: for each token that appears between
an input and an output in observation→action pairs, compute the average
vector from input embedding to output embedding. This displacement IS
the operator's meaning — it's the direction to move in concept space.

Navigation: given a query operand and an operator in context, apply the
operator's displacement to the query's embedding and find the nearest
candidate to the predicted position.

No boundary detection, no role labeling. The environment controls when
to produce output (by calling act()). This module only handles WHERE
to navigate given the current context.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId, COOCCURRENCE,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NavigationStructure:
    """Learned operator displacements in embedding space."""
    operator_displacements: dict[NodeId, list[float]] = field(default_factory=dict)
    operator_confidence: dict[NodeId, int] = field(default_factory=dict)  # example count


# ---------------------------------------------------------------------------
# Learn operator displacements from observation→action pairs
# ---------------------------------------------------------------------------

def learn_displacements(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    min_examples: int = 3,
) -> NavigationStructure:
    """Learn operator displacement vectors from (context → action) pairs.

    For each consecutive pair where the action is a single token:
    1. The action token is the OUTPUT.
    2. In the context, find tokens that also appear as actions elsewhere
       (these are operand-type tokens).
    3. Non-operand tokens between operands are candidate operators.
    4. For each (operand, operator, output) triple, compute the embedding
       displacement: embedding(output) - embedding(operand).
    5. Average across examples to get the operator's displacement vector.

    The key insight: we don't need to label tokens as "operator" or
    "operand" in advance. We discover this from the structure:
    - Operand tokens appear both in context AND as single-token actions.
    - Operator tokens appear in context but NEVER as single-token actions.
    - The displacement from operand to correct output, conditioned on
      which operator is present, gives the operator's meaning.
    """
    if not kg._embeddings:
        return NavigationStructure()

    obs_list = hippo.all_observations()
    obs_recent = obs_list[max(0, since_index):]
    if len(obs_recent) < 3:
        return NavigationStructure()

    # Find tokens that appear as single-token actions (operand-type).
    action_set: set[NodeId] = set()
    for obs in obs_recent:
        if len(obs.token_nids) == 1:
            action_set.add(obs.token_nids[0])

    # For each (context → action) pair where action is operand-type:
    # find all (operand, non-operand, output) triples.
    # operand = context token that's also in action_set
    # non-operand = context token that's NOT in action_set
    # output = the action token
    triple_deltas: dict[NodeId, list[list[float]]] = defaultdict(list)

    for i in range(len(obs_recent) - 1):
        ctx = obs_recent[i]
        act = obs_recent[i + 1]
        if len(act.token_nids) != 1:
            continue
        output_nid = act.token_nids[0]
        if output_nid not in action_set:
            continue

        e_out = kg._embeddings.get(output_nid)
        if e_out is None:
            continue

        # Find operands and non-operands in context.
        ctx_operands = []
        ctx_non_operands = []
        for pos, nid in enumerate(ctx.token_nids):
            if nid in action_set:
                ctx_operands.append((pos, nid))
            else:
                ctx_non_operands.append((pos, nid))

        if not ctx_operands or not ctx_non_operands:
            continue

        # For each non-operand (candidate operator), pair it with the
        # operand that appears just before it (causal order: input
        # precedes operator precedes output).
        for op_pos, op_nid in ctx_non_operands:
            # Find the operand closest before this operator.
            input_nid = None
            for opr_pos, opr_nid in reversed(ctx_operands):
                if opr_pos < op_pos:
                    input_nid = opr_nid
                    break

            if input_nid is None or input_nid == output_nid:
                continue

            e_in = kg._embeddings.get(input_nid)
            if e_in is None or len(e_in) != len(e_out):
                continue

            delta = [e_out[j] - e_in[j] for j in range(len(e_in))]
            triple_deltas[op_nid].append(delta)

    # Average deltas per operator. Only keep operators with enough examples.
    structure = NavigationStructure()
    for op_nid, deltas in triple_deltas.items():
        if len(deltas) < min_examples:
            continue
        n_dims = len(deltas[0])
        avg = [0.0] * n_dims
        for d in deltas:
            for j in range(n_dims):
                avg[j] += d[j]
        for j in range(n_dims):
            avg[j] /= len(deltas)
        structure.operator_displacements[op_nid] = avg
        structure.operator_confidence[op_nid] = len(deltas)

    return structure


# ---------------------------------------------------------------------------
# Navigate: apply operator displacement to query
# ---------------------------------------------------------------------------

def navigate(
    kg: KnowledgeGraph,
    query_nid: NodeId,
    operator_nid: NodeId,
    candidates: list[NodeId],
) -> NodeId | None:
    """Navigate from query in the operator's direction.

    predicted = embedding(query) + displacement(operator)
    Returns the candidate nearest to the predicted position.
    Returns None if embeddings or displacement not available.
    """
    nav = kg._causal_structure
    if nav is None:
        return None

    eq = kg._embeddings.get(query_nid)
    if eq is None:
        return None

    disp = nav.operator_displacements.get(operator_nid)
    if disp is None or len(disp) != len(eq):
        return None

    predicted = [eq[i] + disp[i] for i in range(len(eq))]

    best_nid = None
    best_dist = float('inf')
    for cand in candidates:
        if cand == query_nid:
            continue
        ec = kg._embeddings.get(cand)
        if ec is None or len(ec) != len(predicted):
            continue
        dist = math.sqrt(sum((p - c) ** 2 for p, c in zip(predicted, ec)))
        if dist < best_dist:
            best_dist = dist
            best_nid = cand

    return best_nid


# ---------------------------------------------------------------------------
# Main: discover and cache navigation structure
# ---------------------------------------------------------------------------

def discover_navigation(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
) -> dict[str, Any]:
    """Learn operator displacements and cache on KG.

    Called during consolidation.
    """
    structure = learn_displacements(kg, hippo, since_index=since_index)
    kg._causal_structure = structure

    return {
        "operators_with_displacement": len(structure.operator_displacements),
        "total_examples": sum(structure.operator_confidence.values()),
    }
