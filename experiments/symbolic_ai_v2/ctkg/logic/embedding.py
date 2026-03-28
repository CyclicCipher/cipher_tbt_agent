"""
Concept embedding — interpretable continuous vectors from FCA.

Each token gets a vector of length |concepts|, where each dimension
is a formal concept discovered by FCA. The value in each dimension is
the token's continuous membership in that concept: what fraction of
the concept's observations does this token appear in?

Binary FCA: token is in concept or not (0/1).
Continuous: token appears in 80% of concept's observations → 0.8.

This gives an interpretable embedding where:
- Each dimension has a known meaning (the concept's extent + intent)
- Distance between tokens = distance between their concept vectors
- The FCA lattice hierarchy gives algebraic structure ON the dimensions
- Multi-threshold persistence gives scale separation

The embedding IS the enriched hom-object. Two tokens' distance in
this space is their distance in the enriched category.

No domain-specific code. Operates on abstract NodeIds and observations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId, COOCCURRENCE,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus
from experiments.symbolic_ai_v2.ctkg.logic.fca import (
    FormalConcept, multi_threshold_fca,
)


# ---------------------------------------------------------------------------
# Concept embedding computation
# ---------------------------------------------------------------------------

def compute_concept_embeddings(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    min_persistence: int = 2,
    min_concept_size: int = 2,
    max_concept_size: int = 10,
    max_observations: int = 500,
) -> dict[str, Any]:
    """Compute concept membership vectors for all tokens.

    1. Get FCA concepts (already computed or recompute).
    2. For each concept, collect its intent (observation set).
    3. For each token, compute membership in each concept:
       membership(token, concept) = |intent(concept) ∩ obs(token)| / |intent(concept)|
    4. Store as kg._embeddings: {NodeId: list[float]}
       and kg._embedding_concepts: list of concept extents (the dimension labels).

    Returns stats.
    """
    # Get observations for membership computation.
    all_obs = hippo.all_observations()
    obs_slice = all_obs[max(0, since_index):]
    if len(obs_slice) > max_observations:
        obs_slice = obs_slice[-max_observations:]
    if not obs_slice:
        kg._embeddings = {}
        kg._embedding_concepts = []
        return {"embedding_dims": 0, "tokens_embedded": 0}

    obs_base = max(0, since_index)
    if len(all_obs[obs_base:]) > max_observations:
        obs_base = len(all_obs) - max_observations

    # Build token → observation set index.
    token_obs: dict[NodeId, set[int]] = {}
    for obs_idx, obs in enumerate(obs_slice):
        abs_idx = obs_base + obs_idx
        for nid in obs.token_nids:
            if nid not in token_obs:
                token_obs[nid] = set()
            token_obs[nid].add(abs_idx)

    # Get FCA concepts.
    result = multi_threshold_fca(
        kg, hippo, since_index=since_index,
        min_persistence=min_persistence,
        max_observations=max_observations,
    )

    # Filter to persistent, right-sized concepts.
    concepts: list[FormalConcept] = []
    for fc in result.concepts:
        if fc.persistence < min_persistence:
            continue
        if not (min_concept_size <= len(fc.extent) <= max_concept_size):
            continue
        if not fc.intent:
            continue
        concepts.append(fc)

    if not concepts:
        kg._embeddings = {}
        kg._embedding_concepts = []
        return {"embedding_dims": 0, "tokens_embedded": 0}

    # Compute continuous membership for each token in each concept.
    # membership(token, concept) = |intent(concept) ∩ obs(token)| / |intent(concept)|
    n_dims = len(concepts)
    embeddings: dict[NodeId, list[float]] = {}

    for nid in token_obs:
        vec = []
        for fc in concepts:
            if not fc.intent:
                vec.append(0.0)
                continue
            overlap = len(fc.intent & token_obs.get(nid, set()))
            membership = overlap / len(fc.intent)
            vec.append(membership)
        embeddings[nid] = vec

    # Store on the knowledge graph.
    kg._embeddings = embeddings
    kg._embedding_concepts = [fc.extent for fc in concepts]

    return {
        "embedding_dims": n_dims,
        "tokens_embedded": len(embeddings),
    }


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def embedding_distance(
    kg: KnowledgeGraph,
    a: NodeId,
    b: NodeId,
) -> float:
    """Euclidean distance between two tokens in concept embedding space.

    Returns float('inf') if either token has no embedding.
    """
    ea = kg._embeddings.get(a)
    eb = kg._embeddings.get(b)
    if ea is None or eb is None:
        return float('inf')
    if len(ea) != len(eb):
        return float('inf')
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(ea, eb)))


def embedding_similarity(
    kg: KnowledgeGraph,
    a: NodeId,
    b: NodeId,
) -> float:
    """Cosine similarity between two tokens in concept embedding space.

    Returns 0.0 if either token has no embedding or norms are zero.
    Range: [-1, 1]. Higher = more similar.
    """
    ea = kg._embeddings.get(a)
    eb = kg._embeddings.get(b)
    if ea is None or eb is None:
        return 0.0

    dot = sum(x * y for x, y in zip(ea, eb))
    norm_a = math.sqrt(sum(x * x for x in ea))
    norm_b = math.sqrt(sum(x * x for x in eb))

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def nearest_in_embedding(
    kg: KnowledgeGraph,
    query: NodeId,
    candidates: list[NodeId],
    metric: str = "cosine",
) -> NodeId | None:
    """Find the candidate nearest to query in embedding space.

    metric: "cosine" (similarity, higher=closer) or "euclidean" (distance, lower=closer).
    """
    if not candidates:
        return None

    eq = kg._embeddings.get(query)
    if eq is None:
        return None

    best_nid = None
    best_score = float('-inf') if metric == "cosine" else float('inf')

    for cand in candidates:
        if cand == query:
            continue
        if metric == "cosine":
            score = embedding_similarity(kg, query, cand)
            if score > best_score:
                best_score = score
                best_nid = cand
        else:
            score = embedding_distance(kg, query, cand)
            if score < best_score:
                best_score = score
                best_nid = cand

    return best_nid


# ---------------------------------------------------------------------------
# Successor via embedding: find the nearest candidate to the successor's
# expected position in embedding space.
# ---------------------------------------------------------------------------

def successor_via_embedding(
    kg: KnowledgeGraph,
    query: NodeId,
    candidates: list[NodeId],
) -> NodeId | None:
    """Predict successor by embedding geometry.

    If the query has a discovered successor in _discovered_succ, use the
    successor's embedding to find the nearest candidate (handles cases
    where the direct successor isn't a candidate but a nearby token is).

    If no direct successor, find candidates that are close to the query
    AND in the same direction as the typical successor displacement
    (the average vector from tokens to their successors).
    """
    if not kg._embeddings or not candidates:
        return None

    eq = kg._embeddings.get(query)
    if eq is None:
        return None

    # Strategy 1: if query has a discovered successor with an embedding,
    # find the candidate nearest to that successor's position.
    # Only return if the nearest candidate is NOT the query itself
    # (which would mean the successor is closer to the query than to
    # any forward neighbor — a degenerate case).
    direct_succ = kg._discovered_succ.get(query)
    if direct_succ is not None:
        es = kg._embeddings.get(direct_succ)
        if es is not None:
            result = nearest_in_embedding(kg, direct_succ, candidates, "cosine")
            if result is not None and result != query:
                return result

    # Strategy 2 would compute successor direction from known pairs,
    # but with 189 noisy dimensions the displacement vectors are unreliable.
    # Return None to fall through to PMI/co-occurrence layers.
    return None
