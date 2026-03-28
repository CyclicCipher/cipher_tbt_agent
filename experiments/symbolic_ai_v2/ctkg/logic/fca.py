"""
Formal Concept Analysis — domain-general structure discovery.

Bottom-up hierarchical lattice construction: compute token intents
(which observations each token appears in), then build upward by
intent intersection. Each level discovers coarser structure.

Level 0: individual token closures (atoms)
Level 1: pairwise intent intersections
Level 2: triple intersections (meets of level-1 concepts)
...
Level k: meets of level-(k-1) concepts

The Hasse diagram (subconcept ordering) comes free from the
construction — each concept's parents are the level-(k-1) concepts
it was built from.

Multi-threshold persistence: instead of running the full lattice at
multiple thresholds, we filter the incidence data by edge weight
threshold and track which concepts survive across thresholds. Concepts
persisting at all thresholds are the grid-cell scale-module analog.

No function in this file inspects token labels or values. It operates
on NodeIds and observation indices only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId, COOCCURRENCE,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FormalConcept:
    """One node in the concept lattice."""
    extent: frozenset[NodeId]       # set of token NodeIds
    intent: frozenset[int]          # set of observation indices
    level: int = 0                  # level in the bottom-up construction
    persistence: int = 1            # how many thresholds this survives
    node_id: NodeId | None = None   # KG node once materialized


@dataclass
class FCAResult:
    """Result of a multi-threshold FCA run."""
    concepts: list[FormalConcept]
    subconcept_pairs: list[tuple[int, int]]  # (child_idx, parent_idx)
    persistent_concepts: list[FormalConcept]


# ---------------------------------------------------------------------------
# Intent computation from observations
# ---------------------------------------------------------------------------

def _compute_token_intents(
    hippo: Hippocampus,
    kg: KnowledgeGraph,
    since_index: int = 0,
    threshold: float = 0.0,
    max_observations: int = 500,
    max_tokens: int = 200,
) -> dict[NodeId, frozenset[int]]:
    """Compute intent(T) = set of observation indices containing token T.

    With threshold > 0: T is "in" observation O only if all pairwise
    co-occurrence edge weights between T and other tokens in O exceed
    the threshold.

    Returns {NodeId: frozenset of observation indices}.
    """
    all_obs = hippo.all_observations()
    obs_slice = all_obs[max(0, since_index):]
    if len(obs_slice) > max_observations:
        obs_slice = obs_slice[-max_observations:]
    if not obs_slice:
        return {}

    # Collect token frequencies, keep top max_tokens.
    token_freq: dict[NodeId, int] = {}
    for obs in obs_slice:
        for nid in obs.token_nids:
            token_freq[nid] = token_freq.get(nid, 0) + 1

    sorted_tokens = sorted(token_freq.keys(), key=lambda n: -token_freq[n])
    if len(sorted_tokens) > max_tokens:
        sorted_tokens = sorted_tokens[:max_tokens]
    token_set = set(sorted_tokens)

    if not token_set:
        return {}

    # Compute absolute observation base index.
    obs_base = max(0, since_index)
    if len(all_obs[obs_base:]) > max_observations:
        obs_base = len(all_obs) - max_observations

    # Build intent for each token.
    intents: dict[NodeId, set[int]] = {nid: set() for nid in token_set}

    for obs_idx, obs in enumerate(obs_slice):
        abs_idx = obs_base + obs_idx
        obs_nids = [nid for nid in obs.token_nids if nid in token_set]

        for nid in obs_nids:
            if threshold <= 0.0:
                intents[nid].add(abs_idx)
            else:
                passes = True
                for other_nid in obs_nids:
                    if other_nid == nid:
                        continue
                    e1 = kg.edge(nid, other_nid)
                    e2 = kg.edge(other_nid, nid)
                    w1 = e1.weight if e1 is not None else 0.0
                    w2 = e2.weight if e2 is not None else 0.0
                    if max(w1, w2) < threshold:
                        passes = False
                        break
                if passes:
                    intents[nid].add(abs_idx)

    return {nid: frozenset(s) for nid, s in intents.items() if s}


def _close_extent(
    candidate: frozenset[NodeId],
    token_intents: dict[NodeId, frozenset[int]],
    all_tokens: set[NodeId],
) -> tuple[frozenset[NodeId], frozenset[int]]:
    """Compute the closure of a candidate extent.

    closure(S) = extent(intent(S)):
    1. intent(S) = intersection of intents of all members of S
    2. extent(intent(S)) = all tokens whose intent is a superset of intent(S)

    Returns (closed_extent, shared_intent).
    """
    if not candidate:
        return frozenset(), frozenset()

    # Step 1: intent = intersection of all member intents.
    member_intents = [token_intents[nid] for nid in candidate
                      if nid in token_intents]
    if not member_intents:
        return frozenset(), frozenset()

    shared_intent = member_intents[0]
    for mi in member_intents[1:]:
        shared_intent = shared_intent & mi
        if not shared_intent:
            return frozenset(), frozenset()

    # Step 2: extent = all tokens whose intent contains shared_intent.
    closed_extent = frozenset(
        nid for nid in all_tokens
        if nid in token_intents and shared_intent <= token_intents[nid]
    )

    return closed_extent, shared_intent


# ---------------------------------------------------------------------------
# Bottom-up hierarchical lattice construction
# ---------------------------------------------------------------------------

def build_lattice_bottom_up(
    token_intents: dict[NodeId, frozenset[int]],
    max_extent_size: int = 10,
    max_concepts: int = 500,
) -> tuple[list[FormalConcept], list[tuple[int, int]]]:
    """Build the concept lattice bottom-up by intent intersection.

    Level 0: close each individual token → atom concepts.
    Level k: for each pair of level-(k-1) concepts, intersect intents
    and close. If the result is new, add it at level k.

    Stops when no new concepts are generated or max_concepts reached.

    Returns (concepts, hasse_pairs) where hasse_pairs are direct
    subconcept→superconcept edges as (child_idx, parent_idx).
    """
    all_tokens = set(token_intents.keys())
    if not all_tokens:
        return [], []

    # Dedup: map extent → concept index.
    extent_to_idx: dict[frozenset[NodeId], int] = {}
    concepts: list[FormalConcept] = []
    # Parent tracking for Hasse diagram.
    # parent_of[idx] = set of concept indices that this concept was built from.
    parent_of: dict[int, set[int]] = {}

    def _add_concept(extent: frozenset[NodeId], intent: frozenset[int],
                     level: int, parents: set[int] | None = None) -> int | None:
        """Add a concept if new. Returns index or None if duplicate/filtered."""
        if extent in extent_to_idx:
            return extent_to_idx[extent]
        if len(extent) > max_extent_size:
            return None
        if not intent:
            return None
        if len(concepts) >= max_concepts:
            return None

        idx = len(concepts)
        concepts.append(FormalConcept(
            extent=extent, intent=intent, level=level,
        ))
        extent_to_idx[extent] = idx
        if parents is not None:
            parent_of[idx] = parents
        return idx

    # --- Level 0: atom concepts (individual token closures) ---
    for nid in sorted(all_tokens):
        if nid not in token_intents or not token_intents[nid]:
            continue
        extent, intent = _close_extent(frozenset({nid}), token_intents, all_tokens)
        if extent and intent:
            _add_concept(extent, intent, level=0)

    if not concepts:
        return [], []

    # --- Levels 1+: pairwise meets of previous-level concepts ---
    prev_level_start = 0
    prev_level_end = len(concepts)
    level = 1

    while level <= max_extent_size and len(concepts) < max_concepts:
        new_this_level = 0

        # Try all pairs: one from previous level, one from any level.
        # This ensures we discover concepts that are meets of non-adjacent atoms.
        prev_level_indices = list(range(prev_level_start, prev_level_end))
        all_indices = list(range(len(concepts)))

        for i in prev_level_indices:
            ci = concepts[i]
            for j in all_indices:
                if j >= i:
                    continue  # avoid duplicate pairs and self-meet
                cj = concepts[j]

                # Quick reject: if one intent is a subset of the other,
                # intersection gives the smaller — already discovered.
                if ci.intent <= cj.intent or cj.intent <= ci.intent:
                    continue

                # Intersect intents and close.
                shared = ci.intent & cj.intent
                if not shared:
                    continue

                # Compute extent of the shared intent.
                new_extent = frozenset(
                    nid for nid in all_tokens
                    if nid in token_intents and shared <= token_intents[nid]
                )

                if not new_extent or new_extent in extent_to_idx:
                    continue
                if len(new_extent) > max_extent_size:
                    continue

                # Verify closure: extent(intent(new_extent)) should equal new_extent.
                # Since we computed new_extent as extent(shared), and shared is
                # already an intersection of closed intents, the closure is exact.
                result = _add_concept(new_extent, shared, level=level,
                                      parents={i, j})
                if result is not None:
                    new_this_level += 1

                if len(concepts) >= max_concepts:
                    break
            if len(concepts) >= max_concepts:
                break

        if new_this_level == 0:
            break

        prev_level_start = prev_level_end
        prev_level_end = len(concepts)
        level += 1

    # --- Build Hasse diagram ---
    # Direct subconcept: A < B iff A.extent ⊂ B.extent and no C with A < C < B.
    # Optimization: sort by extent size, check only adjacent sizes.
    sorted_by_size = sorted(range(len(concepts)),
                            key=lambda i: len(concepts[i].extent))

    hasse: list[tuple[int, int]] = []
    for si, i in enumerate(sorted_by_size):
        ei = concepts[i].extent
        # Look for the smallest proper supersets.
        for sj in range(si + 1, len(sorted_by_size)):
            j = sorted_by_size[sj]
            ej = concepts[j].extent
            if not (ei < ej):
                continue
            # Check directness: no concept k with ei ⊂ ek ⊂ ej.
            is_direct = True
            for sk in range(si + 1, sj):
                k = sorted_by_size[sk]
                ek = concepts[k].extent
                if ei < ek < ej:
                    is_direct = False
                    break
            if is_direct:
                hasse.append((i, j))

    return concepts, hasse


# ---------------------------------------------------------------------------
# Multi-threshold persistence
# ---------------------------------------------------------------------------

THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]


def multi_threshold_fca(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    thresholds: list[float] | None = None,
    max_observations: int = 500,
    max_tokens: int = 200,
    max_extent_size: int = 10,
    max_concepts: int = 500,
    min_persistence: int = 2,
) -> FCAResult:
    """Run bottom-up FCA at multiple thresholds and find persistent concepts.

    A concept is persistent if the same extent appears at
    >= min_persistence different thresholds.
    """
    if thresholds is None:
        thresholds = list(THRESHOLDS)

    extent_to_thresholds: dict[frozenset[NodeId], list[float]] = {}
    extent_to_concept: dict[frozenset[NodeId], FormalConcept] = {}

    for t in sorted(thresholds):
        intents = _compute_token_intents(
            hippo, kg, since_index=since_index, threshold=t,
            max_observations=max_observations, max_tokens=max_tokens,
        )
        if not intents:
            continue

        concepts_at_t, _ = build_lattice_bottom_up(
            intents, max_extent_size=max_extent_size,
            max_concepts=max_concepts,
        )

        for fc in concepts_at_t:
            if fc.extent not in extent_to_thresholds:
                extent_to_thresholds[fc.extent] = []
                extent_to_concept[fc.extent] = fc
            extent_to_thresholds[fc.extent].append(t)

    # Build final concept list with persistence.
    all_concepts: list[FormalConcept] = []
    for extent, ts in extent_to_thresholds.items():
        fc = extent_to_concept[extent]
        fc.persistence = len(ts)
        all_concepts.append(fc)

    # Hasse diagram on the merged concept set.
    sorted_by_size = sorted(range(len(all_concepts)),
                            key=lambda i: len(all_concepts[i].extent))
    subconcept_pairs: list[tuple[int, int]] = []
    for si, i in enumerate(sorted_by_size):
        ei = all_concepts[i].extent
        for sj in range(si + 1, len(sorted_by_size)):
            j = sorted_by_size[sj]
            ej = all_concepts[j].extent
            if not (ei < ej):
                continue
            is_direct = True
            for sk in range(si + 1, sj):
                k = sorted_by_size[sk]
                ek = all_concepts[k].extent
                if ei < ek < ej:
                    is_direct = False
                    break
            if is_direct:
                subconcept_pairs.append((i, j))

    persistent = [fc for fc in all_concepts if fc.persistence >= min_persistence]

    return FCAResult(
        concepts=all_concepts,
        subconcept_pairs=subconcept_pairs,
        persistent_concepts=persistent,
    )


# ---------------------------------------------------------------------------
# Convenience: single-threshold FCA (used by tests)
# ---------------------------------------------------------------------------

def run_fca_at_threshold(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    threshold: float = 0.0,
    max_observations: int = 500,
    max_tokens: int = 200,
) -> list[FormalConcept]:
    """Run bottom-up FCA at a single threshold. Returns formal concepts."""
    intents = _compute_token_intents(
        hippo, kg, since_index=since_index, threshold=threshold,
        max_observations=max_observations, max_tokens=max_tokens,
    )
    if not intents:
        return []
    concepts, _ = build_lattice_bottom_up(intents)
    return concepts


# ---------------------------------------------------------------------------
# Incidence matrix (kept for test compatibility)
# ---------------------------------------------------------------------------

def build_incidence_matrix(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    threshold: float = 0.0,
    max_observations: int = 500,
    max_tokens: int = 200,
) -> tuple[list[str], list[str], list[tuple[bool, ...]]]:
    """Build the binary incidence matrix (for tests and diagnostics).

    Returns (object_names, property_names, bools) — same format as the
    concepts library expects, but we no longer depend on that library
    for lattice computation.
    """
    intents = _compute_token_intents(
        hippo, kg, since_index=since_index, threshold=threshold,
        max_observations=max_observations, max_tokens=max_tokens,
    )
    if not intents:
        return [], [], []

    # Collect all observation indices.
    all_obs_indices: set[int] = set()
    for s in intents.values():
        all_obs_indices |= s
    if not all_obs_indices:
        return [], [], []

    token_list = sorted(intents.keys())
    obs_list = sorted(all_obs_indices)
    obs_set_lookup = {idx: i for i, idx in enumerate(obs_list)}

    objects = [str(nid) for nid in token_list]
    properties = [f"obs_{idx}" for idx in obs_list]

    bools: list[tuple[bool, ...]] = []
    for nid in token_list:
        row = [False] * len(obs_list)
        for obs_idx in intents[nid]:
            row[obs_set_lookup[obs_idx]] = True
        bools.append(tuple(row))

    return objects, properties, bools


# ---------------------------------------------------------------------------
# Main entry point: discover and materialize FCA structure
# ---------------------------------------------------------------------------

def discover_fca_structure(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    min_persistence: int = 2,
    min_concept_size: int = 2,
    max_concept_size: int = 10,
) -> dict[str, Any]:
    """Discover and materialize FCA concepts in the KnowledgeGraph.

    Called by Consolidation.consolidate(). Creates concept nodes and
    edges encoding the type hierarchy (subconcept ordering) and cocone
    maps (member tokens -> concept node).

    Returns stats dict.
    """
    result = multi_threshold_fca(
        kg, hippo, since_index=since_index, min_persistence=min_persistence,
        max_extent_size=max_concept_size,
    )

    concepts_discovered = len(result.concepts)
    concepts_materialized = 0
    cocone_edges = 0
    hierarchy_edges = 0

    materialized_indices: dict[int, NodeId] = {}

    for idx, fc in enumerate(result.concepts):
        if fc.persistence < min_persistence:
            continue
        if not (min_concept_size <= len(fc.extent) <= max_concept_size):
            continue

        concept_key = ("__fca_concept__", fc.extent)
        concept_nid = kg.get_or_create(concept_key)
        fc.node_id = concept_nid
        materialized_indices[idx] = concept_nid

        node = kg.node(concept_nid)
        if node is not None:
            node.resting = max(node.resting, 0.1 * fc.persistence)

        concepts_materialized += 1

        for member_nid in fc.extent:
            if member_nid not in kg._nodes:
                continue
            edge = kg.get_or_create_edge(member_nid, concept_nid, role=COOCCURRENCE)
            edge.strengthen(min(0.1 * fc.persistence, 0.8))
            cocone_edges += 1

    for child_idx, parent_idx in result.subconcept_pairs:
        if child_idx in materialized_indices and parent_idx in materialized_indices:
            child_nid = materialized_indices[child_idx]
            parent_nid = materialized_indices[parent_idx]
            edge = kg.get_or_create_edge(child_nid, parent_nid, role=COOCCURRENCE)
            child_fc = result.concepts[child_idx]
            parent_fc = result.concepts[parent_idx]
            strength = min(child_fc.persistence, parent_fc.persistence)
            edge.strengthen(min(0.1 * strength, 0.8))
            hierarchy_edges += 1

    return {
        "concepts_discovered": concepts_discovered,
        "concepts_materialized": concepts_materialized,
        "persistent_count": len(result.persistent_concepts),
        "cocone_edges": cocone_edges,
        "hierarchy_edges": hierarchy_edges,
    }
