"""Formal Concept Analysis (FCA) on local chunk concept matrices.

After each segment boundary, the current chunk contains at most W symbols
(W is the empirical segment size, not a parameter).  FCA on the W × R binary
concept matrix finds all formal concepts — closed (symbol_set, edge_type_set)
pairs.  Each concept is a Galois connection = an adjunction in the CTKG.

Complexity: O(2^R · K · R) where K = symbols in chunk, R = edge type count.
  Typical:   K ≤ 20, R ≤ 8  →  2^8 × 20 × 8 = 40,960 operations, microseconds.
  Worst case (3D 26-conn+time): R=28 → 2^28 ≈ 268M.  For R>16, switch to
  the NextConcepts algorithm (stubbed below); for validation (R≤8) this is fine.

References:
  Wille 1982 — Restructuring Lattice Theory (FCA original paper)
  RESEARCH.md §"Formal Concept Analysis as adjunction discovery"
"""

from __future__ import annotations

from typing import Optional


# ── Concept matrix construction ───────────────────────────────────────────────

def chunk_concept_matrix(
    chunk: list[tuple[int, Optional[int]]],
    n_edge_types: int,
) -> tuple[list[int], list[int], list[list[bool]]]:
    """Build the FCA concept matrix from a chunk buffer.

    chunk         : list of (symbol_id, incoming_edge_type_int | None)
    n_edge_types  : total number of registered edge types

    Returns:
      objects   : list of distinct symbol IDs in the chunk
      attributes: list of edge type integers observed in the chunk
      matrix    : matrix[i][j] = True iff objects[i] appeared with attributes[j]
                  (as incoming edge) in this chunk

    Note: edge_type = None (first element) is excluded from the attribute set.
    """
    # Collect which edge types each symbol was observed with
    obj_to_attrs: dict[int, set[int]] = {}
    for sid, etype in chunk:
        if etype is None:
            continue
        if sid not in obj_to_attrs:
            obj_to_attrs[sid] = set()
        obj_to_attrs[sid].add(etype)

    # Also include symbols that only appear as the first element (no edge)
    # so they appear as objects with an empty attribute set
    for sid, etype in chunk:
        if sid not in obj_to_attrs:
            obj_to_attrs[sid] = set()

    objects: list[int] = sorted(obj_to_attrs.keys())
    # Only include edge types that actually appear in this chunk
    attrs_present: set[int] = set()
    for a_set in obj_to_attrs.values():
        attrs_present |= a_set
    attributes: list[int] = sorted(attrs_present)

    if not objects or not attributes:
        return objects, attributes, []

    obj_idx  = {o: i for i, o in enumerate(objects)}
    attr_idx = {a: j for j, a in enumerate(attributes)}

    K = len(objects)
    R = len(attributes)
    matrix: list[list[bool]] = [[False] * R for _ in range(K)]
    for sid, a_set in obj_to_attrs.items():
        i = obj_idx[sid]
        for a in a_set:
            if a in attr_idx:
                matrix[i][attr_idx[a]] = True

    return objects, attributes, matrix


# ── FCA algorithm ─────────────────────────────────────────────────────────────

def formal_concepts(
    objects: list,
    attributes: list,
    matrix: list[list[bool]],
) -> list[tuple[frozenset, frozenset]]:
    """Find all formal concepts in a binary context.

    A formal concept is a maximal (object_set, attribute_set) pair such that:
      - every object in the set has every attribute in the set
      - no object or attribute can be added while preserving the above

    Algorithm: enumerate all 2^|attributes| attribute subsets, compute the
    closure, keep canonical (closed) ones.  Suitable for |attributes| ≤ 16.
    For larger R, see _formal_concepts_nextconcepts() below.

    Returns a list of (frozenset of objects, frozenset of attributes).
    """
    K = len(objects)
    R = len(attributes)
    if K == 0 or R == 0:
        return []

    # Represent each object as a bitmask over attributes
    obj_masks: list[int] = [0] * K
    for i in range(K):
        for j in range(R):
            if matrix[i][j]:
                obj_masks[i] |= (1 << j)

    # Full attribute mask
    full_mask = (1 << R) - 1

    concepts: list[tuple[frozenset, frozenset]] = []
    seen_closures: set[int] = set()

    if R <= 20:
        # Enumerate all 2^R attribute subsets
        for attr_mask in range(full_mask + 1):
            # Attribute-to-object derivation: objects that have ALL attributes in attr_mask
            a_set: list[int] = []
            for i in range(K):
                if (obj_masks[i] & attr_mask) == attr_mask:
                    a_set.append(i)

            # Object-to-attribute derivation: attributes shared by ALL objects in a_set
            if not a_set:
                closed_attr_mask = full_mask  # no objects → all attributes trivially shared
            else:
                closed_attr_mask = full_mask
                for i in a_set:
                    closed_attr_mask &= obj_masks[i]

            # Canonical check: only keep if this is the closure of attr_mask
            if closed_attr_mask != attr_mask:
                continue
            if closed_attr_mask in seen_closures:
                continue
            seen_closures.add(closed_attr_mask)

            # Second derivation: objects with ALL closed attributes
            obj_set = frozenset(
                objects[i] for i in range(K)
                if (obj_masks[i] & closed_attr_mask) == closed_attr_mask
            )
            attr_set = frozenset(
                attributes[j] for j in range(R)
                if (closed_attr_mask >> j) & 1
            )
            concepts.append((obj_set, attr_set))
    else:
        # R > 20: fall back to object-enumeration (2^K subsets)
        # This is appropriate when K << R, e.g. small chunks with many edge types
        concepts = _formal_concepts_object_enum(objects, attributes, matrix, obj_masks)

    return concepts


def _formal_concepts_object_enum(
    objects: list,
    attributes: list,
    matrix: list[list[bool]],
    obj_masks: list[int],
) -> list[tuple[frozenset, frozenset]]:
    """Object-subset enumeration FCA for R > 20.  O(2^K · K · R)."""
    K = len(objects)
    R = len(attributes)
    full_attr_mask = (1 << R) - 1

    # Each attribute as a bitmask over objects
    attr_obj_masks: list[int] = [0] * R
    for j in range(R):
        for i in range(K):
            if matrix[i][j]:
                attr_obj_masks[j] |= (1 << i)

    concepts: list[tuple[frozenset, frozenset]] = []
    seen: set[int] = set()

    for obj_mask in range(1 << K):
        # Derive attributes shared by all objects in obj_mask
        if obj_mask == 0:
            attr_mask = full_attr_mask
        else:
            attr_mask = full_attr_mask
            for i in range(K):
                if (obj_mask >> i) & 1:
                    attr_mask &= obj_masks[i]

        # Derive objects that have all attributes in attr_mask
        if attr_mask == 0:
            closed_obj_mask = (1 << K) - 1
        else:
            closed_obj_mask = (1 << K) - 1
            for j in range(R):
                if (attr_mask >> j) & 1:
                    closed_obj_mask &= attr_obj_masks[j]

        if closed_obj_mask in seen:
            continue
        seen.add(closed_obj_mask)

        obj_set = frozenset(objects[i] for i in range(K) if (closed_obj_mask >> i) & 1)
        attr_set = frozenset(
            attributes[j] for j in range(R) if (attr_mask >> j) & 1
        )
        concepts.append((obj_set, attr_set))

    return concepts


# ── Convenience ───────────────────────────────────────────────────────────────

def concepts_from_chunk(
    chunk: list[tuple[int, Optional[int]]],
    n_edge_types: int,
) -> list[tuple[frozenset, frozenset]]:
    """One-shot: build concept matrix from chunk and run FCA.

    Returns a list of (frozenset[symbol_id], frozenset[edge_type_int]) formal concepts.
    """
    objects, attributes, matrix = chunk_concept_matrix(chunk, n_edge_types)
    return formal_concepts(objects, attributes, matrix)
