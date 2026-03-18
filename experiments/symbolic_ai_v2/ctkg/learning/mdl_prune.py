"""
Phase 7: domain-indexed MDL pruning of the MorphismGraph.

A morphism is retained iff its presence reduces the description length:

    support * log2(support) > lambda_prune * |body| * log2(max(vocab_size, 2))

where support = evidence_count, |body| = len(morphism.body), and vocab_size is
the number of distinct atoms in the Hankel vocabulary.

This is a conservative criterion: low-support morphisms (vacuous generalisations)
are pruned; high-support morphisms (genuine patterns) are retained.

Identity morphisms are never pruned regardless of support.

The pruning returns a new MorphismGraph with the same objects but only the
retained morphisms (plus identities).  The original graph is not modified.

See CTKG_ARCHITECTURE.md §Phase 7 for the full specification.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph,
    CTKGMorphism,
)

if TYPE_CHECKING:
    from experiments.symbolic_ai_v2.ctkg.core.rewrite import RewriteRule


def mdl_prune(
    morphism_graph: MorphismGraph,
    vocab_size: int = 20,
    lambda_prune: float = 0.05,
    min_support: int = 2,
) -> MorphismGraph:
    """Prune low-utility morphisms from a MorphismGraph.

    Parameters
    ----------
    morphism_graph:
        The MorphismGraph to prune (not modified in place).
    vocab_size:
        Number of distinct atom types in the vocabulary.  Used to compute the
        description-length penalty per symbol.  Default 20.
    lambda_prune:
        MDL regularisation coefficient.  Higher = more aggressive pruning.
        Default 0.05.
    min_support:
        Minimum evidence_count for a morphism to be retained at all,
        regardless of the MDL criterion.  Default 2.

    Returns
    -------
    A new MorphismGraph with the same objects as the input and only the
    morphisms that pass both the min_support and MDL criteria.
    """
    new_mg = MorphismGraph()

    # Copy all objects (and their auto-created identities)
    obj_map: dict[int, int] = {}  # old obj_id → new obj_id
    for obj in morphism_graph.objects():
        new_obj = new_mg.add_object(obj.concept, label=obj.label)
        obj_map[obj.obj_id] = new_obj.obj_id

    # Compute MDL threshold variables
    log_vocab = math.log2(max(vocab_size, 2))

    # Copy morphisms that pass the pruning criterion
    for m in morphism_graph.morphisms(include_identity=False):
        if not _should_keep(m, lambda_prune, log_vocab, min_support):
            continue

        # Map source/target to new object IDs
        new_src = obj_map.get(m.source)
        new_tgt = obj_map.get(m.target)
        if new_src is None or new_tgt is None:
            continue

        new_body = [obj_map.get(oid, oid) for oid in m.body]

        new_mg.add_morphism(
            source_id=new_src,
            target_id=new_tgt,
            body=new_body,
            evidence=m.evidence_count,
            morph_type=m.morph_type,
            confidence=m.confidence,
        )

    return new_mg


def _should_keep(
    m: CTKGMorphism,
    lambda_prune: float,
    log_vocab: float,
    min_support: int,
) -> bool:
    """Return True if morphism m should be retained."""
    # Hard support floor
    if m.evidence_count < min_support:
        return False

    # MDL criterion: benefit must exceed penalty
    support = m.evidence_count
    body_len = max(len(m.body), 1)

    benefit = support * math.log2(max(support, 1))
    penalty = lambda_prune * body_len * log_vocab

    return benefit >= penalty


# ---------------------------------------------------------------------------
# Phase VII — semantic deduplication and storage policy
# ---------------------------------------------------------------------------


def _normalize_morph_type(morph_type: str, norm_map: dict[str, str]) -> str:
    """Normalize a morph_type string using a head→head substitution map.

    morph_type strings use '∘' as composition separator. Each component
    is looked up in norm_map and replaced if found.
    """
    if not norm_map:
        return morph_type
    parts = morph_type.split('∘')
    return '∘'.join(norm_map.get(p.strip(), p.strip()) for p in parts)


def _build_norm_map(rules: list[RewriteRule]) -> dict[str, str]:
    """Build a head→head normalization map from rewrite rules.

    For each rule lhs_head → rhs_head (where both are function heads, not
    variables), record lhs_head → rhs_head so that surface-form aliases
    (sq → pow, etc.) can be normalized before grouping morphisms.
    """
    norm_map: dict[str, str] = {}
    for rule in rules:
        lhs_head = rule.lhs.head if not rule.lhs.is_var else None
        rhs_head = rule.rhs.head if not rule.rhs.is_var else None
        if lhs_head and rhs_head and lhs_head != rhs_head:
            norm_map[lhs_head] = rhs_head
    return norm_map


def semantic_deduplicate(
    mg: MorphismGraph,
    rules: list[RewriteRule] | None = None,
) -> MorphismGraph:
    """Remove semantically duplicate morphisms from a MorphismGraph.

    Two morphisms are considered duplicates if they have the same:
    - source concept label
    - target concept label
    - normalized morph_type (after applying rewrite rule head→head aliases)

    Among each group of duplicates, the morphism with the highest evidence_count
    is retained. Ties are broken by morph_id (lower = older, prefer higher
    evidence as it reflects more corpus support).

    Identity morphisms are always preserved.

    Parameters
    ----------
    mg:
        Input MorphismGraph (not modified).
    rules:
        Optional list of RewriteRules used to build the normalization map.
        If None or empty, morph_type strings are compared as-is.

    Returns
    -------
    A new MorphismGraph with duplicate morphisms removed.
    """
    if rules is None:
        rules = []

    norm_map = _build_norm_map(rules)

    # Build label index for source/target lookup
    obj_labels: dict[int, str] = {obj.obj_id: obj.label for obj in mg.objects()}

    # Group non-identity morphisms by (src_label, tgt_label, norm_type)
    groups: dict[tuple, list[CTKGMorphism]] = defaultdict(list)
    for m in mg.morphisms(include_identity=False):
        src_label = obj_labels.get(m.source, str(m.source))
        tgt_label = obj_labels.get(m.target, str(m.target))
        norm_type = _normalize_morph_type(m.morph_type, norm_map)
        key = (src_label, tgt_label, norm_type)
        groups[key].append(m)

    # Determine which morph_ids to retain (best representative per group)
    keep_ids: set[int] = set()
    for group in groups.values():
        best = max(group, key=lambda m: (m.evidence_count, -m.morph_id))
        keep_ids.add(best.morph_id)

    # Build new graph
    new_mg = MorphismGraph()
    obj_map: dict[int, int] = {}
    for obj in mg.objects():
        new_obj = new_mg.add_object(obj.concept, label=obj.label)
        obj_map[obj.obj_id] = new_obj.obj_id

    for m in mg.morphisms(include_identity=False):
        if m.morph_id not in keep_ids:
            continue
        new_src = obj_map.get(m.source)
        new_tgt = obj_map.get(m.target)
        if new_src is None or new_tgt is None:
            continue
        new_body = [obj_map.get(oid, oid) for oid in m.body]
        new_mg.add_morphism(
            source_id=new_src,
            target_id=new_tgt,
            body=new_body,
            evidence=m.evidence_count,
            morph_type=m.morph_type,
            confidence=m.confidence,
        )

    return new_mg


def compute_storage_policy(
    mg: MorphismGraph,
    rules: list[RewriteRule] | None = None,
    k_steps: int = 5,
) -> dict[str, bool]:
    """Determine which morphisms must be stored vs. can be recomputed.

    A morphism is considered *reconstructible* if its morph_type can be
    expressed as a ≤ k_steps composition of other morphisms already in the
    graph (i.e., it is a composite that adds no information beyond its parts).

    Morphisms that are reconstructible are marked False (no need to store
    explicitly); those that are atomic or deep are marked True (must store).

    The policy is returned as a dict mapping morph_type → bool (store=True).
    Identity morphisms are always False (trivially reconstructible).

    Parameters
    ----------
    mg:
        MorphismGraph to analyse.
    rules:
        Optional rewrite rules (unused currently, reserved for future
        semantic analysis of morph_type components).
    k_steps:
        Maximum composition depth below which a morphism is considered
        reconstructible.  Default 5.

    Returns
    -------
    dict mapping each morph_type string → True (store) / False (reconstruct).
    """
    # Collect all known morph_types in the graph
    all_types: set[str] = set()
    for m in mg.morphisms(include_identity=False):
        all_types.add(m.morph_type)

    policy: dict[str, bool] = {}
    for mt in all_types:
        # Count composition depth by splitting on '∘'
        parts = [p.strip() for p in mt.split('∘') if p.strip()]
        depth = len(parts)
        # A single-component morph_type is atomic — must store
        # A multi-component type with depth ≤ k_steps is reconstructible
        # from its parts if all parts are present in the graph
        if depth == 1:
            policy[mt] = True  # atomic, must store
        elif depth <= k_steps:
            # Reconstructible iff each component is itself in the graph
            component_types = {
                m.morph_type for m in mg.morphisms(include_identity=False)
            }
            all_present = all(p in component_types for p in parts)
            policy[mt] = not all_present  # store only if parts are missing
        else:
            # Too deep — store as a single unit to avoid explosion
            policy[mt] = True

    return policy
