"""
Expression laws as CTKG morphisms — Phase 1 of the Einstein Roadmap.

An ExprLaw stores a (pattern → conclusion) Expr pair as an EXPR_LAW self-loop
morphism on a named object in the MorphismGraph.  The morphism payload is the
(pattern, conclusion) tuple.

Iron Law compliance
-------------------
All operator identity is carried by NodeId inside the Expr (Expr.head is a
NodeId, not a string).  No string comparisons on operator content occur inside
this module.  The only string↔NodeId conversions happen at the public API
boundary (law_label for object lookup, variable names in bindings dicts).

Bitter Lesson compliance
------------------------
Every operation is a structural graph traversal.  A law stored with operator
NodeId 42 (which encodes 'mul') behaves identically when the same structure is
stored with operator NodeId 9001 (which encodes '⊕').  The cage tests in
test_expr_law.py verify this across 10 independent anonymous symbol tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph,
    MorphId,
    ObjectId,
)
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import Expr, match, substitute

_EXPR_LAW_TYPE = "EXPR_LAW"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class ExprLaw:
    """An algebraic law stored as an EXPR_LAW morphism.

    Attributes
    ----------
    pattern    : LHS expression; may contain var() nodes.
    conclusion : RHS expression; may contain var() nodes.
    morph_id   : identity of the morphism in the MorphismGraph.
    """
    pattern:    Expr
    conclusion: Expr
    morph_id:   MorphId


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def add_expr_law(
    mg: MorphismGraph,
    law_label: str,
    pattern: Expr,
    conclusion: Expr,
) -> ExprLaw:
    """Store (pattern → conclusion) as an EXPR_LAW self-loop in mg.

    If an identical law (same pattern and conclusion) already exists under
    law_label, the existing ExprLaw is returned without duplication.

    Parameters
    ----------
    mg          : graph to write into.
    law_label   : string label for the anchor object (e.g. "newton_second").
    pattern     : LHS Expr (may contain var() nodes).
    conclusion  : RHS Expr (may contain var() nodes).

    Returns
    -------
    ExprLaw wrapping the new (or existing) morphism.
    """
    anchor_id = _ensure_object(mg, law_label)

    # Deduplication: if identical (pattern, conclusion) already stored, reuse.
    for m in mg.source_morphisms(anchor_id, morph_type=_EXPR_LAW_TYPE):
        p, c = m.payload
        if p == pattern and c == conclusion:
            return ExprLaw(pattern=p, conclusion=c, morph_id=m.morph_id)

    m = mg.add_morphism(
        anchor_id,
        anchor_id,
        morph_type=_EXPR_LAW_TYPE,
        evidence=1,
        payload=(pattern, conclusion),
    )
    return ExprLaw(pattern=pattern, conclusion=conclusion, morph_id=m.morph_id)


def query_expr_laws(mg: MorphismGraph, law_label: str) -> list[ExprLaw]:
    """Return all ExprLaws stored under law_label.

    Returns an empty list when no object with that label exists.
    """
    anchor_id = _find_object(mg, law_label)
    if anchor_id is None:
        return []
    result: list[ExprLaw] = []
    for m in mg.source_morphisms(anchor_id, morph_type=_EXPR_LAW_TYPE):
        p, c = m.payload
        result.append(ExprLaw(pattern=p, conclusion=c, morph_id=m.morph_id))
    return result


def match_law(law: ExprLaw, expr: Expr) -> Optional[dict[str, Expr]]:
    """Match expr against law.pattern.

    Returns a bindings dict {var_name: Expr} on success, or None on failure.
    Delegates entirely to term_algebra.match — no string dispatch.
    """
    return match(law.pattern, expr)


def apply_law(law: ExprLaw, bindings: dict[str, Expr]) -> Expr:
    """Substitute bindings into law.conclusion.

    Delegates entirely to term_algebra.substitute — no string dispatch.
    """
    return substitute(law.conclusion, bindings)


def match_and_apply(
    mg: MorphismGraph,
    law_label: str,
    expr: Expr,
) -> Optional[Expr]:
    """Convenience: find the first law under law_label that matches expr and apply it.

    Returns the instantiated conclusion, or None if no law matches.
    Tries laws in storage order; the first match wins.
    """
    for law in query_expr_laws(mg, law_label):
        bindings = match_law(law, expr)
        if bindings is not None:
            return apply_law(law, bindings)
    return None


# ---------------------------------------------------------------------------
# Structural renaming (used by cage tests)
# ---------------------------------------------------------------------------

def rename_expr(expr: Expr, nid_map: dict[int, int]) -> Expr:
    """Return a copy of expr with all NodeIds renamed according to nid_map.

    NodeIds not present in nid_map are left unchanged.  Variable flag is
    preserved.  Used by compliance tests to replace named operator NodeIds
    with anonymous Unicode NodeIds, verifying symbol-invariant behaviour.
    """
    new_head = nid_map.get(expr.head, expr.head)
    if not expr.args:
        return Expr(head=new_head, args=(), is_var=expr.is_var)
    new_args = tuple(rename_expr(a, nid_map) for a in expr.args)
    return Expr(head=new_head, args=new_args, is_var=expr.is_var)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_object(mg: MorphismGraph, label: str) -> ObjectId:
    for obj in mg.objects():
        if obj.label == label:
            return obj.obj_id
    return mg.add_object(concept=None, label=label).obj_id


def _find_object(mg: MorphismGraph, label: str) -> Optional[ObjectId]:
    for obj in mg.objects():
        if obj.label == label:
            return obj.obj_id
    return None
