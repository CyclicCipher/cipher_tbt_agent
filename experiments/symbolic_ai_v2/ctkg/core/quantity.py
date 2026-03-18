"""
Continuous quantities and unit-typed nodes — Phase 3 of the Einstein Roadmap.

QuantityNode
------------
A typed real-valued measurement: (value: float, unit_id: NodeId).
The unit is an opaque NodeId — not a string.  Two QuantityNode objects with
the same float value but different unit_ids are DISTINCT (they refer to the
same numeric magnitude in different measurement systems).

EvalContext
-----------
A dict-like container mapping NodeId → Python callable.  This is the ONLY
place where operator semantics live.  The evaluator dispatches on NodeId,
not on string names — Iron Law compliant.

Unit registry
-------------
Unit conversions are stored as UNIT_CONV morphisms in a MorphismGraph.
Each morphism carries a float conversion factor as its payload.
`register_unit_conversion(mg, from_id, to_id, factor)` stores the morphism.
`get_unit_factor(mg, from_id, to_id)` retrieves it.
`convert_quantity(mg, qty, target_unit)` converts a QuantityNode.

Iron Law compliance
-------------------
No string comparisons on operator names or unit names anywhere in this module.
All operator identity is by NodeId.  Unit identity is by NodeId.
The cage tests verify zero-variance across 10 anonymous symbol tables.

Bitter Lesson compliance
------------------------
The EvalContext is data, not code.  A formula `⊕(⊗(k, m), a)` evaluated with
the mapping {⊕: lambda a,b: a*b, ⊗: lambda a,b: a*b} produces the same
result as `mul(mul(k, m), a)` with the standard mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH, NodeId
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import Expr
from experiments.symbolic_ai_v2.ctkg.core.expr_law import _ensure_object, _find_object

_UNIT_CONV_TYPE = "UNIT_CONV"


# ---------------------------------------------------------------------------
# QuantityNode
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantityNode:
    """A typed real-valued measurement.

    Attributes
    ----------
    value   : the numeric magnitude (float).
    unit_id : NodeId identifying the unit type.  Two QuantityNodes with the
              same value but different unit_ids are NOT equal — they measure
              different things in different units.

    Examples
    --------
    >>> m_nid  = TOKEN_GRAPH.encode('m')   # metres
    >>> mm_nid = TOKEN_GRAPH.encode('mm')  # millimetres
    >>> QuantityNode(1.0, m_nid) != QuantityNode(1.0, mm_nid)  # True
    >>> QuantityNode(1.0, m_nid) != QuantityNode(1000.0, mm_nid)  # semantically same; structurally distinct
    """
    value:   float
    unit_id: NodeId


# ---------------------------------------------------------------------------
# EvalContext
# ---------------------------------------------------------------------------

@dataclass
class EvalContext:
    """Operator dispatch table for numeric expression evaluation.

    operators : dict[NodeId, Callable]
        Maps each operator NodeId to its Python callable.
        Callables receive positional float arguments matching the operator
        arity and return a float.

    Example
    -------
    >>> mul_nid = TOKEN_GRAPH.encode('mul')
    >>> ctx = EvalContext({mul_nid: lambda a, b: a * b})
    >>> eval_expr(node('mul', atom('3'), atom('5')), {}, ctx)
    15.0
    """
    operators: dict[int, Callable] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------

def eval_expr(
    expr:     Expr,
    bindings: dict[str, float],
    ctx:      EvalContext,
) -> float:
    """Numerically evaluate an Expr tree.

    Dispatch is on NodeId (Expr.head) — no string comparisons.

    Parameters
    ----------
    expr     : the expression to evaluate.  May contain var() nodes (looked
               up in `bindings`) and atom nodes (parsed as float literals or
               looked up in `bindings` if not parseable as float).
    bindings : dict mapping variable/atom names → float values.
    ctx      : EvalContext holding the operator callables.

    Returns
    -------
    float — the evaluated result.

    Raises
    ------
    ValueError  — if a var() is unbound, an atom cannot be interpreted as
                  float, or no callable is registered for an operator.
    """
    if expr.is_var:
        name = TOKEN_GRAPH.decode(expr.head)
        if name in bindings:
            return float(bindings[name])
        raise ValueError(f"eval_expr: unbound variable {name!r}")

    if not expr.args:
        # Leaf atom: look up in bindings first, then try float literal
        name = TOKEN_GRAPH.decode(expr.head)
        if name in bindings:
            return float(bindings[name])
        try:
            return float(name)
        except ValueError:
            raise ValueError(
                f"eval_expr: atom {name!r} is not a float literal and not in bindings"
            )

    # Internal node: dispatch on NodeId
    op_id = expr.head
    callable_ = ctx.operators.get(op_id)
    if callable_ is None:
        raise ValueError(
            f"eval_expr: no operator registered for NodeId {op_id} "
            f"(token={TOKEN_GRAPH.decode(op_id)!r})"
        )
    arg_vals = [eval_expr(arg, bindings, ctx) for arg in expr.args]
    return float(callable_(*arg_vals))


# ---------------------------------------------------------------------------
# Unit registry
# ---------------------------------------------------------------------------

def register_unit_conversion(
    mg:       MorphismGraph,
    from_id:  NodeId,
    to_id:    NodeId,
    factor:   float,
) -> None:
    """Store a unit conversion factor as a UNIT_CONV morphism.

    Adds a directed edge from_unit → to_unit with payload=factor, meaning:
        quantity_in_to_unit = quantity_in_from_unit * factor

    Example: metres → millimetres, factor=1000.0
        QuantityNode(1.0, m_nid) → QuantityNode(1000.0, mm_nid)

    Parameters
    ----------
    mg      : graph to write into.
    from_id : NodeId of the source unit.
    to_id   : NodeId of the target unit.
    factor  : multiplicative conversion factor.
    """
    from_obj_id = _unit_object(mg, from_id)
    to_obj_id   = _unit_object(mg, to_id)
    # Dedup: if this conversion already exists, update factor; else add
    for m in mg.source_morphisms(from_obj_id, morph_type=_UNIT_CONV_TYPE):
        if m.target == to_obj_id:
            # Already registered; leave as-is (idempotent)
            return
    mg.add_morphism(
        from_obj_id,
        to_obj_id,
        morph_type=_UNIT_CONV_TYPE,
        evidence=1,
        payload=float(factor),
    )


def get_unit_factor(
    mg:      MorphismGraph,
    from_id: NodeId,
    to_id:   NodeId,
) -> Optional[float]:
    """Return the direct conversion factor from from_id to to_id, or None.

    Only checks direct (single-hop) conversions.  Multi-hop is not yet
    supported (Phase 4+).
    """
    from_obj_id = _find_unit_object(mg, from_id)
    if from_obj_id is None:
        return None
    to_obj_id = _find_unit_object(mg, to_id)
    if to_obj_id is None:
        return None
    for m in mg.source_morphisms(from_obj_id, morph_type=_UNIT_CONV_TYPE):
        if m.target == to_obj_id:
            return float(m.payload)
    return None


def convert_quantity(
    mg:         MorphismGraph,
    qty:        QuantityNode,
    target_unit: NodeId,
) -> Optional[QuantityNode]:
    """Convert a QuantityNode to a different unit using the graph's conversion morphisms.

    Returns a new QuantityNode with the converted value, or None if no
    conversion path is registered.
    """
    if qty.unit_id == target_unit:
        return qty
    factor = get_unit_factor(mg, qty.unit_id, target_unit)
    if factor is None:
        return None
    return QuantityNode(value=qty.value * factor, unit_id=target_unit)


# ---------------------------------------------------------------------------
# Continuous surprise
# ---------------------------------------------------------------------------

def continuous_surprise(
    predicted: float,
    observed:  float,
    sigma:     float = 1.0,
) -> float:
    """Squared-error surprise for continuous observations.

    Returns (predicted - observed)² / σ², analogous to the discrete KL
    surprise used in the token domain.  Higher values indicate more surprise.

    Parameters
    ----------
    predicted : model's predicted float value.
    observed  : actual observed float value.
    sigma     : noise standard deviation (default 1.0).

    Returns
    -------
    float ≥ 0.
    """
    return ((predicted - observed) / sigma) ** 2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unit_label(unit_id: NodeId) -> str:
    return f"__unit_{unit_id}"


def _unit_object(mg: MorphismGraph, unit_id: NodeId) -> int:
    """Get or create a graph object representing this unit NodeId."""
    label = _unit_label(unit_id)
    return _ensure_object(mg, label)


def _find_unit_object(mg: MorphismGraph, unit_id: NodeId) -> Optional[int]:
    label = _unit_label(unit_id)
    return _find_object(mg, label)
