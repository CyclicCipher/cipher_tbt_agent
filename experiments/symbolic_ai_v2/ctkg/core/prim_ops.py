"""
Primitive arithmetic operation registry — Phase 10 of the Einstein Roadmap.

Defines a fixed set of typed primitive operations (PRIM_MUL, PRIM_ADD, etc.)
that compose_search uses to build expression trees during law discovery.

Iron Law compliance: morph_type strings like "PRIM_MUL" are structural meta-labels
(not domain operator name tokens). The beam search dispatches on morph_type and
NodeId, never on domain vocabulary strings.

Bitter Lesson compliance: the EvalContext maps NodeIds to callables. A renamed
PRIM_MUL NodeId evaluates identically to the original — the morph_type string
"PRIM_MUL" is only used to locate the callable, not as domain knowledge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH, NodeId
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext

# Morph-type string constants for the 9 primitives
PRIM_MUL  = "PRIM_MUL"
PRIM_ADD  = "PRIM_ADD"
PRIM_SUB  = "PRIM_SUB"
PRIM_DIV  = "PRIM_DIV"
PRIM_POW  = "PRIM_POW"
PRIM_NEG  = "PRIM_NEG"
PRIM_SQRT = "PRIM_SQRT"
PRIM_INV  = "PRIM_INV"
PRIM_SQ   = "PRIM_SQ"

# (morph_type, arity, fn)
_PRIM_DEFS: list[tuple[str, int, Callable]] = [
    (PRIM_MUL,  2, lambda a, b: a * b),
    (PRIM_ADD,  2, lambda a, b: a + b),
    (PRIM_SUB,  2, lambda a, b: a - b),
    (PRIM_DIV,  2, lambda a, b: a / b if b != 0.0 else float("nan")),
    (PRIM_POW,  2, lambda a, b: float(a ** b) if abs(b) < 10 else float("nan")),
    (PRIM_NEG,  1, lambda a: -a),
    (PRIM_SQRT, 1, lambda a: math.sqrt(a) if a >= 0.0 else float("nan")),
    (PRIM_INV,  1, lambda a: 1.0 / a if a != 0.0 else float("nan")),
    (PRIM_SQ,   1, lambda a: a * a),
]

# Public lookup: morph_type -> fn
PRIM_FNS: dict[str, Callable] = {mt: fn for mt, _, fn in _PRIM_DEFS}


@dataclass(frozen=True)
class PrimSpec:
    """Specification for one primitive operation.

    Attributes
    ----------
    nid        : NodeId = TOKEN_GRAPH.encode(morph_type). Stable across calls.
    morph_type : structural meta-label string (e.g. "PRIM_MUL").
    arity      : 1 or 2.
    """
    nid:       NodeId
    morph_type: str
    arity:     int


def get_prim_specs() -> list[PrimSpec]:
    """Return PrimSpec for all 9 primitive operations.

    NodeIds are assigned by TOKEN_GRAPH.encode(morph_type) and are stable
    across calls within a single process (TOKEN_GRAPH uses a shared registry).
    """
    return [
        PrimSpec(nid=TOKEN_GRAPH.encode(mt), morph_type=mt, arity=arity)
        for mt, arity, _ in _PRIM_DEFS
    ]


def make_prim_ctx() -> EvalContext:
    """Build an EvalContext covering all 9 primitive operations.

    Maps each primitive's NodeId to its Python callable. No domain operators
    are included — this context is for compose_search's internal evaluation only.
    """
    return EvalContext({
        TOKEN_GRAPH.encode(mt): fn
        for mt, _, fn in _PRIM_DEFS
    })
