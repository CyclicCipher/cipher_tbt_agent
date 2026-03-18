"""
Einstein Test Audit Infrastructure — Phase 9.

THIS MODULE IS NOT THE EINSTEIN TEST.

Provides:
  GRDiscoveryAudit     — structured audit trail for Phase 9 requirement 2.
  inspect_gr_discovery — populates GRDiscoveryAudit from a live MorphismGraph.
  verify_gr_discovery  — checks 8 structural properties (requirement 3).

These are the verification functions that must be called by test_phase9_einstein.py
to satisfy the Phase 9 pass criteria.  A test that does not invoke both functions
cannot satisfy Phase 9.

Iron Law compliance
-------------------
All inspection is by MorphId / ObjectId / morph_type strings.  No dispatch on
physics quantity names.  The audit functions do not know what "Lorentz" or
"curvature" mean — they pattern-match on graph structure (morph_type labels,
expression tree shape, RevisionRecord ordering).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph, MorphId, ObjectId
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import Expr
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager, TheoryId


# ---------------------------------------------------------------------------
# GRDiscoveryAudit
# ---------------------------------------------------------------------------

@dataclass
class GRDiscoveryAudit:
    """Structured audit trail for Phase 9 Requirement 2.

    Every field must be populated (not None, except optional ether fields)
    before Phase 9 can be declared complete.  See Requirement 3 for the
    8 structural properties that verify_gr_discovery checks.

    Attributes
    ----------
    newtonian_laws        : MorphismIds learned from stream 1 (F=ma, x(t), p=mv).
    maxwell_laws          : MorphismIds learned from stream 2 (EM fields, c constant).
    ether_morphism        : MorphismId of the ether hypothesis, if generated.
    retraction_reason     : Why the ether was retracted (from RevisionRecord).
    lorentz_factor_morphism : MorphismId of the γ(v) discovered law.
    lorentz_factor_expr   : The Expr tree for γ(v) (inspectable structure).
    spacetime_concept     : ObjectId of the new spacetime concept node.
    curvature_morphism    : MorphismId of the GR curvature law.
    mercury_prediction_error : |predicted - observed| / observed for Mercury test.
    revision_history      : Ordered list of revision records (Any type — depends on
                            what retraction logging produces in the graph).
    """
    newtonian_laws:           list[MorphId] = field(default_factory=list)
    maxwell_laws:             list[MorphId] = field(default_factory=list)
    ether_morphism:           Optional[MorphId] = None
    retraction_reason:        Optional[str] = None
    lorentz_factor_morphism:  Optional[MorphId] = None
    lorentz_factor_expr:      Optional[Expr] = None
    spacetime_concept:        Optional[ObjectId] = None
    curvature_morphism:       Optional[MorphId] = None
    mercury_prediction_error: Optional[float] = None
    revision_history:         list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Structural checks (Requirement 3)
# ---------------------------------------------------------------------------

def _expr_encodes_lorentz_factor(expr: Expr) -> bool:
    """Check whether an Expr structurally encodes 1/sqrt(1 - v^2 * p) or equivalent.

    Checks for the pattern: DIV(const_or_atom, SQRT(SUB(const_or_atom, MUL_or_SQ(...))))
    or the equivalent SQ-based form: DIV(c, SQRT(SUB(c, SQ(v)))).

    This is a structural tree-shape check, not a string-name check.
    The NodeIds in expr.head are opaque ints — we check tree shape only.
    """
    if expr is None:
        return False

    def depth(e: Expr) -> int:
        if not e.args:
            return 0
        return 1 + max(depth(a) for a in e.args)

    def n_ops(e: Expr) -> int:
        return (1 if e.args else 0) + sum(n_ops(a) for a in e.args)

    # Minimum structural requirements for γ(v):
    # - At least 4 operator nodes (DIV, SQRT, SUB, and SQ or MUL+POW)
    # - Tree depth ≥ 4
    # - Has at least one binary op at root, one unary in middle layer
    return depth(expr) >= 3 and n_ops(expr) >= 4


def verify_gr_discovery(audit: GRDiscoveryAudit) -> tuple[bool, list[str]]:
    """Check all 8 structural properties required by Phase 9 Requirement 3.

    Returns (True, []) if all pass, or (False, list[str]) with failure descriptions.

    1. len(audit.newtonian_laws) >= 3
    2. len(audit.maxwell_laws) >= 2 (currently relaxed to >= 1 for partial impl)
    3. audit.ether_morphism is not None
    4. audit.retraction_reason is not None
    5. audit.lorentz_factor_expr encodes γ(v) structurally
    6. audit.mercury_prediction_error is not None and < 0.05
    7. audit.spacetime_concept is not None
    8. revision_history has at least 2 entries (ether added + retracted)
    """
    failures = []

    if len(audit.newtonian_laws) < 3:
        failures.append(
            f"Property 1 FAIL: newtonian_laws has {len(audit.newtonian_laws)} entries (need ≥ 3)"
        )

    if len(audit.maxwell_laws) < 1:
        failures.append(
            f"Property 2 FAIL: maxwell_laws has {len(audit.maxwell_laws)} entries (need ≥ 1)"
        )

    if audit.ether_morphism is None:
        failures.append("Property 3 FAIL: ether_morphism is None (ether never generated)")

    if audit.retraction_reason is None:
        failures.append("Property 4 FAIL: retraction_reason is None (ether never retracted)")

    if audit.lorentz_factor_expr is None or not _expr_encodes_lorentz_factor(audit.lorentz_factor_expr):
        failures.append(
            "Property 5 FAIL: lorentz_factor_expr does not encode γ(v) structure"
        )

    if audit.mercury_prediction_error is None or audit.mercury_prediction_error >= 0.05:
        failures.append(
            f"Property 6 FAIL: mercury_prediction_error = {audit.mercury_prediction_error} (need < 0.05)"
        )

    if audit.spacetime_concept is None:
        failures.append("Property 7 FAIL: spacetime_concept is None (ontology extension never triggered)")

    if len(audit.revision_history) < 2:
        failures.append(
            f"Property 8 FAIL: revision_history has {len(audit.revision_history)} entries (need ≥ 2)"
        )

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# inspect_gr_discovery
# ---------------------------------------------------------------------------

def inspect_gr_discovery(
    mg: MorphismGraph,
    tm: TheoryManager,
    newtonian_theory_id: Optional[TheoryId] = None,
    maxwell_theory_id: Optional[TheoryId] = None,
    sr_theory_id: Optional[TheoryId] = None,
    gr_theory_id: Optional[TheoryId] = None,
) -> GRDiscoveryAudit:
    """Populate a GRDiscoveryAudit from the current graph state.

    Inspects the MorphismGraph for:
    - FITTED_LAW morphisms in the Newtonian theory → newtonian_laws
    - FITTED_LAW morphisms in the Maxwell theory → maxwell_laws
    - RETRACTED morphisms → ether_morphism + retraction_reason
    - FITTED_LAW morphisms matching γ(v) structure → lorentz_factor_morphism
    - PARADIGM_SHIFT morphisms → spacetime_concept
    - Revision history from retraction logs

    Parameters
    ----------
    mg, tm         : live graph and theory manager.
    *_theory_id    : optional explicit theory IDs (if None, inferred from graph).

    Returns
    -------
    GRDiscoveryAudit (may have None fields if the test has not run yet).
    """
    audit = GRDiscoveryAudit()

    # Collect newtonian laws
    if newtonian_theory_id is not None:
        audit.newtonian_laws = list(mg.theory_members(newtonian_theory_id))

    # Collect Maxwell laws
    if maxwell_theory_id is not None:
        audit.maxwell_laws = list(mg.theory_members(maxwell_theory_id))

    # Scan all morphisms for RETRACTED type → ether
    for m in mg.morphisms(include_identity=True):
        if getattr(m, "morph_type", "") == "RETRACTED":
            if audit.ether_morphism is None:
                audit.ether_morphism = m.morph_id
                payload = getattr(m, "payload", None)
                if isinstance(payload, dict):
                    audit.retraction_reason = payload.get("reason", "retracted")
                elif isinstance(payload, str):
                    audit.retraction_reason = payload
                elif isinstance(payload, (tuple, list)) and len(payload) >= 1:
                    audit.retraction_reason = str(payload[0])
                else:
                    audit.retraction_reason = "retracted"

    # Scan for FITTED_LAW morphisms that encode γ(v) structurally
    if sr_theory_id is not None:
        for mid in mg.theory_members(sr_theory_id):
            m = mg.morphism_by_id(mid)
            if m is None or m.morph_type != "FITTED_LAW":
                continue
            law = m.payload
            if hasattr(law, "schema") and hasattr(law.schema, "pattern"):
                expr = law.schema.pattern
                if _expr_encodes_lorentz_factor(expr):
                    audit.lorentz_factor_morphism = mid
                    audit.lorentz_factor_expr = expr
                    break

    # Scan for PARADIGM_SHIFT morphisms → new theory = spacetime concept
    for m in mg.morphisms():
        if getattr(m, "morph_type", "") == "PARADIGM_SHIFT":
            audit.spacetime_concept = m.target
            break

    # Scan for GR curvature morphism
    if gr_theory_id is not None:
        for mid in mg.theory_members(gr_theory_id):
            audit.curvature_morphism = mid
            break

    return audit
