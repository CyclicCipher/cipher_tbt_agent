"""
Evidence-Triggered Theory Revision Pipeline — Phase 16 of the Einstein Roadmap.

auto_revise_on_anomaly integrates:
  residual check → discover_law → retract old → add replacement → preservation check

Iron Law: no dispatch on theory or morphism names. Works identically for
any domain given the right observations and context.

Bitter Lesson compliance
------------------------
No physical law names are hardcoded. The pipeline uses numerical residuals
and MSE thresholds, not domain-specific checks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph, MorphId
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import FittedLaw, add_fitted_law, predict_continuous
from experiments.symbolic_ai_v2.ctkg.core.prim_ops import PrimSpec, get_prim_specs, make_prim_ctx
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext, eval_expr
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager, TheoryId
from experiments.symbolic_ai_v2.ctkg.inference.revision import ClosedLoopReviser
from experiments.symbolic_ai_v2.ctkg.inference.retract import RetractEngine
from experiments.symbolic_ai_v2.ctkg.inference.preservation import (
    PredictionLedger,
    check_preservation,
)
from experiments.symbolic_ai_v2.ctkg.inference.compose_search import discover_law

_FITTED_LAW_TYPE = "FITTED_LAW"


@dataclass
class RevisionPipelineResult:
    """Result of auto_revise_on_anomaly.

    Attributes
    ----------
    accepted         : True if the revision was applied and passed preservation.
    retracted_mid    : MorphId of the old morphism that was retracted (if any).
    replacement_mid  : MorphId of the new replacement morphism (if any).
    discovered_law   : FittedLaw discovered from anomalous obs (if any).
    rejection_reason : String describing why revision was rejected (if not accepted).
    """
    accepted: bool
    retracted_mid: Optional[MorphId] = None
    replacement_mid: Optional[MorphId] = None
    discovered_law: Optional[FittedLaw] = None
    rejection_reason: Optional[str] = None


def auto_revise_on_anomaly(
    theory_id: TheoryId,
    anomalous_obs: list[tuple[dict, float]],
    ctx: EvalContext,
    mg: MorphismGraph,
    tm: TheoryManager,
    rev: ClosedLoopReviser,
    eng: RetractEngine,
    ledger: PredictionLedger,
    prim_ctx: Optional[EvalContext] = None,
    prim_specs: Optional[list[PrimSpec]] = None,
    max_depth: int = 4,
    beam_width: int = 60,
    anomaly_threshold: float = 0.1,
    fit_threshold: float = 0.05,
    tolerance: float = 0.05,
    label: str = "__auto_revised__",
    extra_atom_values: Optional[list[float]] = None,
) -> RevisionPipelineResult:
    """Evidence-triggered theory revision.

    Algorithm
    ---------
    1. Check residual of current theory on anomalous_obs.
       If residual < anomaly_threshold: return "not_anomalous".

    2. Discover replacement law from anomalous_obs.
       If law.residual > fit_threshold: return "poor_fit".

    3. Find the worst-performing morphism in theory_id (the one to retract).
       If no morphisms: return "no_candidate".

    4. Retract the old morphism, add the replacement.

    5. Check preservation against ledger entries.
       If preservation fails: undo (un-retract old, veto new).
       Return "preservation_failed" if preservation fails.

    6. Return accepted=True.

    Parameters
    ----------
    theory_id        : the theory to potentially revise.
    anomalous_obs    : observations that triggered the revision.
    ctx              : EvalContext for theory predictions.
    mg, tm           : graph and theory manager.
    rev              : ClosedLoopReviser (holds _vetoed set).
    eng              : RetractEngine.
    ledger           : PredictionLedger — contains prior correct predictions.
    prim_ctx         : EvalContext for discover_law. Defaults to ctx.
    prim_specs       : PrimSpec list for discover_law.
    max_depth        : max expression tree depth.
    beam_width       : beam width for discover_law.
    anomaly_threshold: minimum relative residual to consider anomalous.
    fit_threshold    : maximum residual for the discovered law to be accepted.
    tolerance        : relative error threshold for preservation check.
    label            : morphism label for the new law.
    extra_atom_values: extra fixed constants for discover_law.

    Returns
    -------
    RevisionPipelineResult.
    """
    if prim_ctx is None:
        prim_ctx = ctx
    if prim_specs is None:
        prim_specs = get_prim_specs()

    # Step 1: Check residual on anomalous observations
    residual = _compute_residual(theory_id, anomalous_obs, ctx, tm)
    if residual < anomaly_threshold:
        return RevisionPipelineResult(
            accepted=False,
            rejection_reason="not_anomalous",
        )

    # Step 2: Discover replacement law
    law = discover_law(
        anomalous_obs,
        prim_ctx=prim_ctx,
        prim_specs=prim_specs,
        max_depth=max_depth,
        beam_width=beam_width,
        extra_atom_values=extra_atom_values,
    )

    if law.residual > fit_threshold:
        return RevisionPipelineResult(
            accepted=False,
            discovered_law=law,
            rejection_reason="poor_fit",
        )

    # Step 3: Find old morphism to retract (worst performer on anomalous obs)
    old_mid = _find_worst_morphism(theory_id, anomalous_obs, ctx, mg, tm, rev)
    if old_mid is None:
        # No retractable morphism found; just add the new law
        new_mid = add_fitted_law(mg, f"{label}_replacement", law)
        tm.assign_morphism(new_mid, theory_id)
        return RevisionPipelineResult(
            accepted=True,
            replacement_mid=new_mid,
            discovered_law=law,
        )

    # Step 4: Retract old, add new
    eng.retract_morphism(old_mid, theory_id, reason="auto_revise: anomaly detected")
    new_mid = add_fitted_law(mg, f"{label}_replacement", law)
    tm.assign_morphism(new_mid, theory_id)

    # Step 5: Check preservation against ledger
    examples = ledger.correct_examples(theory_id)
    if examples:
        pres = check_preservation(theory_id, examples, mg, tm, ctx, tolerance=tolerance)
        if not pres.passed:
            # Undo: un-retract old, veto new
            rev._vetoed.discard(old_mid)
            rev._vetoed.add(new_mid)
            return RevisionPipelineResult(
                accepted=False,
                retracted_mid=old_mid,
                replacement_mid=new_mid,
                discovered_law=law,
                rejection_reason=f"preservation_failed: {pres.reason}",
            )

    return RevisionPipelineResult(
        accepted=True,
        retracted_mid=old_mid,
        replacement_mid=new_mid,
        discovered_law=law,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_residual(
    theory_id: TheoryId,
    observations: list[tuple[dict, float]],
    ctx: EvalContext,
    tm: TheoryManager,
) -> float:
    """Compute mean relative squared residual of theory on observations.

    Returns float('inf') if the theory cannot predict any observation.
    """
    errors = []
    for inp, obs in observations:
        pred = tm.predict_under_theory(theory_id, inp, ctx)
        if pred is None:
            errors.append(float("inf"))
            continue
        denom = abs(obs) + 1e-12
        rel_err = (pred - obs) ** 2 / (denom ** 2)
        errors.append(rel_err)

    if not errors or all(math.isinf(e) for e in errors):
        return float("inf")
    finite_errors = [e for e in errors if math.isfinite(e)]
    return sum(finite_errors) / len(finite_errors) if finite_errors else float("inf")


def _find_worst_morphism(
    theory_id: TheoryId,
    anomalous_obs: list[tuple[dict, float]],
    ctx: EvalContext,
    mg: MorphismGraph,
    tm: TheoryManager,
    rev: ClosedLoopReviser,
) -> Optional[MorphId]:
    """Find the FITTED_LAW morphism in theory_id with the highest anomaly error.

    Returns None if no non-vetoed FITTED_LAW morphism exists.
    """
    member_ids = mg.theory_members(theory_id)
    worst_mid: Optional[MorphId] = None
    worst_error: float = -1.0

    for mid in member_ids:
        if mid in rev._vetoed:
            continue
        m = mg.morphism_by_id(mid)
        if m is None or m.morph_type != _FITTED_LAW_TYPE:
            continue
        law: FittedLaw = m.payload
        total_error = 0.0
        count = 0
        for inp, obs in anomalous_obs:
            try:
                pred = predict_continuous(law, inp, ctx)
                total_error += abs(pred - obs)
                count += 1
            except (ValueError, KeyError, TypeError):
                pass
        if count == 0:
            continue
        avg_error = total_error / count
        if avg_error > worst_error:
            worst_error = avg_error
            worst_mid = mid

    return worst_mid
