"""
Paradigm Shift — Phase 10 of the Einstein Roadmap (A-8).

When an anomaly cannot be absorbed into the existing theory — even after
RetractEngine retraction, ClosedLoopReviser revision, latent abduction, and
multi-anomaly coverage all fail — the system must propose a NEW theory cluster
to cover it.  This is the Left Kan Extension in categorical terms: extend the
current functor to a new object class that was outside its original domain.

Motivation: Einstein test
--------------------------
After Michelson-Morley, the anomaly (null ether drift) is irreconcilable with
Newtonian mechanics and the luminiferous ether theory.  No single morphism
retraction or latent variable fixes it.  The correct move is to CREATE a new
theory (Special Relativity) that:
  1. Covers the anomaly.
  2. Reduces to the old theory in the appropriate limit.

The bridge morphism (Newton → SR) encodes the limiting relationship.  In the
CTKG this is a PARADIGM_SHIFT edge: old_theory_id → new_theory_id.

Design
------
ParadigmShiftResult
    old_theory_id    : the theory that could not absorb the anomaly.
    new_theory_id    : the fresh theory cluster created for the anomaly.
    bridge_morph_id  : MorphId of the PARADIGM_SHIFT morphism.
    explanation      : Optional[LatentHypothesis] — hypothesis in new theory.
    anomaly_coverage : fraction of anomaly sets explained by new theory.

propose_paradigm_shift(theory_id, anomalies, schema_g_list, schema_h, ctx,
                        mg, tm, new_theory_name, label_prefix, min_coverage)
    1. Create a new theory node (new_theory_id).
    2. Fit a latent hypothesis within the new theory for the anomalies.
    3. Score coverage of the hypothesis over the anomaly sets.
    4. If coverage ≥ min_coverage, store as PARADIGM_SHIFT morphism.
    5. Return ParadigmShiftResult, or None if the fit fails.

Left Kan Extension view
    The old theory F: C → D is a functor from known objects to observations.
    The anomaly lives in a new region X outside C's domain.
    propose_paradigm_shift computes the left Kan extension LanF(X):
    a new functor F': C ∪ {X} → D that agrees with F on C and extends
    to cover X.
    In code: the new theory node IS X; the morphisms in it ARE LanF(X).

Iron Law compliance
-------------------
No token names used.  All dispatch via ObjectId / MorphId / morph_type strings.

Defect probe
------------
A naive system adds the anomalous observations as new morphisms IN THE EXISTING
theory (contaminating it, breaking preservation for class-A examples).
The correct system:
  1. Creates a NEW theory cluster.
  2. Adds the hypothesis ONLY to the new cluster.
  3. Links with a typed bridge morphism.
The probe verifies that the old theory's morphism set is UNCHANGED after
propose_paradigm_shift, and that class-A predictions are unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph,
    MorphId,
    ObjectId,
)
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import predict_continuous
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager, TheoryId
from experiments.symbolic_ai_v2.ctkg.inference.latent import (
    LatentHypothesis,
    hypothesise_latent,
)
from experiments.symbolic_ai_v2.ctkg.inference.coverage import score_coverage

_PARADIGM_SHIFT_TYPE = "PARADIGM_SHIFT"
_DEFAULT_MIN_COVERAGE = 0.5


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class ParadigmShiftResult:
    """The result of a paradigm shift proposal.

    Attributes
    ----------
    old_theory_id    : the theory whose morphisms could not absorb the anomaly.
    new_theory_id    : the fresh theory cluster created for the anomaly class.
    bridge_morph_id  : MorphId of the PARADIGM_SHIFT morphism (old → new).
    explanation      : the LatentHypothesis fitted within the new theory.
                       None if the new theory uses a simpler representation.
    anomaly_coverage : fraction of anomaly sets explained by the new theory.
    wired_morphisms  : MorphIds of WIRED_TO morphisms added via wires_to param.
    """
    old_theory_id:    TheoryId
    new_theory_id:    TheoryId
    bridge_morph_id:  MorphId
    explanation:      Optional[LatentHypothesis]
    anomaly_coverage: float
    wired_morphisms:  list[MorphId] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def propose_paradigm_shift(
    theory_id:       TheoryId,
    anomaly_sets:    list[list[tuple[dict, float]]],
    schema_g_list:   list[SchematicLaw],
    schema_h:        SchematicLaw,
    ctx:             EvalContext,
    mg:              MorphismGraph,
    tm:              TheoryManager,
    new_theory_name: str   = "__paradigm__",
    label_prefix:    str   = "__ps__",
    min_coverage:    float = _DEFAULT_MIN_COVERAGE,
    tolerance:       float = 0.10,
    mdl_per_param:   float = 2.0,
    wires_to:        Optional[list[MorphId]] = None,
) -> Optional[ParadigmShiftResult]:
    """Propose a new theory cluster for an irreconcilable anomaly.

    Algorithm
    ---------
    1. Create a new theory node new_theory_id (named new_theory_name).
    2. Combine all anomaly sets into one training corpus.
    3. For each schema_g candidate, fit a LatentHypothesis within new_theory_id.
    4. Select the hypothesis with lowest MDL (all candidates have the same
       training data so coverage is expected to be equal — MDL tiebreak).
    5. Score coverage of best hypothesis over the individual anomaly_sets.
    6. If coverage ≥ min_coverage, store as PARADIGM_SHIFT bridge morphism
       (source=theory_id, target=new_theory_id) and return result.
    7. Otherwise return None.

    Key invariant
    -------------
    The old theory_id is NEVER modified.  No morphisms are added to it.
    The anomaly explanation lives exclusively in new_theory_id.  The bridge
    morphism is the only connection between the two theory clusters.

    Parameters
    ----------
    theory_id        : the theory that could not absorb the anomaly.
    anomaly_sets     : list of anomaly sets (each a list of (input, obs) pairs).
    schema_g_list    : candidate SchematicLaw templates for g (input → latent).
    schema_h         : SchematicLaw template for h (latent → output).
    ctx              : EvalContext.
    mg, tm           : graph and theory manager.
    new_theory_name  : name for the new theory cluster.
    label_prefix     : prefix for morphism labels in the new theory.
    min_coverage     : minimum coverage fraction for the shift to be proposed.
    tolerance        : relative error threshold for coverage scoring.
    mdl_per_param    : MDL cost per free parameter.

    Returns
    -------
    ParadigmShiftResult, or None if no adequate hypothesis can be fitted.
    """
    if not anomaly_sets:
        return None

    combined: list[tuple[dict, float]] = []
    for obs_set in anomaly_sets:
        combined.extend(obs_set)
    if not combined:
        return None

    # Step 1: create new theory cluster — NO morphisms added to old theory
    new_theory_id = tm.register_theory(new_theory_name)

    # Step 2 + 3: fit candidates in the NEW theory
    best_hyp: Optional[LatentHypothesis] = None
    best_mdl  = float("inf")

    for i, schema_g in enumerate(schema_g_list):
        hyp = hypothesise_latent(
            combined, schema_g, schema_h, ctx, mg, tm, new_theory_id,
            label_prefix=f"{label_prefix}_{i}",
            mdl_per_param=mdl_per_param,
        )
        if hyp is None:
            continue
        if hyp.mdl_score < best_mdl:
            best_hyp = hyp
            best_mdl = hyp.mdl_score

    if best_hyp is None:
        return None

    # Step 4: score coverage
    cov = score_coverage(best_hyp, anomaly_sets, ctx, tolerance=tolerance)
    if cov.coverage < min_coverage:
        return None

    # Step 5: store PARADIGM_SHIFT bridge morphism (old → new)
    payload = (
        theory_id,
        new_theory_id,
        best_hyp.morph_id,
        cov.coverage,
    )
    bridge = mg.add_morphism(
        theory_id,
        new_theory_id,
        morph_type=_PARADIGM_SHIFT_TYPE,
        evidence=len(combined),
        morph_concept_id=new_theory_id,
        payload=payload,
    )

    # Wire new concept node to specified morphisms from related theories
    wired: list[MorphId] = []
    if wires_to:
        for target_mid in wires_to:
            wm = mg.add_morphism(
                new_theory_id,
                target_mid,
                morph_type="WIRED_TO",
                evidence=1,
                morph_concept_id=new_theory_id,
                payload={"wired_source": theory_id, "wired_target": target_mid},
            )
            wired.append(wm.morph_id)

    return ParadigmShiftResult(
        old_theory_id=theory_id,
        new_theory_id=new_theory_id,
        bridge_morph_id=bridge.morph_id,
        explanation=best_hyp,
        anomaly_coverage=cov.coverage,
        wired_morphisms=wired,
    )


# ---------------------------------------------------------------------------
# Graph query
# ---------------------------------------------------------------------------

def query_paradigm_shifts(
    mg:        MorphismGraph,
    theory_id: TheoryId,
) -> list[ParadigmShiftResult]:
    """Return all PARADIGM_SHIFT morphisms originating from theory_id."""
    result: list[ParadigmShiftResult] = []

    for m in mg.source_morphisms(theory_id, morph_type=_PARADIGM_SHIFT_TYPE):
        old_id, new_id, hyp_morph_id, coverage = m.payload

        # Retrieve the latent hypothesis
        lm = mg.morphism_by_id(hyp_morph_id)
        if lm is None:
            continue

        try:
            lat_id, mid_g, mid_h, residual, mdl_score = lm.payload
        except (TypeError, ValueError):
            continue

        m_g = mg.morphism_by_id(mid_g)
        m_h = mg.morphism_by_id(mid_h)
        if m_g is None or m_h is None:
            continue

        hyp = LatentHypothesis(
            latent_id=lat_id,
            input_law=m_g.payload,
            output_law=m_h.payload,
            residual=residual,
            mdl_score=mdl_score,
            morph_id=hyp_morph_id,
        )
        result.append(ParadigmShiftResult(
            old_theory_id=old_id,
            new_theory_id=new_id,
            bridge_morph_id=m.morph_id,
            explanation=hyp,
            anomaly_coverage=coverage,
        ))

    return result
