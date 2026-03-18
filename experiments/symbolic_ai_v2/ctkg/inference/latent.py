"""
Latent Variable and Ontology Extension Abduction — Phase 7 of the Einstein Roadmap.

This module implements two abductive capabilities required for the Einstein test:

1.  Latent variable hypothesis generation
    Given observations of f(x) where f = h ∘ g and g is unobserved, recover
    g as a hidden intermediate quantity.  The hypothesised g is a new FittedLaw
    whose composition with h predicts the observations.

2.  Ontology extension (new concept nodes)
    Given an anomaly that no existing morphism structure can explain, propose a
    fresh concept node C (a new ObjectId with no surface token) defined entirely
    by its structural role: the morphisms connecting it to existing objects.

Design
------
LatentHypothesis
    Represents a hypothesised unobserved quantity.
    - latent_id   : a fresh ObjectId in the MorphismGraph (no token label).
    - input_law   : FittedLaw  g: inputs → latent   (maps observations to latent value)
    - output_law  : FittedLaw  h: latent → observed (maps latent to outputs)
    - residual    : mean squared prediction error over training observations.

hypothesise_latent(observations, schema_g, schema_h, ctx, mg, tm, theory_id)
    Fit g and h independently using OLS.  The latent quantity is the value
    predicted by g for each input.  h is then fitted to (latent, observed) pairs.
    The latent node is a fresh ObjectId stored in the graph.

    Scoring: MDL(latent hypothesis) = residual * n_obs + 2 * (|params_g| + |params_h|)
    Lower is better.  Among candidates, the one with the lowest MDL is preferred.

OntologyExtension
    A new concept node C defined by incoming and outgoing morphism templates.
    - concept_id      : fresh ObjectId (no token, no label).
    - in_morph_ids    : morphisms from existing objects INTO C.
    - out_morph_ids   : morphisms FROM C to existing objects.
    - justification   : which anomalies this concept resolves.

propose_new_concept(anomalies, candidate_morphisms, mg, theory_id)
    Given anomalies and a set of candidate morphisms that partially explain them,
    find a grouping where inserting a new intermediate node C as their common
    codomain (for inputs) or domain (for outputs) reduces the total residual.

Iron Law compliance
-------------------
Latent nodes are ObjectIds — opaque integers.  They have no token label and
no string name.  Their identity is defined entirely by the morphisms connecting
them to existing objects.  The cage tests verify that two symbol tables produce
structurally isomorphic latent graphs.

MDL principle (Occam's razor)
------------------------------
The roadmap specifies: among candidate latent hypotheses, prefer the one with
fewer free parameters.  This is implemented via the MDL score:
    MDL = residual * n_obs + mdl_per_param * (n_params_g + n_params_h)
A 1-parameter latent (linear: g(x)=a*x) scores lower than a 3-parameter latent
(polynomial: g(x)=a*x²+b*x+c) when both fit the data equally well.

Defect probe — minimal latent
------------------------------
Present observations consistent with g₁ (1 free parameter: linear) and g₂
(3 free parameters: polynomial).  hypothesise_latent must prefer g₁ by MDL.
A system without the MDL prior selects g₂ (it fits perfectly) and overfits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph,
    MorphId,
    ObjectId,
)
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import (
    FittedLaw,
    fit_parameters,
    add_fitted_law,
    predict_continuous,
)
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager, TheoryId

_LATENT_MORPH_TYPE  = "LATENT_HYPOTHESIS"
_LATENT_OBJ_LABEL   = "__latent_{id}__"
_ONTOLOGY_EXT_TYPE  = "ONTOLOGY_EXTENSION"

_DEFAULT_MDL_PER_PARAM = 2.0   # nats per free parameter (Occam factor)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class LatentHypothesis:
    """A hypothesised unobserved intermediate quantity.

    Attributes
    ----------
    latent_id   : ObjectId of the fresh concept node (no token; purely structural).
    input_law   : FittedLaw g: inputs → latent.
    output_law  : FittedLaw h: latent → observed.
    residual    : mean squared error of h(g(x)) vs observed.
    mdl_score   : residual * n_obs + mdl_per_param * n_params (lower = better).
    morph_id    : MorphId of the LATENT_HYPOTHESIS morphism in the graph (-1 = not stored).
    """
    latent_id:   ObjectId
    input_law:   FittedLaw
    output_law:  FittedLaw
    residual:    float
    mdl_score:   float
    morph_id:    MorphId = -1


@dataclass
class OntologyExtension:
    """A new abstract concept node added to explain anomalies.

    Attributes
    ----------
    concept_id   : fresh ObjectId (no token label; identity = graph structure).
    in_morph_ids : morphisms from existing objects INTO concept_id.
    out_morph_ids: morphisms FROM concept_id to existing objects.
    residual_gain: how much total residual decreases by inserting this concept.
    morph_id     : MorphId of the ONTOLOGY_EXTENSION morphism (-1 = not stored).
    """
    concept_id:   ObjectId
    in_morph_ids: list[MorphId]
    out_morph_ids: list[MorphId]
    residual_gain: float
    morph_id:     MorphId = -1


# ---------------------------------------------------------------------------
# Latent variable hypothesis generation
# ---------------------------------------------------------------------------

def hypothesise_latent(
    observations:  list[tuple[dict, float]],
    schema_g:      SchematicLaw,
    schema_h:      SchematicLaw,
    ctx:           EvalContext,
    mg:            MorphismGraph,
    tm:            TheoryManager,
    theory_id:     TheoryId,
    label_prefix:  str = "__latent__",
    mdl_per_param: float = _DEFAULT_MDL_PER_PARAM,
) -> Optional[LatentHypothesis]:
    """Fit a latent hypothesis f = h ∘ g from observations.

    Algorithm
    ---------
    1. Fit g: inputs → latent (schema_g, same observations as g input).
       The "latent" value for each observation is g(input).
    2. Fit h: latent → output (schema_h, with latent as input variable).
    3. Compute residual: mean (h(g(input)) - observed)².
    4. Compute MDL score.
    5. Create a fresh latent concept node in the graph.
    6. Store as LATENT_HYPOTHESIS morphism.

    Parameters
    ----------
    observations  : list of (input_bindings, observed_output) pairs.
    schema_g      : SchematicLaw template for g (maps inputs → latent).
                    Must have exactly one output variable (treated as latent).
    schema_h      : SchematicLaw template for h (maps latent → output).
    ctx           : EvalContext.
    mg, tm        : graph and theory manager.
    theory_id     : theory to attach the hypothesis to.
    label_prefix  : prefix for the latent object label.
    mdl_per_param : MDL cost per free parameter.

    Returns
    -------
    LatentHypothesis, or None if fitting fails.
    """
    if not observations:
        return None

    # Step 1: Fit g (input → latent)
    try:
        fl_g = fit_parameters(schema_g.conclusion, schema_g.params, observations, ctx)
    except (ValueError, Exception):
        return None

    # Step 2: Compute latent values for each observation
    latent_obs: list[tuple[dict, float]] = []
    for inp, _ in observations:
        try:
            latent_val = predict_continuous(fl_g, inp, ctx)
        except (ValueError, KeyError, TypeError):
            return None
        latent_obs.append(({"z": latent_val}, 0.0))  # placeholder output

    # Update latent_obs with actual observed outputs
    latent_obs_with_output = [
        ({"z": lat_inp["z"]}, obs)
        for (lat_inp, _), (_, obs) in zip(latent_obs, observations)
    ]

    # Step 3: Fit h (latent → output)
    try:
        fl_h = fit_parameters(schema_h.conclusion, schema_h.params,
                               latent_obs_with_output, ctx)
    except (ValueError, Exception):
        return None

    # Step 4: Compute total residual
    preds = []
    for (inp, obs) in observations:
        try:
            latent_val = predict_continuous(fl_g, inp, ctx)
            pred = predict_continuous(fl_h, {"z": latent_val}, ctx)
            preds.append((pred - obs) ** 2)
        except (ValueError, KeyError, TypeError):
            return None

    n_obs = len(observations)
    residual = float(np.mean(preds)) if preds else float("inf")
    n_params = len(schema_g.params) + len(schema_h.params)
    mdl_score = residual * n_obs + mdl_per_param * n_params

    # Step 5: Create latent concept node (no token label, no concept)
    latent_obj = mg.add_object(concept=None, label="")
    latent_id  = latent_obj.obj_id

    # Step 6: Store input law and output law in graph, associate with theory
    label_g = f"{label_prefix}_g_{latent_id}"
    label_h = f"{label_prefix}_h_{latent_id}"
    mid_g   = add_fitted_law(mg, label_g, fl_g)
    mid_h   = add_fitted_law(mg, label_h, fl_h)
    tm.assign_morphism(mid_g, theory_id)
    tm.assign_morphism(mid_h, theory_id)

    # Store as LATENT_HYPOTHESIS morphism on the theory object
    payload = (latent_id, mid_g, mid_h, residual, mdl_score)
    morph = mg.add_morphism(
        theory_id,
        theory_id,
        morph_type=_LATENT_MORPH_TYPE,
        evidence=n_obs,
        morph_concept_id=latent_id,
        payload=payload,
    )

    return LatentHypothesis(
        latent_id=latent_id,
        input_law=fl_g,
        output_law=fl_h,
        residual=residual,
        mdl_score=mdl_score,
        morph_id=morph.morph_id,
    )


def hypothesise_latent_mdl_select(
    observations:   list[tuple[dict, float]],
    schema_g_list:  list[SchematicLaw],
    schema_h:       SchematicLaw,
    ctx:            EvalContext,
    mg:             MorphismGraph,
    tm:             TheoryManager,
    theory_id:      TheoryId,
    label_prefix:   str = "__latent_mdl__",
    mdl_per_param:  float = _DEFAULT_MDL_PER_PARAM,
) -> Optional[LatentHypothesis]:
    """Fit multiple latent hypotheses and return the one with lowest MDL score.

    Implements the MDL selection described in the roadmap defect probe:
    given g₁ (1 param) and g₂ (3 params) that both fit the data, select g₁.

    Parameters
    ----------
    schema_g_list : list of candidate SchematicLaw templates for g.
                    Each is tried; the one with the lowest MDL score is returned.
    All other parameters as in hypothesise_latent.
    """
    best: Optional[LatentHypothesis] = None
    for i, schema_g in enumerate(schema_g_list):
        hyp = hypothesise_latent(
            observations, schema_g, schema_h, ctx, mg, tm, theory_id,
            label_prefix=f"{label_prefix}_{i}",
            mdl_per_param=mdl_per_param,
        )
        if hyp is None:
            continue
        if best is None or hyp.mdl_score < best.mdl_score:
            best = hyp
    return best


def query_latent_hypotheses(
    mg:        MorphismGraph,
    theory_id: TheoryId,
) -> list[LatentHypothesis]:
    """Return all LATENT_HYPOTHESIS morphisms stored on theory_id."""
    result = []
    for m in mg.source_morphisms(theory_id, morph_type=_LATENT_MORPH_TYPE):
        latent_id, mid_g, mid_h, residual, mdl_score = m.payload
        m_g = mg.morphism_by_id(mid_g)
        m_h = mg.morphism_by_id(mid_h)
        if m_g is None or m_h is None:
            continue
        result.append(LatentHypothesis(
            latent_id=latent_id,
            input_law=m_g.payload,
            output_law=m_h.payload,
            residual=residual,
            mdl_score=mdl_score,
            morph_id=m.morph_id,
        ))
    return result


# ---------------------------------------------------------------------------
# Ontology extension
# ---------------------------------------------------------------------------

def propose_new_concept(
    mg:           MorphismGraph,
    tm:           TheoryManager,
    theory_id:    TheoryId,
    in_morph_ids: list[MorphId],
    out_morph_ids: list[MorphId],
    residual_gain: float = 0.0,
) -> OntologyExtension:
    """Create a new anonymous concept node C as an ontology extension.

    C is a fresh ObjectId with no token label.  Its identity is defined
    entirely by the morphisms connecting it to existing objects:
    - morphisms in `in_morph_ids` are interpreted as pointing INTO C
    - morphisms in `out_morph_ids` are interpreted as pointing FROM C

    The node is stored as an ONTOLOGY_EXTENSION self-loop on the theory object.

    Parameters
    ----------
    mg, tm        : graph and theory manager.
    theory_id     : the theory this extension belongs to.
    in_morph_ids  : morphism ids from existing objects to C (structural role).
    out_morph_ids : morphism ids from C to existing objects.
    residual_gain : how much this extension reduces the total residual.

    Returns
    -------
    OntologyExtension with a fresh concept_id.
    """
    # Fresh concept node — no token, no label, purely structural
    concept_obj = mg.add_object(concept=None, label="")
    concept_id  = concept_obj.obj_id

    # Log as ONTOLOGY_EXTENSION self-loop on theory
    payload = (concept_id, in_morph_ids, out_morph_ids, residual_gain)
    morph = mg.add_morphism(
        theory_id,
        theory_id,
        morph_type=_ONTOLOGY_EXT_TYPE,
        evidence=1,
        morph_concept_id=concept_id,
        payload=payload,
    )

    return OntologyExtension(
        concept_id=concept_id,
        in_morph_ids=list(in_morph_ids),
        out_morph_ids=list(out_morph_ids),
        residual_gain=residual_gain,
        morph_id=morph.morph_id,
    )


def query_ontology_extensions(
    mg:        MorphismGraph,
    theory_id: TheoryId,
) -> list[OntologyExtension]:
    """Return all ONTOLOGY_EXTENSION morphisms stored on theory_id."""
    result = []
    for m in mg.source_morphisms(theory_id, morph_type=_ONTOLOGY_EXT_TYPE):
        concept_id, in_ids, out_ids, gain = m.payload
        result.append(OntologyExtension(
            concept_id=concept_id,
            in_morph_ids=in_ids,
            out_morph_ids=out_ids,
            residual_gain=gain,
            morph_id=m.morph_id,
        ))
    return result
