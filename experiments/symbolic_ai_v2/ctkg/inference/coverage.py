"""
Multi-Anomaly Coverage Abduction — Phase 8 of the Einstein Roadmap.

Implements A-6: given multiple anomaly sets that might share a single explanation,
find the latent hypothesis that covers the largest fraction of anomalies.

Design
------
Problem statement
    We have K anomaly sets A_1 … A_K, each a list of (input, observed) pairs.
    We suspect there is a single latent generating process f = h ∘ g that
    explains most of them.  A minority (noise, different regime) may not be
    explained.

    Goal: find the hypothesis H* = argmax_{H} coverage(H, {A_i})
    where coverage(H, {A_i}) = |{i : H explains A_i}| / K.

    Tiebreak: among hypotheses with equal coverage, prefer lower MDL score.

Key invariant (Iron Law)
    No string names.  EvalContext keys are NodeIds (opaque ints).  The
    algorithm never inspects or compares token strings.

Defect probe
    A naive system fits one hypothesis per anomaly set (O(K) hypotheses, no
    shared explanation).  The correct system:
    1. Combines all anomaly sets into a single training corpus.
    2. Fits one hypothesis per schema_g candidate.
    3. Scores each hypothesis against individual anomaly sets.
    4. Returns the single hypothesis with highest coverage.

    This single-hypothesis constraint is what distinguishes abduction (find
    the common cause) from mere curve-fitting (explain each anomaly
    independently).

Types
-----
CoverageResult
    coverage : fraction of anomaly sets explained by hypothesis H (0.0–1.0).
    n_explained : number of anomaly sets with relative error ≤ tolerance.
    residuals_per_set : per-set mean relative error.

CoverageReport
    Full report: hypothesis + coverage result + whether it was selected.

Functions
---------
score_coverage(hypothesis, anomaly_sets, ctx, tolerance)
    Score a single hypothesis against K anomaly sets.

select_best_covering_hypothesis(hypotheses, anomaly_sets, ctx, tolerance)
    Given pre-fitted hypotheses, return the one with best coverage.
    Tiebreak: lowest MDL score.

multi_anomaly_abduction(anomaly_sets, schema_g_list, schema_h, ctx,
                         mg, tm, theory_id, tolerance, label_prefix)
    Full pipeline:
    1. Combine all anomaly sets → combined training corpus.
    2. For each schema_g candidate, call hypothesise_latent on combined corpus.
    3. Score each hypothesis for coverage over individual anomaly sets.
    4. Return best-coverage hypothesis (MDL tiebreak).

    The selected hypothesis is stored in the graph as a COVERAGE_ABDUCTION
    morphism so it can be retrieved via query_coverage_abductions().

query_coverage_abductions(mg, theory_id)
    Return all COVERAGE_ABDUCTION morphisms stored on theory_id.
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
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import (
    FittedLaw,
    predict_continuous,
)
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.inference.latent import (
    LatentHypothesis,
    hypothesise_latent,
)
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager, TheoryId

_COVERAGE_MORPH_TYPE = "COVERAGE_ABDUCTION"
_DEFAULT_TOLERANCE   = 0.10   # relative error threshold for "explained"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class CoverageResult:
    """Per-set coverage measurement for one latent hypothesis.

    Attributes
    ----------
    n_explained      : number of anomaly sets explained (rel-err ≤ tolerance).
    n_total          : total number of anomaly sets.
    coverage         : n_explained / n_total (0.0 – 1.0).
    residuals_per_set: mean relative error for each set (length = n_total).
    tolerance        : the relative-error threshold used.
    """
    n_explained:       int
    n_total:           int
    coverage:          float
    residuals_per_set: list[float]
    tolerance:         float


@dataclass
class CoverageReport:
    """Full report pairing a hypothesis with its coverage result.

    Attributes
    ----------
    hypothesis : the latent hypothesis being evaluated.
    coverage   : CoverageResult for this hypothesis.
    selected   : True iff this hypothesis was chosen by the selector.
    morph_id   : MorphId of the COVERAGE_ABDUCTION morphism (-1 = not stored).
    """
    hypothesis: LatentHypothesis
    coverage:   CoverageResult
    selected:   bool = False
    morph_id:   MorphId = -1


# ---------------------------------------------------------------------------
# Coverage scoring
# ---------------------------------------------------------------------------

def score_coverage(
    hypothesis:   LatentHypothesis,
    anomaly_sets: list[list[tuple[dict, float]]],
    ctx:          EvalContext,
    tolerance:    float = _DEFAULT_TOLERANCE,
) -> CoverageResult:
    """Score coverage of *hypothesis* over *anomaly_sets*.

    For each anomaly set, compute the mean relative prediction error of
    h(g(input)) vs observed.  An anomaly set is "explained" if the mean
    relative error is ≤ tolerance.

    Parameters
    ----------
    hypothesis   : a fitted LatentHypothesis (has .input_law and .output_law).
    anomaly_sets : list of anomaly sets; each set is a list of (input, obs).
    ctx          : EvalContext.
    tolerance    : maximum mean relative error to consider an anomaly "explained".

    Returns
    -------
    CoverageResult.
    """
    n_explained   = 0
    residuals: list[float] = []

    for obs_set in anomaly_sets:
        if not obs_set:
            residuals.append(float("inf"))
            continue

        errors: list[float] = []
        for inp, obs in obs_set:
            try:
                z    = predict_continuous(hypothesis.input_law,  inp,        ctx)
                pred = predict_continuous(hypothesis.output_law, {"z": z},   ctx)
                denom = abs(obs) + 1e-12
                errors.append(abs(pred - obs) / denom)
            except (ValueError, KeyError, TypeError):
                errors.append(float("inf"))

        mean_err = float(np.mean(errors)) if errors else float("inf")
        residuals.append(mean_err)
        if mean_err <= tolerance:
            n_explained += 1

    n_total  = len(anomaly_sets)
    coverage = n_explained / max(n_total, 1)

    return CoverageResult(
        n_explained=n_explained,
        n_total=n_total,
        coverage=coverage,
        residuals_per_set=residuals,
        tolerance=tolerance,
    )


# ---------------------------------------------------------------------------
# Best-hypothesis selector
# ---------------------------------------------------------------------------

def select_best_covering_hypothesis(
    hypotheses:   list[LatentHypothesis],
    anomaly_sets: list[list[tuple[dict, float]]],
    ctx:          EvalContext,
    tolerance:    float = _DEFAULT_TOLERANCE,
) -> Optional[LatentHypothesis]:
    """Return the hypothesis that covers the most anomaly sets.

    Among hypotheses tied on n_explained, the one with the lower MDL score
    is preferred (Occam's razor).

    Parameters
    ----------
    hypotheses   : list of LatentHypothesis objects to compare.
    anomaly_sets : list of anomaly sets; each is a list of (input, obs) pairs.
    ctx          : EvalContext.
    tolerance    : relative error threshold for "explained".

    Returns
    -------
    The best LatentHypothesis, or None if the list is empty.
    """
    best_hyp: Optional[LatentHypothesis] = None
    best_n:   int   = -1
    best_mdl: float = float("inf")

    for hyp in hypotheses:
        cov = score_coverage(hyp, anomaly_sets, ctx, tolerance=tolerance)
        if (cov.n_explained > best_n or
                (cov.n_explained == best_n and hyp.mdl_score < best_mdl)):
            best_hyp  = hyp
            best_n    = cov.n_explained
            best_mdl  = hyp.mdl_score

    return best_hyp


# ---------------------------------------------------------------------------
# Full multi-anomaly abduction pipeline
# ---------------------------------------------------------------------------

def multi_anomaly_abduction(
    anomaly_sets:  list[list[tuple[dict, float]]],
    schema_g_list: list[SchematicLaw],
    schema_h:      SchematicLaw,
    ctx:           EvalContext,
    mg:            MorphismGraph,
    tm:            TheoryManager,
    theory_id:     TheoryId,
    tolerance:     float = _DEFAULT_TOLERANCE,
    label_prefix:  str   = "__multi_abd__",
    mdl_per_param: float = 2.0,
) -> Optional[LatentHypothesis]:
    """Fit a single latent hypothesis that covers the most anomaly sets.

    Algorithm
    ---------
    1. For each (schema_g, anomaly_set_i) pair, fit a candidate latent
       hypothesis on that individual anomaly set.  This gives at most
       |schema_g_list| × |anomaly_sets| candidate hypotheses.
    2. Score each candidate against ALL anomaly sets.
    3. Return the single candidate with highest coverage (MDL tiebreak).
    4. Store it as a COVERAGE_ABDUCTION morphism on theory_id.

    Rationale for fitting per-set rather than on the combined corpus
    ----------------------------------------------------------------
    When anomaly sets have different generating processes (e.g. 6*x vs 10*x),
    combining them before fitting yields an averaged hypothesis (≈7*x) that
    explains none of the sets well.  Fitting on each set individually and then
    scoring coverage against all sets lets the winning generating process
    "vote itself in" by explaining the most sets.

    Key invariant
    -------------
    Only ONE hypothesis is returned (the shared explanation).  A system that
    returns K hypotheses (one per anomaly set) fails the defect probe.

    Parameters
    ----------
    anomaly_sets  : K anomaly sets; each is a list of (input_bindings, output).
    schema_g_list : candidate SchematicLaw templates for g (input → latent).
    schema_h      : SchematicLaw template for h (latent → output).
    ctx           : EvalContext.
    mg, tm        : graph and theory manager.
    theory_id     : theory to attach the hypothesis to.
    tolerance     : relative error threshold for "explained" (default 0.10).
    label_prefix  : prefix for morphism labels.
    mdl_per_param : MDL cost per free parameter.

    Returns
    -------
    The best LatentHypothesis (or None if no hypothesis could be fitted).
    """
    if not anomaly_sets:
        return None

    # Step 1: fit one hypothesis per (schema_g, anomaly_set) pair
    fitted: list[LatentHypothesis] = []
    for i, schema_g in enumerate(schema_g_list):
        for j, obs_set in enumerate(anomaly_sets):
            if not obs_set:
                continue
            hyp = hypothesise_latent(
                obs_set, schema_g, schema_h, ctx, mg, tm, theory_id,
                label_prefix=f"{label_prefix}_{i}_{j}",
                mdl_per_param=mdl_per_param,
            )
            if hyp is not None:
                fitted.append(hyp)

    if not fitted:
        return None

    # Step 2 + 3: select best by coverage across all anomaly sets (MDL tiebreak)
    best = select_best_covering_hypothesis(
        fitted, anomaly_sets, ctx, tolerance=tolerance,
    )
    if best is None:
        return None

    # Step 5: store as COVERAGE_ABDUCTION self-loop on theory
    cov = score_coverage(best, anomaly_sets, ctx, tolerance=tolerance)
    payload = (
        best.latent_id,
        best.morph_id,
        cov.n_explained,
        cov.n_total,
        cov.coverage,
    )
    morph = mg.add_morphism(
        theory_id,
        theory_id,
        morph_type=_COVERAGE_MORPH_TYPE,
        evidence=sum(len(s) for s in anomaly_sets),
        morph_concept_id=best.latent_id,
        payload=payload,
    )
    _ = morph  # stored; callers can query via query_coverage_abductions()

    return best


# ---------------------------------------------------------------------------
# Graph query
# ---------------------------------------------------------------------------

def query_coverage_abductions(
    mg:        MorphismGraph,
    theory_id: TheoryId,
) -> list[CoverageReport]:
    """Return all COVERAGE_ABDUCTION morphisms stored on theory_id.

    Returns a list of CoverageReport objects (without re-scoring; uses
    the scores saved in the morphism payload).
    """
    result: list[CoverageReport] = []

    for m in mg.source_morphisms(theory_id, morph_type=_COVERAGE_MORPH_TYPE):
        latent_id, latent_morph_id, n_explained, n_total, coverage = m.payload

        # Reconstruct a minimal CoverageResult from stored values
        cov = CoverageResult(
            n_explained=n_explained,
            n_total=n_total,
            coverage=coverage,
            residuals_per_set=[],   # not stored; re-score if needed
            tolerance=_DEFAULT_TOLERANCE,
        )

        # Retrieve the underlying LatentHypothesis morphism
        lm = mg.morphism_by_id(latent_morph_id)
        if lm is None or lm.payload is None:
            continue

        # lm.payload is a LatentHypothesis payload tuple stored by latent.py:
        # (latent_id, mid_g, mid_h, residual, mdl_score)
        try:
            lat_id2, mid_g, mid_h, residual, mdl_score = lm.payload
        except (TypeError, ValueError):
            continue

        m_g = mg.morphism_by_id(mid_g)
        m_h = mg.morphism_by_id(mid_h)
        if m_g is None or m_h is None:
            continue

        hyp = LatentHypothesis(
            latent_id=lat_id2,
            input_law=m_g.payload,
            output_law=m_h.payload,
            residual=residual,
            mdl_score=mdl_score,
            morph_id=latent_morph_id,
        )
        result.append(CoverageReport(hypothesis=hyp, coverage=cov, selected=True))

    return result
