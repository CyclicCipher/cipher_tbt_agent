"""
Parameter fitting for SchematicLaws — Phase 3 of the Einstein Roadmap.

FittedLaw
---------
A SchematicLaw whose free parameters have been fitted to numerical observations.
Carries the fitted float values, a mean-squared-error residual, and a
MorphismGraph morphism id for persistence.

fit_parameters
--------------
Given:
  - conclusion: Expr  — the formula for the output (with var() nodes for both
    parameters and input variables)
  - param_names: frozenset[str]  — which var names are the free parameters
  - observations: list[(input_bindings: dict[str, float], output: float)]
  - ctx: EvalContext  — operator dispatch table (NodeId → callable)

Finds parameter values that minimise the mean squared residual.

Algorithm: builds the OLS design matrix A where A[i, j] = eval(formula,
{inputs_i, p_j=1.0, all_other_params=0.0}, ctx).  This is exact for formulas
that are LINEAR in each parameter independently (e.g. k*x, k1*x + k2*y).
For Phase 3's target laws (scaling, F=ma, Hooke, Ohm) all parameters are linear.

Iron Law compliance
-------------------
No string comparisons on parameter names inside fit_parameters.  Parameter
names are used only as dict keys for the bindings dict (which is the public
API boundary where str↔NodeId conversion is appropriate).

Bitter Lesson compliance
------------------------
The formula Expr uses NodeId for all operators.  A formula with anonymous
operator '⊕' evaluates identically to one with 'mul' if the EvalContext maps
both to the same callable.  The cage tests verify this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph, MorphId
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext, eval_expr
from experiments.symbolic_ai_v2.ctkg.core.expr_law import _ensure_object, _find_object

_FITTED_LAW_TYPE = "FITTED_LAW"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class FittedLaw:
    """A SchematicLaw with numerically fitted parameter values.

    Attributes
    ----------
    schema   : the structural SchematicLaw this was fitted from.
    params   : dict mapping parameter var-names → fitted float values.
    residual : mean squared error over training observations.
    morph_id : MorphismGraph morphism id (-1 = not stored in any graph).
    """
    schema:   SchematicLaw
    params:   dict[str, float]
    residual: float
    morph_id: MorphId = -1


# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------

def fit_parameters(
    conclusion:   object,          # Expr — the formula to fit
    param_names:  frozenset,
    observations: list,            # list[(dict[str,float], float)]
    ctx:          EvalContext,
) -> FittedLaw:
    """Fit parameter values to numerical observations using OLS.

    Constructs the OLS design matrix A where:
        A[i, j] = eval_expr(conclusion, {**inputs_i, p_j: 1.0,
                                          other_params: 0.0}, ctx)

    This gives the contribution of parameter p_j to the output at observation i,
    assuming the formula is linear in each parameter independently.
    Then fits params via least-squares: A @ params_vector ≈ targets.

    Parameters
    ----------
    conclusion   : Expr formula for the output.
    param_names  : frozenset of var-name strings that are the free parameters.
    observations : list of (input_bindings_dict, target_float) pairs.
    ctx          : EvalContext supplying operator callables.

    Returns
    -------
    FittedLaw with fitted params and MSE residual.

    Raises
    ------
    ValueError if observations is empty.
    """
    if not observations:
        raise ValueError("fit_parameters: observations list is empty")

    param_list = sorted(param_names)   # deterministic column order
    n_params = len(param_list)
    n_obs    = len(observations)

    targets = np.array([tgt for _, tgt in observations], dtype=float)

    if n_params == 0:
        # No free parameters: just evaluate and compute residual
        preds = np.array(
            [eval_expr(conclusion, inp, ctx) for inp, _ in observations],
            dtype=float,
        )
        mse = float(np.mean((preds - targets) ** 2))
        # Build a dummy schema-free FittedLaw
        from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
        from experiments.symbolic_ai_v2.ctkg.core.term_algebra import var as _var
        dummy_schema = SchematicLaw(
            pattern=conclusion, conclusion=conclusion,
            params=frozenset(), variables=frozenset(), evidence=n_obs,
        )
        return FittedLaw(schema=dummy_schema, params={}, residual=mse)

    # Build the OLS design matrix
    # Column j: evaluate formula with p_j = 1.0 and all other params = 0.0
    A = np.zeros((n_obs, n_params), dtype=float)
    for j, p in enumerate(param_list):
        unit_bindings = {q: (1.0 if q == p else 0.0) for q in param_list}
        for i, (inputs, _) in enumerate(observations):
            A[i, j] = eval_expr(conclusion, {**inputs, **unit_bindings}, ctx)

    # Least-squares solve
    fitted_vec, _, _, _ = np.linalg.lstsq(A, targets, rcond=None)
    fitted = {p: float(fitted_vec[j]) for j, p in enumerate(param_list)}

    # Compute MSE
    preds = A @ fitted_vec
    mse = float(np.mean((preds - targets) ** 2))

    # Build a minimal SchematicLaw wrapper (schema is passed in separately via
    # add_fitted_law; this stub satisfies the FittedLaw contract)
    from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
    stub_schema = SchematicLaw(
        pattern=conclusion, conclusion=conclusion,
        params=frozenset(param_names), variables=frozenset(), evidence=n_obs,
    )
    return FittedLaw(schema=stub_schema, params=fitted, residual=mse)


def fit_law(
    schema:      SchematicLaw,
    observations: list,           # list[(dict[str,float], float)]
    ctx:         EvalContext,
) -> FittedLaw:
    """Convenience wrapper: fit a SchematicLaw's parameters and return FittedLaw.

    Uses schema.conclusion as the formula and schema.params as the parameter names.

    Parameters
    ----------
    schema       : the SchematicLaw to fit.
    observations : list of (input_bindings_dict, target_float) pairs.
                   Keys must match the variable names in schema.variables and
                   the parameter names in schema.params.
    ctx          : EvalContext.

    Returns
    -------
    FittedLaw with schema set to the input schema.
    """
    fl = fit_parameters(schema.conclusion, schema.params, observations, ctx)
    return FittedLaw(
        schema=schema,
        params=fl.params,
        residual=fl.residual,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_continuous(
    fitted:        FittedLaw,
    input_bindings: dict,          # dict[str, float]
    ctx:           EvalContext,
) -> float:
    """Predict the output for new inputs using the fitted law.

    Substitutes the fitted parameter values and evaluates the formula.

    Parameters
    ----------
    fitted        : a FittedLaw with fitted.params and fitted.schema.conclusion.
    input_bindings: dict mapping input variable names → float values.
    ctx           : EvalContext.

    Returns
    -------
    float — predicted output value.
    """
    all_bindings = {**input_bindings, **fitted.params}
    return eval_expr(fitted.schema.conclusion, all_bindings, ctx)


# ---------------------------------------------------------------------------
# Graph storage
# ---------------------------------------------------------------------------

def add_fitted_law(
    mg:        MorphismGraph,
    law_label: str,
    law:       FittedLaw,
) -> MorphId:
    """Store a FittedLaw as a FITTED_LAW self-loop morphism in mg.

    Deduplicates by (schema.pattern, params) — if an identical law is already
    stored under law_label, returns the existing morphism id.
    """
    anchor_id = _ensure_object(mg, law_label)

    for m in mg.source_morphisms(anchor_id, morph_type=_FITTED_LAW_TYPE):
        stored: FittedLaw = m.payload
        if (stored.schema.pattern == law.schema.pattern
                and stored.params == law.params):
            return m.morph_id

    m = mg.add_morphism(
        anchor_id, anchor_id,
        morph_type=_FITTED_LAW_TYPE,
        evidence=1,
        payload=law,
    )
    return m.morph_id


def query_fitted_laws(
    mg:        MorphismGraph,
    law_label: str,
) -> list[FittedLaw]:
    """Return all FittedLaws stored under law_label."""
    anchor_id = _find_object(mg, law_label)
    if anchor_id is None:
        return []
    return [
        m.payload
        for m in mg.source_morphisms(anchor_id, morph_type=_FITTED_LAW_TYPE)
    ]
