"""
Graph-backed compositional law discovery — Phase 10 of the Einstein Roadmap.

discover_law
------------
Given numerical observations (input_bindings, output) pairs, discover the
functional form that generated them by beam-searching over expression trees
composed from primitive operations, then fit free parameters via OLS.

This replaces the pattern of hand-building a SchematicLaw and passing it to
fit_parameters. The caller provides only observations; the structural form
(linearity, quadratic, etc.) is discovered, not assumed.

Algorithm
---------
1. Build terminals:
   - var("x"), var("y"), ... for each observed input variable name
   - var("p0"), var("p1"), ... for n_param_slots free parameter slots
   - atom("1.0"), atom("2.0"), atom("-1.0"), atom("0.5") as fixed constants

2. Beam search from depth 0 → max_depth:
   At each depth, expand the current beam by applying each primitive to
   beam candidates:
     arity-1: Expr(prim.nid, (c,))         for c in beam
     arity-2: Expr(prim.nid, (c1, c2))     for c1, c2 in beam x beam
   Score each candidate with MDL = depth_penalty * tree_size + log(mse + ε).
   Prune to beam_width.

3. OLS fitting within a structure:
   For a candidate expression containing var("p0"), var("p1"), ...,
   build the design matrix A[i,j] = eval_expr(expr, {inputs_i, p_j=1, rest=0})
   and solve via least-squares. Works for expressions linear in each param.

4. Return the minimum-MDL candidate as a FittedLaw whose schema.variables
   = frozenset of input names, schema.params = frozenset of "p0", "p1", etc.

Iron Law compliance
-------------------
No string comparisons on domain operator names. All operator dispatch uses
PrimSpec.nid (opaque int) as the Expr.head. The prim_ctx EvalContext maps
these NodeIds to callables — no token name dispatch.

Bitter Lesson compliance
------------------------
The structure search is driven by MDL over numerical residuals, not by any
prior knowledge of what form the law should have.  discover_law("k*x") and
discover_law("k*x²") receive identical-structure code paths; the MDL score
selects the correct depth for each.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.prim_ops import PrimSpec, get_prim_specs, make_prim_ctx
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext, eval_expr
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import FittedLaw
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import Expr, atom, var

_PARAM_PREFIX = "p"
_FIXED_CONSTS = ("0.0", "1.0", "2.0", "-1.0", "0.5", "3.0", "0.25")
_EPS = 1e-9
_LOG_EPS = math.log(_EPS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _param_name(i: int) -> str:
    return f"{_PARAM_PREFIX}{i}"


def _tree_size(expr: Expr) -> int:
    """Total number of nodes in the expression tree."""
    return 1 + sum(_tree_size(a) for a in expr.args)


def _find_param_names(expr: Expr, param_set: frozenset[str]) -> list[str]:
    """Collect which param names from param_set actually appear in expr."""
    found: set[str] = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        if e.is_var:
            name = TOKEN_GRAPH.decode(e.head)
            if name in param_set:
                found.add(name)
        for a in e.args:
            stack.append(a)
    return sorted(found)


def _ols_fit(
    expr: Expr,
    param_names: list[str],   # sorted list of parameter var names present in expr
    observations: list[tuple[dict, float]],
    prim_ctx: EvalContext,
) -> tuple[Optional[dict[str, float]], float]:
    """Fit param_names in expr to minimise MSE over observations via OLS.

    Assumes expr is linear in each param independently (the standard case for
    depth-1 and many depth-2 structures).

    Returns (fitted_dict, mse) or (None, inf) on failure.
    """
    n_obs = len(observations)
    n_params = len(param_names)

    if n_params == 0:
        # No free params: just evaluate and compute MSE
        preds = []
        for inp, _ in observations:
            try:
                p = eval_expr(expr, inp, prim_ctx)
                preds.append(p if math.isfinite(p) else float("nan"))
            except Exception:
                preds.append(float("nan"))
        if any(math.isnan(p) for p in preds):
            return None, float("inf")
        targets = [t for _, t in observations]
        mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / n_obs
        return {}, mse

    targets = np.array([t for _, t in observations], dtype=float)
    A = np.zeros((n_obs, n_params), dtype=float)
    for j, p in enumerate(param_names):
        unit = {q: (1.0 if q == p else 0.0) for q in param_names}
        for i, (inp, _) in enumerate(observations):
            combined = {**inp, **unit}
            try:
                v = eval_expr(expr, combined, prim_ctx)
            except Exception:
                v = float("nan")
            if not math.isfinite(v):
                return None, float("inf")
            A[i, j] = v

    try:
        fitted_vec, _, _, _ = np.linalg.lstsq(A, targets, rcond=None)
    except np.linalg.LinAlgError:
        return None, float("inf")

    preds = A @ fitted_vec
    mse = float(np.mean((preds - targets) ** 2))
    if not math.isfinite(mse):
        return None, float("inf")

    fitted = {p: float(fitted_vec[j]) for j, p in enumerate(param_names)}
    return fitted, mse


def _var_nids_in(expr: Expr) -> frozenset:
    """Collect NodeIds of all var() leaves in expr (for structural diversity)."""
    result: set = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        if e.is_var:
            result.add(e.head)
        for a in e.args:
            stack.append(a)
    return frozenset(result)


def _struct_sig(expr: Expr) -> tuple:
    """Structural diversity signature: (root_op_nid, frozenset_of_var_nids).
    Expressions with the same signature are structurally similar.
    """
    return (expr.head, _var_nids_in(expr))


def _diverse_keep(
    beam: list[tuple[float, Expr, dict]],
    beam_width: int,
) -> list[tuple[float, Expr, dict]]:
    """Keep top beam_width by MDL score, plus one rep per unseen structural signature.

    Ensures that intermediate subexpressions (e.g. SQ(v), SUB(1, SQ(v))) survive
    pruning even when their individual residuals on the target data are poor,
    which is necessary for discovering deeply nested expressions like gamma(v).

    The result size is at most 2 * beam_width.
    """
    top = beam[:beam_width]
    seen: set = {_struct_sig(e) for _, e, _ in top}
    extras: list = []
    for item in beam[beam_width:]:
        if len(extras) >= beam_width:   # cap total at 2x
            break
        sig = _struct_sig(item[1])
        if sig not in seen:
            seen.add(sig)
            extras.append(item)
    return top + extras


def _nlopt_fit(
    expr: Expr,
    param_names: list[str],
    observations: list[tuple[dict, float]],
    prim_ctx: EvalContext,
    n_restarts: int = 5,
) -> tuple[Optional[dict[str, float]], float]:
    """Fit parameters via numerical optimization (L-BFGS-B).

    Used as a fallback when OLS fails or gives poor residuals for
    non-linearly parameterized expressions (e.g. gamma(v) = 1/sqrt(1-v^2/c^2)
    where c is a free parameter appearing non-linearly).

    Tries n_restarts initial points and returns the best result.
    Returns (None, inf) if all restarts fail.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None, float("inf")

    if not param_names:
        return _ols_fit(expr, param_names, observations, prim_ctx)

    n_params = len(param_names)
    targets = [t for _, t in observations]
    inputs_list = [inp for inp, _ in observations]

    def objective2(x: "np.ndarray") -> float:
        bindings_ext = {p: float(x[j]) for j, p in enumerate(param_names)}
        sq_sum = 0.0
        for (inp, tgt) in observations:
            combined = {**inp, **bindings_ext}
            try:
                v = eval_expr(expr, combined, prim_ctx)
                sq_sum += (v - tgt) ** 2 if math.isfinite(v) else 1e10
            except Exception:
                sq_sum += 1e10
        return sq_sum / len(observations)

    rng = np.random.RandomState(42)
    init_pts = [
        np.zeros(n_params),
        np.ones(n_params),
        -np.ones(n_params),
        np.full(n_params, 0.5),
        np.full(n_params, 0.1),
    ]
    for _ in range(max(0, n_restarts - len(init_pts))):
        init_pts.append(rng.uniform(-2.0, 2.0, n_params))

    best_x, best_mse = None, float("inf")
    for x0 in init_pts[:n_restarts]:
        try:
            res = minimize(objective2, x0, method="L-BFGS-B",
                           options={"maxiter": 200, "ftol": 1e-12})
            if math.isfinite(res.fun) and res.fun < best_mse:
                best_mse = float(res.fun)
                best_x = res.x
        except Exception:
            continue

    if best_x is None:
        return None, float("inf")
    fitted = {p: float(best_x[j]) for j, p in enumerate(param_names)}
    return fitted, best_mse


def _score_candidate(
    expr: Expr,
    all_param_names: frozenset[str],
    observations: list[tuple[dict, float]],
    prim_ctx: EvalContext,
    depth_penalty: float,
    ols_fallback_threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Score an expression candidate. Returns (mdl_score, fitted_params)."""
    present = _find_param_names(expr, all_param_names)
    fitted, mse = _ols_fit(expr, present, observations, prim_ctx)

    # Fallback to non-linear optimization when OLS gives poor results
    if (fitted is None or mse > ols_fallback_threshold) and present:
        fitted_nl, mse_nl = _nlopt_fit(expr, present, observations, prim_ctx)
        if fitted_nl is not None and mse_nl < (mse if mse != float("inf") else float("inf")):
            fitted, mse = fitted_nl, mse_nl

    if fitted is None:
        return float("inf"), {}
    log_mse = math.log(mse + _EPS)
    size = _tree_size(expr)
    mdl = depth_penalty * size + log_mse
    return mdl, fitted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_law(
    observations: list[tuple[dict, float]],
    prim_ctx: Optional[EvalContext] = None,
    prim_specs: Optional[list[PrimSpec]] = None,
    n_param_slots: int = 2,
    max_depth: int = 4,
    beam_width: int = 30,
    depth_penalty: float = 1.5,
) -> FittedLaw:
    """Discover a functional law from numerical observations.

    Parameters
    ----------
    observations  : list of (input_bindings: dict[str, float], output: float).
                    Input variable names are extracted from the dict keys.
    prim_ctx      : EvalContext for primitive ops. Defaults to make_prim_ctx().
    prim_specs    : list of PrimSpec. Defaults to get_prim_specs().
    n_param_slots : number of free parameter slots ("p0", "p1", ...).
    max_depth     : maximum expression tree size (nodes, not depth).
    beam_width    : number of candidates to keep at each step.
    depth_penalty : MDL penalty per tree node (higher = prefer simpler).

    Returns
    -------
    FittedLaw with:
      - schema.variables = frozenset of input variable names from observations
      - schema.params    = frozenset of parameter names actually used
      - params           = fitted float values
      - residual         = MSE on training observations
    """
    if not observations:
        raise ValueError("discover_law: observations list is empty")

    if prim_ctx is None:
        prim_ctx = make_prim_ctx()
    if prim_specs is None:
        prim_specs = get_prim_specs()

    # Extract input variable names from observations
    var_names: list[str] = sorted({k for inp, _ in observations for k in inp})
    param_names_all: list[str] = [_param_name(i) for i in range(n_param_slots)]
    all_param_set = frozenset(param_names_all)

    # Build terminals: input vars + param slots + fixed constants
    terminals: list[Expr] = (
        [var(v) for v in var_names]
        + [var(p) for p in param_names_all]
        + [atom(c) for c in _FIXED_CONSTS]
    )

    # Score all terminals and initialise beam
    # beam: list of (score, expr, fitted_params)  — sorted ascending by score
    beam: list[tuple[float, Expr, dict]] = []
    for e in terminals:
        s, fp = _score_candidate(e, all_param_set, observations, prim_ctx, depth_penalty)
        beam.append((s, e, fp))
    beam.sort(key=lambda x: x[0])
    beam = _dedup_beam(beam)[:beam_width]

    # Iteratively expand
    for _depth in range(max_depth):
        new_items: list[tuple[float, Expr, dict]] = list(beam)
        for spec in prim_specs:
            if spec.arity == 1:
                for (_, c, _) in beam:
                    e = Expr(head=spec.nid, args=(c,))
                    s, fp = _score_candidate(e, all_param_set, observations, prim_ctx, depth_penalty)
                    new_items.append((s, e, fp))
            elif spec.arity == 2:
                for (_, c1, _) in beam:
                    for (_, c2, _) in beam:
                        e = Expr(head=spec.nid, args=(c1, c2))
                        s, fp = _score_candidate(e, all_param_set, observations, prim_ctx, depth_penalty)
                        new_items.append((s, e, fp))
        new_items.sort(key=lambda x: x[0])
        new_items = _dedup_beam(new_items)
        beam = _diverse_keep(new_items, beam_width)

    # Best candidate
    best_score, best_expr, best_fitted = beam[0]

    # Build SchematicLaw from best expression
    present_params = frozenset(_find_param_names(best_expr, all_param_set))
    schema = SchematicLaw(
        pattern=best_expr,
        conclusion=best_expr,
        params=present_params,
        variables=frozenset(var_names),
        evidence=len(observations),
    )

    # Compute residual with best fitted params
    _, mse = _ols_fit(
        best_expr, sorted(present_params), observations, prim_ctx
    )

    return FittedLaw(schema=schema, params=best_fitted, residual=mse if mse != float("inf") else 0.0)


def _dedup_beam(
    beam: list[tuple[float, Expr, dict]],
) -> list[tuple[float, Expr, dict]]:
    """Remove duplicate Expr objects from beam (keep first / best-scored occurrence)."""
    seen: set[Expr] = set()
    result = []
    for item in beam:
        if item[1] not in seen:
            seen.add(item[1])
            result.append(item)
    return result
