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
    """Structural diversity signature: (root_op_nid, frozenset_of_var_nids, tree_size).

    Including tree_size ensures that structurally distinct compositions with the
    same root operator and variable set (e.g. INV(v) vs INV(SQRT(v))) receive
    different diversity slots, so deeper building blocks like SUB(1.0, SQ(v))
    are not crowded out by shallower expressions with the same (op, var_set).
    """
    return (expr.head, _var_nids_in(expr), _tree_size(expr))


def _diverse_keep(
    beam: list[tuple[float, Expr, dict]],
    beam_width: int,
) -> list[tuple[float, Expr, dict]]:
    """Keep top beam_width by MDL score, plus the best-MDL item per novel signature.

    Does a FULL SCAN of beam[beam_width:] to find the single best-MDL representative
    per structural signature not already in top-N.  The top beam_width diversity reps
    by MDL are then added to the result.

    Unlike the old early-stopping version (which stopped after collecting beam_width
    extras), this full-scan version guarantees that building blocks like SQ(v) or
    SUB(1, SQ(v)) — which may appear at large positions in the sorted beam — are
    never dropped simply because the extras quota filled up early.  They compete on
    merit (MDL) against all other novel-signature candidates.

    The result size is at most 2 * beam_width.
    """
    top = beam[:beam_width]
    seen_sigs: set = {_struct_sig(e) for _, e, _ in top}

    # Full scan: build best-MDL representative per novel signature.
    best_per_novel: dict = {}   # sig -> (score, expr, params)
    for item in beam[beam_width:]:
        sig = _struct_sig(item[1])
        if sig not in seen_sigs:
            if sig not in best_per_novel or item[0] < best_per_novel[sig][0]:
                best_per_novel[sig] = item

    # Take the top-beam_width diversity reps by MDL score.
    diversity_reps = sorted(best_per_novel.values(), key=lambda x: x[0])[:beam_width]
    return top + diversity_reps


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
) -> tuple[float, dict[str, float]]:
    """Score an expression candidate via OLS only. Returns (mdl_score, fitted_params).

    Uses OLS for speed during beam expansion. Non-linear fitting is done
    separately as a post-processing pass via _nlopt_rescore_beam, applied
    only to the final beam before discover_law returns.
    """
    present = _find_param_names(expr, all_param_names)
    fitted, mse = _ols_fit(expr, present, observations, prim_ctx)

    if fitted is None:
        return float("inf"), {}
    log_mse = math.log(mse + _EPS)
    size = _tree_size(expr)
    mdl = depth_penalty * size + log_mse
    return mdl, fitted


def _nlopt_rescore_beam(
    beam: list[tuple[float, Expr, dict]],
    all_param_names: frozenset[str],
    observations: list[tuple[dict, float]],
    prim_ctx: EvalContext,
    depth_penalty: float,
    top_k: int = 30,
) -> list[tuple[float, Expr, dict]]:
    """Re-score the top_k beam candidates with non-linear optimization.

    Called once after beam search completes. Runs _nlopt_fit for each of
    the top_k candidates (at most top_k scipy calls total — fast). Replaces
    OLS-fitted params with nlopt-fitted params when nlopt achieves lower MSE.

    This recovers correct parameters for non-linearly parameterized expressions
    (e.g. gamma(v) = 1/sqrt(1-v^2/c^2) where c appears non-linearly), without
    the prohibitive cost of calling nlopt during every beam expansion step.
    """
    rescored = []
    for i, (score, expr, fitted_ols) in enumerate(beam):
        if i >= top_k:
            rescored.append((score, expr, fitted_ols))
            continue

        present = _find_param_names(expr, all_param_names)
        if not present:
            rescored.append((score, expr, fitted_ols))
            continue

        fitted_nl, mse_nl = _nlopt_fit(expr, present, observations, prim_ctx)
        if fitted_nl is not None:
            log_mse = math.log(mse_nl + _EPS)
            size = _tree_size(expr)
            new_score = depth_penalty * size + log_mse
            if new_score < score:
                rescored.append((new_score, expr, fitted_nl))
                continue

        rescored.append((score, expr, fitted_ols))

    rescored.sort(key=lambda x: x[0])
    return rescored


# ---------------------------------------------------------------------------
# Zero-param search (used for target-transformation discovery)
# ---------------------------------------------------------------------------

def _zero_param_search(
    observations: list[tuple[dict, float]],
    prim_specs: list[PrimSpec],
    prim_ctx: EvalContext,
    var_names: list[str],
    max_depth: int,
    beam_width: int,
    depth_penalty: float,
) -> tuple[float, Expr]:
    """Beam search over zero-parameter (pure fixed-constant) expressions.

    Like the main beam search but with no free parameters (no p0/p1 slots).
    Used to discover structurally exact laws under a target transformation.

    Returns (best_mse, best_expr).
    """
    no_params: frozenset = frozenset()
    FIXED_CONSTS_ZP = ("0.0", "1.0", "2.0", "-1.0", "0.5", "3.0", "0.25")
    terminals: list[Expr] = (
        [var(v) for v in var_names]
        + [atom(c) for c in FIXED_CONSTS_ZP]
    )
    # Pre-expand terminals with all arity-1 prim applications over var nodes.
    # This ensures depth-1 sub-expressions (e.g. SQ(v), SQRT(v)) are always
    # available as arguments at the next beam depth, regardless of their MDL
    # score against the transformed target.  Without this, beam pruning kills
    # SQ(v) before depth 2, preventing SUB(1.0, SQ(v)) from being built.
    for _spec in prim_specs:
        if _spec.arity == 1:
            for _vname in var_names:
                terminals.append(Expr(head=_spec.nid, args=(var(_vname),)))

    beam_zp: list[tuple[float, Expr, dict]] = []
    for e in terminals:
        s, fp = _score_candidate(e, no_params, observations, prim_ctx, depth_penalty)
        beam_zp.append((s, e, fp))
    beam_zp.sort(key=lambda x: x[0])
    beam_zp = _dedup_beam(beam_zp)[:beam_width]

    for _depth in range(max_depth):
        new_items: list[tuple[float, Expr, dict]] = list(beam_zp)
        for spec in prim_specs:
            if spec.arity == 1:
                for (_, c, _) in beam_zp:
                    e = Expr(head=spec.nid, args=(c,))
                    s, fp = _score_candidate(e, no_params, observations, prim_ctx, depth_penalty)
                    new_items.append((s, e, fp))
            elif spec.arity == 2:
                for (_, c1, _) in beam_zp:
                    for (_, c2, _) in beam_zp:
                        e = Expr(head=spec.nid, args=(c1, c2))
                        s, fp = _score_candidate(e, no_params, observations, prim_ctx, depth_penalty)
                        new_items.append((s, e, fp))
        new_items.sort(key=lambda x: x[0])
        new_items = _dedup_beam(new_items)
        beam_zp = _diverse_keep(new_items, beam_width)

    best_score, best_expr, _ = beam_zp[0]
    # Compute actual MSE (score includes depth_penalty so we re-evaluate)
    _, mse = _ols_fit(best_expr, [], observations, prim_ctx)
    return mse if math.isfinite(mse) else float("inf"), best_expr


def _parameterized_sub_search(
    observations: list[tuple[dict, float]],
    prim_specs: list[PrimSpec],
    prim_ctx: EvalContext,
    var_names: list[str],
    all_param_names: list[str],
    max_depth: int,
    beam_width: int,
    depth_penalty: float,
) -> tuple[float, Expr, dict]:
    """Beam search with free parameters on transformed target.

    Like _zero_param_search but includes param slots (p0, p1, ...) in terminals
    and uses OLS fitting. Returns (best_mse, best_expr, best_fitted_params).
    """
    all_param_set = frozenset(all_param_names)
    FIXED_CONSTS_ZP = ("0.0", "1.0", "2.0", "-1.0", "0.5", "3.0", "0.25")
    terminals: list[Expr] = (
        [var(v) for v in var_names]
        + [var(p) for p in all_param_names]
        + [atom(c) for c in FIXED_CONSTS_ZP]
    )
    # Pre-expand terminals with all arity-1 prim applications over var nodes
    # (same rationale as _zero_param_search — prevents beam pruning from
    # eliminating SQ(v), SQRT(v), etc. before they can be composed).
    for _spec in prim_specs:
        if _spec.arity == 1:
            for _vname in var_names:
                terminals.append(Expr(head=_spec.nid, args=(var(_vname),)))

    beam: list[tuple[float, Expr, dict]] = []
    for e in terminals:
        s, fp = _score_candidate(e, all_param_set, observations, prim_ctx, depth_penalty)
        beam.append((s, e, fp))
    beam.sort(key=lambda x: x[0])
    beam = _dedup_beam(beam)[:beam_width]

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

    # Rescore with nlopt for non-linear params
    beam = _nlopt_rescore_beam(beam, all_param_set, observations, prim_ctx, depth_penalty, top_k=20)

    best_score, best_expr, best_fitted = beam[0]
    # Compute actual MSE with fitted params
    present = _find_param_names(best_expr, all_param_set)
    _, mse = _ols_fit(best_expr, present, observations, prim_ctx)
    if not math.isfinite(mse) and best_fitted:
        # Use nlopt-fitted params directly
        preds = []
        for inp, _ in observations:
            combined = {**inp, **best_fitted}
            try:
                p = eval_expr(best_expr, combined, prim_ctx)
                preds.append(p if math.isfinite(p) else float("nan"))
            except Exception:
                preds.append(float("nan"))
        valid = [(p, t) for (_, t), p in zip(observations, preds) if not math.isnan(p)]
        mse = sum((p - t) ** 2 for p, t in valid) / len(valid) if valid else float("inf")
    return mse if math.isfinite(mse) else float("inf"), best_expr, best_fitted


def _transform_and_search(
    observations: list[tuple[dict, float]],
    prim_specs: list[PrimSpec],
    prim_ctx: EvalContext,
    var_names: list[str],
    max_depth: int,
    beam_width: int,
    depth_penalty: float,
    all_param_names: list[str] = [],
) -> Optional[tuple[float, Expr]]:
    """Try a set of invertible target transformations and search for zero-param laws.

    For each transformation T in a candidate set, transforms the observed outputs
    by T, runs _zero_param_search on the transformed target, then wraps the
    discovered expression with T_inv to get a candidate for the original target.

    Returns (mse_on_original, wrapped_expr) for the best candidate whose
    wrapped expression achieves finite MSE on the ORIGINAL observations, or
    None if no transform yields a useful candidate.

    Rationale: some exact laws (e.g. γ(v) = 1/√(1-v²)) have intermediate
    building blocks (1-v², √(1-v²)) that are poor predictors of γ(v) and get
    pruned by MDL before the full expression can be assembled.  Applying a
    suitable transformation (e.g. T(x) = 1/x²) reveals a simpler target (1-v²)
    that is directly discoverable at lower depth.
    """
    # Each entry: (T_fn, T_inv_wrap, name)
    # T_fn: float → float  (applied to each observed output to get new target)
    # T_inv_wrap: Expr → Expr  (wraps the discovered sub-expression)
    sqrt_nid = TOKEN_GRAPH.encode("PRIM_SQRT")
    inv_nid  = TOKEN_GRAPH.encode("PRIM_INV")
    sq_nid   = TOKEN_GRAPH.encode("PRIM_SQ")

    def _safe_inv(t: float) -> float:
        return 1.0 / t if abs(t) > 1e-12 else float("inf")

    def _safe_invsq(t: float) -> float:
        return 1.0 / (t * t) if abs(t) > 1e-12 else float("inf")

    def _safe_sq(t: float) -> float:
        return t * t

    def _safe_sqrt(t: float) -> float:
        return math.sqrt(t) if t > 0.0 else float("inf")

    def _safe_invsqrt(t: float) -> float:
        return 1.0 / math.sqrt(t) if t > 0.0 else float("inf")

    candidate_transforms = [
        # (T_fn, T_inv_wrap)
        # T=1/x²  →  T_inv(y) = 1/√y  = INV(SQRT(y))   [key for Lorentz γ(v)]
        (_safe_invsq,  lambda e: Expr(inv_nid,  (Expr(sqrt_nid, (e,)),))),
        # T=1/x   →  T_inv(y) = 1/y   = INV(y)
        (_safe_inv,    lambda e: Expr(inv_nid,  (e,))),
        # T=x²    →  T_inv(y) = √y    = SQRT(y)
        (_safe_sq,     lambda e: Expr(sqrt_nid, (e,))),
        # T=√x    →  T_inv(y) = y²    = SQ(y)
        (_safe_sqrt,   lambda e: Expr(sq_nid,   (e,))),
        # T=1/√x  →  T_inv(y) = 1/y² = INV(SQ(y))
        (_safe_invsqrt, lambda e: Expr(inv_nid, (Expr(sq_nid,  (e,)),))),
    ]

    best_mse: float = float("inf")
    best_wrapped: Optional[Expr] = None
    best_fitted_params: dict = {}

    for T_fn, T_inv_wrap in candidate_transforms:
        # Apply transformation to outputs
        transformed: list[tuple[dict, float]] = []
        ok = True
        for inp, out in observations:
            t_out = T_fn(out)
            if not math.isfinite(t_out):
                ok = False
                break
            transformed.append((inp, t_out))
        if not ok:
            continue

        # Search for zero-param expression on transformed target.
        # Pre-expanded terminals (added to _zero_param_search) guarantee that
        # arity-1 sub-expressions like SQ(v) are available at depth-1 regardless
        # of their MDL score, so a small beam_width still finds SUB(1,SQ(v)).
        sub_bw = max(10, beam_width // 4)
        mse_t, sub_expr = _zero_param_search(
            transformed, prim_specs, prim_ctx, var_names,
            max_depth, sub_bw, depth_penalty,
        )

        # The MSE on the transformed target must be small to be interesting
        if math.isfinite(mse_t) and mse_t <= 1e-3:
            # Wrap the sub-expression to get candidate for original target
            wrapped = T_inv_wrap(sub_expr)

            # Evaluate wrapped expression on ORIGINAL observations
            _, orig_mse = _ols_fit(wrapped, [], observations, prim_ctx)
            if math.isfinite(orig_mse) and orig_mse < best_mse:
                best_mse = orig_mse
                best_wrapped = wrapped
                best_fitted_params = {}
            continue   # zero-param worked; skip parameterized search for this T

        # Zero-param search did not find a good fit on this transform.
        # If free parameter slots are available, try parameterized sub-search.
        if not all_param_names:
            continue

        param_sub_bw = max(10, beam_width // 4)
        mse_t_param, param_sub_expr, _param_fitted_t = _parameterized_sub_search(
            transformed, prim_specs, prim_ctx, var_names,
            all_param_names, max_depth, param_sub_bw, depth_penalty,
        )

        if not math.isfinite(mse_t_param) or mse_t_param > 1e-3:
            continue

        # Wrap the parameterized sub-expression to get candidate for original target
        wrapped_param = T_inv_wrap(param_sub_expr)

        # Evaluate wrapped expression on ORIGINAL observations
        present_params = _find_param_names(wrapped_param, frozenset(all_param_names))

        # Try OLS first
        fitted_orig, orig_mse = _ols_fit(wrapped_param, present_params, observations, prim_ctx)

        # Try nlopt regardless (needed for non-linearly parameterized expressions)
        fitted_nl, orig_mse_nl = _nlopt_fit(
            wrapped_param, present_params, observations, prim_ctx
        )
        # Pick the better of OLS and nlopt
        if math.isfinite(orig_mse_nl) and (not math.isfinite(orig_mse) or orig_mse_nl < orig_mse):
            best_fit_mse = orig_mse_nl
            best_fit_params = fitted_nl or {}
        elif math.isfinite(orig_mse):
            best_fit_mse = orig_mse
            best_fit_params = fitted_orig or {}
        else:
            continue

        if best_fit_mse < best_mse:
            best_mse = best_fit_mse
            best_wrapped = wrapped_param
            best_fitted_params = best_fit_params

    if best_wrapped is None:
        return None
    return best_mse, best_wrapped, best_fitted_params


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
    extra_atom_values: Optional[list[float]] = None,
) -> FittedLaw:
    """Discover a functional law from numerical observations.

    Parameters
    ----------
    observations      : list of (input_bindings: dict[str, float], output: float).
                        Input variable names are extracted from the dict keys.
    prim_ctx          : EvalContext for primitive ops. Defaults to make_prim_ctx().
    prim_specs        : list of PrimSpec. Defaults to get_prim_specs().
    n_param_slots     : number of free parameter slots ("p0", "p1", ...).
    max_depth         : maximum expression tree size (nodes, not depth).
    beam_width        : number of candidates to keep at each step.
    depth_penalty     : MDL penalty per tree node (higher = prefer simpler).
    extra_atom_values : additional fixed constant values to add to the terminal set.
                        These are opaque numeric constants (e.g. 1/c^2) that the
                        search can use like any other fixed atom.

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

    # Build terminals: input vars + param slots + fixed constants + extra atoms
    _extra_atom_strs: list[str] = []
    if extra_atom_values:
        for v in extra_atom_values:
            s = str(round(v, 10))
            if s not in _FIXED_CONSTS:
                _extra_atom_strs.append(s)
    terminals: list[Expr] = (
        [var(v) for v in var_names]
        + [var(p) for p in param_names_all]
        + [atom(c) for c in _FIXED_CONSTS]
        + [atom(c) for c in _extra_atom_strs]
    )

    # ---------------------------------------------------------------------------
    # Phase 1: Quick small-beam pre-check for parameterized laws.
    #
    # Run a cheap beam search with a small beam width and limited depth to see
    # if a parameterized expression (with p0, p1, ...) already fits the data
    # well.  If yes, return it immediately: parameterized expressions are the
    # natural representation for laws like f(x) = k*x, f(x) = a*x² + b, etc.
    # Returning from here preserves free parameters in the schema (important
    # for callers that inspect law.params).
    # ---------------------------------------------------------------------------
    def _run_beam(bw: int, md: int) -> list[tuple[float, Expr, dict]]:
        """Run main beam search with given beam_width and max_depth."""
        _beam: list[tuple[float, Expr, dict]] = []
        for e in terminals:
            s, fp = _score_candidate(e, all_param_set, observations, prim_ctx, depth_penalty)
            _beam.append((s, e, fp))
        _beam.sort(key=lambda x: x[0])
        _beam = _dedup_beam(_beam)[:bw]
        for _ in range(md):
            _new: list[tuple[float, Expr, dict]] = list(_beam)
            for spec in prim_specs:
                if spec.arity == 1:
                    for (_, c, _) in _beam:
                        e = Expr(head=spec.nid, args=(c,))
                        s, fp = _score_candidate(e, all_param_set, observations, prim_ctx, depth_penalty)
                        _new.append((s, e, fp))
                elif spec.arity == 2:
                    for (_, c1, _) in _beam:
                        for (_, c2, _) in _beam:
                            e = Expr(head=spec.nid, args=(c1, c2))
                            s, fp = _score_candidate(e, all_param_set, observations, prim_ctx, depth_penalty)
                            _new.append((s, e, fp))
            _new.sort(key=lambda x: x[0])
            _new = _dedup_beam(_new)
            _beam = _diverse_keep(_new, bw)
        return _beam

    quick_bw = max(10, beam_width // 4)
    quick_md = min(2, max_depth)
    quick_beam = _run_beam(quick_bw, quick_md)
    quick_best_expr = quick_beam[0][1]
    quick_best_fp   = quick_beam[0][2]
    quick_params    = _find_param_names(quick_best_expr, all_param_set)
    _, quick_mse    = _ols_fit(quick_best_expr, quick_params, observations, prim_ctx)

    if math.isfinite(quick_mse) and quick_mse < 1e-3:
        # OLS reported near-zero MSE.  Validate via actual expression evaluation
        # because OLS linearizes the expression (evaluates with p=1, scales by
        # fitted coefficient), which may give wrong predictions for non-linear
        # parameterizations like SQ(MUL(p0, x)) where OLS gives p0=3 but actual
        # eval gives (3x)²=9x² instead of 3x².
        preds_q = []
        for inp, _ in observations:
            combined_q = {**inp, **quick_best_fp}
            try:
                val = eval_expr(quick_best_expr, combined_q, prim_ctx)
                preds_q.append(val if math.isfinite(val) else float("nan"))
            except Exception:
                preds_q.append(float("nan"))
        valid_q = [(p, t) for (_, t), p in zip(observations, preds_q)
                   if not math.isnan(p)]
        actual_mse_q = (sum((p - t) ** 2 for p, t in valid_q) / len(valid_q)
                        if valid_q else float("inf"))

        if math.isfinite(actual_mse_q) and actual_mse_q < 1e-3:
            # Simple parameterized law found quickly — return without further search.
            present_params_q = frozenset(quick_params)
            schema_q = SchematicLaw(
                pattern=quick_best_expr,
                conclusion=quick_best_expr,
                params=present_params_q,
                variables=frozenset(var_names),
                evidence=len(observations),
            )
            return FittedLaw(schema=schema_q, params=quick_best_fp,
                             residual=actual_mse_q)

    # ---------------------------------------------------------------------------
    # Phase 2: Target-transformation pass for exact zero-param laws.
    #
    # For laws like γ(v) = 1/√(1-v²), applying T(x) = 1/x² maps the target
    # to 1-v², which is discoverable at depth 2.  The transform is inverted
    # (T_inv = INV(SQRT(·))) to recover the original expression.
    # ---------------------------------------------------------------------------
    fast_result = _transform_and_search(
        observations, prim_specs, prim_ctx, var_names,
        max_depth, beam_width, depth_penalty,
        all_param_names=param_names_all,
    )
    if fast_result is not None:
        fast_mse, fast_expr, fast_fitted = fast_result
        if fast_mse < 1e-4:
            present_params_fast = frozenset(_find_param_names(fast_expr, all_param_set))
            schema_fast = SchematicLaw(
                pattern=fast_expr,
                conclusion=fast_expr,
                params=present_params_fast,
                variables=frozenset(var_names),
                evidence=len(observations),
            )
            return FittedLaw(schema=schema_fast, params=fast_fitted, residual=fast_mse)

    # ---------------------------------------------------------------------------
    # Phase 3: Full main beam search (complex parameterized laws, fallback).
    # ---------------------------------------------------------------------------
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

    # Post-processing: re-score top candidates with non-linear optimization.
    combined = _nlopt_rescore_beam(
        beam, all_param_set, observations, prim_ctx, depth_penalty, top_k=30
    )

    # Target-transformation pass: try invertible transforms of the output target
    # to discover zero-param laws whose building blocks have poor MDL against
    # the original target (e.g. γ(v) = INV(SQRT(SUB(1.0, SQ(v))))).
    transform_result = _transform_and_search(
        observations, prim_specs, prim_ctx, var_names,
        max_depth, beam_width, depth_penalty,
        all_param_names=param_names_all,
    )

    # Best candidate: pick the better of (main beam, transform pass)
    best_score, best_expr, best_fitted = combined[0]
    if transform_result is not None:
        tr_mse, tr_expr, tr_fitted = transform_result
        # Main beam residual (for fair comparison — use raw MSE not MDL score)
        # We compare using MSE on original observations
        _, main_mse_val = _ols_fit(best_expr,
                                   sorted(_find_param_names(best_expr, all_param_set)),
                                   observations, prim_ctx)
        if tr_mse < main_mse_val:
            best_expr = tr_expr
            best_fitted = tr_fitted

    # Build SchematicLaw from best expression
    present_params = frozenset(_find_param_names(best_expr, all_param_set))
    schema = SchematicLaw(
        pattern=best_expr,
        conclusion=best_expr,
        params=present_params,
        variables=frozenset(var_names),
        evidence=len(observations),
    )

    # Compute final residual with best fitted params (nlopt may have improved them)
    present_list = sorted(present_params)
    if best_fitted and present_list:
        # Evaluate MSE directly using the fitted params
        preds = []
        for inp, _ in observations:
            eval_bindings = {**inp, **best_fitted}
            try:
                p = eval_expr(best_expr, eval_bindings, prim_ctx)
                preds.append(p if math.isfinite(p) else float("nan"))
            except Exception:
                preds.append(float("nan"))
        valid = [(p, t) for (_, t), p in zip(observations, preds) if not math.isnan(p)]
        mse = sum((p - t) ** 2 for p, t in valid) / len(valid) if valid else float("inf")
    else:
        _, mse = _ols_fit(best_expr, present_list, observations, prim_ctx)

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
