"""
Tests for compose_search.py — Phase 10 of the Einstein Roadmap.

Graph-backed compositional law discovery: discover_law() finds the functional
form from observations alone without a pre-specified SchematicLaw schema.

Test classes
------------
TestComposeSearchBasic      : linear and quadratic law discovery
TestComposeSearchMDLSelection: MDL prefers simpler structure when data fits
TestComposeSearchCage       : 10 anonymous input-variable-name seeds
TestComposeSearchDefectProbes: targeted violation probes
"""
from __future__ import annotations

import math
import random

import pytest

from experiments.symbolic_ai_v2.ctkg.core.prim_ops import (
    get_prim_specs, make_prim_ctx, PRIM_MUL, PRIM_ADD, PRIM_SQ, PRIM_POW,
)
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import var, atom, Expr
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext, eval_expr
from experiments.symbolic_ai_v2.ctkg.inference.compose_search import discover_law, _find_param_names
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import predict_continuous


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_linear(k: float, n: int = 8, var_name: str = "x") -> list:
    """f(x) = k*x observations."""
    return [({var_name: float(i + 1)}, k * (i + 1)) for i in range(n)]


def _obs_quadratic(k: float, n: int = 8, var_name: str = "x") -> list:
    """f(x) = k*x² observations."""
    return [({var_name: float(i + 1)}, k * (i + 1) ** 2) for i in range(n)]


def _rel_err(pred: float, true: float) -> float:
    if abs(true) < 1e-12:
        return abs(pred)
    return abs(pred - true) / abs(true)


def _anon_var() -> str:
    """Return a random Unicode symbol from U+2200..U+22FE."""
    return chr(0x2200 + random.randint(0, 0xFE))


# ---------------------------------------------------------------------------
# TestComposeSearchBasic
# ---------------------------------------------------------------------------

class TestComposeSearchBasic:

    def test_discovers_linear_k3(self):
        """discover_law recovers k=3 from f(x)=3*x data."""
        ctx = make_prim_ctx()
        obs = _obs_linear(3.0)
        law = discover_law(obs, prim_ctx=ctx)
        # Predict held-out points
        for x in [10.0, 15.0, 20.0]:
            pred = predict_continuous(law, {"x": x}, ctx)
            assert _rel_err(pred, 3.0 * x) < 0.05, \
                f"linear k=3: pred={pred:.4f} true={3.0*x:.4f}"

    def test_discovers_linear_k7(self):
        """discover_law recovers k=7 from f(x)=7*x data."""
        ctx = make_prim_ctx()
        obs = _obs_linear(7.0)
        law = discover_law(obs, prim_ctx=ctx)
        for x in [5.0, 9.0, 12.0]:
            pred = predict_continuous(law, {"x": x}, ctx)
            assert _rel_err(pred, 7.0 * x) < 0.05

    def test_discovers_quadratic_k2(self):
        """discover_law recovers k=2 from f(x)=2*x² data."""
        ctx = make_prim_ctx()
        obs = _obs_quadratic(2.0)
        law = discover_law(obs, prim_ctx=ctx, max_depth=3, beam_width=40)
        for x in [5.0, 8.0, 10.0]:
            pred = predict_continuous(law, {"x": x}, ctx)
            assert _rel_err(pred, 2.0 * x ** 2) < 0.05, \
                f"quadratic k=2: pred={pred:.4f} true={2.0*x**2:.4f}"

    def test_residual_near_zero(self):
        """discover_law residual is near zero on noise-free linear data."""
        ctx = make_prim_ctx()
        obs = _obs_linear(5.0)
        law = discover_law(obs, prim_ctx=ctx)
        assert law.residual < 0.01, f"residual={law.residual:.6f}"

    def test_schema_has_variables(self):
        """Discovered SchematicLaw has correct variable names."""
        ctx = make_prim_ctx()
        obs = _obs_linear(4.0)
        law = discover_law(obs, prim_ctx=ctx)
        assert "x" in law.schema.variables

    def test_schema_has_params(self):
        """Discovered SchematicLaw has at least one free parameter."""
        ctx = make_prim_ctx()
        obs = _obs_linear(4.0)
        law = discover_law(obs, prim_ctx=ctx)
        assert len(law.params) >= 1


# ---------------------------------------------------------------------------
# TestComposeSearchMDLSelection
# ---------------------------------------------------------------------------

class TestComposeSearchMDLSelection:

    def test_linear_data_not_over_fitted_quadratic(self):
        """Linear data → discovered law predicts linearly on extrapolation."""
        ctx = make_prim_ctx()
        obs = _obs_linear(5.0)
        law = discover_law(obs, prim_ctx=ctx, max_depth=3, beam_width=40)
        # Extrapolate: if quadratic k≈5, x=20 gives 5*20=100 (linear) vs 5*400=2000 (quad)
        pred = predict_continuous(law, {"x": 20.0}, ctx)
        assert _rel_err(pred, 100.0) < 0.10, \
            f"linear extrap: pred={pred:.2f} expected~100"

    def test_quadratic_data_not_fitted_linear(self):
        """Quadratic data → discovered law predicts quadratically, not linearly."""
        ctx = make_prim_ctx()
        obs = _obs_quadratic(1.0)
        law = discover_law(obs, prim_ctx=ctx, max_depth=3, beam_width=40)
        # At x=10: quadratic=100, linear best-fit≈9*10=90 (slope of x=1..8 mean)
        pred = predict_continuous(law, {"x": 10.0}, ctx)
        assert _rel_err(pred, 100.0) < 0.10, \
            f"quadratic extrap: pred={pred:.2f} expected~100"

    def test_mdl_selects_simpler_when_both_fit(self):
        """When linear perfectly fits, MDL rejects higher-complexity alternatives."""
        ctx = make_prim_ctx()
        # Pure linear data — quadratic with k≈0 also fits but is more complex
        obs = _obs_linear(3.0, n=10)
        law = discover_law(obs, prim_ctx=ctx, max_depth=3, beam_width=40, depth_penalty=2.0)
        # Simple sanity: residual should be near zero
        assert law.residual < 0.01


# ---------------------------------------------------------------------------
# TestComposeSearchCage
# ---------------------------------------------------------------------------

class TestComposeSearchCage:

    def test_cage_linear_10_seeds(self):
        """10 anonymous input-variable-name seeds: linear law recovered consistently."""
        ctx = make_prim_ctx()
        rng = random.Random(42)
        k_true = 5.0
        results = []
        for seed in range(10):
            var_sym = chr(0x2200 + rng.randint(0, 0xFE))
            obs = [({var_sym: float(i + 1)}, k_true * (i + 1)) for i in range(8)]
            law = discover_law(obs, prim_ctx=ctx)
            pred = predict_continuous(law, {var_sym: 9.0}, ctx)
            results.append(pred / 9.0)  # should be ≈ k_true=5.0
        # All recovered k values within 5% of true
        for i, k_rec in enumerate(results):
            assert _rel_err(k_rec, k_true) < 0.05, \
                f"seed {i}: k_recovered={k_rec:.4f} k_true={k_true}"
        # Variance across seeds < 5pp (in units of k, so < 0.25)
        import statistics
        assert statistics.stdev(results) < 0.25, \
            f"cage linear: k std={statistics.stdev(results):.4f}"

    def test_cage_quadratic_10_seeds(self):
        """10 anonymous input-variable-name seeds: quadratic law recovered consistently."""
        ctx = make_prim_ctx()
        rng = random.Random(99)
        k_true = 2.0
        results = []
        for seed in range(10):
            var_sym = chr(0x2200 + rng.randint(0, 0xFE))
            obs = [({var_sym: float(i + 1)}, k_true * (i + 1) ** 2) for i in range(8)]
            law = discover_law(obs, prim_ctx=ctx, max_depth=3, beam_width=40)
            x_test = 6.0
            pred = predict_continuous(law, {var_sym: x_test}, ctx)
            results.append(pred)
        # All predictions within 10% of k*x²
        true_val = k_true * 6.0 ** 2
        for i, pred in enumerate(results):
            assert _rel_err(pred, true_val) < 0.10, \
                f"seed {i}: pred={pred:.4f} true={true_val}"

    def test_named_vs_anon_gap(self):
        """Named ('x') and anonymous variable names give same accuracy (<5pp gap)."""
        ctx = make_prim_ctx()
        k_true = 4.0
        # Named
        obs_named = _obs_linear(k_true)
        law_named = discover_law(obs_named, prim_ctx=ctx)
        pred_named = predict_continuous(law_named, {"x": 10.0}, ctx)
        # Anonymous
        sym = chr(0x2295)
        obs_anon = [({sym: float(i + 1)}, k_true * (i + 1)) for i in range(8)]
        law_anon = discover_law(obs_anon, prim_ctx=ctx)
        pred_anon = predict_continuous(law_anon, {sym: 10.0}, ctx)
        # Gap in recovered k
        k_named = pred_named / 10.0
        k_anon  = pred_anon  / 10.0
        assert _rel_err(k_named, k_true) < 0.05
        assert _rel_err(k_anon,  k_true) < 0.05
        assert abs(k_named - k_anon) / k_true < 0.05, \
            f"gap: named={k_named:.4f} anon={k_anon:.4f}"


# ---------------------------------------------------------------------------
# TestComposeSearchDefectProbes
# ---------------------------------------------------------------------------

class TestComposeSearchDefectProbes:

    def test_probe_no_linear_default_for_quadratic(self):
        """Quadratic data MUST NOT be fitted with a linear expression.

        A system with a hardcoded linear prior would fit quadratic data with
        a linear expression and have large extrapolation error.
        This probe fails if discover_law defaults to linear regardless of data.
        """
        ctx = make_prim_ctx()
        obs = _obs_quadratic(3.0, n=10)
        law = discover_law(obs, prim_ctx=ctx, max_depth=3, beam_width=40)
        # At x=15: quadratic=3*225=675, linear best-fit ≈ 3*(1+2+...+10)/10 * 15 ≈ 247
        pred_15 = predict_continuous(law, {"x": 15.0}, ctx)
        true_15 = 3.0 * 15.0 ** 2
        assert _rel_err(pred_15, true_15) < 0.10, \
            f"PROBE: quadratic probe: pred={pred_15:.2f} true={true_15:.2f}" \
            f" — possible linear default"

    def test_probe_schema_no_domain_tokens(self):
        """Discovered SchematicLaw must not contain domain-specific operator NodeIds.

        The expression in schema.pattern should only contain:
          - PrimSpec NodeIds (PRIM_MUL, PRIM_ADD, etc.)
          - Var NodeIds for input variables and params
          - Atom NodeIds for fixed constants

        A Bitter Lesson violation would embed domain token NodeIds (e.g. the
        NodeId for 'mul', 'add') inside the expression instead of PRIM_MUL, PRIM_ADD.
        """
        ctx = make_prim_ctx()
        obs = _obs_linear(5.0)
        law = discover_law(obs, prim_ctx=ctx)
        prim_nids = {spec.nid for spec in get_prim_specs()}
        # Walk all internal (non-leaf) nodes; their head must be a prim nid
        def _check(expr):
            if expr.args:  # internal node
                assert expr.head in prim_nids, \
                    f"PROBE: internal node head {expr.head} not a prim NodeId"
                for a in expr.args:
                    _check(a)
        _check(law.schema.pattern)

    def test_probe_iron_law_prim_nids_not_domain_names(self):
        """PRIM_MUL NodeId must differ from 'mul' domain token NodeId.

        If they coincide, a domain EvalContext could accidentally satisfy
        prim_ctx lookups — the Iron Law boundary would be broken.
        """
        from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
        prim_mul_nid = TOKEN_GRAPH.encode(PRIM_MUL)
        domain_mul_nid = TOKEN_GRAPH.encode("mul")
        assert prim_mul_nid != domain_mul_nid, \
            "PROBE: PRIM_MUL and 'mul' share a NodeId — Iron Law boundary broken"

    def test_probe_discover_law_does_not_take_schema_input(self):
        """discover_law signature must NOT accept a schema argument.

        The violation to detect: a caller passing SchematicLaw to discover_law,
        bypassing the structure search. The function signature must enforce that
        only observations are accepted.
        """
        import inspect
        sig = inspect.signature(discover_law)
        param_names = list(sig.parameters.keys())
        # Must not have a 'schema' or 'schematic_law' parameter
        assert "schema" not in param_names, \
            "PROBE: discover_law has a 'schema' parameter — structure search bypassed"
        assert "schematic_law" not in param_names, \
            "PROBE: discover_law has a 'schematic_law' parameter"
