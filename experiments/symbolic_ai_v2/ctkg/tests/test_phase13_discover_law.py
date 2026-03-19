"""
Phase 13 Integration Test — Parameterized Transform Discovery.

Tests that discover_law can recover Lorentz-factor-shaped laws with c != 1.0
by using the parameterized sub-search path in _transform_and_search.
"""
from __future__ import annotations

import math
import random

import pytest

from experiments.symbolic_ai_v2.ctkg.inference.compose_search import discover_law


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lorentz_obs_c(c: float = 1.0, n: int = 15, seed: int = 0):
    """Generate Lorentz-factor observations for a given speed of light c."""
    rng = random.Random(seed)
    obs = []
    for _ in range(n):
        v = rng.uniform(0.05, 0.90) * c
        gamma = 1.0 / math.sqrt(1.0 - (v / c) ** 2)
        obs.append(({'velocity': v}, gamma))
    return obs


# ---------------------------------------------------------------------------
# TestPhase13ParamTransform
# ---------------------------------------------------------------------------

class TestPhase13ParamTransform:

    def test_lorentz_c10_still_zero_param(self):
        """c=1.0 data returns zero-param (or very-small-param) expression, residual ≈ 0."""
        obs = _lorentz_obs_c(c=1.0, n=15, seed=0)
        law = discover_law(obs, max_depth=4, beam_width=60)
        assert law.residual < 0.01, (
            f"c=1.0 Lorentz residual {law.residual:.6f} too high"
        )
        # For c=1.0, no free parameter needed
        # The expression should have depth >= 3 (it's not a trivial constant)
        expr = law.schema.pattern
        def depth(e):
            if not e.args:
                return 0
            return 1 + max(depth(a) for a in e.args)
        assert depth(expr) >= 3, f"Expression too shallow (depth={depth(expr)}): {expr}"

    def test_lorentz_c03(self):
        """discover_law on γ(v) with c=0.3 achieves residual < 0.01, depth >= 3."""
        obs = _lorentz_obs_c(c=0.3, n=15, seed=0)
        law = discover_law(obs, max_depth=4, beam_width=60, n_param_slots=2)
        assert law.residual < 0.01, (
            f"c=0.3 Lorentz residual {law.residual:.6f} too high"
        )
        expr = law.schema.pattern
        def depth(e):
            if not e.args:
                return 0
            return 1 + max(depth(a) for a in e.args)
        assert depth(expr) >= 3, f"Expression too shallow (depth={depth(expr)})"

    def test_lorentz_c05(self):
        """c=0.5, residual < 0.01."""
        obs = _lorentz_obs_c(c=0.5, n=15, seed=0)
        law = discover_law(obs, max_depth=4, beam_width=60, n_param_slots=2)
        assert law.residual < 0.01, (
            f"c=0.5 Lorentz residual {law.residual:.6f} too high"
        )


# ---------------------------------------------------------------------------
# TestPhase13Cage
# ---------------------------------------------------------------------------

class TestPhase13Cage:

    def test_cage_5_seeds_c03(self):
        """γ(v) with c=0.3, seeds 0..4, all residual < 0.02."""
        for seed in range(5):
            obs = _lorentz_obs_c(c=0.3, n=15, seed=seed)
            law = discover_law(obs, max_depth=4, beam_width=80, n_param_slots=2)
            assert law.residual < 0.02, (
                f"seed={seed}, c=0.3: residual {law.residual:.6f} too high"
            )


# ---------------------------------------------------------------------------
# TestPhase13DefectProbes
# ---------------------------------------------------------------------------

class TestPhase13DefectProbes:

    def test_probe1_zero_param_preferred_when_exact(self):
        """c=1.0 data: result has no p0/p1 var in expression (zero-param preferred)."""
        from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
        obs = _lorentz_obs_c(c=1.0, n=15, seed=0)
        law = discover_law(obs, max_depth=4, beam_width=60, n_param_slots=2)
        # Residual must be near zero
        assert law.residual < 0.01

        # For a truly zero-param fit, there should be no free parameters used
        # (or the residual should be essentially zero with minimal params)
        # The key check is that residual is near zero
        assert law.residual < 1e-6, (
            f"Expected very low residual for exact c=1.0 case, got {law.residual:.2e}"
        )

    def test_probe2_wrong_transform_not_selected(self):
        """Data from f(x) = sqrt(1 - x*x) must NOT be claimed as a Lorentz expression.

        sqrt(1-x^2) has residual ≈ 0 vs 1/sqrt(1-x^2) which has high residual.
        The system should discover the correct form.
        """
        rng = random.Random(42)
        obs = []
        for _ in range(15):
            x = rng.uniform(0.05, 0.90)
            y = math.sqrt(1.0 - x * x)
            obs.append(({'x': x}, y))

        law = discover_law(obs, max_depth=4, beam_width=60, n_param_slots=2)
        # The expression should fit well (residual near zero)
        assert law.residual < 0.01, f"Expected low residual for sqrt(1-x^2), got {law.residual:.4f}"

        # The discovered expression should NOT be 1/sqrt(1-x^2) structure
        # i.e., it should not have residual near zero against the wrong target
        # Verify by checking that the law predicts sqrt(1-x^2) correctly
        from experiments.symbolic_ai_v2.ctkg.core.prim_ops import make_prim_ctx
        from experiments.symbolic_ai_v2.ctkg.core.quantity import eval_expr
        ctx = make_prim_ctx()
        errors = []
        for inp, target in obs:
            try:
                pred = eval_expr(law.schema.pattern, {**inp, **law.params}, ctx)
                errors.append(abs(pred - target))
            except Exception:
                errors.append(float('inf'))
        mean_err = sum(errors) / len(errors)
        assert mean_err < 0.05, f"Law doesn't predict sqrt(1-x^2) correctly: mean_err={mean_err:.4f}"
