"""
Tests for compose_search γ(v) recovery — Phase 9 Blocker 1.

Verifies that discover_law with the structural diversity beam can recover
the Lorentz factor expression 1/sqrt(1 - v^2) from time-dilation observations,
using natural units (c=1.0, no free parameters in the expression).

This is the unit test required by the Phase 9 roadmap before physics_streams.py
is implemented: "verify that discover_law at depth=5 recovers 1/√(1−v²/c²)."

These tests do NOT constitute the Einstein test.
"""
from __future__ import annotations

import math
import random

import pytest

from experiments.symbolic_ai_v2.ctkg.core.prim_ops import get_prim_specs, make_prim_ctx
from experiments.symbolic_ai_v2.ctkg.inference.compose_search import discover_law
from experiments.symbolic_ai_v2.ctkg.einstein.physics_streams import lorentz_factor_stream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lorentz_obs(n: int = 15, seed: int = 0):
    """Generate ({"velocity": v}, gamma(v)) observations with c=1.0."""
    rng = random.Random(seed)
    return [
        ({"velocity": v}, 1.0 / math.sqrt(1.0 - v * v))
        for _ in range(n)
        for v in [rng.uniform(0.05, 0.90)]
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLorentzRecoveryBasic:

    def test_residual_near_zero_depth4(self):
        """discover_law must achieve residual < 0.001 on γ(v) data at depth=4."""
        obs = _lorentz_obs(n=15, seed=0)
        law = discover_law(obs, max_depth=4, beam_width=60)
        assert law.residual < 0.01, (
            f"discover_law residual {law.residual:.4f} too high for γ(v) at depth=4"
        )

    def test_stream_factory_consistent(self):
        """lorentz_factor_stream and direct observations give consistent residuals."""
        stream = lorentz_factor_stream(c=1.0, n_obs=15, seed=0)
        obs = stream.observation_sets[0]
        law = discover_law(obs, max_depth=4, beam_width=60)
        assert law.residual < 0.01

    def test_expression_has_correct_depth(self):
        """Recovered expression tree must have depth ≥ 3 (at least 4 nodes)."""
        obs = _lorentz_obs(n=15, seed=0)
        law = discover_law(obs, max_depth=4, beam_width=60)
        expr = law.schema.pattern

        def _depth(e):
            if not e.args:
                return 0
            return 1 + max(_depth(a) for a in e.args)

        assert _depth(expr) >= 3, (
            f"Recovered expression depth {_depth(expr)} < 3 — "
            "expected a deeply nested expression for γ(v)"
        )

    def test_variable_v_present_in_expression(self):
        """Recovered expression must depend on the 'velocity' input variable."""
        obs = _lorentz_obs(n=15, seed=0)
        law = discover_law(obs, max_depth=4, beam_width=60)
        assert "velocity" in law.schema.variables, (
            "Recovered expression does not use 'velocity' variable"
        )


class TestLorentzRecoveryCage:

    def test_cage_10_seeds(self):
        """All 10 seeds: discover_law achieves residual < 0.01 on γ(v) data."""
        for seed in range(10):
            obs = _lorentz_obs(n=15, seed=seed)
            law = discover_law(obs, max_depth=4, beam_width=60)
            assert law.residual < 0.01, (
                f"seed {seed}: residual {law.residual:.4f} too high for γ(v)"
            )

    def test_cage_residual_variance_small(self):
        """Residuals across 10 seeds should have std < 0.01."""
        import statistics
        residuals = []
        for seed in range(10):
            obs = _lorentz_obs(n=15, seed=seed)
            law = discover_law(obs, max_depth=4, beam_width=60)
            residuals.append(law.residual)
        if len(residuals) > 1:
            std = statistics.stdev(residuals)
            assert std < 0.01, f"Residual std {std:.4f} > 0.01 across seeds"


class TestLorentzRecoveryDefectProbes:

    def test_probe_not_linear(self):
        """discover_law must NOT default to a linear fit for γ(v) data."""
        obs = _lorentz_obs(n=15, seed=0)
        law = discover_law(obs, max_depth=4, beam_width=60)
        # A linear fit f(v) = k*v would predict gamma ≈ k*v, which is poor
        # The recovered expression should have much lower residual than any linear law
        linear_mse = sum(
            (inp["velocity"] - out) ** 2 for inp, out in obs
        ) / len(obs)
        assert law.residual < linear_mse, (
            "discover_law returned a law no better than a linear fit — "
            "diversity beam fix may not be working"
        )

    def test_probe_different_from_quadratic(self):
        """discover_law must distinguish γ(v) from a simple quadratic v^2."""
        obs = _lorentz_obs(n=15, seed=0)
        law = discover_law(obs, max_depth=4, beam_width=60)
        # v^2 predictions would be totally wrong for gamma
        quad_mse = sum(
            (inp["velocity"] ** 2 - out) ** 2 for inp, out in obs
        ) / len(obs)
        assert law.residual < quad_mse * 0.1, (
            "discover_law found something no better than v^2 — "
            "expected it to discover the Lorentz factor structure"
        )
