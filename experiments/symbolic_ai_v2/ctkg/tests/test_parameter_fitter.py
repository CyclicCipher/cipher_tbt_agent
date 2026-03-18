"""
Phase 3 gate tests — Continuous Quantities and Numerical Fitting.

Four test classes:

1. TestQuantityNode        — QuantityNode identity, unit registry, eval_expr,
                             continuous_surprise.
2. TestParameterFitter     — OLS fitting: single param, multi-param, noisy
                             data, predict_continuous, graph storage.
3. TestPhase3Roadmap       — IDA I-5 continuous: recover F=ma within 1%
                             parameter error from 10 noisy observations.
4. TestBitterLessonCage    — 10 anonymous symbol tables; fitted parameter
                             values must agree within 1% across all seeds.
                             Variance < 5 pp.
5. TestDefectProbe         — unit confusion: QuantityNode(1.0, m) ≠
                             QuantityNode(1.0, mm); fitting without unit
                             conversion gives wrong k; with conversion gives
                             correct k.  Run with anonymous unit NodeIds so
                             string matching on "m" / "mm" cannot distinguish.
"""
from __future__ import annotations

import math
import random
import unicodedata

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom, node, var
from experiments.symbolic_ai_v2.ctkg.core.expr_law import rename_expr
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import (
    SchematicLaw, discover_parametric_law,
)
from experiments.symbolic_ai_v2.ctkg.core.quantity import (
    QuantityNode, EvalContext, eval_expr,
    register_unit_conversion, get_unit_factor, convert_quantity,
    continuous_surprise,
)
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import (
    FittedLaw, fit_parameters, fit_law,
    predict_continuous, add_fitted_law, query_fitted_laws,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _mg() -> MorphismGraph:
    return MorphismGraph()


def _mul_ctx(*op_tokens: str) -> EvalContext:
    """EvalContext mapping each op_token → multiplication callable."""
    return EvalContext({
        TOKEN_GRAPH.encode(t): (lambda a, b: a * b)
        for t in op_tokens
    })


_UNICODE_OPS = [
    chr(i) for i in range(0x2200, 0x22FF)
    if unicodedata.category(chr(i)) not in ('Cn', 'Co')
]


def _fresh_sym(roles: list[str], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    return dict(zip(roles, rng.sample(_UNICODE_OPS, len(roles))))


def _nid(sym: dict[str, str]) -> dict[int, int]:
    return {TOKEN_GRAPH.encode(s): TOKEN_GRAPH.encode(t) for s, t in sym.items()}


# ---------------------------------------------------------------------------
# 1. QuantityNode, EvalContext, unit registry, continuous_surprise
# ---------------------------------------------------------------------------

class TestQuantityNode:

    # QuantityNode identity
    def test_same_value_same_unit_equal(self):
        nid = TOKEN_GRAPH.encode('m')
        assert QuantityNode(1.0, nid) == QuantityNode(1.0, nid)

    def test_same_value_different_unit_not_equal(self):
        m_nid  = TOKEN_GRAPH.encode('m')
        mm_nid = TOKEN_GRAPH.encode('mm')
        assert QuantityNode(1.0, m_nid) != QuantityNode(1.0, mm_nid)

    def test_different_value_same_unit_not_equal(self):
        nid = TOKEN_GRAPH.encode('m')
        assert QuantityNode(1.0, nid) != QuantityNode(2.0, nid)

    def test_quantity_node_is_hashable(self):
        nid = TOKEN_GRAPH.encode('kg')
        s = {QuantityNode(3.0, nid), QuantityNode(3.0, nid)}
        assert len(s) == 1

    # Unit registry
    def test_register_and_retrieve_conversion(self):
        mg = _mg()
        m_nid  = TOKEN_GRAPH.encode('m')
        mm_nid = TOKEN_GRAPH.encode('mm')
        register_unit_conversion(mg, m_nid, mm_nid, 1000.0)
        factor = get_unit_factor(mg, m_nid, mm_nid)
        assert factor == pytest.approx(1000.0)

    def test_absent_conversion_returns_none(self):
        mg = _mg()
        m_nid = TOKEN_GRAPH.encode('m')
        s_nid = TOKEN_GRAPH.encode('s')
        assert get_unit_factor(mg, m_nid, s_nid) is None

    def test_convert_quantity(self):
        mg = _mg()
        m_nid  = TOKEN_GRAPH.encode('m')
        mm_nid = TOKEN_GRAPH.encode('mm')
        register_unit_conversion(mg, m_nid, mm_nid, 1000.0)
        qty = QuantityNode(2.0, m_nid)
        converted = convert_quantity(mg, qty, mm_nid)
        assert converted == QuantityNode(2000.0, mm_nid)

    def test_convert_same_unit_noop(self):
        mg = _mg()
        nid = TOKEN_GRAPH.encode('kg')
        qty = QuantityNode(5.0, nid)
        assert convert_quantity(mg, qty, nid) is qty

    def test_convert_no_path_returns_none(self):
        mg = _mg()
        m_nid  = TOKEN_GRAPH.encode('m')
        kg_nid = TOKEN_GRAPH.encode('kg')
        assert convert_quantity(mg, QuantityNode(1.0, m_nid), kg_nid) is None

    def test_unit_conversion_idempotent(self):
        """Registering the same conversion twice doesn't duplicate morphisms."""
        mg = _mg()
        m_nid  = TOKEN_GRAPH.encode('m')
        mm_nid = TOKEN_GRAPH.encode('mm')
        register_unit_conversion(mg, m_nid, mm_nid, 1000.0)
        register_unit_conversion(mg, m_nid, mm_nid, 1000.0)
        factor = get_unit_factor(mg, m_nid, mm_nid)
        assert factor == pytest.approx(1000.0)

    # EvalContext / eval_expr
    def test_eval_float_atom(self):
        ctx = EvalContext()
        assert eval_expr(atom('3'), {}, ctx) == pytest.approx(3.0)

    def test_eval_variable_binding(self):
        ctx = EvalContext()
        assert eval_expr(var('x'), {'x': 7.0}, ctx) == pytest.approx(7.0)

    def test_eval_binary_mul(self):
        mul_nid = TOKEN_GRAPH.encode('mul')
        ctx = EvalContext({mul_nid: lambda a, b: a * b})
        expr = node('mul', atom('3'), atom('5'))
        assert eval_expr(expr, {}, ctx) == pytest.approx(15.0)

    def test_eval_nested(self):
        mul_nid = TOKEN_GRAPH.encode('mul')
        ctx = EvalContext({mul_nid: lambda a, b: a * b})
        # mul(mul(2, 3), 4) = 24
        expr = node('mul', node('mul', atom('2'), atom('3')), atom('4'))
        assert eval_expr(expr, {}, ctx) == pytest.approx(24.0)

    def test_eval_variable_in_formula(self):
        mul_nid = TOKEN_GRAPH.encode('mul')
        ctx = EvalContext({mul_nid: lambda a, b: a * b})
        expr = node('mul', var('x'), atom('5'))
        assert eval_expr(expr, {'x': 3.0}, ctx) == pytest.approx(15.0)

    def test_eval_unbound_var_raises(self):
        ctx = EvalContext()
        with pytest.raises(ValueError, match="unbound variable"):
            eval_expr(var('z'), {}, ctx)

    def test_eval_unknown_op_raises(self):
        nid = TOKEN_GRAPH.encode('unknown_op')
        ctx = EvalContext()  # no operators registered
        with pytest.raises(ValueError, match="no operator registered"):
            eval_expr(node('unknown_op', atom('1'), atom('2')), {}, ctx)

    def test_eval_atom_not_float_raises(self):
        ctx = EvalContext()
        with pytest.raises(ValueError, match="not a float literal"):
            eval_expr(atom('xyz'), {}, ctx)

    # continuous_surprise
    def test_surprise_zero_when_exact(self):
        assert continuous_surprise(3.0, 3.0) == pytest.approx(0.0)

    def test_surprise_positive_when_wrong(self):
        assert continuous_surprise(5.0, 3.0) > 0.0

    def test_surprise_symmetric(self):
        assert continuous_surprise(5.0, 3.0) == pytest.approx(
            continuous_surprise(3.0, 5.0)
        )

    def test_surprise_scales_with_sigma(self):
        # surprise with sigma=2 should be (1/4) of surprise with sigma=1
        assert continuous_surprise(5.0, 3.0, sigma=2.0) == pytest.approx(
            continuous_surprise(5.0, 3.0, sigma=1.0) / 4.0
        )


# ---------------------------------------------------------------------------
# 2. Parameter fitter functional tests
# ---------------------------------------------------------------------------

class TestParameterFitter:

    def _mul_ctx(self) -> EvalContext:
        return EvalContext({TOKEN_GRAPH.encode('mul'): lambda a, b: a * b})

    def _add_ctx(self) -> EvalContext:
        return EvalContext({
            TOKEN_GRAPH.encode('mul'): lambda a, b: a * b,
            TOKEN_GRAPH.encode('add'): lambda a, b: a + b,
        })

    # Single linear parameter — exact data
    def test_fit_single_param_exact(self):
        """k * x, exact observations → k fitted exactly."""
        ctx = self._mul_ctx()
        formula = node('mul', var('k'), var('x'))
        obs = [(({'x': float(x)}, 2.0 * x)) for x in range(1, 6)]
        fl = fit_parameters(formula, frozenset({'k'}), obs, ctx)
        assert fl.params['k'] == pytest.approx(2.0, rel=1e-6)
        assert fl.residual == pytest.approx(0.0, abs=1e-10)

    def test_fit_single_param_larger_k(self):
        ctx = self._mul_ctx()
        formula = node('mul', var('k'), var('x'))
        obs = [({'x': float(x)}, 7.0 * x) for x in range(1, 8)]
        fl = fit_parameters(formula, frozenset({'k'}), obs, ctx)
        assert fl.params['k'] == pytest.approx(7.0, rel=1e-6)

    def test_fit_multi_param_linear(self):
        """k1*x + k2*y — two linear parameters."""
        add_nid = TOKEN_GRAPH.encode('add')
        mul_nid = TOKEN_GRAPH.encode('mul')
        ctx = EvalContext({add_nid: lambda a, b: a + b,
                           mul_nid: lambda a, b: a * b})
        formula = node('add',
                       node('mul', var('k1'), var('x')),
                       node('mul', var('k2'), var('y')))
        obs = [
            ({'x': 1.0, 'y': 2.0}, 3.0 * 1 + 5.0 * 2),
            ({'x': 3.0, 'y': 1.0}, 3.0 * 3 + 5.0 * 1),
            ({'x': 2.0, 'y': 4.0}, 3.0 * 2 + 5.0 * 4),
            ({'x': 5.0, 'y': 3.0}, 3.0 * 5 + 5.0 * 3),
        ]
        fl = fit_parameters(formula, frozenset({'k1', 'k2'}), obs, ctx)
        assert fl.params['k1'] == pytest.approx(3.0, rel=1e-4)
        assert fl.params['k2'] == pytest.approx(5.0, rel=1e-4)

    # With noise
    def test_fit_single_param_with_noise(self):
        """k=2.0, Gaussian noise σ=0.05 on 20 obs → recover within 1%."""
        rng = random.Random(42)
        ctx = self._mul_ctx()
        formula = node('mul', var('k'), var('x'))
        true_k = 2.0
        obs = [
            ({'x': float(x)}, true_k * x + rng.gauss(0, 0.05))
            for x in range(1, 21)
        ]
        fl = fit_parameters(formula, frozenset({'k'}), obs, ctx)
        assert fl.params['k'] == pytest.approx(true_k, rel=0.01)

    # No parameters
    def test_fit_no_params_residual(self):
        """Formula with no free parameters: just compute residual."""
        ctx = self._mul_ctx()
        # formula = mul(3, x) — no param vars
        formula = node('mul', atom('3'), var('x'))
        obs = [({'x': float(x)}, 3.0 * x) for x in range(1, 5)]
        fl = fit_parameters(formula, frozenset(), obs, ctx)
        assert fl.params == {}
        assert fl.residual == pytest.approx(0.0, abs=1e-10)

    # predict_continuous
    def test_predict_continuous(self):
        ctx = self._mul_ctx()
        formula = node('mul', var('k'), var('x'))
        obs = [({'x': float(x)}, 4.0 * x) for x in range(1, 6)]
        fl = fit_parameters(formula, frozenset({'k'}), obs, ctx)
        pred = predict_continuous(fl, {'x': 10.0}, ctx)
        assert pred == pytest.approx(4.0 * 10.0, rel=1e-4)

    # Graph storage
    def test_add_and_query_fitted_law(self):
        mg = _mg()
        ctx = self._mul_ctx()
        formula = node('mul', var('k'), var('x'))
        obs = [({'x': float(x)}, 3.0 * x) for x in range(1, 4)]
        fl = fit_parameters(formula, frozenset({'k'}), obs, ctx)
        mid = add_fitted_law(mg, 'spring_law', fl)
        laws = query_fitted_laws(mg, 'spring_law')
        assert len(laws) == 1
        assert laws[0].params['k'] == pytest.approx(3.0, rel=1e-4)

    def test_add_fitted_law_dedup(self):
        mg = _mg()
        ctx = self._mul_ctx()
        formula = node('mul', var('k'), var('x'))
        obs = [({'x': float(x)}, 3.0 * x) for x in range(1, 4)]
        fl = fit_parameters(formula, frozenset({'k'}), obs, ctx)
        mid1 = add_fitted_law(mg, 'spring_law', fl)
        mid2 = add_fitted_law(mg, 'spring_law', fl)
        assert mid1 == mid2
        assert len(query_fitted_laws(mg, 'spring_law')) == 1

    def test_query_absent_label(self):
        mg = _mg()
        assert query_fitted_laws(mg, 'nonexistent') == []


# ---------------------------------------------------------------------------
# 3. IDA I-5 continuous — recover F=ma within 1% parameter error
# ---------------------------------------------------------------------------

class TestPhase3Roadmap:
    """
    IDA benchmark I-5 extended to continuous domains.

    Law: F = k * m * a  (Newton's second law, where k = 1.0 exactly)

    Given 10 (m_value, a_value, F_value) triples with Gaussian noise σ=0.05,
    recover k to within 1% of the true value 1.0.
    """

    def _make_fma_formula(self) -> tuple:
        """Returns (formula_expr, ctx, param_names)."""
        mul_nid = TOKEN_GRAPH.encode('mul')
        ctx = EvalContext({mul_nid: lambda a, b: a * b})
        # F = k * (m * a) = mul(k, mul(m, a))
        formula = node('mul', var('k'), node('mul', var('m'), var('a')))
        return formula, ctx, frozenset({'k'})

    def _noisy_fma_obs(self, n: int, true_k: float, sigma: float,
                        seed: int) -> list:
        rng = random.Random(seed)
        obs = []
        for _ in range(n):
            m_val = rng.uniform(0.5, 5.0)
            a_val = rng.uniform(0.1, 10.0)
            F_val = true_k * m_val * a_val + rng.gauss(0, sigma)
            obs.append(({'m': m_val, 'a': a_val}, F_val))
        return obs

    def test_recover_k_within_1_percent(self):
        formula, ctx, param_names = self._make_fma_formula()
        obs = self._noisy_fma_obs(n=10, true_k=1.0, sigma=0.05, seed=7)
        fl = fit_parameters(formula, param_names, obs, ctx)
        assert fl.params['k'] == pytest.approx(1.0, rel=0.01), (
            f"Fitted k={fl.params['k']:.6f}, expected ≈1.0 within 1%"
        )

    def test_recover_k_2_0_within_1_percent(self):
        """Same formula structure with a different true k."""
        formula, ctx, param_names = self._make_fma_formula()
        obs = self._noisy_fma_obs(n=10, true_k=2.0, sigma=0.1, seed=13)
        fl = fit_parameters(formula, param_names, obs, ctx)
        assert fl.params['k'] == pytest.approx(2.0, rel=0.01)

    def test_prediction_matches_law(self):
        """After fitting, predict on novel (m, a) → within 5% of true F."""
        formula, ctx, param_names = self._make_fma_formula()
        obs = self._noisy_fma_obs(n=20, true_k=1.0, sigma=0.02, seed=3)
        fl = fit_parameters(formula, param_names, obs, ctx)
        # Novel input not in training
        pred = predict_continuous(fl, {'m': 3.0, 'a': 4.0}, ctx)
        assert pred == pytest.approx(1.0 * 3.0 * 4.0, rel=0.05)

    def test_residual_is_small(self):
        """MSE residual should be well below 1.0 on noisy data."""
        formula, ctx, param_names = self._make_fma_formula()
        obs = self._noisy_fma_obs(n=10, true_k=1.0, sigma=0.05, seed=99)
        fl = fit_parameters(formula, param_names, obs, ctx)
        assert fl.residual < 0.1, f"MSE residual too large: {fl.residual}"

    def test_fit_law_convenience(self):
        """fit_law() passes the schema through correctly."""
        formula, ctx, _ = self._make_fma_formula()
        # Build a SchematicLaw stub
        schema = SchematicLaw(
            pattern=node('mul', var('k'), node('mul', var('m'), var('a'))),
            conclusion=formula,
            params=frozenset({'k'}),
            variables=frozenset({'m', 'a'}),
            evidence=10,
        )
        obs = self._noisy_fma_obs(n=10, true_k=1.0, sigma=0.05, seed=5)
        fl = fit_law(schema, obs, ctx)
        assert fl.schema is schema
        assert fl.params['k'] == pytest.approx(1.0, rel=0.01)


# ---------------------------------------------------------------------------
# 4. Bitter Lesson cage — 10 anonymous symbol tables
# ---------------------------------------------------------------------------

class TestBitterLessonCage:
    """
    Replace 'mul' with an anonymous Unicode operator in both the formula Expr
    and the EvalContext.  The fitted k must agree within 1% across all 10 seeds.
    Variance of k̂ across seeds must be < 5 pp of the true value.
    """

    def _build_anon_formula_and_ctx(self, sym: dict[str, str]):
        """Return (formula, ctx, param_names) with 'mul' replaced by anonymous symbol."""
        anon_mul = sym['mul']
        anon_nid = TOKEN_GRAPH.encode(anon_mul)
        ctx = EvalContext({anon_nid: lambda a, b: a * b})
        # formula = anon_op(k, anon_op(m, a))
        nid_map = {TOKEN_GRAPH.encode('mul'): anon_nid}
        named_formula = node('mul', var('k'), node('mul', var('m'), var('a')))
        anon_formula = rename_expr(named_formula, nid_map)
        return anon_formula, ctx, frozenset({'k'})

    def _noisy_obs(self, n: int, seed: int) -> list:
        rng = random.Random(seed)
        return [
            ({'m': rng.uniform(0.5, 5.0), 'a': rng.uniform(0.1, 10.0)},
             1.0 * rng.uniform(0.5, 5.0) * rng.uniform(0.1, 10.0)
             + rng.gauss(0, 0.05))
            for _ in range(n)
        ]

    def _noisy_fma_obs(self, n: int, seed: int) -> list:
        rng = random.Random(seed)
        return [
            ({'m': m, 'a': a}, 1.0 * m * a + rng.gauss(0, 0.05))
            for m, a in [
                (rng.uniform(0.5, 5.0), rng.uniform(0.1, 10.0))
                for _ in range(n)
            ]
        ]

    def test_cage_10_seeds_within_1_percent(self):
        """Fitted k must be within 1% of 1.0 for all 10 anonymous symbol seeds."""
        obs = self._noisy_fma_obs(10, seed=42)  # same observations for all seeds
        for seed in range(10):
            sym = _fresh_sym(['mul'], seed)
            formula, ctx, param_names = self._build_anon_formula_and_ctx(sym)
            fl = fit_parameters(formula, param_names, obs, ctx)
            assert fl.params['k'] == pytest.approx(1.0, rel=0.01), (
                f"seed {seed}: k={fl.params['k']:.6f} not within 1% of 1.0"
            )

    def test_cage_zero_variance(self):
        """Variance of fitted k across 10 seeds < 5e-4 (well below 5% of 1.0)."""
        obs = self._noisy_fma_obs(10, seed=7)
        fitted_ks = []
        for seed in range(10):
            sym = _fresh_sym(['mul'], seed)
            formula, ctx, param_names = self._build_anon_formula_and_ctx(sym)
            fl = fit_parameters(formula, param_names, obs, ctx)
            fitted_ks.append(fl.params['k'])

        mean_k = sum(fitted_ks) / len(fitted_ks)
        variance = sum((k - mean_k) ** 2 for k in fitted_ks) / len(fitted_ks)
        assert variance < 5e-4, (
            f"k variance across seeds too high: {variance:.2e}. "
            f"Fitted values: {[f'{k:.6f}' for k in fitted_ks]}"
        )

    def test_cage_prediction_same_across_seeds(self):
        """Predictions for novel input agree within 1% across all 10 seeds."""
        obs = self._noisy_fma_obs(15, seed=99)
        predictions = []
        for seed in range(10):
            sym = _fresh_sym(['mul'], seed)
            formula, ctx, param_names = self._build_anon_formula_and_ctx(sym)
            fl = fit_parameters(formula, param_names, obs, ctx)
            pred = predict_continuous(fl, {'m': 3.0, 'a': 4.0}, ctx)
            predictions.append(pred)

        mean_pred = sum(predictions) / len(predictions)
        for i, p in enumerate(predictions):
            assert p == pytest.approx(mean_pred, rel=0.01), (
                f"seed {i}: prediction {p:.4f} differs from mean {mean_pred:.4f}"
            )


# ---------------------------------------------------------------------------
# 5. Defect probe — unit confusion
# ---------------------------------------------------------------------------

class TestDefectProbe:
    """
    Defect probe: a system that stores QuantityNode values as raw floats
    (ignoring unit_id) will:
      (a) Equate QuantityNode(1.0, m) with QuantityNode(1.0, mm) — WRONG
      (b) Fit different k values for the metres stream vs the mm stream — WRONG

    A correct system:
      (a) QuantityNode(1.0, m) != QuantityNode(1.0, mm)  — distinct unit_ids
      (b) Fit k_metres ≈ 100 N/m; k_mm ≈ 0.1 N/mm without conversion
          After converting mm → m, fit k_metres_from_mm ≈ 100 N/m  — same!

    The probe runs with anonymous unit NodeIds (not strings 'm'/'mm') so
    string matching on "m" cannot distinguish them.
    """

    def _spring_obs_metres(self) -> list:
        """F = 100 * x_metres: 5 observations with no noise."""
        return [(({'x': float(i)}, 100.0 * i)) for i in range(1, 6)]

    def _spring_obs_mm(self) -> list:
        """F = 100 * x_metres = 0.1 * x_mm: same law in mm."""
        return [(({'x': float(i) * 1000}, 100.0 * i)) for i in range(1, 6)]

    def _spring_formula(self) -> tuple:
        mul_nid = TOKEN_GRAPH.encode('mul')
        ctx = EvalContext({mul_nid: lambda a, b: a * b})
        formula = node('mul', var('k'), var('x'))
        return formula, ctx

    # (a) Identity check
    def test_quantity_node_distinct_units(self):
        m_nid  = TOKEN_GRAPH.encode('__unit_m_probe')
        mm_nid = TOKEN_GRAPH.encode('__unit_mm_probe')
        assert QuantityNode(1.0, m_nid) != QuantityNode(1.0, mm_nid)

    def test_quantity_node_anonymous_units_distinct(self):
        """Anonymous unit NodeIds that encode different tokens must still be distinct."""
        # Use arbitrary unique symbols for units
        sym = _fresh_sym(['unit_a', 'unit_b'], seed=0)
        a_nid = TOKEN_GRAPH.encode(sym['unit_a'])
        b_nid = TOKEN_GRAPH.encode(sym['unit_b'])
        assert a_nid != b_nid, "The anonymous unit NodeIds must differ"
        assert QuantityNode(1.0, a_nid) != QuantityNode(1.0, b_nid)

    # (b) Fitting without unit conversion gives wrong k for mm data
    def test_fit_mm_stream_without_conversion_gives_wrong_k(self):
        """Fitting the mm stream with the raw float values gives k ≈ 0.1, not 100."""
        formula, ctx = self._spring_formula()
        obs_mm = self._spring_obs_mm()
        fl = fit_parameters(formula, frozenset({'k'}), obs_mm, ctx)
        # Should be ~0.1 (0.1 N/mm), NOT 100 N/m
        assert fl.params['k'] == pytest.approx(0.1, rel=0.01), (
            f"Without unit conversion, k should be 0.1 N/mm, got {fl.params['k']:.4f}"
        )

    def test_fit_m_stream_gives_correct_k(self):
        """Fitting the metres stream gives k ≈ 100 N/m."""
        formula, ctx = self._spring_formula()
        obs_m = self._spring_obs_metres()
        fl = fit_parameters(formula, frozenset({'k'}), obs_m, ctx)
        assert fl.params['k'] == pytest.approx(100.0, rel=0.01)

    # (c) With unit conversion, both streams give the same k
    def test_unit_conversion_reconciles_streams(self):
        """After converting mm → m, k from mm stream ≈ k from m stream."""
        mg = _mg()
        m_nid  = TOKEN_GRAPH.encode('__unit_m_probe')
        mm_nid = TOKEN_GRAPH.encode('__unit_mm_probe')
        register_unit_conversion(mg, mm_nid, m_nid, factor=0.001)

        formula, ctx = self._spring_formula()

        # Convert mm observations to metres before fitting
        obs_mm = self._spring_obs_mm()  # x in mm
        obs_mm_converted = [
            ({'x': x_mm * 0.001}, F)          # multiply by factor 0.001
            for (inp, F) in obs_mm
            for x_mm in [inp['x']]
        ]
        fl_converted = fit_parameters(formula, frozenset({'k'}),
                                      obs_mm_converted, ctx)
        assert fl_converted.params['k'] == pytest.approx(100.0, rel=0.01), (
            f"After conversion, k should be 100 N/m, got {fl_converted.params['k']:.4f}"
        )

    # (d) Anonymous unit NodeIds — string matching on "m"/"mm" cannot help
    def test_anonymous_units_still_distinct(self):
        """
        Replace 'm' and 'mm' with anonymous Unicode unit labels.
        QuantityNode identity is still by NodeId, not by string.
        """
        for seed in range(10):
            sym = _fresh_sym(['unit_metres', 'unit_mm'], seed)
            m_nid  = TOKEN_GRAPH.encode(sym['unit_metres'])
            mm_nid = TOKEN_GRAPH.encode(sym['unit_mm'])
            # The two NodeIds must be distinct (different tokens → different NodeIds)
            assert m_nid != mm_nid
            q_m  = QuantityNode(1.0, m_nid)
            q_mm = QuantityNode(1.0, mm_nid)
            assert q_m != q_mm, (
                f"seed {seed}: QuantityNodes with different anonymous unit NodeIds "
                f"must not be equal"
            )

    def test_anonymous_fitting_same_wrong_k_for_mm(self):
        """
        With anonymous 'mul' operator: k from mm stream is still ~0.1, not 100.
        The anonymous symbols don't change the numerical result.
        """
        for seed in range(10):
            sym = _fresh_sym(['mul'], seed)
            anon_nid = TOKEN_GRAPH.encode(sym['mul'])
            ctx = EvalContext({anon_nid: lambda a, b: a * b})
            nid_map = {TOKEN_GRAPH.encode('mul'): anon_nid}
            formula = rename_expr(node('mul', var('k'), var('x')), nid_map)

            obs_mm = self._spring_obs_mm()
            fl = fit_parameters(formula, frozenset({'k'}), obs_mm, ctx)
            assert fl.params['k'] == pytest.approx(0.1, rel=0.01), (
                f"seed {seed}: expected k≈0.1 for mm stream, got {fl.params['k']:.4f}"
            )
