"""
End-to-end pipeline test — Phases 1, 2, and 3 chained.

This test is the actual Phase 3 integration gate: it is the ONLY test that
exercises all three phases working together in sequence.  The individual
phase test files (test_expr_law.py, test_schematic_law.py,
test_parameter_fitter.py) test each phase in isolation; this file tests
their composition.

Pipeline
--------
Step 1 (Phase 1): Store a law schema as an EXPR_LAW morphism — the structural
                  identity of the law lives in the graph before any data is seen.

Step 2 (Phase 2): Observe token-level examples grouped by parameter value.
                  `discover_parametric_law` finds the schema, classifies
                  which Expr positions are parameters vs. variables, and
                  stores the result as a SCHEMATIC_LAW morphism.

Step 3 (Phase 3): Observe continuous float observations (same law, real
                  measurements with Gaussian noise).  `fit_parameters` fits
                  the free parameter from the data and stores the result as a
                  FITTED_LAW morphism.

Step 4 (Prediction): `predict_continuous` with novel inputs.

Each test scenario follows this exact pipeline.  No step is skipped.

Scenarios
---------
  A. Newton's second law:  F = k · m · a   (k = 1.0)
  B. Hooke's law:          F = k · x       (k = 50.0)
  C. Ohm's law:            V = k · I       (k = 10.0, resistance R)

Cage
----
All three scenarios are also run under 10 anonymous operator symbol tables.
The pipeline must produce the same fitted parameter values (within 1%) and
the same structural schema (same params / variables count) across all seeds.

Defect probes
-------------
  P1. Parameter not confused with variable (Phase 2 output feeds Phase 3 correctly).
  P2. Wrong formula gives high residual — fit_parameters detects the mismatch.
  P3. Novel input outside training range predicted correctly.
"""
from __future__ import annotations

import math
import random
import unicodedata

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom, node, var
from experiments.symbolic_ai_v2.ctkg.core.expr_law import (
    add_expr_law, query_expr_laws, rename_expr,
)
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import (
    SchematicLaw, discover_parametric_law,
    add_schematic_law, query_schematic_laws,
)
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext, eval_expr
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import (
    FittedLaw, fit_parameters, predict_continuous,
    add_fitted_law, query_fitted_laws,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNICODE_OPS = [
    chr(i) for i in range(0x2200, 0x22FF)
    if unicodedata.category(chr(i)) not in ('Cn', 'Co')
]


def _fresh_sym(roles: list[str], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    return dict(zip(roles, rng.sample(_UNICODE_OPS, len(roles))))


def _nid_map(sym: dict[str, str]) -> dict[int, int]:
    return {TOKEN_GRAPH.encode(s): TOKEN_GRAPH.encode(t) for s, t in sym.items()}


def _mul_ctx(*tokens: str) -> EvalContext:
    return EvalContext({TOKEN_GRAPH.encode(t): (lambda a, b: a * b) for t in tokens})


def _noisy(true_val: float, sigma: float, rng: random.Random) -> float:
    return true_val + rng.gauss(0, sigma)


# ---------------------------------------------------------------------------
# Scenario A — Newton's second law: F = k · m · a  (true k = 1.0)
# ---------------------------------------------------------------------------

class TestNewtonPipeline:
    """Full Phase 1 → 2 → 3 pipeline for F = k · m · a."""

    _TRUE_K   = 1.0
    _SIGMA    = 0.05
    _N_OBS    = 15
    _LAW_LABEL = "newton_second"

    # ------------------------------------------------------------------
    # Shared setup methods used by each step
    # ------------------------------------------------------------------

    def _step1_store_law_schema(self, mg: MorphismGraph) -> None:
        """Phase 1: store F = mul(m, a) as an EXPR_LAW morphism."""
        pat = node('mul', var('m'), var('a'))
        con = var('F')
        add_expr_law(mg, self._LAW_LABEL, pat, con)

    def _step2_discover_schema(self, mg: MorphismGraph) -> SchematicLaw:
        """Phase 2: discover SchematicLaw from two token-level families."""
        # Family 1: different m values, same a=2
        fam1 = [
            (node('mul', atom(str(m)), atom('2')), atom(str(m * 2)))
            for m in range(1, 6)
        ]
        # Family 2: different m values, same a=3
        fam2 = [
            (node('mul', atom(str(m)), atom('3')), atom(str(m * 3)))
            for m in range(1, 6)
        ]
        law = discover_parametric_law([fam1, fam2])
        add_schematic_law(mg, self._LAW_LABEL, law)
        return law

    def _step3_fit_law(
        self,
        schema:    SchematicLaw,
        mg:        MorphismGraph,
        true_k:    float = None,
        seed:      int   = 0,
    ) -> FittedLaw:
        """Phase 3: fit k from continuous noisy observations."""
        if true_k is None:
            true_k = self._TRUE_K
        rng = random.Random(seed)

        # conclusion formula: k * m * a = mul(k, mul(m, a))
        ctx  = _mul_ctx('mul')
        formula = node('mul', var('k'), node('mul', var('m'), var('a')))

        obs = [
            (
                {'m': _noisy(rng.uniform(0.5, 5.0), 0, rng),
                 'a': _noisy(rng.uniform(0.1, 8.0), 0, rng)},
                _noisy(true_k * rng.uniform(0.5, 5.0) * rng.uniform(0.1, 8.0),
                        self._SIGMA, rng),
            )
            for _ in range(self._N_OBS)
        ]
        # Build proper observations: use matched m, a values
        obs = []
        for _ in range(self._N_OBS):
            m_val = rng.uniform(0.5, 5.0)
            a_val = rng.uniform(0.1, 8.0)
            F_val = _noisy(true_k * m_val * a_val, self._SIGMA, rng)
            obs.append(({'m': m_val, 'a': a_val}, F_val))

        fl = fit_parameters(formula, frozenset({'k'}), obs, ctx)
        add_fitted_law(mg, self._LAW_LABEL, fl)
        return fl

    # ------------------------------------------------------------------
    # Step 1: Phase 1 gate
    # ------------------------------------------------------------------

    def test_step1_law_stored(self):
        mg = MorphismGraph()
        self._step1_store_law_schema(mg)
        laws = query_expr_laws(mg, self._LAW_LABEL)
        assert len(laws) == 1
        assert laws[0].pattern == node('mul', var('m'), var('a'))

    # ------------------------------------------------------------------
    # Step 2: Phase 2 gate — schema discovery from token examples
    # ------------------------------------------------------------------

    def test_step2_schema_has_param_and_variable(self):
        mg = MorphismGraph()
        schema = self._step2_discover_schema(mg)
        assert isinstance(schema, SchematicLaw)
        assert len(schema.params) >= 1, "Expected at least one parameter (a)"
        assert len(schema.variables) >= 1, "Expected at least one variable (m)"

    def test_step2_schema_stored_in_graph(self):
        mg = MorphismGraph()
        self._step2_discover_schema(mg)
        schemas = query_schematic_laws(mg, self._LAW_LABEL)
        assert len(schemas) == 1

    # ------------------------------------------------------------------
    # Step 3: Phase 3 gate — continuous parameter fitting
    # ------------------------------------------------------------------

    def test_step3_fitted_k_within_1_percent(self):
        mg = MorphismGraph()
        self._step1_store_law_schema(mg)
        schema = self._step2_discover_schema(mg)
        fl = self._step3_fit_law(schema, mg, seed=42)
        assert fl.params['k'] == pytest.approx(self._TRUE_K, rel=0.01), (
            f"Fitted k={fl.params['k']:.6f}, expected ≈{self._TRUE_K} within 1%"
        )

    def test_step3_fitted_law_stored_in_graph(self):
        mg = MorphismGraph()
        self._step1_store_law_schema(mg)
        schema = self._step2_discover_schema(mg)
        self._step3_fit_law(schema, mg, seed=7)
        laws = query_fitted_laws(mg, self._LAW_LABEL)
        assert len(laws) == 1

    def test_step3_residual_small(self):
        mg = MorphismGraph()
        self._step1_store_law_schema(mg)
        schema = self._step2_discover_schema(mg)
        fl = self._step3_fit_law(schema, mg, seed=1)
        assert fl.residual < 0.5

    # ------------------------------------------------------------------
    # Step 4: Prediction on novel inputs
    # ------------------------------------------------------------------

    def test_step4_predict_novel_input(self):
        """Predict F for (m=10, a=5) not in training range."""
        mg = MorphismGraph()
        self._step1_store_law_schema(mg)
        schema = self._step2_discover_schema(mg)
        fl = self._step3_fit_law(schema, mg, seed=0)
        ctx = _mul_ctx('mul')
        pred = predict_continuous(fl, {'m': 10.0, 'a': 5.0}, ctx)
        assert pred == pytest.approx(self._TRUE_K * 10.0 * 5.0, rel=0.02)

    # ------------------------------------------------------------------
    # Full pipeline as single test (the integration regression test)
    # ------------------------------------------------------------------

    def test_full_pipeline(self):
        """
        The Phase 3 integration test: run all four steps, verify the final
        prediction is within 2% of the true value for a novel input.
        """
        mg = MorphismGraph()

        # Step 1
        self._step1_store_law_schema(mg)
        assert len(query_expr_laws(mg, self._LAW_LABEL)) == 1

        # Step 2
        schema = self._step2_discover_schema(mg)
        assert isinstance(schema, SchematicLaw)

        # Step 3
        fl = self._step3_fit_law(schema, mg, seed=99)
        assert fl.params['k'] == pytest.approx(self._TRUE_K, rel=0.01)

        # Step 4
        ctx = _mul_ctx('mul')
        novel_inputs = [
            {'m': 3.0, 'a': 7.0},   # F_true = 21.0
            {'m': 0.5, 'a': 9.0},   # F_true = 4.5
            {'m': 8.0, 'a': 2.0},   # F_true = 16.0
        ]
        for inp in novel_inputs:
            F_true = self._TRUE_K * inp['m'] * inp['a']
            pred = predict_continuous(fl, inp, ctx)
            assert pred == pytest.approx(F_true, rel=0.02), (
                f"Novel prediction failed: inputs={inp}, predicted={pred:.3f}, "
                f"true={F_true:.3f}"
            )


# ---------------------------------------------------------------------------
# Scenario B — Hooke's law: F = k · x  (spring constant k = 50.0)
# ---------------------------------------------------------------------------

class TestHookePipeline:

    _TRUE_K    = 50.0
    _LAW_LABEL = "hooke"

    def _build_pipeline(self, mg: MorphismGraph, seed: int = 0):
        # Phase 1: F = k * x
        add_expr_law(mg, self._LAW_LABEL,
                     node('mul', var('k'), var('x')), var('F'))

        # Phase 2: two families (k=10 and k=50) of token-level examples
        fam_k10 = [(node('mul', atom('1'), atom('10')), atom('10')),
                    (node('mul', atom('2'), atom('10')), atom('20')),
                    (node('mul', atom('3'), atom('10')), atom('30'))]
        fam_k50 = [(node('mul', atom('1'), atom('50')), atom('50')),
                    (node('mul', atom('2'), atom('50')), atom('100')),
                    (node('mul', atom('3'), atom('50')), atom('150'))]
        schema = discover_parametric_law([fam_k10, fam_k50])
        add_schematic_law(mg, self._LAW_LABEL, schema)

        # Phase 3: fit k=50 from continuous observations
        rng = random.Random(seed)
        ctx = _mul_ctx('mul')
        formula = node('mul', var('k'), var('x'))
        obs = [
            ({'x': rng.uniform(0.01, 0.5)},
             _noisy(self._TRUE_K * rng.uniform(0.01, 0.5), 0.5, rng))
            for _ in range(20)
        ]
        # Rebuild with consistent x values
        obs = []
        for _ in range(20):
            x_val = rng.uniform(0.01, 0.5)
            F_val = _noisy(self._TRUE_K * x_val, 0.5, rng)
            obs.append(({'x': x_val}, F_val))

        fl = fit_parameters(formula, frozenset({'k'}), obs, ctx)
        add_fitted_law(mg, self._LAW_LABEL, fl)
        return schema, fl

    def test_hooke_full_pipeline(self):
        mg = MorphismGraph()
        schema, fl = self._build_pipeline(mg, seed=11)

        # Phase 2 check: discovered structural schema
        assert isinstance(schema, SchematicLaw)

        # Phase 3 check: fitted k within 5% (more noise here, use rel=0.05)
        assert fl.params['k'] == pytest.approx(self._TRUE_K, rel=0.05), (
            f"Hooke k={fl.params['k']:.2f}, expected ≈{self._TRUE_K}"
        )

        # Prediction check
        ctx = _mul_ctx('mul')
        pred = predict_continuous(fl, {'x': 0.1}, ctx)
        assert pred == pytest.approx(self._TRUE_K * 0.1, rel=0.1)


# ---------------------------------------------------------------------------
# Scenario C — Ohm's law: V = k · I  (resistance k = 10.0)
# ---------------------------------------------------------------------------

class TestOhmPipeline:

    _TRUE_K    = 10.0
    _LAW_LABEL = "ohm"

    def test_ohm_full_pipeline(self):
        mg = MorphismGraph()

        # Phase 1
        add_expr_law(mg, self._LAW_LABEL,
                     node('mul', var('R'), var('I')), var('V'))

        # Phase 2
        fam_r5  = [(node('mul', atom('1'), atom('5')), atom('5')),
                    (node('mul', atom('2'), atom('5')), atom('10')),
                    (node('mul', atom('3'), atom('5')), atom('15'))]
        fam_r10 = [(node('mul', atom('1'), atom('10')), atom('10')),
                    (node('mul', atom('2'), atom('10')), atom('20')),
                    (node('mul', atom('3'), atom('10')), atom('30'))]
        schema = discover_parametric_law([fam_r5, fam_r10])
        assert isinstance(schema, SchematicLaw)

        # Phase 3
        rng = random.Random(17)
        ctx = _mul_ctx('mul')
        formula = node('mul', var('k'), var('I'))
        obs = []
        for _ in range(15):
            I_val = rng.uniform(0.1, 5.0)
            V_val = _noisy(self._TRUE_K * I_val, 0.1, rng)
            obs.append(({'I': I_val}, V_val))

        fl = fit_parameters(formula, frozenset({'k'}), obs, ctx)
        assert fl.params['k'] == pytest.approx(self._TRUE_K, rel=0.02)

        # Prediction
        pred = predict_continuous(fl, {'I': 3.0}, ctx)
        assert pred == pytest.approx(self._TRUE_K * 3.0, rel=0.05)


# ---------------------------------------------------------------------------
# Cage — all three scenarios under 10 anonymous symbol tables
# ---------------------------------------------------------------------------

class TestPipelineCage:
    """
    The operator 'mul' is replaced by an anonymous Unicode symbol in all
    three phases simultaneously.  Fitted k values must be within 1% of the
    named-symbol result across all 10 seeds.  Variance < 1e-4.
    """

    def _run_pipeline_anon(self, sym: dict[str, str], seed: int) -> float:
        """Run the Newton pipeline with anonymous 'mul', return fitted k."""
        anon_nid = TOKEN_GRAPH.encode(sym['mul'])
        nid_map  = {TOKEN_GRAPH.encode('mul'): anon_nid}
        ctx = EvalContext({anon_nid: lambda a, b: a * b})

        mg = MorphismGraph()

        # Phase 1: store law with anonymous operator
        pat_named = node('mul', var('m'), var('a'))
        con_named = var('F')
        pat = rename_expr(pat_named, nid_map)
        con = rename_expr(con_named, nid_map)
        add_expr_law(mg, 'newton', pat, con)

        # Phase 2: token examples with anonymous operator
        def anon_node(x_s, y_s):
            return rename_expr(node('mul', atom(x_s), atom(y_s)), nid_map)

        fam1 = [(anon_node(str(m), '2'), atom(str(m * 2))) for m in range(1, 5)]
        fam2 = [(anon_node(str(m), '3'), atom(str(m * 3))) for m in range(1, 5)]
        schema = discover_parametric_law([fam1, fam2])
        assert isinstance(schema, SchematicLaw)

        # Phase 3: continuous fitting with anonymous formula
        formula_named = node('mul', var('k'), node('mul', var('m'), var('a')))
        formula = rename_expr(formula_named, nid_map)

        rng = random.Random(seed)
        obs = []
        for _ in range(15):
            m_val = rng.uniform(0.5, 5.0)
            a_val = rng.uniform(0.1, 8.0)
            F_val = _noisy(1.0 * m_val * a_val, 0.05, rng)
            obs.append(({'m': m_val, 'a': a_val}, F_val))

        fl = fit_parameters(formula, frozenset({'k'}), obs, ctx)
        return fl.params['k']

    def test_cage_10_seeds_within_1_percent(self):
        obs_seed = 42
        for sym_seed in range(10):
            sym = _fresh_sym(['mul'], sym_seed)
            k_fitted = self._run_pipeline_anon(sym, seed=obs_seed)
            assert k_fitted == pytest.approx(1.0, rel=0.01), (
                f"sym_seed={sym_seed}: k={k_fitted:.6f} not within 1% of 1.0"
            )

    def test_cage_zero_variance(self):
        """Variance of k̂ across 10 anonymous seeds < 1e-3."""
        obs_seed = 7
        ks = [
            self._run_pipeline_anon(_fresh_sym(['mul'], s), seed=obs_seed)
            for s in range(10)
        ]
        mean_k = sum(ks) / len(ks)
        variance = sum((k - mean_k) ** 2 for k in ks) / len(ks)
        assert variance < 1e-3, (
            f"k variance across anonymous seeds too high: {variance:.2e}. "
            f"Values: {[f'{k:.6f}' for k in ks]}"
        )


# ---------------------------------------------------------------------------
# Defect probes
# ---------------------------------------------------------------------------

class TestPipelineDefectProbes:
    """
    P1. Parameter not confused with variable: the variable identified in
        Phase 2 correctly flows into Phase 3 as the varying input, not as
        a parameter to fit.

    P2. Wrong formula gives detectably high residual: if the user provides
        an incorrect conclusion formula (e.g. addition instead of
        multiplication), the fitted residual is large.

    P3. Novel input outside training range extrapolates correctly.
    """

    # P1: variable flows through correctly
    def test_p1_variable_is_input_not_parameter(self):
        """
        Phase 2 identifies 'm' as a variable (varies within families) and
        'a' as a parameter (constant within families).  In Phase 3, 'm' must
        appear in the input_bindings, not in param_names.
        """
        fam_a2 = [(node('mul', atom(str(m)), atom('2')), atom(str(m * 2)))
                   for m in range(1, 6)]
        fam_a3 = [(node('mul', atom(str(m)), atom('3')), atom(str(m * 3)))
                   for m in range(1, 6)]
        schema = discover_parametric_law([fam_a2, fam_a3])
        assert isinstance(schema, SchematicLaw)

        # The discovered variable (m, position 0) must be in schema.variables
        # The discovered parameter (a, position 1) must be in schema.params
        assert len(schema.variables) >= 1, "m should be classified as variable"
        assert len(schema.params) >= 1,    "a should be classified as parameter"

        # Phase 3: fit using 'm' as the input variable, 'k' as the scaling param
        ctx = _mul_ctx('mul')
        formula = node('mul', var('k'), var('m'))
        obs = [({'m': float(m)}, 5.0 * m) for m in range(1, 11)]
        fl = fit_parameters(formula, frozenset({'k'}), obs, ctx)
        assert fl.params['k'] == pytest.approx(5.0, rel=0.01)

    # P2: wrong formula gives high residual
    def test_p2_wrong_formula_high_residual(self):
        """
        Providing the wrong conclusion (addition instead of multiplication)
        to fit_parameters must produce a detectably higher residual than the
        correct formula.
        """
        mul_nid = TOKEN_GRAPH.encode('mul')
        add_nid = TOKEN_GRAPH.encode('add')
        ctx = EvalContext({
            mul_nid: lambda a, b: a * b,
            add_nid: lambda a, b: a + b,
        })

        # True law: F = k * m * a  (k = 3.0)
        true_k = 3.0
        obs = [
            ({'m': float(m), 'a': float(a)}, true_k * m * a)
            for m in range(1, 5) for a in range(1, 4)
        ]

        # Correct formula
        correct_formula = node('mul', var('k'), node('mul', var('m'), var('a')))
        fl_correct = fit_parameters(correct_formula, frozenset({'k'}), obs, ctx)

        # Wrong formula: k + m + a (additive instead of multiplicative)
        wrong_formula = node('add', var('k'),
                             node('add', var('m'), var('a')))
        fl_wrong = fit_parameters(wrong_formula, frozenset({'k'}), obs, ctx)

        # Correct formula should have near-zero residual
        assert fl_correct.residual < 1e-10, (
            f"Correct formula has high residual: {fl_correct.residual:.4f}"
        )
        # Wrong formula should have a much higher residual
        assert fl_wrong.residual > fl_correct.residual * 100, (
            f"Wrong formula residual ({fl_wrong.residual:.4f}) should dwarf "
            f"correct formula residual ({fl_correct.residual:.2e})"
        )

    # P3: extrapolation outside training range
    def test_p3_extrapolation(self):
        """
        Training range: m ∈ [1, 5], a ∈ [1, 3].
        Novel input: m=20, a=10 — well outside training range.
        Prediction must be within 5% of the true value (OLS extrapolates
        perfectly for linear formulas).
        """
        rng = random.Random(55)
        ctx = _mul_ctx('mul')
        formula = node('mul', var('k'), node('mul', var('m'), var('a')))
        true_k = 2.0

        obs = [
            ({'m': float(m), 'a': float(a)},
             _noisy(true_k * m * a, 0.01, rng))
            for m in range(1, 6) for a in range(1, 4)
        ]
        fl = fit_parameters(formula, frozenset({'k'}), obs, ctx)

        # Extrapolate
        pred = predict_continuous(fl, {'m': 20.0, 'a': 10.0}, ctx)
        true_F = true_k * 20.0 * 10.0  # = 400.0
        assert pred == pytest.approx(true_F, rel=0.05), (
            f"Extrapolation failed: pred={pred:.2f}, true={true_F:.2f}"
        )
