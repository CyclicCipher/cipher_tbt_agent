"""Tests for SkeletonStore — Phase XXIV.

Gate conditions:
- _normalize_sq correctly rewrites sq tokens
- _extract_skel extracts skeleton and NNO values
- _build_template builds a slot-indexed template
- _reconstruct rebuilds the full output
- _try_synth_slot finds the correct NNOExpr for each pattern
- SkeletonStore.learn + predict correctly generalises the power rule
  (d pow x n → mul n pow x pred(n)) to OOD values of n
"""

from __future__ import annotations

import pytest
from experiments.symbolic_ai_v2.ctkg.learning.skeleton_lambda import (
    SkeletonStore,
    NNOExpr,
    _normalize_sq,
    _extract_skel,
    _build_template,
    _reconstruct,
    _try_synth_slot,
    _eval_nno_expr,
)


# ---------------------------------------------------------------------------
# Minimal NNO helpers for testing
# ---------------------------------------------------------------------------

def _make_succ_pred():
    """Return (succ_map, pred_map) for digits 0-9."""
    succ_map = {str(i): str(i + 1) for i in range(9)}
    pred_map = {v: k for k, v in succ_map.items()}
    return succ_map, pred_map


NNO_SET = frozenset(str(i) for i in range(10))

SUCC_MAP, PRED_MAP = _make_succ_pred()


# ---------------------------------------------------------------------------
# Minimal engine stub
# ---------------------------------------------------------------------------

class _StubEngine:
    """Minimal ComposeEngine stub for testing: supports mul only."""

    def compute_tup(self, op: str, a: tuple, b: tuple) -> tuple | None:
        if op == 'mul' and len(a) == 1 and len(b) == 1:
            try:
                result = int(a[0]) * int(b[0])
                return (str(result),) if result < 10 else None
            except ValueError:
                return None
        if op == 'div' and len(a) == 1 and len(b) == 1:
            try:
                av, bv = int(a[0]), int(b[0])
                if bv == 0 or av % bv != 0:
                    return None
                return (str(av // bv),)
            except ValueError:
                return None
        return None


ENGINE = _StubEngine()


# ---------------------------------------------------------------------------
# _normalize_sq
# ---------------------------------------------------------------------------

class TestNormalizeSq:
    def test_no_sq(self):
        assert _normalize_sq(['d', 'pow', 'x', '3']) == ['d', 'pow', 'x', '3']

    def test_sq_x(self):
        assert _normalize_sq(['d', 'sq', 'x']) == ['d', 'pow', 'x', '2']

    def test_sq_in_output(self):
        assert _normalize_sq(['mul', '3', 'sq', 'x']) == ['mul', '3', 'pow', 'x', '2']

    def test_multiple_sq(self):
        result = _normalize_sq(['sq', 'x', 'sq', 'x'])
        assert result == ['pow', 'x', '2', 'pow', 'x', '2']

    def test_empty(self):
        assert _normalize_sq([]) == []

    def test_sq_at_end_no_operand(self):
        # sq at end with no following token — leave as-is
        assert _normalize_sq(['x', 'sq']) == ['x', 'sq']

    def test_sq_operand_is_digit(self):
        # sq 4 → pow 4 2
        assert _normalize_sq(['sq', '4']) == ['pow', '4', '2']


# ---------------------------------------------------------------------------
# _extract_skel
# ---------------------------------------------------------------------------

class TestExtractSkel:
    def test_all_structural(self):
        skel, nno = _extract_skel(['pow', 'x'], NNO_SET)
        assert skel == ('pow', 'x')
        assert nno == []

    def test_mixed(self):
        skel, nno = _extract_skel(['pow', 'x', '3'], NNO_SET)
        assert skel == ('pow', 'x')
        assert nno == ['3']

    def test_multiple_nno(self):
        skel, nno = _extract_skel(['mul', '2', 'pow', 'x', '3'], NNO_SET)
        assert skel == ('mul', 'pow', 'x')
        assert nno == ['2', '3']

    def test_all_nno(self):
        skel, nno = _extract_skel(['3', '4'], NNO_SET)
        assert skel == ()
        assert nno == ['3', '4']

    def test_empty(self):
        skel, nno = _extract_skel([], NNO_SET)
        assert skel == ()
        assert nno == []


# ---------------------------------------------------------------------------
# _build_template and _reconstruct
# ---------------------------------------------------------------------------

class TestBuildAndReconstruct:
    def test_no_nno(self):
        tmpl, nno = _build_template(['eq', 'mul', 'x'], NNO_SET)
        assert tmpl == ('eq', 'mul', 'x')
        assert nno == []

    def test_single_slot(self):
        tmpl, nno = _build_template(['eq', '3'], NNO_SET)
        assert tmpl == ('eq', 0)
        assert nno == ['3']

    def test_two_slots(self):
        tmpl, nno = _build_template(['step', '3', 'ans', 'mul', '3', 'pow', 'x', '2'], NNO_SET)
        assert tmpl == ('step', 0, 'ans', 'mul', 1, 'pow', 'x', 2)
        assert nno == ['3', '3', '2']

    def test_reconstruct_identity(self):
        tmpl = ('step', 0, 'ans', 'mul', 1, 'pow', 'x', 2)
        out_nno = ['4', '4', '3']
        result = _reconstruct(tmpl, out_nno)
        assert result == ['step', '4', 'ans', 'mul', '4', 'pow', 'x', '3']

    def test_reconstruct_empty_nno(self):
        tmpl = ('eq', 'x')
        result = _reconstruct(tmpl, [])
        assert result == ['eq', 'x']

    def test_reconstruct_missing_slot_returns_empty(self):
        tmpl = ('eq', 0, 'pow', 'x', 1)
        result = _reconstruct(tmpl, ['3'])  # missing slot 1
        assert result == []


# ---------------------------------------------------------------------------
# _try_synth_slot
# ---------------------------------------------------------------------------

class TestTrySynthSlot:
    def test_const(self):
        pairs = [(('3',), '2'), (('4',), '2'), (('5',), '2')]
        expr = _try_synth_slot(pairs, SUCC_MAP, PRED_MAP, ENGINE, n_in=1)
        assert expr is not None
        assert expr.kind == 'const'
        assert expr.args[0] == '2'

    def test_id(self):
        pairs = [(('3',), '3'), (('4',), '4'), (('5',), '5')]
        expr = _try_synth_slot(pairs, SUCC_MAP, PRED_MAP, ENGINE, n_in=1)
        assert expr is not None
        assert expr.kind == 'id'
        assert expr.args[0] == 0

    def test_pred(self):
        pairs = [(('3',), '2'), (('4',), '3'), (('5',), '4')]
        expr = _try_synth_slot(pairs, SUCC_MAP, PRED_MAP, ENGINE, n_in=1)
        assert expr is not None
        assert expr.kind == 'pred'

    def test_succ(self):
        pairs = [(('3',), '4'), (('4',), '5'), (('5',), '6')]
        expr = _try_synth_slot(pairs, SUCC_MAP, PRED_MAP, ENGINE, n_in=1)
        assert expr is not None
        assert expr.kind == 'succ'

    def test_bfm_mul(self):
        # out = in[0] * in[1]
        pairs = [(('2', '3'), '6'), (('3', '2'), '6'), (('2', '4'), '8')]
        expr = _try_synth_slot(pairs, SUCC_MAP, PRED_MAP, ENGINE, n_in=2)
        assert expr is not None
        assert expr.kind == 'bfm'
        assert expr.args[0] == 'mul'

    def test_no_match(self):
        # No simple expression: output is a random string
        pairs = [(('3',), 'x'), (('4',), 'y')]
        expr = _try_synth_slot(pairs, SUCC_MAP, PRED_MAP, ENGINE, n_in=1)
        assert expr is None

    def test_empty_pairs(self):
        expr = _try_synth_slot([], SUCC_MAP, PRED_MAP, ENGINE, n_in=1)
        assert expr is None


# ---------------------------------------------------------------------------
# SkeletonStore — derivative power rule generalisation
# ---------------------------------------------------------------------------

class TestSkeletonStoreDerivatives:
    """Gate test: SkeletonStore learns d(x^n) = n*x^(n-1) from n=2..6 and
    generalises to n=7,8,9 (OOD).

    Training corpus uses eq-format derivative sequences (after sq-normalization
    is applied internally by SkeletonStore):
        d sq x    eq mul 2 x          (→ d pow x 2 eq mul 2 x)
        d pow x 3 eq mul 3 sq x       (→ d pow x 3 eq mul 3 pow x 2)
        d pow x 4 eq mul 4 pow x 3
        d pow x 5 eq mul 5 pow x 4
        d pow x 6 eq mul 6 pow x 5
    """

    @pytest.fixture
    def store(self):
        corpus = [
            ['d', 'sq', 'x', 'eq', 'mul', '2', 'x'],
            ['d', 'pow', 'x', '3', 'eq', 'mul', '3', 'sq', 'x'],
            ['d', 'pow', 'x', '4', 'eq', 'mul', '4', 'pow', 'x', '3'],
            ['d', 'pow', 'x', '5', 'eq', 'mul', '5', 'pow', 'x', '4'],
            ['d', 'pow', 'x', '6', 'eq', 'mul', '6', 'pow', 'x', '5'],
        ]
        s = SkeletonStore()
        s.learn(corpus, SUCC_MAP, ENGINE)
        return s

    def test_predicts_eq_token(self, store):
        # Given: d pow x 3, predict: eq
        prefix = ['d', 'pow', 'x', '3']
        result = store.predict(prefix, SUCC_MAP, ENGINE)
        assert result is not None
        assert 'eq' in result

    def test_predicts_mul_after_eq(self, store):
        # Given: d pow x 4 eq, predict: mul
        prefix = ['d', 'pow', 'x', '4', 'eq']
        result = store.predict(prefix, SUCC_MAP, ENGINE)
        assert result is not None
        assert 'mul' in result

    def test_predicts_coefficient(self, store):
        # Given: d pow x 5 eq mul, predict: 5
        prefix = ['d', 'pow', 'x', '5', 'eq', 'mul']
        result = store.predict(prefix, SUCC_MAP, ENGINE)
        assert result is not None
        assert '5' in result

    def test_ood_n7(self, store):
        """OOD: d pow x 7 should generalise to eq mul 7 pow x 6."""
        # Predict eq
        prefix = ['d', 'pow', 'x', '7']
        r = store.predict(prefix, SUCC_MAP, ENGINE)
        assert r is not None and 'eq' in r, f"Expected 'eq', got {r}"

        # Predict mul after eq
        prefix = ['d', 'pow', 'x', '7', 'eq']
        r = store.predict(prefix, SUCC_MAP, ENGINE)
        assert r is not None and 'mul' in r, f"Expected 'mul', got {r}"

        # Predict 7 (coefficient)
        prefix = ['d', 'pow', 'x', '7', 'eq', 'mul']
        r = store.predict(prefix, SUCC_MAP, ENGINE)
        assert r is not None and '7' in r, f"Expected '7', got {r}"

        # Predict pow
        prefix = ['d', 'pow', 'x', '7', 'eq', 'mul', '7']
        r = store.predict(prefix, SUCC_MAP, ENGINE)
        assert r is not None and 'pow' in r, f"Expected 'pow', got {r}"

        # Predict x
        prefix = ['d', 'pow', 'x', '7', 'eq', 'mul', '7', 'pow']
        r = store.predict(prefix, SUCC_MAP, ENGINE)
        assert r is not None and 'x' in r, f"Expected 'x', got {r}"

        # Predict 6 (exponent = pred(7))
        prefix = ['d', 'pow', 'x', '7', 'eq', 'mul', '7', 'pow', 'x']
        r = store.predict(prefix, SUCC_MAP, ENGINE)
        assert r is not None and '6' in r, f"Expected '6', got {r}"

    def test_sq_input_normalised(self, store):
        """d sq x (training case) should still work via sq-normalization."""
        prefix = ['d', 'sq', 'x', 'eq']
        r = store.predict(prefix, SUCC_MAP, ENGINE)
        assert r is not None and 'mul' in r, f"Expected 'mul', got {r}"

    def test_eos_at_end(self, store):
        """After predicting the full output, predict <eos>."""
        # n=4: full output after eq is 'mul 4 pow x 3'
        prefix = ['d', 'pow', 'x', '4', 'eq', 'mul', '4', 'pow', 'x', '3']
        r = store.predict(prefix, SUCC_MAP, ENGINE)
        assert r is not None and '<eos>' in r, f"Expected '<eos>', got {r}"

    def test_miss_for_unknown_op(self, store):
        result = store.predict(['unk', 'pow', 'x', '3'], SUCC_MAP, ENGINE)
        assert result is None


# ---------------------------------------------------------------------------
# SkeletonStore — integral rule (div-based synthesis)
# ---------------------------------------------------------------------------

class TestSkeletonStoreIntegrals:
    """Gate: int mul rm x dx → step R ans mul R pow x 2 where R = rm/2."""

    @pytest.fixture
    def store(self):
        # int mul 2 x dx step 1 ans mul 1 sq x   (r=1, rm=2)
        # int mul 4 x dx step 2 ans mul 2 sq x   (r=2, rm=4)
        # int mul 6 x dx step 3 ans mul 3 sq x   (r=3, rm=6)
        corpus = [
            ['int', 'mul', '2', 'x', 'dx', 'step', '1', 'ans', 'mul', '1', 'sq', 'x', '<eos>'],
            ['int', 'mul', '4', 'x', 'dx', 'step', '2', 'ans', 'mul', '2', 'sq', 'x', '<eos>'],
            ['int', 'mul', '6', 'x', 'dx', 'step', '3', 'ans', 'mul', '3', 'sq', 'x', '<eos>'],
        ]
        s = SkeletonStore()
        s.learn(corpus, SUCC_MAP, ENGINE)
        return s

    def test_predicts_step_token(self, store):
        prefix = ['int', 'mul', '4', 'x', 'dx']
        r = store.predict(prefix, SUCC_MAP, ENGINE)
        assert r is not None and 'step' in r, f"Expected 'step', got {r}"

    def test_ood_rm8(self, store):
        """OOD: int mul 8 x dx → step 4 ans mul 4 pow x 2."""
        prefix = ['int', 'mul', '8', 'x', 'dx']
        r = store.predict(prefix, SUCC_MAP, ENGINE)
        assert r is not None and 'step' in r, f"step: {r}"

        prefix = ['int', 'mul', '8', 'x', 'dx', 'step']
        r = store.predict(prefix, SUCC_MAP, ENGINE)
        assert r is not None and '4' in r, f"coeff 4: {r}"
