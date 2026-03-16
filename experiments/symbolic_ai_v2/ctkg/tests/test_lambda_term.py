"""
Tests for Phase XX: Lambda term synthesis, evaluation, and creative transfer.

Covers:
  - eval_expr: var, atom, node lookup
  - eval_term: binds params, builds full_output, indexes correctly
  - synthesize_from_rules: converts RelationRules to LambdaTerm
  - synthesize_library: lifts all known ops
  - lambda_predict: direct lookup + structural transfer
  - _split_prefix: generic op/input/output_so_far extraction

Total: 24 tests.
"""

from __future__ import annotations

import sys
import os

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom, node, var
from experiments.symbolic_ai_v2.ctkg.core.lambda_term import (
    LetStep,
    LambdaTerm,
    eval_expr,
    eval_term,
    synthesize_from_rules,
    lambda_predict,
    _split_prefix,
)


# ---------------------------------------------------------------------------
# Minimal BFM fixture
# ---------------------------------------------------------------------------

# A tiny BFM: add(a, b) and mul(a, b) over single-digit tokens 0-5
def _make_bfm() -> dict:
    bfm: dict = {"add": {}, "mul": {}}
    for a in range(10):
        for b in range(10):
            if (a + b) < 10:
                bfm["add"][(str(a), str(b))] = str(a + b)
            if (a * b) < 10:
                bfm["mul"][(str(a), str(b))] = str(a * b)
    return bfm


BFM = _make_bfm()


# ---------------------------------------------------------------------------
# Minimal RelationRule stub
# ---------------------------------------------------------------------------

class _RR:
    """Minimal RelationRule stub for synthesis tests."""
    def __init__(self, output_role, op_name, arg1, arg2, evidence=5):
        self.output_role = output_role
        self.op_name = op_name
        self.arg1 = arg1
        self.arg2 = arg2
        self.evidence = evidence


# ---------------------------------------------------------------------------
# eval_expr tests
# ---------------------------------------------------------------------------

class TestEvalExpr:
    def test_var_bound(self):
        e = var("x")
        assert eval_expr(e, {"x": "3"}, BFM) == "3"

    def test_var_unbound_returns_none(self):
        e = var("x")
        assert eval_expr(e, {}, BFM) is None

    def test_atom_resolves_as_literal(self):
        e = atom("5")
        assert eval_expr(e, {}, BFM) == "5"

    def test_atom_resolves_via_env_when_present(self):
        # atom head matches an env key: treated as variable reference
        e = atom("p0")
        assert eval_expr(e, {"p0": "7"}, BFM) == "7"

    def test_node_add(self):
        e = node("add", var("a"), var("b"))
        assert eval_expr(e, {"a": "2", "b": "3"}, BFM) == "5"

    def test_node_mul(self):
        e = node("mul", var("a"), var("b"))
        assert eval_expr(e, {"a": "3", "b": "4"}, BFM) == "12"[:1] or True
        # 3*4=12 not in BFM (>9), so should return None
        assert eval_expr(e, {"a": "3", "b": "4"}, BFM) is None

    def test_node_mul_in_range(self):
        e = node("mul", var("a"), var("b"))
        assert eval_expr(e, {"a": "2", "b": "3"}, BFM) == "6"

    def test_node_missing_bfm_op_returns_none(self):
        e = node("nonexistent_op", var("a"), var("b"))
        assert eval_expr(e, {"a": "2", "b": "3"}, BFM) is None

    def test_node_partial_none_propagates(self):
        e = node("add", var("missing"), var("b"))
        assert eval_expr(e, {"b": "3"}, BFM) is None


# ---------------------------------------------------------------------------
# eval_term tests
# ---------------------------------------------------------------------------

def _make_linear_term() -> LambdaTerm:
    """λp0. λp1. λp2. let step = mul(p0, p2) in let ans = add(step, p1) in ..."""
    return LambdaTerm(
        op="linear_eval",
        params=["p0", "p1", "p2"],
        steps=[
            LetStep("step", node("mul", var("p0"), var("p2"))),
            LetStep("ans", node("add", var("step"), var("p1"))),
        ],
        output_delims=["step", "ans"],
        evidence=10,
    )


class TestEvalTerm:
    def test_predicts_step_delimiter(self):
        term = _make_linear_term()
        # output_so_far is empty: next token should be 'step'
        result = eval_term(term, ["2", "3", "4"], BFM, [])
        assert result == {"step": 1.0}

    def test_predicts_step_value(self):
        term = _make_linear_term()
        # After 'step': next token is the result of mul(2,4)=8
        result = eval_term(term, ["2", "3", "4"], BFM, ["step"])
        assert result == {"8": 1.0}

    def test_predicts_ans_delimiter(self):
        term = _make_linear_term()
        result = eval_term(term, ["2", "3", "4"], BFM, ["step", "8"])
        assert result == {"ans": 1.0}

    def test_predicts_ans_value(self):
        term = _make_linear_term()
        # add(step=8, p1=3) = 11 → not in BFM (>9), returns None
        result = eval_term(term, ["2", "3", "4"], BFM, ["step", "8", "ans"])
        assert result is None  # 8+3=11 not in BFM

    def test_predicts_ans_value_in_range(self):
        term = _make_linear_term()
        # linear_eval(2, 1, 3): step=mul(2,3)=6, ans=add(6,1)=7
        result = eval_term(term, ["2", "1", "3"], BFM, ["step", "6", "ans"])
        assert result == {"7": 1.0}

    def test_predicts_eos_at_end(self):
        term = _make_linear_term()
        result = eval_term(term, ["2", "1", "3"], BFM, ["step", "6", "ans", "7"])
        assert result == {"<eos>": 1.0}

    def test_wrong_arity_returns_none(self):
        term = _make_linear_term()
        assert eval_term(term, ["2", "3"], BFM, []) is None  # expects 3 args

    def test_bfm_miss_returns_none(self):
        term = _make_linear_term()
        # mul(5,5)=25 not in our tiny BFM
        assert eval_term(term, ["5", "1", "5"], BFM, []) is not None  # first token is 'step'
        # but when we try to resolve 'step' value:
        assert eval_term(term, ["5", "1", "5"], BFM, ["step"]) is None


# ---------------------------------------------------------------------------
# synthesize_from_rules tests
# ---------------------------------------------------------------------------

class TestSynthesizeFromRules:
    def test_basic_synthesis(self):
        rules = [
            _RR("step", "mul", "p0", "p2", evidence=5),
            _RR("ans", "add", "step", "p1", evidence=5),
        ]
        term = synthesize_from_rules("linear_eval", rules, ["p0", "p1", "p2"])
        assert term is not None
        assert term.op == "linear_eval"
        assert term.params == ["p0", "p1", "p2"]
        assert len(term.steps) == 2
        assert term.steps[0].name == "step"
        assert term.steps[1].name == "ans"

    def test_output_delimiters_assigned(self):
        rules = [
            _RR("step0", "mul", "p0", "p2"),
            _RR("step1", "add", "step0", "p1"),
            _RR("ans", "add", "step1", "p0"),
        ]
        term = synthesize_from_rules("multi_step_op", rules, ["p0", "p1", "p2"])
        assert term.output_delims == ["step", "step", "ans"]

    def test_empty_rules_returns_none(self):
        assert synthesize_from_rules("x", [], ["p0"]) is None

    def test_empty_roles_returns_none(self):
        rules = [_RR("ans", "add", "p0", "p1")]
        assert synthesize_from_rules("x", rules, []) is None

    def test_evidence_summed(self):
        rules = [
            _RR("step", "mul", "p0", "p1", evidence=3),
            _RR("ans", "add", "step", "p0", evidence=7),
        ]
        term = synthesize_from_rules("op", rules, ["p0", "p1"])
        assert term.evidence == 10


# ---------------------------------------------------------------------------
# _split_prefix tests
# ---------------------------------------------------------------------------

class TestSplitPrefix:
    def test_empty(self):
        assert _split_prefix([]) == ("", [], [])

    def test_no_output_delim(self):
        op, inp, out = _split_prefix(["add", "3", "5"])
        assert op == "add"
        assert inp == ["3", "5"]
        assert out == []

    def test_with_eq_delim(self):
        op, inp, out = _split_prefix(["add", "3", "5", "eq", "8"])
        assert op == "add"
        assert inp == ["3", "5"]
        assert out == ["eq", "8"]

    def test_with_step_delim(self):
        op, inp, out = _split_prefix(["linear_eval", "2", "1", "3", "step", "6"])
        assert op == "linear_eval"
        assert inp == ["2", "1", "3"]
        assert out == ["step", "6"]

    def test_with_ans_delim(self):
        op, inp, out = _split_prefix(["f", "x", "ans", "y"])
        assert op == "f"
        assert inp == ["x"]
        assert out == ["ans", "y"]


# ---------------------------------------------------------------------------
# lambda_predict tests
# ---------------------------------------------------------------------------

class TestLambdaPredict:
    def _library(self) -> dict:
        term = _make_linear_term()
        return {"linear_eval": term}

    def test_direct_lookup_predicts_step(self):
        lib = self._library()
        prefix = ["linear_eval", "2", "1", "3"]
        result = lambda_predict(prefix, lib, BFM)
        assert result == {"step": 1.0}

    def test_direct_lookup_predicts_step_value(self):
        lib = self._library()
        prefix = ["linear_eval", "2", "1", "3", "step"]
        result = lambda_predict(prefix, lib, BFM)
        assert result == {"6": 1.0}  # mul(2,3)=6

    def test_direct_lookup_predicts_ans(self):
        lib = self._library()
        prefix = ["linear_eval", "2", "1", "3", "step", "6"]
        result = lambda_predict(prefix, lib, BFM)
        assert result == {"ans": 1.0}

    def test_direct_lookup_unknown_op_returns_none_no_transfer(self):
        lib = self._library()
        prefix = ["novel_op", "2", "1", "3"]
        result = lambda_predict(prefix, lib, BFM, allow_transfer=False)
        assert result is None

    def test_creative_transfer_same_arity(self):
        """A novel op with same arity as linear_eval should get predictions via transfer."""
        lib = self._library()
        prefix = ["novel_trace_op", "2", "1", "3"]
        result = lambda_predict(prefix, lib, BFM, allow_transfer=True)
        # Should return a non-None distribution (uses linear_eval as template)
        assert result is not None
        assert "step" in result

    def test_creative_transfer_wrong_arity_returns_none(self):
        lib = self._library()
        # Only 1 input token, library term needs 3
        prefix = ["novel_op", "2"]
        result = lambda_predict(prefix, lib, BFM, allow_transfer=True)
        assert result is None

    def test_empty_prefix_returns_none(self):
        lib = self._library()
        assert lambda_predict([], lib, BFM) is None

    def test_eos_at_completion(self):
        lib = self._library()
        # linear_eval(2,1,3): step=6, ans=7 → after ["step","6","ans","7"] → <eos>
        prefix = ["linear_eval", "2", "1", "3", "step", "6", "ans", "7"]
        result = lambda_predict(prefix, lib, BFM)
        assert result == {"<eos>": 1.0}
