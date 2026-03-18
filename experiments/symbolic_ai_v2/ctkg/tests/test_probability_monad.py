"""
Tests for Phase XXII: Probability monad and enrichment over [0, 1].

Covers:
  - RelationRule.confidence property: total_obs=0 → 1.0, n_match/total
  - RelationRule.evaluate() returns dict[tuple[NodeId], float] not Optional[str]
  - evaluate() returns {} on miss (analogous to previous None)
  - evaluate() confidence value matches rule.confidence
  - LetStep.confidence field
  - eval_term: Kleisli confidence propagation (product across steps)
  - eval_term: partial-chain confidence (only consumed steps contribute)
  - predict_from_relation_rules / predict_alternatives_from_rules compatibility

Total: 22 tests.
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

from experiments.symbolic_ai_v2.ctkg.core.term_algebra import node, var
from experiments.symbolic_ai_v2.ctkg.core.node import (
    TOKEN_GRAPH, enc,
    EOS_NODE, STEP_NODE, ANS_NODE,
)
from experiments.symbolic_ai_v2.ctkg.core.lambda_term import (
    LetStep,
    LambdaTerm,
    eval_term,
    lambda_predict,
)
from experiments.symbolic_ai_v2.ctkg.learning.relation_store import (
    RelationRule,
    Relation,
    discover_relation_rules,
    predict_from_relation_rules,
    predict_alternatives_from_rules,
)


# ---------------------------------------------------------------------------
# Engine fixture
# ---------------------------------------------------------------------------

def _make_bfm():
    bfm: dict = {'add': {}, 'mul': {}}
    for a in range(10):
        for b in range(10):
            if a + b < 10:
                bfm['add'][(str(a), str(b))] = str(a + b)
            if a * b < 10:
                bfm['mul'][(str(a), str(b))] = str(a * b)
    return bfm


class _MockEngine:
    def __init__(self, bfm: dict) -> None:
        self._bfm = bfm

    def compute(self, op: str, a: str, b: str):
        result = self._bfm.get(op, {}).get((a, b))
        return (result,) if result is not None else None

    def compute_tup(self, op: str, a_tup: tuple, b_tup: tuple):
        if len(a_tup) == 1 and len(b_tup) == 1:
            return self.compute(op, a_tup[0], b_tup[0])
        return None

    def known_ops(self) -> list:
        return list(self._bfm.keys())


ENGINE = _MockEngine(_make_bfm())


# ---------------------------------------------------------------------------
# RelationRule.confidence
# ---------------------------------------------------------------------------

class TestRelationRuleConfidence:
    def test_default_confidence_is_one(self):
        rr = RelationRule(output_role=enc('eq'), op_name=enc('add'), arg1=enc('p0'), arg2=enc('p1'))
        assert rr.confidence == 1.0

    def test_confidence_zero_total_obs(self):
        rr = RelationRule(output_role=enc('eq'), op_name=enc('add'), arg1=enc('p0'), arg2=enc('p1'),
                          evidence=5, total_obs=0)
        assert rr.confidence == 1.0

    def test_confidence_full_match(self):
        rr = RelationRule(output_role=enc('eq'), op_name=enc('add'), arg1=enc('p0'), arg2=enc('p1'),
                          evidence=10, total_obs=10)
        assert rr.confidence == pytest.approx(1.0)

    def test_confidence_partial_match(self):
        rr = RelationRule(output_role=enc('eq'), op_name=enc('add'), arg1=enc('p0'), arg2=enc('p1'),
                          evidence=7, total_obs=10)
        assert rr.confidence == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# RelationRule.evaluate() returns dict[tuple[NodeId,...], float]
# ---------------------------------------------------------------------------

class TestRelationRuleEvaluate:
    def _rule(self, evidence=10, total_obs=10):
        return RelationRule(
            output_role=enc('eq'), op_name=enc('add'), arg1=enc('p0'), arg2=enc('p1'),
            evidence=evidence, total_obs=total_obs,
        )

    def _rv(self, a, b):
        """Build role_values dict with NodeId keys and single-element tuple values."""
        return {enc('p0'): (enc(a),), enc('p1'): (enc(b),)}

    def test_returns_dict(self):
        rr = self._rule()
        result = rr.evaluate(self._rv('2', '3'), ENGINE)
        assert isinstance(result, dict)

    def test_hit_returns_correct_value(self):
        rr = self._rule()
        result = rr.evaluate(self._rv('2', '3'), ENGINE)
        assert (enc('5'),) in result

    def test_hit_confidence_value(self):
        rr = self._rule(evidence=8, total_obs=10)
        result = rr.evaluate(self._rv('2', '3'), ENGINE)
        assert result == {(enc('5'),): pytest.approx(0.8)}

    def test_miss_returns_empty_dict(self):
        rr = self._rule()
        # 9+9=18 not computable by engine
        result = rr.evaluate(self._rv('9', '9'), ENGINE)
        assert result == {}

    def test_missing_arg_returns_empty_dict(self):
        rr = self._rule()
        result = rr.evaluate({enc('p0'): (enc('2'),)}, ENGINE)  # p1 missing
        assert result == {}

    def test_default_confidence_one(self):
        rr = RelationRule(output_role=enc('eq'), op_name=enc('add'), arg1=enc('p0'), arg2=enc('p1'),
                          evidence=5)   # total_obs=0 → confidence=1.0
        result = rr.evaluate(self._rv('1', '2'), ENGINE)
        assert result == {(enc('3'),): pytest.approx(1.0)}


# ---------------------------------------------------------------------------
# LetStep.confidence
# ---------------------------------------------------------------------------

class TestLetStepConfidence:
    def test_default_confidence_is_one(self):
        step = LetStep(enc('ans'), node('add', var('p0'), var('p1')))
        assert step.confidence == 1.0

    def test_custom_confidence(self):
        step = LetStep(enc('ans'), node('add', var('p0'), var('p1')), confidence=0.8)
        assert step.confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# eval_term: Kleisli confidence propagation
# ---------------------------------------------------------------------------

def _make_linear_term(conf_step=1.0, conf_ans=1.0):
    """λp0 p1 p2. let step=mul(p0,p2) in let ans=add(step,p1)."""
    return LambdaTerm(
        op=enc('linear_eval'),
        params=[enc('p0'), enc('p1'), enc('p2')],
        steps=[
            LetStep(enc('step'), node('mul', var('p0'), var('p2')), confidence=conf_step),
            LetStep(enc('ans'), node('add', var('step'), var('p1')), confidence=conf_ans),
        ],
        output_delims=[STEP_NODE, ANS_NODE],
        evidence=10,
    )


class TestEvalTermKleisli:
    def test_delimiter_before_step_has_full_confidence(self):
        term = _make_linear_term(conf_step=0.8, conf_ans=0.9)
        result = eval_term(term, ['2', '1', '3'], ENGINE, [])
        assert result == {STEP_NODE: pytest.approx(1.0)}

    def test_step_value_multiplied_by_step_confidence(self):
        term = _make_linear_term(conf_step=0.8, conf_ans=0.9)
        result = eval_term(term, ['2', '1', '3'], ENGINE, [STEP_NODE])
        assert result == {enc('6'): pytest.approx(0.8)}

    def test_ans_delimiter_product_of_both(self):
        term = _make_linear_term(conf_step=0.8, conf_ans=0.9)
        result = eval_term(term, ['2', '1', '3'], ENGINE, [STEP_NODE, enc('6')])
        assert result == {ANS_NODE: pytest.approx(0.8)}

    def test_ans_value_product_of_both_confidences(self):
        term = _make_linear_term(conf_step=0.8, conf_ans=0.9)
        result = eval_term(term, ['2', '1', '3'], ENGINE, [STEP_NODE, enc('6'), ANS_NODE])
        assert result == {enc('7'): pytest.approx(0.72)}

    def test_eos_has_full_product(self):
        term = _make_linear_term(conf_step=0.8, conf_ans=0.9)
        result = eval_term(term, ['2', '1', '3'], ENGINE, [STEP_NODE, enc('6'), ANS_NODE, enc('7')])
        assert result == {EOS_NODE: pytest.approx(0.72)}

    def test_unit_confidence_unchanged(self):
        term = _make_linear_term()
        result = eval_term(term, ['2', '1', '3'], ENGINE, [STEP_NODE])
        assert result == {enc('6'): pytest.approx(1.0)}


# ---------------------------------------------------------------------------
# predict_from_relation_rules / predict_alternatives compatibility
# ---------------------------------------------------------------------------

def _make_store_and_rels():
    from experiments.symbolic_ai_v2.ctkg.learning.relation_store import RelationStore
    seqs = [['add', str(a), str(b), 'eq', str(a + b)]
            for a in range(4) for b in range(4) if a + b < 10]
    store = RelationStore()
    store.update_batch(seqs)
    rels = store.get_relations('add')
    return store, rels


class TestRelationStoreProbabilityMonad:
    def test_predict_from_rules_still_works(self):
        store, rels = _make_store_and_rels()
        rules = discover_relation_rules(rels, ENGINE)
        rules_by_op = {enc('add'): rules}
        result = predict_from_relation_rules(['add', '2', '3'], store, rules_by_op, ENGINE)
        assert result is not None
        assert 'eq' in result
        assert '5' in result

    def test_predict_alternatives_returns_weighted_list(self):
        store, rels = _make_store_and_rels()
        rules = discover_relation_rules(rels, ENGINE)
        rules_by_op = {enc('add'): rules}
        alts = predict_alternatives_from_rules(['add', '1', '2'], store, rules_by_op, ENGINE)
        assert isinstance(alts, list)
        assert len(alts) > 0
        for tok_list, w in alts:
            assert isinstance(tok_list, list)
            assert w > 0

    def test_discover_relation_rules_sets_total_obs(self):
        _, rels = _make_store_and_rels()
        rules = discover_relation_rules(rels, ENGINE)
        assert rules
        for rule in rules:
            assert rule.total_obs > 0
            assert rule.confidence <= 1.0
            assert rule.confidence > 0.0
