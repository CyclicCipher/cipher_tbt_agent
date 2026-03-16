"""
Tests for Phase XXII: Probability monad and enrichment over [0, 1].

Covers:
  - RelationRule.confidence property: total_obs=0 → 1.0, n_match/total
  - RelationRule.evaluate() returns dict[str, float] not Optional[str]
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
# BFM fixture
# ---------------------------------------------------------------------------

def _make_bfm():
    bfm = {'add': {}, 'mul': {}}
    for a in range(10):
        for b in range(10):
            if a + b < 10:
                bfm['add'][(str(a), str(b))] = str(a + b)
            if a * b < 10:
                bfm['mul'][(str(a), str(b))] = str(a * b)
    return bfm


BFM = _make_bfm()


# ---------------------------------------------------------------------------
# RelationRule.confidence
# ---------------------------------------------------------------------------

class TestRelationRuleConfidence:
    def test_default_confidence_is_one(self):
        rr = RelationRule(output_role='eq', op_name='add', arg1='p0', arg2='p1')
        assert rr.confidence == 1.0

    def test_confidence_zero_total_obs(self):
        rr = RelationRule(output_role='eq', op_name='add', arg1='p0', arg2='p1',
                          evidence=5, total_obs=0)
        assert rr.confidence == 1.0

    def test_confidence_full_match(self):
        rr = RelationRule(output_role='eq', op_name='add', arg1='p0', arg2='p1',
                          evidence=10, total_obs=10)
        assert rr.confidence == pytest.approx(1.0)

    def test_confidence_partial_match(self):
        rr = RelationRule(output_role='eq', op_name='add', arg1='p0', arg2='p1',
                          evidence=7, total_obs=10)
        assert rr.confidence == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# RelationRule.evaluate() returns dict[str, float]
# ---------------------------------------------------------------------------

class TestRelationRuleEvaluate:
    def _rule(self, evidence=10, total_obs=10):
        return RelationRule(
            output_role='eq', op_name='add', arg1='p0', arg2='p1',
            evidence=evidence, total_obs=total_obs,
        )

    def test_returns_dict(self):
        rr = self._rule()
        result = rr.evaluate({'p0': '2', 'p1': '3'}, BFM)
        assert isinstance(result, dict)

    def test_hit_returns_correct_value(self):
        rr = self._rule()
        result = rr.evaluate({'p0': '2', 'p1': '3'}, BFM)
        assert '5' in result

    def test_hit_confidence_value(self):
        rr = self._rule(evidence=8, total_obs=10)
        result = rr.evaluate({'p0': '2', 'p1': '3'}, BFM)
        assert result == {'5': pytest.approx(0.8)}

    def test_miss_returns_empty_dict(self):
        rr = self._rule()
        # 9+9=18 not in BFM
        result = rr.evaluate({'p0': '9', 'p1': '9'}, BFM)
        assert result == {}

    def test_missing_arg_returns_empty_dict(self):
        rr = self._rule()
        result = rr.evaluate({'p0': '2'}, BFM)  # p1 missing
        assert result == {}

    def test_default_confidence_one(self):
        rr = RelationRule(output_role='eq', op_name='add', arg1='p0', arg2='p1',
                          evidence=5)   # total_obs=0 → confidence=1.0
        result = rr.evaluate({'p0': '1', 'p1': '2'}, BFM)
        assert result == {'3': pytest.approx(1.0)}


# ---------------------------------------------------------------------------
# LetStep.confidence
# ---------------------------------------------------------------------------

class TestLetStepConfidence:
    def test_default_confidence_is_one(self):
        step = LetStep('ans', node('add', var('p0'), var('p1')))
        assert step.confidence == 1.0

    def test_custom_confidence(self):
        step = LetStep('ans', node('add', var('p0'), var('p1')), confidence=0.8)
        assert step.confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# eval_term: Kleisli confidence propagation
# ---------------------------------------------------------------------------

def _make_linear_term(conf_step=1.0, conf_ans=1.0):
    """λp0 p1 p2. let step=mul(p0,p2) in let ans=add(step,p1)."""
    return LambdaTerm(
        op='linear_eval',
        params=['p0', 'p1', 'p2'],
        steps=[
            LetStep('step', node('mul', var('p0'), var('p2')), confidence=conf_step),
            LetStep('ans', node('add', var('step'), var('p1')), confidence=conf_ans),
        ],
        output_delims=['step', 'ans'],
        evidence=10,
    )


class TestEvalTermKleisli:
    def test_delimiter_before_step_has_full_confidence(self):
        # Before any step is evaluated, acc_conf = 1.0
        term = _make_linear_term(conf_step=0.8, conf_ans=0.9)
        result = eval_term(term, ['2', '1', '3'], BFM, [])
        # delimiter 'step' returned before step expression evaluated
        assert result == {'step': pytest.approx(1.0)}

    def test_step_value_multiplied_by_step_confidence(self):
        # After consuming 'step': acc_conf = conf_step = 0.8
        term = _make_linear_term(conf_step=0.8, conf_ans=0.9)
        result = eval_term(term, ['2', '1', '3'], BFM, ['step'])
        assert result == {'6': pytest.approx(0.8)}

    def test_ans_delimiter_product_of_both(self):
        # After consuming 'step', '6': delimiter 'ans' returned with acc_conf
        # = conf_step (ans delimiter doesn't multiply yet — step evaluated, ans not yet)
        term = _make_linear_term(conf_step=0.8, conf_ans=0.9)
        result = eval_term(term, ['2', '1', '3'], BFM, ['step', '6'])
        assert result == {'ans': pytest.approx(0.8)}

    def test_ans_value_product_of_both_confidences(self):
        # After 'step','6','ans': acc_conf = conf_step * conf_ans = 0.8 * 0.9
        term = _make_linear_term(conf_step=0.8, conf_ans=0.9)
        result = eval_term(term, ['2', '1', '3'], BFM, ['step', '6', 'ans'])
        assert result == {'7': pytest.approx(0.72)}

    def test_eos_has_full_product(self):
        term = _make_linear_term(conf_step=0.8, conf_ans=0.9)
        result = eval_term(term, ['2', '1', '3'], BFM, ['step', '6', 'ans', '7'])
        assert result == {'<eos>': pytest.approx(0.72)}

    def test_unit_confidence_unchanged(self):
        # Default confidence=1.0: results are still probability 1.0
        term = _make_linear_term()
        result = eval_term(term, ['2', '1', '3'], BFM, ['step'])
        assert result == {'6': pytest.approx(1.0)}


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
        rules = discover_relation_rules(rels, BFM)
        rules_by_op = {'add': rules}
        result = predict_from_relation_rules(['add', '2', '3'], store, rules_by_op, BFM)
        assert result is not None
        assert 'eq' in result
        assert '5' in result

    def test_predict_alternatives_returns_weighted_list(self):
        store, rels = _make_store_and_rels()
        rules = discover_relation_rules(rels, BFM)
        rules_by_op = {'add': rules}
        alts = predict_alternatives_from_rules(['add', '1', '2'], store, rules_by_op, BFM)
        assert isinstance(alts, list)
        assert len(alts) > 0
        # Each alternative is (token_list, weight)
        for tok_list, w in alts:
            assert isinstance(tok_list, list)
            assert w > 0

    def test_discover_relation_rules_sets_total_obs(self):
        _, rels = _make_store_and_rels()
        rules = discover_relation_rules(rels, BFM)
        assert rules
        for rule in rules:
            assert rule.total_obs > 0
            assert rule.confidence <= 1.0
            assert rule.confidence > 0.0
