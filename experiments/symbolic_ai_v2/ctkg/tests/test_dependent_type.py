"""
Tests for Phase XXI: Dependent type system.

Covers:
  - _walk_chain: zero-detection, cyclic chain, ordering
  - infer_token_types: NNO ordinals, carry tokens, structural tokens, unknown
  - TypeTerm.is_compatible_with: same/different tags, ordinal matching
  - types_compatible_under_bijection: anonymization theorem
  - RelationRule.arg1_type / arg2_type / output_type assignment via type_context
  - discover_relation_rules with type_context assigns NNO_DIGIT type tags

Total: 28 tests.
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

from experiments.symbolic_ai_v2.ctkg.core.dependent_type import (
    TypeTerm,
    NNO_DIGIT,
    NNO_CARRY,
    STRUCTURAL,
    UNKNOWN,
    _walk_chain,
    infer_token_types,
    types_compatible_under_bijection,
    token_type,
    rule_type_tag,
)
from experiments.symbolic_ai_v2.ctkg.learning.relation_store import (
    RelationRule,
    discover_relation_rules,
    Relation,
)


# Anonymization bijection from anon_math_benchmark
DIGIT_PERM = {'0': 'g', '1': 'b', '2': 'h', '3': 'e', '4': 'c',
              '5': 'f', '6': 'a', '7': 'i', '8': 'd', '9': 'j'}

SUCC_STD = {str(i): str(i + 1) for i in range(9)}   # 0→1, ..., 8→9
SUCC_ANON = {DIGIT_PERM[str(i)]: DIGIT_PERM[str(i + 1)] for i in range(9)}


# ---------------------------------------------------------------------------
# _walk_chain
# ---------------------------------------------------------------------------

class TestWalkChain:
    def test_linear_chain_starts_at_zero(self):
        chain = _walk_chain(SUCC_STD)
        assert chain[0] == '0'

    def test_linear_chain_ordered(self):
        chain = _walk_chain(SUCC_STD)
        assert chain == [str(i) for i in range(10)]

    def test_single_element(self):
        chain = _walk_chain({'a': 'b'})
        assert chain == ['a', 'b']

    def test_cyclic_chain_length(self):
        # Fully cyclic: no element without a predecessor
        cyc = {'0': '1', '1': '2', '2': '0'}
        chain = _walk_chain(cyc)
        assert len(chain) == 3


# ---------------------------------------------------------------------------
# infer_token_types
# ---------------------------------------------------------------------------

class TestInferTokenTypes:
    def test_nno_digits_assigned(self):
        types = infer_token_types(SUCC_STD)
        for i in range(10):
            assert types[str(i)].tag == 'NNO_DIGIT'

    def test_ordinals_correct(self):
        types = infer_token_types(SUCC_STD)
        for i in range(10):
            assert types[str(i)].ordinal == i

    def test_structural_token(self):
        types = infer_token_types(SUCC_STD)
        assert types['step'].tag == 'STRUCTURAL'
        assert types['ans'].tag == 'STRUCTURAL'

    def test_carry_token_not_in_chain(self):
        # '0' and '1' ARE in the standard chain, so they get NNO_DIGIT
        types = infer_token_types(SUCC_STD)
        assert types['0'].tag == 'NNO_DIGIT'
        assert types['1'].tag == 'NNO_DIGIT'

    def test_custom_carry_not_in_chain(self):
        # Custom carry token not present in the chain
        types = infer_token_types({'a': 'b', 'b': 'c'},
                                  carry_tokens=frozenset({'x'}))
        assert types['x'].tag == 'NNO_CARRY'

    def test_unknown_for_unrecognized_token(self):
        types = infer_token_types(SUCC_STD)
        assert token_type('zzz', types) == UNKNOWN

    def test_anonymized_chain_gets_same_ordinals(self):
        types_anon = infer_token_types(SUCC_ANON)
        for i in range(9):   # succ chain 0→1...8→9 only, 9 has no successor
            anon_tok = DIGIT_PERM[str(i)]
            assert types_anon[anon_tok].ordinal == i


# ---------------------------------------------------------------------------
# TypeTerm.is_compatible_with
# ---------------------------------------------------------------------------

class TestTypeTermCompatibility:
    def test_same_nno_digit_ordinal(self):
        assert NNO_DIGIT(3).is_compatible_with(NNO_DIGIT(3))

    def test_different_nno_digit_ordinals(self):
        assert not NNO_DIGIT(3).is_compatible_with(NNO_DIGIT(5))

    def test_nno_digit_universal_vs_specific(self):
        # ordinal=None is universally quantified — compatible with any ordinal
        assert NNO_DIGIT(None).is_compatible_with(NNO_DIGIT(7))
        assert NNO_DIGIT(7).is_compatible_with(NNO_DIGIT(None))

    def test_different_tags_incompatible(self):
        assert not NNO_DIGIT(0).is_compatible_with(NNO_CARRY)
        assert not NNO_CARRY.is_compatible_with(STRUCTURAL)

    def test_structural_compatible_with_structural(self):
        assert STRUCTURAL.is_compatible_with(STRUCTURAL)

    def test_unknown_incompatible_with_nno(self):
        assert not UNKNOWN.is_compatible_with(NNO_DIGIT(0))


# ---------------------------------------------------------------------------
# types_compatible_under_bijection (anonymization theorem)
# ---------------------------------------------------------------------------

class TestAnonymizationTheorem:
    def test_standard_and_anon_types_compatible(self):
        types_std = infer_token_types(SUCC_STD)
        types_anon = infer_token_types(SUCC_ANON)
        assert types_compatible_under_bijection(types_std, types_anon, DIGIT_PERM)

    def test_broken_bijection_fails(self):
        types_std = infer_token_types(SUCC_STD)
        types_anon = infer_token_types(SUCC_ANON)
        # Swap two entries in the bijection to break ordinal alignment
        bad_perm = dict(DIGIT_PERM)
        bad_perm['0'], bad_perm['1'] = bad_perm['1'], bad_perm['0']
        assert not types_compatible_under_bijection(types_std, types_anon, bad_perm)

    def test_empty_bijection_trivially_passes(self):
        types_std = infer_token_types(SUCC_STD)
        types_anon = infer_token_types(SUCC_ANON)
        assert types_compatible_under_bijection(types_std, types_anon, {})


# ---------------------------------------------------------------------------
# RelationRule type annotation integration
# ---------------------------------------------------------------------------

def _make_bfm():
    bfm = {'add': {}}
    for a in range(10):
        for b in range(10):
            if a + b < 10:
                bfm['add'][(str(a), str(b))] = str(a + b)
    return bfm


def _make_add_relations():
    """Use RelationStore to parse add sequences so positional roles are assigned."""
    from experiments.symbolic_ai_v2.ctkg.learning.relation_store import RelationStore
    seqs = [['add', str(a), str(b), 'eq', str(a + b)]
            for a in range(4) for b in range(4) if a + b < 10]
    store = RelationStore()
    store.update_batch(seqs)
    return store.get_relations('add')


class TestRelationRuleTypes:
    def test_without_type_context_types_are_none(self):
        bfm = _make_bfm()
        rels = _make_add_relations()
        rules = discover_relation_rules(rels, bfm)
        assert rules
        rule = rules[0]
        assert rule.arg1_type is None
        assert rule.arg2_type is None
        assert rule.output_type is None

    def test_with_type_context_types_assigned(self):
        bfm = _make_bfm()
        rels = _make_add_relations()
        type_ctx = infer_token_types(SUCC_STD)
        rules = discover_relation_rules(rels, bfm, type_context=type_ctx)
        assert rules
        rule = rules[0]
        assert rule.arg1_type is not None
        assert rule.arg1_type.tag == 'NNO_DIGIT'
        assert rule.output_type is not None
        assert rule.output_type.tag == 'NNO_DIGIT'

    def test_type_ordinals_universally_quantified(self):
        # Rule type annotations have ordinal=None (universally quantified)
        bfm = _make_bfm()
        rels = _make_add_relations()
        type_ctx = infer_token_types(SUCC_STD)
        rules = discover_relation_rules(rels, bfm, type_context=type_ctx)
        assert rules
        rule = rules[0]
        # Rules are universally quantified — ordinal=None
        assert rule.arg1_type.ordinal is None
        assert rule.output_type.ordinal is None
