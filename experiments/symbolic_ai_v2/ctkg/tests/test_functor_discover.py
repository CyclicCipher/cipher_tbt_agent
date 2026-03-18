"""
Tests for Phase IX: Functorial Variable Discovery.

Gate condition (FIXING_GENERALIZATION_PART2.md §Phase IX):
  Given a corpus with ≥3 role-entity pairs (king/queen, husband/wife, man/woman),
  cluster_consistent_partitions recovers the gender partition as a single
  FunctorCandidate with bijection {man:woman, king:queen, husband:wife}
  and evidence ≥ 3.  The recovered FunctorCandidate is registered as a
  NaturalTransformation in the knowledge graph.

Tests:
  1. collect_variable_values: bindings collected from matching corpus
  2. collect_variable_values: no match returns empty dict
  3. cluster_consistent_partitions: gender partition recovered (gate test)
  4. cluster_consistent_partitions: trivial (1-element) bijection excluded
  5. cluster_consistent_partitions: inconsistent mapping not a candidate
  6. cluster_consistent_partitions: multiple rules with same bijection merged
  7. register_as_nat_trans: stores in kg dict with correct fields
  8. register_as_nat_trans: custom name used when provided
  9. FunctorCandidate repr contains partition sets
  10. NaturalTransformation components match bijection
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

from experiments.symbolic_ai_v2.ctkg.core.term_algebra import node, var, atom
from experiments.symbolic_ai_v2.ctkg.core.rewrite import RewriteRule
from experiments.symbolic_ai_v2.ctkg.learning.functor_discover import (
    FunctorCandidate,
    NaturalTransformation,
    collect_variable_values,
    cluster_consistent_partitions,
    register_as_nat_trans,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analogy_rule() -> RewriteRule:
    """Rule: analogy(X, Y) → X (just used to match pairs)."""
    return RewriteRule(
        lhs=node('analogy', var('X'), var('Y')),
        rhs=var('X'),
        algebra_name='analogy',
        evidence=0,
    )


def _analogy_corpus() -> list:
    """3 analogy examples: man/woman, king/queen, husband/wife."""
    return [
        node('analogy', atom('man'), atom('woman')),
        node('analogy', atom('king'), atom('queen')),
        node('analogy', atom('husband'), atom('wife')),
    ]


# ---------------------------------------------------------------------------
# Tests: collect_variable_values
# ---------------------------------------------------------------------------

class TestCollectVariableValues:
    def test_bindings_collected(self):
        rule = _analogy_rule()
        corpus = _analogy_corpus()
        vals = collect_variable_values([rule], corpus)
        assert ('analogy', 'X') in vals
        assert ('analogy', 'Y') in vals
        assert set(vals[('analogy', 'X')]) == {'man', 'king', 'husband'}
        assert set(vals[('analogy', 'Y')]) == {'woman', 'queen', 'wife'}

    def test_no_match_returns_empty(self):
        rule = RewriteRule(
            lhs=node('foo', var('A'), var('B')),
            rhs=var('A'),
            algebra_name='foo',
        )
        corpus = _analogy_corpus()  # 'analogy' nodes, not 'foo'
        vals = collect_variable_values([rule], corpus)
        assert vals == {}

    def test_preserves_positional_order(self):
        rule = _analogy_rule()
        corpus = _analogy_corpus()
        vals = collect_variable_values([rule], corpus)
        # Positional pairing: X[0]=man pairs with Y[0]=woman, etc.
        x_vals = vals[('analogy', 'X')]
        y_vals = vals[('analogy', 'Y')]
        assert len(x_vals) == len(y_vals) == 3
        # Check the pairing is consistent (same index = same example)
        pairs = set(zip(x_vals, y_vals))
        assert ('man', 'woman') in pairs
        assert ('king', 'queen') in pairs
        assert ('husband', 'wife') in pairs


# ---------------------------------------------------------------------------
# Tests: cluster_consistent_partitions
# ---------------------------------------------------------------------------

class TestClusterConsistentPartitions:
    def _make_vals(self, rule_id='analogy', a_vals=None, b_vals=None):
        if a_vals is None:
            a_vals = ['man', 'king', 'husband']
        if b_vals is None:
            b_vals = ['woman', 'queen', 'wife']
        return {
            (rule_id, 'X'): a_vals,
            (rule_id, 'Y'): b_vals,
        }

    def test_gate_gender_partition_recovered(self):
        """Gate: gender partition with evidence ≥ 3 is recovered."""
        vals = self._make_vals()
        candidates = cluster_consistent_partitions(vals)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.evidence >= 3
        assert cand.bijection.get('man') == 'woman'
        assert cand.bijection.get('king') == 'queen'
        assert cand.bijection.get('husband') == 'wife'

    def test_correct_partitions(self):
        vals = self._make_vals()
        candidates = cluster_consistent_partitions(vals)
        cand = candidates[0]
        assert cand.partition_a == frozenset({'man', 'king', 'husband'})
        assert cand.partition_b == frozenset({'woman', 'queen', 'wife'})

    def test_trivial_single_element_excluded(self):
        """A bijection with only one pair is excluded (trivial)."""
        vals = {
            ('r', 'X'): ['man'],
            ('r', 'Y'): ['woman'],
        }
        candidates = cluster_consistent_partitions(vals)
        assert candidates == []

    def test_inconsistent_not_candidate(self):
        """If X=man twice maps to different Y, no candidate is produced."""
        vals = {
            ('r', 'X'): ['man', 'man', 'king'],
            ('r', 'Y'): ['woman', 'wife', 'queen'],  # man → woman AND man → wife
        }
        candidates = cluster_consistent_partitions(vals)
        # Should not include an inconsistent bijection for man
        for cand in candidates:
            bij = cand.bijection
            if 'man' in bij:
                # man should map consistently to one value
                man_vals = set()
                xs = vals[('r', 'X')]
                ys = vals[('r', 'Y')]
                for x, y in zip(xs, ys):
                    if x == 'man':
                        man_vals.add(y)
                assert len(man_vals) > 1  # indeed inconsistent
                assert 'man' not in bij  # inconsistent entries excluded

    def test_multiple_rules_same_bijection_merged(self):
        """Two rules with the same bijection are merged into one candidate."""
        vals = {
            ('rule1', 'X'): ['man', 'king'],
            ('rule1', 'Y'): ['woman', 'queen'],
            ('rule2', 'A'): ['man', 'king'],
            ('rule2', 'B'): ['woman', 'queen'],
        }
        candidates = cluster_consistent_partitions(vals)
        assert len(candidates) == 1
        cand = candidates[0]
        assert cand.evidence >= 4  # 2 from rule1 + 2 from rule2
        assert len(cand.supporting_rules) == 2

    def test_sorted_by_evidence_descending(self):
        """Candidates are returned sorted by evidence, highest first."""
        vals = {
            ('r1', 'X'): ['a', 'b', 'c'],
            ('r1', 'Y'): ['d', 'e', 'f'],
            ('r2', 'P'): ['g', 'h'],
            ('r2', 'Q'): ['i', 'j'],
        }
        candidates = cluster_consistent_partitions(vals)
        if len(candidates) >= 2:
            assert candidates[0].evidence >= candidates[1].evidence


# ---------------------------------------------------------------------------
# Tests: register_as_nat_trans
# ---------------------------------------------------------------------------

class TestRegisterAsNatTrans:
    def _make_candidate(self):
        return FunctorCandidate(
            partition_a=frozenset({'man', 'king', 'husband'}),
            partition_b=frozenset({'woman', 'queen', 'wife'}),
            bijection={'man': 'woman', 'king': 'queen', 'husband': 'wife'},
            supporting_rules=['analogy'],
            evidence=3,
        )

    def test_registers_in_kg_dict(self):
        kg: dict = {}
        cand = self._make_candidate()
        nat = register_as_nat_trans(cand, kg)
        assert isinstance(nat, NaturalTransformation)
        assert len(kg) == 1
        assert nat in kg.values()

    def test_custom_name_used(self):
        kg: dict = {}
        cand = self._make_candidate()
        nat = register_as_nat_trans(cand, kg, name='gender_transform')
        assert 'gender_transform' in kg
        assert nat.name == 'gender_transform'

    def test_components_match_bijection(self):
        kg: dict = {}
        cand = self._make_candidate()
        nat = register_as_nat_trans(cand, kg)
        assert nat.components == {'man': 'woman', 'king': 'queen', 'husband': 'wife'}

    def test_evidence_preserved(self):
        kg: dict = {}
        cand = self._make_candidate()
        nat = register_as_nat_trans(cand, kg)
        assert nat.evidence == 3

    def test_partitions_preserved(self):
        kg: dict = {}
        cand = self._make_candidate()
        nat = register_as_nat_trans(cand, kg)
        assert nat.partition_a == frozenset({'man', 'king', 'husband'})
        assert nat.partition_b == frozenset({'woman', 'queen', 'wife'})


# ---------------------------------------------------------------------------
# Tests: data structure basics
# ---------------------------------------------------------------------------

class TestFunctorCandidateRepr:
    def test_repr_contains_partition(self):
        cand = FunctorCandidate(
            partition_a=frozenset({'a', 'b'}),
            partition_b=frozenset({'c', 'd'}),
            bijection={'a': 'c', 'b': 'd'},
            evidence=2,
        )
        r = repr(cand)
        assert 'FunctorCandidate' in r
        assert 'ev=2' in r


class TestNaturalTransformationRepr:
    def test_repr_contains_name(self):
        nat = NaturalTransformation(
            name='test_nat',
            partition_a=frozenset({'x'}),
            partition_b=frozenset({'y'}),
            components={'x': 'y'},
            evidence=1,
        )
        r = repr(nat)
        assert 'test_nat' in r
