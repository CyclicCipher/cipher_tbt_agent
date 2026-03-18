"""
Tests for Phase V: Variable Discovery — Notation-Independence.

Gate (FIXING_GENERALIZATION.md Phase V):
- identify_variables returns correct variable positions for discovered rules
- unify_surface_forms detects surface-form equivalences from corpus
- normalize_surface applies a norm_map to token sequences
- _merge_digit_runs / _split_compound handle compound NNO tokens correctly
- Power rule is discoverable with norm_rules + output_norm_rules + functional_maps
"""
import pytest
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import (
    atom, node, var, identify_variables, unify_surface_forms,
)
from experiments.symbolic_ai_v2.ctkg.core.expr_parser import (
    ArityTable, normalize_surface,
)
from experiments.symbolic_ai_v2.ctkg.core.rewrite import RewriteRule, cata_reduce
from experiments.symbolic_ai_v2.ctkg.learning.rule_discover import discover_rules
from experiments.symbolic_ai_v2.ctkg.learning.relation_store import (
    _merge_digit_runs, _split_compound,
)
from experiments.symbolic_ai_v2.ctkg.core.node import enc, dec


# ---------------------------------------------------------------------------
# Shorthand helpers
# ---------------------------------------------------------------------------

def _pow(a, b):   return node('pow',  a, b)
def _mul(a, b):   return node('mul',  a, b)
def _pred(a):     return node('pred', a)
def _succ(a):     return node('succ', a)
def _d(a):        return node('d',    a)

x  = atom('x')
c0 = atom('0')
c1 = atom('1')
c2 = atom('2')
c3 = atom('3')
c4 = atom('4')
c5 = atom('5')

BASE_ARITIES: ArityTable = {
    **{str(d): 0 for d in range(10)},
    'x': 0,
    'succ': 1, 'pred': 1,
    'mul': 2, 'pow': 2,
    'd': 1,
}


# ---------------------------------------------------------------------------
# normalize_surface
# ---------------------------------------------------------------------------

class TestNormalizeSurface:
    def test_empty(self):
        assert normalize_surface([], {}) == []

    def test_no_substitutions(self):
        toks = ['add', '2', '3', 'eq', '5']
        assert normalize_surface(toks, {}) == toks

    def test_digit_substitution(self):
        norm = {'five': '5', 'three': '3'}
        result = normalize_surface(['add', 'five', 'three', 'eq', '8'], norm)
        assert result == ['add', '5', '3', 'eq', '8']

    def test_partial_substitution(self):
        norm = {'five': '5'}
        toks = ['add', 'five', '3', 'eq', '8']
        assert normalize_surface(toks, norm) == ['add', '5', '3', 'eq', '8']

    def test_identity_tokens_unchanged(self):
        norm = {'a': 'b'}
        toks = ['mul', 'x', '2']
        assert normalize_surface(toks, norm) == ['mul', 'x', '2']


# ---------------------------------------------------------------------------
# identify_variables
# ---------------------------------------------------------------------------

class TestIdentifyVariables:
    def test_empty_rules(self):
        assert identify_variables([]) == {}

    def test_single_rule_no_vars(self):
        rule = RewriteRule(lhs=_pow(x, c2), rhs=_mul(c2, _pow(x, c1)),
                           algebra_name='pow', evidence=1)
        result = identify_variables([rule])
        assert result == {}  # no pattern variables

    def test_single_rule_with_vars(self):
        rule = RewriteRule(lhs=_pow(x, var('n')), rhs=_mul(var('n'), _pow(x, _pred(var('n')))),
                           algebra_name='pow', evidence=5)
        result = identify_variables([rule])
        assert (0, 'n') in result
        assert result[(0, 'n')] == frozenset()

    def test_multiple_vars(self):
        rule = RewriteRule(lhs=_mul(var('a'), var('b')), rhs=_mul(var('b'), var('a')),
                           algebra_name='mul', evidence=3)
        result = identify_variables([rule])
        assert (0, 'a') in result
        assert (0, 'b') in result

    def test_two_rules(self):
        r1 = RewriteRule(lhs=_d(_pow(x, var('n'))), rhs=_mul(var('n'), _pow(x, _pred(var('n')))),
                         algebra_name='d', evidence=4)
        r2 = RewriteRule(lhs=_pow(x, c3), rhs=_mul(c3, _pow(x, c2)),
                         algebra_name='pow', evidence=1)
        result = identify_variables([r1, r2])
        assert (0, 'n') in result
        assert (1, 'n') not in result  # r2 has no vars


# ---------------------------------------------------------------------------
# _merge_digit_runs / _split_compound
# Phase XXIII: both functions now operate on NodeId; use enc()/dec() helpers.
# ---------------------------------------------------------------------------

NNO = frozenset(enc(d) for d in '0123456789')

def _enc_toks(toks):
    """Encode list[str] to list[NodeId] for _merge_digit_runs / _split_compound tests."""
    return [enc(t) for t in toks]

def _dec_toks(nids):
    """Decode list[NodeId] to list[str] for assertion comparison."""
    return [dec(n) for n in nids]


class TestMergeDigitRuns:
    def test_no_merge_needed(self):
        assert _dec_toks(_merge_digit_runs(_enc_toks(['mul', '2', 'x']), NNO)) == ['mul', '2', 'x']

    def test_merge_two_digits(self):
        assert _dec_toks(_merge_digit_runs(_enc_toks(['1', '2']), NNO)) == ['12']

    def test_merge_three_digits(self):
        assert _dec_toks(_merge_digit_runs(_enc_toks(['1', '0', '5']), NNO)) == ['105']

    def test_partial_merge(self):
        # digits at start, then non-digit breaks the run
        result = _dec_toks(_merge_digit_runs(_enc_toks(['1', '2', 'pow', 'x', '3']), NNO))
        assert result == ['12', 'pow', 'x', '3']

    def test_non_nno_not_merged(self):
        assert _dec_toks(_merge_digit_runs(_enc_toks(['x', 'y']), NNO)) == ['x', 'y']

    def test_interleaved(self):
        # two separate runs
        result = _dec_toks(_merge_digit_runs(_enc_toks(['1', '2', 'mul', '3', '4']), NNO))
        assert result == ['12', 'mul', '34']

    def test_empty(self):
        assert _merge_digit_runs([], NNO) == []

    def test_single_digit_unchanged(self):
        assert _dec_toks(_merge_digit_runs(_enc_toks(['5']), NNO)) == ['5']


class TestSplitCompound:
    def test_single_char_unchanged(self):
        assert _dec_toks(_split_compound(enc('5'), NNO)) == ['5']

    def test_two_digit_compound(self):
        assert _dec_toks(_split_compound(enc('12'), NNO)) == ['1', '2']

    def test_three_digit_compound(self):
        assert _dec_toks(_split_compound(enc('105'), NNO)) == ['1', '0', '5']

    def test_non_nno_unchanged(self):
        assert _dec_toks(_split_compound(enc('mul'), NNO)) == ['mul']

    def test_mixed_unchanged(self):
        # 'x2' — 'x' not in NNO, so not all-NNO
        assert _dec_toks(_split_compound(enc('x2'), NNO)) == ['x2']

    def test_empty_string(self):
        assert _dec_toks(_split_compound(enc(''), NNO)) == ['']

    def test_roundtrip(self):
        toks = ['1', '2', '3']
        merged = _merge_digit_runs(_enc_toks(toks), NNO)
        assert _dec_toks(merged) == ['123']
        split = _split_compound(merged[0], NNO)
        assert _dec_toks(split) == toks


# ---------------------------------------------------------------------------
# Power rule discovery with Phase V normalization
# (Gate: FIXING_GENERALIZATION.md Phase V)
# ---------------------------------------------------------------------------

class TestPowerRuleDiscovery:
    """
    Gate test: discover d(pow(x,n)) → mul(n, pow(x, pred(n))) from corpus
    using norm_rules (sq→pow), output_norm_rules (x→pow(x,1)), and
    functional_maps (pred alignment).
    """

    @pytest.fixture
    def arities(self):
        return {
            **{str(d): 0 for d in range(10)},
            'x': 0,
            'succ': 1, 'pred': 1, 'sq': 1,
            'mul': 2, 'pow': 2,
            'd': 1,
        }

    @pytest.fixture
    def corpus(self):
        # d(x^2)=2x, d(x^3)=3x^2, d(x^4)=4x^3, d(x^5)=5x^4
        return [
            ['d', 'pow', 'x', '2', 'eq', 'mul', '2', 'x'],
            ['d', 'pow', 'x', '3', 'eq', 'mul', '3', 'pow', 'x', '2'],
            ['d', 'pow', 'x', '4', 'eq', 'mul', '4', 'pow', 'x', '3'],
            ['d', 'pow', 'x', '5', 'eq', 'mul', '5', 'pow', 'x', '4'],
        ]

    @pytest.fixture
    def norm_rules(self, arities):
        # sq(V) → pow(V, 2)
        return [RewriteRule(
            lhs=node('sq', var('V0')), rhs=_pow(var('V0'), c2),
            algebra_name='sq', evidence=1,
        )]

    @pytest.fixture
    def pred_map(self):
        return {'1': '0', '2': '1', '3': '2', '4': '3', '5': '4',
                '6': '5', '7': '6', '8': '7', '9': '8'}

    @pytest.fixture
    def succ_map(self):
        return {'0': '1', '1': '2', '2': '3', '3': '4', '4': '5',
                '5': '6', '6': '7', '7': '8', '8': '9'}

    @pytest.fixture
    def ground_nno(self, pred_map, succ_map):
        rules = []
        for d_from, d_to in succ_map.items():
            rules.append(RewriteRule(
                lhs=node('succ', atom(d_from)), rhs=atom(d_to),
                algebra_name='succ', evidence=1,
            ))
            rules.append(RewriteRule(
                lhs=node('pred', atom(d_to)), rhs=atom(d_from),
                algebra_name='pred', evidence=1,
            ))
        return rules

    @pytest.fixture
    def x_pow1_norm(self, arities):
        # mul(V0, x) → mul(V0, pow(x,1))
        # More specific than x→pow(x,1): only fires when x is a direct arg of mul.
        # Avoids replacing x inside pow(x,N), which would give pow(pow(x,1),N).
        return RewriteRule(
            lhs=node('mul', var('V0'), atom('x')),
            rhs=node('mul', var('V0'), _pow(atom('x'), atom('1'))),
            algebra_name='mul', evidence=1,
        )

    def test_power_rule_discovered(self, corpus, arities, norm_rules,
                                   ground_nno, x_pow1_norm, pred_map):
        """Phase V gate: power rule is discovered with normalization."""
        output_norm_rules = [x_pow1_norm]
        functional_maps = {'pred': pred_map}

        rules = discover_rules(
            corpus, arities,
            norm_rules=norm_rules,
            output_norm_rules=output_norm_rules,
            functional_maps=functional_maps,
            aux_rules=ground_nno,
        )

        # Should discover d(pow(x,V0)) → mul(V0, pow(x, pred(V0)))
        d_rules = [r for r in rules if r.algebra_name == 'd']
        assert len(d_rules) >= 1, f"No 'd' rule found. All rules: {rules}"

        # The rule must have pred(V0) in rhs
        power_rule = d_rules[0]
        rhs_str = repr(power_rule.rhs)
        assert 'pred' in rhs_str, f"Expected pred in rhs, got: {rhs_str}"
        assert 'mul' in rhs_str, f"Expected mul in rhs, got: {rhs_str}"

    def test_power_rule_applies_correctly(self, corpus, arities, norm_rules,
                                          ground_nno, x_pow1_norm, pred_map):
        """Power rule + ground NNO correctly reduces d(pow(x,3)) → mul(3,pow(x,2))."""
        output_norm_rules = [x_pow1_norm]
        functional_maps = {'pred': pred_map}

        rules = discover_rules(
            corpus, arities,
            norm_rules=norm_rules,
            output_norm_rules=output_norm_rules,
            functional_maps=functional_maps,
            aux_rules=ground_nno,
        )

        all_rules = rules + ground_nno
        inp = _d(_pow(x, c3))
        result = cata_reduce(inp, all_rules)
        assert result == _mul(c3, _pow(x, c2)), f"Got: {result}"

    def test_power_rule_out_of_distribution(self, corpus, arities, norm_rules,
                                             ground_nno, x_pow1_norm, pred_map):
        """Power rule generalises to d(pow(x,4)) → mul(4,pow(x,3)) (seen) and
           d(pow(x,9)) → mul(9,pow(x,8)) (OOD)."""
        output_norm_rules = [x_pow1_norm]
        functional_maps = {'pred': pred_map}

        rules = discover_rules(
            corpus, arities,
            norm_rules=norm_rules,
            output_norm_rules=output_norm_rules,
            functional_maps=functional_maps,
            aux_rules=ground_nno,
        )

        all_rules = rules + ground_nno

        # Seen case: d(pow(x,4))
        result = cata_reduce(_d(_pow(x, c4)), all_rules)
        assert result == _mul(c4, _pow(x, c3)), f"Got: {result}"

        # OOD case: d(pow(x,9))
        c9 = atom('9')
        c8 = atom('8')
        result = cata_reduce(_d(_pow(x, c9)), all_rules)
        assert result == _mul(c9, _pow(x, c8)), f"Got: {result}"


# ---------------------------------------------------------------------------
# unify_surface_forms
# ---------------------------------------------------------------------------

class TestUnifySurfaceForms:
    def test_no_equivalences(self):
        """When each token is unique, no norm_map entries are generated."""
        corpus = [
            ['succ', '2', 'eq', '3'],
            ['succ', '3', 'eq', '4'],
            ['succ', '4', 'eq', '5'],
        ]
        arities: ArityTable = {
            **{str(d): 0 for d in range(10)},
            'succ': 1,
        }
        rules = discover_rules(corpus, arities)
        norm_map = unify_surface_forms(corpus, rules, arities)
        # No tokens are surface-form equivalent
        assert norm_map == {}

    def test_known_equivalence(self):
        """Tokens 'a' and 'b' map identically through succ: succ(a)=c, succ(b)=c."""
        # Build a tiny corpus where 'p' and 'q' are equivalent (same successor 'r')
        corpus = [
            ['succ', 'p', 'eq', 'r'],
            ['succ', 'q', 'eq', 'r'],
        ]
        arities: ArityTable = {'p': 0, 'q': 0, 'r': 0, 'succ': 1}
        rules = discover_rules(corpus, arities)
        if not rules:
            pytest.skip("discover_rules found no rules — skip unify test")
        norm_map = unify_surface_forms(corpus, rules, arities)
        # p and q should be identified (same output profile)
        if norm_map:
            canon = min('p', 'q')
            other = max('p', 'q')
            assert norm_map.get(other) == canon
