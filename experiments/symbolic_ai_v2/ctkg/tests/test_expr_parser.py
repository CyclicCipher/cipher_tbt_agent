"""Tests for ctkg/core/expr_parser.py — Phase II gate."""
import pytest
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom, node, var
from experiments.symbolic_ai_v2.ctkg.core.expr_parser import (
    ArityTable, TERMINATORS, parse, parse_full,
    unparse, split_on_terminators,
)
from experiments.symbolic_ai_v2.corpus.math_generator import (
    successor_seqs, derivative_seqs,
)


# ---------------------------------------------------------------------------
# Minimal hand-crafted arity table for most tests
# ---------------------------------------------------------------------------

BASE_ARITIES: ArityTable = {
    **{str(d): 0 for d in range(10)},
    'x': 0, 'C': 0, 'half': 0, 'third': 0,
    'succ': 1, 'pred': 1, 'sq': 1, 'sqrt': 1,
    'add': 2, 'sub': 2, 'mul': 2, 'pow': 2,
    'd': 1,
    'int': 1,
}


# ---------------------------------------------------------------------------
# split_on_terminators
# ---------------------------------------------------------------------------

class TestSplit:
    def test_simple(self):
        tokens = ['add', '2', '3', 'eq', '5']
        segs = split_on_terminators(tokens)
        assert segs == [['add', '2', '3'], ['5']]

    def test_multiple_terminators(self):
        tokens = ['pow', '6', '2', 'eq', 'step', '6', 'ans', '3', '6']
        segs = split_on_terminators(tokens)
        assert segs == [['pow', '6', '2'], ['6'], ['3', '6']]

    def test_dx_terminator(self):
        tokens = ['int', 'sq', 'x', 'dx', 'eq', 'mul', 'third', 'pow', 'x', '3', 'C']
        segs = split_on_terminators(tokens)
        assert segs[0] == ['int', 'sq', 'x']

    def test_empty_input(self):
        assert split_on_terminators([]) == []


# ---------------------------------------------------------------------------
# parse / unparse
# ---------------------------------------------------------------------------

class TestParse:
    def test_atom(self):
        assert parse(['5'], BASE_ARITIES) == atom('5')

    def test_unary(self):
        result = parse(['succ', '3'], BASE_ARITIES)
        assert result == node('succ', atom('3'))

    def test_binary(self):
        result = parse(['add', '2', '3'], BASE_ARITIES)
        assert result == node('add', atom('2'), atom('3'))

    def test_nested(self):
        # add(mul(2, x), 3)  — roadmap key test
        result = parse(['add', 'mul', '2', 'x', '3'], BASE_ARITIES)
        assert result == node('add', node('mul', atom('2'), atom('x')), atom('3'))

    def test_unknown_token(self):
        assert parse(['unknownop', '2'], BASE_ARITIES) is None

    def test_trailing_token(self):
        # Trailing '5' makes parse fail (ambiguous)
        assert parse(['add', '2', '3', '5'], BASE_ARITIES) is None

    def test_d_unary(self):
        result = parse(['d', 'pow', 'x', '3'], BASE_ARITIES)
        assert result == node('d', node('pow', atom('x'), atom('3')))

    def test_d_sq(self):
        result = parse(['d', 'sq', 'x'], BASE_ARITIES)
        assert result == node('d', node('sq', atom('x')))

    def test_d_mul(self):
        result = parse(['d', 'mul', '2', 'x'], BASE_ARITIES)
        assert result == node('d', node('mul', atom('2'), atom('x')))


class TestUnparse:
    def test_atom(self):
        assert unparse(atom('5')) == ['5']

    def test_unary(self):
        assert unparse(node('succ', atom('3'))) == ['succ', '3']

    def test_binary(self):
        assert unparse(node('add', atom('2'), atom('3'))) == ['add', '2', '3']

    def test_nested(self):
        e = node('add', node('mul', atom('2'), atom('x')), atom('3'))
        assert unparse(e) == ['add', 'mul', '2', 'x', '3']

    def test_d_pow(self):
        e = node('d', node('pow', atom('x'), atom('3')))
        assert unparse(e) == ['d', 'pow', 'x', '3']


class TestRoundTrip:
    """parse(unparse(expr)) == expr and unparse(parse(seq)) == seq."""

    def test_atom_roundtrip(self):
        seq = ['7']
        e = parse(seq, BASE_ARITIES)
        assert unparse(e) == seq

    def test_nested_roundtrip(self):
        seq = ['d', 'mul', '3', 'pow', 'x', '3']
        e = parse(seq, BASE_ARITIES)
        assert e is not None
        assert unparse(e) == seq

    def test_expr_roundtrip(self):
        e = node('mul', atom('3'), node('sq', atom('x')))
        assert parse(unparse(e), BASE_ARITIES) == e


# ---------------------------------------------------------------------------
# parse_full
# ---------------------------------------------------------------------------

class TestParseFull:
    def test_simple_add(self):
        seq = ['add', '2', '3', 'eq', '5']
        inp, out = parse_full(seq, BASE_ARITIES)
        assert inp == node('add', atom('2'), atom('3'))
        assert out == atom('5')

    def test_derivative_seq(self):
        seq = ['d', 'pow', 'x', '3', 'eq', 'mul', '3', 'sq', 'x']
        inp, out = parse_full(seq, BASE_ARITIES)
        assert inp == node('d', node('pow', atom('x'), atom('3')))
        assert out == node('mul', atom('3'), node('sq', atom('x')))

    def test_int_with_dx(self):
        # int sq x dx eq mul third pow x 3 C
        # 'dx' is terminator; 'C' needs to be in arities
        seq = ['int', 'sq', 'x', 'dx', 'eq', 'mul', 'third', 'pow', 'x', '3', 'C']
        inp, out = parse_full(seq, BASE_ARITIES)
        assert inp == node('int', node('sq', atom('x')))
        # output has trailing 'C' — 'mul third pow x 3' uses 5 tokens, 'C' left over
        # So out should be None (trailing token) OR we parse mul(third, pow(x, 3))
        # and C is left over → None
        # Let's check: 'mul third pow x 3 C' → mul(third, pow(x, 3)) consumes 5 tokens,
        # 'C' is left → parse returns None
        # (This is expected — the constant 'C' in integrals is structurally separate)
        # parse_full takes the LAST segment which is ['mul', 'third', 'pow', 'x', '3', 'C']
        # That's 6 tokens; mul uses 5 → trailing 'C' → None
        assert out is None  # trailing C makes RHS unparseable as a single tree

    def test_multi_digit_output_skipped(self):
        # mul 3 4 eq 1 2 → output is multi-digit '12'; parse fails → None
        seq = ['mul', '3', '4', 'eq', '1', '2']
        inp, out = parse_full(seq, BASE_ARITIES)
        assert inp == node('mul', atom('3'), atom('4'))
        assert out is None   # '1', '2' → parse gets atom('1'), '2' trailing → None



# ---------------------------------------------------------------------------
# Round-trip on real corpus sequences
# ---------------------------------------------------------------------------

class TestCorpusRoundTrip:
    """Roadmap gate: unparse(parse(seq, arities)) == seq for parseable sequences."""

    def _check_roundtrip(self, seqs, arities):
        passed = 0
        total = 0
        for seq in seqs:
            segs = split_on_terminators(seq, TERMINATORS)
            for seg in segs:
                total += 1
                expr = parse(seg, arities)
                if expr is not None:
                    assert unparse(expr) == seg, (
                        f"Round-trip failure: {seg} → {expr} → {unparse(expr)}"
                    )
                    passed += 1
        return passed, total

    def test_derivative_roundtrip(self):
        """Derivatives round-trip with BASE_ARITIES (known arity table)."""
        tr_d, te_d = derivative_seqs()
        all_seqs = tr_d + te_d
        passed, total = self._check_roundtrip(all_seqs, BASE_ARITIES)
        # Expect ≥90% — multi-digit outputs (12, 15) account for ~10% failures
        assert passed / total >= 0.90, (
            f"Round-trip pass rate too low: {passed}/{total} = {passed/total:.1%}"
        )

    def test_successor_roundtrip(self):
        """Single-digit examples (n=0..9) all round-trip; multi-digit are skipped."""
        train, test = successor_seqs()
        all_seqs = train + test
        # Only segments without multi-digit ambiguity parse successfully.
        passed, total = self._check_roundtrip(all_seqs, BASE_ARITIES)
        # At least the n=0..9 cases round-trip: 10 succ + 9 pred = 19 inputs
        # + 19 outputs = 38 segments out of 200. Threshold ≥15%.
        assert passed / total >= 0.15, (
            f"Round-trip pass rate too low: {passed}/{total} = {passed/total:.1%}"
        )
