"""Tests for the scratchpad framework.

Run: python -m pytest experiments/scratchpad/test_scratchpad.py -v
"""

import random

import torch

from experiments.scratchpad import (
    Vocab, Step, Problem, Grader, problems_to_tensors, split_problems,
)
from experiments.scratchpad.generators.counting import (
    QueryCountingGenerator, CombinedCountingGenerator,
)
from experiments.scratchpad.generators.arithmetic import (
    SingleDigitArithmeticGenerator,
    TwoDigitSingleArithmeticGenerator,
    TwoDigitArithmeticGenerator,
)


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------

class TestVocab:
    def test_special_tokens_reserved(self):
        v = Vocab()
        assert v.PAD == 0
        assert v.WORK == 1
        assert v.NOTE == 2
        assert v.SEP == 3

    def test_add_and_lookup(self):
        v = Vocab()
        tid = v.add('foo')
        assert v['foo'] == tid
        assert v.decode(tid) == 'foo'

    def test_add_idempotent(self):
        v = Vocab()
        a = v.add('x')
        b = v.add('x')
        assert a == b

    def test_len_grows(self):
        v = Vocab()
        n0 = len(v)
        v.add('new_token')
        assert len(v) == n0 + 1

    def test_contains(self):
        v = Vocab()
        v.add('yes')
        assert 'yes' in v
        assert 'no' not in v

    def test_decode_sequence(self):
        v = Vocab()
        v.add('A')
        v.add('B')
        s = v.decode_sequence([v['A'], v['B']])
        assert s == 'A B'


# ---------------------------------------------------------------------------
# Problem + Step
# ---------------------------------------------------------------------------

class TestProblem:
    def test_work_tokens(self):
        p = Problem(
            question=[10, 11],
            steps=[Step('a', [1, 2]), Step('b', [3])],
        )
        assert p.work_tokens == [1, 2, 3]

    def test_n_result_excludes_work_marker(self):
        p = Problem(
            question=[10],
            steps=[Step('a', [1, 2]), Step('b', [3])],
        )
        assert p.n_result == 3  # just work tokens, no WORK marker

    def test_to_tokens(self):
        v = Vocab()
        v.add('Q')
        v.add('A1')
        v.add('A2')
        p = Problem(
            question=[v['Q']],
            steps=[Step('s', [v['A1'], v['A2']])],
        )
        toks = p.to_tokens(v)
        assert toks == [v['Q'], v.WORK, v['A1'], v['A2']]

    def test_to_tokens_with_notepad(self):
        v = Vocab()
        v.add('Q')
        v.add('W')
        v.add('N')
        p = Problem(
            question=[v['Q']],
            steps=[Step('s', [v['W']])],
            notepad=[v['N']],
        )
        toks = p.to_tokens(v)
        assert toks == [v['Q'], v.WORK, v['W'], v.NOTE, v['N']]


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class TestGrader:
    def setup_method(self):
        self.grader = Grader()

    def test_perfect_score(self):
        p = Problem(
            question=[0],
            steps=[Step('a', [1, 2]), Step('b', [3])],
        )
        r = self.grader.grade(p, [1, 2, 3])
        assert r['overall'] == 1.0
        assert r['exact_match'] == 1

    def test_partial_score(self):
        p = Problem(
            question=[0],
            steps=[Step('a', [1, 2], weight=1.0), Step('b', [3], weight=1.0)],
        )
        r = self.grader.grade(p, [1, 9, 3])  # 1 wrong in step 'a'
        assert r['per_step']['a'] == 0.5
        assert r['per_step']['b'] == 1.0
        assert r['exact_match'] == 0

    def test_all_wrong(self):
        p = Problem(
            question=[0],
            steps=[Step('a', [1, 2])],
        )
        r = self.grader.grade(p, [9, 9])
        assert r['overall'] == 0.0
        assert r['exact_match'] == 0

    def test_ungraded_steps_skipped(self):
        p = Problem(
            question=[0],
            steps=[
                Step('cue', [10], grading='ungraded'),
                Step('ans', [5], weight=1.0),
            ],
        )
        r = self.grader.grade(p, [10, 5])
        assert 'cue' not in r['per_step']
        assert r['per_step']['ans'] == 1.0
        assert r['overall'] == 1.0

    def test_weighted_scoring(self):
        p = Problem(
            question=[0],
            steps=[
                Step('easy', [1], weight=1.0),
                Step('hard', [2], weight=3.0),
            ],
        )
        # easy correct, hard wrong
        r = self.grader.grade(p, [1, 9])
        assert r['per_step']['easy'] == 1.0
        assert r['per_step']['hard'] == 0.0
        assert abs(r['overall'] - 0.25) < 1e-9  # 1*1/(1+3) = 0.25

    def test_output_too_short(self):
        p = Problem(
            question=[0],
            steps=[Step('a', [1, 2, 3])],
        )
        r = self.grader.grade(p, [1])  # only 1 of 3 tokens
        assert r['per_step']['a'] < 1.0


# ---------------------------------------------------------------------------
# Counting generators
# ---------------------------------------------------------------------------

class TestQueryCounting:
    def test_enumerate_count(self):
        g = QueryCountingGenerator()
        assert len(g.enumerate_all()) == 100

    def test_n_result(self):
        v = Vocab()
        g = QueryCountingGenerator()
        probs = g.generate([(3, 2)], 10, v)
        for p in probs:
            assert p.n_result == 2  # query_tok + count

    def test_answer_correctness(self):
        v = Vocab()
        g = QueryCountingGenerator()
        random.seed(0)
        probs = g.generate([(4, 6)], 100, v)
        for p in probs:
            # Find which query was asked
            query_step = p.steps[0]  # 'query' step
            count_step = p.steps[1]  # 'count' step
            query_tok = query_step.tokens[0]
            count_tok = count_step.tokens[0]
            if query_tok == v['DOT']:
                assert count_tok == v['4']
            else:
                assert count_tok == v['6']


class TestCombinedCounting:
    def test_n_result(self):
        v = Vocab()
        g = CombinedCountingGenerator()
        probs = g.generate([(5, 3)], 5, v)
        for p in probs:
            assert p.n_result == 4  # DOT d TEN t

    def test_answer_correctness(self):
        v = Vocab()
        g = CombinedCountingGenerator()
        probs = g.generate([(7, 1)], 5, v)
        for p in probs:
            assert p.steps[1].tokens == [v['7']]  # dot_count
            assert p.steps[3].tokens == [v['1']]  # ten_count


# ---------------------------------------------------------------------------
# Arithmetic generators
# ---------------------------------------------------------------------------

class TestSingleDigit:
    def test_enumerate_count(self):
        g = SingleDigitArithmeticGenerator()
        specs = g.enumerate_all()
        assert len(specs) == 155  # 100 add + 55 sub

    def test_n_result(self):
        v = Vocab()
        g = SingleDigitArithmeticGenerator()
        probs = g.generate([(3, '+', 8)], 5, v)
        for p in probs:
            assert p.n_result == 2  # carry + ones

    def test_addition_correctness(self):
        v = Vocab()
        g = SingleDigitArithmeticGenerator()
        probs = g.generate([(7, '+', 8)], 1, v)  # 7+8=15
        p = probs[0]
        assert p.steps[0].tokens == [v['1']]  # carry
        assert p.steps[1].tokens == [v['5']]  # ones

    def test_subtraction_correctness(self):
        v = Vocab()
        g = SingleDigitArithmeticGenerator()
        probs = g.generate([(9, '-', 4)], 1, v)  # 9-4=5
        p = probs[0]
        assert p.steps[0].tokens == [v['0']]  # carry
        assert p.steps[1].tokens == [v['5']]  # ones


class TestTwoDigitArithmetic:
    def test_enumerate_count(self):
        g = TwoDigitArithmeticGenerator()
        specs = g.enumerate_all()
        assert len(specs) == 12195  # 8100 + 4095

    def test_n_result(self):
        v = Vocab()
        g = TwoDigitArithmeticGenerator()
        probs = g.generate([(23, '+', 48)], 1, v)
        assert probs[0].n_result == 21  # 8+1+8+1+3

    def test_addition_23_48(self):
        """23 + 48 = 71. Ones: 3+8+0=11 (carry 1). Tens: 2+4+1=07."""
        v = Vocab()
        g = TwoDigitArithmeticGenerator()
        probs = g.generate([(23, '+', 48)], 1, v)
        p = probs[0]
        decoded = v.decode_sequence(p.to_tokens(v))
        assert '3 + 8 + 0 = 1 1' in decoded  # ones column
        assert '2 + 4 + 1 = 0 7' in decoded  # tens column
        assert decoded.endswith('0 7 1')       # answer = 071

    def test_subtraction_50_13(self):
        """50 - 13 = 37. Ones: 0-3-0 borrow, digit 7. Tens: 5-1-1=3."""
        v = Vocab()
        g = TwoDigitArithmeticGenerator()
        probs = g.generate([(50, '-', 13)], 1, v)
        p = probs[0]
        decoded = v.decode_sequence(p.to_tokens(v))
        assert '0 - 3 - 0 = 1 7' in decoded  # borrow in ones
        assert '5 - 1 - 1 = 0 3' in decoded  # tens with borrow
        assert decoded.endswith('0 3 7')

    def test_no_carry_addition(self):
        """11 + 22 = 33. No carries."""
        v = Vocab()
        g = TwoDigitArithmeticGenerator()
        probs = g.generate([(11, '+', 22)], 1, v)
        p = probs[0]
        decoded = v.decode_sequence(p.to_tokens(v))
        assert '1 + 2 + 0 = 0 3' in decoded  # ones
        assert '1 + 2 + 0 = 0 3' in decoded  # tens (same by coincidence)
        assert decoded.endswith('0 3 3')

    def test_subtraction_no_borrow(self):
        """85 - 23 = 62. No borrows."""
        v = Vocab()
        g = TwoDigitArithmeticGenerator()
        probs = g.generate([(85, '-', 23)], 1, v)
        p = probs[0]
        decoded = v.decode_sequence(p.to_tokens(v))
        assert '5 - 3 - 0 = 0 2' in decoded
        assert '8 - 2 - 0 = 0 6' in decoded
        assert decoded.endswith('0 6 2')

    def test_max_carry(self):
        """99 + 99 = 198. Both columns carry."""
        v = Vocab()
        g = TwoDigitArithmeticGenerator()
        probs = g.generate([(99, '+', 99)], 1, v)
        p = probs[0]
        decoded = v.decode_sequence(p.to_tokens(v))
        assert '9 + 9 + 0 = 1 8' in decoded  # ones
        assert '9 + 9 + 1 = 1 9' in decoded  # tens
        assert decoded.endswith('1 9 8')


class TestTwoDigitSingle:
    def test_enumerate_count(self):
        g = TwoDigitSingleArithmeticGenerator()
        assert len(g.enumerate_all()) == 1800

    def test_zero_padded_input(self):
        """Second operand is zero-padded: 23 + 8 → 2 3 + 0 8."""
        v = Vocab()
        g = TwoDigitSingleArithmeticGenerator()
        probs = g.generate([(23, '+', 8)], 1, v)
        decoded = v.decode_sequence(probs[0].to_tokens(v))
        assert decoded.startswith('2 3 + 0 8 WORK')


# ---------------------------------------------------------------------------
# Tensor conversion + split
# ---------------------------------------------------------------------------

class TestTensors:
    def test_shape_and_padding(self):
        v = Vocab()
        g = SingleDigitArithmeticGenerator()
        probs = g.generate([(1, '+', 2)], 5, v)
        seqs, nr = problems_to_tensors(probs, v, seq_len=48)
        assert seqs.shape == (5, 48)
        assert nr == 2
        # First tokens should be PAD
        assert seqs[0, 0].item() == v.PAD

    def test_split_problems_disjoint(self):
        v = Vocab()
        g = SingleDigitArithmeticGenerator()
        data = split_problems(g, v, n_train=100, n_test=50, seed=42)
        assert data['train_seqs'].shape[0] == 100
        assert data['test_seqs'].shape[0] == 50
        assert data['n_result_tokens'] == 2
        assert data['n_train_specs'] + data['n_test_specs'] == 155


# ---------------------------------------------------------------------------
# Grader + arithmetic integration
# ---------------------------------------------------------------------------

class TestGraderArithmeticIntegration:
    def test_grade_correct_column_scratchpad(self):
        v = Vocab()
        g = TwoDigitArithmeticGenerator()
        probs = g.generate([(23, '+', 48)], 1, v)
        p = probs[0]
        grader = Grader()
        r = grader.grade(p, p.work_tokens)
        assert r['exact_match'] == 1
        assert r['overall'] == 1.0

    def test_grade_wrong_carry_propagates(self):
        """Wrong carry in ones column should cause wrong tens column too."""
        v = Vocab()
        g = TwoDigitArithmeticGenerator()
        probs = g.generate([(23, '+', 48)], 1, v)
        p = probs[0]
        work = list(p.work_tokens)
        # Flip carry_out in ones column (position 6 in 8-token column)
        # Column: 3 + 8 + 0 = [1] 1 — carry is at index 6
        work[6] = v['0']  # wrong carry
        grader = Grader()
        r = grader.grade(p, work)
        assert r['per_step']['ones_col'] < 1.0
        assert r['exact_match'] == 0
