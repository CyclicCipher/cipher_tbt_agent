"""Arithmetic problem generators with column-by-column scratchpad.

Stage 3 — Single-digit +/-:  a OP b WORK carry ones  (2 tokens, IS the column)
Stage 4 — Two-digit ± single: column scratchpad with carries
Stage 5 — Two-digit ± two-digit: full column scratchpad

Stages 4-5 scratchpad format (column-by-column, right to left):
    WORK ones_a OP ones_b OP carry_in = carry_out ones_result
         SEP tens_a OP tens_b OP carry_out_prev = carry_out tens_result
         SEP hundreds tens ones

Each column step reuses Stage 3's single-digit format. Carries are explicit
tokens, so the model never needs to hold them in hidden state.

Example addition:    23 + 48 WORK 3 + 8 + 0 = 1 1 SEP 2 + 4 + 1 = 0 7 SEP 0 7 1
Example subtraction: 50 - 03 WORK 0 - 3 - 0 = 1 7 SEP 5 - 0 - 1 = 0 4 SEP 0 4 7

For subtraction borrowing: underflow → borrow=1, digit = raw + 10
  e.g. 0 - 3 = -3 → borrow 1, digit = 10-3 = 7
"""

import random
from typing import Any, List

from ..framework import Problem, ProblemGenerator, Step, Vocab


def _setup_arithmetic_vocab(vocab: Vocab) -> None:
    """Ensure arithmetic tokens exist in vocab."""
    for d in range(10):
        vocab.add(str(d))
    vocab.add('+')
    vocab.add('-')
    vocab.add('=')


def _column_op(a_digit: int, b_digit: int, op: str, carry_in: int,
               vocab: Vocab) -> tuple:
    """Compute one column and return (steps, carry_out, result_digit).

    Returns a list of Steps for this column plus the carry/borrow out.
    """
    if op == '+':
        raw = a_digit + b_digit + carry_in
        carry_out = raw // 10
        result_digit = raw % 10
    else:  # '-'
        raw = a_digit - b_digit - carry_in
        if raw < 0:
            carry_out = 1
            result_digit = raw + 10
        else:
            carry_out = 0
            result_digit = raw

    # Tokens: a_digit OP b_digit OP carry_in = carry_out result_digit
    col_tokens = [
        vocab[str(a_digit)], vocab[op],
        vocab[str(b_digit)], vocab[op],
        vocab[str(carry_in)], vocab['='],
        vocab[str(carry_out)], vocab[str(result_digit)],
    ]

    return col_tokens, carry_out, result_digit


class SingleDigitArithmeticGenerator(ProblemGenerator):
    """Stage 3: single-digit +/-.

    Input: a OP b
    Work: WORK carry ones

    This IS one column operation. The 2-token output (carry, ones) is the
    same format used within each column step in Stages 4-5.

    155 problems: 100 addition + 55 subtraction (a >= b).
    """

    @property
    def name(self) -> str:
        return 'single_digit'

    @property
    def is_fact_stage(self) -> bool:
        return True  # arithmetic facts must be memorized, not generalized

    def enumerate_all(self) -> List[Any]:
        problems = []
        for a in range(10):
            for b in range(10):
                problems.append((a, '+', b))
                if a >= b:
                    problems.append((a, '-', b))
        return problems

    def generate(self, specs: List[Any], n_samples: int,
                 vocab: Vocab) -> List[Problem]:
        _setup_arithmetic_vocab(vocab)
        problems = []
        for _ in range(n_samples):
            a, op, b = random.choice(specs)
            if op == '+':
                res = a + b
            else:
                res = a - b

            carry = res // 10 if op == '+' else 0
            ones = res % 10 if op == '+' else res

            question = [vocab[str(a)], vocab[op], vocab[str(b)]]
            problems.append(Problem(
                question=question,
                steps=[
                    Step('carry', [vocab[str(carry)]], weight=0.5),
                    Step('ones', [vocab[str(ones)]], weight=0.5),
                ],
            ))
        return problems


class TwoDigitSingleArithmeticGenerator(ProblemGenerator):
    """Stage 4: two-digit ± single-digit with column scratchpad.

    Input: a1 a0 OP 0 b0
    Work: WORK a0 OP b0 OP 0 = c1 r0 SEP a1 OP 0 OP c1 = c2 r1 SEP c2 r1 r0

    Second operand zero-padded to 2 digits for consistency with Stage 5.
    1800 problems: 90 * 10 * 2.
    """

    @property
    def name(self) -> str:
        return 'two_digit_single'

    def enumerate_all(self) -> List[Any]:
        problems = []
        for a in range(10, 100):
            for b in range(10):
                problems.append((a, '+', b))
                problems.append((a, '-', b))
        return problems

    def generate(self, specs: List[Any], n_samples: int,
                 vocab: Vocab) -> List[Problem]:
        _setup_arithmetic_vocab(vocab)
        problems = []
        for _ in range(n_samples):
            a, op, b = random.choice(specs)
            a1, a0 = a // 10, a % 10
            b1, b0 = 0, b

            # Question: a1 a0 OP 0 b0
            question = [
                vocab[str(a1)], vocab[str(a0)],
                vocab[op],
                vocab[str(b1)], vocab[str(b0)],
            ]

            # Column 1: ones
            col1_toks, c1, r0 = _column_op(a0, b0, op, 0, vocab)
            # Column 2: tens
            col2_toks, c2, r1 = _column_op(a1, b1, op, c1, vocab)

            # Final answer: c2 r1 r0
            answer_toks = [vocab[str(c2)], vocab[str(r1)], vocab[str(r0)]]

            problems.append(Problem(
                question=question,
                steps=[
                    Step('ones_col', col1_toks, weight=0.25),
                    Step('sep1', [vocab.SEP], grading='ungraded'),
                    Step('tens_col', col2_toks, weight=0.25),
                    Step('sep2', [vocab.SEP], grading='ungraded'),
                    Step('answer', answer_toks, weight=0.5),
                ],
            ))
        return problems


class TwoDigitArithmeticGenerator(ProblemGenerator):
    """Stage 5: two-digit ± two-digit with column scratchpad.

    Input: a1 a0 OP b1 b0
    Work: WORK a0 OP b0 OP 0 = c1 r0 SEP a1 OP b1 OP c1 = c2 r1 SEP c2 r1 r0

    ~12195 problems: 8100 addition + ~4095 subtraction (a >= b).
    """

    @property
    def name(self) -> str:
        return 'two_digit'

    def enumerate_all(self) -> List[Any]:
        problems = []
        for a in range(10, 100):
            for b in range(10, 100):
                problems.append((a, '+', b))
                if a >= b:
                    problems.append((a, '-', b))
        return problems

    def generate(self, specs: List[Any], n_samples: int,
                 vocab: Vocab) -> List[Problem]:
        _setup_arithmetic_vocab(vocab)
        problems = []
        for _ in range(n_samples):
            a, op, b = random.choice(specs)
            a1, a0 = a // 10, a % 10
            b1, b0 = b // 10, b % 10

            question = [
                vocab[str(a1)], vocab[str(a0)],
                vocab[op],
                vocab[str(b1)], vocab[str(b0)],
            ]

            col1_toks, c1, r0 = _column_op(a0, b0, op, 0, vocab)
            col2_toks, c2, r1 = _column_op(a1, b1, op, c1, vocab)
            answer_toks = [vocab[str(c2)], vocab[str(r1)], vocab[str(r0)]]

            problems.append(Problem(
                question=question,
                steps=[
                    Step('ones_col', col1_toks, weight=0.25),
                    Step('sep1', [vocab.SEP], grading='ungraded'),
                    Step('tens_col', col2_toks, weight=0.25),
                    Step('sep2', [vocab.SEP], grading='ungraded'),
                    Step('answer', answer_toks, weight=0.5),
                ],
            ))
        return problems
