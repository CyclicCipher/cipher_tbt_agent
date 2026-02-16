"""
Compositional arithmetic task generators for curriculum learning.

Each stage builds on the previous:
  Stage 1: Magnitude comparison (digit ordering)
  Stage 2: Successor/predecessor (+1/-1)
  Stage 3: Single-digit arithmetic (+, -, *, /)
  Stage 4: Two-digit arithmetic (compose place value + operation + carry)
  Stage 5: PEMDAS (compose operations with precedence)

All generators return (n_samples, seq_len) long tensors, left-padded with PAD.
Token encoding: digit d -> token d+1.  See VOCAB below.

Design rationale (see CONTINUATION.md):
  - Each stage has near-100% coverage at the algorithmic level
  - Train/test split is by *operand combination*, not random sample
  - Fixed-width results (zero-padded) so all samples in a stage have identical layout
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Vocabulary (22 tokens)
# ---------------------------------------------------------------------------

VOCAB = {
    'PAD': 0,
    '0': 1, '1': 2, '2': 3, '3': 4, '4': 5,
    '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
    '+': 11, '-': 12, '*': 13, '/': 14,
    '=': 15, '>': 16, '<': 17,
    'TRUE': 18, 'FALSE': 19,
    '(': 20, ')': 21,
}

VOCAB_SIZE = 22

ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}


def digit_to_token(d: int) -> int:
    """Convert digit 0-9 to token ID."""
    return d + 1


def decode_tokens(tokens) -> str:
    """Convert token IDs to human-readable string."""
    if isinstance(tokens, Tensor):
        tokens = tokens.tolist()
    return ' '.join(ID_TO_TOKEN.get(t, f'?{t}') for t in tokens)


def _pad_left(tokens: List[int], seq_len: int) -> List[int]:
    """Left-pad with PAD to seq_len."""
    pad_n = seq_len - len(tokens)
    assert pad_n >= 0, f"Tokens too long ({len(tokens)}) for seq_len={seq_len}"
    return [VOCAB['PAD']] * pad_n + tokens


# ---------------------------------------------------------------------------
# Stage 1: Magnitude Comparison
# ---------------------------------------------------------------------------

def _enumerate_comparisons() -> List[Tuple]:
    """All single-digit comparison problems.

    Returns list of (a, cmp_op, b, result_str) where cmp_op in {'>', '<'}
    and result_str in {'TRUE', 'FALSE'}.  200 total (10*10*2).
    """
    problems = []
    for a in range(10):
        for b in range(10):
            problems.append((a, '>', b, 'TRUE' if a > b else 'FALSE'))
            problems.append((a, '<', b, 'TRUE' if a < b else 'FALSE'))
    return problems


def generate_comparison(n_samples: int, seq_len: int = 16,
                        problems: Optional[List] = None) -> Tensor:
    """[PAD..., a, CMP, b, RESULT]  where RESULT in {TRUE, FALSE}."""
    if problems is None:
        problems = _enumerate_comparisons()
    seqs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    for i in range(n_samples):
        a, cmp, b, res = random.choice(problems)
        tokens = [digit_to_token(a), VOCAB[cmp], digit_to_token(b), VOCAB[res]]
        seqs[i] = torch.tensor(_pad_left(tokens, seq_len))
    return seqs


# ---------------------------------------------------------------------------
# Stage 2: Successor / Predecessor
# ---------------------------------------------------------------------------

def _enumerate_successors() -> List[Tuple]:
    """All valid +1/-1 problems (18 total)."""
    problems = []
    for a in range(9):          # 0+1 .. 8+1
        problems.append((a, '+', a + 1))
    for a in range(1, 10):      # 1-1 .. 9-1
        problems.append((a, '-', a - 1))
    return problems


def generate_successor(n_samples: int, seq_len: int = 16,
                       problems: Optional[List] = None) -> Tensor:
    """[PAD..., a, OP, 1, =, result]  where OP in {+, -}."""
    if problems is None:
        problems = _enumerate_successors()
    seqs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    for i in range(n_samples):
        a, op, res = random.choice(problems)
        tokens = [digit_to_token(a), VOCAB[op], digit_to_token(1),
                  VOCAB['='], digit_to_token(res)]
        seqs[i] = torch.tensor(_pad_left(tokens, seq_len))
    return seqs


# ---------------------------------------------------------------------------
# Stage 3: Single-Digit Arithmetic
# ---------------------------------------------------------------------------

def _enumerate_single_digit(ops=None) -> List[Tuple]:
    """All valid single-digit arithmetic problems.

    Returns (a, op_str, b, result_int).  Result can be 0-81.
    """
    if ops is None:
        ops = ['+', '-', '*', '/']
    problems = []
    for a in range(10):
        for b in range(10):
            for op in ops:
                if op == '+':
                    problems.append((a, op, b, a + b))
                elif op == '-' and a >= b:
                    problems.append((a, op, b, a - b))
                elif op == '*':
                    problems.append((a, op, b, a * b))
                elif op == '/' and b > 0 and a % b == 0:
                    problems.append((a, op, b, a // b))
    return problems


def generate_single_digit(n_samples: int, seq_len: int = 16,
                          problems: Optional[List] = None,
                          ops=None) -> Tensor:
    """[PAD..., a, OP, b, =, d1, d2]  (result always 2 digits, zero-padded)."""
    if problems is None:
        problems = _enumerate_single_digit(ops)
    seqs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    for i in range(n_samples):
        a, op, b, res = random.choice(problems)
        tokens = [digit_to_token(a), VOCAB[op], digit_to_token(b), VOCAB['='],
                  digit_to_token(res // 10), digit_to_token(res % 10)]
        seqs[i] = torch.tensor(_pad_left(tokens, seq_len))
    return seqs


# ---------------------------------------------------------------------------
# Stage 4: Two-Digit Arithmetic
# ---------------------------------------------------------------------------

def _enumerate_two_digit(ops=None) -> List[Tuple]:
    """All valid two-digit (10-99) arithmetic problems.

    Returns (a, op_str, b, result_int).  Default ops: +, -.
    """
    if ops is None:
        ops = ['+', '-']
    problems = []
    for a in range(10, 100):
        for b in range(10, 100):
            for op in ops:
                if op == '+':
                    problems.append((a, op, b, a + b))
                elif op == '-' and a >= b:
                    problems.append((a, op, b, a - b))
    return problems


def generate_two_digit(n_samples: int, seq_len: int = 16,
                       problems: Optional[List] = None,
                       ops=None) -> Tensor:
    """[PAD..., a1, a2, OP, b1, b2, =, r1, r2, r3]  (result always 3 digits)."""
    if problems is None:
        problems = _enumerate_two_digit(ops)
    seqs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    for i in range(n_samples):
        a, op, b, res = random.choice(problems)
        tokens = [
            digit_to_token(a // 10), digit_to_token(a % 10),
            VOCAB[op],
            digit_to_token(b // 10), digit_to_token(b % 10),
            VOCAB['='],
            digit_to_token(res // 100),
            digit_to_token((res % 100) // 10),
            digit_to_token(res % 10),
        ]
        seqs[i] = torch.tensor(_pad_left(tokens, seq_len))
    return seqs


# ---------------------------------------------------------------------------
# Stage 5: PEMDAS
# ---------------------------------------------------------------------------

def _apply_op(a: int, op: str, b: int):
    if op == '+': return a + b
    if op == '-': return a - b
    if op == '*': return a * b
    if op == '/' and b != 0 and a % b == 0: return a // b
    return None


def _eval_pemdas(a: int, op1: str, b: int, op2: str, c: int):
    """Evaluate  a op1 b op2 c  with standard precedence (* before +/-)."""
    prec = {'+': 1, '-': 1, '*': 2}
    if prec.get(op2, 0) > prec.get(op1, 0):
        right = _apply_op(b, op2, c)
        return None if right is None else _apply_op(a, op1, right)
    else:
        left = _apply_op(a, op1, b)
        return None if left is None else _apply_op(left, op2, c)


def _enumerate_pemdas() -> List[Tuple]:
    """All valid  a OP1 b OP2 c  problems (digits 1-9, ops +,-,*)."""
    ops = ['+', '-', '*']
    problems = []
    for a in range(1, 10):
        for b in range(1, 10):
            for c in range(1, 10):
                for op1 in ops:
                    for op2 in ops:
                        res = _eval_pemdas(a, op1, b, op2, c)
                        if res is not None and 0 <= res < 1000:
                            problems.append((a, op1, b, op2, c, res))
    return problems


def generate_pemdas(n_samples: int, seq_len: int = 16,
                    problems: Optional[List] = None) -> Tensor:
    """[PAD..., a, OP1, b, OP2, c, =, r1, r2, r3]  (result always 3 digits)."""
    if problems is None:
        problems = _enumerate_pemdas()
    seqs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    for i in range(n_samples):
        a, op1, b, op2, c, res = random.choice(problems)
        tokens = [
            digit_to_token(a), VOCAB[op1],
            digit_to_token(b), VOCAB[op2],
            digit_to_token(c), VOCAB['='],
            digit_to_token(res // 100),
            digit_to_token((res % 100) // 10),
            digit_to_token(res % 10),
        ]
        seqs[i] = torch.tensor(_pad_left(tokens, seq_len))
    return seqs


# ---------------------------------------------------------------------------
# Stage registry and data splitting
# ---------------------------------------------------------------------------

STAGE_CONFIG = {
    1: dict(name='comparison',    enumerate=_enumerate_comparisons, generate=generate_comparison,   n_result=1),
    2: dict(name='successor',     enumerate=_enumerate_successors,  generate=generate_successor,    n_result=1),
    3: dict(name='single_digit',  enumerate=_enumerate_single_digit, generate=generate_single_digit, n_result=2),
    4: dict(name='two_digit',     enumerate=_enumerate_two_digit,   generate=generate_two_digit,    n_result=3),
    5: dict(name='pemdas',        enumerate=_enumerate_pemdas,      generate=generate_pemdas,       n_result=3),
}


def get_stage_data(stage: int, n_train: int = 5000, n_test: int = 1000,
                   test_fraction: float = 0.2, seq_len: int = 16,
                   seed: int = 42) -> Dict:
    """Generate train/test data with held-out operand combinations.

    The problem space is enumerated, shuffled with a fixed seed, then split
    so test operand combinations are *never* seen during training.

    Returns dict with train_seqs, test_seqs, n_result_tokens, vocab_size,
    n_train_problems, n_test_problems, stage.
    """
    cfg = STAGE_CONFIG[stage]
    all_problems = cfg['enumerate']()

    rng = random.Random(seed)
    shuffled = list(all_problems)
    rng.shuffle(shuffled)

    n_held = max(1, int(len(shuffled) * test_fraction))
    test_problems = shuffled[:n_held]
    train_problems = shuffled[n_held:]

    train_seqs = cfg['generate'](n_train, seq_len=seq_len, problems=train_problems)
    test_seqs = cfg['generate'](n_test, seq_len=seq_len, problems=test_problems)

    return dict(
        train_seqs=train_seqs,
        test_seqs=test_seqs,
        n_result_tokens=cfg['n_result'],
        vocab_size=VOCAB_SIZE,
        stage=stage,
        n_train_problems=len(train_problems),
        n_test_problems=len(test_problems),
    )
