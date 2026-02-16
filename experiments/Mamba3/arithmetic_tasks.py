"""
Compositional arithmetic task generators for curriculum learning.

Revised curriculum (4 stages):
  Stage 1: Mixed counting  — DOT/TEN tokens in random order, no interleaving
  Stage 2: Single-digit +/- — 2-digit zero-padded output
  Stage 3: Two-digit ± single-digit — 3-digit output (bridge)
  Stage 4: Two-digit ± two-digit — 3-digit output (composition test)

All generators return (n_samples, seq_len) long tensors, left-padded with PAD.
Token encoding: digit d -> token d+1.  See VOCAB below.

Design principles:
  - Every stage has enough problems for meaningful train/test splits
  - No free answers in the input (no interleaved counting)
  - Counting stage uses randomized DOT/TEN order (no run-length shortcut)
  - Each stage teaches a skill that directly composes into later stages
  - Consistent output format within result-size groups
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Vocabulary (25 tokens)
# ---------------------------------------------------------------------------

VOCAB = {
    'PAD': 0,
    '0': 1, '1': 2, '2': 3, '3': 4, '4': 5,
    '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
    '+': 11, '-': 12, '*': 13, '/': 14,
    '=': 15, '>': 16, '<': 17,
    'TRUE': 18, 'FALSE': 19,
    '(': 20, ')': 21,
    'DOT': 22, 'TEN': 23, 'NEXT': 24,
}

VOCAB_SIZE = 25

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
# Stage 1: Mixed Counting (grounding digits in quantities)
# ---------------------------------------------------------------------------

def _enumerate_mixed_counting() -> List[Tuple]:
    """All (dot_count, ten_count) pairs, 0-9 each.  100 total.

    Each pair represents a two-digit number: 10*ten_count + dot_count.
    Grounds digit tokens in concrete quantities.
    """
    return [(d, t) for d in range(10) for t in range(10)]


def generate_mixed_counting(n_samples: int, seq_len: int = 48,
                            problems: Optional[List] = None) -> Tensor:
    """[PAD..., <shuffled DOTs and TENs>, =, tens_digit, ones_digit]

    DOT and TEN tokens are randomly interleaved (no fixed ordering).
    No running counts — the model must actually count each token type.

    Example: 3 DOTs + 2 TENs → DOT TEN DOT TEN DOT = 2 3
    Example: 0 DOTs + 4 TENs → TEN TEN TEN TEN = 4 0
    Example: 5 DOTs + 0 TENs → DOT DOT DOT DOT DOT = 0 5
    Example: 0 DOTs + 0 TENs → = 0 0

    Max length: 9 + 9 + 1 + 2 = 21 tokens.
    """
    if problems is None:
        problems = _enumerate_mixed_counting()
    seqs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    for i in range(n_samples):
        d, t = random.choice(problems)
        input_tokens = [VOCAB['DOT']] * d + [VOCAB['TEN']] * t
        random.shuffle(input_tokens)
        tokens = input_tokens + [VOCAB['='], digit_to_token(t), digit_to_token(d)]
        seqs[i] = torch.tensor(_pad_left(tokens, seq_len))
    return seqs


# ---------------------------------------------------------------------------
# Stage 2: Single-Digit Arithmetic (+, -)
# ---------------------------------------------------------------------------

def _enumerate_single_digit() -> List[Tuple]:
    """All valid single-digit +/- problems.

    Addition: all (a, b) pairs, a+b in 0-18.  100 total.
    Subtraction: a >= b only, a-b in 0-9.  55 total.
    Grand total: 155 problems.
    """
    problems = []
    for a in range(10):
        for b in range(10):
            problems.append((a, '+', b, a + b))
            if a >= b:
                problems.append((a, '-', b, a - b))
    return problems


def generate_single_digit(n_samples: int, seq_len: int = 48,
                          problems: Optional[List] = None) -> Tensor:
    """[PAD..., a, OP, b, =, d1, d2]  (result always 2 digits, zero-padded)."""
    if problems is None:
        problems = _enumerate_single_digit()
    seqs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    for i in range(n_samples):
        a, op, b, res = random.choice(problems)
        tokens = [digit_to_token(a), VOCAB[op], digit_to_token(b), VOCAB['='],
                  digit_to_token(res // 10), digit_to_token(res % 10)]
        seqs[i] = torch.tensor(_pad_left(tokens, seq_len))
    return seqs


# ---------------------------------------------------------------------------
# Stage 3: Two-Digit ± Single-Digit (bridge)
# ---------------------------------------------------------------------------

def _enumerate_two_digit_single() -> List[Tuple]:
    """All valid two-digit ± single-digit problems.

    a in 10-99, b in 0-9.  OP in {+, -}.
    Subtraction always valid (a >= 10 > 9 >= b).
    Total: 90 * 10 * 2 = 1800 problems.
    """
    problems = []
    for a in range(10, 100):
        for b in range(10):
            problems.append((a, '+', b, a + b))
            problems.append((a, '-', b, a - b))
    return problems


def generate_two_digit_single(n_samples: int, seq_len: int = 48,
                              problems: Optional[List] = None) -> Tensor:
    """[PAD..., a1, a2, OP, 0, b, =, r1, r2, r3]  (result always 3 digits).

    Second operand zero-padded to match two-digit format for consistency
    with Stage 4.  Teaches multi-digit I/O while reusing single-digit skill.
    """
    if problems is None:
        problems = _enumerate_two_digit_single()
    seqs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    for i in range(n_samples):
        a, op, b, res = random.choice(problems)
        tokens = [
            digit_to_token(a // 10), digit_to_token(a % 10),
            VOCAB[op],
            digit_to_token(0), digit_to_token(b),
            VOCAB['='],
            digit_to_token(res // 100),
            digit_to_token((res % 100) // 10),
            digit_to_token(res % 10),
        ]
        seqs[i] = torch.tensor(_pad_left(tokens, seq_len))
    return seqs


# ---------------------------------------------------------------------------
# Stage 4: Two-Digit ± Two-Digit (composition test)
# ---------------------------------------------------------------------------

def _enumerate_two_digit() -> List[Tuple]:
    """All valid two-digit +/- problems.

    a, b in 10-99.  Addition: 90*90 = 8100.
    Subtraction: a >= b only, ~4095.
    Total: ~12195 problems.
    """
    problems = []
    for a in range(10, 100):
        for b in range(10, 100):
            problems.append((a, '+', b, a + b))
            if a >= b:
                problems.append((a, '-', b, a - b))
    return problems


def generate_two_digit(n_samples: int, seq_len: int = 48,
                       problems: Optional[List] = None) -> Tensor:
    """[PAD..., a1, a2, OP, b1, b2, =, r1, r2, r3]  (result always 3 digits)."""
    if problems is None:
        problems = _enumerate_two_digit()
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
# Stage registry and data splitting
# ---------------------------------------------------------------------------

STAGE_CONFIG = {
    1: dict(name='mixed_counting',   enumerate=_enumerate_mixed_counting,   generate=generate_mixed_counting,   n_result=2),
    2: dict(name='single_digit',     enumerate=_enumerate_single_digit,     generate=generate_single_digit,     n_result=2),
    3: dict(name='two_digit_single', enumerate=_enumerate_two_digit_single, generate=generate_two_digit_single, n_result=3),
    4: dict(name='two_digit',        enumerate=_enumerate_two_digit,        generate=generate_two_digit,        n_result=3),
}


MIN_PROBLEMS_FOR_SPLIT = 30  # below this, use all problems for both train and test


def get_stage_data(stage: int, n_train: int = 5000, n_test: int = 1000,
                   test_fraction: float = 0.2, seq_len: int = 48,
                   seed: int = 42) -> Dict:
    """Generate train/test data with held-out operand combinations.

    The problem space is enumerated, shuffled with a fixed seed, then split
    so test operand combinations are *never* seen during training.

    For stages with very few problems (< MIN_PROBLEMS_FOR_SPLIT), ALL
    problems are used for both train and test.  These foundational stages
    exist to be overlearned building blocks — the meaningful generalization
    test happens at later stages.

    Returns dict with train_seqs, test_seqs, n_result_tokens, vocab_size,
    n_train_problems, n_test_problems, stage.
    """
    cfg = STAGE_CONFIG[stage]
    all_problems = cfg['enumerate']()

    rng = random.Random(seed)
    shuffled = list(all_problems)
    rng.shuffle(shuffled)

    if len(shuffled) < MIN_PROBLEMS_FOR_SPLIT:
        # Too few problems for a meaningful held-out split.
        # Train and test on the full problem set.
        train_problems = shuffled
        test_problems = shuffled
    else:
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
