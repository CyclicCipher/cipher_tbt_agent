"""
Synthetic data generation for JEPA energy reasoning experiments.

Stage 1: Structured sequences for masked prediction (JEPA backbone validation)
Stage 2: Pattern induction for few-shot rule discovery (Langevin validation)
"""

import random
from typing import List, Optional, Callable, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def generate_mask(batch_size: int, seq_len: int, mask_ratio: float = 0.2,
                  device: torch.device = torch.device('cpu')) -> Tensor:
    """Generate random binary mask. True = masked (to be predicted).

    Args:
        batch_size: Number of samples.
        seq_len: Sequence length.
        mask_ratio: Fraction of positions to mask.
        device: Target device.

    Returns:
        mask: (batch_size, seq_len) boolean tensor.
    """
    n_mask = max(1, int(seq_len * mask_ratio))
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        idx = torch.randperm(seq_len, device=device)[:n_mask]
        mask[i, idx] = True
    return mask


def generate_causal_mask(batch_size: int, seq_len: int, mask_ratio: float = 0.2,
                         device: torch.device = torch.device('cpu')) -> Tensor:
    """Generate mask biased toward later positions (causal-friendly).

    Mamba3 is causal: position t only sees 0..t-1. Masking early
    positions gives the predictor almost no context. This mask samples
    from the latter half of the sequence, ensuring each masked position
    has ample causal context.

    Args:
        batch_size: Number of samples.
        seq_len: Sequence length.
        mask_ratio: Fraction of positions to mask.
        device: Target device.

    Returns:
        mask: (batch_size, seq_len) boolean tensor.
    """
    n_mask = max(1, int(seq_len * mask_ratio))
    # Only mask from the second half of the sequence
    start = seq_len // 2
    candidates = seq_len - start
    n_mask = min(n_mask, candidates)

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        idx = torch.randperm(candidates, device=device)[:n_mask] + start
        mask[i, idx] = True
    return mask


def generate_last_token_mask(batch_size: int, seq_len: int,
                             device: torch.device = torch.device('cpu')) -> Tensor:
    """Mask only the last position (for pattern induction answer)."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[:, -1] = True
    return mask


# ---------------------------------------------------------------------------
# Stage 1a: Arithmetic sequences
# ---------------------------------------------------------------------------

def generate_arithmetic(n_samples: int, seq_len: int, vocab_size: int) -> Tensor:
    """Generate arithmetic sequences: a, a+d, a+2d, ... (mod vocab_size).

    Each sample has a random start value and step size. The resulting
    sequence is periodic (wraps via modular arithmetic), giving the
    encoder a predictable pattern to learn.

    Returns:
        sequences: (n_samples, seq_len) long tensor with values in [0, vocab_size).
    """
    a = torch.randint(0, vocab_size, (n_samples, 1))
    d = torch.randint(1, vocab_size, (n_samples, 1))
    pos = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)
    return (a + d * pos) % vocab_size


# ---------------------------------------------------------------------------
# Stage 1b: Multi-rule sequences
# ---------------------------------------------------------------------------

def generate_multi_rule(n_samples: int, seq_len: int, vocab_size: int) -> Tensor:
    """Generate sequences where the first half follows rule A, second half rule B.

    Rule A: arithmetic with step d1.
    Rule B: arithmetic with step d2 (different), starting from where A ended.

    Forces the encoder to detect the rule change mid-sequence.

    Returns:
        sequences: (n_samples, seq_len) long tensor.
    """
    half = seq_len // 2
    seqs = torch.zeros(n_samples, seq_len, dtype=torch.long)

    a = torch.randint(0, vocab_size, (n_samples,))
    d1 = torch.randint(1, vocab_size, (n_samples,))
    d2 = torch.randint(1, vocab_size, (n_samples,))
    # Ensure d2 != d1
    same = d2 == d1
    d2[same] = (d2[same] + 1) % vocab_size
    d2[d2 == 0] = 1

    for t in range(half):
        seqs[:, t] = (a + d1 * t) % vocab_size
    mid = (a + d1 * half) % vocab_size
    for t in range(half, seq_len):
        seqs[:, t] = (mid + d2 * (t - half)) % vocab_size

    return seqs


# ---------------------------------------------------------------------------
# Stage 1c: Interleaved sequences
# ---------------------------------------------------------------------------

def generate_interleaved(n_samples: int, seq_len: int,
                         vocab_size: int) -> Tensor:
    """Generate two interleaved arithmetic sequences: A1 B1 A2 B2 ...

    Even positions follow rule A (start a1, step d1).
    Odd positions follow rule B (start a2, step d2).
    Forces encoder to track two independent patterns simultaneously.

    Returns:
        sequences: (n_samples, seq_len) long tensor.
    """
    seqs = torch.zeros(n_samples, seq_len, dtype=torch.long)

    a1 = torch.randint(0, vocab_size, (n_samples,))
    d1 = torch.randint(1, vocab_size, (n_samples,))
    a2 = torch.randint(0, vocab_size, (n_samples,))
    d2 = torch.randint(1, vocab_size, (n_samples,))

    n_even = (seq_len + 1) // 2
    n_odd = seq_len // 2

    for i in range(n_even):
        seqs[:, 2 * i] = (a1 + d1 * i) % vocab_size
    for i in range(n_odd):
        seqs[:, 2 * i + 1] = (a2 + d2 * i) % vocab_size

    return seqs


# ---------------------------------------------------------------------------
# Stage 2: Pattern induction (few-shot rule discovery)
# ---------------------------------------------------------------------------

# Default rule set. Each rule: (name, function(x, vocab_size) -> y)
DEFAULT_RULES: List[Tuple[str, Callable[[int, int], int]]] = [
    ("double",     lambda x, v: (2 * x) % v),
    ("shift3",     lambda x, v: (x + 3) % v),
    ("square",     lambda x, v: (x * x) % v),
    ("complement", lambda x, v: (v - 1 - x) % v),
    ("shift7",     lambda x, v: (x + 7) % v),
]


def generate_pattern_induction(
    n_samples: int,
    n_examples: int,
    vocab_size: int,
    seq_len: Optional[int] = None,
    rules: Optional[List[Tuple[str, Callable]]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate few-shot function learning tasks.

    Each sample: [x1, f(x1), x2, f(x2), ..., xq, f(xq)]
    padded on the left with zeros to seq_len.

    The last token f(xq) is the answer to be predicted.

    Args:
        n_samples: Number of samples.
        n_examples: Number of example (x, f(x)) pairs before the query.
        vocab_size: Token vocabulary size. Tokens in [0, vocab_size).
            Token 0 is used as PAD.
        seq_len: Total sequence length (padded). If None, uses minimal
            length = 2 * n_examples + 2.
        rules: List of (name, fn) tuples. fn(x, vocab_size) -> y.

    Returns:
        sequences: (n_samples, seq_len) long tensor. Left-padded with 0.
        targets: (n_samples,) long tensor. The answer token f(xq).
        rule_indices: (n_samples,) long tensor. Which rule was used.
    """
    if rules is None:
        rules = DEFAULT_RULES

    task_len = 2 * n_examples + 2  # pairs + query_x + answer
    if seq_len is None:
        seq_len = task_len
    assert seq_len >= task_len, f"seq_len {seq_len} < task_len {task_len}"
    pad_len = seq_len - task_len

    sequences = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.zeros(n_samples, dtype=torch.long)
    rule_indices = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        rule_idx = random.randint(0, len(rules) - 1)
        _, rule_fn = rules[rule_idx]
        rule_indices[i] = rule_idx

        # Sample distinct x values from [1, vocab_size) (avoid 0 = PAD)
        n_needed = n_examples + 1
        if n_needed > vocab_size - 1:
            xs = [random.randint(1, vocab_size - 1) for _ in range(n_needed)]
        else:
            xs = random.sample(range(1, vocab_size), n_needed)

        # Fill example pairs after padding
        for j in range(n_examples):
            sequences[i, pad_len + 2 * j] = xs[j]
            sequences[i, pad_len + 2 * j + 1] = rule_fn(xs[j], vocab_size)

        # Query
        xq = xs[-1]
        sequences[i, -2] = xq
        answer = rule_fn(xq, vocab_size)
        sequences[i, -1] = answer
        targets[i] = answer

    return sequences, targets, rule_indices


# ---------------------------------------------------------------------------
# Convenience: get data for a given stage
# ---------------------------------------------------------------------------

def get_stage_data(
    stage: str,
    n_train: int = 5000,
    n_test: int = 1000,
    seq_len: int = 64,
    vocab_size: int = 16,
    n_examples: int = 5,
    n_rules: int = 5,
) -> dict:
    """Generate train/test data for a given stage.

    Args:
        stage: One of '1a', '1b', '1c', '2'.
        n_train, n_test: Sample counts.
        seq_len: Sequence length.
        vocab_size: Vocabulary size.
        n_examples: (Stage 2) number of example pairs.
        n_rules: (Stage 2) number of rules to use.

    Returns:
        dict with keys: train_seqs, test_seqs, and for stage 2:
            train_targets, test_targets, train_rules, test_rules.
    """
    generators = {
        '1a': generate_arithmetic,
        '1b': generate_multi_rule,
        '1c': generate_interleaved,
    }

    if stage in generators:
        gen = generators[stage]
        train_seqs = gen(n_train, seq_len, vocab_size)
        test_seqs = gen(n_test, seq_len, vocab_size)
        return dict(train_seqs=train_seqs, test_seqs=test_seqs)

    elif stage == '2':
        rules = DEFAULT_RULES[:n_rules]
        train_seqs, train_tgt, train_ri = generate_pattern_induction(
            n_train, n_examples, vocab_size, seq_len=seq_len, rules=rules)
        test_seqs, test_tgt, test_ri = generate_pattern_induction(
            n_test, n_examples, vocab_size, seq_len=seq_len, rules=rules)
        return dict(
            train_seqs=train_seqs, test_seqs=test_seqs,
            train_targets=train_tgt, test_targets=test_tgt,
            train_rules=train_ri, test_rules=test_ri,
        )
    else:
        raise ValueError(f"Unknown stage: {stage!r}")
