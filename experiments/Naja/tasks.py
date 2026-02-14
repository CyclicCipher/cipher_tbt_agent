"""
Ablation task generators for Naja.

Each task is designed to test a specific architectural feature:
  - Associative recall  → delta rule (selective erase+write)
  - Parity tracking      → PoPE orthogonal pair (rotation)
  - Multi-scale memory   → per-channel decay (fast+slow channels)
  - Permutation tracking → full architecture (Householder products)

All generators return:
    sequences: (n_samples, seq_len) long tensor, values in [0, vocab_size)
    Token 0 is reserved for PAD (left-padding).
    Next-step prediction: model predicts sequences[:, t+1] from sequences[:, :t+1].
"""

import random
from typing import Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Associative Recall
# ---------------------------------------------------------------------------

def generate_associative_recall(
    n_samples: int,
    n_pairs: int,
    vocab_size: int,
    seq_len: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Key-value binding + retrieval task.

    Tests the delta rule: store key-value pairs, then retrieve by key.
    The delta rule's targeted erase+write should excel here — it can
    overwrite the state in the key's direction with the value.

    Layout: [k1, v1, k2, v2, ..., kN, vN, SEP, kQ, vQ]
    SEP = vocab_size - 1 (reserved separator token).
    The model must predict vQ given kQ after seeing the pairs.

    Left-padded with 0 to seq_len.

    Args:
        n_samples: Number of samples.
        n_pairs: Number of key-value pairs before the query.
        vocab_size: Vocabulary size. Token 0 = PAD, token vocab_size-1 = SEP.
        seq_len: Total sequence length. If 0, uses minimal length.

    Returns:
        sequences: (n_samples, seq_len) long tensor.
        targets: (n_samples,) long tensor — the answer value.
    """
    SEP = vocab_size - 1
    usable = list(range(1, vocab_size - 1))  # tokens for keys/values
    task_len = 2 * n_pairs + 3  # pairs + SEP + query_key + query_value
    if seq_len <= 0:
        seq_len = task_len
    assert seq_len >= task_len
    pad_len = seq_len - task_len

    sequences = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        # Sample distinct keys
        if n_pairs + 1 <= len(usable):
            keys = random.sample(usable, n_pairs)
        else:
            keys = [random.choice(usable) for _ in range(n_pairs)]
        values = [random.choice(usable) for _ in range(n_pairs)]

        # Fill pairs
        for j in range(n_pairs):
            sequences[i, pad_len + 2 * j] = keys[j]
            sequences[i, pad_len + 2 * j + 1] = values[j]

        # Separator
        sequences[i, pad_len + 2 * n_pairs] = SEP

        # Query: pick a random pair to ask about
        q_idx = random.randint(0, n_pairs - 1)
        sequences[i, -2] = keys[q_idx]
        sequences[i, -1] = values[q_idx]
        targets[i] = values[q_idx]

    return sequences, targets


# ---------------------------------------------------------------------------
# Parity Tracking
# ---------------------------------------------------------------------------

def generate_parity_tracking(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[Tensor, Tensor]:
    """Binary parity tracking task.

    Tests PoPE orthogonal pair: the model must track running parity
    of a binary input stream. Two Householder reflections compose into
    a rotation — exactly what parity toggling requires.

    Layout: [b1, p1, b2, p2, ..., bN, pN]
    where b_i ∈ {1, 2} (binary input: 1=zero, 2=one)
    and p_i ∈ {3, 4} (parity: 3=even, 4=odd)

    The model predicts p_i at each step from the preceding sequence.

    Args:
        n_samples: Number of samples.
        seq_len: Total sequence length (must be even).
        vocab_size: Must be >= 5 (tokens 0-4 used).

    Returns:
        sequences: (n_samples, seq_len) long tensor.
        targets: (n_samples,) long tensor — parity of last position.
    """
    assert vocab_size >= 5, "Need >= 5 tokens for parity task"
    assert seq_len % 2 == 0, "seq_len must be even for parity"

    n_steps = seq_len // 2
    sequences = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        parity = 0  # 0=even, 1=odd
        for t in range(n_steps):
            bit = random.randint(0, 1)  # 0 or 1
            sequences[i, 2 * t] = bit + 1  # token 1 or 2
            parity ^= bit
            sequences[i, 2 * t + 1] = parity + 3  # token 3 or 4

        targets[i] = parity + 3  # last parity token

    return sequences, targets


# ---------------------------------------------------------------------------
# Multi-Scale Memory
# ---------------------------------------------------------------------------

def generate_multi_scale_memory(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[Tensor, Tensor]:
    """Multi-scale memory recall task.

    Tests per-channel decay: channels with α≈1 must retain a distant
    cue, while channels with α≈0 handle recent context.

    Layout: [CUE, noise..., noise..., MARKER, ANSWER]
    - CUE is placed at position 1 (after PAD at 0): a token from [1, V//2)
    - Noise fills positions 2 to seq_len-3: random tokens from [V//2, V-1)
    - MARKER at position seq_len-2: token = vocab_size - 1 (signals recall)
    - ANSWER at position seq_len-1: equals CUE

    The model must remember CUE across many noise tokens and reproduce it
    after seeing MARKER. Only per-channel decay with α≈1 channels can do this.

    Args:
        n_samples: Number of samples.
        seq_len: Total sequence length (>= 4).
        vocab_size: Vocabulary size.

    Returns:
        sequences: (n_samples, seq_len) long tensor.
        targets: (n_samples,) long tensor — the cue token to recall.
    """
    assert seq_len >= 4
    MARKER = vocab_size - 1
    cue_range = max(1, vocab_size // 2 - 1)
    noise_lo = vocab_size // 2
    noise_hi = vocab_size - 2

    sequences = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        cue = random.randint(1, cue_range)
        sequences[i, 0] = 0  # PAD
        sequences[i, 1] = cue

        # Fill noise
        for t in range(2, seq_len - 2):
            sequences[i, t] = random.randint(noise_lo, noise_hi)

        sequences[i, -2] = MARKER
        sequences[i, -1] = cue
        targets[i] = cue

    return sequences, targets


# ---------------------------------------------------------------------------
# Permutation Tracking
# ---------------------------------------------------------------------------

def generate_permutation_tracking(
    n_samples: int,
    n_elements: int,
    n_swaps: int,
    vocab_size: int,
    seq_len: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Permutation tracking via swap operations.

    Tests full Naja architecture: Householder products can represent
    arbitrary orthogonal transformations, which includes permutations.

    Layout: [e1, e2, ..., eK, SEP, s1a, s1b, s2a, s2b, ..., SEP, Q, A]
    - Initial arrangement: elements e1..eK (tokens 1..K)
    - SEP = vocab_size - 1
    - Each swap (sa, sb) means "swap positions sa and sb" (tokens 1..K)
    - Query Q: a position (token 1..K)
    - Answer A: which element is at position Q after all swaps

    Args:
        n_samples: Number of samples.
        n_elements: K, number of elements being permuted.
        n_swaps: Number of swap operations.
        vocab_size: Must be > n_elements + 1.
        seq_len: Total length. If 0, uses minimal.

    Returns:
        sequences: (n_samples, seq_len) long tensor.
        targets: (n_samples,) long tensor — element at queried position.
    """
    SEP = vocab_size - 1
    task_len = n_elements + 1 + 2 * n_swaps + 1 + 2  # init + SEP + swaps + SEP + Q + A
    if seq_len <= 0:
        seq_len = task_len
    assert seq_len >= task_len
    assert vocab_size > n_elements + 1
    pad_len = seq_len - task_len

    sequences = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        # Initial arrangement: positions 1..K hold elements 1..K
        perm = list(range(1, n_elements + 1))

        # Write initial arrangement
        for j in range(n_elements):
            sequences[i, pad_len + j] = perm[j]
        sequences[i, pad_len + n_elements] = SEP

        # Apply swaps
        offset = pad_len + n_elements + 1
        for s in range(n_swaps):
            a = random.randint(0, n_elements - 1)
            b = random.randint(0, n_elements - 1)
            while b == a and n_elements > 1:
                b = random.randint(0, n_elements - 1)
            # Swap
            perm[a], perm[b] = perm[b], perm[a]
            # Record swap (1-indexed positions)
            sequences[i, offset + 2 * s] = a + 1
            sequences[i, offset + 2 * s + 1] = b + 1

        # Second separator
        sequences[i, -3] = SEP

        # Query: random position (1-indexed)
        q_pos = random.randint(0, n_elements - 1)
        sequences[i, -2] = q_pos + 1
        sequences[i, -1] = perm[q_pos]
        targets[i] = perm[q_pos]

    return sequences, targets


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

ABLATION_TASKS = {
    'associative_recall': {
        'fn': generate_associative_recall,
        'tests_feature': 'delta_rule',
        'default_args': dict(n_pairs=8, vocab_size=32, seq_len=32),
    },
    'parity': {
        'fn': generate_parity_tracking,
        'tests_feature': 'pope_perp',
        'default_args': dict(seq_len=32, vocab_size=16),
    },
    'multi_scale': {
        'fn': generate_multi_scale_memory,
        'tests_feature': 'per_channel_decay',
        'default_args': dict(seq_len=64, vocab_size=32),
    },
    'permutation_3': {
        'fn': generate_permutation_tracking,
        'tests_feature': 'full',
        'default_args': dict(n_elements=3, n_swaps=4, vocab_size=16, seq_len=24),
    },
    'permutation_4': {
        'fn': generate_permutation_tracking,
        'tests_feature': 'full',
        'default_args': dict(n_elements=4, n_swaps=6, vocab_size=16, seq_len=32),
    },
}


def get_task_data(task_name: str, n_train: int = 5000, n_test: int = 1000,
                  **overrides) -> dict:
    """Generate train/test data for an ablation task.

    Returns:
        dict with train_seqs, test_seqs, train_targets, test_targets.
    """
    info = ABLATION_TASKS[task_name]
    fn = info['fn']
    kwargs = {**info['default_args'], **overrides}

    train_seqs, train_tgt = fn(n_train, **kwargs)
    test_seqs, test_tgt = fn(n_test, **kwargs)

    return dict(
        train_seqs=train_seqs, test_seqs=test_seqs,
        train_targets=train_tgt, test_targets=test_tgt,
        vocab_size=kwargs.get('vocab_size', 16),
        seq_len=train_seqs.shape[1],
    )
