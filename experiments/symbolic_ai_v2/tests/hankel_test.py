"""Tests for core/hankel.py — HankelEstimator (Phase A).

8 tests covering:
  1. observe accumulates H_2 counts correctly
  2. matrix(1) row sums equal left-context occurrence counts
  3. build_state_space(rank=2) recovers 2 states from a 2-state bigram corpus
  4. build_state_space automatic rank ≤ min(rows, cols)
  5. Held-out log-likelihood is higher with k=2 than k=1 on a structured corpus
  6. Two identical corpora produce identical state spaces (determinism)
  7. observe call count scales linearly with sequence length
  8. build_state_space on a 10K-token corpus completes in < 5 seconds
"""

import math
import time

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from experiments.symbolic_ai_v2.core.hankel import HankelEstimator


# ── Test 1: observe accumulates H_2 counts ────────────────────────────────────

def test_observe_h2_count():
    """observe on ['a','b','c','a','b','c'] gives H_2[('a','b'),'c'] == 2."""
    he = HankelEstimator(k_max=2)
    he.observe(['a', 'b', 'c', 'a', 'b', 'c'])
    row_keys, col_keys, mat = he.matrix(2)

    # Find row for ('a','b') and column for 'c'
    assert ('a', 'b') in row_keys, "('a','b') should be a row key"
    assert 'c' in col_keys, "'c' should be a column key"

    r = row_keys.index(('a', 'b'))
    c = col_keys.index('c')
    assert mat[r, c] == 2.0, (
        f"Expected H_2[('a','b'),'c'] == 2, got {mat[r, c]}"
    )


# ── Test 2: matrix(1) row sums equal occurrence counts ────────────────────────

def test_matrix1_row_sums():
    """Each row of H_1 sums to the number of times that atom appeared as a
    left context (i.e., the number of times it was followed by anything)."""
    seq = ['a', 'b', 'c', 'a', 'b', 'c']
    he = HankelEstimator(k_max=1)
    he.observe(seq)
    row_keys, col_keys, mat = he.matrix(1)

    # Count how many times each left-context atom was followed by something
    expected_row_sums = {}
    for i in range(1, len(seq)):
        left = (seq[i - 1],)
        expected_row_sums[left] = expected_row_sums.get(left, 0) + 1

    for i, rk in enumerate(row_keys):
        actual_sum = mat[i].sum()
        expected_sum = expected_row_sums.get(rk, 0)
        assert actual_sum == expected_sum, (
            f"Row sum for {rk}: expected {expected_sum}, got {actual_sum}"
        )


# ── Test 3: 2-state corpus gives rank=2 ───────────────────────────────────────

def test_build_state_space_two_states():
    """A corpus with two independent sub-sequences ({a,b} and {c,d}) should
    produce a state space of rank 2 when rank=2 is requested."""
    he = HankelEstimator(k_max=2)
    # Two non-overlapping cyclic patterns
    seq_ab = ['a', 'b'] * 10
    seq_cd = ['c', 'd'] * 10
    he.observe(seq_ab)
    he.observe(seq_cd)

    ss = he.build_state_space(k_max=1, rank=2)
    assert ss.rank == 2, f"Expected rank 2, got {ss.rank}"

    # The two patterns should be distinguishable in the 2D state space:
    # encode(('a',)) and encode(('c',)) should not be parallel
    v_a = ss.encode(('a',))
    v_c = ss.encode(('c',))
    assert not np.allclose(v_a, v_c, atol=1e-6), (
        "State vectors for ('a',) and ('c',) should differ"
    )


# ── Test 4: automatic rank ≤ min(rows, cols) ──────────────────────────────────

def test_build_state_space_rank_bounded():
    """Automatic rank selection cannot exceed min(n_contexts, n_atoms)."""
    rng = np.random.default_rng(42)
    atoms = [str(i) for i in range(8)]
    he = HankelEstimator(k_max=1)
    # 50 random length-20 sequences over 8 atoms
    for _ in range(50):
        seq = rng.choice(atoms, size=20).tolist()
        he.observe(seq)

    ss = he.build_state_space(k_max=1)
    row_keys, col_keys, H1 = he.matrix(1)
    max_rank = min(len(row_keys), len(col_keys))
    assert ss.rank <= max_rank, (
        f"rank {ss.rank} exceeds min(rows, cols) = {max_rank}"
    )


# ── Test 5: k=2 gives better held-out log-likelihood than k=1 ────────────────

def test_higher_k_lower_perplexity():
    """On a corpus with clear 2nd-order dependencies, k=2 predictions have
    higher log-likelihood on held-out data than k=1 predictions."""
    # Pattern: 'x' → 'a' → 'b' → 'x' → ...
    #          'y' → 'a' → 'c' → 'y' → ...
    # With k=1 context ('a',) → 'b' or 'c' (ambiguous, ~50/50)
    # With k=2 context ('x','a') → 'b' always; ('y','a') → 'c' always
    train_x = ['x', 'a', 'b'] * 20
    train_y = ['y', 'a', 'c'] * 20

    he1 = HankelEstimator(k_max=1)
    he1.observe(train_x)
    he1.observe(train_y)
    ss1 = he1.build_state_space(k_max=1)

    he2 = HankelEstimator(k_max=2)
    he2.observe(train_x)
    he2.observe(train_y)
    ss2 = he2.build_state_space(k_max=2)

    # Held-out test: after 'x','a' → expect 'b'
    def log_likelihood(ss, ctx, target):
        dist = ss.predict_dist(ctx)
        p = dist.get(target, 1e-9)
        return math.log2(p)

    # k=1 prediction for 'a' → 'b' (ambiguous)
    ll1 = log_likelihood(ss1, ('a',), 'b')
    # k=2 prediction for ('x','a') → 'b' (certain)
    ll2 = log_likelihood(ss2, ('x', 'a'), 'b')

    assert ll2 > ll1, (
        f"k=2 log-likelihood ({ll2:.3f}) should exceed k=1 ({ll1:.3f})"
    )


# ── Test 6: determinism — identical corpora give identical state spaces ────────

def test_determinism():
    """Two HankelEstimators trained on the same sequences produce the same
    state space (up to numerical precision and SVD sign conventions)."""
    seqs = [
        ['a', 'b', 'c', 'a'],
        ['b', 'c', 'a', 'b'],
        ['c', 'a', 'b', 'c'],
    ]

    def build(seqs_):
        he = HankelEstimator(k_max=2)
        for s in seqs_:
            he.observe(s)
        return he.build_state_space(k_max=2, rank=3)

    ss1 = build(seqs)
    ss2 = build(seqs)

    # Singular values must be identical (not sign-dependent)
    assert np.allclose(ss1.S, ss2.S, atol=1e-10), (
        f"Singular values differ: {ss1.S} vs {ss2.S}"
    )
    assert ss1.rank == ss2.rank
    assert ss1.row_index == ss2.row_index
    assert ss1.col_index == ss2.col_index


# ── Test 7: observe is O(k_max × len(seq)) per call ──────────────────────────

def test_observe_linear_time():
    """observe(seq) time scales linearly with len(seq): doubling the sequence
    length approximately doubles the execution time (within 4× for noise)."""
    import time

    atoms = [str(i % 5) for i in range(1000)]

    he_short = HankelEstimator(k_max=4)
    he_long = HankelEstimator(k_max=4)

    short_seq = atoms[:200]
    long_seq = atoms[:1000]

    # Warm up
    HankelEstimator(k_max=4).observe(['a', 'b', 'c'])

    t0 = time.perf_counter()
    he_short.observe(short_seq)
    t_short = time.perf_counter() - t0

    t0 = time.perf_counter()
    he_long.observe(long_seq)
    t_long = time.perf_counter() - t0

    # 5× longer sequence should be < 25× slower (generous for timer noise)
    ratio = len(long_seq) / len(short_seq)  # 5
    time_ratio = t_long / (t_short + 1e-9)
    assert time_ratio < ratio * 5, (
        f"Time ratio {time_ratio:.1f} too high for length ratio {ratio}"
    )


# ── Test 8: 10K-token corpus builds state space in < 5 seconds ───────────────

def test_build_state_space_performance():
    """build_state_space on a 10K-token corpus completes in < 5 seconds."""
    rng = np.random.default_rng(7)
    atoms = list('abcdefghijklmnop')  # 16 atoms
    seq = rng.choice(atoms, size=10_000).tolist()

    he = HankelEstimator(k_max=4)
    he.observe(seq)

    t0 = time.perf_counter()
    ss = he.build_state_space(k_max=4)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"build_state_space took {elapsed:.2f}s (limit: 5s)"
    assert ss.rank >= 1
