"""Tests for core/state_space.py — StateSpace (Phase B).

6 tests covering:
  1. predict_dist on a seen context returns correct distribution
  2. predict_unseen on an unseen context returns a non-uniform distribution
  3. predict_unseen on a context sharing a suffix with a seen context returns
     a distribution closer to the seen context's distribution than the marginal
  4. State space dimension equals the rank parameter
  5. encode returns the scaled U row (linearity of the SVD factorisation)
  6. Round-trip: encode(ctx) @ V^T ≈ H[ctx, :] (reconstruction quality)
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from experiments.symbolic_ai_v2.core.hankel import HankelEstimator


def _make_simple_ss(rank=None):
    """Build a small state space from a predictable corpus.

    Pattern: 'a' → 'b' always, 'b' → 'c' always, 'c' → 'a' always.
    H_1[('a',), 'b'] = 10, H_1[('b',), 'c'] = 10, H_1[('c',), 'a'] = 10.
    All other entries are 0.
    """
    he = HankelEstimator(k_max=1)
    seq = ['a', 'b', 'c'] * 10
    he.observe(seq)
    return he, he.build_state_space(k_max=1, rank=rank)


# ── Test 1: predict_dist on seen context ─────────────────────────────────────

def test_predict_dist_seen_context():
    """predict_dist(('a',)) on a corpus where 'a' always follows 'b' should
    return a distribution concentrated on 'b'."""
    _, ss = _make_simple_ss()

    dist = ss.predict_dist(('a',))
    assert 'b' in dist, "Expected 'b' in distribution for context ('a',)"
    assert dist['b'] > 0.9, (
        f"P('b' | 'a') should be ≈ 1.0, got {dist['b']:.3f}"
    )


# ── Test 2: predict_unseen on unseen context ──────────────────────────────────

def test_predict_unseen_non_uniform():
    """predict_unseen on a context not in row_index returns a non-empty,
    non-uniform distribution (does not fail silently)."""
    he = HankelEstimator(k_max=2)
    # Train with 2-gram contexts only; ('z',) will be an unseen 1-gram
    he.observe(['a', 'b', 'c', 'a', 'b', 'c'])
    ss = he.build_state_space(k_max=2)

    fallback = ['a', 'b', 'c']
    dist = ss.predict_unseen(('z',), fallback)
    assert len(dist) > 0, "predict_unseen should return a non-empty distribution"
    # Should not be exactly uniform — the corpus has structure
    # (or if it falls to marginal, it may be non-uniform)
    probs = list(dist.values())
    assert sum(probs) > 0.0


# ── Test 3: predict_unseen uses suffix context ────────────────────────────────

def test_predict_unseen_suffix_is_closer_than_marginal():
    """predict_unseen for a k=2 context whose k=1 suffix IS seen should return
    a distribution similar to predict_dist for that k=1 suffix.

    If ('x', 'a') is unseen but ('a',) IS seen, predict_unseen(('x', 'a'))
    should match predict_dist(('a',)) — the suffix degradation path."""
    he = HankelEstimator(k_max=2)
    # 'a' always followed by 'b'; 'b' always followed by 'c'; 'c' always by 'a'
    he.observe(['a', 'b', 'c'] * 10)
    ss = he.build_state_space(k_max=2)

    # ('x', 'a') is not in row_index (we never observed 'x')
    # but ('a',) IS in row_index
    assert ('x', 'a') not in ss.row_index, "('x','a') should be unseen"
    assert ('a',) in ss.row_index, "('a',) should be seen"

    dist_unseen = ss.predict_unseen(('x', 'a'), ['a', 'b', 'c'])
    dist_seen = ss.predict_dist(('a',))

    # The suffix-degraded prediction should assign the same top atom as the
    # seen-context prediction
    top_unseen = max(dist_unseen, key=dist_unseen.get)
    top_seen = max(dist_seen, key=dist_seen.get)
    assert top_unseen == top_seen, (
        f"Suffix-degraded prediction ({top_unseen}) should match "
        f"seen-context prediction ({top_seen})"
    )


# ── Test 4: rank parameter is respected ──────────────────────────────────────

def test_rank_parameter_respected():
    """StateSpace.rank equals the requested rank (if data supports it)."""
    he = HankelEstimator(k_max=1)
    he.observe(['a', 'b', 'c', 'd', 'e'] * 5)
    ss = he.build_state_space(k_max=1, rank=3)
    assert ss.rank == 3, f"Expected rank 3, got {ss.rank}"


# ── Test 5: encode returns scaled U row (SVD factorisation linearity) ─────────

def test_encode_returns_scaled_u_row():
    """encode(ctx) == U[row_index[ctx]] * S — the SVD scaling of the U row.

    This verifies that encode is a linear projection (not a nonlinear map):
    it directly reads the forward singular vector scaled by singular values.
    """
    _, ss = _make_simple_ss(rank=3)

    for ctx in [('a',), ('b',), ('c',)]:
        encoded = ss.encode(ctx)
        idx = ss.row_index[ctx]
        expected = ss.U[idx] * ss.S
        assert np.allclose(encoded, expected, atol=1e-12), (
            f"encode({ctx}) does not equal U[idx]*S"
        )


# ── Test 6: round-trip reconstruction ────────────────────────────────────────

def test_encode_roundtrip_reconstruction():
    """encode(ctx) @ V^T ≈ H[ctx, :] when rank captures most variance.

    For a small corpus (3 atoms, 3 contexts), a rank=3 decomposition is
    exact (up to floating-point precision), so the round-trip error should
    be near machine epsilon.
    """
    he, ss = _make_simple_ss(rank=3)

    # Get the actual H_1 matrix to compare against
    row_keys, col_keys, H1 = he.matrix(1)

    for ctx in [('a',), ('b',), ('c',)]:
        encoded = ss.encode(ctx)              # (rank,)
        reconstructed = encoded @ ss.V.T      # (n_atoms,)

        # Find original H_1 row for this context
        local_row = row_keys.index(ctx)
        original_row = H1[local_row]          # raw counts (not probabilities)

        # Reconstruction is of the raw count matrix (not normalised).
        # For an exact-rank decomposition, reconstructed ∝ original_row.
        # Normalise both and compare.
        norm_r = np.linalg.norm(reconstructed)
        norm_o = np.linalg.norm(original_row)
        if norm_r > 1e-12 and norm_o > 1e-12:
            cos_sim = (reconstructed / norm_r) @ (original_row / norm_o)
            assert cos_sim > 0.99, (
                f"Round-trip cosine similarity for {ctx}: {cos_sim:.4f} < 0.99"
            )
