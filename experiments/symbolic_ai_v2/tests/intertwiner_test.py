"""Tests for reasoning/intertwiner.py — cross-domain intertwiner (Phase C).

6 tests covering:
  1. Two identical domains produce identity intertwiner (low alignment error)
  2. Two structurally unrelated domains produce higher alignment error
  3. transfer_predict gives the correct top token when domains are isomorphic
  4. find_intertwiner is invariant to orthogonal rotations of the state space
  5. Roundtrip: eta_bwd ∘ eta_fwd ≈ identity on shared atom embeddings
  6. Runtime: find_intertwiner with rank ≤ 50 completes in < 1 second
"""

import copy
import time

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from experiments.symbolic_ai_v2.core.hankel import HankelEstimator
from experiments.symbolic_ai_v2.core.state_space import StateSpace
from experiments.symbolic_ai_v2.reasoning.intertwiner import (
    find_intertwiner,
    transfer_predict,
    alignment_error,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _arith_ss(rank=4):
    """State space from arithmetic sequences: N1 + N2 = SUM."""
    he = HankelEstimator(k_max=3)
    for n1 in range(1, 6):
        for n2 in range(1, 6):
            seq = [str(n1), 'add', str(n2), 'eq', str(n1 + n2)]
            for _ in range(4):
                he.observe(seq)
    return he.build_state_space(k_max=3, rank=rank)


def _nl_ss(rank=4):
    """State space from NL arithmetic sequences: N1 et N2 quot SUM."""
    he = HankelEstimator(k_max=3)
    for n1 in range(1, 6):
        for n2 in range(1, 6):
            seq = [str(n1), 'et', str(n2), 'est', str(n1 + n2)]
            for _ in range(4):
                he.observe(seq)
    return he.build_state_space(k_max=3, rank=rank)


_DIGITS = [str(i) for i in range(1, 11)]


# ── Test 1: Identical domains → low alignment error ──────────────────────────

def test_identical_domains_low_error():
    """Two state spaces trained on the same corpus should align almost
    perfectly: alignment_error ≈ 0."""
    ss1 = _arith_ss(rank=4)
    ss2 = _arith_ss(rank=4)   # same corpus, deterministic SVD → same V

    shared = [σ for σ in _DIGITS if σ in ss1.col_index and σ in ss2.col_index]
    assert shared, "Digits should appear in both state spaces"

    eta = find_intertwiner(ss1, ss2, shared)
    err = alignment_error(ss1, ss2, eta, shared)

    # Identical corpora → perfect alignment
    # Without normalisation lstsq(V, V) = I, so residual = ||V @ I - V|| = 0
    assert err < 1e-10, f"alignment_error for identical domains: {err:.2e} (expected ≈ 0)"


# ── Test 2: Unrelated domains → higher alignment error ───────────────────────

def test_unrelated_domains_higher_error():
    """Domains trained on structurally different corpora should have higher
    alignment error than two copies of the same corpus."""
    ss_math = _arith_ss(rank=4)
    ss_nl = _nl_ss(rank=4)

    shared = [σ for σ in _DIGITS if σ in ss_math.col_index and σ in ss_nl.col_index]
    assert shared

    # Identical-domain baseline error
    ss_math2 = _arith_ss(rank=4)
    eta_same = find_intertwiner(ss_math, ss_math2, shared)
    err_same = alignment_error(ss_math, ss_math2, eta_same, shared)

    # Cross-domain error
    eta_cross = find_intertwiner(ss_math, ss_nl, shared)
    err_cross = alignment_error(ss_math, ss_nl, eta_cross, shared)

    assert err_cross > err_same, (
        f"Cross-domain error ({err_cross:.4f}) should exceed "
        f"same-domain error ({err_same:.4f})"
    )


# ── Test 3: transfer_predict gives correct top token ─────────────────────────

def test_transfer_predict_correct_top_token():
    """An isomorphic NL domain (same structure, different token names)
    should transfer to produce the correct arithmetic answer.

    Both domains follow the pattern: N1 <connector> N2 <query> = N1+N2.
    The intertwiner aligns the digit representations; the target domain
    then predicts the correct sum.
    """
    # Both domains have identical structure — just different connector tokens
    # Domain 1 (source): '2', 'plus', '3', 'gives', '5'
    # Domain 2 (target): '2', 'add', '3', 'eq', '5'
    # 4-token format: n1 <connector> n2 sum — so k=3 context (n1, conn, n2) → sum uniquely
    he_src = HankelEstimator(k_max=3)
    he_tgt = HankelEstimator(k_max=3)

    for n1 in range(1, 5):
        for n2 in range(1, 5):
            s = n1 + n2
            src_seq = [str(n1), 'plus', str(n2), str(s)]
            tgt_seq = [str(n1), 'add',  str(n2), str(s)]
            for _ in range(6):
                he_src.observe(src_seq)
                he_tgt.observe(tgt_seq)

    ss_src = he_src.build_state_space(k_max=3, rank=6)
    ss_tgt = he_tgt.build_state_space(k_max=3, rank=6)

    shared = [σ for σ in _DIGITS
              if σ in ss_src.col_index and σ in ss_tgt.col_index]
    assert shared

    eta = find_intertwiner(ss_src, ss_tgt, shared)

    # Context ('2', 'plus', '3') at i=3 in ['2','plus','3','5'] uniquely predicts '5'
    ctx = ('2', 'plus', '3')
    assert ctx in ss_src.row_index, f"{ctx} not in source row_index"
    dist = transfer_predict(ctx, ss_src, ss_tgt, eta)
    assert dist, "transfer_predict should return non-empty distribution"

    top = max(dist, key=dist.get)
    assert top == '5', (
        f"Expected top prediction '5' for ('2','plus','3') → got '{top}' "
        f"(distribution: {dict(sorted(dist.items(), key=lambda x: -x[1])[:5])})"
    )


# ── Test 4: Invariance to orthogonal rotation of source state space ───────────

def test_orthogonal_invariance():
    """Rotating the source state space by an orthogonal matrix O should not
    change the transfer predictions, since the intertwiner absorbs O^T.

    If ss_rot has V_rot = V @ O and U_rot = U @ O, it represents the same
    WFA in a rotated basis.  find_intertwiner(ss_rot, ss_tgt) should produce
    η_rot ≈ η @ O^T, so that η_rot @ encode_rot(ctx) = η @ encode(ctx).
    """
    ss_src = _arith_ss(rank=4)
    ss_tgt = _nl_ss(rank=4)

    shared = [σ for σ in _DIGITS
              if σ in ss_src.col_index and σ in ss_tgt.col_index]

    # Apply random orthogonal rotation to source state space
    rng = np.random.default_rng(123)
    Q, _ = np.linalg.qr(rng.standard_normal((ss_src.rank, ss_src.rank)))

    ss_rot = copy.copy(ss_src)
    ss_rot = StateSpace(
        rank=ss_src.rank,
        U=ss_src.U @ Q,
        V=ss_src.V @ Q,
        S=ss_src.S,
        row_index=ss_src.row_index,
        col_index=ss_src.col_index,
        col_keys=ss_src.col_keys,
        k_max=ss_src.k_max,
    )

    eta_orig = find_intertwiner(ss_src, ss_tgt, shared)
    eta_rot  = find_intertwiner(ss_rot, ss_tgt, shared)

    # Both should give the same transfer prediction for any seen context
    ctx = ('2', 'add', '3', 'eq')
    if ctx not in ss_src.row_index:
        # Use any seen context
        ctx = next(iter(ss_src.row_index))

    dist_orig = transfer_predict(ctx, ss_src, ss_tgt, eta_orig)
    dist_rot  = transfer_predict(ctx, ss_rot, ss_tgt, eta_rot)

    if dist_orig and dist_rot:
        top_orig = max(dist_orig, key=dist_orig.get)
        top_rot  = max(dist_rot,  key=dist_rot.get)
        assert top_orig == top_rot, (
            f"Rotation changed top prediction: {top_orig} → {top_rot}"
        )


# ── Test 5: Roundtrip ─────────────────────────────────────────────────────────

def test_roundtrip_approximate_identity():
    """eta_bwd ∘ eta_fwd acts as approximate identity on shared atom embeddings.

    eta_fwd maps S_Math → S_NL; eta_bwd maps S_NL → S_Math.
    For each shared atom σ: (eta_bwd @ eta_fwd) @ V_math[σ] ≈ V_math[σ]
    (up to the rank truncation and normalisation of the two intertwiners).
    """
    ss_math = _arith_ss(rank=4)
    ss_nl   = _nl_ss(rank=4)

    shared = [σ for σ in _DIGITS
              if σ in ss_math.col_index and σ in ss_nl.col_index]
    assert len(shared) >= 2, "Need at least 2 shared atoms for roundtrip test"

    eta_fwd = find_intertwiner(ss_math, ss_nl,   shared)  # math → nl
    eta_bwd = find_intertwiner(ss_nl,   ss_math, shared)  # nl → math

    # Compose: eta_bwd @ eta_fwd should approximately recover the source embedding
    composed = eta_bwd @ eta_fwd  # (r_math, r_math)

    for σ in shared:
        v = ss_math.V[ss_math.col_index[σ], :]   # (r_math,) — shared atom embedding
        # After normalisation, the roundtrip can only be approximate;
        # check that the cosine similarity between v and composed @ v is high
        roundtrip = composed @ v
        norm_v = np.linalg.norm(v)
        norm_r = np.linalg.norm(roundtrip)
        if norm_v > 1e-10 and norm_r > 1e-10:
            cos = np.dot(v, roundtrip) / (norm_v * norm_r)
            assert cos > 0.5, (
                f"Roundtrip cosine for σ='{σ}': {cos:.3f} < 0.5"
            )


# ── Test 6: Runtime < 1 second for rank ≤ 50 ─────────────────────────────────

def test_find_intertwiner_performance():
    """find_intertwiner with rank=50 state spaces completes in < 1 second."""
    rng = np.random.default_rng(42)
    rank = 50
    n_atoms = 100

    # Random state spaces with rank=50, n_atoms=100
    def random_ss(rank, n_atoms):
        U = rng.standard_normal((n_atoms, rank))
        V = rng.standard_normal((n_atoms, rank))
        S = np.sort(rng.uniform(0.1, 1.0, rank))[::-1]
        atoms = [str(i) for i in range(n_atoms)]
        return StateSpace(
            rank=rank,
            U=U,
            V=V,
            S=S,
            row_index={(a,): i for i, a in enumerate(atoms)},
            col_index={a: i for i, a in enumerate(atoms)},
            col_keys=atoms,
            k_max=1,
        )

    ss1 = random_ss(rank, n_atoms)
    ss2 = random_ss(rank, n_atoms)
    shared = [str(i) for i in range(min(20, n_atoms))]

    t0 = time.perf_counter()
    eta = find_intertwiner(ss1, ss2, shared)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, f"find_intertwiner took {elapsed:.3f}s (limit: 1s)"
    assert eta.shape == (rank, rank)
