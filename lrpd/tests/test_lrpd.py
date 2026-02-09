"""Tests for the LRPD library.

Each test creates small random LRPD matrices and validates against
brute-force full-matrix computation.

Run with: python -m pytest lrpd/tests/test_lrpd.py -v --tb=short
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import pytest

from lrpd.woodbury import woodbury_inverse_components, lrpd_solve, lrpd_matvec, lrpd_inv_diag
from lrpd.log_det import lrpd_log_det
from lrpd.schur import schur_complement_diag, schur_complement_full_diag
from lrpd.online_update import online_lrpd_update
from lrpd.alt_decompose import alt_decompose, alt_decompose_from_factors, nystrom_alt_decompose
from lrpd.psd_bound import psd_upper_bound


def _make_random_lrpd(n=50, k=10):
    """Create random d > 0 and U, return (d, U, A) where A = diag(d) + U U^T."""
    d = torch.rand(n) * 5 + 0.1
    U = torch.randn(n, k) * 0.5
    A = torch.diag(d) + U @ U.T
    return d, U, A


def _make_random_psd(n=50, rank=None):
    """Create a random PSD matrix with given rank (or full rank)."""
    if rank is None:
        rank = n
    F = torch.randn(rank, n)
    A = F.T @ F + 0.01 * torch.eye(n)
    return A


# ========================================================================
# Woodbury tests
# ========================================================================

class TestWoodbury:
    def test_inverse_matches_full(self):
        d, U, A = _make_random_lrpd(30, 8)
        A_inv_full = torch.inverse(A)
        d_inv, C_inv, d_inv_U = woodbury_inverse_components(d, U)
        A_inv_woodbury = torch.diag(d_inv) - d_inv_U @ C_inv @ d_inv_U.T
        assert torch.allclose(A_inv_woodbury, A_inv_full, atol=1e-4), \
            f"Max diff: {(A_inv_woodbury - A_inv_full).abs().max():.2e}"

    def test_solve_matches_full(self):
        d, U, A = _make_random_lrpd(30, 8)
        B = torch.randn(5, 30)
        expected = B @ torch.inverse(A)
        result = lrpd_solve(d, U, B)
        assert torch.allclose(result, expected, atol=1e-4), \
            f"Max diff: {(result - expected).abs().max():.2e}"

    def test_matvec_matches_full(self):
        d, U, A = _make_random_lrpd(30, 8)
        x = torch.randn(30)
        expected = A @ x
        result = lrpd_matvec(d, U, x)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_matvec_batched(self):
        d, U, A = _make_random_lrpd(30, 8)
        x = torch.randn(10, 30)
        expected = x @ A.T
        result = lrpd_matvec(d, U, x)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_inv_diag_matches_full(self):
        d, U, A = _make_random_lrpd(30, 8)
        expected = torch.inverse(A).diag()
        result = lrpd_inv_diag(d, U)
        assert torch.allclose(result, expected, atol=1e-4), \
            f"Max diff: {(result - expected).abs().max():.2e}"

    def test_zero_U(self):
        d = torch.rand(20) * 5 + 0.1
        U = torch.zeros(20, 5)
        result = lrpd_inv_diag(d, U)
        assert torch.allclose(result, 1.0 / d, atol=1e-4)


# ========================================================================
# Log-determinant tests
# ========================================================================

class TestLogDet:
    def test_log_det_matches_full(self):
        d, U, A = _make_random_lrpd(30, 8)
        expected = torch.logdet(A)
        result = lrpd_log_det(d, U)
        assert torch.allclose(result, expected, atol=1e-3), \
            f"Expected {expected:.6f}, got {result:.6f}"

    def test_zero_U(self):
        d = torch.rand(20) * 5 + 0.1
        U = torch.zeros(20, 5)
        expected = torch.log(d).sum()
        result = lrpd_log_det(d, U)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_positive_d_gives_finite(self):
        d = torch.rand(100) * 10 + 0.01
        U = torch.randn(100, 20) * 0.3
        result = lrpd_log_det(d, U)
        assert torch.isfinite(result)


# ========================================================================
# Schur complement tests
# ========================================================================

class TestSchur:
    def test_schur_diag_matches_full(self):
        d, U, A = _make_random_lrpd(30, 8)
        M = torch.randn(10, 30)
        expected = torch.diag(M @ A @ M.T)
        result = schur_complement_diag(M, d, U)
        assert torch.allclose(result, expected, atol=1e-4), \
            f"Max diff: {(result - expected).abs().max():.2e}"

    def test_schur_full_diag(self):
        d, U, A = _make_random_lrpd(30, 8)
        M = torch.randn(10, 30)
        eta3_diag = torch.rand(10) * 100 + 50
        expected = eta3_diag - torch.diag(M @ A @ M.T)
        result = schur_complement_full_diag(eta3_diag, M, d, U)
        assert torch.allclose(result, expected, atol=1e-3)


# ========================================================================
# Online update tests
# ========================================================================

class TestOnlineUpdate:
    def test_full_rank_exact(self):
        """If k >= n, online update with alpha=1 should give F^T F exactly."""
        n = 20
        batch = 30
        F = torch.randn(batch, n)
        d = torch.zeros(n)
        U = torch.zeros(n, n)
        d_new, U_new = online_lrpd_update(d, U, F, alpha=1.0, beta=0.0,
                                           residual_mode="fitc", rank_k=n)
        reconstructed = torch.diag(d_new) + U_new @ U_new.T
        target = F.T @ F
        assert torch.allclose(reconstructed, target, atol=1e-3), \
            f"Max diff: {(reconstructed - target).abs().max():.2e}"

    def test_fitc_preserves_diagonal(self):
        """FITC mode should preserve exact diagonal of target."""
        n = 30
        batch = 20
        k = 10
        F = torch.randn(batch, n)
        d = torch.rand(n) * 0.1
        U = torch.randn(n, k) * 0.1
        target = 0.7 * (torch.diag(d) + U @ U.T) + 0.3 * F.T @ F
        d_new, U_new = online_lrpd_update(d, U, F, alpha=0.3, beta=0.7,
                                           residual_mode="fitc", rank_k=k)
        reconstructed_diag = d_new + (U_new ** 2).sum(dim=1)
        target_diag = target.diag()
        assert torch.allclose(reconstructed_diag, target_diag, atol=1e-3), \
            f"Max diag diff: {(reconstructed_diag - target_diag).abs().max():.2e}"

    def test_spectral_psd_bound(self):
        """Spectral mode: diag(d) + U U^T >= target in PSD sense."""
        n = 30
        batch = 40
        k = 10
        F = torch.randn(batch, n)
        d = torch.rand(n) * 0.1
        U = torch.randn(n, k) * 0.1
        target = 0.7 * (torch.diag(d) + U @ U.T) + 0.3 * F.T @ F
        d_new, U_new = online_lrpd_update(d, U, F, alpha=0.3, beta=0.7,
                                           residual_mode="spectral", rank_k=k)
        approx = torch.diag(d_new) + U_new @ U_new.T
        diff = approx - target
        eigvals = torch.linalg.eigvalsh(diff)
        assert eigvals.min() >= -1e-4, \
            f"PSD violation: min eigenvalue = {eigvals.min():.2e}"

    def test_gershgorin_psd_bound(self):
        """Gershgorin mode: diag(d) + U U^T >= target in PSD sense."""
        n = 30
        batch = 40
        k = 10
        F = torch.randn(batch, n)
        d = torch.rand(n) * 0.1
        U = torch.randn(n, k) * 0.1
        target = 0.7 * (torch.diag(d) + U @ U.T) + 0.3 * F.T @ F
        d_new, U_new = online_lrpd_update(d, U, F, alpha=0.3, beta=0.7,
                                           residual_mode="gershgorin", rank_k=k)
        approx = torch.diag(d_new) + U_new @ U_new.T
        diff = approx - target
        eigvals = torch.linalg.eigvalsh(diff)
        assert eigvals.min() >= -1e-4, \
            f"PSD violation: min eigenvalue = {eigvals.min():.2e}"


# ========================================================================
# Alt decomposition tests
# ========================================================================

class TestAltDecompose:
    def test_convergence_monotonic(self):
        """Frobenius error should decrease with iterations."""
        A = _make_random_psd(40)
        errors = []
        for iters in [1, 5, 20]:
            d, U = alt_decompose(A, rank_k=10, max_iter=iters)
            err = torch.norm(A - torch.diag(d) - U @ U.T, p='fro').item()
            errors.append(err)
        assert errors[0] >= errors[1] >= errors[2], \
            f"Non-monotonic: {errors}"

    def test_exact_for_low_rank(self):
        """If A is rank-k, Alt should recover it perfectly."""
        n = 30
        k = 5
        F = torch.randn(k, n)
        A = F.T @ F
        d, U = alt_decompose(A, rank_k=k, max_iter=50)
        reconstructed = torch.diag(d) + U @ U.T
        assert torch.allclose(reconstructed, A, atol=1e-4), \
            f"Max diff: {(reconstructed - A).abs().max():.2e}"

    def test_better_than_naive(self):
        """Alt should give lower error than naive (1 iteration)."""
        A = _make_random_psd(40)
        k = 10
        d_naive, U_naive = alt_decompose(A, rank_k=k, max_iter=1)
        err_naive = torch.norm(A - torch.diag(d_naive) - U_naive @ U_naive.T, p='fro').item()
        d_alt, U_alt = alt_decompose(A, rank_k=k, max_iter=50)
        err_alt = torch.norm(A - torch.diag(d_alt) - U_alt @ U_alt.T, p='fro').item()
        assert err_alt <= err_naive + 1e-6, \
            f"Alt ({err_alt:.4e}) worse than naive ({err_naive:.4e})"

    def test_from_factors_matches_direct(self):
        """alt_decompose_from_factors should give same result as alt_decompose."""
        n = 30
        batch = 50
        k = 8
        F = torch.randn(batch, n)
        A = F.T @ F
        d1, U1 = alt_decompose(A, rank_k=k, max_iter=20)
        d2, U2 = alt_decompose_from_factors(F, rank_k=k, max_iter=20)
        err1 = torch.norm(A - torch.diag(d1) - U1 @ U1.T, p='fro').item()
        err2 = torch.norm(A - torch.diag(d2) - U2 @ U2.T, p='fro').item()
        assert abs(err1 - err2) / (err1 + 1e-16) < 0.1, \
            f"Factor-based error {err2:.4e} vs direct {err1:.4e}"

    def test_nystrom_alt(self):
        """Nystrom Alt should produce reasonable approximation."""
        n = 50
        A = _make_random_psd(n)
        k = 10
        d_det, U_det = alt_decompose(A, rank_k=k, max_iter=20)
        err_det = torch.norm(A - torch.diag(d_det) - U_det @ U_det.T, p='fro').item()
        d_nys, U_nys = nystrom_alt_decompose(
            lambda x: A @ x, A.diag(), n, rank_k=k, sketch_size=2*k, max_iter=20
        )
        err_nys = torch.norm(A - torch.diag(d_nys) - U_nys @ U_nys.T, p='fro').item()
        assert err_nys < err_det * 3.0, \
            f"Nystrom error {err_nys:.4e} too far from deterministic {err_det:.4e}"


# ========================================================================
# PSD bound tests
# ========================================================================

class TestPSDBound:
    def test_psd_upper_bound(self):
        """diag(d) + U U^T - A must be PSD."""
        A = _make_random_psd(40)
        d, U, info = psd_upper_bound(A, rank_k=10)
        diff = torch.diag(d) + U @ U.T - A
        eigvals = torch.linalg.eigvalsh(diff)
        assert eigvals.min() >= -1e-5, \
            f"PSD violation: min eigenvalue = {eigvals.min():.2e}"

    def test_tighter_than_naive_spectral(self):
        """Alt-based bound should inflate d less than 1-iteration spectral."""
        A = _make_random_psd(40)
        k = 10
        d_naive, U_naive = alt_decompose(A, rank_k=k, max_iter=1)
        R_naive = A - torch.diag(d_naive) - U_naive @ U_naive.T
        naive_inflation = max(0, -torch.linalg.eigvalsh(R_naive).min().item())
        d_alt, U_alt, info = psd_upper_bound(A, rank_k=k, max_iter=50)
        assert info['inflation'] <= naive_inflation + 1e-6, \
            f"Alt inflation ({info['inflation']:.4e}) > naive ({naive_inflation:.4e})"

    def test_low_rank_matrix_zero_inflation(self):
        """For rank-k matrix, Alt should need zero inflation."""
        n = 30
        k = 5
        F = torch.randn(k, n)
        A = F.T @ F
        d, U, info = psd_upper_bound(A, rank_k=k, max_iter=50)
        assert info['inflation'] < 1e-4, \
            f"Inflation should be ~0 for rank-k matrix, got {info['inflation']:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
