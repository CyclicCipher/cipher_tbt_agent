"""
PSD upper bound LRPD decomposition (stretch goal).

Finds (d, U) such that diag(d) + U U^T >= A in Loewner order,
while keeping the bound as tight as possible.

Strategy:
  1. Run Alt decomposition to get tight (d*, U*) minimizing ||A - d* - U* U*^T||_F
  2. Compute residual R = A - diag(d*) - U* U*^T (generally indefinite)
  3. Find lambda_min(R) (most negative eigenvalue)
  4. Inflate: d_final = d* + max(0, -lambda_min(R))

This inflation is tighter than Gershgorin because Alt's d* already
absorbs most of the spectral tail. The residual norm is bounded by
lambda_{k+1} of the original matrix and shrinks with Alt iterations.
"""

import torch
from typing import Tuple

from .alt_decompose import alt_decompose


def psd_upper_bound(
    A: torch.Tensor,
    rank_k: int,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Find (d, U) such that diag(d) + U U^T >= A in Loewner order.

    Args:
        A: [n, n] symmetric PSD matrix
        rank_k: target rank
        max_iter: Alt iterations
        tol: convergence tolerance

    Returns:
        d: [n] inflated diagonal
        U: [n, k] low-rank factor
        info: dict with diagnostics (inflation amount, residual norm, etc.)
    """
    d, U = alt_decompose(A, rank_k, max_iter, tol)

    # Compute residual
    residual = A - torch.diag(d) - U @ U.T
    residual = 0.5 * (residual + residual.T)

    # Find most negative eigenvalue
    eigvals = torch.linalg.eigvalsh(residual)
    lambda_min = eigvals.min().item()

    inflation = max(0.0, -lambda_min)
    d_inflated = d + inflation

    # Verify PSD guarantee
    check = torch.diag(d_inflated) + U @ U.T - A
    check_eigvals = torch.linalg.eigvalsh(check)

    info = {
        'inflation': inflation,
        'residual_spectral_norm': max(abs(eigvals.min().item()), abs(eigvals.max().item())),
        'alt_frob_error': torch.norm(residual, p='fro').item(),
        'psd_guarantee_min_eig': check_eigvals.min().item(),
    }

    return d_inflated, U, info


def psd_upper_bound_from_eigendecomp(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    rank_k: int,
    alt_refine_iters: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """PSD upper bound when eigendecomposition of A is already available.

    Uses Alt refinement starting from the spectral initialization to
    tighten the diagonal before inflating.

    Args:
        eigenvalues: [n] eigenvalues of A
        eigenvectors: [n, n] corresponding eigenvectors
        rank_k: target rank
        alt_refine_iters: Alt iterations to refine the decomposition

    Returns:
        d: [n] inflated diagonal
        U: [n, k] low-rank factor
        info: dict with diagnostics
    """
    A = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    return psd_upper_bound(A, rank_k, max_iter=alt_refine_iters)
