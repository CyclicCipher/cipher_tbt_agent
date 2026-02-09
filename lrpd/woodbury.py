"""
Woodbury identity operations for LRPD matrices.

For A = diag(d) + U U^T:
  A^{-1} = diag(1/d) - diag(1/d) U C^{-1} U^T diag(1/d)
  where C = I_k + U^T diag(1/d) U  [k x k, cheap to invert]

Cost: O(n k^2) instead of O(n^3) for full inverse.
"""

import torch
from typing import Tuple


def woodbury_inverse_components(
    d: torch.Tensor,
    U: torch.Tensor,
    reg: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Woodbury components for (diag(d) + U U^T)^{-1}.

    Args:
        d: [n] diagonal, must be > 0
        U: [n, k] low-rank factor
        reg: regularization added to C for numerical stability

    Returns:
        d_inv:   [n]     1/d
        C_inv:   [k, k]  (I_k + U^T diag(1/d) U)^{-1}
        d_inv_U: [n, k]  diag(1/d) @ U
    """
    d_inv = 1.0 / d
    d_inv_U = d_inv.unsqueeze(1) * U
    k = U.shape[1]
    C = torch.eye(k, device=d.device, dtype=d.dtype) + U.T @ d_inv_U
    C = C + reg * torch.eye(k, device=d.device, dtype=d.dtype)
    L = torch.linalg.cholesky(C)
    C_inv = torch.cholesky_inverse(L)
    return d_inv, C_inv, d_inv_U


def lrpd_solve(
    d: torch.Tensor,
    U: torch.Tensor,
    B: torch.Tensor,
    reg: float = 1e-6,
) -> torch.Tensor:
    """Compute B @ (diag(d) + U U^T)^{-1} via Woodbury.

    Args:
        d: [n]
        U: [n, k]
        B: [m, n]
        reg: regularization for C

    Returns: [m, n]
    """
    d_inv, C_inv, d_inv_U = woodbury_inverse_components(d, U, reg)
    B_dinv = B * d_inv.unsqueeze(0)
    B_dinv_U = B_dinv @ U
    correction = (B_dinv_U @ C_inv) @ d_inv_U.T
    return B_dinv - correction


def lrpd_matvec(
    d: torch.Tensor,
    U: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute (diag(d) + U U^T) @ x without materializing the full matrix.

    Args:
        d: [n]
        U: [n, k]
        x: [n] or [batch, n]

    Returns: same shape as x
    """
    if x.dim() == 1:
        return d * x + U @ (U.T @ x)
    else:
        return d.unsqueeze(0) * x + (x @ U) @ U.T


def lrpd_inv_diag(
    d: torch.Tensor,
    U: torch.Tensor,
    reg: float = 1e-6,
) -> torch.Tensor:
    """Compute diag((diag(d) + U U^T)^{-1}) via Woodbury.

    Args:
        d: [n]
        U: [n, k]
        reg: regularization for C

    Returns: [n]
    """
    d_inv, C_inv, d_inv_U = woodbury_inverse_components(d, U, reg)
    correction = (d_inv_U @ C_inv) * d_inv_U
    return d_inv - correction.sum(dim=1)
