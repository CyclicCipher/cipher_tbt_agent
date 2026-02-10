"""
Online/streaming LRPD update.

Given old (d, U) representing diag(d) + U U^T, and new batch data F,
produce new (d_new, U_new) approximating:

    target = beta * (diag(d) + U U^T) + alpha * F^T F + alpha_prior * diag(d_prior)

Algorithm:
  1. Form combined factor A = [sqrt(beta) * U | sqrt(alpha) * F^T]
  2. Eigendecompose Gram B = A^T A
  3. Keep top-k eigenvectors -> U_new = A @ Q_topk
  4. Absorb dropped eigenspace residual into d via chosen mode
  5. d_new = beta * d + alpha_prior * d_prior + residual_correction
"""

import torch
from typing import Optional, Tuple


def _fitc_residual(A: torch.Tensor, eigenvectors: torch.Tensor, actual_k: int) -> torch.Tensor:
    """FITC: absorb diag(residual) = row-squared-norms of A @ Q_drop.

    Preserves exact diagonal of the target matrix.
    NO PSD guarantee on the approximation error.
    """
    if actual_k >= eigenvectors.shape[1]:
        return torch.zeros(A.shape[0], device=A.device, dtype=A.dtype)
    Q_drop = eigenvectors[:, :-actual_k]
    A_drop = A @ Q_drop
    return (A_drop ** 2).sum(dim=1)


def _gershgorin_residual(A: torch.Tensor, eigenvectors: torch.Tensor, actual_k: int) -> torch.Tensor:
    """Gershgorin: d_i = sum_j |R_ij| where R = A_drop @ A_drop^T.

    Guarantees diag(d) >= R in PSD sense (diagonally dominant).
    Conservative -- inflates d more than necessary.
    """
    if actual_k >= eigenvectors.shape[1]:
        return torch.zeros(A.shape[0], device=A.device, dtype=A.dtype)
    Q_drop = eigenvectors[:, :-actual_k]
    A_drop = A @ Q_drop
    R = A_drop @ A_drop.T
    return R.abs().sum(dim=1)


def _spectral_residual(eigenvalues: torch.Tensor, actual_k: int, n: int, device: torch.device) -> torch.Tensor:
    """Spectral norm: d = max(dropped eigenvalues) * ones(n).

    Guarantees diag(d) >= R in PSD sense (uniform bound).
    Most conservative -- same inflation for all dimensions.
    """
    if actual_k >= len(eigenvalues):
        return torch.zeros(n, device=device)
    max_dropped = eigenvalues[:-actual_k].max()
    return torch.full((n,), max_dropped.item(), device=device)


def online_lrpd_update(
    d: torch.Tensor,
    U: torch.Tensor,
    F: torch.Tensor,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    d_prior: Optional[torch.Tensor] = None,
    alpha_prior: float = 0.0,
    residual_mode: str = "fitc",
    rank_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Online update of LRPD representation.

    Args:
        d: [n] old diagonal
        U: [n, k] old low-rank factor
        F: [batch, n] new data (contributes F^T F to the target)
        alpha: weight on new data
        beta: weight on old (d, U). Default: 1 - alpha.
        d_prior: [n] optional prior diagonal to mix in
        alpha_prior: weight on prior diagonal
        residual_mode: "fitc" | "gershgorin" | "spectral" | "none"
        rank_k: output rank. Default: U.shape[1]

    Returns:
        d_new: [n]
        U_new: [n, rank_k]
    """
    if beta is None:
        beta = 1.0 - alpha
    if rank_k is None:
        rank_k = U.shape[1]

    n = d.shape[0]
    device = d.device

    sqrt_beta = torch.sqrt(torch.clamp(torch.tensor(beta, device=device), min=0.0))
    sqrt_alpha = torch.sqrt(torch.tensor(alpha, device=device))

    A_old = sqrt_beta * U
    A_new = sqrt_alpha * F.T
    A = torch.cat([A_old, A_new], dim=1)

    # Eigendecompose Gram matrix B = A^T A
    B = A.T @ A
    eigenvalues, eigenvectors = torch.linalg.eigh(B)

    # Top-k (last k since eigh sorts ascending)
    actual_k = min(rank_k, len(eigenvalues))
    top_eigenvalues = torch.clamp(eigenvalues[-actual_k:], min=0.0)
    top_eigenvectors = eigenvectors[:, -actual_k:]

    # U_new = A @ Q_topk
    U_new = A @ top_eigenvectors

    # Compute residual correction
    if residual_mode == "fitc":
        residual_d = _fitc_residual(A, eigenvectors, actual_k)
    elif residual_mode == "gershgorin":
        residual_d = _gershgorin_residual(A, eigenvectors, actual_k)
    elif residual_mode == "spectral":
        residual_d = _spectral_residual(eigenvalues, actual_k, n, device)
    elif residual_mode == "none":
        residual_d = torch.zeros(n, device=device)
    else:
        raise ValueError(f"Unknown residual_mode: {residual_mode}")

    # d_new = beta * d_old + alpha_prior * d_prior + residual
    d_new = beta * d + residual_d
    if d_prior is not None and alpha_prior > 0:
        d_new = d_new + alpha_prior * d_prior
    d_new = torch.clamp(d_new, min=1e-8)

    # Pad U if actual_k < rank_k
    if actual_k < rank_k:
        padding = torch.zeros(n, rank_k - actual_k, device=device, dtype=U.dtype)
        U_new = torch.cat([U_new, padding], dim=1)

    return d_new, U_new
