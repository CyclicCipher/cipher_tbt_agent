"""
Efficient Schur complement diagonal for LRPD matrices.

Given M [m, n], d [n], U [n, k]:
  diag(M (diag(d) + U U^T) M^T) = (M^2) @ d + ((M @ U)^2).sum(dim=1)

Cost: O(m*n + m*k), no materialization of n x n or m x m matrices.
"""

import torch


def schur_complement_diag(
    M: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """Compute diag(M (diag(d) + U U^T) M^T) efficiently.

    Args:
        M: [m, n]
        d: [n]
        U: [n, k]

    Returns: [m]
    """
    diag_part = (M ** 2) @ d
    MU = M @ U
    lowrank_part = (MU ** 2).sum(dim=1)
    return diag_part + lowrank_part


def schur_complement_full_diag(
    eta3_diag: torch.Tensor,
    M: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """Compute diag(eta3) - diag(M eta1 M^T), the MNW Schur complement diagonal.

    For valid MNW, the result (Phi diagonal) must be > 0.

    Args:
        eta3_diag: [m]
        M: [m, n]
        d: [n]
        U: [n, k]

    Returns: [m] Phi diagonal
    """
    return eta3_diag - schur_complement_diag(M, d, U)
