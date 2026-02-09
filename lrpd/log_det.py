"""
Log-determinant of LRPD matrices via the matrix determinant lemma.

log|diag(d) + U U^T| = sum(log(d)) + log|I_k + U^T diag(1/d) U|

Cost: O(n*k + k^3) via Cholesky of the k x k matrix.
"""

import torch


def lrpd_log_det(
    d: torch.Tensor,
    U: torch.Tensor,
    reg: float = 1e-6,
) -> torch.Tensor:
    """Compute log|diag(d) + U U^T| via the matrix determinant lemma.

    Args:
        d: [n], must be > 0
        U: [n, k]
        reg: regularization for the k x k core matrix

    Returns: scalar tensor
    """
    log_d_sum = torch.log(d).sum()

    k = U.shape[1]
    if k == 0:
        return log_d_sum

    d_inv = 1.0 / d
    d_inv_U = d_inv.unsqueeze(1) * U
    C = torch.eye(k, device=d.device, dtype=d.dtype) + U.T @ d_inv_U
    C = C + reg * torch.eye(k, device=d.device, dtype=d.dtype)
    L = torch.linalg.cholesky(C)
    log_det_C = 2.0 * torch.log(torch.diag(L)).sum()

    return log_d_sum + log_det_C
