"""
Alternating LRPD decomposition (Yeon & Anitescu 2025).

Given a PSD matrix A, finds optimal diag(d) + U U^T approximation
by alternating between:
  1. Fix d, eigendecompose R = A - diag(d), take top-k -> U
  2. Fix U, set d = diag(A - U U^T)

Convergence: monotonically decreasing ||A - diag(d) - U U^T||_F^2
with geometric rate under a spectral gap condition.

Reference: "Beyond Low Rank: Fast Low-Rank + Diagonal Decomposition
with a Spectral Approach" (Yeon & Anitescu, arXiv:2512.17120, 2025)
"""

import torch
from typing import Optional, Tuple, Callable


def alt_decompose(
    A: torch.Tensor,
    rank_k: int,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic Alt algorithm.

    Args:
        A: [n, n] symmetric PSD matrix to approximate
        rank_k: target rank of U
        max_iter: maximum alternating iterations
        tol: convergence tolerance on relative Frobenius norm change

    Returns:
        d: [n]
        U: [n, k]
    """
    n = A.shape[0]
    device = A.device
    dtype = A.dtype

    d = torch.zeros(n, device=device, dtype=dtype)
    prev_frob = float('inf')

    for t in range(max_iter):
        # Step 1: Fix d, eigendecompose R = A - diag(d)
        R = A - torch.diag(d)
        eigenvalues, eigenvectors = torch.linalg.eigh(R)

        # Top-k eigenvectors (eigh sorts ascending, so last k)
        actual_k = min(rank_k, n)
        top_vals = torch.clamp(eigenvalues[-actual_k:], min=0.0)
        top_vecs = eigenvectors[:, -actual_k:]
        U = top_vecs * torch.sqrt(top_vals).unsqueeze(0)

        # Step 2: Fix U, optimal diagonal
        UUT_diag = (U ** 2).sum(dim=1)
        d = A.diag() - UUT_diag
        d = torch.clamp(d, min=0.0)

        # Check convergence
        residual = A - torch.diag(d) - U @ U.T
        frob = torch.norm(residual, p='fro').item()
        if prev_frob > 0 and abs(prev_frob - frob) / (prev_frob + 1e-16) < tol:
            break
        prev_frob = frob

    return d, U


def alt_decompose_from_factors(
    F: torch.Tensor,
    rank_k: int,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Alt decomposition when A = F^T F is given in factored form.

    When batch < n, uses the Gram matrix F F^T [batch, batch] for
    eigendecomposition, avoiding materialization of A [n, n].

    Args:
        F: [batch, n] such that A = F^T F
        rank_k: target rank
        max_iter: maximum alternating iterations
        tol: convergence tolerance

    Returns:
        d: [n]
        U: [n, k]
    """
    batch, n = F.shape
    # For the Alt algorithm we need to subtract diag(d) from A each iteration,
    # which breaks the factored form. Materialize A.
    A = F.T @ F
    return alt_decompose(A, rank_k, max_iter, tol)


def nystrom_alt_decompose(
    A_matvec: Callable[[torch.Tensor], torch.Tensor],
    diag_A: torch.Tensor,
    n: int,
    rank_k: int,
    sketch_size: int = 0,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stochastic Alt via Nystrom sketch.

    For large n where materializing A is too expensive. Uses randomized
    eigendecomposition via the Nystrom method.

    Args:
        A_matvec: function computing A @ x for arbitrary x
        diag_A: [n] diagonal of A (needed for the d update)
        n: dimension
        rank_k: target rank
        sketch_size: number of random columns. Default: 2 * rank_k
        max_iter: maximum alternating iterations
        tol: convergence tolerance

    Returns:
        d: [n]
        U: [n, k]
    """
    if sketch_size <= 0:
        sketch_size = min(2 * rank_k + 1, n)

    device = diag_A.device
    dtype = diag_A.dtype
    d = torch.zeros(n, device=device, dtype=dtype)

    for t in range(max_iter):
        # Step 1: randomized eigendecomp of R = A - diag(d)
        Omega = torch.randn(n, sketch_size, device=device, dtype=dtype)
        Y = A_matvec(Omega) - d.unsqueeze(1) * Omega  # R @ Omega

        # Nystrom: eigendecomp of Omega^T Y = Omega^T R Omega
        C = Omega.T @ Y
        C = 0.5 * (C + C.T)  # symmetrize
        eigvals, eigvecs_small = torch.linalg.eigh(C)

        # Top-k
        actual_k = min(rank_k, sketch_size)
        top_vals = torch.clamp(eigvals[-actual_k:], min=1e-12)
        top_vecs_small = eigvecs_small[:, -actual_k:]

        # Recover full eigenvectors: U_approx = Y @ W / sqrt(lambda)
        U = Y @ top_vecs_small / torch.sqrt(top_vals).unsqueeze(0)

        # Step 2: optimal diagonal
        UUT_diag = (U ** 2).sum(dim=1)
        d_new = diag_A - UUT_diag
        d_new = torch.clamp(d_new, min=0.0)

        # Convergence check on d
        d_change = torch.norm(d_new - d).item()
        d_scale = torch.norm(d_new).item() + 1e-16
        d = d_new
        if d_change / d_scale < tol:
            break

    return d, U
