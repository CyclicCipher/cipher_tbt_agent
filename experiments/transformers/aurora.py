"""Aurora — vendored VERBATIM from https://github.com/tilde-research/aurora-release (MIT, (c) 2026 Tilde Research).

Copied rather than paraphrased so the comparison is against the authors' actual update rule, not my reading of it. Source
files `src/polar.py` and `src/aurora.py` at `main`; blog + arXiv 2606.27715.

WHAT IT IS. A Muon variant. Muon orthogonalises the momentum before applying it — replacing the update `M` by the polar
factor `UVᵀ` of its SVD, so every direction moves at the same scale. Aurora's observation is that for a TALL matrix the
plain polar factor spreads the update unevenly across ROWS (neurons): some neurons get much more of the step than others.
It fixes that by iteratively approximating a projection onto the intersection of the ROW-OBLIQUE manifold (all rows the
same norm) and the STIEFEL manifold (orthonormal columns) — a diagonal preconditioner `D` rescaling rows, re-solved for
`pp_iterations` rounds, driving each row's squared norm toward `n/m` without giving up polar-factor precision.

Square or wide matrices fall back to standard Muon, so in a transformer the distinctive path is taken only by the tall
weights — here `qkv` (3d×d) and the first MLP layer (4d×d).

`polar` is CANS-12: nine Chebyshev-optimised cubic Newton–Schulz iterations (arXiv:2506.10935) then three classic
(1.5, −0.5) ones, run in FP32 with the input normalised so the spectral norm is ≤ 1 and the iteration converges.

NOTE FOR CALLERS: this is a FUNCTIONAL update, not a `torch.optim.Optimizer` — it owns the weight write (decoupled weight
decay, then `W -= eta·update`), the caller owns the momentum buffers, and with `nesterov=True` it MUTATES `G` in place.
"""

import torch


@torch.no_grad()
def polar(G: torch.Tensor) -> torch.Tensor:
    """Approximate the polar factor with CANS-12 Newton-Schulz.

    Args:
        G: input matrix of shape [..., m, n].

    Returns:
        An FP32 approximation to polar(G) with the same shape.
    """
    assert G.ndim >= 2
    # Run CANS in FP32
    X = G.to(torch.float32)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm <= 1 so the iteration converges to polar.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    coefficients = (
        (5.182503604966906, -5.178098480082684),
        (2.586120737395915, -0.6479542005271643),
        (2.567364126726186, -0.6454968804392178),
        (2.520560084348265, -0.6393528082067044),
        (2.410759275435182, -0.6248683598710716),
        (2.1883348130094173, -0.5952022073798908),
        (1.8595760874873613, -0.5504490972723968),
        (1.589020160467417, -0.5126569802066718),
        (1.5051653981684994, -0.5007377068751799),
        (1.5, -0.5),
        (1.5, -0.5),
        (1.5, -0.5),
    )
    for a, b in coefficients:
        A = X @ X.mT
        X = a * X + b * A @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.no_grad()
def aurora(
    W,
    G,
    momentum,
    eta=0.05,
    weight_decay=0.025,
    mu=0.95,
    nesterov=True,
    pp_iterations=2,
    pp_beta=0.5,
    eps=1e-7,
):
    if W.ndim != 2:
        raise ValueError(f"aurora expects 2D weight tensors, got shape {tuple(W.shape)}")
    if G.shape != W.shape:
        raise ValueError(f"G shape {tuple(G.shape)} must match W shape {tuple(W.shape)}")
    if momentum.shape != W.shape:
        raise ValueError(f"momentum shape {tuple(momentum.shape)} must match W shape {tuple(W.shape)}")
    if not (0.0 < mu < 1.0):
        raise ValueError(f"mu must be in (0, 1), got {mu}")
    if eta <= 0.0:
        raise ValueError(f"eta must be positive, got {eta}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if pp_iterations < 1:
        raise ValueError(f"pp_iterations must be >= 1, got {pp_iterations}")
    if pp_beta <= 0.0:
        raise ValueError(f"pp_beta must be positive, got {pp_beta}")

    # SGD-momentum (Nesterov by default).
    momentum.lerp_(G, 1 - mu)
    # Clone when not using Nesterov to avoid scaling the momentum buffer in-place below.
    update = G.lerp_(momentum, mu) if nesterov else momentum.clone()
    # Aurora's leverage-uniform polar via diagonal preconditioning.
    m, n = update.size(-2), update.size(-1)
    if m <= n:
        # Square/wide: standard Muon
        update = polar(update)
    else:
        G32 = update.to(torch.float32)
        target_row_sq = n / m
        row_norm = G32.norm(dim=-1, keepdim=True).clamp_(min=eps)
        D = 1.0 / row_norm
        for k in range(pp_iterations):
            U = polar(D * G32)
            if k < pp_iterations - 1:
                row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
                D = D * (target_row_sq / row_sq).pow(pp_beta)
        update = U
    # Spectral aspect-ratio scaling (Muon convention).
    update *= max(1, G.size(-2) / G.size(-1)) ** 0.5
    if not update.isfinite().all():
        raise RuntimeError(
            f"aurora produced non-finite update for parameter of shape {tuple(W.shape)}. "
            "Check for NaN/Inf in gradients or an ill-conditioned weight matrix."
        )
    # Decoupled weight decay then apply.
    W.mul_(1 - eta * weight_decay)
    W.add_(update, alpha=-eta)
    return W
