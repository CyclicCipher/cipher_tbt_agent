"""Muon and Aurora optimizers (matrix-aware), with AdamW for the rest.

Muon (Jordan et al.): orthogonalize the momentum of each 2D weight matrix via a
Newton-Schulz iteration (an approximate polar factor that pushes all singular
values toward 1), then take a spectrally-normalized step. Embeddings, the LM head,
norms, and other non-matrix params use AdamW.

Aurora (Tilde Research, blog.tilderesearch.com/blog/aurora): a "leverage-aware"
improvement on Muon. Muon's orthogonalized update inherits row-norm anisotropy from
the gradient, so low-leverage rows (neurons) stay starved -> "neuron death" in tall
MLP matrices. Aurora alternates row-normalization (to target row norm sqrt(n/m),
with a damped-EMA of row scales) with the polar factor, forcing uniform row norms
*and* (approximate) semi-orthogonality. Claim 1: a tall matrix cannot be exactly
column-orthogonal AND uniform-unit-row-norm, so the target is sqrt(n/m), not 1.

Matched to the official repo (github.com/tilde-research/aurora-release, src/aurora.py,
src/polar.py): EMA+Nesterov momentum (mu=0.95), the "simple quintic" polar
(a=2,b=-1.5,c=0.5, 12 iters, Frobenius-normalized), the D-refinement row-balancing
loop (target row energy n/m, damping pp_beta=0.5, pp_iterations=2), Muon-style
spectral scaling max(1,m/n)^0.5, and decoupled weight decay (default 0.025).
Verified empirically (tests): Aurora revives starved rows + uniformizes row norms
vs plain polar (Muon). The polar is float32 here for CPU/stability (repo uses bf16).
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------- orthogonalization
def newton_schulz(G: torch.Tensor, steps: int = 12, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal polar factor (repo's "simple quintic", 12 iters).

    Iterates X <- a X + (b A + c A^2) X with A = X X^T and FIXED coefficients
    a=2, b=-1.5, c=0.5 -- the gentle monotone quintic p(s)=2s-1.5s^3+0.5s^5 with
    p(1)=1, p'(1)=0, driving singular values toward 1. Frobenius-normalized first
    (spectral norm <= 1 => convergence). float32.
    """
    assert G.ndim == 2, "newton_schulz expects a 2D matrix"
    a, b, c = 2.0, -1.5, 0.5
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:                      # iterate on the orientation with fewer rows
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def aurora_transform(M: torch.Tensor, K: int = 2, beta: float = 0.5,
                     ns_steps: int = 12, eps: float = 1e-7) -> torch.Tensor:
    """Aurora row-balancing (repo src/aurora.py). Find a row-scaling D so that
    polar(D * M) has uniform row energy (target n/m), by fixed-point refinement:

        D_0 = 1 / rownorm(M)
        for k in 0..K-1:  U = polar(D * M);  if k<K-1: D *= (n/m / rowsq(U))^beta

    The constraint (Claim 1): a tall matrix can't be column-orthogonal AND have
    uniform unit rows, so the target row energy is n/m, not 1.

    On SQUARE matrices the row-balancing is trivial (target n/m = 1, and a square
    orthogonal matrix already has unit rows), so Aurora reduces to plain polar
    (Muon). The speedrun (PR #284) routes square q/k/v/o projections to Muon for
    exactly this reason; we short-circuit here to the same effect.
    """
    transposed = M.shape[0] < M.shape[1]     # canonicalize tall (m >= n)
    G = (M.T if transposed else M).float()
    m, n = G.shape
    if m == n:                               # square -> Aurora == Muon
        U = newton_schulz(G, steps=ns_steps, eps=eps)
        return (U.T if transposed else U).to(M.dtype)
    target = n / m
    D = 1.0 / G.norm(dim=1).clamp_min(eps)   # (m,) initial row scaling
    U = None
    for k in range(K):
        U = newton_schulz(D.unsqueeze(1) * G, steps=ns_steps, eps=eps)
        if k < K - 1:
            row_sq = (U * U).sum(dim=1).clamp_min(eps)
            D = D * (target / row_sq).pow(beta)
    if transposed:
        U = U.T
    return U.to(M.dtype)


# ---------------------------------------------------------------- the optimizer
class MuonAuroraAdamW(torch.optim.Optimizer):
    """Hybrid optimizer. Param groups carry ``use_muon``:
      * use_muon=True  -> Muon ("muon") or Aurora ("aurora") update for 2D matrices.
      * use_muon=False -> standard AdamW (embeddings, head, norms, biases).
    Build the two groups in the training script (route 2D non-embedding/head weights
    to the muon group, everything else to the adam group).
    """

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):  # noqa: C901
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_group(group)
            else:
                self._adamw_group(group)
        return loss

    def _muon_group(self, group) -> None:
        lr = group["lr"]
        mu = group.get("momentum", 0.95)
        wd = group.get("weight_decay", 0.025)
        nesterov = group.get("nesterov", True)
        variant = group.get("variant", "aurora")
        ns_steps = group.get("ns_steps", 12)
        K = group.get("aurora_K", 2)
        beta = group.get("aurora_beta", 0.5)
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            if g.ndim != 2:  # safety: only matrices here
                continue
            st = self.state[p]
            if "buf" not in st:
                st["buf"] = torch.zeros_like(p)
            buf = st["buf"]
            buf.mul_(mu).add_(g, alpha=1 - mu)            # EMA momentum
            upd = g.add(buf, alpha=mu) if nesterov else buf   # Nesterov: G + mu*momentum
            if variant == "aurora":
                u = aurora_transform(upd, K=K, beta=beta, ns_steps=ns_steps)
            else:  # muon: plain polar (same NS as Aurora; isolates the row-balancing)
                u = newton_schulz(upd, steps=ns_steps)
            scale = (max(p.shape) / min(p.shape)) ** 0.5  # Muon spectral scale max(1,m/n)^0.5
            if wd != 0.0:
                p.mul_(1 - lr * wd)                       # decoupled weight decay
            p.add_(u, alpha=-lr * scale)

    def _adamw_group(self, group) -> None:
        lr = group["lr"]
        b1, b2 = group.get("betas", (0.9, 0.999))
        eps = group.get("eps", 1e-8)
        wd = group.get("weight_decay", 0.0)
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "step" not in st:
                st["step"] = 0
                st["m"] = torch.zeros_like(p)
                st["v"] = torch.zeros_like(p)
            st["step"] += 1
            t = st["step"]
            m, v = st["m"], st["v"]
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            m_hat = m / (1 - b1 ** t)
            v_hat = v / (1 - b2 ** t)
            if wd != 0.0:
                p.mul_(1 - lr * wd)
            p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)


def split_matrix_params(model):
    """Route params: 2D weights that are NOT embeddings/head -> muon group; rest -> adam.
    Returns (matrix_params, other_params)."""
    matrix, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.ndim == 2 and ("embed" not in name) and ("head" not in name) and ("pos" not in name)
        (matrix if is_matrix else other).append(p)
    return matrix, other
