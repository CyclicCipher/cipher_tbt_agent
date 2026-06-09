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

SOURCE CAVEAT: the only reference is the Tilde blog (no full paper/appendix). The
core transforms here are verified empirically (singular values -> 1; Aurora's row
norms more uniform than Muon's -- see tests). But the exact LR-scaling constant and
wide-matrix handling are best-effort; treat the learning rate as something to sweep,
and compare against Muon (the point of the ladder Adam -> Muon -> Aurora).
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------- orthogonalization
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal polar factor of a 2D matrix (Muon's quintic NS).

    Iterates X <- a X + (b A + c A^2) X with A = X X^T, which drives the singular
    values of X toward 1. Input is Frobenius-normalized first so the spectral norm
    is <= 1 (NS convergence condition). Computed in float32.
    """
    assert G.ndim == 2, "newton_schulz expects a 2D matrix"
    a, b, c = 3.4445, -4.7750, 2.0315
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
                     ns_steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Aurora's alternating projection: K rounds of (damped row-normalize) + polar."""
    Mf = M.float()
    transposed = Mf.shape[0] < Mf.shape[1]   # work tall (m >= n) for the row-norm logic
    if transposed:
        Mf = Mf.T
    m, n = Mf.shape
    X = Mf / (Mf.norm() + eps)
    D = torch.ones(m, device=Mf.device, dtype=Mf.dtype)   # diagonal row-scale EMA (as vector)
    target = (n / m) ** 0.5
    for _ in range(K):
        r = X.norm(dim=1).clamp_min(eps)                  # row norms
        D = D.pow(beta) * r.pow(1.0 - beta)               # damped EMA of row scales
        X = target * (X / D.unsqueeze(1))                 # row-normalize to sqrt(n/m)
        X = newton_schulz(X, steps=ns_steps, eps=eps)     # re-orthogonalize
    if transposed:
        X = X.T
    return X.to(M.dtype)


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
        mom = group.get("momentum", 0.9)
        wd = group.get("weight_decay", 0.0)
        variant = group.get("variant", "aurora")
        ns_steps = group.get("ns_steps", 5)
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
            buf.mul_(mom).add_(g, alpha=1 - mom)          # EMA momentum (Aurora blog)
            if variant == "aurora":
                u = aurora_transform(buf, K=K, beta=beta, ns_steps=ns_steps)
                # Aurora LR scaling: n / ||U||_F  (n = output/col count of the update)
                scale = p.shape[1] / (u.norm() + 1e-7)
            else:  # muon
                u = newton_schulz(buf, steps=ns_steps)
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
            if wd != 0.0:
                p.mul_(1 - lr * wd)                        # decoupled weight decay
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
