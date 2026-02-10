"""
AdaWoodbury: Adam + rank-1 LRPD curvature correction.

Adam's diagonal second-moment + rank-1 cross-parameter curvature direction
via online PCA of the gradient stream. The step is computed via the
Woodbury identity in O(n) time with O(n) extra memory (one curvature vector).

Step computation:
  H ≈ diag(√v + ε) + α·uu^T            (LRPD Hessian approximation)
  step = H^{-1}·m_hat                   (Woodbury inversion)
       = m_hat/D - α·(u^T(m_hat/D))/(1 + α·u^T(u/D)) · u/D
  where D = √v_hat + ε

When α=0, reduces exactly to Adam. The rank-1 correction modulates the
step along the dominant cross-parameter curvature direction, reducing
it where there's excess correlated curvature beyond what the diagonal
captures.

α is estimated adaptively: ratio of actual curvature along u to what
the diagonal predicts. When ratio > 1, there's excess off-diagonal
curvature → correction kicks in.

Memory: 3n per parameter (m, v, u) vs Adam's 2n (m, v). At 1B params:
12GB vs 8GB — 50% overhead.

Compute: 3 extra dot products per step per parameter tensor (negligible).
"""

import torch
from torch.optim.optimizer import Optimizer


class AdaWoodbury(Optimizer):
    """Adam with rank-1 Woodbury curvature correction.

    Args:
        params: iterable of parameters
        lr: learning rate (default: 1e-3)
        betas: coefficients for first/second moment EMA (default: (0.9, 0.999))
        beta_u: EMA coefficient for curvature direction update (default: 0.99)
        alpha: base strength of rank-1 correction (default: 1.0).
            Multiplied by the estimated excess curvature ratio.
            Set to 0.0 for pure Adam behavior.
        eps: numerical stability (default: 1e-8)
        weight_decay: AdamW-style decoupled weight decay (default: 0.0)
        warmup_steps: pure Adam steps while u converges (default: 100)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta_u=0.99,
                 alpha=1.0, eps=1e-8, weight_decay=0.0, warmup_steps=100):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, beta_u=beta_u, alpha=alpha,
                        eps=eps, weight_decay=weight_decay,
                        warmup_steps=warmup_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            beta_u = group['beta_u']
            base_alpha = group['alpha']
            eps = group['eps']
            wd = group['weight_decay']
            warmup = group['warmup_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # --- Initialize state ---
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    # Curvature direction (per-parameter tensor, unit norm)
                    state['u'] = torch.randn_like(p.data)
                    state['u'].div_(state['u'].norm().clamp(min=eps))
                    state['proj_sq_ema'] = 0.0

                state['step'] += 1
                t = state['step']
                m, v, u = state['m'], state['v'], state['u']

                # --- AdamW decoupled weight decay ---
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)

                # --- Adam moment updates ---
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)
                denom = v_hat.sqrt().add_(eps)

                # Adam step (baseline)
                adam_step = m_hat / denom

                # --- Online PCA: update curvature direction ---
                use_correction = (base_alpha > 0 and p.numel() > 1)

                if use_correction:
                    g_flat = grad.flatten()
                    u_flat = u.flatten()

                    # Streaming power iteration: u → top eigenvector of E[gg^T]
                    proj = torch.dot(g_flat, u_flat)
                    u_new = u_flat.mul(beta_u).add_(g_flat, alpha=(1.0 - beta_u) * proj.item())
                    u_norm = u_new.norm()
                    if u_norm > eps:
                        u_new.div_(u_norm)
                    u.copy_(u_new.view_as(u))

                    # Curvature along u: EMA of (g^T u)²
                    proj_sq = proj.item() ** 2
                    state['proj_sq_ema'] = (beta_u * state['proj_sq_ema']
                                            + (1.0 - beta_u) * proj_sq)
                    pse_bc = state['proj_sq_ema'] / (1.0 - beta_u ** t)

                # --- Woodbury correction (after warmup) ---
                if use_correction and t > warmup:
                    # What the diagonal already explains along u
                    u_flat = u.flatten()
                    diag_curv = torch.dot(u_flat * u_flat, v_hat.flatten()).item()

                    # Excess curvature ratio: > 0 means off-diagonal structure
                    ratio = pse_bc / (diag_curv + eps)
                    alpha_eff = max(0.0, ratio - 1.0) * base_alpha

                    if alpha_eff > eps:
                        # Woodbury: (D + α uu^T)^{-1} m = m/D - α(u^T(m/D))/(1+α u^T(u/D)) · u/D
                        u_scaled = u / denom
                        c = torch.dot(u.flatten(), adam_step.flatten())
                        s = torch.dot(u.flatten(), u_scaled.flatten())
                        correction = alpha_eff * c.item() / (1.0 + alpha_eff * s.item())
                        p.data.add_(adam_step, alpha=-lr)
                        p.data.add_(u_scaled, alpha=lr * correction)
                    else:
                        p.data.add_(adam_step, alpha=-lr)
                else:
                    p.data.add_(adam_step, alpha=-lr)

        return loss

    def get_diagnostics(self):
        """Return per-parameter curvature diagnostics.

        Returns a dict mapping parameter id to:
          - alpha_eff: effective correction strength
          - ratio: curvature ratio (>1 = excess off-diagonal curvature)
          - u_norm: norm of curvature direction (should be ~1.0)
        """
        diags = {}
        for group in self.param_groups:
            beta_u = group['beta_u']
            base_alpha = group['alpha']
            eps = group['eps']
            warmup = group['warmup_steps']

            for p in group['params']:
                state = self.state.get(p, {})
                if not state or p.numel() <= 1:
                    continue

                t = state['step']
                u = state['u']
                v = state['v']

                if t <= warmup:
                    diags[id(p)] = {'alpha_eff': 0.0, 'ratio': 0.0,
                                    'u_norm': u.norm().item(), 'step': t}
                    continue

                beta1, beta2 = group['betas']
                v_hat = v / (1.0 - beta2 ** t)
                u_flat = u.flatten()
                diag_curv = torch.dot(u_flat * u_flat, v_hat.flatten()).item()
                pse_bc = state['proj_sq_ema'] / (1.0 - beta_u ** t)
                ratio = pse_bc / (diag_curv + eps)
                alpha_eff = max(0.0, ratio - 1.0) * base_alpha

                diags[id(p)] = {
                    'alpha_eff': alpha_eff,
                    'ratio': ratio,
                    'u_norm': u.norm().item(),
                    'step': t,
                }

        return diags
