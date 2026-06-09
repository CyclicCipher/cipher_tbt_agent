"""SNR-gated AdamW (Litman & Guo 2026, arXiv:2605.01172).

Core idea: a parameter's gradient is *signal* if its mean across the minibatch is
large relative to its variance (coherent across examples = the source/program);
it is *noise* if dominated by per-example variance (idiosyncratic = the sample =
memorization). Gate out the noise directions:

    update parameter k  iff  mu_k^2 > sigma_k^2 / (b - 1)        (b = batch size)

where mu_k is the mean and sigma_k^2 the variance of that parameter's gradient.
Two estimators:
  * "ema"      -- mu = EMA of grad (Adam's m); sigma^2 = EMA of (g - m)^2 (one extra
                  buffer). Cheap, what the paper ships as a one-liner.
  * "faithful" -- mu, sigma^2 from true within-batch per-example gradients
                  (`per_example_snr_gate`, via torch.func). Costly but exact.

Also exposes a validation-free generalization signal (the predicted population-risk
improvement rate): sum of max(0, mu^2 - sigma^2/(b-1)) over parameters.

Deviations from the paper, noted honestly: the gate multiplies the *parameter step*
(not the moment update -- a minor difference), weight decay is applied ungated
(decoupled regularizer), and `gate_warmup` steps run ungated so the EMA estimates
can populate before gating (avoids a dead start where everything looks like noise).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


class SNRAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 batch_size=128, var_beta=0.99, mode="ema", gate_warmup=50):
        if mode not in ("none", "ema", "faithful"):
            raise ValueError(f"mode must be none|ema|faithful, got {mode!r}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        batch_size=batch_size, var_beta=var_beta, mode=mode,
                        gate_warmup=gate_warmup)
        super().__init__(params, defaults)
        self._external_gate: dict[torch.Tensor, torch.Tensor] = {}
        self.last_risk: float = 0.0          # validation-free pop-risk-rate signal
        self.last_gate_frac: float = 1.0     # fraction of params updated this step

    def set_external_gate(self, gate: dict[torch.Tensor, torch.Tensor], risk: float) -> None:
        """For mode='faithful': supply per-parameter masks computed from per-example grads."""
        self._external_gate = gate
        self.last_risk = risk

    @torch.no_grad()
    def step(self, closure=None):  # noqa: C901
        loss = closure() if closure is not None else None
        risk = 0.0
        kept = 0
        total = 0
        for group in self.param_groups:
            b1, b2 = group["betas"]
            b = group["batch_size"]
            mode = group["mode"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            vb, warm = group["var_beta"], group["gate_warmup"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                    st["s"] = torch.zeros_like(p)  # gradient-variance EMA (ema mode)
                st["step"] += 1
                t = st["step"]
                m, v, s = st["m"], st["v"], st["s"]

                if mode == "ema":  # variance proxy uses m BEFORE this step's update
                    s.mul_(vb).addcmul_(g - m, g - m, value=1 - vb)
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)

                if mode == "none" or t <= warm:
                    q = None  # ungated
                elif mode == "ema":
                    s_hat = s / (1 - vb ** t)
                    excess = m_hat * m_hat - s_hat / (b - 1)
                    q = (excess > 0).to(p.dtype)
                    risk += excess.clamp_min(0).sum().item()
                else:  # faithful: mask supplied externally
                    q = self._external_gate.get(p)
                    if q is None:
                        q = torch.ones_like(p)

                step_dir = m_hat / (v_hat.sqrt() + eps)
                if q is not None:
                    step_dir = q * step_dir
                    kept += int(q.sum().item())
                    total += q.numel()
                else:
                    kept += p.numel()
                    total += p.numel()
                p.add_(step_dir, alpha=-lr)
                if wd != 0.0:                       # decoupled weight decay (ungated)
                    p.add_(p, alpha=-lr * wd)

        if mode == "ema":
            self.last_risk = risk
        self.last_gate_frac = kept / max(1, total)
        return loss


def per_example_snr_gate(model, input_ids, targets, loss_mask, batch_size):
    """True within-batch per-example SNR gate for mode='faithful'.

    Computes per-example gradients via torch.func, then per parameter:
        gate_k = 1 if mu_k^2 > var_k/(b-1) else 0
    Returns (gate_dict keyed by the model's parameter tensors, risk float).
    Cost: one vectorized per-example backward (b-fold) -- the "faithful but costly" path.
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_one(prm, ids1, tgt1, m1):
        logits = functional_call(model, (prm, buffers), (ids1.unsqueeze(0),))
        vocab = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, vocab), tgt1.reshape(-1), reduction="none")
        ce = ce.reshape(m1.shape)
        return (ce * m1).sum() / m1.sum().clamp_min(1.0)

    per_ex = vmap(grad(loss_one), in_dims=(None, 0, 0, 0))(params, input_ids, targets, loss_mask)

    gate: dict[torch.Tensor, torch.Tensor] = {}
    risk = 0.0
    denom = batch_size - 1
    name_to_param = dict(model.named_parameters())
    for name, ge in per_ex.items():       # ge: (b, *param_shape)
        mu = ge.mean(0)
        var = ge.var(0, unbiased=False)
        excess = mu * mu - var / denom
        p = name_to_param[name]
        gate[p] = (excess > 0).to(p.dtype)
        risk += excess.clamp_min(0).sum().item()
    return gate, risk
