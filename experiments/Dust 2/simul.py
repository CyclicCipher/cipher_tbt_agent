"""T3 (THERMODYNAMICS.md §2, fluctuation–dissipation): every site perturbed at once, in every pass, with no clean pass
and no cached activations.

Each draw is one full forward of the batch with independent noise added at EVERY linear output (embedding, q/k/v,
attention projection, MLP hidden, MLP output, head logits). Each site's weights are updated by the three-factor rule
Σ_t (reward_t − chunk mean) · a_t x_tᵀ, where:
- a_t is the noise the site received;
- x_t is the input it actually saw in THAT draw;
- reward_t = −(token t's loss).

Nothing is stored between draws or between layers beyond the current pass. Cost = K forward passes per step.
"""
from __future__ import annotations

import torch

from model import rms


def noisy_pass(m, idx, tgt, sig, sig_head, g, local=False):
    """One forward of N = Kc*B rows (idx, tgt already tiled) with noise at every site. Returns per-token losses (N, T) and
    {site: (a, x)} with a = the noise (N, T, D_out) and x = the site's input in this pass (N, T, D_in)."""
    N, T = idx.shape
    rec = {}

    def nz(shape, dt):
        return torch.randn(shape, generator=g, device=idx.device, dtype=dt)

    e = m.wte[idx] + m.wpe[:T][None]
    a = nz(e.shape, e.dtype)
    rec[("emb",)] = (a, None)
    x = rms(e + sig * a)
    for b in range(m.L):
        u = rms(x)
        qkv = u @ m.Wqkv[b].T
        a = nz(qkv.shape, qkv.dtype)
        rec[("qkv", b)] = (a, u)
        o = m.attend(qkv + sig * a)
        p = o @ m.Wproj[b].T
        a = nz(p.shape, p.dtype)
        rec[("proj", b)] = (a, o)
        x = x + p + sig * a
        u = rms(x)
        h = u @ m.Wfc[b].T
        a = nz(h.shape, h.dtype)
        rec[("fc", b)] = (a, u)
        z = m.act(h + sig * a, b)
        mo = z @ m.Wout[b].T
        a = nz(mo.shape, mo.dtype)
        rec[("out", b)] = (a, z)
        x = x + mo + sig * a
        if local and b in m.aux_blocks:                       # Experiment 5 W3: an auxiliary head, also perturbed
            ua = rms(x)
            za = ua @ m.Waux[str(b)].T
            a = nz(za.shape, za.dtype)
            rec[("aux", b)] = (a, ua)
            rec["loss_aux", b] = m.tok_loss(za + sig_head * a, tgt)
    u = rms(x)
    zl = u @ m.Whead.T
    a = nz(zl.shape, zl.dtype)
    rec[("head",)] = (a, u)
    loss = m.tok_loss(zl + sig_head * a, tgt)                 # tgt is already tiled to N rows
    return loss, rec


@torch.no_grad()
def score_of(m, s, local):
    """Which loss scores site s: the final loss, or (local) the first auxiliary head at or after the site's block."""
    if not local or s[0] == "head":
        return None
    b = s[1] if s[0] != "emb" else 0
    nxt = [e for e in m.aux_blocks if e >= b]
    return nxt[0] if nxt else None


def estimate_simul(model_fp32, idx, tgt, K, sig, sig_head, chunk, g, shadow_model, local=False):
    """Returns {parameter: gradient estimate of the mean per-token loss} and the cost in forward-equivalents."""
    m = shadow_model
    B, T = idx.shape
    N = B * T
    acc = {}
    for j in range(0, K, chunk):
        kc = min(chunk, K - j)
        I = idx.repeat(kc, 1)
        Tg = tgt.repeat(kc, 1)
        loss, rec = noisy_pass(m, I, Tg, sig, sig_head, g, local)
        rewards = {}
        for key in [None] + list(m.aux_blocks):
            lt = loss if key is None else rec.get(("loss_aux", key))
            if lt is None:
                continue
            r = -lt.view(kc, B, T)
            rewards[key] = r - r.mean(0, keepdim=True)        # chunk-mean baseline: no clean pass
        for s, v in rec.items():
            if s[0] == "loss_aux":
                continue
            a, x = v
            r = rewards[s[1]] if s[0] == "aux" else rewards[score_of(m, s, local)]
            sg = sig_head if s[0] in ("head", "aux") else sig
            a = a.view(kc, B, T, -1)
            ra = r[..., None].to(a.dtype) * a                     # (kc, B, T, D_out)
            if s[0] == "emb":
                ge = ra.float().sum(0).reshape(-1, m.d)
                acc.setdefault("wte", torch.zeros_like(model_fp32.wte)).index_add_(0, idx.reshape(-1), ge)
                acc.setdefault("wpe", torch.zeros_like(model_fp32.wpe)).index_add_(
                    0, torch.arange(T, device=idx.device).repeat(B), ge)
                continue
            G = torch.einsum("kbtd,kbti->di", ra, x.view(kc, B, T, -1)).float() / sg
            acc[s] = acc.get(s, 0) + G
    grads = {}
    for s, G in acc.items():
        if s == "wte":
            grads[model_fp32.wte] = -G / (K * sig * N)
        elif s == "wpe":
            grads[model_fp32.wpe] = -G / (K * sig * N)
        else:
            grads[model_fp32.weight_of(s)] = -G / (K * N)
    return grads, float(K)
