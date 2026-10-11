"""A tiny GPT in the shape of Dust's model, written so that every linear OUTPUT is a named SITE.

Sites (where noise can be added and an error estimated):
  ("emb",)        token + position embedding, before the norm                 -> wte (scatter-add), wpe
  ("qkv", b)      the q, k, v projections of block b                          -> Wqkv[b]
  ("o", b)        the attention output before the projection (a hub, no weight; Dust's `o@b`)
  ("proj", b)     the attention projection, a residual writer                -> Wproj[b]
  ("fc", b)       the MLP's hidden layer, before its activation               -> Wfc[b]
  ("out", b)      the MLP's output projection, a residual writer             -> Wout[b]
  ("head",)       the vocabulary logits, before the soft-cap                  -> Whead
  ("aux", b)      an auxiliary next-character head read after block b (local losses, Gemini idea 3) -> Waux[b]

`forward_cache` runs the clean pass and keeps every site's output and every linear layer's input; `resume(site, value)`
runs the rest of the network from a site, given that site's (perturbed) output for Kc stacked copies of the batch, and
returns per-token losses. Pre-norm blocks, parameter-free RMSNorm, ReLU² MLP, no biases, logit soft-cap 15 (as Dust);
learned absolute positions instead of rotary.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def rms(x):
    return F.rms_norm(x, (x.shape[-1],))


def _attn_half(x, Wqkv, Wproj, H):
    N, T, d = x.shape
    q, k, v = (rms(x) @ Wqkv.T).view(N, T, 3, H, d // H).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(N, T, d)
    return x + o @ Wproj.T


def _mlp_half(x, Wfc, Wout, topk_frac, theta=None):
    h = rms(x) @ Wfc.T
    z = F.relu(h if theta is None else h - theta).square()
    if topk_frac:
        k = max(1, int(round(topk_frac * h.shape[-1])))
        z = z * (h >= h.topk(k, dim=-1).values[..., -1:])
    return x + z @ Wout.T


_COMPILED = {}


def _halves(fast):
    if not fast:
        return _attn_half, _mlp_half
    if not _COMPILED:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 64
        _COMPILED["a"] = torch.compile(_attn_half, dynamic=False)
        _COMPILED["m"] = torch.compile(_mlp_half, dynamic=False)
    return _COMPILED["a"], _COMPILED["m"]


class TinyGPT(nn.Module):
    fast = False                                   # torch.compile the rerun path (apparatus; same maths)

    def __init__(self, V, d=64, L=4, H=4, T=64, mlp=4, act="relu2", aux_every=0, topk_frac=0.0, ev=False):
        super().__init__()
        self.ev = ev                               # event-driven hidden units: z = (h - theta_j)_+^2, per-unit threshold
        if ev:
            self.register_buffer("theta", torch.zeros(L, mlp * d))
        self.V, self.d, self.L, self.H, self.T, self.m = V, d, L, H, T, mlp * d
        self.act_kind, self.topk_frac = act, topk_frac
        s = math.sqrt(3) / math.sqrt(d)
        u = lambda o, i, sc: nn.Parameter(torch.empty(o, i).uniform_(-sc, sc))
        self.wte = nn.Parameter(torch.randn(V, d))
        self.wpe = nn.Parameter(torch.randn(T, d))
        self.Wqkv = nn.ParameterList([u(3 * d, d, s) for _ in range(L)])
        self.Wproj = nn.ParameterList([nn.Parameter(torch.zeros(d, d)) for _ in range(L)])
        self.Wfc = nn.ParameterList([u(self.m, d, s) for _ in range(L)])
        self.Wout = nn.ParameterList([nn.Parameter(torch.zeros(d, self.m)) for _ in range(L)])
        self.Whead = nn.Parameter(torch.randn(V, d) * 1e-3)
        # auxiliary heads after blocks aux_every-1, 2*aux_every-1, ... (never after the last block)
        self.aux_blocks = [b for b in range(L - 1) if aux_every and (b + 1) % aux_every == 0]
        self.Waux = nn.ParameterDict({str(b): nn.Parameter(torch.randn(V, d) * 1e-3) for b in self.aux_blocks})

    # ---- parameters by site -------------------------------------------------------------------------------------------
    def weight_of(self, site):
        k = site[0]
        return dict(qkv=lambda: self.Wqkv[site[1]], proj=lambda: self.Wproj[site[1]], fc=lambda: self.Wfc[site[1]],
                    out=lambda: self.Wout[site[1]], head=lambda: self.Whead, aux=lambda: self.Waux[str(site[1])])[k]()

    def weight_sites(self):
        s = [("emb",)]
        for b in range(self.L):
            s += [("qkv", b), ("proj", b), ("fc", b), ("out", b)]
        return s + [("head",)] + [("aux", b) for b in self.aux_blocks]

    # ---- pieces -------------------------------------------------------------------------------------------------------
    def act(self, h, b=None):
        if self.ev:
            return F.relu(h - self.theta[b].to(h.dtype)).square()
        z = F.relu(h).square()
        if self.topk_frac:
            k = max(1, int(round(self.topk_frac * h.shape[-1])))
            thr = h.topk(k, dim=-1).values[..., -1:]
            z = z * (h >= thr)
        return z

    def margin(self, h, b=None):
        """How far each hidden unit is from being active (<= 0: active): ReLU² -> -h; top-k -> max(threshold, 0) - h;
        event-driven -> theta_j - h."""
        if self.ev:
            return self.theta[b].to(h.dtype) - h
        if self.topk_frac:
            k = max(1, int(round(self.topk_frac * h.shape[-1])))
            thr = h.topk(k, dim=-1).values[..., -1:].clamp(min=0)
            return thr - h
        return -h

    def attend(self, qkv):
        N, T, _ = qkv.shape
        q, k, v = qkv.view(N, T, 3, self.H, self.d // self.H).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return o.transpose(1, 2).reshape(N, T, self.d)

    def logits(self, x, W):
        z = rms(x) @ W.T
        return z

    @staticmethod
    def cap(z):
        return 15 * torch.tanh(z / 15)

    def tok_loss(self, z, tgt):
        """z: (N, T, V) pre-cap logits; tgt (B, T) is tiled to N rows. Returns (N, T)."""
        N, T, V = z.shape
        t = tgt.repeat(N // tgt.shape[0], 1)
        return F.cross_entropy(self.cap(z).reshape(-1, V).float(), t.reshape(-1), reduction="none").view(N, T)

    # ---- the clean pass -----------------------------------------------------------------------------------------------
    def forward_cache(self, idx, tgt, retain=False):
        """Returns dict c: site outputs c[site], linear inputs c["in", site], residual-stream inputs c["xa", b] (before
        the attention half) and c["xm", b] (before the MLP half), per-token losses c["loss"] (B, T), aux losses."""
        B, T = idx.shape
        c = dict(B=B, T=T, idx=idx, tgt=tgt)

        def keep(site, t):
            if retain:
                t.retain_grad()
            c[site] = t
            return t

        e = keep(("emb",), self.wte[idx] + self.wpe[:T][None])
        x = rms(e)
        for b in range(self.L):
            c["xa", b] = x
            c["in", ("qkv", b)] = u = rms(x)
            qkv = keep(("qkv", b), u @ self.Wqkv[b].T)
            o = keep(("o", b), self.attend(qkv))
            c["in", ("proj", b)] = o
            x = x + keep(("proj", b), o @ self.Wproj[b].T)
            c["xm", b] = x
            c["in", ("fc", b)] = u = rms(x)
            h = keep(("fc", b), u @ self.Wfc[b].T)
            c["in", ("out", b)] = z = self.act(h, b)
            x = x + keep(("out", b), z @ self.Wout[b].T)
            if b in self.aux_blocks:
                c["in", ("aux", b)] = rms(x)
                za = keep(("aux", b), rms(x) @ self.Waux[str(b)].T)
                c["loss_aux", b] = self.tok_loss(za, tgt)
        c["xL"] = x
        c["in", ("head",)] = rms(x)
        z = keep(("head",), rms(x) @ self.Whead.T)
        c["loss"] = self.tok_loss(z, tgt)
        return c

    # ---- resuming from a site -----------------------------------------------------------------------------------------
    def resume(self, site, val, c, stop_aux=None):
        """val: (N, T, D) = the site's output for N = Kc*B rows (draw-major). Runs the rest of the network and returns
        per-token losses (N, T): the final loss, or the auxiliary loss after block `stop_aux` if given."""
        k = site[0]
        rep = val.shape[0] // c["B"]
        R = lambda t: t.repeat(rep, 1, 1)
        if k == "head":
            return self.tok_loss(val, c["tgt"])
        if k == "aux":
            return self.tok_loss(val, c["tgt"])
        if k == "emb":
            x, b0, half = rms(val), 0, "a"
        elif k == "qkv":
            b = site[1]
            x, b0, half = R(c["xa", b]) + self.attend(val) @ self.Wproj[b].T, b, "m"
        elif k == "o":
            b = site[1]
            x, b0, half = R(c["xa", b]) + val @ self.Wproj[b].T, b, "m"
        elif k == "proj":
            b = site[1]
            x, b0, half = R(c["xa", b]) + val, b, "m"
        elif k == "fc":
            b = site[1]
            x, b0, half = R(c["xm", b]) + self.act(val, b) @ self.Wout[b].T, b, "end"
        elif k == "out":
            b = site[1]
            x, b0, half = R(c["xm", b]) + val, b, "end"
        else:
            raise ValueError(site)
        b = b0
        ah, mh = _halves(self.fast)
        while b < self.L:
            if half == "a":
                x = ah(x, self.Wqkv[b], self.Wproj[b], self.H)
                half = "m"
            if half == "m":
                x = mh(x, self.Wfc[b], self.Wout[b], self.topk_frac, self.theta[b].to(x.dtype) if self.ev else None)
                half = "end"
            if stop_aux is not None and b == stop_aux:
                return self.tok_loss(rms(x) @ self.Waux[str(b)].T, c["tgt"])
            b, half = b + 1, "a"
        return self.tok_loss(rms(x) @ self.Whead.T, c["tgt"])

    # ---- cost (multiply-adds per token, counted once per forward) -------------------------------------------------------
    def flops_tok(self):
        d, m, T = self.d, self.m, self.T
        attn_half = 3 * d * d + d * d + T * d       # qkv, proj, attention (causal: ~T/2 keys x2 for scores and values)
        mlp_half = 2 * d * m
        return dict(attn=attn_half, mlp=mlp_half, head=d * self.V, fwd=self.L * (attn_half + mlp_half) + d * self.V)

    def resume_cost(self, site, stop_aux=None):
        """Fraction of one full forward that a draw at `site` reruns."""
        f = self.flops_tok()
        k = site[0]
        if k in ("head", "aux"):
            return 0.0
        b = 0 if k == "emb" else site[1]
        cost = 0.0
        if k in ("emb", "qkv", "o", "proj"):
            cost += f["attn"] if k in ("emb", "qkv") else (f["attn"] - 3 * self.d * self.d - self.T * self.d) * (k == "o")
            cost += f["mlp"]
        else:
            cost += f["mlp"] / 2 if k == "fc" else 0.0
        last = self.L - 1 if stop_aux is None else stop_aux
        cost += (last - b) * (f["attn"] + f["mlp"])
        cost += f["head"]
        return cost / f["fwd"]
