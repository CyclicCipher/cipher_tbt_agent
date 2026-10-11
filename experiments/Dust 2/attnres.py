"""TinyGPT with Block Attention Residuals (Kimi Team 2026, arXiv 2603.15031; notes in ../ziplearn/refs/attention_residuals.md).

One AttnRes block per transformer block. There is no fixed residual sum: before each sublayer (attention, MLP) and
before the head, the input is a softmax mix over SOURCES:
  - b_0 = the normalised embedding, b_1 … b_n = the completed blocks' outputs (each the sum of its two sublayer outputs);
  - after a block's first sublayer, also that block's partial sum.
Mix scores s_i = w_l · RMSNorm(source_i), one pseudo-query w_l per sublayer and one for the head, zero-initialised (so
every mix starts uniform), values = the raw sources.

For Dust, each mix's scores are a site ("mix", l): a linear map of the keys by w_l, so its weight update is the outer
product of the estimated score error with the keys, like any other linear layer. It needs no derivative of anything.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from model import TinyGPT, rms


class ARGPT(TinyGPT):
    def __init__(self, V, **kw):
        super().__init__(V, **kw)
        self.n_mix = 2 * self.L + 1
        self.Wmix = nn.ParameterList([nn.Parameter(torch.zeros(1, self.d)) for _ in range(self.n_mix)])

    # ---- sites ----------------------------------------------------------------------------------------------------------
    def weight_of(self, site):
        if site[0] == "mix":
            return self.Wmix[site[1]]
        return super().weight_of(site)

    def weight_sites(self):
        return super().weight_sites() + [("mix", l) for l in range(self.n_mix)]

    def site_grad(self, s, err, c, N):
        """Weight gradient for a site from its output error (B, T, D_out) and cached input."""
        if s[0] == "mix":
            return torch.einsum("bts,btsd->d", err, c["in", s].float())[None] / N
        return None

    # ---- the mix ----------------------------------------------------------------------------------------------------------
    def mix(self, l, srcs, scores=None):
        V = torch.stack(srcs, 2)                                         # (N, T, S, d)
        Kk = rms(V)
        s = Kk @ self.Wmix[l][0].to(Kk.dtype) if scores is None else scores   # (N, T, S)
        a = torch.softmax(s.float(), -1).to(V.dtype)
        return (a[..., None] * V).sum(2), s, Kk, V

    def attn_sub(self, b, h):
        return self.attend(rms(h) @ self.Wqkv[b].T) @ self.Wproj[b].T

    def mlp_sub(self, b, h):
        return self.act(rms(h) @ self.Wfc[b].T, b) @ self.Wout[b].T

    # ---- the clean pass ----------------------------------------------------------------------------------------------------
    def forward_cache(self, idx, tgt, retain=False):
        B, T = idx.shape
        c = dict(B=B, T=T, idx=idx, tgt=tgt)

        def keep(site, t):
            if retain:
                t.retain_grad()
            c[site] = t
            return t

        e = keep(("emb",), self.wte[idx] + self.wpe[:T][None])
        blocks = [rms(e)]
        for b in range(self.L):
            c["blocks", b] = list(blocks)
            h, s, Kk, V = self.mix(2 * b, blocks)
            keep(("mix", 2 * b), s)
            c["in", ("mix", 2 * b)], c["V", 2 * b] = Kk, V
            c["in", ("qkv", b)] = u = rms(h)
            qkv = keep(("qkv", b), u @ self.Wqkv[b].T)
            o = keep(("o", b), self.attend(qkv))
            c["in", ("proj", b)] = o
            p = keep(("proj", b), o @ self.Wproj[b].T)
            c["partial", b] = p
            h, s, Kk, V = self.mix(2 * b + 1, blocks + [p])
            keep(("mix", 2 * b + 1), s)
            c["in", ("mix", 2 * b + 1)], c["V", 2 * b + 1] = Kk, V
            c["in", ("fc", b)] = u = rms(h)
            hh = keep(("fc", b), u @ self.Wfc[b].T)
            c["in", ("out", b)] = z = self.act(hh, b)
            m = keep(("out", b), z @ self.Wout[b].T)
            blocks = blocks + [p + m]
        c["blocks", self.L] = list(blocks)
        h, s, Kk, V = self.mix(2 * self.L, blocks)
        keep(("mix", 2 * self.L), s)
        c["in", ("mix", 2 * self.L)], c["V", 2 * self.L] = Kk, V
        c["xL"] = h
        c["in", ("head",)] = rms(h)
        z = keep(("head",), rms(h) @ self.Whead.T)
        c["loss"] = self.tok_loss(z, tgt)
        return c

    # ---- resuming from a site ----------------------------------------------------------------------------------------------
    def _run(self, b, blocks, p=None, after_attn=False, tgt=None):
        """Continue from block b. If after_attn, block b's attention output p is known: do its MLP sublayer next."""
        while b < self.L:
            if not after_attn:
                h, _, _, _ = self.mix(2 * b, blocks)
                p = self.attn_sub(b, h)
            h, _, _, _ = self.mix(2 * b + 1, blocks + [p])
            blocks = blocks + [p + self.mlp_sub(b, h)]
            b, after_attn = b + 1, False
        h, _, _, _ = self.mix(2 * self.L, blocks)
        return self.tok_loss(rms(h) @ self.Whead.T, tgt)

    def resume(self, site, val, c, stop_aux=None):
        k = site[0]
        rep = val.shape[0] // c["B"]
        R = lambda t: t.repeat(rep, *([1] * (t.dim() - 1)))
        Rb = lambda bl: [R(t) for t in bl]
        tgt = c["tgt"]
        if k == "head":
            return self.tok_loss(val, tgt)
        if k == "emb":
            return self._run(0, [rms(val)], tgt=tgt)
        if k == "mix":
            l = site[1]
            V = R(c["V", l])
            a = torch.softmax(val.float(), -1).to(V.dtype)
            h = (a[..., None] * V).sum(2)
            if l == 2 * self.L:
                return self.tok_loss(rms(h) @ self.Whead.T, tgt)
            b = l // 2
            blocks = Rb(c["blocks", b])
            if l % 2 == 0:                                    # attention mix of block b
                return self._run(b, blocks, self.attn_sub(b, h), after_attn=True, tgt=tgt)
            p = R(c["partial", b])
            return self._run(b + 1, blocks + [p + self.mlp_sub(b, h)], tgt=tgt)
        b = site[1]
        blocks = Rb(c["blocks", b])
        if k in ("qkv", "o", "proj"):
            if k == "qkv":
                p = self.attend(val) @ self.Wproj[b].T
            elif k == "o":
                p = val @ self.Wproj[b].T
            else:
                p = val
            return self._run(b, blocks, p, after_attn=True, tgt=tgt)
        p = R(c["partial", b])
        m = self.act(val, b) @ self.Wout[b].T if k == "fc" else val
        return self._run(b + 1, blocks + [p + m], tgt=tgt)

    def resume_cost(self, site, stop_aux=None):
        if site[0] == "mix":
            l = site[1]
            if l == 2 * self.L:
                return 0.0
            return super().resume_cost(("qkv", l // 2) if l % 2 == 0 else ("fc", l // 2))
        return super().resume_cost(site, stop_aux)
