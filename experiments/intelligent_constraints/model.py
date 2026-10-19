"""A tiny causal transformer whose MLPs are either plain key-value memories or DAGGER memories.

Plain MLP (key-value memory, Geva et al. 2021):   y = V gelu(K x + b_k)
Dagger MLP (the converse read is the transpose):  y = V gelu(K x + b_k) + K^T gelu(V^T x + b_r)

K is (m, d), V is (d, m). In the dagger MLP the second term reads the SAME parameters backwards: V's columns act as keys
and K's rows as values. A unit trained to map cue x_a to value v_b is therefore also cued by v_b and writes back x_a.
In the linear case this is Kosko's bidirectional associative memory (1988): forward read W, backward read W^T.
`rev=False` switches the transpose read off (the ablation in Experiment 1).

`frozen=True` (Experiment 1b) fixes every token's code at a random N(0, 1) vector, so facts cannot be stored in the
embedding table and must go into the weights. The answer logits are then s * LN(x) . code / sqrt(d), with a learned
scalar s; untied, the output codes are learned at the same scale.

`seg_ids` (hippocampus H1b): a fixed per-position segment id (e.g. the slot within a retrieved episode) whose learned
embedding is added to the input, so a role shared by many positions is learned once.

`rope=True` (hippocampus H1e): rotary position encoding on queries and keys instead of learned absolute positions,
so a fixed offset ("two tokens earlier") is the same operation at every position.

`transport="sum"` (Experiment 1c) removes attention: the MLP blocks read the fixed sum of the two token codes, so an
entity arrives at the MLP as exactly its own code. `set_mlp_off(True)` zeroes every MLP's output (the per-run check of
how much of the facts the MLPs hold).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class MLP(nn.Module):
    def __init__(self, d, m, dagger, n_layer):
        super().__init__()
        self.dagger = dagger
        self.rev = True
        self.off = False
        self.K = nn.Parameter(torch.randn(m, d) * 0.02)
        self.V = nn.Parameter(torch.randn(d, m) * 0.02 / math.sqrt(2 * n_layer))
        self.bk = nn.Parameter(torch.zeros(m))
        self.bv = nn.Parameter(torch.zeros(d))
        if dagger:
            self.br = nn.Parameter(torch.zeros(m))

    def forward(self, x):
        if self.off:
            return torch.zeros_like(x)
        y = Fn.gelu(x @ self.K.T + self.bk) @ self.V.T
        if self.dagger and self.rev:
            y = y + Fn.gelu(x @ self.V + self.br) @ self.K
        return y + self.bv


class Block(nn.Module):
    def __init__(self, d, n_head, m, dagger, n_layer, attn=True):
        super().__init__()
        self.attn = attn
        if attn:
            self.ln1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
            nn.init.normal_(self.qkv.weight, std=0.02)
            nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(2 * n_layer))
        self.ln2 = nn.LayerNorm(d)
        self.n_head = n_head
        self.mlp = MLP(d, m, dagger, n_layer)

    def forward(self, x, rope=None):
        if not self.attn:
            return x + self.mlp(self.ln2(x))
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (t.view(B, T, self.n_head, d // self.n_head).transpose(1, 2) for t in (q, k, v))
        if rope is not None:                                   # rotary: rotate q, k by their positions
            cos, sin = rope
            q, k = (torch.cat([t[..., ::2] * cos - t[..., 1::2] * sin, t[..., ::2] * sin + t[..., 1::2] * cos], -1)
                    for t in (q, k))
        a = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.ln2(x))


class Net(nn.Module):
    def __init__(self, vocab, d=128, n_layer=2, n_head=4, m=256, dagger=False, tied=True, ctx=2, frozen=False,
                 transport="attn", seg_ids=None, rope=False):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.transport = transport
        self.rope = rope
        if transport == "attn" and not rope:
            self.pos = nn.Embedding(ctx, d)
        self.frozen = frozen
        nn.init.normal_(self.tok.weight, std=1.0 if frozen else 0.02)
        if frozen:
            self.tok.weight.requires_grad_(False)
            self.scale = nn.Parameter(torch.ones(()))
        if transport == "attn" and not rope:
            nn.init.normal_(self.pos.weight, std=0.02)
        if rope:
            hd = d // n_head
            inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
            ang = torch.arange(ctx).float()[:, None] * inv[None, :]          # (ctx, hd/2)
            self.register_buffer("rope_cos", ang.cos(), persistent=False)
            self.register_buffer("rope_sin", ang.sin(), persistent=False)
        self.blocks = nn.ModuleList(Block(d, n_head, m, dagger, n_layer, attn=(transport == "attn"))
                                    for _ in range(n_layer))
        self.lnf = nn.LayerNorm(d)
        self.tied = tied
        if not tied:
            self.out = nn.Parameter(torch.randn(vocab, d) * (1.0 if frozen else 0.02))
        self.has_seg = seg_ids is not None
        if self.has_seg:                         # created last, so configurations without it initialise as before
            self.register_buffer("seg_ids", torch.tensor(seg_ids))
            self.seg = nn.Embedding(int(max(seg_ids)) + 1, d)
            nn.init.normal_(self.seg.weight, std=0.02)

    def set_rev(self, on):
        for b in self.blocks:
            b.mlp.rev = on

    def set_mlp_off(self, on):
        for b in self.blocks:
            b.mlp.off = on

    def forward(self, idx):
        rope = None
        if self.transport == "sum":
            x = self.tok(idx).sum(1, keepdim=True)
        elif self.rope:
            x = self.tok(idx)
            T = idx.shape[1]
            rope = (self.rope_cos[:T], self.rope_sin[:T])
        else:
            x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        if self.has_seg:
            x = x + self.seg(self.seg_ids[:idx.shape[1]])
        for b in self.blocks:
            x = b(x, rope)
        W = self.tok.weight if self.tied else self.out
        logits = self.lnf(x[:, -1]) @ W.T        # logits for the answer after the last prompt token
        return logits * self.scale / W.shape[1] ** 0.5 if self.frozen else logits
