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
        self.K = nn.Parameter(torch.randn(m, d) * 0.02)
        self.V = nn.Parameter(torch.randn(d, m) * 0.02 / math.sqrt(2 * n_layer))
        self.bk = nn.Parameter(torch.zeros(m))
        self.bv = nn.Parameter(torch.zeros(d))
        if dagger:
            self.br = nn.Parameter(torch.zeros(m))

    def forward(self, x):
        y = Fn.gelu(x @ self.K.T + self.bk) @ self.V.T
        if self.dagger and self.rev:
            y = y + Fn.gelu(x @ self.V + self.br) @ self.K
        return y + self.bv


class Block(nn.Module):
    def __init__(self, d, n_head, m, dagger, n_layer):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(2 * n_layer))
        self.n_head = n_head
        self.mlp = MLP(d, m, dagger, n_layer)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (t.view(B, T, self.n_head, d // self.n_head).transpose(1, 2) for t in (q, k, v))
        a = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.ln2(x))


class Net(nn.Module):
    def __init__(self, vocab, d=128, n_layer=2, n_head=4, m=256, dagger=False, tied=True, ctx=2, frozen=False):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.frozen = frozen
        nn.init.normal_(self.tok.weight, std=1.0 if frozen else 0.02)
        if frozen:
            self.tok.weight.requires_grad_(False)
            self.scale = nn.Parameter(torch.ones(()))
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(d, n_head, m, dagger, n_layer) for _ in range(n_layer))
        self.lnf = nn.LayerNorm(d)
        self.tied = tied
        if not tied:
            self.out = nn.Parameter(torch.randn(vocab, d) * (1.0 if frozen else 0.02))

    def set_rev(self, on):
        for b in self.blocks:
            b.mlp.rev = on

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        W = self.tok.weight if self.tied else self.out
        logits = self.lnf(x[:, -1]) @ W.T        # logits for the answer after the last prompt token
        return logits * self.scale / W.shape[1] ** 0.5 if self.frozen else logits
