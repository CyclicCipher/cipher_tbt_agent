"""
test_transformer_grokking.py (the real version; Gemini's original is gemini_original/test_transformer_grokking.py)
Addition mod N with real adelic structure (REAL_ADELIC.md, part 2).

Embedding (the genuine adelic coordinates of x/N, any N): the circle (cos, sin)(2 pi x / N), its image at the real
place, and for each p^k exactly dividing N the k base-p digits of its p-adic part (x * (N/p^k)^-1 mod p^k). For
N = 105 this is Gemini's embedding up to relabelling; for a prime N it is a lookup table plus the circle.

Loss (adelic metric losses on the ONE output distribution P over Z/N):
    CE(P, y) + E_P[circle distance to y] + sum over p | N of E_P[p-adic distance to y]
The local predictions are marginals of P, so they agree with the global one by construction; no consistency
("Tate") penalty is needed, and the product formula holds for the answer automatically.

Model, optimizer, split, epochs: Gemini's (one attention + one MLP sublayer with AttnRes depth routing, RoPE,
AdamW 2e-3, weight decay 0.01, dropout 0.1, half the pairs for training, 120 epochs).
    python test_transformer_grokking.py --moduli 105 81 97 --seeds 0 1 2
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import adelic as A

D_MODEL, N_HEADS, D_FF = 128, 4, 256
BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, DROPOUT, EPOCHS = 512, 2e-3, 0.01, 0.1, 120
OUT = "runs/real_grok.jsonl"
ARMS = {   # name: (embedding, use the circle loss, use the p-adic losses)
    "ce": ("plain", False, False), "arch": ("plain", True, False), "padic": ("plain", False, True),
    "adelic": ("plain", True, True), "emb_ce": ("adelic", False, False), "emb_adelic": ("adelic", True, True)}


# ---- model (Gemini's, unchanged) --------------------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x):
        cos, sin = self.cos[:x.shape[2]][None, None], self.sin[:x.shape[2]][None, None]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=DROPOUT):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d_model, d_model, bias=False) for _ in range(4))
        self.rope = RotaryEmbedding(self.head_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, S, D = x.shape
        q, k, v = (f(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2) for f in (self.q_proj, self.k_proj, self.v_proj))
        q, k = self.rope(q), self.rope(k)
        attn = self.drop(F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1))
        return self.out_proj((attn @ v).transpose(1, 2).contiguous().view(B, S, D))


class DepthAttentionResidualRouter(nn.Module):
    def __init__(self, d_model, max_layers=4):
        super().__init__()
        self.layer_queries = nn.Parameter(torch.randn(max_layers, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, history, layer):
        if len(history) == 1:
            return history[0]
        stacked = torch.stack(history, 1)
        alpha = F.softmax((self.k_proj(stacked) * self.layer_queries[layer]).sum(-1) / math.sqrt(stacked.shape[-1]), 1)
        return (alpha.unsqueeze(-1) * stacked).sum(1)


# ---- embeddings -------------------------------------------------------------------------------------------------------
class PlainEmbedding(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.emb = nn.Embedding(N + 1, D_MODEL)                          # token N is '='

    def forward(self, x):
        return self.emb(x)


class AdelicEmbedding(nn.Module):
    """One block of the model width per place (the real place first), as in Gemini's layout."""
    def __init__(self, N):
        super().__init__()
        self.N, self.places = N, A.places(N)
        d = D_MODEL // (1 + len(self.places))
        self.proj_inf = nn.Linear(2, D_MODEL - d * len(self.places))
        self.digits = nn.ModuleList([nn.ModuleList([nn.Embedding(p, d) for _ in range(k)]) for p, k, _ in self.places])

    def forward(self, x):
        is_eq = x == self.N
        x = torch.where(is_eq, torch.zeros_like(x), x)
        theta = 2 * math.pi * x.float() / self.N
        blocks = [self.proj_inf(torch.stack([theta.cos(), theta.sin()], -1))]
        for (p, k, u), embs in zip(self.places, self.digits):
            y = (x * u) % p ** k
            blocks.append(sum(e((y // p ** j) % p) for j, e in enumerate(embs)))
        h = torch.cat(blocks, -1)
        return torch.where(is_eq.unsqueeze(-1), torch.zeros_like(h), h)


class ModularTransformer(nn.Module):
    def __init__(self, N, embedding):
        super().__init__()
        self.embed = AdelicEmbedding(N) if embedding == "adelic" else PlainEmbedding(N)
        self.drop = nn.Dropout(DROPOUT)
        self.norm1, self.norm2, self.final_norm = nn.RMSNorm(D_MODEL), nn.RMSNorm(D_MODEL), nn.RMSNorm(D_MODEL)
        self.attn = RoPEMultiHeadAttention(D_MODEL, N_HEADS)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(D_FF, D_MODEL))
        self.router = DepthAttentionResidualRouter(D_MODEL)
        self.head = nn.Linear(D_MODEL, N)

    def forward(self, x):
        history = [self.drop(self.embed(x))]
        history.append(self.attn(self.norm1(self.router(history, 1))))
        history.append(self.mlp(self.norm2(self.router(history, 2))))
        return self.head(self.final_norm(self.router(history, 3))[:, 2])


# ---- the adelic metric loss -------------------------------------------------------------------------------------------
class AdelicLoss(nn.Module):
    def __init__(self, N, circle, padic):
        super().__init__()
        dists = ([A.circle_distance(N)] if circle else []) + ([A.padic_distance(N, p, k) for p, k, _ in A.places(N)]
                                                              if padic else [])
        self.register_buffer("D", torch.tensor(sum(dists), dtype=torch.float32) if dists else torch.zeros(N, N))

    def forward(self, logits, y):
        return F.cross_entropy(logits, y) + (logits.softmax(-1) * self.D[y]).sum(-1).mean()


def run(N, arm, seed, dev):
    embedding, circle, padic = ARMS[arm]
    a = torch.arange(N).repeat_interleave(N)
    b = torch.arange(N).repeat(N)
    y = (a + b) % N
    x = torch.stack([a, b, torch.full_like(a, N)], 1)
    torch.manual_seed(42)                                   # Gemini's split
    idx = torch.randperm(len(x))
    tr, va = idx[:len(x) // 2], idx[len(x) // 2:]
    torch.manual_seed(seed)
    model = ModularTransformer(N, embedding).to(dev)
    loss_fn = AdelicLoss(N, circle, padic).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(TensorDataset(x[tr], y[tr]), batch_size=BATCH_SIZE, shuffle=True)
    xv, yv = x[va].to(dev), y[va].to(dev)
    curve = []
    for _ in range(EPOCHS):
        model.train()
        correct = 0
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            logits = model(bx)
            loss = loss_fn(logits, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(-1) == by).sum().item()
        model.eval()
        with torch.no_grad():
            val = (model(xv).argmax(-1) == yv).float().mean().item()
        curve.append(dict(train=correct / len(tr), val=val))
    return curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--moduli", type=int, nargs="+", default=[105, 81, 97])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    have = {(r["N"], r["arm"], r["seed"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    dev = torch.device("cuda")
    for seed in a.seeds:
        for N in a.moduli:
            for arm in a.arms:
                if (N, arm, seed) in have:
                    continue
                t0 = time.time()
                curve = run(N, arm, seed, dev)
                vals = [c["val"] for c in curve]
                hit = next((i + 1 for i, v in enumerate(vals) if v >= 0.99), None)
                with open(OUT, "a") as f:
                    f.write(json.dumps(dict(N=N, arm=arm, seed=seed, curve=curve, seconds=time.time() - t0)) + "\n")
                print(f"N {N:3d} {arm:10s} seed {seed}: first >= 99% at epoch {hit}; val at 10/30/120 "
                      f"{[round(vals[i], 3) for i in (9, 29, 119)]}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
