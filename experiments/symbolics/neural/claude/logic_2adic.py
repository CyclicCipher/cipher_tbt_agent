"""
logic_2adic.py: the real version of Gemini's neural/gemini/test_transformer_discrete_logic.py
Boolean expression trees with a real 2-adic attention bias (docs/REAL_ADELIC.md, part 3).

A node's address (its branch sequence from the root) names a 2-adic ball: the 2-adic integers whose first digits are
that sequence. The nodes below it are the smaller balls inside it. Two 2-adic structures live on this tree of balls:
- the path length between balls (the Bruhat-Tits tree metric of Q_2): Gemini's bias, mislabelled "ultrametric";
- the ultrametric on the tree's ends, whose trace on nodes is the 2-adic valuation v = depth(LCA(i, j)), the
  Gromov product from the root. The real arm's bias is its log form, bias(i, j) = |beta| * depth(LCA(i, j)), so a
  node attends more to its own subtree (a node and all its descendants share v = its depth).
The node-loss weights 2^-depth are the Haar measure of each node's ball, so they stay. Padding is masked here
(Gemini's version gave padding the strongest bias by accident).

Arms: bias in {real, geo (Gemini's path-length bias), none} x supervision in {nodes (every subtree's value),
root (the answer only)}. Model, data and training as Gemini's.
    python -m neural.claude.logic_2adic --arms real_root none_root geo_root real_nodes      (from experiments/symbolics)
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

NUM_LAYERS, D_MODEL, N_HEADS, D_FF = 5, 128, 4, 512
MAX_TREE_DEPTH, NUM_SAMPLES, MAX_SEQ_LEN = 6, 8000, 25
BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, DROPOUT, EPOCHS = 256, 1.5e-3, 0.01, 0.1, 120
BINARY_OPS, UNARY_OPS, VALUES = ["AND", "OR", "XOR"], ["NOT"], ["0", "1"]
TOKEN_TO_ID = {"<PAD>": 0, "0": 1, "1": 2, "AND": 3, "OR": 4, "XOR": 5, "NOT": 6, "=": 7}
OUT = "runs/real_logic.jsonl"


# ---- data (Gemini's generator and parser, unchanged) ------------------------------------------------------------------
def generate_tree(depth, max_depth=5):
    if depth >= max_depth or (depth > 1 and random.random() < 0.18):
        return [random.choice(VALUES)]
    if random.random() < 0.2:
        return ["NOT"] + generate_tree(depth + 1, max_depth)
    return [random.choice(BINARY_OPS)] + generate_tree(depth + 1, max_depth) + generate_tree(depth + 1, max_depth)


def parse(tokens):
    """Each node's value and its branch sequence from the root."""
    stack, vals = [], [0] * len(tokens)
    for i in reversed(range(len(tokens))):
        t = tokens[i]
        if t in VALUES:
            vals[i] = int(t)
        elif t in UNARY_OPS:
            vals[i] = 1 - stack.pop()
        else:
            v1, v2 = stack.pop(), stack.pop()
            vals[i] = {"AND": v1 & v2, "OR": v1 | v2, "XOR": v1 ^ v2}[t]
        stack.append(vals[i])
    paths = [()] * len(tokens)

    def size(start):
        need, cur = 1, start
        while need > 0:
            need += 1 if tokens[cur] in BINARY_OPS else (-1 if tokens[cur] in VALUES else 0)
            cur += 1
        return cur - start

    def assign(i, prefix):
        paths[i] = prefix
        if tokens[i] in UNARY_OPS:
            assign(i + 1, prefix + (0,))
        elif tokens[i] in BINARY_OPS:
            assign(i + 1, prefix + (0,))
            assign(i + 1 + size(i + 1), prefix + (1,))

    assign(0, ())
    return paths, vals


def lca_depth(p, q):
    n = 0
    for a, b in zip(p, q):
        if a != b:
            break
        n += 1
    return n


def build_dataset():
    random.seed(42)
    rows, seen = [], set()
    while len(rows) < NUM_SAMPLES:
        expr = generate_tree(1, MAX_TREE_DEPTH - 1)
        key = " ".join(expr)
        if key in seen or len(expr) + 1 > MAX_SEQ_LEN:
            continue
        seen.add(key)
        paths, vals = parse(expr)
        paths = paths + [()]                                              # '=' sits at the root's address
        n = len(paths)
        geo = torch.zeros(MAX_SEQ_LEN, MAX_SEQ_LEN)
        val2 = torch.zeros(MAX_SEQ_LEN, MAX_SEQ_LEN)
        for i in range(n):
            for j in range(n):
                v = lca_depth(paths[i], paths[j])
                val2[i, j] = v
                if i != j:
                    geo[i, j] = 1.0 - 2.0 ** -(len(paths[i]) + len(paths[j]) - 2 * v)
        bp = torch.full((MAX_SEQ_LEN, MAX_TREE_DEPTH), -1, dtype=torch.long)
        for i, p in enumerate(paths):
            for d, bit in enumerate(p[:MAX_TREE_DEPTH]):
                bp[i, d] = bit
        pad = MAX_SEQ_LEN - n
        rows.append(([TOKEN_TO_ID[t] for t in expr + ["="]] + [0] * pad, vals + [vals[0]] + [0] * pad,
                     [len(p) for p in paths] + [0] * pad, [1.0] * n + [0.0] * pad, bp, geo, val2, n - 1, vals[0]))
    cols = list(zip(*rows))
    out = [torch.tensor(cols[0]), torch.tensor(cols[1]), torch.tensor(cols[2]), torch.tensor(cols[3]),
           torch.stack(cols[4]), torch.stack(cols[5]), torch.stack(cols[6]), torch.tensor(cols[7]), torch.tensor(cols[8])]
    split = int(0.75 * NUM_SAMPLES)
    return [t[:split] for t in out], [t[split:] for t in out]


# ---- model (Gemini's, with the bias as an argument and padding masked) ------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=32):
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


class BiasedAttention(nn.Module):
    def __init__(self, sign):
        super().__init__()
        self.head_dim = D_MODEL // N_HEADS
        self.q, self.k, self.v, self.out = (nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(4))
        self.rope = RotaryEmbedding(self.head_dim)
        self.beta = nn.Parameter(torch.tensor(1.2))
        self.sign = sign                                                  # +1: valuation (closer = larger); -1: distance
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, bias, valid):
        B, S, D = x.shape
        q, k, v = (f(x).view(B, S, N_HEADS, self.head_dim).transpose(1, 2) for f in (self.q, self.k, self.v))
        q, k = self.rope(q), self.rope(k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) + self.sign * self.beta.abs() * bias.unsqueeze(1)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))
        return self.out((attn @ v).transpose(1, 2).contiguous().view(B, S, D))


class DepthAttentionResidualRouter(nn.Module):
    def __init__(self, max_layers):
        super().__init__()
        self.layer_queries = nn.Parameter(torch.randn(max_layers, D_MODEL) * 0.02)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, history, layer):
        if len(history) == 1:
            return history[0]
        stacked = torch.stack(history, 1)
        alpha = F.softmax((self.k_proj(stacked) * self.layer_queries[layer]).sum(-1) / math.sqrt(D_MODEL), 1)
        return (alpha.unsqueeze(-1) * stacked).sum(1)


class LogicTransformer(nn.Module):
    def __init__(self, sign):
        super().__init__()
        self.embed = nn.Embedding(len(TOKEN_TO_ID), D_MODEL)
        self.branch_embed = nn.Parameter(torch.randn(MAX_TREE_DEPTH, 2, D_MODEL) * 0.02)   # the address's 2-adic digits
        self.drop = nn.Dropout(DROPOUT)
        self.attn = nn.ModuleList([BiasedAttention(sign) for _ in range(NUM_LAYERS)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Dropout(DROPOUT),
                                                nn.Linear(D_FF, D_MODEL)) for _ in range(NUM_LAYERS)])
        self.norms = nn.ModuleList([nn.RMSNorm(D_MODEL) for _ in range(2 * NUM_LAYERS + 1)])
        self.router = DepthAttentionResidualRouter(2 * NUM_LAYERS + 2)
        self.unembed = nn.Linear(D_MODEL, 2)

    def forward(self, x, bp, bias, valid, use_paths):
        h = self.embed(x)
        if use_paths:
            h = h + ((bp == 0).float().unsqueeze(-1) * self.branch_embed[:, 0] +
                     (bp == 1).float().unsqueeze(-1) * self.branch_embed[:, 1]).sum(2)
        history, layer = [self.drop(h)], 1
        for i in range(NUM_LAYERS):
            history.append(self.attn[i](self.norms[2 * i](self.router(history, layer)), bias, valid))
            history.append(self.mlp[i](self.norms[2 * i + 1](self.router(history, layer + 1))))
            layer += 2
        return self.unembed(self.norms[-1](self.router(history, layer)))


def run(arm, seed, train, val, dev):
    bias_kind, supervision = arm.split("_")
    torch.manual_seed(seed)
    model = LogicTransformer(sign=1.0 if bias_kind == "real" else -1.0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def batch(t):
        x, y, d, m, bp, geo, val2, pos, root = [u.to(dev) for u in t]
        bias = {"real": val2, "geo": geo, "none": torch.zeros_like(geo)}[bias_kind]
        valid = m > 0
        if supervision == "nodes":
            w = torch.pow(2.0, -d.float()) * m                            # each node's ball's Haar measure
        else:
            w = torch.zeros_like(m)
            w[torch.arange(len(pos)), pos] = 1.0
        return x, y, bp, bias, valid, w, pos, root

    use_paths = bias_kind != "none"
    loader = DataLoader(TensorDataset(*train), batch_size=BATCH_SIZE, shuffle=True)
    vb = batch(val)
    curve = []
    for _ in range(EPOCHS):
        model.train()
        correct, n = 0, 0
        for t in loader:
            x, y, bp, bias, valid, w, pos, root = batch(t)
            logits = model(x, bp, bias, valid, use_paths)
            ce = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1), reduction="none").view_as(w)
            loss = (ce * w).sum() / (w.sum() + 1e-8)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits[torch.arange(len(pos)), pos].argmax(-1) == root).sum().item()
            n += len(pos)
        model.eval()
        with torch.no_grad():
            x, y, bp, bias, valid, w, pos, root = vb
            hits = []
            for i in range(0, len(x), 512):
                sl = slice(i, i + 512)
                lg = model(x[sl], bp[sl], bias[sl], valid[sl], use_paths)
                hits.append(lg[torch.arange(len(pos[sl])), pos[sl]].argmax(-1) == root[sl])
        curve.append(dict(train=correct / n, val=torch.cat(hits).float().mean().item()))
    return curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["real_root", "none_root", "geo_root", "real_nodes"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = p.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    have = {(r["arm"], r["seed"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    train, val = build_dataset()
    print(f"{len(train[0])} train, {len(val[0])} val; majority class {max(val[8].float().mean(), 1 - val[8].float().mean()):.3f}",
          flush=True)
    dev = torch.device("cuda")
    for seed in a.seeds:
        for arm in a.arms:
            if (arm, seed) in have:
                continue
            t0 = time.time()
            curve = run(arm, seed, train, val, dev)
            vals = [c["val"] for c in curve]
            hit = next((i + 1 for i, v in enumerate(vals) if v >= 0.95), None)
            with open(OUT, "a") as f:
                f.write(json.dumps(dict(arm=arm, seed=seed, curve=curve, seconds=time.time() - t0)) + "\n")
            print(f"{arm:10s} seed {seed}: val at 10/30/120 {[round(vals[i], 3) for i in (9, 29, 119)]}  first >= 95%: "
                  f"epoch {hit}  train end {curve[-1]['train']:.3f}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
