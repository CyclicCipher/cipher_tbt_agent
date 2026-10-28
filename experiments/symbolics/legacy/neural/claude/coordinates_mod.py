"""Computed coordinates for addition mod 105 (docs/COORDINATES.md): hypothesize a commutative group, complete the table
by associativity, diagonalize the shift operators, and hand the resulting characters to the grokking transformer.
    python -m neural.claude.coordinates_mod      (from experiments/symbolics)
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from neural.claude import grokking_adelic as G

N, OUT = 105, "runs/coordinates_mod.json"


def split():
    a = torch.arange(N).repeat_interleave(N)
    b = torch.arange(N).repeat(N)
    y = (a + b) % N
    x = torch.stack([a, b, torch.full_like(a, N)], 1)
    torch.manual_seed(42)                                   # the grokking controls' split
    idx = torch.randperm(len(x))
    return x, y, idx[:len(x) // 2], idx[len(x) // 2:]


def complete(T):
    """Fill the table under the hypothesis 'commutative group': commutativity and associativity to a fixed point.
    Returns (table, contradictions, passes)."""
    T = T.copy()
    contradictions, passes = 0, 0
    while True:
        passes += 1
        before = (T >= 0).sum()
        both = (T >= 0) & (T.T >= 0)
        contradictions += int(np.sum(both & (T != T.T)) // 2)
        T = np.where((T < 0) & (T.T >= 0), T.T, T)
        for a in range(len(T)):
            B = np.nonzero(T[a] >= 0)[0]
            C = T[a, B]                                     # a + b = c
            Fm, Em = T[B], T[C]                             # b + d = f, c + d = e
            ok = (Fm >= 0) & (Em >= 0)
            f, e = Fm[ok], Em[ok]
            if len(f) == 0:
                continue
            known = T[a, f] >= 0
            contradictions += int(np.sum(known & (T[a, f] != e)))
            new = ~known
            if new.any():
                fu, first = np.unique(f[new], return_index=True)
                T[a, fu] = e[new][first]
        if (T >= 0).sum() == before:
            return T, contradictions, passes


def characters(T, rng):
    """Eigenvectors of a random combination of the shift operators, normalized at the identity element."""
    n = len(T)
    ident = int(np.nonzero([np.all(T[e] == np.arange(n)) for e in range(n)])[0][0])
    S = np.zeros((n, n))
    for a, w in enumerate(rng.normal(size=n)):
        S[T[a], np.arange(n)] += w                          # P_a sends e_b to e_{a+b}
    vals, vecs = np.linalg.eig(S.T)                         # functions chi on elements: rows of S.T act on them
    chis = vecs / vecs[ident]                               # chi(identity) = 1
    return chis.T, ident


def order(chi, tol=1e-6):
    for k in range(1, len(chi) + 1):
        if np.max(np.abs(chi ** k - 1)) < tol:
            return k
    return None


def coordinates():
    x, y, tr, va = split()
    T = -np.ones((N, N), int)
    T[x[tr, 0].numpy(), x[tr, 1].numpy()] = y[tr].numpy()
    t0 = time.time()
    full, contra, passes = complete(T)
    res = dict(seen=int((T >= 0).sum()), filled=int((full >= 0).sum()), contradictions=contra, passes=passes,
               completion_seconds=time.time() - t0)
    pred = full[x[va, 0].numpy(), x[va, 1].numpy()]
    res["completion_heldout_accuracy"] = float(np.mean(pred == y[va].numpy()))
    rng = np.random.default_rng(0)                          # control: a random operation table, same seen half
    R = rng.integers(0, N, (N, N))
    TR = -np.ones((N, N), int)
    TR[x[tr, 0].numpy(), x[tr, 1].numpy()] = R[x[tr, 0].numpy(), x[tr, 1].numpy()]
    _, contra_r, _ = complete(TR)
    res["random_table_contradictions"] = contra_r
    chis, ident = characters(full, rng)
    true = np.exp(2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N)   # true chi_k(x) = e^{2 pi i k x / N}
    dev = [float(np.min(np.max(np.abs(true - c[None, :]), axis=1))) for c in chis]
    res["max_deviation_from_true_characters"] = max(dev)
    hom = max(float(np.max(np.abs(c[full] - np.outer(c, c)))) for c in chis)
    res["max_homomorphism_error"] = hom
    orders = [order(c) for c in chis]
    res["character_orders"] = sorted(set(o for o in orders if o))
    by_order = {}
    for c, o in zip(chis, orders):
        if o:
            by_order.setdefault(o, c)
    prime_powers = [o for o in by_order if o > 1 and len({p for p in range(2, o + 1) if o % p == 0 and all(p % q for q in range(2, p))}) == 1]
    maximal = [o for o in prime_powers if not any(o2 != o and o2 % o == 0 for o2 in prime_powers)]
    chosen = sorted(maximal) + [max(by_order)]
    res["generating_orders"] = chosen
    feats = np.concatenate([np.stack([np.cos(np.angle(by_order[o])), np.sin(np.angle(by_order[o]))], 1) for o in chosen], 1)
    return res, feats


class FeatureEmbedding(nn.Module):
    def __init__(self, feats):
        super().__init__()
        self.register_buffer("F", torch.tensor(np.vstack([feats, np.zeros((1, feats.shape[1]))]), dtype=torch.float32))
        self.proj = nn.Linear(feats.shape[1], G.D_MODEL)

    def forward(self, x):
        h = self.proj(self.F[x])
        return torch.where((x == N).unsqueeze(-1), torch.zeros_like(h), h)


def train(feats, seed, dev):
    x, y, tr, va = split()
    torch.manual_seed(seed)
    model = G.ModularTransformer(N, "plain").to(dev)
    model.embed = FeatureEmbedding(feats).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=G.LEARNING_RATE, weight_decay=G.WEIGHT_DECAY)
    loader = DataLoader(TensorDataset(x[tr], y[tr]), batch_size=G.BATCH_SIZE, shuffle=True)
    xv, yv = x[va].to(dev), y[va].to(dev)
    curve = []
    for _ in range(G.EPOCHS):
        model.train()
        for bx, by in loader:
            loss = F.cross_entropy(model(bx.to(dev)), by.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            curve.append((model(xv).argmax(-1) == yv).float().mean().item())
    return curve


def main():
    res, feats = coordinates()
    print(json.dumps(res, indent=1), flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res["runs"] = {}
    for seed in (0, 1, 2):
        t0 = time.time()
        curve = train(feats, seed, dev)
        hit = next((i + 1 for i, v in enumerate(curve) if v >= 0.99), None)
        res["runs"][seed] = dict(first99=hit, final=curve[-1], seconds=time.time() - t0, curve=curve)
        print(f"computed coordinates, seed {seed}: first >= 99% at epoch {hit}, final {curve[-1]:.4f}, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"))


if __name__ == "__main__":
    main()


# ---- exploratory (after K5 failed): the same computed coordinates, as lookup tables -------------------------------
class TableEmbedding(nn.Module):
    """Computed residue classes (read off each prime-power character's value, r = angle * p / 2 pi) as learned lookup
    tables, plus the computed largest-order character as a (cos, sin) circle: the layout of the residue embedding,
    with every coordinate computed from the table instead of given."""
    def __init__(self, classes, orders, circle):
        super().__init__()
        d = G.D_MODEL // (1 + len(orders))
        self.register_buffer("cls", torch.tensor(np.vstack([classes, np.zeros((1, classes.shape[1]), int)]), dtype=torch.long))
        self.register_buffer("circ", torch.tensor(np.vstack([circle, np.zeros((1, 2))]), dtype=torch.float32))
        self.tables = nn.ModuleList([nn.Embedding(p, d) for p in orders])
        self.proj = nn.Linear(2, G.D_MODEL - d * len(orders))

    def forward(self, x):
        h = torch.cat([self.proj(self.circ[x])] + [t(self.cls[x][..., k]) for k, t in enumerate(self.tables)], -1)
        return torch.where((x == N).unsqueeze(-1), torch.zeros_like(h), h)


def exploratory():
    x, y, tr, va = split()
    T = -np.ones((N, N), int)
    T[x[tr, 0].numpy(), x[tr, 1].numpy()] = y[tr].numpy()
    full, _, _ = complete(T)
    chis, _ = characters(full, np.random.default_rng(0))
    by_order = {}
    for c in chis:
        o = order(c)
        if o:
            by_order.setdefault(o, c)
    orders = [3, 5, 7]                                       # the computed generating orders (K4), largest excluded
    assert all(o in by_order for o in orders)
    classes = np.stack([np.round(np.angle(by_order[p]) * p / (2 * np.pi)).astype(int) % p for p in orders], 1)
    circle = np.stack([np.cos(np.angle(by_order[max(by_order)])), np.sin(np.angle(by_order[max(by_order)]))], 1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for seed in (0, 1, 2):
        t0 = time.time()
        _, yy, tr_, va_ = split()
        torch.manual_seed(seed)
        model = G.ModularTransformer(N, "plain").to(dev)
        model.embed = TableEmbedding(classes, orders, circle).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=G.LEARNING_RATE, weight_decay=G.WEIGHT_DECAY)
        loader = DataLoader(TensorDataset(x[tr_], yy[tr_]), batch_size=G.BATCH_SIZE, shuffle=True)
        xv, yv = x[va_].to(dev), yy[va_].to(dev)
        curve = []
        for _ in range(G.EPOCHS):
            model.train()
            for bx, by in loader:
                loss = F.cross_entropy(model(bx.to(dev)), by.to(dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                curve.append((model(xv).argmax(-1) == yv).float().mean().item())
        hit = next((i + 1 for i, v in enumerate(curve) if v >= 0.99), None)
        out[seed] = dict(first99=hit, final=curve[-1], curve=curve)
        print(f"computed coordinates as tables, seed {seed}: first >= 99% at epoch {hit}, final {curve[-1]:.4f}, {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open("runs/coordinates_mod_tables.json", "w"))


if __name__ == "__main__" and os.environ.get("COORD_EXPLORATORY"):
    exploratory()
