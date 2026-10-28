"""Controls for the two transformer experiments (docs/CONTROLS.md): which component carries each result?

grok:  neural/gemini/test_transformer_grokking.py (addition mod 105, 50% of pairs for training). Arms:
       as_written  - the adelic embedding (circle + residues mod 3, 5, 7) and the residue heads + Tate term
       adelic_ce   - the adelic embedding, cross-entropy on the answer only
       plain_ce    - a learned token embedding, cross-entropy on the answer only (the standard grokking setup)
       plain_heads - a learned token embedding, with the residue heads (the loss without the embedding)
logic: neural/gemini/test_transformer_discrete_logic.py (Boolean expression trees). Arms:
       as_written      - tree-path embedding + tree-distance attention bias, every node's value supervised
       no_structure    - neither the path embedding nor the distance bias, every node supervised
       root_only       - the tree structure, only the answer supervised
       plain_root_only - neither, only the answer supervised (the plain baseline)

Everything else (model, optimizer, epochs, data, split) is the original's. Full curves are logged, no early stop.
    python -m neural.claude.controls_transformers grok --seeds 0 1 2      (from experiments/symbolics)
    python -m neural.claude.controls_transformers logic --seeds 0
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from neural.gemini import test_transformer_discrete_logic as L
from neural.gemini import test_transformer_grokking as G

OUT = "runs"


def done(path):
    if not os.path.exists(path):
        return set()
    return {(r["arm"], r["seed"]) for r in map(json.loads, open(path))}


# ---- grok -------------------------------------------------------------------------------------------------------------
class PlainEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(G.MODULUS + 1, G.D_MODEL)

    def forward(self, x):
        return self.emb(x)


def grok_run(arm, seed, dev):
    a = torch.arange(G.MODULUS).repeat_interleave(G.MODULUS)
    b = torch.arange(G.MODULUS).repeat(G.MODULUS)
    y = (a + b) % G.MODULUS
    x = torch.stack([a, b, torch.full_like(a, G.MODULUS)], 1)
    torch.manual_seed(42)                                   # the original's split
    idx = torch.randperm(len(x))
    split = len(x) // 2
    tr, va = idx[:split], idx[split:]
    torch.manual_seed(seed)
    model = G.AdelicModularTransformer().to(dev)
    if arm.startswith("plain"):
        model.adelic_embed = PlainEmbedding().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=G.LEARNING_RATE, weight_decay=G.WEIGHT_DECAY)
    hasse = G.AdelicHasseLoss()

    def loss_fn(p, t):
        if arm in ("as_written", "plain_heads"):
            return hasse(p, t)                              # the Tate term has no gradient (argmax); kept as written
        return F.cross_entropy(p["infty"], t)

    loader = DataLoader(TensorDataset(x[tr], y[tr]), batch_size=G.BATCH_SIZE, shuffle=True)
    xv, yv = x[va].to(dev), y[va].to(dev)
    curve = []
    for epoch in range(G.EPOCHS):
        model.train()
        correct = 0
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            p = model(bx)
            loss = loss_fn(p, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (p["infty"].argmax(-1) == by).sum().item()
        model.eval()
        with torch.no_grad():
            pv = model(xv)
            val = (pv["infty"].argmax(-1) == yv).float().mean().item()
            local = {q: (pv[f"p{q}"].argmax(-1) == yv % q).float().mean().item() for q in (3, 5, 7)}
        curve.append(dict(train=correct / len(tr), val=val, **{f"val_mod{q}": v for q, v in local.items()}))
    return curve


# ---- logic ------------------------------------------------------------------------------------------------------------
def logic_tensors():
    data = L.build_dataset(num_samples=L.NUM_SAMPLES, max_seq_len=25)
    split = int(0.75 * len(data))

    def stack(batch):
        return (torch.tensor([s[0] for s in batch]), torch.tensor([s[1] for s in batch]),
                torch.tensor([s[2] for s in batch]), torch.tensor([s[3] for s in batch], dtype=torch.float32),
                torch.stack([s[4] for s in batch]), torch.stack([s[5] for s in batch]),
                torch.tensor([s[6] for s in batch]), torch.tensor([s[7] for s in batch]))

    return stack(data[:split]), stack(data[split:])


def logic_run(arm, seed, dev, train, val):
    structure = arm in ("as_written", "root_only")
    every_node = arm in ("as_written", "no_structure")
    torch.manual_seed(seed)
    model = L.AdelicLogicTransformer(vocab_size=len(L.TOKEN_TO_ID)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=L.LEARNING_RATE, weight_decay=L.WEIGHT_DECAY)
    crit = L.AdelicFiltrationLoss()

    def prep(x, y, d, m, bp, d2, pos, root):
        if not structure:
            bp, d2 = torch.full_like(bp, -1), torch.zeros_like(d2)
        if not every_node:
            m = torch.zeros_like(m)
            m[torch.arange(len(pos)), pos] = 1.0
        return x, y, d, m, bp, d2, pos, root

    loader = DataLoader(TensorDataset(*train), batch_size=L.BATCH_SIZE, shuffle=True)
    vx = [t.to(dev) for t in prep(*val)]
    curve = []
    for epoch in range(L.EPOCHS):
        model.train()
        correct, n = 0, 0
        for batch in loader:
            x, y, d, m, bp, d2, pos, root = [t.to(dev) for t in prep(*batch)]
            logits = model(x, bp, d2)
            loss = crit(logits, y, d, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits[torch.arange(len(pos)), pos].argmax(-1) == root).sum().item()
            n += len(pos)
        model.eval()
        with torch.no_grad():
            x, y, d, m, bp, d2, pos, root = vx
            acc = []
            for i in range(0, len(x), 512):
                lg = model(x[i:i + 512], bp[i:i + 512], d2[i:i + 512])
                acc.append(lg[torch.arange(len(pos[i:i + 512])), pos[i:i + 512]].argmax(-1) == root[i:i + 512])
            curve.append(dict(train=correct / n, val=torch.cat(acc).float().mean().item()))
    return curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=("grok", "logic"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--arms", nargs="+", default=None)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda")
    path = f"{OUT}/controls_{a.task}.jsonl"
    arms = a.arms or (["as_written", "adelic_ce", "plain_ce", "plain_heads"] if a.task == "grok"
                      else ["as_written", "no_structure", "root_only", "plain_root_only"])
    have = done(path)
    if a.task == "logic":
        train, val = logic_tensors()
        majority = max(val[7].float().mean().item(), 1 - val[7].float().mean().item())
        print(f"logic: {len(train[0])} train, {len(val[0])} val; majority-class val accuracy {majority:.3f}", flush=True)
    for seed in a.seeds:
        for arm in arms:
            if (arm, seed) in have:
                continue
            t0 = time.time()
            curve = grok_run(arm, seed, dev) if a.task == "grok" else logic_run(arm, seed, dev, train, val)
            vals = [c["val"] for c in curve]
            hit = next((i + 1 for i, v in enumerate(vals) if v >= (0.99 if a.task == "grok" else 0.95)), None)
            rec = dict(arm=arm, seed=seed, curve=curve, seconds=time.time() - t0)
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{a.task} {arm:16s} seed {seed}: val at epochs 1/5/10/30/120 "
                  f"{[round(vals[i], 3) for i in (0, 4, 9, 29, len(vals) - 1)]}  first >= threshold: epoch {hit}  "
                  f"train end {curve[-1]['train']:.3f}  {rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
