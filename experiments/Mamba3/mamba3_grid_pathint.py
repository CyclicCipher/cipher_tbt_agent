"""Does the REAL Mamba3 block path-integrate? — grid ability, tested in isolation.

The realignment check: wm_discover used a bare inline rotation, not the actual Mamba3 block. Here we
test the genuine `Mamba3Block` on the one thing the grid substrate must do — **path-integrate a stream
of self-motions and track position** (modular cumulative sum). This is exactly Mamba3's headline
state-tracking claim (complex SSM via data-dependent RoPE), so stock Mamba3 should already do it. We
test stock first and only reach for a grid modification (unit-magnitude decay) if stock forgets.

Setup (analogous to grid_ssm.navigation_acc, now through the real block): a 1-D torus of size m; each
action is a step in {−2,−1,0,+1,+2}; feed the action sequence as tokens; read the running position out
of the per-step Mamba3 state. Train on random sequences, test on FRESH sequences (and longer ones) —
zero-shot composition = genuine integration, not memorized transitions.

FINDING: stock Mamba3 path-integrates natively (≈0.99 at training length, generalizes to fresh
sequences). A GridSSM-style "unit-magnitude / pure-rotation" modification (zeroing the decay) made it
*worse* at every length — Mamba3's state is a decaying accumulation of rotated inputs, not a single
rotating vector, so the decay helps (bounds the state, recency-weighting). So we use the STOCK block as
the grid/path-integration substrate; no rotation-transition modification is warranted.

Tiny + GPU; seqlen must be a multiple of chunk_size (SSD constraint).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba3_block import Mamba3Block, Mamba3Config  # noqa: E402

STEPS = torch.tensor([-2, -1, 0, 1, 2])
N_ACTIONS = len(STEPS)


def action_seqs(n, L, m, seed=0):
    """Random action sequences and their true running positions on a torus of size m (start at 0)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, N_ACTIONS, (n, L), generator=g)
    steps = STEPS[a]
    pos = (torch.cumsum(steps, dim=1)) % m                       # (n, L) running position
    return a, pos


class Mamba3PathInt(nn.Module):
    """Action tokens → Mamba3 → per-step position readout. The SSM state is the (grid) position code."""

    def __init__(self, m, cfg: Mamba3Config):
        super().__init__()
        self.m = m
        self.emb = nn.Embedding(N_ACTIONS, cfg.d_model)
        self.blocks = nn.ModuleList([Mamba3Block(cfg) for _ in range(cfg.n_layer)])
        self.head = nn.Linear(cfg.d_model, m)

    def forward(self, actions):
        x = self.emb(actions)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)                                       # (B, L, m)


def train(m=12, d_model=64, n_layer=2, L=16, n=4096, iters=600, lr=2e-3, bs=256,
          seed=0, device=None, verbose=True):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Mamba3Config(d_model=d_model, d_state=16, headdim=16, chunk_size=4,
                       n_layer=n_layer, mimo_rank=1, use_pope=True)
    model = Mamba3PathInt(m, cfg).to(dev)
    a, pos = action_seqs(n, L, m, seed=seed)
    a, pos = a.to(dev), pos.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)
    for it in range(iters):
        idx = torch.randint(0, n, (bs,), generator=gen)
        logits = model(a[idx])
        loss = F.cross_entropy(logits.reshape(-1, m), pos[idx].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if verbose and (it % 100 == 0 or it == iters - 1):
            acc = (logits.argmax(-1) == pos[idx]).float().mean().item()
            print(f"  it {it:4d}  loss {loss.item():6.4f}  train-acc {acc:5.3f}")
    return model


@torch.no_grad()
def evaluate(model, m, L, n=2000, seed=999):
    dev = next(model.parameters()).device
    a, pos = action_seqs(n, L, m, seed=seed)
    a, pos = a.to(dev), pos.to(dev)
    logits = model(a)
    pred = logits.argmax(-1)
    per_step = (pred == pos).float().mean().item()                # accuracy over all positions
    final = (pred[:, -1] == pos[:, -1]).float().mean().item()     # final-position accuracy
    return {"all_steps_acc": per_step, "final_acc": final, "chance": 1.0 / m}


if __name__ == "__main__":
    m = 12
    print(f"=== Mamba3 path-integration on Z/{m} (stock, trained L=16) ===")
    model = train(m=m, L=16)
    print("  fresh L=16:", evaluate(model, m, 16))
    print("  fresh L=24 (longer, generalization):", evaluate(model, m, 24))
