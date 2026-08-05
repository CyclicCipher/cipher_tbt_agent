"""THE LEARNED CANONICALISER: recover the object from behaviour, without being told the group.

WHY THIS IS NOW THE MAIN LINE. `canon2prog.py` retracted the target-based reading of this whole arc: given the CANONICAL
state, both the distance (0.714) and the PROGRAM (0.733) are learnable and generalise, while from behavioural samples the
program is 0.000 everywhere. So the one hard step is **recovering the canonical form from behaviour** — which output
position corresponds to which input position, under an unknown value map. A matching/binding problem.

`localise.py` and `canonical.py` solved it by HAND: brute-force the 10 affine value maps, match columns, verify. Exact
(87/87) and cheap, but it is domain knowledge — it uses the fact that the group factors as `S6 x Aff(Z5)`. This file asks
whether the same thing can be LEARNED with no such knowledge.

THE METHOD, and the reason it needs no labels. A canonical form is a COMPLETE INVARIANT: same object -> same code,
different objects -> different codes. That is exactly a contrastive objective, and **the positives are free because we can
execute**: sample a transformation, draw two INDEPENDENT demonstration sets from it, and those two views are a positive
pair. Negatives are views of other transformations in the batch. Nothing about permutations, value maps or group structure
appears anywhere — only the ability to run the forward model, which is the component that always works.

THE ONE INDUCTIVE BIAS, stated rather than smuggled: the encoder MEAN-POOLS over demonstrations, so it is permutation-
invariant in them. That is a fact about the data (demonstrations are exchangeable), not about the group — the analogue of
using a convolution on images. It is asserted below, not assumed.

WHAT IS MEASURED, against known ceilings and floors:
  1. RETRIEVAL   — a fresh view of a held-out transformation, matched against a bank of codes. Direct test of "does the
                   code identify the object". Chance is 1/|bank|.
  2. CANONICAL PROBE — can `(idx, vmp)` be decoded from the frozen code? If yes, the encoder has REDISCOVERED the
                   canonical form without being told it exists. This is the sharpest result available.
  3. DISTANCE PROBE  — frozen code -> distance class, directly comparable to:
                       exact canonical state  0.714   (`canonical.py`, the ceiling)
                       raw behaviour          0.276   (`costtogo.py`, the floor)

CONTROL: the identical probes on a RANDOM-INIT encoder. Any embedding supports some probing, so the contrastive training
has to beat its own untrained initialisation or it contributed nothing.

Held-out transformations never appear in contrastive training NOR in probe training.

Usage:  python experiments/transformers/canonicaliser.py
"""
from __future__ import annotations

import argparse
import collections
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigroup import NAMES, cayley, compile_progs, stratify
from h1_lid import L, V

MAXM = 5


class Block(nn.Module):
    """BIDIRECTIONAL attention — this is an encoder, so a position must be able to see the whole demonstration. (The
    causal block in `h1_lid.py` would let early positions see almost nothing, which is the opposite of what a matching
    problem needs.)"""

    def __init__(self, d, n_head):
        super().__init__()
        self.h, self.hd = n_head, d // n_head
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = (z.view(B, T, self.h, self.hd).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v)                    # no causal mask
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.n2(x))


class Encoder(nn.Module):
    """(B, K demos, 2L tokens) -> a unit-norm code. Two pooling schemes, and the difference between them is a hypothesis
    about what the matching problem needs:

      mean  — a transformer over each demonstration INDEPENDENTLY, then mean-pool across demonstrations.
      set   — the same per-demonstration transformer, then ATTENTION ACROSS the demonstration embeddings, then pool.
      cross — all K demonstrations concatenated into ONE sequence with the positional code repeated per block.

    Recovering the permutation means finding, for each output position, the input position that matches it CONSISTENTLY
    ACROSS EVERY demonstration — an INTERSECTION of per-demonstration constraints. Mean-pooling can only AVERAGE those
    constraints, and averaging is not intersecting; the hand-written `recover` in `canonical.py` does an explicit `all()`
    across demonstrations. `set` is the fix: per-demo encoding preserves the within-demo pairing of x[j] with y[j], and
    the second stage lets demonstrations interact so a consistency constraint can be computed rather than averaged.

    ⚠ `cross` IS BROKEN AND KEPT ONLY AS A RECORD. Giving every demonstration block the same positional code makes the
    demo index invisible to attention — which was the point, for permutation invariance — but it also makes token j of
    demo 1 indistinguishable from token j of demo 5, so the model cannot pair x[j] with y[j] WITHIN a demonstration
    either. It destroys exactly the structure the task needs, and it collapses: cos same = cos diff = 1.000 with InfoNCE
    stuck at ln(batch) under two learning rates. A fix that removes the signal is not a test of the hypothesis.

    ALL THREE stay permutation-invariant over demonstrations (no positional code on the demo axis); asserted in main()."""

    def __init__(self, d=128, n_layer=3, n_head=4, dim=64, pool="mean"):
        super().__init__()
        self.pool = pool
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, 2 * L, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.across = nn.ModuleList([Block(d, n_head) for _ in range(2)]) if pool == "set" else None
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, dim))

    def forward(self, xy):
        B, K, T = xy.shape
        if self.pool == "cross":
            h = self.emb(xy.reshape(B, K * T)) + self.pos.repeat(1, K, 1)
            for b in self.blocks:
                h = b(h)
            return F.normalize(self.head(self.norm(h).mean(1)), dim=-1)
        h = self.emb(xy.reshape(B * K, T)) + self.pos
        for b in self.blocks:
            h = b(h)
        h = self.norm(h).mean(1).reshape(B, K, -1)                      # one embedding per demonstration
        if self.pool == "set":
            for b in self.across:                                       # demonstrations attend to EACH OTHER
                h = b(h)
        return F.normalize(self.head(h.mean(1)), dim=-1)


def views(sel, n_demos, tabs, dev, g):
    """A demonstration set for each selected transformation: inputs drawn fresh, outputs computed by the forward model."""
    idx_t, vmp_t = tabs
    B = sel.shape[0]
    x = torch.randint(0, V, (B, n_demos, L), generator=g, device=dev)
    gi = idx_t[sel][:, None, :].expand(B, n_demos, L)
    y = vmp_t[sel][:, None, :].expand(B, n_demos, V).gather(2, x.gather(2, gi))
    return torch.cat([x, y], dim=2)                                    # (B, K, 2L)


def info_nce(z1, z2, temp):
    logits = z1 @ z2.T / temp
    lab = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))


@torch.no_grad()
def codes_for(enc, ws, n_demos, dev, seed, reps=4):
    """A stable code per transformation: the mean over `reps` independent demonstration sets, re-normalised. If the
    encoder is a good canonicaliser this averaging changes almost nothing, which the invariance number below reports."""
    tabs = compile_progs(ws, dev)
    sel = torch.arange(len(ws), device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    z = sum(enc(views(sel, n_demos, tabs, dev, g)) for _ in range(reps))
    return F.normalize(z, dim=-1)


@torch.no_grad()
def retrieval(enc, ws, n_demos, dev, seed):
    """One fresh view per held-out transformation, matched against the bank of codes. Top-1 accuracy; chance = 1/len(ws)."""
    bank = codes_for(enc, ws, n_demos, dev, seed)
    tabs = compile_progs(ws, dev)
    g = torch.Generator(device=dev).manual_seed(seed + 777)
    q = enc(views(torch.arange(len(ws), device=dev), n_demos, tabs, dev, g))
    pred = (q @ bank.T).argmax(1)
    same = (q * bank).sum(-1).mean().item()                            # cosine between two views of the SAME function
    diff = ((q @ bank.T).sum() - (q * bank).sum()) / (len(ws) * (len(ws) - 1))
    return (pred == torch.arange(len(ws), device=dev)).float().mean().item(), same, float(diff)


def probe(train_z, train_y, held_z, held_y, n_cls, dev, steps=1500, lr=3e-3):
    """A small MLP probe on FROZEN codes. `train_y` is (N, n_out) of class indices; returns exact-match over all outputs."""
    n_out = train_y.shape[1]
    net = nn.Sequential(nn.Linear(train_z.shape[1], 256), nn.GELU(), nn.Linear(256, n_out * n_cls)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        logit = net(train_z).reshape(-1, n_out, n_cls)
        loss = F.cross_entropy(logit.reshape(-1, n_cls), train_y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        def ex(z, y):
            p = net(z).reshape(-1, n_out, n_cls).argmax(-1)
            return (p == y).all(-1).float().mean().item()
        return ex(train_z, train_y), ex(held_z, held_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--held", type=int, default=400)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256, help="transformations per step; also the number of negatives")
    ap.add_argument("--demos", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--pool", default="mean", choices=["mean", "set", "cross"])
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_cap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    _probe_x, table = cayley(dev, args.seed)
    progs = [w for w in table.values() if 0 < len(w) <= MAXM]
    g0 = torch.Generator().manual_seed(args.seed)
    by = collections.defaultdict(list)
    for w in progs:
        by[len(w)].append(w)
    held, pool = [], []
    frac = args.held / len(progs)
    for d, ws in sorted(by.items()):
        order = torch.randperm(len(ws), generator=g0).tolist()
        cut = min(len(ws) - 1, max(3, int(round(frac * len(ws)))))
        held += [ws[i] for i in order[:cut]]
        pool += [ws[i] for i in order[cut:]]
    order = torch.randperm(len(pool), generator=g0).tolist()
    train = [pool[i] for i in order][:args.n_train]
    print(f"device {dev} | universe depth<={MAXM}: {len(progs)} | train {len(train)} | held-out {len(held)}")
    print(f"contrastive: 2 independent demonstration sets per transformation, {args.batch} negatives, temp {args.temp}")
    print("NO group structure is used anywhere -- only the ability to execute a transformation.\n")

    enc = Encoder(dim=args.dim, pool=args.pool).to(dev)
    # The permutation-invariance over demonstrations is ASSERTED, not assumed.
    tabs_tr = compile_progs(train, dev)
    gchk = torch.Generator(device=dev).manual_seed(11)
    vv = views(torch.arange(8, device=dev), args.demos, tabs_tr, dev, gchk)
    with torch.no_grad():
        a = enc(vv)
        b = enc(vv[:, torch.randperm(args.demos, generator=gchk, device=dev)])
    assert torch.allclose(a, b, atol=1e-5), "encoder is not permutation-invariant over demonstrations"
    print("encoder verified permutation-invariant over demonstrations")

    rand_enc = Encoder(dim=args.dim, pool=args.pool).to(dev)                            # the untrained control, kept aside
    rand_enc.load_state_dict(enc.state_dict())

    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    g = torch.Generator(device=dev).manual_seed(args.seed)
    n_tr = len(train)
    t0 = time.time()
    for _ in range(args.steps):
        # WITHOUT replacement: a duplicate transformation in the batch would be a false negative.
        sel = torch.randperm(n_tr, generator=g, device=dev)[:args.batch]
        v1, v2 = views(sel, args.demos, tabs_tr, dev, g), views(sel, args.demos, tabs_tr, dev, g)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            loss = info_nce(enc(v1).float(), enc(v2).float(), args.temp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        sched.step()
    print(f"encoder trained in {time.time() - t0:.0f}s | final InfoNCE {loss.item():.4f}\n")

    # The probe is a READ-OUT, so it gets ALL the training transformations. A stratified subsample left it ~60 points for
    # an 11-output target and it simply memorised them (train 1.000 / held-out 0.000 for both encoders), which measures
    # the probe's capacity rather than the code's content. Retrieval keeps a modest bank so chance stays interpretable.
    tr_s, he_s = train, stratify(held, args.eval_cap)
    bank = stratify(held, 100)
    def targets(ws):
        it, vt = compile_progs(ws, dev)
        canon = torch.cat([it, vt], dim=1)
        dist = torch.tensor([[len(w)] for w in ws], device=dev)
        return canon, dist

    # The canonical state is probed in TWO HALVES, because they are not equally hard and lumping them hides which one
    # failed. `vmp` is a global value shift, readable off any single position; `idx` is the PERMUTATION, and recovering
    # it is the matching/binding problem this whole file exists to test.
    print(f"{'encoder':<12}{'retr(tr)':>10}{'retr(held)':>11}{'cos same':>10}{'cos diff':>10}"
          f"{'IDX probe':>13}{'VMP probe':>13}{'DIST probe':>13}")
    for name, e in (("random", rand_enc), ("contrastive", enc)):
        r_tr, _s, _d = retrieval(e, stratify(train, 100), args.demos, dev, args.seed)
        r, same, diff = retrieval(e, bank, args.demos, dev, args.seed)
        ztr = codes_for(e, tr_s, args.demos, dev, args.seed)
        zhe = codes_for(e, he_s, args.demos, dev, args.seed)
        c_tr, c_he = targets(tr_s)[0], targets(he_s)[0]
        d_tr, d_he = targets(tr_s)[1], targets(he_s)[1]
        ie = probe(ztr, c_tr[:, :L], zhe, c_he[:, :L], L, dev)          # the permutation
        ve = probe(ztr, c_tr[:, L:], zhe, c_he[:, L:], V, dev)          # the value map
        de = probe(ztr, d_tr, zhe, d_he, MAXM + 1, dev)
        print(f"{name:<12}{r_tr:>10.3f}{r:>11.3f}{same:>10.3f}{diff:>10.3f}"
              f"{ie[0]:>6.3f}/{ie[1]:<6.3f}{ve[0]:>6.3f}/{ve[1]:<6.3f}{de[0]:>6.3f}/{de[1]:<6.3f}")

    print(f"\nretrieval: fresh view matched against a bank of {len(he_s)} HELD-OUT codes; chance {1 / len(he_s):.4f}.")
    print("'cos same' is the cosine between two views of the SAME transformation, 'cos diff' between different ones --")
    print("a canonical form wants same~1 and diff~0. Probe columns are train/HELD-OUT exact-match on FROZEN codes.")
    print("CANON probe = can (idx, vmp) be decoded from the code, i.e. did the encoder rediscover the canonical form")
    print("without being told it exists. DIST probe compares to: exact canonical 0.714 (ceiling), raw behaviour 0.276.")


if __name__ == "__main__":
    main()
