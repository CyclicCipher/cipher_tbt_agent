"""Experiment 1 runner (EXPERIMENT_1.md): one arm per invocation — train, probe in-context extrapolation during training
(M3), then score the in-context curve (M1), its out-of-range version (M4), and write runs/e1/<arm>.json.

Arms: A0 (next-step + rollouts), A1 (particles, energy score), A2 (density per horizon), A2shuf (A2 on order-destroyed
sequences). Budget: <= 2 minutes per arm, training plus evaluation.

Usage:  python run_e1.py --arm A2 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import generators as G
from heads import K, Backbone, MixtureHead, ParticleHead, energy_score, features, mixture_logpdf, mixture_sample, targets

HERE = Path(__file__).resolve().parent
L, PER_FAMILY = 48, 18
NS = [1, 2, 4, 8, 16, 32]
BUCKETS = {"k1": (1, 1), "k2-4": (2, 4), "k5-8": (5, 8), "k9-16": (9, 16)}
BANDWIDTHS = [0.003, 0.01, 0.03, 0.1]
LN2, LOG_FLOOR = math.log(2), math.log(0.01)


class Model(torch.nn.Module):
    def __init__(self, arm, d, layers):
        super().__init__()
        self.arm = arm
        self.bb = Backbone(d=d, layers=layers)
        self.head = ParticleHead(d) if arm == "A1" else MixtureHead(d, 1 if arm == "A0" else K)
        self.particles = 8

    def loss(self, p):
        h = self.bb(p)
        Y, m = targets(p)
        if self.arm == "A1":
            X = self.head(h, p, P=self.particles)
            valid = m[:, :, 0] > 0
            return energy_score(X.float(), Y, m)[valid].mean()
        logw, mu, logs = self.head(h, p)
        if self.arm == "A0":
            lp = mixture_logpdf(logw[:, :, 0].float(), mu[:, :, 0].float(), logs[:, :, 0].float(), Y[:, :, 0])
            return -(lp * m[:, :, 0]).sum() / m[:, :, 0].sum()
        lp = mixture_logpdf(logw.float(), mu.float(), logs.float(), Y)
        return -(lp * m).sum() / m.sum()


# ------------------------------------------------------------------------------------------------------- scoring
def bits(logp_nats):
    """log2 density with 1% uniform mixed in (uniform density = 1 on the unit square)."""
    return torch.logaddexp(logp_nats + math.log(0.99), torch.full_like(logp_nats, LOG_FLOOR)) / LN2


def kde_logpdf(S, y, bw):
    """S (N, M, K, 2) samples, y (N, K, 2) -> (N, K) log density of a Gaussian KDE with bandwidth bw."""
    d2 = ((S - y[:, None]) ** 2).sum(-1)
    return torch.logsumexp(-0.5 * d2 / bw ** 2, 1) - math.log(S.shape[1]) - math.log(2 * math.pi * bw ** 2)


@torch.no_grad()
def predict(model, p, n, M=32):
    """For context size n: either ('density', (N, K) log density of the true next K points) or ('samples', (N, M, K, 2))."""
    model.eval()
    y = p[:, n:n + K]
    if model.arm in ("A2", "A2shuf"):
        logw, mu, logs = model.head(model.bb(p[:, :n]), p[:, :n])
        out = ("density", mixture_logpdf(logw[:, -1].float(), mu[:, -1].float(), logs[:, -1].float(), y))
    elif model.arm == "A1":
        X = model.head(model.bb(p[:, :n]), p[:, :n], P=M)
        out = ("samples", X[:, -1].float())
    else:                                                   # A0: autoregressive rollouts, incremental (k/v cache)
        tf = model.bb.tf
        cur = p[:, :n].repeat_interleave(M, 0)
        caches = tf.new_caches()
        h = tf.norm(tf.forward_embedded(model.bb.inp(features(cur)), caches, 0))
        new = []
        for k in range(K):
            logw, mu, logs = model.head(h[:, -1:], cur[:, -1:])
            nxt = mixture_sample(logw[:, 0, 0].float(), mu[:, 0, 0].float(), logs[:, 0, 0].float())
            new.append(nxt)
            cur = nxt[:, None]
            if k + 1 < K:
                h = tf.norm(tf.forward_embedded(model.bb.inp(features(cur)), caches, n + k))
        out = ("samples", torch.stack(new, 1).view(p.shape[0], M, K, 2))
    model.train()
    return out, y


def gain_table(model, p, fam, bw=None, ns=NS, preds=None):
    """{family: {n: {bucket: mean bits}}}; for sample-based arms `bw` maps bucket -> bandwidth. `preds` reuses
    predictions already computed by `predict` (n -> ((kind, v), y))."""
    res = {}
    for n in ns:
        (kind, v), y = preds[n] if preds is not None else predict(model, p, n)
        if kind == "samples":
            lp = torch.stack([kde_logpdf(v[:, :, [k - 1]], y[:, [k - 1]], bw[next(b for b, (lo, hi) in BUCKETS.items() if lo <= k <= hi)])[:, 0]
                              for k in range(1, K + 1)], 1)
        else:
            lp = v
        b = bits(lp)
        for i, f in enumerate(G.FAMILIES):
            sel = fam == i
            if sel.any():
                res.setdefault(f, {})[n] = {name: round(float(b[sel][:, lo - 1:hi].mean()), 3) for name, (lo, hi) in BUCKETS.items()}
    return res


def choose_bandwidths(model, p):
    """Per horizon bucket, the KDE bandwidth with the best mean score on a validation batch (n = 8 and 16); each
    prediction is computed once and scored at every bandwidth."""
    preds = {n: predict(model, p, n) for n in (8, 16)}
    best = {}
    for name, (lo, hi) in BUCKETS.items():
        scores = [sum(float(bits(kde_logpdf(v[:, :, lo - 1:hi], y[:, lo - 1:hi], bw)).mean()) for (_, v), y in preds.values())
                  for bw in BANDWIDTHS]
        best[name] = BANDWIDTHS[max(range(len(BANDWIDTHS)), key=lambda i: scores[i])]
    return best


def structured_mean(tab, n, bucket):
    return round(sum(tab[f][n][bucket] for f in G.STRUCTURED) / len(G.STRUCTURED), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A0", "A1", "A2", "A2shuf"], required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probes", type=int, default=10)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    train, _ = G.batch(args.steps * PER_FAMILY, L, g, dev)                     # one fresh task per training example
    train = train[torch.randperm(len(train), generator=g, device=dev)]
    ge = torch.Generator(device=dev).manual_seed(10_000 + args.seed)
    test, test_f = G.batch(64, L, ge, dev, split="eval")
    val, _ = G.batch(32, L, ge, dev, split="eval")
    probe, probe_f = G.batch(32, L, ge, dev, split="eval")
    ood, ood_f = G.batch(64, L, ge, dev, ood=True, split="eval")
    model = Model(args.arm, args.d, args.layers).to(dev)
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / 100 if s < 100 else 0.5 * (1 + math.cos(math.pi * (s - 100) / max(1, args.steps - 100))))
    B = PER_FAMILY * len(G.FAMILIES)
    probes, t_train = [], 0.0
    every = max(1, args.steps // args.probes)
    for step in range(args.steps):
        t1 = time.time()
        p = train[step * B:(step + 1) * B]
        if args.arm == "A2shuf":
            p = p.gather(1, torch.argsort(torch.rand(p.shape[:2], device=dev), 1)[..., None].expand(-1, -1, 2))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(p)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        t_train += time.time() - t1
        if (step + 1) % every == 0:
            tab = gain_table(model, probe, probe_f, bw={b: 0.03 for b in BUCKETS}, ns=[8])
            probes.append(dict(step=step + 1, loss=round(loss.item(), 4), far_n8=structured_mean(tab, 8, "k9-16"),
                               k1_n8=structured_mean(tab, 8, "k1"),
                               per_family={f: tab[f][8]["k9-16"] for f in tab}))
            print(f"  step {step + 1:5d}  loss {loss.item():8.4f}  n=8 gain (structured): k1 {probes[-1]['k1_n8']:+.2f}"
                  f"  k9-16 {probes[-1]['far_n8']:+.2f}  [{time.time() - t0:5.1f}s]", flush=True)
    bw = choose_bandwidths(model, val) if args.arm in ("A0", "A1") else None
    tab = gain_table(model, test, test_f, bw)
    tab_ood = gain_table(model, ood, ood_f, bw, ns=[16])
    total = time.time() - t0
    print(f"\n{args.arm}: {n_params:,} params | train {t_train:.0f}s | total {total:.0f}s | bandwidths {bw}")
    print("bits gained over uniform, mean of structured families, by context size n (rows) and horizon (cols)")
    print("   n  " + "  ".join(f"{b:>6s}" for b in BUCKETS))
    for n in NS:
        print(f"  {n:2d}  " + "  ".join(f"{structured_mean(tab, n, b):+6.2f}" for b in BUCKETS))
    print("per family, n=16:  " + "  ".join(f"{f} k1 {tab[f][16]['k1']:+.2f} far {tab[f][16]['k9-16']:+.2f}" for f in G.FAMILIES))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(args=vars(args), params=n_params, train_s=round(t_train, 1), total_s=round(total, 1), bandwidths=bw,
                       probes=probes, test=tab, ood=tab_ood), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
