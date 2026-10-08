"""Experiment 2 runner (EXPERIMENT_2.md): train the fixed base model for a WALL-CLOCK budget with one bucket-weighting
rule (curriculum.py), probe every 10% of the budget, then score the in-context curve with run_e1's evaluation.

Usage:  python run_e2.py --arm GA --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import generators as G
from curriculum import ARMS, HGROUP, N_BUCKETS
from heads import mixture_logpdf, targets, with_uniform
from run_e1 import BUCKETS, NS, PER_FAMILY, L, Model, gain_table

NR = ["circle", "spiral", "lissajous", "billiard"]


def term_nll(model, p):
    """Per-term negative log-likelihood (B, L, K) and validity mask, for the A2 query head with a uniform component."""
    h = model.bb(p)
    Y, m = targets(p)
    logw, mu, logs = model.head(h, p)
    lu = model.head.last_log_u
    lp = with_uniform(mixture_logpdf(logw.float(), mu.float(), logs.float(), Y), lu.float() if lu is not None else None)
    return -lp, m


def bucket_stats(nll, m, fam):
    """ids (B, L, K) bucket of each term; per-bucket sums and counts over valid terms."""
    ids = fam[:, None, None] * 4 + HGROUP.to(nll.device)[None, None, :]
    ids = ids.expand_as(nll)
    v = m > 0
    sums = torch.zeros(N_BUCKETS, device=nll.device).index_add(0, ids[v], nll[v])
    counts = torch.zeros(N_BUCKETS, device=nll.device).index_add(0, ids[v], torch.ones_like(nll[v]))
    return ids, sums, counts


def four_fam(tab, n, b="k9-16"):
    return round(sum(tab[f][n][b] for f in NR) / len(NR), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=70.0, help="seconds of training (wall-clock)")
    ap.add_argument("--lr", type=float, default=1.6e-2)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    max_steps = 4000
    train, train_f = G.batch(max_steps * PER_FAMILY, L, g, dev)
    perm = torch.randperm(len(train), generator=g, device=dev)
    train, train_f = train[perm], train_f[perm]
    ge = torch.Generator(device=dev).manual_seed(10_000 + args.seed)
    test, test_f = G.batch(64, L, ge, dev, split="eval")
    probe, probe_f = G.batch(32, L, ge, dev, split="eval")
    model = Model("A2", 64, 3, diff=True, head="query", unif=True).to(dev)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    cur = ARMS[args.arm](dev)
    B = PER_FAMILY * len(G.FAMILIES)
    probes, step, t_train, ga_s, next_probe = [], 0, 0.0, 0.0, 0.1
    torch.cuda.synchronize()
    while t_train < args.budget and step < max_steps:
        t1 = time.time()
        frac = t_train / args.budget
        lr = args.lr * ((step + 1) / 100 if step < 100 else 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))
        for gp in opt.param_groups:
            gp["lr"] = lr
        p, fam = train[step * B:(step + 1) * B], train_f[step * B:(step + 1) * B]
        if args.arm == "GA":
            cur.snapshot(step, params)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nll, m = term_nll(model, p)
        ids, sums, counts = bucket_stats(nll, m, fam)
        L_b = sums / counts.clamp(min=1)
        cur.observe(step, L_b)
        if args.arm == "AF":
            mk = cur.term_mask(step, nll, ids, m)
            loss = (nll * mk).sum() / mk.sum()
        else:
            c = cur.multipliers()
            loss = (c * sums).sum() / (c * counts).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if args.arm == "GA" and cur.due(step):
            t2 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nll2, m2 = term_nll(model, p)
            _, sums2, counts2 = bucket_stats(nll2, m2, fam)
            cur.refresh(step, sums2 / counts2.clamp(min=1), params, opt)
            torch.cuda.synchronize()
            ga_s += time.time() - t2
        torch.cuda.synchronize()
        t_train += time.time() - t1
        if t_train >= next_probe * args.budget:
            next_probe += 0.1
            tab = gain_table(model, probe, probe_f, ns=[8])
            c = cur.multipliers().view(7, 4)
            probes.append(dict(step=step, t=round(t_train, 1), loss=round(loss.item(), 4), far_n8=four_fam(tab, 8),
                               rotation_k1_n8=tab["rotation"][8]["k1"],
                               c_family={f: round(float(c[i].mean()), 3) for i, f in enumerate(G.FAMILIES)},
                               c_horizon={b: round(float(c[:, j].mean()), 3) for j, b in enumerate(BUCKETS)}))
            print(f"  step {step:5d}  t {t_train:5.1f}s  loss {loss.item():7.4f}  4-fam far n=8 {probes[-1]['far_n8']:+.2f}"
                  f"  rotation k1 n=8 {probes[-1]['rotation_k1_n8']:+.2f}  c_family "
                  + " ".join(f"{k[:3]}{v:.2f}" for k, v in probes[-1]["c_family"].items()), flush=True)
    tab = gain_table(model, test, test_f)
    total = time.time() - t0
    worst_unif = min(tab["uniform"][n][b] for n in NS[1:] for b in BUCKETS)
    print(f"\n{args.arm} seed {args.seed}: {step} steps in {t_train:.0f}s (GA bookkeeping {ga_s:.1f}s) | total {total:.0f}s")
    print("  4-fam far: " + "  ".join(f"n={n} {four_fam(tab, n):+.2f}" for n in (8, 16, 32))
          + f"  | rotation k1 n=16 {tab['rotation'][16]['k1']:+.2f} | uniform worst n>=2 {worst_unif:+.2f}"
          + f" | pen far n=16 {tab['pen'][16]['k9-16']:+.2f}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(args=vars(args), steps=step, train_s=round(t_train, 1), ga_s=round(ga_s, 1), total_s=round(total, 1),
                       probes=probes, test=tab), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
