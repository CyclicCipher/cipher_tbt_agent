"""Train the tiny GPT on Latin characters by backprop or by a Dust-style estimator (estimators.py).

Usage:  python train.py --method dust --steps 300 --lr 3e-3 --set K=32 noise=orth --json runs/e1/orth_s0.json
        python train.py --method bp --steps 3000 --save runs/ckpt_bp.pt
Adam (0.9, 0.99) for every method, 20 warmup steps, cosine decay to zero. Validation loss (nats per character) on 8
fixed held-out batches. Cost is counted in forward-pass equivalents of one training batch (backprop: 3 per step).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import data
import recall
from estimators import Cfg, backprop, estimate
from attnres import ARGPT
from model import TinyGPT


@torch.no_grad()
def val_loss(model, vb):
    """Mean loss per SCORED token (recall scores only the answers)."""
    return sum((model.forward_cache(i, t)["loss"].sum() / (t >= 0).sum()).item() for i, t in vb) / len(vb)


def make_opt(model, kind, lr, mom=0.95, emb_mult=1.0, mix_mult=1.0):
    """Adam (0.9, 0.99) as before, or SGD with momentum and a separate learning-rate multiplier for the embeddings.
    AttnRes pseudo-queries can get their own multiplier (mix_mult)."""
    if kind == "adam":
        mix = list(getattr(model, "Wmix", []))
        if mix and mix_mult != 1.0:
            rest = [p for p in model.parameters() if all(p is not q for q in mix)]
            return torch.optim.Adam([dict(params=rest, mult=1.0), dict(params=mix, mult=mix_mult)], lr=lr,
                                    betas=(0.9, 0.99))
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    emb = [model.wte, model.wpe]
    rest = [p for p in model.parameters() if all(p is not e for e in emb)]
    return torch.optim.SGD([dict(params=rest, mult=1.0), dict(params=emb, mult=emb_mult)], lr=lr, momentum=mom)


def parse_cfg(pairs):
    cfg = Cfg()
    for kv in pairs:
        k, v = kv.split("=", 1)
        if k.startswith("sig."):
            cfg.sig[k[4:]] = float(v)
            continue
        cur = getattr(cfg, k)
        setattr(cfg, k, (v in ("1", "True", "true")) if isinstance(cur, bool) else type(cur)(v))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["bp", "dust"], required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--task", choices=["latin", "recall"], default="latin")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--aux_every", type=int, default=0)
    ap.add_argument("--aux_weight", type=float, default=0.0, help="bp only: weight of the auxiliary losses")
    ap.add_argument("--topk", type=float, default=0.0)
    ap.add_argument("--attnres", action="store_true", help="Block Attention Residuals (attnres.py)")
    ap.add_argument("--mix_lr", type=float, default=1.0, help="AttnRes: learning-rate multiplier for the pseudo-queries")
    ap.add_argument("--fast", action="store_true", help="torch.compile the rerun path (apparatus)")
    ap.add_argument("--amp", default="bf16", help="bp only: '', 'bf16' or 'fp16' autocast (dust: --set amp=bf16)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--mom", type=float, default=0.95, help="SGD momentum")
    ap.add_argument("--emb_mult", type=float, default=1.0, help="SGD: lr multiplier for wte/wpe (Dust uses a large one)")
    ap.add_argument("--sig_sched", default="", help="T1: 'start,end' -- rerun-site σ annealed geometrically over the run")
    ap.add_argument("--save", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.json and Path(args.json).exists():          # never re-run an unchanged experiment (the user's rule)
        print(f"{args.json} exists: not re-run")
        return
    dev = "cuda"
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    gv = torch.Generator(device=dev).manual_seed(999)
    if args.task == "latin":
        tr, va = data.load(dev)
        vb = [data.batch(va, 64, args.T, gv) for _ in range(8)]
        get = lambda: data.batch(tr, args.B, args.T, g)
        gtb = torch.Generator(device=dev).manual_seed(4242)
        tb = [data.batch(tr, 64, args.T, gtb) for _ in range(8)]
        V = data.V
    else:
        vb = [recall.batch(64, args.T, gv) for _ in range(8)]
        get = lambda: recall.batch(args.B, args.T, g)
        tb = []
        V = recall.V
    model = (ARGPT if args.attnres else TinyGPT)(V, d=args.d, L=args.L, H=args.H, T=args.T, aux_every=args.aux_every,
                    topk_frac=args.topk).to(dev)
    cfg = parse_cfg(args.set)
    TinyGPT.fast = args.fast
    opt = make_opt(model, args.opt, args.lr, args.mom, args.emb_mult, args.mix_lr)
    state, log, cost = {}, [], 0.0
    t0 = time.time()
    v0 = val_loss(model, vb)
    log.append(dict(step=0, t=0.0, cost=0.0, val=round(v0, 4)))
    print(f"step     0  val {v0:.4f}", flush=True)
    for step in range(1, args.steps + 1):
        lr = args.lr * (step / 20 if step <= 20 else 0.5 * (1 + math.cos(math.pi * (step - 20) / (args.steps - 20))))
        for gp in opt.param_groups:
            gp["lr"] = lr * gp.get("mult", 1.0)
        idx, tgt = get()
        if args.sig_sched:
            s0, s1 = map(float, args.sig_sched.split(","))
            for k in ("emb", "o", "proj", "fc", "out", "qkv"):
                cfg.sig[k] = s0 * (s1 / s0) ** ((step - 1) / max(1, args.steps - 1))
        if args.method == "bp":
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dict(bf16=torch.bfloat16, fp16=torch.float16).get(args.amp, torch.float32),
                                enabled=bool(args.amp)):
                c = model.forward_cache(idx, tgt)
                loss = c["loss"].mean() + sum(args.aux_weight * c["loss_aux", b].mean() for b in model.aux_blocks)
            loss.backward()
            cost += 3.0
        else:
            grads, st = estimate(model, idx, tgt, cfg, state, g)
            for p in model.parameters():
                p.grad = grads.get(p)
            cost += st["cost"]
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            torch.cuda.synchronize()
            v = val_loss(model, vb)
            trl = round(val_loss(model, tb), 4) if tb else None
            log.append(dict(step=step, t=round(time.time() - t0, 1), cost=round(cost), val=round(v, 4), train=trl))
            print(f"step {step:5d}  t {time.time() - t0:6.1f}s  cost {cost:9.0f} fwd  val {v:.4f}", flush=True)
    total = time.time() - t0
    print(f"done: {args.method} {' '.join(args.set)} | val {log[-1]['val']:.4f} | {total:.0f}s | {cost:.0f} fwd-equiv")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(args=vars(args), cfg={k: v for k, v in cfg.__dict__.items()}, log=log, secs=round(total, 1),
                       cost=round(cost), val=log[-1]["val"]), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
