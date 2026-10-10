"""Gradient-quality diagnostic: cosine between an estimator's weight gradients and backprop's, per weight matrix, on
fixed batches, at a model state (fresh init, or a checkpoint trained by backprop for a few seconds).

Usage:  python measure.py --ckpt runs/ckpt_bp20s.pt --K 32 --noise gauss --batches 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import data
import recall
from estimators import Cfg, backprop, estimate
from model import TinyGPT
from train import parse_cfg


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-30))


def names(model):
    out = {model.wte: "wte", model.wpe: "wpe", model.Whead: "head"}
    for b in range(model.L):
        out[model.Wqkv[b]] = f"qkv{b}"
        out[model.Wproj[b]] = f"proj{b}"
        out[model.Wfc[b]] = f"fc{b}"
        out[model.Wout[b]] = f"out{b}"
    for b in model.aux_blocks:
        out[model.Waux[str(b)]] = f"aux{b}"
    return out


def build(args, dev):
    model = TinyGPT(data.V if args.task == "latin" else recall.V, d=args.d, L=args.L, H=args.H, T=args.T, aux_every=args.aux_every,
                    topk_frac=args.topk).to(dev)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=dev)
        model.load_state_dict(sd, strict=False)
    return model


def participation(e):
    """Effective rank (participation ratio) of the per-token errors e (B, T, D): (sum λ)^2 / sum λ^2."""
    x = e.reshape(-1, e.shape[-1]).float()
    lam = torch.linalg.eigvalsh(x.T @ x).clamp(min=0)
    return float(lam.sum() ** 2 / (lam ** 2).sum().clamp(min=1e-30))


def measure(model, cfg, batches, seed, dev, aux_weight=0.0, warm=()):
    """Per-matrix cosine to backprop averaged over batches; also the per-token error cosine at each site, and the
    effective rank of the TRUE per-token errors at each site. One estimator state persists over warm + measured batches."""
    g = torch.Generator(device=dev).manual_seed(seed)
    state = {}
    for idx, tgt in warm:
        estimate(model, idx, tgt, cfg, state, g)
    pr = {}
    nm = names(model)
    per = {v: [] for v in nm.values()}
    site_cos = {}
    cost = 0.0
    t0 = time.time()
    for idx, tgt in batches:
        true, terr, _ = backprop(model, idx, tgt, aux_weight)
        if cfg.noise == "oracle":
            for s, e in terr.items():
                parts = [(s, e)] if s[0] != "qkv" else [((k, s[1]), e[..., j * model.d:(j + 1) * model.d])
                                                        for j, k in enumerate("qkv")]
                for key, x in parts:
                    x = x.reshape(-1, x.shape[-1]).float()
                    state[("O", key)] = torch.linalg.eigh(x.T @ x)[1]
        est, st = estimate(model, idx, tgt, cfg, state, g)
        cost = st["cost"]
        for p, gr in est.items():
            if p in true:
                per[nm[p]].append(cos(gr, true[p]))
        for s, e in st["err"].items():
            if s in terr:
                k = "".join(map(str, s))
                site_cos.setdefault(k, []).append(cos(e, terr[s]))
                pr.setdefault(k, []).append(participation(terr[s]))
    torch.cuda.synchronize()
    res = {k: round(sum(v) / len(v), 4) for k, v in per.items() if v}
    return dict(cos=res, mean=round(sum(res.values()) / len(res), 4), site_err_cos={k: round(sum(v) / len(v), 4)
                for k, v in site_cos.items()}, true_err_rank={k: round(sum(v) / len(v), 1) for k, v in pr.items()},
                cost=round(cost, 1), secs=round(time.time() - t0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--task", choices=["latin", "recall"], default="latin")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--aux_every", type=int, default=0)
    ap.add_argument("--topk", type=float, default=0.0)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--warm", type=int, default=0, help="estimator calls before measuring (builds guided's covariance)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="Cfg overrides, e.g. K=64 noise=orth gamma=0.9")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(args.seed)
    model = build(args, dev)
    cfg = parse_cfg(["amp=fp16"] + args.set)
    gb = torch.Generator(device=dev).manual_seed(123)
    if args.task == "latin":
        tr, va = data.load(dev)
        batches = [data.batch(va, args.B, args.T, gb) for _ in range(args.batches + args.warm)]
    else:
        batches = [recall.batch(args.B, args.T, gb) for _ in range(args.batches + args.warm)]
    r = measure(model, cfg, batches[args.warm:], args.seed, dev, warm=batches[:args.warm])
    print(json.dumps(r))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(args=vars(args), cfg={k: v for k, v in cfg.__dict__.items()}, **r), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
