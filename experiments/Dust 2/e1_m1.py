"""Experiment 1, M1 (gradient quality) and M0 (depth): every arm's cosine to backprop at the checkpoints, converted into
equivalent draws K_eq with the baseline curve (runs/e1/curve). Results: runs/e1/m1/<task>_<arm>.json and a summary.

Usage:  python e1_m1.py            (all arms, both tasks, then M0)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

import data
import recall
from estimators import backprop
from measure import measure
from model import TinyGPT
from train import parse_cfg

OUT = Path("runs/e1/m1")
KS = [8, 16, 32, 64, 128, 256, 512]
ARMS = [  # name, cfg overrides, aux_every, baseline K at the same cost
    ("base", [], 0, 32),
    ("lr_rand8", ["noise=lr_rand", "rank=8"], 0, 32),
    ("lr_rand16", ["noise=lr_rand", "rank=16"], 0, 32),
    ("lr_pca8", ["noise=lr_pca", "rank=8"], 0, 32),
    ("orth", ["noise=orth"], 0, 32),
    ("sobol", ["noise=sobol"], 0, 32),
    ("anti", ["noise=anti"], 0, 32),
    ("local32", ["local=1"], 1, 8),
    ("local128", ["local=1", "K=128"], 1, 32),
    ("ent50", ["ent_frac=0.5"], 0, 32),
    ("ent25", ["ent_frac=0.25"], 0, 32),
    ("guided", ["noise=guided", "rank=8", "beta=0.5"], 0, 32),
    ("oracle_b1", ["noise=oracle", "rank=8", "beta=1.0"], 0, 32),
    ("oracle_b05", ["noise=oracle", "rank=8", "beta=0.5"], 0, 32),
]


def k_eq(c, curve):
    """Baseline draws giving mean cosine c, log-linear between measured points (extrapolated at the ends)."""
    xs = [math.log(k) for k in KS]
    ys = [curve[k] for k in KS]
    i = 0 if c < ys[1] else len(KS) - 2 if c >= ys[-2] else max(j for j in range(len(KS) - 1) if ys[j] <= c)
    return math.exp(xs[i] + (c - ys[i]) / (ys[i + 1] - ys[i]) * (xs[i + 1] - xs[i]))


def fit_aux(model, get, steps=300):
    """Local-loss arms need trained auxiliary heads: fit them (backbone frozen) by backprop on their own loss."""
    opt = torch.optim.Adam([p for p in model.Waux.values()], lr=1e-2)
    for _ in range(steps):
        idx, tgt = get()
        c = model.forward_cache(idx, tgt)
        loss = sum(c["loss_aux", b].mean() for b in model.aux_blocks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in model.parameters():
            if all(p is not q for q in model.Waux.values()):
                p.grad = None
        opt.step()


def main():
    dev = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va = data.load(dev)
    summary = {}
    for task, ck, V in [("latin", "runs/ckpt_bp300.pt", data.V), ("recall", "runs/ckpt_recall_bp4000.pt", recall.V)]:
        curve = {k: json.load(open(f"runs/e1/curve/{task}_K{k}.json"))["mean"] for k in KS}
        gb = torch.Generator(device=dev).manual_seed(123)
        mk = (lambda g: data.batch(va, 32, 64, g)) if task == "latin" else (lambda g: recall.batch(32, 64, g))
        warm = [mk(gb) for _ in range(20)]
        gb = torch.Generator(device=dev).manual_seed(123)
        batches = [mk(gb) for _ in range(4)]          # the same 4 batches as the curve
        gt = torch.Generator(device=dev).manual_seed(7)
        gtrain = (lambda: data.batch(tr, 32, 64, gt)) if task == "latin" else (lambda: recall.batch(32, 64, gt))
        for name, sets, aux, kmatch in ARMS:
            if len(sys.argv) > 1 and name not in sys.argv[1:]:
                continue
            torch.manual_seed(0)
            model = TinyGPT(V, aux_every=aux).to(dev)
            model.load_state_dict(torch.load(ck, map_location=dev), strict=False)
            if aux:
                fit_aux(model, gtrain)
            cfg = parse_cfg(["K=32", "amp=fp16"] + sets)
            r = measure(model, cfg, batches, 0, dev, warm=warm if cfg.noise == "guided" else ())
            ke = k_eq(r["mean"], curve)
            r.update(arm=name, task=task, k_eq=round(ke, 1), gain=round(ke / kmatch, 2), k_match=kmatch)
            json.dump(r, open(OUT / f"{task}_{name}.json", "w"), indent=1)
            summary[task, name] = r
            print(f"{task:6s} {name:10s} mean cos {r['mean']:.3f}  K_eq {ke:6.1f}  gain {ke / kmatch:5.2f}  "
                  f"cost {r['cost']:6.1f}  {r['secs']:.1f}s", flush=True)
    # M0: depth. Baseline at L = 8.
    if len(sys.argv) == 1 or "depth" in sys.argv[1:]:
        ck8 = Path("runs/ckpt_bp300_L8.pt")
        model = TinyGPT(data.V, L=8).to(dev)
        model.load_state_dict(torch.load(ck8, map_location=dev))
        gb = torch.Generator(device=dev).manual_seed(123)
        batches = [data.batch(va, 32, 64, gb) for _ in range(4)]
        r = measure(model, parse_cfg(["K=32", "amp=fp16"]), batches, 0, dev)
        json.dump(r, open(OUT / "latin_depth_L8.json", "w"), indent=1)
        e = r["site_err_cos"]
        for kind in ("o", "proj", "out", "fc", "qkv"):
            print(f"depth L=8  {kind:5s} per-token error cos by block: "
                  + " ".join(f"{e[f'{kind}{b}']:.3f}" for b in range(8)))
        print("depth L=8  weight cos by block (out): " + " ".join(f"{r['cos'][f'out{b}']:.3f}" for b in range(8)))


if __name__ == "__main__":
    main()
