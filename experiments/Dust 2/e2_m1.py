"""Experiment 2 (own ideas), the gradient-quality part: every arm's cosine to backprop at the checkpoints, converted into
equivalent draws with the baseline curve of the same model; and the width-scaling measurement (W).

Usage:  python e2_m1.py arms        (O1-O4 and the combination, both tasks)
        python e2_m1.py width       (W: d = 64, 128, 256)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

import data
import recall
from e1_m1 import KS, fit_aux, k_eq
from measure import measure
from model import TinyGPT
from train import parse_cfg

OUT = Path("runs/e2/m1")
TinyGPT.fast = True                                  # compiled rerun path (apparatus; same maths)
COMBO = ["noise=guided", "rank=8", "beta=0.5", "sparse_c=2", "top_guide=0.3", "hub_T=1", "exact_head=1"]
ARMS = [  # name, cfg overrides, model kind (the ReLU² baseline is Experiment 1's runs/e1/m1/<task>_base.json: not re-run)
    ("O1_sparse", ["sparse_c=2"], "relu2"),
    ("O2_exact_head", ["exact_head=1"], "relu2"),
    ("O3_top", ["top_guide=0.3", "exact_head=1"], "relu2"),
    ("O3_top05", ["top_guide=0.5", "exact_head=1"], "relu2"),
    ("O4_hubT", ["hub_T=1"], "relu2"),
    ("O4_hubT_K56", ["hub_T=1", "K=56"], "relu2"),
    ("combo", COMBO, "relu2"),
    ("combo_K56", COMBO + ["K=56"], "relu2"),
    ("top10_O1", ["sparse_c=2"], "top10"),       # the top-10% model's baseline is its curve's K = 32 point
    ("combo_local", COMBO + ["local=1", "K=56"], "relu2_aux"),
    # Experiment 2b (whitened guides)
    ("guided_w", ["noise=guided", "rank=8", "beta=0.5", "whiten=1"], "relu2"),
    ("top_w", ["top_guide=0.3", "exact_head=1", "whiten=1"], "relu2"),
    ("combo_w_K56", COMBO + ["K=56", "whiten=1"], "relu2"),
    ("O4_orth_K56", ["hub_T=1", "K=56", "noise=orth"], "relu2"),
]


def curve_cos(task, kind, dev, batches, ck):
    """The baseline curve of the model in use (Experiment 1's for ReLU²; measured here for the top-k model)."""
    if kind == "relu2":
        return {k: json.load(open(f"runs/e1/curve/{task}_K{k}.json"))["mean"] for k in KS}
    f = OUT / f"curve_{task}_{kind}.json"
    if f.exists():
        return {int(k): v for k, v in json.load(open(f)).items()}
    model = load(task, kind, ck, dev)
    cur = {k: measure(model, parse_cfg([f"K={k}", "amp=fp16"]), batches, 0, dev)["mean"] for k in KS}
    json.dump(cur, open(f, "w"))
    return cur


def load(task, kind, ck, dev, d=64, L=4, aux_every=0):
    V = data.V if task == "latin" else recall.V
    model = TinyGPT(V, d=d, L=L, H=d // 16, topk_frac=0.1 if kind == "top10" else 0.0, aux_every=aux_every).to(dev)
    model.load_state_dict(torch.load(ck, map_location=dev), strict=False)
    return model


def cost_of_K(K):
    """Baseline cost per step (forward-equivalents) is linear in K: 1 + 8.94 K (Experiment 1: K=32 -> 287)."""
    return 1 + (287.0 - 1) / 32 * K


def arms(dev):
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va = data.load(dev)
    for task in ("latin", "recall"):
        gb = torch.Generator(device=dev).manual_seed(123)
        mk = (lambda g: data.batch(va, 32, 64, g)) if task == "latin" else (lambda g: recall.batch(32, 64, g))
        warm = [mk(gb) for _ in range(20)]
        gb = torch.Generator(device=dev).manual_seed(123)
        batches = [mk(gb) for _ in range(4)]
        for name, sets, kind in ARMS:
            if len(sys.argv) > 2 and name not in sys.argv[2:]:
                continue
            base_kind = "relu2" if kind == "relu2_aux" else kind
            ck = {("latin", "relu2"): "runs/ckpt_bp300.pt", ("recall", "relu2"): "runs/ckpt_recall_bp4000.pt",
                  ("latin", "top10"): "runs/ckpt_bp300_top10.pt",
                  ("recall", "top10"): "runs/ckpt_recall_bp4000_top10.pt"}[task, base_kind]
            curve = curve_cos(task, base_kind, dev, batches, ck)
            model = load(task, base_kind, ck, dev, aux_every=1 if kind == "relu2_aux" else 0)
            if kind == "relu2_aux":
                gt = torch.Generator(device=dev).manual_seed(7)
                fit_aux(model, (lambda: data.batch(tr, 32, 64, gt)) if task == "latin" else (lambda: recall.batch(32, 64, gt)))
            cfg = parse_cfg(["K=32", "amp=fp16"] + sets)
            r = measure(model, cfg, batches, 0, dev, warm=warm if cfg.noise == "guided" else ())
            kmatch = (r["cost"] - 1) / ((287.0 - 1) / 32)            # the baseline's K at this arm's cost
            ke = k_eq(r["mean"], curve)
            r.update(arm=name, task=task, k_eq=round(ke, 1), gain=round(ke / kmatch, 2), k_match=round(kmatch, 1))
            json.dump(r, open(OUT / f"{task}_{name}.json", "w"), indent=1)
            print(f"{task:6s} {name:14s} mean cos {r['mean']:.3f}  K_eq {ke:6.1f}  matched K {kmatch:5.1f}  gain "
                  f"{ke / kmatch:5.2f}  cost {r['cost']:6.1f}  | fc0 {r['cos']['fc0']:.3f} out0 {r['cos']['out0']:.3f} "
                  f"proj0 {r['cos']['proj0']:.3f} qkv0 {r['cos']['qkv0']:.3f} wte {r['cos']['wte']:.3f}", flush=True)


def width(dev):
    """W: does the errors' effective rank grow with width, and with it the cost of plain Dust vs a subspace?"""
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va = data.load(dev)
    res = {}
    e1 = lambda n: json.load(open(f"runs/e1/{n}.json"))
    c32 = e1("curve/latin_K32")
    res[64] = row = dict(base_K32=c32["mean"], base_K128=e1("curve/latin_K128")["mean"],
                         base_K512=e1("curve/latin_K512")["mean"], base_cost=c32["cost"], rank=c32["true_err_rank"],
                         err_cos_K32=c32["site_err_cos"], oracle8=e1("m1/latin_oracle_b1")["mean"], oracle8_cost=c32["cost"],
                         guided8=e1("m1/latin_guided")["mean"], guided8_cost=c32["cost"],
                         combo=json.load(open(OUT / "latin_combo.json"))["mean"],
                         combo_cost=json.load(open(OUT / "latin_combo.json"))["cost"], reused="Experiment 1 + arms")
    finish(row, 64)
    for d in (128, 256):
        ck = f"runs/ckpt_bp300_d{d}.pt"
        model = load("latin", "relu2", ck, dev, d=d)
        gb = torch.Generator(device=dev).manual_seed(123)
        warm = [data.batch(va, 32, 64, gb) for _ in range(20)]
        gb = torch.Generator(device=dev).manual_seed(123)
        batches = [data.batch(va, 32, 64, gb) for _ in range(2)]
        row = {}
        for K in (32, 128, 512):
            r = measure(model, parse_cfg([f"K={K}", "amp=fp16"]), batches, 0, dev)
            row[f"base_K{K}"] = r["mean"]
            if K == 32:
                row["base_cost"] = r["cost"]
                row["rank"] = r["true_err_rank"]
                row["err_cos_K32"] = r["site_err_cos"]
        for name, sets, w in [("oracle8", ["noise=oracle", "rank=8", "beta=1.0"], ()),
                              ("guided8", ["noise=guided", "rank=8", "beta=0.5"], warm),
                              ("combo", COMBO, warm)]:
            r = measure(model, parse_cfg(["K=32", "amp=fp16"] + sets), batches, 0, dev, warm=w)
            row[name] = r["mean"]
            row[name + "_cost"] = r["cost"]
        finish(row, d)
        res[d] = row
    json.dump(res, open(OUT / "width.json", "w"), indent=1)


def finish(row, d):
    """Equivalent draws from this width's own curve (log-linear between K = 32, 128, 512), gains, average ranks."""
    xs, ys = [math.log(k) for k in (32, 128, 512)], [row[f"base_K{k}"] for k in (32, 128, 512)]

    def keq(c):
        i = 0 if c < ys[1] else 1
        return math.exp(xs[i] + (c - ys[i]) / (ys[i + 1] - ys[i]) * (xs[i + 1] - xs[i]))
    for name in ("oracle8", "guided8", "combo"):
        row[name + "_gain"] = round(keq(row[name]) / ((row[name + "_cost"] - 1) / ((row["base_cost"] - 1) / 32)), 2)
    rk = row["rank"]
    avg = lambda pre: round(sum(v for k, v in rk.items() if k.rstrip("0123456789") == pre)
                            / max(1, sum(1 for k in rk if k.rstrip("0123456789") == pre)), 1)
    row["rank_avg"] = {p: avg(p) for p in ("o", "proj", "out", "fc", "qkv", "emb")}
    if True:
        print(f"d={d:4d} rank(writers proj/out, hub o, hidden fc) {row['rank_avg']}  base cos K32/128/512 "
              f"{row['base_K32']:.3f}/{row['base_K128']:.3f}/{row['base_K512']:.3f}  oracle8 {row['oracle8']:.3f} "
              f"(gain {row['oracle8_gain']})  guided8 {row['guided8']:.3f} (gain {row['guided8_gain']})  combo "
              f"{row['combo']:.3f} (gain {row['combo_gain']})", flush=True)


if __name__ == "__main__":
    dev = "cuda"
    {"arms": arms, "width": width}[sys.argv[1]](dev)
