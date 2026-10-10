"""A fast proxy for an estimator's training value, without training (PROXY.md): curvature-weighted progress at the best
step size, for plain and Adam-like scaled updates, at a checkpoint; plus the batch-level cosine and error persistence.

Usage:  python proxy.py <checkpoint> <arm> [<arm> ...]      (arms from ARMS below; runs on the CPU by default)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

import data
from e1_m1 import fit_aux
from estimators import estimate
from model import TinyGPT
from train import parse_cfg

OUT = Path("runs/proxy")
COMBO = ["noise=guided", "rank=8", "beta=0.5", "sparse_c=2", "top_guide=0.3", "hub_T=1", "exact_head=1"]
SITES = ("emb", "o", "proj", "fc", "out")
ARMS = {  # name -> (cfg overrides, aux_every)
    "bp": (None, 0),
    "base": ([], 0), "lr_rand": (["noise=lr_rand", "rank=8"], 0), "lr_pca": (["noise=lr_pca", "rank=8"], 0),
    "orth": (["noise=orth"], 0), "sobol": (["noise=sobol"], 0), "anti": (["noise=anti"], 0),
    "local128": (["local=1", "K=128"], 1), "ent50": (["ent_frac=0.5"], 0),
    "guided": (["noise=guided", "rank=8", "beta=0.5"], 0),
    "O1_sparse": (["sparse_c=2"], 0), "O3_top": (["top_guide=0.3", "exact_head=1"], 0),
    "O4_hubT_K56": (["hub_T=1", "K=56"], 0), "combo_K56": (COMBO + ["K=56"], 0),
    "guided_w": (["noise=guided", "rank=8", "beta=0.5", "whiten=1"], 0),
    "top_w": (["top_guide=0.3", "exact_head=1", "whiten=1"], 0), "combo_w_K56": (COMBO + ["K=56", "whiten=1"], 0),
    "O4_orth_K56": (["hub_T=1", "K=56", "noise=orth"], 0),
    # Experiment 3 (thermodynamics)
    "T1_sig01": ([f"sig.{s}=0.1" for s in SITES], 0), "T1_sig04": ([f"sig.{s}=0.4" for s in SITES], 0),
    "T2_boltz1": (["shaping=boltz", "lam=1.0"], 0), "T2_boltz03": (["shaping=boltz", "lam=0.3"], 0),
    "T2_rank": (["shaping=rank"], 0), "T3_simul_s02": (["simul=1", "K=286"], 0),
    "T3_simul_s01": (["simul=1", "K=286", "sig.proj=0.1"], 0),
}


def main_params(m):
    return [m.wte, m.wpe, m.Whead] + [p for b in range(m.L) for p in (m.Wqkv[b], m.Wproj[b], m.Wfc[b], m.Wout[b])]


def flat(d, params):
    return torch.cat([(d[p] if d.get(p) is not None else torch.zeros_like(p)).reshape(-1).float() for p in params])


def grad_of(m, params, batches):
    out = []
    for idx, tgt in batches:
        loss = m.forward_cache(idx, tgt)["loss"].mean()
        out.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, params)]))
    return torch.stack(out)


def make_hvp(m, params, batches):
    sizes = [p.numel() for p in params]

    def hvp(v):
        acc = torch.zeros_like(v)
        for idx, tgt in batches:
            loss = m.forward_cache(idx, tgt)["loss"].mean()
            gr = torch.autograd.grad(loss, params, create_graph=True)
            vs = torch.split(v, sizes)
            gv = sum((a * b.view_as(a)).sum() for a, b in zip(gr, vs))
            acc += torch.cat([h.reshape(-1) for h in torch.autograd.grad(gv, params)])
        return acc / len(batches)
    return hvp


def progress(g, U, hvp):
    """Best-step decrease of a quadratic model for updates U (n, P): (gᵀū)² / 2(ūᵀHū + tr(HΣ))."""
    ub = U.mean(0)
    dev = U - ub
    uHu = float(ub @ hvp(ub))
    trHS = float(sum(d @ hvp(d) for d in dev)) / (len(U) - 1)
    den = uHu + trHS
    return dict(P=float((g @ ub) ** 2 / (2 * den)) if den > 0 else float("nan"), align=float(g @ ub),
                uHu=uHu, trHS=trHS)


def run_arm(name, ck, dev, n=8, warm=5):
    sets, aux = ARMS[name]
    tr, va = data.load(dev)
    m = TinyGPT(data.V, aux_every=aux).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev), strict=False)
    gt = torch.Generator(device=dev).manual_seed(7)
    if aux:
        fit_aux(m, lambda: data.batch(tr, 32, 64, gt))
    params = main_params(m)
    gb = torch.Generator(device=dev).manual_seed(321)
    train_b = [data.batch(tr, 32, 64, gb) for _ in range(n + warm)]
    gv = torch.Generator(device=dev).manual_seed(654)
    val_b = [data.batch(va, 32, 64, gv) for _ in range(8)]
    g = grad_of(m, params, val_b).mean(0)
    gtrue = grad_of(m, params, train_b[warm:])                        # (n, P) per-batch backprop gradients
    hvp = make_hvp(m, params, val_b[:2])
    t0 = time.time()
    if sets is None:
        U = gtrue.clone()
        cost = 3.0
    else:
        cfg = parse_cfg(["K=32"] + (["amp=fp16"] if dev == "cuda" else []) + sets)
        state, est = {}, []
        g_rng = torch.Generator(device=dev).manual_seed(0)
        stateful = cfg.noise == "guided"                              # only the guided subspace carries state
        for idx, tgt in (train_b if stateful else train_b[warm:]):
            gr, st = estimate(m, idx, tgt, cfg, state, g_rng)
            est.append(flat(gr, params))
        U = torch.stack(est[-n:])
        cost = st["cost"]
    secs_est = time.time() - t0
    cos = float(torch.nn.functional.cosine_similarity(U, gtrue, dim=1).mean())
    err = U - gtrue
    persist = float(torch.nn.functional.cosine_similarity(err[:-1], err[1:], dim=1).mean()) if sets else 0.0
    D = 1.0 / (U.pow(2).mean(0).sqrt() + 1e-12)
    r = dict(arm=name, ckpt=ck, cost=cost, cos=round(cos, 4), persist=round(persist, 4),
             sgd=progress(g, U, hvp), adam=progress(g, U * D, hvp), secs_estimates=round(secs_est, 1),
             secs_total=round(time.time() - t0, 1))
    return r


def main():
    torch.set_num_threads(int(os.environ.get("PROXY_THREADS", "6")))
    dev = os.environ.get("PROXY_DEV", "cpu")
    ck = sys.argv[1]
    tag = Path(ck).stem
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[2:]:
        f = OUT / f"{tag}_{name}.json"
        if f.exists():                                                 # never re-run an unchanged evaluation
            print(f"{name}: exists, skipped")
            continue
        r = run_arm(name, ck, dev)
        json.dump(r, open(f, "w"), indent=1)
        print(f"{tag} {name:14s} cos {r['cos']:.3f}  persist {r['persist']:+.3f}  P_sgd {r['sgd']['P']:.3e}  "
              f"P_adam {r['adam']['P']:.3e}  ({r['secs_total']}s)", flush=True)


if __name__ == "__main__":
    main()
