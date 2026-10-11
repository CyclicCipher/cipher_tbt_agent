"""Proxy version 2 (PROXY.md): a 25-step training probe from a checkpoint with the real optimiser; the proxy is the drop
in validation loss. Usage: python probe.py <checkpoint> <arm> [...]   (arms from proxy.ARMS; GPU by default)"""
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
from proxy import ARMS
from train import make_opt, parse_cfg, val_loss

OUT = Path("runs/probe")
STEPS, LR, WARM = 25, 1e-2, 5
OPT = dict(kind=os.environ.get("PROBE_OPT", "adam"), lr=float(os.environ.get("PROBE_LR", "1e-2")),
           emb_mult=float(os.environ.get("PROBE_EMB", "1")))


def probe(name, ck, dev):
    sets, aux = ARMS[name]
    tr, va = data.load(dev)
    torch.manual_seed(0)
    m = TinyGPT(data.V, aux_every=aux).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev), strict=False)
    if aux:
        gt = torch.Generator(device=dev).manual_seed(7)
        fit_aux(m, lambda: data.batch(tr, 32, 64, gt))
    gv = torch.Generator(device=dev).manual_seed(654)
    vb = [data.batch(va, 32, 64, gv) for _ in range(8)]
    gb = torch.Generator(device=dev).manual_seed(321)
    batches = [data.batch(tr, 32, 64, gb) for _ in range(STEPS)]
    v0 = val_loss(m, vb)
    opt = make_opt(m, OPT["kind"], OPT["lr"], 0.95, OPT["emb_mult"])
    cfg = parse_cfg(["K=32", "amp=fp16"] + sets) if sets is not None else None
    state, g = {}, torch.Generator(device=dev).manual_seed(0)
    t0 = time.time()
    for i, (idx, tgt) in enumerate(batches):
        for gp in opt.param_groups:
            gp["lr"] = OPT["lr"] * gp.get("mult", 1.0) * min(1.0, (i + 1) / WARM)
        if cfg is None:
            m.zero_grad(set_to_none=True)
            m.forward_cache(idx, tgt)["loss"].mean().backward()
        else:
            grads, _ = estimate(m, idx, tgt, cfg, state, g)
            for p in m.parameters():
                p.grad = grads.get(p)
        opt.step()
    v1 = val_loss(m, vb)
    return dict(arm=name, ckpt=ck, v0=round(v0, 4), v1=round(v1, 4), drop=round(v0 - v1, 4),
                secs=round(time.time() - t0, 1))


def main():
    TinyGPT.fast = True
    dev = os.environ.get("PROXY_DEV", "cuda")
    ck = sys.argv[1]
    tag = Path(ck).stem + ("" if OPT["kind"] == "adam" and OPT["lr"] == 1e-2 else
                           f"_{OPT['kind']}_lr{OPT['lr']:g}_emb{OPT['emb_mult']:g}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[2:]:
        f = OUT / f"{tag}_{name}.json"
        if f.exists():
            print(f"{name}: exists, skipped")
            continue
        r = probe(name, ck, dev)
        json.dump(r, open(f, "w"), indent=1)
        print(f"{tag} {name:14s} val {r['v0']:.4f} -> {r['v1']:.4f}  drop {r['drop']:+.4f}  ({r['secs']}s)", flush=True)


if __name__ == "__main__":
    main()
