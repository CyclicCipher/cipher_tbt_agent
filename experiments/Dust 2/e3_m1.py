"""Experiment 3, M1: gradient quality of the thermodynamic arms (EXPERIMENT_3.md) at Experiment 1's checkpoints, as gain
over Experiment 1's baseline curve (reused). Usage: python e3_m1.py [arm ...]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import data
import recall
from e1_m1 import KS, k_eq
from measure import measure
from model import TinyGPT
from train import parse_cfg

OUT = Path("runs/e3/m1")
SITES = ("emb", "o", "proj", "fc", "out")
sig = lambda v: [f"sig.{s}={v}" for s in SITES]
ARMS = [
    ("T1_sig01", sig(0.1)),
    ("T1_sig04", sig(0.4)),
    ("T2_boltz1", ["shaping=boltz", "lam=1.0"]),
    ("T2_boltz03", ["shaping=boltz", "lam=0.3"]),
    ("T2_rank", ["shaping=rank"]),
    ("T3_simul_s02", ["simul=1", "K=286"]),
    ("T3_simul_s01", ["simul=1", "K=286", "sig.proj=0.1"]),
]


def main():
    dev = "cuda"
    TinyGPT.fast = True
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va = data.load(dev)
    for task, ck, V in [("latin", "runs/ckpt_bp300.pt", data.V), ("recall", "runs/ckpt_recall_bp4000.pt", recall.V)]:
        curve = {k: json.load(open(f"runs/e1/curve/{task}_K{k}.json"))["mean"] for k in KS}
        gb = torch.Generator(device=dev).manual_seed(123)
        batches = [data.batch(va, 32, 64, gb) if task == "latin" else recall.batch(32, 64, gb) for _ in range(4)]
        for name, sets in ARMS:
            if len(sys.argv) > 1 and name not in sys.argv[1:]:
                continue
            model = TinyGPT(V).to(dev)
            model.load_state_dict(torch.load(ck, map_location=dev))
            r = measure(model, parse_cfg(["K=32", "amp=fp16"] + sets), batches, 0, dev)
            kmatch = (r["cost"] - 1) / ((287.0 - 1) / 32)
            ke = k_eq(r["mean"], curve)
            r.update(arm=name, task=task, k_eq=round(ke, 1), gain=round(ke / kmatch, 2), k_match=round(kmatch, 1))
            json.dump(r, open(OUT / f"{task}_{name}.json", "w"), indent=1)
            c = r["cos"]
            print(f"{task:6s} {name:14s} mean cos {r['mean']:.3f}  gain {ke / kmatch:5.2f}  cost {r['cost']:6.1f}  | "
                  f"fc0 {c['fc0']:.3f} out0 {c['out0']:.3f} proj0 {c['proj0']:.3f} qkv0 {c['qkv0']:.3f} "
                  f"wte {c['wte']:.3f} head {c['head']:.3f}", flush=True)


if __name__ == "__main__":
    main()
