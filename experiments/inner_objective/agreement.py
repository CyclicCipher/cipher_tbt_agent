"""Where do task families agree? For saved snapshots of a run, compute one gradient per (family, half-batch) and report
the across-family / within-family variance ratio per weight coordinate: the fraction of coordinates the F-test keeps at
each threshold c, and which parameter tensors hold the agreed coordinates. Diagnoses why a mask keeps or drops what it
does, with no training.

    python experiments/inner_objective/agreement.py --run runs_affine/A --steps 0 550 1000 1800   (seconds, GPU)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "transformers"))
from h1_lid import Model                                                    # noqa: E402
import tasks as TASKS                                                       # noqa: E402
from train import Flat, GroupGrads, env_batch, group_grads, L               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 550, 1000, 1800])
    ap.add_argument("--per_env", type=int, default=64, help="sequences per family (two halves); more = less within-noise")
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    r = Path(args.run)
    cfg = json.load(open(r / "config.json"))
    tr = torch.load(r / "trace.pt", weights_only=False)
    task = TASKS.make(cfg["task"], cfg["seed"])
    V, K = task.V, cfg["k"]
    model = Model(d_model=cfg["d"], n_layer=cfg["layers"], n_head=cfg["heads"], max_len=K * 2 * L + 2, pos="rope",
                  n_vocab=V).to(dev)
    flat = Flat(model)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda"))
    engine = GroupGrads(model, flat, K, dev, amp, V)
    layout = flat.layout()
    snap_steps = tr["snap_steps"]
    g = torch.Generator(device=dev).manual_seed(7)
    E, half = len(task.train), args.per_env // 2
    cs = [1, 3, 10, 30, 100, 1000]
    out = {}
    print(f"{cfg['arm']} on {task.name}: families {E}, {args.per_env} sequences per family per rep, {args.reps} reps")
    print(f"{'step':>5} | kept fraction at c = " + " ".join(f"{c:>6}" for c in cs) + " | ratio median  p10  p90")
    for s in args.steps:
        i = int(np.argmin([abs(x - s) for x in snap_steps]))
        flat.assign(tr["snaps"][i].to(dev).float())
        acc_a, acc_w = torch.zeros(flat.P, device=dev), torch.zeros(flat.P, device=dev)
        for _ in range(args.reps):
            toks = env_batch(task, task.train, 0, args.per_env, K, dev, g).view(E, 2, half, -1)
            G = group_grads(model, flat, toks, K, amp, engine, V)
            a, b = G[:, 0], G[:, 1]
            gm = 0.5 * (a + b)
            acc_w += 0.25 * ((a - b) ** 2).mean(0)
            acc_a += gm.var(0, unbiased=False)
        within, across = acc_w / args.reps, acc_a / args.reps
        ratio = (across / (within + 1e-12)).cpu()
        kept = {c: float((ratio <= c).float().mean()) for c in cs}
        q = torch.quantile(ratio.float(), torch.tensor([0.5, 0.1, 0.9]))
        per_tensor = {}
        for ent in layout:
            seg = ratio[ent["start"]:ent["end"]]
            per_tensor[ent["name"]] = float((seg <= 3).float().mean())
        top = sorted(per_tensor.items(), key=lambda kv: -kv[1])[:4]
        print(f"{snap_steps[i]:>5} | " + " ".join(f"{kept[c]:6.3f}" for c in cs) +
              f" | {q[0]:.1f} {q[1]:.1f} {q[2]:.1f}   most-agreed tensors (kept at c=3): " +
              ", ".join(f"{n} {v:.2f}" for n, v in top))
        out[snap_steps[i]] = dict(kept=kept, quantiles=q.tolist(), per_tensor=per_tensor)
    json.dump(out, open(r / "agreement.json", "w"), indent=1)


if __name__ == "__main__":
    main()
