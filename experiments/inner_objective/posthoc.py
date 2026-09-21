"""Graded measures computed after the run from each arm's saved model: held-out per-digit accuracy and held-out loss
(bits per output digit), which carry signal when whole-sequence exact match is censored at zero, plus learning speed
on the training families (steps to reach a training-accuracy threshold).

    python experiments/inner_objective/posthoc.py --runs runs/A runs/B runs/Ap runs/Bp
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "transformers"))
from h1_lid import Model, out_mask                        # noqa: E402
import tasks as TASKS                                     # noqa: E402
from train import env_batch                               # noqa: E402

L = TASKS.L


@torch.no_grad()
def graded(model, task, pairs, K, dev, g, n=256):
    """Per-digit accuracy and bits/digit on the LAST demonstration's output (query), averaged over tasks."""
    m = out_mask(K, dev)[1:]
    V = task.V
    acc, bits = [], []
    for pair in pairs:
        tok = env_batch(task, [tuple(pair)], 0, n, K, dev, g)[0]
        logits = model(tok)[:, :-1]
        tgt = tok[:, 1:]
        lg, tg = logits[:, m].reshape(n, K, L, V), tgt[:, m].reshape(n, K, L)
        last_lg, last_tg = lg[:, -1], tg[:, -1]
        acc.append(float((last_lg.argmax(-1) == last_tg).float().mean()))
        bits.append(float(F.cross_entropy(last_lg.reshape(-1, V).float(), last_tg.reshape(-1)) / math.log(2)))
    return float(np.mean(acc)), float(np.mean(bits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--thresholds", default="0.1,0.2,0.3")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ths = [float(x) for x in args.thresholds.split(",")]
    out = {}
    print(f"{'arm':4} | {'held digit acc':>14} {'held bits/digit':>15} | {'train digit acc':>15} {'train bits':>10} | steps to train acc " + " / ".join(f"{t:.1f}" for t in ths))
    for r in args.runs:
        r = Path(r)
        cfg = json.load(open(r / "config.json"))
        tr = torch.load(r / "trace.pt", weights_only=False)
        task = TASKS.make(cfg.get("task", "compose"), cfg["seed"])
        model = Model(d_model=cfg["d"], n_layer=cfg["layers"], n_head=cfg["heads"], max_len=cfg["k"] * 2 * L + 2,
                      pos="rope", n_vocab=cfg["V"]).to(dev)
        model.load_state_dict(tr["model"])
        model.eval()
        g = torch.Generator(device=dev).manual_seed(11)
        ha, hb = graded(model, task, cfg["test_pairs"], cfg["k"], dev, g)
        ta, tb = graded(model, task, cfg["train_pairs"], cfg["k"], dev, g)
        ev = [json.loads(l) for l in open(r / "eval.jsonl")]
        speed = {}
        for t in ths:
            hit = [e["step"] for e in ev if e["train_last_acc"] >= t]
            speed[t] = hit[0] if hit else None
        out[cfg["arm"]] = dict(held_digit_acc=ha, held_bits=hb, train_digit_acc=ta, train_bits=tb, steps_to=speed)
        print(f"{cfg['arm']:4} | {ha:14.3f} {hb:15.3f} | {ta:15.3f} {tb:10.3f} | " + " / ".join(str(speed[t]) for t in ths))
    json.dump(out, open(Path(args.runs[0]).parent / "results" / "posthoc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
