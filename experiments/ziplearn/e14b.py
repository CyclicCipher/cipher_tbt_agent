"""E14b — the gradient arm of E14: the run-2 transformer trained on the 25 compositions in different orders.

Sequential training: each task in turn for `--steps` steps on that task alone (batch 32, in-context K = 8), the
same seed and demonstrations for every order. The PREQUENTIAL total -- the sum over all training steps of the loss in
bits per output digit -- is the network's price for the stream, the analogue of E14's total bits; the final
per-digit bits on all 25 tasks is the analogue of the frozen re-pass. Orders: parts first, wholes first, two random.

    python experiments/ziplearn/e14b.py          (~4 min on the GPU) -> runs/e14/e14b.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inner_objective"))
sys.path.insert(0, str(HERE.parent / "transformers"))
from h1_lid import Model, out_mask                     # noqa: E402
import tasks as TASKS                                  # noqa: E402
from train import env_batch                            # noqa: E402

L = TASKS.L


@torch.no_grad()
def bits_per_digit(model, task, pairs, K, dev, g, n=128):
    m = out_mask(K, dev)[1:]
    tot = 0.0
    for pair in pairs:
        tok = env_batch(task, [pair], 0, n, K, dev, g)[0]
        logits = model(tok)[:, :-1]
        tgt = tok[:, 1:]
        lg, tg = logits[:, m].reshape(n, K, L, task.V), tgt[:, m].reshape(n, K, L)
        tot += float(F.cross_entropy(lg[:, -1].reshape(-1, task.V).float(), tg[:, -1].reshape(-1)) / math.log(2))
    return tot / len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e14"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    task = TASKS.make("compose", args.seed)
    tasks = task.train + task.test
    K, V = args.k, task.V
    value_prims = {5, 6}
    parts = [p for p in tasks if (p[0] in value_prims) or (p[1] in value_prims)]
    wholes = [p for p in tasks if p not in parts]
    rng = np.random.default_rng(args.seed)
    orders = {"parts first": parts + wholes, "wholes first": wholes + parts,
              "random 0": [tasks[j] for j in rng.permutation(len(tasks))], "random 1": [tasks[j] for j in rng.permutation(len(tasks))]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = out_mask(K, dev)[1:]
    res = {}
    print(f"E14b: transformer, {args.steps} steps per task, {len(tasks)} tasks in sequence\n")
    print(f"{'order':<14}{'prequential bits/digit (sum)':>30}{'final bits/digit, all 25':>26}")
    for name, order in orders.items():
        torch.manual_seed(args.seed)
        model = Model(d_model=64, n_layer=3, n_head=4, max_len=K * 2 * L + 2, pos="rope", n_vocab=V).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98))
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda"))
        g = torch.Generator(device=dev).manual_seed(args.seed + 1)
        preq = 0.0
        t0 = time.time()
        for pair in order:
            for _ in range(args.steps):
                tok = env_batch(task, [pair], 0, args.batch, K, dev, g)[0]
                with amp:
                    logits = model(tok)[:, :-1]
                loss = F.cross_entropy(logits[:, m].reshape(-1, V).float(), tok[:, 1:][:, m].reshape(-1))
                preq += float(loss) / math.log(2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        ge = torch.Generator(device=dev).manual_seed(args.seed + 2)
        final = bits_per_digit(model, task, tasks, K, dev, ge)
        res[name] = dict(prequential=preq, final_bits=final, seconds=time.time() - t0)
        print(f"{name:<14}{preq:>30.1f}{final:>26.3f}   ({time.time() - t0:.0f}s)")
    vals = [r["prequential"] for r in res.values()]
    print(f"\ncurriculum gap (worst − best prequential): {max(vals) - min(vals):.1f} = {(max(vals) - min(vals)) / min(vals):.1%} of the best")
    json.dump(res, open(out / "e14b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
