"""E6b — the gradient comparison arm for E6: the run-2 transformer trained on the same sequence of rules, by backprop.

The same three rules as E6 (A = shift 3, B = shift 5, C = 2x+1, then A again), each for a stretch of training steps on
that rule alone, in the in-context format the transformer uses (8 demonstrations of 6 digits, predict the outputs).
After every stretch, every rule seen so far is tested: per-digit and whole-answer accuracy on the LAST demonstration
of fresh sequences (the rule's own 7 earlier demonstrations are in the context, so the test gives the network the same
help E6 gives ZipLearner). Forgetting is the drop on a rule after training on the others.

    python experiments/ziplearn/e6b.py           (~2 min on the GPU) -> runs/e6b/e6b.json
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
RULES = {"A": (1, 3), "B": (1, 5), "C": (2, 1)}


@torch.no_grad()
def test(model, task, K, dev, g, n=256):
    m = out_mask(K, dev)[1:]
    out = {}
    for name, rule in RULES.items():
        tok = env_batch(task, [rule], 0, n, K, dev, g)[0]
        logits = model(tok)[:, :-1]
        tgt = tok[:, 1:]
        lg, tg = logits[:, m].reshape(n, K, L, task.V), tgt[:, m].reshape(n, K, L)
        pred = lg[:, -1].argmax(-1)
        out[name] = dict(digit=float((pred == tg[:, -1]).float().mean()), exact=float((pred == tg[:, -1]).all(-1).float().mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400, help="training steps per stretch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e6b"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = TASKS.make("affine", args.seed)
    K, V = args.k, task.V
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Model(d_model=64, n_layer=3, n_head=4, max_len=K * 2 * L + 2, pos="rope", n_vocab=V).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98))
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda"))
    g = torch.Generator(device=dev).manual_seed(args.seed)
    ge = torch.Generator(device=dev).manual_seed(args.seed + 1)
    order = ["A", "B", "C", "A"]
    m = out_mask(K, dev)[1:]
    print(f"E6b: transformer ({sum(p.numel() for p in model.parameters()):,} weights) trained {args.steps} steps per stretch on "
          f"{' -> '.join(order)}, batch {args.batch}, in-context K={K}\n")
    print(f"{'after stretch':<16}{'A digit/exact':>16}{'B':>14}{'C':>14}")
    log = []
    t0 = time.time()
    for i, name in enumerate(order):
        rule = RULES[name]
        for step in range(args.steps):
            tok = env_batch(task, [rule], 0, args.batch, K, dev, g)[0]
            with amp:
                logits = model(tok)[:, :-1]
            loss = F.cross_entropy(logits[:, m].reshape(-1, V).float(), tok[:, 1:][:, m].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        r = test(model, task, K, dev, ge)
        seen = order[:i + 1]
        log.append(dict(stretch=i + 1, rule=name, results=r, loss=float(loss), seen=seen))
        cells = "".join(f"{(r[k]['digit'], r[k]['exact'])[0]:>9.2f}/{r[k]['exact']:<5.2f}" if k in seen else f"{'':>15}" for k in RULES)
        print(f"{i + 1} ({name}){'':<10}{cells}   loss {float(loss):.3f}  {time.time() - t0:.0f}s")
    # forgetting: for each rule, accuracy right after its own stretch vs at the end
    final = log[-1]["results"]
    forgetting = {}
    for k in RULES:
        own = next(e["results"][k] for e in log if e["rule"] == k)
        forgetting[k] = dict(after_own=own, at_end=final[k], drop_digit=own["digit"] - final[k]["digit"])
    print("\nforgetting (per-digit accuracy right after the rule's own stretch -> at the end): " +
          ", ".join(f"{k}: {v['after_own']['digit']:.2f} -> {v['at_end']['digit']:.2f}" for k, v in forgetting.items()))
    json.dump(dict(log=log, forgetting=forgetting), open(out / "e6b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
