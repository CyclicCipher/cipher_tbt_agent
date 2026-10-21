"""E8, option B: rules inferred in context (amortized inference in activations). (EXPERIMENT_E8.md)

Training: next-observation prediction across many worlds, each world's whole 128-step history in one pass; the
prediction for step t sees only the world's steps before t (incontext.ContextModel). No world-specific weights.
Evaluation: identical to option A's (e8_hier): bits per step and per-rule accuracy against the chunk index (8 steps),
i.e. against experience in the world, on unseen worlds and on held-out parameter combinations.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from e2_world import load, losses, robot_mask
from e8_hier import RULES, summarize, world_index
from e8_score import ChunkScorer
from incontext import ContextModel
from village import SPEC

LN2 = math.log(2)


def world_tensors(z, worlds, dev):
    return [torch.from_numpy(z[x][worlds]).to(dev) for x in ("obs", "gob", "act", "nxt", "gnx")]


def forward_worlds(model, z, worlds, dev):
    o, g, a, nc, ng = world_tensors(z, worlds, dev)
    W, L = a.shape
    flat = lambda x: x.reshape(W * L, *x.shape[2:])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ctx = model.contexts(o, g, a, nc, ng)
        cl, gl = model.predict(flat(o), flat(g), flat(a), flat(ctx))
    return cl, gl, flat(o), flat(nc), flat(ng)


def evaluate(model, z, worlds_all, dev, batch=4, chunk=8):
    T = worlds_all.shape[1] // chunk
    sc = ChunkScorer(z, T)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(worlds_all), batch):
            w = worlds_all[i:i + batch]
            cl, gl, o, nc, ng = forward_worlds(model, z, w, dev)
            _, sur = losses(cl, gl, nc, ng, per_row=True, mask=robot_mask(o, 2))
            W, L = w.shape
            idx = torch.arange(W * L, device=dev).view(W, L)
            for t in range(T):
                sel = idx[:, t * chunk:(t + 1) * chunk].reshape(-1)
                sc.add(t, [x[sel] for x in cl], [x[sel] for x in gl], w[:, t * chunk:(t + 1) * chunk].reshape(-1), sur[sel])
    model.train()
    return sc.result()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--worlds-per-batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--time-limit", type=float, default=420.0)
    p.add_argument("--eval-worlds", type=int, default=128)
    p.add_argument("--data", default="runs/data_family")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    train, test, held = (load(os.path.join(a.data, f)) for f in ("train.npz", "test.npz", "heldout.npz"))
    wtr, wte, whe = world_index(train), world_index(test)[:a.eval_worlds], world_index(held)[:a.eval_worlds]
    model = ContextModel(SPEC).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    json.dump(dict(vars(a), n_params=sum(q.numel() for q in model.parameters())), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    print(f"worlds: train {len(wtr)}, test {len(wte)}, held-out {len(whe)}", flush=True)

    def do_eval(tag):
        for name, z, w in (("test", test, wte), ("heldout", held, whe)):
            r = evaluate(model, z, w, dev)
            s = summarize(r)
            log.write(json.dumps(dict(type="eval", tag=tag, split=name, adapt="context", summary=s, curves=r)) + "\n")
            log.flush()
            print(f"[{tag}] {name:7s} context  bits early {s['bits_early']:.2f} late {s['bits_late']:.2f}  "
                  f"move {s['move'][0]:.2f}->{s['move'][1]:.2f}  turn {s['turn'][0]:.2f}->{s['turn'][1]:.2f}  "
                  f"burn {s['robot_burn'][0]}->{s['robot_burn'][1]}  push {s['push'][0]}->{s['push'][1]}", flush=True)

    do_eval("start")
    t_train, step, half = 0.0, 0, False
    while t_train < a.time_limit:
        t0 = time.time()
        idx = rng.choice(len(wtr), a.worlds_per_batch, replace=False)
        cl, gl, o, nc, ng = forward_worlds(model, train, wtr[idx], dev)
        loss, sur = losses(cl, gl, nc, ng, per_row=True, mask=robot_mask(o, 2))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        torch.cuda.synchronize()
        t_train += time.time() - t0
        if step % 50 == 0:
            bits = float(sur.mean()) / LN2
            log.write(json.dumps(dict(type="train", step=step, t=t_train, bits=bits)) + "\n")
            print(f"step {step}  {t_train:.0f}s  train bits/step {bits:.2f}", flush=True)
        if not half and t_train > a.time_limit / 2:
            do_eval("mid")
            half = True
    do_eval("end")
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))


if __name__ == "__main__":
    main()
