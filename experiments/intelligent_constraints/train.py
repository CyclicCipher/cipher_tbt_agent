"""Train one configuration on the synthetic facts and log full curves.

Usage:  python train.py --mlp dagger --m 128 --seed 0 --json runs/e1/dagger_w128_tied_s0.json
Refuses to run if the --json file exists (never re-run an unchanged experiment).
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

from data import make_facts
from model import Net

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(net, F, names=("fwd", "rev_paired", "rev_oneway", "rev_oneway_clean")):
    net.eval()
    out = {}
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
        for name in names:
            qs = F.evals[name]
            logits = net(torch.tensor([p for p, _, _ in qs], device=DEV)).float()[:, :F.n_ent]
            prob = logits.softmax(-1).cpu().numpy()
            top = prob.argmax(-1)
            hits = mass = leak = 0.0
            for i, (_, correct, lk) in enumerate(qs):
                hits += top[i] in correct
                mass += prob[i, correct].sum()
                leak += prob[i, lk].sum() if lk else 0.0
            n = len(qs)
            out[name] = dict(hits=round(float(hits) / n, 4), mass=round(float(mass) / n, 4), leak=round(float(leak) / n, 4))
    net.train()
    return out


def run(mlp, m, tied, seed, steps=6000, lr=1e-3, batch=512, eval_every=250, d=128, n_layer=2, frozen=False):
    F = make_facts(seed=0)                       # the facts are fixed; the seed varies the model and batch order
    torch.manual_seed(seed)
    net = Net(F.vocab, d=d, n_layer=n_layer, m=m, dagger=(mlp == "dagger"), tied=tied, frozen=frozen).to(DEV)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    X = torch.tensor([p for p, _ in F.train], device=DEV)
    Y = torch.tensor([a for _, a in F.train], device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    n_train_q = len(F.evals["fwd"]) + len(F.evals["rev_paired"])
    log, reach99, t0 = [], None, time.time()
    for step in range(1, steps + 1):
        for grp in opt.param_groups:
            grp["lr"] = lr * min(1.0, step / 200)
        i = torch.randint(0, len(X), (batch,), device=DEV, generator=g)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            loss = Fn.cross_entropy(net(X[i]).float(), Y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            e = evaluate(net, F)
            train_acc = (e["fwd"]["hits"] * len(F.evals["fwd"]) +
                         e["rev_paired"]["hits"] * len(F.evals["rev_paired"])) / n_train_q
            if reach99 is None and train_acc >= 0.99:
                reach99 = step
            log.append(dict(step=step, loss=round(loss.item(), 4), train_acc=round(train_acc, 4), **e))
    res = dict(mlp=mlp, m=m, tied=tied, frozen=frozen, seed=seed, steps=steps, lr=lr, batch=batch, d=d, n_layer=n_layer,
               params=sum(p.numel() for p in net.parameters()), log=log, final=log[-1], reach99=reach99,
               chance=round(sum(len(c) for _, c, _ in F.evals["rev_oneway"]) / len(F.evals["rev_oneway"]) / F.n_ent, 5))
    if mlp == "dagger":                          # ablation: the transpose read switched off at test time
        net.set_rev(False)
        res["final_rev_off"] = evaluate(net, F)
        net.set_rev(True)
    res["secs"] = round(time.time() - t0, 1)
    return res


def save(res, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp", choices=["plain", "dagger"], default="plain")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--untied", action="store_true")
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.json and Path(a.json).exists():
        print(f"{a.json} exists: not re-run")
        raise SystemExit
    r = run(a.mlp, a.m, not a.untied, a.seed, steps=a.steps, frozen=a.frozen)
    print(json.dumps(dict(final=r["final"], rev_off=r.get("final_rev_off"), reach99=r["reach99"], secs=r["secs"])))
    if a.json:
        save(r, a.json)
