"""E7: Bayesian R3 on the one-refutation test (arm K's setup). (EXPERIMENT_E7.md)

From J2's world model (it never learned button_off), learn ONE (or k) button_off instance(s) by the MAP of
[J2's Laplace posterior x the instances' likelihood]: minimize l_new + 1/2 sum Lambda (theta - mu)^2, the instances
counted once, optimized for --iters steps. No replay, no interleaving, no retirement schedule: the prior does it.
Logged every 25 iterations: held-out button_off, the rehearsed case(s), the other rules (forgetting), changed /
unchanged-entry accuracy, and the Bayesian surprise of the update (nats).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from bayes import LaplacePrior, per_sample_fisher
from e2_world import eval_set, evaluate, load, losses, robot_mask
from village import SPEC
from village_events import EVENTS
from worldmodel import WorldModel

OTHERS = ("move", "turn", "mine_work", "mined", "deliver", "theft", "interact_full", "push", "robot_burn",
          "trapped_hurt", "frozen_tick")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/e2/J2_focus_questions")
    p.add_argument("--rule", default="button_off")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-data", type=float, default=614400, help="data the old posterior summarises (each counted once)")
    p.add_argument("--lam0", type=float, default=1e-2)
    p.add_argument("--fisher-samples", type=int, default=2048)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="runs/data")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    dev = torch.device("cuda")
    rng = np.random.default_rng(a.seed)
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    focus = cfg.get("focus", 2)
    train, test = load(os.path.join(a.data, "train.npz")), load(os.path.join(a.data, "test.npz"))
    ev, by_event = eval_set(test)
    model = WorldModel(SPEC, cfg["d"], cfg["layers"], cfg["heads"], pos=cfg["pos"], persist=cfg["persist"],
                       radius=cfg.get("radius", 1)).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev))

    def rows_tensors(rows):
        return [torch.from_numpy(train[x][rows]).to(dev) for x in ("obs", "gob", "act", "nxt", "gnx")]

    def loss_fn(m, rows):
        o, g, act, nx, gx = rows_tensors(rows)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cl, gl, _ = m(o, g, act)
        return losses(cl, gl, nx, gx, mask=robot_mask(o, focus))

    t0 = time.time()
    fisher_rows = rng.choice(len(train["act"]), a.fisher_samples, replace=False)
    fisher = per_sample_fisher(model, loss_fn, ([int(r)] for r in fisher_rows))
    prior = LaplacePrior(model, fisher, a.n_data, a.lam0)
    print(f"Fisher from {a.fisher_samples} single samples in {time.time() - t0:.0f}s", flush=True)

    k_rule = EVENTS.index(a.rule)
    chosen = rng.permutation(np.nonzero((train["events"] >> k_rule) & 1)[0])[:a.k]
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    log = open(a.out, "w")
    for it in range(a.iters + 1):
        if it % 25 == 0:
            r = evaluate(model, ev, by_event, dev)
            tr = evaluate(model, {x: train[x][chosen] for x in ev}, {a.rule: (0, len(chosen)), "_random": (0, len(chosen))}, dev)
            row = dict(iter=it, acc=r[a.rule]["acc"], case_acc=tr[a.rule]["acc"], case_bits=tr[a.rule]["bits"],
                       changed=r["_overall"]["changed_acc"], unchanged=r["_overall"]["unchanged_acc"],
                       surprise_nats=prior.bayesian_surprise(model), others={e: r[e]["acc"] for e in OTHERS})
            log.write(json.dumps(row) + "\n")
            log.flush()
            if it % 100 == 0:
                print(f"iter {it:4d}  held-out {row['acc']:.2f}  case {row['case_acc']:.2f}  theft {row['others']['theft']:.2f}"
                      f"  mined {row['others']['mined']:.2f}  changed {row['changed']:.3f}  unchanged {row['unchanged']:.3f}"
                      f"  bayes-surprise {row['surprise_nats']:.2f} nats", flush=True)
        if it == a.iters:
            break
        model.train()
        loss = loss_fn(model, chosen) * len(chosen) + prior.penalty(model)     # the instances, each counted once
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


if __name__ == "__main__":
    main()
