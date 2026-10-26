"""E2 arm K2: the learning rule in its clean form. Can a trained world model learn a new rule from ONE refutation, without
rationalizing (over-fitting the one case) and without forgetting the old rules? (EXPERIMENT_E2.md)

From J2's model and one (or four) training instances of a rule it never learned:
- the instance is rehearsed (8 copies per batch) while OPEN; it RETIRES once its in-scope surprise has halved, then
  joins the interleaved pool (stop rehearsing when explained);
- the other 24 rows per batch come from either
  "kinds": a pool of past refutations, sampled evenly across surprise kinds (label-free: signature()), or
  "stream": random rows of the training stream (K's setup).
- --always keeps rehearsing (no retirement): K's setup, the over-rehearsal control.
"""
import argparse
import json
import os

import numpy as np
import torch

from lab.e2.e2_world import eval_set, evaluate, load, losses, robot_mask, robot_pos
from nets.worldmodel import WorldModel
from worlds.village import SPEC
from worlds.village_events import EVENTS


def batch(train, rows, dev):
    return [torch.from_numpy(train[x][rows]).to(dev) for x in ("obs", "gob", "act", "nxt", "gnx")]


@torch.no_grad()
def surprise_and_kind(model, train, rows, dev, focus):
    s, k = [], []
    for i in range(0, len(rows), 1024):
        o, g, act, nx, gx = batch(train, rows[i:i + 1024], dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cl, gl, _ = model(o, g, act)
        _, tot, sig = losses(cl, gl, nx, gx, per_row="sig", mask=robot_mask(o, focus), robot_pos=robot_pos(o))
        s.append(tot.cpu().numpy())
        k.append(sig)
    return np.concatenate(s), np.concatenate(k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/e2/J2_focus_questions")
    p.add_argument("--rule", default="button_off")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--pool", default="kinds", choices=("kinds", "stream"))
    p.add_argument("--always", action="store_true")
    p.add_argument("--updates", type=int, default=600)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0, help="which instance(s) are chosen")
    p.add_argument("--data", default="runs/data")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    dev = torch.device("cuda")
    rng = np.random.default_rng(a.seed)
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    train, test = load(os.path.join(a.data, "train.npz")), load(os.path.join(a.data, "test.npz"))
    ev, by_event = eval_set(test)
    model = WorldModel(SPEC, cfg["d"], cfg["layers"], cfg["heads"], pos=cfg["pos"], persist=cfg["persist"],
                       radius=cfg.get("radius", 1)).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev))
    focus = cfg.get("focus", 2)

    # the pool of past refutations: surprising stream rows (above the median), equal numbers per kind
    cand = rng.choice(len(train["act"]), 60000, replace=False)
    s, k = surprise_and_kind(model, train, cand, dev, focus)
    cand, k = cand[s > np.median(s)], k[s > np.median(s)]
    per = max(1, 4096 // len(np.unique(k)))
    pool = np.concatenate([rng.permutation(cand[k == kd])[:per] for kd in np.unique(k)])
    k_rule = EVENTS.index(a.rule)
    chosen = rng.permutation(np.nonzero((train["events"] >> k_rule) & 1)[0])[:a.k]
    s0, _ = surprise_and_kind(model, train, chosen, dev, focus)
    open_ = np.ones(len(chosen), bool)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    log = open(a.out, "w")
    for u in range(a.updates + 1):
        if u % 50 == 0:
            r = evaluate(model, ev, by_event, dev)
            row = dict(update=u, rule=a.rule, acc=r[a.rule]["acc"], open=int(open_.sum()),
                       changed_acc=r["_overall"]["changed_acc"],
                       others={e: r[e]["acc"] for e in ("move", "turn", "theft", "mined", "deliver", "interact_full",
                                                         "robot_burn") if e != a.rule})
            log.write(json.dumps(row) + "\n")
            log.flush()
        if u == a.updates:
            break
        live = chosen[open_] if not a.always else chosen
        reh = rng.choice(live, 8) if len(live) else np.zeros(0, int)
        bg = rng.choice(pool, 32 - len(reh)) if a.pool == "kinds" else rng.integers(0, len(train["act"]), 32 - len(reh))
        if not a.always and not open_.all():                 # retired instances join the interleaved pool
            bg[:2] = rng.choice(chosen[~open_], 2)
        rows = np.concatenate([reh, bg]).astype(int)
        o, g, act, nx, gx = batch(train, rows, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cl, gl, _ = model(o, g, act)
        loss, tot = losses(cl, gl, nx, gx, per_row=True, mask=robot_mask(o, focus))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not a.always and len(reh):                        # retire an instance once its surprise has halved
            st = tot[:len(reh)].cpu().numpy()
            for row_id, sv in zip(reh, st):
                j = np.nonzero(chosen == row_id)[0][0]
                if sv < 0.5 * s0[j]:
                    open_[j] = False
    L = [json.loads(x) for x in open(a.out)]
    print(f"{os.path.basename(a.out)}: {a.rule} acc {[round(x['acc'], 2) for x in L]}  theft "
          f"{[round(x['others']['theft'], 2) for x in L][::3]}  changed {L[0]['changed_acc']:.3f}->{L[-1]['changed_acc']:.3f}"
          f"  open {[x['open'] for x in L][::3]}")


if __name__ == "__main__":
    main()
