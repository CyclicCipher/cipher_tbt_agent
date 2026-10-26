"""E2 arm K: few-shot capacity under unlimited rehearsal. How many DISTINCT refutations of a rule does a trained world
model need, if it may rehearse them as often as it likes? (EXPERIMENT_E2.md)

Start from a saved E2 model (J2), which never learned the rule. For k distinct training instances of the rule, fine-tune
with every batch = `k_per_batch` rows drawn (with replacement) from those k + the rest from the whole stream (so the
other rules are kept). Score the rule on held-out layouts, and the other rules, every --eval-every updates.

    python -m lab.e2.e2_fewshot --model runs/e2/J2_focus_questions --rule push --out runs/e2/K
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from lab.e2.e2_world import eval_set, evaluate, load, losses, robot_mask
from nets.worldmodel import WorldModel
from worlds.village import SPEC
from worlds.village_events import EVENTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rule", required=True)
    p.add_argument("--ks", default="1,4,16,64,256")
    p.add_argument("--updates", type=int, default=600)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--k-per-batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="runs/data")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    train, test = load(os.path.join(a.data, "train.npz")), load(os.path.join(a.data, "test.npz"))
    ev, by_event = eval_set(test)
    k_rule = EVENTS.index(a.rule)
    pool = np.nonzero((train["events"] >> k_rule) & 1)[0]
    rng = np.random.default_rng(0)
    pool = rng.permutation(pool)
    log = open(os.path.join(a.out, f"{a.rule}.jsonl"), "w")
    for k in [int(x) for x in a.ks.split(",")]:
        k = min(k, len(pool))
        chosen = pool[:k]
        model = WorldModel(SPEC, cfg["d"], cfg["layers"], cfg["heads"], pos=cfg["pos"], persist=cfg["persist"],
                           radius=cfg.get("radius", 1)).to(dev)
        model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev))
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        t0 = time.time()
        for u in range(a.updates + 1):
            if u % a.eval_every == 0:
                r = evaluate(model, ev, by_event, dev)
                tr = evaluate(model, {x: train[x][chosen] for x in ev}, {a.rule: (0, k), "_random": (0, k)}, dev)
                row = dict(rule=a.rule, k=k, update=u, acc=r[a.rule]["acc"], bits=r[a.rule]["bits"],
                           train_acc=tr[a.rule]["acc"], train_bits=tr[a.rule]["bits"],
                           changed_acc=r["_overall"]["changed_acc"],
                           others={e: r[e]["acc"] for e in ("move", "turn", "theft", "mined", "push", "robot_burn",
                                                             "button_off") if e in r and e != a.rule})
                log.write(json.dumps(row) + "\n")
                log.flush()
            if u == a.updates:
                break
            rows = np.concatenate([rng.choice(chosen, a.k_per_batch),
                                   rng.integers(0, len(train["act"]), a.batch - a.k_per_batch)])
            o, g, act, nx, gx = (torch.from_numpy(train[x][rows]).to(dev) for x in ("obs", "gob", "act", "nxt", "gnx"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cl, gl, _ = model(o, g, act)
            mask = robot_mask(o, cfg["focus"]) if cfg.get("focus", -1) >= 0 else None
            loss = losses(cl, gl, nx, gx, mask=mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        accs = [json.loads(x) for x in open(os.path.join(a.out, f"{a.rule}.jsonl")) if f'"k": {k},' in x]
        print(f"{a.rule} k={k:4d}: acc over updates {[round(x['acc'], 2) for x in accs]}  "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
