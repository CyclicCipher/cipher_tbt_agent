"""E2 arm M5: wake-sleep with the open set re-scored at each sleep onset (K4's procedure, run periodically in the
stream). (EXPERIMENT_E2.md)

WAKE: learn from the stream as it arrives (I's setup).
SLEEP, every --sleep-every wake updates:
1. score a sample of --score rows of everything seen so far with the CURRENT model;
2. the top 10 % are the open refutations, grouped by surprise kind;
3. --sleep-updates gentle updates (lr --sleep-lr, batch 32) = 8 open rows (kinds chosen uniformly) + 12 retired rows
   (even across kinds, kept across sleeps) + 12 stream rows;
4. an open row retires once its surprise has halved.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from e2_fewshot2 import batch, surprise_and_kind
from e2_world import eval_points, eval_set, evaluate, load, losses, robot_mask
from transitions import EVENTS
from worldmodel import WorldModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wake-decay", type=float, default=1.0)
    p.add_argument("--sleep-every", type=int, default=200)
    p.add_argument("--sleep-updates", type=int, default=100)
    p.add_argument("--sleep-lr", type=float, default=1e-4)
    p.add_argument("--score", type=int, default=20000)
    p.add_argument("--time-limit", type=float, default=285.0)
    p.add_argument("--focus", type=int, default=2)
    p.add_argument("--data", default="runs/data")
    p.add_argument("--ema", type=float, default=0.0, help="evaluate a slow EMA copy of the weights (0 = off)")
    p.add_argument("--probe", action="store_true", help="M6: kinds chosen by generalization to their unrehearsed half")
    p.add_argument("--around-sleep", type=int, default=0, help="evaluate before and after every n-th sleep (0 = off)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    train, test = load(os.path.join(a.data, "train.npz")), load(os.path.join(a.data, "test.npz"))
    L = int(train["lanes"])
    steps = len(train["act"]) // L
    ev, by_event = eval_set(test)
    ev_counts = np.stack([((train["events"] >> k) & 1).reshape(-1, L).sum(1) for k in range(len(EVENTS))], 1).cumsum(0)
    model = WorldModel(128, 4, 4, pos="local", persist=5.0, radius=1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    slow = None
    if a.ema:                                         # the slow "cortex": an EMA of the weights, read out, never trained
        import copy
        slow = copy.deepcopy(model)
    json.dump(dict(vars(a), d=128, layers=4, heads=4, pos="local", persist=5.0, radius=1),
              open(os.path.join(a.out, "config.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    points = set(eval_points(steps))
    retired = {}                                      # kind -> rows
    kind_prog = {}                                    # M6: kind -> EMA of the probe-half surprise drop per sleep
    t_train, n_sleep, n_retired = 0.0, 0, 0

    def train_on(rows, lr):
        for gr in opt.param_groups:
            gr["lr"] = lr
        o, g, act, nx, gx = batch(train, rows, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cl, gl, _ = model(o, g, act)
        loss, tot = losses(cl, gl, nx, gx, per_row=True, mask=robot_mask(o, a.focus))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if slow is not None:
            with torch.no_grad():
                for ps, pf in zip(slow.parameters(), model.parameters()):
                    ps.mul_(a.ema).add_(pf.detach(), alpha=1 - a.ema)
        return tot.cpu().numpy()

    def record(s, tag):
        r = evaluate(slow if slow is not None else model, ev, by_event, dev)
        if slow is not None:
            rf = evaluate(model, ev, by_event, dev)
            r["_fast"] = {e: rf[e]["acc"] for e in ("theft", "interact_full", "push", "button_off", "robot_burn",
                                                     "deliver", "mined")}
            r["_fast"]["changed_acc"] = rf["_overall"]["changed_acc"]
        for k, e in enumerate(EVENTS):
            if e in r:
                r[e]["seen"] = int(ev_counts[s - 1, k]) if s > 0 else 0
        r.update(step=s, transitions=s * L, train_seconds=t_train, tag=tag, sleeps=n_sleep, retired=n_retired,
                 retired_kinds=len(retired))
        log.write(json.dumps(r) + "\n")
        log.flush()
        print(f"step {s:5d} {tag:5s} {t_train:6.1f}s  changed {r['_overall']['changed_acc']:.3f} unchanged "
              f"{r['_overall']['unchanged_acc']:.3f}  theft {r['theft']['acc']:.2f}  burn {r['robot_burn']['acc']:.2f}  "
              f"button {r['button_off']['acc']:.2f}  full {r['interact_full']['acc']:.2f}  retired {n_retired}", flush=True)

    for s in range(steps + 1):
        if s in points:
            record(s, "wake")
        if s == steps or t_train > a.time_limit:
            break
        t1 = time.time()
        lr_w = a.lr * (a.wake_decay + (1 - a.wake_decay) * 0.5 * (1 + np.cos(np.pi * s / steps)))
        train_on(np.arange(s * L, (s + 1) * L), lr_w)
        if (s + 1) % a.sleep_every == 0:              # sleep: re-score, ruminate on what is still unexplained
            seen = (s + 1) * L
            watch = a.around_sleep and (n_sleep % a.around_sleep == a.around_sleep - 1)
            if watch:
                t_train += time.time() - t1
                record(s + 1, "pre")
                t1 = time.time()
            cand = rng.choice(seen, min(a.score, seen), replace=False)
            sv, kd = surprise_and_kind(model, train, cand, dev, a.focus)
            hot = sv > np.quantile(sv, 0.9)
            open_by_kind = {}
            for r_, k_, v_ in zip(cand[hot], kd[hot], sv[hot]):
                open_by_kind.setdefault(int(k_), []).append((int(r_), float(v_)))
            probes = {}
            if a.probe:                                   # M6: hold out half of each kind (>= 4 rows) as its probe
                for k_ in list(open_by_kind):
                    lst = open_by_kind[k_]
                    if len(lst) >= 4:
                        probes[k_] = np.array([r_ for r_, _ in lst[1::2]][:32])
                        open_by_kind[k_] = lst[0::2]
                before = {k_: float(surprise_and_kind(model, train, pr, dev, a.focus)[0].mean()) for k_, pr in probes.items()}
            for _ in range(a.sleep_updates):
                live = [k_ for k_ in open_by_kind if open_by_kind[k_]]
                if not live:
                    break
                if a.probe:
                    w = np.array([max(kind_prog[k_], 0.0) + 0.05 if k_ in kind_prog else 1.0 for k_ in live])
                    chosen = rng.choice(live, 8, p=w / w.sum())
                else:
                    chosen = rng.choice(live, 8)
                picks = [open_by_kind[k_][rng.integers(len(open_by_kind[k_]))] for k_ in chosen]
                rk = list(retired)
                ret_rows = [retired[rk[i]][rng.integers(len(retired[rk[i]]))] for i in rng.integers(0, len(rk), 12)] \
                    if rk else []
                rows = np.array([r_ for r_, _ in picks] + ret_rows + rng.integers(0, seen, 32 - 8 - len(ret_rows)).tolist())
                tot = train_on(rows, a.sleep_lr)
                for (r_, v0), sv_ in zip(picks, tot[:8]):
                    if sv_ < 0.5 * v0:
                        for k_ in live:
                            if (r_, v0) in open_by_kind[k_]:
                                open_by_kind[k_].remove((r_, v0))
                                retired.setdefault(k_, []).append(r_)
                                n_retired += 1
                                break
            if a.probe:                                   # did rehearsing a kind help its UNREHEARSED half?
                for k_, pr in probes.items():
                    d = before[k_] - float(surprise_and_kind(model, train, pr, dev, a.focus)[0].mean())
                    kind_prog[k_] = 0.5 * kind_prog[k_] + 0.5 * d if k_ in kind_prog else d
            n_sleep += 1
            if watch:
                t_train += time.time() - t1
                record(s + 1, "post")
                t1 = time.time()
        torch.cuda.synchronize()
        t_train += time.time() - t1
    record(s, "end")
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))


if __name__ == "__main__":
    main()
