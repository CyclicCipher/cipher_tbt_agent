"""E2 arm M: wake-sleep rumination, K3's learning rule run inside the online stream. (EXPERIMENT_E2.md)

WAKE: learn from the stream as it arrives (I's setup: locality window, persistence prior, robot-local loss), and keep
the rows whose in-scope surprise is in the batch's top 10 % as OPEN refutations, each with its surprise kind
(e2_world.signature) and its surprise on arrival.
SLEEP, every --sleep-every wake updates: --sleep-updates gentle updates (lr --sleep-lr, batch 32), each with
- 8 open refutations, the kinds chosen by learning progress (L3's bandit) plus an exploration bonus, and
- 24 retired refutations, sampled evenly across kinds (interleaving, against forgetting).
An open refutation RETIRES once its surprise has halved since arrival (stop when explained), and is DROPPED after 10
rehearsals without halving (noise cannot be learned).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from e2_world import eval_points, eval_set, evaluate, load, losses, robot_mask, robot_pos
from village import SPEC
from village_events import EVENTS
from worldmodel import WorldModel


class Ruminator:
    def __init__(self, cap_open=4096, cap_ret=16384, patience=10, ema=0.9, explore=0.05, seed=0):
        self.rng = np.random.default_rng(seed)
        self.open = {}                          # row -> [kind, surprise on arrival, rehearsals, last surprise]
        self.ret = {}                           # kind -> list of rows
        self.cap_open, self.cap_ret, self.patience = cap_open, cap_ret, patience
        self.prog, self.tries = np.zeros(229), np.zeros(229)
        self.ema, self.explore = ema, explore
        self.admitted = self.retired = self.dropped = 0
        self.prequential = False                # M2: progress = the drop in NEW arrivals' surprise, per kind
        self.absolute = False                   # M3: progress in absolute nats, from refutations only
        self.uniform = False                    # M4: kinds chosen uniformly
        self.fast, self.slow, self.seen = np.zeros(229), np.zeros(229), np.zeros(229)

    def observe(self, surprise, kinds, a_fast=0.2, a_slow=0.02):
        """Prequential learning progress per kind: EMA of new arrivals' surprise, slow minus fast (> 0 = learning)."""
        for k in np.unique(kinds):
            m = float(surprise[kinds == k].mean())
            if self.seen[k] == 0:
                self.fast[k] = self.slow[k] = m
            else:
                self.fast[k] += a_fast * (m - self.fast[k])
                self.slow[k] += a_slow * (m - self.slow[k])
            self.seen[k] += 1
            if self.prequential:
                d = self.slow[k] - self.fast[k]
                self.prog[k] = d if self.absolute else d / max(self.slow[k], 1e-6)   # M3: bits, not a ratio
                self.tries[k] = self.seen[k]

    def admit(self, rows, surprise, kinds, frac=0.10):
        cut = np.quantile(surprise, 1 - frac)
        for r, s, k in zip(rows, surprise, kinds):
            if s > cut and int(r) not in self.open:
                self.open[int(r)] = [int(k), float(s), 0, float(s)]
                self.admitted += 1
        if len(self.open) > self.cap_open:      # keep the most surprising
            keep = sorted(self.open.items(), key=lambda kv: -kv[1][3])[:self.cap_open]
            self.open = dict(keep)

    def sample(self, n_open=8, n_ret=24, n_stream=0, stream_hi=0):
        rows = []
        if self.open:
            items = list(self.open.items())
            kinds = np.array([v[0] for _, v in items])
            uk = np.unique(kinds)
            w = np.ones(len(uk)) if self.uniform else                 np.maximum(self.prog[uk], 0) + np.where(self.tries[uk] < 5, 1.0, self.explore)
            for kd in self.rng.choice(uk, n_open, p=w / w.sum()):
                cand = [r for (r, v) in items if v[0] == kd]
                rows.append(cand[self.rng.integers(len(cand))])
        n_open_got = len(rows)
        if self.ret:
            ks = list(self.ret)
            for kd in self.rng.choice(len(ks), n_ret):
                lst = self.ret[ks[kd]]
                rows.append(lst[self.rng.integers(len(lst))])
        if n_stream and stream_hi:              # M2: keep the base rate of "nothing changes"
            rows.extend(self.rng.integers(0, stream_hi, n_stream).tolist())
        return np.array(rows, int), n_open_got

    def update(self, rows, surprise, n_open):
        for r, s in zip(rows[:n_open], surprise[:n_open]):
            if r not in self.open:
                continue
            v = self.open[r]
            drop = (v[3] - s) / max(v[3], 1e-6)
            k = v[0]
            if not self.prequential:
                self.prog[k] = self.ema * self.prog[k] + (1 - self.ema) * drop if self.tries[k] else drop
                self.tries[k] += 1
            v[2] += 1
            v[3] = float(s)
            if s < 0.5 * v[1]:
                self.ret.setdefault(k, []).append(r)
                del self.open[r]
                self.retired += 1
            elif v[2] >= self.patience:
                del self.open[r]
                self.dropped += 1
        total = sum(len(x) for x in self.ret.values())
        if total > self.cap_ret:                # forget the oldest of the largest kind
            big = max(self.ret, key=lambda k: len(self.ret[k]))
            self.ret[big] = self.ret[big][1:]

    def stats(self):
        return dict(open=len(self.open), retired_kinds=len(self.ret), retired_pool=sum(len(x) for x in self.ret.values()),
                    admitted=self.admitted, retired=self.retired, dropped=self.dropped,
                    top_kinds=[(int(k), round(float(self.prog[k]), 3), int(self.tries[k]))
                               for k in np.argsort(-np.where(self.tries >= 5, self.prog, -9))[:6]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--focus", type=int, default=2)
    p.add_argument("--sleep-every", type=int, default=200)
    p.add_argument("--sleep-updates", type=int, default=100)
    p.add_argument("--sleep-lr", type=float, default=1e-4)
    p.add_argument("--time-limit", type=float, default=285.0)
    p.add_argument("--prequential", action="store_true", help="M2: progress from new arrivals per kind")
    p.add_argument("--stream-mix", type=int, default=0, help="M2: stream rows per sleep batch")
    p.add_argument("--absolute", action="store_true", help="M3: absolute progress, observed on refutations only")
    p.add_argument("--wake-decay", type=float, default=1.0, help="M4: wake lr decays (cosine) to lr * this by the end")
    p.add_argument("--uniform-kinds", action="store_true", help="M4: open kinds chosen uniformly (K4u)")
    p.add_argument("--data", default="runs/data")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    train, test = load(os.path.join(a.data, "train.npz")), load(os.path.join(a.data, "test.npz"))
    L = int(train["lanes"])
    steps = len(train["act"]) // L
    ev, by_event = eval_set(test)
    ev_counts = np.stack([((train["events"] >> k) & 1).reshape(-1, L).sum(1) for k in range(len(EVENTS))], 1).cumsum(0)
    model = WorldModel(SPEC, a.d, a.layers, a.heads, pos="local", persist=5.0, radius=1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rum = Ruminator()
    rum.prequential = a.prequential
    rum.absolute = a.absolute
    rum.uniform = a.uniform_kinds
    json.dump(dict(vars(a), pos="local", persist=5.0, radius=1), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    points = set(eval_points(steps))

    def step_on(rows, lr):
        for gr in opt.param_groups:
            gr["lr"] = lr
        o, g, act, nx, gx = (torch.from_numpy(train[x][rows]).to(dev) for x in ("obs", "gob", "act", "nxt", "gnx"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cl, gl, _ = model(o, g, act)
        loss, tot, sig = losses(cl, gl, nx, gx, per_row="sig", mask=robot_mask(o, a.focus), robot_pos=robot_pos(o))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return tot.cpu().numpy(), sig

    t_train, sleep_updates, n_rows = 0.0, 0, 0
    for s in range(steps + 1):
        if s in points:
            r = evaluate(model, ev, by_event, dev)
            for k, e in enumerate(EVENTS):
                if e in r:
                    r[e]["seen"] = int(ev_counts[s - 1, k]) if s > 0 else 0
            r.update(step=s, transitions=s * L, train_seconds=t_train, sleep_updates=sleep_updates, rows=n_rows,
                     rumination=rum.stats())
            log.write(json.dumps(r) + "\n")
            log.flush()
            print(f"step {s:5d}  {t_train:6.1f}s  changed acc {r['_overall']['changed_acc']:.3f}  theft {r['theft']['acc']:.2f}"
                  f"  burn {r['robot_burn']['acc']:.2f}  push {r['push']['acc']:.2f}  button {r['button_off']['acc']:.2f}"
                  f"  open {len(rum.open)} retired {rum.retired}", flush=True)
        if s == steps or t_train > a.time_limit:
            break
        t1 = time.time()
        rows = np.arange(s * L, (s + 1) * L)
        lr_w = a.lr * (a.wake_decay + (1 - a.wake_decay) * 0.5 * (1 + np.cos(np.pi * s / steps)))
        tot, sig = step_on(rows, lr_w)                                # wake
        n_rows += len(rows)
        rum.admit(rows, tot, sig)
        if a.absolute:                                                # only refutations are arrivals of a kind
            hot = tot > np.quantile(tot, 0.9)
            rum.observe(tot[hot], sig[hot])
        else:
            rum.observe(tot, sig)
        if (s + 1) % a.sleep_every == 0:                              # sleep
            for _ in range(a.sleep_updates):
                srows, n_open = rum.sample(8, 24 - a.stream_mix, a.stream_mix, (s + 1) * L) if a.stream_mix                     else rum.sample()
                if len(srows) == 0:
                    break
                stot, _ = step_on(srows, a.sleep_lr)
                rum.update(srows, stot, n_open)
                sleep_updates += 1
                n_rows += len(srows)
        torch.cuda.synchronize()
        t_train += time.time() - t1
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))


if __name__ == "__main__":
    main()
