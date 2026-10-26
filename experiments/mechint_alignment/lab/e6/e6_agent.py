"""E6: the one-network agent from a random initialization. (EXPERIMENT_E6.md)

No scripted policy anywhere: the robot's own experience, from a fresh network whose Q is zero (so it babbles).
Each collection step (all lanes in lockstep):
  act   ~ softmax(Q(s, .) / temp) from one trunk pass;
  then the same pass predicts the next state for the action taken -> its in-scope surprise and surprise KIND
  (e2_world.signature); the intrinsic reward is computed from that (--intrinsic):
    lp       : learning progress of the kind (slow EMA - fast EMA of its refutations' surprise, in nats, >= 0)
    surprise : the raw surprise (the noisy-TV control)
    none
Each update (one per collection step, a batch from the replay of the robot's own experience), ONE loss on ONE net:
  next-state cross-entropy (robot-local, persistence prior) + (reward(s,a) - r)^2 + Huber(Q(s,a) - y),
  y = r + r_int + gamma * max_a' Q_target(s', a'), Q_target = a slow EMA copy of the network.
"""
import argparse
import copy
import functools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from worlds import scripted
from lab.e2.e2_world import losses, robot_mask, robot_pos
from nets.onenet import OneNet
from worlds.village import CELL_VOCAB, GLOBAL_VOCAB, N, N_ACTIONS, SPEC, Village

EVAL_SEEDS = [10**9 + i for i in range(64)]


class Progress:
    """Per surprise kind: fast and slow EMAs of the surprise of its refutations (transitions above `thr` nats)."""

    def __init__(self, n=229, fast=0.2, slow=0.02):
        self.fast, self.slow, self.n = np.zeros(n), np.zeros(n), np.zeros(n)
        self.a_fast, self.a_slow = fast, slow

    def observe(self, s, k):
        for kd in np.unique(k):
            m = float(s[k == kd].mean())
            if self.n[kd] == 0:
                self.fast[kd] = self.slow[kd] = m
            else:
                self.fast[kd] += self.a_fast * (m - self.fast[kd])
                self.slow[kd] += self.a_slow * (m - self.slow[kd])
            self.n[kd] += 1

    def lp(self, k):
        return np.maximum(self.slow[k] - self.fast[k], 0.0)


def evaluate(net, make, dev, temp=0.05, seed=0):
    """Held-out layouts: return, delivered, theft, robot deaths, and the fraction of steps spent within 2 cells of a
    villager (the noisy-TV signature: villagers' random steps are the world's main unlearnable surprise)."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    envs = [make(s) for s in EVAL_SEEDS]
    n = len(envs)
    cells = np.zeros((n, N * N, len(CELL_VOCAB)), np.uint8)
    glob = np.zeros((n, len(GLOBAL_VOCAB)), np.uint8)
    ret, near, steps = np.zeros(n), np.zeros(n), np.zeros(n)
    live = np.ones(n, bool)
    while live.any():
        idx = np.nonzero(live)[0]
        for k in idx:
            envs[k].obs(cells[k], glob[k])
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = net.trunk(torch.from_numpy(cells[idx]).to(dev), torch.from_numpy(glob[idx]).to(dev))
            q, _ = net.values(h)
            a = torch.multinomial(F.softmax(q / temp, -1), 1, generator=gen).squeeze(1).cpu().numpy()
        for j, k in enumerate(idx):
            e = envs[k]
            r_, c_ = e.robot["pos"]
            near[k] += any(v["alive"] and max(abs(v["pos"][0] - r_), abs(v["pos"][1] - c_)) <= 2 for v in e.villagers)
            steps[k] += 1
            r, d = e.step(int(a[j]))
            ret[k] += r
            live[k] = not d
    st = [e.summary() for e in envs]
    chest = [s for s in st if s["m_theft"]]
    return dict(ret=float(ret.mean()), delivered=float(np.mean([s["delivered"] for s in st])),
                theft_rate=float(np.mean([s["theft"] > 0 for s in chest])) if chest else None,
                robot_died=float(np.mean([s["robot_died"] for s in st])), near_villager=float((near / steps).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--intrinsic", default="lp", choices=("lp", "surprise", "none"))
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--cap", type=float, default=5.0, help="nats at which the intrinsic reward saturates")
    p.add_argument("--refut-top", type=float, default=0.1, help="the most surprising fraction of each step's transitions = refutations")
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--villagers", type=int, default=2)
    p.add_argument("--lanes", type=int, default=64)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--buffer", type=int, default=200_000)
    p.add_argument("--warmup", type=int, default=5_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.995)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--focus", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--time-limit", type=float, default=270.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    make = functools.partial(Village, size=a.size, max_villagers=a.villagers)
    net = OneNet(SPEC, a.d, a.layers).to(dev)
    targ = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    json.dump(dict(vars(a), n_params=sum(q.numel() for q in net.parameters())), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w")

    # yardsticks on the evaluation layouts (evaluation only, never training data)
    envs = [make(s) for s in EVAL_SEEDS]
    sc = []
    for e in envs:
        r = 0.0
        while not e.done:
            r += e.step(scripted.act(e))[0]
        sc.append(r)
    log.write(json.dumps(dict(type="reference", scripted_ret=float(np.mean(sc)))) + "\n")
    print(f"scripted task policy on the evaluation layouts: {np.mean(sc):.2f}", flush=True)

    F_, G_ = len(CELL_VOCAB), len(GLOBAL_VOCAB)
    B = a.buffer
    buf = dict(c=np.zeros((B, N * N, F_), np.uint8), g=np.zeros((B, G_), np.uint8), a=np.zeros(B, np.int64),
               r=np.zeros(B, np.float32), ri=np.zeros(B, np.float32), d=np.zeros(B, np.float32),
               nc=np.zeros((B, N * N, F_), np.uint8), ng=np.zeros((B, G_), np.uint8))
    size = ptr = 0
    seed_next = a.seed * 10**7
    envs = []
    for _ in range(a.lanes):
        envs.append(make(seed_next))
        seed_next += 1
    cells = np.zeros((a.lanes, N * N, F_), np.uint8)
    glob = np.zeros((a.lanes, G_), np.uint8)
    for k, e in enumerate(envs):
        e.obs(cells[k], glob[k])
    nxt_c, nxt_g = np.zeros_like(cells), np.zeros_like(glob)
    prog = Progress()
    steps, wall, next_eval, upd = 0, 0.0, 0, 0
    acc = dict(wm=0.0, rw=0.0, td=0.0, ri=0.0, sur=0.0, refut=0.0, n=0)
    ep_ret = np.zeros(a.lanes)
    finished = []
    while True:
        if steps >= next_eval or wall > a.time_limit:
            ev = evaluate(net, make, dev)
            top = np.argsort(-prog.lp(np.arange(229)))[:6]
            ev.update(type="eval", steps=steps, wall=wall, updates=upd,
                      top_lp_kinds=[(int(k), round(float(prog.lp(k)), 3), int(prog.n[k])) for k in top])
            log.write(json.dumps(ev) + "\n")
            log.flush()
            print(f"[eval] steps {steps:>7,}  {wall:5.0f}s  return {ev['ret']:6.2f}  delivered {ev['delivered']:.2f}  "
                  f"died {ev['robot_died']:.2f}  near villager {ev['near_villager']:.2f}  theft {ev['theft_rate']}  "
                  f"top LP kinds {ev['top_lp_kinds'][:3]}", flush=True)
            next_eval += a.eval_every
            if wall > a.time_limit:
                break
        t0 = time.time()
        c_t, g_t = torch.from_numpy(cells).to(dev), torch.from_numpy(glob).to(dev)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = net.trunk(c_t, g_t)
            q, _ = net.values(h)
            act_t = torch.multinomial(F.softmax(q / a.temp, -1), 1).squeeze(1)
        act = act_t.cpu().numpy()
        rew = np.zeros(a.lanes, np.float32)
        done = np.zeros(a.lanes, np.float32)
        for k, e in enumerate(envs):
            r, d = e.step(int(act[k]))
            rew[k], done[k] = r, d
            ep_ret[k] += r
            e.obs(nxt_c[k], nxt_g[k])
        with torch.no_grad():                         # the step as a prediction: surprise and its kind
            nc_t, ng_t = torch.from_numpy(nxt_c).to(dev), torch.from_numpy(nxt_g).to(dev)
            cl, gl = net.predict_next(h, c_t, g_t, act_t)
            _, sur, kind = losses(cl, gl, nc_t, ng_t, per_row="sig", mask=robot_mask(c_t, a.focus), robot_pos=robot_pos(c_t))
            sur = sur.cpu().numpy()
        refut = sur > np.quantile(sur, 1 - a.refut_top)
        if refut.any():
            prog.observe(sur[refut], kind[refut])
        if a.intrinsic == "lp":
            ri = np.where(refut, a.beta * np.minimum(prog.lp(kind), a.cap) / a.cap, 0.0)
        elif a.intrinsic == "surprise":
            ri = a.beta * np.minimum(sur, a.cap) / a.cap
        else:
            ri = np.zeros(a.lanes)
        idx = (ptr + np.arange(a.lanes)) % B
        buf["c"][idx], buf["g"][idx], buf["a"][idx], buf["r"][idx] = cells, glob, act, rew
        buf["ri"][idx], buf["d"][idx], buf["nc"][idx], buf["ng"][idx] = ri, done, nxt_c, nxt_g
        ptr, size = (ptr + a.lanes) % B, min(size + a.lanes, B)
        for k, e in enumerate(envs):                   # reset finished episodes
            if done[k]:
                finished.append(ep_ret[k])
                ep_ret[k] = 0.0
                e.reset(seed_next)
                seed_next += 1
            e.obs(cells[k], glob[k]) if done[k] else None
        cells[~done.astype(bool)] = nxt_c[~done.astype(bool)]
        glob[~done.astype(bool)] = nxt_g[~done.astype(bool)]
        steps += a.lanes
        acc["ri"] += float(ri.mean()); acc["sur"] += float(sur.mean()); acc["refut"] += float(refut.mean()); acc["n"] += 1

        if size >= a.warmup:                          # one update on the robot's own replayed experience
            j = rng.integers(0, size, a.batch)
            T = lambda x: torch.from_numpy(buf[x][j]).to(dev)
            c_b, g_b, a_b, r_b, ri_b, d_b, nc_b, ng_b = (T(x) for x in ("c", "g", "a", "r", "ri", "d", "nc", "ng"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = net.trunk(c_b, g_b)
            qa, ra = net.values(h)
            q_sa = qa.gather(1, a_b[:, None]).squeeze(1)
            r_sa = ra.gather(1, a_b[:, None]).squeeze(1)
            cl, gl = net.predict_next(h, c_b, g_b, a_b)
            wm = losses(cl, gl, nc_b, ng_b, mask=robot_mask(c_b, a.focus))
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ht = targ.trunk(nc_b, ng_b)
                qt, _ = targ.values(ht)
                y = r_b + ri_b + a.gamma * (1 - d_b) * qt.max(1).values
            td = F.smooth_l1_loss(q_sa, y)
            rw = F.mse_loss(r_sa, r_b)
            loss = wm + rw + td
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for pt, pn in zip(targ.parameters(), net.parameters()):
                    pt.mul_(a.tau).add_(pn.detach(), alpha=1 - a.tau)
            upd += 1
            acc["wm"] += float(wm.detach()); acc["rw"] += float(rw.detach()); acc["td"] += float(td.detach())
        torch.cuda.synchronize()
        wall += time.time() - t0
        if acc["n"] >= 50:
            n_ = acc.pop("n")
            line = dict(type="train", steps=steps, wall=wall, updates=upd,
                        train_ret=float(np.mean(finished[-64:])) if finished else None,
                        **{k: v / n_ for k, v in acc.items()})
            log.write(json.dumps(line) + "\n")
            acc = dict(wm=0.0, rw=0.0, td=0.0, ri=0.0, sur=0.0, refut=0.0, n=0)
    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
    print(f"done: {steps:,} steps, {upd:,} updates, {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
