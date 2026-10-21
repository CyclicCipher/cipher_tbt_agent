"""E3: does learning from refutations make the robot's own training faster? (EXPERIMENT_E3.md)

PPO alone (--aux 0, the control) against PPO plus the world-model loss on the same trunk (--aux c): the robot's
network predicts the next observation for every action it takes, with the persistence prior, so each step's
refutations shape the features its policy reads. Small world (--size 5), each run <= 5 minutes, the curves logged:
- "train" lines per update: the finished training episodes' return, entropy, KL, world-model loss;
- "eval" lines every --eval-every steps: return and value dimensions on 128 held-out layouts.
"""
import argparse
import functools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from agentnet import AgentNet
from e2_world import losses as wm_losses
from episodes import Roller, net_actor, run_episodes, summarize
from policy import amp
from ppo import gae, norm
from village import SPEC, Village

EVAL_SEEDS = [10**9 + i for i in range(128)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--aux", type=float, default=0.0, help="world-model loss coefficient (0 = PPO only)")
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--villagers", type=int, default=2)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--micro", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--pos", default="rel")
    p.add_argument("--persist", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--time-limit", type=float, default=280.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    make = functools.partial(Village, size=a.size, max_villagers=a.villagers)
    net = AgentNet(SPEC, a.d, a.layers, a.heads, pos=a.pos, persist=a.persist).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    json.dump(dict(vars(a), n_params=sum(q.numel() for q in net.parameters())), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    ref = summarize(run_episodes(EVAL_SEEDS, __import__("episodes").scripted_actor, make=make)[0])
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    log.write(json.dumps(dict(type="reference", **ref)) + "\n")
    roller = Roller(a.envs, a.rollout, a.seed * 10**7, dev, make=make)
    n = a.envs * a.rollout
    step, wall, next_eval, t_start = 0, 0.0, 0, time.time()
    while True:
        if step >= next_eval or wall > a.time_limit:
            net.eval()
            ev = summarize(run_episodes(EVAL_SEEDS, net_actor(net, dev, seed=0), make=make)[0])
            ev.update(type="eval", step=step, wall=wall)
            log.write(json.dumps(ev) + "\n")
            log.flush()
            next_eval += a.eval_every
            print(f"[eval] step {step:>8,}  {wall:5.0f}s  return {ev['ret']:6.2f} (scripted {ref['ret']:.2f})  "
                  f"delivered {ev['delivered']:.2f}  theft rate {ev['theft_rate']}", flush=True)
            if wall > a.time_limit:
                break
        t0 = time.time()
        net.eval()
        roll = roller.collect(net)
        adv, ret = gae(a, roll)
        B = lambda x: torch.from_numpy(x.reshape(n, *x.shape[2:])).to(dev)
        batch = dict(cells=B(roll["cells"]), glob=B(roll["glob"]), ncells=B(roll["nxt_cells"]), nglob=B(roll["nxt_glob"]),
                     act=roll["act"].reshape(-1), adv=adv.reshape(-1), ret=ret.reshape(-1),
                     logp=roll["logp_all"].reshape(n, -1).gather(1, roll["act"].reshape(-1, 1)).squeeze(1))
        net.train()
        stats, count = dict(ent=0.0, kl=0.0, wm=0.0), 0
        for _ in range(a.epochs):
            perm = torch.randperm(n, device=dev)
            for k in range(0, n, a.minibatch):
                mb = perm[k:k + a.minibatch]
                advn = norm(batch["adv"][mb])
                opt.zero_grad(set_to_none=True)
                for j in range(0, a.minibatch, a.micro):
                    idx, w = mb[j:j + a.micro], a.micro / a.minibatch
                    with amp(dev):
                        logits, v, h = net.policy_and_features(batch["cells"][idx], batch["glob"][idx])
                    lp_all = F.log_softmax(logits, -1)
                    lp = lp_all.gather(1, batch["act"][idx, None]).squeeze(1)
                    ent = -(lp_all.exp() * lp_all).sum(-1).mean()
                    ratio = (lp - batch["logp"][idx]).exp()
                    ad = advn[j:j + a.micro]
                    pg = torch.max(-ad * ratio, -ad * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
                    vl = 0.5 * (v - batch["ret"][idx]).pow(2).mean()
                    loss = pg - a.ent * ent + a.vf * vl
                    if a.aux:
                        with amp(dev):
                            cl, gl = net.predict(h, batch["cells"][idx], batch["glob"][idx], batch["act"][idx])
                        wl = wm_losses(cl, gl, batch["ncells"][idx], batch["nglob"][idx])
                        loss = loss + a.aux * wl
                        stats["wm"] += float(wl.detach()) * w
                    (loss * w).backward()
                    stats["ent"] += float(ent.detach()) * w
                    stats["kl"] += float(((ratio - 1) - (lp - batch["logp"][idx])).mean().detach()) * w
                torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad)
                opt.step()
                count += 1
        torch.cuda.synchronize()
        step += n
        wall += time.time() - t0
        fin = roll["finished"]
        line = dict(type="train", step=step, wall=wall, episodes=len(fin),
                    ret=float(np.mean([e["ret"] for e in fin])) if fin else None,
                    delivered=float(np.mean([e["delivered"] for e in fin])) if fin else None,
                    theft=float(np.mean([e["theft"] for e in fin])) if fin else None,
                    t_env=roll["t_env"], **{k: v / count for k, v in stats.items()})
        log.write(json.dumps(line) + "\n")
    print(f"done: {step:,} steps, {wall:.0f}s training, {time.time() - t_start:.0f}s wall")


if __name__ == "__main__":
    main()
