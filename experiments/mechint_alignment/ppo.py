"""E1: PPO on Village v0's task reward, instrumented to measure how efficiently it trains (EXPERIMENT_E1.md).

log.jsonl, one "train" line per update:
- the finished episodes' return and causal stats;
- the reward channel's bits (info.reward_bits);
- policy entropy, approx KL, clip fraction, explained variance;
- wall time split into env / rollout network / update; cumulative FLOPs (estimate).
Every --gns-every updates: the gradient noise scale of the policy loss and of the value loss, at theta_old.
Every --eval-every steps, an "eval" line: the held-out return and value dimensions (256 fixed seeds), the probe
set's information measures, and an fp32 checkpoint for the post-hoc analysis (analyze_e1.py).

GPU job:   python ppo.py --out runs/e1
Benchmark: python ppo.py --out runs/bench --bench 60      (throughput + projected run time, then exit)
Resume:    rerun the same command; it continues from runs/e1/latest.pt (fresh episodes, same seed counter).
"""
import argparse
import collections
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import info
from episodes import EVAL_SEEDS, PROBE_SEEDS, Roller, net_actor, run_episodes, scripted_actor, summarize
from policy import Policy, amp, flops_per_forward, n_params


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--micro", type=int, default=512, help="gradient-accumulation chunk (memory only)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=250_000)
    p.add_argument("--gns-every", type=int, default=10)
    p.add_argument("--bench", type=float, default=0.0, help="seconds")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def losses(net, a, cells, glob, act, old_logp, adv, ret):
    with amp(cells.device):
        logits, v = net(cells, glob)
    logp_all = F.log_softmax(logits, -1)
    logp = logp_all.gather(1, act[:, None]).squeeze(1)
    ent = -(logp_all.exp() * logp_all).sum(-1).mean()
    ratio = (logp - old_logp).exp()
    pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
    vl = 0.5 * (v - ret).pow(2).mean()
    with torch.no_grad():
        kl = ((ratio - 1) - (logp - old_logp)).mean()
        clipped = ((ratio - 1).abs() > a.clip).float().mean()
    return pg, vl, ent, kl, clipped


def norm(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def measure_gns(net, a, batch):
    """B_simple of the policy loss (incl. the entropy bonus) and of the value loss, at theta_old (no step)."""
    params = [p for p in net.parameters()]
    n = batch["act"].shape[0]
    k, b = n // a.minibatch, a.minibatch
    perm = torch.randperm(n, device=batch["act"].device)
    sq, vec = {"pi": 0.0, "v": 0.0}, {"pi": None, "v": None}
    for m in range(k):
        mb = perm[m * b:(m + 1) * b]
        adv = norm(batch["adv"][mb])
        acc = {"pi": [torch.zeros_like(p) for p in params], "v": [torch.zeros_like(p) for p in params]}
        for j in range(0, b, a.micro):
            idx, w = mb[j:j + a.micro], a.micro / b
            pg, vl, ent, _, _ = losses(net, a, batch["cells"][idx], batch["glob"][idx], batch["act"][idx],
                                       batch["logp"][idx], adv[j:j + a.micro], batch["ret"][idx])
            gp = torch.autograd.grad((pg - a.ent * ent) * w, params, retain_graph=True, allow_unused=True)
            gv = torch.autograd.grad(a.vf * vl * w, params, allow_unused=True)
            for key, gs in (("pi", gp), ("v", gv)):
                for s, g in zip(acc[key], gs):
                    if g is not None:
                        s += g
        for key in acc:
            flat = torch.cat([g.flatten() for g in acc[key]]).float()
            sq[key] += float(flat.pow(2).sum())
            vec[key] = flat if vec[key] is None else vec[key] + flat
    return {key: info.gns(sq[key], vec[key], k, b) for key in sq}


def ppo_update(net, opt, a, batch):
    n = batch["act"].shape[0]
    stats = collections.defaultdict(float)
    count = 0
    for _ in range(a.epochs):
        perm = torch.randperm(n, device=batch["act"].device)
        for k in range(0, n, a.minibatch):
            mb = perm[k:k + a.minibatch]
            adv = norm(batch["adv"][mb])
            opt.zero_grad(set_to_none=True)
            for j in range(0, a.minibatch, a.micro):
                idx, w = mb[j:j + a.micro], a.micro / a.minibatch
                pg, vl, ent, kl, clipped = losses(net, a, batch["cells"][idx], batch["glob"][idx], batch["act"][idx],
                                                  batch["logp"][idx], adv[j:j + a.micro], batch["ret"][idx])
                ((pg - a.ent * ent + a.vf * vl) * w).backward()
                for key, x in (("pg", pg), ("vloss", vl), ("ent_nats", ent), ("kl", kl), ("clipfrac", clipped)):
                    stats[key] += float(x.detach()) * w
            stats["gradnorm"] += float(torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad))
            opt.step()
            count += 1
    return {key: v / count for key, v in stats.items()}


def gae(a, roll):
    rew = torch.from_numpy(roll["rew"]).to(roll["val"].device)
    done = torch.from_numpy(roll["done"]).to(roll["val"].device)
    val = roll["val"]
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(rew[0])
    for t in reversed(range(rew.shape[0])):
        nxt = roll["v_last"] if t == rew.shape[0] - 1 else val[t + 1]
        live = 1.0 - done[t]
        last = rew[t] + a.gamma * nxt * live - val[t] + a.gamma * a.lam * live * last
        adv[t] = last
    return adv, adv + val


def evaluate(net, dev, probe):
    net.eval()
    eps, _ = run_episodes(EVAL_SEEDS, net_actor(net, dev, seed=0))
    out = summarize(eps)
    out.update(info.policy_info(info.log_probs(net, dev, probe["cells"], probe["glob"]), probe["acts"]))
    return out


def main(argv=None):
    a = get_args(argv)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    net = Policy(a.d, a.layers, a.heads).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    n_batch = a.envs * a.rollout
    assert n_batch % a.minibatch == 0 and a.minibatch % a.micro == 0
    fwd = flops_per_forward(net)
    state = dict(step=0, update=0, next_seed=a.seed * 10**7, reward_bits=0.0, reward_events=0, flops=0.0,
                 wall=0.0, next_eval=0)
    latest = os.path.join(a.out, "latest.pt")
    if os.path.exists(latest) and not a.bench:
        ck = torch.load(latest, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        torch.set_rng_state(ck["torch_rng"])
        print(f"resumed at step {state['step']}")
    elif not a.bench:
        with open(os.path.join(a.out, "config.json"), "w") as f:
            json.dump(dict(vars(a), n_params=n_params(net), flops_per_forward=fwd), f, indent=1)
        t0 = time.time()
        ref = summarize(run_episodes(EVAL_SEEDS, scripted_actor)[0])
        with open(os.path.join(a.out, "reference.json"), "w") as f:
            json.dump(ref, f, indent=1)
        _, (pc, pg, pa) = run_episodes(PROBE_SEEDS, scripted_actor, collect=True)
        np.savez_compressed(os.path.join(a.out, "probe.npz"), cells=pc, glob=pg, acts=pa)
        print(f"scripted reference: return {ref['ret']:.2f}, probe set {len(pa)} states ({time.time() - t0:.0f}s)")
    probe = dict(np.load(os.path.join(a.out, "probe.npz"))) if not a.bench else None
    roller = Roller(a.envs, a.rollout, state["next_seed"], dev)
    log = open(os.path.join(a.out, "log.jsonl"), "a") if not a.bench else None
    print(f"params {n_params(net):,}  forward {fwd / 1e9:.2f} GFLOP/obs  batch {n_batch}  device {dev}")

    t_start = time.time()
    while state["step"] < a.steps:
        if not a.bench and state["step"] >= state["next_eval"]:
            t0 = time.time()
            ev = evaluate(net, dev, probe)
            ev.update(type="eval", step=state["step"], wall=state["wall"], flops=state["flops"],
                      reward_bits_cum=state["reward_bits"], reward_events_cum=state["reward_events"],
                      eval_seconds=time.time() - t0)
            log.write(json.dumps(ev) + "\n")
            log.flush()
            torch.save(dict(step=state["step"], model=net.state_dict(), cfg=vars(a)),
                       os.path.join(a.out, f"ckpt_{state['step']:010d}.pt"))
            state["next_eval"] += a.eval_every
            print(f"[eval] step {state['step']:>10,}  return {ev['ret']:6.2f}  I(S;A) {ev['I_SA_bits']:.3f} bits  "
                  f"theft {ev['theft_rate']}  button {ev['button_disabled_rate']}")
        t0 = time.time()
        net.eval()
        roll = roller.collect(net)
        adv, ret = gae(a, roll)
        batch = dict(cells=torch.from_numpy(roll["cells"].reshape(n_batch, *roll["cells"].shape[2:])).to(dev),
                     glob=torch.from_numpy(roll["glob"].reshape(n_batch, -1)).to(dev),
                     act=roll["act"].reshape(-1), adv=adv.reshape(-1), ret=ret.reshape(-1),
                     logp=roll["logp_all"].reshape(n_batch, -1).gather(1, roll["act"].reshape(-1, 1)).squeeze(1))
        t1 = time.time()
        net.train()
        g = measure_gns(net, a, batch) if a.gns_every and state["update"] % a.gns_every == 0 else None
        upd = ppo_update(net, opt, a, batch)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_upd = time.time() - t1
        h_r, bits = info.reward_bits(roll["rew"])
        events = int((roll["rew"] > 0).sum())
        val, r_ = roll["val"].reshape(-1), ret.reshape(-1)
        ev_ = float(1 - (r_ - val).var() / (r_.var() + 1e-8))
        state["step"] += n_batch
        state["update"] += 1
        state["reward_bits"] += bits
        state["reward_events"] += events
        state["flops"] += fwd * n_batch * (1 + 3 * a.epochs + (5 if g else 0))
        state["wall"] += time.time() - t0
        state["next_seed"] = roller.next_seed
        fin = roll["finished"]
        line = dict(type="train", step=state["step"], update=state["update"], wall=state["wall"],
                    sps=n_batch / (time.time() - t0), t_env=roll["t_env"], t_net=roll["t_net"], t_upd=t_upd,
                    episodes=len(fin), ret=float(np.mean([e["ret"] for e in fin])) if fin else None,
                    delivered=float(np.mean([e["delivered"] for e in fin])) if fin else None,
                    reward_H_bits=h_r, reward_bits=bits, reward_bits_cum=state["reward_bits"],
                    reward_events=events, reward_events_cum=state["reward_events"], explained_var=ev_,
                    ent_bits=upd["ent_nats"] / info.LN2, flops=state["flops"], gns=g,
                    **{k: v for k, v in upd.items() if k != "ent_nats"})
        if a.bench:
            el = time.time() - t_start
            print(f"update {state['update']}: {line['sps']:.0f} steps/s (env {roll['t_env']:.1f}s, net "
                  f"{roll['t_net']:.1f}s, update {t_upd:.1f}s)")
            if el >= a.bench:
                sps = state["step"] / el
                mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
                print(f"BENCH {sps:.0f} steps/s overall; peak GPU memory {mem:.2f} GiB; "
                      f"{a.steps:,} steps -> {a.steps / sps / 3600:.1f} h (+ evaluations)")
                return
            continue
        log.write(json.dumps(line) + "\n")
        log.flush()
        if state["step"] >= state["next_eval"] or state["step"] >= a.steps:
            torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), state=state,
                            torch_rng=torch.get_rng_state()), latest)
        if state["update"] % 10 == 0:
            print(f"step {state['step']:>10,}  return {line['ret']}  ent {line['ent_bits']:.2f} bits  "
                  f"{line['sps']:.0f} steps/s")
    if not a.bench:
        ev = evaluate(net, dev, probe)
        ev.update(type="eval", step=state["step"], wall=state["wall"], flops=state["flops"],
                  reward_bits_cum=state["reward_bits"], reward_events_cum=state["reward_events"])
        log.write(json.dumps(ev) + "\n")
        torch.save(dict(step=state["step"], model=net.state_dict(), cfg=vars(a)),
                   os.path.join(a.out, f"ckpt_{state['step']:010d}.pt"))
        state["next_eval"] = state["step"] + a.eval_every          # a resume with more --steps goes on from here
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), state=state,
                        torch_rng=torch.get_rng_state()), latest)
        print(f"done: {state['step']:,} steps in {state['wall'] / 3600:.2f} h, final return {ev['ret']:.2f}")


if __name__ == "__main__":
    main()
