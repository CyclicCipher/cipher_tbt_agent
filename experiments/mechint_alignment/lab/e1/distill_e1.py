"""E1 / M5: on-policy distillation of the final PPO policy into a fresh network. Same architecture, batch geometry,
optimizer and epochs as PPO; the training signal is the teacher's action distribution on the student's OWN states
(as in OPSD and DAgger) instead of the reward.

It measures:
(a) the env steps a fully informative signal needs to reach the returns PPO reached: the "RL tax", by milestone;
(b) the prequential code length of the behaviour: on each new batch, BEFORE training on it, the student's
    cross-entropy to the teacher, and KL(teacher || student) (that excludes the teacher's own randomness). Summed
    over all samples, the KL is the excess code length. The area above its final level estimates the behaviour's
    description length for this architecture (epiplexity: Finzi et al. 2026, arXiv:2601.03220, eq. 8);
(c) the distillation loss's gradient noise scale, to compare with PPO's policy loss.

    python -m lab.e1.distill_e1 --run runs/e1 --steps 3000000
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from lab.e1 import info
from nets.policy import Policy, amp
from worlds.episodes import EVAL_SEEDS, PROBE_SEEDS, Roller, net_actor, run_episodes, summarize
from worlds.village import SPEC


def soft_ce(net, cells, glob_, p_teacher):
    with amp(cells.device):
        logits, _ = net(cells, glob_)
    return -(p_teacher * F.log_softmax(logits, -1)).sum(-1).mean()


def measure_gns(net, a, batch):
    params = list(net.parameters())
    n = batch["p"].shape[0]
    k, b = n // a.minibatch, a.minibatch
    perm = torch.randperm(n, device=batch["p"].device)
    sq, vec = 0.0, None
    for m in range(k):
        mb = perm[m * b:(m + 1) * b]
        acc = [torch.zeros_like(p) for p in params]
        for j in range(0, b, a.micro):
            idx = mb[j:j + a.micro]
            gs = torch.autograd.grad(soft_ce(net, batch["cells"][idx], batch["glob"][idx], batch["p"][idx]) * a.micro / b,
                                     params, allow_unused=True)
            for s, g in zip(acc, gs):
                if g is not None:
                    s += g
        flat = torch.cat([g.flatten() for g in acc]).float()
        sq += float(flat.pow(2).sum())
        vec = flat if vec is None else vec + flat
    return info.gns(sq, vec, k, b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="the PPO run folder; its last checkpoint is the teacher")
    p.add_argument("--out", default=None)
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--gns-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    for key in ("envs", "rollout", "minibatch", "micro", "epochs", "lr", "max_grad", "d", "layers", "heads"):
        setattr(a, key, cfg[key])
    a.out = a.out or os.path.join(a.run, "distill")
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)

    teacher = Policy(SPEC, a.d, a.layers, a.heads).to(dev)
    ckpt = sorted(glob.glob(os.path.join(a.run, "ckpt_*.pt")))[-1]
    teacher.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    teacher.eval()
    torch.manual_seed(a.seed)
    student = Policy(SPEC, a.d, a.layers, a.heads).to(dev)
    opt = torch.optim.Adam(student.parameters(), lr=a.lr, eps=1e-5)
    teacher_eval = summarize(run_episodes(EVAL_SEEDS, net_actor(teacher, dev, seed=0))[0])
    _, (pc, pg, _) = run_episodes(PROBE_SEEDS, net_actor(teacher, dev, seed=1), collect=True)
    teacher_probe = info.log_probs(teacher, dev, pc, pg)
    print(f"teacher {os.path.basename(ckpt)}: return {teacher_eval['ret']:.2f}")

    n_batch = a.envs * a.rollout
    roller = Roller(a.envs, a.rollout, 5 * 10**8 + a.seed * 10**7, dev)
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    log.write(json.dumps(dict(type="teacher", ckpt=ckpt, **teacher_eval)) + "\n")
    step, update, wall, next_eval, rows = 0, 0, 0.0, 0, []
    while step <= a.steps:
        if step >= next_eval:
            student.eval()
            ev = summarize(run_episodes(EVAL_SEEDS, net_actor(student, dev, seed=0))[0])
            lp = info.log_probs(student, dev, pc, pg)
            ev.update(type="eval", step=step, wall=wall, KL_teacher_bits=info.kl_bits(teacher_probe, lp),
                      **info.policy_info(lp))
            log.write(json.dumps(ev) + "\n")
            log.flush()
            next_eval += a.eval_every
            print(f"[eval] step {step:>9,}  return {ev['ret']:6.2f}  KL(teacher||student) {ev['KL_teacher_bits']:.3f} bits")
            if step >= a.steps:
                break
        t0 = time.time()
        student.eval()
        roll = roller.collect(student)
        cells = roll["cells"].reshape(n_batch, *roll["cells"].shape[2:])
        globs = roll["glob"].reshape(n_batch, -1)
        lp_t = info.log_probs(teacher, dev, cells, globs).to(dev)
        lp_s = roll["logp_all"].reshape(n_batch, -1)          # the student before training on this batch
        p_t = lp_t.exp()
        pre_ce = float(-(p_t * lp_s).sum(-1).mean() / info.LN2)
        pre_kl = info.kl_bits(lp_t, lp_s)
        h_t = float(-(p_t * lp_t).sum(-1).mean() / info.LN2)
        batch = dict(cells=torch.from_numpy(cells).to(dev), glob=torch.from_numpy(globs).to(dev), p=p_t)
        student.train()
        g = measure_gns(student, a, batch) if a.gns_every and update % a.gns_every == 0 else None
        for _ in range(a.epochs):
            perm = torch.randperm(n_batch, device=dev)
            for k in range(0, n_batch, a.minibatch):
                mb = perm[k:k + a.minibatch]
                opt.zero_grad(set_to_none=True)
                for j in range(0, a.minibatch, a.micro):
                    idx = mb[j:j + a.micro]
                    (soft_ce(student, batch["cells"][idx], batch["glob"][idx], batch["p"][idx]) * a.micro / a.minibatch).backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), a.max_grad)
                opt.step()
        step += n_batch
        update += 1
        wall += time.time() - t0
        fin = roll["finished"]
        row = dict(type="train", step=step, update=update, wall=wall, pre_ce_bits=pre_ce, pre_kl_bits=pre_kl,
                   teacher_H_bits=h_t, n=n_batch, ret=float(np.mean([e["ret"] for e in fin])) if fin else None, gns=g)
        rows.append(row)
        log.write(json.dumps(row) + "\n")

    evals = [json.loads(x) for x in open(os.path.join(a.out, "log.jsonl")) if '"type": "eval"' in x]
    floor = float(np.mean([r["pre_kl_bits"] for r in rows[-3:]]))
    code = dict(cross_entropy_bits=sum(r["pre_ce_bits"] * r["n"] for r in rows),
                excess_kl_bits=sum(r["pre_kl_bits"] * r["n"] for r in rows),
                epiplexity_area_bits=sum((r["pre_kl_bits"] - floor) * r["n"] for r in rows),
                final_kl_floor_bits=floor)
    milestones = {}
    ppo_log = [json.loads(x) for x in open(os.path.join(a.run, "log.jsonl")) if '"type": "eval"' in x]
    ret0 = evals[0]["ret"]
    for frac in (0.5, 0.8, 0.9, 0.95):        # the same absolute return level for both learners
        target = ret0 + frac * (teacher_eval["ret"] - ret0)
        d = next((e["step"] for e in evals if e["ret"] >= target), None)
        q = next((e["step"] for e in sorted(ppo_log, key=lambda e: e["step"]) if e["ret"] >= target), None)
        milestones[f"{frac:.0%} of the gain to the teacher's return ({target:.2f})"] = dict(
            distill_steps=d, ppo_steps=q, rl_tax=(q / d) if d and q else None)
    out = dict(teacher=teacher_eval, final_student=evals[-1], code_length=code, milestones=milestones,
               gns_B_simple=[r["gns"]["B_simple"] for r in rows if r["gns"]])
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(dict(code_length=code, milestones=milestones), indent=1))


if __name__ == "__main__":
    main()
