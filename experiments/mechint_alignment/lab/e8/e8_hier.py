"""E8, option A: rules inferred in the weights -- hierarchical Bayes over a world family. (EXPERIMENT_E8.md)

Each world (random play under its own hidden physics) is processed in chunks of 8 steps:
  1. predict the chunk with the world's current fast parameters (PREQUENTIAL: before adapting on it). With
     --prior-avg, the first --sample-chunks chunks are predicted by the posterior PREDICTIVE: the average over --samples
     draws phi ~ N(phi_w, 1/Lambda_w), Lambda_w = 1/sigma^2 + n_seen * Fbar, with sigma the learned prior spread
     (the Bayes-correct prediction while the world is still uncertain; the likelihood is the chunk's joint mixture);
  2. adapt (--adapt): one Kalman / natural-gradient step on the chunk, counted once,
        phi_w <- phi_w - g / (prior precision + n_seen * Fbar),   prior precision = 1/sigma^2 (--prior-avg) or lam0.
Training maximises the summed prequential log-likelihood of the worlds = their log marginal likelihood under the
hierarchical model (empirical Bayes) over the trunk, the prior mean mu and (with --prior-avg) the prior spread sigma.
First-order in mu; a truncated outer update every --outer-every chunks. --adapt off = one model for all worlds.
Evaluation on unseen worlds and held-out parameter combinations: bits per step and per-rule accuracy against the chunk
index, i.e. against experience in the world.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from lab.e2.e2_world import load, losses, robot_mask
from lab.e8.e8_score import RULES, ChunkScorer
from nets.family import FamilyModel
from worlds.village import SPEC

LN2 = math.log(2)


def world_index(z, length=128):
    """[n_worlds, length] row indices of the worlds that ran their full length, each sorted by step."""
    order = np.lexsort((z["tstep"], z["episode"]))
    ep = z["episode"][order]
    _, start, counts = np.unique(ep, return_index=True, return_counts=True)
    return np.stack([order[s:s + length] for s, c in zip(start, counts) if c >= length])


class Runner:
    def __init__(self, model, data, dev, focus=2, chunk=8, lam0=10.0, clamp=0.1, prior_avg=False, samples=4,
                 sample_chunks=4, em_sigma=False, em_rate=0.2):
        self.m, self.z, self.dev, self.focus, self.chunk = model, data, dev, focus, chunk
        self.lam0, self.clamp = lam0, clamp
        self.prior_avg, self.samples, self.sample_chunks = prior_avg, samples, sample_chunks
        self.em_sigma, self.em_rate = em_sigma, em_rate
        self.Fbar = {k: torch.full_like(v, 1e-3) for k, v in model.mu.items()}

    def tensors(self, rows):
        return [torch.from_numpy(self.z[x][rows]).to(self.dev) for x in ("obs", "gob", "act", "nxt", "gnx")]

    def prior_prec(self, k):
        return torch.exp(-2 * self.m.log_sigma[k]) if (self.prior_avg or self.em_sigma) else self.lam0

    @torch.no_grad()
    def em_update(self, fast, n_total):
        """Empirical Bayes in closed form (the EM M-step of a Gaussian hierarchical prior): the prior variance is the
        between-world variance of the adapted parameters plus their mean remaining posterior variance,
            sigma^2 <- (1 - rate) sigma^2 + rate * [ mean_w (phi_w - mu)^2 + 1 / (1/sigma^2 + n_total Fbar) ]."""
        for k, v in fast.items():
            sig2 = torch.exp(2 * self.m.log_sigma[k])
            post = 1.0 / (1.0 / sig2 + n_total * self.Fbar[k])
            est = ((v.detach() - self.m.mu[k]) ** 2).mean(0) + post
            new = ((1 - self.em_rate) * sig2 + self.em_rate * est).clamp(1e-6, 1.0)
            self.m.log_sigma[k].copy_(0.5 * torch.log(new))

    def kalman_step(self, fast, grads, W, C, n_seen, train=False):
        """One Bayesian step of every world's fast parameters on a chunk (each datum counted once):
        phi <- phi - g_sum / (prior precision + n_seen * Fbar), clamped. grads: of the chunk's mean loss over W*C rows."""
        new = {}
        for k, gk in grads.items():
            gsum = gk.detach() * (W * C)                       # the chunk's summed loss, each datum once
            if train:
                self.Fbar[k].mul_(0.99).add_(0.01 * (gsum ** 2 / C).mean(0))
            prec = self.prior_prec(k)
            prec = prec.detach() if torch.is_tensor(prec) else prec
            step = (gsum / (prec + n_seen * self.Fbar[k])).clamp(-self.clamp, self.clamp)
            new[k] = (fast[k].detach() - step).requires_grad_()
        return new

    def run(self, worlds, adapt, train=False, scorer=None, opt=None, outer_every=16):
        """One pass over a batch of worlds. Returns the mean prequential bits per step."""
        m, C = self.m, self.chunk
        W, T = worlds.shape
        fast = m.fast_init(W)
        wid = torch.arange(W, device=self.dev).repeat_interleave(C)
        n_chunks, bits, pending = T // C, 0.0, False
        for t in range(n_chunks):
            rows = worlds[:, t * C:(t + 1) * C].reshape(-1)
            o, g, a, nx, gx = self.tensors(rows)
            mask = robot_mask(o, self.focus)
            keys = list(fast)
            K = self.samples if (self.prior_avg and t < self.sample_chunks) else 1
            if K > 1:                                        # the posterior predictive, a joint mixture per world
                n_before = t * C
                std = {k: (self.prior_prec(k) + n_before * self.Fbar[k]).rsqrt() for k in keys}
                L_train, L_sur, probs = [], [], None
                for _ in range(K):
                    fk = {k: fast[k] + std[k] * torch.randn_like(fast[k]) for k in keys}
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        cl, gl = m.forward_fast(o, g, a, fk, wid)
                    tr, sr = losses(cl, gl, nx, gx, per_row="rows", mask=mask)
                    L_train.append(-tr.view(W, C).sum(1))
                    L_sur.append(-sr.view(W, C).sum(1))
                    if scorer is not None:
                        p = [x.float().softmax(-1).detach() for x in cl + gl]
                        probs = p if probs is None else [a_ + b_ for a_, b_ in zip(probs, p)]
                lse = lambda L: torch.logsumexp(torch.stack(L), 0) - math.log(K)
                loss = -lse(L_train).sum() / (W * C)
                sur = -lse(L_sur)                            # [W]: each world's chunk, in nats
                bits += float(sur.sum()) / (W * C) / LN2
                if scorer is not None:
                    nc = len(cl)
                    scorer.add(t, [x.log() for x in probs[:nc]], [x.log() for x in probs[nc:]], rows, sur)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cl, gl = m.forward_fast(o, g, a, fast, wid)
                loss, sur = losses(cl, gl, nx, gx, per_row=True, mask=mask)
                bits += float(sur.mean()) / LN2
                if scorer is not None:
                    scorer.add(t, cl, gl, rows, sur)
            if train:
                loss.backward()
                grads = [fast[k].grad for k in keys]
                for k, gk in zip(keys, grads):
                    m.mu[k].grad = gk.sum(0) if m.mu[k].grad is None else m.mu[k].grad + gk.sum(0)
                pending = True
            elif adapt:
                grads = torch.autograd.grad(loss, [fast[k] for k in keys])
            if adapt:
                fast = self.kalman_step(fast, dict(zip(keys, grads)), W, C, (t + 1) * C, train)
            else:
                fast = {k: v.detach().requires_grad_() for k, v in fast.items()}
            if train and ((t + 1) % outer_every == 0 or t == n_chunks - 1):   # truncated meta-update
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                pending = False
        if train and self.em_sigma and adapt:
            self.em_update(fast, n_chunks * C)
        return bits / n_chunks

    def evaluate(self, worlds_all, adapt, batch=32):
        sc = ChunkScorer(self.z, worlds_all.shape[1] // self.chunk)
        self.m.eval()
        torch.manual_seed(1)
        for i in range(0, len(worlds_all), batch):
            self.run(worlds_all[i:i + batch], adapt, train=False, scorer=sc)
        self.m.train()
        return sc.result()


def summarize(r):
    """Early (first 2 chunks), late (last 4 chunks) and whole-world mean bits; per rule, early and late accuracy."""
    def pooled(e, sl):
        c = np.array(r["count"][e][sl]); a = np.array(r["acc"][e][sl])
        return float((a * c).sum() / c.sum()) if c.sum() else None
    out = dict(bits_early=float(np.mean(r["bits"][:2])), bits_late=float(np.mean(r["bits"][-4:])),
               bits_mean=float(np.mean(r["bits"])))
    for e in RULES:
        out[e] = (pooled(e, slice(0, 2)), pooled(e, slice(-4, None)), int(sum(r["count"][e])))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--adapt", action="store_true")
    p.add_argument("--prior-avg", action="store_true")
    p.add_argument("--em-sigma", action="store_true", help="the prior variance by closed-form empirical Bayes (EM)")
    p.add_argument("--em-rate", type=float, default=0.2)
    p.add_argument("--sigma-init", type=float, default=0.3)
    p.add_argument("--resume", default=None, help="a checkpoint to continue training from")
    p.add_argument("--prev-seconds", type=float, default=0.0, help="training seconds already in a checkpoint that does not record them")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--sample-chunks", type=int, default=4)
    p.add_argument("--lam0", type=float, default=10.0)
    p.add_argument("--clamp", type=float, default=0.1)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--outer-every", type=int, default=16)
    p.add_argument("--worlds-per-batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--time-limit", type=float, default=420.0)
    p.add_argument("--eval-worlds", type=int, default=128)
    p.add_argument("--evals", default="start,mid,end")
    p.add_argument("--data", default="runs/data_family")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    train, test, held = (load(os.path.join(a.data, f)) for f in ("train.npz", "test.npz", "heldout.npz"))
    wtr = world_index(train, a.length)
    wte, whe = world_index(test, a.length)[:a.eval_worlds], world_index(held, a.length)[:a.eval_worlds]
    print(f"worlds: train {len(wtr)}, test {len(wte)}, held-out {len(whe)} (length {a.length})", flush=True)
    model = FamilyModel(SPEC).to(dev)
    learn_sigma = a.prior_avg and not a.em_sigma                     # by gradient (E8b) unless set by EM
    params = model.shared_parameters() + list(model.mu.parameters()) + (list(model.log_sigma.parameters()) if learn_sigma else [])
    opt = torch.optim.Adam(params, lr=a.lr)
    kw = dict(lam0=a.lam0, clamp=a.clamp, prior_avg=a.prior_avg, samples=a.samples, sample_chunks=a.sample_chunks,
              em_sigma=a.em_sigma, em_rate=a.em_rate)
    run_tr, run_te, run_he = (Runner(model, z, dev, **kw) for z in (train, test, held))
    t_prev, step_prev = 0.0, 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        run_tr.Fbar = ck["Fbar"]
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        t_prev, step_prev = ck.get("t_train", a.prev_seconds), ck.get("step", 0)
        print(f"resumed from {a.resume}: {t_prev:.0f}s of training, {step_prev} batches", flush=True)
    elif a.em_sigma:
        with torch.no_grad():
            for v in model.log_sigma.values():
                v.fill_(math.log(a.sigma_init))
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w")
    json.dump(vars(a), open(os.path.join(a.out, "config.json" if not a.resume else f"config_resume_{int(t_prev)}.json"), "w"), indent=1)
    evals = a.evals.split(",")

    def do_eval(tag):
        if tag not in evals:
            return
        for name, rn, w in (("test", run_te, wte), ("heldout", run_he, whe)):
            rn.Fbar = run_tr.Fbar
            for adapt in ([True, False] if a.adapt else [False]):
                r = rn.evaluate(w, adapt)
                s = summarize(r)
                log.write(json.dumps(dict(type="eval", tag=tag, split=name, adapt=adapt, summary=s, curves=r)) + "\n")
                log.flush()
                print(f"[{tag}] {name:7s} adapt={adapt!s:5s}  bits early {s['bits_early']:.2f} late {s['bits_late']:.2f} "
                      f"mean {s['bits_mean']:.2f}  move {s['move'][0]:.2f}->{s['move'][1]:.2f}  "
                      f"turn {s['turn'][0]:.2f}->{s['turn'][1]:.2f}", flush=True)

    do_eval("start")
    t_train, step, half_done = t_prev, step_prev, False
    t_stop = t_prev + a.time_limit
    while t_train < t_stop:
        t0 = time.time()
        idx = rng.choice(len(wtr), a.worlds_per_batch, replace=False)
        bits = run_tr.run(wtr[idx], a.adapt, train=True, opt=opt, outer_every=a.outer_every)
        step += 1
        torch.cuda.synchronize()
        t_train += time.time() - t0
        log.write(json.dumps(dict(type="train", step=step, t=t_train, bits=bits)) + "\n")
        if step % 10 == 0:
            sig = float(torch.cat([v.detach().flatten() for v in model.log_sigma.values()]).exp().mean()) if (a.prior_avg or a.em_sigma) else None
            print(f"batch {step}  {t_train:.0f}s  train prequential bits/step {bits:.2f}  prior sigma {sig}", flush=True)
        if not half_done and t_train > t_prev + a.time_limit / 2:
            do_eval("mid")
            half_done = True
    do_eval("end")
    torch.save(dict(model=model.state_dict(), Fbar=run_tr.Fbar, opt=opt.state_dict(), t_train=t_train, step=step),
               os.path.join(a.out, "model.pt"))


if __name__ == "__main__":
    main()
