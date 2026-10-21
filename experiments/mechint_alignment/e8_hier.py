"""E8, option A: rules inferred in the weights -- hierarchical Bayes over a world family. (EXPERIMENT_E8.md)

Each world (128 steps of random play under its own hidden physics) is processed in chunks of 8 steps:
  1. predict the chunk with the world's current fast parameters phi_w (PREQUENTIAL: before adapting on it);
  2. adapt (--adapt): one Kalman / natural-gradient step on the chunk, counted once,
        phi_w <- phi_w - g / (lam0 + n_seen * Fbar)
     g = the gradient of the chunk's summed loss, n_seen = the world's data so far, Fbar = the per-datum Fisher of the
     fast parameters (an EMA over training).
Training maximises the summed prequential log-likelihood of the worlds = their log marginal likelihood under the
hierarchical model (empirical Bayes): Adam on the shared trunk and on the prior mean mu (first-order: d/d mu of each
chunk's loss is taken at phi_t). --adapt off is the no-adaptation baseline (one model for all worlds).
Evaluation on unseen worlds (new permutations) and on held-out parameter combinations: bits per step and per-rule
accuracy against the chunk index, i.e. against experience in the world.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from e2_world import load, losses, robot_mask, stack82
from family import FamilyModel
from transitions import EVENTS

RULES = ("move", "turn", "mine_work", "mined", "theft", "deliver", "interact_full", "push", "robot_burn",
         "trapped_hurt", "button_off", "frozen_tick")
LN2 = math.log(2)


def world_index(z, length=128):
    """[n_worlds, length] row indices of the worlds that ran their full length, each sorted by step."""
    order = np.lexsort((z["tstep"], z["episode"]))
    ep = z["episode"][order]
    _, start, counts = np.unique(ep, return_index=True, return_counts=True)
    return np.stack([order[s:s + length] for s, c in zip(start, counts) if c >= length])


class Runner:
    def __init__(self, model, data, dev, focus=2, chunk=8, lam0=10.0, clamp=0.1):
        self.m, self.z, self.dev, self.focus, self.chunk = model, data, dev, focus, chunk
        self.lam0, self.clamp = lam0, clamp
        self.Fbar = {k: torch.full_like(v, 1e-3) for k, v in model.mu.items()}

    def tensors(self, rows):
        return [torch.from_numpy(self.z[x][rows]).to(self.dev) for x in ("obs", "gob", "act", "nxt", "gnx")]

    def run(self, worlds, adapt, train=False, record=None):
        """One pass over a batch of worlds. Returns the mean prequential bits per step."""
        m, C = self.m, self.chunk
        W, T = worlds.shape
        fast = m.fast_init(W)
        wid = torch.arange(W, device=self.dev).repeat_interleave(C)
        bits = 0.0
        for t in range(T // C):
            rows = worlds[:, t * C:(t + 1) * C].reshape(-1)
            o, g, a, nx, gx = self.tensors(rows)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cl, gl = m.forward_fast(o, g, a, fast, wid)
            loss, sur = losses(cl, gl, nx, gx, per_row=True, mask=robot_mask(o, self.focus))
            bits += float(sur.mean()) / LN2
            if record is not None:
                record(t, cl, gl, rows, sur)
            keys = list(fast)
            if train:
                loss.backward()
                grads = [fast[k].grad for k in keys]
                for k, gk in zip(keys, grads):
                    m.mu[k].grad = gk.sum(0) if m.mu[k].grad is None else m.mu[k].grad + gk.sum(0)
            elif adapt:
                grads = torch.autograd.grad(loss, [fast[k] for k in keys])
            if adapt:
                n_seen = (t + 1) * C
                new = {}
                for k, gk in zip(keys, grads):
                    gsum = gk.detach() * (W * C)                       # the chunk's summed loss, each datum once
                    if train:
                        self.Fbar[k].mul_(0.99).add_(0.01 * (gsum ** 2 / C).mean(0))
                    step = (gsum / (self.lam0 + n_seen * self.Fbar[k])).clamp(-self.clamp, self.clamp)
                    new[k] = (fast[k].detach() - step).requires_grad_()
                fast = new
            elif not train:
                fast = {k: v.detach().requires_grad_() for k, v in fast.items()}
            else:
                fast = {k: v.detach().requires_grad_() for k, v in fast.items()}
        return bits / (T // C)

    def evaluate(self, worlds_all, adapt, batch=32):
        """Per chunk index: mean bits per step, and per rule the fraction of instances whose check entries are right."""
        T = worlds_all.shape[1] // self.chunk
        bits = np.zeros(T)
        hit = {e: np.zeros(T) for e in RULES}
        tot = {e: np.zeros(T) for e in RULES}
        z = self.z

        def record(t, cl, gl, rows, sur):
            bits[t] += float(sur.sum()) / LN2
            pc = torch.stack([x.argmax(-1) for x in cl], -1).cpu()
            pg = torch.stack([x.argmax(-1) for x in gl], -1).cpu()
            pred = stack82(pc, pg)
            tgt = stack82(torch.from_numpy(z["nxt"][rows]).long(), torch.from_numpy(z["gnx"][rows]).long())
            ev = z["events"][rows]
            for e in RULES:
                k = EVENTS.index(e)
                sel = np.nonzero((ev >> k) & 1)[0]
                if len(sel) == 0:
                    continue
                ch = torch.from_numpy(z["checks"][rows[sel], k].astype(np.int64))
                valid = ch[..., 0] >= 0
                rr = torch.from_numpy(sel)[:, None].expand_as(ch[..., 0])
                r_, f_ = ch[..., 0].clamp(min=0), ch[..., 1].clamp(min=0)
                ok = ((pred[rr, r_, f_] == tgt[rr, r_, f_]) | ~valid).all(1)
                hit[e][t] += float(ok.sum())
                tot[e][t] += len(sel)

        self.m.eval()
        n = 0
        for i in range(0, len(worlds_all), batch):
            w = worlds_all[i:i + batch]
            self.run(w, adapt, train=False, record=record)
            n += w.size
        self.m.train()
        rows_per_chunk = n / T
        return dict(bits=(bits / rows_per_chunk).tolist(),
                    acc={e: (hit[e] / np.maximum(tot[e], 1)).tolist() for e in RULES},
                    count={e: tot[e].tolist() for e in RULES})


def summarize(r):
    """Early (chunks 0-1) vs late (chunks 12-15) bits and rule accuracy, pooled over instances."""
    def pooled(e, sl):
        c = np.array(r["count"][e][sl]); a = np.array(r["acc"][e][sl])
        return float((a * c).sum() / c.sum()) if c.sum() else None
    out = dict(bits_early=float(np.mean(r["bits"][:2])), bits_late=float(np.mean(r["bits"][12:])))
    for e in RULES:
        out[e] = (pooled(e, slice(0, 2)), pooled(e, slice(12, 16)), int(sum(r["count"][e])))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--adapt", action="store_true")
    p.add_argument("--lam0", type=float, default=10.0)
    p.add_argument("--clamp", type=float, default=0.1)
    p.add_argument("--worlds-per-batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--time-limit", type=float, default=420.0)
    p.add_argument("--eval-worlds", type=int, default=128)
    p.add_argument("--data", default="runs/data_family")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    train, test, held = (load(os.path.join(a.data, f)) for f in ("train.npz", "test.npz", "heldout.npz"))
    wtr, wte, whe = world_index(train), world_index(test)[:a.eval_worlds], world_index(held)[:a.eval_worlds]
    print(f"worlds: train {len(wtr)}, test {len(wte)}, held-out {len(whe)}", flush=True)
    model = FamilyModel().to(dev)
    opt = torch.optim.Adam(model.shared_parameters() + list(model.mu.parameters()), lr=a.lr)
    run_tr, run_te, run_he = (Runner(model, z, dev, lam0=a.lam0, clamp=a.clamp) for z in (train, test, held))
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)

    def do_eval(tag):
        for name, rn, w in (("test", run_te, wte), ("heldout", run_he, whe)):
            rn.Fbar = run_tr.Fbar
            for adapt in ([True, False] if a.adapt else [False]):
                r = rn.evaluate(w, adapt)
                s = summarize(r)
                log.write(json.dumps(dict(type="eval", tag=tag, split=name, adapt=adapt, summary=s, curves=r)) + "\n")
                log.flush()
                print(f"[{tag}] {name:7s} adapt={adapt!s:5s}  bits early {s['bits_early']:.2f} late {s['bits_late']:.2f}  "
                      f"move {s['move'][0]:.2f}->{s['move'][1]:.2f}  turn {s['turn'][0]:.2f}->{s['turn'][1]:.2f}  "
                      f"burn {s['robot_burn'][0]}->{s['robot_burn'][1]}  push {s['push'][0]}->{s['push'][1]}", flush=True)

    do_eval("start")
    t_train, step, half_done = 0.0, 0, False
    while t_train < a.time_limit:
        t0 = time.time()
        idx = rng.choice(len(wtr), a.worlds_per_batch, replace=False)
        opt.zero_grad(set_to_none=True)
        bits = run_tr.run(wtr[idx], a.adapt, train=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        torch.cuda.synchronize()
        t_train += time.time() - t0
        if step % 20 == 0:
            log.write(json.dumps(dict(type="train", step=step, t=t_train, bits=bits)) + "\n")
        if step % 50 == 0:
            print(f"step {step}  {t_train:.0f}s  train prequential bits/step {bits:.2f}", flush=True)
        if not half_done and t_train > a.time_limit / 2:
            do_eval("mid")
            half_done = True
    do_eval("end")
    torch.save(dict(model=model.state_dict(), Fbar=run_tr.Fbar), os.path.join(a.out, "model.pt"))


if __name__ == "__main__":
    main()
