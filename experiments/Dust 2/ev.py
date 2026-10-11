"""Experiment 4 (EXPERIMENT_4.md): event-driven hidden units, and how far a sparse perturbation cascades.

  python ev.py train 0.05        train EV5 by backprop (thresholds by homeostasis) -> runs/ckpt_ev5.pt
  python ev.py measure           the cascade, avalanche sizes, rerun-cost fraction (DENSE, EV5, EV2) and the
                                 estimator cosines on the EV models

Event-driven unit: z_j = (h_j - theta_j)_+^2. theta starts, block by block, at each unit's (1 - target) quantile of h
over a batch; then after every step theta_j += ETA * (rate_j - target), with rate_j the unit's firing fraction in the
batch (homeostasis: a unit firing too often raises its own threshold). Elementwise, no sort.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

import data
from e1_m1 import k_eq
from measure import measure
from model import TinyGPT, rms
from train import parse_cfg, val_loss

ETA = 0.05
OUT = Path("runs/e4")


def train_ev(target, steps=300, lr=3e-3, seed=0, dev="cuda"):
    torch.manual_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    gv = torch.Generator(device=dev).manual_seed(999)
    tr, va = data.load(dev)
    vb = [data.batch(va, 64, 64, gv) for _ in range(8)]
    m = TinyGPT(data.V, ev=True).to(dev)
    with torch.no_grad():                                  # thresholds at the target quantile, block by block
        idx, tgt = data.batch(tr, 32, 64, g)
        for b in range(m.L):
            h = m.forward_cache(idx, tgt)[("fc", b)].reshape(-1, m.m)
            m.theta[b] = torch.quantile(h, 1 - target, dim=0)
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.99))
    log = []
    for step in range(1, steps + 1):
        f = step / 20 if step <= 20 else 0.5 * (1 + math.cos(math.pi * (step - 20) / (steps - 20)))
        for gp in opt.param_groups:
            gp["lr"] = lr * f
        idx, tgt = data.batch(tr, 32, 64, g)
        c = m.forward_cache(idx, tgt)
        loss = c["loss"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            rates = []
            for b in range(m.L):
                rate = (c[("fc", b)].detach() > m.theta[b]).float().mean((0, 1))
                m.theta[b] += ETA * (rate - target)
                rates.append(float(rate.mean()))
        if step % 50 == 0 or step == steps:
            v = val_loss(m, vb)
            log.append(dict(step=step, val=round(v, 4), rate=[round(r, 4) for r in rates]))
            print(f"step {step:4d}  val {v:.4f}  firing rate by block {' '.join(f'{r:.3f}' for r in rates)}", flush=True)
    name = f"ev{round(target * 100)}"
    torch.save(m.state_dict(), f"runs/ckpt_{name}.pt")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(target=target, eta=ETA, steps=steps, lr=lr, log=log), open(OUT / f"train_{name}.json", "w"), indent=1)


@torch.no_grad()
def record(m, idx, tgt, site_b=None, noise=None):
    """Forward in fp32 recording every layer; optionally add `noise` to block site_b's MLP output (a residual writer)."""
    T = idx.shape[1]
    x = rms(m.wte[idx] + m.wpe[:T][None])
    rec = {}
    for b in range(m.L):
        o = m.attend(rms(x) @ m.Wqkv[b].T)
        rec["o", b] = o
        x = x + o @ m.Wproj[b].T
        rec["xa", b] = x
        h = rms(x) @ m.Wfc[b].T
        z = m.act(h, b)
        rec["h", b], rec["z", b] = h, z
        mo = z @ m.Wout[b].T
        if b == site_b:
            mo = mo + noise
        x = x + mo
        rec["xm", b] = x
    rec["logit"] = rms(x) @ m.Whead.T
    rec["loss"] = m.tok_loss(rec["logit"], tgt)
    return rec


@torch.no_grad()
def rerun_sod(m, tgt, clean, noise, tol):
    """Rerun after block 0's MLP output (+ noise). tol=None: exact. Otherwise SEND-ON-DELTA: after every operation a change
    smaller than tol x the clean tensor's rms is dropped (the clean value is kept). Returns per-token loss, fractions."""
    fr = {}

    def keep(key, y):
        if tol is None:
            return y
        c = clean[key]
        mk = (y - c).abs() > tol * c.pow(2).mean().sqrt()
        fr[f"{key[0]}{key[1]}" if isinstance(key, tuple) else key] = float(mk.float().mean())
        return c + (y - c) * mk
    x = keep(("xm", 0), clean["xm", 0] + noise)
    for b in range(1, m.L):
        o = keep(("o", b), m.attend(rms(x) @ m.Wqkv[b].T))
        x = keep(("xa", b), x + o @ m.Wproj[b].T)
        h = keep(("h", b), rms(x) @ m.Wfc[b].T)
        z = keep(("z", b), m.act(h, b))
        x = keep(("xm", b), x + z @ m.Wout[b].T)
    logit = keep("logit", rms(x) @ m.Whead.T)
    return m.tok_loss(logit, tgt), fr


def fidelity(dev="cuda"):
    """Experiment 4b: exact vs send-on-delta rewards for sparse (4 of 64) and dense (64) probes at block 0's MLP output."""
    from estimators import backprop
    tr, va = data.load(dev)
    gb = torch.Generator(device=dev).manual_seed(123)
    batches = [data.batch(va, 32, 64, gb) for _ in range(2)]
    res = {}
    for name, ck, ev in [("DENSE", "runs/ckpt_bp300.pt", False), ("EV5", "runs/ckpt_ev5.pt", True)]:
        m = TinyGPT(data.V, ev=ev).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev))
        for mc in (4, 64):
            g = torch.Generator(device=dev).manual_seed(0)
            acc = {t: dict(ex=[], sod=[], fr=[]) for t in (1e-4, 1e-3, 1e-2)}
            cos_ex, cos_sod = [], {t: [] for t in acc}
            for idx, tgt in batches:
                B, T = idx.shape
                clean = record(m, idx, tgt)
                true = backprop(m, idx, tgt)[0][m.Wout[0]]
                z0 = clean["z", 0]
                A, Rex, Rsod = [], [], {t: [] for t in acc}
                for k in range(32):
                    a = torch.zeros(B, T, m.d, device=dev)
                    pick = torch.rand(B, T, m.d, generator=g, device=dev).argsort(-1)[..., :mc]
                    a.scatter_(-1, pick, torch.randn(B, T, mc, generator=g, device=dev) * (m.d / mc) ** 0.5)
                    lex, _ = rerun_sod(m, tgt, clean, 0.2 * a, None)
                    A.append(a)
                    Rex.append(clean["loss"] - lex)
                    for t in acc:
                        ls, fr = rerun_sod(m, tgt, clean, 0.2 * a, t)
                        Rsod[t].append(clean["loss"] - ls)
                        acc[t]["ex"].append((clean["loss"] - lex).flatten())
                        acc[t]["sod"].append((clean["loss"] - ls).flatten())
                        acc[t]["fr"].append(fr)
                A = torch.stack(A)

                def est(R):
                    R = torch.stack(R)
                    R = R - R.mean(0, keepdim=True)
                    gh = -(R[..., None] * A).sum(0) / (32 * 0.2)
                    G = torch.einsum("btd,bti->di", gh, z0) / (B * T)
                    return float((G * true).sum() / (G.norm() * true.norm()))
                cos_ex.append(est(Rex))
                for t in acc:
                    cos_sod[t].append(est(Rsod[t]))
            out = {}
            for t, v in acc.items():
                ex, so = torch.cat(v["ex"]), torch.cat(v["sod"])
                corr = float(torch.corrcoef(torch.stack([ex, so]))[0, 1])
                frs = {k: sum(f[k] for f in v["fr"]) / len(v["fr"]) for k in v["fr"][0]}
                out[str(t)] = dict(corr=round(corr, 4), cos=round(sum(cos_sod[t]) / len(cos_sod[t]), 4),
                                   cost_fraction=rerun_cost_fraction(m, frs),
                                   changed_stream=[round(frs[f"xm{b}"], 3) for b in range(m.L)])
            r = dict(cos_exact=round(sum(cos_ex) / len(cos_ex), 4), by_tol=out)
            res[f"{name}_m{mc}"] = r
            print(f"{name} probe m={mc:2d}: exact-reward cos {r['cos_exact']:.3f} | " + " | ".join(
                f"tol {t}: corr {o['corr']:.3f} cos {o['cos']:.3f} cost {o['cost_fraction']:.2f}" for t, o in out.items()),
                flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "fidelity.json", "w"), indent=1)


def changed(a, b, tol=1e-3):
    """Entries that changed by more than tol x the clean tensor's rms."""
    return (a - b).abs() > tol * a.pow(2).mean().sqrt()


def cascade(m, batches, g, m_coords=4, sig=0.2, one_token=None):
    """Perturb m_coords random coordinates of block 0's MLP output per token (or only at token `one_token`)."""
    keys = [("o", b) for b in range(1, m.L)] + [("xa", b) for b in range(m.L)] + \
           [("h", b) for b in range(1, m.L)] + [("z", b) for b in range(1, m.L)] + [("xm", b) for b in range(m.L)]
    frac = {k: [] for k in keys}
    act = {b: [] for b in range(m.L)}
    sizes = []
    for idx, tgt in batches:
        B, T = idx.shape
        clean = record(m, idx, tgt)
        noise = torch.zeros(B, T, m.d, device=idx.device)
        pick = torch.rand(B, T, m.d, generator=g, device=idx.device).argsort(-1)[..., :m_coords]
        noise.scatter_(-1, pick, sig * torch.randn(B, T, m_coords, generator=g, device=idx.device))
        if one_token is not None:
            noise[:, :one_token] = 0
            noise[:, one_token + 1:] = 0
        pert = record(m, idx, tgt, site_b=0, noise=noise)
        for k in keys:
            frac[k].append(float(changed(clean[k], pert[k]).float().mean()))
        for b in range(m.L):
            act[b].append(float((clean["z", b] > 0).float().mean()))
        zc = sum(changed(clean["z", b], pert["z", b]).float().sum(-1) for b in range(1, m.L))  # (B, T)
        sizes.append(zc.sum(-1) if one_token is not None else zc.flatten())
    s = torch.cat(sizes).float()
    avg = {f"{k[0]}{k[1]}": round(sum(v) / len(v), 4) for k, v in frac.items()}
    br = [round(avg[f"z{b + 1}"] / max(avg[f"z{b}"], 1e-9), 3) for b in range(1, m.L - 1)]
    return dict(changed=avg, active={b: round(sum(v) / len(v), 4) for b, v in act.items()}, branching_z=br,
                avalanche=dict(mean=round(float(s.mean()), 1), median=round(float(s.median()), 1),
                               p99=round(float(s.quantile(0.99)), 1), max=round(float(s.max()), 1)))


def rerun_cost_fraction(m, ch):
    """Multiply-adds of a rerun after block 0's MLP output, counting each matmul at its changed-input fraction,
    attention in full; relative to the full rerun."""
    d, mm, T, V = m.d, m.m, m.T, m.V
    full = part = 0.0
    for b in range(1, m.L):
        ops = [(3 * d * d, ch[f"xm{b - 1}"]),                # qkv reads rms(x): a token changes wholly if any entry does
               (T * d, 1.0),                                 # attention itself
               (d * d, ch[f"o{b}"]),                         # projection reads o
               (d * mm, ch[f"xa{b}"]),                       # MLP in reads rms(x after attention)
               (mm * d, ch[f"z{b}"])]                        # MLP out reads z: the event-driven saving
        for f, c in ops:
            full += f
            part += f * c
    full += d * V
    part += d * V * ch[f"xm{m.L - 1}"]
    return round(part / full, 3)


def measure_all(dev="cuda"):
    OUT.mkdir(parents=True, exist_ok=True)
    TinyGPT.fast = True
    tr, va = data.load(dev)
    gb = torch.Generator(device=dev).manual_seed(123)
    batches = [data.batch(va, 32, 64, gb) for _ in range(4)]
    res = {}
    for name, ck, ev in [("DENSE", "runs/ckpt_bp300.pt", False), ("EV5", "runs/ckpt_ev5.pt", True),
                         ("EV2", "runs/ckpt_ev2.pt", True)]:
        m = TinyGPT(data.V, ev=ev).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev))
        g = torch.Generator(device=dev).manual_seed(0)
        r = dict(val=round(val_loss(m, batches), 4))
        r["all_tokens"] = cascade(m, batches, g)
        r["one_token"] = cascade(m, batches, g, one_token=32)
        r["rerun_cost_fraction"] = rerun_cost_fraction(m, r["all_tokens"]["changed"])
        if ev:                                               # estimator quality on the EV model, against its own curve
            curve = {k: measure(m, parse_cfg([f"K={k}", "amp=fp16"]), batches, 0, dev)["mean"]
                     for k in (8, 16, 32, 64, 128, 256, 512)}
            o1 = measure(m, parse_cfg(["K=32", "amp=fp16", "sparse_c=2"]), batches, 0, dev)
            r.update(curve=curve, O1_cos=o1["mean"], O1_gain=round(k_eq(o1["mean"], curve) / 32, 2),
                     O1_fc=[o1["cos"][f"fc{b}"] for b in range(m.L)])
        else:
            r["curve_K32"] = json.load(open("runs/e1/curve/latin_K32.json"))["mean"]   # reused, not re-run
        res[name] = r
        a = r["all_tokens"]
        print(f"{name}: val {r['val']}  active z by block {a['active']}  rerun-cost fraction {r['rerun_cost_fraction']}")
        print(f"   changed (all tokens perturbed): {a['changed']}")
        print(f"   branching (z) {a['branching_z']}  avalanche per token {a['avalanche']}")
        print(f"   one-token perturbation: changed {r['one_token']['changed']}  avalanche per sequence "
              f"{r['one_token']['avalanche']}")
        if ev:
            print(f"   estimator: base cos K32 {r['curve'][32]:.3f}  O1 cos {r['O1_cos']:.3f}  O1 gain {r['O1_gain']}", flush=True)
        json.dump(res, open(OUT / "measure.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train_ev(float(sys.argv[2]))
    elif sys.argv[1] == "fidelity":
        fidelity()
    else:
        measure_all()
