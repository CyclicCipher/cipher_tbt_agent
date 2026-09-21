"""One arm of the inner-objective experiment: the same model, task and outer loss as `h1_lid.py`, with the update
rule swapped.

    A   AdamW on the mean gradient (the baseline)          Ap  A with 3 noise environments in the mix
    B   AdamW on the `ftest+wd` gradient (plan section 4)   Bp  B with 3 noise environments

Every step draws, for each environment (training composition), two independent half-batches and computes one gradient
per half -- both arms see the identical sequences and the identical per-half gradients; the ONLY difference is how the
halves are combined into the gradient AdamW is handed (A: their mean; B: trust environments that agree with themselves,
keep coordinates whose across-environment variance is within `c` times their within-environment variance, decay the
rest toward zero). Everything the analysis needs is recorded: the mask and trust telemetry, full weight snapshots, and
the sufficient statistics of the participation matrix (per-environment gradient vs the update actually taken).

    python experiments/inner_objective/train.py --arm B --steps 2000 --out runs/B
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_T = Path(__file__).resolve().parent.parent / "transformers"
if str(_T) not in sys.path:
    sys.path.insert(0, str(_T))
from h1_lid import Model, out_mask, per_demo_exact  # noqa: E402
import tasks as TASKS  # noqa: E402

L = TASKS.L
N_NOISE = 3


# ── data: one batch per environment, split into two independent halves ─────────────────────────────────────────────
def env_batch(task, pairs, n_noise, n_per_env, K, dev, g):
    """(E, n_per_env, K*2*L) tokens: environments 0..len(pairs)-1 are tasks of `task`, the last `n_noise` are noise
    environments whose outputs are FRESH random digits every time -- learnable by nothing, the noisy TV."""
    E = len(pairs) + n_noise
    x = torch.randint(0, task.V, (E, n_per_env, K, L), generator=g, device=dev)
    y = torch.empty_like(x)
    for e, pair in enumerate(pairs):
        y[e] = task.apply(x[e], pair)
    if n_noise:
        y[len(pairs):] = torch.randint(0, task.V, (n_noise, n_per_env, K, L), generator=g, device=dev)
    return torch.stack([x, y], dim=3).reshape(E, n_per_env, K * 2 * L)


@torch.no_grad()
def trials_to_criterion(model, task, pairs, K, dev, g, crit=0.8, n=256):
    """The bee measure: demonstrations before the query is right >= crit of the time; K (censored) if never."""
    out = []
    for pair in pairs:
        tok = env_batch(task, [pair], 0, n, K, dev, g)[0]
        acc = per_demo_exact(model, tok, K)
        hit = (acc >= crit).nonzero()
        out.append((pair, int(hit[0]) if len(hit) else K, float(acc[-1])))
    return out


# ── per-group gradients ─────────────────────────────────────────────────────────────────────────────────────────────
class Flat:
    """Views into the model's parameters as one flat vector, so masks and statistics are per coordinate."""

    def __init__(self, model):
        self.params = [p for p in model.parameters()]
        self.sizes = [p.numel() for p in self.params]
        self.P = sum(self.sizes)
        self.names = [n for n, _p in model.named_parameters()]
        self.offsets = np.cumsum([0] + self.sizes)

    def get(self):
        return torch.cat([p.detach().reshape(-1) for p in self.params])

    def set_grad(self, flat):
        i = 0
        for p, s in zip(self.params, self.sizes):
            p.grad = flat[i:i + s].view_as(p).clone()
            i += s

    def assign(self, flat):
        i = 0
        with torch.no_grad():
            for p, s in zip(self.params, self.sizes):
                p.copy_(flat[i:i + s].view_as(p))
                i += s

    def scale_masked(self, mask, factor):
        """Multiply masked (mask == True) coordinates by `factor`, in place."""
        i = 0
        with torch.no_grad():
            for p, s in zip(self.params, self.sizes):
                m = mask[i:i + s].view_as(p)
                p.mul_(torch.where(m, torch.full_like(p, factor), torch.ones_like(p)))
                i += s

    def layout(self):
        return [dict(name=n, start=int(a), end=int(b)) for n, a, b in zip(self.names, self.offsets[:-1], self.offsets[1:])]


def group_grads_loop(model, flat, toks, K, amp, V):
    """One gradient per (environment, half) by a plain loop of backward passes: exact, simple, launch-bound (measured
    344 ms/step at 17 environments). The fallback if `vmap` is unavailable."""
    E = toks.shape[0]
    out = torch.empty(E, 2, flat.P, device=toks.device)
    m = out_mask(K, toks.device)[1:]
    for e in range(E):
        for h in range(2):
            tok = toks[e, h]
            with amp:
                logits = model(tok)[:, :-1]
            loss = F.cross_entropy(logits[:, m].reshape(-1, V).float(), tok[:, 1:][:, m].reshape(-1))
            grads = torch.autograd.grad(loss, flat.params)
            out[e, h] = torch.cat([g.reshape(-1) for g in grads])
    return out


class GroupGrads:
    """All 2E per-group gradients in ONE batched forward/backward via `torch.func.vmap(grad(...))`. Same numbers as
    the loop (checked at construction), ~20x the speed. Boolean indexing is replaced by `index_select` because vmap
    cannot batch data-dependent shapes."""

    def __init__(self, model, flat, K, dev, amp, V):
        from torch.func import functional_call, grad, vmap
        self.model, self.flat, self.amp, self.V = model, flat, amp, V
        self.names = flat.names
        self.buffers = dict(model.named_buffers())
        self.idx = out_mask(K, dev)[1:].nonzero().squeeze(1)

        def loss_fn(params, tok):
            logits = functional_call(model, (params, self.buffers), (tok,))[:, :-1]
            lg = logits.index_select(1, self.idx).reshape(-1, self.V).float()
            tg = tok[:, 1:].index_select(1, self.idx).reshape(-1)
            return F.cross_entropy(lg, tg)
        self.vg = vmap(grad(loss_fn), in_dims=(None, 0))
        self.ok = True
        try:                                                     # verify against the loop once, in fp32, then trust it
            toks = torch.randint(0, V, (2, 2, 4, K * 2 * L), device=dev)
            off = torch.autocast("cuda", enabled=False)
            amp_saved, self.amp = self.amp, off
            a, b = self(toks), group_grads_loop(model, flat, toks, K, off, V)
            self.amp = amp_saved
            self.ok = bool(torch.allclose(a, b, rtol=1e-4, atol=1e-6))
        except Exception as e:                                    # noqa: BLE001
            print(f"vmap unavailable ({type(e).__name__}: {str(e)[:80]}); using the loop", flush=True)
            self.ok = False

    def __call__(self, toks):
        E, H, n, T = toks.shape
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        # The fused attention kernels have no vmap batching rule for their backward (functorch loops them per group,
        # silently); the MATH backend is plain matmul + softmax, which vmap batches natively.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with self.amp, sdpa_kernel([SDPBackend.MATH]):
            g = self.vg(params, toks.reshape(E * H, n, T))
        return torch.cat([g[k].reshape(E * H, -1) for k in self.names], dim=1).reshape(E, H, -1).float()


def group_grads(model, flat, toks, K, amp, engine, V):
    if engine is not None and engine.ok:
        return engine(toks)
    return group_grads_loop(model, flat, toks, K, amp, V)


# ── the inner objective ─────────────────────────────────────────────────────────────────────────────────────────────
def combine(G, arm, c, trust_min, eps=1e-12, gate=True, keep_q=0.0, mask_grad=True):
    """(E, 2, P) per-half gradients -> the gradient handed to AdamW, plus telemetry. `gate=False` is the baseline
    rule even in a B arm (the warm-up phase of `--rule warm`)."""
    g1, g2 = G[:, 0], G[:, 1]
    E = G.shape[0]
    if arm in ("A", "Ap") or not gate:
        return 0.5 * (g1 + g2).mean(0), None, torch.ones(E, device=G.device), None, None
    t = F.cosine_similarity(g1, g2, dim=1)                        # agreement of each environment WITH ITSELF
    keep = t > trust_min
    if int(keep.sum()) < 2:
        return torch.zeros(G.shape[-1], device=G.device), torch.ones(G.shape[-1], dtype=torch.bool, device=G.device), t, None, None
    a, b = g1[keep], g2[keep]
    gm = 0.5 * (a + b)
    within = 0.25 * ((a - b) ** 2).mean(0)                        # variance of a half-batch mean, per coordinate
    across = gm.var(0, unbiased=False)
    if keep_q > 0:                                                # data-driven threshold: keep the most-agreed fraction
        ratio = across / (within + eps)
        mask_out = ratio > torch.quantile(ratio.float(), keep_q)
    else:
        mask_out = across > c * within + eps                      # True = masked OUT (environments disagree in value)
    u = gm.mean(0)
    return (u * (~mask_out) if mask_grad else u), mask_out, t, across, within


# ── evaluation ──────────────────────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, task, K, crit, dev, g):
    ttc = trials_to_criterion(model, task, task.test, K, dev, g, crit=crit, n=256)
    held_last = float(np.mean([acc for _p, _h, acc in ttc]))
    held_ttc = [h for _p, h, _a in ttc]
    tr = []
    for pair in task.train:
        tok = env_batch(task, [pair], 0, 64, K, dev, g)[0]
        tr.append(float(per_demo_exact(model, tok, K)[-1]))
    return dict(held_ttc=held_ttc, held_ttc_mean=float(np.mean(held_ttc)), held_last_acc=held_last,
                held_solved=float(np.mean([h < K for h in held_ttc])), train_last_acc=float(np.mean(tr)),
                train_per_env=tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "Ap", "Bp"])
    ap.add_argument("--task", default="compose", choices=["compose", "affine"])
    ap.add_argument("--rule", default="ftest", choices=["ftest", "warm"],
                    help="ftest: gate from step 0 (pre-registered; cannot bootstrap in a real net); "
                         "warm: baseline updates for the first --warm_frac of steps, then ftest+wd")
    ap.add_argument("--warm_frac", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--per_env", type=int, default=16, help="sequences per environment per step (two halves)")
    ap.add_argument("--c", type=float, default=3.0)
    ap.add_argument("--trust", type=float, default=0.2)
    ap.add_argument("--decay_total", type=float, default=3.0, help="masked coordinates shrink by e^-this over a run")
    ap.add_argument("--keep_q", type=float, default=0.0,
                    help="if > 0: keep this fraction of coordinates with the smallest across/within ratio instead of c")
    ap.add_argument("--mask_where", default="grad", choices=["grad", "update"],
                    help="grad: Adam sees the masked gradient; update: Adam sees the mean gradient, masked steps are undone")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crit", type=float, default=0.8)
    ap.add_argument("--snap_every", type=int, default=25)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time_only", type=int, default=0, help="run this many steps, print ms/step, exit")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    task = TASKS.make(args.task, args.seed)
    train, test, V = task.train, task.test, task.V
    n_noise = N_NOISE if args.arm.endswith("p") else 0
    E = len(train) + n_noise
    model = Model(d_model=args.d, n_layer=args.layers, n_head=args.heads, max_len=args.k * 2 * L + 2, pos="rope",
                  n_vocab=V).to(dev)
    flat = Flat(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = max(1, args.steps // 20)
    lr_at = lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda"))
    g = torch.Generator(device=dev).manual_seed(args.seed)
    ge = torch.Generator(device=dev).manual_seed(args.seed + 99)
    delta = args.decay_total / args.steps                          # per-step shrink of a masked coordinate at full LR
    engine = GroupGrads(model, flat, args.k, dev, amp, V)
    print(f"per-group gradients: {'vmap (verified against the loop)' if engine.ok else 'loop'}", flush=True)
    half = args.per_env // 2
    print(f"arm {args.arm} | params {flat.P:,} | envs {E} ({n_noise} noise) | steps {args.steps} | per-env {args.per_env} "
          f"| c {args.c} trust {args.trust} | dev {dev}", flush=True)
    prims = [[bool(task.contains(pair, i)) for pair in train] for i in range(len(task.primitives))]
    json.dump(dict(vars(args), params=flat.P, layout=flat.layout(), train_pairs=train, test_pairs=test, V=V,
                   primitives=task.primitives, prims=prims, E=E, n_noise=n_noise,
                   env_names=[task.describe(p) for p in train]), open(out / "config.json", "w"))

    # participation statistics: per environment e and coordinate, sums over steps of g_e, g_e*d, g_e^2; and of d, d^2
    S_g = torch.zeros(E, flat.P, device=dev)
    S_gd = torch.zeros(E, flat.P, device=dev)
    S_gg = torch.zeros(E, flat.P, device=dev)
    S_d = torch.zeros(flat.P, device=dev)
    S_dd = torch.zeros(flat.P, device=dev)
    mask_count = torch.zeros(flat.P, device=dev)                  # how often each coordinate was masked out
    snaps, snap_steps = [flat.get().half().cpu()], [0]
    logf = open(out / "train.jsonl", "w")
    evalf = open(out / "eval.jsonl", "w")
    t0 = time.time()
    for step in range(args.steps):
        toks = env_batch(task, train, n_noise, args.per_env, args.k, dev, g).view(E, 2, half, -1)
        G = group_grads(model, flat, toks, args.k, amp, engine, V)
        gate = not (args.rule == "warm" and step < args.warm_frac * args.steps)
        grad, mask_out, trust, across, within = combine(G, args.arm, args.c, args.trust, gate=gate, keep_q=args.keep_q,
                                                        mask_grad=(args.mask_where == "grad"))
        before = flat.get()
        flat.set_grad(grad)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if mask_out is not None and mask_out.any():
            if args.mask_where == "update":
                flat.assign(torch.where(mask_out, before, flat.get()))   # undo the step on disagreed coordinates
            flat.scale_masked(mask_out, 1.0 - delta * lr_at(step))
            mask_count += mask_out.float()
        after = flat.get()
        d = after - before
        gm_all = 0.5 * (G[:, 0] + G[:, 1])                        # each environment's full-batch gradient
        S_g += gm_all
        S_gd += gm_all * d[None, :]
        S_gg += gm_all ** 2
        S_d += d
        S_dd += d ** 2
        if args.time_only and step + 1 == args.time_only:
            torch.cuda.synchronize() if dev == "cuda" else None
            print(f"{args.time_only} steps in {time.time() - t0:.1f}s -> {1000 * (time.time() - t0) / args.time_only:.0f} ms/step")
            return
        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                loss_now = float(F.cross_entropy(
                    model(toks[:, 0].reshape(-1, toks.shape[-1]))[:, :-1][:, out_mask(args.k, dev)[1:]].reshape(-1, V).float(),
                    toks[:, 0].reshape(-1, toks.shape[-1])[:, 1:][:, out_mask(args.k, dev)[1:]].reshape(-1)))
            rec = dict(step=step + 1, loss=loss_now, grad_norm=float(gn), lr=opt.param_groups[0]["lr"],
                       trust=[round(float(x), 3) for x in trust], wall=time.time() - t0)
            if mask_out is not None:
                rec.update(masked_frac=float(mask_out.float().mean()),
                           across_med=float(across.median()) if across is not None else None,
                           within_med=float(within.median()) if within is not None else None)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if (step + 1) % args.snap_every == 0:
            snaps.append(after.half().cpu())
            snap_steps.append(step + 1)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = evaluate(model, task, args.k, args.crit, dev, ge)
            ev.update(step=step + 1, wall=time.time() - t0)
            evalf.write(json.dumps(ev) + "\n")
            evalf.flush()
            print(f"  step {step + 1:5d} | loss {loss_now:.3f} | train acc {ev['train_last_acc']:.3f} | held acc "
                  f"{ev['held_last_acc']:.3f} solved {ev['held_solved']:.2f} ttc {ev['held_ttc_mean']:.2f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
    n = args.steps
    torch.save(dict(S_g=S_g.cpu(), S_gd=S_gd.cpu(), S_gg=S_gg.cpu(), S_d=S_d.cpu(), S_dd=S_dd.cpu(), n=n,
                    mask_count=mask_count.cpu(), snaps=torch.stack(snaps), snap_steps=snap_steps,
                    model=model.state_dict()), out / "trace.pt")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
