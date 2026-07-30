"""ALLOCATION COST: what does it cost to hold N operations, and does it matter whether they can SHARE a basis?

THE FINDING THIS EXISTS TO EXPLAIN. `engine.py` trained one model on `+` and `x` and generalisation moved from step ~2000
to ~6000 — a 3x delay — while the circuits themselves did NOT degrade (11 of 96 row-dimensions still carry each
operation, keep-only accuracy 0.887 / 0.957). Per-operation examples per step were IDENTICAL to the single-op run and the
two circuits used 22 of 96 available dimensions, so the delay is neither data dilution nor capacity. The extra 4000 steps
bought an ALLOCATION: two non-interfering subspaces plus a routing function. This file asks what that allocation costs as
a function of (a) how many operations there are and (b) whether they are mathematically forced to duplicate.

THE TWO AXES, SEPARATED — which is the point, since `engine.py` confounded them.

  * NUMBER of operations, sharing held CONSTANT and maximal. Every Z-linear map `(alpha*a + beta*b) mod M` is diagonal in
    the SAME value-index Fourier basis, so a family of them can in principle share one coordinate system entirely and
    only the routing count grows. Coefficients are chosen coprime to M so each map is balanced in each argument.
        --ops lin:1:1 / lin:1:1,lin:1:-1 / +lin:1:5,lin:5:1 / +four more
    If steps-to-generalise grows with N even here, the cost is ROUTING. If it stays flat, routing is nearly free and
    `engine.py`'s 3x was about the basis conflict instead.

  * COMPATIBILITY, number held constant at two.
        share the value basis:  --ops lin:1:1,lin:1:-1     (a+b and a-b: same basis, one operand's phase conjugated)
        share the log basis:    --ops mul,div              (a*b and a/b: same basis, likewise)
        share NOTHING:          --ops lin:1:1,mul          (no basis diagonalises + and x simultaneously)
    This is the recruit-versus-duplicate question asked where the answer is forced and known in advance, so what is
    measured is whether the OPTIMISER exploits sharing when it is available.

  * SEQUENTIAL versus JOINT (`--seq`). Train the first operation alone for half the budget, then all of them. If
    sequential is much cheaper the cost was interference during joint learning; if it is not, the cost was discovering
    the routing. Watch the first operation's accuracy after the switch — catastrophic forgetting would show here.

MEASUREMENTS. Steps-to-90% per operation (the trace, at `--trace` resolution); spectra in both bases; per-basis ablation
crossed with per-operation accuracy, rank-matched as in `arith.py`; and the neuron-routing matrix — for each operation,
ablate the decile of MLP neurons most selective for it and measure EVERY operation, so the off-diagonal says whether the
populations are disjoint.

Domain is the units 1..p-1 for every operation so all spectra are FFTs over the same p-1 embedding rows. Linear ops are
mod p-1 and multiplicative ops mod p, so both live on cyclic groups of the same ORDER and cannot be told apart by that.

Usage:  python experiments/transformers/alloc.py --ops lin:1:1,lin:1:-1
        python experiments/transformers/alloc.py --ops mul,div
        python experiments/transformers/alloc.py --ops lin:1:1,mul
        python experiments/transformers/alloc.py --ops lin:1:1,lin:1:-1,lin:1:5,lin:5:1
        python experiments/transformers/alloc.py --ops lin:1:1,mul --seq 1
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from arith import keep_freqs, log_table, primitive_root, random_basis, real_rank, spectrum, sparsity
from h1_lid import Model


def make_ops(specs, p, dev):
    """Each spec becomes (label, callable, expected basis). `lin:a:b` is Z-linear mod p-1 and therefore diagonal in the
    VALUE Fourier basis; `mul`/`div` are the multiplicative group and diagonal in the DISCRETE-LOG basis."""
    inv = torch.tensor([0] + [pow(x, p - 2, p) for x in range(1, p)], device=dev)
    ops = []
    for s in specs:
        if s in ("mul", "div"):
            fn = ((lambda a, b: (a * b) % p) if s == "mul" else (lambda a, b: (a * inv[b]) % p))
            ops.append((s, fn, "log"))
        else:
            _, al, be = s.split(":")
            al, be = int(al), int(be)
            for co in (al, be):
                if math.gcd(abs(co), p - 1) != 1:
                    raise ValueError(f"{s}: coefficient {co} is not coprime to {p - 1}, so the map is unbalanced")
            ops.append((s, (lambda a, b, al=al, be=be: (al * a + be * b) % (p - 1)), "value"))
    return ops


def build(ops, p, dev, train_frac, seed):
    a = torch.arange(1, p, device=dev).repeat_interleave(p - 1)
    b = torch.arange(1, p, device=dev).repeat(p - 1)
    toks, cs, tags = [], [], []
    for i, (_lbl, fn, _basis) in enumerate(ops):
        toks.append(torch.stack([a, torch.full_like(a, p + i), b, torch.full_like(a, p + len(ops))], dim=1))
        cs.append(fn(a, b))
        tags.append(torch.full_like(a, i))
    tok, c, tag = torch.cat(toks), torch.cat(cs), torch.cat(tags)
    g = torch.Generator(device=dev).manual_seed(seed)
    perm = torch.randperm(len(tok), generator=g, device=dev)
    cut = int(train_frac * len(tok))
    def per_op(ix):
        return [(tok[ix][tag[ix] == i], c[ix][tag[ix] == i]) for i in range(len(ops))]
    return (tok[perm[:cut]], c[perm[:cut]], tag[perm[:cut]]), per_op(perm[:cut]), per_op(perm[cut:])


class Neurons:
    def __init__(self, model):
        self.act, self.mask = None, None
        model.blocks[0].mlp[1].register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        self.act = out.detach()
        return None if self.mask is None else out * self.mask


@torch.no_grad()
def acc(model, data):
    return (model(data[0])[:, -1].argmax(-1) == data[1]).float().mean().item() if len(data[0]) else 0.0


@torch.no_grad()
def routing_matrix(model, nrn, tests, frac, seed):
    """For each operation, the decile of neurons most selective FOR it, ablated, measured on every operation. A diagonal
    of collapses with an intact off-diagonal means disjoint populations; a full row of collapses means shared compute.
    `mags[i]` is mean activation magnitude at the answer position under operation i."""
    mags = []
    for data in tests:
        model(data[0])
        mags.append(nrn.act[:, -1].abs().mean(0).float())
    M = torch.stack(mags)                                        # (n_ops, d_mlp)
    d = M.shape[1]
    n_sel = max(1, int(frac * d))
    rows = []
    for i in range(len(tests)):
        others = torch.cat([M[:i], M[i + 1:]]).mean(0)
        sel = (M[i] - others) / (M[i] + others).clamp(min=1e-9)
        pop = torch.topk(sel, n_sel).indices
        m = torch.ones(d, device=M.device)
        m[pop] = 0.0
        nrn.mask = m
        rows.append([acc(model, t) for t in tests])
        nrn.mask = None
    gen = torch.Generator(device=M.device).manual_seed(seed)
    m = torch.ones(d, device=M.device)
    m[torch.randperm(d, generator=gen, device=M.device)[:n_sel]] = 0.0
    nrn.mask = m
    ctrl = [acc(model, t) for t in tests]
    nrn.mask = None
    return rows, ctrl, n_sel


@torch.no_grad()
def basis_ablation(model, idx, freqs, tests, seed):
    E0 = model.emb.weight.detach().clone()
    rws = E0[idx].float()
    n = rws.shape[0]
    mu = rws.mean(0, keepdim=True)
    Q = random_basis(n, real_rank(freqs, n), E0.device, torch.Generator(device=E0.device).manual_seed(seed))
    proj = Q @ (Q.T @ (rws - mu))
    out = {}
    for name, new in (("keep named", keep_freqs(rws, freqs, True)), ("keep random", mu + proj),
                      ("drop named", keep_freqs(rws, freqs, False)), ("drop random", rws - proj)):
        model.emb.weight[idx] = new.to(E0.dtype)
        out[name] = [acc(model, t) for t in tests]
        model.emb.weight.copy_(E0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--ops", default="lin:1:1,mul")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--trace", type=int, default=500)
    ap.add_argument("--crit", type=float, default=0.9, help="accuracy defining 'generalised', for steps-to-criterion")
    ap.add_argument("--seq", type=int, default=0, help="1 = train ops[0] alone for half the budget, then all ops")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    p = args.p

    order, _ = log_table(p, primitive_root(p))
    ops = make_ops(args.ops.split(","), p, dev)
    train, tr_ops, te_ops = build(ops, p, dev, args.train_frac, args.seed)
    labels = [o[0] for o in ops]
    print(f"device {dev} | p={p} | ops {labels} | bases {[o[2] for o in ops]} | "
          f"{len(train[0])} train / {sum(len(t[0]) for t in te_ops)} test | seq={bool(args.seq)} | steps={args.steps}")

    model = Model(d_model=args.d_model, n_layer=1, n_head=4, max_len=8, pos="learned",
                  n_vocab=p + len(ops) + 1).to(dev)
    nrn = Neurons(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98),
                            fused=(dev == "cuda"))
    warm = max(1, args.steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 1.0)
    trainer = torch.compile(model) if args.compile and dev == "cuda" else model
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.bf16 and dev == "cuda"))

    # SEQUENTIAL: the first half sees only ops[0]. Implemented as a mask over the training set rather than a second
    # dataset, so the joint phase is bit-identical to the joint run apart from where the weights started.
    first_only = train[2] == 0
    hit = [None] * len(ops)
    t0 = time.time()
    for s in range(args.steps):
        if args.seq and s < args.steps // 2:
            tk, cc = train[0][first_only], train[1][first_only]
        else:
            tk, cc = train[0], train[1]
        with amp:
            logits = trainer(tk)[:, -1]
        loss = F.cross_entropy(logits.float(), cc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if args.trace and (s + 1) % args.trace == 0:
            a_now = [acc(model, t) for t in te_ops]
            for i, v in enumerate(a_now):
                if hit[i] is None and v >= args.crit:
                    hit[i] = s + 1
            if (s + 1) % (args.trace * 4) == 0:
                print(f"   step {s + 1:>6}  " + "  ".join(f"{l} {v:.3f}" for l, v in zip(labels, a_now)))
    secs = time.time() - t0

    print(f"\ntrained in {secs:.0f}s | final loss {loss.item():.4f}")
    print(f"(1) STEPS TO {args.crit:.0%}   " + "   ".join(
        f"{l} {('%6d' % h) if h else ' never'}" for l, h in zip(labels, hit)))
    print(f"    TEST acc      " + "   ".join(f"{l} {acc(model, t):.3f}" for l, t in zip(labels, te_ops)))
    if min(acc(model, t) for t in te_ops) < 0.5:
        print("    !! an operation did not generalise -- everything below is UNINTERPRETABLE, not a result.")

    E = model.emb.weight.detach()
    bases = (("value index", torch.arange(1, p, device=dev)), ("DISCRETE LOG", torch.tensor(order, device=dev)))
    print(f"\n(2) SPECTRA   {'top-'+str(args.k):>9}{'eff. freqs':>13}   strongest")
    named = {}
    for name, idx in bases:
        top, pr, idxs = sparsity(spectrum(E[idx]), args.k)
        named[name] = idxs
        print(f"    {name:<14}{top:>7.3f}{pr:>13.1f}   {idxs}")

    print(f"\n(3) BASIS ABLATION -- top {args.k} frequencies of each basis, rank-matched random controls")
    print(f"    {'basis':<14}{'condition':<14}" + "".join(f"{l:>10}" for l in labels))
    for name, idx in bases:
        r = basis_ablation(model, idx, named[name], te_ops, args.seed)
        for cond in ("keep named", "keep random", "drop named", "drop random"):
            print(f"    {name if cond == 'keep named' else '':<14}{cond:<14}"
                  + "".join(f"{v:>10.3f}" for v in r[cond]))

    rows, ctrl, n_sel = routing_matrix(model, nrn, te_ops, args.frac, args.seed)
    print(f"\n(4) NEURON ROUTING -- ablate the {n_sel}-neuron decile most selective for each op, measure all ops")
    print(f"    {'ablated':<20}" + "".join(f"{l:>10}" for l in labels))
    for l, row in zip(labels, rows):
        print(f"    {'sel. for ' + l:<20}" + "".join(f"{v:>10.3f}" for v in row))
    print(f"    {'random (control)':<20}" + "".join(f"{v:>10.3f}" for v in ctrl))
    print("    A collapsing DIAGONAL with an intact off-diagonal means disjoint populations (routing). A full row of")
    print("    collapses means the computation is shared and selectivity is not where the switch lives.")


if __name__ == "__main__":
    main()
