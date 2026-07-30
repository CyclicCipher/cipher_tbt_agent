"""INSTINCT: does pre-supplying the COORDINATE SYSTEM remove the allocation cost?

THE CLAIM UNDER TEST, and it is two claims at once. `arith.py` found that an operation is a change of basis in which it
becomes coordinate-wise, which suggested that a structured initialisation should supply a BASIS rather than a rule — an
"instinct" that training can still revise. `engine.py` then found that holding two operations costs a 3x delay in
generalisation which is neither data dilution nor capacity, and therefore looks like the cost of SEARCHING for an
allocation. Put together: **if the delay is the basis search, initialising the embedding with the bases already present
should remove most of it.** One run tests the initialisation idea and the scale hypothesis simultaneously.

WHY THIS IS NOT HANDING OVER THE ANSWER, which is the whole design problem. Supplying the FULL Fourier basis supplies
nothing — a random matrix expressed in a rotated basis is still a random matrix, and the information is not in the basis
but in the SPARSITY. So the init supplies a superposition of `m` frequencies per basis, and **which frequencies is chosen
at random**, deliberately not the ones a trained model turns out to use. That is legitimate because *which* frequencies is
arbitrary: different seeds pick different sets and all of them work, so the structure ("the code is sparse in these two
coordinate systems") is supplied while the answer (what to compute, how to route, and which of the supplied modes to
actually lean on) is not. Nothing about the operations, their group structure, or the routing is written in.

THE CONTROL THAT DECIDES IT — `--init shuffled`. The same construction, then the ROWS ARE PERMUTED. That preserves every
marginal statistic of the initialisation — the singular values, the rank, the sinusoidal shape, the scale — and destroys
only the correspondence between a row's position and its value, which is exactly what makes it a coordinate system. If
`fourier` helps and `shuffled` does not, the benefit is the coordinate system. If both help, the benefit was low-rank
structure or scale and the basis story is wrong.

ARMS
    --init random     the baseline: default initialisation, i.e. `engine.py`
    --init fourier    m random value-index frequencies + m random discrete-log frequencies, superposed
    --init shuffled   the same matrix with its rows permuted (the falsifier)
    --init value      only the value-index modes -- should help `+` and not `x`
    --init log        only the discrete-log modes -- should help `x` and not `+`

The last two are the sharper prediction: a supplied basis should accelerate exactly the operation whose structure it
matches, and be neutral for the other. A general speedup for both under a one-basis init would mean the init is helping
for some reason other than the one claimed.

Usage:  python experiments/transformers/instinct.py --init random
        python experiments/transformers/instinct.py --init fourier
        python experiments/transformers/instinct.py --init shuffled
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from alloc import Neurons, acc, build, make_ops
from arith import log_table, primitive_root, spectrum, sparsity
from h1_lid import Model


def fourier_init(model, p, order, kind, m, gen, dev):
    """Overwrite the VALUE-token embedding rows with a superposition of `m` randomly chosen frequencies per requested
    basis, rescaled to the standard deviation the default initialisation would have had — so the arms differ in
    STRUCTURE and not in scale, which would otherwise be a confound on the learning rate."""
    E = model.emb.weight.data
    d = E.shape[1]
    n = p - 1
    target_std = E[1:p].std().item()
    rows = torch.zeros(n, d, device=dev)
    nyq = n // 2
    picked = {}
    for basis in (("value",) if kind == "value" else ("log",) if kind == "log" else ("value", "log")):
        freqs = (torch.randperm(nyq - 1, generator=gen, device=dev)[:m] + 1).tolist()
        picked[basis] = sorted(freqs)
        # Row position in this basis: its VALUE for the value index, its DISCRETE LOG for the log index.
        if basis == "value":
            pos = torch.arange(n, device=dev, dtype=torch.float32)
        else:
            pos = torch.empty(n, device=dev, dtype=torch.float32)
            for i, x in enumerate(order):                      # order[i] is the unit whose discrete log is i
                pos[x - 1] = float(i)
        for f in freqs:
            th = 2 * math.pi * f * pos / n
            A = torch.randn(1, d, generator=gen, device=dev)
            B = torch.randn(1, d, generator=gen, device=dev)
            rows += th.cos()[:, None] * A + th.sin()[:, None] * B
    rows *= target_std / rows.std()
    if kind == "shuffled":                                     # THE FALSIFIER: identical statistics, no coordinate system
        rows = rows[torch.randperm(n, generator=gen, device=dev)]
    E[1:p] = rows
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--ops", default="lin:1:1,mul")
    ap.add_argument("--init", default="fourier", choices=["random", "fourier", "shuffled", "value", "log"])
    ap.add_argument("--m", type=int, default=5, help="frequencies supplied per basis, chosen at RANDOM")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--trace", type=int, default=500)
    ap.add_argument("--crit", type=float, default=0.9)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    p = args.p

    order, _ = log_table(p, primitive_root(p))
    ops = make_ops(args.ops.split(","), p, dev)
    train, _tr_ops, te_ops = build(ops, p, dev, args.train_frac, args.seed)
    labels = [o[0] for o in ops]

    model = Model(d_model=args.d_model, n_layer=1, n_head=4, max_len=8, pos="learned",
                  n_vocab=p + len(ops) + 1).to(dev)
    picked = {}
    if args.init != "random":
        gen = torch.Generator(device=dev).manual_seed(args.seed + 12345)
        picked = fourier_init(model, p, order, args.init, args.m, gen, dev)
    print(f"device {dev} | p={p} | ops {labels} | init={args.init}"
          + (f" m={args.m} supplied {picked}" if picked else "") + f" | steps={args.steps}")
    E0 = model.emb.weight.detach()
    for name, idx in (("value index", torch.arange(1, p, device=dev)),
                      ("DISCRETE LOG", torch.tensor(order, device=dev))):
        top, pr, idxs = sparsity(spectrum(E0[idx]), args.k)
        print(f"   init spectrum {name:<14} top-{args.k} {top:.3f}  eff.freqs {pr:>5.1f}   {idxs}")

    nrn = Neurons(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98),
                            fused=(dev == "cuda"))
    warm = max(1, args.steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 1.0)
    trainer = torch.compile(model) if args.compile and dev == "cuda" else model
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.bf16 and dev == "cuda"))
    hit = [None] * len(ops)
    t0 = time.time()
    for s in range(args.steps):
        with amp:
            logits = trainer(train[0])[:, -1]
        loss = F.cross_entropy(logits.float(), train[1])
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
    print(f"\ntrained in {time.time() - t0:.0f}s | final loss {loss.item():.4f}")
    print(f"init={args.init:<9} STEPS TO {args.crit:.0%}  " + "   ".join(
        f"{l} {('%6d' % h) if h else ' never'}" for l, h in zip(labels, hit)))
    print(f"{'':<14}FINAL acc        " + "   ".join(f"{l} {acc(model, t):.3f}" for l, t in zip(labels, te_ops)))
    E = model.emb.weight.detach()
    for name, idx in (("value index", torch.arange(1, p, device=dev)),
                      ("DISCRETE LOG", torch.tensor(order, device=dev))):
        top, pr, idxs = sparsity(spectrum(E[idx]), args.k)
        print(f"{'':<14}final spectrum {name:<14} eff.freqs {pr:>5.1f}   {idxs}")
    print("\nCLAIM: `fourier` reaches criterion in far fewer steps than `random`, and `shuffled` does NOT -- which is")
    print("what distinguishes 'a coordinate system was supplied' from 'a low-rank matrix of the right scale was supplied'.")


if __name__ == "__main__":
    main()
