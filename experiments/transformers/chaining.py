"""CHAINING CONSISTENCY: an objective that separates composing from memorising.

WHY ERM CANNOT. Let a task be `(a,b)`. A MEMORISER stores `g_t` per task; a COMPOSER has `g_(a,b) = φ_b ∘ φ_a`. On the
labelled training tasks the two are EXTENSIONALLY IDENTICAL — same outputs everywhere the loss can see — so no objective
that is a function of behaviour-on-training-tasks can prefer either. That is why the optimiser, architecture and data axes
all came back null (`apply_vs_mix.py`, `diversity.py`): they were all varying things the loss was blind to.

THE OBJECTIVE. Ground the PARTS with supervision and constrain the WHOLES with a label-free consistency term — run a
composition two ways and penalise the disagreement:

    L = L_sup(labelled)  +  lambda * CE( M(D_ab, x),  sg[ M(D_b, M(D_a, x)) ] )

For a composer this is zero by construction. For a memoriser, its output on an unlabelled composition is unconstrained by
`L_sup`, so the term is zero only if its answer happens to coincide with the chained one — and the gradient drives it
there. This is stronger than a tie-breaking regulariser: the chained prediction is a MANUFACTURED TARGET, so the loss
*constructs* composer-behaviour where no label exists. ERM's problem is that the two hypotheses agree everywhere it looks;
this enlarges the region where they disagree, to everything reachable by chaining, and supervises there for free.

It is a different axis from task diversity: more LABELLED compositions each get memorised independently (4→42 bought
nothing), while chained ones are TIED TO THE PRIMITIVES.

THREE-WAY SPLIT, because the honest failure mode is transduction:
  * LABELLED   — every primitive, plus some compositions. Supervised loss.
  * UNLABELLED — compositions presented ONLY through the consistency term. No supervised gradient on their queries.
  * EVAL       — compositions NEVER presented in any form. The real test.
"Unlabelled" is precise here and worth stating: the demonstrations for such a task are of course correct pairs (they are
the in-context data), but no supervised gradient is ever taken on its query. The existing failure is that the model scores
0.02–0.08 on held-out tasks WHILE BEING SHOWN their demonstrations, so demonstrations are exactly what does not suffice.

BUILT-IN FALSIFIERS, in the same run rather than as follow-ups:
  * `--no_prim_labels` — the chain ungrounded. If the gain survives this, the mechanism story is wrong, because the whole
    argument is that supervision on the parts is what keeps the chain from converging on coherent nonsense.
  * `--lam 0` — must recover the current null.
  * EVAL vs UNLABELLED accuracy — if gains appear only on the unlabelled pool, we have shown transduction, not
    composition, and it must be reported that way.

Usage:  python experiments/transformers/chaining.py --arms full,lam0,noprim
"""
from __future__ import annotations

import argparse
import itertools
import math
import time

import torch
import torch.nn.functional as F

from apply_vs_mix import Mix
from diversity import NAMES, PRIMS
from h1_lid import L, V


def apply_task(s, task):
    for i in task:
        s = PRIMS[NAMES[i]](s)
    return s


def build_splits(seed: int):
    """Primitives, then compositions, deduplicated ACROSS BOTH LENGTHS. A length-2 composition can equal a length-1
    primitive (`rot1∘rot1 = rot2`), and if that is not caught an 'unseen' evaluation task can secretly be a labelled
    primitive under another name — which would manufacture exactly the result this is testing for."""
    g = torch.Generator().manual_seed(seed)
    probe = torch.randint(0, V, (24, L), generator=g)
    ident = tuple(probe.flatten().tolist())
    seen, prims = {}, []
    for i in range(len(NAMES)):                                  # primitives claim their signatures FIRST
        sig = tuple(apply_task(probe, (i,)).flatten().tolist())
        if sig not in seen:
            seen[sig] = (i,)
            prims.append((i,))
    comps = []
    for a, b in itertools.product(range(len(NAMES)), repeat=2):
        sig = tuple(apply_task(probe, (a, b)).flatten().tolist())
        if sig == ident or sig in seen:                          # drop no-ops and anything equal to a primitive
            continue
        seen[sig] = (a, b)
        comps.append((a, b))
    order = torch.randperm(len(comps), generator=g).tolist()
    comps = [comps[i] for i in order]
    # Split in THIRDS from whatever the dedup actually leaves, rather than from a hardcoded count. The cross-length dedup
    # removes 9 of the 62 two-compositions because they coincide with a primitive, and a fixed size silently over-ran.
    n = len(comps) // 3
    return prims, comps[:n], comps[n:2 * n], comps[2 * n:3 * n]


def make(B, K, tasks, dev, g, fixed=None):
    idx = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, len(tasks), (B,), generator=g, device=dev))
    x = torch.randint(0, V, (B, K, L), generator=g, device=dev)
    y = torch.empty_like(x)
    for t in idx.unique():
        m = idx == t
        y[m] = apply_task(x[m], tasks[int(t)])
    return x[:, :-1], y[:, :-1], x[:, -1], y[:, -1]


def demos_for(B, K, task, dev, g):
    """Demonstrations of ONE given task — needed to run a primitive on demand inside the chain."""
    x = torch.randint(0, V, (B, K - 1, L), generator=g, device=dev)
    return x, apply_task(x, task)


@torch.no_grad()
def chained_target(model, ab, B, K, qx, dev, g):
    """Run the composition the OTHER way: primitive `a`, then primitive `b`, on the model's own intermediate output. This
    whole function is the stop-gradient — the target is manufactured, never differentiated through."""
    a, b = (ab[0],), (ab[1],)
    dax, day = demos_for(B, K, a, dev, g)
    mid = model(dax, day, qx).argmax(-1)
    dbx, dby = demos_for(B, K, b, dev, g)
    return model(dbx, dby, mid).argmax(-1)


@torch.no_grad()
def accuracy(model, tasks, dev, K, n=384):
    hits = []
    for t in range(len(tasks)):
        dx, dy, qx, qy = make(n, K, tasks, dev, None, fixed=t)
        hits.append((model(dx, dy, qx).argmax(-1) == qy).all(-1).float().mean().item())
    return sum(hits) / len(hits), sum(h >= 0.8 for h in hits)


def run(name, args, splits, dev):
    prims, lab, unl, ev = splits
    torch.manual_seed(args.seed)
    model = Mix().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98),
                            fused=(dev == "cuda"))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    g = torch.Generator(device=dev).manual_seed(args.seed)

    # The supervised pool. `no_prim` removes the PRIMITIVES from supervision, which ungrounds the chain — the falsifier.
    sup = (lab if args.no_prim else prims + lab)
    t0 = time.time()
    for step in range(args.steps):
        dx, dy, qx, qy = make(args.batch, args.k, sup, dev, g)
        loss = F.cross_entropy(model(dx, dy, qx).reshape(-1, V), qy.reshape(-1))
        # GROUND THE PARTS FIRST. The consistency term is switched on only after `cwarm` of training. This is not a
        # tuning knob bolted on after a bad result -- it is the mechanism story enforced in time. The whole argument is
        # that supervision on the primitives is what stops the chain converging on coherent nonsense, and at step 0 there
        # are no learned primitives to ground it: the model distils its own garbage, and that attractor beat the
        # supervised loss outright. Measured without it, at lambda=1: 0.000 on EVERYTHING, primitives included, against a
        # working 0.997 control. Predicted as failure mode 1 in the proposal; the proposed mitigation (just add L_sup)
        # was not enough.
        if args.lam > 0 and unl and step >= args.cwarm * args.steps:
            t = int(torch.randint(0, len(unl), (1,), generator=g, device=dev))
            cx, cy, cqx, _ = make(args.cbatch, args.k, unl, dev, g, fixed=t)
            tgt = chained_target(model, unl[t], args.cbatch, args.k, cqx, dev, g)
            loss = loss + args.lam * F.cross_entropy(model(cx, cy, cqx).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    pr = accuracy(model, prims, dev, args.k)
    la = accuracy(model, lab, dev, args.k)
    un = accuracy(model, unl, dev, args.k)
    evl = accuracy(model, ev, dev, args.k)
    print(f"{name:<10}{pr[0]:>10.3f}{la[0]:>10.3f}{un[0]:>12.3f}{un[1]:>4}/{len(unl):<4}"
          f"{evl[0]:>12.3f}{evl[1]:>4}/{len(ev):<4}{time.time()-t0:>8.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--cbatch", type=int, default=128, help="batch for the consistency term")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_prim", action="store_true")
    p.add_argument("--cwarm", type=float, default=0.5,
                   help="fraction of training before the consistency term switches on")
    p.add_argument("--arms", default="full,lam0,noprim")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    splits = build_splits(args.seed)
    prims, lab, unl, ev = splits
    print(f"device {dev} | {len(prims)} primitives, {len(lab)} labelled, {len(unl)} unlabelled, "
          f"{len(ev)} EVAL (never presented) | steps={args.steps}")
    print(f"\n{'arm':<10}{'prims':>10}{'labelled':>10}{'UNLABELLED':>12}{'solved':>8}{'EVAL':>12}{'solved':>8}{'secs':>8}")
    for arm in args.arms.split(","):
        a = argparse.Namespace(**vars(args))
        a.lam = 0.0 if arm == "lam0" else args.lam
        a.no_prim = (arm == "noprim")
        run(arm, a, splits, dev)
    print("\nCLAIM: `full` lifts EVAL above `lam0`. FALSIFIERS: if `noprim` matches `full` the mechanism story is wrong")
    print("(the chain was never grounded); if UNLABELLED lifts but EVAL does not, that is transduction, not composition.")


if __name__ == "__main__":
    main()
