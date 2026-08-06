"""MATCHING AS ARCHITECTURE: a Sinkhorn assignment layer whose OUTPUT IS the correspondence.

WHY. `canonicaliser.py` measured the split precisely: contrastive training on demonstrations recovers the value map
(0.689 held-out, against a 0.44-0.54 random-encoder floor) and NEVER the permutation (0.000/0.000, at or below random,
under two architectures). And it is not a generalisation failure -- retrieval on TRAINING transformations is 0.059, so
the objective does not solve its own training task. `vmp` is a global, position-independent property readable off any
single position; `idx` is a CORRESPONDENCE, and that is the binding/matching operation the whole line is stuck on.

The prescription that measurement produced was: stop hoping a generic encoder INDUCES matching, and SUPPLY it as
architecture. That is this file. Instead of pooling to an opaque code and probing whether a permutation can be read out
of it, the model emits a doubly-stochastic matrix -- Sinkhorn's projection of a score matrix onto the Birkhoff polytope --
which IS a soft permutation. Nothing has to discover that a correspondence is the right kind of object.

Same lesson as `project_an_operation_is_a_basis`: supply the structure the operation needs and let training fill in which
one, rather than expecting the structure itself to be learned.

THE OBJECTIVE IS LABEL-FREE, and this is what keeps it honest. `localise.py`'s hand-written `recover` brute-forces the 10
affine value maps and matches columns -- exact, but it USES the fact that the group factors as `S6 x Aff(Z5)`. Here
neither `idx` nor `vmp` is ever supplied. Both unknowns are parameterised as assignments, and the only signal is
RECONSTRUCTION: with `y[a] = vmp[x[idx[a]]]`, a candidate `(P, M)` predicts

    log p(y[a] = v)  =  logsumexp_b ( logP[a, b] + logM[x[b], v] )

and is scored by how well that predicts demonstrations we generate by executing. Exactly the `canonicaliser.py` premise
-- the forward model is the component that always works -- with the assignment structure added.

Both unknowns are assignments and each is inferred THROUGH the other, which is the part the contrastive setup could not
express: you cannot tell which input position feeds output `a` without a value map, and you cannot read the value map off
a position whose source you do not know. Sinkhorn makes that joint constraint differentiable.

TWO ARMS, because they answer different questions.
  solve     -- per-transformation optimisation. No training set, no generalisation: free logits for `(P, M)`, Adam on
               that transformation's own demonstrations. This is the CEILING -- does assignment + reconstruction suffice
               to recover a correspondence AT ALL with no group knowledge? A differentiable, group-agnostic stand-in for
               the hand-written `recover`. If this fails, no amortised version can work.
  amortise  -- the actual learned canonicaliser. A network reads the demonstrations and EMITS the score matrix in one
               forward pass, trained across transformations and evaluated on held-out ones. This is the arm directly
               comparable to `canonicaliser.py`'s 0.000.

TWO CONTROLS, each isolating one claim.
  --agg embed    reproduces the `canonicaliser.py` failure INSIDE this architecture: average the per-demonstration
                 embeddings FIRST, then score pairs. NOTES.md's diagnosis was "averaging is not intersecting"; summing
                 per-demonstration PAIR scores is the intersection (a product of per-demo compatibilities), and pooling
                 embeddings first is not. If `pair` works and `embed` does not, the fix is located at the aggregation,
                 not at Sinkhorn.
  --no_sinkhorn  row-softmax instead of doubly-stochastic. Keeps the pairwise scoring and drops the assignment
                 constraint, so it separates "score pairs" from "constrain to a permutation".

The demonstration sweep is the third measurement and is free in `solve`: how many demonstrations does a correspondence
NEED? Two input positions are indistinguishable when they carry identical values across every demonstration, which at
V=5 happens with probability 5^-K per pair -- so there is a genuine identifiability floor at small K, and the curve
should show it rather than have it hidden inside an average.

Usage:  python experiments/transformers/sinkhorn.py --mode solve
        python experiments/transformers/sinkhorn.py --mode amortise
"""
from __future__ import annotations

import argparse
import collections
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigroup import cayley, compile_progs, stratify
from canonicaliser import Block, views                      # reuse, do not reimplement
from h1_lid import L, V

MAXM = 5


def sinkhorn(log_alpha, n_iter=20, tau=0.5):
    """Project a score matrix onto the Birkhoff polytope IN LOG SPACE: alternate row and column normalisation, which
    converges to the doubly-stochastic matrix closest to `exp(log_alpha/tau)` in KL. Ends on a ROW normalisation so the
    rows are exact distributions -- the reconstruction below reads them as `p(source | output position)`.

    `tau` is the entropy knob: as `tau -> 0` the result approaches a hard permutation but the gradient vanishes, so it
    stays moderate here and the argmax is taken only at evaluation."""
    z = log_alpha / tau
    for _ in range(n_iter):
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)
    return z - torch.logsumexp(z, dim=-1, keepdim=True)


def recon_nll(lp, lm, x, y, n_iter, tau, doubly=True):
    """The one objective. `lp` (B,L,L) scores output-position <- input-position, `lm` (B,V,V) scores input-value ->
    output-value; `x`,`y` are (B,K,L) demonstrations. Marginalising the unknown source position gives a proper
    log-distribution over each output value, and the loss is its NLL on what the forward model actually produced.

    No label enters: `idx` and `vmp` are never referenced, only the (x, y) pairs that executing produces."""
    P = sinkhorn(lp, n_iter, tau) if doubly else torch.log_softmax(lp, dim=-1)
    M = sinkhorn(lm, n_iter, tau) if doubly else torch.log_softmax(lm, dim=-1)
    B = x.shape[0]
    lmx = M[torch.arange(B, device=x.device)[:, None, None], x]          # (B,K,L,V): the value map at each input slot
    pred = torch.logsumexp(P[:, None, :, :, None] + lmx[:, :, None, :, :], dim=3)
    return F.nll_loss(pred.reshape(-1, V), y.reshape(-1)), P, M


def score(P, M, idx_t, vmp_t):
    """Exact-match on BOTH halves of the canonical form, plus per-slot partial credit (an all-or-nothing number cannot
    tell 'no signal' from 'five of six positions right')."""
    pi, pv = P.argmax(-1), M.argmax(-1)
    return ((pi == idx_t).all(-1).float().mean().item(), (pi == idx_t).float().mean().item(),
            (pv == vmp_t).all(-1).float().mean().item(), (pv == vmp_t).float().mean().item())


def solve(xy, steps, lr, n_iter, tau, doubly=True, seed=0):
    """Per-transformation optimisation: free logits, Adam, that transformation's own demonstrations. Every
    transformation in the batch is solved independently and in parallel (the loss is a sum of per-item terms and the
    parameters do not interact), so the whole held-out set costs one run."""
    B = xy.shape[0]
    g = torch.Generator(device=xy.device).manual_seed(seed)
    lp = (0.01 * torch.randn(B, L, L, generator=g, device=xy.device)).requires_grad_(True)
    lm = (0.01 * torch.randn(B, V, V, generator=g, device=xy.device)).requires_grad_(True)
    opt = torch.optim.Adam([lp, lm], lr=lr)
    x, y = xy[:, :, :L], xy[:, :, L:]
    for _ in range(steps):
        loss, _, _ = recon_nll(lp, lm, x, y, n_iter, tau, doubly)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        loss, P, M = recon_nll(lp, lm, x, y, n_iter, tau, doubly)
    return P, M, loss.item()


class Matcher(nn.Module):
    """The amortised canonicaliser: demonstrations -> a score matrix, in one forward pass.

    A bidirectional transformer runs over each demonstration's `2L` tokens, with a positional code that DISTINGUISHES
    input slot j from output slot j -- the thing `canonicaliser.py`'s `cross` pooling destroyed, taking the within-demo
    pairing with it. Then, per demonstration, every output slot scores every input slot; those `(B,K,L,L)` scores are
    SUMMED over demonstrations, which in log space is the product of per-demonstration compatibilities: the INTERSECTION
    constraint, computed where it can actually be computed. `--agg embed` averages the embeddings first instead, which is
    the failure mode, kept runnable as the control.

    The value map comes off a pooled code through an MLP, because that half was never the problem (0.689 held-out)."""

    def __init__(self, d=128, n_layer=2, n_head=4, agg="pair"):
        super().__init__()
        self.agg = agg
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, 2 * L, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d)
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.mhead = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, V * V))
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, xy):
        B, K, T = xy.shape
        h = self.emb(xy.reshape(B * K, T)) + self.pos
        for b in self.blocks:
            h = b(h)
        h = self.norm(h).reshape(B, K, T, -1)
        hin, hout = h[:, :, :L], h[:, :, L:]
        if self.agg == "pair":                                           # score pairs per demo, THEN intersect
            s = torch.einsum("bkid,bkjd->bkij", self.q(hout), self.k(hin)) * self.scale
            lp = s.sum(1)
        else:                                                            # average embeddings first -- the control
            lp = torch.einsum("bid,bjd->bij", self.q(hout.mean(1)), self.k(hin.mean(1))) * self.scale
        return lp, self.mhead(h.mean((1, 2))).reshape(B, V, V)


def split(progs, n_held, n_train, seed):
    """Depth-stratified, because a prefix of a depth-ordered list samples only shallow transformations -- an apparatus
    bug this line has already paid for twice."""
    g0 = torch.Generator().manual_seed(seed)
    by = collections.defaultdict(list)
    for w in progs:
        by[len(w)].append(w)
    held, pool = [], []
    frac = n_held / len(progs)
    for _d, ws in sorted(by.items()):
        order = torch.randperm(len(ws), generator=g0).tolist()
        cut = min(len(ws) - 1, max(3, int(round(frac * len(ws)))))
        held += [ws[i] for i in order[:cut]]
        pool += [ws[i] for i in order[cut:]]
    order = torch.randperm(len(pool), generator=g0).tolist()
    return [pool[i] for i in order][:n_train], held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="solve", choices=["solve", "amortise"])
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--held", type=int, default=400)
    ap.add_argument("--eval_cap", type=int, default=200)
    ap.add_argument("--demos", type=int, default=8)
    ap.add_argument("--sweep", default="1,2,4,8,16")
    ap.add_argument("--sink_iter", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--no_sinkhorn", action="store_true")
    ap.add_argument("--agg", default="pair", choices=["pair", "embed"])
    ap.add_argument("--solve_steps", type=int, default=400)
    ap.add_argument("--solve_lr", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    doubly = not args.no_sinkhorn

    _probe_x, table = cayley(dev, args.seed)
    progs = [w for w in table.values() if 0 < len(w) <= MAXM]
    train, held = split(progs, args.held, args.n_train, args.seed)
    he = stratify(held, args.eval_cap)
    print(f"device {dev} | universe depth<={MAXM}: {len(progs)} | train {len(train)} | held-out {len(held)} "
          f"| eval {len(he)}")
    print(f"assignment: {'SINKHORN doubly-stochastic' if doubly else 'row-softmax ONLY (ablated)'}, "
          f"tau {args.tau}, {args.sink_iter} iters")
    print("objective: reconstruction of executed demonstrations. `idx` and `vmp` are NEVER supplied.\n")

    tabs_he = compile_progs(he, dev)
    idx_he, vmp_he = tabs_he
    sel_he = torch.arange(len(he), device=dev)

    if args.mode == "solve":
        print(f"{'demos':>6}{'IDX exact':>12}{'IDX slot':>11}{'VMP exact':>12}{'VMP slot':>11}{'NLL':>9}{'secs':>7}")
        for k in [int(s) for s in args.sweep.split(",")]:
            g = torch.Generator(device=dev).manual_seed(args.seed + k)
            xy = views(sel_he, k, tabs_he, dev, g)
            t0 = time.time()
            P, M, nll = solve(xy, args.solve_steps, args.solve_lr, args.sink_iter, args.tau, doubly, args.seed)
            ie, isl, ve, vsl = score(P, M, idx_he, vmp_he)
            print(f"{k:>6}{ie:>12.3f}{isl:>11.3f}{ve:>12.3f}{vsl:>11.3f}{nll:>9.4f}{time.time() - t0:>7.0f}")
        print("\nIDX exact is the number to read: `canonicaliser.py` scored 0.000 on it with a contrastive code, and the")
        print("hand-written `recover` in `localise.py` scores 1.000 using the group factorisation this file never sees.")
        print("Per-slot columns give partial credit, so a near-miss is distinguishable from no signal at all.")
        return

    tabs_tr = compile_progs(train, dev)
    net = Matcher(agg=args.agg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    g = torch.Generator(device=dev).manual_seed(args.seed)
    n_tr = len(train)

    @torch.no_grad()
    def evaluate(xy, idx_t, vmp_t):
        lp, lm = net(xy)
        _, P, M = recon_nll(lp, lm, xy[:, :, :L], xy[:, :, L:], args.sink_iter, args.tau, doubly)
        return score(P, M, idx_t, vmp_t)

    g_he = torch.Generator(device=dev).manual_seed(args.seed + 777)
    xy_he = views(sel_he, args.demos, tabs_he, dev, g_he)
    # The SAME evaluation on TRAINING transformations, which is the diagnostic that broke `canonicaliser.py` open: there
    # retrieval on train (0.059) matched held-out (0.053), showing the objective never solved its own training task, so
    # the failure was not about generalisation at all. A train/held gap and a train/held plateau want different fixes.
    tr_s = stratify(train, args.eval_cap)
    tabs_trs = compile_progs(tr_s, dev)
    g_tr = torch.Generator(device=dev).manual_seed(args.seed + 778)
    xy_tr = views(torch.arange(len(tr_s), device=dev), args.demos, tabs_trs, dev, g_tr)
    row = lambda tag, nll, t, h, s: print(
        f"{tag:>6}{nll:>11}{t[0]:>8.3f}/{h[0]:<7.3f}{t[1]:>8.3f}/{h[1]:<7.3f}"
        f"{t[2]:>8.3f}/{h[2]:<7.3f}{t[3]:>8.3f}/{h[3]:<7.3f}{s:>7.0f}")
    print(f"{'step':>6}{'train NLL':>11}{'IDX exact':>16}{'IDX slot':>16}{'VMP exact':>16}{'VMP slot':>16}{'secs':>7}")
    row("init", "-", evaluate(xy_tr, *tabs_trs), evaluate(xy_he, idx_he, vmp_he), 0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        sel = torch.randperm(n_tr, generator=g, device=dev)[:args.batch]
        xy = views(sel, args.demos, tabs_tr, dev, g)
        lp, lm = net(xy)
        loss, _, _ = recon_nll(lp, lm, xy[:, :, :L], xy[:, :, L:], args.sink_iter, args.tau, doubly)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, args.steps // 6) == 0 or step == args.steps:
            row(str(step), f"{loss.item():.4f}", evaluate(xy_tr, *tabs_trs),
                evaluate(xy_he, idx_he, vmp_he), time.time() - t0)
    print(f"\nEach cell is TRAIN/HELD-OUT, one forward pass, {args.demos} demonstrations each, agg={args.agg}.")
    print("Comparison: contrastive code IDX 0.000/0.000 (`canonicaliser.py`), VMP 0.689 held-out.")
    print("A chance permutation is 1/720 = 0.0014 exact and 1/6 = 0.167 per slot; chance VMP is 1/120 and 1/5 = 0.200.")


if __name__ == "__main__":
    main()
