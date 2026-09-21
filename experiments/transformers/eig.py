"""EIG WITHOUT ENUMERATION: information gain measured over the SOLVER'S OWN hypothesis space.

WHY THIS AND NOT `bits.py`. `bits.py` computes the exact posterior by enumerating 2350 transformations and scoring all
15625 candidate probes. It gave this line its first real denominator, and it does not scale past a small closed universe
— `detective.py` refuses to enumerate for exactly that reason.

Worse, step 1b showed enumeration does not merely fail to scale, it answers a DIFFERENT QUESTION. The exact posterior
and the prior-free Sinkhorn solver rank probe sets in OPPOSITE order: the hand-designed all-distinct probe is the
posterior's best (0.00 bits left) and the solver's worst (0.110 recovery), and random probes are the reverse. The reason
is provable rather than empirical — if a probe's values are all distinct then for ANY permutation `idx'` there is a value
map `vmp'` reproducing the output (`vmp'[v] = y[(x∘idx')⁻¹(v)]`, well-defined since `x∘idx'` is a bijection), so an
all-distinct probe constrains the permutation not at all. Only REPEATED values break the confound. The posterior escapes
it by knowing the answer is one of 2350 specific pairs; a solver searching 720 × 120 cannot.

**So expected information gain has to be computed over the hypothesis space of whoever will read the answer.** Probe
quality is not intrinsic. That is what makes this file a prerequisite for the choosing-beats-sampling test rather than a
follow-up to it.

THE METHOD — sample the posterior instead of enumerating it. Run the solver R times from different initialisations on
the observations so far. Where the data determines the answer they agree; where it does not they spread, and that spread
IS a sample from the solver's posterior. Then score a candidate probe by how much the samples DISAGREE about what it
would return:

    EIG(x)  ≈  H( { h(x) : h ~ posterior } )

Maximal when the surviving hypotheses predict different outcomes, zero when they all predict the same thing — which is
exactly expected information gain when the observation is deterministic given the hypothesis, and is Query-by-Committee
/ BALD. Nothing is enumerated on either side: hypotheses are SAMPLED (R of them) and candidates are SAMPLED (C of them),
so both axes scale by drawing more rather than by listing everything.

Before any observation there is nothing to fit, so the R hypotheses are drawn from the prior — uniform permutations of
positions and of values — which is the solver's own prior stated honestly, and is where its disagreement with the
enumerating observer begins.

WHAT IS MEASURED. Solver recovery after a budget of probes, chosen by sampled EIG vs random vs the hand-designed set,
AND the bits each leaves for the enumerating observer. Both columns are reported for every arm precisely because step 1b
showed one does not predict the other.

Usage:  python experiments/transformers/eig.py
"""
from __future__ import annotations

import argparse
import math
import time

import torch

from bigroup import cayley, compile_progs, stratify
from bits import NCODE, respond as posterior_codes
from h1_lid import L, V
from sinkhorn import solve

MAXM = 5


def prior_samples(n, R, dev, g):
    """R hypotheses per task from the solver's own prior: uniform permutations of the L positions and of the V values.

    A permutation of values rather than an arbitrary map, because the solver's `M` is doubly stochastic and its argmax is
    therefore a permutation — the prior has to be the one the solver actually searches, not a more convenient one."""
    idx = torch.rand(n, R, L, generator=g, device=dev).argsort(-1)
    vmp = torch.rand(n, R, V, generator=g, device=dev).argsort(-1)
    return idx, vmp


def posterior_samples(xy, R, solver_steps, dev, seed):
    """R independent solver runs per task, batched into ONE call. Restarts differ because `solve` seeds its own noise
    across the whole batch, so stacking R copies of each task gives R genuinely different initialisations for free."""
    n = xy.shape[0]
    rep = xy[:, None].expand(n, R, *xy.shape[1:]).reshape(n * R, *xy.shape[1:])
    P, M, _ = solve(rep, solver_steps, 0.1, 20, 0.5, True, seed)
    return P.argmax(-1).reshape(n, R, L), M.argmax(-1).reshape(n, R, V)


def predict(idx_h, vmp_h, cands, pw):
    """Response CODE each sampled hypothesis predicts for each candidate probe: `(n, R, C)`."""
    n, R, _ = idx_h.shape
    C = cands.shape[0]
    xg = cands[:, idx_h].permute(1, 2, 0, 3)                       # (n, R, C, L) — the value at each permuted position
    y = vmp_h[:, :, None].expand(n, R, C, V).gather(3, xg)
    return (y * pw).sum(-1)


def disagreement(codes):
    """Plug-in entropy of the predicted-response distribution, per task and candidate: `(n, C)`.

    `p_i` is the fraction of samples agreeing with sample `i`, so `H = -(1/R) Σ_i log p_i` — computed by pairwise
    equality, which is cheap because R is small and avoids materialising a histogram over 15625 possible responses."""
    n, R, C = codes.shape
    c = codes.permute(0, 2, 1)                                     # (n, C, R)
    same = (c[:, :, :, None] == c[:, :, None, :]).float().sum(-1)  # (n, C, R): how many agree with each sample
    return -(torch.log(same / R)).mean(-1)


def choose(idx_h, vmp_h, cands, pw, used):
    """The probe the sampled posterior most disagrees about, EXCLUDING any already asked.

    The exclusion is not tidiness. Measured: the restart-ensemble collapses fast — 11.88 distinct hypotheses among R=12
    before any observation, 3.47 by the third probe — and once it agrees, disagreement is flat across candidates and the
    argmax is arbitrary. Without this guard 98.5% of tasks re-ask a question they have already asked, which is why
    sampled EIG LOST to random at budget 4 while winning at budget 2."""
    d = disagreement(predict(idx_h, vmp_h, cands, pw))
    if used:
        d = d.scatter(1, torch.stack(used, dim=1), float("-inf"))
    j = d.argmax(-1)
    return cands[j], j


def run_policy(policy, tabs, budget, R, cands, pw, dev, args, g):
    """Sequentially select `budget` probes per task, observing the true response after each. Returns the probe set."""
    idx_t, vmp_t = tabs
    n = idx_t.shape[0]
    sel = torch.arange(n, device=dev)
    chosen, used = [], []
    for b in range(budget):
        if policy == "random":
            x = torch.randint(0, V, (n, L), generator=g, device=dev)
        elif policy == "designed":
            x = torch.cat([torch.arange(V, device=dev), torch.tensor([b % V], device=dev)]).expand(n, L)
        else:
            if not chosen:                                          # nothing observed yet ⇒ sample the PRIOR
                ih, vh = prior_samples(n, R, dev, g)
            else:                                                   # otherwise sample the posterior given observations
                X = torch.stack(chosen, dim=1)
                ys = torch.stack([vmp_t[sel].gather(1, X[:, k].gather(1, idx_t[sel])) for k in range(X.shape[1])], 1)
                ih, vh = posterior_samples(torch.cat([X, ys], dim=2), R, args.solver_steps, dev, args.seed + b)
            x, j = choose(ih, vh, cands, pw, used)
            used.append(j)
        chosen.append(x)
    return torch.stack(chosen, dim=1)                               # (n, budget, L)


def evaluate(X, tabs, all_idx, all_vmp, uni_sel, pw, args):
    """Both columns, because step 1b showed neither predicts the other: what the SOLVER recovers from these probes, and
    how many bits they leave the ENUMERATING observer."""
    idx_t, vmp_t = tabs
    n, B, _ = X.shape
    sel = torch.arange(n, device=X.device)
    ys = torch.stack([vmp_t[sel].gather(1, X[:, b].gather(1, idx_t[sel])) for b in range(B)], dim=1)
    P, M, _ = solve(torch.cat([X, ys], dim=2), args.eval_steps, 0.1, 20, 0.5, True, args.seed)
    ie = ((P.argmax(-1) == idx_t).all(-1) & (M.argmax(-1) == vmp_t).all(-1)).float().mean().item()
    isl = (P.argmax(-1) == idx_t).float().mean().item()
    keep = torch.ones(n, all_idx.shape[0], dtype=torch.bool, device=X.device)
    for b in range(B):
        code = posterior_codes(all_idx, all_vmp, X[:, b], pw)
        keep &= code == code.gather(1, uni_sel[:, None])
    return ie, isl, torch.log2(keep.sum(1).float().clamp(min=1.0)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=150)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--samples", type=int, default=12, help="R: hypotheses drawn from the solver's posterior")
    ap.add_argument("--cands", type=int, default=512, help="C: candidate probes sampled per decision")
    ap.add_argument("--solver_steps", type=int, default=120)
    ap.add_argument("--eval_steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    _p, table = cayley(dev, args.seed)
    progs = [w for w in table.values() if 0 < len(w) <= MAXM]
    all_idx, all_vmp = compile_progs(progs, dev)
    pw = (V ** torch.arange(L, device=dev))
    ws = stratify(progs, args.tasks)
    where = {tuple(w): i for i, w in enumerate(progs)}
    uni_sel = torch.tensor([where[tuple(w)] for w in ws], device=dev)
    tabs = compile_progs(ws, dev)
    print(f"device {dev} | universe {len(progs)} ({math.log2(len(progs)):.2f} bits) | {len(ws)} tasks "
          f"| R={args.samples} posterior samples, C={args.cands} candidates — NOTHING enumerated")

    g = torch.Generator(device=dev).manual_seed(args.seed)
    cands = torch.randint(0, V, (args.cands, L), generator=g, device=dev)
    print(f"\n{'probes':<12}{'budget':>8}{'SOLVER exact':>14}{'solver slot':>13}{'bits left':>11}{'secs':>7}")
    for policy in ("random", "designed", "eig"):
        for B in (2, args.budget):
            t0 = time.time()
            X = run_policy(policy, tabs, B, args.samples, cands, pw, dev, args, g)
            ie, isl, bits = evaluate(X, tabs, all_idx, all_vmp, uni_sel, pw, args)
            dist = torch.tensor([[len(set(r)) for r in X[:, b].tolist()] for b in range(B)]).float().mean()
            print(f"{policy:<12}{B:>8}{ie:>14.3f}{isl:>13.3f}{bits:>11.2f}{time.time() - t0:>7.0f}"
                  f"   distinct {dist:.2f}/6")

    print("\nSOLVER exact = both halves of the canonical form recovered by the prior-free solver — the observer this")
    print("file's EIG is computed FOR. 'bits left' is the ENUMERATING observer's residual, shown alongside because")
    print("step 1b measured that the two disagree: designed probes are best for one and worst for the other.")
    print("Reference points: 8 RANDOM demonstrations give the solver 1.000 (`sinkhorn.py`); random 2 give ~0.505.")


if __name__ == "__main__":
    main()
