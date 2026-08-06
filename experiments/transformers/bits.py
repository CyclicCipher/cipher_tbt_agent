"""THE BIT MEASUREMENT, and an agent that actually uses it.

WHAT WAS WRONG WITH THE LINE UNTIL NOW. Every claim about probe informativeness has been inferred BACKWARDS from answer
accuracy — `detective.py`'s 0.246 vs 0.008 vs 0.969 — which confounds three things: how good the question was, how good
the answerer was, and how sensitive the answerer is to being fed a query distribution it did not train on (measured, and
it is severe: an off-policy answerer scored 0.000). And the proxy standing in for informativeness was "distinct values
out of 6", which is not bits and does not become bits by being correlated with them.

Meanwhile the framing this line is named for — Gwern's *Death Note: L, Anonymity & Eluding Entropy* — is entirely about
choosing the observation that maximally SPLITS a hypothesis space. We derived the 11.2 bits by hand, hand-designed the
probes it implies, and then checked whether a model rediscovered them. That is TESTING FOR L, not implementing L. No
split was ever computed.

THIS FILE COMPUTES SPLITS. The universe is 2350 transformations, each a `(idx, vmp)` pair, so the posterior after a set
of probes is EXACT and cheap: keep every transformation whose response matches what was observed. `log2` of that count is
the remaining entropy, measured rather than argued, with no model and nothing trained.

THE AGENT. Given the posterior, greedy expected-information-gain probe selection is L's move stated literally:

    partition the surviving hypotheses by the response each would give to candidate x
    EIG(x) = log2|H| - SUM_r p(r) log2|H_r|          (the expected posterior entropy, subtracted)
    ask the x that maximises it, observe, discard everything inconsistent

The candidate space is `5^6 = 15625` and is enumerated EXHAUSTIVELY, so this is the true greedy optimum, not a sample of
it. That matters because we do not currently know the ceiling: a response carries up to log2(15625) = 13.9 bits, more
than the 11.2 needed, so ONE perfect probe could in principle suffice — and the pigeonhole duplicate says it cannot.
Nobody has computed which is right.

⚠ **THIS IS DELIBERATELY THE ENUMERATING VERSION, and that is a trade, not an oversight.** `detective.py` refuses to
enumerate on purpose — the bitter-lesson claim is that informativeness should EMERGE from outcome reward. Enumeration
does not scale past a small closed universe. What it buys is the thing an emergent policy cannot supply: the CEILING,
in bits, for what perfect play looks like here. Without it, "the model reached 0.246" has no denominator — we cannot say
whether its queries were half as good as possible or a twentieth, and we cannot tell a bad policy from a hard problem.

WHAT IS SCORED. Three policies, on the same held-out transformations, measured in bits remaining rather than downstream
accuracy: RANDOM, the HAND-DESIGNED probes from `detective.py`, and GREEDY EIG. The first probe is shared by every task
under any policy (no information has arrived yet, so nothing can condition on it), which is also why `detective.py`'s
adaptivity statistic only makes sense from probe 2 onward.

Usage:  python experiments/transformers/bits.py
"""
from __future__ import annotations

import argparse
import math
import time

import torch

from bigroup import cayley, compile_progs, stratify
from canon2prog import split
from h1_lid import L, V

MAXM = 5
NCODE = V ** L                                          # a response is L digits base V -- 15625 distinct outcomes


def respond(idx_t, vmp_t, X, pw):
    """Apply every transformation to every probe: `(N,L),(N,V)` x `(C,L)` -> `(C,N)` response CODES.

    `y[a] = vmp[x[idx[a]]]` for each, packed base-V into one integer so a whole response is a single comparable token."""
    C, N = X.shape[0], idx_t.shape[0]
    xg = X[:, None, :].expand(C, N, L).gather(2, idx_t[None].expand(C, N, L))
    y = vmp_t[None].expand(C, N, V).gather(2, xg)
    return (y * pw).sum(-1)


def eig(codes):
    """Expected information gain of each candidate, in bits. `codes` is `(C, N)` over the SURVIVING hypotheses.

    Grouping the survivors by the response they predict IS the partition; a candidate that splits them into many even
    groups is informative, one that sends them all to the same response tells you nothing and scores 0."""
    C, N = codes.shape
    cnt = torch.zeros(C, NCODE, device=codes.device)
    cnt.scatter_add_(1, codes, torch.ones(C, N, device=codes.device))
    elog = ((cnt / N) * torch.log2(cnt.clamp(min=1.0))).sum(1)     # SUM_r p(r) log2|H_r|; empty groups contribute 0
    return math.log2(N) - elog


def best_probe(idx_t, vmp_t, surv, cands, pw, chunk):
    """The greedy optimum over the FULL candidate set, chunked to fit. Returns `(probe, its EIG)`."""
    bi, bv = idx_t[surv], vmp_t[surv]
    best_g, best_x = -1.0, None
    for s in range(0, cands.shape[0], chunk):
        blk = cands[s:s + chunk]
        g = eig(respond(bi, bv, blk, pw))
        j = int(g.argmax())
        if float(g[j]) > best_g:
            best_g, best_x = float(g[j]), blk[j]
    return best_x, best_g


def designed_probe(b, dev):
    """`detective.py`'s hand-derived probe: all V values present (the maximum at L=6), with the DUPLICATED VALUE moved
    between probes so the second resolves the ambiguity the first must leave."""
    return torch.cat([torch.arange(V, device=dev), torch.tensor([b % V], device=dev)])


def run(policy, idx_t, vmp_t, tasks, budget, cands, pw, dev, chunk, g, supplied=None):
    """One episode per task: choose a probe, observe the TRUE response, discard every hypothesis inconsistent with it.
    Returns bits remaining after each probe, averaged over tasks, and the mean probes-to-identification."""
    N = idx_t.shape[0]
    traces, solved = [], []
    shared = None                                                  # probe 1 cannot depend on the task: nothing observed
    for t in tasks:
        surv = torch.arange(N, device=dev)
        row, hit = [math.log2(N)], budget + 1
        for b in range(budget):
            if policy == "random":
                x = torch.randint(0, V, (L,), generator=g, device=dev)
            elif policy == "designed":
                x = designed_probe(b, dev)
            elif policy == "learned":
                x = supplied[int(t)][min(b, supplied[int(t)].shape[0] - 1)]
            elif b == 0 and shared is not None:
                x = shared
            else:
                x, _ = best_probe(idx_t, vmp_t, surv, cands, pw, chunk)
                if b == 0:
                    shared = x
            code = respond(idx_t, vmp_t, x[None], pw)[0]           # every hypothesis's prediction
            surv = surv[code[surv] == code[int(t)]]                # keep only those matching the TRUE response
            row.append(math.log2(len(surv)))
            if len(surv) == 1 and hit > budget:
                hit = b + 1
        traces.append(row)
        solved.append(hit)
    avg = [sum(r[i] for r in traces) / len(traces) for i in range(budget + 1)]
    return avg, sum(1 for s in solved if s <= budget) / len(solved), shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--held", type=int, default=400)
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--cands", type=int, default=0, help="0 = enumerate all 5^6; otherwise sample this many")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--probes", default="", help="a detective.py --dump_probes file, scored as a 'learned' policy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    _probe, table = cayley(dev, args.seed)
    progs = [w for w in table.values() if 0 < len(w) <= MAXM]
    idx_t, vmp_t = compile_progs(progs, dev)
    N = len(progs)
    pw = (V ** torch.arange(L, device=dev))
    print(f"device {dev} | universe {N} transformations = {math.log2(N):.2f} bits to identify")

    # The candidate set, enumerated: every length-L string over V symbols, as base-V counting.
    allc = torch.arange(NCODE, device=dev)
    cands = torch.stack([(allc // (V ** j)) % V for j in range(L)], dim=1)
    if args.cands:
        cands = cands[torch.randperm(NCODE, device=dev)[:args.cands]]
    print(f"candidate probes: {cands.shape[0]} (exhaustive = {NCODE})")

    # With `--probes`, the task set comes FROM the dump so every policy is scored on exactly the transformations the
    # learned one faced (`detective.py` splits differently from `canon2prog.split`, so its held-out set is its own).
    supplied, policies = None, ["random", "designed", "greedy"]
    if args.probes:
        blob = torch.load(args.probes)
        where = {tuple(w): i for i, w in enumerate(progs)}
        pairs = [(where[w], blob["probes"][j].to(dev)) for j, w in enumerate(blob["progs"]) if w in where]
        pairs = pairs[:args.tasks]
        supplied, tasks = dict(pairs), [t for t, _ in pairs]
        policies.append("learned")
        print(f"learned probes: {len(pairs)} tasks x {blob['probes'].shape[1]} probes from {args.probes}")
    else:
        _train, held = split(progs, args.seed, args.held, args.n_train)
        hset = {tuple(w) for w in stratify(held, args.tasks)}
        tasks = [i for i, w in enumerate(progs) if tuple(w) in hset][:args.tasks]
    print(f"scored on {len(tasks)} held-out transformations, budget {args.budget} probes\n")

    g = torch.Generator(device=dev).manual_seed(args.seed)
    hdr = "".join(f"{'after ' + str(i):>10}" for i in range(args.budget + 1))
    print(f"{'policy':<10}{hdr}{'identified':>12}{'secs':>7}")
    firsts = {}
    for policy in policies:
        t0 = time.time()
        avg, ident, shared = run(policy, idx_t, vmp_t, tasks, args.budget, cands, pw, dev, args.chunk, g, supplied)
        firsts[policy] = shared
        print(f"{policy:<10}" + "".join(f"{b:>10.2f}" for b in avg) + f"{ident:>12.3f}{time.time() - t0:>7.0f}")

    print("\nEvery cell is BITS OF HYPOTHESIS SPACE REMAINING (log2 of the surviving count), averaged over tasks;")
    print("0.00 means the transformation is uniquely identified. 'identified' is the fraction reaching 0 within budget.")
    if firsts.get("greedy") is not None:
        d = designed_probe(0, dev)
        print(f"\ngreedy's first probe {firsts['greedy'].tolist()} vs the hand-derived {d.tolist()}")
        for nm, x in (("greedy", firsts["greedy"]), ("designed", d)):
            e = float(eig(respond(idx_t, vmp_t, x[None], pw))[0])
            print(f"  {nm:<9} EIG {e:.2f} bits, {len(set(x.tolist()))}/{L} distinct values")
    print(f"\nA single response carries at most log2({NCODE}) = {math.log2(NCODE):.1f} bits, which EXCEEDS the "
          f"{math.log2(N):.1f} needed —")
    print("so the number of probes required is set by what a probe can actually distinguish, not by counting bits.")
    print("Reference for `detective.py`: its learned policy averaged 2.63 distinct values, random 3.72, designed 5.00.")


if __name__ == "__main__":
    main()
