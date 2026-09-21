"""CANONICALISE THE OUTPUT: quotient the target space by functional equivalence.

THE ASYMMETRY THIS FIXES. `endtoend.py` recovers a canonical INPUT — one code per function — and then hands it to a
next-token model whose TARGET is one particular writing of the program. So the input was quotiented by equivalence and
the output was not, and the objective is still scoring surface form. We papered over it in the EVALUATION:
`canon2prog.py` scores by FUNCTION, "an equivalent writing counts", built precisely because token scoring marks correct
answers wrong for being phrased differently. The patch belongs in the objective.

AND THE AMBIGUITY IS NOT SMALL, which was the open question. `cayley` stores each function's MINIMAL program, so targets
are already canonical by length — but ties among equally-short programs are broken arbitrarily by BFS order. Enumerated:

    13 primitives, 2350 functions at depth <= 5
    minimal-length programs per function:  mean 12.53, max 120
    fraction of functions with MORE THAN ONE minimal writing:  0.902

So for 90% of functions the model is pushed toward one of ~12 equally valid, equally short answers, and penalised for
emitting any of the other eleven. That is the next-token blindspot in its concrete local form: the objective scores the
SPLIT between writings, which is a fact about BFS visit order, not about the world.

THE FIX, and why this particular one. Training on a random class member each step optimises `E_p[log P(p)]`, which
forces the model to SPREAD mass over the class — the opposite of what is wanted, since greedy decoding then emits the
mode of a spread distribution and may not complete any valid program. The right objective is the class MARGINAL:

    loss = -log SUM_{p in class} P(p | canonical state)

which is minimised by putting all mass on whichever single member the model finds easiest. It is indifferent to WHICH
writing wins, and that indifference is exactly what "quotient the output" means. Estimated over up to 16 sampled members
per function, which is a lower bound on the true class marginal and therefore a valid objective to descend.

MEASURED, against the same split and architecture:
  1. FUNCTIONAL accuracy, free-running, held-out — the number `canon2prog.py` reports as 0.733 and `endtoend.py` as
     0.762 with a single-writing target.
  2. CLASS MASS — the probability the model assigns to its whole equivalence class versus to the one BFS writing. This
     is the direct measurement of how much of the residual error was the objective splitting probability across
     equivalent answers rather than the model not knowing the function.

Usage:  python experiments/transformers/quotient.py
"""
from __future__ import annotations

import argparse
import collections
import math
import time

import torch
import torch.nn.functional as F

from bigroup import NAMES, PRIMS, cayley, compile_progs, stratify
from canon2prog import HALT, MAXM, PROG0, STATE, N_VOCAB_PROG, split, state_tokens
from h1_lid import L, V, Model

CMAX = 16                                    # class members kept per function; a lower bound on the class marginal


def equivalence_classes():
    """Every function's FULL set of minimal-length programs, by BFS over the Cayley graph keeping ALL parents rather
    than the first one found. `cayley` keeps only the first, which is where the arbitrary choice enters."""
    pos = {n: all(bool((PRIMS[n](torch.full((1, L), c)) == c).all()) for c in range(V)) for n in NAMES}
    start = (tuple(range(L)), tuple(range(V)))

    def step(st, k):
        i, v = st
        n = NAMES[k]
        if pos[n]:
            return (tuple(PRIMS[n](torch.tensor(i).unsqueeze(0)).squeeze(0).tolist()), v)
        return (i, tuple(PRIMS[n](torch.tensor(v).unsqueeze(0)).squeeze(0).tolist()))

    seen, classes, frontier = {start: 0}, {}, {start: [()]}
    for depth in range(1, MAXM + 1):
        nxt = collections.defaultdict(list)
        for st, ws in frontier.items():
            for k in range(len(NAMES)):
                s2 = step(st, k)
                if s2 in seen and seen[s2] < depth:
                    continue
                seen[s2] = depth
                nxt[s2] += [w + (k,) for w in ws]
        classes.update(nxt)
        frontier = nxt
    return classes


def pack(ws, cls_of, dev):
    """Each function's class as padded tensors: programs `(n, CMAX, MAXM)`, lengths `(n, CMAX)`, size `(n,)`."""
    P = torch.zeros(len(ws), CMAX, MAXM, dtype=torch.long)
    Ln = torch.ones(len(ws), CMAX, dtype=torch.long)
    N = torch.ones(len(ws), dtype=torch.long)
    for i, w in enumerate(ws):
        cls = cls_of[tuple(w)][:CMAX]
        N[i] = len(cls)
        for j, p in enumerate(cls):
            P[i, j, :len(p)] = torch.tensor(p)
            Ln[i, j] = len(p)
    return P.to(dev), Ln.to(dev), N.to(dev)


def seq_and_mask(state, prog, ln, dev):
    """`[canonical state | program | HALT | filler]`, with a mask over the program tokens and its HALT."""
    B = prog.shape[0]
    body = torch.cat([PROG0 + prog, torch.full((B, 1), HALT, device=dev)], dim=1)
    pos = torch.arange(MAXM + 1, device=dev)[None, :]
    stream = torch.where(pos < ln[:, None], body, torch.full_like(body, HALT))
    tok = torch.cat([state, stream], dim=1)
    msk = torch.zeros_like(tok, dtype=torch.bool)
    msk[:, STATE:] = pos <= ln[:, None]
    return tok, msk


def seq_logprob(model, tok, msk):
    """Total log-probability of each sequence's program region — the quantity the class marginal sums over."""
    logits = model(tok)[:, :-1]
    lp = torch.log_softmax(logits.float(), dim=-1).gather(2, tok[:, 1:, None]).squeeze(2)
    return (lp * msk[:, 1:].float()).sum(1)


@torch.no_grad()
def class_mass(model, state, P, Ln, N, dev):
    """Probability on the WHOLE class versus on its first (BFS-chosen) writing — how much the single-target objective
    was leaving on the table."""
    n = state.shape[0]
    st = state[:, None].expand(n, CMAX, STATE).reshape(n * CMAX, STATE)
    tok, msk = seq_and_mask(st, P.reshape(-1, MAXM), Ln.reshape(-1), dev)
    lp = seq_logprob(model, tok, msk).reshape(n, CMAX)
    valid = torch.arange(CMAX, device=dev)[None, :] < N[:, None]
    tot = torch.logsumexp(lp.masked_fill(~valid, float("-inf")), dim=1)
    return tot.exp().mean().item(), lp[:, 0].exp().mean().item()


@torch.no_grad()
def functional_acc(model, tabs, dev):
    """Free-running greedy emission from the canonical state, scored by FUNCTION — an equivalent writing counts."""
    n = tabs[0].shape[0]
    seq = state_tokens(tabs, torch.arange(n, device=dev))
    gen = torch.zeros(n, 0, dtype=torch.long, device=dev)
    for _ in range(MAXM + 1):
        nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
        seq, gen = torch.cat([seq, nxt], dim=1), torch.cat([gen, nxt], dim=1)
    hits = 0
    for t, row in enumerate(gen.tolist()):
        ids = []
        for tk in row:
            if tk < PROG0 or tk >= HALT:
                break
            ids.append(tk - PROG0)
        if not ids or len(ids) > MAXM:
            continue
        gi, gv = compile_progs([tuple(ids)], dev)
        if torch.equal(gi[0], tabs[0][t]) and torch.equal(gv[0], tabs[1][t]):
            hits += 1
    return hits / n


def train(mode, tabs, P, Ln, N, dev, args):
    model = Model(max_len=STATE + MAXM + 3, pos="rope", n_vocab=N_VOCAB_PROG).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    g = torch.Generator(device=dev).manual_seed(args.seed)
    n_fn, S = tabs[0].shape[0], (1 if mode == "single" else args.members)
    t0 = time.time()
    for _ in range(args.steps):
        sel = torch.randint(0, n_fn, (args.batch,), generator=g, device=dev)
        state = state_tokens(tabs, sel)
        if mode == "single":
            pick = torch.zeros(args.batch, 1, dtype=torch.long, device=dev)       # the one BFS writing
        else:
            # Sample S members WITH replacement; duplicates only inflate a term of the logsumexp slightly and cost
            # nothing in correctness, whereas rejection-sampling distinct members would need a per-row loop.
            pick = (torch.rand(args.batch, S, generator=g, device=dev) * N[sel][:, None]).long()
        prog = P[sel][torch.arange(args.batch, device=dev)[:, None], pick]        # (B, S, MAXM)
        ln = Ln[sel][torch.arange(args.batch, device=dev)[:, None], pick]         # (B, S)
        st = state[:, None].expand(args.batch, S, STATE).reshape(-1, STATE)
        tok, msk = seq_and_mask(st, prog.reshape(-1, MAXM), ln.reshape(-1), dev)
        lp = seq_logprob(model, tok, msk).reshape(args.batch, S)
        loss = -(torch.logsumexp(lp, dim=1) if mode == "class" else lp[:, 0]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--held", type=int, default=400)
    ap.add_argument("--eval_cap", type=int, default=150)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    _p, table = cayley(dev, args.seed)
    progs = [w for w in table.values() if 0 < len(w) <= MAXM]
    classes = equivalence_classes()
    # RE-KEY. `cayley` identifies a function by its action on a random probe; `equivalence_classes` identifies it by
    # `(idx, vmp)`. Those are different key spaces, and looking one up with the other silently returns a class of size 1
    # for every function — which is what the first run of this file reported (mean 1.00 against a true 12.53).
    _it, _vt = compile_progs(progs, dev)
    cls_of = {tuple(w): classes.get((tuple(_it[i].tolist()), tuple(_vt[i].tolist())), [tuple(w)])
              for i, w in enumerate(progs)}
    train_ws, held = split(progs, args.seed, args.held, args.n_train)
    he = stratify(held, args.eval_cap)
    sz = [len(cls_of[tuple(w)]) for w in progs]
    print(f"device {dev} | {len(progs)} functions | train {len(train_ws)} | held-out {len(held)} | eval {len(he)}")
    print(f"minimal writings per function: mean {sum(sz)/len(sz):.2f}, max {max(sz)}, "
          f"fraction >1 {sum(1 for s in sz if s > 1)/len(sz):.3f}\n")

    tabs_tr, tabs_he = compile_progs(train_ws, dev), compile_progs(he, dev)
    Ptr, Ltr, Ntr = pack(train_ws, cls_of, dev)
    Phe, Lhe, Nhe = pack(he, cls_of, dev)
    print(f"{'target':<10}{'FUNCTIONAL (held-out)':>23}{'class mass':>13}{'single mass':>13}{'secs':>7}")
    for mode in ("single", "class"):
        model, secs = train(mode, tabs_tr, Ptr, Ltr, Ntr, dev, args)
        acc = functional_acc(model, tabs_he, dev)
        cm, sm = class_mass(model, state_tokens(tabs_he, torch.arange(len(he), device=dev)), Phe, Lhe, Nhe, dev)
        print(f"{mode:<10}{acc:>23.3f}{cm:>13.3f}{sm:>13.3f}{secs:>7.0f}")

    print("\n'single' is the existing objective: one arbitrary minimal writing per function, which is what `cayley`'s")
    print("BFS happened to reach first. 'class' is the marginal over ALL its minimal writings, so the model may put its")
    print("mass wherever it likes within the class. Both are scored FUNCTIONALLY, free-running, on held-out functions.")
    print("Reference: 0.733 (`canon2prog.py`) / 0.762 (`endtoend.py`), both single-writing targets.")


if __name__ == "__main__":
    main()
