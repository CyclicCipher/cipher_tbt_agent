"""THE GATE: does the CANONICAL STATE alone make factorisation learnable, or was it the change of TARGET?

WHY THIS RUNS BEFORE ANYTHING ELSE. `canonical.py` learned distance-to-identity from an 11-token canonical group element
and reached held-out exact 0.724, after the same target learned from BEHAVIOUR reached only 0.276. It is tempting to read
that as "canonicalisation fixed it". But two things changed across this line's experiments, not one:

    (1) the INPUT   -- from a behavioural sample (8 demonstrations) to a canonical group element
    (2) the TARGET  -- from the PROGRAM (discontinuous) to the DISTANCE (smooth)

Every earlier 0.000 (`decompose.py`, `bigroup.py`) had a behavioural input AND a program target, so it cannot separate
them. This file holds the input fixed at the canonical state and varies ONLY the target, training two models with the same
architecture, the same steps and the same split:

    canonical state -> DISTANCE     (the `canonical.py` result, re-derived here so the split is provably identical)
    canonical state -> PROGRAM      (the question)

WHAT EACH OUTCOME MEANS, fixed in advance:
  * PROGRAM fails while DISTANCE succeeds  =>  the TARGET did the work. Canonicalisation was necessary and not
    sufficient, the discontinuous-target reading of this whole line stands, and for a real domain the priority is the
    cost-to-go rather than the state representation.
  * PROGRAM succeeds                       =>  canonicalisation ALONE was sufficient, and **the central claim of this
    line is partly retracted**: the program would have been learnable all along given a non-behavioural input, and every
    0.000 in `decompose.py` / `bigroup.py` was about the INPUT being a sample rather than about the target being
    discontinuous. That is the branch that makes this worth running first.

Scoring is by FUNCTION (`PROG-fn`): the emitted program is compared to the target by its canonical tables, so an
equivalent writing counts as correct and multimodality in the target cannot cause an undercount. Decoding is free-running.

Usage:  python experiments/transformers/canon2prog.py
"""
from __future__ import annotations

import argparse
import collections
import math
import time

import torch
import torch.nn.functional as F

from bigroup import NAMES, cayley, compile_progs, stratify
from h1_lid import L, V, Model

MAXM = 5
IDX0, VMP0 = 0, L                      # canonical state: idx values 0..L-1, then vmp values L..L+V-1
STATE = L + V                          # 11 tokens of state
PROG0 = STATE                          # program tokens follow
HALT = PROG0 + len(NAMES)
DIST0 = STATE                          # for the distance model, classes reuse the block after the state
N_VOCAB_PROG = HALT + 1
N_VOCAB_DIST = STATE + MAXM + 1


def split(progs, seed, held_n, n_train):
    """The split used by `canonical.py`, reproduced exactly so the two results are comparable."""
    g = torch.Generator().manual_seed(seed)
    by = collections.defaultdict(list)
    for w in progs:
        by[len(w)].append(w)
    held, pool = [], []
    frac = held_n / len(progs)
    for d, ws in sorted(by.items()):
        order = torch.randperm(len(ws), generator=g).tolist()
        cut = min(len(ws) - 1, max(3, int(round(frac * len(ws)))))
        held += [ws[i] for i in order[:cut]]
        pool += [ws[i] for i in order[cut:]]
    order = torch.randperm(len(pool), generator=g).tolist()
    return [pool[i] for i in order][:n_train], held


def state_tokens(tabs, sel):
    idx_t, vmp_t = tabs
    return torch.cat([idx_t[sel] + IDX0, vmp_t[sel] + VMP0], dim=1)


def make_dist(B, tabs, lens, dev, g, fixed=None):
    n = tabs[0].shape[0]
    sel = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, n, (B,), generator=g, device=dev))
    return torch.cat([state_tokens(tabs, sel), (DIST0 + lens[sel])[:, None]], dim=1)


def make_prog(B, tabs, comp, lens, dev, g, fixed=None):
    """Canonical state, then the program and HALT. Everything past the first HALT is filler and is masked out."""
    n = tabs[0].shape[0]
    sel = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, n, (B,), generator=g, device=dev))
    body = torch.cat([PROG0 + comp[sel], torch.full((B, 1), HALT, device=dev)], dim=1)
    pos = torch.arange(MAXM + 1, device=dev)[None, :]
    cut = lens[sel][:, None]
    stream = torch.where(pos < cut, body, torch.full_like(body, HALT))
    return torch.cat([state_tokens(tabs, sel), stream], dim=1), cut.squeeze(1)


def train_model(kind, tabs, comp, lens, dev, args):
    n_vocab = N_VOCAB_PROG if kind == "prog" else N_VOCAB_DIST
    n_tok = STATE + (MAXM + 1 if kind == "prog" else 1)
    model = Model(max_len=n_tok + 2, pos="rope", n_vocab=n_vocab).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98),
                            fused=(dev == "cuda"))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    trainer = torch.compile(model) if dev == "cuda" else model
    g = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()
    for _ in range(args.steps):
        if kind == "dist":
            tok = make_dist(args.batch, tabs, lens, dev, g)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logits = trainer(tok)[:, -2]
            loss = F.cross_entropy(logits.float(), tok[:, -1])
        else:
            tok, cut = make_prog(args.batch, tabs, comp, lens, dev, g)
            msk = torch.zeros_like(tok, dtype=torch.bool)
            pos = torch.arange(MAXM + 1, device=dev)[None, :]
            msk[:, STATE:] = pos <= cut[:, None]                 # only the program and its HALT are targets
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logits = trainer(tok)[:, :-1]
            m = msk[:, 1:]
            loss = F.cross_entropy(logits[m].float(), tok[:, 1:][m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model, time.time() - t0, loss.item()


@torch.no_grad()
def eval_dist(model, ws, dev):
    tabs = compile_progs(ws, dev)
    lens = torch.tensor([len(w) for w in ws], device=dev)
    toks = state_tokens(tabs, torch.arange(len(ws), device=dev))
    pred = model(toks)[:, -1, DIST0:DIST0 + MAXM + 1].argmax(-1).float()
    return (pred == lens.float()).float().mean().item(), (pred - lens.float()).abs().mean().item()


@torch.no_grad()
def eval_prog(model, ws, dev, n=32):
    """Free-running program emission from the canonical state, scored by FUNCTION: the emitted program's canonical
    tables must equal the target's, so an equivalent writing counts."""
    tabs = compile_progs(ws, dev)
    fn, first = [], []
    for t, w in enumerate(ws):
        toks = state_tokens(tabs, torch.full((n,), t, dtype=torch.long, device=dev))
        seq = toks
        gen = torch.zeros(n, 0, dtype=torch.long, device=dev)
        for _ in range(MAXM + 1):
            nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
            seq, gen = torch.cat([seq, nxt], dim=1), torch.cat([gen, nxt], dim=1)
        want = (tabs[0][t], tabs[1][t])
        hits = 0
        for row in gen.tolist():
            ids = []
            for tk in row:
                if tk < PROG0 or tk >= HALT:
                    break
                ids.append(tk - PROG0)
            if not ids or len(ids) > MAXM:
                continue
            gi, gv = compile_progs([tuple(ids)], dev)
            if torch.equal(gi[0], want[0]) and torch.equal(gv[0], want[1]):
                hits += 1
        fn.append(hits / n)
        first.append((gen[:, 0] == PROG0 + w[0]).float().mean().item())
    return sum(fn) / len(fn), sum(first) / len(first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--held", type=int, default=400)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_cap", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    _probe, table = cayley(dev, args.seed)
    progs = [w for w in table.values() if 0 < len(w) <= MAXM]
    train, held = split(progs, args.seed, args.held, args.n_train)
    print(f"device {dev} | universe depth<={MAXM}: {len(progs)} transformations | train {len(train)} | "
          f"held-out {len(held)} | INPUT is the {STATE}-token canonical state in BOTH arms")

    tabs = compile_progs(train, dev)
    comp = torch.tensor([[w[min(j, len(w) - 1)] for j in range(MAXM)] for w in train], device=dev)
    lens = torch.tensor([len(w) for w in train], device=dev)
    tr_s, he_s = stratify(train, args.eval_cap), stratify(held, args.eval_cap)

    print(f"\n{'target':<12}{'train':>10}{'HELD-OUT':>11}{'extra':>26}{'secs':>7}")
    md, sd, ld = train_model("dist", tabs, comp, lens, dev, args)
    a_tr, m_tr = eval_dist(md, tr_s, dev)
    a_he, m_he = eval_dist(md, he_s, dev)
    print(f"{'DISTANCE':<12}{a_tr:>10.3f}{a_he:>11.3f}{'MAE ' + f'{m_tr:.2f} / {m_he:.2f}':>26}{sd:>7.0f}")

    mp, sp, lp = train_model("prog", tabs, comp, lens, dev, args)
    p_tr, f_tr = eval_prog(mp, tr_s, dev)
    p_he, f_he = eval_prog(mp, he_s, dev)
    print(f"{'PROGRAM':<12}{p_tr:>10.3f}{p_he:>11.3f}"
          f"{'first-prim ' + f'{f_tr:.3f} / {f_he:.3f}':>26}{sp:>7.0f}")

    print(f"\nBoth arms: same architecture, same {args.steps} steps, same split, same {STATE}-token canonical input.")
    print("DISTANCE is exact-match on the distance class; PROGRAM is FUNCTIONAL match (an equivalent writing counts),")
    print(f"free-running. Chance: distance {1 / (MAXM + 1):.3f}; first primitive {1 / len(NAMES):.3f}.")
    print("\nVERDICT: if PROGRAM held-out is near zero while DISTANCE is high, the TARGET did the work and the")
    print("discontinuous-target reading stands. If PROGRAM held-out is high, canonicalisation alone was sufficient")
    print("and this line's central claim is partly retracted.")


if __name__ == "__main__":
    main()
