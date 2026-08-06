"""BEHAVIOUR -> PROGRAM, end to end, with nothing hand-derived in the middle.

THE COMPOSITION THIS TESTS. `decompose.py` and `bigroup.py` measured behaviour -> program at **0.000** in every condition,
across two universes and a 36x coverage sweep. `canon2prog.py` then showed the target was never the problem: from a
CANONICAL state the program is learnable at **0.733** held-out. So the missing step was behaviour -> canonical state, and
`sinkhorn.py` now does it — exactly at 8 demonstrations, label-free, with no group knowledge.

At K=8 the composition needs no run: recovery is bit-identical to `compile_progs`, so `canon2prog.py`'s number carries
over by construction. What is genuinely unknown is the IMPERFECT regime. This file substitutes the RECOVERED canonical
state for the true one at every point on the demonstration curve and measures what the program model does with it.

THE PREDICTION, STATED BEFORE RUNNING, because this line has twice paid for attributing an effect after the fact.
**I expect a CLIFF, not graceful degradation — end-to-end accuracy ~ P(recovery exactly right) x 0.733.** The reason is
structural: a canonical state that is wrong in one position is not a NOISY description of the right transformation, it is
an EXACT description of a different one. The model will read it confidently and emit that other transformation's program,
which scores zero. Five-sixths of a permutation is worth nothing.

If that holds it is the flip side of the whole canonicalisation win, and worth as much: **a canonical representation has
no metric.** Its value comes from being exact, and there is no partial credit to be had — which is precisely why the
recovery step had to reach 1.000 rather than merely improve, and why `sinkhorn.py`'s data-efficiency curve (how many
demonstrations to reach exact) is the number that matters rather than its per-slot accuracy.

If instead accuracy sits ABOVE the multiplicative prediction, partial canonical information is usable, and that would be
the more interesting result: it would mean the program model tolerates a corrupted state, and the recovery step could be
allowed to be approximate.

WHAT IS HELD FIXED. The program model is trained ONCE, on TRUE canonical states of the TRAINING transformations — exactly
`canon2prog.py`'s setup, so its held-out 0.733 is the reference line. Only the INPUT at evaluation changes. The targets
are always the true tables, so a recovered state that names a different function scores zero, as it should.

Usage:  python experiments/transformers/endtoend.py
"""
from __future__ import annotations

import argparse
import time
import types

import torch

from bigroup import cayley, compile_progs, stratify
from canon2prog import HALT, MAXM, PROG0, STATE, split, state_tokens, train_model
from canonicaliser import views
from h1_lid import L
from sinkhorn import solve

_ = STATE                                                    # imported for the token-layout contract it documents


@torch.no_grad()
def eval_prog_from(model, in_tabs, true_tabs, dev):
    """Emit a program from the SUPPLIED canonical state, score it against the TRUE function.

    The split between `in_tabs` and `true_tabs` is the whole measurement: `canon2prog.py`'s `eval_prog` uses one table
    for both, which is right when the state is exact and hides everything when it is not.

    One sample per transformation, not 32: decoding is greedy and the input is fixed, so all 32 are identical — the
    redundancy `canon2prog.py` already flagged as wasteful."""
    n_task = in_tabs[0].shape[0]
    toks = state_tokens(in_tabs, torch.arange(n_task, device=dev))
    seq, gen = toks, torch.zeros(n_task, 0, dtype=torch.long, device=dev)
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
        if torch.equal(gi[0], true_tabs[0][t]) and torch.equal(gv[0], true_tabs[1][t]):
            hits += 1
    return hits / n_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--held", type=int, default=400)
    ap.add_argument("--eval_cap", type=int, default=150)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sweep", default="1,2,3,4,8")
    ap.add_argument("--solve_steps", type=int, default=400)
    ap.add_argument("--solve_lr", type=float, default=0.1)
    ap.add_argument("--sink_iter", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    _probe, table = cayley(dev, args.seed)
    progs = [w for w in table.values() if 0 < len(w) <= MAXM]
    train, held = split(progs, args.seed, args.held, args.n_train)
    he = stratify(held, args.eval_cap)
    print(f"device {dev} | universe depth<={MAXM}: {len(progs)} | train {len(train)} | held-out {len(held)} "
          f"| eval {len(he)}")
    print("program model trained ONCE on TRUE canonical states (canon2prog's setup); only the EVAL input varies.\n")

    tabs = compile_progs(train, dev)
    comp = torch.tensor([[w[min(j, len(w) - 1)] for j in range(MAXM)] for w in train], device=dev)
    lens = torch.tensor([len(w) for w in train], device=dev)
    targs = types.SimpleNamespace(steps=args.steps, batch=args.batch, lr=args.lr, seed=args.seed)
    model, secs, _loss = train_model("prog", tabs, comp, lens, dev, targs)
    print(f"program model trained in {secs:.0f}s")

    true_he = compile_progs(he, dev)
    sel = torch.arange(len(he), device=dev)
    ref = eval_prog_from(model, true_he, true_he, dev)
    print(f"reference: TRUE canonical state in, held-out functional accuracy {ref:.3f} "
          f"(canon2prog reports 0.733)\n")

    print(f"{'demos':>6}{'IDX exact':>11}{'VMP exact':>11}{'BOTH':>8}{'end-to-end':>12}"
          f"{'predicted':>11}{'secs':>7}")
    for k in [int(s) for s in args.sweep.split(",")]:
        g = torch.Generator(device=dev).manual_seed(args.seed + k)
        xy = views(sel, k, true_he, dev, g)
        t0 = time.time()
        P, M, _nll = solve(xy, args.solve_steps, args.solve_lr, args.sink_iter, args.tau, True, args.seed)
        rec = (P.argmax(-1), M.argmax(-1))
        ok_i = (rec[0] == true_he[0]).all(-1)
        ok_v = (rec[1] == true_he[1]).all(-1)
        both = (ok_i & ok_v).float().mean().item()
        acc = eval_prog_from(model, rec, true_he, dev)
        print(f"{k:>6}{ok_i.float().mean().item():>11.3f}{ok_v.float().mean().item():>11.3f}{both:>8.3f}"
              f"{acc:>12.3f}{both * ref:>11.3f}{time.time() - t0:>7.0f}")

    print("\n'BOTH' is the fraction where the recovered state is exactly right in BOTH halves; 'predicted' is BOTH x the")
    print("reference, i.e. the CLIFF hypothesis — a wrong canonical state names a different function and scores zero.")
    print("End-to-end ABOVE predicted means partial canonical information survives; equal means it does not.")
    print(f"\nBaseline this replaces: behaviour -> program measured 0.000 held-out (`decompose.py`, `bigroup.py`).")
    print(f"Nothing in this pipeline is hand-derived: {L}-slot correspondence and value map both come from `sinkhorn.py`,")
    print("which sees only executed demonstrations and never the group factorisation `localise.py` relies on.")


if __name__ == "__main__":
    main()
