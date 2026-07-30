"""CAN IDENTIFICATION BE LEARNED? The coverage sweep, on decomposition ALONE.

THE DEFLATIONARY HYPOTHESIS THIS TESTS, and it should be tested before any clever mechanism is built. `identify.py`
showed the model executes a named program at 0.997 and never once names one correctly (0.000). The interesting reading is
that identification needs a different kind of machinery. The BORING reading is that it needs different DATA:

    EXECUTION composes. Thirty supervised tasks teach it, because the primitives are shared across every task, so a
    novel combination reuses machinery that thirty tasks already trained.
    IDENTIFICATION may not compose. Thirty supervised tasks may teach it only thirty (demonstrations -> program) labels,
    with nothing shared to generalise from.

If that is the whole story, identification is fixable by COVERAGE — and coverage is free, because a system that can
execute can generate its own labelled identification data (sample a program, run it, keep the pair). That is wake-sleep /
analysis-by-synthesis used as a LEARNING algorithm rather than an inference one, and its biological analogue is replay
(`reference_exploration_replay`). No new labels and no new mechanism required.

So this file isolates decomposition and sweeps how much of the program space it is trained on.

THE TASK, stripped to the one ability. Given `n_demos` demonstrations `[in, out]` of an unfamiliar function, emit the
PROGRAM: `ID_1 ... ID_m HALT`. No blocks, no intermediates, no execution — the model is asked only to name the parts.
Twelve primitives give **174 distinct functions** of length <= 3, so unlike the 5-primitive setting there is room for a
real coverage sweep. The held-out set is FIXED across every condition and training sets are NESTED, so nothing varies but
supply — the design `diversity.py` used, for the same reason.

WHAT EACH OUTCOME MEANS, fixed in advance:
  * identification generalises as coverage grows  =>  the deflationary story is right. The original failure was that
    identification saw 30 of 155 programs with nothing shared to generalise from, and self-generated data fixes it. No
    proposal mechanism needed, and `search.py`'s enumeration is a scaffold rather than the answer.
  * it stays flat even at near-total coverage  =>  identification is ITSELF a compositional problem that gradient
    learning does not crack, which is a much stronger result: it would mean amortised decomposition is not learnable
    here, and that search or a structured proposal mechanism is the only route rather than a convenience.

MEASUREMENT NOTE. A function can have several writings, so scoring the emitted program against one canonical writing
would undercount. `PROG-fn` compares the emitted program's FUNCTION SIGNATURE to the target's, so an equivalent
decomposition counts as correct; `PROG-ex` is the stricter exact-writing match, reported alongside. Chance for ID@1 is
1/12 = 0.083.

Usage:  python experiments/transformers/decompose.py --n_train 10
        python experiments/transformers/decompose.py --n_train 134
"""
from __future__ import annotations

import argparse
import itertools
import math
import time

import torch
import torch.nn.functional as F

from chaining import apply_task
from diversity import NAMES
from h1_lid import L, V, Model

MAXM = 3


def universe(seed, n_test=40):
    """Every DISTINCT function writable in <= MAXM primitives over all 12, deduplicated across lengths with the shorter
    writings claiming their signatures first. Returns a FIXED held-out set and a NESTED training pool, plus a
    signature table used to score equivalent decompositions as correct."""
    g = torch.Generator().manual_seed(seed)
    probe = torch.randint(0, V, (24, L), generator=g)
    ident = tuple(probe.flatten().tolist())
    sig_of, seen, keep = {}, {}, []
    for m in range(1, MAXM + 1):
        for prog in itertools.product(range(len(NAMES)), repeat=m):
            s = tuple(apply_task(probe, prog).flatten().tolist())
            sig_of[prog] = s
            if s == ident or s in seen:
                continue
            seen[s] = prog
            keep.append(prog)
    order = torch.randperm(len(keep), generator=g).tolist()
    keep = [keep[i] for i in order]
    return keep[:n_test], keep[n_test:], sig_of


def make(B, n_demos, tasks, dev, g, comp, lens, fixed=None):
    """`n_demos` demonstrations of `[in, out]`, then the PROGRAM and HALT. Everything past the first HALT is filler and
    is masked out of the loss, so there is nothing to copy."""
    ID0, HALT = V, V + len(NAMES)
    sel = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, len(tasks), (B,), generator=g, device=dev))
    x = torch.randint(0, V, (B, n_demos, L), generator=g, device=dev)
    y = torch.empty_like(x)
    for t in sel.unique():                       # the demonstrations are the only evidence the model gets
        m = sel == t
        y[m] = apply_task(x[m], tasks[int(t)])
    demos = torch.stack([x, y], dim=2).reshape(B, n_demos * 2 * L)
    prog = torch.cat([ID0 + comp[sel], torch.full((B, 1), HALT, device=dev)], dim=1)
    pos = torch.arange(MAXM + 1, device=dev)[None, :]
    cut = lens[sel][:, None]
    stream = torch.where(pos < cut, prog, torch.full_like(prog, HALT))
    return torch.cat([demos, stream], dim=1), cut.squeeze(1)


def loss_mask(B, n_demos, cut, dev):
    """Targets: every demonstration's output block (which is what forces the model to MODEL the function rather than
    pattern-match the program), plus the program up to and including the first HALT."""
    n = n_demos * 2 * L + MAXM + 1
    m = torch.zeros(B, n, dtype=torch.bool, device=dev)
    for k in range(n_demos):
        m[:, k * 2 * L + L:(k + 1) * 2 * L] = True
    pos = torch.arange(MAXM + 1, device=dev)[None, :]
    m[:, n_demos * 2 * L:] = pos <= cut[:, None]
    return m


@torch.no_grad()
def evaluate(model, tasks, dev, n_demos, comp_of, lens_of, sig_of, n=64):
    """Free-running: the model emits its own program tokens and consumes them. Reports ID@1, exact-writing match,
    FUNCTION match (an equivalent decomposition counts) and whether it got the length right."""
    if not tasks:
        return dict(id1=0.0, ex=0.0, fn=0.0, ln=0.0)
    ID0, HALT = V, V + len(NAMES)
    comp, lens = comp_of(tasks), lens_of(tasks)
    id1, ex, fn, ln = [], [], [], []
    for t, task in enumerate(tasks):
        tok, _cut = make(n, n_demos, tasks, dev, None, comp, lens, fixed=t)
        seq = tok[:, :n_demos * 2 * L]
        gen = torch.zeros(n, 0, dtype=torch.long, device=dev)
        for _ in range(MAXM + 1):
            nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
            seq, gen = torch.cat([seq, nxt], dim=1), torch.cat([gen, nxt], dim=1)
        want_sig = sig_of[task]
        id1.append((gen[:, 0] == ID0 + int(comp[t][0])).float().mean().item())
        e = f = l = 0
        for row in gen.tolist():
            ids = []
            for tk in row:
                if tk == HALT or tk < ID0:
                    break
                ids.append(tk - ID0)
            l += int(len(ids) == len(task))
            if not ids:
                continue
            e += int(tuple(ids) == tuple(task))
            f += int(sig_of.get(tuple(ids)) == want_sig)
        ex.append(e / n)
        fn.append(f / n)
        ln.append(l / n)
    N = len(tasks)
    return dict(id1=sum(id1) / N, ex=sum(ex) / N, fn=sum(fn) / N, ln=sum(ln) / N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=134, help="distinct programs the identifier is trained on")
    ap.add_argument("--n_test", type=int, default=40, help="held-out programs, FIXED across every condition")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--demos", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_max", type=int, default=40, help="TRAIN tasks evaluated (HELD-OUT is always evaluated in full)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    test, pool, sig_of = universe(args.seed, args.n_test)
    train = pool[:args.n_train]
    n_vocab = V + len(NAMES) + 1
    n_tok = args.demos * 2 * L + MAXM + 1
    print(f"device {dev} | DECOMPOSITION ONLY (no execution) | {len(NAMES)} primitives, "
          f"{len(test) + len(pool)} distinct functions | train {len(train)} (nested) | test {len(test)} (FIXED) | "
          f"seq {n_tok} | steps={args.steps}")

    comp_of = lambda ts: torch.tensor(
        [[t[min(j, len(t) - 1)] for j in range(MAXM)] for t in ts], device=dev)
    lens_of = lambda ts: torch.tensor([len(t) for t in ts], device=dev)
    tr_comp, tr_lens = comp_of(train), lens_of(train)

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
        tok, cut = make(args.batch, args.demos, train, dev, g, tr_comp, tr_lens)
        msk = loss_mask(args.batch, args.demos, cut, dev)[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = trainer(tok)[:, :-1]
        loss = F.cross_entropy(logits[msk].float(), tok[:, 1:][msk])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    print(f"\n|train|={len(train):<4} trained in {time.time() - t0:.0f}s | final loss {loss.item():.4f}")
    print(f"{'pool':<14}{'n':>5}{'ID@1':>8}{'PROG-fn':>9}{'PROG-ex':>9}{'LEN':>8}")
    # TRAIN is SUBSAMPLED for evaluation (it only has to confirm the model fit what it was given); HELD-OUT is always
    # evaluated in full, because it is the measurement.
    for name, tasks in (("TRAIN" + (f" (of {len(train)})" if len(train) > args.eval_max else ""),
                         train[:args.eval_max]), ("HELD-OUT", test)):
        r = evaluate(model, tasks, dev, args.demos, comp_of, lens_of, sig_of)
        print(f"{name:<14}{len(tasks):>5}{r['id1']:>8.3f}{r['fn']:>9.3f}{r['ex']:>9.3f}{r['ln']:>8.3f}")
    print(f"\nChance for ID@1 is 1/{len(NAMES)} = {1 / len(NAMES):.3f}. PROG-fn counts an EQUIVALENT decomposition as")
    print("correct (signature match); PROG-ex demands the exact writing. All numbers FREE-RUNNING.")
    print("CLAIM: if HELD-OUT PROG-fn rises with |train|, identification is learnable and the earlier failure was")
    print("coverage -- fixable by self-generated data, since a system that can execute can label its own programs.")
    print("If it stays flat at near-total coverage, amortised decomposition is NOT learnable here and search or a")
    print("structured proposal mechanism is the only route.")


if __name__ == "__main__":
    main()
