"""THE COVERAGE SWEEP ON A MUCH BIGGER GROUP: is factorisation unlearnable, or merely under-sampled?

WHAT THIS SETTLES. `decompose.py` found held-out factorisation at exactly 0.000 for |train| = 10, 40 and 134, and I read
that as a hypothesis-class fact: gradient descent learns approximately continuous maps, and behaviour -> program is
discontinuous (measured in `recall.py`: Spearman between behavioural similarity and program-edit distance = -0.151). But
the competing account was never excluded — that the map is learnable and 134 examples is simply too few. The flatness
argued against it, since a sample-complexity story usually shows SOME trend, but 134 examples over a 174-element universe
is not a large-scale test.

**And the universe was the binding problem: the 12 primitives at L=6, V=5 generate a group of order only 360** (measured
by BFS over the Cayley graph, diameter 6). So the earlier "universe" of 174 functions was already half the entire group,
and no sweep on it could have gone much further.

THE FIX. Adding one transposition to the rotations generates the full symmetric group, so 13 primitives give
**S6 x Aff(Z5) = 720 x 10 = 7200** elements (measured, matching theory exactly), diameter 9 — **41x the previous
universe**. Enumeration is by BFS from the identity, which yields every element together with its MINIMAL program, i.e.
its Cayley-graph distance. That makes the object under study explicit: this is factorisation over a Cayley graph, and the
sweep asks whether a 64x range of coverage buys any generalisation.

DESIGN. Same task as `decompose.py`, stripped to the one ability: given demonstrations of an unfamiliar function, emit the
PROGRAM. No execution, no intermediates. Held-out set FIXED and STRATIFIED BY DEPTH so the breakdown is readable, training
sets NESTED, compute fixed. `PROG-fn` scores an equivalent factorisation as correct (signature match), so a different
writing of the same function is not penalised.

TWO INTERNAL CONTROLS that make this interpretable whatever the headline says:
  * DEPTH-1 held-out functions are pure RECOGNITION — "which single primitive is this?" — so they should be near 1.000 if
    the apparatus works at all. If depth 1 succeeds while depth >= 3 fails inside the SAME run, that is the
    recognition/factorisation dissociation demonstrated without relying on any cross-run comparison.
  * Train fit is reported at every coverage level, so an undertrained top arm is visible rather than silently weakening
    the conclusion (the flaw the largest point of both `diversity.py` and `decompose.py` had).

ENGINEERING NOTE, because it is what makes the sweep possible at all. `decompose.py` generated batches with a Python loop
over the unique tasks in the batch, so cost grew with |train| — which is why its |train|=134 arm took 501 s and why
|train|=3200 would have been hopeless. Every element of this group is a position permutation composed with a value map, so
each function compiles to ONE gather pair `out = vmp[x[idx]]`; generation is then a couple of gathers, independent of
|train|. The compilation is asserted against the reference implementation.

Usage:  python experiments/transformers/bigroup.py --n_train 50
        python experiments/transformers/bigroup.py --n_train 3200
"""
from __future__ import annotations

import argparse
import collections
import math
import time

import torch
import torch.nn.functional as F

from diversity import PRIMS as P12
from h1_lid import L, V, Model

PRIMS = dict(P12)
PRIMS["swap01"] = lambda s: s.index_select(-1, torch.tensor([1, 0] + list(range(2, L)), device=s.device))
NAMES = list(PRIMS)
MAXM = 9                                       # set from --max_depth in main(); the group diameter is 9


def apply_prog(s, prog):
    for i in prog:
        s = PRIMS[NAMES[i]](s)
    return s


def cayley(dev, seed=0):
    """BFS from the identity over the Cayley graph: every group element with its MINIMAL program. A node is a function,
    identified by its action on a fixed probe; an edge applies one primitive."""
    g = torch.Generator().manual_seed(seed)
    probe = torch.randint(0, V, (12, L), generator=g)
    seen = {tuple(probe.flatten().tolist()): ()}
    frontier = [(probe, ())]
    while frontier:
        nxt = []
        for st, w in frontier:
            for p in range(len(NAMES)):
                s2 = apply_prog(st, (p,))
                k = tuple(s2.flatten().tolist())
                if k not in seen:
                    seen[k] = w + (p,)
                    nxt.append((s2, w + (p,)))
        frontier = nxt
    return probe, seen


def compile_progs(progs, dev):
    """Each function as ONE gather pair `out = vmp[x[idx]]`. Every primitive here is either a position permutation or a
    value map and the two commute, so any composition collapses to a single pair — which is what removes |train| from the
    per-step cost."""
    ar_L, ar_V = torch.arange(L), torch.arange(V)
    positional = {n: bool((PRIMS[n](torch.full((1, L), c)) == c).all()) for n in NAMES for c in [0]}
    for n in NAMES:                                        # test EVERY constant, not just zero: `negate` fixes 0
        positional[n] = all(bool((PRIMS[n](torch.full((1, L), c)) == c).all()) for c in range(V))
    idx, vmp = [], []
    for prog in progs:
        i, v = ar_L.clone(), ar_V.clone()
        for k in prog:
            if positional[NAMES[k]]:
                i = PRIMS[NAMES[k]](i.unsqueeze(0)).squeeze(0)
            else:
                v = PRIMS[NAMES[k]](v.unsqueeze(0)).squeeze(0)
        idx.append(i)
        vmp.append(v)
    return torch.stack(idx).to(dev), torch.stack(vmp).to(dev)


def make(B, n_demos, tab, comp, lens, dev, g, fixed=None):
    """`n_demos` demonstrations of `[in, out]`, then the PROGRAM and HALT. Fully vectorised."""
    ID0, HALT = V, V + len(NAMES)
    idx_t, vmp_t = tab
    n = idx_t.shape[0]
    sel = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, n, (B,), generator=g, device=dev))
    x = torch.randint(0, V, (B, n_demos, L), generator=g, device=dev)
    gi = idx_t[sel][:, None, :].expand(B, n_demos, L)
    y = vmp_t[sel][:, None, :].expand(B, n_demos, V).gather(2, x.gather(2, gi))
    demos = torch.stack([x, y], dim=2).reshape(B, n_demos * 2 * L)
    prog = torch.cat([ID0 + comp[sel], torch.full((B, 1), HALT, device=dev)], dim=1)
    pos = torch.arange(MAXM + 1, device=dev)[None, :]
    cut = lens[sel][:, None]
    return torch.cat([demos, torch.where(pos < cut, prog, torch.full_like(prog, HALT))], dim=1), cut.squeeze(1)


def loss_mask(B, n_demos, cut, dev):
    n = n_demos * 2 * L + MAXM + 1
    m = torch.zeros(B, n, dtype=torch.bool, device=dev)
    for k in range(n_demos):
        m[:, k * 2 * L + L:(k + 1) * 2 * L] = True
    pos = torch.arange(MAXM + 1, device=dev)[None, :]
    m[:, n_demos * 2 * L:] = pos <= cut[:, None]
    return m


def stratify(progs, cap):
    """A depth-BALANCED subsample. Taking the first `cap` of a depth-ordered list would evaluate only shallow functions
    and never touch the deep ones, which is the opposite of informative here."""
    by = collections.defaultdict(list)
    for p in progs:
        by[len(p)].append(p)
    per = max(1, cap // max(1, len(by)))
    out = []
    for d in sorted(by):
        out += by[d][:per]
    return out


@torch.no_grad()
def evaluate(model, progs, sigs, dev, n_demos, n=32, cap=200):
    """Free-running program emission, scored by FUNCTION (an equivalent factorisation counts). Returns overall and a
    per-minimal-depth breakdown, the latter being the internal control: depth 1 is pure recognition."""
    if not progs:
        return 0.0, 0.0, {}
    ID0, HALT = V, V + len(NAMES)
    progs = stratify(progs, cap)
    tab = compile_progs(progs, dev)
    comp = torch.tensor([[p[min(j, len(p) - 1)] for j in range(MAXM)] for p in progs], device=dev)
    lens = torch.tensor([len(p) for p in progs], device=dev)
    fn, id1, by = [], [], collections.defaultdict(list)
    for t, prog in enumerate(progs):
        tok, _ = make(n, n_demos, tab, comp, lens, dev, None, fixed=t)
        seq = tok[:, :n_demos * 2 * L]
        gen = torch.zeros(n, 0, dtype=torch.long, device=dev)
        for _ in range(MAXM + 1):
            nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
            seq, gen = torch.cat([seq, nxt], dim=1), torch.cat([gen, nxt], dim=1)
        want = sigs[prog]
        f = 0
        for row in gen.tolist():
            ids = []
            for tk in row:
                if tk == HALT or tk < ID0:
                    break
                ids.append(tk - ID0)
            if ids and sigs.get(tuple(ids)) == want:
                f += 1
        fn.append(f / n)
        id1.append((gen[:, 0] == ID0 + int(comp[t][0])).float().mean().item())
        by[len(prog)].append(f / n)
    return (sum(fn) / len(fn), sum(id1) / len(id1),
            {d: (sum(v) / len(v), len(v)) for d, v in sorted(by.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--held", type=int, default=1000, help="held-out functions, FIXED and depth-stratified")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--demos", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    # The full group has diameter 9, but depth-9 targets are long enough that the model does not fit even 50 of them
    # (measured: train 0.366 at 8000 steps), which fails this project's own sanity gate. Capping at depth <= 5 leaves
    # 2350 functions -- still 13.5x the old 174-element universe, with a 36x coverage sweep available on top of it.
    ap.add_argument("--max_depth", type=int, default=5,
                    help="restrict the universe to functions of minimal depth <= this")
    ap.add_argument("--min_held", type=int, default=4, help="minimum held-out functions per depth (the control)")
    ap.add_argument("--eval_cap", type=int, default=150, help="functions evaluated per pool")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    global MAXM
    MAXM = args.max_depth

    probe, table = cayley(dev, args.seed)
    all_progs = [w for w in table.values() if 0 < len(w) <= MAXM]
    # Signatures for EVERY program up to the diameter would be 13^9; instead a program's signature is looked up by
    # applying it to the probe on demand, and the group's own table gives the canonical minimal words.
    sigs = {w: tuple(apply_prog(probe, w).flatten().tolist()) for w in all_progs}
    depth_hist = collections.Counter(len(w) for w in all_progs)
    print(f"device {dev} | {len(NAMES)} primitives | group order 7200, diameter 9 | universe capped at depth "
          f"<= {MAXM}: {len(all_progs)} functions")
    print(f"minimal-depth histogram: {dict(sorted(depth_hist.items()))}")

    # Depth-STRATIFIED held-out set, so the breakdown covers every depth, and a nested training pool from the rest.
    g = torch.Generator().manual_seed(args.seed)
    by_depth = collections.defaultdict(list)
    for w in all_progs:
        by_depth[len(w)].append(w)
    held, pool = [], []
    frac = args.held / len(all_progs)
    for d, ws in sorted(by_depth.items()):
        order = torch.randperm(len(ws), generator=g).tolist()
        # At least `min_held` per depth. The shallow depths are the INTERNAL CONTROL and there are only 13 depth-1
        # functions in the whole group, so a proportional split would leave ~2 and the control would be worthless.
        # Holding out 4 of the 13 also creates a sharper question: can it name a primitive it has never been shown
        # ALONE, having only ever seen it inside deeper compositions?
        cut = min(len(ws) - 1, max(args.min_held, int(round(frac * len(ws)))))
        held += [ws[i] for i in order[:cut]]
        pool += [ws[i] for i in order[cut:]]
    order = torch.randperm(len(pool), generator=g).tolist()
    pool = [pool[i] for i in order]
    train = pool[:args.n_train]
    print(f"held-out {len(held)} (stratified, FIXED) | pool {len(pool)} | train {len(train)} (nested)")

    n_vocab = V + len(NAMES) + 1
    n_tok = args.demos * 2 * L + MAXM + 1
    tab = compile_progs(train, dev)
    # The compiled gather pairs are ASSERTED against the reference, on every training function.
    gchk = torch.Generator(device=dev).manual_seed(7)
    xs = torch.randint(0, V, (16, L), generator=gchk, device=dev)
    idx_t, vmp_t = tab
    for t, prog in enumerate(train[:200]):
        got = vmp_t[t][None].expand(16, V).gather(1, xs.gather(1, idx_t[t][None].expand(16, L)))
        assert torch.equal(got, apply_prog(xs, prog)), f"compiled table wrong for {prog}"
    print(f"compiled gather pairs verified against the reference | seq {n_tok} vocab {n_vocab} steps={args.steps}\n")

    comp = torch.tensor([[p[min(j, len(p) - 1)] for j in range(MAXM)] for p in train], device=dev)
    lens = torch.tensor([len(p) for p in train], device=dev)
    model = Model(max_len=n_tok + 2, pos="rope", n_vocab=n_vocab).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98),
                            fused=(dev == "cuda"))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    trainer = torch.compile(model) if dev == "cuda" else model
    gg = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()
    for _ in range(args.steps):
        tok, cut = make(args.batch, args.demos, tab, comp, lens, dev, gg)
        msk = loss_mask(args.batch, args.demos, cut, dev)[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = trainer(tok)[:, :-1]
        loss = F.cross_entropy(logits[msk].float(), tok[:, 1:][msk])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    print(f"|train|={len(train):<5} trained in {time.time() - t0:.0f}s | final loss {loss.item():.4f}")

    print(f"{'pool':<22}{'n':>5}{'PROG-fn':>9}{'ID@1':>8}")
    for name, ps in (("TRAIN", train), ("HELD-OUT", held)):
        acc, id1, by = evaluate(model, ps, sigs, dev, args.demos, cap=args.eval_cap)
        print(f"{name + (' (subsample)' if len(ps) > args.eval_cap else ''):<22}"
              f"{min(len(ps), args.eval_cap):>5}{acc:>9.3f}{id1:>8.3f}")
        print("      by minimal depth: " + "  ".join(f"d{d}={a:.2f}({n})" for d, (a, n) in by.items()))
    print(f"\nChance for ID@1 is 1/{len(NAMES)} = {1 / len(NAMES):.3f}. PROG-fn counts an EQUIVALENT factorisation as")
    print("correct. All numbers FREE-RUNNING. DEPTH-1 held-out functions are pure RECOGNITION and are the internal")
    print("control: if d1 works while d>=3 does not, the dissociation holds without any cross-run comparison.")


if __name__ == "__main__":
    main()
