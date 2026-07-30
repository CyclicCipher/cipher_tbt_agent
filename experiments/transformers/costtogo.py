"""COST-TO-GO: learn the DISTANCE, search the PROGRAM.

THE SPLIT THIS FILE TESTS, and it is the one every measurement in this line has been pointing at:

    forward model (apply a named primitive)      SMOOTH      learned at 0.997   (`identify.py`)
    scalar property of the decomposition         SMOOTH      learned at 0.84    (`halt.py`, `identify.py`, `bigroup.py`)
    the program itself                        DISCONTINUOUS  0.000 everywhere   (`decompose.py`, `bigroup.py`)

So stop trying to amortise the answer and amortise the DISTANCE instead. This is DeepCubeA's move (McAleer, Agostinelli,
Shmakov & Baldi, Nat Mach Intell 2019), on what is literally this problem — Rubik's cube is factoring a group element into
generators, and they solve it by learning a cost-to-go and using it as a heuristic in weighted A*, never learning the move
sequence. It is also a direct correction to `decompose.py`: that file used self-generated data to amortise the ANSWER and
was flat at 0.000 over a 36x coverage range; the same data engine aimed at the DISTANCE is a smooth target.

WHAT IS BEING RISKED, stated up front so the result can be negative. Distance is 1-Lipschitz in GRAPH distance — adjacent
nodes differ by exactly one — but the network does not see the graph, it sees behaviour. Whether distance is smooth in the
BEHAVIOURAL representation is exactly the open question, and it is the same question that killed retrieval in `recall.py`
(where behavioural similarity tracked program distance at only rho = -0.151). If distance is as behaviourally
discontinuous as the program is, the heuristic fails too and the whole DeepCubeA route closes for this problem class.

DESIGN. The group is `S6 x Aff(Z5)` = 7200 elements (13 primitives, BFS over the Cayley graph gives exact minimal
distances), capped at depth <= 5 for tractability as in `bigroup.py`. The heuristic sees `n_demos` interleaved
`[current, goal]` sequence pairs and predicts the distance as a class in 0..MAXM. Training samples a transformation from a
TRAINING subset of the group and a random current state; **held-out transformations never appear in heuristic training**,
and search will query the heuristic on them constantly, so this is a genuine generalisation test.

THREE BASELINES, because a learned heuristic has to beat both doing nothing and doing something free:
  * `zero`     — h = 0, i.e. uniform-cost search. The uninformed floor.
  * `hamming`  — h = mean Hamming distance between current and goal, hand-coded and free. **If the learned heuristic does
                 not beat this, learning contributed nothing** — and Hamming is a plausible proxy, so this is the baseline
                 that matters.
  * `learned`  — the network.
Reference point from `localise.py`: uninformed meet-in-the-middle solved held-out functions at 1.000 with 29.3 expansions
(on the smaller depth<=3 universe), and forward enumeration needed 69.9.

Usage:  python experiments/transformers/costtogo.py
"""
from __future__ import annotations

import argparse
import collections
import heapq
import math
import time

import torch
import torch.nn.functional as F

from bigroup import NAMES, PRIMS, apply_prog, cayley, compile_progs, stratify
from h1_lid import L, V, Model

MAXM = 5


def dist_token(d):
    return V + d


def make(B, n_demos, tab, lens, dev, g, fixed=None):
    """Interleaved `[current, goal]` pairs, then the DISTANCE class at the final position. `current` is a random state and
    `goal` is that state under the sampled transformation, so the pair determines the residual transformation exactly and
    its minimal length is the label."""
    idx_t, vmp_t = tab
    n = idx_t.shape[0]
    sel = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, n, (B,), generator=g, device=dev))
    cur = torch.randint(0, V, (B, n_demos, L), generator=g, device=dev)
    gi = idx_t[sel][:, None, :].expand(B, n_demos, L)
    goal = vmp_t[sel][:, None, :].expand(B, n_demos, V).gather(2, cur.gather(2, gi))
    pairs = torch.stack([cur, goal], dim=2).reshape(B, n_demos * 2 * L)
    return torch.cat([pairs, dist_token(lens[sel])[:, None]], dim=1)


def encode(cur, goal, dev):
    """The same layout, for querying the heuristic at search time. `cur`/`goal` are (B, n_demos, L)."""
    B = cur.shape[0]
    return torch.stack([cur, goal], dim=2).reshape(B, -1)


@torch.no_grad()
def h_learned(model, cur, goal, dev):
    """Expected distance under the model's class posterior — a graded value rather than the argmax, which matters for a
    search priority where ties are common."""
    tok = encode(cur, goal, dev)
    p = model(tok)[:, -1, V:V + MAXM + 1].softmax(-1)
    return (p * torch.arange(MAXM + 1, device=dev, dtype=p.dtype)).sum(-1)


def h_hamming(cur, goal, _dev):
    return (cur != goal).float().mean(dim=(1, 2)) * MAXM      # scaled to the same range, so lam means the same thing


def h_zero(cur, goal, _dev):
    return torch.zeros(cur.shape[0], device=cur.device)


def best_first(prog_target, prims_tabs, hfn, dev, n_demos, lam, cap, sigs, seed=0):
    """Best-first search over programs. State = the current demo sequences; goal = their images under the target
    transformation. Children of a state are its images under each primitive, and the heuristic is evaluated on ALL
    children in one batch. Returns (found, expansions)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randint(0, V, (n_demos, L), generator=g, device=dev)
    goal = apply_prog(x, prog_target)
    want = sigs[prog_target]
    gl = goal[None].expand(len(NAMES), n_demos, L)

    key = lambda s: tuple(s.flatten().tolist())
    h0 = float(hfn(x[None], goal[None], dev)[0])
    heap = [(h0, 0, key(x), x, ())]
    seen = {key(x)}
    expansions = 0
    while heap and expansions < cap:
        _pri, gc, k, st, prog = heapq.heappop(heap)
        if torch.equal(st, goal):
            if sigs.get(prog) == want:
                return True, expansions
        if gc >= MAXM:
            continue
        expansions += 1
        kids = torch.stack([apply_prog(st, (p,)) for p in range(len(NAMES))])     # (n_prims, n_demos, L)
        hs = hfn(kids, gl, dev)
        for p in range(len(NAMES)):
            kk = key(kids[p])
            if kk in seen:
                continue
            seen.add(kk)
            heapq.heappush(heap, (gc + 1 + lam * float(hs[p]), gc + 1, kk, kids[p], prog + (p,)))
    return False, expansions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1200, help="transformations the HEURISTIC is trained on")
    ap.add_argument("--held", type=int, default=400)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--demos", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=2.0, help="heuristic weight in weighted A*")
    ap.add_argument("--cap", type=int, default=400, help="expansion budget per task")
    ap.add_argument("--eval_cap", type=int, default=90, help="held-out functions searched")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    probe, table = cayley(dev, args.seed)
    progs = [w for w in table.values() if 0 < len(w) <= MAXM]
    sigs = {w: tuple(apply_prog(probe, w).flatten().tolist()) for w in progs}
    sigs[()] = tuple(probe.flatten().tolist())
    hist = collections.Counter(len(w) for w in progs)
    print(f"device {dev} | group order 7200 (diameter 9) | universe depth<={MAXM}: {len(progs)} transformations")
    print(f"minimal-depth histogram: {dict(sorted(hist.items()))}")

    g = torch.Generator().manual_seed(args.seed)
    by = collections.defaultdict(list)
    for w in progs:
        by[len(w)].append(w)
    held, pool = [], []
    frac = args.held / len(progs)
    for d, ws in sorted(by.items()):
        order = torch.randperm(len(ws), generator=g).tolist()
        cut = min(len(ws) - 1, max(3, int(round(frac * len(ws)))))
        held += [ws[i] for i in order[:cut]]
        pool += [ws[i] for i in order[cut:]]
    order = torch.randperm(len(pool), generator=g).tolist()
    train = [pool[i] for i in order][:args.n_train]
    print(f"heuristic trained on {len(train)} transformations | {len(held)} held out (never in heuristic training)\n")

    n_vocab = V + MAXM + 1
    n_tok = args.demos * 2 * L + 1
    tab = compile_progs(train, dev)
    lens = torch.tensor([len(w) for w in train], device=dev)
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
        tok = make(args.batch, args.demos, tab, lens, dev, gg)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = trainer(tok)[:, -2]                        # predicts the final (distance) token
        loss = F.cross_entropy(logits.float(), tok[:, -1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    print(f"heuristic trained in {time.time() - t0:.0f}s | final loss {loss.item():.4f}")

    # HOW GOOD IS THE HEURISTIC ITSELF, before any search? Exact-match and mean absolute error, on train and held-out.
    @torch.no_grad()
    def h_quality(ws):
        tb, ln = compile_progs(ws, dev), torch.tensor([len(w) for w in ws], device=dev)
        ex, mae = [], []
        for t in range(len(ws)):
            tok = make(64, args.demos, tb, ln, dev, None, fixed=t)
            p = model(tok)[:, -2, V:V + MAXM + 1]
            pred = p.argmax(-1).float()
            ex.append((pred == len(ws[t])).float().mean().item())
            mae.append((pred - len(ws[t])).abs().mean().item())
        return sum(ex) / len(ex), sum(mae) / len(mae)
    # BOTH samples are depth-STRATIFIED. `held` is built depth-ordered, so a prefix slice would evaluate only shallow
    # functions while a shuffled `train` slice is depth-representative — and the depth mismatch alone would inflate the
    # apparent generalisation gap. Same prefix-of-an-ordered-list trap as in `bigroup.py`.
    tr_ex, tr_mae = h_quality(stratify(train, 120))
    he_ex, he_mae = h_quality(stratify(held, 120))
    print(f"\nHEURISTIC QUALITY (depth-stratified)   train exact {tr_ex:.3f} MAE {tr_mae:.2f}   "
          f"HELD-OUT exact {he_ex:.3f} MAE {he_mae:.2f}   (chance exact ~{1 / (MAXM + 1):.3f})")

    tasks = stratify(held, args.eval_cap)
    print(f"\nSEARCH on {len(tasks)} HELD-OUT transformations | weighted A*, lam={args.lam}, cap={args.cap}")
    print(f"{'heuristic':<12}{'found':>8}{'expansions':>12}{'exp (solved)':>14}")
    for label, hfn in (("zero", h_zero), ("hamming", h_hamming), ("learned", lambda c, gl, d: h_learned(model, c, gl, d))):
        rs = [best_first(t, None, hfn, dev, args.demos, args.lam, args.cap, sigs, args.seed) for t in tasks]
        solved = [r[1] for r in rs if r[0]]
        print(f"{label:<12}{sum(r[0] for r in rs) / len(rs):>8.3f}{sum(r[1] for r in rs) / len(rs):>12.1f}"
              f"{(sum(solved) / len(solved) if solved else float('nan')):>14.1f}")
    print("\n'expansions' averages over ALL tasks (unsolved ones hit the cap); 'exp (solved)' averages over solved ones")
    print("only, which is the honest cost-per-success. CLAIM: learned << hamming << zero on expansions at equal found.")
    print("If learned ~ hamming, the network contributed nothing over a free hand-coded proxy.")
    print("Reference: uninformed meet-in-the-middle (localise.py) solved 1.000 at 29.3 expansions on the depth<=3 universe.")


if __name__ == "__main__":
    main()
