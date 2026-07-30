"""IDENTIFY vs APPLY: is the missing operation naming the part, or applying it?

THE BRIDGE BETWEEN THE TWO LINES, and the reason it is worth one experiment. `engine.py` showed that routing on a GIVEN
discrete key is exact — swap the operation token and the model computes the other operation at 1.000 — and `alloc.py`
showed such routing is nearly free when the circuit can be shared. Meanwhile the composition line's model builds
WHOLE-task keys perfectly well (supervised ~0.95) and cannot factor an observed whole into PART keys: it knows a held-out
3-composition needs three steps (DEPTH 0.842) and cannot produce the first one (FIRST 0.344). **Counting the parts while
being unable to name them is the signature of estimating a scalar rather than factoring a function.**

So the hypothesis: the missing operation is IDENTIFICATION, not APPLICATION. The scratchpad never failed to apply a
primitive — it failed to work out WHICH primitive step one is.

HOW THIS IS MADE DECIDABLE. Give the decode stream an explicit primitive-identity token before each block, exactly the
status the operation token has in `engine.py`, and then vary whether that token is an INPUT or an OUTPUT:

    stream:  ID_1  b_1  ID_2  b_2  ...  ID_m  b_m  HALT

  --ids none        no identity tokens at all. The baseline, i.e. `halt.py`.
  --ids given       identity tokens are INPUTS — supplied at train AND test, excluded from the loss. Measures
                    EXECUTION GIVEN THE DECOMPOSITION. This is a DIAGNOSTIC AND AN UPPER BOUND, NOT A SOLUTION: for a
                    held-out task it hands over where to break, which is the rigging this line has otherwise avoided.
                    It is legitimate *as a diagnostic* because it isolates one ability, and it is labelled as such.
  --ids predicted   identity tokens are OUTPUTS — supervised on labelled tasks only, and at test the model emits them
                    itself and consumes its own. Measures the whole loop, and yields ID accuracy directly.

WHAT EACH OUTCOME MEANS, fixed in advance:
  * `given` >> `none`  =>  execution-given-decomposition works, so the bottleneck is IDENTIFICATION. The two lines are
                           one finding, and `engine.py`'s "routing on a given key is easy" transfers.
  * `given` ~ `none`   =>  even with the decomposition handed over the model cannot execute it, APPLICATION really is
                           missing, and the engine analogy fails. That would retract the framing.

THE MEASUREMENT MAKING THIS WORTH MORE THAN THE ARMS. With the key explicit, `predicted` reports **ID accuracy per step**
— "which primitive is step one?" as a discrete 1-of-5 choice — separately from whether the block was computed correctly.
FIRST=0.344 in `halt.py` conflated "did not know which primitive" with "knew it and applied it wrong". These separate
them. Chance is 1/5 = 0.200, and that is the number ID accuracy has to beat to mean anything.

HALT is a value of the identity slot rather than a separate mechanism: the model emits either a primitive id or HALT
there, so the no-arity property of `halt.py` is preserved — nothing tells it how many steps a task takes.

Usage:  python experiments/transformers/identify.py --ids none
        python experiments/transformers/identify.py --ids given
        python experiments/transformers/identify.py --ids predicted
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from arity import build_splits
from chaining import apply_task
from diversity import NAMES
from h1_lid import L, V, Model
from scratchpad import compile_tasks

MAXM = 3
PRIMS = "rot1,reverse,swap_pairs,add1,negate"


def token_ids(n_prims):
    """Digits are 0..V-1; identity tokens are V+j for compact primitive index j; HALT is the last."""
    return V, V + n_prims, V + n_prims + 1          # (first ID token, HALT, vocab size)


def make(B, K, tasks, dev, g, tabs, lens, comp, ids, fixed=None):
    """(K-1) demonstrations of `[in, out]`, the query input, then the interleaved chain. Demonstrations NEVER carry an
    intermediate or an identity — only the query's decode stream does."""
    n_prims = len(PRIMS.split(","))
    ID0, HALT, _ = token_ids(n_prims)
    step = L + 1 if ids != "none" else L
    sel = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, len(tasks), (B,), generator=g, device=dev))
    x = torch.randint(0, V, (B, K, L), generator=g, device=dev)
    def block(tab):
        idx_tab, vmp_tab = tab
        gi = idx_tab[sel][:, None, :].expand(B, K, L)
        return vmp_tab[sel][:, None, :].expand(B, K, V).gather(2, x.gather(2, gi))
    blocks = [block(t) for t in tabs]                       # prefixes 1..MAXM (they repeat past a task's own length)
    out = blocks[-1]
    demos = torch.stack([x[:, :-1], out[:, :-1]], dim=2).reshape(B, (K - 1) * 2 * L)
    parts = []
    for j in range(MAXM):
        if ids != "none":
            parts.append(ID0 + comp[sel, j][:, None])       # the j-th primitive's compact id, per sequence
        parts.append(blocks[j][:, -1])
    chain = torch.cat(parts + [torch.full((B, 1), HALT, device=dev)], dim=1)
    cut = (lens[sel] * step)[:, None]                       # where HALT goes: after the task's OWN number of steps
    pos = torch.arange(MAXM * step + 1, device=dev)[None, :]
    stream = torch.where(pos < cut, chain, torch.full_like(chain, HALT))
    return torch.cat([demos, x[:, -1], stream], dim=1), out[:, -1], cut.squeeze(1)


def loss_mask(B, K, cut, dev, ids):
    """Targets: demonstration outputs, the digit blocks, and the first HALT. Identity positions are targets ONLY when
    they are outputs (`predicted`); under `given` they are inputs and must be excluded, or the arm would be secretly
    training the very ability it is meant to hold constant."""
    step = L + 1 if ids != "none" else L
    base = (K - 1) * 2 * L + L
    m = torch.zeros(B, base + MAXM * step + 1, dtype=torch.bool, device=dev)
    for k in range(K - 1):
        m[:, k * 2 * L + L:(k + 1) * 2 * L] = True
    pos = torch.arange(MAXM * step + 1, device=dev)[None, :]
    keep = pos <= cut[:, None]
    if ids == "given":
        keep = keep & ~((pos % step == 0) & (pos < MAXM * step))     # identity slots are inputs, not targets
    m[:, base:] = keep
    return m


@torch.no_grad()
def rollout(model, tok, K, comp_row, m_row, ids, cap):
    """Free-running. Under `given`, the identity slots are INJECTED with the true primitive (and HALT after the task's
    own length) instead of being sampled; every digit is always the model's own."""
    dev = tok.device
    B = tok.shape[0]
    n_prims = len(PRIMS.split(","))
    ID0, HALT, _ = token_ids(n_prims)
    step = L + 1 if ids != "none" else L
    seq = tok[:, :(K - 1) * 2 * L + L]
    gen = torch.zeros(B, 0, dtype=torch.long, device=dev)
    at = torch.full((B,), -1, dtype=torch.long, device=dev)
    for i in range(cap):
        nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
        if ids == "given" and i % step == 0:
            j = i // step
            forced = (ID0 + int(comp_row[j])) if j < m_row else HALT
            nxt = torch.full_like(nxt, forced)
        seq, gen = torch.cat([seq, nxt], dim=1), torch.cat([gen, nxt], dim=1)
        at = torch.where((nxt.squeeze(1) == HALT) & (at < 0), torch.full_like(at, i), at)
        if bool((at >= 0).all()):
            break
    return gen, at


@torch.no_grad()
def evaluate(model, tasks, dev, K, tabs_of, lens_of, comp_of, ids, cap, n=128):
    """ANSWER / DEPTH / FIRST-block free-running, plus ID accuracy at step 1 and over the whole program."""
    if not tasks:
        return dict(ans=0.0, solved=0, depth=0.0, first=0.0, id1=0.0, prog=0.0, forced=0.0)
    n_prims = len(PRIMS.split(","))
    ID0, HALT, _ = token_ids(n_prims)
    step = L + 1 if ids != "none" else L
    base = (K - 1) * 2 * L + L
    tabs, lens, comp = tabs_of(tasks), lens_of(tasks), comp_of(tasks)
    ok, dep, fst, id1, prog, forced = [], [], [], [], [], []
    for t in range(len(tasks)):
        m = len(tasks[t])
        tok, ans, _cut = make(n, K, tasks, dev, None, tabs, lens, comp, ids, fixed=t)
        gen, at = rollout(model, tok, K, comp[t], m, ids, cap)
        idx = (at[:, None] - L + torch.arange(L, device=dev)[None, :]).clamp(min=0)
        ok.append((((gen.gather(1, idx) == ans).all(-1)) & (at >= L)).float().mean().item())
        dep.append((at == m * step).float().mean().item())
        off = 1 if ids != "none" else 0                       # the first digit block starts after the first ID slot
        fst.append((gen[:, off:off + L] == tok[:, base + off:base + off + L]).all(-1).float().mean().item())
        if ids == "predicted":
            id1.append((gen[:, 0] == ID0 + int(comp[t][0])).float().mean().item())
            want = torch.tensor([ID0 + int(comp[t][j]) for j in range(m)], device=dev)
            got = gen[:, [j * step for j in range(m)]] if gen.shape[1] > (m - 1) * step else None
            prog.append(0.0 if got is None else ((got == want).all(-1) & (at == m * step)).float().mean().item())
        lo = base + (m - 1) * step + off
        forced.append((model(tok)[:, lo - 1:lo + L - 1].argmax(-1) == ans).all(-1).float().mean().item())
    N = len(tasks)
    return dict(ans=sum(ok) / N, solved=sum(h >= 0.8 for h in ok), depth=sum(dep) / N, first=sum(fst) / N,
                id1=(sum(id1) / N if id1 else float("nan")), prog=(sum(prog) / N if prog else float("nan")),
                forced=sum(forced) / N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="predicted", choices=["none", "given", "predicted"])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prims", default=PRIMS)
    ap.add_argument("--gate", type=float, default=0.8)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    names = args.prims.split(",")
    gidx = [NAMES.index(nm) for nm in names]
    compact = {g: j for j, g in enumerate(gidx)}              # global primitive index -> compact id token
    sup, held = build_splits(args.seed, gidx, max_len=MAXM)
    sup_all = sup[1] + sup[2] + sup[3]
    step = L + 1 if args.ids != "none" else L
    n_tok = (args.k - 1) * 2 * L + L + MAXM * step + 1
    _, _, n_vocab = token_ids(len(names))
    cap = MAXM * step + 1
    print(f"device {dev} | ids={args.ids} | supervised {len(sup[1])} prims + {len(sup[2])} pairs + {len(sup[3])} "
          f"triples = {len(sup_all)} | held-out {len(held[2])} pairs + {len(held[3])} triples | seq {n_tok} "
          f"| vocab {n_vocab} | steps={args.steps}")

    tabs_of = lambda ts: [compile_tasks([t[:min(j + 1, len(t))] for t in ts], dev) for j in range(MAXM)]
    lens_of = lambda ts: torch.tensor([len(t) for t in ts], device=dev)
    # Compact primitive ids per task, padded by repeating the last so the tensor is rectangular; positions past a task's
    # own length are never read, because HALT is written at `m*step`.
    comp_of = lambda ts: torch.tensor(
        [[compact[t[min(j, len(t) - 1)]] for j in range(MAXM)] for t in ts], device=dev)
    sup_tab, sup_len, sup_comp = tabs_of(sup_all), lens_of(sup_all), comp_of(sup_all)

    # VERIFY the generator before training on it: every digit block must equal the reference on the right prefix, every
    # identity slot must name the right primitive, and HALT must land at m*step.
    g0 = torch.Generator(device=dev).manual_seed(7)
    base = (args.k - 1) * 2 * L + L
    for t, task in enumerate(sup_all):
        tok, ans, cut = make(8, args.k, sup_all, dev, g0, sup_tab, sup_len, sup_comp, args.ids, fixed=t)
        qx = tok[:, base - L:base]
        assert int(cut[0]) == len(task) * step, f"HALT misplaced for {task}"
        off = 1 if args.ids != "none" else 0
        for j in range(len(task)):
            at = base + j * step
            if args.ids != "none":
                assert bool((tok[:, at] == V + compact[task[j]]).all()), f"identity slot {j} wrong for {task}"
            blk = tok[:, at + off:at + off + L]
            assert torch.equal(blk, apply_task(qx, task[:j + 1])), f"block {j} wrong for {task}"
        assert torch.equal(tok[:, base + (len(task) - 1) * step + off:base + len(task) * step], ans) if step == L \
            else True, "answer misplaced"
    print("generator verified: blocks match the reference, identity slots name the right primitive, HALT at m*step")

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
        tok, _a, cut = make(args.batch, args.k, sup_all, dev, g, sup_tab, sup_len, sup_comp, args.ids)
        msk = loss_mask(args.batch, args.k, cut, dev, args.ids)[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = trainer(tok)[:, :-1]
        loss = F.cross_entropy(logits[msk].float(), tok[:, 1:][msk])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    rows = [("supervised prims", sup[1]), ("supervised pairs", sup[2]), ("supervised triples", sup[3]),
            ("HELD-OUT pairs", held[2]), ("HELD-OUT triples", held[3])]
    got = {}
    print(f"\n{'pool':<20}{'n':>4}{'ANSWER':>9}{'solved':>9}{'DEPTH':>7}{'FIRST':>7}{'ID@1':>7}{'PROG':>7}{'forced':>8}")
    for name, tasks in rows:
        r = evaluate(model, tasks, dev, args.k, tabs_of, lens_of, comp_of, args.ids, cap)
        got[name] = r
        print(f"{name:<20}{len(tasks):>4}{r['ans']:>9.3f}{r['solved']:>5}/{len(tasks):<3}{r['depth']:>7.3f}"
              f"{r['first']:>7.3f}{r['id1']:>7.3f}{r['prog']:>7.3f}{r['forced']:>8.3f}")
    print(f"\nids={args.ids} trained in {time.time() - t0:.0f}s | ANSWER/DEPTH/FIRST/ID@1/PROG are FREE-RUNNING.")
    print("ID@1 = the emitted identity for step 1 is the right primitive (chance = 0.200); PROG = the whole emitted")
    print("program is right. Both are NaN unless ids=predicted, where the identities are the model's own output.")
    sups = ("supervised prims", "supervised pairs", "supervised triples")
    worst = min(got[s]["forced"] for s in sups)
    print(f"sanity gate {'PASSED' if worst >= args.gate else 'FAILED'} (worst supervised pool {worst:.3f} teacher-forced)")
    print("BASELINE for HELD-OUT triples ANSWER: halt.py 0.048, one-shot control 0.113.")


if __name__ == "__main__":
    main()
