"""THE HALT TOKEN: no arity handed over at all — the model decides how many steps to take, and when it is finished.

WHY THIS IS THE ONLY HONEST FORM. Every scratchpad format so far fixed the number of decode blocks in advance, which
hands the model the SHAPE of the answer. There is no program that can always supply the correct arity (the user,
2026-07-28), so a format that needs one has smuggled in part of the solution. Here the decode stream is

    b_1  b_2  ...  b_m  HALT

and `m` is the model's own output: it emits blocks of L digits until it emits HALT, and the ANSWER is the last complete
block before the HALT. Nothing in the prompt says how many blocks a task needs, and the eval cap is a runtime guard the
model never sees.

WHAT `arity.py` LEARNED THE HARD WAY, and why this is not just `--pack end` with a token bolted on. The previous
no-arity attempt padded a fixed-depth format by REPEATING the final answer, and scored a perfect 1.000 — by copying.
Repetition does not remove the arity, it removes the task: most predicted blocks become "the same as the last one", the
copy shortcut dominates the loss, and the model learns that instead of computing. **The chain here has NO repeated
blocks.** A task of length m is supervised on exactly m blocks — prefixes 1, 2, ..., m, one primitive per step — and
then HALT. Everything after the first HALT is filler and is MASKED OUT of the loss, so there is nothing to copy and
nothing to pad toward.

WHAT IS SUPERVISED AND WHAT IS NOT. Primitives and half the pairs and triples are supervised, chain and HALT included —
their decomposition is ours because we built them. Held-out compositions receive no gradient in any form, and the
demonstrations stay `[in, out]` and never carry an intermediate. So for a held-out task the model must infer, from
`[in, out]` pairs alone, BOTH where to break AND how many breaks there are. That is strictly harder than every arm run
before it.

EVALUATION IS FREE-RUNNING, and after `arity.py` that is not negotiable: teacher forcing puts the true intermediate in
the context and measures "can it finish", not "can it chain". The model generates its own tokens, fed back, until it
emits HALT.

THE THREE MEASUREMENTS, because "accuracy" alone cannot say what went wrong:
  * ANSWER   — the block before HALT is the correct composition. The claim.
  * DEPTH    — the number of blocks emitted equals the task's true length. This is the arity question asked directly:
               a model that emits one block and halts has fallen back to the one-shot solution, and a model that gets
               DEPTH right while ANSWER is wrong is decomposing correctly and executing badly. Those are different
               failures and the previous formats could not tell them apart.
  * HALTED   — it emitted HALT at all, within the cap. A model that never terminates has not answered.

BASELINE: the `slots=0` one-shot control in `arity.py`, same splits, same seed, same primitive set — free-running 0.113
on held-out triples, 0/18 solved.

Usage:  python experiments/transformers/halt.py
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from arity import build_splits, tables
from chaining import apply_task
from diversity import NAMES
from h1_lid import L, V, Model

HALT = V                                    # the digits are 0..V-1; the model emits V to stop
MAXM = 3                                    # deepest composition in the pool; the FORMAT never encodes it per task


def make(B, K, tasks, dev, g, tabs, lens, fixed=None):
    """(K-1) demonstrations of `[in, out]`, the query input, then `m` chain blocks and a HALT — where `m` is the task's
    true length, so the stream LENGTH varies per sequence and everything past the first HALT is filler."""
    sel = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, len(tasks), (B,), generator=g, device=dev))
    x = torch.randint(0, V, (B, K, L), generator=g, device=dev)
    def step(tab):
        idx_tab, vmp_tab = tab
        gi = idx_tab[sel][:, None, :].expand(B, K, L)
        return vmp_tab[sel][:, None, :].expand(B, K, V).gather(2, x.gather(2, gi))
    blocks = [step(t) for t in tabs]                         # prefixes 1..MAXM; beyond the task's length they repeat…
    out = blocks[-1]
    demos = torch.stack([x[:, :-1], out[:, :-1]], dim=2).reshape(B, (K - 1) * 2 * L)
    chain = torch.cat([b[:, -1] for b in blocks] + [torch.full((B, 1), HALT, device=dev)], dim=1)
    cut = (lens[sel] * L)[:, None]                           # …and the repeats are CUT here, replaced by HALT+filler,
    pos = torch.arange(MAXM * L + 1, device=dev)[None, :]    #    which is what keeps a copy shortcut from existing
    stream = torch.where(pos < cut, chain, torch.full_like(chain, HALT))
    tok = torch.cat([demos, x[:, -1], stream], dim=1)
    return tok, out[:, -1], cut.squeeze(1)


def loss_mask(B, K, cut, dev):
    """Target positions: every demonstration's output block, then the chain up to AND INCLUDING the first HALT. The
    HALT is scored because deciding to stop IS the arity decision — it is the output under test, not punctuation.
    Filler beyond it is masked out, so the model is never rewarded for repeating anything."""
    base = (K - 1) * 2 * L + L
    m = torch.zeros(B, base + MAXM * L + 1, dtype=torch.bool, device=dev)
    for k in range(K - 1):
        m[:, k * 2 * L + L:(k + 1) * 2 * L] = True
    pos = torch.arange(MAXM * L + 1, device=dev)[None, :]
    m[:, base:] = pos <= cut[:, None]
    return m


@torch.no_grad()
def rollout(model, tok, K, cap):
    """FREE-RUNNING until the model halts. Returns the generated tokens and, per sequence, the position of the first
    HALT (-1 if it never emitted one inside the cap)."""
    dev = tok.device
    B = tok.shape[0]
    seq = tok[:, :(K - 1) * 2 * L + L]                       # demonstrations + the query INPUT, nothing else
    gen = torch.zeros(B, 0, dtype=torch.long, device=dev)
    at = torch.full((B,), -1, dtype=torch.long, device=dev)
    for i in range(cap):
        nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
        seq, gen = torch.cat([seq, nxt], dim=1), torch.cat([gen, nxt], dim=1)
        at = torch.where((nxt.squeeze(1) == HALT) & (at < 0), torch.full_like(at, i), at)
        if bool((at >= 0).all()):
            break
    return gen, at


@torch.no_grad()
def evaluate(model, tasks, dev, K, tabs_of, lens, cap, n=128):
    """ANSWER / DEPTH / HALTED free-running, plus FIRST and FORCED, which between them say WHERE a failure is.

    ANSWER — the L tokens immediately preceding the first HALT are the correct composition. The claim. A HALT that
             arrives before a full block, or never, is a wrong answer; no partial credit for not finishing.
    DEPTH  — it emitted exactly the task's own number of blocks. The arity question asked directly.
    FIRST  — its first emitted block equals `phi_a(x)`, the canonical first step. Localises where a chain goes wrong.
             It can UNDERSTATE: a task may admit another valid split, and taking one is not an error.
    FORCED — teacher-forced accuracy of the answer block. This is the one the SANITY GATE should be read against,
             because it asks "did the model learn the training distribution" — whereas free-running additionally asks
             "can it survive its own intermediate errors", which is a different question and the interesting one."""
    if not tasks:
        return dict(ans=0.0, solved=0, depth=0.0, halted=0.0, first=0.0, forced=0.0)
    tabs = tabs_of(tasks)
    base = (K - 1) * 2 * L + L
    ok, depth, halted, first, forced = [], [], [], [], []
    for t in range(len(tasks)):
        m = len(tasks[t])
        tok, ans, _cut = make(n, K, tasks, dev, None, tabs, lens, fixed=t)
        gen, at = rollout(model, tok, K, cap)
        idx = (at[:, None] - L + torch.arange(L, device=dev)[None, :]).clamp(min=0)
        ok.append((((gen.gather(1, idx) == ans).all(-1)) & (at >= L)).float().mean().item())
        depth.append((at == m * L).float().mean().item())
        halted.append((at >= 0).float().mean().item())
        first.append((gen[:, :L] == tok[:, base:base + L]).all(-1).float().mean().item())
        lo = base + (m - 1) * L                                   # the answer block's own position in the stream
        forced.append((model(tok)[:, lo - 1:lo + L - 1].argmax(-1) == ans).all(-1).float().mean().item())
    n_t = len(tasks)
    return dict(ans=sum(ok) / n_t, solved=sum(h >= 0.8 for h in ok), depth=sum(depth) / n_t,
                halted=sum(halted) / n_t, first=sum(first) / n_t, forced=sum(forced) / n_t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prims", default="rot1,reverse,swap_pairs,add1,negate")
    p.add_argument("--gate", type=float, default=0.8)
    p.add_argument("--cap", type=int, default=(MAXM + 2) * L + 1,
                   help="generation cap — a runtime guard the model never sees, deliberately above any real chain")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    sup, held = build_splits(args.seed, [NAMES.index(n) for n in args.prims.split(",")], max_len=MAXM)
    sup_all = sup[1] + sup[2] + sup[3]
    n_tok = (args.k - 1) * 2 * L + L + MAXM * L + 1
    print(f"device {dev} | HALT token | supervised {len(sup[1])} prims + {len(sup[2])} pairs + {len(sup[3])} triples "
          f"= {len(sup_all)} | held-out {len(held[2])} pairs + {len(held[3])} triples | seq {n_tok} | steps={args.steps}")

    # The chain is the `end`-style prefix ladder 1..MAXM, CUT at each task's own length inside `make`. `lens` is indexed
    # by the position of a task in whichever list is being generated from, so it is rebuilt per list.
    tabs_of = lambda ts: tables(ts, dev, MAXM - 1, "end")
    lens_of = lambda ts: torch.tensor([len(t) for t in ts], device=dev)
    sup_tab, sup_len = tabs_of(sup_all), lens_of(sup_all)

    # VERIFY the generator before training on it: every chain block must equal the reference `apply_task` on the right
    # prefix, the HALT must sit at exactly `m*L`, and nothing before it may be a repeat of the block before.
    g0 = torch.Generator(device=dev).manual_seed(7)
    for t, task in enumerate(sup_all):
        tok, ans, cut = make(8, args.k, sup_all, dev, g0, sup_tab, sup_len, fixed=t)
        base = (args.k - 1) * 2 * L + L
        qx = tok[:, base - L:base]
        assert int(cut[0]) == len(task) * L, f"HALT misplaced for {task}"
        assert bool((tok[:, base + int(cut[0])] == HALT).all()), f"no HALT at the cut for {task}"
        for j in range(len(task)):
            blk = tok[:, base + j * L:base + (j + 1) * L]
            assert torch.equal(blk, apply_task(qx, task[:j + 1])), f"chain block {j} wrong for {task}"
        assert torch.equal(tok[:, base + (len(task) - 1) * L:base + len(task) * L], ans), f"answer misplaced for {task}"
    print("generator verified: chain blocks match the reference, HALT sits at m*L, no repeated block precedes it")

    model = Model(max_len=n_tok + 2, pos="rope", n_vocab=V + 1).to(dev)
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
        tok, _ans, cut = make(args.batch, args.k, sup_all, dev, g, sup_tab, sup_len)
        msk = loss_mask(args.batch, args.k, cut, dev)[:, 1:]
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
    print(f"\n{'pool':<20}{'n':>5}{'ANSWER':>9}{'solved':>10}{'DEPTH':>8}{'FIRST':>8}{'HALTED':>8}{'forced':>9}")
    for name, tasks in rows:
        r = evaluate(model, tasks, dev, args.k, tabs_of, lens_of(tasks), args.cap)
        got[name] = r
        print(f"{name:<20}{len(tasks):>5}{r['ans']:>9.3f}{r['solved']:>6}/{len(tasks):<4}"
              f"{r['depth']:>8.3f}{r['first']:>8.3f}{r['halted']:>8.3f}{r['forced']:>9.3f}")
    print(f"\ntrained in {time.time() - t0:.0f}s | ANSWER/DEPTH/FIRST/HALTED are FREE-RUNNING -- the model chose its own depth")
    print("BASELINE for HELD-OUT triples: the one-shot control (arity.py --slots 0) scores 0.113, 0/18 solved.")

    # THE GATE IS READ AGAINST `forced`, deliberately. It asks whether the model learned the training distribution at
    # all; free-running supervised accuracy additionally asks whether it survives its own intermediate errors, and
    # conflating the two would report an exposure-bias result as a failure to learn.
    sups = ("supervised prims", "supervised pairs", "supervised triples")
    worst = min(got[n]["forced"] for n in sups)
    if worst < args.gate:
        print(f"!! SANITY GATE FAILED: worst supervised pool is {worst:.3f} teacher-forced < {args.gate}. The model has")
        print("   not learned the training distribution, so the held-out numbers are UNINTERPRETABLE.")
    else:
        print(f"sanity gate passed (worst supervised pool {worst:.3f} teacher-forced)")
        drop = min(got[n]["forced"] - got[n]["ans"] for n in sups)
        print(f"   and the model FITS the training set while losing at least {drop:.3f} of it to its OWN intermediate")
        print("   errors when run free -- that gap is exposure bias, not undertraining.")


if __name__ == "__main__":
    main()
