"""ARITY: does the scratchpad gain survive 3-compositions, and does it need the slot count to MATCH?

WHAT `scratchpad.py` LEFT OPEN. Decoding `[intermediate, output]` instead of `[output]` took held-out compositions from
0.046 to 0.363 (0 -> 7/34 solved) — the first inductive gain in this line. But the format has exactly ONE intermediate
slot, which hands the model the ARITY: tasks break into two steps. It is never told WHERE to break, and the
demonstrations never show an intermediate, so the decomposition is still inferred — but the DEPTH is not.

THE TEST. Move to 3-compositions, which by construction need three applications (any 3-writing that happens to equal a
1- or 2-composition is deduplicated away, so a surviving 3-task is genuinely irreducible in this primitive set). Then
vary ONLY the number of decode slots:

    slots = 0   one shot                       the control, exactly `scratchpad.py --pad 0`
    slots = 1   one intermediate               the scratchpad format, now too SHALLOW for a 3-composition
    slots = 2   two intermediates              depth matches composition length

THE PREDICTION, stated before running. If the mechanism is "composing m operations needs m applications", then
`slots=2` >> `slots=1` ~ `slots=0` on held-out 3-compositions. If `slots=1` does just as well, the depth argument is
wrong and the gain came from something else — the format merely giving the model somewhere to put a partial result, say
— and that would retract the reasoning `scratchpad.py` was built on.

HOW THE CHAIN IS PACKED OVER THE SLOTS. With `slots` intermediates there are `slots+1` decode blocks, and a task of
length m has m applications to distribute over them:

  front (default) — the LAST blocks each do one primitive, and the excess is absorbed at the FRONT. Block j computes the
    first `max(0, m - (slots - j))` primitives. A task shorter than the slots idles first (a primitive under slots=2
    decodes `[x, x, phi_p(x)]`); a task LONGER than the slots must do a COMPOUND first step (a 3-composition under
    slots=1 decodes `[phi_b(phi_a(x)), phi_c(...)]`). That compound step is precisely the arity mismatch: the last step
    stays "apply one primitive" for every task, so the read-out is aligned exactly as in `scratchpad.py`, and the ONLY
    thing that breaks is whether every step is a single local operation.

  end (`--pack end`) — block j computes the first `min(j+1, m)` primitives, so the chain runs from the START and REPEATS
    once finished. THIS IS THE NO-ARITY FORM, and it is the one that eventually has to work: there is no program that
    can always hand the correct arity, so the depth has to be the model's to choose. With `slots` set ABOVE the deepest
    task, the slot count carries no information about any particular task — every task gets the same generous budget and
    must decide for itself how many steps are real and when to stop changing. It is strictly harder than `front`,
    because nothing forces the intermediate blocks to be used at all: only the last block is scored, so "answer in one
    shot at block 0, then copy" is available and is exactly the control's solution.

WHAT IS STILL HANDED OVER even under `--pack end`: a generous UPPER BOUND on depth. Removing that last bound needs the
model to emit its own halt, which is the next step and not this one.

WHAT IS NEVER HANDED OVER, in either mode: the demonstrations remain `[in, out]` exactly as the environment provides
them and never carry an intermediate. Intermediate supervision exists only for the LABELLED training tasks, whose
decomposition we chose when we built them.

THE PRIMITIVE SET IS CHOSEN BY COUNTING, not by taste — the same discipline `h1_lid.py` needed. All 12 primitives give
174 distinct functions up to length 3 (12 + 53 + 109), which puts 92 tasks in the supervised pool, and at that size the
model does not fit its own training distribution: measured, the one-shot control reached 0.171 on supervised PRIMITIVES
and 0.048 on supervised triples, which makes every held-out number uninterpretable (the sanity-gate failure that
`h1_lid.py` and `apply_vs_mix.py` both hit). Five primitives give 56 distinct functions (5 + 15 + 36) and a supervised
pool of 30 — the size `scratchpad.py` actually fit. The gate below is enforced rather than eyeballed.

THE OUTCOME (2026-07-28), because it did not answer the question it was built for and instead invalidated the result it
was built to extend. The no-arity arm returned 1.000 on every held-out pool, which is not a result — under `end` packing
the block before the answer IS the answer, and evaluation was a single teacher-forced forward pass, so copying scored
perfectly. That exposed the general form of the bug: teacher forcing puts the TRUE intermediate in the model's context,
so it never had to produce one. Re-run free-running (`rollout`), NO arm beats the one-shot control and not one held-out
task is solved anywhere; `scratchpad.py`'s 0.363 becomes 0.024 against a 0.047 control. The arity question is therefore
moot as posed — you cannot ask whether the slot count must match when no arm chains at all. What survives is the `forced`
column read as what it actually measures: given a correct partial result, the model completes a NOVEL composition at
0.48–0.59 against 0.113. It can execute one local step out of distribution and cannot sequence two of its own. Full
account in `NOTES.md`.

Usage:  python experiments/transformers/arity.py --slots 2
        python experiments/transformers/arity.py --slots 4 --pack end
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
from scratchpad import compile_tasks, rollout


def build_splits(seed: int, prims, max_len: int = 3):
    """Every DISTINCT function writable as a composition of up to `max_len` primitives, deduplicated ACROSS lengths and
    with the shorter writings claiming their signatures first. The cross-length dedup is what makes the experiment mean
    anything: `rot1.rot1.rot1 = rot3` and `rot1.rot2 = rot3` would otherwise let a "3-composition" be solvable in one
    application, and the whole question is whether depth is needed.

    Each length is then split in half into SUPERVISED and HELD-OUT, except the primitives, which are all supervised —
    they are the parts the chain is grounded on."""
    g = torch.Generator().manual_seed(seed)
    probe = torch.randint(0, V, (24, L), generator=g)
    ident = tuple(probe.flatten().tolist())
    seen, by_len = {}, {}
    for m in range(1, max_len + 1):
        keep = []
        for task in itertools.product(prims, repeat=m):
            sig = tuple(apply_task(probe, task).flatten().tolist())
            if sig == ident or sig in seen:
                continue
            seen[sig] = task
            keep.append(task)
        order = torch.randperm(len(keep), generator=g).tolist()
        by_len[m] = [keep[i] for i in order]
    sup, held = {}, {}
    for m, tasks in by_len.items():
        cut = len(tasks) if m == 1 else len(tasks) // 2
        sup[m], held[m] = tasks[:cut], tasks[cut:]
    return sup, held


def prefix(task, j, slots, pack):
    """How much of `task` the j-th decode block computes. One line, and it is the whole difference between the arms."""
    k = max(0, len(task) - (slots - j)) if pack == "front" else min(j + 1, len(task))
    return task[:k]


def tables(tasks, dev, slots, pack):
    """One gather pair per decode block. `compile_tasks` handles any length, including the empty prefix (= identity)."""
    return [compile_tasks([prefix(t, j, slots, pack) for t in tasks], dev) for j in range(slots + 1)]


def make(B, K, tasks, dev, g, tabs, fixed=None):
    """(K-1) demonstrations of `[in, out]`, then the query input followed by `slots+1` decoded blocks. The
    demonstrations NEVER carry an intermediate — that would reveal the decomposition of an unseen task."""
    sel = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, len(tasks), (B,), generator=g, device=dev))
    x = torch.randint(0, V, (B, K, L), generator=g, device=dev)
    def step(tab):
        idx_tab, vmp_tab = tab
        gi = idx_tab[sel][:, None, :].expand(B, K, L)
        return vmp_tab[sel][:, None, :].expand(B, K, V).gather(2, x.gather(2, gi))
    blocks = [step(t) for t in tabs]
    out = blocks[-1]                                       # the last block is the ANSWER under both packings
    demos = torch.stack([x[:, :-1], out[:, :-1]], dim=2).reshape(B, (K - 1) * 2 * L)
    return torch.cat([demos, x[:, -1]] + [b[:, -1] for b in blocks], dim=1), out[:, -1]


def masks(K, slots, dev):
    """Positions carrying a PREDICTION, and the ANSWER block accuracy is scored on."""
    n = (K - 1) * 2 * L + (slots + 2) * L
    m = torch.zeros(n, dtype=torch.bool, device=dev)
    for k in range(K - 1):
        m[k * 2 * L + L:(k + 1) * 2 * L] = True
    m[(K - 1) * 2 * L + L:] = True                         # every decoded block of the query
    final = torch.zeros(n, dtype=torch.bool, device=dev)
    final[n - L:] = True
    return m, final


@torch.no_grad()
def verify(tasks, dev, tabs, slots, pack, n=32):
    """Check the VECTORISED gather tables against the reference `apply_task`, block by block. This exists because the
    same vectorisation silently corrupted four tasks once — `negate` was classified as a position permutation on the
    strength of fixing the all-zeros sequence — and nothing in the results would have looked wrong."""
    g = torch.Generator(device=dev).manual_seed(7)
    x = torch.randint(0, V, (n, L), generator=g, device=dev)
    for t, task in enumerate(tasks):
        for j, (idx_tab, vmp_tab) in enumerate(tabs):
            got = vmp_tab[t][None].expand(n, V).gather(1, x.gather(1, idx_tab[t][None].expand(n, L)))
            want = apply_task(x, prefix(task, j, slots, pack))
            if not torch.equal(got, want):
                raise AssertionError(f"generator mismatch: task {task}, block {j}, pack={pack}")


@torch.no_grad()
def accuracy(model, tasks, dev, K, slots, pack, n=256):
    """TWO numbers, because they are two different claims and conflating them is how this experiment nearly lied.

    FREE-RUNNING is the real one, and the only one the environment could ever supply: demonstrations and a query input
    go in, the model generates its own intermediates, the answer comes off the end of its own chain.

    TEACHER-FORCED scores the same final block with the TRUE intermediates sitting in the context. That is not a test of
    chaining — it is a test of the LAST step given a correct partial result handed over for free, so it measures
    something strictly easier than the task. Under `--pack end` it is not even that: the block before the answer is by
    construction IDENTICAL to the answer, so copying scores 1.000 and did. `scratchpad.py` reported only this number,
    which is why its 0.363 has to be read as "can finish, given the intermediate" rather than "can chain".

    Both are printed so the gap is visible instead of assumed away."""
    if not tasks:
        return (0.0, 0), 0.0
    tabs = tables(tasks, dev, slots, pack)
    _, final = masks(K, slots, dev)
    free, forced = [], []
    for t in range(len(tasks)):
        tok, ans = make(n, K, tasks, dev, None, tabs, fixed=t)
        free.append((rollout(model, tok, K, slots) == ans).all(-1).float().mean().item())
        got = model(tok)[:, :-1].argmax(-1)[:, final[1:]].reshape(n, L)
        forced.append((got == ans).all(-1).float().mean().item())
    return (sum(free) / len(free), sum(h >= 0.8 for h in free)), sum(forced) / len(forced)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slots", type=int, default=2, help="number of INTERMEDIATE decode blocks (0 = one-shot control)")
    p.add_argument("--pack", default="front", choices=["front", "end"],
                   help="front = excess absorbed at the front (arity handed); end = chain from the start and repeat (NO arity)")
    p.add_argument("--prims", default="rot1,reverse,swap_pairs,add1,negate",
                   help="primitive subset, CHOSEN BY COUNTING so the supervised pool is one the model fits")
    p.add_argument("--gate", type=float, default=0.8,
                   help="minimum supervised accuracy below which the held-out numbers are declared uninterpretable")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    sup, held = build_splits(args.seed, [NAMES.index(n) for n in args.prims.split(",")])
    sup_all = sup[1] + sup[2] + sup[3]
    n_tok = (args.k - 1) * 2 * L + (args.slots + 2) * L
    print(f"device {dev} | slots={args.slots} pack={args.pack} | supervised "
          f"{len(sup[1])} prims + {len(sup[2])} pairs + {len(sup[3])} triples = {len(sup_all)} | "
          f"held-out {len(held[2])} pairs + {len(held[3])} triples | seq {n_tok} | steps={args.steps}")

    sup_tab = tables(sup_all, dev, args.slots, args.pack)
    verify(sup_all, dev, sup_tab, args.slots, args.pack)
    for m in (2, 3):
        verify(held[m], dev, tables(held[m], dev, args.slots, args.pack), args.slots, args.pack)
    print("generator verified against the reference on every task and every decode block")

    model = Model(max_len=n_tok + 2, pos="rope").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98),
                            fused=(dev == "cuda"))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    trainer = torch.compile(model) if dev == "cuda" else model
    g = torch.Generator(device=dev).manual_seed(args.seed)
    m, _ = masks(args.k, args.slots, dev)
    t0 = time.time()
    for _ in range(args.steps):
        tok, _ans = make(args.batch, args.k, sup_all, dev, g, sup_tab)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = trainer(tok)[:, :-1]
        loss = F.cross_entropy(logits[:, m[1:]].reshape(-1, V).float(), tok[:, 1:][:, m[1:]].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    rows = [("supervised prims", sup[1]), ("supervised pairs", sup[2]), ("supervised triples", sup[3]),
            ("HELD-OUT pairs", held[2]), ("HELD-OUT triples", held[3])]
    got = {}
    print(f"\n{'pool':<20}{'n':>5}{'FREE-RUN':>10}{'solved':>10}{'forced':>9}")
    for name, tasks in rows:
        (acc, solved), forced = accuracy(model, tasks, dev, args.k, args.slots, args.pack)
        got[name] = acc
        print(f"{name:<20}{len(tasks):>5}{acc:>10.3f}{solved:>6}/{len(tasks):<4}{forced:>9.3f}")
    print(f"\nslots={args.slots} pack={args.pack} trained in {time.time() - t0:.0f}s")
    print("FREE-RUN is the claim (model generates its own intermediates); `forced` supplies the TRUE ones and is a")
    print("strictly easier diagnostic -- under pack=end it is pure copying, since the block before the answer equals it.")

    # THE SANITY GATE, enforced rather than eyeballed. Held-out accuracy cannot be compared across arms that differ in
    # how well they fit the TRAINING set, and an arm that fits nothing censors every held-out task at the same value.
    worst = min(got[n] for n in ("supervised prims", "supervised pairs", "supervised triples"))
    if worst < args.gate:
        print(f"!! SANITY GATE FAILED: worst supervised pool is {worst:.3f} < {args.gate}. The model has not learned the")
        print("   training distribution, so the held-out numbers below it are UNINTERPRETABLE -- not a null result.")
    else:
        print(f"sanity gate passed (worst supervised pool {worst:.3f})")
    print("HELD-OUT TRIPLES is the claim: 3-compositions never supervised, irreducible to fewer applications by dedup.")


if __name__ == "__main__":
    main()
