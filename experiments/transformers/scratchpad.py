"""CHAINING AT INFERENCE TIME: the model emits the intermediate, then the answer.

WHY. `chaining.py` made chaining a TRAINING SIGNAL and it worked transductively and only transductively — unlabelled
compositions went 0.021 → 0.593 while never-presented ones stayed at 0.018. The diagnosis was exact: at inference the
model still answered in ONE SHOT from demonstrations of `(a,b)`, so chained targets taught the ANSWERS and not the
PROCEDURE. It memorised the manufactured labels. To learn the procedure, chaining has to be the computation performed at
inference, which is also what the depth argument demands — composing *m* operations needs *m* applications, and a
one-shot read-out has one.

THE CHANGE. The model decodes `[intermediate, output]` instead of `[output]`. For a composition `(a,b)` on input `x` the
target is `[φ_a(x), φ_b(φ_a(x))]`; for a primitive `p` it is `[x, φ_p(x)]` — identity, then the primitive. So the SECOND
decode step is always the same operation, "apply one primitive to the intermediate", exercised by every task in training.
That is the LID claim made concrete: each local step stays in-distribution while the composite is novel.

WHAT IS NOT HANDED OVER, which is the whole methodological point. The DEMONSTRATIONS remain `[in, out]` exactly as the
environment provides them — they never show an intermediate. Showing intermediates in the demonstrations would reveal the
decomposition of an unseen task at test time, which is precisely the rigging this line has been avoiding. Intermediate
supervision exists only for the LABELLED training tasks, whose decomposition we chose when we built them. On a held-out
composition the model must infer where to break, unsupervised, from `[in, out]` pairs alone.

CONTROL. The same model, same data, same steps, decoding `[output]` directly. The only difference is whether the
computation has an intermediate step, so a gap is that and nothing else.

Usage:  python experiments/transformers/scratchpad.py --pad 1     (scratchpad)
        python experiments/transformers/scratchpad.py --pad 0     (control)
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from chaining import apply_task, build_splits
from diversity import NAMES, PRIMS
from h1_lid import L, V, Model


def compile_tasks(tasks, dev):
    """Each task as ONE gather pair, computed once. Every primitive here is EITHER a position permutation (rotations,
    reverse, swap-pairs) OR a value map (shifts, negate) — verified below rather than assumed from the names — and the two
    kinds commute, so any composition is exactly `out = vmap[ x[..., idx] ]`. That replaces a Python loop over ~29 tasks
    per batch, which had become the bottleneck at 62 ms/step against the model's ~24."""
    ar_L = torch.arange(L)
    ar_V = torch.arange(V)
    # Positional iff it fixes EVERY constant sequence. Testing only the all-zeros sequence classified `negate` as
    # positional, because (V-0) % V = 0 — negate happens to fix zero — and four tasks were then generated wrongly. Caught
    # by verifying the vectorised generator against the reference `apply_task`, which is why that check exists.
    positional = {}
    for n, f in PRIMS.items():
        positional[n] = all(bool((f(torch.full((1, L), c, dtype=torch.long)) == c).all()) for c in range(V))
    idx, vmp = [], []
    for task in tasks:
        i, v = ar_L.clone(), ar_V.clone()
        for k in task:
            name = NAMES[k]
            if positional[name]:
                i = PRIMS[name](i.unsqueeze(0)).squeeze(0)
            else:
                v = PRIMS[name](v.unsqueeze(0)).squeeze(0)
        idx.append(i)
        vmp.append(v)
    return torch.stack(idx).to(dev), torch.stack(vmp).to(dev)


def make(B, K, tasks, dev, g, pad, fixed=None, tab=None):
    """(K−1) demonstrations of `[in, out]`, then the query. With `pad`, the query decodes `[in, mid, out]`; without,
    `[in, out]`. The demonstrations NEVER carry an intermediate — that would give away the decomposition."""
    (fidx, fvmp), (widx, wvmp) = tab
    sel = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, len(tasks), (B,), generator=g, device=dev))
    x = torch.randint(0, V, (B, K, L), generator=g, device=dev)
    def apply(idx_tab, vmp_tab):
        gi = idx_tab[sel][:, None, :].expand(B, K, L)
        return vmp_tab[sel][:, None, :].expand(B, K, V).gather(2, x.gather(2, gi))
    mid = apply(fidx, fvmp)                               # a primitive's first step is the IDENTITY, so that the second
    out = apply(widx, wvmp)                               #   step is "apply one primitive" for EVERY task
    demos = torch.stack([x[:, :-1], out[:, :-1]], dim=2).reshape(B, (K - 1) * 2 * L)
    tail = ([x[:, -1], mid[:, -1], out[:, -1]] if pad else [x[:, -1], out[:, -1]])
    return torch.cat([demos] + tail, dim=1), out[:, -1]


def tables(tasks, dev):
    """Gather tables for the FIRST step (identity for a primitive) and for the WHOLE task."""
    firsts = [((t[0],) if len(t) == 2 else ()) for t in tasks]
    return compile_tasks(firsts, dev), compile_tasks(tasks, dev)


def masks(K, pad, dev):
    """Positions that carry a PREDICTION: every demonstration's output block, and the query's decoded blocks."""
    n = (K - 1) * 2 * L + (3 if pad else 2) * L
    m = torch.zeros(n, dtype=torch.bool, device=dev)
    for k in range(K - 1):
        m[k * 2 * L + L:(k + 1) * 2 * L] = True
    m[(K - 1) * 2 * L + L:] = True                        # mid+out (or just out) of the query
    final = torch.zeros(n, dtype=torch.bool, device=dev)
    final[n - L:] = True                                  # the ANSWER block, which is what accuracy is scored on
    return m, final


@torch.no_grad()
def rollout(model, tok, K, slots):
    """FREE-RUNNING generation: cut the sequence back to the demonstrations plus the query INPUT, then let the model
    produce every decoded block itself, feeding its own tokens back. The answer comes off the end of its OWN chain.

    ADDED AFTER THE FACT, and it overturned this file's headline. The original measurement was a single teacher-forced
    forward pass over a sequence that ALREADY CONTAINED the true intermediate, so the model was handed `phi_a(x)` for a
    held-out task and only had to apply the second step. That is not chaining; it is finishing. The environment can
    never supply an intermediate, so this is the only evaluation the claim was ever entitled to."""
    seq = tok[:, :(K - 1) * 2 * L + L]
    for _ in range((slots + 1) * L):
        seq = torch.cat([seq, model(seq)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return seq[:, -L:]


@torch.no_grad()
def accuracy(model, tasks, dev, K, pad, n=256):
    """Free-running accuracy (the claim) and teacher-forced accuracy (the strictly easier diagnostic), side by side."""
    tab = tables(tasks, dev)
    _, final = masks(K, pad, dev)
    free, forced = [], []
    for t in range(len(tasks)):
        tok, ans = make(n, K, tasks, dev, None, pad, fixed=t, tab=tab)
        free.append((rollout(model, tok, K, pad) == ans).all(-1).float().mean().item())
        got = model(tok)[:, :-1].argmax(-1)[:, final[1:]].reshape(n, L)
        forced.append((got == ans).all(-1).float().mean().item())
    return sum(free) / len(free), sum(h >= 0.8 for h in free), sum(forced) / len(forced)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pad", type=int, default=1, help="1 = decode [mid, out]; 0 = decode [out] (control)")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    prims, lab, unl, ev = build_splits(args.seed)
    sup = prims + lab                                     # intermediates are supervised ONLY here
    held = unl + ev                                       # every composition never supervised, in any form
    n_tok = (args.k - 1) * 2 * L + (3 if args.pad else 2) * L
    print(f"device {dev} | scratchpad={bool(args.pad)} | supervised {len(sup)} tasks "
          f"({len(prims)} primitives + {len(lab)} compositions), held-out {len(held)} | seq {n_tok} | steps={args.steps}")

    model = Model(max_len=n_tok + 2, pos="rope").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98),
                            fused=(dev == "cuda"))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    trainer = torch.compile(model) if dev == "cuda" else model
    g = torch.Generator(device=dev).manual_seed(args.seed)
    m, _ = masks(args.k, args.pad, dev)
    sup_tab = tables(sup, dev)
    t0 = time.time()
    for _ in range(args.steps):
        tok, _ans = make(args.batch, args.k, sup, dev, g, args.pad, tab=sup_tab)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = trainer(tok)[:, :-1]
        loss = F.cross_entropy(logits[:, m[1:]].reshape(-1, V).float(), tok[:, 1:][:, m[1:]].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    pr = accuracy(model, prims, dev, args.k, args.pad)
    la = accuracy(model, lab, dev, args.k, args.pad)
    hl = accuracy(model, held, dev, args.k, args.pad)
    print(f"\n{'arm':<14}{'prims':>9}{'labelled':>11}{'HELD-OUT':>11}{'solved':>10}{'hl forced':>11}{'secs':>8}")
    print(f"{'scratchpad' if args.pad else 'control':<14}{pr[0]:>9.3f}{la[0]:>11.3f}{hl[0]:>11.3f}"
          f"{hl[1]:>6}/{len(held):<4}{hl[2]:>11.3f}{time.time()-t0:>8.0f}")
    print("\nHELD-OUT here is every composition never supervised in any form -- the unlabelled and eval pools of")
    print("`chaining.py` combined, since neither receives a gradient in this experiment.")
    print("All accuracies are FREE-RUNNING (the model generates its own intermediate). `hl forced` is the old")
    print("teacher-forced number, which handed the model the TRUE intermediate and is reported only as the diagnostic.")


if __name__ == "__main__":
    main()
