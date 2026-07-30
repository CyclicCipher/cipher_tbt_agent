"""HYPOTHESISE-AND-TEST: use the executor as its own verifier.

WHY. `identify.py` split the composition failure cleanly: told which primitives to apply, the model chains three it has
never seen composed at 0.997 free-running; asked to name them, it never once emits the right program (0.000). Forward
execution 0.997, inverse identification 0.000. Analysis-by-synthesis exploits exactly that asymmetry — do not invert,
HYPOTHESISE a decomposition, render it forward, compare against what was observed, keep what survives.

WHAT MAKES THIS DIFFERENT FROM `identify.py --ids given`, and it is the whole point. That arm HANDED OVER the true
program, so it was a diagnostic and an upper bound. Here the program is DERIVED, using only what the environment
supplies: the demonstrations. A candidate is scored by whether running it reproduces a demonstration the model was not
shown. Nothing about the true decomposition is used at any point.

⚠ **ENUMERATION DOES NOT SCALE, AND A POSITIVE RESULT HERE DOES NOT SOLVE THE PROBLEM** (the user's note, 2026-07-28,
recorded because it is the correct reading of whatever number comes out). The candidate set is every program of length
<= 3 over 5 primitives = 155, which is exhaustively checkable; the space is |prims|^depth and exhaustive search dies
immediately past a toy. So what a success establishes is (a) that the forward/inverse asymmetry is real and exploitable,
and (b) a CEILING for what any smarter proposal mechanism could achieve. It does not provide the proposal mechanism, and
the proposal distribution is where the actual science is — see `decompose.py` for whether identification can be learned
instead, and the brain candidates (difference-scoring / prediction-error localisation / context-cued memory sampling) for
what would replace enumeration.

VERIFICATION NEEDS A HELD-OUT DEMONSTRATION, which forces one design change. To score a candidate we need an input whose
true output we know and which is NOT in the model's context — otherwise a candidate could be "verified" by the model
copying the answer out of its own prompt, every candidate would score alike, and the search would silently degenerate.
So the demonstrations are split: `n_ctx` of them go in the context, one is held back as the PROBE. That means the model
must accept a variable number of demonstrations, so it is TRAINED with the count sampled per batch. (The alternative —
verifying against a demonstration that is in context — has the copying confound, and while it would be self-diagnosing,
a confound that has to be argued away is worse than one designed out.)

SCORING. A wrong program reproduces a 6-digit probe by chance with probability 5^-6 ~ 6e-5, so a single probe is already
a decisive filter and no threshold tuning is involved. Ties are broken by SHORTEST program first — a mild Occam prior,
reported separately so its contribution is visible rather than assumed.

Usage:  python experiments/transformers/search.py
"""
from __future__ import annotations

import argparse
import itertools
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
STEP = L + 1                                  # one identity slot then one block of L digits


def make(B, n_ctx, tasks, dev, g, comp, lens, tabs, fixed=None):
    """`n_ctx` demonstrations of `[in, out]`, then the query input, then `ID_j b_j ...` and HALT. Identity slots are
    INPUTS here (never in the loss) — this model's job is execution, and identification is what the search supplies."""
    ID0, HALT = V, V + len(PRIMS.split(","))
    sel = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, len(tasks), (B,), generator=g, device=dev))
    x = torch.randint(0, V, (B, n_ctx + 1, L), generator=g, device=dev)
    def block(tab):
        i_t, v_t = tab
        gi = i_t[sel][:, None, :].expand(B, n_ctx + 1, L)
        return v_t[sel][:, None, :].expand(B, n_ctx + 1, V).gather(2, x.gather(2, gi))
    blocks = [block(t) for t in tabs]
    out = blocks[-1]
    demos = torch.stack([x[:, :-1], out[:, :-1]], dim=2).reshape(B, n_ctx * 2 * L)
    parts = []
    for j in range(MAXM):
        parts.append(ID0 + comp[sel, j][:, None])
        parts.append(blocks[j][:, -1])
    chain = torch.cat(parts + [torch.full((B, 1), HALT, device=dev)], dim=1)
    cut = (lens[sel] * STEP)[:, None]
    pos = torch.arange(MAXM * STEP + 1, device=dev)[None, :]
    stream = torch.where(pos < cut, chain, torch.full_like(chain, HALT))
    return torch.cat([demos, x[:, -1], stream], dim=1), out[:, -1], cut.squeeze(1)


def loss_mask(B, n_ctx, cut, dev):
    base = n_ctx * 2 * L + L
    m = torch.zeros(B, base + MAXM * STEP + 1, dtype=torch.bool, device=dev)
    for k in range(n_ctx):
        m[:, k * 2 * L + L:(k + 1) * 2 * L] = True
    pos = torch.arange(MAXM * STEP + 1, device=dev)[None, :]
    m[:, base:] = (pos <= cut[:, None]) & ~((pos % STEP == 0) & (pos < MAXM * STEP))
    return m


@torch.no_grad()
def run_program(model, prompt, progs, lens):
    """Execute one program per row, free-running on the DIGITS and injecting the identity slots. `prompt` is
    `[demos, query_input]`; returns the final block before each row's HALT."""
    dev = prompt.device
    B = prompt.shape[0]
    ID0, HALT = V, V + len(PRIMS.split(","))
    seq = prompt
    gen = torch.zeros(B, 0, dtype=torch.long, device=dev)
    at = torch.full((B,), -1, dtype=torch.long, device=dev)
    for i in range(MAXM * STEP + 1):
        nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
        if i % STEP == 0:
            j = i // STEP
            inj = torch.where(torch.full((B,), j, device=dev) < lens, ID0 + progs[:, min(j, MAXM - 1)],
                              torch.full((B,), HALT, device=dev))
            nxt = inj[:, None]
        seq, gen = torch.cat([seq, nxt], dim=1), torch.cat([gen, nxt], dim=1)
        at = torch.where((nxt.squeeze(1) == HALT) & (at < 0), torch.full_like(at, i), at)
    idx = (at[:, None] - L + torch.arange(L, device=dev)[None, :]).clamp(min=0)
    return gen.gather(1, idx), at >= L


@torch.no_grad()
def search(model, task, dev, n_ctx, cands, cand_t, cand_len, n=16, chunk=32):
    """For ONE task: draw `n` independent demonstration sets, hold one demonstration back as the probe, score every
    candidate by whether executing it reproduces that probe, then answer the real query with the winner."""
    g = torch.Generator(device=dev).manual_seed(1234)
    n_prims = len(PRIMS.split(","))
    # Fresh data for this task, generated directly rather than through `make`, because the search needs the probe pair
    # and the query pair kept separate and explicitly labelled.
    xs = torch.randint(0, V, (n, n_ctx + 2, L), generator=g, device=dev)
    ys = apply_task(xs.reshape(-1, L), task).reshape(n, n_ctx + 2, L)
    demos = torch.stack([xs[:, :n_ctx], ys[:, :n_ctx]], dim=2).reshape(n, n_ctx * 2 * L)
    probe_in, probe_out = xs[:, n_ctx], ys[:, n_ctx]
    query_in, query_out = xs[:, n_ctx + 1], ys[:, n_ctx + 1]

    scores = torch.zeros(n, len(cands), device=dev)
    for lo in range(0, len(cands), chunk):
        hi = min(lo + chunk, len(cands))
        c = hi - lo
        prompt = torch.cat([demos, probe_in], dim=1)[:, None, :].expand(n, c, -1).reshape(n * c, -1)
        progs = cand_t[lo:hi][None].expand(n, c, MAXM).reshape(n * c, MAXM)
        lens = cand_len[lo:hi][None].expand(n, c).reshape(n * c)
        got, ok = run_program(model, prompt, progs, lens)
        hit = (got == probe_out[:, None, :].expand(n, c, L).reshape(n * c, L)).all(-1) & ok
        scores[:, lo:hi] = hit.float().reshape(n, c)

    # Occam tie-break: among candidates that reproduce the probe, prefer the SHORTEST program.
    best = (scores * 1000 - cand_len[None, :].float()).argmax(1)
    n_match = scores.sum(1)
    prompt = torch.cat([demos, query_in], dim=1)
    got, ok = run_program(model, prompt, cand_t[best], cand_len[best])
    answer = ((got == query_out).all(-1) & ok).float().mean().item()
    # Did the search find a program that is the SAME FUNCTION as the task? Checked by signature, so an equivalent
    # writing counts -- otherwise a correct decomposition under another name would be scored as a failure.
    probe = torch.randint(0, V, (24, L), generator=g, device=dev)
    want = tuple(apply_task(probe, task).flatten().tolist())
    same = sum(1 for b in best.tolist()
               if tuple(apply_task(probe, cands[b]).flatten().tolist()) == want) / n
    return answer, same, n_match.mean().item(), (n_match == 0).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ctx_lo", type=int, default=5)
    ap.add_argument("--ctx_hi", type=int, default=7, help="demonstration count sampled per batch, inclusive")
    ap.add_argument("--n_ctx", type=int, default=6, help="context demonstrations at search time; one more is the probe")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=16, help="independent demonstration sets per task")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    names = PRIMS.split(",")
    gidx = [NAMES.index(nm) for nm in names]
    compact = {gi: j for j, gi in enumerate(gidx)}
    sup, held = build_splits(args.seed, gidx, max_len=MAXM)
    sup_all = sup[1] + sup[2] + sup[3]

    cands = [tuple(c) for m in range(1, MAXM + 1) for c in itertools.product(gidx, repeat=m)]
    cand_t = torch.tensor([[compact[c[min(j, len(c) - 1)]] for j in range(MAXM)] for c in cands], device=dev)
    cand_len = torch.tensor([len(c) for c in cands], device=dev)
    n_vocab = V + len(names) + 1
    max_ctx = args.ctx_hi
    print(f"device {dev} | executor: identity slots are INPUTS, demo count sampled from "
          f"[{args.ctx_lo},{args.ctx_hi}] | supervised {len(sup_all)} tasks | held-out {len(held[2])} pairs + "
          f"{len(held[3])} triples | {len(cands)} candidate programs | steps={args.steps}")

    tabs_of = lambda ts: [compile_tasks([t[:min(j + 1, len(t))] for t in ts], dev) for j in range(MAXM)]
    lens_of = lambda ts: torch.tensor([len(t) for t in ts], device=dev)
    comp_of = lambda ts: torch.tensor([[compact[t[min(j, len(t) - 1)]] for j in range(MAXM)] for t in ts], device=dev)
    sup_tab, sup_len, sup_comp = tabs_of(sup_all), lens_of(sup_all), comp_of(sup_all)

    n_tok_max = max_ctx * 2 * L + L + MAXM * STEP + 1
    model = Model(max_len=n_tok_max + 2, pos="rope", n_vocab=n_vocab).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98),
                            fused=(dev == "cuda"))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    # Compiled for training only. The demonstration count varies per batch, so this recompiles once per distinct
    # sequence length (three of them) — cheap, and the alternative was 40 ms/step. Evaluation runs the uncompiled
    # `model` because the search feeds many different batch shapes.
    trainer = torch.compile(model) if dev == "cuda" else model
    g = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()
    for s in range(args.steps):
        n_ctx = int(torch.randint(args.ctx_lo, args.ctx_hi + 1, (1,), generator=g, device=dev))
        tok, _a, cut = make(args.batch, n_ctx, sup_all, dev, g, sup_comp, sup_len, sup_tab)
        msk = loss_mask(args.batch, n_ctx, cut, dev)[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = trainer(tok)[:, :-1]
        loss = F.cross_entropy(logits[msk].float(), tok[:, 1:][msk])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    print(f"executor trained in {time.time() - t0:.0f}s | final loss {loss.item():.4f}")

    t1 = time.time()
    print(f"\n{'pool':<20}{'n':>4}{'ANSWER':>9}{'solved':>9}{'found fn':>10}{'#matching':>11}{'no match':>10}")
    for name, tasks in (("supervised prims", sup[1]), ("supervised pairs", sup[2]),
                        ("supervised triples", sup[3]), ("HELD-OUT pairs", held[2]), ("HELD-OUT triples", held[3])):
        if not tasks:
            continue
        rs = [search(model, t, dev, args.n_ctx, cands, cand_t, cand_len, n=args.n) for t in tasks]
        a = sum(r[0] for r in rs) / len(rs)
        print(f"{name:<20}{len(tasks):>4}{a:>9.3f}{sum(r[0] >= 0.8 for r in rs):>5}/{len(tasks):<3}"
              f"{sum(r[1] for r in rs) / len(rs):>10.3f}{sum(r[2] for r in rs) / len(rs):>11.1f}"
              f"{sum(r[3] for r in rs) / len(rs):>10.3f}")
    print(f"\nsearch took {time.time() - t1:.0f}s | ANSWER = the real query answered with the SELECTED program;")
    print("'found fn' = the selected program is the same FUNCTION as the task (by signature, so an equivalent writing")
    print("counts); '#matching' = candidates reproducing the probe; 'no match' = the search found nothing at all.")
    print("BASELINES for HELD-OUT triples: halt.py 0.048, identify.py predicted 0.020, one-shot control 0.113.")
    print("CEILING reference: identify.py --ids given, which was HANDED the program, reached 0.997.")
    print("\nAND THE STANDING CAVEAT: 155 candidates is exhaustively checkable and |prims|^depth is not. This measures")
    print("whether the asymmetry is exploitable and sets a ceiling; it does not supply a proposal mechanism.")


if __name__ == "__main__":
    main()
