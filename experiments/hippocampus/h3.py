"""Experiment H3: consolidation ("sleep") from the store into the weights (EXPERIMENT_H3.md).

A reader trained with the learned expansion (H1c, arm E1) writes world 0 into its store. Sleep then fine-tunes it with
an EMPTY memory on consolidation examples, interleaved 1:1 with rehearsal of the reading skill on the training worlds:
  F  forward only: each stored episode (a, r, b) replayed as the question [a, r] -> b
  I  forward + the model's own inferences: each episode is also put alone in memory, the reader answers the reverse
     question [b, r°]; a confident answer (p >= CONF) becomes the target of the empty-memory question [b, r°]
After sleep: what the weights know without memory (both directions), whether reading survived (world 0 and a held-out
world), and how much of the store can be evicted. An episode counts as consolidated when the weights alone give
p >= CONS_P to it in BOTH directions. The combined system reads only episodes that match both the question's entity and
relation (overlap >= THETA_COMB bits); with none, the memory is empty and the weights answer.

Usage:  python h3.py --arm I --seed 4      -> runs/h3/I_s4.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from h1 import (CTX, DEV, MEM, MODEL, N_ENT, N_REL, PAD, R, SEP, TEST_WORLD, VOCAB, Net, eval_mem, eval_sets,
                gated_write, make_facts, opt_for, score, step, store_factory, training_set)

HERE = Path(__file__).parent
OUT, READERS = HERE / "runs" / "h3", HERE / "runs" / "h1c"
SLEEP_STEPS, SLEEP_LR, HALF = 2000, 5e-4, 128
CONF, CONS_P, THETA_COMB = 0.5, 0.1, 60.0
HELDOUT_WORLD = 50


def empty_ctx(P):
    n = len(P)
    return torch.cat([torch.full((n, 1), MEM), torch.full((n, 4 * R), PAD), P.cpu()], 1).to(DEV)


def single_ctx(eps, P):
    n = len(P)
    ep = torch.cat([eps.cpu(), torch.full((n, 1), SEP)], 1)
    return torch.cat([torch.full((n, 1), MEM), torch.full((n, 4 * (R - 1)), PAD), ep, P.cpu()], 1).to(DEV)


@torch.no_grad()
def predict(net, ctx):
    net.eval()
    with torch.autocast(DEV, dtype=torch.bfloat16):
        p = net(ctx).float()[:, :N_ENT].softmax(-1)
    net.train()
    return p


def eval_empty(net, sets):
    out = {}
    for name, s in sets.items():
        p = predict(net, empty_ctx(torch.tensor([q for q, _, _ in s])))
        hits, mass = score(p.log(), s)
        out[name] = dict(hits=hits, mass=mass)
    return out


def written_store(net, Store, world, seed, gen):
    F = make_facts(seed=world)
    order = torch.randperm(len(F.facts), generator=torch.Generator().manual_seed(seed)).tolist()
    st = Store(VOCAB, "sparse")
    gated_write(net, st, [F.facts[j] for j in order], gen, familiarity=True)
    return F, st


def combined(net, store, sets, gen):
    """Read only fully matching episodes (overlap >= THETA_COMB); none -> empty memory, so the weights answer."""
    import h1
    saved = h1.THETA
    h1.THETA = THETA_COMB
    try:
        return eval_mem(net, store, sets, gen)
    finally:
        h1.THETA = saved


def run_sleep(arm, seed):
    t0 = time.time()
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    net = Net(VOCAB, ctx=CTX, **MODEL).to(DEV)
    net.load_state_dict(torch.load(READERS / f"reader_E1_s{seed}.pt", map_location=DEV))
    Store = store_factory(expand=True, read_order="ranked")
    F0, store = written_store(net, Store, TEST_WORLD, seed, gen)
    sets = eval_sets(F0)
    F50, store50 = written_store(net, Store, HELDOUT_WORLD, seed, gen)
    sets50 = eval_sets(F50)
    before = dict(empty=eval_empty(net, sets), memory=eval_mem(net, store, sets, gen),
                  combined=combined(net, store, sets, gen), heldout=eval_mem(net, store50, sets50, gen))

    rows = store.rows.cpu()                                       # (n, 3): a, relation token, b
    cX = [empty_ctx(rows[:, :2])]
    cY = [rows[:, 2].to(DEV)]
    inference = None
    if arm == "I":
        Prev = torch.stack([rows[:, 2], rows[:, 1] + N_REL], 1)   # [b, r°]
        p = predict(net, single_ctx(rows, Prev))
        conf, ans = p.max(-1)
        keep = conf >= CONF
        rev = {}
        for a, r, b in F0.facts:
            rev.setdefault((r, b), set()).add(a)
        truth = [int(ans[i]) in rev[(int(rows[i, 1]) - N_ENT, int(rows[i, 2]))] for i in range(len(rows))]
        inference = dict(generated=len(rows), kept=int(keep.sum()),
                         correct_all=round(sum(truth) / len(truth), 4),
                         correct_kept=round(sum(t for t, k in zip(truth, keep.tolist()) if k) / max(1, int(keep.sum())), 4))
        cX.append(empty_ctx(Prev[keep.cpu()]))
        cY.append(ans[keep].to(DEV))
    cX, cY = torch.cat(cX), torch.cat(cY)
    X, Y = training_set(Store, "sparse", gen)                     # rehearsal of the reading skill

    opt = opt_for(net, SLEEP_LR, fused=True)
    train_net = torch.compile(net)
    net.train()
    for t in range(1, SLEEP_STEPS + 1):
        i = torch.randint(0, len(cX), (HALF,), device=DEV, generator=gen)
        j = torch.randint(0, len(X), (HALF,), device=DEV, generator=gen)
        step(train_net, opt, torch.cat([cX[i], X[j]]), torch.cat([cY[i], Y[j]]), SLEEP_LR, t, total=SLEEP_STEPS)

    after = dict(empty=eval_empty(net, sets), memory=eval_mem(net, store, sets, gen),
                 combined=combined(net, store, sets, gen), heldout=eval_mem(net, store50, sets50, gen))
    pf = predict(net, empty_ctx(rows[:, :2]))[torch.arange(len(rows)), rows[:, 2].to(DEV)]
    pr = predict(net, empty_ctx(torch.stack([rows[:, 2], rows[:, 1] + N_REL], 1)))[torch.arange(len(rows)),
                                                                                   rows[:, 0].to(DEV)]
    cons = ((pf >= CONS_P) & (pr >= CONS_P)).nonzero().flatten().tolist()
    n_before = len(store)
    store.remove(cons)
    evicted = dict(rows_before=n_before, consolidated=len(cons), rows_after=len(store),
                   fwd_consolidated=int((pf >= CONS_P).sum()), rev_consolidated=int((pr >= CONS_P).sum()),
                   combined_after_eviction=combined(net, store, sets, gen))
    return dict(arm=arm, seed=seed, sleep_steps=SLEEP_STEPS, sleep_lr=SLEEP_LR, half=HALF, conf=CONF, cons_p=CONS_P,
                theta_comb=THETA_COMB, consolidation_examples=len(cX), inference=inference, before=before,
                after=after, eviction=evicted, secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["F", "I"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    out = OUT / f"{a.arm}_s{a.seed}.json"
    if out.exists():
        print(f"{out} exists: not re-run")
        raise SystemExit
    r = run_sleep(a.arm, a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(out, "w"), indent=1)
    print(json.dumps(r, indent=1))
