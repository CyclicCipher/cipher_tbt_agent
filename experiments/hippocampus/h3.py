"""Experiment H3: consolidation ("sleep") from the store into the weights (EXPERIMENT_H3.md).

A reader trained with the learned expansion (H1c, arm E1) writes world 0 into its store. Sleep then fine-tunes it with
an EMPTY memory on consolidation examples, interleaved 1:1 with rehearsal of the reading skill on the training worlds:
  F  forward only: each stored episode (a, r, b) replayed as the question [a, r] -> b
  I  forward + the model's own inferences: for each episode the reader answers the reverse question [b, r°] through
     its normal read path (retrieval from the store); a confident answer (p >= CONF) becomes the target of the
     empty-memory question [b, r°]
After sleep: what the weights know without memory (both directions), whether reading survived (world 0 and a held-out
world), and how much of the store can be evicted. An episode counts as consolidated when the weights alone give
p >= CONS_P to it in BOTH directions. The combined system reads only episodes that match both the question's entity and
relation: if the best stored overlap is >= THETA_FULL bits (a full match), the question is read normally (12
episodes, as the reader was trained); otherwise the memory is EMPTY and the weights answer. (A half-empty context is
never shown: the readers were never trained on one and fail on it.)

Usage:  python h3.py --arm I --seed 4      -> runs/h3/I_s4.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from h1 import (CTX, DEV, MEM, MODEL, N_ENT, N_REL, PAD, R, TEST_WORLD, VOCAB, Net, contexts, eval_mem, eval_sets,
                gated_write, make_facts, opt_for, score, step, store_factory, training_set)

HERE = Path(__file__).parent
OUT, READERS = HERE / "runs" / "h3", HERE / "runs" / "h1c"
SLEEP_STEPS, SLEEP_LR, HALF = 2000, 5e-4, 128
CONF, CONS_P, THETA_FULL = 0.5, 0.1, 60.0
HELDOUT_WORLD = 50


def empty_ctx(P):
    n = len(P)
    return torch.cat([torch.full((n, 1), MEM), torch.full((n, 4 * R), PAD), P.cpu()], 1).to(DEV)


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
    """A full match in the store -> the normal read; none -> the empty memory, so the weights answer."""
    out = {}
    for name, s in sets.items():
        P = torch.tensor([q for q, _, _ in s])
        full = store.best_overlap(P) >= THETA_FULL
        ctx = torch.where(full[:, None], contexts(store, P, gen)[0], empty_ctx(P))
        hits, mass = score(predict(net, ctx).log(), s)
        out[name] = dict(hits=hits, mass=mass, read_fraction=round(full.float().mean().item(), 4))
    return out


def consolidation_examples(net, store, gen, inferences=True):
    """Empty-memory questions for every stored episode: forward [a, r] -> b, and (inferences) the reverse [b, r°] ->
    the reader's own answer through its normal read path, kept if p >= CONF. Returns (cX, cY, (conf, ans, keep))."""
    rows = store.rows.cpu()
    cX, cY, out = [empty_ctx(rows[:, :2])], [rows[:, 2].to(DEV)], None
    if inferences:
        Prev = torch.stack([rows[:, 2], rows[:, 1] + N_REL], 1)
        conf, ans = predict(net, contexts(store, Prev, gen)[0]).max(-1)
        keep = conf >= CONF
        cX.append(empty_ctx(Prev[keep.cpu()]))
        cY.append(ans[keep].to(DEV))
        out = (conf, ans, keep)
    return torch.cat(cX), torch.cat(cY), out


def sleep_train(net, cX, cY, X, Y, gen, steps, cons, rehearse, pseudo=None):
    """Sleep: `steps` updates, each `cons` consolidation examples + `rehearse` examples of the reading task, and
    (H3d) `pseudo` = (pX, pY, k): k examples of what the weights already knew before this sleep."""
    opt = opt_for(net, SLEEP_LR, fused=True)
    train_net = torch.compile(net)
    net.train()
    for t in range(1, steps + 1):
        i = torch.randint(0, len(cX), (cons,), device=DEV, generator=gen)
        j = torch.randint(0, len(X), (rehearse,), device=DEV, generator=gen)
        bx, by = [cX[i], X[j]], [cY[i], Y[j]]
        if pseudo is not None:
            pX, pY, k = pseudo
            m = torch.randint(0, len(pX), (k,), device=DEV, generator=gen)
            bx.append(pX[m])
            by.append(pY[m])
        step(train_net, opt, torch.cat(bx), torch.cat(by), SLEEP_LR, t, total=steps)


def what_weights_know(net, n_ent=N_ENT, n_rel=N_REL, conf=CONF):
    """H3d pseudo-rehearsal: the weights' confident (p >= conf) empty-memory answers over the whole question space,
    every (entity, relation) forward and every (entity, converse) reverse question. Returns (pX, pY) on DEV."""
    P = torch.tensor([(e, N_ENT + r) for e in range(n_ent) for r in range(2 * n_rel)])
    X_, Y_ = [], []
    for i in range(0, len(P), 2000):
        part = P[i:i + 2000]
        c, a = predict(net, empty_ctx(part)).max(-1)
        keep = (c >= conf).cpu()
        X_.append(empty_ctx(part[keep]))
        Y_.append(a[keep.to(DEV)])
    return torch.cat(X_), torch.cat(Y_)


def self_check(net, rows):
    """For each stored episode (a, r, b): is the weights-only top answer the stored one, forward and reverse? Also the
    probabilities the weights give to the stored answers."""
    qf = predict(net, empty_ctx(rows[:, :2]))
    qr = predict(net, empty_ctx(torch.stack([rows[:, 2], rows[:, 1] + N_REL], 1)))
    ar = torch.arange(len(rows))
    pf, pr = qf[ar, rows[:, 2].to(DEV)], qr[ar, rows[:, 0].to(DEV)]
    return qf.argmax(-1).cpu() == rows[:, 2], qr.argmax(-1).cpu() == rows[:, 0], pf, pr


def run_sleep(arm, seed, steps=SLEEP_STEPS, cons=HALF, rehearse=HALF, evict_rule="p"):
    """H3 defaults (2,000 steps, 128 + 128, evict at p >= CONS_P). H3b: longer sleep, other mixes, and the
    self-checked rule (evict only if the weights' top answer is the stored answer in both directions)."""
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
    cX, cY, gen_out = consolidation_examples(net, store, gen, inferences=(arm == "I"))
    inference = None
    if arm == "I":
        conf, ans, keep = gen_out
        rev = {}
        for a, r, b in F0.facts:
            rev.setdefault((r, b), set()).add(a)
        truth = [int(ans[i]) in rev[(int(rows[i, 1]) - N_ENT, int(rows[i, 2]))] for i in range(len(rows))]
        inference = dict(generated=len(rows), kept=int(keep.sum()),
                         correct_all=round(sum(truth) / len(truth), 4),
                         correct_kept=round(sum(t for t, k in zip(truth, keep.tolist()) if k) / max(1, int(keep.sum())), 4))
    X, Y = training_set(Store, "sparse", gen)                     # rehearsal of the reading skill
    sleep_train(net, cX, cY, X, Y, gen, steps, cons, rehearse)

    after = dict(empty=eval_empty(net, sets), memory=eval_mem(net, store, sets, gen),
                 combined=combined(net, store, sets, gen), heldout=eval_mem(net, store50, sets50, gen))
    top_f, top_r, pf, pr = self_check(net, rows)
    p_rule = (pf >= CONS_P) & (pr >= CONS_P)
    self_rule = (top_f & top_r).to(DEV)
    chosen = self_rule if evict_rule == "self" else p_rule
    cons_ = chosen.nonzero().flatten().tolist()
    n_before = len(store)
    store.remove(cons_)
    evicted = dict(rule=evict_rule, rows_before=n_before, consolidated=len(cons_), rows_after=len(store),
                   fwd_consolidated=int((pf >= CONS_P).sum()), rev_consolidated=int((pr >= CONS_P).sum()),
                   p_rule_both=int(p_rule.sum()), self_rule_both=int(self_rule.sum()),
                   top_fwd=int(top_f.sum()), top_rev=int(top_r.sum()),
                   combined_after_eviction=combined(net, store, sets, gen))
    return dict(arm=arm, seed=seed, sleep_steps=steps, sleep_lr=SLEEP_LR, cons_batch=cons, rehearse_batch=rehearse,
                conf=CONF, cons_p=CONS_P,
                theta_full=THETA_FULL, consolidation_examples=len(cX), inference=inference, before=before,
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
