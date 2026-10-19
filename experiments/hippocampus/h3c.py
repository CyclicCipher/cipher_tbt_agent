"""Experiment H3c: sleep inside the continual stream (EXPERIMENT_H3C.md).

The stream: the union of worlds 100-101 in a fixed random order, 24 phases, 1,000 Zipf(1) queries per phase, a store
capped at CAP rows with value eviction (H2's rule). Questions are asked through the combined switch (a full match in
the store -> the normal read; none -> an empty memory, so the weights answer).
  value     no sleep
  sleep     after phases 6, 12, 18, 24: consolidate the store's episodes (forward + the reader's own reverse
            inferences), SLEEP steps at 3:1 with rehearsal of reading, then self-checked eviction (H3b)
  uncapped  no cap, no sleep (reference)
At the end: per-fact accuracy on EVERY stream fact (forward and reverse, combined system), the weights alone, the
held-out world's reading, and how well the weights still hold the facts consolidated at the first sleep.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from h1 import CTX, DEV, MODEL, N_ENT, VOCAB, Net, contexts, conv, eval_mem, eval_sets, gated_write, make_facts, rel, \
    store_factory, training_set
from h2 import credit, question
from h3 import (READERS, HELDOUT_WORLD, THETA_FULL, consolidation_examples, empty_ctx, predict, self_check,
                sleep_train, written_store)

HERE = Path(__file__).parent
OUT = HERE / "runs" / "h3c"
WORLDS, PHASES, QUERIES, CAP, ZIPF = (100, 101), 24, 1000, 1500, 1.0
SLEEP_AT, SLEEP_STEPS, CONS, REH = (6, 12, 18, 24), 1000, 192, 64


def world(stream_seed):
    facts = set()
    for w in WORLDS:
        facts |= set(make_facts(seed=w).facts)
    facts = sorted(facts)
    rng = random.Random(stream_seed)
    rng.shuffle(facts)
    ranks = list(range(1, len(facts) + 1))
    rng.shuffle(ranks)
    pop = {f: 1.0 / r ** ZIPF for f, r in zip(facts, ranks)}
    fwd, rev = {}, {}
    for a, r, b in facts:
        fwd.setdefault((a, r), set()).add(b)
        rev.setdefault((r, b), set()).add(a)
    return facts, pop, fwd, rev


def ask(net, store, qs, gen):
    """The combined switch. Returns hits, predictions and the read row ids (-1 rows where the memory was empty)."""
    P = torch.tensor([q[0] for q in qs])
    full = store.best_overlap(P) >= THETA_FULL
    ctx, ids = contexts(store, P, gen)
    ctx = torch.where(full[:, None], ctx, empty_ctx(P))
    ids = torch.where(full[:, None], ids, torch.full_like(ids, -1))
    pred = predict(net, ctx).argmax(-1).tolist()
    return [int(pred[i] in q[1]) for i, q in enumerate(qs)], pred, ids


def score_all(net, store, qs, gen, weights_only=False):
    hits = []
    for i in range(0, len(qs), 1000):
        part = qs[i:i + 1000]
        if weights_only:
            P = torch.tensor([q[0] for q in part])
            pred = predict(net, empty_ctx(P)).argmax(-1).tolist()
            hits += [int(pred[k] in q[1]) for k, q in enumerate(part)]
        else:
            hits += ask(net, store, part, gen)[0]
    return hits


def value_evict(store, cap):
    n = len(store) - cap
    if n <= 0:
        return 0
    order = sorted(range(len(store)), key=lambda i: ((store.evidence[i] + 1) / (store.step - store.born[i] + QUERIES),
                                                      store.born[i]))
    store.remove(order[:n])
    return n


def run(arm, seed, stream_seed):
    t0 = time.time()
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(stream_seed)
    rng = random.Random(stream_seed)
    net = Net(VOCAB, ctx=CTX, **MODEL).to(DEV)
    net.load_state_dict(torch.load(READERS / f"reader_E1_s{seed}.pt", map_location=DEV))
    Store = store_factory(expand=True, read_order="ranked")
    F50, store50 = written_store(net, Store, HELDOUT_WORLD, seed, gen)
    sets50 = eval_sets(F50)
    heldout_before = eval_mem(net, store50, sets50, gen)
    X = Y = None
    if arm == "sleep":
        X, Y = training_set(Store, "sparse", gen)
    facts, pop, fwd, rev = world(stream_seed)
    chunks = [facts[i * len(facts) // PHASES:(i + 1) * len(facts) // PHASES] for i in range(PHASES)]
    store = Store(VOCAB, "sparse")
    cap = None if arm == "uncapped" else CAP
    seen, log, per_fact, sleeps, first_sleep_facts = [], [], {}, [], []
    tot = {"fwd": [0, 0], "rev": [0, 0]}
    for ph, chunk in enumerate(chunks, 1):
        net.eval()
        w = gated_write(net, store, chunk, gen, familiarity=True)
        seen += chunk
        ev = value_evict(store, cap) if cap else 0
        picks = rng.choices(seen, weights=[pop[f] for f in seen], k=QUERIES)
        qs = [question(f, "fwd" if k % 2 == 0 else "rev", fwd, rev) for k, f in enumerate(picks)]
        ph_hits = {"fwd": [0, 0], "rev": [0, 0]}
        for i in range(0, QUERIES, 500):
            part = qs[i:i + 500]
            hits, pred, ids = ask(net, store, part, gen)
            for q, h in zip(part, hits):
                ph_hits[q[2]][0] += h
                ph_hits[q[2]][1] += 1
                pf_ = per_fact.setdefault((q[3], q[2]), [0, 0])
                pf_[0] += h
                pf_[1] += 1
            credit(store, part, pred, ids)
            store.step += len(part)
        for k in tot:
            tot[k][0] += ph_hits[k][0]
            tot[k][1] += ph_hits[k][1]
        sl = None
        if arm == "sleep" and ph in SLEEP_AT:
            rows = store.rows.cpu()
            cX, cY, _ = consolidation_examples(net, store, gen, inferences=True)
            sleep_train(net, cX, cY, X, Y, gen, SLEEP_STEPS, CONS, REH)
            top_f, top_r, _, _ = self_check(net, rows)
            out = (top_f & top_r).nonzero().flatten().tolist()
            if not sleeps:
                first_sleep_facts = [(int(rows[i, 0]), int(rows[i, 1]) - N_ENT, int(rows[i, 2])) for i in out]
            store.remove(out)
            sl = dict(phase=ph, rows=len(rows), consolidation_examples=len(cX), evicted=len(out))
            sleeps.append(sl)
        log.append(dict(phase=ph, written=len(w), value_evicted=ev, rows=len(store), sleep=sl,
                        online_fwd=round(ph_hits["fwd"][0] / ph_hits["fwd"][1], 4),
                        online_rev=round(ph_hits["rev"][0] / ph_hits["rev"][1], 4)))
    net.eval()
    every = [question(f, k, fwd, rev) for f in facts for k in ("fwd", "rev")]
    comb = score_all(net, store, every, gen)
    wonly = score_all(net, store, every, gen, weights_only=True)

    def split(h):
        f = [x for x, q in zip(h, every) if q[2] == "fwd"]
        r = [x for x, q in zip(h, every) if q[2] == "rev"]
        return round(sum(f) / len(f), 4), round(sum(r) / len(r), 4)
    retention = None
    if first_sleep_facts:
        rq = [question(f, k, fwd, rev) for f in first_sleep_facts for k in ("fwd", "rev")]
        retention = split(score_all(net, store, rq, gen, weights_only=True)) if rq else None
        retention = None if retention is None else dict(facts=len(first_sleep_facts), fwd=retention[0], rev=retention[1])
    pf = {k: round(sum(h / n for (f, kk), (h, n) in per_fact.items() if kk == k) /
                   max(1, sum(1 for (f, kk) in per_fact if kk == k)), 4) for k in ("fwd", "rev")}
    cf, cr = split(comb)
    wf, wr = split(wonly)
    return dict(arm=arm, seed=seed, stream_seed=stream_seed, facts=len(facts), cap=cap, sleep_at=SLEEP_AT,
                sleep_steps=SLEEP_STEPS, cons=CONS, rehearse=REH, log=log, sleeps=sleeps,
                online_fwd=round(tot["fwd"][0] / tot["fwd"][1], 4), online_rev=round(tot["rev"][0] / tot["rev"][1], 4),
                per_fact_queried_fwd=pf["fwd"], per_fact_queried_rev=pf["rev"],
                final_all_combined=dict(fwd=cf, rev=cr), final_all_weights=dict(fwd=wf, rev=wr),
                first_sleep_retention=retention, rows_final=len(store),
                heldout_before=heldout_before, heldout_after=eval_mem(net, store50, sets50, gen),
                secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["value", "sleep", "uncapped"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stream_seed", type=int, default=0)
    a = ap.parse_args()
    out = OUT / f"{a.arm}_s{a.seed}_ss{a.stream_seed}.json"
    if out.exists():
        print(f"{out} exists: not re-run")
        raise SystemExit
    r = run(a.arm, a.seed, a.stream_seed)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "log"}, indent=1))
