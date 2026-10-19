"""Experiment H2: a continual stream into the store, uncapped or capped with an eviction policy (EXPERIMENT_H2.md).

Step 1  `python h2.py --reader --seed 2`   trains a B2 reader (H1b's configuration, a new seed), saves its weights to
        runs/h2/reader_s2.pt (git-ignored) and its world-0 results to runs/h2/reader_s2.json (an H1b replication).
Step 2  `python h2.py --policy value --seed 2`   runs the stream with that reader -> runs/h2/stream_value_s2.json.

The stream: the union of worlds 100-103 (11,930 facts), in a fixed random order, split into 24 phases. Each phase writes
its facts through the gate (familiarity, then surprise), evicts down to the capacity, then answers 2,000 queries (half
forward, half reverse) sampled from all facts written so far with Zipf(1) popularity. Queries are answered BEFORE any
credit is given; a row earns evidence when it supplied the answer the reader gave (no ground truth used).
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

from h1 import (CTX, DEV, H1B, H1B_STEPS, MODEL, N_ENT, VOCAB, Net, contexts, conv, gated_write, make_facts, rel,
                run_memory)
from store import Store

HERE = Path(__file__).parent
OUT = HERE / "runs" / "h2"
STREAM_WORLDS, STREAM_SEED = (100, 101, 102, 103), 0
PHASES, QUERIES, CAP, ZIPF = 24, 2000, 3000, 1.0
PRIOR_AGE = QUERIES                                   # value = (evidence + 1) / (age + PRIOR_AGE), age in queries


def stream_world():
    facts = set()
    for w in STREAM_WORLDS:
        facts |= set(make_facts(seed=w).facts)
    facts = sorted(facts)
    rng = random.Random(STREAM_SEED)
    rng.shuffle(facts)                                                # arrival order
    ranks = list(range(1, len(facts) + 1))
    rng.shuffle(ranks)
    pop = {f: 1.0 / r ** ZIPF for f, r in zip(facts, ranks)}         # popularity, independent of arrival time
    fwd, rev = {}, {}
    for a, r, b in facts:
        fwd.setdefault((a, r), set()).add(b)
        rev.setdefault((r, b), set()).add(a)
    return facts, pop, fwd, rev


def ask(net, store, qs, gen):
    """qs: list of (prompt, correct set, kind, fact). Returns hits per query and the predicted entity per query."""
    P = torch.tensor([q[0] for q in qs])
    ctx, ids = contexts(store, P, gen)
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
        pred = net(ctx).float()[:, :N_ENT].argmax(-1).tolist()
    hits = [int(pred[i] in q[1]) for i, q in enumerate(qs)]
    return hits, pred, ids


def question(f, kind, fwd, rev):
    a, r, b = f
    if kind == "fwd":
        return ((a, rel(r)), fwd[(a, r)], kind, f)
    return ((b, conv(r)), rev[(r, b)], kind, f)


def credit(store, qs, pred, ids):
    """Evidence +1 for each read row whose answer slot holds the reader's answer (forward: (a, r, pred); reverse:
    (pred, r, b))."""
    rows = store.rows.tolist()
    for i, q in enumerate(qs):
        a, r, b = q[3]
        for j in ids[i].tolist():
            if j < 0:
                continue
            t = rows[j]
            if (q[2] == "fwd" and t == [a, rel(r), pred[i]]) or (q[2] == "rev" and t == [pred[i], rel(r), b]):
                store.evidence[j] += 1
                store.last[j] = store.step


def evict(store, policy, cap, pop, rng):
    n = len(store) - cap
    if n <= 0:
        return 0
    idx = list(range(len(store)))
    if policy == "fifo":
        order = sorted(idx, key=lambda i: store.born[i])
    elif policy == "random":
        order = idx[:]
        rng.shuffle(order)
    elif policy == "value":
        order = sorted(idx, key=lambda i: ((store.evidence[i] + 1) / (store.step - store.born[i] + PRIOR_AGE),
                                           store.born[i]))
    elif policy == "oracle":                                         # an upper bound: true popularity, not observable
        rows = store.rows.tolist()
        order = sorted(idx, key=lambda i: pop[(rows[i][0], rows[i][1] - N_ENT, rows[i][2])])
    store.remove(order[:n])
    return n


def run_stream(policy, seed):
    t0 = time.time()
    net = Net(VOCAB, ctx=CTX, **MODEL).to(DEV)
    net.load_state_dict(torch.load(OUT / f"reader_s{seed}.pt", map_location=DEV))
    net.eval()
    gen = torch.Generator(device=DEV).manual_seed(STREAM_SEED)
    rng = random.Random(STREAM_SEED)
    facts, pop, fwd, rev = stream_world()
    chunks = [facts[i * len(facts) // PHASES:(i + 1) * len(facts) // PHASES] for i in range(PHASES)]
    probe = [question(f, k, fwd, rev) for f in chunks[0] for k in ("fwd", "rev")]
    store = Store(VOCAB, "sparse")
    cap = None if policy == "uncapped" else CAP
    seen, log = [], []
    tot = {"fwd": [0, 0], "rev": [0, 0]}
    for ph, chunk in enumerate(chunks, 1):
        w = gated_write(net, store, chunk, gen, familiarity=True)
        seen += chunk
        evicted = evict(store, policy, cap, pop, rng) if cap else 0
        weights = [pop[f] for f in seen]
        picks = rng.choices(seen, weights=weights, k=QUERIES)
        qs = [question(f, "fwd" if k % 2 == 0 else "rev", fwd, rev) for k, f in enumerate(picks)]
        ph_hits = {"fwd": [0, 0], "rev": [0, 0]}
        for i in range(0, QUERIES, 500):                             # answer, then credit, batch by batch
            part = qs[i:i + 500]
            hits, pred, ids = ask(net, store, part, gen)
            for q, h in zip(part, hits):
                ph_hits[q[2]][0] += h
                ph_hits[q[2]][1] += 1
            credit(store, part, pred, ids)
            store.step += len(part)
        for k in tot:
            tot[k][0] += ph_hits[k][0]
            tot[k][1] += ph_hits[k][1]
        ph_probe, _, pids = ask(net, store, probe, gen)
        good = [{j for j in store.where.get((q[3][0], rel(q[3][1]), q[3][2]), [])} for q in probe]
        recall = [any(int(j) in good[i] for j in pids[i].tolist()) for i in range(len(probe))]
        pf = [h for h, q in zip(ph_probe, probe) if q[2] == "fwd"]
        pr = [h for h, q in zip(ph_probe, probe) if q[2] == "rev"]
        rr = [x for x, q in zip(recall, probe) if q[2] == "rev"]
        log.append(dict(phase=ph, written=len(w), evicted=evicted, rows=len(store),
                        online_fwd=round(ph_hits["fwd"][0] / ph_hits["fwd"][1], 4),
                        online_rev=round(ph_hits["rev"][0] / ph_hits["rev"][1], 4),
                        probe_fwd=round(sum(pf) / len(pf), 4), probe_rev=round(sum(pr) / len(pr), 4),
                        probe_rev_recall=round(sum(rr) / len(rr), 4)))
    return dict(policy=policy, seed=seed, cap=cap, phases=PHASES, queries=QUERIES, zipf=ZIPF, prior_age=PRIOR_AGE,
                facts=len(facts), log=log,
                online_fwd=round(tot["fwd"][0] / tot["fwd"][1], 4), online_rev=round(tot["rev"][0] / tot["rev"][1], 4),
                written_total=sum(l["written"] for l in log), rows_final=len(store),
                bytes_final=len(store) * store.row_bytes(), secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", action="store_true")
    ap.add_argument("--policy", choices=["uncapped", "fifo", "random", "value", "oracle"])
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (f"reader_s{a.seed}.json" if a.reader else f"stream_{a.policy}_s{a.seed}.json")
    if out.exists():
        print(f"{out} exists: not re-run")
        raise SystemExit
    if a.reader:
        r = run_memory("sparse", a.seed, steps=H1B_STEPS, save=OUT / f"reader_s{a.seed}.pt", **H1B["B2"])
        r["arm"] = "B2 (H2 reader)"
    else:
        r = run_stream(a.policy, a.seed)
    print(json.dumps({k: v for k, v in r.items() if k != "log"}, indent=1))
    json.dump(r, open(out, "w"), indent=1)
