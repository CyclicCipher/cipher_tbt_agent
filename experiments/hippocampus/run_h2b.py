"""Experiment H2b driver: learned query expansion against crowding (EXPERIMENT_H2B.md). Reuses existing results.

1. world 0 with the expansion, for the H2 readers (seeds 2, 3): does retrieval that matches the relation lift reverse
   reading, including the reader stuck on the plateau?
2. streams: policies uncapped and value x readers 2, 3 x stream seeds 0, 1, 2, with and without the expansion
   (stream seed 0 without expansion is H2's result, reused).
"""
import json
import random
import time

import torch

from expand import expansion_table
from h1 import CTX, DEV, MODEL, VOCAB, Net, eval_mem, eval_sets, gated_write, make_facts, TEST_WORLD
from h2 import OUT as H2OUT, run_stream
from store import Store

OUT = H2OUT.parent / "h2b"
SEEDS, POLICIES, STREAM_SEEDS = (2, 3), ("uncapped", "value"), (0, 1, 2)


def world0(seed):
    t0 = time.time()
    net = Net(VOCAB, ctx=CTX, **MODEL).to(DEV)
    net.load_state_dict(torch.load(H2OUT / f"reader_s{seed}.pt", map_location=DEV))
    net.eval()
    gen = torch.Generator(device=DEV).manual_seed(seed)
    F0 = make_facts(seed=TEST_WORLD)
    sets = eval_sets(F0)
    order = torch.randperm(len(F0.facts), generator=torch.Generator().manual_seed(seed)).tolist()
    store = Store(VOCAB, "sparse")
    store.expansion = expansion_table()
    w = gated_write(net, store, [F0.facts[j] for j in order], gen, familiarity=True)
    return dict(seed=seed, expand=True, written=len(w), final=eval_mem(net, store, sets, gen),
                final_oracle=eval_mem(net, store, sets, gen, oracle=True), secs=round(time.time() - t0, 1))


def get(path, fn):
    if path.exists():
        return json.load(open(path))
    r = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(path, "w"), indent=1)
    print(f"ran {path.name} in {r['secs']}s", flush=True)
    return r


def stream_path(p, s, ss, x):
    if ss == 0 and not x:
        return H2OUT / f"stream_{p}_s{s}.json"                       # H2's run
    return OUT / f"stream_{p}_s{s}_ss{ss}{'_x' if x else ''}.json"


def main():
    expansion_table()
    w0 = {s: get(OUT / f"world0_x_s{s}.json", lambda s=s: world0(s)) for s in SEEDS}
    st = {(p, s, ss, x): get(stream_path(p, s, ss, x), lambda p=p, s=s, ss=ss, x=x: run_stream(p, s, ss, x))
          for s in SEEDS for p in POLICIES for ss in STREAM_SEEDS for x in (False, True)}
    print("\nworld 0 (reader seed): REV one-way clean without -> with expansion | recall | fwd")
    for s in SEEDS:
        base = json.load(open(H2OUT / f"reader_s{s}.json"))["final"]
        f = w0[s]["final"]
        print(f"  {s}: {base['rev_oneway_clean']['hits']:.3f} -> {f['rev_oneway_clean']['hits']:.3f} | "
              f"{base['rev_oneway_clean']['recall']:.3f} -> {f['rev_oneway_clean']['recall']:.3f} | "
              f"{base['fwd_oneway']['hits']:.3f} -> {f['fwd_oneway']['hits']:.3f}")
    print("\nstreams: policy seed ss exp | online fwd rev mean | per-fact fwd rev | probe rev @6 -> @24 "
          "(rev recall @24) | probe fwd @24 (fwd recall @24) | rows")
    for (p, s, ss, x), r in st.items():
        L = r["log"]
        pff, pfr = r.get("per_fact_fwd", float("nan")), r.get("per_fact_rev", float("nan"))
        fr = L[-1].get("probe_fwd_recall", float("nan"))
        print(f"  {p:8s} {s} {ss} {'x' if x else '-'} | {r['online_fwd']:.3f} {r['online_rev']:.3f} "
              f"{(r['online_fwd'] + r['online_rev']) / 2:.3f} | {pff:.3f} {pfr:.3f} | {L[5]['probe_rev']:.3f} -> "
              f"{L[-1]['probe_rev']:.3f} ({L[-1]['probe_rev_recall']:.2f}) | {L[-1]['probe_fwd']:.3f} ({fr:.2f}) | "
              f"{r['rows_final']}")
    print("\nmeans over stream seeds (online mean of fwd and rev):")
    for s in SEEDS:
        for p in POLICIES:
            for x in (False, True):
                v = [(st[(p, s, ss, x)]["online_fwd"] + st[(p, s, ss, x)]["online_rev"]) / 2 for ss in STREAM_SEEDS]
                print(f"  reader {s} {p:8s} {'exp' if x else '---'}: {sum(v) / len(v):.3f}  ({', '.join(f'{u:.3f}' for u in v)})")


if __name__ == "__main__":
    main()
