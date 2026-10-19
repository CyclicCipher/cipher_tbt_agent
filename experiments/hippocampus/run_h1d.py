"""Experiment H1d driver: can a reader learn content matching when rank order carries nothing? (EXPERIMENT_H1D.md)

W1: warm start from the E1 reader of the same seed, then 4,500 steps with a SHUFFLED read order.
M1: from scratch, 4,500 steps with a MIXED read order (each question ranked or shuffled at random).
Each trained reader is then evaluated on world 0 under both read orders. Reuses existing results.
"""
import json
import time
from pathlib import Path

import torch

from h1 import (CTX, DEV, H1B, H1B_STEPS, MODEL, TEST_WORLD, VOCAB, Net, eval_mem, eval_sets, gated_write,
                make_facts, run_memory, store_factory)

HERE = Path(__file__).parent
OUT = HERE / "runs" / "h1d"
E1 = HERE / "runs" / "h1c"
ARMS = {"W1": lambda s: dict(expand=True, read_order="shuffled", init_from=E1 / f"reader_E1_s{s}.pt"),
        "M1": lambda s: dict(expand=True, read_order="mixed")}
SEEDS = (4, 5)


def both_orders(path, seed):
    """World 0 with the reader at `path`, under the ranked and the shuffled read order."""
    t0 = time.time()
    net = Net(VOCAB, ctx=CTX, **MODEL).to(DEV)
    net.load_state_dict(torch.load(path, map_location=DEV))
    net.eval()
    out = {}
    for order in ("ranked", "shuffled"):
        gen = torch.Generator(device=DEV).manual_seed(seed)
        F0 = make_facts(seed=TEST_WORLD)
        Store = store_factory(expand=True, read_order=order)
        store = Store(VOCAB, "sparse")
        perm = torch.randperm(len(F0.facts), generator=torch.Generator().manual_seed(seed)).tolist()
        gated_write(net, store, [F0.facts[j] for j in perm], gen, familiarity=True)
        out[order] = eval_mem(net, store, eval_sets(F0), gen, classify=True)
    out["secs"] = round(time.time() - t0, 1)
    return out


def get(path, fn):
    if path.exists():
        return json.load(open(path))
    r = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(path, "w"), indent=1)
    print(f"ran {path.name} in {r['secs']}s", flush=True)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for a in ARMS:
        for s in SEEDS:
            ck = OUT / f"reader_{a}_s{s}.pt"
            train = get(OUT / f"reader_{a}_s{s}.json",
                        lambda a=a, s=s, ck=ck: dict(run_memory("sparse", s, steps=H1B_STEPS, save=ck, **H1B["B2"],
                                                                **ARMS[a](s)), arm=a))
            ev = get(OUT / f"eval_{a}_s{s}.json", lambda ck=ck, s=s: both_orders(ck, s))
            rows[(a, s)] = (train, ev)
    base = {s: get(OUT / f"eval_E1_s{s}.json", lambda s=s: both_orders(E1 / f"reader_E1_s{s}.pt", s)) for s in SEEDS}
    print("\nreader | world 0 REV one-way clean: ranked / shuffled | fwd: ranked / shuffled | shuffled reverse classes")
    for s in SEEDS:
        e = base[s]
        print(f"  E1 s{s} (reference) | {e['ranked']['rev_oneway_clean']['hits']:.3f} / {e['shuffled']['rev_oneway_clean']['hits']:.3f} | "
              f"{e['ranked']['fwd_oneway']['hits']:.3f} / {e['shuffled']['fwd_oneway']['hits']:.3f} | "
              f"{e['shuffled']['rev_oneway_clean'].get('classes')}")
    for (a, s), (t, e) in rows.items():
        print(f"  {a} s{s} | {e['ranked']['rev_oneway_clean']['hits']:.3f} / {e['shuffled']['rev_oneway_clean']['hits']:.3f} | "
              f"{e['ranked']['fwd_oneway']['hits']:.3f} / {e['shuffled']['fwd_oneway']['hits']:.3f} | "
              f"{e['shuffled']['rev_oneway_clean'].get('classes')} | train {t['secs']}s")


if __name__ == "__main__":
    main()
