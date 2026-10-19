"""Experiment H1c driver: readers trained WITH the learned expansion (EXPERIMENT_H1C.md). Reuses existing results.

E1: expansion, ranked read order (best episode last), seeds 4-7.  E2: expansion, shuffled read order, seeds 4-5.
Then streams with the E1 readers of seeds 4 and 5: uncapped and value, stream seeds 0-2, expansion on.
"""
import json
from pathlib import Path

from h1 import H1B, H1B_STEPS, run_memory
from h2 import run_stream

OUT = Path(__file__).parent / "runs" / "h1c"
ARMS = {"E1": dict(expand=True, read_order="ranked"), "E2": dict(expand=True, read_order="shuffled")}
SEEDS = {"E1": (4, 5, 6, 7), "E2": (4, 5)}
STREAM_READERS, POLICIES, STREAM_SEEDS = (4, 5), ("uncapped", "value"), (0, 1, 2)


def get(path, fn):
    if path.exists():
        return json.load(open(path))
    r = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(path, "w"), indent=1)
    print(f"ran {path.name} in {r['secs']}s", flush=True)
    return r


def main():
    readers = {(a, s): get(OUT / f"reader_{a}_s{s}.json",
                           lambda a=a, s=s: dict(run_memory("sparse", s, steps=H1B_STEPS,
                                                            save=OUT / f"reader_{a}_s{s}.pt", **H1B["B2"], **ARMS[a]),
                                                 arm=a))
               for a in ARMS for s in SEEDS[a]}
    streams = {(p, s, ss): get(OUT / f"stream_{p}_E1s{s}_ss{ss}.json",
                               lambda p=p, s=s, ss=ss: run_stream(p, s, ss, expand=True,
                                                                  reader=OUT / f"reader_E1_s{s}.pt"))
               for s in STREAM_READERS for p in POLICIES for ss in STREAM_SEEDS}
    print("\nreaders, world 0: arm seed | fwd | REV one-way clean (recall) | reverse classes | reverse curve | secs")
    for (a, s), r in readers.items():
        f = r["final"]
        cl = f["rev_oneway_clean"].get("classes", {})
        curve = " ".join(f"{l['rev_oneway_clean']['hits']:.2f}" for l in r["log"])
        print(f"  {a} {s} | {f['fwd_oneway']['hits']:.3f} | {f['rev_oneway_clean']['hits']:.3f} "
              f"({f['rev_oneway_clean']['recall']:.2f}) | {cl} | {curve} | {r['secs']}")
    print("\nstreams (E1 readers, expansion): policy reader ss | online fwd rev mean | per-fact fwd rev mean | "
          "probe rev @6 -> @24 (recall) | rows")
    for (p, s, ss), r in streams.items():
        L = r["log"]
        print(f"  {p:8s} {s} {ss} | {r['online_fwd']:.3f} {r['online_rev']:.3f} {(r['online_fwd'] + r['online_rev']) / 2:.3f} | "
              f"{r['per_fact_fwd']:.3f} {r['per_fact_rev']:.3f} {(r['per_fact_fwd'] + r['per_fact_rev']) / 2:.3f} | "
              f"{L[5]['probe_rev']:.3f} -> {L[-1]['probe_rev']:.3f} ({L[-1]['probe_rev_recall']:.2f}) | {r['rows_final']}")


if __name__ == "__main__":
    main()
