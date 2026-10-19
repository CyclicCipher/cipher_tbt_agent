"""Experiment H2 driver: the two readers, then every policy for each, then the summary. Existing results are reused."""
import json

from h1 import H1B, H1B_STEPS, run_memory
from h2 import OUT, run_stream

SEEDS = (2, 3)
POLICIES = ("uncapped", "fifo", "random", "value", "oracle")


def get(path, fn):
    if path.exists():
        return json.load(open(path))
    r = fn()
    json.dump(r, open(path, "w"), indent=1)
    print(f"ran {path.name} in {r['secs']}s", flush=True)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    readers = {s: get(OUT / f"reader_s{s}.json",
                      lambda s=s: dict(run_memory("sparse", s, steps=H1B_STEPS, save=OUT / f"reader_s{s}.pt",
                                                  **H1B["B2"]), arm="B2 (H2 reader)")) for s in SEEDS}
    streams = {(p, s): get(OUT / f"stream_{p}_s{s}.json", lambda p=p, s=s: run_stream(p, s))
               for s in SEEDS for p in POLICIES}
    print("\nreaders on world 0 (H1b replication): seed | fwd | REV one-way clean | duplicates | secs")
    for s, r in readers.items():
        f = r["final"]
        print(f"  {s} | {f['fwd_oneway']['hits']:.3f} | {f['rev_oneway_clean']['hits']:.3f} | "
              f"{r['gate']['written_again']} | {r['secs']}")
    print("\nstreams: policy seed | online fwd rev mean | probe rev @phase6 -> @24 (recall @24) | probe fwd @24 | "
          "rows | written | secs")
    for (p, s), r in streams.items():
        L = r["log"]
        print(f"  {p:8s} {s} | {r['online_fwd']:.3f} {r['online_rev']:.3f} {(r['online_fwd'] + r['online_rev']) / 2:.3f} | "
              f"{L[5]['probe_rev']:.3f} -> {L[-1]['probe_rev']:.3f} ({L[-1]['probe_rev_recall']:.2f}) | "
              f"{L[-1]['probe_fwd']:.3f} | {r['rows_final']} | {r['written_total']} | {r['secs']}")


if __name__ == "__main__":
    main()
