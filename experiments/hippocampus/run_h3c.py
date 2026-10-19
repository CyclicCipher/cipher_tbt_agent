"""Experiments H3c and H3d driver: value / sleep / uncapped / sleep_pr x E1 readers 4, 5 x stream seeds 0, 1.
Reuses existing results (H3c's runs are not repeated)."""
import json

from h3c import OUT, run

ARMS = ("value", "sleep", "uncapped", "sleep_pr", "sleep_pr_fix", "sleep_pr_dose")   # sleep_pr: H3d; *_fix/_dose: H3e
SEEDS, STREAM_SEEDS = (4, 5), (0, 1)


def get(arm, seed, ss):
    p = OUT / f"{arm}_s{seed}_ss{ss}.json"
    if p.exists():
        return json.load(open(p))
    r = run(arm, seed, ss)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(p, "w"), indent=1)
    print(f"ran {p.name} in {r['secs']}s", flush=True)
    return r


def main():
    runs = {(a, s, ss): get(a, s, ss) for s in SEEDS for ss in STREAM_SEEDS for a in ARMS}
    print("\narm reader ss | FINAL all facts, combined fwd rev mean | weights only fwd rev | online fwd rev mean | "
          "rows | sleeps evicted | first-sleep retention | held-out rev before->after | secs")
    for (a, s, ss), r in runs.items():
        c, w = r["final_all_combined"], r["final_all_weights"]
        ret = r["first_sleep_retention"]
        ev = [x["evicted"] for x in r["sleeps"]]
        hb, ha = r["heldout_before"]["rev_oneway_clean"]["hits"], r["heldout_after"]["rev_oneway_clean"]["hits"]
        print(f"  {a:8s} {s} {ss} | {c['fwd']:.3f} {c['rev']:.3f} {(c['fwd'] + c['rev']) / 2:.3f} | {w['fwd']:.3f} {w['rev']:.3f} | "
              f"{r['online_fwd']:.3f} {r['online_rev']:.3f} {(r['online_fwd'] + r['online_rev']) / 2:.3f} | {r['rows_final']} | "
              f"{ev} | {ret} | {hb:.3f}->{ha:.3f} | {r['secs']}")


if __name__ == "__main__":
    main()
