"""Experiment H3b driver: longer sleep and self-checked eviction (EXPERIMENT_H3B.md). Reuses existing results."""
import json

from h3 import OUT, run_sleep

ARMS = {"I5k": dict(steps=5000, cons=128, rehearse=128, evict_rule="self"),       # 1:1 consolidation : rehearsal
        "I5k31": dict(steps=5000, cons=192, rehearse=64, evict_rule="self")}      # 3:1
SEEDS = (4, 5)


def get(arm, seed):
    p = OUT / f"{arm}_s{seed}.json"
    if p.exists():
        return json.load(open(p))
    r = run_sleep("I", seed, **ARMS[arm])
    r["arm"] = arm
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(p, "w"), indent=1)
    print(f"ran {p.name} in {r['secs']}s", flush=True)
    return r


def h(d, k):
    return f"{d[k]['hits']:.3f}"


def main():
    runs = {(a, s): get(a, s) for a in ARMS for s in SEEDS}
    print("\narm seed | WEIGHTS ONLY fwd rev after | memory rev before->after | held-out rev before->after | "
          "evicted (self rule) / p-rule would / of rows | combined fwd rev: after sleep -> after eviction (read share) | secs")
    for (a, s), r in runs.items():
        b, f, e = r["before"], r["after"], r["eviction"]
        ce = e["combined_after_eviction"]
        print(f"  {a:6s} {s} | {h(f['empty'], 'fwd_oneway')} {h(f['empty'], 'rev_oneway_clean')} | "
              f"{h(b['memory'], 'rev_oneway_clean')}->{h(f['memory'], 'rev_oneway_clean')} | "
              f"{h(b['heldout'], 'rev_oneway_clean')}->{h(f['heldout'], 'rev_oneway_clean')} | "
              f"{e['consolidated']} / {e['p_rule_both']} / {e['rows_before']} | "
              f"{h(f['combined'], 'fwd_oneway')} {h(f['combined'], 'rev_oneway_clean')} -> "
              f"{h(ce, 'fwd_oneway')} {h(ce, 'rev_oneway_clean')} ({ce['rev_oneway_clean']['read_fraction']:.2f}) | {r['secs']}")


if __name__ == "__main__":
    main()
