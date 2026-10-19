"""Experiment H3 driver: sleep arms F and I for the E1 readers of seeds 4 and 5. Reuses existing results."""
import json

from h3 import OUT, run_sleep

ARMS, SEEDS = ("F", "I"), (4, 5)


def get(arm, seed):
    p = OUT / f"{arm}_s{seed}.json"
    if p.exists():
        return json.load(open(p))
    r = run_sleep(arm, seed)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(p, "w"), indent=1)
    print(f"ran {p.name} in {r['secs']}s", flush=True)
    return r


def h(d, name):
    return f"{d[name]['hits']:.3f}"


def main():
    runs = {(a, s): get(a, s) for a in ARMS for s in SEEDS}
    print("\narm seed | inference kept/correct | WEIGHTS ONLY fwd rev (before -> after) | memory rev before -> after | "
          "held-out rev before -> after | consolidated fwd/rev/both of rows -> rows | combined fwd rev: before sleep, "
          "after sleep, after eviction | secs")
    for (a, s), r in runs.items():
        b, f, e, inf = r["before"], r["after"], r["eviction"], r["inference"]
        ce = e["combined_after_eviction"]
        infs = f"{inf['kept']}/{inf['correct_kept']}" if inf else "-"
        print(f"  {a} {s} | {infs} | {h(b['empty'], 'fwd_oneway')} {h(b['empty'], 'rev_oneway_clean')} -> "
              f"{h(f['empty'], 'fwd_oneway')} {h(f['empty'], 'rev_oneway_clean')} | "
              f"{h(b['memory'], 'rev_oneway_clean')} -> {h(f['memory'], 'rev_oneway_clean')} | "
              f"{h(b['heldout'], 'rev_oneway_clean')} -> {h(f['heldout'], 'rev_oneway_clean')} | "
              f"{e['fwd_consolidated']}/{e['rev_consolidated']}/{e['consolidated']} of {e['rows_before']} -> {e['rows_after']} | "
              f"{h(b['combined'], 'fwd_oneway')} {h(b['combined'], 'rev_oneway_clean')}, "
              f"{h(f['combined'], 'fwd_oneway')} {h(f['combined'], 'rev_oneway_clean')}, "
              f"{h(ce, 'fwd_oneway')} {h(ce, 'rev_oneway_clean')} (read {ce['rev_oneway_clean']['read_fraction']:.2f}) | {r['secs']}")


if __name__ == "__main__":
    main()
