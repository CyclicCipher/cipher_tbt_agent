"""Experiment H1e driver: readers with rotary positions (EXPERIMENT_H1E.md). Reuses existing results.

R_sh: RoPE, expansion, SHUFFLED read order, from scratch (E2's setting with RoPE).
R_rk: RoPE, expansion, ranked read order, from scratch (E1's setting with RoPE).
Each evaluated on world 0 under both read orders.
"""
import json
from pathlib import Path

from h1 import H1B, H1B_STEPS, run_memory
from run_h1d import both_orders

HERE = Path(__file__).parent
OUT = HERE / "runs" / "h1e"
ARMS = {"R_sh": dict(expand=True, read_order="shuffled", rope=True),
        "R_rk": dict(expand=True, read_order="ranked", rope=True)}
SEEDS = (4, 5)


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
            t = get(OUT / f"reader_{a}_s{s}.json",
                    lambda a=a, s=s, ck=ck: dict(run_memory("sparse", s, steps=H1B_STEPS, save=ck, **H1B["B2"],
                                                            **ARMS[a]), arm=a))
            e = get(OUT / f"eval_{a}_s{s}.json", lambda ck=ck, s=s: both_orders(ck, s, rope=True))
            rows[(a, s)] = (t, e)
    print("\nreader | world 0 REV one-way clean: ranked / shuffled | fwd: ranked / shuffled | shuffled reverse classes | "
          "reverse curve (training order) | secs")
    for (a, s), (t, e) in rows.items():
        curve = " ".join(f"{l['rev_oneway_clean']['hits']:.2f}" for l in t["log"])
        print(f"  {a} s{s} | {e['ranked']['rev_oneway_clean']['hits']:.3f} / {e['shuffled']['rev_oneway_clean']['hits']:.3f} | "
              f"{e['ranked']['fwd_oneway']['hits']:.3f} / {e['shuffled']['fwd_oneway']['hits']:.3f} | "
              f"{e['shuffled']['rev_oneway_clean'].get('classes')} | {curve} | {t['secs']}")


if __name__ == "__main__":
    main()
