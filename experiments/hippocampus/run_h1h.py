"""Experiment H1h driver: does non-causal attention let the reader learn FORWARD content matching? (EXPERIMENT_H1H.md)

Both arms: PoPE, the learned expansion, SHUFFLED read order, from scratch, batch 128 x 5,000 steps (sized from a
measured end-to-end run to fit the 2-minute rule), seeds 6-9 (paired with H1g).
  PC  causal attention (as in H1f/H1g)      PN  non-causal attention over the whole input
"""
import json
from pathlib import Path

from h1 import H1B, run_memory
from run_h1d import both_orders

HERE = Path(__file__).parent
OUT = HERE / "runs" / "h1h"
ARMS = {"PC": True, "PN": False}                                     # arm -> causal
SEEDS = (6, 7, 8, 9)
CFG = dict(steps=5000, batch=128, eval_every=1250, expand=True, read_order="shuffled", pope=True)


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
    for a, causal in ARMS.items():
        for s in SEEDS:
            ck = OUT / f"reader_{a}_s{s}.pt"
            t = get(OUT / f"reader_{a}_s{s}.json",
                    lambda s=s, ck=ck, causal=causal, a=a: dict(run_memory("sparse", s, save=ck, causal=causal,
                                                                           **H1B["B2"], **CFG), arm=a))
            e = get(OUT / f"eval_{a}_s{s}.json",
                    lambda ck=ck, s=s, causal=causal: both_orders(ck, s, pope=True, keep_pred=True, causal=causal))
            rows[(a, s)] = (t, e)
    print("\nreader | shuffled: REV / FWD | ranked: REV / FWD | shuffled fwd classes | curves rev | fwd | secs")
    for (a, s), (t, e) in rows.items():
        sh, rk = e["shuffled"], e["ranked"]
        cr = " ".join(f"{l['rev_oneway_clean']['hits']:.2f}" for l in t["log"])
        cf = " ".join(f"{l['fwd_oneway']['hits']:.2f}" for l in t["log"])
        print(f"  {a} s{s} | {sh['rev_oneway_clean']['hits']:.3f} / {sh['fwd_oneway']['hits']:.3f} | "
              f"{rk['rev_oneway_clean']['hits']:.3f} / {rk['fwd_oneway']['hits']:.3f} | {sh['fwd_oneway']['classes']} | "
              f"{cr} | {cf} | {t['secs']}")


if __name__ == "__main__":
    main()
