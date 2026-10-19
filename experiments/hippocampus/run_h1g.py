"""Experiment H1g driver: how reliable is PoPE content matching, and do more updates help? (EXPERIMENT_H1G.md)

Both arms: PoPE, the learned expansion, SHUFFLED read order, from scratch, seeds 6-9 (new seeds).
  PS4  batch 256, 4,000 steps, evaluated every 500 steps (H1f's P_sh recipe)
  PS8  batch 128, 8,000 steps, evaluated every 1,000 steps (the same examples, twice the updates)
Each trained reader is evaluated on world 0 under both read orders, with forward and reverse error classes and the
predictions saved. Reuses existing results.
"""
import json
from pathlib import Path

from h1 import H1B, run_memory
from run_h1d import both_orders

HERE = Path(__file__).parent
OUT = HERE / "runs" / "h1g"
ARMS = {"PS4": dict(steps=4000, batch=256, eval_every=500), "PS8": dict(steps=8000, batch=128, eval_every=1000)}
SEEDS = (6, 7, 8, 9)
COMMON = dict(expand=True, read_order="shuffled", pope=True)


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
    for a, cfg in ARMS.items():
        for s in SEEDS:
            ck = OUT / f"reader_{a}_s{s}.pt"
            t = get(OUT / f"reader_{a}_s{s}.json",
                    lambda cfg=cfg, s=s, ck=ck, a=a: dict(run_memory("sparse", s, save=ck, **H1B["B2"], **COMMON, **cfg),
                                                          arm=a))
            e = get(OUT / f"eval_{a}_s{s}.json", lambda ck=ck, s=s: both_orders(ck, s, pope=True, keep_pred=True))
            rows[(a, s)] = (t, e)
    print("\nreader | shuffled: REV / FWD | ranked: REV / FWD | shuffled classes rev ; fwd | curves rev | fwd | secs")
    for (a, s), (t, e) in rows.items():
        sh, rk = e["shuffled"], e["ranked"]
        cr = " ".join(f"{l['rev_oneway_clean']['hits']:.2f}" for l in t["log"])
        cf = " ".join(f"{l['fwd_oneway']['hits']:.2f}" for l in t["log"])
        print(f"  {a} s{s} | {sh['rev_oneway_clean']['hits']:.3f} / {sh['fwd_oneway']['hits']:.3f} | "
              f"{rk['rev_oneway_clean']['hits']:.3f} / {rk['fwd_oneway']['hits']:.3f} | "
              f"{sh['rev_oneway_clean']['classes']} ; {sh['fwd_oneway']['classes']} | {cr} | {cf} | {t['secs']}")


if __name__ == "__main__":
    main()
