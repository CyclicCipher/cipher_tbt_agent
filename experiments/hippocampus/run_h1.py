"""Experiment H1 driver: every arm and seed, then the summary table. Existing results are reused, never re-run."""
import json
from pathlib import Path

from h1 import run_memory, run_weights

OUT = Path(__file__).parent / "runs" / "h1"
CONFIGS = ([("B", None, s) for s in (0, 1)] + [("C", None, s) for s in (0, 1)] +
           [("A", e, s) for e in (1, 8, 64) for s in (0, 1)])


def get(arm, e, seed):
    p = OUT / (f"A_e{e}_s{seed}.json" if arm == "A" else f"{arm}_s{seed}.json")
    if p.exists():
        return json.load(open(p))
    r = run_weights(e, seed) if arm == "A" else run_memory("sparse" if arm == "B" else "dense", seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(p, "w"), indent=1)
    print(f"ran {p.name} in {r['secs']}s", flush=True)
    return r


def main():
    rows = [(arm, e, s, get(arm, e, s)) for arm, e, s in CONFIGS]
    print(f"\n{'arm':8s} s | fwd 1-way hits/mass | REV 1-way clean hits/mass recall | rev paired | "
          f"oracle rev | leak rev | gate: 1st again (by kind) | secs")
    for arm, e, s, r in rows:
        f = r["final"]
        lab = f"A e={e}" if arm == "A" else arm
        if arm == "A":
            print(f"{lab:8s} {s} |   {f['fwd_oneway']['hits']:.3f} {f['fwd_oneway']['mass']:.3f}     |   "
                  f"{f['rev_oneway_clean']['hits']:.3f} {f['rev_oneway_clean']['mass']:.3f}   -    | "
                  f"{f['rev_paired']['hits']:.3f}      |    -       |    -     |            -              | {r['secs']}")
        else:
            g, o, lk = r["gate"], r["final_oracle"], r["leak_control"]
            print(f"{lab:8s} {s} |   {f['fwd_oneway']['hits']:.3f} {f['fwd_oneway']['mass']:.3f}     |   "
                  f"{f['rev_oneway_clean']['hits']:.3f} {f['rev_oneway_clean']['mass']:.3f} "
                  f"{f['rev_oneway_clean']['recall']:.3f} | {f['rev_paired']['hits']:.3f}      |   "
                  f"{o['rev_oneway_clean']['hits']:.3f}    |  {lk['rev_oneway_clean']['hits']:.3f}   | "
                  f"{g['written_first']} {g['written_again']} {g['again_by_kind']} | {r['secs']}")


if __name__ == "__main__":
    main()
