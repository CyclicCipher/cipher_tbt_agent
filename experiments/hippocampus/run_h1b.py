"""Experiment H1b driver: arms B2 and B3, two seeds each, then the summary with H1's B for reference. Reuses results."""
import json
from pathlib import Path

from h1 import H1B, H1B_STEPS, run_memory

HERE = Path(__file__).parent
OUT = HERE / "runs" / "h1b"


def get(arm, seed):
    p = OUT / f"{arm}_s{seed}.json"
    if p.exists():
        return json.load(open(p))
    r = run_memory("sparse", seed, steps=H1B_STEPS, **H1B[arm])
    r["arm"] = arm
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(p, "w"), indent=1)
    print(f"ran {p.name} in {r['secs']}s", flush=True)
    return r


def main():
    rows = [(arm, s, get(arm, s)) for arm in ("B2", "B3") for s in (0, 1)]
    rows = [("B (H1)", s, json.load(open(HERE / "runs" / "h1" / f"B_s{s}.json"))) for s in (0, 1)] + rows
    print(f"\n{'arm':7s} s | fwd hits mass | REV 1-way clean hits mass recall | oracle rev | leak rev | "
          f"gate: 1st / again (surprise-only again) | secs")
    for arm, s, r in rows:
        f, o, lk, g = r["final"], r["final_oracle"], r["leak_control"], r["gate"]
        so = g.get("again_stats", {}).get("surprise_only", "-")
        print(f"{arm:7s} {s} | {f['fwd_oneway']['hits']:.3f} {f['fwd_oneway']['mass']:.3f} | "
              f"{f['rev_oneway_clean']['hits']:.3f} {f['rev_oneway_clean']['mass']:.3f} "
              f"{f['rev_oneway_clean']['recall']:.3f} | {o['rev_oneway_clean']['hits']:.3f}      | "
              f"{lk['rev_oneway_clean']['hits']:.3f}    | {g['written_first']} / {g['written_again']} ({so}) | {r['secs']}")
    for arm, s, r in rows[2:]:
        print(arm, s, "curve (step: fwd/rev):",
              " ".join(f"{l['step']}:{l['fwd_oneway']['hits']:.2f}/{l['rev_oneway_clean']['hits']:.2f}" for l in r["log"]))


if __name__ == "__main__":
    main()
