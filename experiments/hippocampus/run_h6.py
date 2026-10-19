"""Experiment H6 driver: arms none / recent / retrieve x seeds 0, 1; then the summary. Reuses existing results."""
import json

from h6 import OUT, STEPS, run

ARMS, SEEDS = ("none", "recent", "retrieve"), (0, 1)


def get(arm, seed):
    p = OUT / f"{arm}_s{seed}.json"
    if p.exists():
        return json.load(open(p))
    r = run(arm, seed, STEPS)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(p, "w"), indent=1)
    print(f"ran {p.name} in {r['secs']}s", flush=True)
    return r


def main():
    runs = {(a, s): get(a, s) for a in ARMS for s in SEEDS}
    print("\narm seed | bits/char: heldout  caesar  mhg | second reading: first -> second (ratio) | final train loss | secs")
    for (a, s), r in runs.items():
        st, sr = r["streams"], r["second_reading"]
        print(f"  {a:8s} {s} | {st['heldout']['bpc']:.3f}  {st['caesar']['bpc']:.3f}  {st['mhg']['bpc']:.3f} | "
              f"{sr['first_bpc']:.3f} -> {sr['second_bpc']:.3f} ({sr['ratio']:.3f}) | {r['log'][-1]['loss_bits']:.3f} | {r['secs']}")
    print("\nmeans over seeds (bits/char):")
    for a in ARMS:
        m = {k: sum(runs[(a, s)]["streams"][k]["bpc"] for s in SEEDS) / len(SEEDS) for k in ("heldout", "caesar", "mhg")}
        ratio = sum(runs[(a, s)]["second_reading"]["ratio"] for s in SEEDS) / len(SEEDS)
        print(f"  {a:8s} heldout {m['heldout']:.3f}  caesar {m['caesar']:.3f}  mhg {m['mhg']:.3f}  second-reading ratio {ratio:.3f}")


if __name__ == "__main__":
    main()
