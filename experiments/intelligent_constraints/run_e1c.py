"""Experiment 1c driver: weight decay x transport, with the per-run MLP-off check. Existing results are reused.

    python run_e1c.py   -> runs/e1c/
"""
import json
from pathlib import Path

from train import run, save

OUT = Path(__file__).parent / "runs" / "e1c"
M = 256
ARMS = {"attn": {"A": ("plain", M), "B": ("dagger", M), "C": ("plain", 2 * M)},
        "sum": {"A": ("plain", M), "B": ("dagger", M)}}
WDS = (0.01, 0.1, 1.0)
SEEDS = (0, 1)
STEPS = 10000


def get(transport, mlp, w, wd, seed):
    p = OUT / f"{transport}_{mlp}_w{w}_wd{wd}_s{seed}.json"
    if p.exists():
        return json.load(open(p))
    r = run(mlp, w, True, seed, steps=STEPS, frozen=True, transport=transport, wd=wd, decay="matrices")
    save(r, p)
    print(f"ran {p.name}: train {r['final']['train_acc']}  one-way clean {r['final']['rev_oneway_clean']['hits']}"
          f"  {r['secs']}s", flush=True)
    return r


def main():
    hdr = (f"{'trans':5s} {'wd':>5s} arm s  train reach99 | fwd   revP  | 1-way: hits  mass   leak | "
           f"MLP off: fwd | transpose off: 1-way revP | secs")
    rows = []
    for transport, arms in ARMS.items():
        for wd in WDS:
            for arm, (mlp, w) in arms.items():
                for seed in SEEDS:
                    rows.append((transport, wd, arm, seed, get(transport, mlp, w, wd, seed)))
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for transport, wd, arm, seed, r in rows:
        fin, off, mo = r["final"], r.get("final_rev_off"), r["final_mlp_off"]
        c = fin["rev_oneway_clean"]
        o = f"{off['rev_oneway_clean']['hits']:.3f}  {off['rev_paired']['hits']:.3f}" if off else "  -      -  "
        print(f"{transport:5s} {wd:5.2f}  {arm}  {seed}  {fin['train_acc']:.3f} {str(r['reach99']):>6s} | "
              f"{fin['fwd']['hits']:.3f} {fin['rev_paired']['hits']:.3f} | {c['hits']:.3f}  {c['mass']:.3f}  "
              f"{c['leak']:.3f} |    {mo['fwd']['hits']:.3f}    |       {o}      | {r['secs']}")
    print(f"\nchance hits@1 on one-way questions: {rows[-1][4]['chance']}")


if __name__ == "__main__":
    main()
